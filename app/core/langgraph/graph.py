"""LangGraph Agent.
核心功能
    1.把一次聊天请求编排成完整的 LangGraph Agent 流程
    2.包括会话状态、长期记忆、LLM 调用、工具调用、检查点持久化、流式输出和中断恢复

一句话：
    1.service.py 负责“怎么调用 LLM”
    2.graph.py 负责“什么时候调用 LLM、是否调用工具、如何恢复会话，以及如何把整个 Agent 流程串起来”

核心方法：
    接口层
        get_response() - 普通聊天业务入口
        get_stream_response() - 流式聊天业务入口
            ↓
    图执行层
        graph.ainvoke()
        graph.astream()
            ↓
    节点编排层
        _chat() - 聊天节点
        _tool_call() - 工具调用节点
            ↓
    基础能力层
        LLMService
        MemoryService
        AsyncPostgresSaver
"""

import asyncio
from typing import (
    AsyncGenerator,
    Optional,
    cast, Any,
)
from urllib.parse import quote_plus

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RetryPolicy,
    StateSnapshot,
)
from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import langfuse_callback_handler
from app.core.prompts import load_system_prompt
from app.schemas import (
    GraphState, Message
)
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.utils.graph import (
    dump_messages,
    extract_text_content,
    prepare_messages,
    process_llm_response,
)

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]


class LangGraphAgent:
    # 初始化 LangGraph Agent
    def __init__(self):
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[PostgresConnPool] = None
        self._graph: Optional[CompiledStateGraph] = None

    # 成员方法：获取 PostgreSQL 连接池
    async def _get_connection_pool(self) -> AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None:
        if self._connection_pool is None:
            try:
                max_size = settings.POSTGRES_POOL_SIZE
                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(  # type: ignore
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
            except Exception as e:
                raise e
        return self._connection_pool

    # 成员方法：获取图
    async def _get_graph(self) -> CompiledStateGraph[Any, None, Any, Any] | None:
        if self._graph is None:
            self._graph = await self.create_graph()
        return self._graph

    # 成员方法：处理消息
    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # 仅保留 assistant 和 user 消息
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    # 成员方法：定义聊天节点
    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """处理聊天状态并生成响应.

        参数：
            state (GraphState): 当前会话状态.
            config (RunnableConfig): 本次调用的可运行配置.

        返回：
            Command: 包含更新状态和下一个待执行节点的命令对象.
        """
        # 获取当前大模型LLM
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )

        # 获取当前会话的用户名和线程ID
        username = config.get("metadata", {}).get("username")
        thread_id = config.get("configurable", {}).get("thread_id")

        # 获取系统提示词
        SYSTEM_PROMPT = load_system_prompt(username=username, long_term_memory=state.long_term_memory)
        messages = prepare_messages(state.messages, SYSTEM_PROMPT)

        try:
            # 指标监测：LLM 推理时长
            with llm_inference_duration_seconds.labels(model=model_name).time():
                # 调用大模型LLM：带自动重试和循环回退能力
                response_message = await self.llm_service.call(dump_messages(messages))
            response_message = process_llm_response(response_message)

            # 根据是否存在工具调用确定下一个节点
            if isinstance(response_message, AIMessage) and response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "所有大模型调用均失败",
                session_id=thread_id,
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"无法获取 LLM 响应: {str(e)}")

    # 成员方法：定义工具节点
    async def _tool_call(self, state: GraphState) -> Command:
        """处理最后一条消息中的工具调用.

        参数：
            state: 包含消息和工具调用的当前 Agent 状态.

        返回：
            Command: 包含更新消息并返回聊天节点路由的命令对象.
        """
        tool_calls = state.messages[-1].tool_calls

        async def _execute_tool(tool_call: dict) -> ToolMessage:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            return ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

        # 存在多个工具调用时并发执行
        if len(tool_calls) == 1:
            outputs = [await _execute_tool(tool_calls[0])]
        else:
            outputs = list(await asyncio.gather(*[_execute_tool(tc) for tc in tool_calls]))

        return Command(update={"messages": outputs}, goto="chat")

    # 对外提供接口：创建图
    async def create_graph(self) -> CompiledStateGraph[Any, None, Any, Any] | None:
        if self._graph is None:
            try:
                # 1.定义状态：GraphState

                # 2.定义节点：chat, tool_call

                # 3.构建图Graph
                graph_builder = StateGraph(GraphState)
                # 3.1 添加节点1
                graph_builder.add_node("chat", self._chat, destinations=("tool_call", END))
                # 3.2 添加节点2
                graph_builder.add_node(
                    "tool_call",
                    self._tool_call,
                    destinations=("chat",),
                    retry_policy=RetryPolicy(max_attempts=3),
                )
                # 3.3 设置入口和出口节点
                graph_builder.set_entry_point("chat")
                graph_builder.set_finish_point("chat")

                # 4.连接池无法创建时抛出异常；没有检查点保存器就不提供服务.
                connection_pool = await self._get_connection_pool()
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()

                # 5.编译图
                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )
            except Exception as e:
                raise e

        return self._graph

    # 对外提供接口：非流式响应
    async def get_response(
            self,
            messages: list[Message],
            session_id: str,
            user_id: Optional[str] = None,
            username: Optional[str] = None,
    ) -> list[Message]:
        """获取 LLM 响应.

        参数：
            messages (list[Message]): 要发送给 LLM 的消息.
            session_id (str): 会话 ID.
            user_id (Optional[str]): 用户 ID.
            username (Optional[str]): 用户显示名称.

        返回：
            list[Message]: LLM 响应.
        """
        # 1. 获取LangGraph图
        graph = await self._get_graph()

        # 2. 配置回调处理程序
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if langfuse_callback_handler else []

        # 3. 配置可运行参数
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

        try:
            # 并行读取会话状态checkpointer
            # 并行搜索长期记忆：先查缓存，缓存未命中时查询pgvector
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, messages[-1].content),
            )

            if state.next:  # 如果存在中断的图
                response = await graph.ainvoke(
                    Command(resume=messages[-1].content),
                    config=config,
                )
            else:
                relevant_memory = relevant_memory or "No relevant memory found."
                response = await graph.ainvoke(
                    input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                    config=config,
                )

            # 检查本次调用期间图是否被中断
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("图执行已中断", session_id=session_id, interrupt_value=str(interrupt_value))
                return [Message(role="assistant", content=str(interrupt_value))]

            openai_msgs = cast(list[dict], convert_to_openai_messages(response["messages"]))
            asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
            return self.__process_messages(response["messages"])

        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("图执行已中断", session_id=session_id, interrupt_value=str(interrupt_value))
            return [Message(role="assistant", content=str(interrupt_value))]

        except Exception as e:
            logger.exception("获取非流式响应失败", error=str(e), session_id=session_id)
            raise

    # 对外提供接口：流式响应
    async def get_stream_response(
            self,
            messages: list[Message],
            session_id: str,
            user_id: Optional[str] = None,
            username: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """获取 LLM 的流式响应.

        参数：
            messages (list[Message]): 要发送给 LLM 的消息.
            session_id (str): 会话 ID.
            user_id (Optional[str]): 用户 ID.
            username (Optional[str]): 用户显示名称.

        生成：
            str: LLM 响应的令牌.
        """
        # 1. 获取LangGraph图
        graph = await self._get_graph()

        # 2. 配置回调处理程序
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if langfuse_callback_handler else []

        # 3. 配置可运行参数
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

        try:
            # 并发执行状态检查和记忆搜索，节省 200-500 毫秒
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, messages[-1].content),
            )

            if state.next:
                logger.info("正在恢复中断的流式图执行", session_id=session_id, next_nodes=state.next)
                graph_input = Command(resume=messages[-1].content)
            else:
                relevant_memory = relevant_memory or "No relevant memory found."
                graph_input = {"messages": dump_messages(messages), "long_term_memory": relevant_memory}

            async for token, _ in graph.astream(
                    graph_input,
                    config,
                    stream_mode="messages",
            ):
                if not isinstance(token, (AIMessage, AIMessageChunk)):
                    continue

                text = extract_text_content(token.content)
                if text:
                    yield text

            # 流式处理完成后检查中断状态或更新记忆
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("流式图执行已中断", session_id=session_id, interrupt_value=str(interrupt_value))
                yield str(interrupt_value)
            elif state.values and "messages" in state.values:
                openai_msgs = cast(list[dict], convert_to_openai_messages(state.values["messages"]))
                asyncio.create_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("流式图执行已中断", session_id=session_id, interrupt_value=str(interrupt_value))
            yield str(interrupt_value)
        except Exception as stream_error:
            logger.exception("流式响应处理失败", error=str(stream_error), session_id=session_id)
            raise stream_error

    # 对外提供接口：获取聊天历史
    async def get_chat_history(self, session_id: str) -> list[Message]:
        graph = await self._get_graph()
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__process_messages(state.values["messages"]) if state.values else []

    # 对外提供接口：清空聊天历史
    async def clear_chat_history(self, session_id: str) -> None:
        try:
            # 确保连接池已在当前事件循环中初始化
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            # 在一次 pipeline 往返中批量执行所有 DELETE 操作
            async with conn_pool.connection() as conn:
                async with conn.pipeline():
                    for table in settings.CHECKPOINT_TABLES:
                        await conn.execute(
                            sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                            (session_id,),
                        )
                logger.info("checkpoint清理成功", tables=settings.CHECKPOINT_TABLES, session_id=session_id)

        except Exception as e:
            logger.error("清理聊天历史失败", session_id=session_id, error=str(e))
            raise
