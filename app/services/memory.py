"""基于mem0.AsyncMemory、PostgreSQL pgvector、Valkey/Redis 或内存缓存实现的长期记忆服务
主要功能：负责保存、检索用户历史信息，例如用户偏好、背景资料和长期上下文
它保存的是长期用户记忆，而 LangGraph 的 PostgreSQL Checkpointer 保存的是会话状态和聊天历史，两者职责不同

核心接口：
_get_memory()：延迟初始化记忆组件
search()：搜索用户相关的记忆
add()：将消息添加到用户的长期记忆

工作流程：
    用户消息
      ↓
    memory_service.search()
      ↓
    得到相关长期记忆
      ↓
    写入 GraphState.long_term_memory
      ↓
    拼接到系统提示词
      ↓
    调用 LLM
      ↓
    memory_service.add() 保存本轮对话
"""

from mem0 import AsyncMemory

from app.core.cache import (
    cache_key,
    cache_service,
)
from app.core.config import settings
from app.core.logging import logger


class MemoryService:

    def __init__(self):
        """初始化记忆服务."""
        self._memory: AsyncMemory | None = None

    # 延迟初始化记忆组件：配置 pgvector 数据库连接、OpenAI LLM 和嵌入模型
    async def _get_memory(self) -> AsyncMemory | None:
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return None
        if not settings.LONG_TERM_MEMORY_EMBEDDER_API_KEY or not settings.LONG_TERM_MEMORY_EMBEDDER_BASE_URL:
            logger.warning(
                "长期记忆配置缺失，已跳过初始化",
                required=["LONG_TERM_MEMORY_EMBEDDER_API_KEY", "LONG_TERM_MEMORY_EMBEDDER_BASE_URL"],
            )
            return None
        if self._memory is None:
            # AsyncMemory 是 mem0 提供的异步长期记忆管理对象，负责把用户对话转换成可长期保存、可语义检索的用户记忆
            self._memory = AsyncMemory.from_config(
                config_dict={
                    "vector_store": {  # 保存和搜索向量，当前使用 PostgreSQL + pgvector
                        "provider": "pgvector",
                        "config": {
                            "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                            "embedding_model_dims": settings.LONG_TERM_MEMORY_EMBEDDER_DIMENSIONS,
                            "dbname": settings.POSTGRES_DB,
                            "user": settings.POSTGRES_USER,
                            "password": settings.POSTGRES_PASSWORD,
                            "host": settings.POSTGRES_HOST,
                            "port": settings.POSTGRES_PORT,
                        },
                    },
                    "llm": {  # 判断对话中哪些内容应该保存为记忆
                        "provider": "openai",
                        "config": {
                            "model": settings.LONG_TERM_MEMORY_MODEL,
                            "api_key": settings.LONG_TERM_MEMORY_LLM_API_KEY,
                            "openai_base_url": settings.LONG_TERM_MEMORY_LLM_BASE_URL,
                        },
                    },
                    "embedder": {  # 用于生成向量的嵌入模型
                        "provider": "openai",
                        "config": {
                            "model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL,
                            "api_key": settings.LONG_TERM_MEMORY_EMBEDDER_API_KEY,
                            "openai_base_url": settings.LONG_TERM_MEMORY_EMBEDDER_BASE_URL,
                        },
                    },
                }
            )
        return self._memory

    async def initialize(self) -> None:
        """预热 mem0 AsyncMemory 实例及其 pgvector 连接池.

        在应用启动时调用一次，避免首次 search() 或 add() 承担 from_config 和
        pgvector.list_cols() 初始化产生的约 130 毫秒冷启动开销.
        """
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return
        await self._get_memory()

    async def search(self, user_id: str | None, query: str) -> str:
        """搜索用户相关的记忆.

        先检查缓存；缓存未命中时查询 mem0 并缓存结果.

        查询成功时返回格式化的记忆字符串；失败或未提供 user_id 时返回空字符串.
        匿名会话会跳过长期记忆，避免共享同一个用户分区.
        """
        if user_id is None:
            return ""
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return ""
        try:
            # 优先检查redis长期记忆缓存
            key = cache_key("memory", str(user_id), query)
            cached = await cache_service.get(key)
            if cached is not None:
                logger.info("长期记忆搜索命中缓存", user_id=user_id, cache_key=key)
                return cached
            logger.info("长期记忆搜索未命中缓存", user_id=user_id, cache_key=key)

            # 再查询mem0长期记忆，在PostgreSQL + pgvector中
            memory = await self._get_memory()
            if memory is None:
                return ""
            results = await memory.search(query=query, filters={"user_id": str(user_id)})
            memories = results.get("results", []) if isinstance(results, dict) else []
            result = "\n".join([f"* {r['memory']}" for r in memories])

            # 缓存成功结果到redis
            if result:
                await cache_service.set(key, result)
                logger.info("长期记忆搜索结果已写入缓存", user_id=user_id, cache_key=key)
            else:
                logger.info("长期记忆搜索结果为空，跳过缓存写入", user_id=user_id, cache_key=key)

            return result
        except Exception as e:
            logger.exception("搜索长期记忆失败", user_id=user_id, error=str(e))
            return ""

    async def add(self, user_id: str | None, messages: list[dict], metadata: dict | None = None) -> None:
        """将消息添加到用户的长期记忆.

        当 ``user_id`` 为 ``None`` 时不执行任何操作，原因参见 ``search``.
        """
        if user_id is None:
            return
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return
        try:
            memory = await self._get_memory()
            if memory is None:
                return
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
        except Exception as e:
            logger.exception("更新长期记忆失败", user_id=user_id, error=str(e))


memory_service = MemoryService()
