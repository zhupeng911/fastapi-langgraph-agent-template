from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)


# 定义图状态
# add_messages(LangGraph专用)
# 语法：messages: Annotated[list[BaseMessage],add_messages]
# - 专门为消息设计的Reducer
# - 新旧消息合并时，相同 ID 的消息会覆盖（用新消息替换旧消息），不同 ID 的则追加
# - 支持 LangGraph 的消息管理特性，如 MessageState、消息更新等

class GraphState(BaseModel):
    # 消息列表：相同ID覆盖（用新消息替换旧消息）不同ID则追加
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list, description="会话消息列表"
    )

    # 长期记忆，策略：默认覆盖
    long_term_memory: str = Field(default="", description="会话长期记忆")
