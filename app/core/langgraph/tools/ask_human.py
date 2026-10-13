"""LangGraph 的人在回路确认工具.
功能：在执行敏感操作前暂停图执行并请求用户确认，例如删除数据、发送邮件、购买商品或执行任何不可逆操作.
"""

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_human(question: str) -> str:
    user_response = interrupt(question)
    return str(user_response)
