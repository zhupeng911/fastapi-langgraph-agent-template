"""预初始化 LLM 实例的模型注册表.

只负责“模型登记和获取”，不负责真正调用、重试或回退.
"""

from typing import (
    Any,
    Dict,
    List,
    cast,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings

_API_KEY = SecretStr(settings.OPENAI_API_KEY)


class LLMRegistry:
    """包含预初始化实例的可用 LLM 模型注册表.

    该类维护 LLM 配置列表，并提供按名称获取模型及覆盖可选参数的方法.
    """

    # 索引 0 是默认模型，也是循环回退链的起点。
    LLMS: List[Dict[str, Any]] = [
        {
            "name": "deepseek-flash",
            "llm": ChatOpenAI(
                model="deepseek-flash",
                api_key=_API_KEY,
                max_completion_tokens=settings.MAX_TOKENS,
            ),
        },
        {
            "name": "deepseek-v4-pro",
            "llm": ChatOpenAI(
                model="deepseek-v4-pro",
                api_key=_API_KEY,
                max_completion_tokens=settings.MAX_TOKENS,
            ),
        },
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """按名称获取 LLM，并支持覆盖可选参数.

        提供 kwargs 时会返回应用这些覆盖参数的新 ChatOpenAI 实例，不修改注册表中的共享实例.

        参数：
            model_name: 要获取的模型名称.
            **kwargs: 用于覆盖默认模型配置的可选参数.

        返回：
            BaseChatModel 实例.

        异常：
            ValueError: model_name 不存在于 LLMS 时抛出.
        """
        # 按名称获取模型注册项
        model = next((e for e in cls.LLMS if e["name"] == model_name), None)

        # 如果模型注册项不存在，则抛出 ValueError
        if not model:
            available_model = ", ".join(e["name"] for e in cls.LLMS)
            raise ValueError(f"model '{model_name}' not found in registry. 可用模型列表: {available_model}")

        # 如果提供了 kwargs，则使用注册项中的模型 ID 创建一个新实例
        if kwargs:
            base_llm = cast(ChatOpenAI, model["llm"])
            return ChatOpenAI(
                model=base_llm.model_name,
                api_key=_API_KEY,
                max_completion_tokens=settings.MAX_TOKENS,
                **kwargs,
            )
        return model["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """按注册顺序返回全部模型名称.

        返回：
            List[str]: 模型名称列表.
        """
        return [e["name"] for e in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """返回指定索引处的模型注册项，索引越界时回到索引 0.

        参数：
            index: LLMS 中的索引.

        返回：
            模型注册项字典.
        """
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]
