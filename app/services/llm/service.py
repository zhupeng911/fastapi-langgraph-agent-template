"""将一次 LLM 调用封装成“总超时控制 + 单模型重试 + 多模型回退”的统一服务。

设计亮点：
1. 统一接口：提供统一的接口 ``call()``，无论默认路径还是覆盖参数调用模式，都通过 ``call()`` 进行调用.
2. 总超时控制：使用 ``asyncio.wait_for`` 进行总超时控制，默认 60s.
3. 单模型重试：使用 ``tenacity`` 进行单模型重试，默认最多重试 3 次，等待策略是指数退避.
4. 多模型回退：在单模型调用失败后，自动切换到下一个模型进行调用，最多尝试所有注册的模型.
5. 工具绑定：支持工具绑定，绑定的工具会持续存在，直到调用 ``_switch_to_next_model()`` 进行模型切换.
6. 结构化输出：支持结构化输出，通过 ``with_structured_output()`` 进行结构化输出.

整体调用链路：
call()
  └── asyncio.wait_for，总超时控制
        └── _call_with_fallback()
              └── _fallback_loop()
                    └── _invoke_with_retry()
                          └── llm.ainvoke(messages)

调用模式:默认路径 + 覆盖参数调用模式
await llm_service.call(messages)
await llm_service.call(messages, model_name="deepseek-v4-pro", temperature=0.2)
"""

import asyncio
import logging
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.services.llm.registry import LLMRegistry

T = TypeVar("T", bound=BaseModel)


class LLMService:
    # 初始化，根据配置加载默认模型.
    def __init__(self):
        self._llm: Any = None
        self._current_model_index: int = 0
        self._bound_tools: List = []
        all_names = LLMRegistry.get_all_names()

        try:
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
        except ValueError as e:
            self._llm = LLMRegistry.LLMS[0]["llm"]
            self._current_model_index = 0
            logger.info("配置的模型不存在，回退到注册表第一个模型....", model=all_names[0])

    # 核心：统一调用入口
    async def call(
            self,
            messages: LanguageModelInput,
            model_name: Optional[str] = None,
            response_format: Optional[Type[BaseModel]] = None,
            **model_kwargs: Any,
    ) -> Union[BaseMessage, BaseModel]:
        try:
            return await asyncio.wait_for(
                self._call_with_fallback(messages, model_name, response_format, model_kwargs),
                timeout=settings.LLM_TOTAL_TIMEOUT,  # 总超时控制, 默认60s
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"llm call timed out after {settings.LLM_TOTAL_TIMEOUT}s total budget")

    # 负责调用 LLM、自动重试、模型切换、工具绑定和结构化输出.
    async def _call_with_fallback(
            self,
            messages: LanguageModelInput,
            model_name: Optional[str],
            response_format: Optional[Type[BaseModel]],
            model_kwargs: dict,
    ) -> Union[BaseMessage, BaseModel]:
        """构建路径专用策略，并委托给共享回退循环.

        一次性路径（设置任意覆盖参数）：
            ``get_target`` 每次尝试都会创建新的注册表实例.
            ``advance`` 递增本地索引，不会修改 ``self._llm``.

        默认路径（没有覆盖参数）：
            ``get_target`` 返回绑定工具的 ``self._llm``.
            ``advance`` 调用 ``_switch_to_next_model``，确保工具绑定持续存在.
        """

        # 内部函数：根据索引获取目标 LLM
        def _override_target(idx: int) -> Any:
            base = LLMRegistry.get(LLMRegistry.LLMS[idx]["name"], **model_kwargs)
            return base.with_structured_output(response_format) if response_format else base

        # 内部函数：根据索引获取默认 LLM
        def _default_target(_: int) -> Any:
            return self._llm

        # 内部函数：根据索引获取默认 LLM
        def _default_advance(_: int) -> Optional[int]:
            return self._current_model_index if self._switch_to_next_model() else None

        # 获取目标 LLM
        if model_name or response_format or model_kwargs:
            all_names = LLMRegistry.get_all_names()
            if model_name and model_name not in all_names:
                raise ValueError(
                    f"model '{model_name}' not found in registry. available models: {', '.join(all_names)}"
                )
            start = all_names.index(model_name) if model_name else self._current_model_index
            total = len(LLMRegistry.LLMS)
            get_target: Callable[[int], Any] = _override_target

            def _override_advance(idx: int) -> Optional[int]:
                return (idx + 1) % total

            advance: Callable[[int], Optional[int]] = _override_advance
        else:
            start = self._current_model_index
            get_target = _default_target
            advance = _default_advance

        return await self._fallback_loop(messages, start, get_target, advance)

    # 模型循环切换
    def _switch_to_next_model(self) -> bool:
        try:
            next_index = (self._current_model_index + 1) % len(LLMRegistry.LLMS)
            next_model = LLMRegistry.get_model_at_index(next_index)
            self._current_model_index = next_index
            self._llm = next_model["llm"]
            if self._bound_tools:
                self._llm = self._llm.bind_tools(self._bound_tools)
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
            return False

    def bind_tools(self, tools: List) -> "LLMService":
        if self._llm:
            self._bound_tools = tools
            self._llm = self._llm.bind_tools(tools)
        return self

    def get_llm(self) -> Any:
        """返回当前实际使用的 LLM 实例。"""
        return self._llm

    # 多模型循环回退
    # 回退顺序：deepseek-flash -> deepseek-v4-pro -> RuntimeError
    # 回退顺序由 LLMRegistry.LLMS 的顺序决定，最多尝试一轮所有模型，避免无限循环
    async def _fallback_loop(
            self,
            messages: LanguageModelInput,
            start: int,
            get_target: Callable[[int], Any],
            advance: Callable[[int], Optional[int]],
    ) -> Any:
        total = len(LLMRegistry.LLMS)
        current = start
        models_tried = 0
        last_error: Optional[Exception] = None

        for models_tried in range(1, total + 1):
            try:
                return await self._invoke_with_retry(get_target(current), messages)
            except OpenAIError as e:
                last_error = e
                if models_tried >= total:
                    break
                next_idx = advance(current)
                if next_idx is None:
                    break
                current = next_idx

        raise RuntimeError(
            f"failed to get response from llm after trying {models_tried} models. last error: {str(last_error)}"
        )

    # 单模型自动重试
    # 重试策略：最多重试 3 次，等待策略是指数退避：每次重试间隔约 2s -> 4s -> 8s
    # 重试条件：RateLimitError, APITimeoutError, APIError，其他错误不会重试，直接模型回退.
    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _invoke_with_retry(self, llm: Any, messages: LanguageModelInput) -> Any:
        try:
            response = await llm.ainvoke(messages)
            return response
        except (RateLimitError, APITimeoutError, APIError) as e:
            raise
        except OpenAIError as e:
            raise


# 全局单例，整个应用通常共享同一个 LLMService
llm_service = LLMService()
