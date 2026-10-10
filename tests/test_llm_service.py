"""LLMService 的离线单元测试。"""

import asyncio
from typing import Any

import pytest
from openai import APIError, OpenAIError
from pydantic import BaseModel
from tenacity import stop_after_attempt, wait_none

import app.services.llm.service as service_module
from app.services.llm.service import LLMService


class FakeStructuredLLM:
    """模拟 LangChain 的结构化输出包装器。"""

    def __init__(self, base: "FakeLLM", schema: type[BaseModel]):
        self.base = base
        self.schema = schema

    async def ainvoke(self, messages: Any) -> BaseModel:
        self.base.structured_calls.append((messages, self.schema))
        await self.base.ainvoke(messages)
        return self.schema(answer="structured-response")


class FakeLLM:
    """按预设队列返回结果或抛出异常的异步模型桩。"""

    def __init__(self, name: str, responses: list[Any] | None = None):
        self.name = name
        self.responses = list(responses or [])
        self.calls: list[Any] = []
        self.bound_tools: list[Any] | None = None
        self.structured_calls: list[tuple[Any, type[BaseModel]]] = []
        self.override_kwargs: list[dict[str, Any]] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        response = self.responses.pop(0) if self.responses else f"{self.name}-response"
        if isinstance(response, BaseException):
            raise response
        return response

    def bind_tools(self, tools: list[Any]) -> "FakeLLM":
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema: type[BaseModel]) -> FakeStructuredLLM:
        return FakeStructuredLLM(self, schema)


def run(coroutine: Any) -> Any:
    """在不依赖 pytest-asyncio 的情况下执行异步测试。"""
    return asyncio.run(coroutine)


def api_error(message: str = "temporary failure") -> APIError:
    """创建不依赖真实 HTTP 请求的 APIError。"""
    return APIError(message, request=object(), body=None)


def configure_registry(monkeypatch: pytest.MonkeyPatch, default_model: str, models: list[FakeLLM]) -> None:
    """用测试模型替换注册表，并保留注册表的索引逻辑。"""
    entries = [{"name": model.name, "llm": model} for model in models]
    model_by_name = {model.name: model for model in models}

    monkeypatch.setattr(service_module.LLMRegistry, "LLMS", entries)
    monkeypatch.setattr(service_module.settings, "DEFAULT_LLM_MODEL", default_model)

    def fake_get(model_name: str, **kwargs: Any) -> FakeLLM:
        if model_name not in model_by_name:
            available = ", ".join(model_by_name)
            raise ValueError(f"model '{model_name}' not found in registry. available models: {available}")
        model = model_by_name[model_name]
        if kwargs:
            model.override_kwargs.append(kwargs)
        return model

    monkeypatch.setattr(service_module.LLMRegistry, "get", fake_get)

# 测试初始化逻辑
def test_init_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a")
    model_b = FakeLLM("model-b")
    configure_registry(monkeypatch, "model-b", [model_a, model_b])

    service = LLMService()

    assert service._llm is model_b
    assert service._current_model_index == 1


# 测试调用逻辑
def test_call_uses_default_model_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a", responses=["answer"])
    configure_registry(monkeypatch, "model-a", [model_a])
    service = LLMService()
    messages = [("human", "hello")]

    result = run(service.call(messages))

    assert result == "answer"
    assert model_a.calls == [messages]


def test_call_with_model_name_uses_override_model_without_mutating_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_a = FakeLLM("model-a")
    model_b = FakeLLM("model-b", responses=["override-answer"])
    configure_registry(monkeypatch, "model-a", [model_a, model_b])
    service = LLMService()
    messages = [("human", "hello")]

    result = run(service.call(messages, model_name="model-b", temperature=0.1))

    assert result == "override-answer"
    assert model_a.calls == []
    assert model_b.calls == [messages]
    assert model_b.override_kwargs == [{"temperature": 0.1}]
    assert service._llm is model_a
    assert service._current_model_index == 0


class AnswerSchema(BaseModel):
    answer: str


def test_call_with_response_format_returns_structured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a", responses=["raw-response"])
    configure_registry(monkeypatch, "model-a", [model_a])
    service = LLMService()

    result = run(service.call([("human", "hello")], response_format=AnswerSchema))

    assert isinstance(result, AnswerSchema)
    assert result.answer == "structured-response"
    assert model_a.structured_calls[0][1] is AnswerSchema


def test_bind_tools_rebinds_tools_after_default_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_a = FakeLLM("model-a", responses=[OpenAIError("model-a unavailable")])
    model_b = FakeLLM("model-b", responses=["fallback-answer"])
    configure_registry(monkeypatch, "model-a", [model_a, model_b])
    service = LLMService()
    tools = [object()]

    service.bind_tools(tools)
    result = run(service.call([("human", "hello")]))

    assert result == "fallback-answer"
    assert model_a.bound_tools is tools
    assert model_b.bound_tools is tools
    assert service._llm is model_b
    assert service._current_model_index == 1


def test_retry_repeats_api_error_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a", responses=[api_error(), api_error(), "retried-answer"])
    configure_registry(monkeypatch, "model-a", [model_a])
    service = LLMService()

    retrying_call = service._invoke_with_retry.retry_with(
        stop=stop_after_attempt(3),
        wait=wait_none(),
    )
    result = run(retrying_call(model_a, [("human", "hello")]))

    assert result == "retried-answer"
    assert len(model_a.calls) == 3


def test_non_retryable_openai_errors_fallback_then_raise_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_a = FakeLLM("model-a", responses=[OpenAIError("model-a failed")])
    model_b = FakeLLM("model-b", responses=[OpenAIError("model-b failed")])
    configure_registry(monkeypatch, "model-a", [model_a, model_b])
    service = LLMService()

    with pytest.raises(RuntimeError, match="after trying 2 models"):
        run(service.call([("human", "hello")]))

    assert len(model_a.calls) == 1
    assert len(model_b.calls) == 1


def test_call_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a")
    configure_registry(monkeypatch, "model-a", [model_a])
    service = LLMService()

    with pytest.raises(ValueError, match="model 'missing-model' not found"):
        run(service.call([("human", "hello")], model_name="missing-model"))


def test_call_translates_total_timeout_to_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    model_a = FakeLLM("model-a")
    configure_registry(monkeypatch, "model-a", [model_a])
    service = LLMService()
    monkeypatch.setattr(service_module.settings, "LLM_TOTAL_TIMEOUT", 7)

    async def fake_wait_for(awaitable: Any, timeout: int) -> Any:
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(service_module.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(RuntimeError, match="timed out after 7s"):
        run(service.call([("human", "hello")]))
