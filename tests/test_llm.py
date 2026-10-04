"""Exercises GroqLLM through the real Groq SDK with a mocked HTTP transport (no network, no API key)."""
import json

import groq
import httpx
import pytest

from app import config
from app.config import Settings
from app.llm import REASONING_HEADROOM, GroqLLM, build_llm


def _llm(handler, model="llama-3.3-70b-versatile"):
    llm = GroqLLM("test-key", model)
    llm.client = groq.Groq(api_key="test-key", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return llm


def _ok(text):
    return httpx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def _bad_request(message):
    return httpx.Response(400, json={"error": {"message": message, "type": "invalid_request_error"}})


def test_request_shape_and_text_extraction():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok("<sql>SELECT 1</sql>")

    out = _llm(handler).complete("sys prompt", "user prompt", temperature=0.0, max_tokens=123)
    assert out == "<sql>SELECT 1</sql>"
    assert seen["model"] == "llama-3.3-70b-versatile"          # the model comes from config, not from the client
    assert seen["messages"] == [{"role": "system", "content": "sys prompt"}, {"role": "user", "content": "user prompt"}]
    assert seen["max_completion_tokens"] == 123 and seen["temperature"] == 0.0
    assert "reasoning_effort" not in seen


def test_gpt_oss_gets_low_reasoning_effort_and_token_headroom():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _ok("ok")

    _llm(handler, model="openai/gpt-oss-120b").complete("s", "u", max_tokens=1000)
    assert seen["model"] == "openai/gpt-oss-120b"
    assert seen["reasoning_effort"] == "low" and seen["include_reasoning"] is False
    assert seen["max_completion_tokens"] == 1000 + REASONING_HEADROOM


def test_retries_without_optional_params_if_model_rejects_them():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return _bad_request("`temperature` is not supported for this model")
        return _ok("fine")

    assert _llm(handler).complete("s", "u") == "fine"
    assert len(bodies) == 2 and "temperature" not in bodies[1]


def test_other_400_errors_are_not_swallowed():
    def handler(request):
        return _bad_request("model does not exist")

    with pytest.raises(groq.BadRequestError):
        _llm(handler).complete("s", "u")


def test_inline_think_blocks_are_stripped():
    def handler(request):
        return _ok("<think>maybe <sql>DROP TABLE x</sql></think>\n<sql>SELECT 1</sql>")

    assert _llm(handler, model="qwen/qwen3-32b").complete("s", "u") == "<sql>SELECT 1</sql>"


def test_empty_content_returns_empty_string():
    def handler(request):
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": None}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    assert _llm(handler).complete("s", "u") == ""


def test_build_llm_uses_provider_and_model_from_settings():
    llm = build_llm(Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", groq_api_key="k"))
    assert isinstance(llm, GroqLLM) and llm.model == "openai/gpt-oss-20b"
    assert build_llm(Settings(llm_provider="groq", groq_api_key=None)) is None


def test_build_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        build_llm(Settings(llm_provider="nope", groq_api_key="k"))


def test_settings_read_provider_and_model_from_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)   # ignore a developer's real .env
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_model, s.groq_api_key) == ("groq", "llama-3.1-8b-instant", "gsk_test")
    monkeypatch.delenv("LLM_MODEL")
    assert Settings.from_env().llm_model == "openai/gpt-oss-120b"       # documented default