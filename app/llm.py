"""Thin LLM interface so the pipeline never depends on a specific provider (and tests can use a fake).

The provider and model come from configuration (LLM_PROVIDER / LLM_MODEL in .env); nothing here is model-specific
except small compatibility handling for Groq's reasoning models.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.config import Settings

SUPPORTED_PROVIDERS = ("groq",)

# Reasoning models spend completion tokens on hidden thinking before the answer; give them room on top of the
# caller's max_tokens so the visible answer is not cut off (you are only billed for tokens actually generated).
REASONING_HEADROOM = 2048
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLM(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str):
        import groq

        self._groq = groq
        self.client = groq.Groq(api_key=api_key, timeout=45.0, max_retries=2)
        self.model = model

    @property
    def _is_gpt_oss(self) -> bool:
        return self.model.startswith("openai/gpt-oss")

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
        )
        optional: dict = {"temperature": temperature}
        if self._is_gpt_oss:
            kwargs["max_completion_tokens"] = max_tokens + REASONING_HEADROOM
            optional.update(reasoning_effort="low", include_reasoning=False)
        try:
            resp = self.client.chat.completions.create(**kwargs, **optional)
        except self._groq.BadRequestError as e:
            # A model may reject temperature / reasoning options; retry once with the plain request.
            if not any(word in str(e).lower() for word in ("temperature", "reasoning")):
                raise
            resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        # Some reasoning models (e.g. Qwen) inline their thinking; the pipeline must only see the answer.
        return _THINK_BLOCK.sub("", text).strip()


def build_llm(settings: "Settings") -> LLM | None:
    """Create the LLM named by LLM_PROVIDER / LLM_MODEL. Returns None when no API key is configured."""
    provider = settings.llm_provider.strip().lower()
    if provider == "groq":
        return GroqLLM(settings.groq_api_key, settings.llm_model) if settings.groq_api_key else None
    raise ValueError(f"Unsupported LLM_PROVIDER {settings.llm_provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}")