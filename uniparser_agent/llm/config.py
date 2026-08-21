"""Shared LLM configuration (OpenAI-compatible, no hardcoded defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible LLM settings.

    Required fields have no library defaults: set via constructor, CLI, or
    ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``.
    """

    api_key: str
    base_url: str
    model: str
    timeout: float = 3600.0
    max_tokens: int = 81920
    enable_thinking: bool = False
    extra_body: dict[str, Any] | None = None

    def resolved_extra_body(self) -> dict[str, Any] | None:
        """Return request ``extra_body``, applying Qwen thinking kwargs when needed."""
        if self.extra_body is not None:
            return self.extra_body
        if self.enable_thinking or "qwen" in self.model.lower():
            return {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }
        return None

    def meta(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "extra_body": self.resolved_extra_body(),
        }


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def resolve_llm_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
    extra_body: dict[str, Any] | None = None,
    config: LLMConfig | None = None,
) -> LLMConfig:
    """Resolve LLM settings: explicit args / ``config`` override ``OPENAI_*`` env.

    Raises:
        ValueError: if api_key, base_url, or model is missing after resolution.
    """
    if config is not None:
        overrides: dict[str, Any] = {}
        if api_key is not None:
            overrides["api_key"] = api_key
        if base_url is not None:
            overrides["base_url"] = base_url
        if model is not None:
            overrides["model"] = model
        if timeout is not None:
            overrides["timeout"] = timeout
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        if enable_thinking is not None:
            overrides["enable_thinking"] = enable_thinking
        if extra_body is not None:
            overrides["extra_body"] = extra_body
        resolved = replace(config, **overrides) if overrides else config
    else:
        resolved = LLMConfig(
            api_key=(api_key if api_key is not None else _env("OPENAI_API_KEY")),
            base_url=(base_url if base_url is not None else _env("OPENAI_BASE_URL")).rstrip("/"),
            model=(model if model is not None else _env("OPENAI_MODEL")),
            timeout=3600.0 if timeout is None else timeout,
            max_tokens=81920 if max_tokens is None else max_tokens,
            enable_thinking=False if enable_thinking is None else enable_thinking,
            extra_body=extra_body,
        )

    missing: list[str] = []
    if not resolved.api_key:
        missing.append("OPENAI_API_KEY (or --api-key / LLMConfig.api_key)")
    if not resolved.base_url:
        missing.append("OPENAI_BASE_URL (or --base-url / LLMConfig.base_url)")
    if not resolved.model:
        missing.append("OPENAI_MODEL (or --model / LLMConfig.model)")
    if missing:
        raise ValueError(
            "Missing required LLM config: "
            + "; ".join(missing)
            + ". Set environment variables or pass them explicitly."
        )

    return replace(resolved, base_url=resolved.base_url.rstrip("/"))
