"""VQA LLM client (OpenAI-compatible via shared llm module)."""

from __future__ import annotations

from typing import Any

from uniparser_agent.llm import LLMConfig, OpenAICompatLLM, resolve_llm_config


class VQALLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool = False,
        extra_body: dict[str, Any] | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        self._llm = OpenAICompatLLM(
            config=config,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

    @property
    def api_key(self) -> str:
        return self._llm.api_key

    @property
    def base_url(self) -> str:
        return self._llm.base_url

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def timeout(self) -> float:
        return self._llm.timeout

    @property
    def max_tokens(self) -> int:
        return self._llm.max_tokens

    @property
    def enable_thinking(self) -> bool:
        return self._llm.enable_thinking

    def chat(self, *, system_prompt: str, user_content: str) -> str:
        return self._llm.chat(system_prompt=system_prompt, user_content=user_content)

    def meta(self) -> dict[str, Any]:
        return self._llm.meta()


__all__ = ["VQALLMClient", "resolve_llm_config", "LLMConfig"]
