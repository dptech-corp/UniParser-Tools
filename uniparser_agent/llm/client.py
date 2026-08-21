"""OpenAI-compatible chat client."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from uniparser_agent.llm.config import LLMConfig, resolve_llm_config


class OpenAICompatLLM:
    """Thin wrapper around ``openai.OpenAI`` chat completions."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.config = resolve_llm_config(
            config=config,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    @property
    def api_key(self) -> str:
        return self.config.api_key

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def max_tokens(self) -> int:
        return self.config.max_tokens

    @property
    def enable_thinking(self) -> bool:
        return self.config.enable_thinking

    def chat(self, *, system_prompt: str, user_content: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.config.max_tokens,
        }
        extra = self.config.resolved_extra_body()
        if extra is not None:
            kwargs["extra_body"] = extra
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content")
        return content

    def meta(self) -> dict[str, Any]:
        return self.config.meta()
