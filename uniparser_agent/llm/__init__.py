"""Shared OpenAI-compatible LLM helpers for uniparser_agent."""

from uniparser_agent.llm.client import OpenAICompatLLM
from uniparser_agent.llm.config import LLMConfig, resolve_llm_config


__all__ = [
    "LLMConfig",
    "OpenAICompatLLM",
    "resolve_llm_config",
]
