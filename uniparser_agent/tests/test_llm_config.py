"""Unit tests for shared LLM config resolution."""

from __future__ import annotations

import pytest

from uniparser_agent.llm.config import LLMConfig, resolve_llm_config


def test_resolve_from_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.com/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    cfg = resolve_llm_config()
    assert cfg.api_key == "sk-env"
    assert cfg.base_url == "http://example.com/v1"
    assert cfg.model == "gpt-test"
    assert cfg.enable_thinking is False


def test_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    cfg = resolve_llm_config(
        api_key="sk-cli",
        base_url="http://cli/v1",
        model="cli-model",
        enable_thinking=True,
    )
    assert cfg.api_key == "sk-cli"
    assert cfg.base_url == "http://cli/v1"
    assert cfg.model == "cli-model"
    assert cfg.enable_thinking is True


def test_config_object_with_partial_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    base = LLMConfig(
        api_key="sk-base",
        base_url="http://base/v1",
        model="base-model",
    )
    cfg = resolve_llm_config(config=base, model="override-model")
    assert cfg.api_key == "sk-base"
    assert cfg.model == "override-model"


def test_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_llm_config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        resolve_llm_config()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        resolve_llm_config()


def test_legacy_env_names_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("API_KEY", "legacy-key")
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    monkeypatch.setenv("BASE_URL", "http://legacy/v1")
    monkeypatch.setenv("MODEL_NAME", "legacy-model")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_llm_config()


def test_qwen_extra_body_injected() -> None:
    cfg = LLMConfig(
        api_key="sk",
        base_url="https://api.openai.com/v1",
        model="qwen-test-model",
        enable_thinking=False,
    )
    assert cfg.resolved_extra_body() == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_non_qwen_no_extra_body() -> None:
    cfg = LLMConfig(
        api_key="sk",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )
    assert cfg.resolved_extra_body() is None


def test_enable_thinking_forces_extra_body() -> None:
    cfg = LLMConfig(
        api_key="sk",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        enable_thinking=True,
    )
    assert cfg.resolved_extra_body() == {
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_explicit_extra_body_wins() -> None:
    cfg = LLMConfig(
        api_key="sk",
        base_url="https://api.openai.com/v1",
        model="Qwen3",
        extra_body={"foo": 1},
    )
    assert cfg.resolved_extra_body() == {"foo": 1}
