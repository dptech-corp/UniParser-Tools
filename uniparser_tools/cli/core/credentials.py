from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import yaml
from click.core import ParameterSource


CONFIG_DIR = Path.home() / ".uniparser"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
API_KEY_URL = "https://uniparser.dp.tech/"


@dataclass(frozen=True)
class ApiKeySource:
    api_key: str | None
    source: str  # "flag", "env", "config", ""


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_api_key_from_config() -> str | None:
    key = load_config().get("api_key")
    if key is None:
        return None
    text = str(key).strip()
    return text or None


def _root_context(ctx: click.Context) -> click.Context:
    while ctx.parent is not None:
        ctx = ctx.parent
    return ctx


def resolve_api_key_source(ctx: click.Context | None = None) -> ApiKeySource:
    if ctx is not None:
        root = _root_context(ctx)
        api_key = ((root.obj or {}).get("api_key") or "").strip()
        if api_key:
            try:
                param_source = root.get_parameter_source("api_key")
            except LookupError:
                param_source = None
            if param_source == ParameterSource.COMMANDLINE:
                return ApiKeySource(api_key=api_key, source="flag")
            if param_source == ParameterSource.ENVIRONMENT:
                return ApiKeySource(api_key=api_key, source="env")
            config_key = read_api_key_from_config()
            if config_key and api_key == config_key:
                return ApiKeySource(api_key=api_key, source="config")
            env_key = (os.getenv("UNIPARSER_API_KEY") or "").strip()
            if env_key and api_key == env_key:
                return ApiKeySource(api_key=api_key, source="env")
            return ApiKeySource(api_key=api_key, source="flag")

    env_key = (os.getenv("UNIPARSER_API_KEY") or "").strip()
    if env_key:
        return ApiKeySource(api_key=env_key, source="env")
    config_key = read_api_key_from_config()
    if config_key:
        return ApiKeySource(api_key=config_key, source="config")
    return ApiKeySource(api_key=None, source="")


def save_api_key(api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config["api_key"] = api_key.strip()
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


def mask_api_key(api_key: str) -> str:
    text = api_key.strip()
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"
