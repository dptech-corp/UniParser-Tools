"""Environment configuration for the UniParser MCP server."""

from __future__ import annotations

import os
from pathlib import Path

from uniparser_mcp.defaults import DEFAULT_PREVIEW_CHARS, UNIPARSER_BASE_URL


def get_api_key() -> str | None:
    key = (os.environ.get("UNIPARSER_API_KEY") or "").strip()
    return key or None


def get_base_url() -> str:
    raw = (os.environ.get("UNIPARSER_BASE_URL") or "").strip().rstrip("/")
    return raw or UNIPARSER_BASE_URL


def get_output_root() -> Path:
    raw = (os.environ.get("OUTPUT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Uni-Parser-Skill").expanduser().resolve()


def get_preview_chars() -> int:
    raw = (os.environ.get("UNIPARSER_PREVIEW_CHARS") or "").strip()
    if not raw:
        return DEFAULT_PREVIEW_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_PREVIEW_CHARS
    return max(0, value)
