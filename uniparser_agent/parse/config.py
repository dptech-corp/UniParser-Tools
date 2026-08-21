"""UniParser API configuration."""

from __future__ import annotations

import os


DEFAULT_BASE_URL = "https://uniparser.dp.tech"


def get_api_key() -> str:
    key = (os.environ.get("UNIPARSER_API_KEY") or "").strip()
    if not key:
        raise ValueError("UNIPARSER_API_KEY is not set.")
    return key


def get_base_url() -> str:
    return (os.environ.get("UNIPARSER_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
