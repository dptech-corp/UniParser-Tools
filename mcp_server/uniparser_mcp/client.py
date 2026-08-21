"""UniParserClient factory."""

from __future__ import annotations

from uniparser_mcp.config import get_api_key, get_base_url
from uniparser_tools.api.clients import UniParserClient


def get_client() -> UniParserClient:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("UNIPARSER_API_KEY is not set")
    return UniParserClient(get_base_url(), api_key)
