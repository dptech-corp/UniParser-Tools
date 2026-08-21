"""UniParser MCP server entrypoint."""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.fastmcp import FastMCP

from uniparser_mcp.instructions import SERVER_INSTRUCTIONS
from uniparser_mcp.tools.register import register_tools


mcp = FastMCP("UniParser", instructions=SERVER_INSTRUCTIONS)
register_tools(mcp)


def _resolve_mcp_transport() -> Literal["stdio", "sse", "streamable-http"]:
    import os

    raw = (os.environ.get("UNIPARSER_MCP_TRANSPORT") or "stdio").strip().lower()
    if raw in ("http", "streamable-http", "streamable_http"):
        return "streamable-http"
    if raw == "sse":
        return "sse"
    if raw == "stdio":
        return "stdio"
    raise ValueError(f"UNIPARSER_MCP_TRANSPORT invalid: {raw!r}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport=_resolve_mcp_transport())
