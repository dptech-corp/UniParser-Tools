"""Poll UniParser jobs until completion."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from uniparser_mcp.defaults import PENDING_STATUSES, POLL_INTERVAL_SEC, POLL_TIMEOUT_SEC, UNDEFINED_MAX_POLLS
from uniparser_mcp.errors import parse_error, token_not_found_error
from uniparser_mcp.schemas import ErrorResult
from uniparser_tools.api.clients import UniParserClient


async def poll_until_success(client: UniParserClient, token: str) -> dict[str, Any] | ErrorResult:
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    last: dict[str, Any] = {}
    undefined_polls = 0

    while time.monotonic() < deadline:
        last = await asyncio.to_thread(
            client.get_result,
            token,
            content=False,
            objects=False,
            pages_dict=False,
            pages_tree=False,
        )
        status = last.get("status")
        if status == "success":
            return last
        if status == "error":
            return parse_error("get_result_poll", last)
        if status == "undefined":
            undefined_polls += 1
            if undefined_polls >= UNDEFINED_MAX_POLLS:
                return token_not_found_error(token, attempts=undefined_polls)
            await asyncio.sleep(POLL_INTERVAL_SEC)
            continue
        if status in PENDING_STATUSES:
            undefined_polls = 0
            await asyncio.sleep(POLL_INTERVAL_SEC)
            continue
        return parse_error("get_result_poll", last)

    return parse_error(
        "get_result_poll",
        {
            "status": "error",
            "description": f"Timed out after {POLL_TIMEOUT_SEC}s waiting for parsing to finish.",
            "token": token,
            "last_status": last.get("status"),
        },
    )
