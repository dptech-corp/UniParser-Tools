import asyncio
from unittest.mock import MagicMock

import pytest
from uniparser_mcp.pipeline.poll import poll_until_success


def test_undefined_token_stops_after_bounded_checks(monkeypatch: pytest.MonkeyPatch):
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("uniparser_mcp.pipeline.poll.asyncio.sleep", no_sleep)
    client = MagicMock()
    client.get_result.return_value = {"status": "undefined"}

    result = asyncio.run(poll_until_success(client, "missing-token"))

    assert result.ok is False
    assert result.error.code == "TOKEN_NOT_FOUND"
    assert result.error.token == "missing-token"
    assert client.get_result.call_count == 3


def test_pending_statuses_still_reach_success(monkeypatch: pytest.MonkeyPatch):
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("uniparser_mcp.pipeline.poll.asyncio.sleep", no_sleep)
    client = MagicMock()
    client.get_result.side_effect = [
        {"status": "waiting"},
        {"status": "processing"},
        {"status": "success"},
    ]

    result = asyncio.run(poll_until_success(client, "valid-token"))

    assert result == {"status": "success"}
    assert client.get_result.call_count == 3
