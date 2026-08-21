"""Release/v1.3 compatibility tests for the standalone agent HTTP client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from uniparser_agent.parse.api_client import UniParserApiClient


class _Response:
    def __init__(
        self,
        payload: dict[str, Any] | None,
        *,
        status_code: int = 200,
        reason: str = "OK",
        text: str = "",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.reason = reason
        self.text = text

    def json(self) -> dict[str, Any]:
        if self.payload is None:
            raise ValueError("not json")
        return self.payload


class _Session:
    def __init__(
        self,
        responses: list[_Response] | None = None,
        *,
        error: requests.RequestException | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_trigger_url_uses_v13_payload_and_sync_timeout() -> None:
    session = _Session([_Response({"status": "success", "token": "server-token"})])
    client = UniParserApiClient(
        "https://example.com/",
        "key",
        request_timeout=(1, 2),
        sync_request_timeout=(3, 4),
        session=session,  # type: ignore[arg-type]
    )

    client.trigger_url(
        "tos://bucket/document.pdf",
        trigger_kwargs={
            "preset_layout": [[{"type": "textual"}]],
            "model_version": "v1.3",
            "padding_snip": True,
        },
        server_generated_token=True,
    )

    payload = session.calls[0][2]["json"]
    assert payload["timeout"] == 1800
    assert payload["inplace_update"] is False
    assert payload["model_version"] == "v1.3"
    assert payload["preset_layout"] == '[[{"type": "textual"}]]'
    assert payload["token"] is None
    assert "padding_snip" not in payload
    assert session.calls[0][2]["timeout"] == (3, 4)


def test_trigger_file_uses_short_timeout_for_async_request(tmp_path: Path) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF")
    session = _Session([_Response({"status": "success", "token": "task-token"})])
    client = UniParserApiClient(
        "https://example.com",
        "key",
        request_timeout=(1, 2),
        sync_request_timeout=(3, 4),
        session=session,  # type: ignore[arg-type]
    )

    client.trigger_file(str(source), trigger_kwargs={"sync": False})

    assert session.calls[0][2]["timeout"] == (1, 2)
    assert session.calls[0][2]["data"]["padding_snip"] is True


def test_transport_normalizes_http_error_and_redacts_query_credentials() -> None:
    session = _Session(
        [
            _Response(
                {"description": ("upload denied for https://tos.example.com/upload?X-Tos-Signature=SECRET")},
                status_code=403,
                reason="Forbidden",
            )
        ]
    )
    client = UniParserApiClient(
        "https://example.com",
        "key",
        session=session,  # type: ignore[arg-type]
    )

    result = client.get_result("task-token")

    assert result["status"] == "error"
    assert result["http_status"] == 403
    assert result["token"] == "task-token"
    assert "SECRET" not in result["description"]
    assert "?<redacted>" in result["description"]


def test_transport_redacts_request_exception() -> None:
    session = _Session(
        error=requests.ConnectionError("failed for https://tos.example.com/upload?X-Tos-Signature=SECRET")
    )
    client = UniParserApiClient(
        "https://example.com",
        "key",
        session=session,  # type: ignore[arg-type]
    )

    result = client.get_result("task-token")

    assert result["status"] == "error"
    assert "SECRET" not in result["description"]
    assert result["error_type"] == "ConnectionError"
