"""Lightweight release/v1.3 HTTP transport for the standalone agent package."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests


RequestTimeout = float | tuple[float, float | None]

DEFAULT_REQUEST_TIMEOUT: RequestTimeout = (10.0, 60.0)
DEFAULT_SYNC_REQUEST_TIMEOUT: RequestTimeout = (10.0, 1860.0)


def _redact_url_queries(value: str) -> str:
    """Remove bearer-style query strings from URLs included in diagnostics."""
    return re.sub(r"(?P<url>(?:https?://|/)[^\s?]+)\?[^\s]+", r"\g<url>?<redacted>", value)


def _redact_diagnostic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_diagnostic_value(item) for item in value]
    if isinstance(value, str):
        return _redact_url_queries(value)
    return value


class UniParserHTTPTransport:
    """Authenticated, reusable HTTP transport aligned with the main v1.3 client."""

    def __init__(
        self,
        host: str,
        api_key: str,
        *,
        request_timeout: RequestTimeout = DEFAULT_REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("host must be a valid http or https URL")
        if not api_key:
            raise ValueError("api_key can not be empty")

        self.host = host.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.session = session or requests.Session()
        self._owns_session = session is None

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout: RequestTimeout | None = None,
        error_message: str = "request failed",
        token: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("X-API-Key", self.api_key)
        url = path if path.startswith(("http://", "https://")) else f"{self.host}/{path.lstrip('/')}"

        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=self.request_timeout if timeout is None else timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            result: dict[str, Any] = {
                "status": "error",
                "message": error_message,
                "description": _redact_url_queries(str(exc)),
                "error_type": type(exc).__name__,
            }
            if token is not None:
                result["token"] = token
            return result

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code >= 400:
            if isinstance(payload, dict):
                result = _redact_diagnostic_value(payload)
                result.setdefault("status", "error")
                result.setdefault("description", response.reason or error_message)
            else:
                result = {
                    "status": "error",
                    "description": response.reason or error_message,
                    "body": _redact_url_queries(response.text),
                }
            result["http_status"] = response.status_code
            if token is not None:
                result.setdefault("token", token)
            return result

        if isinstance(payload, dict):
            return payload
        return {
            "status": "error",
            "message": error_message,
            "description": "response body is not valid JSON",
            "body": _redact_url_queries(response.text),
            **({"token": token} if token is not None else {}),
        }

    def close(self) -> None:
        if self._owns_session:
            self.session.close()
