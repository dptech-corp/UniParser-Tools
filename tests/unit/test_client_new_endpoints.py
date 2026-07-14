"""Unit tests for the account endpoint methods (``me`` / ``list_api_keys``)
on ``UniParserClient``.

These cover the wrapper's contract — endpoint composition, request shape,
error-dict synthesis on transport / HTTP failure — without hitting the
network. Live behaviour is in ``tests/integration/test_client_live.py``.

Pattern mirrors the existing ``tests/unit/test_client.py`` so the suite stays
internally consistent: each method gets a transport-failure case (synthesised
``ConnectionError`` via ``monkeypatch``) and a happy-path case where we stub
out ``requests.get`` to return a tiny ``DummyResponse``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from uniparser_tools.api import clients as clients_mod
from uniparser_tools.api.clients import UniParserClient


def _raise_conn_err(*args, **kwargs):
    import requests as _requests

    raise _requests.ConnectionError("simulated")


class DummyResponse:
    """Minimal stand-in for ``requests.Response`` covering what _get_json reads."""

    def __init__(
        self,
        json_body: Any = None,
        status_code: int = 200,
        text: str = "",
        reason: str = "OK",
        raw_text: Optional[str] = None,
    ):
        self._json_body = json_body
        self.status_code = status_code
        self.reason = reason
        # Mirror ``requests.Response.text`` semantics: prefer raw_text if
        # given, otherwise fall back to a serialised dump of json_body, then
        # to the supplied ``text``.
        if raw_text is not None:
            self.text = raw_text
        elif json_body is not None:
            self.text = json.dumps(json_body)
        else:
            self.text = text

    def json(self):
        if self._json_body is None:
            # Trigger the same JSONDecodeError path that requests would.
            raise json.decoder.JSONDecodeError("no body", "", 0)
        return self._json_body


# --------------------------------------------------------------------------- #
# Endpoint composition
# --------------------------------------------------------------------------- #


class TestNewEndpointComposition:
    """The wrapper appends the documented path; trailing slashes don't leak."""

    def test_me_endpoint(self):
        c = UniParserClient(host="https://example.com", api_key="k")
        assert c.me_endpoint == "https://example.com/users/me"

    def test_api_keys_endpoint(self):
        c = UniParserClient(host="https://example.com", api_key="k")
        assert c.api_keys_endpoint == "https://example.com/api-keys"


# --------------------------------------------------------------------------- #
# Transport-failure → structured error dict
# --------------------------------------------------------------------------- #


class TestNewEndpointTransportFailure:
    """On connection failure every wrapper returns ``{"status": "error",
    "description": ...}`` instead of raising — matches the existing methods'
    contract so callers don't need a try/except per call."""

    def test_me_returns_error_dict(self, monkeypatch):
        monkeypatch.setattr(clients_mod.requests, "get", _raise_conn_err)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.me()
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "description" in result

    def test_list_api_keys_returns_error_dict(self, monkeypatch):
        monkeypatch.setattr(clients_mod.requests, "get", _raise_conn_err)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.list_api_keys()
        assert isinstance(result, dict)
        assert result.get("status") == "error"


# --------------------------------------------------------------------------- #
# HTTP-error response → structured error dict (surfaces server detail)
# --------------------------------------------------------------------------- #


class TestNewEndpointHTTPErrorPropagation:
    def test_401_surfaces_http_status_and_body(self, monkeypatch):
        """401 with a FastAPI ``{"detail": "..."}`` body — wrapper preserves
        the JSON body so the caller can show "认证失败" to the user."""

        def _401(*args, **kwargs):
            return DummyResponse(
                json_body={"detail": "认证失败：未找到有效的用户"},
                status_code=401,
                reason="Unauthorized",
            )

        monkeypatch.setattr(clients_mod.requests, "get", _401)
        c = UniParserClient(host="https://example.com", api_key="bad-key")
        result = c.me()
        assert result["status"] == "error"
        assert result["http_status"] == 401
        assert result["body"] == {"detail": "认证失败：未找到有效的用户"}

    def test_402_payment_required_json_detail(self, monkeypatch):
        """402 with a FastAPI ``{"detail": "..."}`` body — the wrapper
        preserves the JSON body so the caller can surface the server reason."""

        def _402(*args, **kwargs):
            return DummyResponse(
                json_body={"detail": "billing_blocked: 您有未结清的扣费记录，请充值后重试"},
                status_code=402,
                reason="Payment Required",
            )

        monkeypatch.setattr(clients_mod.requests, "get", _402)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.me()
        assert result["status"] == "error"
        assert result["http_status"] == 402
        assert "billing_blocked" in result["body"]["detail"]

    def test_5xx_with_non_json_body_falls_back_to_text(self, monkeypatch):
        """A 503 from an upstream proxy may not be JSON; we still produce
        a structured error dict and surface the text in ``body``."""

        def _503(*args, **kwargs):
            return DummyResponse(
                status_code=503,
                reason="Service Unavailable",
                raw_text="<html><body>502 Bad Gateway</body></html>",
            )

        monkeypatch.setattr(clients_mod.requests, "get", _503)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.me()
        assert result["status"] == "error"
        assert result["http_status"] == 503
        assert "Bad Gateway" in result["body"]


# --------------------------------------------------------------------------- #
# Happy-path body parsing + parameter wiring
# --------------------------------------------------------------------------- #


class TestNewEndpointSuccess:
    def test_me_returns_body_verbatim(self, monkeypatch):
        body = {
            "id": "11111111-1111-1111-1111-111111111111",
            "username": "alice",
            "email": "alice@example.com",
            "is_active": True,
            "balance": "5.00",
        }
        monkeypatch.setattr(clients_mod.requests, "get", lambda *a, **kw: DummyResponse(json_body=body))
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.me()
        assert result == body

    def test_list_api_keys_returns_list_verbatim(self, monkeypatch):
        body: List[Dict] = [
            {"id": "a", "key_identifier": "api_key_alice", "permissions": [], "status": "active"},
            {"id": "b", "key_identifier": "bohrium_key_alice", "permissions": [], "status": "active"},
        ]
        monkeypatch.setattr(clients_mod.requests, "get", lambda *a, **kw: DummyResponse(json_body=body))
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.list_api_keys()
        assert result == body
        # And it really is a list (not coerced to a dict), so caller can iterate.
        assert isinstance(result, list)


class TestNewEndpointParameterWiring:
    """Capture the URL+params the wrapper sends — protects against a future
    refactor that drops a header or sends the wrong type."""

    def _capture(self, captured: dict):
        def _get(url, headers=None, params=None, timeout=None, **kw):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            captured["timeout"] = timeout
            return DummyResponse(json_body={})

        return _get

    def test_all_account_methods_send_api_key_header(self, monkeypatch):
        """X-API-Key must be on every account call — server gates them
        with the same auth middleware as the parsing endpoints."""
        captured: dict = {}
        monkeypatch.setattr(clients_mod.requests, "get", self._capture(captured))
        c = UniParserClient(host="https://example.com", api_key="my-secret")
        for fn in (c.me, c.list_api_keys):
            captured.clear()
            fn()
            assert captured["headers"] == {"X-API-Key": "my-secret"}, fn.__name__


# --------------------------------------------------------------------------- #
# Body-shape edge cases
# --------------------------------------------------------------------------- #


class TestNewEndpointBodyShape:
    def test_2xx_with_non_json_body_returns_message_dict(self, monkeypatch):
        """A misconfigured edge / upstream proxy could return 200 + HTML.
        Wrapper should not raise; should produce a structured fallback."""
        monkeypatch.setattr(
            clients_mod.requests,
            "get",
            lambda *a, **kw: DummyResponse(status_code=200, raw_text="not json"),
        )
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.me()
        assert isinstance(result, dict)
        # The fallback path sets ``status=error`` and surfaces ``message``.
        assert result.get("status") == "error"
        assert result.get("message") == "not json"
