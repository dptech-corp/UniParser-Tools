"""Unit tests for ``UniParserClient`` that do not hit the network.

The real HTTP calls are covered in ``tests/integration`` and are skipped
unless credentials are provided.
"""

from __future__ import annotations

import json

import pytest

from uniparser_tools.api import clients as clients_mod
from uniparser_tools.api.clients import UniParserClient
from uniparser_tools.common.constant import ThirdPartyFormatter


class _Response:
    status_code = 200
    text = ""
    reason = "OK"

    def json(self):
        return {"status": "success"}


class _ErrorResponse(_Response):
    status_code = 403
    text = "denied"
    reason = "Forbidden"


class TestClientConstruction:
    def test_rejects_empty_api_key(self) -> None:
        with pytest.raises(AssertionError):
            UniParserClient(host="https://example.com", api_key="")

    def test_rejects_non_http_host(self) -> None:
        with pytest.raises(AssertionError):
            UniParserClient(host="example.com", api_key="k")

    def test_endpoints_compose_correctly(self) -> None:
        c = UniParserClient(host="https://example.com/", api_key="k")
        assert c.host == "https://example.com"
        assert c.trigger_file_endpoint.endswith("/trigger-file-async")
        assert c.trigger_url_endpoint.endswith("/trigger-url-async")
        assert c.trigger_snip_endpoint.endswith("/trigger-snip-async")
        assert c.get_result_endpoint.endswith("/get-result")
        assert c.get_formatted_endpoint.endswith("/get-formatted")
        assert c.get_third_party_output_endpoint.endswith("/get-third-party-output")


class TestTokenHelpers:
    def test_to_token_is_deterministic(self) -> None:
        c = UniParserClient(host="https://e.com", api_key="secret")
        t1 = c.to_token("/abs/path/file.pdf")
        t2 = c.to_token("/abs/path/file.pdf")
        assert t1 == t2

    def test_to_token_varies_across_keys(self) -> None:
        c1 = UniParserClient(host="https://e.com", api_key="secret-A")
        c2 = UniParserClient(host="https://e.com", api_key="secret-B")
        assert c1.to_token("same.pdf") != c2.to_token("same.pdf")

    def test_validate_token_accepts_hex(self) -> None:
        c = UniParserClient(host="https://e.com", api_key="k")
        c.validate_token(c.to_token("x.pdf"))

    def test_validate_token_rejects_illegal_chars(self) -> None:
        c = UniParserClient(host="https://e.com", api_key="k")
        with pytest.raises(AssertionError):
            c.validate_token("has spaces!")

    def test_validate_token_rejects_empty(self) -> None:
        c = UniParserClient(host="https://e.com", api_key="k")
        with pytest.raises(AssertionError):
            c.validate_token("")


class TestClientErrorShapes:
    """When the underlying request raises, we expect structured error dicts."""

    def _raise_conn_err(self, *args, **kwargs):
        import requests as _requests

        raise _requests.ConnectionError("simulated")

    def test_health_returns_error_dict_on_request_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(clients_mod.requests, "get", self._raise_conn_err)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.health()
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "description" in result

    def test_version_returns_error_dict_on_request_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(clients_mod.requests, "get", self._raise_conn_err)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.version()
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "description" in result

    def test_health_uses_requests_reason_for_http_errors(self, monkeypatch) -> None:
        monkeypatch.setattr(clients_mod.requests, "get", lambda *args, **kwargs: _ErrorResponse())
        result = UniParserClient(host="https://example.com", api_key="k").health()
        assert result == {
            "status": "error",
            "http_status": 403,
            "description": "Forbidden",
            "body": "denied",
        }

    def test_trigger_file_returns_error_dict_on_request_failure(self, monkeypatch, tmp_path) -> None:
        p = tmp_path / "dummy.pdf"
        p.write_bytes(b"%PDF-1.4 tiny")
        monkeypatch.setattr(clients_mod.requests, "post", self._raise_conn_err)
        c = UniParserClient(host="https://example.com", api_key="k")
        result = c.trigger_file(file_path=str(p))
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert "token" in result


class TestClientPayloads:
    def test_trigger_file_sends_new_options(self, monkeypatch, tmp_path) -> None:
        captured = {}

        def post(*args, **kwargs):
            captured.update(kwargs)
            return _Response()

        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(clients_mod.requests, "post", post)

        client = UniParserClient(host="https://example.com", api_key="k")
        result = client.trigger_file(
            str(path),
            admin_debug=True,
            timeout=12,
            table_cls=True,
            padding_snip=False,
            inplace_update=True,
            preset_layout=[[{"type": "table"}]],
        )

        assert result["status"] == "success"
        assert captured["timeout"] == 12
        assert captured["data"]["admin_debug"] is True
        assert captured["data"]["table_cls"] is True
        assert captured["data"]["padding_snip"] is False
        assert captured["data"]["inplace_update"] is True
        assert json.loads(captured["data"]["preset_layout"]) == [[{"type": "table"}]]

    def test_result_methods_send_return_half(self, monkeypatch) -> None:
        payloads = []

        def post(*args, **kwargs):
            payloads.append(kwargs["json"])
            return _Response()

        monkeypatch.setattr(clients_mod.requests, "post", post)
        client = UniParserClient(host="https://example.com", api_key="k")

        client.get_result("token", return_half=True)
        client.get_formatted("token", return_half=True)

        assert payloads[0]["return_half"] is True
        assert payloads[1]["return_half"] is True

    def test_third_party_output_payload(self, monkeypatch) -> None:
        captured = {}

        def post(*args, **kwargs):
            captured.update(kwargs)
            return _Response()

        monkeypatch.setattr(clients_mod.requests, "post", post)
        client = UniParserClient(host="https://example.com", api_key="k")

        result = client.get_third_party_output("token")

        assert result["status"] == "success"
        assert captured["json"] == {"token": "token", "formatter": ThirdPartyFormatter.MinerU}
