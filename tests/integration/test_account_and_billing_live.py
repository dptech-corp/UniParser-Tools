"""Live integration tests for the account endpoints (``/users/me`` and
``/api-keys``) plus the auth error paths.

Skipped automatically unless ``UNIPARSER_TEST_API_KEY`` and
``UNIPARSER_TEST_HOST`` are set (same gate as ``test_client_live.py``).

These tests are intentionally tolerant of the credential's account type:
the test API key may belong to a legacy local-balance user OR a
Bohrium-bound user; assertions check for "structurally valid response"
shapes that hold in either case.
"""

from __future__ import annotations

import pytest


@pytest.mark.live
class TestAccountEndpointsLive:
    """``/users/me`` and ``/api-keys`` — basic account-management surface
    used by the demo home page's account panel."""

    def test_me_returns_user_record(self, live_client) -> None:
        result = live_client.me()
        assert isinstance(result, dict), result
        # Server returns the User row. Don't assert on every field — the
        # response_model can grow. Anchor on stable identifiers.
        # On failure, dump the nested body explicitly so a ResponseValidationError
        # detail (often truncated in pytest's default repr) is fully visible.
        assert "id" in result and result["id"], result.get("body", result)
        assert "username" in result and result["username"], result
        # Neither shape from our wrapper's error path should be present.
        assert "http_status" not in result, result
        # /users/me with a valid key should never produce an error dict;
        # if it does, something is wrong server-side and we want to know.
        assert result.get("status") != "error", result

    def test_list_api_keys_returns_list(self, live_client) -> None:
        result = live_client.list_api_keys()
        # The test API key was issued to the test user — they should have
        # at least one API key (the one we're authenticating with).
        assert isinstance(result, list), result
        assert len(result) >= 1, result
        # Each entry is sanitised — no raw secret.
        for item in result:
            assert "id" in item
            assert "key_identifier" in item
            # masked_key is optional (may be None for very-short legacy keys
            # or when decrypt fails); the secret-bearing 'raw_key' MUST not
            # be present on the list endpoint.
            assert "raw_key" not in item, item


@pytest.mark.live
class TestErrorPathsLive:
    """End-to-end auth + error-path checks that touch the new middleware."""

    def test_bad_api_key_returns_401(self, api_host: str | None) -> None:
        """The auth middleware's fingerprint-miss fallback (C-B3 fix) should
        still 401 cleanly on a totally-unknown key. We never let a malformed
        key trigger a 5xx."""
        if not api_host:
            pytest.skip("UNIPARSER_TEST_HOST not set")
        from uniparser_tools.api.clients import UniParserClient

        bad_client = UniParserClient(host=api_host, api_key="up_definitely_invalid_key_xxx")
        result = bad_client.me()
        # Wrapper surfaces 401 as an error dict.
        assert isinstance(result, dict), result
        assert result.get("status") == "error", result
        assert result.get("http_status") in (401, 403), result
