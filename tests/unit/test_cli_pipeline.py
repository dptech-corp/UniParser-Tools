from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from uniparser_tools.cli.core.input import InputKind, ResolvedInput
from uniparser_tools.cli.core.pipeline import poll_until_success, trigger_input


def _local_pdf(path: Path) -> ResolvedInput:
    return ResolvedInput(
        kind=InputKind.FILE,
        source_stem=path.stem,
        raw=str(path),
        path=path,
    )


def test_local_pdf_uses_direct_upload_with_server_token(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.trigger_file.return_value = {"status": "success", "token": "server-token"}

    result, stage = trigger_input(client, _local_pdf(pdf), trigger_kwargs={"sync": True})

    assert result == {"status": "success", "token": "server-token"}
    assert stage == "trigger_file"
    client.trigger_file.assert_called_once_with(
        file_path=str(pdf),
        token=None,
        server_generated_token=True,
        http_timeout=(60.0, 1860.0),
        sync=True,
    )
    client.trigger_url.assert_not_called()


def test_async_local_pdf_uses_direct_upload_timeout(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.trigger_file.return_value = {"status": "success", "token": "server-token"}

    result, stage = trigger_input(client, _local_pdf(pdf), trigger_kwargs={"sync": False})

    assert result == {"status": "success", "token": "server-token"}
    assert stage == "trigger_file"
    client.trigger_file.assert_called_once_with(
        file_path=str(pdf),
        token=None,
        server_generated_token=True,
        http_timeout=(60.0, 60.0),
        sync=False,
    )
    client.trigger_url.assert_not_called()


def test_undefined_token_stops_after_bounded_checks(monkeypatch, capsys) -> None:
    client = MagicMock()
    client.get_result.return_value = {"status": "undefined"}
    monkeypatch.setattr("uniparser_tools.cli.core.pipeline.time.sleep", lambda _: None)

    result = poll_until_success(client, "missing-token")

    assert result == 1
    assert client.get_result.call_count == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "TOKEN_NOT_FOUND"
    assert payload["token"] == "missing-token"


def test_pending_statuses_still_reach_success(monkeypatch) -> None:
    client = MagicMock()
    client.get_result.side_effect = [
        {"status": "waiting"},
        {"status": "processing"},
        {"status": "success"},
    ]
    monkeypatch.setattr("uniparser_tools.cli.core.pipeline.time.sleep", lambda _: None)

    result = poll_until_success(client, "valid-token")

    assert result == {"status": "success"}
    assert client.get_result.call_count == 3
