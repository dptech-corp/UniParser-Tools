import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

from uniparser_mcp.pipeline.parse import run_parse
from uniparser_mcp.schemas import ParseRequest


def test_failed_trigger_does_not_attempt_token_recovery(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    client = MagicMock()
    client.trigger_file.return_value = {
        "status": "error",
        "token": "diagnostic-token",
        "message": "trigger failed",
    }

    out = tmp_path / "out"
    req = ParseRequest(file_path=str(pdf), output_dir=str(out))
    result = asyncio.run(run_parse(client, req))

    assert result.ok is False
    assert result.error.code == "PARSE_ERROR"
    assert result.error.token == "diagnostic-token"
    saved_error = json.loads((out / "trigger_error.json").read_text(encoding="utf-8"))
    assert saved_error["token"] == "diagnostic-token"
    client.trigger_file.assert_called_once()
    client.get_result.assert_not_called()
    client.get_formatted.assert_not_called()


def test_success_parse_writes_trigger_meta(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    client = MagicMock()
    client.trigger_file.return_value = {"status": "success", "token": "tok456"}
    client.get_result.side_effect = [
        {"status": "success"},
        {"status": "success"},
    ]
    client.get_formatted.return_value = {
        "status": "success",
        "content": "# Title",
    }

    out = tmp_path / "out"
    req = ParseRequest(file_path=str(pdf), output_dir=str(out))
    result = asyncio.run(run_parse(client, req))

    assert result.ok is True
    assert result.trigger_meta_path is not None
    assert Path(result.trigger_meta_path).is_file()
    trigger_args, trigger_kwargs = client.trigger_file.call_args
    assert trigger_args == (str(pdf.resolve()),)
    assert trigger_kwargs["token"] is None
    assert trigger_kwargs["server_generated_token"] is True
    assert trigger_kwargs["http_timeout"] == (60.0, 1860.0)


def test_async_local_pdf_uses_direct_upload_timeout(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.trigger_file.return_value = {"status": "error", "description": "stop"}

    req = ParseRequest(file_path=str(pdf), output_dir=str(tmp_path / "out"), async_mode=True)
    result = asyncio.run(run_parse(client, req))

    assert result.ok is False
    _, trigger_kwargs = client.trigger_file.call_args
    assert trigger_kwargs["token"] is None
    assert trigger_kwargs["server_generated_token"] is True
    assert trigger_kwargs["http_timeout"] == (60.0, 60.0)
    assert trigger_kwargs["sync"] is False


def test_direct_upload_transport_failure_uses_upload_error(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = MagicMock()
    client.trigger_file.return_value = {
        "status": "error",
        "description": "The write operation timed out",
        "error_type": "WriteTimeout",
    }

    req = ParseRequest(file_path=str(pdf), output_dir=str(tmp_path / "out"))
    result = asyncio.run(run_parse(client, req))

    assert result.ok is False
    assert result.error.code == "UPLOAD_ERROR"
    assert result.error.stage == "trigger_file"
    assert result.error.token is None


def test_url_trigger_uses_server_generated_token(tmp_path: Path):
    client = MagicMock()
    client.trigger_url.return_value = {"status": "error", "description": "stop"}

    req = ParseRequest(pdf_url="https://example.com/paper.pdf", output_dir=str(tmp_path / "out"))
    result = asyncio.run(run_parse(client, req))

    assert result.ok is False
    _, trigger_kwargs = client.trigger_url.call_args
    assert trigger_kwargs["token"] is None
    assert trigger_kwargs["server_generated_token"] is True


def test_existing_output_is_preserved_and_result_uses_suffixed_sibling(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")

    client = MagicMock()
    client.trigger_file.return_value = {"status": "success", "token": "tok789"}
    client.get_result.side_effect = [
        {"status": "success"},
        {"status": "success", "pages_tree": {}},
    ]
    client.get_formatted.return_value = {"status": "success", "content": "# New"}
    req = ParseRequest(file_path=str(pdf), output_dir=str(existing))
    result = asyncio.run(run_parse(client, req))

    assert result.ok is True
    assert Path(result.output_dir) == tmp_path / "existing_1"
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert Path(result.markdown_path).read_text(encoding="utf-8") == "# New"
    client.trigger_file.assert_called_once()
