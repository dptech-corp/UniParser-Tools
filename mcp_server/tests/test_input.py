from pathlib import Path

from uniparser_mcp.input import InputKind, resolve_request
from uniparser_mcp.schemas import ParseRequest


def test_resolve_pdf_url():
    req = ParseRequest(pdf_url="https://example.com/paper.pdf")
    resolved = resolve_request(req)
    assert resolved.kind == InputKind.URL
    assert resolved.token_seed == "https://example.com/paper.pdf"
    assert resolved.source_stem == "paper"


def test_resolve_missing_file(tmp_path: Path):
    req = ParseRequest(file_path=str(tmp_path / "missing.pdf"))
    assert "not found" in resolve_request(req).lower()


def test_resolve_image(tmp_path: Path):
    image = tmp_path / "fig.png"
    image.write_bytes(b"png")
    req = ParseRequest(image_path=str(image))
    resolved = resolve_request(req)
    assert resolved.kind == InputKind.IMAGE
    assert resolved.path == image.resolve()
