import pytest
from uniparser_mcp.schemas import ParseModeChoice, ParseRequest, TextualChoice


def test_parse_request_requires_exactly_one_input():
    with pytest.raises(ValueError):
        ParseRequest()
    with pytest.raises(ValueError):
        ParseRequest(file_path="/a.pdf", pdf_url="https://x.com/a.pdf")
    req = ParseRequest(file_path="/tmp/paper.pdf")
    assert req.file_path == "/tmp/paper.pdf"


def test_scientific_paper_defaults():
    req = ParseRequest(pdf_url="https://example.com/paper.pdf")
    assert req.textual == TextualChoice.ocr_hq
    assert req.equation == ParseModeChoice.ocr_hq
    assert req.chart == ParseModeChoice.base64
    assert req.molecule == ParseModeChoice.ocr_fast
