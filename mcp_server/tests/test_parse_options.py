from uniparser_mcp.parse_options import SCIENTIFIC_PAPER_DEFAULTS, resolve_trigger_kwargs
from uniparser_mcp.schemas import ParseModeChoice, TextualChoice

from uniparser_tools.common.constant import ParseMode, ParseModeTextual


def test_scientific_paper_defaults_match_cli():
    assert SCIENTIFIC_PAPER_DEFAULTS["textual"] == ParseModeTextual.OCRHighQuality
    assert SCIENTIFIC_PAPER_DEFAULTS["chart"] == ParseMode.DumpBase64
    assert SCIENTIFIC_PAPER_DEFAULTS["molecule"] == ParseMode.OCRFast


def test_resolve_trigger_kwargs_sync():
    kwargs = resolve_trigger_kwargs(
        sync=False,
        textual=TextualChoice.ocr_hq,
        equation=ParseModeChoice.ocr_hq,
        table=ParseModeChoice.ocr_hq,
        chart=ParseModeChoice.base64,
        figure=ParseModeChoice.base64,
        expression=ParseModeChoice.base64,
        molecule=ParseModeChoice.ocr_fast,
    )
    assert kwargs["sync"] is False
    assert kwargs["equation"] == ParseMode.OCRHighQuality
