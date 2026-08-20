from uniparser_tools.common.constant import LayoutType
from uniparser_tools.utils.format_utils import parse_inline_text


def test_parse_inline_text_keeps_code_markup_as_text() -> None:
    text = "Use <code>CCO</code> and `NCC` as code."

    assert parse_inline_text(text) == ([text], [LayoutType.Text])


def test_parse_inline_text_still_recognizes_explicit_molecule_markup() -> None:
    assert parse_inline_text("***CCO***") == (["CCO"], [LayoutType.Molecule])
