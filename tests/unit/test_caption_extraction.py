from __future__ import annotations

from uniparser_tools.common.constant import LayoutType
from uniparser_tools.common.dataclass import BBox, TextualResult
from uniparser_tools.tools.caption_extraction.main import clean_flatten_pages


class LegacyTextualResult(TextualResult):
    @property
    def plain(self) -> str:
        return self.text


def _textual_result(
    *, types: list[LayoutType], textual_result_cls: type[TextualResult] = TextualResult
) -> TextualResult:
    item = textual_result_cls(
        token="token",
        page=0,
        block=1,
        bbox=BBox(0, 0, 1, 1),
        conf=1.0,
        page_size=(100, 100),
        type=LayoutType.Reference,
        bboxes=[BBox(0, 0, 0.2, 0.1), BBox(0.2, 0, 0.8, 0.1), BBox(0.8, 0, 1, 0.1)],
        contents=["prefix ", "https://doi.org/10.1/example ", "suffix"],
        types=types or [LayoutType.Text] * 3,
    )
    if not types:
        # Simulate a TextualResult produced by an older version without inline types.
        item.types = []
    return item


def test_clean_flatten_pages_filters_types_with_doi_content() -> None:
    item = _textual_result(types=[LayoutType.Text, LayoutType.Text, LayoutType.Molecule])

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.contents == ["prefix ", "suffix"]
    assert cleaned.bboxes == [BBox(0, 0, 0.2, 0.1), BBox(0.8, 0, 1, 0.1)]
    assert cleaned.types == [LayoutType.Text, LayoutType.Molecule]


def test_clean_flatten_pages_accepts_empty_legacy_types() -> None:
    item = _textual_result(types=[], textual_result_cls=LegacyTextualResult)

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.contents == ["prefix ", "suffix"]
    assert cleaned.bboxes == [BBox(0, 0, 0.2, 0.1), BBox(0.8, 0, 1, 0.1)]
    assert cleaned.types == []
