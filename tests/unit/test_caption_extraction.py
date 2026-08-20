from __future__ import annotations

from uniparser_tools.common.constant import LayoutType
from uniparser_tools.common.dataclass import BBox, TextualResult
from uniparser_tools.tools.caption_extraction.main import clean_flatten_pages


def _textual_result(
    *,
    types: list[LayoutType],
    bboxes: list[BBox] | None = None,
) -> TextualResult:
    item = TextualResult(
        token="token",
        page=0,
        block=1,
        bbox=BBox(0, 0, 1, 1),
        conf=1.0,
        page_size=(100, 100),
        type=LayoutType.Reference,
        bboxes=([BBox(0, 0, 0.2, 0.1), BBox(0.2, 0, 0.8, 0.1), BBox(0.8, 0, 1, 0.1)] if bboxes is None else bboxes),
        contents=["prefix ", "https://doi.org/10.1/example ", "suffix"],
        types=types,
    )
    return item


def test_clean_flatten_pages_filters_types_with_doi_content() -> None:
    item = _textual_result(types=[LayoutType.Text, LayoutType.Text, LayoutType.Molecule])

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.contents == ["prefix ", "suffix"]
    assert cleaned.bboxes == [BBox(0, 0, 0.2, 0.1), BBox(0.8, 0, 1, 0.1)]
    assert cleaned.types == [LayoutType.Text, LayoutType.Molecule]


def test_clean_flatten_pages_preserves_text_when_bboxes_are_empty() -> None:
    item = _textual_result(
        types=[LayoutType.Text, LayoutType.Text, LayoutType.Molecule],
        bboxes=[],
    )

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.text == "prefix suffix"
    assert cleaned.contents == ["prefix ", "suffix"]
    assert cleaned.bboxes == []
    assert cleaned.types == [LayoutType.Text, LayoutType.Molecule]


def test_clean_flatten_pages_supports_legacy_text_only_result() -> None:
    text = "Smith et al. https://doi.org/10.1/example Nature 2020."
    item = TextualResult(
        token="token",
        page=0,
        block=1,
        bbox=BBox(0, 0, 1, 1),
        conf=1.0,
        page_size=(100, 100),
        type=LayoutType.Reference,
        text=text,
    )

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.text == text
    assert cleaned.contents == [text]
    assert cleaned.bboxes == []
    assert cleaned.types == [LayoutType.Text]


def test_clean_flatten_pages_supports_structured_contents_without_bboxes() -> None:
    contents = [
        "在软件管理的领域里存在着被称作“依赖地狱”的死亡之谷。  ",
        r"\frac{dy}{dx}=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}",
        "   开放源码软件所广泛使用的  ",
        r"c^{2}=a^{2}+b^{2}-2ab\cos C",
        "  公共 <b>not html</b><img src=x onerror=test> literal @@MATH_TOKEN_0@@ and  ",
        r"x^{2}",
        " ; literal @@CODE_TOKEN_0@@ and `code`",
    ]
    types = [
        LayoutType.Text,
        LayoutType.EquationInline,
        LayoutType.Text,
        LayoutType.EquationInline,
        LayoutType.Text,
        LayoutType.EquationInline,
        LayoutType.Text,
    ]
    rich_text = (
        "在软件管理的领域里存在着被称作“依赖地狱”的死亡之谷。  "
        r"\(\frac{dy}{dx}=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}\)"
        "   开放源码软件所广泛使用的  "
        r"\(c^{2}=a^{2}+b^{2}-2ab\cos C\)"
        "  公共 <b>not html</b><img src=x onerror=test> literal @@MATH_TOKEN_0@@ and  "
        r"\(x^{2}\)"
        " ; literal @@CODE_TOKEN_0@@ and `code`"
    )
    item = TextualResult(
        token="token",
        page=0,
        block=1,
        bbox=BBox(0, 0, 1, 1),
        conf=1.0,
        page_size=(100, 100),
        type=LayoutType.Paragraph,
        bboxes=[],
        contents=contents,
        types=types,
        text=rich_text,
    )

    cleaned = clean_flatten_pages([[item]])[0][0]

    assert cleaned.contents == contents
    assert cleaned.bboxes == []
    assert cleaned.types == types
    assert r"\(\frac{dy}{dx}=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}\)" in cleaned.text
    assert r"\(c^{2}=a^{2}+b^{2}-2ab\cos C\)" in cleaned.text
    assert "<b>not html</b><img src=x onerror=test>" in cleaned.text
    assert "@@MATH_TOKEN_0@@" in cleaned.text
    assert "@@CODE_TOKEN_0@@ and `code`" in cleaned.text
