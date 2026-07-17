"""Tests for ``uniparser_tools.utils.convert.build_item`` and ``dict2obj``.

These functions drive the ``pages_dict`` -> dataclass conversion used by
most downstream consumers of the SDK, so we exercise each branch of the
``build_item`` dispatcher.
"""

from __future__ import annotations

import pytest

from tests.utils import make_reaction_dict
from uniparser_tools.common.dataclass import (
    ChartResult,
    EquationResult,
    ExpressionResult,
    FigureResult,
    GroupedResult,
    LayoutItem,
    MoleculeResult,
    TabularResult,
    TextualResult,
)
from uniparser_tools.utils.convert import build_item, dict2obj


BASE_BLOCK = dict(
    token="t",
    page=0,
    block=0,
    conf=1.0,
    bbox=[0.0, 0.0, 1.0, 1.0],
    page_size=(100, 100),
    type="paragraph",
)


@pytest.mark.parametrize(
    "extra,expected_cls",
    [
        ({"reactions": [make_reaction_dict()]}, ExpressionResult),
        (
            {
                "placeholders": ["##P0##"],
                "contents": ["c1"],
                "html": "<table><tr><td>##P0##</td></tr></table>",
                "bboxes": [[0, 0, 1, 1]],
                "labels": [0],
                "type": "table",
            },
            TabularResult,
        ),
        ({"markush": False, "smi": "CCO", "esmi": "CCO", "caption": "ethanol"}, MoleculeResult),
        ({"data": "a|b\n1|2"}, ChartResult),
        ({"desc": "a cat"}, FigureResult),
        ({"latex_repr": "a+b"}, EquationResult),
        ({"text": "hello", "bboxes": [], "contents": []}, TextualResult),
    ],
)
def test_build_item_dispatches_by_payload_shape(extra, expected_cls) -> None:
    block = {**BASE_BLOCK, **extra}
    if "pages" in block:
        block.pop("pages")
    item = build_item(dict(block))
    assert isinstance(item, expected_cls)


def test_build_item_grouped_recurses() -> None:
    child = {**BASE_BLOCK, "text": "child", "bboxes": [], "contents": []}
    block = {**BASE_BLOCK, "type": "tablegroup", "items": [child], "level": 1}
    item = build_item(block)
    assert isinstance(item, GroupedResult)
    assert len(item.items) == 1
    assert isinstance(item.items[0], TextualResult)


def test_build_item_falls_back_to_layout_item() -> None:
    block = {**BASE_BLOCK, "type": "hline"}
    item = build_item(block)
    assert isinstance(item, LayoutItem)


def test_build_item_strips_pages_key() -> None:
    block = {**BASE_BLOCK, "text": "x", "bboxes": [], "contents": [], "pages": [1, 2, 3]}
    item = build_item(block)
    assert isinstance(item, TextualResult)
    assert not hasattr(item, "pages")


def test_dict2obj_returns_nested_list() -> None:
    pages = [
        [
            {**BASE_BLOCK, "text": "a", "bboxes": [], "contents": []},
            {**BASE_BLOCK, "type": "hline"},
        ],
        [
            {**BASE_BLOCK, "latex_repr": "E=mc^2", "type": "equation"},
        ],
    ]
    result = dict2obj(pages)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 1
    assert isinstance(result[0][0], TextualResult)
    assert isinstance(result[0][1], LayoutItem)
    assert isinstance(result[1][0], EquationResult)
