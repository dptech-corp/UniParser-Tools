"""Tests for ``uniparser_tools.common.constant``.

These invariants were previously enforced only in an ``if __name__ == \"__main__\"``
block inside the module; they're lifted to real tests here.
"""

from __future__ import annotations

import pytest

from uniparser_tools.common.constant import (
    EntityTypes,
    FunctionalTypes,
    GroupedTypes,
    IgnoreTypes,
    Language,
    LayoutType,
    OrderingMethod,
    ParseMode,
    SemanticType,
    TableBBoxType,
    TextualTypes,
    ThirdPartyFormatter,
    to_semantic,
)


class TestStrEnum:
    def test_language_mixes_in_str(self) -> None:
        assert Language.Chinese_Simplified == "zh-hans"
        assert Language.Chinese_Simplified in ["zh-hans"]
        assert Language.Chinese_Simplified in {"zh-hans": 1}
        assert Language("zh-hans") is Language.Chinese_Simplified
        assert Language["Chinese_Simplified"] is Language.Chinese_Simplified

    def test_layouttype_equals_string(self) -> None:
        assert LayoutType.Title == "title"
        assert LayoutType("title") is LayoutType.Title

    def test_ordering_method_membership(self) -> None:
        assert "xy_cut" in list(OrderingMethod)
        assert "xy_cut" in OrderingMethod
        assert OrderingMethod("gap_tree") is OrderingMethod.GapTree

    def test_third_party_formatter(self) -> None:
        assert ThirdPartyFormatter.MinerU == "mineru"


class TestIntEnum:
    def test_table_bbox_type_from_int(self) -> None:
        assert TableBBoxType(0) is TableBBoxType.Table

    def test_parse_mode_bool_coercion(self) -> None:
        assert ParseMode(False) is ParseMode.Disable
        assert ParseMode(True) is ParseMode.OCRFast
        assert bool(ParseMode.Disable) is False
        assert bool(ParseMode.OCRFast) is True

    def test_parse_mode_ordering(self) -> None:
        assert min(ParseMode.Disable, ParseMode.OCRFast, ParseMode.OCRHighQuality) is ParseMode.Disable


class TestLayoutPartition:
    def test_layout_partition_is_exhaustive_and_disjoint(self) -> None:
        partitions = [TextualTypes, EntityTypes, GroupedTypes, IgnoreTypes, FunctionalTypes]
        union_count = sum(len(p) for p in partitions)
        assert union_count == len(LayoutType), (
            f"Layout partitions have {union_count} entries but LayoutType has {len(LayoutType)}"
        )
        union_set: set[LayoutType] = set()
        for part in partitions:
            for item in part:
                assert item not in union_set, f"{item} listed in multiple partitions"
                union_set.add(item)
        assert union_set == set(LayoutType)

    @pytest.mark.parametrize("layout", list(LayoutType))
    def test_to_semantic_maps_every_layout_type(self, layout: LayoutType) -> None:
        result = to_semantic(layout)
        assert result in SemanticType
