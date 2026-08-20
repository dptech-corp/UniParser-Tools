import logging

from uniparser_tools.common.constant import LayoutType, OrderingMethod
from uniparser_tools.common.dataclass import BBox, GroupedResult, LayoutItem
from uniparser_tools.order import structure_order


def _item(block: int, item_type: LayoutType, bbox: BBox) -> LayoutItem:
    return LayoutItem(
        token="token",
        page=0,
        block=block,
        bbox=bbox,
        conf=1.0,
        page_size=(100, 100),
        type=item_type,
    )


def test_build_page_tree_uses_xy_cut_exp_for_group_children(monkeypatch) -> None:
    parent = _item(0, LayoutType.Group, BBox(0, 0, 1, 1))
    child = _item(1, LayoutType.Image, BBox(0.1, 0.1, 0.2, 0.2))
    calls = []

    def capture_sort(self, items, method="xy_cut", reversed=False, **kwargs):
        calls.append((method, reversed, kwargs))
        return items, list(range(len(items)))

    monkeypatch.setattr(structure_order.StructureOrder, "sort", capture_sort)

    structure_order.build_page_tree([parent, child])

    assert calls == [
        (
            OrderingMethod.XYCutExp,
            False,
            {"line_height": 1, "primary": "x", "sequential": False},
        )
    ]


def test_build_page_tree_promotes_figure_parent_to_figure_group() -> None:
    parent = _item(0, LayoutType.Figure, BBox(0, 0, 1, 1))
    child = _item(1, LayoutType.Image, BBox(0.1, 0.1, 0.2, 0.2))

    result = structure_order.build_page_tree([parent, child])[0]

    assert isinstance(result, GroupedResult)
    assert result.type == LayoutType.FigureGroup
    assert result.method == "grouped-2"
    assert result.items == [parent, child]


def test_ensure_unique_item_ids_warns_for_nested_duplicates(caplog) -> None:
    first = _item(1, LayoutType.Image, BBox(0, 0, 0.4, 0.4))
    duplicate = _item(1, LayoutType.Image, BBox(0.5, 0.5, 0.9, 0.9))
    group = GroupedResult.clone(
        first,
        type=LayoutType.Group,
        items=[first, duplicate],
    )

    with caplog.at_level(logging.WARNING):
        structure_order.ensure_unique_item_ids([group], "test")

    assert "Duplicate item id 0_1_image found during test" in caplog.text


def test_intra_page_sorting_runs_final_unique_id_check(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        structure_order,
        "ensure_unique_item_ids",
        lambda items, stage: calls.append((items, stage)),
    )

    assert structure_order.intra_page_sorting([[]], OrderingMethod.XYCutExp) == [[]]
    assert calls == [([], "intra_page_sorting final check")]
