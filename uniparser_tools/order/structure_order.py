from copy import deepcopy
from typing import List

from uniparser_tools.common.constant import Direction, LayoutType, LayoutTypeBot, LayoutTypeTop, OrderingMethod
from uniparser_tools.common.dataclass import GroupedResult, LayoutItem, SemanticItem, TextualResult
from uniparser_tools.order.gap_tree import GapTree  # noqa
from uniparser_tools.order.naive_order import order_float_bboxes
from uniparser_tools.order.xy_cut import xycut  # noqa
from uniparser_tools.order.xy_cut_exp import xycut_expanded  # noqa
from uniparser_tools.utils.log import get_root_logger  # noqa


class StructureOrder:
    def sort(self, blocks_in_single_page: List[TextualResult], method="xy_cut", reversed=False, **kwargs):
        if method == OrderingMethod.Naive:
            width_threshold: float = kwargs.get("width_threshold", 0.1)
            height_threshold: float = kwargs.get("height_threshold", 0.1)
            mode: str = kwargs.get("mode", "col")
            texts: bool = kwargs.get("texts", False)
            if texts:
                block_texts = [b.plain for b in blocks_in_single_page]
            else:
                block_texts = None
            if reversed:
                bboxes = [deepcopy(b.bbox).transpose((1, 1), Direction.Rotate_180).xyxy for b in blocks_in_single_page]
            else:
                bboxes = [b.bbox.xyxy for b in blocks_in_single_page]
            sorted_indices = order_float_bboxes(
                float_bboxes=bboxes,
                texts=block_texts,
                width_threshold=width_threshold,
                height_threshold=height_threshold,
                mode=mode,
            )
            if reversed:
                sorted_indices = sorted_indices[::-1]
            return [blocks_in_single_page[idx] for idx in sorted_indices], sorted_indices
        elif method == OrderingMethod.GapTree:
            if reversed:
                return GapTree(lambda b: deepcopy(b.bbox).transpose((1, 1), Direction.Rotate_180).xyxy).sort(
                    blocks_in_single_page
                )[::-1], None
            else:
                return GapTree(lambda b: b.bbox.xyxy).sort(blocks_in_single_page), None
        elif method == OrderingMethod.XYCut:
            if reversed:
                sorted_indices = xycut(
                    [
                        (deepcopy(b.bbox).transpose((1, 1), Direction.Rotate_180) * b.page_size)
                        .shrink(20, b.page_size)
                        .xyxy_int
                        for b in blocks_in_single_page
                    ]
                )[::-1]
            else:
                sorted_indices = xycut(
                    [(b.bbox * b.page_size).shrink(20, b.page_size).xyxy_int for b in blocks_in_single_page]
                )
            return [blocks_in_single_page[idx] for idx in sorted_indices], sorted_indices
        elif method == OrderingMethod.XYCutExp:
            if reversed:
                # not tested
                reversed_items = []
                for item in blocks_in_single_page:
                    item_ = deepcopy(item)
                    item_.bbox = item_.bbox.transpose((1, 1), Direction.Rotate_180)
                    reversed_items.append(item_)
                sorted_indices = xycut_expanded(reversed_items)[::-1]
            else:
                sorted_indices = xycut_expanded(blocks_in_single_page)
            return [blocks_in_single_page[idx] for idx in sorted_indices], sorted_indices
        else:
            raise ValueError(f"Unknown method: {method}")


def count_items(item: SemanticItem) -> int:
    if isinstance(item, dict) and "items" in item:
        return 1 + sum(count_items(child) for child in item["items"])
    elif isinstance(item, GroupedResult):
        return 1 + sum(count_items(child) for child in item.items)
    else:
        return 1


def build_page_tree(
    page: List[SemanticItem], thresh: float = 0.95, merge_group: bool = False, flat: bool = True
) -> List[SemanticItem]:
    base = 1e-2
    n = len(page)
    parent = [None] * n

    areas = []
    level = []
    for item in page:
        areas.append(item.bbox.area)
        if item.type in [LayoutTypeBot.Group]:
            level.append(5)
        elif item.type in [LayoutTypeBot.Image]:
            level.append(4)
        elif item.type in [
            LayoutTypeTop.FigureGroup,
        ]:
            level.append(3)
        elif item.type in [
            LayoutTypeTop.Figure,
            LayoutTypeTop.Expression,
            LayoutTypeTop.Chart,
        ]:
            level.append(2)
        elif item.type in [
            LayoutTypeTop.MoleculeGroup,
        ]:
            level.append(1)
        else:
            level.append(0)

    # 选父节点：满足 IOF >= thresh 且面积最小
    for i in range(n):
        candidates: List[int] = []
        for j in range(n):
            # same node
            if i == j:
                continue
            # not allowed group
            if level[j] == 0:
                continue
            if areas[j] + level[j] * base > areas[i] + level[i] * base and page[j].bbox.iof(page[i].bbox) >= thresh:
                candidates.append(j)
        if candidates:
            try:
                areas_w = [areas[j] + level[j] * base for j in candidates]
                # 这里有一个旋转问题，正常是左上，旋转90在左下
                dists_w = [
                    page[j].bbox.tl.distance_to(page[i].bbox.tl, method="manhattan") + level[j] * base
                    for j in candidates
                ]
                weights = [x[0] + x[1] for x in zip(areas_w, dists_w)]
                parent[i] = candidates[min(range(len(candidates)), key=lambda j: weights[j])]
            except Exception:
                get_root_logger().exception("Error in building page tree")
                areas_w = [areas[j] + level[j] * base for j in candidates]
                parent[i] = candidates[min(range(len(candidates)), key=lambda j: areas_w[j])]

    # 构建树节点
    nodes = [{"item": page[i], "children": []} for i in range(n)]
    roots = []
    for i, p in enumerate(parent):
        if p is not None:
            nodes[p]["children"].append(nodes[i])
        else:
            roots.append(nodes[i])

    def build_node(node, level=1):
        self_item: SemanticItem = node["item"]
        if node["children"]:
            # 递归处理每个子节点
            children = [build_node(child, level + 1) for child in node["children"]]
            reversed = self_item.type != LayoutType.Group
            sorted_children, _ = StructureOrder().sort(children, method=OrderingMethod.XYCut, reversed=reversed)
            bbox = deepcopy(self_item.bbox)
            if len(sorted_children) >= 2 and merge_group:
                union_bbox = None
                for child in sorted_children:
                    if union_bbox is None:
                        union_bbox = deepcopy(child.bbox)
                    else:
                        union_bbox = union_bbox.union(child.bbox)
                if union_bbox != bbox:
                    allowd_pixel = 10
                    if union_bbox == (
                        (self_item.bbox * self_item.page_size).shrink(allowd_pixel, [-1, -1]) / self_item.page_size
                    ).union(union_bbox):
                        bbox = union_bbox
            if isinstance(self_item, LayoutItem):
                return GroupedResult.clone(self_item, bbox=bbox, items=sorted_children, level=level)
            else:
                if self_item.type in [
                    LayoutTypeTop.Figure,
                    LayoutTypeTop.Expression,
                    LayoutTypeTop.Chart,
                ]:
                    return GroupedResult.clone(
                        self_item, type=LayoutTypeTop.FigureGroup, bbox=bbox, items=[self_item, *sorted_children]
                    )
                elif self_item.type in [LayoutTypeBot.Image]:
                    return GroupedResult.clone(
                        self_item, type=LayoutTypeBot.Group, bbox=bbox, items=[self_item, *sorted_children]
                    )
                else:
                    return GroupedResult.clone(self_item, bbox=bbox, items=[self_item, *sorted_children])
        else:
            return self_item

    page_tree: List[SemanticItem] = [build_node(root, 1) for root in roots]

    if flat:
        # cn = sum(count_items(node) for node in page_tree)
        # assert cn == n, f"Page tree construction error: {cn} != {n}"
        pass
    return page_tree


def set_item_order(page: List[SemanticItem]):
    # 设置每个 item 的 order，递归处理 GroupedResult
    def _set_order(items: List[SemanticItem], start=0):
        order = start
        for item in items:
            if isinstance(item, GroupedResult):
                # 递归对子项设置 order
                item.order = order
                order = _set_order(item.items, order + 1)
            else:
                item.order = order
                order += 1
        return order

    _set_order(page)
