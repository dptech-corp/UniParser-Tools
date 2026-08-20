from copy import deepcopy
from typing import Dict, List, Union

import numpy as np

from uniparser_tools.common.constant import Direction, LayoutType, LayoutTypeBot, LayoutTypeTop, OrderingMethod
from uniparser_tools.common.dataclass import GroupedResult, LayoutItem, SemanticItem, TextualResult
from uniparser_tools.order.gap_tree import GapTree  # noqa
from uniparser_tools.order.naive_order import order_float_bboxes
from uniparser_tools.order.xy_cut_exp import xycut, xycut_expanded  # noqa
from uniparser_tools.utils.bbox import compute_iou
from uniparser_tools.utils.log import get_root_logger  # noqa


class StructureOrder:
    def sort(self, blocks_in_single_page: List[TextualResult], method="xy_cut", reversed=False, **kwargs):
        if len(blocks_in_single_page) <= 1:
            return blocks_in_single_page, list(range(len(blocks_in_single_page)))

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
            line_height: int = kwargs.get("line_height", 0)
            if reversed:
                # not tested
                reversed_items = []
                for item in blocks_in_single_page:
                    item_ = deepcopy(item)
                    item_.bbox = item_.bbox.transpose((1, 1), Direction.Rotate_180)
                    reversed_items.append(item_)
                sorted_indices = xycut(reversed_items, line_height)[::-1]
            else:
                sorted_indices = xycut(blocks_in_single_page, line_height)
            return [blocks_in_single_page[idx] for idx in sorted_indices], sorted_indices
        elif method == OrderingMethod.XYCutExp:
            line_height: int = kwargs.get("line_height", 0)
            sequential: bool = kwargs.get("sequential", False)
            primary: str = kwargs.get("primary", "y")
            if reversed:
                # not tested
                reversed_items = []
                for item in blocks_in_single_page:
                    item_ = deepcopy(item)
                    item_.bbox = item_.bbox.transpose((1, 1), Direction.Rotate_180)
                    reversed_items.append(item_)
                sorted_indices = xycut_expanded(reversed_items, line_height, primary, sequential)[::-1]
            else:
                sorted_indices = xycut_expanded(blocks_in_single_page, line_height, primary, sequential)
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


def ensure_unique_item_ids(items: List[SemanticItem], stage: str) -> None:
    """Warn about duplicate ``page_block_type`` values in a nested item tree."""
    seen = {}

    def visit(item: SemanticItem, path: str) -> None:
        item_id = (item.page, item.block, item.type)
        previous_path = seen.get(item_id)
        if previous_path is not None:
            readable_id = f"{item.page}_{item.block}_{item.type.value}"
            get_root_logger().warning(
                f"Duplicate item id {readable_id} found during {stage}: {previous_path} and {path}"
            )
        else:
            seen[item_id] = path
        if isinstance(item, GroupedResult):
            for child_idx, child in enumerate(item.items):
                visit(child, f"{path}.items[{child_idx}]")

    for item_idx, item in enumerate(items):
        visit(item, f"items[{item_idx}]")


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
            if self_item.type == LayoutType.Group:
                sorted_children, _ = StructureOrder().sort(
                    children,
                    method=OrderingMethod.XYCutExp,
                    reversed=False,
                    line_height=1,
                    primary="x",
                    sequential=False,
                )
            else:
                sorted_children, _ = StructureOrder().sort(
                    children,
                    method=OrderingMethod.XYCut,
                    reversed=True,
                    line_height=1,
                )
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
            if self_item.type in [
                LayoutTypeTop.Figure,
                LayoutTypeTop.Expression,
                LayoutTypeTop.Chart,
            ]:
                return GroupedResult.clone(
                    self_item,
                    type=LayoutTypeTop.FigureGroup,
                    bbox=bbox,
                    items=[self_item, *sorted_children],
                    level=level,
                    method="grouped-2",
                )
            elif isinstance(self_item, LayoutItem):
                return GroupedResult.clone(
                    self_item,
                    bbox=bbox,
                    items=sorted_children,
                    level=level,
                    method="grouped-1",
                )
            elif isinstance(self_item, GroupedResult):
                return GroupedResult.clone(
                    self_item,
                    bbox=bbox,
                    items=[*self_item.items, *sorted_children],
                    level=level,
                    method="default",
                )
            else:
                if self_item.type in [LayoutTypeBot.Image]:
                    return GroupedResult.clone(
                        self_item,
                        type=LayoutTypeBot.Group,
                        bbox=bbox,
                        items=[self_item, *sorted_children],
                        level=level,
                        method="grouped-3",
                    )
                else:
                    # This is a synthetic container around ``self_item``. Keeping
                    # its original type would duplicate page_block_type between
                    # the parent and the first child.
                    return GroupedResult.clone(
                        self_item,
                        type=LayoutType.Group,
                        bbox=bbox,
                        items=[self_item, *sorted_children],
                        level=level,
                        method="grouped-4",
                    )
        else:
            return self_item

    page_tree: List[SemanticItem] = [build_node(root, 1) for root in roots]

    if flat:
        # cn = sum(count_items(node) for node in page_tree)
        # assert cn == n, f"Page tree construction error: {cn} != {n}"
        pass
    return page_tree


def flatten_item(item: Union[Dict, SemanticItem]) -> List[Union[Dict, SemanticItem]]:
    item = deepcopy(item)
    if isinstance(item, dict) and "items" in item:
        items = item.pop("items")
        item["items"] = []
        return [item] + [ii for i in items for ii in flatten_item(i)]
    elif isinstance(item, GroupedResult):
        items = item.items
        item.items = []  # inplace clear
        return [item] + [ii for i in items for ii in flatten_item(i)]
    else:
        return [item]


def flatten_page(page: List[Union[Dict, SemanticItem]]) -> List[Union[Dict, SemanticItem]]:
    return [i for item in page for i in flatten_item(item)]


def rerank_page(page: List[SemanticItem]):
    # 输入的page为List[SemanticItem]，每个item带有order，根据order重新排序page
    page.sort(key=lambda x: x.order)
    for item in page:
        if isinstance(item, GroupedResult):
            rerank_page(item.items)
    return page


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


def get_columns_type(items: list[LayoutItem]) -> str:
    """
    简单判断页面布局是单栏还是多栏
    返回 "single" 或 "multi"
    """
    if len(items) <= 1:
        return "single"

    # 计算所有 item 的水平投影区间
    projections_x = []
    projections_y = []
    min_x, max_x = float("inf"), float("-inf")
    for it in items:
        x1 = it.bbox.x1 * it.page_size[0]
        x2 = it.bbox.x2 * it.page_size[0]
        projections_x.append((x1, x2))
        min_x = min(min_x, x1)
        max_x = max(max_x, x2)

        y1 = it.bbox.y1 * it.page_size[1]
        y2 = it.bbox.y2 * it.page_size[1]
        projections_y.append((y1 + 3, y2 - 3))

    # 合并区间
    projections_x.sort()
    projections_y.sort()

    iou_x = compute_iou(
        np.array([[start, 0, end, 1] for start, end in projections_x]),
        np.array([[min_x, 0, max_x, 1]]),
    )
    avg_iou_x = np.median(np.max(iou_x, axis=1))

    iou_y = compute_iou(
        np.array([[start, 0, end, 1] for start, end in projections_y]),
        np.array([[start, 0, end, 1] for start, end in projections_y]),
    )
    iou_y[np.diag_indices_from(iou_y)] = -1
    avg_iou_y = np.sum(np.max(iou_y, axis=1)) / len(items)

    delta = 2 * len(items) / it.page_size[1]
    if avg_iou_y < 1e-6:
        return "single_1"
    elif avg_iou_y < delta and avg_iou_x > 0.5:
        return "single_2"
    elif avg_iou_y < 2 * delta and avg_iou_x > 0.5:
        return "single_3"
    else:
        return "multi"


def intra_page_sorting(ordered_pages: List[List[SemanticItem]], default_method: OrderingMethod):
    """
    intra page sorting, bbox is normalized to [0, 1]
    Args:
        ordered_pages: List[List[SemanticItem]]
        default_method: OrderingMethod, default is OrderingMethod.XYCutExp

    Returns:
        List[List[SemanticItem]]
    """
    num_pages = len(ordered_pages)

    # Order before everything
    for page_id in range(num_pages):
        page = ordered_pages[page_id]
        if not page:
            continue

        token = page[0].token
        main_content_items_bot: List[SemanticItem] = []
        main_content_items_top: List[SemanticItem] = []
        margin_content_items: List[SemanticItem] = []

        for item in page:
            if item.type in [
                LayoutType.PageHeader,
                LayoutType.PageFooter,
                LayoutType.PageBar,
                LayoutType.PageNote,
                LayoutType.PageNumber,
                LayoutType.Watermark,
            ]:
                margin_content_items.append(item)
            else:
                # split top and bottom
                if item.type in LayoutTypeTop:
                    main_content_items_top.append(item)
                elif item.type in LayoutTypeBot:
                    main_content_items_bot.append(item)
                else:
                    get_root_logger().warning("未知的布局类型: %s", item.type)
                    main_content_items_bot.append(item)
        try:
            main_content_items_top = build_page_tree(main_content_items_top, 0.9, merge_group=False, flat=True)
        except Exception:
            get_root_logger().exception(f"{token} Page {page_id} build top tree failed!")

        main_content_items = main_content_items_top + main_content_items_bot
        try:
            main_content_items = build_page_tree(main_content_items, 0.9, merge_group=False, flat=False)
        except Exception:
            get_root_logger().exception(f"{token} Page {page_id} build tree failed!")

        if page[-1].order >= 0:  # already ordered
            sorted_all_items = rerank_page(main_content_items + margin_content_items)
            set_item_order(sorted_all_items)
            ordered_pages[page_id] = sorted_all_items
            continue

        columns_type = get_columns_type(main_content_items)

        # remove top or bottom hline, e.g. 2411.01770/page_010
        # 在引入splits 后效果不大，可以注释
        hlines: List[SemanticItem] = []
        for idx in range(len(main_content_items) - 1, -1, -1):
            if main_content_items[idx].type == LayoutType.HLine:
                hlines.append(main_content_items.pop(idx))
        if len(main_content_items) and len(hlines):
            min_y1 = min([item.bbox.y1 for item in main_content_items])
            max_y2 = max([item.bbox.y2 for item in main_content_items])
            for hline in hlines:
                if hline.bbox.y2 < min_y1 or hline.bbox.y1 > max_y2:
                    margin_content_items.append(hline)
                else:
                    main_content_items.append(hline)
        else:
            main_content_items += hlines

        # 将页栏添加到正文
        for idx in range(len(margin_content_items) - 1, -1, -1):
            if margin_content_items[idx].type == LayoutType.PageBar:
                # 页栏在左侧或右侧时，不添加到正文
                if margin_content_items[idx].bbox.x2 < 0.25 or margin_content_items[idx].bbox.x1 > 0.75:
                    continue
                main_content_items.append(margin_content_items.pop(idx))

        defaults_methods = [OrderingMethod.GapTree, OrderingMethod.Naive]
        if default_method not in defaults_methods:
            defaults_methods = [default_method] + defaults_methods
        for method in defaults_methods:
            try:
                if method == OrderingMethod.XYCutExp:
                    kwargs = dict(primary="x", line_height=1, sequential=columns_type != "single_1")
                else:
                    kwargs = dict()
                sorted_main_items, _ = StructureOrder().sort(main_content_items, method=method, **kwargs)
                if method != defaults_methods[0]:
                    get_root_logger().debug(f"{token} Page {page_id} sort using {method} success.")
                break
            except Exception:
                get_root_logger().exception(f"{token} Error")
                for item in main_content_items:
                    get_root_logger().debug(f"{token} Page {page_id} {item}")
                get_root_logger().info(
                    f"{token} Page {page_id} {[(item.bbox * item.page_size).xyxy_int for item in main_content_items]}"
                )
                get_root_logger().info(f"{token} Page {page_id} sort using {method} failed, try next method.")
        else:
            sorted_main_items = main_content_items
            get_root_logger().info(
                f"{token} OrderingMethod all failed, bboxes: {[(b.bbox * b.page_size).xyxy_int for b in page]}"
            )

        # 将页栏从正文移除
        for idx in range(len(sorted_main_items) - 1, -1, -1):
            if sorted_main_items[idx].type == LayoutType.PageBar:
                margin_content_items.append(sorted_main_items.pop(idx))

        try:
            sorted_margin_items, _ = StructureOrder().sort(margin_content_items, method=OrderingMethod.XYCut)
        except Exception:
            get_root_logger().exception(f"{token} Error")
            sorted_margin_items = margin_content_items

        sorted_all_items = sorted_main_items + sorted_margin_items
        set_item_order(sorted_all_items)
        ordered_pages[page_id] = sorted_all_items
    ensure_unique_item_ids(
        [item for page in ordered_pages for item in page],
        "intra_page_sorting final check",
    )
    return ordered_pages
