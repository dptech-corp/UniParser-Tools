import json
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union

import fitz  # PyMuPDF
from fitz.utils import get_pixmap
from PIL import Image

from uniparser_tools.common.constant import LayoutType, LayoutTypeBot, LayoutTypeTop, OrderingMethod
from uniparser_tools.common.dataclass import (
    BBox,
    ExpressionResult,
    GroupedResult,
    Item,
    SemanticItem,
    TabularResult,
    TextualResult,
)
from uniparser_tools.order.structure_order import StructureOrder, count_items, set_item_order
from uniparser_tools.utils.convert import dict2obj
from uniparser_tools.utils.log import get_root_logger
from uniparser_tools.utils.processor import (
    clean_scientific_text,
    find_figure_caption_kws,
    flat_layout,
    is_head_of_paragraph,
    is_tail_of_paragraph,
    recursive_required_content,
    recursive_required_items,
    tree_repr,
    truncate_string,
)


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
                # get_root_logger().debug([i, page[i].type, candidates, areas_w, dists_w, weights, parent[i]])
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
                        get_root_logger().debug(
                            f"merge group form {bbox * self_item.page_size} to {union_bbox * self_item.page_size}"
                        )
                        bbox = union_bbox
            return GroupedResult.clone(self_item, bbox=bbox, items=sorted_children, level=level)
        else:
            return self_item

    page_tree: List[SemanticItem] = [build_node(root, 1) for root in roots]

    if flat:
        cn = sum(count_items(node) for node in page_tree)
        assert cn == n, f"Page tree construction error: {cn} != {n}"
    return page_tree


def clean_flatten_pages(pages: List[List[SemanticItem]]):
    ignore_types = [
        LayoutType.HLine,
        LayoutType.PageHeader,
        LayoutType.PageFooter,
        LayoutType.PageNumber,
        LayoutType.PageBar,
        LayoutType.Watermark,
    ]

    new_pages = []
    for page in pages:
        new_page = []

        ignore_bboxes: List[BBox] = []
        for item in page:
            if item.type in ignore_types:
                ignore_bboxes.append(item.bbox)

            # 允许提取图片型表格
            if item.type == LayoutType.Table:
                item.type = LayoutType.Image
            elif item.type == LayoutType.TableCaption:
                item.type = LayoutType.ImageCaption
            elif item.type == LayoutType.TableFootnote:
                item.type = LayoutType.ImageFootnote

        for item in page:
            if item.type in ignore_types:
                # remove hline/pageheader/pagefooter/pagenumber/pagebar/watermark
                continue
            elif any(bbox.iou(item.bbox) > 0.9 for bbox in ignore_bboxes):
                continue
            elif item.plain.lower().startswith("doi:10.") or item.plain.lower().startswith("https://doi.org/"):
                # flat and remove item which plain startswith doi:10.
                continue
            else:
                if isinstance(item, TextualResult):
                    if "doi:10." in item.plain.lower() or "https://doi.org/" in item.plain.lower():
                        if item.contents:
                            new_bboxes, new_contents = [], []
                            for bbox, line in zip(item.bboxes, item.contents):
                                if line.lower().startswith("doi:10.") or line.lower().startswith("https://doi.org/"):
                                    pass
                                else:
                                    new_bboxes.append(bbox)
                                    new_contents.append(line)
                            item.bboxes = new_bboxes
                            item.contents = new_contents
                            item.text = " ".join(item.contents)
                        else:
                            item.text = re.sub(r"(?:doi:|https://doi\.org/)10\.\S+", "", item.text, flags=re.I)
                    item.text = item.text.replace("F IGU R E", "FIGURE")  # preprint_ck_2311.14410v2
                    item.text = clean_scientific_text(item.text)
                new_page.append(item)
        new_pages.append(new_page)
    return new_pages


def refine_page_tree(items: List[Item]) -> List[Item]:
    for idx in range(len(items)):
        item = items[idx]
        if isinstance(item, GroupedResult):
            # remove group which only contains one item
            if count_items(item) == 2:
                items[idx] = item.items[0]
                continue
            if count_items(item) == 1:
                items[idx] = SemanticItem.clone(item)
                continue
            # append isolated legends to the only image or figure group

            # 处理没有直属Image但是有直属Chart等
            item_types = [it.type for it in item.items]
            if LayoutType.Image not in item_types:
                ious = []
                for it in item.items:
                    if it.type in [LayoutType.Chart, LayoutType.Figure, LayoutType.Expression, LayoutType.FigureGroup]:
                        ious.append(it.bbox.iou(item.bbox))
                    else:
                        ious.append(0)
                max_iou = max(ious)
                max_iou_idx = ious.index(max(ious))
                if max_iou > 0.85:
                    item.items[max_iou_idx] = GroupedResult.clone(
                        item=item.items[max_iou_idx], type=LayoutType.Image, items=[item.items[max_iou_idx]]
                    )
                    get_root_logger().warning(
                        f"Force to set {item.items[max_iou_idx].type} to image: {item.page = } {item.order = } {max_iou = }"
                    )
                elif sum([iou > 0 for iou in ious]) == 1:
                    # if max_iou > 0.35:
                    item.items[max_iou_idx] = GroupedResult.clone(
                        item=item.items[max_iou_idx], type=LayoutType.Image, items=[item.items[max_iou_idx]]
                    )
                    get_root_logger().warning(
                        f"Force to set {item.items[max_iou_idx].type} to image: {item.page = } {item.order = } {max_iou = }"
                    )
                    # else:
                    #     get_root_logger().warning(f"NOT! Force to set {item.items[max_iou_idx].type} to image: {item.page = } {item.order = } {max_iou = }")

            item_types = [it.type for it in item.items]
            if LayoutType.Legend in item_types:
                if item_types.count(LayoutType.Image) == 1:
                    legends = [it for it in item.items if it.type == LayoutType.Legend]
                    image_idx = item_types.index(LayoutType.Image)
                    if not isinstance(item.items[image_idx], GroupedResult):
                        item.items[image_idx] = GroupedResult.clone(
                            item.items[image_idx], items=[item.items[image_idx]]
                        )
                    image: GroupedResult = item.items[image_idx]
                    for legend in legends:
                        image.bbox = image.bbox.union(legend.bbox)
                        image.items.append(legend)

                elif item_types.count(LayoutType.FigureGroup) == 1:
                    legends = [it for it in item.items if it.type == LayoutType.Legend]
                    figgroup_idx = item_types.index(LayoutType.FigureGroup)
                    if not isinstance(item.items[figgroup_idx], GroupedResult):
                        item.items[figgroup_idx] = GroupedResult.clone()
                    figgroup: GroupedResult = item.items[figgroup_idx]
                    for legend in legends:
                        figgroup.bbox = figgroup.bbox.union(legend.bbox)
                        figgroup.items.append(legend)
                else:
                    pass

            # 对于单独出现的且直属image的chart figure修改为figuregroup包含的单元素
            item_types = [it.type for it in item.items]
            if LayoutType.Image in item_types:
                idxs = [
                    idx
                    for idx, it in enumerate(item.items)
                    if it.type == LayoutType.Image and isinstance(it, GroupedResult) and len(it.items) > 0
                ]
                if len(idxs) > 1:
                    get_root_logger().warning(
                        f"image group contains multiple images: {item.page = } {item.order = } {idxs = }"
                    )
                elif len(idxs) == 1:
                    idx = idxs[0]
                    item_types_of_image = [item.type for item in item.items[idx].items]
                    for sub_idx, item_type in enumerate(item_types_of_image):
                        if item_type in [LayoutType.Chart, LayoutType.Figure, LayoutType.Expression]:
                            item.items[idx].items[sub_idx] = GroupedResult.clone(
                                item=item.items[idx].items[sub_idx],
                                type=LayoutType.FigureGroup,
                                items=[item.items[idx].items[sub_idx]],
                            )
    return items


def reorder_pages(page_objs: List[List[Item]]) -> List[List[SemanticItem]]:
    token = "unknown"
    num_pages = len(page_objs)

    page_results: List[Dict[int, Item]] = [{} for _ in range(num_pages)]
    for page_idx, page in enumerate(page_objs):
        for item_ in page:
            item = deepcopy(item_)
            item.bbox *= item.page_size
            if hasattr(item, "bboxes"):
                for idx in range(len(item.bboxes)):
                    item.bboxes[idx] *= item.page_size
            page_results[page_idx][item.block] = item

    ordered_pages: List[List[SemanticItem]] = [[] for _ in range(num_pages)]

    for page_id in range(num_pages):
        ordered_pages[page_id] = list(page_results[page_id].values())

    # Order before everything
    for page_id in range(num_pages):
        page = ordered_pages[page_id]
        if not page:
            continue

        main_content_items_bot: List[Item] = []
        main_content_items_top: List[Item] = []
        margin_content_items: List[Item] = []

        for item in page:
            item.bbox /= item.page_size
            if hasattr(item, "bboxes"):
                item: Union[TabularResult, TextualResult]
                for j in range(len(item.bboxes)):
                    item.bboxes[j] /= item.page_size
            if isinstance(item, ExpressionResult):
                for reaction in item.reactions:
                    for comp in reaction:
                        comp.bbox /= item.page_size
            if item.type in [
                LayoutType.PageHeader,
                LayoutType.PageFooter,
                LayoutType.PageBar,
                LayoutType.PageNote,
                LayoutType.PageNumber,
                LayoutType.Watermark,
                LayoutType.Section,
            ]:
                # ignore Background
                # margin_content_items.append(item)
                pass
            else:
                # split top and bottom
                if item.type in LayoutTypeTop:
                    main_content_items_top.append(item)
                elif item.type in LayoutTypeBot:
                    main_content_items_bot.append(item)

        if main_content_items_top:
            try:
                main_content_items_top = build_page_tree(main_content_items_top, 0.6, merge_group=False, flat=False)
            except Exception:
                get_root_logger().exception(f"{token} Page {page_id} build top tree failed!")

        main_content_items = main_content_items_top + main_content_items_bot
        if main_content_items:
            try:
                main_content_items = build_page_tree(main_content_items, 0.6, merge_group=True, flat=False)
            except Exception:
                get_root_logger().exception(f"{token} Page {page_id} build tree failed!")

            try:
                main_content_items = refine_page_tree(main_content_items)
            except Exception:
                get_root_logger().exception(f"{token} Page {page_id} refine tree failed!")

            defaults_method = [OrderingMethod.XYCutExp, OrderingMethod.GapTree, OrderingMethod.Naive]
            for method in defaults_method:
                try:
                    sorted_main_items, _ = StructureOrder().sort(main_content_items, method=method)
                    break
                except Exception:
                    get_root_logger().exception(f"{token} Error")
                    get_root_logger().debug(f"{token} Page {page_id} sort using {method} failed, try next method.")
            else:
                sorted_main_items = main_content_items
                get_root_logger().debug(
                    f"{token} OrderingMethod all failed, bboxes: {[(b.bbox * b.page_size).xyxy_int for b in page]}"
                )

        else:
            sorted_main_items = []

        if margin_content_items:
            try:
                sorted_margin_items, _ = StructureOrder().sort(margin_content_items, method=OrderingMethod.XYCut)
            except Exception:
                get_root_logger().exception(f"{token} Error")
                sorted_margin_items = margin_content_items
        else:
            sorted_margin_items = []

        sorted_all_items = sorted_main_items + sorted_margin_items
        set_item_order(sorted_all_items)
        ordered_pages[page_id] = sorted_all_items

    return ordered_pages


class ImageConcatType(Enum):
    SingleFull = "single_full"  # 单页单栏图文对
    CrossPage = "cross_page"  # 跨页图文对
    CrossColumn = "cross_column"  # 跨栏图文对


@dataclass
class ImageWithCaption:
    main_image: Image.Image
    caption_image: Image.Image
    group_image: Image.Image

    image_concat_type: ImageConcatType

    captions: List[str]
    contexts: List[str]
    keywords: List[str]
    subfigures_info: List[Dict]
    task: Dict


def continue_next_column_image_caption(pages: List[List[SemanticItem]]):
    for i in range(len(pages)):
        if not pages[i]:
            continue
        for j in range(len(pages[i]) - 1):
            try:
                last_this_col = pages[i][j]
                first_next_col = pages[i][j + 1]
                if isinstance(last_this_col, GroupedResult):
                    prev_name = f"Page.{last_this_col.page} Order.{last_this_col.order}"
                    next_name = f"Page.{first_next_col.page} Order.{first_next_col.order}"
                    get_root_logger().debug(f"Checking {prev_name} and {next_name}")

                    last_this_col_content = recursive_required_content(last_this_col)
                    if last_this_col_content:
                        if not is_tail_of_paragraph(last_this_col_content):
                            get_root_logger().debug(
                                f"{prev_name} not ends with tail of paragraph: {repr(truncate_string(last_this_col_content))}"
                            )
                        else:
                            continue
                    else:
                        get_root_logger().debug(f"{prev_name} group caption not found.")

                    first_next_col_content = recursive_required_content(first_next_col)
                    if first_next_col_content:
                        if not is_head_of_paragraph(first_next_col_content):
                            get_root_logger().debug(
                                f"{next_name} not starts with head of paragraph: {repr(truncate_string(first_next_col_content))}"
                            )
                        elif not last_this_col_content:
                            get_root_logger().debug(f"{prev_name} group caption not found.")
                        else:
                            continue

                    # check almost aligned in x axis
                    x1, _, x2, _ = last_this_col.bbox.xyxy
                    x1n, _, x2n, _ = first_next_col.bbox.xyxy
                    iou = (min(x2, x2n) - max(x1, x1n)) / (max(x2, x2n) - min(x1, x1n) + 1e-5)
                    if iou > 0:
                        get_root_logger().warning(
                            f"Skip appending non-cross-column caption: {prev_name} with {next_name}, iou={iou:.2f}"
                        )
                        continue

                    if first_next_col.type in [LayoutType.ImageCaption, LayoutType.ImageFootnote, LayoutType.Paragraph]:
                        # last_this_col.items.append(first_next_col)
                        pages[i][j] = GroupedResult.clone(last_this_col, items=[last_this_col, first_next_col])
                        get_root_logger().debug(
                            f"=> Appended caption from next column: {next_name} to {prev_name} caption: {truncate_string(first_next_col.plain)}"
                        )
                    elif isinstance(first_next_col, GroupedResult):
                        get_root_logger().warning(
                            f"Skip appending GroupedResult as caption: {next_name} to {prev_name}"
                        )
                        # get_root_logger().debug(f"Group item: {first_next_page}")
                        continue
            except Exception:
                get_root_logger().exception(f"Error when checking page {i}")

    return pages


def continue_next_page_image_caption(pages: List[List[SemanticItem]]):
    for i in range(len(pages) - 1):
        try:
            if not pages[i] or not pages[i + 1]:
                continue
            last_this_page = pages[i][-1]
            first_next_page = pages[i + 1][0]
            if isinstance(last_this_page, GroupedResult):
                prev_name = f"Page.{last_this_page.page} Order.{last_this_page.order}"
                next_name = f"Page.{first_next_page.page} Order.{first_next_page.order}"
                get_root_logger().debug(f"Checking {prev_name} and {next_name}")

                last_this_page_content = recursive_required_content(last_this_page)
                if last_this_page_content:
                    if not is_tail_of_paragraph(last_this_page_content):
                        get_root_logger().debug(
                            f"{prev_name} not ends with tail of paragraph: {repr(truncate_string(last_this_page_content))}"
                        )
                    else:
                        continue
                else:
                    get_root_logger().debug(f"{prev_name} group caption not found.")

                first_next_page_content = recursive_required_content(first_next_page)
                if first_next_page_content:
                    if not is_head_of_paragraph(first_next_page_content):
                        get_root_logger().debug(
                            f"{next_name} not starts with head of paragraph: {repr(truncate_string(first_next_page_content))}"
                        )
                    elif not last_this_page_content:
                        get_root_logger().debug(f"{prev_name} group caption not found.")
                    else:
                        continue

                # check almost aligned in x axis
                x1, _, x2, _ = last_this_page.bbox.xyxy
                x1n, _, x2n, _ = first_next_page.bbox.xyxy
                iou = (min(x2, x2n) - max(x1, x1n)) / (max(x2, x2n) - min(x1, x1n) + 1e-5)
                if iou < 0.5:
                    get_root_logger().warning(
                        f"Skip appending non-aligned caption: {prev_name} with {next_name}, iou={iou:.2f}"
                    )
                    continue

                if first_next_page.type in [LayoutType.ImageCaption, LayoutType.ImageFootnote, LayoutType.Paragraph]:
                    last_this_page.items.append(first_next_page)
                    last_this_page.bbox.x1 = min(x1, x1n)
                    last_this_page.bbox.x2 = max(x2, x2n)
                    get_root_logger().debug(
                        f"=> Appended caption from next page: {next_name} to {prev_name} caption: {truncate_string(first_next_page.plain)}"
                    )
                elif isinstance(first_next_page, GroupedResult):
                    get_root_logger().warning(f"Skip appending GroupedResult as caption: {next_name} to {prev_name}")
                    # get_root_logger().debug(f"Group item: {first_next_page}")
                    continue
        except Exception:
            get_root_logger().exception(f"Error when checking page {i}")

    return pages


def continue_paragraphs(texts: List[TextualResult]):
    # 使用链表结构构建连续自然段
    class Node:
        def __init__(self, item: Item):
            self.item = item
            self.next: Node = None

    # 构建链表节点
    nodes = [Node(item) for item in texts]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    # 合并连续自然段
    merged_items: List[Item] = []
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        item = node.item
        # 判断是否为可合并类型
        if item.type in [
            LayoutType.Paragraph,
            LayoutType.Caption,
            LayoutType.ImageCaption,
            LayoutType.FigureCaption,
            LayoutType.TableCaption,
            LayoutType.ExpressionCaption,
        ] and not is_tail_of_paragraph(item.plain):
            group = [item]
            cur = node
            while cur.next:
                next_item = cur.next.item
                if next_item.type in [
                    LayoutType.Paragraph,
                    LayoutType.Caption,
                    LayoutType.ImageCaption,
                    LayoutType.FigureCaption,
                    LayoutType.TableCaption,
                    LayoutType.ExpressionCaption,
                ] and not is_head_of_paragraph(next_item.plain):
                    group.append(next_item)
                    cur = cur.next
                    idx += 1
                else:
                    break
            if len(group) > 1:
                merged = GroupedResult.clone(item, items=group)
                merged_items.append(merged)
                get_root_logger().debug(
                    f"concat {len(group)} items on page {item.page}: "
                    + "=>".join([repr(truncate_string(i.plain)) for i in group])
                )
            else:
                get_root_logger().debug(
                    f"skip page {item.page} paragraph {item.order}: {repr(truncate_string(item.plain))}"
                )
                merged_items.append(item)
        else:
            merged_items.append(item)
        idx += 1
    return merged_items


def get_sub_figures(group: GroupedResult):
    # 设置每个 item 的 order，递归处理 GroupedResult
    all_sub_figures: List[SemanticItem] = []

    def _get_sub_figures(items: List[SemanticItem]):
        for item in items:
            if item.type == LayoutType.FigureGroup:
                all_sub_figures.append(item)
                continue
            if hasattr(item, "items"):
                _get_sub_figures(item.items)

    _get_sub_figures(group.items)
    return all_sub_figures


def main(token: str, pdf_path: str, json_path: str, save_dir: str = None, dpi=300, log_level="DEBUG"):
    get_root_logger(log_level=log_level).debug("=" * 100)
    get_root_logger().debug(f"Processing info: \n{pdf_path = }, \n{json_path = }")

    if not os.path.exists(json_path):
        get_root_logger().error("No json file for %s", token)
        return None

    task = dict(
        token=token,
        pdf_path=pdf_path,
        json_path=json_path,
    )

    if save_dir:
        save_dir: Path = Path(save_dir) / token
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        extracted: Dict = {}
        main_content_types = [
            LayoutType.Paragraph,
            LayoutType.Title,  # as break point
        ]  # continued
        allowed_caption_types = [
            LayoutType.ImageCaption,
            LayoutType.ImageFootnote,
            LayoutType.TableCaption,
            LayoutType.TableFootnote,
            LayoutType.Title,
            LayoutType.Paragraph,
        ]
        other_type_groups = [
            LayoutType.Group,
            LayoutType.Equation,
            LayoutType.EquationID,
            LayoutType.Table,
            LayoutType.TableCaption,
            LayoutType.TableFootnote,
            LayoutType.Expression,
            LayoutType.Paragraph,
            LayoutType.PageNote,
        ]

        get_root_logger().debug(f"Loading json from {json_path}")
        pages = dict2obj(json.load(open(json_path, "r", encoding="utf-8")))  # 转为对象

        get_root_logger().debug(f"Loaded {len(pages)} pages and flatting items ...")
        pages = clean_flatten_pages([[i for item in page for i in flat_layout(item)] for page in pages])

        # 重排 page 内 item 顺序
        get_root_logger().debug("=" * 30 + "reorder_pages" + "=" * 30)
        pages = reorder_pages(pages)

        # 合并跨页的 caption
        get_root_logger().debug("=" * 30 + "continue_next_column_image_caption" + "=" * 30)
        pages = continue_next_column_image_caption(pages)
        get_root_logger().debug("=" * 30 + "continue_next_page_image_caption" + "=" * 30)
        pages = continue_next_page_image_caption(pages)

        # 合并跨页的 paragraph
        get_root_logger().debug("=" * 30 + "continue_paragraphs" + "=" * 30)
        paragraphs: List[TextualResult] = [item for page in pages for item in page if item.type in main_content_types]
        paragraphs = [item for item in continue_paragraphs(paragraphs) if item.type in [LayoutType.Paragraph]]
        paragraphs_kws = [find_figure_caption_kws([p.plain]) for p in paragraphs]
        get_root_logger().debug(f"Continued {len(paragraphs)} paragraphs")

        kept_caption_kws = []
        ignored_caption_kws = []
        all_caption_kws = list(find_figure_caption_kws([item.plain for page in pages for item in page]))
        get_root_logger().debug(f"All caption keywords: {all_caption_kws}")

        # 遍历每一页
        doc = fitz.Document(pdf_path)
        for page_idx, page in enumerate(pages):
            # 获取 Group
            get_root_logger().debug("=" * 30 + f" Page {page_idx} " + "=" * 30)
            groups: List[GroupedResult] = [item for item in page if isinstance(item, GroupedResult)]

            for g_idx, group in enumerate(groups):
                group_name = f"Page.{group.page:03d}-Group.{g_idx:02d}"
                item_types = [it.type for it in group.items]
                item_pages = [it.page for it in group.items]

                flat_items = flat_layout(group)
                flat_item_types = [it.type for it in flat_items]
                flat_item_pages = [it.page for it in flat_items]
                unique_item_pages = list(set(flat_item_pages))

                captions = [clean_scientific_text(it.plain) for it in group.items if it.type in allowed_caption_types]
                keywords = find_figure_caption_kws(captions)
                if not keywords:
                    other_types = set(flat_item_types) - set(other_type_groups)
                    if not other_types:
                        continue
                    get_root_logger().warning(f"{token} Skip no-caption group: {group_name} with {captions = }")
                    continue

                keyword = keywords[0]  # only keep the first keyword
                group_name += f"-Cap.{keyword}"
                ignored_caption_kws.append(keyword)

                if LayoutType.Image not in flat_item_types and LayoutType.FigureGroup not in flat_item_types:
                    other_types = set(flat_item_types) - set(other_type_groups)
                    if not other_types:
                        continue
                    get_root_logger().warning(f"{token} Skip non-image group: {group_name} with types {other_types}")
                    continue
                if len(unique_item_pages) == 2:
                    if min([item_pages.count(p) for p in unique_item_pages]) > 1:
                        get_root_logger().error(f"{token} Skip multi-page group: {group_name} with pages {item_pages}")
                        continue
                    get_root_logger().debug(f"{token} Found multi-page group: {group_name} with pages {item_pages}")
                elif len(unique_item_pages) > 2:
                    get_root_logger().error(f"{token} Skip multi-page group: {group_name} with pages {item_pages}")
                    continue

                image_main = [it for it in group.items if it.type in [LayoutType.Image]]
                image_desc = [it for it in group.items if it.type in allowed_caption_types]

                if not image_main:
                    image_main = [it for it in group.items if it.type in [LayoutType.FigureGroup]]

                if not image_main:
                    get_root_logger().warning(
                        f"{token} Skip non-image group: {group_name} with types {item_types}, {image_main}"
                    )
                    continue

                if not image_desc:
                    get_root_logger().warning(
                        f"{token} Skip no-caption group: {group_name} with types {item_types}, {image_desc}"
                    )
                    continue

                if len(image_main) > 1:
                    overlap = False
                    union_image_bbox = image_main[0].bbox
                    for image in image_main[1:]:
                        if union_image_bbox.iou(image.bbox) != 0:
                            overlap = True
                            break
                        union_image_bbox = union_image_bbox.union(image.bbox)
                    image_main = [
                        GroupedResult.clone(
                            group,
                            items=image_main,
                            bbox=union_image_bbox,
                            type=LayoutType.Image,
                        )
                    ]
                    if overlap:
                        get_root_logger().warning(
                            f"{token} Skip overlapped multi-image group: {group_name} with {len(image_main)} images: \n{tree_repr(GroupedResult.clone(group, items=image_main), verbose=True)}"
                        )
                        continue

                get_root_logger().debug(f"Processing Group {group_name}")
                union_desc_bbox = image_desc[0].bbox
                for desc in image_desc[1:]:
                    union_desc_bbox = union_desc_bbox.union(desc.bbox)
                desc_iou_with_image = union_desc_bbox.iou(image_main[0].bbox, axis="x")
                if desc_iou_with_image == 0:
                    get_root_logger().debug(
                        f"{token} Desc iou with image: {union_desc_bbox = }, {image_main[0].bbox = }, {desc_iou_with_image}"
                    )

                # raw_group = group.clone(group)
                # group = refine_group(group)
                # if not group:
                #     get_root_logger().warning(f"{token} Skip refined group: {group_name} with {item_types} items")
                #     continue

                # 裁剪 PDF 区域
                page = doc[group.page]
                max_dpi = min(dpi, max(1, int(4096 * 72 / max(page.rect.width, page.rect.height))))  # max 4096 pixels
                if len(unique_item_pages) == 1 and desc_iou_with_image > 0:
                    image_concat_type = ImageConcatType.SingleFull
                    group_clip: BBox = group.bbox * [page.rect.width, page.rect.height] + tuple(page.rect.top_left)
                    pix = get_pixmap(page, clip=fitz.Rect(*group_clip.xyxy), dpi=max_dpi)
                    group_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    if save_dir:
                        pix.save(save_dir / f"{group_name}.group.png")
                    group_size = [pix.width, pix.height]

                    size_ratio: tuple[float, float] = [pix.width / group.bbox.width, pix.height / group.bbox.height]

                    image_main = image_main[0]
                    image_main_clip: BBox = image_main.bbox * [page.rect.width, page.rect.height] + tuple(
                        page.rect.top_left
                    )
                    pix = get_pixmap(page, clip=fitz.Rect(*image_main_clip.xyxy), dpi=max_dpi)
                    if save_dir:
                        pix.save(save_dir / f"{group_name}.image.png")
                    else:
                        main_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    # image_offset_x, image_offset_y = ((image_main.bbox.tl - group.bbox.tl) * size_ratio).ceil.tuple
                    # image_bbox = [image_offset_x, image_offset_y, image_offset_x + pix.width, image_offset_y + pix.height]
                    image_bbox = [math.ceil(i) for i in ((image_main.bbox - group.bbox.tl) * size_ratio).xyxy]

                    image_desc_bbox = image_desc[0].bbox
                    for desc in image_desc[1:]:
                        image_desc_bbox = image_desc_bbox.union(desc.bbox)
                    image_desc_clip: BBox = image_desc_bbox * [page.rect.width, page.rect.height] + tuple(
                        page.rect.top_left
                    )
                    pix = get_pixmap(page, clip=fitz.Rect(*image_desc_clip.xyxy), dpi=max_dpi)
                    if save_dir:
                        pix.save(save_dir / f"{group_name}.caption.png")
                    else:
                        caption_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                else:
                    if desc_iou_with_image > 0:
                        image_concat_type = ImageConcatType.CrossPage
                    else:
                        image_concat_type = ImageConcatType.CrossColumn
                    get_root_logger().debug(
                        f"Processing {image_concat_type} group: {group_name} with pages {flat_item_pages}"
                    )
                    group_clip: BBox = group.bbox * [page.rect.width, page.rect.height] + tuple(page.rect.top_left)
                    pix = get_pixmap(page, clip=fitz.Rect(*group_clip.xyxy), dpi=max_dpi)

                    size_ratio = [pix.width / group.bbox.width, pix.height / group.bbox.height]

                    image_main = image_main[0]
                    image_main_clip: BBox = image_main.bbox * [page.rect.width, page.rect.height] + tuple(
                        page.rect.top_left
                    )
                    pix = get_pixmap(page, clip=fitz.Rect(*image_main_clip.xyxy), dpi=max_dpi)
                    main_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    if save_dir:
                        pix.save(save_dir / f"{group_name}.image.png")

                    split_images = [main_image]
                    split_offsets = [(image_main.bbox.tl - group.bbox.tl) * size_ratio]

                    image_desc_imgs = []
                    for idx, image_desc_ in enumerate(image_desc):
                        page_ = doc[image_desc_.page]
                        image_desc_clip: BBox = image_desc_.bbox * [page_.rect.width, page_.rect.height] + tuple(
                            page_.rect.top_left
                        )
                        pix = get_pixmap(page_, clip=fitz.Rect(*image_desc_clip.xyxy), dpi=max_dpi)
                        if save_dir:
                            pix.save(save_dir / f"{group_name}.caption.{idx}.png")
                        image_desc_img_ = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        image_desc_imgs.append(image_desc_img_)
                        if image_concat_type == ImageConcatType.CrossPage:
                            split_offsets.append((image_desc_.bbox.tl - group.bbox.tl) * size_ratio)
                        else:
                            split_offsets.append((0, None))

                    split_images += image_desc_imgs
                    max_width = max(img.width for img in split_images)
                    total_height = sum(img.height for img in split_images)
                    group_image = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
                    y_offset = 0
                    for img, (xf, yf) in zip(split_images, split_offsets):
                        get_root_logger().debug([xf, yf, y_offset])
                        group_image.paste(img, (int(xf), y_offset))
                        y_offset += img.height
                    if save_dir:
                        group_image.save(save_dir / f"{group_name}.group.png")
                    else:
                        caption_image = group_image.crop(
                            (
                                0,
                                main_image.height,
                                max_width,
                                group_image.height,
                            )
                        )
                    group_size = [group_image.width, group_image.height]
                    # image_offset_x, image_offset_y = ((image_main.bbox.tl - group.bbox.tl) * size_ratio).ceil.tuple
                    # image_bbox = [image_offset_x, image_offset_y, image_offset_x + split_images[0].width, image_offset_y + split_images[0].height]
                    image_bbox = [math.ceil(i) for i in ((image_main.bbox - group.bbox.tl) * size_ratio).xyxy]

                # 剪裁子图区域
                sub_figures = get_sub_figures(group)
                # check iou when sub_figures is a single item
                if len(sub_figures) == 1:
                    iou = sub_figures[0].bbox.iou(image_main.bbox)
                    if iou > 0.9:
                        # get_root_logger().debug(f"Sub figure {sub_figures[0].bbox} is too similar to main figure {image_main.bbox}")
                        sub_figures = []

                subfigures_info = []
                for idx, item in enumerate(sub_figures):
                    sub_figure_types = [i.type for i in flat_layout(item)]
                    if "chart" in sub_figure_types:
                        sub_figure_type = "chart"
                    elif "table" in sub_figure_types:
                        sub_figure_type = "table"
                    elif "expression" in sub_figure_types:
                        sub_figure_type = "chemical reaction"
                    elif "molecule" in sub_figure_types:
                        sub_figure_type = "molecule"
                    else:
                        sub_figure_type = "figure"

                    legends_items = recursive_required_items(item, [LayoutType.Legend])
                    if len(legends_items) == 1:
                        legend_item = legends_items[0]
                        legend_bbox: BBox = [
                            math.ceil(i) for i in ((legend_item.bbox - group.bbox.tl) * size_ratio).xyxy
                        ]
                        legend_info = dict(bbox=legend_bbox, legend=clean_scientific_text(legend_item.plain.strip()))
                    elif len(legends_items) > 1:
                        get_root_logger().warning(f"Multiple legends found in group: {group_name}")
                        legend_info = dict(
                            legend=clean_scientific_text(" ".join([i.plain.strip() for i in legends_items]))
                        )
                    else:
                        legend_info = {}
                    subfigures_info.append(
                        dict(
                            type=sub_figure_type,
                            caption=clean_scientific_text(
                                recursive_required_content(
                                    item, [LayoutType.ImageCaption, LayoutType.FigureCaption, LayoutType.Paragraph]
                                )
                            ),
                            bbox=[
                                math.ceil(i) for i in ((item.bbox - group.bbox.tl) * size_ratio).xyxy
                            ],  # 归一化到 group bbox
                            legend_info=legend_info,
                        )
                    )

                # get_root_logger().debug(f"items: {captions}")

                kept_caption_kws.append(keyword)
                ignored_caption_kws.remove(keyword)

                # 检索相关 Paragraph
                hit_indices = []
                for i, p_kws in enumerate(paragraphs_kws):
                    if keyword in p_kws:
                        hit_indices.append(i)
                #         get_root_logger().debug(f"Matched: Page.{p.page} Order.{p.order} Para.{i} - {p.plain}")
                # get_root_logger().debug(f"hit_indices: {hit_indices}")

                all_indices: List[int] = hit_indices
                if all_indices:
                    contexts = [clean_scientific_text(paragraphs[i].plain) for i in all_indices]
                else:
                    contexts = []
                    get_root_logger().warning(f"No context found for keyword: {keyword}")

                caption_dict = dict(
                    group_size=group_size,
                    image_bbox=image_bbox,
                    image_concat_type=image_concat_type.value,
                    captions=captions,
                    contexts=contexts,
                    keywords=[keyword],
                    subfigures_info=subfigures_info,
                    structure=tree_repr(group),
                    task=dict(
                        token=token,
                        pdf_path=pdf_path,
                        json_path=json_path,
                    ),
                    verbose=dict(
                        dpi=max_dpi,
                        group=group.dump_dict(),
                        sub_figures=[sub_fig.dump_dict() for sub_fig in sub_figures],
                    ),
                )

                if save_dir:
                    json.dump(
                        caption_dict,
                        open(save_dir / f"{group_name}.json", "w", encoding="utf-8"),
                        ensure_ascii=False,
                        indent=4,
                    )

                    extracted[f"{token}#{group.page:03d}#{g_idx:02d}"] = dict(
                        image=f"{token}/{group_name}.image.png",
                        raw_caption=" ".join(captions),
                        context=contexts,
                        image_bbox=image_bbox,
                        subfigures_info=subfigures_info,
                    )
                else:
                    extraction = ImageWithCaption(
                        main_image=main_image,
                        caption_image=caption_image,
                        group_image=group_image,
                        image_concat_type=image_concat_type,
                        captions=captions,
                        contexts=contexts,
                        keywords=[keyword],
                        subfigures_info=subfigures_info,
                        task=task,
                    )
                    extracted[f"{token}#{group.page:03d}#{g_idx:02d}"] = extraction

        doc.close()

        extract_ratio = -1
        if all_caption_kws:
            main_caption_kws = [i for i in all_caption_kws if "$" not in i]
            if len(main_caption_kws):
                kept_main_caption_kws = [i for i in kept_caption_kws if "$" not in i]
                extract_ratio = len(kept_main_caption_kws) / len(main_caption_kws)

        global_info = {
            "任务索引": token,
            "原始文件路径": pdf_path,
            "原始结果路径": json_path,
            "全文检索-全部图名": ", ".join(all_caption_kws),
            "图文对检索-提取成功的图名": ", ".join(kept_caption_kws),
            "图文对检索-提取失败的图名": ", ".join(ignored_caption_kws),
            "提取图文对数目": len(extracted),
            "提取比例": f"{100 * extract_ratio:.2f}%" if extract_ratio >= 0 else "-",
        }
        if save_dir:
            json.dump(
                global_info,
                open(save_dir / "global_info.json", "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4,
            )

        get_root_logger().debug(f"All caption keywords: {all_caption_kws}")
        get_root_logger().debug(f"Kept captions: {kept_caption_kws}, skipped captions: {ignored_caption_kws}")

        return dict(token=token, extract_ratio=extract_ratio, extracted=extracted, global_info=global_info)

    except Exception:
        get_root_logger().exception(f"Error processing {token}")
        return None
