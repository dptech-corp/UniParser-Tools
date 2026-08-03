# from https://github.com/Sanster/xy-cut.git
# Implement XY Cut Algorithm used in 《XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding》

from collections import Counter
from typing import List

import numpy as np
from scipy.signal import find_peaks  # noqa

from uniparser_tools.common.dataclass import BBox, Item, LayoutType  # noqa
from uniparser_tools.utils.bbox import compute_iou  # noqa


PADDING = 5


def projection_by_bboxes(
    boxes: np.array,
    axis: int,
    length: int = None,
    count: bool = False,
    padding: bool = True,
) -> np.ndarray:
    """
     通过一组 bbox 获得投影直方图，最后以 per-pixel 形式输出

    Args:
        boxes: [N, 4]
        axis: 0-x坐标向水平方向投影， 1-y坐标向垂直方向投影
        length: 投影方向坐标的最大值
        count: 是否计数，如果计数则为1，否则为bbox的实际高度（相对于axis）

    Returns:
        1D 投影直方图，长度为投影方向坐标的最大值(我们不需要图片的实际边长，因为只是要找文本框的间隔)

    """
    assert axis in [0, 1]
    if length is None:
        length = np.max(boxes[:, axis::2]) + 1
    if padding:
        length += 2 * PADDING
    res = np.zeros(length, dtype=int)
    if count:
        height = 1
    else:
        height = boxes[:, 3 - axis] - boxes[:, 1 - axis]
    np.add.at(res, boxes[:, axis], height)
    np.add.at(res, boxes[:, axis + 2], -height)
    res = np.cumsum(res)[:-1]  # remove last element, res.length = length - 1
    return res


# https://dothinking.github.io/2021-06-19-递归投影分割算法/#:~:text=递归投影分割（Recursive XY,，可以划分段落、行。
def split_projection_profile(arr_values: np.array, min_value: float, min_gap: float):
    """Split projection profile:

    ```
                              ┌──┐
         arr_values           │  │       ┌─┐     ───
             ┌──┐             │  │       │ │      |
             │  │             │  │ ┌───┐ │ │   min_value
             │  │<- min_gap ->│  │ │   │ │ │      |
         ────┴──┴─────────────┴──┴─┴───┴─┴─┴──────┴───
         0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
    ```

    Args:
        arr_values (np.array): 1-d array representing the projection profile.
        min_value (float): Ignore the profile if `arr_value` is less than `min_value`.
        min_gap (float): Ignore the gap if less than this value.

    Returns:
        tuple: Start indexes and end indexes of split groups.
    """
    # all indexes with projection height exceeding the threshold
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return

    # find zero intervals between adjacent projections
    # |  |                    ||
    # ||||<- zero-interval -> |||||
    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    # convert to index of projection range:
    # the start index of zero interval is the end index of projection
    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
    arr_end += 1  # end index will be excluded as index slice

    return arr_start, arr_end


def sticky_items(items: List[Item], *, offset: int = 5, axis: str = "both") -> None:
    """
    原地把 items 的 bbox 做“sticky”对齐。
    axis = 'x'  | 'y'  | 'both'  表示要吸哪条边。
    offset: 最大吸合距离（像素）
    """
    if not items:
        return

    # 1. 收集四条边
    bboxes = np.array([it.p_bbox.xyxy for it in items])  # (N,4)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # 2. 定义 helper：把一维坐标分组→众数→统一赋值
    def _snap(coords: np.ndarray) -> np.ndarray:
        order = np.argsort(coords)
        groups = []
        for idx in order:
            val = coords[idx]
            # 找头/尾差距 ≤ offset 的组
            for g in groups:
                if abs(val - coords[g[0]]) <= offset or abs(val - coords[g[-1]]) <= offset:
                    g.append(idx)
                    break
            else:
                groups.append([idx])

        new_coords = coords.copy()
        for g in groups:
            vals = coords[g]
            cnt = Counter(vals)
            max_freq = max(cnt.values())
            modes = [v for v, c in cnt.items() if c == max_freq]
            snap_val = int(np.mean(modes)) if len(modes) > 1 else int(modes[0])
            new_coords[g] = snap_val
        return new_coords

    # 3. 按 axis 吸合
    if axis in ("x", "both"):
        snapped_x1 = _snap(x1)
        snapped_x2 = _snap(x2)
        bboxes[:, 0] = np.minimum(snapped_x1, snapped_x2)  # x1
        bboxes[:, 2] = np.maximum(snapped_x1, snapped_x2)  # x2
    if axis in ("y", "both"):
        snapped_y1 = _snap(y1)
        snapped_y2 = _snap(y2)
        bboxes[:, 1] = np.minimum(snapped_y1, snapped_y2)  # y1
        bboxes[:, 3] = np.maximum(snapped_y1, snapped_y2)  # y2

    # 4. 写回对象
    new_items = []
    for it, new_box in zip(items, bboxes):
        new_box = BBox(*new_box)
        new_items.append(it.clone(it, bbox=new_box / it.page_size))
    return new_items


def recursive_xy_cut(
    boxes: np.ndarray,
    indices: List[int],
    res: List[int],
    pos_y=None,
    min_value_x: int = 0,
    min_gap_x: int = 1,
    min_value_y: int = 0,
    min_gap_y: int = 1,
    level: int = 0,
):
    """

    Args:
        boxes: (N, 4)
        indices: 递归过程中始终表示 box 在原始数据中的索引
        res: 保存输出结果, reading order indices

    """
    # 向 y 轴投影
    assert len(boxes) == len(indices)

    _indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[_indices]
    y_sorted_indices = indices[_indices]

    # debug_vis(y_sorted_boxes, y_sorted_indices)

    if pos_y is None:
        y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
        pos_y = split_projection_profile(y_projection, min_value_y, min_gap_y)

    if not pos_y:
        return

    arr_y0, arr_y1 = pos_y
    for r0, r1 in zip(arr_y0, arr_y1):
        # [r0, r1] 表示按照水平切分，有 bbox 的区域，对这些区域会再进行垂直切分
        _indices = (r0 <= y_sorted_boxes[:, 1]) & (y_sorted_boxes[:, 1] < r1)

        y_sorted_boxes_chunk = y_sorted_boxes[_indices]
        y_sorted_indices_chunk = y_sorted_indices[_indices]

        _indices = y_sorted_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_sorted_boxes_chunk[_indices]
        x_sorted_indices_chunk = y_sorted_indices_chunk[_indices]

        # 往 x 方向投影
        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        pos_x = split_projection_profile(x_projection, min_value_x, min_gap_x)
        if not pos_x:
            continue

        arr_x0, arr_x1 = pos_x
        if len(arr_x0) == 1:
            # x 方向无法切分
            res.extend(x_sorted_indices_chunk)
            continue

        # x 方向上能分开，继续递归调用
        for c0, c1 in zip(arr_x0, arr_x1):
            _indices = (c0 <= x_sorted_boxes_chunk[:, 0]) & (x_sorted_boxes_chunk[:, 0] < c1)
            recursive_xy_cut(
                x_sorted_boxes_chunk[_indices],
                x_sorted_indices_chunk[_indices],
                res,
                min_value_x=min_value_x,
                min_gap_x=min_gap_x,
                min_value_y=min_value_y,
                min_gap_y=min_gap_y,
                level=level + 1,
            )
    return res


def recursive_xy_cut_priority(
    boxes: np.ndarray,
    indices: List[int],
    res: List[int],
    primary: str = "x",
    splits_y: List[bool] = [],
    splits_x: List[bool] = [],
    min_value_x: int = 0,
    min_gap_x: int = 1,
    min_value_y: int = 0,
    min_gap_y: int = 1,
    level: int = 0,
    sequential: bool = False,
):
    """通用递归 XY 切分：可以选择 primary 为 'y'（行优先，保持原有行为）或 'x'（列优先）。

    当 primary=='x' 时，先按 x 投影切分，再对每个 x 区间按 y 投影切分，并在需要时递归。
    当 primary=='y' 时，先按 y 投影切分，再对每个 y 区间按 x 投影切分，并在需要时递归。
    """
    assert len(boxes) == len(indices)

    if primary == "x":
        axis_pri = 0
        axis_sec = 1
        min_value_pri, min_gap_pri = min_value_x, min_gap_x
        min_value_sec, min_gap_sec = min_value_y, min_gap_y
    else:
        axis_pri = 1
        axis_sec = 0
        min_value_pri, min_gap_pri = min_value_y, min_gap_y
        min_value_sec, min_gap_sec = min_value_x, min_gap_x

    _idx = boxes[:, axis_pri].argsort()
    sorted_boxes_pri = boxes[_idx]
    sorted_indices_pri = indices[_idx]

    projection_pri = projection_by_bboxes(boxes=sorted_boxes_pri, axis=axis_pri)
    pos_pri = split_projection_profile(projection_pri, min_value_pri, min_gap_pri)

    if not pos_pri:
        return

    arr_pri_0, arr_pri_1 = pos_pri
    if sequential and len(arr_pri_0) > 2:
        arr_pri_0 = arr_pri_0[:2]
        arr_pri_1 = arr_pri_1[[0, -1]]
    if axis_pri == 1 and len(splits_y):
        arr_pri_0 = [0] + splits_y
        arr_pri_1 = [i - 1 for i in splits_y] + [np.max(boxes[:, axis_pri + 2])]
    for c0, c1 in zip(arr_pri_0, arr_pri_1):
        _mask = (c0 <= sorted_boxes_pri[:, axis_pri]) & (sorted_boxes_pri[:, axis_pri] < c1)
        if not _mask.any():
            continue

        chunk_boxes = sorted_boxes_pri[_mask]
        chunk_indices = sorted_indices_pri[_mask]

        _idx2 = chunk_boxes[:, axis_sec].argsort()
        sorted_boxes_sec = chunk_boxes[_idx2]
        sorted_indices_sec = chunk_indices[_idx2]

        projection_sec = projection_by_bboxes(boxes=sorted_boxes_sec, axis=axis_sec)
        pos_sec = split_projection_profile(projection_sec, min_value_sec, min_gap_sec)
        if not pos_sec:
            continue

        arr_sec_0, arr_sec_1 = pos_sec
        if len(arr_sec_0) == 1:
            # axis_sec 方向无法进一步切分，直接按 axis_sec 排序输出
            res.extend(sorted_indices_sec)
            continue
        if sequential and len(arr_sec_0) > 2:
            arr_sec_0 = arr_sec_0[:2]
            arr_sec_1 = arr_sec_1[[0, -1]]
        if axis_sec == 1 and len(splits_y):
            arr_sec_0 = [0] + splits_y
            arr_sec_1 = [i - 1 for i in splits_y] + [np.max(boxes[:, axis_sec + 2])]

        for r0, r1 in zip(arr_sec_0, arr_sec_1):
            _mask2 = (r0 <= sorted_boxes_sec[:, axis_sec]) & (sorted_boxes_sec[:, axis_sec] < r1)
            if not _mask2.any():
                continue
            recursive_xy_cut_priority(
                sorted_boxes_sec[_mask2],
                sorted_indices_sec[_mask2],
                res,
                primary="x",
                min_value_x=min_value_x,
                min_gap_x=min_gap_x,
                min_value_y=min_value_y,
                min_gap_y=min_gap_y,
                level=level + 1,
                sequential=sequential,
            )
    return res


def xycut(items: List[Item], line_height: int = 0):
    # auto remove zero size bboxes
    if not items:
        return []

    items = sticky_items(items, offset=5, axis="both")

    if not line_height:
        line_heights = [max(items[0].page_size[1] / 80, 10)]
        for item in items:
            if item.type != LayoutType.HLine:
                line_heights.append(item.bbox.height * item.page_size[1])
            if hasattr(item, "bboxes"):
                for bbox in item.bboxes:
                    line_heights.append(bbox.height * item.page_size[1])
        line_height = max(1, np.min(line_heights))

    boxes = np.asarray([(b.bbox * b.page_size).shrink(2, b.page_size, axis="y").xyxy_int for b in items])

    # fix zero size bboxes
    valid_indices = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) > 0
    boxes[:, 2:] = np.where(valid_indices[:, None], boxes[:, 2:], boxes[:, 2:] + 1)

    indices: List[int] = np.arange(len(boxes))
    res: List[int] = []
    recursive_xy_cut(boxes, indices, res, min_gap_y=line_height)
    assert len(res) == len(set(res)) == len(items), (len(res), len(set(res)), len(items))
    return res


def xycut_expanded(items: List[Item], line_height: int = 0, primary: str = "y", sequential=False):
    """与 xycut 功能等价，但可以选择 primary='y'（默认，行优先）或 'x'（列优先）。
    sequential: False => single 1
    sequential: True => other, one by one

    返回 reading order indices 列表。
    """
    if not items:
        return []

    items = sticky_items(items, offset=5, axis="both")
    min_x, max_x = min([item.bbox.x1 for item in items]), max([item.bbox.x2 for item in items])
    content_width = max_x - min_x  # 0 ~ 1
    content_width = max(content_width, 0.6)

    if not line_height:
        line_heights = [max(items[0].page_size[1] / 80, 10)]
        for item in items:
            if item.type != LayoutType.HLine:
                line_heights.append(item.bbox.height * item.page_size[1])
            if hasattr(item, "bboxes"):
                for bbox in item.bboxes:
                    line_heights.append(bbox.height * item.page_size[1])
        line_height = max(1, np.min(line_heights))

    splits_y = []
    for b in items:
        if b.type == LayoutType.HLine and b.r_bbox.width > content_width * 0.5:
            splits_y.append(int(b.bbox.y1 * b.page_size[1]))
        elif b.type == LayoutType.Group:
            if sequential and b.r_bbox.width > content_width * 0.6:
                splits_y.append(int(b.bbox.y1 * b.page_size[1]))
                splits_y.append(int(b.bbox.y2 * b.page_size[1]))
            elif not sequential and b.r_bbox.width > content_width * 0.5:
                splits_y.append(int(b.bbox.y1 * b.page_size[1]))
                splits_y.append(int(b.bbox.y2 * b.page_size[1]))
    splits_y = sorted(splits_y)

    boxes = np.asarray([(b.bbox * b.page_size).shrink(2, b.page_size, axis="y").xyxy_int for b in items])

    # fix zero size bboxes
    valid_indices = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) > 0
    boxes[:, 2:] = np.where(valid_indices[:, None], boxes[:, 2:], boxes[:, 2:] + 1)

    indices: List[int] = np.arange(len(boxes))
    res: List[int] = []
    recursive_xy_cut_priority(
        boxes,
        indices,
        res,
        primary=primary,
        min_gap_y=line_height,
        sequential=sequential,
        splits_y=splits_y,
    )
    # res = list(dict.fromkeys(res))  # remove duplicates and keep order
    assert len(res) == len(set(res)) == len(items), (len(res), len(set(res)), len(items))
    return res
