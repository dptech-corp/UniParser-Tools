# from https://github.com/Sanster/xy-cut.git
# Implement XY Cut Algorithm used in 《XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding》

from copy import deepcopy  # noqa
from typing import List, Tuple

import numpy as np
from scipy.signal import find_peaks  # noqa

from uniparser_tools.common.dataclass import Item, LayoutType  # noqa
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


def expand_items(items: List[Item]):
    if not items:
        return items
    page_size = items[0].page_size
    bboxes = np.asarray([(b.bbox * b.page_size).xyxy_int for b in items])

    proj_x = projection_by_bboxes(boxes=bboxes, axis=0, length=page_size[0] + 1)

    diff = np.diff(proj_x, n=1, axis=0, prepend=0)
    pos_diff = np.maximum(diff, 0)
    neg_diff = np.maximum(-diff, 0)

    height_threshold = page_size[1] / 20
    pos_peaks, pos_info = find_peaks(np.concatenate([np.zeros((1,)), pos_diff]), height=height_threshold)
    neg_peaks, neg_info = find_peaks(np.concatenate([np.zeros((1,)), neg_diff]), height=height_threshold)
    pos_peaks = pos_peaks - 1  # 因为前面 concat 了一个 0
    neg_peaks = neg_peaks - 1

    # neg_peak_heights = neg_info.get("peak_heights", None)
    # pos_peak_heights = pos_info.get("peak_heights", None)

    if not pos_peaks.shape[0]:
        pos_peaks = [int(bboxes[:, 0].min())]
        # pos_peak_heights = np.array([0])
    if not neg_peaks.shape[0]:
        neg_peaks = [int(bboxes[:, 2].max())]
        # neg_peak_heights = np.array([0])

    # merge nearby peaks
    # merged_pos_index = [np.argmax(pos_peak_heights)]
    # merged_pos_peaks = [pos_peaks[merged_pos_index[0]]]
    # for i, (p, v) in enumerate(zip(pos_peaks, pos_peak_heights)):
    #     if i == merged_pos_index[0]:
    #         continue
    #     if p - merged_pos_peaks[-1] < PADDING:
    #         if v > pos_peak_heights[merged_pos_index[-1]]:
    #             merged_pos_peaks[-1] = p
    #             merged_pos_index[-1] = i
    #     else:
    #         merged_pos_peaks.append(p)
    #         merged_pos_index.append(i)
    # pos_peaks = np.sort(np.array(merged_pos_peaks))
    # pos_peak_heights = np.sort(np.array([pos_peak_heights[i] for i in merged_pos_index]))
    # debug_print(f"merged pos_peaks: {pos_peaks}")

    # merged_neg_index = [np.argmax(neg_peak_heights)]
    # merged_neg_peaks = [neg_peaks[merged_neg_index[0]]]
    # for i, (p, v) in enumerate(zip(neg_peaks, neg_peak_heights)):
    #     if i == merged_neg_index[0]:
    #         continue
    #     if merged_neg_peaks[-1] - p < PADDING:
    #         if v > neg_peak_heights[merged_neg_index[-1]]:
    #             merged_neg_peaks[-1] = p
    #             merged_neg_index[-1] = i
    #     else:
    #         merged_neg_peaks.append(p)
    #         merged_neg_index.append(i)
    # neg_peaks = np.sort(np.array(merged_neg_peaks))
    # neg_peak_heights = np.sort(np.array([neg_peak_heights[i] for i in merged_neg_index]))
    # debug_print(f"merged neg_peaks: {neg_peaks}")

    most_left: int = max(0, pos_peaks[0] - PADDING)
    most_right: int = min(neg_peaks[-1] + PADDING, page_size[0])

    expand_left_bboxes = []
    expand_right_bboxes = []
    for idx, item in enumerate(items):
        bbox_ = item.bbox * item.page_size
        if (bbox_.x1 - most_left) > bbox_.width or bbox_.x1 <= most_left:  # item.type != LayoutType.HLine and
            expand_left_bboxes.append(bbox_.xyxy)
        else:
            bbox_.x2 = bbox_.x1 - 1
            bbox_.x1 = most_left
            expand_left_bboxes.append(bbox_.xyxy)

        bbox_ = item.bbox * item.page_size
        if (most_right - bbox_.x2) > bbox_.width or bbox_.x2 >= most_right:  # item.type != LayoutType.HLine and
            expand_right_bboxes.append(bbox_.xyxy)
        else:
            bbox_.x1 = bbox_.x2 + 1
            bbox_.x2 = most_right
            expand_right_bboxes.append(bbox_.xyxy)

    # check iou
    # raw_iou = compute_iou(bboxes, bboxes)
    # raw_iou[np.eye(raw_iou.shape[0], dtype=bool)] = 0
    raw_iou = np.zeros((len(items), len(items)), dtype=float)

    left_iou = compute_iou(np.asarray(expand_left_bboxes), bboxes)
    right_iou = compute_iou(np.asarray(expand_right_bboxes), bboxes)

    expand_items: List[Item] = []
    for idx, (item, iou_l, iou_r, r_iou) in enumerate(zip(items, left_iou, right_iou, raw_iou)):
        max_iou_l = np.max(iou_l[r_iou == 0])  # 忽略已经重合的框
        max_iou_r = np.max(iou_r[r_iou == 0])

        bbox = item.bbox * item.page_size
        if max_iou_l == 0:
            bbox.x1 = most_left
        if max_iou_r == 0:
            bbox.x2 = most_right

        bbox = bbox / item.page_size
        item_ = item.clone(item, bbox=bbox)
        expand_items.append(item_)
    return expand_items


def page_split_projection_profile(expanded_bboxes, page_size: Tuple[int, int], line_height: float = 20):
    most_left = np.min(expanded_bboxes[:, 0])
    most_right = np.max(expanded_bboxes[:, 2])

    proj_y = projection_by_bboxes(boxes=expanded_bboxes, axis=1, length=page_size[1] + 1)
    proj_y_count = projection_by_bboxes(boxes=expanded_bboxes, axis=1, length=page_size[1] + 1, count=1)

    arr_start_y, arr_end_y = split_projection_profile(proj_y, 0, 1)

    try:
        # single col and full width
        proj_y_sc = (proj_y_count == 1) & (proj_y == most_right - most_left)
        arr_start_y_sc, arr_end_y_sc = split_projection_profile(proj_y_sc, 0, 1)

    except Exception:
        arr_start_y_sc = np.array([0])
        arr_end_y_sc = np.array([len(proj_y)])

    try:
        # full width gap and gap width > line_height
        proj_y_gap = proj_y_count == 0
        arr_start_y_gap, arr_end_y_gap = split_projection_profile(proj_y_gap, 0, 1)

        keep = (arr_end_y_gap - arr_start_y_gap) > 3 * line_height
        arr_start_y_gap = arr_start_y_gap[keep]
        arr_end_y_gap = arr_end_y_gap[keep]

    except Exception:
        arr_start_y_gap = np.array([])
        arr_end_y_gap = np.array([])

    # merge arr_start_y, arr_end_y if not inside arr_start_y_, arr_end_y_
    parts = np.asarray(
        sorted(
            {
                0,
                *arr_start_y_sc.tolist(),
                *arr_end_y_sc.tolist(),
                *arr_start_y_gap.tolist(),
                *arr_end_y_gap.tolist(),
                page_size[1] - 1,
            }
        )
    )
    arr_start_y_sc = parts[:-1]
    arr_end_y_sc = parts[1:]

    h_ = np.asarray([arr_start_y_sc, np.zeros_like(arr_start_y_sc), arr_end_y_sc, np.ones_like(arr_end_y_sc)]).T
    h = np.asarray([arr_start_y, np.zeros_like(arr_start_y), arr_end_y, np.ones_like(arr_end_y)]).T

    iou = compute_iou(h_, h)

    keep = np.max(iou, axis=1) > 0

    arr_start_y_sc = arr_start_y_sc[keep]
    arr_end_y_sc = arr_end_y_sc[keep]

    #  = h[iou[keep] > 0]
    overlap_x1 = (iou[keep] > 0) * h[:, 0][None, ...]
    arr_masked = np.ma.masked_where(overlap_x1 == 0, overlap_x1)
    arr_start_y_sc = arr_masked.min(axis=1).filled(0)  # 全0行返回0

    overlap_x2 = (iou[keep] > 0) * h[:, 2][None, ...]
    arr_masked = np.ma.masked_where(overlap_x2 == 0, overlap_x2)
    arr_end_y_sc = arr_masked.max(axis=1).filled(page_size[1] - 1)  # 全0行返回0

    arr_start_y_sc = np.unique(arr_start_y_sc)
    arr_end_y_sc = np.unique(arr_end_y_sc)
    return arr_start_y_sc, arr_end_y_sc


def recursive_xy_cut(
    boxes: np.ndarray, indices: List[int], res: List[int], pos_y=None, min_value: int = 0, min_gap: int = 1
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
        pos_y = split_projection_profile(y_projection, min_value, min_gap)

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
        pos_x = split_projection_profile(x_projection, min_value, min_gap)
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
            recursive_xy_cut(x_sorted_boxes_chunk[_indices], x_sorted_indices_chunk[_indices], res)
    return res


def xycut(items: List[Item]):
    # auto remove zero size bboxes
    if not items:
        return []
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
    recursive_xy_cut(boxes, indices, res, min_gap=line_height)
    assert len(res) == len(set(res)) == len(items), (len(res), len(set(res)), len(items))
    return res


def xycut_expanded(items: List[Item]):
    # auto remove zero size bboxes
    if not items:
        return []
    line_heights = [max(items[0].page_size[1] / 80, 10)]
    page_size = items[0].page_size
    for item in items:
        if item.type != LayoutType.HLine:
            line_heights.append(item.bbox.height * item.page_size[1])
        if hasattr(item, "bboxes"):
            for bbox in item.bboxes:
                line_heights.append(bbox.height * item.page_size[1])
    line_height = max(1, np.min(line_heights))

    expanded_items = expand_items(items)
    expanded_bboxes = np.asarray(
        [(b.bbox * b.page_size).shrink(10, b.page_size, axis="y").xyxy_int for b in expanded_items]
    )

    # fix zero size bboxes
    valid_indices = np.prod(expanded_bboxes[:, 2:] - expanded_bboxes[:, :2], axis=1) > 0
    expanded_bboxes[:, 2:] = np.where(valid_indices[:, None], expanded_bboxes[:, 2:], expanded_bboxes[:, 2:] + 1)

    _indices = expanded_bboxes[:, 1].argsort()
    y_sorted_boxes = expanded_bboxes[_indices]
    pos_y = page_split_projection_profile(y_sorted_boxes, page_size, line_height)

    indices: List[int] = np.arange(len(items))
    res: List[int] = []
    recursive_xy_cut(expanded_bboxes, indices, res, pos_y)
    assert len(res) == len(set(res)) == len(items), (len(res), len(set(res)), len(items))
    return res
