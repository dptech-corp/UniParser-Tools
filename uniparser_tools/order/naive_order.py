from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from uniparser_tools.common.dataclass import FLOAT_4


@dataclass
class FloatBBoxObject:
    bbox: FLOAT_4
    text: str
    idx: int


class DisjointSet:
    def __init__(self, n: int):
        self.parents = list(range(n))

    def find(self, x: int) -> int:
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x: int, y: int):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parents[max(x_root, y_root)] = min(x_root, y_root)

    def split(self) -> Dict[int, List[int]]:
        d = {}
        for i, f in enumerate(self.parents):
            d.setdefault(f, []).append(i)
        return d


def get_bbox_of_bboxes(bboxes: List[FLOAT_4]) -> FLOAT_4:
    assert len(bboxes) > 0
    x1s, y1s, x2s, y2s = zip(*bboxes)
    return min(x1s), min(y1s), max(x2s), max(y2s)


def get_intersection_between_lines(line1: Tuple[float, float], line2: Tuple[float, float]) -> float:
    return min(max(line1), max(line2)) - max(min(line1), min(line2))


def split_float_bboxes(
    float_bboxes: List[FLOAT_4],
    threshold: float,
    mode: str,
) -> List[List[int]]:
    row_wise = mode == "row"
    # x1, y1, x2, y2 = get_bbox_of_bboxes(float_bboxes)
    # w, h = x2 - x1, y2 - y1
    n_bbox = len(float_bboxes)
    disjoint_set = DisjointSet(n_bbox)
    for i in range(n_bbox - 1):
        for j in range(i + 1, n_bbox):
            ix1, iy1, ix2, iy2 = float_bboxes[i]
            jx1, jy1, jx2, jy2 = float_bboxes[j]
            h_intersection = get_intersection_between_lines((iy1, iy2), (jy1, jy2))
            w_intersection = get_intersection_between_lines((ix1, ix2), (jx1, jx2))
            iw, ih = ix2 - ix1, iy2 - iy1
            jw, jh = jx2 - jx1, jy2 - jy1
            span = min(ih, jh) if row_wise else min(iw, jw)
            intersection, s_intersection = (
                (h_intersection, w_intersection) if row_wise else (w_intersection, h_intersection)
            )
            if intersection > threshold * span:
                disjoint_set.union(i, j)
    dict_int_list = disjoint_set.split()
    if row_wise:
        keys = sorted(dict_int_list.keys(), key=lambda x: float_bboxes[x][1])
    else:
        keys = sorted(dict_int_list.keys(), key=lambda x: float_bboxes[x][0])
    return [dict_int_list[k] for k in keys]


def split_bbox_objects(
    float_bbox_objects: List[FloatBBoxObject],
    threshold: float,
    mode: str,
) -> List[List[FloatBBoxObject]]:
    float_bboxes = [obj.bbox for obj in float_bbox_objects]
    groups = split_float_bboxes(float_bboxes, threshold, mode)
    return [[float_bbox_objects[i] for i in group] for group in groups]


class FloatBBoxSplit:
    def __init__(
        self,
        float_bbox_objects: List[FloatBBoxObject],
        dict_threshold: Dict[str, float],
        mode: str,
        panic: bool,
    ):
        self.float_bbox_objects = float_bbox_objects
        self.mode = mode
        self.splits: Optional[List[FloatBBoxSplit]] = None
        if len(float_bbox_objects) > 1:
            groups_float_bbox_objects = split_bbox_objects(float_bbox_objects, dict_threshold[mode], mode)
            next_panic = len(groups_float_bbox_objects) == 1
            if not panic or not next_panic:
                self.splits = [
                    FloatBBoxSplit(
                        group_float_bbox_objects,
                        dict_threshold,
                        "col" if mode == "row" else "row",
                        next_panic,
                    )
                    for group_float_bbox_objects in groups_float_bbox_objects
                ]

    def __repr__(self) -> str:
        if self.splits is None:
            if len(self.float_bbox_objects) == 1:
                return str(self.float_bbox_objects[0].bbox)
            return " + ".join([str(obj.bbox) for obj in self.float_bbox_objects])
        return f"({self.mode}, {self.splits})"

    @property
    def num_splits(self) -> int:
        if self.splits is None:
            return 1
        return len(self.splits)

    @property
    def bbox(self) -> FLOAT_4:
        return get_bbox_of_bboxes([obj.bbox for obj in self.float_bbox_objects])

    @property
    def sorted_float_bbox_objects(self) -> List[FloatBBoxObject]:
        if self.splits is None:
            return self.float_bbox_objects
        if self.mode == "col":
            return sum([split.sorted_float_bbox_objects for split in self.splits], [])
        split_groups = []
        split_groups_n_col = []
        temp_n_col = 1
        for i, split in enumerate(self.splits):
            if i == 0:
                pass
            elif split.num_splits == 1:
                if split.close_to(self.splits[split_groups[-1][-1]]):
                    split_groups[-1].append(i)
                    continue
            elif len(split.splits) == temp_n_col:
                if (
                    self.splits[split_groups[-1][-1]].num_splits == 1
                    and split.close_to(self.splits[split_groups[-1][-1]])
                    or split.each_col_close_to(self.splits[split_groups[-1][-1]])
                ):
                    split_groups[-1].append(i)
                    continue
            temp_n_col = split.num_splits
            split_groups.append([i])
            split_groups_n_col.append(temp_n_col)

        sorted_float_bbox_objects = []
        for split_group, n_col in zip(split_groups, split_groups_n_col):
            for i in range(n_col):
                for idx in split_group:
                    split: FloatBBoxSplit = self.splits[idx]
                    if split.num_splits == 1 and i == 0:
                        sorted_float_bbox_objects.extend(split.sorted_float_bbox_objects)
                    elif split.num_splits > i:
                        sorted_float_bbox_objects.extend(split.splits[i].sorted_float_bbox_objects)
        return sorted_float_bbox_objects

    def close_to(self, split: FloatBBoxSplit, threshold=0.03) -> bool:
        ix1, iy1, ix2, iy2 = self.bbox
        jx1, jy1, jx2, jy2 = split.bbox
        h_intersection = get_intersection_between_lines((iy1, iy2), (jy1, jy2))
        return h_intersection > -threshold

    def each_col_close_to(self, split: FloatBBoxSplit, threshold=0.02) -> bool:
        if self.num_splits == 1:
            return False
        if split.num_splits != self.num_splits:
            return False
        for i in range(self.num_splits):
            ix1, iy1, ix2, iy2 = self.splits[i].bbox
            jx1, jy1, jx2, jy2 = split.splits[i].bbox
            h_intersection = get_intersection_between_lines((iy1, iy2), (jy1, jy2))
            w_intersection = get_intersection_between_lines((ix1, ix2), (jx1, jx2))
            iw, _ = ix2 - ix1, iy2 - iy1
            jw, _ = jx2 - jx1, jy2 - jy1
            if w_intersection < 0.8 * min(iw, jw) or h_intersection < -threshold:
                return False
        return True


def order_float_bboxes(
    float_bboxes: List[FLOAT_4],
    width_threshold: float = 0.1,
    height_threshold: float = 0.1,
    texts: Optional[List[str]] = None,
    mode: str = "row",
) -> List[int]:
    n_bbox = len(float_bboxes)
    float_bbox_objects = [FloatBBoxObject(float_bboxes[i], "" if texts is None else texts[i], i) for i in range(n_bbox)]
    main_split = FloatBBoxSplit(
        float_bbox_objects,
        {"row": height_threshold, "col": width_threshold},
        mode,
        False,
    )
    sorted_float_bbox_objects = main_split.sorted_float_bbox_objects
    return [obj.idx for obj in sorted_float_bbox_objects]
