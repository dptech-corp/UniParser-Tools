from typing import List, Tuple

import cv2
import numpy as np

from uniparser_tools.common.dataclass import BBox


def get_mini_boxes(contour: List) -> List[Tuple[float, float]]:
    bounding_box = cv2.minAreaRect(np.array(contour, dtype=np.float32))
    points = sorted(cv2.boxPoints(bounding_box).tolist(), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box


def compute_overlap(boxes1, boxes2):
    """
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns:
        - (N, M) array of intersection areas
        - (N, 1, 4) array of boxes1
        - (1, M, 4) array of boxes2
    """
    # N = boxes1.shape[0]
    # M = boxes2.shape[0]

    # Expand dimensions to (N, 1, 4) and (1, M, 4) to broadcast
    boxes1 = np.expand_dims(boxes1, 1)  # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, 0)  # (1, M, 4)

    # Compute intersection coordinates
    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    # Compute intersection area
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h  # (N, M)

    return inter_area, boxes1, boxes2


def compute_iof(boxes1, boxes2):
    """
    boxes1: (N, 4), background boxes
    boxes2: (M, 4), foreground boxes
    Returns: (N, M) array of intersection areas
    if iof[row, col] == 1, then boxes1[row] fully contains boxes2[col]
    """
    inter_area, boxes1, boxes2 = compute_overlap(boxes1, boxes2)

    # Compute area of foreground boxes (boxes1)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (1, M)

    # Avoid division by zero
    area2 = np.maximum(area2, 1e-6)
    result = inter_area / area2  # N, M

    return result


def compute_iou(boxes1, boxes2):
    """
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) array of intersection areas
    """
    inter_area, boxes1, boxes2 = compute_overlap(boxes1, boxes2)

    # Compute area of foreground boxes (boxes1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (N, 1)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (1, M)

    # Avoid division by zero
    union = area1 + area2 - inter_area  # N, M
    union = np.maximum(union, 1e-6)
    result = inter_area / union  # N, M

    return result


def assign(
    bboxes1: List[BBox],  # foreground @ iof
    bboxes2: List[BBox],  # background @ iof
    method: str = "iou",
    threshold: float = 0.7,
    axis: int = 1,
) -> List[int]:
    """
    Assign bboxes1 and bboxes2 based on their overlap and axis
    """
    bboxes1: np.ndarray = np.array([[b.x1, b.y1, b.x2, b.y2] for b in bboxes1])  # M x 4
    bboxes2: np.ndarray = np.array([[b.x1, b.y1, b.x2, b.y2] for b in bboxes2])  # N x 4

    # calc iou
    intersection = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:]) - np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
    intersection = np.maximum(intersection, 0)
    intersection = intersection[:, :, 0] * intersection[:, :, 1]  # M x N x 2

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # M

    if method == "iou":
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # N
        union = area1[:, None] + area2 - intersection
        iou = intersection / (union + 1e-15)  # M x N
    elif method == "iof":
        iou = intersection / (area1[:, None] + 1e-15)  # M x N
    else:
        raise ValueError(f"Unknown method: {method}")

    # assign
    idx: np.ndarray = np.argmax(iou, axis=axis)
    val = np.max(iou, axis=axis)
    idx[val <= threshold] = -1
    hits: List[int] = idx.astype(int).tolist()
    return hits


if __name__ == "__main__":
    bboxes1 = [
        BBox(0, 0, 50, 50),
        BBox(0, 0, 80, 100),
        BBox(0, 100, 80, 200),
        BBox(0, 200, 80, 300),
        # BBox(0, 300, 100, 400),
        # BBox(0, 400, 100, 500),
        # BBox(0, 500, 100, 600),
        # BBox(0, 600, 100, 700),
        # BBox(0, 700, 100, 800),
        # BBox(0, 800, 100, 900),
        # BBox(0, 900, 100, 999),
    ]
    bboxes2 = [
        BBox(0, 0, 100, 100),
        BBox(0, 100, 100, 200),
        BBox(0, 200, 100, 300),
        BBox(0, 300, 100, 400),
        BBox(0, 400, 100, 500),
        BBox(0, 500, 100, 600),
        BBox(0, 600, 100, 700),
        BBox(0, 700, 100, 800),
        BBox(0, 800, 100, 900),
        BBox(0, 900, 100, 999),
    ]
    print(assign(bboxes1, bboxes2, method="iou", axis=1))

    print(assign(bboxes1, bboxes2, method="iof", axis=1))

    print(compute_iof(np.array([b.xyxy for b in bboxes1]), np.array([b.xyxy for b in bboxes1])))
