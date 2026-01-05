# utils/geometry.py
import cv2
import numpy as np

def box_in_polygon(box, polygon, mode="center"):
    """
    判断检测框是否落在多边形 ROI 内

    :param box: [x1, y1, x2, y2]
    :param polygon: [[x,y], [x,y], ...]
    :param mode:
        - "center": 框中心点在 ROI 内（推荐，快）
        - "iou": 框与 ROI 有交集（慢，不建议实时用）
    :return: bool
    """

    if not polygon:
        return True

    x1, y1, x2, y2 = box

    if mode == "center":
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return _point_in_polygon((cx, cy), polygon)

    elif mode == "iou":
        return _box_polygon_intersect(box, polygon)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

def _point_in_polygon(point, polygon):
    """
    使用 cv2.pointPolygonTest
    """
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0


def _box_polygon_intersect(box, polygon):
    """
    判断 box 是否与 polygon 有交集（mask 法，慢）
    """
    x1, y1, x2, y2 = box

    poly = np.array(polygon, dtype=np.int32)
    mask_poly = np.zeros((1080, 1920), dtype=np.uint8)
    mask_box = np.zeros_like(mask_poly)

    cv2.fillPoly(mask_poly, [poly], 255)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, -1)

    inter = cv2.bitwise_and(mask_poly, mask_box)
    return np.any(inter > 0)
