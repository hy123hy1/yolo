import numpy as np
import cv2

# 判断是否有交集
def is_intersect(box_rect, fence_poly, frame_shape):
    box_poly = np.array(box_rect, dtype=np.int32)
    fence_poly = np.array(fence_poly, dtype=np.int32)
    mask1 = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    mask2 = np.zeros_like(mask1)
    cv2.fillPoly(mask1, [box_poly], 255)
    cv2.fillPoly(mask2, [fence_poly], 255)
    intersection = cv2.bitwise_and(mask1, mask2)
    return np.any(intersection > 0)

# 解析坐标，获取矩形框
def rect_to_polygon(rect, frame_width, frame_height):

    base_area = {"width": 960, "height": 540}

    # 按实际视频大小映射到逻辑区域
    scale_x = frame_width / base_area["width"]
    scale_y = frame_height / base_area["height"]

    # 映射后的实际视频坐标
    x = rect["x"] * scale_x
    y = rect["y"] * scale_y
    w = rect["width"] * scale_x
    h = rect["height"] * scale_y

    return [
        [int(x), int(y)],  # 左上
        [int(x + w), int(y)],  # 右上
        [int(x + w), int(y + h)],  # 右下
        [int(x), int(y + h)]  # 左下
    ]


