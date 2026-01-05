import cv2
import numpy as np
from yolo.YOLODetector.app_yolo import parse_yolo_outputs


CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class YOLODetector:
    def __init__(self, session, input_size=640, conf_thres=0.5, iou_thres=0.45):
        self.session = session
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def preprocess(self, frame):
        ori_h, ori_w = frame.shape[:2]
        scale = min(self.input_size / ori_w, self.input_size / ori_h)

        new_w, new_h = int(ori_w * scale), int(ori_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (self.input_size, self.input_size), swapRB=True
        )

        meta = {
            "ori_shape": (ori_h, ori_w),
            "scale": scale,
            "pad": (pad_x, pad_y)
        }
        return blob, meta

    def infer(self, blob):
        return self.session.infer(feeds=[blob], mode="static")

    def postprocess(self, outputs, meta):
        """
        返回格式:
        [(x1, y1, x2, y2, conf, cls_id, label), ...]
        """
        ori_h, ori_w = meta["ori_shape"]
        scale = meta["scale"]
        pad_x, pad_y = meta["pad"]

        detections = parse_yolo_outputs(outputs)  # 原始 640x640 坐标

        results = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            if conf < self.conf_thres:
                continue

            rx1 = int((x1 - pad_x) / scale)
            ry1 = int((y1 - pad_y) / scale)
            rx2 = int((x2 - pad_x) / scale)
            ry2 = int((y2 - pad_y) / scale)

            rx1 = max(0, min(rx1, ori_w - 1))
            ry1 = max(0, min(ry1, ori_h - 1))
            rx2 = max(0, min(rx2, ori_w - 1))
            ry2 = max(0, min(ry2, ori_h - 1))

            label = CLASSES.get(int(cls_id), "unknown")
            results.append((rx1, ry1, rx2, ry2, float(conf), int(cls_id), label))

        return results

    def detect(self, frame):
        blob, meta = self.preprocess(frame)
        outputs = self.infer(blob)
        return self.postprocess(outputs, meta)
