# detectors/helmet.py
from .base import BaseDetector
from utils.geometry import box_in_polygon

class HelmetDetector(BaseDetector):
    """
    安全帽检测
    - 支持自定义 YOLO 模型
    - ROI 支持
    - 事件触发
    """

    def __init__(self, config, logger=None, yolo_model=None):
        super().__init__(config, "HELMET", logger=logger, yolo_model=yolo_model)

        cfg = config["HELMET"]
        self.person_class = cfg.get("person_class", "person")
        self.helmet_class = cfg.get("helmet_class", "helmet")
        self.no_helmet_class = cfg.get("no_helmet_class", "no_helmet")
        self.min_frames = int(cfg.get("min_frames", 20))
        self.end_frames = int(cfg.get("end_frames", 20))

    def _filter(self, detections):
        """
        返回不戴安全帽的人员列表
        """
        no_helmet_persons = []

        for det in detections:
            if det["name"] == self.person_class:
                # 判断 ROI
                if self.enable_roi and not box_in_polygon(det["xyxy"], self.roi):
                    continue

                # 检测该 person 是否戴帽子
                for sub_det in detections:
                    if sub_det["name"] == self.no_helmet_class:
                        # 可以加一个简单 IoU 判断
                        # 假设 xyxy = [x1,y1,x2,y2]
                        px1, py1, px2, py2 = det["xyxy"]
                        hx1, hy1, hx2, hy2 = sub_det["xyxy"]
                        # 简单中心点判断
                        cx = (px1 + px2) / 2
                        cy = (py1 + py2) / 2
                        if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                            no_helmet_persons.append(det)
                            break

        return no_helmet_persons
