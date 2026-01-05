# detectors/counting.py
from .base import BaseDetector
from utils.geometry import box_in_polygon

class CountingDetector(BaseDetector):
    """
    人员计数
    - 支持 ROI 区域
    - 可统计每帧人数或累计人数
    """

    def __init__(self, config, logger=None, yolo_model=None):
        super().__init__(config, "COUNTING", logger=logger, yolo_model=yolo_model)
        cfg = config["COUNTING"]
        self.person_class = cfg.get("person_class", "person")
        self.enable_roi = cfg.getboolean("enable_roi", False)

        # 运行时状态
        self.total_count = 0

    def _filter(self, detections):
        """
        返回 ROI 内的人员列表
        """
        persons = []
        for det in detections:
            if det["name"] != self.person_class:
                continue
            if self.enable_roi and not box_in_polygon(det["xyxy"], self.roi):
                continue
            persons.append(det)
        return persons

    def process(self, frame, detections, context=None):
        persons = self._filter(detections)
        self.total_count += len(persons)

        # if self.logger:
        #     self.logger.info(f"[COUNTING] Current frame persons: {len(persons)}, Total: {self.total_count}")
