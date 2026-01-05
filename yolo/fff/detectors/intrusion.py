# detectors/intrusion.py
from .base import BaseDetector
from utils.geometry import box_in_polygon

class IntrusionDetector(BaseDetector):
    def __init__(self, config, logger=None, yolo_model=None):
        super().__init__(config, "INTRUSION", logger=logger, yolo_model=yolo_model)
        self.person_class = config["INTRUSION"].get("person_class", "person")

    def _filter(self, detections):
        persons = []
        for det in detections:
            if det["name"] != self.person_class:
                continue
            if self.enable_roi and not box_in_polygon(det["xyxy"], self.roi):
                continue
            persons.append(det)
        return persons
