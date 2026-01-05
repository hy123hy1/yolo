from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf=0.5, imgsz=640, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        dets = []
        for r in results:
            for b in r.boxes:
                dets.append({
                    "xyxy": b.xyxy[0].cpu().numpy(),
                    "conf": float(b.conf),
                    "cls": int(b.cls),
                    "name": self.model.names[int(b.cls)]
                })
        return dets


class YOLOManager:
    """
    管理多个 YOLODetector 实例
    """
    def __init__(self):
        self.models = {}  # model_path -> YOLODetector

    def get_model(self, model_path, conf=0.5, imgsz=640, device="cpu"):
        if model_path not in self.models:
            self.models[model_path] = YOLODetector(model_path, conf, imgsz, device)
        return self.models[model_path]
