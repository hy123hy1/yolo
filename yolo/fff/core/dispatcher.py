# core/dispatcher.py
from detectors.intrusion import IntrusionDetector
from detectors.helmet import HelmetDetector
from detectors.counting import CountingDetector
# from utils.logger import get_logger

# 映射配置节名到算法类
DETECTOR_MAP = {
    "INTRUSION": IntrusionDetector,
    "HELMET": HelmetDetector,
    "COUNTING": CountingDetector,
}

class AlgorithmDispatcher:
    def __init__(self, config, yolo_mgr, logger):
        """
        :param config: configparser.ConfigParser 对象
        :param yolo_mgr: YoloManager 对象，用于加载多个 YOLO 模型
        """
        self.config = config
        self.algorithms = {}  # {"intrusion": {"instance": ..., "model": ...}}
        self.logger = logger
        self.yolo_mgr = yolo_mgr
        self.auto_register()

    def register(self, name, detector_cls, yolo_model):
        """注册算法"""
        if detector_cls is None:
            self.logger(f"Detector class for '{name}' is None, skipping registration")
            return

        instance = detector_cls(self.config, self.logger, yolo_model)
        self.algorithms[name] = {"instance": instance, "model": yolo_model}
        self.logger.info(f"Registered algorithm: {name}")

    def auto_register(self):
        """根据配置文件自动注册启用的算法"""
        for section in self.config.sections():
            section_upper = section.upper()
            if section_upper not in DETECTOR_MAP:
                continue

            cfg = self.config[section]
            enable = cfg.getboolean("enable", True)
            if not enable:
                self.logger.info(f"Algorithm {section} disabled in config, skipping")
                continue

            detector_cls = DETECTOR_MAP.get(section_upper)
            if detector_cls is None:
                self.logger.warning(f"No detector class found for section {section}, skipping")
                continue

            # 支持自定义模型路径，否则使用全局 YOLO 模型
            model_path = cfg.get("model", self.config["YOLO"]["model"])
            yolo_model = self.yolo_mgr.get_model(model_path=model_path)

            self.register(section.lower(), detector_cls, yolo_model)

    def run_all(self, frame, context=None):
        """执行所有注册算法"""
        for name, algo in self.algorithms.items():
            det_instance = algo["instance"]
            model = algo["model"]
            dets = model.detect(frame)
            det_instance.process(frame, dets, context=context)

