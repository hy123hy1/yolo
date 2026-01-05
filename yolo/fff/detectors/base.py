# detectors/base.py
import datetime
from utils.geometry import box_in_polygon
from utils.storage import save_image, save_video
import threading

class BaseDetector:
    def __init__(self, config, section_name, logger=None, yolo_model=None):
        self.config = config
        self.logger = logger
        self.model = yolo_model

        cfg = config[section_name]
        self.enable = cfg.getboolean("enable", True)
        self.enable_roi = cfg.getboolean("enable_roi", False)
        self.roi = eval(cfg.get("roi", "[]"))

        self.save_image_flag = cfg.getboolean("save_image", True)
        self.save_video_flag = cfg.getboolean("save_video", True)
        self.min_frames = cfg.getint("min_frames", 20)
        self.end_frames = cfg.getint("end_frames", 20)

        # 状态机
        self.active = False
        self.frame_counter = 0
        self.disappear_counter = 0
        self.start_time = None

        # 预留帧缓冲，可在子类或父类实现
        self.frame_buffer = []

    def process(self, frame, detections, context=None):
        """每帧调用"""
        filtered = self._filter(detections)

        if filtered:
            self.frame_counter += 1
            self.disappear_counter = 0
        else:
            self.frame_counter = 0
            self.disappear_counter += 1

        # self.logger.info(f"frame_counter: {self.frame_counter}, disappear_counter: {self.disappear_counter}")
        print(f"frame_counter: {self.frame_counter}, disappear_counter: {self.disappear_counter}")

        # === 触发事件 ===
        if not self.active and self.frame_counter >= self.min_frames:
            self._start_event(frame, filtered)

        # === 结束事件 ===
        if self.active and self.disappear_counter >= self.end_frames:
            self._end_event()

    # ---------------- 内部方法 ----------------
    def _filter(self, detections):
        """子类实现自己的过滤逻辑"""
        raise NotImplementedError

    def _start_event(self, frame, filtered_dets):
        """事件开始"""
        self.active = True
        self.start_time = datetime.datetime.now()
        if self.save_image_flag:
            save_image(frame, self.config, prefix=self.__class__.__name__.lower())
        if self.save_video_flag:
            h, w = frame.shape[:2]
            threading.Thread(
                target=save_video,
                args=(self.frame_buffer, None, self.config, self.__class__.__name__.lower(), (w, h)),
                daemon=True
            ).start()
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} event started")

    def _end_event(self):
        """事件结束"""
        self.active = False
        self.frame_counter = 0
        self.disappear_counter = 0
        self.start_time = None
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} event ended")
