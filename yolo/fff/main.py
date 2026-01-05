# main.py
import configparser
from core.video import VideoStream
from core.yolo import YOLOManager
from core.dispatcher import AlgorithmDispatcher
from utils.logger import get_logger

def load_config():
    cfg = configparser.ConfigParser()
    cfg.read("config.ini", encoding="utf-8")
    return cfg

def main():
    # ---------------- 配置 & 日志 ----------------
    config = load_config()
    logger = get_logger(config, name=__name__)
    logger.info("System starting...")

    # ---------------- 视频流 ----------------
    stream = VideoStream(config)

    # ---------------- YOLO 管理器 ----------------
    yolo_mgr = YOLOManager()

    # ---------------- 算法调度器 ----------------
    dispatcher = AlgorithmDispatcher(config, yolo_mgr, logger)

    # 通过配置文件自动注册算法
    # dispatcher.auto_register(yolo_mgr)

    # ---------------- 主循环 ----------------
    while True:
        frame = stream.read()
        if frame is None:
            continue
        dispatcher.run_all(frame)


if __name__ == "__main__":
    main()
