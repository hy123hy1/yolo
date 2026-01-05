import logging
import os
from logging.handlers import RotatingFileHandler

_LOGGER_CACHE = {}

def setup_logger(config):
    level = config["GLOBAL"]["log_level"]

def get_logger(config, name="ai_detection"):
    """
    获取全局 logger（基于 config.ini）
    """
    global _LOGGER_CACHE

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    log_cfg = config["LOG"]

    log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_dir = log_cfg.get("dir", "logs")
    filename = log_cfg.get("filename", "ai_detection.log")
    max_bytes = log_cfg.getint("max_bytes", 10 * 1024 * 1024)
    backup_count = log_cfg.getint("backup_count", 5)
    enable_console = log_cfg.getboolean("console", True)

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # 防止重复加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # ---------- 文件日志 ----------
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, filename),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ---------- 控制台日志 ----------
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _LOGGER_CACHE[name] = logger
    return logger
