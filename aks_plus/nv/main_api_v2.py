"""
Video Analytics System V2 - 高性能版本
基于生产者-消费者模式重构的实时视频分析系统

主要优化:
1. 生产者-消费者解耦：帧读取和检测分离，检测不阻塞帧读取
2. 有界队列：防止内存无限增长
3. 线程池限制：视频生成使用线程池，避免线程爆炸
4. 独立检测器实例：每个流有独立的检测器，避免共享状态问题
5. 环形缓冲区：减少内存分配和拷贝

API接口:
- POST /set_fence: 设置围栏并启动检测流
- POST /delete_stream: 停止流并删除围栏
- GET /status: 获取系统状态（包含性能指标）
"""
import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple, Callable
from multiprocessing import Process
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import psutil
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from video_analytics.config.settings import AppConfig
from video_analytics.engines.factory import create_infer_engine
from video_analytics.engines.ultralytics_engine import YOLOV8_CLASSES, SAFETY_HELMET_CLASSES
from video_analytics.detectors.intrusion_detector import IntrusionDetector, FenceRegion
from video_analytics.detectors.helmet_detector import HelmetDetector
from video_analytics.detectors.overcrowd_detector import OvercrowdDetector
from video_analytics.detectors.new_detector import NewDetector
from video_analytics.detectors.base_detector import BaseDetector

# V2 组件
from video_analytics.core.stream_processor_v2 import StreamManagerV2, StreamConfig
from video_analytics.services.video_service_v2 import VideoServiceV2, VideoConfig
from video_analytics.services.storage_service import StorageServiceFactory
from video_analytics.services.alarm_service import AlarmServiceFactory


# =========================
# 配置
# =========================
CONFIG_PATH = "./cfg/config.json"
app = Flask(__name__)
CORS(app)

# 全局状态
class SystemState:
    def __init__(self):
        self.stream_manager: Optional[StreamManagerV2] = None
        self.detectors: Dict[str, BaseDetector] = {}  # 检测器模板
        self.config: Optional[AppConfig] = None
        self.active_fence_streams: Dict[str, Dict] = {}
        self.fence_dict: Dict[str, List] = {}
        self.lock = threading.Lock()

        # 性能统计
        self.start_time = datetime.now()
        self.total_requests = 0

system_state = SystemState()


# =========================
# 检测器工厂函数
# =========================
def create_detector_factory(algo_type: str, config: AppConfig) -> Callable[[], BaseDetector]:
    """
    创建检测器工厂函数
    每个流调用工厂创建独立的检测器实例，避免共享状态问题
    """
    def factory() -> Optional[BaseDetector]:
        try:
            if algo_type == "1":
                # 闯入检测器
                engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    iou=config.model.iou,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id,
                    fp16=config.model.fp16
                )
                detector = IntrusionDetector(
                    engine=engine,
                    config={
                        "min_frames": config.detection.intrusion_min_frames,
                        "confidence": config.detection.intrusion_confidence,
                        "cooldown_seconds": config.detection.intrusion_cooldown,
                        "target_classes": [0]
                    }
                )
                # 如果有围栏配置，应用它
                return detector

            elif algo_type == "2":
                # 安全帽检测器
                person_engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id
                )

                if config.model.helmet_model_path and os.path.exists(config.model.helmet_model_path):
                    helmet_engine = create_infer_engine(
                        model_path=config.model.helmet_model_path,
                        backend='ultralytics',
                        input_size=config.model.input_size,
                        confidence=0.3,
                        classes=SAFETY_HELMET_CLASSES,
                        device_id=config.model.device_id,
                        verbose=False
                    )
                else:
                    helmet_engine = person_engine

                return HelmetDetector(
                    person_engine=person_engine,
                    helmet_engine=helmet_engine,
                    config={
                        "min_frames": config.detection.helmet_min_frames,
                        "person_confidence": config.detection.helmet_confidence,
                        "helmet_confidence": 0.5,
                        "cooldown_seconds": config.detection.helmet_cooldown,
                        "crop_padding": config.detection.helmet_crop_padding
                    }
                )

            elif algo_type == "3":
                # 超员检测器
                engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id
                )
                return OvercrowdDetector(
                    engine=engine,
                    config={
                        "max_people": config.detection.overcrowd_max_people,
                        "duration_threshold": config.detection.overcrowd_duration,
                        "cooldown_seconds": config.detection.overcrowd_cooldown,
                        "confidence": config.detection.overcrowd_confidence
                    }
                )

            elif algo_type == "4":
                # 新检测器
                engine = create_infer_engine(
                    model_path=config.model.person_model_path,
                    backend=config.model.backend,
                    input_size=config.model.input_size,
                    confidence=config.model.confidence,
                    classes=YOLOV8_CLASSES,
                    device_id=config.model.device_id
                )
                return NewDetector(
                    engine=engine,
                    config={
                        "min_frames": config.detection.new_min_frames,
                        "confidence": config.detection.new_confidence,
                        "cooldown_seconds": config.detection.new_cooldown,
                    }
                )

            else:
                logger.error(f"Unknown algorithm type: {algo_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create detector {algo_type}: {e}")
            return None

    return factory


# =========================
# 画框服务 (保持不变)
# =========================
class FFmpegCapture:
    """FFmpeg拉流捕获"""
    def __init__(self, rtsp_url: str, width: int, height: int, init_wait: float = 3.0):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.start()
        if init_wait > 0:
            time.sleep(init_wait)

    def start(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-max_delay", "500000",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-i", self.rtsp_url,
            "-an",
            "-c:v", "rawvideo",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1"
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def read(self, retry: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        for attempt in range(retry):
            try:
                if self.process.poll() is not None:
                    return False, None

                raw = self.process.stdout.read(self.frame_size)

                if len(raw) == self.frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    ).copy()
                    return True, frame
                elif len(raw) == 0:
                    time.sleep(0.05)
                    continue
                else:
                    time.sleep(0.05)
                    continue
            except Exception as e:
                logger.error(f"FFmpegCapture read error: {e}")
                return False, None

        return False, None

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()
            self.process = None

    def is_healthy(self) -> bool:
        return self.process is not None and self.process.poll() is None


def get_stream_size(rtsp_url: str) -> Tuple[int, int]:
    """获取视频流分辨率"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        rtsp_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    w = data["streams"][0]["width"]
    h = data["streams"][0]["height"]
    return w, h


def rect_to_polygon(rect: Dict, default_area: Dict, frame_width: int, frame_height: int) -> List[List[int]]:
    """将矩形转换为多边形点坐标"""
    base_width = default_area.get("width", 960)
    base_height = default_area.get("height", 540)

    scale_x = frame_width / base_width
    scale_y = frame_height / base_height

    x = rect["x"] * scale_x
    y = rect["y"] * scale_y
    w = rect["width"] * scale_x
    h = rect["height"] * scale_y

    return [
        [int(x), int(y)],
        [int(x + w), int(y)],
        [int(x + w), int(y + h)],
        [int(x), int(y + h)]
    ]


def fence_worker(camera_id: str, rtsp_url: str, fence_area: List, output_host: str = "192.168.1.61"):
    """画框工作进程"""
    import queue as queue_module

    logger.info(f"[FenceWorker] 启动画框进程 camera_id={camera_id}")

    try:
        width, height = get_stream_size(rtsp_url)
        cap = FFmpegCapture(rtsp_url, width, height, init_wait=3.0)
        fps = 25

        rtsp_push = f"rtsp://{output_host}:554/Streaming/Channels/{camera_id}"

        def start_ffmpeg():
            cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-fflags", "+genpts+discardcorrupt",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-g", str(fps),
                "-keyint_min", str(fps),
                "-sc_threshold", "0",
                "-pix_fmt", "yuv420p",
                "-f", "rtsp",
                "-rtsp_transport", "tcp",
                rtsp_push
            ]
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )

        push_proc = start_ffmpeg()
        frame_queue = queue_module.Queue(maxsize=50)
        stop_flag = threading.Event()

        def read_loop():
            bad_count = 0
            reconnect_count = 0
            nonlocal cap, push_proc
            success_count = 0

            while not stop_flag.is_set():
                ret, frame = cap.read(retry=3)

                if not ret or frame is None:
                    bad_count += 1

                    if bad_count % 25 == 0:
                        healthy = cap.is_healthy()
                        logger.warning(f"[FenceWorker] {camera_id} 读取帧失败({bad_count}), 进程健康={healthy}")

                    if bad_count > 50 or not cap.is_healthy():
                        logger.warning(f"[FenceWorker] {camera_id} 拉流异常，准备重连...")
                        cap.release()
                        time.sleep(min(2 ** reconnect_count, 10))
                        reconnect_count += 1
                        logger.info(f"[FenceWorker] {camera_id} 第{reconnect_count}次重连...")
                        cap = FFmpegCapture(rtsp_url, width, height, init_wait=2.0)
                        bad_count = 0
                        success_count = 0
                    continue

                bad_count = 0
                success_count += 1

                if success_count > 100:
                    if reconnect_count > 0:
                        logger.info(f"[FenceWorker] {camera_id} 流已稳定，重置重连计数")
                    reconnect_count = 0
                    success_count = 0

                pts = np.array(fence_area, np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame)

        def push_loop():
            nonlocal push_proc
            while not stop_flag.is_set():
                try:
                    frame = frame_queue.get(timeout=1)
                except queue_module.Empty:
                    continue

                if push_proc.poll() is not None:
                    logger.warning(f"[FenceWorker] 推流FFmpeg挂了，重启 {camera_id}")
                    push_proc = start_ffmpeg()
                    continue

                try:
                    push_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    push_proc = start_ffmpeg()
                except Exception as e:
                    logger.error(f"[FenceWorker] 推流异常 {camera_id}: {e}")
                    push_proc = start_ffmpeg()

        t1 = threading.Thread(target=read_loop, daemon=True)
        t2 = threading.Thread(target=push_loop, daemon=True)
        t1.start()
        t2.start()

        while not stop_flag.is_set():
            time.sleep(1)

        stop_flag.set()
        cap.release()
        try:
            push_proc.kill()
        except:
            pass

    except Exception as e:
        logger.error(f"[FenceWorker] 异常 {camera_id}: {e}")
        import traceback
        traceback.print_exc()


def stop_fence_worker(camera_id: str):
    """停止画框进程"""
    with system_state.lock:
        info = system_state.active_fence_streams.get(camera_id)
        if info:
            proc = info.get("process")
            if proc and proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()

            try:
                for p in psutil.process_iter(["pid", "name", "cmdline"]):
                    if "ffmpeg" in p.info["name"] and str(camera_id) in " ".join(p.info["cmdline"] or []):
                        p.kill()
            except:
                pass

            system_state.active_fence_streams.pop(camera_id, None)
            system_state.fence_dict.pop(camera_id, None)
            logger.info(f"[FenceWorker] 已停止 {camera_id}")


# =========================
# 系统初始化
# =========================
def load_config(path: str = CONFIG_PATH) -> AppConfig:
    """加载配置文件"""
    config = AppConfig.from_file(path)

    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                old_config = json.load(f)

            if "server_endpoints" in old_config:
                config.alarm.endpoints_intrusion = old_config["server_endpoints"]
            if "server_endpoints2" in old_config:
                config.alarm.endpoints_helmet = old_config["server_endpoints2"]
            if "server_endpoints3" in old_config:
                config.alarm.endpoints_overcrowd = old_config["server_endpoints3"]
        except Exception as e:
            logger.error(f"[Config] Failed to load old config: {e}")

    return config


def initialize_system() -> Tuple[StreamManagerV2, Dict]:
    """初始化系统组件 V2"""
    logger.info("=" * 60)
    logger.info("[System] Initializing Video Analytics System V2")
    logger.info("[System] Optimizations: Producer-Consumer + ThreadPool + CircularBuffer")
    logger.info("=" * 60)

    config = load_config()
    system_state.config = config

    # 创建服务
    logger.info("[System] Creating services...")

    storage_service = StorageServiceFactory.create(
        service_type=config.storage.type,
        endpoint=config.storage.endpoint,
        access_key=config.storage.access_key,
        secret_key=config.storage.secret_key,
        secure=config.storage.secure,
        bucket_name=config.storage.bucket_name,
        public_url=config.storage.public_url,
    )

    alarm_service = AlarmServiceFactory.create(
        service_type=config.alarm.type,
        endpoints=config.alarm.endpoints,
        timeout=config.alarm.timeout,
        retry_count=config.alarm.retry_count
    )

    # V2 视频服务（使用线程池）
    video_service = VideoServiceV2(
        storage_service=storage_service,
        config=VideoConfig(
            fps=config.stream.fps,
            pre_seconds=config.stream.pre_buffer_seconds,
            post_seconds=config.stream.post_record_seconds,
            max_concurrent_generations=3,  # 限制并发视频生成
            use_memory_encoding=True
        )
    )

    # V2 流管理器
    logger.info("[System] Creating stream manager V2...")
    stream_manager = StreamManagerV2(
        storage_service=storage_service,
        alarm_service=alarm_service,
        video_service=video_service
    )

    # 注册检测器工厂（每个流独立实例）
    logger.info("[System] Registering detector factories...")
    for algo_type in ["1", "2", "3", "4"]:
        factory = create_detector_factory(algo_type, config)
        stream_manager.register_detector_factory(algo_type, factory)

    logger.info("[System] Initialization complete!")
    logger.info("=" * 60)

    return stream_manager, {}


# =========================
# Flask API 路由
# =========================

@app.route("/")
def index():
    """根路由"""
    return jsonify({
        "status": "running",
        "service": "Video Analytics V2 (High Performance)",
        "version": "2.0",
        "optimizations": [
            "Producer-Consumer Architecture",
            "Bounded Frame Queue",
            "Circular Frame Buffer",
            "ThreadPool for Video Generation",
            "Independent Detector Instances"
        ]
    })


@app.route("/set_fence", methods=["POST"])
def set_fence():
    """
    设置围栏并启动检测流

    V2优化:
    - 独立检测器实例，避免线程安全问题
    - 有界队列防止内存无限增长
    - 环形缓冲区减少内存拷贝
    """
    system_state.total_requests += 1
    data = request.get_json() or {}

    algorithm_type = data.get("algorithmType")
    rtsp_url = data.get("url")
    rect = data.get("fence_area")
    default_area = data.get("default_area", {"width": 960, "height": 540})
    camera_id = str(data.get("cam_id"))

    if not all([algorithm_type, rtsp_url, camera_id]):
        return jsonify({
            "status": "error",
            "message": "缺少必要参数: algorithmType, url, cam_id"
        }), 400

    algo_map = {1: "1", 2: "2", 3: "3", 4: "4"}
    algo_type_str = algo_map.get(algorithm_type)
    if not algo_type_str:
        return jsonify({
            "status": "error",
            "message": f"无效的 algorithmType: {algorithm_type}"
        }), 400

    logger.info(f"[API] 请求启动 camera_id={camera_id}, algo={algo_type_str}")

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({
                "status": "error",
                "message": "无法打开RTSP流"
            }), 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not width or not height:
            return jsonify({
                "status": "error",
                "message": "无法获取视频分辨率"
            }), 500

        logger.info(f"[API] 视频分辨率: {width}x{height}")

        if rect:
            fence_area = rect_to_polygon(rect, default_area, width, height)
        else:
            fence_area = [[0, 0], [width, 0], [width, height], [0, height]]

        logger.info(f"[API] 围栏坐标: {fence_area}")

        with system_state.lock:
            if camera_id in system_state.active_fence_streams:
                logger.info(f"[API] 摄像头 {camera_id} 已存在，停止旧流")
                stop_fence_worker(camera_id)
                system_state.stream_manager.remove_stream(camera_id)
                time.sleep(0.5)

            # V2: 每个流独立创建检测器实例，围栏配置通过其他方式传递
            # 注意：V2版本需要改进围栏配置的传递方式
            # 这里简化处理，围栏配置需要在检测器工厂中处理

            config = system_state.config
            stream_config = StreamConfig(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                ip_address="",
                algorithm_types={algo_type_str},
                fps=config.stream.fps,
                skip_frames=config.stream.skip_frames,
                max_reconnect=config.stream.max_reconnect,
                pre_buffer_seconds=config.stream.pre_buffer_seconds,
                enable_display=False,
                frame_queue_size=10,  # V2: 有界队列
                detection_queue_size=5
            )

            success = system_state.stream_manager.add_stream(stream_config)
            if not success:
                return jsonify({
                    "status": "error",
                    "message": "启动检测流失败"
                }), 500

            logger.info(f"[API] 检测流已启动 camera_id={camera_id}")

            # 启动画框进程
            p = Process(
                target=fence_worker,
                args=(camera_id, rtsp_url, fence_area),
                daemon=True
            )
            p.start()

            system_state.active_fence_streams[camera_id] = {
                "process": p,
                "url": rtsp_url,
                "algorithm_type": algo_type_str,
                "fence_area": fence_area
            }
            system_state.fence_dict[camera_id] = fence_area

        output_url = f"rtsp://192.168.1.61:554/Streaming/Channels/{camera_id}"
        logger.info(f"[API] 画框流输出地址: {output_url}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id,
            "algorithm_type": algo_type_str,
            "output_url": output_url,
            "detection_started": True,
            "version": "v2"
        })

    except Exception as e:
        logger.exception(f"处理请求失败: {e}")
        return jsonify({
            "status": "error",
            "message": f"处理请求失败: {str(e)}"
        }), 500


@app.route("/delete_stream", methods=["POST"])
def delete_stream():
    """停止流并删除围栏"""
    system_state.total_requests += 1
    data = request.get_json() or {}
    camera_id = str(data.get("cam_id"))

    if not camera_id:
        return jsonify({
            "status": "error",
            "message": "cam_id 不能为空"
        }), 400

    try:
        with system_state.lock:
            stop_fence_worker(camera_id)
            if system_state.stream_manager:
                system_state.stream_manager.remove_stream(camera_id)

        logger.info(f"[API] 已删除摄像头 {camera_id}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id
        })

    except Exception as e:
        logger.exception(f"删除流失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def get_status():
    """获取系统状态（V2增强版，包含性能指标）"""
    with system_state.lock:
        fence_streams = list(system_state.active_fence_streams.keys())

        # V2: 获取详细的流统计
        detection_stats = {}
        if system_state.stream_manager:
            detection_stats = system_state.stream_manager.get_all_stats()

        # V2: 获取视频服务统计
        video_stats = {}
        if system_state.stream_manager and hasattr(system_state.stream_manager.video, 'get_stats'):
            video_stats = system_state.stream_manager.video.get_stats()

        # 系统运行时间
        uptime = (datetime.now() - system_state.start_time).total_seconds()

    return jsonify({
        "status": "running",
        "version": "v2",
        "uptime_seconds": int(uptime),
        "total_requests": system_state.total_requests,
        "active_fence_streams": fence_streams,
        "stream_count": len(fence_streams),
        "detection_stats": detection_stats,
        "video_service_stats": video_stats,
        "optimizations": {
            "architecture": "Producer-Consumer",
            "frame_queue": "Bounded (10 frames)",
            "buffer": "Circular (reduced memory copy)",
            "video_generation": "ThreadPool (max 3 concurrent)",
            "detector_instances": "Independent per stream"
        }
    })


@app.route("/performance", methods=["GET"])
def get_performance():
    """获取详细的性能指标"""
    stats = {}
    if system_state.stream_manager:
        for cam_id, cam_stats in system_state.stream_manager.get_all_stats().items():
            stats[cam_id] = {
                "fps": cam_stats.get("fps", 0),
                "detection_fps": cam_stats.get("detection_fps", 0),
                "frame_count": cam_stats.get("frame_count", 0),
                "detection_count": cam_stats.get("detection_count", 0),
                "dropped_frames": cam_stats.get("dropped_frames", 0),
                "avg_detection_latency_ms": cam_stats.get("avg_detection_latency_ms", 0),
                "error_count": cam_stats.get("error_count", 0)
            }

    return jsonify({
        "streams": stats,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime": int((datetime.now() - system_state.start_time).total_seconds())
        }
    })


# =========================
# 信号处理
# =========================
def force_exit(signum, frame):
    """强制退出处理"""
    logger.info("[Main] FORCE EXIT...")
    try:
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except:
                pass
    except:
        pass
    os._exit(1)


import signal
signal.signal(signal.SIGINT, force_exit)
signal.signal(signal.SIGTERM, force_exit)


# =========================
# 主程序
# =========================
def main():
    """主函数"""
    stream_manager, _ = initialize_system()
    system_state.stream_manager = stream_manager

    port = 5005

    logger.info(f"[Main] Starting API server V2 on port {port}")
    logger.info(f"[Main] API endpoints:")
    logger.info(f"  - POST http://0.0.0.0:{port}/set_fence")
    logger.info(f"  - POST http://0.0.0.0:{port}/delete_stream")
    logger.info(f"  - GET  http://0.0.0.0:{port}/status")
    logger.info(f"  - GET  http://0.0.0.0:{port}/performance")
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
