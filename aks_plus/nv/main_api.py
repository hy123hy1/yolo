"""
Video Analytics System with Fence API
集画框服务与目标检测服务于一体

API接口:
- POST /set_fence: 设置围栏并启动检测流
- POST /delete_stream: 停止流并删除围栏
"""
import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from multiprocessing import Process
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import psutil
from flask import Flask, request, jsonify
from flask_cors import CORS

from video_analytics.config.settings import AppConfig
from video_analytics.engines.factory import create_infer_engine
from video_analytics.engines.ultralytics_engine import YOLOV8_CLASSES, SAFETY_HELMET_CLASSES
from video_analytics.detectors.intrusion_detector import IntrusionDetector, FenceRegion
from video_analytics.detectors.helmet_detector import HelmetDetector
from video_analytics.detectors.overcrowd_detector import OvercrowdDetector
from video_analytics.core.stream_processor import StreamManager, StreamConfig
from video_analytics.services.storage_service import StorageServiceFactory
from video_analytics.services.alarm_service import AlarmServiceFactory
from video_analytics.services.video_service import VideoService, VideoConfig


# =========================
# 配置
# =========================
CONFIG_PATH = "./cfg/config.json"
app = Flask(__name__)
CORS(app)

# 全局状态
class SystemState:
    def __init__(self):
        self.stream_manager: Optional[StreamManager] = None
        self.detectors: Dict[str, any] = {}
        self.config: Optional[AppConfig] = None
        self.active_fence_streams: Dict[str, Dict] = {}  # camera_id -> 画框进程信息
        self.fence_dict: Dict[str, List] = {}  # camera_id -> fence_area points
        self.lock = threading.Lock()

system_state = SystemState()


# =========================
# 画框服务相关 (从 push.py 迁移)
# =========================
class FFmpegCapture:
    """FFmpeg拉流捕获 - 增强容错版"""
    def __init__(self, rtsp_url: str, width: int, height: int, init_wait: float = 3.0):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.start()
        # 等待ffmpeg稳定输出
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
            "-probesize", "32",      # 减少探测时间
            "-analyzeduration", "0",  # 减少分析时间
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
        """读取帧，支持重试"""
        for attempt in range(retry):
            try:
                # 检查进程是否还在运行
                if self.process.poll() is not None:
                    return False, None

                # Windows不支持select用于管道，使用直接读取
                raw = self.process.stdout.read(self.frame_size)

                if len(raw) == self.frame_size:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    ).copy()
                    return True, frame
                elif len(raw) == 0:
                    # ffmpeg可能结束了或还没准备好
                    time.sleep(0.05)
                    continue
                else:
                    # 数据不完整，丢弃并等待
                    time.sleep(0.05)
                    continue
            except Exception as e:
                print(f"[FFmpegCapture] read error: {e}")
                return False, None

        return False, None

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()
            self.process = None

    def is_healthy(self) -> bool:
        """检查进程是否健康"""
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
    """
    将矩形转换为多边形点坐标

    Args:
        rect: {"x": int, "y": int, "width": int, "height": int}
        default_area: {"width": int, "height": int} - 前端基准尺寸
        frame_width: 实际视频宽度
        frame_height: 实际视频高度

    Returns:
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 顺时针从左上开始
    """
    base_width = default_area.get("width", 960)
    base_height = default_area.get("height", 540)

    # 按实际视频大小映射
    scale_x = frame_width / base_width
    scale_y = frame_height / base_height

    x = rect["x"] * scale_x
    y = rect["y"] * scale_y
    w = rect["width"] * scale_x
    h = rect["height"] * scale_y

    return [
        [int(x), int(y)],          # 左上
        [int(x + w), int(y)],      # 右上
        [int(x + w), int(y + h)],  # 右下
        [int(x), int(y + h)]       # 左下
    ]


def fence_worker(camera_id: str, rtsp_url: str, fence_area: List, output_host: str = "192.168.1.61"):
    """
    画框工作进程 - 拉流、画红框、推流

    Args:
        camera_id: 摄像头ID
        rtsp_url: 原始RTSP流地址
        fence_area: 围栏多边形点坐标 [[x1,y1], [x2,y2], ...]
        output_host: 推流目标主机
    """
    import queue

    print(f"[FenceWorker] 启动画框进程 camera_id={camera_id}")

    try:
        width, height = get_stream_size(rtsp_url)
        cap = FFmpegCapture(rtsp_url, width, height, init_wait=3.0)
        fps = 25

        rtsp_push = f"rtsp://{output_host}:554/Streaming/Channels/{camera_id}"

        # 启动推流FFmpeg
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
        frame_queue = queue.Queue(maxsize=50)
        stop_flag = threading.Event()

        def read_loop():
            bad_count = 0
            reconnect_count = 0
            nonlocal cap, push_proc

            # 连续成功计数，用于确认流稳定
            success_count = 0

            while not stop_flag.is_set():
                ret, frame = cap.read(retry=3)

                if not ret or frame is None:
                    bad_count += 1

                    # 每30秒打印一次日志，避免刷屏
                    if bad_count % 25 == 0:
                        healthy = cap.is_healthy()
                        print(f"[FenceWorker] {camera_id} 读取帧失败({bad_count}), 进程健康={healthy}")

                    # 增加阈值到50（约2秒），给RTSP流更多缓冲时间
                    if bad_count > 50 or not cap.is_healthy():
                        print(f"[FenceWorker] {camera_id} 拉流异常，准备重启... (bad_count={bad_count})")
                        cap.release()
                        time.sleep(min(2 ** reconnect_count, 10))
                        reconnect_count += 1
                        print(f"[FenceWorker] {camera_id} 第{reconnect_count}次重连...")
                        cap = FFmpegCapture(rtsp_url, width, height, init_wait=2.0)
                        bad_count = 0
                        success_count = 0
                    continue

                # 成功读取
                bad_count = 0
                success_count += 1

                # 成功读取一定帧数后重置重连计数（表示流已稳定）
                if success_count > 100:
                    if reconnect_count > 0:
                        print(f"[FenceWorker] {camera_id} 流已稳定，重置重连计数")
                    reconnect_count = 0
                    success_count = 0

                # 画围栏红框
                pts = np.array(fence_area, np.int32)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

                # 丢帧策略入队
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame)

        def push_loop():
            nonlocal push_proc
            while not stop_flag.is_set():
                try:
                    frame = frame_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if push_proc.poll() is not None:
                    print(f"[FenceWorker] 推流FFmpeg挂了，重启 {camera_id}")
                    push_proc = start_ffmpeg()
                    continue

                try:
                    push_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    push_proc = start_ffmpeg()
                except Exception as e:
                    print(f"[FenceWorker] 推流异常 {camera_id}: {e}")
                    push_proc = start_ffmpeg()

        # 启动线程
        t1 = threading.Thread(target=read_loop, daemon=True)
        t2 = threading.Thread(target=push_loop, daemon=True)
        t1.start()
        t2.start()

        # 保持进程运行
        while not stop_flag.is_set():
            time.sleep(1)

        # 清理
        stop_flag.set()
        cap.release()
        try:
            push_proc.kill()
        except:
            pass

    except Exception as e:
        print(f"[FenceWorker] 异常 {camera_id}: {e}")
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

            # 清理残留ffmpeg进程
            try:
                for p in psutil.process_iter(["pid", "name", "cmdline"]):
                    if "ffmpeg" in p.info["name"] and str(camera_id) in " ".join(p.info["cmdline"] or []):
                        p.kill()
            except:
                pass

            system_state.active_fence_streams.pop(camera_id, None)
            system_state.fence_dict.pop(camera_id, None)
            print(f"[FenceWorker] 已停止 {camera_id}")


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
            print(f"[Config] Failed to load old config: {e}")

    return config


def initialize_system() -> Tuple[StreamManager, Dict]:
    """初始化系统组件"""
    print("=" * 60)
    print("[System] Initializing Video Analytics System with API")
    print("=" * 60)

    config = load_config()
    system_state.config = config

    # 1. 创建推理引擎
    print("[System] Creating inference engines...")

    person_engine = create_infer_engine(
        model_path=config.model.person_model_path,
        backend=config.model.backend,
        input_size=config.model.input_size,
        confidence=config.model.confidence,
        iou=config.model.iou,
        classes=YOLOV8_CLASSES,
        device_id=config.model.device_id,
        fp16=config.model.fp16
    )
    person_engine.warmup(num_runs=5)

    # 安全帽检测引擎
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
        helmet_engine.warmup(num_runs=3)
        print("[System] Helmet detection engine loaded")
    else:
        helmet_engine = person_engine
        print("[System] Using person engine for helmet detection")

    # 2. 创建检测器
    print("[System] Creating detectors...")

    detectors = {}

    # 闯入检测器 (1号算法)
    intrusion_detector = IntrusionDetector(
        engine=person_engine,
        config={
            "min_frames": config.detection.intrusion_min_frames,
            "confidence": config.detection.intrusion_confidence,
            "cooldown_seconds": config.detection.intrusion_cooldown,
            "target_classes": [0]
        }
    )
    detectors["1"] = intrusion_detector

    # 安全帽检测器 (2号算法)
    helmet_detector = HelmetDetector(
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
    detectors["2"] = helmet_detector

    # 超员检测器 (3号算法)
    overcrowd_detector = OvercrowdDetector(
        engine=person_engine,
        config={
            "max_people": config.detection.overcrowd_max_people,
            "duration_threshold": config.detection.overcrowd_duration,
            "cooldown_seconds": config.detection.overcrowd_cooldown,
            "confidence": config.detection.overcrowd_confidence
        }
    )
    detectors["3"] = overcrowd_detector

    # 3. 创建服务
    print("[System] Creating services...")

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

    video_service = VideoService(
        storage_service=storage_service,
        config=VideoConfig(
            fps=config.stream.fps,
            pre_seconds=config.stream.pre_buffer_seconds,
            post_seconds=config.stream.post_record_seconds
        )
    )

    # 4. 创建流管理器
    print("[System] Creating stream manager...")
    stream_manager = StreamManager(
        storage_service=storage_service,
        alarm_service=alarm_service,
        video_service=video_service
    )

    # 注册检测器
    for algo_type, detector in detectors.items():
        stream_manager.register_detector(algo_type, detector)

    print("[System] Initialization complete!")
    print("=" * 60)

    return stream_manager, detectors


# =========================
# Flask API 路由
# =========================

@app.route("/")
def index():
    """根路由"""
    return jsonify({
        "status": "running",
        "service": "Video Analytics with Fence API",
        "version": "2.0"
    })


@app.route("/set_fence", methods=["POST"])
def set_fence():
    """
    设置围栏并启动检测流

    Request Body:
    {
        "fence_area": {"x": 350, "y": 350, "width": 600, "height": 300},
        "default_area": {"width": 960, "height": 540},
        "url": "rtsp://...",
        "algorithmType": 2,  // 1=闯入, 2=安全帽, 3=超员
        "cam_id": "2027671157372751872"
    }
    """
    data = request.get_json() or {}

    # 参数解析
    algorithm_type = data.get("algorithmType")
    rtsp_url = data.get("url")
    rect = data.get("fence_area")
    default_area = data.get("default_area", {"width": 960, "height": 540})
    camera_id = str(data.get("cam_id"))

    # 参数校验
    if not all([algorithm_type, rtsp_url, camera_id]):
        return jsonify({
            "status": "error",
            "message": "缺少必要参数: algorithmType, url, cam_id"
        }), 400

    # 算法类型映射: 前端 1,2,3 -> 内部 "1","2","3"
    algo_map = {1: "1", 2: "2", 3: "3"}
    algo_type_str = algo_map.get(algorithm_type)
    if not algo_type_str:
        return jsonify({
            "status": "error",
            "message": f"无效的 algorithmType: {algorithm_type}, 应为 1, 2 或 3"
        }), 400

    print(f"[API] 请求启动 camera_id={camera_id}, algo={algo_type_str}, url={rtsp_url}")

    try:
        # 获取视频分辨率
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

        print(f"[API] 视频分辨率: {width}x{height}")

        # 计算围栏多边形坐标
        if rect:
            fence_area = rect_to_polygon(rect, default_area, width, height)
        else:
            # 默认全图
            fence_area = [[0, 0], [width, 0], [width, height], [0, height]]

        print(f"[API] 围栏坐标: {fence_area}")

        with system_state.lock:
            # 1. 如果已存在，先停止旧流
            if camera_id in system_state.active_fence_streams:
                print(f"[API] 摄像头 {camera_id} 已存在，停止旧流")
                stop_fence_worker(camera_id)
                system_state.stream_manager.remove_stream(camera_id)
                time.sleep(0.5)

            # 2. 设置电子围栏（关键！）
            if algo_type_str == "1" and "1" in system_state.detectors:
                # 闯入检测需要设置围栏
                intrusion_detector = system_state.detectors["1"]
                intrusion_detector.set_fence_from_points(camera_id, fence_area)
                print(f"[API] 已设置闯入检测围栏 camera_id={camera_id}")

            # 3. 启动检测流（事件驱动！无需等待轮询）
            config = system_state.config
            stream_config = StreamConfig(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                ip_address="",  # 可从URL解析或前端传入
                algorithm_types={algo_type_str},
                fps=config.stream.fps,
                skip_frames=config.stream.skip_frames,
                max_reconnect=config.stream.max_reconnect,
                pre_buffer_seconds=config.stream.pre_buffer_seconds,
                enable_display=False
            )

            success = system_state.stream_manager.add_stream(stream_config)
            if not success:
                return jsonify({
                    "status": "error",
                    "message": "启动检测流失败"
                }), 500

            print(f"[API] 检测流已启动 camera_id={camera_id}")

            # 4. 启动画框进程
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
        print(f"[API] 画框流输出地址: {output_url}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id,
            "algorithm_type": algo_type_str,
            "output_url": output_url,
            "detection_started": True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"处理请求失败: {str(e)}"
        }), 500


@app.route("/delete_stream", methods=["POST"])
def delete_stream():
    """
    停止流并删除围栏

    Request Body:
    {
        "cam_id": "2027671157372751872"
    }
    """
    data = request.get_json() or {}
    camera_id = str(data.get("cam_id"))

    if not camera_id:
        return jsonify({
            "status": "error",
            "message": "cam_id 不能为空"
        }), 400

    try:
        with system_state.lock:
            # 1. 停止画框进程
            stop_fence_worker(camera_id)

            # 2. 停止检测流
            if system_state.stream_manager:
                system_state.stream_manager.remove_stream(camera_id)

            # 3. 清除围栏配置
            if "1" in system_state.detectors:
                system_state.detectors["1"].clear_fence(camera_id)

        print(f"[API] 已删除摄像头 {camera_id}")

        return jsonify({
            "status": "success",
            "camera_id": camera_id
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def get_status():
    """获取系统状态"""
    with system_state.lock:
        fence_streams = list(system_state.active_fence_streams.keys())
        detection_stats = system_state.stream_manager.get_all_stats() if system_state.stream_manager else {}

    return jsonify({
        "status": "running",
        "active_fence_streams": fence_streams,
        "detection_stats": detection_stats
    })


# =========================
# 信号处理 - 确保能立即退出
# =========================
def force_exit(signum, frame):
    """强制退出处理"""
    print("\n[Main] FORCE EXIT...")
    # 直接杀死所有子进程
    try:
        import psutil
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except:
                pass
    except:
        pass
    # 立即退出
    os._exit(1)

# 注册信号处理
import signal
signal.signal(signal.SIGINT, force_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, force_exit)  # kill


# =========================
# 主程序
# =========================
def main():
    """主函数"""
    # 初始化系统
    stream_manager, detectors = initialize_system()
    system_state.stream_manager = stream_manager
    system_state.detectors = detectors

    # 启动Flask服务
    config = system_state.config
    port = 5005

    print(f"[Main] Starting API server on port {port}")
    print(f"[Main] API endpoints:")
    print(f"  - POST http://0.0.0.0:{port}/set_fence")
    print(f"  - POST http://0.0.0.0:{port}/delete_stream")
    print(f"  - GET  http://0.0.0.0:{port}/status")
    print("=" * 60)

    # 使用多线程模式支持并发请求
    # 注意：Flask的reloader会干扰信号处理，必须禁用
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
