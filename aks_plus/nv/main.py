"""
Video Analytics System - Main Entry Point
NVIDIA GPU Edition (TensorRT/ONNX/PyTorch)
"""
import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import List, Tuple, Set, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import cv2

from video_analytics.config.settings import AppConfig, default_config
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
# 配置加载
# =========================
def load_config(path: str = "./cfg/config.json") -> AppConfig:
    """加载配置文件"""
    config = AppConfig.from_file(path)

    # 从旧版配置文件加载报警端点
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
            if "server_endpoints4" in old_config:
                config.alarm.endpoints = old_config["server_endpoints4"]
        except Exception as e:
            print(f"[Config] Failed to load old config: {e}")

    return config


# =========================
# 摄像头API
# =========================
def get_cameras(api_url: str = "http://172.21.3.141:8080/aks-mkaqjcyj/cameraMonitoring/selectSxtInfo"
                ) -> List[Tuple[str, str, str, Set[str]]]:
    """
    从API获取摄像头列表

    Returns:
        List[(camera_id, rtsp_url, ip_address, algorithm_types)]
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(api_url, headers=headers, timeout=10)

        if resp.status_code != 200:
            print(f"[CameraAPI] Request failed: {resp.status_code}")
            return []

        data = resp.json()
        cameras_data = data.get("data", [])
        print(f"[CameraAPI] Found {len(cameras_data)} camera configs")

        valid_cameras = []
        for item in cameras_data:
            algorithmtypes = item.get("algorithmtypes")
            if algorithmtypes:
                # 解析算法类型 "1,2" -> {"1", "2"}
                algo_set = set(algorithmtypes.split(","))
                valid_cameras.append((
                    str(item.get("id")),
                    item.get("originalRtsp"),
                    item.get("ipAddress", ""),
                    algo_set
                ))

        return valid_cameras

    except Exception as e:
        print(f"[CameraAPI] Error: {e}")
        return []


# =========================
# 系统初始化
# =========================
def initialize_system(config: AppConfig) -> Tuple[StreamManager, dict]:
    """
    初始化系统组件

    Returns:
        (stream_manager, detectors_dict)
    """
    print("=" * 60)
    print("[System] Initializing Video Analytics System")
    print(f"[System] Backend: {config.model.backend}")
    print("=" * 60)

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

    # 预热
    print("[System] Warming up engines...")
    person_engine.warmup(num_runs=5)

    # 安全帽检测引擎 (使用自定义训练的模型，类别为 helmet, head)
    if config.model.helmet_model_path and os.path.exists(config.model.helmet_model_path):
        # 安全帽模型使用Ultralytics引擎，类别为 helmet(0), head(1)
        helmet_engine = create_infer_engine(
            model_path=config.model.helmet_model_path,
            backend='ultralytics',  # 强制使用Ultralytics接口
            input_size=config.model.input_size,
            confidence=0.3,  # 安全帽检测使用较低置信度
            classes=SAFETY_HELMET_CLASSES,  # {0: 'helmet', 1: 'head'}
            device_id=config.model.device_id,
            verbose=False
        )
        helmet_engine.warmup(num_runs=3)
        print("[System] Helmet detection engine loaded with classes: helmet, head")
    else:
        # 复用人员检测模型
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
        # base_path=config.storage.local_path
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

    # 注意：如果需要设置电子围栏，请在此处添加
    # 示例：为摄像头设置闯入检测围栏
    # intrusion_detector = detectors.get("1")
    # if intrusion_detector:
    #     # 方法1: 使用多边形顶点 (支持任意形状)
    #     intrusion_detector.set_fence_from_points(
    #         camera_id="your_camera_id",  # 替换为实际摄像头ID
    #         points=[(100, 100), (500, 100), (500, 400), (100, 400)]  # 左上角顺时针
    #     )
    #     # 方法2: 使用预定义区域 (画面对角线的10%~90%)
    #     # intrusion_detector.set_fence_full_frame("your_camera_id", margin=0.1)

    print("[System] Initialization complete!")
    print("=" * 60)

    return stream_manager, detectors


# =========================
# 主程序
# =========================
def main():
    """主函数"""
    # 加载配置
    config = load_config("./cfg/config.json")

    # 初始化系统
    stream_manager, detectors = initialize_system(config)

    # 围栏配置示例 (实际应从API或配置文件加载)
    # intrusion_detector = detectors["1"]
    # intrusion_detector.set_fence_from_points(
    #     camera_id="camera_001",
    #     points=[(100, 100), (500, 100), (500, 400), (100, 400)]
    # )

    # 摄像头轮询配置
    poll_interval = 600  # 10分钟
    bad_camera_timeout = 900  # 15分钟
    bad_cameras = {}

    print("[Main] Starting camera poll loop...")

    try:
        while True:
            try:
                # 获取摄像头列表
                # cameras = get_cameras()
                cameras = [('1997855044199911425', 'rtsp://admin:sxhbjq123@192.168.1.2:554/Streaming/Channels/101', '192.168.1.2', '1')]
                print(f"[Main] Found {len(cameras)} valid cameras")

                # 处理每个摄像头
                for camera_id, rtsp_url, ip_address, algo_types in cameras:
                    # 跳过坏流
                    if camera_id in bad_cameras:
                        if time.time() - bad_cameras[camera_id] < bad_camera_timeout:
                            continue
                        else:
                            del bad_cameras[camera_id]

                    # 检查流是否已在运行
                    stats = stream_manager.get_stream_stats(camera_id)
                    if stats is None:
                        # 新流，启动
                        stream_config = StreamConfig(
                            camera_id=camera_id,
                            rtsp_url=rtsp_url,
                            ip_address=ip_address,
                            algorithm_types=algo_types,
                            fps=config.stream.fps,
                            skip_frames=config.stream.skip_frames,
                            max_reconnect=config.stream.max_reconnect,
                            pre_buffer_seconds=config.stream.pre_buffer_seconds,
                            enable_display=config.stream.enable_display
                        )

                        success = stream_manager.add_stream(stream_config)
                        if not success:
                            bad_cameras[camera_id] = time.time()
                            print(f"[Main] Marked as bad camera: {camera_id}")

                # 清理已停止的流
                all_stats = stream_manager.get_all_stats()
                current_ids = {c[0] for c in cameras}
                for camera_id in list(all_stats.keys()):
                    if camera_id not in current_ids:
                        stream_manager.remove_stream(camera_id)
                        print(f"[Main] Removed stopped stream: {camera_id}")

                # 打印统计
                print("\n[Stats] Current streams:")
                for cid, stat in all_stats.items():
                    print(f"  {cid}: state={stat['state']}, frames={stat['frame_count']}, "
                          f"events={stat['event_count']}, fps={stat['fps']:.1f}")

            except Exception as e:
                print(f"[Main] Poll loop error: {e}")

            # 等待下一轮
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")

    finally:
        # 停止所有流
        stream_manager.stop_all()
        print("[Main] System shutdown complete")


if __name__ == "__main__":
    main()
