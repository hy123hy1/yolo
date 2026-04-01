"""
Stream Processor - 流处理器
重构后的RTSP流处理架构
"""
import os
import sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;1000000|err_detect;ignore_err|max_delay;300000"

from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import threading
import time
import traceback

import cv2
import numpy as np

from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle, DetectionEvent
)
from video_analytics.services.storage_service import BaseStorageService
from video_analytics.services.alarm_service import BaseAlarmService
from video_analytics.services.video_service import VideoService
from video_analytics.core.state_machine import EventState


class StreamState(Enum):
    """流状态"""
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """流配置"""
    camera_id: str
    rtsp_url: str
    ip_address: str
    algorithm_types: Set[str] = field(default_factory=set)  # "1", "2", "3"
    fps: int = 25
    skip_frames: int = 0          # 跳帧数 (每N帧检测一次)
    max_reconnect: int = 5
    reconnect_interval: float = 2.0
    reconnect_backoff: float = 2.0
    pre_buffer_seconds: int = 3
    enable_display: bool = False


@dataclass
class StreamStats:
    """流统计信息"""
    frame_count: int = 0
    detection_count: int = 0
    event_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    start_time: Optional[datetime] = None
    last_frame_time: Optional[datetime] = None
    fps: float = 0.0


class StreamProcessor:
    """
    流处理器

    职责:
    - 管理RTSP连接和重连
    - 帧缓存管理
    - 调用检测器管道
    - 事件处理和报警
    - 视频生成和上传

    架构:
    ```
    RTSP Stream -> Frame Buffer -> Detector Pipeline -> Event Handler -> Alarm/Video
                       ^              (1,2,3号算法)
                       |
                   Reconnect
    ```
    """

    def __init__(
        self,
        config: StreamConfig,
        detectors: Dict[str, BaseDetector],
        storage_service: BaseStorageService,
        alarm_service: BaseAlarmService,
        video_service: VideoService,
        on_event: Optional[Callable[[str, DetectionEvent, DetectionResultBundle], None]] = None
    ):
        """
        初始化流处理器

        Args:
            config: 流配置
            detectors: 检测器字典 {"1": detector1, "2": detector2, ...}
            storage_service: 存储服务
            alarm_service: 报警服务
            video_service: 视频服务
            on_event: 事件回调函数 (camera_id, event, result)
        """
        self.config = config
        self.detectors = detectors
        self.storage = storage_service
        self.alarm = alarm_service
        self.video = video_service
        self.on_event = on_event

        # 状态
        self._state = StreamState.IDLE
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # 帧缓存
        self._frame_buffer: deque = deque(
            maxlen=config.pre_buffer_seconds * config.fps
        )

        # 统计
        self.stats = StreamStats()

        # 活跃事件跟踪 (用于视频生成)
        self._active_events: Dict[str, Dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._state in [StreamState.RUNNING, StreamState.RECONNECTING]

    def start(self) -> bool:
        """
        启动流处理

        Returns:
            bool: 是否成功启动
        """
        if self.is_running:
            print(f"[StreamProcessor] Already running: {self.config.camera_id}")
            return True

        # 检查RTSP可用性
        if not self._check_rtsp_available(self.config.rtsp_url):
            print(f"[StreamProcessor] RTSP not available: {self.config.rtsp_url}")
            self._state = StreamState.ERROR
            return False

        self._stop_event.clear()
        self._state = StreamState.CONNECTING
        self.stats.start_time = datetime.now()

        self._thread = threading.Thread(
            target=self._run,
            name=f"Stream-{self.config.camera_id}",
            daemon=True
        )
        self._thread.start()

        print(f"[StreamProcessor] Started: {self.config.camera_id}")
        return True

    def stop(self, timeout: float = 5.0):
        """
        停止流处理

        Args:
            timeout: 等待超时(秒)
        """
        if not self.is_running:
            return

        print(f"[StreamProcessor] Stopping: {self.config.camera_id}")
        self._stop_event.set()
        self._state = StreamState.STOPPED

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        print(f"[StreamProcessor] Stopped: {self.config.camera_id}")

    def _run(self):
        """主运行循环"""
        reconnect_count = 0
        backoff = self.config.reconnect_interval

        while not self._stop_event.is_set():
            try:
                # 连接流
                cap = self._connect()
                if cap is None:
                    # 连接失败，尝试重连
                    reconnect_count += 1
                    if reconnect_count > self.config.max_reconnect:
                        print(f"[StreamProcessor] Max reconnect exceeded: {self.config.camera_id}")
                        self._state = StreamState.ERROR
                        break

                    time.sleep(min(backoff, 30))
                    backoff *= self.config.reconnect_backoff
                    continue

                # 重置重连计数
                reconnect_count = 0
                backoff = self.config.reconnect_interval
                self._state = StreamState.RUNNING

                # 处理帧
                self._process_stream(cap)

                cap.release()

            except Exception as e:
                print(f"[StreamProcessor] Error in {self.config.camera_id}: {e}")
                traceback.print_exc()
                self.stats.error_count += 1
                time.sleep(1)

        self._state = StreamState.STOPPED

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """连接RTSP流"""
        self._state = StreamState.CONNECTING
        print(f"[StreamProcessor] Connecting: {self.config.rtsp_url}")

        cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"[StreamProcessor] Connection failed: {self.config.rtsp_url}")
            return None

        print(f"[StreamProcessor] Connected: {self.config.camera_id}")
        return cap

    def _process_stream(self, cap: cv2.VideoCapture):
        """处理视频流"""
        frame_counter = 0
        last_fps_time = time.time()
        fps_frame_count = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"[StreamProcessor] Frame read failed: {self.config.camera_id}")
                self._state = StreamState.RECONNECTING
                break

            # 更新统计
            frame_counter += 1
            fps_frame_count += 1
            self.stats.frame_count += 1
            self.stats.last_frame_time = datetime.now()

            # 更新FPS
            now = time.time()
            if now - last_fps_time >= 1.0:
                self.stats.fps = fps_frame_count / (now - last_fps_time)
                fps_frame_count = 0
                last_fps_time = now

            # 添加到帧缓存
            self._frame_buffer.append(frame.copy())

            # 跳帧检测
            if frame_counter % (self.config.skip_frames + 1) != 0:
                continue

            # 执行检测
            self._process_frame(frame)

            # 显示 (调试用)
            if self.config.enable_display:
                cv2.imshow(f"Camera {self.config.camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def _process_frame(self, frame: np.ndarray):
        """处理单帧"""
        # 创建检测上下文
        context = DetectionContext(
            camera_id=self.config.camera_id,
            rtsp_url=self.config.rtsp_url,
            ip_address=self.config.ip_address,
            frame=frame,
            timestamp=datetime.now(),
            frame_buffer=list(self._frame_buffer),
            fps=self.stats.fps or self.config.fps
        )

        # 根据算法类型执行不同检测器
        for algo_type in self.config.algorithm_types:
            detector = self.detectors.get(algo_type)
            if detector is None:
                print(f"[StreamProcessor] Warning: Detector {algo_type} not found")
                continue

            try:
                result = detector.process(context)
                self.stats.detection_count += 1

                # 调试输出 (每100帧打印一次)
                if self.stats.frame_count % 100 == 0:
                    debug_info = result.debug_info if hasattr(result, 'debug_info') else {}
                    print(f"[Debug][Camera {self.config.camera_id}][Algo {algo_type}] "
                          f"triggered={result.triggered}, "
                          f"detections={len(result.detections)}, "
                          f"debug={debug_info}")

                # 处理事件
                if result.triggered and result.event:
                    self._handle_event(algo_type, result, context)

            except Exception as e:
                print(f"[StreamProcessor] Detector error {algo_type}: {e}")
                traceback.print_exc()

    def _handle_event(
        self,
        algo_type: str,
        result: DetectionResultBundle,
        context: DetectionContext
    ):
        """处理检测事件"""
        event = result.event
        if event is None:
            return

        print(f"[StreamProcessor] Event triggered: {algo_type} - {event.event_type.value}")
        self.stats.event_count += 1

        # 上传图片
        if result.visualized_frame is not None:
            upload_result = self.storage.upload_image(
                result.visualized_frame,
                self.config.camera_id,
                label=event.event_type.value
            )
            if upload_result.success:
                event.image_url = upload_result.url

        # 异步生成视频
        self._async_generate_video(algo_type, event, context)

        # 发送报警
        self.alarm.send_alarm(event)

        # 回调
        if self.on_event:
            try:
                self.on_event(self.config.camera_id, event, result)
            except Exception as e:
                print(f"[StreamProcessor] Event callback error: {e}")

    def _async_generate_video(
        self,
        algo_type: str,
        event: DetectionEvent,
        context: DetectionContext
    ):
        """异步生成视频"""
        def on_video_complete(upload_result):
            if upload_result.success:
                event.video_url = upload_result.url
                print(f"[StreamProcessor] Video uploaded: {upload_result.url}")
            else:
                print(f"[StreamProcessor] Video upload failed: {upload_result.error_message}")

        self.video.async_generate_and_upload(
            pre_frames=context.frame_buffer,
            rtsp_url=self.config.rtsp_url,
            camera_id=self.config.camera_id,
            on_complete=on_video_complete,
            fps=self.config.fps
        )

    def _check_rtsp_available(self, rtsp_url: str, timeout: float = 5.0) -> bool:
        """检查RTSP流是否可用"""
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))

            start = time.time()
            while time.time() - start < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return True
                time.sleep(0.1)

            cap.release()
            return False

        except Exception as e:
            print(f"[StreamProcessor] RTSP check error: {e}")
            return False

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "camera_id": self.config.camera_id,
            "state": self._state.value,
            "frame_count": self.stats.frame_count,
            "detection_count": self.stats.detection_count,
            "event_count": self.stats.event_count,
            "error_count": self.stats.error_count,
            "fps": self.stats.fps,
            "run_time": (datetime.now() - self.stats.start_time).total_seconds() if self.stats.start_time else 0
        }


class StreamManager:
    """
    流管理器
    管理多个StreamProcessor
    """

    def __init__(
        self,
        storage_service: BaseStorageService,
        alarm_service: BaseAlarmService,
        video_service: VideoService
    ):
        self.storage = storage_service
        self.alarm = alarm_service
        self.video = video_service

        # 流处理器字典 {camera_id: StreamProcessor}
        self._processors: Dict[str, StreamProcessor] = {}

        # 检测器模板 {algo_type: detector}
        self._detector_templates: Dict[str, BaseDetector] = {}

        # 锁
        self._lock = threading.Lock()

    def register_detector(self, algo_type: str, detector: BaseDetector):
        """注册检测器模板"""
        self._detector_templates[algo_type] = detector

    def add_stream(self, config: StreamConfig) -> bool:
        """添加流"""
        with self._lock:
            if config.camera_id in self._processors:
                print(f"[StreamManager] Stream already exists: {config.camera_id}")
                return False

            # 为当前流创建检测器实例 (每个流独立的状态机)
            detectors = {}
            for algo_type in config.algorithm_types:
                template = self._detector_templates.get(algo_type)
                if template:
                    # 注意：这里应该创建新的实例或确保线程安全
                    # 简化处理：直接使用模板 (假设检测器是状态less的，除了状态机)
                    detectors[algo_type] = template

            processor = StreamProcessor(
                config=config,
                detectors=detectors,
                storage_service=self.storage,
                alarm_service=self.alarm,
                video_service=self.video,
                on_event=self._on_event
            )

            self._processors[config.camera_id] = processor

            # 启动
            return processor.start()

    def remove_stream(self, camera_id: str, timeout: float = 5.0):
        """移除流"""
        with self._lock:
            processor = self._processors.pop(camera_id, None)
            if processor:
                processor.stop(timeout)
                print(f"[StreamManager] Stream removed: {camera_id}")

    def update_stream(self, config: StreamConfig):
        """更新流配置 (先移除再添加)"""
        self.remove_stream(config.camera_id)
        time.sleep(0.5)
        self.add_stream(config)

    def get_stream_stats(self, camera_id: str) -> Optional[dict]:
        """获取流统计"""
        processor = self._processors.get(camera_id)
        if processor:
            return processor.get_stats()
        return None

    def get_all_stats(self) -> Dict[str, dict]:
        """获取所有流统计"""
        return {
            cid: proc.get_stats()
            for cid, proc in self._processors.items()
        }

    def stop_all(self, timeout: float = 10.0):
        """停止所有流"""
        with self._lock:
            for camera_id, processor in list(self._processors.items()):
                processor.stop(timeout)
            self._processors.clear()

    def _on_event(self, camera_id: str, event: DetectionEvent, result: DetectionResultBundle):
        """事件回调"""
        # 可以在这里添加全局事件处理逻辑
        pass
