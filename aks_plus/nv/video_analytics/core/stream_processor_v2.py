"""
Stream Processor V2 - 高性能流处理器
使用生产者-消费者模式解耦帧读取和检测

架构:
```
RTSP Stream -> FrameReader(生产者) -> FrameQueue -> DetectionWorker(消费者) -> EventHandler
                ↓                                              ↓
         FrameBuffer(环形)                          Detector Pipeline
```
"""
import os
import sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;1000000|err_detect;ignore_err|max_delay;300000"

from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import threading
import time
import traceback
import logging

import cv2
import numpy as np

from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle, DetectionEvent
)
from video_analytics.services.storage_service import BaseStorageService
from video_analytics.services.alarm_service import BaseAlarmService
from video_analytics.services.video_service import VideoService
from video_analytics.core.state_machine import EventState

# 配置日志
logger = logging.getLogger(__name__)


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
    algorithm_types: Set[str] = field(default_factory=set)
    fps: int = 25
    skip_frames: int = 0
    max_reconnect: int = 5
    reconnect_interval: float = 2.0
    reconnect_backoff: float = 2.0
    pre_buffer_seconds: int = 3
    enable_display: bool = False
    # 性能优化配置
    frame_queue_size: int = 10  # 帧队列大小（有界队列防止内存无限增长）
    detection_queue_size: int = 5  # 检测队列大小


@dataclass
class StreamStats:
    """流统计信息"""
    frame_count: int = 0
    detection_count: int = 0
    event_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    dropped_frames: int = 0  # 丢弃的帧数（队列满）
    start_time: Optional[datetime] = None
    last_frame_time: Optional[datetime] = None
    last_detection_time: Optional[datetime] = None
    fps: float = 0.0
    detection_fps: float = 0.0  # 检测帧率
    avg_detection_latency: float = 0.0  # 平均检测延迟(ms)


@dataclass
class FramePackage:
    """帧包 - 包含帧数据和元数据"""
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    frame_buffer_snapshot: Optional[List[np.ndarray]] = None  # 懒加载的帧缓冲快照


class CircularFrameBuffer:
    """
    环形帧缓冲区
    使用预分配内存避免频繁分配
    """

    def __init__(self, capacity: int, frame_shape: Optional[Tuple] = None):
        self.capacity = capacity
        self.frame_shape = frame_shape
        self._buffer: List[Optional[np.ndarray]] = [None] * capacity
        self._index = 0
        self._count = 0
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray):
        """添加帧到缓冲区"""
        with self._lock:
            # 如果位置已有数据，先释放
            if self._buffer[self._index] is not None:
                del self._buffer[self._index]

            # 存储帧的浅拷贝（引用），不复制数据
            self._buffer[self._index] = frame
            self._index = (self._index + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)

    def get_snapshot(self) -> List[np.ndarray]:
        """
        获取当前缓冲区的快照
        返回按时间顺序排列的帧列表
        """
        with self._lock:
            if self._count < self.capacity:
                # 缓冲区未满，返回有效部分
                return [f for f in self._buffer[:self._count] if f is not None]
            else:
                # 缓冲区已满，按时间顺序重组
                result = []
                for i in range(self.capacity):
                    idx = (self._index + i) % self.capacity
                    if self._buffer[idx] is not None:
                        result.append(self._buffer[idx])
                return result

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer = [None] * self.capacity
            self._index = 0
            self._count = 0


class StreamProcessorV2:
    """
    高性能流处理器 V2

    核心优化:
    1. 生产者-消费者模式：帧读取和检测解耦
    2. 有界队列：防止内存无限增长
    3. 环形缓冲区：减少内存分配和拷贝
    4. 独立检测器实例：避免共享状态的线程安全问题
    5. 异步事件处理：检测不阻塞帧读取
    """

    def __init__(
        self,
        config: StreamConfig,
        detector_factory: Callable[[str], BaseDetector],  # 工厂函数，创建独立检测器实例
        storage_service: BaseStorageService,
        alarm_service: BaseAlarmService,
        video_service: VideoService,
        on_event: Optional[Callable[[str, DetectionEvent, DetectionResultBundle], None]] = None
    ):
        self.config = config
        self.detector_factory = detector_factory
        self.storage = storage_service
        self.alarm = alarm_service
        self.video = video_service
        self.on_event = on_event

        # 状态
        self._state = StreamState.IDLE
        self._stop_event = threading.Event()

        # 有界队列 - 帧读取 -> 检测
        self._frame_queue: deque = deque(maxlen=config.frame_queue_size)
        self._queue_lock = threading.Lock()
        self._queue_sem = threading.Semaphore(0)  # 用于通知检测线程

        # 环形帧缓冲区（用于视频生成）
        self._frame_buffer: Optional[CircularFrameBuffer] = None

        # 每个算法类型的独立检测器实例
        self._detectors: Dict[str, BaseDetector] = {}

        # 统计
        self.stats = StreamStats()
        self._detection_latency_history: deque = deque(maxlen=100)  # 延迟历史

        # 线程
        self._reader_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None

        # 活跃事件跟踪
        self._active_events: Dict[str, Dict[str, Any]] = {}
        self._events_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._state in [StreamState.RUNNING, StreamState.RECONNECTING]

    def start(self) -> bool:
        """启动流处理"""
        if self.is_running:
            logger.warning(f"Stream already running: {self.config.camera_id}")
            return True

        # 检查RTSP可用性
        if not self._check_rtsp_available(self.config.rtsp_url):
            logger.error(f"RTSP not available: {self.config.rtsp_url}")
            self._state = StreamState.ERROR
            return False

        # 创建独立检测器实例
        self._detectors = {}
        for algo_type in self.config.algorithm_types:
            detector = self.detector_factory(algo_type)
            if detector:
                self._detectors[algo_type] = detector
                logger.info(f"Created detector instance for {algo_type}")

        self._stop_event.clear()
        self._state = StreamState.CONNECTING
        self.stats.start_time = datetime.now()

        # 启动读取线程（生产者）
        self._reader_thread = threading.Thread(
            target=self._frame_reader_loop,
            name=f"FrameReader-{self.config.camera_id}",
            daemon=True
        )
        self._reader_thread.start()

        # 启动检测线程（消费者）
        self._detection_thread = threading.Thread(
            target=self._detection_worker_loop,
            name=f"DetectionWorker-{self.config.camera_id}",
            daemon=True
        )
        self._detection_thread.start()

        logger.info(f"StreamProcessorV2 started: {self.config.camera_id}")
        return True

    def stop(self, timeout: float = 5.0, force: bool = False):
        """停止流处理"""
        if not self.is_running:
            return

        logger.info(f"Stopping stream: {self.config.camera_id}")
        self._stop_event.set()
        self._state = StreamState.STOPPED

        # 等待读取线程
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=timeout / 2)

        # 等待检测线程
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=timeout / 2)

        # 清理资源
        if self._frame_buffer:
            self._frame_buffer.clear()

        # 清理检测器
        self._detectors.clear()

        logger.info(f"Stream stopped: {self.config.camera_id}")

    def _frame_reader_loop(self):
        """帧读取循环（生产者）"""
        reconnect_count = 0
        backoff = self.config.reconnect_interval
        frame_counter = 0
        last_fps_time = time.time()
        fps_frame_count = 0

        # 初始化帧缓冲区（需要知道帧尺寸）
        frame_shape = None

        while not self._stop_event.is_set():
            try:
                # 连接流
                cap = self._connect()
                if cap is None:
                    reconnect_count += 1
                    if reconnect_count > self.config.max_reconnect:
                        logger.error(f"Max reconnect exceeded: {self.config.camera_id}")
                        self._state = StreamState.ERROR
                        break

                    time.sleep(min(backoff, 30))
                    backoff *= self.config.reconnect_backoff
                    continue

                # 获取帧尺寸并初始化缓冲区
                if frame_shape is None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_shape = (height, width, 3)
                    buffer_capacity = self.config.pre_buffer_seconds * self.config.fps
                    self._frame_buffer = CircularFrameBuffer(buffer_capacity, frame_shape)
                    logger.info(f"Frame buffer initialized: {buffer_capacity} frames @ {width}x{height}")

                # 重置重连计数
                reconnect_count = 0
                backoff = self.config.reconnect_interval
                self._state = StreamState.RUNNING

                # 读取帧循环
                while not self._stop_event.is_set():
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        logger.warning(f"Frame read failed: {self.config.camera_id}")
                        self._state = StreamState.RECONNECTING
                        break

                    frame_counter += 1
                    fps_frame_count += 1
                    self.stats.frame_count += 1
                    self.stats.last_frame_time = datetime.now()

                    # 更新FPS统计
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        self.stats.fps = fps_frame_count / (now - last_fps_time)
                        fps_frame_count = 0
                        last_fps_time = now

                    # 添加到环形缓冲区（使用引用，不复制）
                    self._frame_buffer.append(frame)

                    # 跳帧逻辑：只将需要检测的帧放入队列
                    if frame_counter % (self.config.skip_frames + 1) == 0:
                        # 创建帧包（浅拷贝帧数据）
                        frame_pkg = FramePackage(
                            frame=frame,  # 不复制，直接引用
                            timestamp=datetime.now(),
                            frame_number=frame_counter
                        )

                        # 放入有界队列（非阻塞，满则丢弃最旧的）
                        with self._queue_lock:
                            if len(self._frame_queue) >= self.config.frame_queue_size:
                                self._frame_queue.popleft()  # 丢弃最旧的帧
                                self.stats.dropped_frames += 1
                                logger.debug(f"Frame dropped (queue full): {self.config.camera_id}")

                            self._frame_queue.append(frame_pkg)
                            self._queue_sem.release()  # 通知检测线程

                cap.release()

            except Exception as e:
                logger.error(f"Frame reader error: {e}")
                self.stats.error_count += 1
                time.sleep(1)

        self._state = StreamState.STOPPED

    def _detection_worker_loop(self):
        """检测工作循环（消费者）"""
        frame_counter = 0
        last_fps_time = time.time()
        detection_count = 0

        while not self._stop_event.is_set():
            try:
                # 等待帧数据（带超时，方便检查停止信号）
                acquired = self._queue_sem.acquire(timeout=0.5)
                if not acquired:
                    continue

                # 获取帧包
                with self._queue_lock:
                    if not self._frame_queue:
                        continue
                    frame_pkg = self._frame_queue.popleft()

                # 执行检测
                start_time = time.time()
                self._process_frame_package(frame_pkg)
                detection_latency = (time.time() - start_time) * 1000  # ms

                # 更新统计
                detection_count += 1
                self._detection_latency_history.append(detection_latency)

                # 计算检测FPS
                frame_counter += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    self.stats.detection_fps = detection_count / (now - last_fps_time)
                    self.stats.avg_detection_latency = sum(self._detection_latency_history) / len(self._detection_latency_history)
                    detection_count = 0
                    last_fps_time = now

            except Exception as e:
                logger.error(f"Detection worker error: {e}")
                traceback.print_exc()
                self.stats.error_count += 1

    def _process_frame_package(self, frame_pkg: FramePackage):
        """处理帧包"""
        self.stats.detection_count += 1
        self.stats.last_detection_time = datetime.now()

        # 懒加载帧缓冲快照（只在需要时复制）
        frame_buffer_snapshot = None

        # 对每个算法类型执行检测
        for algo_type, detector in self._detectors.items():
            try:
                # 只在需要时获取帧缓冲快照
                if frame_buffer_snapshot is None:
                    frame_buffer_snapshot = self._frame_buffer.get_snapshot()

                # 创建检测上下文
                context = DetectionContext(
                    camera_id=self.config.camera_id,
                    rtsp_url=self.config.rtsp_url,
                    ip_address=self.config.ip_address,
                    frame=frame_pkg.frame,
                    timestamp=frame_pkg.timestamp,
                    frame_buffer=frame_buffer_snapshot,
                    fps=self.stats.fps or self.config.fps
                )

                # 执行检测
                result = detector.process(context)

                # 处理事件
                if result.triggered and result.event:
                    self._handle_event(algo_type, result, context)

            except Exception as e:
                logger.error(f"Detector error {algo_type}: {e}")
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

        logger.info(f"Event triggered: {algo_type} - {event.event_type.value}")
        self.stats.event_count += 1

        # 上传图片（异步，不阻塞）
        if result.visualized_frame is not None:
            try:
                upload_result = self.storage.upload_image(
                    result.visualized_frame,
                    self.config.camera_id,
                    label=event.event_type.value
                )
                if upload_result.success:
                    event.image_url = upload_result.url
            except Exception as e:
                logger.error(f"Image upload failed: {e}")

        # 异步生成视频
        self._async_generate_video(algo_type, event, context)

        # 发送报警
        try:
            self.alarm.send_alarm(event)
        except Exception as e:
            logger.error(f"Alarm send failed: {e}")

        # 回调
        if self.on_event:
            try:
                self.on_event(self.config.camera_id, event, result)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

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
                logger.info(f"Video uploaded: {upload_result.url}")
            else:
                logger.error(f"Video upload failed: {upload_result.error_message}")

        # 复制帧数据（避免引用被修改）
        pre_frames = [f.copy() for f in context.frame_buffer] if context.frame_buffer else []

        self.video.async_generate_and_upload(
            pre_frames=pre_frames,
            rtsp_url=self.config.rtsp_url,
            camera_id=self.config.camera_id,
            on_complete=on_video_complete,
            fps=self.config.fps
        )

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """连接RTSP流"""
        self._state = StreamState.CONNECTING
        logger.info(f"Connecting: {self.config.rtsp_url}")

        cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            logger.error(f"Connection failed: {self.config.rtsp_url}")
            return None

        logger.info(f"Connected: {self.config.camera_id}")
        return cap

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
            logger.error(f"RTSP check error: {e}")
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
            "dropped_frames": self.stats.dropped_frames,
            "fps": round(self.stats.fps, 2),
            "detection_fps": round(self.stats.detection_fps, 2),
            "avg_detection_latency_ms": round(self.stats.avg_detection_latency, 2),
            "run_time": (datetime.now() - self.stats.start_time).total_seconds() if self.stats.start_time else 0
        }


class StreamManagerV2:
    """
    流管理器 V2
    管理多个 StreamProcessorV2
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

        # 流处理器字典
        self._processors: Dict[str, StreamProcessorV2] = {}

        # 检测器工厂 {algo_type: factory_function}
        self._detector_factories: Dict[str, Callable[[], BaseDetector]] = {}

        # 锁
        self._lock = threading.Lock()

    def register_detector_factory(self, algo_type: str, factory: Callable[[], BaseDetector]):
        """
        注册检测器工厂

        Args:
            algo_type: 算法类型 ("1", "2", "3")
            factory: 创建检测器实例的工厂函数
        """
        self._detector_factories[algo_type] = factory
        logger.info(f"Registered detector factory for {algo_type}")

    def _create_detector(self, algo_type: str) -> Optional[BaseDetector]:
        """创建检测器实例"""
        factory = self._detector_factories.get(algo_type)
        if factory:
            try:
                return factory()
            except Exception as e:
                logger.error(f"Failed to create detector {algo_type}: {e}")
        return None

    def add_stream(self, config: StreamConfig) -> bool:
        """添加流"""
        with self._lock:
            if config.camera_id in self._processors:
                logger.warning(f"Stream already exists: {config.camera_id}")
                return False

            processor = StreamProcessorV2(
                config=config,
                detector_factory=self._create_detector,
                storage_service=self.storage,
                alarm_service=self.alarm,
                video_service=self.video,
                on_event=self._on_event
            )

            self._processors[config.camera_id] = processor

            return processor.start()

    def remove_stream(self, camera_id: str, timeout: float = 5.0):
        """移除流"""
        with self._lock:
            processor = self._processors.pop(camera_id, None)
            if processor:
                processor.stop(timeout)
                logger.info(f"Stream removed: {camera_id}")

    def update_stream(self, config: StreamConfig):
        """更新流配置"""
        self.remove_stream(config.camera_id)
        time.sleep(0.5)
        return self.add_stream(config)

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

    def stop_all(self, timeout: float = 10.0, force: bool = False):
        """停止所有流"""
        with self._lock:
            for camera_id, processor in list(self._processors.items()):
                processor.stop(timeout, force=force)
            self._processors.clear()

    def _on_event(self, camera_id: str, event: DetectionEvent, result: DetectionResultBundle):
        """事件回调"""
        pass
