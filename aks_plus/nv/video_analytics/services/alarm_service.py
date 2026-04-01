"""
Alarm Service - 报警服务抽象
解耦HTTP告警推送
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import threading
import queue
import time

import requests

from video_analytics.detectors.base_detector import DetectionEvent


@dataclass
class AlarmConfig:
    """报警配置"""
    endpoints: List[str] = None           # 告警接收端点
    timeout: int = 10                     # 请求超时(秒)
    retry_count: int = 3                  # 重试次数
    retry_interval: float = 1.0           # 重试间隔(秒)
    batch_size: int = 1                   # 批量大小
    batch_interval: float = 0.0           # 批量发送间隔(秒)

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []


@dataclass
class AlarmResult:
    """报警发送结果"""
    success: bool
    endpoint: str = ""
    status_code: int = 0
    response: str = ""
    error_message: str = ""
    retry_count: int = 0


class BaseAlarmService(ABC):
    """
    报警服务抽象基类

    职责:
    - 接收检测事件
    - 格式化报警数据
    - 发送到告警接收端
    - 处理重试逻辑
    """

    def __init__(self, config: AlarmConfig):
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """初始化服务"""
        pass

    @abstractmethod
    def send_alarm(self, event: DetectionEvent) -> List[AlarmResult]:
        """
        发送单条报警

        Args:
            event: 检测事件

        Returns:
            List[AlarmResult]: 各端点的发送结果
        """
        pass

    def send_alarms_batch(self, events: List[DetectionEvent]) -> List[List[AlarmResult]]:
        """
        批量发送报警

        Args:
            events: 事件列表

        Returns:
            每组发送结果
        """
        results = []
        for event in events:
            result = self.send_alarm(event)
            results.append(result)
        return results

    def _format_event(self, event: DetectionEvent) -> Dict[str, Any]:
        """格式化事件为报警数据结构"""
        return event.to_dict()


class HttpAlarmService(BaseAlarmService):
    """
    HTTP报警服务
    通过HTTP POST发送JSON数据
    """

    def _initialize(self):
        """初始化HTTP会话"""
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "VideoAnalytics/2.0",
            "Accept": "application/json"
        })

    def send_alarm(self, event: DetectionEvent) -> List[AlarmResult]:
        """发送HTTP报警"""
        results = []
        data = self._format_event(event)

        for endpoint in self.config.endpoints:
            result = self._send_to_endpoint(endpoint, data)
            results.append(result)

        return results

    def _send_to_endpoint(
        self,
        endpoint: str,
        data: Dict[str, Any],
        retry_count: int = 0
    ) -> AlarmResult:
        """发送到单个端点，带重试"""
        max_retries = self.config.retry_count

        while retry_count <= max_retries:
            try:
                response = self.session.post(
                    endpoint,
                    json=data,
                    timeout=self.config.timeout
                )

                # 检查响应
                if response.status_code == 200:
                    return AlarmResult(
                        success=True,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        response=response.text[:200],
                        retry_count=retry_count
                    )
                else:
                    # 非200响应，尝试重试
                    if retry_count < max_retries:
                        retry_count += 1
                        time.sleep(self.config.retry_interval)
                        continue

                    return AlarmResult(
                        success=False,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        response=response.text[:200],
                        error_message=f"HTTP {response.status_code}",
                        retry_count=retry_count
                    )

            except requests.exceptions.Timeout:
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_interval)
                    continue

                return AlarmResult(
                    success=False,
                    endpoint=endpoint,
                    error_message="Request timeout",
                    retry_count=retry_count
                )

            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_interval)
                    continue

                return AlarmResult(
                    success=False,
                    endpoint=endpoint,
                    error_message=str(e),
                    retry_count=retry_count
                )

        return AlarmResult(
            success=False,
            endpoint=endpoint,
            error_message="Max retries exceeded",
            retry_count=retry_count
        )

    def release(self):
        """释放资源"""
        if self.session:
            self.session.close()


class AsyncAlarmService(BaseAlarmService):
    """
    异步报警服务
    使用后台线程批量发送报警
    """

    def __init__(self, config: AlarmConfig, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._http_service = None
        super().__init__(config)

    def _initialize(self):
        """初始化异步服务"""
        self._http_service = HttpAlarmService(self.config)
        self._start_worker()

    def _start_worker(self):
        """启动后台工作线程"""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self._worker_thread.start()
        print("[AsyncAlarmService] Worker thread started")

    def _worker_loop(self):
        """工作线程循环"""
        batch = []
        last_send_time = time.time()

        while not self._stop_event.is_set():
            try:
                # 非阻塞获取
                event = self._queue.get(timeout=0.1)
                batch.append(event)

            except queue.Empty:
                # 检查是否需要发送批量
                if batch and (time.time() - last_send_time) >= self.config.batch_interval:
                    self._send_batch(batch)
                    batch = []
                    last_send_time = time.time()
                continue

            # 检查批量大小
            if len(batch) >= self.config.batch_size:
                self._send_batch(batch)
                batch = []
                last_send_time = time.time()

        # 处理剩余
        if batch:
            self._send_batch(batch)

    def _send_batch(self, events: List[DetectionEvent]):
        """发送批量报警"""
        for event in events:
            try:
                self._http_service.send_alarm(event)
            except Exception as e:
                print(f"[AsyncAlarmService] Send failed: {e}")

    def send_alarm(self, event: DetectionEvent) -> List[AlarmResult]:
        """
        异步发送报警
        将事件放入队列，立即返回
        """
        try:
            self._queue.put_nowait(event)
            # 立即返回成功（实际发送在后台）
            return [AlarmResult(success=True, endpoint="queued")]
        except queue.Full:
            return [AlarmResult(success=False, error_message="Queue full")]

    def stop(self):
        """停止服务"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._http_service:
            self._http_service.release()


class ConsoleAlarmService(BaseAlarmService):
    """
    控制台报警服务 (用于调试)
    仅打印报警信息到控制台
    """

    def _initialize(self):
        pass

    def send_alarm(self, event: DetectionEvent) -> List[AlarmResult]:
        """打印报警到控制台"""
        print("=" * 60)
        print(f"[ALARM] {event.event_type.value.upper()}")
        print(f"  Camera: {event.camera_id}")
        print(f"  Time: {event.timestamp}")
        print(f"  Objects: {len(event.objects)}")
        print(f"  Image: {event.image_url}")
        print(f"  Video: {event.video_url}")
        print("=" * 60)

        return [AlarmResult(success=True, endpoint="console")]


class AlarmServiceFactory:
    """报警服务工厂"""

    @staticmethod
    def create(
        service_type: str = "http",
        **kwargs
    ) -> BaseAlarmService:
        """
        创建报警服务

        Args:
            service_type: 服务类型 ('http', 'async', 'console')
            **kwargs: 配置参数

        Returns:
            BaseAlarmService: 报警服务实例
        """
        config = AlarmConfig(**kwargs)

        if service_type == "http":
            return HttpAlarmService(config)
        elif service_type == "async":
            return AsyncAlarmService(config)
        elif service_type == "console":
            return ConsoleAlarmService(config)
        else:
            raise ValueError(f"Unsupported alarm type: {service_type}")
