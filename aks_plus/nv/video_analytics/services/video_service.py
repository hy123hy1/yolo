"""
Video Service - 视频处理服务
负责视频录制、编码、生成告警视频
"""
import os
import io
import tempfile
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import threading
import time

import numpy as np
import cv2

from .storage_service import BaseStorageService, UploadResult


@dataclass
class VideoConfig:
    """视频配置"""
    fps: int = 25
    pre_seconds: int = 3       # 事件前录制秒数
    post_seconds: int = 3      # 事件后录制秒数
    codec: str = "avc1"        # 视频编码器
    quality: int = 20          # 视频质量 (0-100, 越小越好)
    max_retries: int = 3


class VideoService:
    """
    视频服务

    职责:
    - 管理帧缓存 (pre-buffer)
    - 生成告警视频
    - 上传视频到存储
    - 异步处理视频生成
    """

    def __init__(
        self,
        storage_service: BaseStorageService,
        config: VideoConfig = None
    ):
        self.storage = storage_service
        self.config = config or VideoConfig()
        self._codec = cv2.VideoWriter_fourcc(*self.config.codec)

    def create_frame_buffer(self, max_seconds: Optional[int] = None) -> deque:
        """
        创建帧缓存队列

        Args:
            max_seconds: 最大缓存秒数

        Returns:
            deque: 帧缓存队列
        """
        max_len = (max_seconds or self.config.pre_seconds) * self.config.fps
        return deque(maxlen=max_len)

    def generate_video(
        self,
        pre_frames: List[np.ndarray],
        post_frames: List[np.ndarray],
        fps: Optional[int] = None
    ) -> Optional[bytes]:
        """
        生成视频字节流

        Args:
            pre_frames: 事件前帧列表
            post_frames: 事件后帧列表
            fps: 帧率

        Returns:
            bytes: 视频字节流，失败返回None
        """
        if not pre_frames and not post_frames:
            print("[VideoService] No frames to encode")
            return None

        fps = fps or self.config.fps
        all_frames = list(pre_frames) + list(post_frames)

        try:
            # 获取视频尺寸
            sample_frame = all_frames[0]
            h, w = sample_frame.shape[:2]

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # 创建视频写入器
            writer = cv2.VideoWriter(tmp_path, self._codec, fps, (w, h))
            if not writer.isOpened():
                print("[VideoService] Failed to create video writer")
                return None

            # 写入帧
            for frame in all_frames:
                writer.write(frame)

            writer.release()

            # 读取视频字节
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()

            # 删除临时文件
            os.remove(tmp_path)

            print(f"[VideoService] Video generated: {len(all_frames)} frames, {len(video_bytes)} bytes")
            return video_bytes

        except Exception as e:
            print(f"[VideoService] Video generation failed: {e}")
            return None

    def upload_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None
    ) -> UploadResult:
        """
        上传视频

        Args:
            video_bytes: 视频字节流
            camera_id: 摄像头ID
            timestamp: 时间戳

        Returns:
            UploadResult: 上传结果
        """
        return self.storage.upload_video(video_bytes, camera_id, timestamp)

    def generate_and_upload(
        self,
        pre_frames: List[np.ndarray],
        post_frames: List[np.ndarray],
        camera_id: str,
        fps: Optional[int] = None
    ) -> UploadResult:
        """
        生成并上传视频

        Args:
            pre_frames: 事件前帧
            post_frames: 事件后帧
            camera_id: 摄像头ID
            fps: 帧率

        Returns:
            UploadResult: 上传结果
        """
        video_bytes = self.generate_video(pre_frames, post_frames, fps)

        if video_bytes is None:
            return UploadResult(success=False, error_message="Video generation failed")

        return self.upload_video(video_bytes, camera_id)

    def async_generate_and_upload(
        self,
        pre_frames: List[np.ndarray],
        rtsp_url: str,
        camera_id: str,
        on_complete: Optional[Callable[[UploadResult], None]] = None,
        fps: Optional[int] = None
    ):
        """
        异步生成并上传视频

        在后台线程中:
        1. 录制post_frames
        2. 生成视频
        3. 上传视频
        4. 调用回调函数

        Args:
            pre_frames: 事件前帧
            rtsp_url: RTSP流地址 (用于录制post_frames)
            camera_id: 摄像头ID
            on_complete: 完成回调函数
            fps: 帧率
        """
        thread = threading.Thread(
            target=self._async_worker,
            args=(pre_frames, rtsp_url, camera_id, on_complete, fps),
            daemon=True
        )
        thread.start()

    def _async_worker(
        self,
        pre_frames: List[np.ndarray],
        rtsp_url: str,
        camera_id: str,
        on_complete: Optional[Callable[[UploadResult], None]],
        fps: Optional[int]
    ):
        """异步工作线程"""
        try:
            # 录制post frames
            post_frames = self._record_post_frames(rtsp_url, self.config.post_seconds, fps)

            # 生成并上传视频
            result = self.generate_and_upload(pre_frames, post_frames, camera_id, fps)

            # 调用回调
            if on_complete:
                on_complete(result)

        except Exception as e:
            print(f"[VideoService] Async worker error: {e}")
            if on_complete:
                on_complete(UploadResult(success=False, error_message=str(e)))

    def _record_post_frames(
        self,
        rtsp_url: str,
        seconds: int,
        fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        从RTSP流录制指定秒数的帧

        Args:
            rtsp_url: RTSP地址
            seconds: 录制秒数
            fps: 帧率

        Returns:
            List[np.ndarray]: 录制的帧列表
        """
        fps = fps or self.config.fps
        total_frames = fps * seconds
        frames = []

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[VideoService] Failed to open RTSP: {rtsp_url}")
            return frames

        while len(frames) < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())

        cap.release()
        print(f"[VideoService] Recorded {len(frames)} post frames")
        return frames


class FrameRecorder:
    """
    帧录制器
    用于实时录制视频流到帧缓存
    """

    def __init__(self, max_seconds: int = 10, fps: int = 25):
        self.max_seconds = max_seconds
        self.fps = fps
        self.frame_buffer = deque(maxlen=max_seconds * fps)
        self._recording = False
        self._thread = None

    def start(self, rtsp_url: str):
        """开始后台录制"""
        if self._recording:
            return

        self._recording = True
        self._thread = threading.Thread(
            target=self._record_loop,
            args=(rtsp_url,),
            daemon=True
        )
        self._thread.start()

    def _record_loop(self, rtsp_url: str):
        """录制循环"""
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        while self._recording:
            ret, frame = cap.read()
            if ret:
                self.frame_buffer.append(frame)
            else:
                time.sleep(0.01)

        cap.release()

    def stop(self):
        """停止录制"""
        self._recording = False
        if self._thread:
            self._thread.join(timeout=1)

    def get_snapshot(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        if self.frame_buffer:
            return self.frame_buffer[-1].copy()
        return None

    def get_buffer_copy(self) -> List[np.ndarray]:
        """获取缓存副本"""
        return [f.copy() for f in self.frame_buffer]
