"""
Storage Service - 存储服务抽象
解耦MinIO等存储依赖
"""
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import io
import os

import numpy as np
import cv2


@dataclass
class StorageConfig:
    """存储配置"""
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False
    bucket_name: str = "yolo"
    public_url: str = ""  # 公共访问URL前缀


@dataclass
class UploadResult:
    """上传结果"""
    success: bool
    object_name: str = ""
    url: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = None


class BaseStorageService(ABC):
    """
    存储服务抽象基类

    提供统一的存储接口，支持:
    - 图片上传
    - 视频上传
    - 元数据管理
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """初始化存储客户端"""
        pass

    @abstractmethod
    def upload_image(
        self,
        image: np.ndarray,
        camera_id: str,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """
        上传图片

        Args:
            image: OpenCV图像 (BGR格式)
            camera_id: 摄像头ID
            label: 标签/分类
            metadata: 元数据

        Returns:
            UploadResult: 上传结果
        """
        pass

    @abstractmethod
    def upload_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """
        上传视频

        Args:
            video_bytes: 视频字节流
            camera_id: 摄像头ID
            timestamp: 时间戳
            metadata: 元数据

        Returns:
            UploadResult: 上传结果
        """
        pass

    @abstractmethod
    def delete_object(self, object_name: str) -> bool:
        """删除对象"""
        pass

    def _generate_object_name(
        self,
        camera_id: str,
        label: str,
        extension: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """生成对象名称"""
        ts = timestamp or datetime.now()
        timestamp_str = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        label_part = f"{label}/" if label else ""
        return f"{camera_id}/{label_part}{timestamp_str}.{extension}"


class MinioStorageService(BaseStorageService):
    """
    MinIO存储服务实现
    """

    def _initialize(self):
        """初始化MinIO客户端"""
        try:
            from minio import Minio
            self.client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )
            # 确保bucket存在
            if not self.client.bucket_exists(self.config.bucket_name):
                self.client.make_bucket(self.config.bucket_name)
                print(f"[MinIO] Created bucket: {self.config.bucket_name}")
        except ImportError:
            raise ImportError("MinIO not installed. Please install: pip install minio")
        except Exception as e:
            print(f"[MinIO] Initialization warning: {e}")
            self.client = None

    def upload_image(
        self,
        image: np.ndarray,
        camera_id: str,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """上传图片到MinIO"""
        if self.client is None:
            return UploadResult(
                success=False,
                error_message="MinIO client not initialized"
            )

        try:
            # 编码图片
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                return UploadResult(success=False, error_message="Image encoding failed")

            # 生成对象名
            object_name = self._generate_object_name(camera_id, label, "jpg")

            # 转换为字节流
            image_bytes = io.BytesIO(buffer.tobytes())

            # 上传
            self.client.put_object(
                bucket_name=self.config.bucket_name,
                object_name=object_name,
                data=image_bytes,
                length=len(buffer),
                content_type="image/jpeg",
                metadata=metadata or {}
            )

            # 构建URL
            url = f"{self.config.public_url}/{self.config.bucket_name}/{object_name}"

            print(f"[MinIO] Image uploaded: {object_name}")

            return UploadResult(
                success=True,
                object_name=object_name,
                url=url,
                metadata=metadata
            )

        except Exception as e:
            print(f"[MinIO] Upload failed: {e}")
            return UploadResult(success=False, error_message=str(e))

    def upload_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """上传视频到MinIO"""
        if self.client is None:
            return UploadResult(
                success=False,
                error_message="MinIO client not initialized"
            )

        try:
            # 生成对象名
            ts = timestamp or datetime.now()
            object_name = f"{camera_id}/{ts.strftime('%Y%m%d_%H%M%S')}.mp4"

            # 上传
            video_stream = io.BytesIO(video_bytes)
            self.client.put_object(
                bucket_name=self.config.bucket_name,
                object_name=object_name,
                data=video_stream,
                length=len(video_bytes),
                content_type="video/mp4",
                metadata=metadata or {}
            )

            # 构建URL
            url = f"{self.config.public_url}/{self.config.bucket_name}/{object_name}"

            print(f"[MinIO] Video uploaded: {object_name}")

            return UploadResult(
                success=True,
                object_name=object_name,
                url=url,
                metadata=metadata
            )

        except Exception as e:
            print(f"[MinIO] Upload failed: {e}")
            return UploadResult(success=False, error_message=str(e))

    def delete_object(self, object_name: str) -> bool:
        """删除对象"""
        if self.client is None:
            return False

        try:
            self.client.remove_object(self.config.bucket_name, object_name)
            return True
        except Exception as e:
            print(f"[MinIO] Delete failed: {e}")
            return False


class LocalStorageService(BaseStorageService):
    """
    本地文件系统存储服务 (用于调试)
    """

    def __init__(self, config: StorageConfig, base_path: str = "./output"):
        self.base_path = base_path
        super().__init__(config)

    def _initialize(self):
        """初始化本地存储目录"""
        os.makedirs(self.base_path, exist_ok=True)

    def upload_image(
        self,
        image: np.ndarray,
        camera_id: str,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """保存图片到本地"""
        try:
            # 生成文件名
            object_name = self._generate_object_name(camera_id, label, "jpg")
            file_path = os.path.join(self.base_path, object_name)

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存图片
            cv2.imwrite(file_path, image)

            print(f"[LocalStorage] Image saved: {file_path}")

            return UploadResult(
                success=True,
                object_name=object_name,
                url=file_path,
                metadata=metadata
            )

        except Exception as e:
            print(f"[LocalStorage] Save failed: {e}")
            return UploadResult(success=False, error_message=str(e))

    def upload_video(
        self,
        video_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """保存视频到本地"""
        try:
            # 生成文件名
            ts = timestamp or datetime.now()
            object_name = f"{camera_id}/{ts.strftime('%Y%m%d_%H%M%S')}.mp4"
            file_path = os.path.join(self.base_path, object_name)

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存视频
            with open(file_path, 'wb') as f:
                f.write(video_bytes)

            print(f"[LocalStorage] Video saved: {file_path}")

            return UploadResult(
                success=True,
                object_name=object_name,
                url=file_path,
                metadata=metadata
            )

        except Exception as e:
            print(f"[LocalStorage] Save failed: {e}")
            return UploadResult(success=False, error_message=str(e))

    def delete_object(self, object_name: str) -> bool:
        """删除本地文件"""
        try:
            file_path = os.path.join(self.base_path, object_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"[LocalStorage] Delete failed: {e}")
            return False


class StorageServiceFactory:
    """存储服务工厂"""

    @staticmethod
    def create(
        service_type: str = "minio",
        **kwargs
    ) -> BaseStorageService:
        """
        创建存储服务

        Args:
            service_type: 服务类型 ('minio', 'local', 's3')
            **kwargs: 配置参数

        Returns:
            BaseStorageService: 存储服务实例
        """
        config = StorageConfig(**kwargs)

        if service_type == "minio":
            return MinioStorageService(config)
        # elif service_type == "local":
        #     base_path = kwargs.get("base_path", "./output")
        #     return LocalStorageService(config, base_path)
        else:
            raise ValueError(f"Unsupported storage type: {service_type}")
