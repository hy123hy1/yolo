"""
Configuration Settings
集中管理所有配置
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """模型配置"""
    person_model_path: str = "models/yolov8n.pt"  # 人员检测模型 (官方YOLOv8)
    helmet_model_path: str = "models/safehat.pt"   # 安全帽检测模型 (自定义训练)
    backend: str = "ultralytics"                      # ultralytics/tensorrt/onnx/torch
    input_size: tuple = (640, 640)
    confidence: float = 0.4
    iou: float = 0.45
    device_id: int = 0
    fp16: bool = False                              # PyTorch后端通常不用FP16


@dataclass
class DetectionConfig:
    """检测配置"""
    # 闯入检测
    intrusion_min_frames: int = 25
    intrusion_cooldown: float = 60.0
    intrusion_confidence: float = 0.5

    # 安全帽检测
    helmet_min_frames: int = 25
    helmet_cooldown: float = 60.0
    helmet_confidence: float = 0.4
    helmet_crop_padding: float = 0.2

    # 超员检测
    overcrowd_max_people: int = 15
    overcrowd_duration: float = 2.0
    overcrowd_cooldown: float = 60.0
    overcrowd_confidence: float = 0.4


@dataclass
class StorageConfig:
    """存储配置"""
    type: str = "minio"  # minio/local
    endpoint: str = "192.168.1.61:9000"
    access_key: str = "admin"
    secret_key: str = "12345678"
    secure: bool = False
    bucket_name: str = "yolo"
    public_url: str = "192.168.1.61:9000"
    local_path: str = "./output"


@dataclass
class AlarmConfig:
    """报警配置"""
    type: str = "http"  # http/async/console
    endpoints: List[str] = field(default_factory=list)
    endpoints_intrusion: List[str] = field(default_factory=list)
    endpoints_helmet: List[str] = field(default_factory=list)
    endpoints_overcrowd: List[str] = field(default_factory=list)
    timeout: int = 10
    retry_count: int = 3


@dataclass
class StreamConfig:
    """流处理配置"""
    fps: int = 25
    skip_frames: int = 0           # 跳帧数
    max_reconnect: int = 5
    reconnect_interval: float = 2.0
    pre_buffer_seconds: int = 3
    post_record_seconds: int = 3
    enable_display: bool = False


@dataclass
class AppConfig:
    """应用配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

    @classmethod
    def from_file(cls, path: str) -> 'AppConfig':
        """从JSON文件加载配置"""
        if not os.path.exists(path):
            print(f"[Config] Config file not found: {path}, using defaults")
            return cls()

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建配置"""
        return cls(
            model=ModelConfig(**data.get('model', {})),
            detection=DetectionConfig(**data.get('detection', {})),
            storage=StorageConfig(**data.get('storage', {})),
            alarm=AlarmConfig(**data.get('alarm', {})),
            stream=StreamConfig(**data.get('stream', {}))
        )

    def to_file(self, path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'model': self.model.__dict__,
            'detection': self.detection.__dict__,
            'storage': self.storage.__dict__,
            'alarm': self.alarm.__dict__,
            'stream': self.stream.__dict__
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 默认配置实例
default_config = AppConfig()
