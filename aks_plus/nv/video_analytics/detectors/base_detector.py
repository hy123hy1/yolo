"""
Base Detector Abstraction
Provides unified interface for all detection scenarios
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import cv2
from datetime import datetime

from video_analytics.engines.base_engine import BaseInferEngine, DetectionResult


class DetectionEventType(Enum):
    """检测事件类型"""
    INTRUSION = "intrusion"          # 闯入
    NO_HELMET = "no_helmet"          # 未戴安全帽
    OVERCROWD = "overcrowd"          # 人员超员
    GATHERING = "gathering"          # 人员聚集
    FENCE_BREACH = "fence_breach"    # 电子围栏
    UNKNOWN = "unknown"


@dataclass
class DetectionEvent:
    """检测事件数据"""
    event_type: DetectionEventType
    camera_id: str
    timestamp: datetime
    objects: List[Dict[str, Any]] = field(default_factory=list)
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "event_type": self.event_type.value,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "objects": self.objects,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class DetectionContext:
    """检测上下文信息"""
    camera_id: str
    rtsp_url: str
    ip_address: str
    frame: np.ndarray
    timestamp: datetime
    frame_buffer: List[np.ndarray] = field(default_factory=list)
    fps: float = 25.0
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResultBundle:
    """检测结果包"""
    triggered: bool                          # 是否触发检测
    event: Optional[DetectionEvent] = None   # 检测事件
    detections: List[DetectionResult] = field(default_factory=list)  # 原始检测结果
    visualized_frame: Optional[np.ndarray] = None  # 可视化后的帧
    debug_info: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """
    检测器抽象基类

    职责分离:
    - BaseInferEngine: 负责推理 (YOLO等)
    - BaseDetector: 负责解析检测结果、触发业务逻辑

    使用示例:
        detector = IntrusionDetector(
            engine=TensorRTInferEngine("yolov8n.engine"),
            config={"min_frames": 25, "confidence": 0.5}
        )

        result = detector.process(frame, context)
        if result.triggered:
            print(f"Event triggered: {result.event}")
    """

    def __init__(
        self,
        engine: BaseInferEngine,
        config: Optional[Dict[str, Any]] = None,
        event_type: DetectionEventType = DetectionEventType.UNKNOWN
    ):
        """
        初始化检测器

        Args:
            engine: 推理引擎实例
            config: 检测器配置
            event_type: 事件类型标识
        """
        self.engine = engine
        self.config = config or {}
        self.event_type = event_type

        # 性能统计
        self._process_count = 0
        self._trigger_count = 0
        self._total_process_time = 0.0

    @abstractmethod
    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """
        处理单帧图像

        Args:
            context: 检测上下文

        Returns:
            DetectionResultBundle: 检测结果
        """
        pass

    def _infer(self, frame: np.ndarray) -> Tuple[List[DetectionResult], Any]:
        """
        执行推理

        Args:
            frame: 输入图像

        Returns:
            detections: 检测结果列表
            infer_context: 推理上下文
        """
        detections, infer_context = self.engine.infer(frame)
        return detections, infer_context

    def _filter_by_class(
        self,
        detections: List[DetectionResult],
        class_ids: List[int],
        min_confidence: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        按类别过滤检测结果

        Args:
            detections: 原始检测结果
            class_ids: 保留的类别ID列表
            min_confidence: 最小置信度

        Returns:
            过滤后的检测结果
        """
        min_conf = min_confidence or self.config.get("confidence", 0.4)

        filtered = []
        for det in detections:
            if det.class_id in class_ids and det.conf >= min_conf:
                filtered.append(det)

        return filtered

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制检测框

        Args:
            frame: 原始图像
            detections: 检测结果
            color: 框颜色 (B, G, R)
            thickness: 线宽

        Returns:
            绘制后的图像
        """
        img = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            label = f"{det.class_name or det.class_id} {det.conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # 标签背景
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # 标签文字
            cv2.putText(
                img, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1
            )

        return img

    def _create_event(
        self,
        context: DetectionContext,
        objects: List[Dict[str, Any]],
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionEvent:
        """
        创建检测事件

        Args:
            context: 检测上下文
            objects: 检测到的对象列表
            confidence: 置信度
            metadata: 元数据

        Returns:
            DetectionEvent实例
        """
        return DetectionEvent(
            event_type=self.event_type,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            objects=objects,
            confidence=confidence,
            metadata=metadata or {}
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self._total_process_time / self._process_count if self._process_count > 0 else 0

        return {
            "process_count": self._process_count,
            "trigger_count": self._trigger_count,
            "avg_process_time": avg_time,
            "trigger_rate": self._trigger_count / self._process_count if self._process_count > 0 else 0
        }

    def reset_stats(self):
        """重置统计"""
        self._process_count = 0
        self._trigger_count = 0
        self._total_process_time = 0.0


class MultiStageDetector(BaseDetector):
    """
    多级检测器基类
    用于需要多阶段推理的场景 (如: 先检测人，再检测安全帽)
    """

    def __init__(
        self,
        primary_engine: BaseInferEngine,
        secondary_engine: Optional[BaseInferEngine] = None,
        config: Optional[Dict[str, Any]] = None,
        event_type: DetectionEventType = DetectionEventType.UNKNOWN
    ):
        """
        初始化多级检测器

        Args:
            primary_engine: 主推理引擎 (如: 人员检测)
            secondary_engine: 次级推理引擎 (如: 安全帽检测)
        """
        super().__init__(primary_engine, config, event_type)
        self.secondary_engine = secondary_engine

    def _crop_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        padding: float = 0.0
    ) -> np.ndarray:
        """
        裁剪图像区域

        Args:
            frame: 原始图像
            bbox: (x1, y1, x2, y2)
            padding: 外扩比例

        Returns:
            裁剪后的图像
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # 应用padding
        if padding > 0:
            box_w = x2 - x1
            box_h = y2 - y1
            x1 = max(0, x1 - box_w * padding)
            y1 = max(0, y1 - box_h * padding)
            x2 = min(w, x2 + box_w * padding)
            y2 = min(h, y2 + box_h * padding)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        return frame[y1:y2, x1:x2]

    def _secondary_infer(
        self,
        region: np.ndarray
    ) -> Tuple[List[DetectionResult], Any]:
        """
        执行次级推理

        Args:
            region: 裁剪后的区域图像

        Returns:
            检测结果和上下文
        """
        if self.secondary_engine is None:
            return [], None

        return self.secondary_engine.infer(region)


class DetectorPipeline:
    """
    检测器管道
    串行执行多个检测器
    """

    def __init__(self, detectors: List[BaseDetector]):
        self.detectors = detectors

    def process(self, context: DetectionContext) -> List[DetectionResultBundle]:
        """
        依次执行所有检测器

        Args:
            context: 检测上下文

        Returns:
            所有检测结果列表
        """
        results = []

        for detector in self.detectors:
            result = detector.process(context)
            results.append(result)

        return results

    def add_detector(self, detector: BaseDetector):
        """添加检测器"""
        self.detectors.append(detector)

    def remove_detector(self, detector_type: type):
        """移除指定类型的检测器"""
        self.detectors = [d for d in self.detectors if not isinstance(d, detector_type)]
