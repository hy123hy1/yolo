"""
Base Inference Engine Abstraction
Provides unified interface for different inference backends
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum


class BackendType(Enum):
    """支持的推理后端类型"""
    TENSORRT = "tensorrt"
    ONNXRUNTIME = "onnxruntime"
    PYTORCH = "pytorch"


@dataclass
class DetectionResult:
    """标准检测结果格式"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: Optional[str] = None

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """返回xyxy格式坐标"""
        return self.x1, self.y1, self.x2, self.y2

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """返回xywh格式坐标"""
        x = (self.x1 + self.x2) / 2
        y = (self.y1 + self.y2) / 2
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return x, y, w, h


@dataclass
class InferenceContext:
    """推理上下文信息"""
    original_shape: Tuple[int, int]  # 原始图像尺寸 (H, W)
    input_shape: Tuple[int, int]     # 输入模型尺寸 (H, W)
    scale: float                     # 缩放比例
    pad_x: int                       # x方向padding
    pad_y: int                       # y方向padding
    preprocessing_time: float        # 预处理耗时
    inference_time: float            # 推理耗时
    postprocessing_time: float       # 后处理耗时


class BaseInferEngine(ABC):
    """
    推理引擎抽象基类

    统一接口设计:
    - 输入: BGR numpy数组 (OpenCV格式)
    - 输出: List[DetectionResult] 标准检测结果
    - 内部自动完成: 预处理 -> 推理 -> NMS后处理

    使用示例:
        engine = TensorRTInferEngine("model.engine", confidence=0.4, iou=0.45)
        detections = engine.infer(frame)
        for det in detections:
            print(f"Detected {det.class_name} at {det.to_xyxy()} with conf {det.conf}")
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence: float = 0.4,
        iou: float = 0.45,
        max_detections: int = 300,
        classes: Optional[Dict[int, str]] = None,
        device_id: int = 0,
        **kwargs
    ):
        """
        初始化推理引擎

        Args:
            model_path: 模型文件路径
            input_size: 模型输入尺寸 (H, W)
            confidence: 置信度阈值
            iou: NMS IoU阈值
            max_detections: 最大检测数量
            classes: 类别名称映射 {class_id: class_name}
            device_id: GPU设备ID
        """
        self.model_path = model_path
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.classes = classes or {}
        self.device_id = device_id
        self.kwargs = kwargs

        # 性能统计
        self._stats = {
            'inference_count': 0,
            'total_pre_time': 0.0,
            'total_infer_time': 0.0,
            'total_post_time': 0.0,
        }

        # 子类实现初始化
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """子类实现模型加载和初始化"""
        pass

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, InferenceContext]:
        """
        预处理图像

        Args:
            image: BGR格式numpy数组 (H, W, 3)

        Returns:
            blob: 预处理后的数据，可直接输入模型
            context: 推理上下文，用于后处理坐标映射
        """
        pass

    @abstractmethod
    def _inference(self, blob: np.ndarray) -> Any:
        """
        执行推理

        Args:
            blob: 预处理后的输入数据

        Returns:
            raw_outputs: 模型原始输出
        """
        pass

    @abstractmethod
    def _postprocess(self, raw_outputs: Any, context: InferenceContext) -> List[DetectionResult]:
        """
        后处理：解析模型输出并应用NMS

        Args:
            raw_outputs: 模型原始输出
            context: 推理上下文

        Returns:
            detections: 检测结果列表
        """
        pass

    def infer(self, image: np.ndarray) -> Tuple[List[DetectionResult], InferenceContext]:
        """
        执行完整推理流程

        Args:
            image: BGR格式numpy数组 (H, W, 3)

        Returns:
            detections: 检测结果列表
            context: 推理上下文信息
        """
        import time

        # 预处理
        t0 = time.perf_counter()
        blob, context = self._preprocess(image)
        t1 = time.perf_counter()

        # 推理
        raw_outputs = self._inference(blob)
        t2 = time.perf_counter()

        # 后处理
        detections = self._postprocess(raw_outputs, context)
        t3 = time.perf_counter()

        # 更新统计
        context.preprocessing_time = t1 - t0
        context.inference_time = t2 - t1
        context.postprocessing_time = t3 - t2

        self._stats['inference_count'] += 1
        self._stats['total_pre_time'] += context.preprocessing_time
        self._stats['total_infer_time'] += context.inference_time
        self._stats['total_post_time'] += context.postprocessing_time

        return detections, context

    def infer_batch(self, images: List[np.ndarray]) -> List[Tuple[List[DetectionResult], InferenceContext]]:
        """
        批量推理（子类可覆盖以实现真正的batch推理）

        Args:
            images: 图像列表

        Returns:
            每帧的检测结果和上下文列表
        """
        # 默认串行处理，子类可优化为真正的batch推理
        results = []
        for img in images:
            results.append(self.infer(img))
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        count = self._stats['inference_count']
        if count == 0:
            return {
                'inference_count': 0,
                'avg_preprocessing_time': 0.0,
                'avg_inference_time': 0.0,
                'avg_postprocessing_time': 0.0,
                'avg_total_time': 0.0,
                'fps': 0.0,
            }

        avg_pre = self._stats['total_pre_time'] / count
        avg_infer = self._stats['total_infer_time'] / count
        avg_post = self._stats['total_post_time'] / count
        avg_total = avg_pre + avg_infer + avg_post

        return {
            'inference_count': count,
            'avg_preprocessing_time': avg_pre,
            'avg_inference_time': avg_infer,
            'avg_postprocessing_time': avg_post,
            'avg_total_time': avg_total,
            'fps': 1.0 / avg_total if avg_total > 0 else 0.0,
        }

    def reset_stats(self):
        """重置性能统计"""
        self._stats = {
            'inference_count': 0,
            'total_pre_time': 0.0,
            'total_infer_time': 0.0,
            'total_post_time': 0.0,
        }

    def warmup(self, num_runs: int = 10):
        """预热模型"""
        dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for _ in range(num_runs):
            self.infer(dummy_image)
        self.reset_stats()

    @abstractmethod
    def release(self):
        """释放资源"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class LetterBoxPreprocessor:
    """
    LetterBox预处理工具类
    保持图像长宽比，短边填充灰色(114,114,114)
    """

    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        执行letterbox预处理

        Returns:
            padded: 填充后的图像 (input_size[0], input_size[1], 3)
            scale: 缩放比例
            pad_x: x方向padding
            pad_y: y方向padding
        """
        ori_h, ori_w = image.shape[:2]
        target_h, target_w = self.input_size

        # 等比例缩放
        scale = min(target_w / ori_w, target_h / ori_h)
        new_w = int(ori_w * scale)
        new_h = int(ori_h * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充画布
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # 计算padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        # 放置缩放后的图像
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        return padded, scale, pad_x, pad_y

    def reverse_transform(self, x1: float, y1: float, x2: float, y2: float,
                          scale: float, pad_x: int, pad_y: int,
                          ori_w: int, ori_h: int) -> Tuple[int, int, int, int]:
        """
        将坐标从padded图像映射回原始图像

        Returns:
            (rx1, ry1, rx2, ry2): 原始图像上的坐标
        """
        # 去padding
        x1 -= pad_x
        y1 -= pad_y
        x2 -= pad_x
        y2 -= pad_y

        # 反缩放
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        # 裁剪到边界
        rx1 = int(max(0, min(ori_w - 1, round(x1))))
        ry1 = int(max(0, min(ori_h - 1, round(y1))))
        rx2 = int(max(0, min(ori_w - 1, round(x2))))
        ry2 = int(max(0, min(ori_h - 1, round(y2))))

        return rx1, ry1, rx2, ry2


# 导入OpenCV用于预处理
import cv2
