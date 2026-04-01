"""
Ultralytics YOLO Inference Engine
使用Ultralytics官方接口加载YOLO模型
适用于没有TensorRT环境的情况
"""
import os
import warnings
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2
import threading
import time

from .base_engine import BaseInferEngine, DetectionResult, InferenceContext, LetterBoxPreprocessor

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    warnings.warn(f"Ultralytics not available: {e}. UltralyticsEngine will not function.")


class UltralyticsEngine(BaseInferEngine):
    """
    Ultralytics YOLO推理引擎

    特性:
    - 使用Ultralytics官方YOLO接口
    - 自动处理预处理后处理
    - 支持YOLOv8/v9/v10等模型
    - 线程安全

    使用示例:
        engine = UltralyticsEngine(
            model_path="yolov8n.pt",
            input_size=(640, 640),
            confidence=0.4
        )
        detections, context = engine.infer(frame)
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence: float = 0.4,
        iou: float = 0.45,
        max_detections: int = 300,
        classes: Optional[dict] = None,
        device_id: int = 0,
        half: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """
        初始化Ultralytics引擎

        Args:
            model_path: YOLO模型路径 (.pt)
            input_size: 输入尺寸
            confidence: 置信度阈值
            iou: NMS IoU阈值
            max_detections: 最大检测数
            classes: 类别名称映射
            device_id: GPU设备ID
            half: 是否使用FP16
            verbose: 是否输出详细信息
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics is not available. Please install: pip install ultralytics")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")

        self.half = half and self._check_cuda()
        self.device = f'cuda:{device_id}' if self._check_cuda() and device_id >= 0 else 'cpu'
        self.verbose = verbose

        # Ultralytics模型
        self.model = None
        self.model_names = {}  # 类别名称映射

        # 预处理工具 (用于获取坐标转换参数)
        self.preprocessor = LetterBoxPreprocessor(input_size)

        # 线程锁
        self._thread_lock = threading.Lock()

        super().__init__(
            model_path=model_path,
            input_size=input_size,
            confidence=confidence,
            iou=iou,
            max_detections=max_detections,
            classes=classes,
            device_id=device_id,
            **kwargs
        )

    def _check_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _initialize(self):
        """加载Ultralytics模型"""
        print(f"[Ultralytics] Loading model: {self.model_path}")

        # 加载模型
        self.model = YOLO(self.model_path)

        # 切换到指定设备
        if self.device != 'cpu':
            self.model.to(self.device)

        # 获取类别名称
        self.model_names = self.model.names if hasattr(self.model, 'names') else {}

        # 如果提供了自定义类别映射，合并
        if self.classes:
            self.model_names.update(self.classes)

        print(f"[Ultralytics] Model loaded: {self.model_path}")
        print(f"[Ultralytics] Device: {self.device}")
        print(f"[Ultralytics] Classes: {self.model_names}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, InferenceContext]:
        """
        预处理 (Ultralytics自动处理，这里仅记录上下文)
        """
        ori_h, ori_w = image.shape[:2]

        # 获取letterbox参数用于后处理坐标映射
        _, scale, pad_x, pad_y = self.preprocessor.preprocess(image)

        context = InferenceContext(
            original_shape=(ori_h, ori_w),
            input_shape=self.input_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            preprocessing_time=0.0,
            inference_time=0.0,
            postprocessing_time=0.0
        )

        # 直接返回原始图像，Ultralytics内部处理预处理
        return image, context

    def _inference(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Ultralytics推理

        Args:
            image: BGR numpy数组

        Returns:
            Ultralytics Results对象列表
        """
        with self._thread_lock:
            # Ultralytics预测
            results = self.model.predict(
                image,
                imgsz=self.input_size[0],
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_detections,
                device=self.device,
                half=self.half,
                verbose=self.verbose
            )

        return results

    def _postprocess(self, results: List[Any], context: InferenceContext) -> List[DetectionResult]:
        """
        解析Ultralytics结果

        Args:
            results: Ultralytics Results列表
            context: 推理上下文

        Returns:
            List[DetectionResult]: 标准检测结果
        """
        detections = []

        if not results:
            return detections

        # 取第一个结果 (单张图片)
        result = results[0]

        if result.boxes is None:
            return detections

        # 获取检测框
        boxes = result.boxes

        for i in range(len(boxes)):
            # 获取坐标 (xyxy格式)
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            # 获取置信度
            conf = float(boxes.conf[i].cpu().numpy())

            # 获取类别
            cls_id = int(boxes.cls[i].cpu().numpy())

            # 获取类别名称
            cls_name = self.model_names.get(cls_id, f"class_{cls_id}")

            # 创建DetectionResult
            detections.append(DetectionResult(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=conf,
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections

    def infer(self, image: np.ndarray) -> Tuple[List[DetectionResult], InferenceContext]:
        """
        执行推理 (重写以支持Ultralytics的特殊处理)
        """
        t0 = time.perf_counter()

        # 预处理 (记录上下文)
        _, context = self._preprocess(image)
        t1 = time.perf_counter()

        # 推理
        results = self._inference(image)
        t2 = time.perf_counter()

        # 后处理
        detections = self._postprocess(results, context)
        t3 = time.perf_counter()

        # 更新上下文时间
        context.preprocessing_time = t1 - t0
        context.inference_time = t2 - t1
        context.postprocessing_time = t3 - t2

        # 更新统计
        self._stats['inference_count'] += 1
        self._stats['total_pre_time'] += context.preprocessing_time
        self._stats['total_infer_time'] += context.inference_time
        self._stats['total_post_time'] += context.postprocessing_time

        return detections, context

    def release(self):
        """释放资源"""
        if self.model:
            del self.model
            self.model = None

        # 清理CUDA缓存
        if self.device != 'cpu':
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

        print("[Ultralytics] Model released")

    @classmethod
    def from_yolov8(cls, model_path: str, device: str = 'cuda:0', **kwargs) -> 'UltralyticsEngine':
        """
        从YOLOv8模型创建引擎

        Args:
            model_path: YOLOv8模型路径 (.pt)
            device: 设备 ('cuda:0', 'cpu', etc.)

        Returns:
            UltralyticsEngine实例
        """
        device_id = int(device.split(':')[1]) if ':' in device else 0
        if device.startswith('cpu'):
            device_id = -1

        return cls(
            model_path=model_path,
            device_id=device_id,
            **kwargs
        )


# YOLOv8官方类别定义
YOLOV8_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

# 安全帽检测类别 (自定义模型)
SAFETY_HELMET_CLASSES = {
    0: 'helmet',
    1: 'head'
}
