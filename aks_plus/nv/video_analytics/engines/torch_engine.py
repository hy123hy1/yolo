"""
PyTorch Inference Engine Implementation
For debugging and development with native PyTorch models
"""
import os
import warnings
from typing import List, Tuple, Any, Optional, Union
import numpy as np
import cv2
import threading

from .base_engine import BaseInferEngine, DetectionResult, InferenceContext, LetterBoxPreprocessor

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    warnings.warn(f"PyTorch not available: {e}. TorchInferEngine will not function.")


class TorchInferEngine(BaseInferEngine):
    """
    PyTorch推理引擎

    特性:
    - 原生PyTorch模型支持 (.pt, .pth)
    - 支持TorchScript (.torchscript, .ts)
    - 自动混合精度 (AMP)
    - 推理模式优化 (torch.inference_mode)
    - 适合调试和开发

    使用示例:
        # 从YOLOv8加载
        engine = TorchInferEngine.from_yolov8("yolov8n.pt", device='cuda:0')

        # 从自定义模型加载
        engine = TorchInferEngine(
            model_path="custom_model.pth",
            model_class=MyModel,
            device='cuda:0'
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
        model_class: Optional[type] = None,
        model_loader: Optional[str] = 'auto',  # 'auto', 'yolov8', 'torchscript', 'pickle'
        half: bool = False,  # FP16推理
        **kwargs
    ):
        """
        初始化PyTorch推理引擎

        Args:
            model_path: PyTorch模型文件路径
            model_class: 模型类 (用于自定义模型)
            model_loader: 模型加载方式
            half: 是否使用FP16半精度
            device_id: GPU设备ID
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install torch.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch model file not found: {model_path}")

        self.model_class = model_class
        self.model_loader = model_loader
        self.half = half and torch.cuda.is_available()
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'

        # PyTorch组件
        self.model = None
        self.model_type = None  # 'yolov8', 'torchscript', 'native'

        # 预处理工具
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

    def _initialize(self):
        """初始化PyTorch模型"""
        # 自动检测模型类型
        self.model_type = self._detect_model_type()

        if self.model_type == 'yolov8':
            self._load_yolov8()
        elif self.model_type == 'torchscript':
            self._load_torchscript()
        else:
            self._load_native_model()

        # 设置推理模式
        self.model.eval()
        if self.half:
            self.model = self.model.half()

        print(f"[PyTorch] Model loaded: {self.model_path}")
        print(f"[PyTorch] Type: {self.model_type}")
        print(f"[PyTorch] Device: {self.device}")
        print(f"[PyTorch] FP16: {self.half}")

    def _detect_model_type(self) -> str:
        """检测模型类型"""
        ext = os.path.splitext(self.model_path)[1].lower()

        if ext in ['.torchscript', '.ts', '.ptl']:
            return 'torchscript'

        # 尝试检测YOLOv8
        try:
            ckpt = torch.load(self.model_path, map_location='cpu')
            if isinstance(ckpt, dict):
                if 'model' in ckpt and hasattr(ckpt['model'], 'args'):
                    return 'yolov8'
                if any(key in ckpt for key in ['model', 'state_dict', 'network']):
                    return 'native'
        except:
            pass

        # 默认尝试YOLOv8
        return 'yolov8'

    def _load_yolov8(self):
        """加载YOLOv8模型"""
        try:
            # 尝试使用ultralytics
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            # 切换到指定设备
            self.model.to(self.device)
        except ImportError:
            # 手动加载YOLOv8
            ckpt = torch.load(self.model_path, map_location='cpu')
            if isinstance(ckpt, dict):
                self.model = ckpt.get('model', ckpt)
            else:
                self.model = ckpt
            self.model = self.model.to(self.device)
            self.model.float()  # 确保是float32

    def _load_torchscript(self):
        """加载TorchScript模型"""
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model = self.model.to(self.device)

    def _load_native_model(self):
        """加载原生PyTorch模型"""
        ckpt = torch.load(self.model_path, map_location=self.device)

        if self.model_class:
            self.model = self.model_class()
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
            else:
                state_dict = ckpt
            self.model.load_state_dict(state_dict)
        else:
            if isinstance(ckpt, dict):
                self.model = ckpt.get('model', ckpt.get('state_dict', ckpt))
            else:
                self.model = ckpt

        self.model = self.model.to(self.device)

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, InferenceContext]:
        """
        PyTorch预处理
        输入: BGR numpy (H, W, 3)
        输出: GPU Tensor [1, 3, H, W]
        """
        ori_h, ori_w = image.shape[:2]

        # LetterBox预处理
        padded, scale, pad_x, pad_y = self.preprocessor.preprocess(image)

        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        # 归一化到[0, 1]
        tensor = tensor.float() / 255.0

        if self.half:
            tensor = tensor.half()

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

        return tensor, context

    def _inference(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """PyTorch推理"""
        with self._thread_lock:
            with torch.inference_mode():
                if self.model_type == 'yolov8' and hasattr(self.model, 'predict'):
                    # YOLOv8 API
                    outputs = self.model.model(tensor)
                else:
                    outputs = self.model(tensor)

        # 转换为numpy
        if isinstance(outputs, torch.Tensor):
            return [outputs.cpu().numpy()]
        elif isinstance(outputs, (list, tuple)):
            return [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
        else:
            return [outputs]

    def _postprocess(self, raw_outputs: List[np.ndarray], context: InferenceContext) -> List[DetectionResult]:
        """解析YOLO输出并应用NMS"""
        if not raw_outputs:
            return []

        output = raw_outputs[0]

        # 处理不同输出格式
        if output.ndim == 3:
            if output.shape[1] == 84:  # YOLOv8 [batch, 84, 8400]
                detections = self._parse_yolov8_output(output[0], context)
            elif output.shape[2] == 6:  # [batch, num_dets, 6]
                detections = self._parse_decoded_output(output[0], context)
            else:
                detections = self._parse_generic_output(output, context)
        elif output.ndim == 2:
            if output.shape[1] == 6:
                detections = self._parse_decoded_output(output, context)
            elif output.shape[0] == 84:  # [84, 8400]
                detections = self._parse_yolov8_output(output, context)
            else:
                detections = self._parse_generic_output(output, context)
        else:
            detections = self._parse_generic_output(output, context)

        # NMS
        if len(detections) > 0:
            detections = self._apply_nms(detections)

        return detections

    def _parse_yolov8_output(self, output: np.ndarray, context: InferenceContext) -> List[DetectionResult]:
        """解析YOLOv8输出 [84, 8400]"""
        ori_h, ori_w = context.original_shape

        predictions = output.T if output.shape[0] == 84 else output

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences > self.confidence
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        detections = []
        for i in range(len(boxes)):
            rx1, ry1, rx2, ry2 = self.preprocessor.reverse_transform(
                x1[i], y1[i], x2[i], y2[i],
                context.scale, context.pad_x, context.pad_y,
                ori_w, ori_h
            )

            class_id = int(class_ids[i])
            detections.append(DetectionResult(
                x1=float(rx1),
                y1=float(ry1),
                x2=float(rx2),
                y2=float(ry2),
                conf=float(confidences[i]),
                class_id=class_id,
                class_name=self.classes.get(class_id)
            ))

        return detections

    def _parse_decoded_output(self, output: np.ndarray, context: InferenceContext) -> List[DetectionResult]:
        """解析已解码的输出 [N, 6]"""
        detections = []
        for det in output:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, class_id = det[:6]
            if conf < self.confidence:
                continue

            class_id = int(class_id)
            detections.append(DetectionResult(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=float(conf),
                class_id=class_id,
                class_name=self.classes.get(class_id)
            ))
        return detections

    def _parse_generic_output(self, output: np.ndarray, context: InferenceContext) -> List[DetectionResult]:
        """通用输出解析"""
        try:
            if output.size % 6 == 0:
                dets = output.reshape(-1, 6)
                return self._parse_decoded_output(dets, context)
        except:
            pass
        return []

    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """应用NMS"""
        if not detections:
            return []

        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        scores = np.array([d.conf for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence,
            self.iou
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, tuple):
            indices = indices[0]

        return [detections[int(i)] for i in indices.flatten()][:self.max_detections]

    def release(self):
        """释放资源"""
        if self.model:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        print("[PyTorch] Model released")

    @classmethod
    def from_yolov8(cls, model_path: str, device: str = 'cuda:0', **kwargs) -> 'TorchInferEngine':
        """
        从YOLOv8模型创建引擎

        Args:
            model_path: YOLOv8模型路径 (.pt)
            device: 设备

        Returns:
            TorchInferEngine实例
        """
        device_id = int(device.split(':')[1]) if ':' in device else 0
        if device.startswith('cpu'):
            device_id = -1

        return cls(
            model_path=model_path,
            model_loader='yolov8',
            device_id=device_id,
            **kwargs
        )

    @classmethod
    def from_torchscript(cls, model_path: str, device: str = 'cuda:0', **kwargs) -> 'TorchInferEngine':
        """
        从TorchScript模型创建引擎

        Args:
            model_path: TorchScript模型路径 (.torchscript, .ts)
            device: 设备

        Returns:
            TorchInferEngine实例
        """
        device_id = int(device.split(':')[1]) if ':' in device else 0
        if device.startswith('cpu'):
            device_id = -1

        return cls(
            model_path=model_path,
            model_loader='torchscript',
            device_id=device_id,
            **kwargs
        )


# 便捷创建函数
def create_torch_engine(
    model_path: str,
    input_size: Tuple[int, int] = (640, 640),
    device: str = 'cuda:0',
    half: bool = False,
    **kwargs
) -> TorchInferEngine:
    """
    便捷创建PyTorch引擎

    Args:
        model_path: 模型路径
        input_size: 输入尺寸
        device: 设备 ('cuda:0', 'cpu', etc.)
        half: 是否使用FP16

    Returns:
        TorchInferEngine实例
    """
    device_id = int(device.split(':')[1]) if ':' in device else 0
    if device.startswith('cpu'):
        device_id = -1

    return TorchInferEngine(
        model_path=model_path,
        input_size=input_size,
        device_id=device_id,
        half=half,
        **kwargs
    )
