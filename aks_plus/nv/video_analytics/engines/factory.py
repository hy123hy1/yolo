"""
Inference Engine Factory
Provides unified interface to create different inference backends
"""
from typing import Tuple, Optional, Dict, Any
import os

from .base_engine import BaseInferEngine, BackendType


def create_infer_engine(
    model_path: str,
    backend: str = 'tensorrt',
    input_size: Tuple[int, int] = (640, 640),
    confidence: float = 0.4,
    iou: float = 0.45,
    classes: Optional[Dict[int, str]] = None,
    device_id: int = 0,
    **kwargs
) -> BaseInferEngine:
    """
    创建推理引擎工厂函数

    根据模型文件类型和后端类型自动选择合适的推理引擎

    Args:
        model_path: 模型文件路径
        backend: 推理后端 ('tensorrt', 'onnx', 'torch', 'auto')
        input_size: 输入尺寸 (H, W)
        confidence: 置信度阈值
        iou: NMS IoU阈值
        classes: 类别名称映射
        device_id: GPU设备ID
        **kwargs: 额外参数传递给具体引擎

    Returns:
        BaseInferEngine: 推理引擎实例

    Raises:
        ValueError: 不支持的模型格式或后端
        FileNotFoundError: 模型文件不存在

    示例:
        # TensorRT (推荐生产环境)
        engine = create_infer_engine(
            model_path="yolov8n.engine",
            backend='tensorrt',
            input_size=(640, 640),
            fp16=True
        )

        # ONNX Runtime
        engine = create_infer_engine(
            model_path="yolov8n.onnx",
            backend='onnx',
            use_cuda=True,
            use_tensorrt=False
        )

        # PyTorch (调试)
        engine = create_infer_engine(
            model_path="yolov8n.pt",
            backend='torch',
            device='cuda:0'
        )
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 自动检测后端
    if backend == 'auto':
        backend = _detect_backend(model_path)
        print(f"[Factory] Auto-detected backend: {backend}")

    backend = backend.lower()

    # 创建对应引擎
    if backend in ['tensorrt', 'trt']:
        from .tensorrt_engine import TensorRTInferEngine
        return TensorRTInferEngine(
            model_path=model_path,
            input_size=input_size,
            confidence=confidence,
            iou=iou,
            classes=classes,
            device_id=device_id,
            **kwargs
        )

    elif backend in ['onnx', 'onnxruntime', 'ort']:
        from .onnx_engine import ONNXInferEngine, create_onnx_engine
        return create_onnx_engine(
            model_path=model_path,
            input_size=input_size,
            confidence=confidence,
            iou=iou,
            classes=classes,
            device_id=device_id,
            **kwargs
        )

    elif backend in ['torch', 'pytorch', 'pt']:
        from .torch_engine import TorchInferEngine, create_torch_engine
        return create_torch_engine(
            model_path=model_path,
            input_size=input_size,
            confidence=confidence,
            iou=iou,
            classes=classes,
            device_id=device_id,
            **kwargs
        )

    elif backend in ['ultralytics', 'yolo', 'ultra']:
        from .ultralytics_engine import UltralyticsEngine, YOLOV8_CLASSES
        # 如果没有提供类别，使用YOLOv8默认类别
        if classes is None:
            classes = YOLOV8_CLASSES
        return UltralyticsEngine(
            model_path=model_path,
            input_size=input_size,
            confidence=confidence,
            iou=iou,
            classes=classes,
            device_id=device_id,
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}. "
                         f"Supported: tensorrt, onnx, torch, ultralytics, auto")


def _detect_backend(model_path: str, prefer_ultralytics: bool = True) -> str:
    """根据文件扩展名自动检测后端"""
    ext = os.path.splitext(model_path)[1].lower()

    # 检查Ultralytics是否可用 (YOLO模型优先使用Ultralytics)
    if prefer_ultralytics and ext in ['.pt', '.pth']:
        try:
            from ultralytics import YOLO
            return 'ultralytics'
        except ImportError:
            pass

    backend_map = {
        '.engine': 'tensorrt',
        '.trt': 'tensorrt',
        '.onnx': 'onnx',
        '.pt': 'torch',
        '.pth': 'torch',
        '.torchscript': 'torch',
        '.ts': 'torch',
    }

    if ext in backend_map:
        return backend_map[ext]

    # 默认尝试TensorRT -> ONNX -> Ultralytics -> PyTorch
    base = os.path.splitext(model_path)[0]
    if os.path.exists(base + '.engine'):
        return 'tensorrt'
    elif os.path.exists(base + '.onnx'):
        return 'onnx'
    elif os.path.exists(base + '.pt'):
        try:
            from ultralytics import YOLO
            return 'ultralytics'
        except ImportError:
            return 'torch'

    raise ValueError(f"Cannot auto-detect backend for: {model_path}")


# YOLO标准类别定义
YOLO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 安全帽检测类别
SAFETY_HELMET_CLASSES = {
    0: 'helmet',
    1: 'head',
    2: 'person'
}

# 引擎能力检测
def check_backend_availability() -> Dict[str, bool]:
    """
    检查各推理后端是否可用

    Returns:
        Dict[str, bool]: 各后端可用状态
    """
    availability = {
        'tensorrt': False,
        'onnx': False,
        'torch': False,
    }

    # 检查TensorRT
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        availability['tensorrt'] = True
    except ImportError:
        pass

    # 检查ONNX Runtime
    try:
        import onnxruntime as ort
        availability['onnx'] = True
    except ImportError:
        pass

    # 检查PyTorch
    try:
        import torch
        availability['torch'] = True
    except ImportError:
        pass

    return availability


# 模型转换工具
def convert_model(
    input_path: str,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    opset_version: int = 12
) -> str:
    """
    模型转换工具

    支持转换:
    - PyTorch (.pt) -> ONNX (.onnx)
    - ONNX (.onnx) -> TensorRT (.engine)

    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        input_size: 输入尺寸
        opset_version: ONNX opset版本

    Returns:
        output_path: 输出模型路径
    """
    input_ext = os.path.splitext(input_path)[1].lower()
    output_ext = os.path.splitext(output_path)[1].lower()

    # PT -> ONNX
    if input_ext in ['.pt', '.pth'] and output_ext == '.onnx':
        return _convert_pt_to_onnx(input_path, output_path, input_size, opset_version)

    # ONNX -> TensorRT
    elif input_ext == '.onnx' and output_ext in ['.engine', '.trt']:
        from .tensorrt_engine import create_tensorrt_engine
        create_tensorrt_engine(
            onnx_path=input_path,
            engine_path=output_path,
            input_size=input_size,
            force_rebuild=True
        )
        return output_path

    else:
        raise ValueError(f"Unsupported conversion: {input_ext} -> {output_ext}")


def _convert_pt_to_onnx(
    pt_path: str,
    onnx_path: str,
    input_size: Tuple[int, int],
    opset_version: int
) -> str:
    """PyTorch模型转ONNX"""
    import torch

    # 加载模型
    model = torch.load(pt_path, map_location='cpu')
    if isinstance(model, dict):
        model = model.get('model', model)

    model.eval()

    # 创建dummy输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )

    print(f"[Factory] Converted {pt_path} -> {onnx_path}")
    return onnx_path
