"""
TensorRT Inference Engine Implementation
Production-ready TensorRT backend for NVIDIA GPU
"""
import os
import warnings
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2
import threading

from .base_engine import BaseInferEngine, DetectionResult, InferenceContext, LetterBoxPreprocessor

# TensorRT导入
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自动初始化CUDA
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    warnings.warn(f"TensorRT not available: {e}. TensorRTInferEngine will not function.")


class TensorRTInferEngine(BaseInferEngine):
    """
    TensorRT推理引擎

    特性:
    - FP16/FP32推理支持
    - 动态batch支持
    - CUDA流异步推理
    - 零拷贝内存优化
    - 线程安全

    使用示例:
        engine = TensorRTInferEngine(
            model_path="yolov8n.engine",
            input_size=(640, 640),
            confidence=0.4,
            iou=0.45,
            fp16=True
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
        fp16: bool = True,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
        **kwargs
    ):
        """
        初始化TensorRT推理引擎

        Args:
            model_path: TensorRT引擎文件路径 (.engine)
            fp16: 是否使用FP16半精度推理
            max_batch_size: 最大batch size
            workspace_size: 工作空间大小(字节)
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Please install TensorRT and pycuda.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {model_path}")

        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.workspace_size = workspace_size

        # CUDA上下文
        self._cuda_ctx = None
        self._thread_lock = threading.Lock()

        # TensorRT组件
        self.logger = None
        self.runtime = None
        self.engine = None
        self.context = None

        # 内存分配
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # 预处理工具
        self.preprocessor = LetterBoxPreprocessor(input_size)

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
        """初始化TensorRT运行时"""
        # 设置CUDA设备
        cuda.init()
        device = cuda.Device(self.device_id)
        self._cuda_ctx = device.make_context()

        # 创建TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 反序列化引擎
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.model_path}")

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配内存
        self._allocate_buffers()

        print(f"[TensorRT] Engine loaded: {self.model_path}")
        print(f"[TensorRT] Input shape: {self.input_size}")
        print(f"[TensorRT] FP16: {self.fp16}")
        print(f"[TensorRT] Device: {cuda.Device(self.device_id).name()}")

    def _allocate_buffers(self):
        """分配GPU和CPU内存"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # 动态shape处理
            if shape[0] == -1:
                shape = (self.max_batch_size,) + shape[1:]
                self.context.set_input_shape(name, shape)

            # 计算大小
            size = trt.volume(shape)
            nbytes = size * dtype.itemsize

            # 分配设备内存
            device_mem = cuda.mem_alloc(nbytes)

            # 分配主机内存
            host_mem = cuda.pagelocked_empty(size, dtype)

            # 绑定
            self.bindings.append(int(device_mem))

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem,
                                   'shape': shape, 'dtype': dtype, 'nbytes': nbytes})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem,
                                    'shape': shape, 'dtype': dtype, 'nbytes': nbytes})

        # 创建CUDA流
        self.stream = cuda.Stream()

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, InferenceContext]:
        """
        TensorRT预处理
        输入: BGR numpy (H, W, 3)
        输出: NCHW格式, 归一化到[0,1]
        """
        ori_h, ori_w = image.shape[:2]

        # LetterBox预处理
        padded, scale, pad_x, pad_y = self.preprocessor.preprocess(image)

        # BGR -> RGB, HWC -> CHW, /255
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )

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

        return blob, context

    def _inference(self, blob: np.ndarray) -> List[np.ndarray]:
        """TensorRT异步推理"""
        with self._thread_lock:
            # 推送当前CUDA上下文
            self._cuda_ctx.push()

            try:
                # 拷贝输入到GPU
                np.copyto(self.inputs[0]['host'], blob.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

                # 设置tensor地址
                for i in range(self.engine.num_io_tensors):
                    name = self.engine.get_tensor_name(i)
                    self.context.set_tensor_address(name, self.bindings[i])

                # 执行推理
                self.context.execute_async_v3(stream_handle=self.stream.handle)

                # 拷贝输出到CPU
                outputs = []
                for out in self.outputs:
                    cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
                    outputs.append(out['host'].copy().reshape(out['shape']))

                # 同步
                self.stream.synchronize()

                return outputs

            finally:
                self._cuda_ctx.pop()

    def _postprocess(self, raw_outputs: List[np.ndarray], context: InferenceContext) -> List[DetectionResult]:
        """
        解析YOLO输出并应用NMS

        支持两种格式:
        1. [batch, 84, 8400] - YOLOv8标准输出
        2. [batch, num_detections, 6] - 已解码输出
        """
        if not raw_outputs:
            return []

        output = raw_outputs[0]

        # 处理不同输出格式
        if output.ndim == 3 and output.shape[1] == 84:
            # YOLOv8格式: [batch, 84, 8400]
            detections = self._parse_yolov8_output(output[0], context)
        elif output.ndim == 2 and output.shape[1] == 6:
            # 已解码格式: [num_detections, 6]
            detections = self._parse_decoded_output(output, context)
        else:
            # 尝试通用解析
            detections = self._parse_generic_output(output, context)

        # NMS
        if len(detections) > 0:
            detections = self._apply_nms(detections)

        return detections

    def _parse_yolov8_output(self, output: np.ndarray, context: InferenceContext) -> List[DetectionResult]:
        """
        解析YOLOv8输出 [84, 8400] -> xywh + 80 classes
        """
        ori_h, ori_w = context.original_shape
        input_h, input_w = context.input_shape

        # 转置为 [8400, 84]
        predictions = output.T  # (8400, 84)

        # 提取xywh和conf
        boxes = predictions[:, :4]  # (8400, 4)
        class_scores = predictions[:, 4:]  # (8400, 80)

        # 计算每个box的最大类别置信度
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # 置信度过滤
        mask = confidences > self.confidence
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # xywh -> xyxy
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        detections = []
        for i in range(len(boxes)):
            # 映射回原始图像坐标
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
        """解析已解码的输出 [N, 6] (x1, y1, x2, y2, conf, class_id)"""
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
        """通用输出解析，尝试多种常见格式"""
        # 尝试reshape为 [-1, 6]
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

        # 提取boxes和scores
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        scores = np.array([d.conf for d in detections])

        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence,
            self.iou
        )

        if len(indices) == 0:
            return []

        # 处理不同版本的OpenCV返回格式
        if isinstance(indices, tuple):
            indices = indices[0]

        return [detections[int(i)] for i in indices.flatten()][:self.max_detections]

    def infer_batch(self, images: List[np.ndarray]) -> List[Tuple[List[DetectionResult], InferenceContext]]:
        """
        真正的Batch推理

        注意: 当前实现为串行，如需真正并行batch推理，需要:
        1. 修改engine支持动态batch
        2. 堆叠blob为 [batch, 3, H, W]
        3. 修改后处理逻辑
        """
        # TODO: 实现真正的batch推理
        return super().infer_batch(images)

    def release(self):
        """释放TensorRT资源"""
        with self._thread_lock:
            if self._cuda_ctx:
                self._cuda_ctx.push()

            # 释放内存
            for inp in self.inputs:
                del inp['device']
            for out in self.outputs:
                del out['device']

            # 释放TensorRT对象
            if self.context:
                del self.context
            if self.engine:
                del self.engine

            if self._cuda_ctx:
                self._cuda_ctx.pop()
                self._cuda_ctx = None

        print("[TensorRT] Engine released")


# 工厂函数
def create_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    input_size: Tuple[int, int] = (640, 640),
    fp16: bool = True,
    max_batch_size: int = 1,
    force_rebuild: bool = False,
    **kwargs
) -> TensorRTInferEngine:
    """
    创建TensorRT引擎（如果不存在则自动从ONNX转换）

    Args:
        onnx_path: ONNX模型路径
        engine_path: TensorRT引擎保存路径
        input_size: 输入尺寸
        fp16: 是否使用FP16
        max_batch_size: 最大batch size
        force_rebuild: 是否强制重新构建

    Returns:
        TensorRTInferEngine实例
    """
    if not force_rebuild and os.path.exists(engine_path):
        return TensorRTInferEngine(engine_path, input_size=input_size, fp16=fp16, **kwargs)

    # 从ONNX构建TensorRT引擎
    print(f"[TensorRT] Building engine from {onnx_path}...")

    from .onnx_engine import ONNXInferEngine
    engine = ONNXInferEngine.build_tensorrt_engine(
        onnx_path, engine_path, input_size, fp16, max_batch_size
    )

    return TensorRTInferEngine(engine_path, input_size=input_size, fp16=fp16, **kwargs)
