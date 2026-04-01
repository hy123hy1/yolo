"""
ONNX Runtime Inference Engine Implementation
Cross-platform inference backend with CUDA/TensorRT execution providers
"""
import os
import warnings
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2
import threading

from .base_engine import BaseInferEngine, DetectionResult, InferenceContext, LetterBoxPreprocessor

# ONNX Runtime导入
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    warnings.warn(f"ONNX Runtime not available: {e}. ONNXInferEngine will not function.")


class ONNXInferEngine(BaseInferEngine):
    """
    ONNX Runtime推理引擎

    特性:
    - 多执行提供程序支持 (CUDA, TensorRT, CPU)
    - 自动优化图
    - 线程安全
    - 动态输入形状

    使用示例:
        engine = ONNXInferEngine(
            model_path="yolov8n.onnx",
            input_size=(640, 640),
            confidence=0.4,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
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
        providers: Optional[List[str]] = None,
        session_options: Optional['ort.SessionOptions'] = None,
        **kwargs
    ):
        """
        初始化ONNX Runtime推理引擎

        Args:
            model_path: ONNX模型文件路径
            providers: 执行提供程序列表，优先级从高到低
                      可选: 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
            session_options: 自定义SessionOptions
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available. Please install onnxruntime-gpu.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        self.providers = providers or self._get_default_providers(device_id)
        self.session_options = session_options or self._create_session_options()
        self.device_id = device_id

        # ONNX组件
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None

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

    def _get_default_providers(self, device_id: int) -> List[str]:
        """获取默认执行提供程序"""
        providers = []

        # 检查TensorRT
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            providers.append('TensorrtExecutionProvider')

        # 检查CUDA
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')

        # CPU兜底
        providers.append('CPUExecutionProvider')

        return providers

    def _create_session_options(self) -> 'ort.SessionOptions':
        """创建优化的SessionOptions"""
        sess_options = ort.SessionOptions()

        # 图优化级别: 启用所有优化
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 线程数设置
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        # 内存优化
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        return sess_options

    def _initialize(self):
        """初始化ONNX Runtime会话"""
        # 配置provider选项
        provider_options = []

        for provider in self.providers:
            if provider == 'CUDAExecutionProvider':
                provider_options.append({
                    'device_id': self.device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                })
            elif provider == 'TensorrtExecutionProvider':
                provider_options.append({
                    'device_id': self.device_id,
                    'trt_max_workspace_size': 2147483648,  # 2GB
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache',
                })
            else:
                provider_options.append({})

        # 创建推理会话
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=self.session_options,
            providers=self.providers,
            provider_options=provider_options
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"[ONNX] Model loaded: {self.model_path}")
        print(f"[ONNX] Input: {self.input_name} {self.input_shape}")
        print(f"[ONNX] Outputs: {self.output_names}")
        print(f"[ONNX] Providers: {self.session.get_providers()}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, InferenceContext]:
        """
        ONNX预处理
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
        """ONNX推理"""
        with self._thread_lock:
            outputs = self.session.run(
                self.output_names,
                {self.input_name: blob}
            )
        return outputs

    def _postprocess(self, raw_outputs: List[np.ndarray], context: InferenceContext) -> List[DetectionResult]:
        """解析YOLO输出并应用NMS"""
        if not raw_outputs:
            return []

        # 处理不同数量的输出
        if len(raw_outputs) == 1:
            output = raw_outputs[0]
        else:
            # 多输出情况，选择合适的一个
            output = self._select_main_output(raw_outputs)

        # 处理不同输出格式
        if output.ndim == 3:
            if output.shape[1] == 84:  # YOLOv8格式 [batch, 84, 8400]
                detections = self._parse_yolov8_output(output[0], context)
            elif output.shape[2] == 6:  # [batch, num_dets, 6]
                detections = self._parse_decoded_output(output[0], context)
            else:
                detections = self._parse_generic_output(output, context)
        elif output.ndim == 2:
            if output.shape[1] == 6:
                detections = self._parse_decoded_output(output, context)
            else:
                detections = self._parse_generic_output(output, context)
        else:
            detections = self._parse_generic_output(output, context)

        # NMS
        if len(detections) > 0:
            detections = self._apply_nms(detections)

        return detections

    def _select_main_output(self, outputs: List[np.ndarray]) -> np.ndarray:
        """从多输出中选择主输出"""
        # 优先选择维度为3且形状合理的输出
        for out in outputs:
            if out.ndim == 3 and out.shape[1] in [84, 80, 5, 6]:
                return out
        # 默认返回第一个
        return outputs[0]

    def _parse_yolov8_output(self, output: np.ndarray, context: InferenceContext) -> List[DetectionResult]:
        """解析YOLOv8输出 [84, 8400]"""
        ori_h, ori_w = context.original_shape

        # 转置为 [8400, 84]
        predictions = output.T

        # 提取boxes和scores
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]

        # 获取每个box的最佳类别
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

    @staticmethod
    def build_tensorrt_engine(
        onnx_path: str,
        engine_path: str,
        input_size: Tuple[int, int] = (640, 640),
        fp16: bool = True,
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30
    ) -> str:
        """
        使用ONNX Runtime TensorRT provider构建引擎
        (注意: 这不是原生的TensorRT引擎，需要TensorrtExecutionProvider)

        Args:
            onnx_path: ONNX模型路径
            engine_path: 引擎缓存路径
            input_size: 输入尺寸
            fp16: 是否使用FP16
            max_batch_size: 最大batch size
            workspace_size: 工作空间大小

        Returns:
            engine_path: 引擎路径
        """
        os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else '.', exist_ok=True)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        provider_options = [{
            'device_id': 0,
            'trt_max_workspace_size': workspace_size,
            'trt_fp16_enable': fp16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.dirname(engine_path) or './',
            'trt_profile_min_shapes': f'images:1x3x{input_size[0]}x{input_size[1]}',
            'trt_profile_max_shapes': f'images:{max_batch_size}x3x{input_size[0]}x{input_size[1]}',
            'trt_profile_opt_shapes': f'images:1x3x{input_size[0]}x{input_size[1]}',
        }]

        # 创建会话会触发引擎构建
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=['TensorrtExecutionProvider'],
            provider_options=provider_options
        )

        print(f"[ONNX] TensorRT engine built and cached")
        return engine_path

    def release(self):
        """释放资源"""
        if self.session:
            del self.session
            self.session = None
        print("[ONNX] Session released")


# 便捷创建函数
def create_onnx_engine(
    model_path: str,
    input_size: Tuple[int, int] = (640, 640),
    use_cuda: bool = True,
    use_tensorrt: bool = False,
    device_id: int = 0,
    **kwargs
) -> ONNXInferEngine:
    """
    便捷创建ONNX引擎

    Args:
        model_path: ONNX模型路径
        input_size: 输入尺寸
        use_cuda: 是否使用CUDA
        use_tensorrt: 是否使用TensorRT provider (需要ONNX Runtime TensorRT)
        device_id: GPU设备ID

    Returns:
        ONNXInferEngine实例
    """
    providers = []

    if use_tensorrt and 'TensorrtExecutionProvider' in ort.get_available_providers():
        providers.append('TensorrtExecutionProvider')

    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')

    providers.append('CPUExecutionProvider')

    return ONNXInferEngine(
        model_path=model_path,
        input_size=input_size,
        providers=providers,
        device_id=device_id,
        **kwargs
    )
