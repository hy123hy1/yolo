# 昇腾NPU 迁移到 NVIDIA GPU 迁移指南

## 目录
1. [架构对比](#架构对比)
2. [快速开始](#快速开始)
3. [详细迁移步骤](#详细迁移步骤)
4. [核心代码对比](#核心代码对比)
5. [性能优化建议](#性能优化建议)
6. [常见问题](#常见问题)

---

## 架构对比

### 原昇腾架构 (test.py)
```
InferSession (OM模型)
    ↓
parse_yolo_outputs (手动解析)
    ↓
全局状态变量 (defaultdict)
    ↓
直接处理业务逻辑
```

### 新NVIDIA架构
```
BaseInferEngine (抽象接口)
    ├── TensorRTInferEngine (生产)
    ├── ONNXInferEngine (备选)
    └── TorchInferEngine (调试)
            ↓
    BaseDetector (业务抽象)
        ├── IntrusionDetector (闯入)
        ├── HelmetDetector (安全帽)
        └── OvercrowdDetector (超员)
                ↓
        EventStateMachine (状态管理)
                ↓
        StreamProcessor (流处理)
                ↓
        Services (存储/报警/视频)
```

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖 (TensorRT优先)
pip install tensorrt pycuda opencv-python numpy requests

# 或 ONNX Runtime (备选)
pip install onnxruntime-gpu opencv-python numpy requests

# 或 PyTorch (调试)
pip install torch torchvision opencv-python numpy requests
```

### 2. 模型转换

```bash
# 从ONNX转换为TensorRT (推荐)
python -c "
from video_analytics.engines.tensorrt_engine import create_tensorrt_engine
create_tensorrt_engine(
    onnx_path='yolov8n.onnx',
    engine_path='yolov8n.engine',
    input_size=(640, 640),
    fp16=True
)
"
```

### 3. 运行系统

```bash
python main.py
```

---

## 详细迁移步骤

### 步骤1: 替换InferSession为统一引擎

**原代码:**
```python
from ais_bench.infer.interface import InferSession

session = InferSession(device_id=0, model_path="yolov8n.om")
outs = session.infer(feeds=[blob], mode="static")
```

**新代码:**
```python
from video_analytics.engines.factory import create_infer_engine

# 方式1: TensorRT (推荐)
engine = create_infer_engine(
    model_path="yolov8n.engine",
    backend="tensorrt",
    input_size=(640, 640),
    confidence=0.4,
    iou=0.45
)

# 方式2: ONNX Runtime (备选)
engine = create_infer_engine(
    model_path="yolov8n.onnx",
    backend="onnx",
    input_size=(640, 640)
)

# 方式3: PyTorch (调试)
engine = create_infer_engine(
    model_path="yolov8n.pt",
    backend="torch",
    device_id=0
)

# 统一推理接口
detections, context = engine.infer(frame)
```

### 步骤2: 重构检测逻辑为Detector类

**原代码:**
```python
# 全局状态变量
intrusion_state = defaultdict(lambda: {"active": False, ...})
intrusion_frame_counter = defaultdict(int)

def detect_intrusion(frame, camera_id, ...):
    # 手动预处理
    blob = cv2.dnn.blobFromImage(...)
    outs = session.infer(feeds=[blob], mode="static")
    detections = parse_yolo_outputs(outs, frame.shape)

    # 手动状态管理
    if valid_dets:
        intrusion_frame_counter[camera_id] += 1
    else:
        intrusion_frame_counter[camera_id] = 0

    if intrusion_frame_counter[camera_id] < MIN_FRAMES:
        return False, []

    # 事件处理逻辑...
```

**新代码:**
```python
from video_analytics.detectors.intrusion_detector import IntrusionDetector
from video_analytics.core.state_machine import EventStateMachine

# 创建检测器 (每个类型一个实例)
detector = IntrusionDetector(
    engine=engine,
    config={
        "min_frames": 25,
        "confidence": 0.5,
        "cooldown_seconds": 60
    }
)

# 设置电子围栏 (可选)
detector.set_fence_from_points(
    camera_id="camera_001",
    points=[(100, 100), (500, 100), (500, 400), (100, 400)]
)

# 处理帧
from video_analytics.detectors.base_detector import DetectionContext

context = DetectionContext(
    camera_id=camera_id,
    rtsp_url=rtsp_url,
    ip_address=ip,
    frame=frame,
    timestamp=datetime.now(),
    frame_buffer=frame_buffer,
    fps=25.0
)

result = detector.process(context)

if result.triggered:
    print(f"Event: {result.event}")
    # result.event 包含所有事件信息
    # result.visualized_frame 是可视化后的图像
```

### 步骤3: 替换全局状态变量为状态机

**原代码:**
```python
# 多个分散的全局变量
intrusion_state = defaultdict(lambda: {...})
intrusion_frame_counter = defaultdict(int)
intrusion_disappear_counter = defaultdict(int)
last_alarm_time = {}
```

**新代码:**
```python
from video_analytics.core.state_machine import EventStateMachine

# 统一的状态机 (封装在每个Detector内部)
state_machine = EventStateMachine(
    min_trigger_frames=25,    # 防抖帧数
    min_end_frames=25,        # 结束确认帧数
    cooldown_seconds=60       # 冷却时间
)

# 每帧更新
state = state_machine.update(detected=True)

if state == EventState.TRIGGERED:
    print("事件首次触发!")
elif state == EventState.ONGOING:
    print("事件持续中...")
elif state == EventState.COOLDOWN:
    print("事件结束，冷却中")
```

### 步骤4: 重构流处理为StreamProcessor

**原代码:**
```python
def process_stream(camera_id, rtsp_url, ...):
    cap = cv2.VideoCapture(rtsp_url)
    frame_buffer = deque(maxlen=FPS * PRE_SECONDS)

    while True:
        ret, frame = cap.read()
        if not ret:
            # 重连逻辑
            ...

        frame_buffer.append(frame)

        # 检测逻辑
        if "1" in algorithmtypes:
            detect_intrusion(frame, ...)
        if "2" in algorithmtypes:
            check_helmet_and_alert(frame, ...)

        # 手动处理报警上传
        ...
```

**新代码:**
```python
from video_analytics.core.stream_processor import StreamManager, StreamConfig
from video_analytics.services.storage_service import StorageServiceFactory
from video_analytics.services.alarm_service import AlarmServiceFactory
from video_analytics.services.video_service import VideoService

# 创建服务
storage = StorageServiceFactory.create(
    service_type="minio",
    endpoint="172.21.3.141:8084",
    access_key="root",
    secret_key="12345678"
)

alarm = AlarmServiceFactory.create(
    service_type="http",
    endpoints=["http://172.21.3.141:8080/alarm"]
)

video = VideoService(storage)

# 创建流管理器
stream_manager = StreamManager(storage, alarm, video)

# 注册检测器
stream_manager.register_detector("1", intrusion_detector)
stream_manager.register_detector("2", helmet_detector)
stream_manager.register_detector("3", overcrowd_detector)

# 添加流
stream_config = StreamConfig(
    camera_id="camera_001",
    rtsp_url="rtsp://...",
    ip_address="192.168.1.100",
    algorithm_types={"1", "2"},  # 启用1号+2号算法
    fps=25,
    skip_frames=0
)

stream_manager.add_stream(stream_config)
```

### 步骤5: 配置迁移

**原配置 (config.json):**
```json
{
  "server_endpoints": ["http://..."],
  "confidence_threshold": 0.5
}
```

**新配置 (cfg/config.json):**
```json
{
  "model": {
    "person_model_path": "models/yolov8n.engine",
    "helmet_model_path": "models/safehat.engine",
    "backend": "tensorrt",
    "confidence": 0.4,
    "iou": 0.45,
    "device_id": 0,
    "fp16": true
  },
  "detection": {
    "intrusion_min_frames": 25,
    "intrusion_cooldown": 60,
    "helmet_min_frames": 25,
    "overcrowd_max_people": 15
  },
  "storage": {
    "type": "minio",
    "endpoint": "172.21.3.141:8084",
    "access_key": "root",
    "secret_key": "12345678"
  },
  "alarm": {
    "type": "http",
    "endpoints_intrusion": ["http://..."],
    "endpoints_helmet": ["http://..."]
  }
}
```

---

## 核心代码对比

### 推理接口对比

| 操作 | 昇腾 (原) | NVIDIA (新) |
|------|----------|-------------|
| 初始化 | `InferSession(device_id=0, model_path="x.om")` | `create_infer_engine(model_path="x.engine", backend="tensorrt")` |
| 预处理 | 手动letterbox + blob | 自动内部处理 |
| 推理 | `session.infer(feeds=[blob], mode="static")` | `engine.infer(frame)` |
| 后处理 | 手动parse_yolo_outputs | 自动NMS + 坐标映射 |
| 输出 | 原始numpy数组 | `List[DetectionResult]` |

### 状态管理对比

| 功能 | 昇腾 (原) | NVIDIA (新) |
|------|----------|-------------|
| 防抖计数 | `intrusion_frame_counter[camera_id] += 1` | `EventStateMachine.update(detected=True)` |
| 事件状态 | 多个defaultdict变量 | 统一`EventState`枚举 |
| 冷却时间 | 手动管理时间戳 | 内置`cooldown_seconds` |
| 状态流转 | 手写if-else | 自动状态机管理 |

---

## 性能优化建议

### 1. TensorRT优化

```python
# 使用FP16半精度
engine = create_infer_engine(
    model_path="yolov8n.engine",
    backend="tensorrt",
    fp16=True  # 速度提升约2倍
)

# 预热
engine.warmup(num_runs=10)
```

### 2. 批量推理 (高级)

```python
# 多路流共享引擎时，可开启异步推理
# TODO: 实现真正的batch推理
```

### 3. GPU内存优化

```python
# 监控GPU内存
import pycuda.driver as cuda

context.pop()
cuda.Context.synchronize()
```

### 4. 多流并行

```python
# 每个流独立线程，充分利用多核CPU
stream_manager = StreamManager(...)
# 自动管理多流并发
```

---

## 常见问题

### Q1: TensorRT引擎构建失败?

```bash
# 检查TensorRT安装
python -c "import tensorrt; print(tensorrt.__version__)"

# 从ONNX重新构建
python -c "
from video_analytics.engines.factory import convert_model
convert_model('yolov8n.onnx', 'yolov8n.engine')
"
```

### Q2: CUDA内存不足?

```python
# 减少并行流数量
# 或使用更小的batch size
engine = create_infer_engine(
    model_path="yolov8n.engine",
    max_batch_size=1
)
```

### Q3: 如何回退到ONNX?

```python
# 只需修改backend参数
engine = create_infer_engine(
    model_path="yolov8n.onnx",  # 改为onnx文件
    backend="onnx"              # 改为onnx后端
)
```

### Q4: 原OM模型能用吗?

```bash
# OM是昇腾专用格式，需要转换为ONNX
# 使用ATC工具或重新导出:
# 1. 从原始框架导出ONNX
# 2. 从ONNX转换为TensorRT
```

### Q5: 如何调试?

```python
# 使用PyTorch后端可调试
engine = create_infer_engine(
    model_path="yolov8n.pt",
    backend="torch"
)

# 或使用控制台报警服务
alarm = AlarmServiceFactory.create(service_type="console")

# 或使用本地存储
storage = StorageServiceFactory.create(service_type="local")
```

---

## 迁移检查清单

- [ ] 安装NVIDIA驱动 (>= 520)
- [ ] 安装CUDA (>= 11.8)
- [ ] 安装TensorRT (>= 8.6) 或 ONNX Runtime
- [ ] 转换OM模型为ONNX/TensorRT格式
- [ ] 配置cfg/config.json
- [ ] 测试单路流
- [ ] 测试多路流
- [ ] 验证报警功能
- [ ] 验证视频生成功能
- [ ] 性能对比测试

---

## 总结

| 特性 | 原昇腾实现 | 新NVIDIA实现 |
|------|-----------|-------------|
| 代码复杂度 | 高 (全部耦合) | 低 (模块化) |
| 推理性能 | 昇腾NPU优化 | TensorRT GPU优化 |
| 可维护性 | 差 (全局变量) | 好 (类封装) |
| 可扩展性 | 差 | 好 (插件化) |
| 多后端支持 | 否 | 是 (TRT/ONNX/Torch) |
| 单元测试 | 困难 | 容易 |
