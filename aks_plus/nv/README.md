# Video Analytics System - NVIDIA GPU Edition

基于NVIDIA GPU (TensorRT/ONNX/PyTorch/Ultralytics) 的实时视频分析系统，从昇腾NPU迁移重构而来。

## 特性

- **多后端推理**: Ultralytics (推荐)、TensorRT (生产)、ONNXRuntime (备选)、PyTorch (调试)
- **统一推理接口**: `BaseInferEngine` 抽象层，无缝切换后端
- **业务解耦**: 检测器、状态机、流处理完全分离
- **三大检测场景**: 人员闯入、安全帽检测、人员超员
- **状态机管理**: 统一的事件生命周期管理 (防抖、冷却)
- **服务化架构**: 存储、报警、视频服务完全解耦

## 项目结构

```
video_analytics/
├── core/                      # 核心组件
│   ├── state_machine.py      # 事件状态机
│   └── stream_processor.py   # 流处理器/管理器
├── engines/                   # 推理引擎
│   ├── base_engine.py        # 抽象基类
│   ├── tensorrt_engine.py    # TensorRT实现
│   ├── onnx_engine.py        # ONNXRuntime实现
│   ├── torch_engine.py       # PyTorch实现
│   └── factory.py            # 工厂函数
├── detectors/                 # 检测器
│   ├── base_detector.py      # 抽象基类
│   ├── intrusion_detector.py # 闯入检测
│   ├── helmet_detector.py    # 安全帽检测
│   └── overcrowd_detector.py # 超员检测
├── services/                  # 服务层
│   ├── storage_service.py    # 存储服务 (MinIO/本地)
│   ├── alarm_service.py      # 报警服务 (HTTP/异步)
│   └── video_service.py      # 视频服务
├── config/                    # 配置
│   └── settings.py           # 配置管理
└── __init__.py

main.py                        # 主入口
MIGRATION_GUIDE.md            # 迁移指南
```

## 快速开始

### 1. 环境安装

**推荐方案 (Ultralytics - 最简单)**
```bash
# 安装Ultralytics (自动安装PyTorch)
pip install ultralytics

# 基础依赖
pip install opencv-python numpy requests minio
```

**其他方案 (可选)**
```bash
# TensorRT (生产环境最佳性能)
pip install tensorrt pycuda

# ONNX Runtime (跨平台)
pip install onnxruntime-gpu
```

### 2. 模型准备

**方式1: 使用Ultralytics官方模型 (推荐)**
```bash
# 下载YOLOv8n模型
mkdir -p models
# 首次运行时会自动下载，或手动下载:
# wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -O models/yolov8n.pt
```

**方式2: TensorRT引擎 (生产环境)**
```bash
# 将 .engine 文件放入 models/ 目录
# 或使用ONNX转换:
python -c "
from video_analytics.engines.factory import convert_model
convert_model('yolov8n.onnx', 'models/yolov8n.engine')
"
```

### 3. 配置

创建 `cfg/config.json` (参考 `cfg/config.json.example`):

**Ultralytics配置 (推荐)**
```json
{
  "model": {
    "person_model_path": "models/yolov8n.pt",
    "helmet_model_path": "models/safehat.pt",
    "backend": "ultralytics",
    "confidence": 0.4,
    "device_id": 0
  },
  "detection": {
    "intrusion_min_frames": 25,
    "helmet_min_frames": 25,
    "overcrowd_max_people": 15
  },
  "storage": {
    "type": "local",
    "local_path": "./output"
  },
  "alarm": {
    "type": "console"
  }
}
```

**TensorRT配置 (生产环境)**
```json
{
  "model": {
    "person_model_path": "models/yolov8n.engine",
    "helmet_model_path": "models/safehat.engine",
    "backend": "tensorrt",
    "confidence": 0.4,
    "device_id": 0,
    "fp16": true
  }
}
```

### 4. 运行

```bash
python main.py
```

## 核心类使用示例

### 推理引擎

```python
from video_analytics.engines.factory import create_infer_engine

# TensorRT
engine = create_infer_engine(
    model_path="yolov8n.engine",
    backend="tensorrt",
    confidence=0.4
)

# 推理
detections, context = engine.infer(frame)
for det in detections:
    print(f"Detected {det.class_name} at {det.to_xyxy()}")
```

### 检测器

```python
from video_analytics.detectors.intrusion_detector import IntrusionDetector

# 创建检测器
detector = IntrusionDetector(engine, config={"min_frames": 25})

# 设置围栏
detector.set_fence_from_points("cam_001", [
    (100, 100), (500, 100), (500, 400), (100, 400)
])

# 处理帧
result = detector.process(context)
if result.triggered:
    print(f"Event: {result.event}")
```

### 状态机

```python
from video_analytics.core.state_machine import EventStateMachine, EventState

sm = EventStateMachine(min_trigger_frames=25, cooldown_seconds=60)
state = sm.update(detected=True)

if state == EventState.TRIGGERED:
    print("Event started!")
elif state == EventState.COOLDOWN:
    print("Event ended")
```

## 推理后端对比

| 后端 | 速度 | 易用性 | 适用场景 |
|------|------|--------|----------|
| **Ultralytics** | ⭐⭐ | ⭐⭐⭐ | **推荐 - 简单高效** |
| TensorRT | ⭐⭐⭐ | ⭐ | 生产环境极限性能 |
| ONNXRuntime | ⭐⭐ | ⭐⭐ | 跨平台部署 |
| PyTorch | ⭐ | ⭐⭐⭐ | 调试/原型 |

切换后端只需修改配置:
```python
# Ultralytics (推荐 - 自动处理YOLO模型)
backend="ultralytics", model_path="model.pt"

# TensorRT (生产环境)
backend="tensorrt", model_path="model.engine"

# ONNX
backend="onnx", model_path="model.onnx"
```

## 迁移指南

详见 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## 性能测试

```python
from video_analytics.engines.factory import create_infer_engine

engine = create_infer_engine("yolov8n.engine", backend="tensorrt")

# 预热
engine.warmup(10)

# 查看统计
print(engine.get_stats())
# {
#   'inference_count': 1000,
#   'avg_inference_time': 0.003,
#   'fps': 333.3
# }
```

## License

MIT
