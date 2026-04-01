# 快速开始指南

## 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装Ultralytics和相关依赖
pip install ultralytics opencv-python numpy requests minio
```

### 2. 准备模型

```bash
# 创建模型目录
mkdir models

# 下载YOLOv8n官方模型 (会自动下载，或手动放置)
# 官方模型地址: https://github.com/ultralytics/assets/releases/

# 将您的模型放入models目录:
# - models/yolov8n.pt        (人员检测)
# - models/safehat.pt        (安全帽检测，自定义训练)
```

### 3. 配置文件

```bash
# 复制示例配置
copy cfg\config.json.example cfg\config.json

# 编辑 cfg/config.json，修改为您的摄像头信息
```

**最小化配置示例** (`cfg/config.json`):
```json
{
  "model": {
    "person_model_path": "models/yolov8n.pt",
    "helmet_model_path": "models/safehat.pt",
    "backend": "ultralytics"
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

### 4. 运行测试

```bash
# 测试Ultralytics引擎
python test_ultralytics.py

# 运行主程序
python main.py
```

## 模型说明

### 人员检测模型 (yolov8n.pt)
- **来源**: Ultralytics官方YOLOv8n模型
- **类别**: 80类COCO数据集 (person, car, bicycle等)
- **用途**: 人员闯入检测、人员超员检测

### 安全帽检测模型 (safehat.pt)
- **来源**: 自定义训练 (基于YOLOv8n)
- **类别**:
  - `0: helmet` - 佩戴安全帽
  - `1: head` - 未佩戴安全帽
- **用途**: 安全帽检测

## 算法编号说明

| 编号 | 算法 | 说明 |
|------|------|------|
| 1 | 人员闯入 | 电子围栏检测 |
| 2 | 安全帽检测 | 检测是否佩戴安全帽 |
| 3 | 人员超员 | 检测人数是否超过阈值 |

摄像头算法配置格式: `"1"` 或 `"1,2"` 或 `"1,2,3"`

## 常见问题

### Q: 没有GPU可以使用吗?
A: 可以，Ultralytics会自动使用CPU:
```python
# 在config.json中设置
device_id: -1  # 使用CPU
```

### Q: 如何测试单张图片?
```python
from video_analytics.engines.factory import create_infer_engine
import cv2

engine = create_infer_engine("models/yolov8n.pt", backend="ultralytics")
frame = cv2.imread("test.jpg")
detections, _ = engine.infer(frame)
for det in detections:
    print(f"{det.class_name}: {det.conf:.2f}")
```

### Q: 如何添加电子围栏?
```python
# 在main.py中初始化检测器后添加
intrusion_detector = detectors["1"]
intrusion_detector.set_fence_from_points(
    camera_id="your_camera_id",
    points=[(100, 100), (500, 100), (500, 400), (100, 400)]  # 多边形顶点
)
```
