# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Video Analytics System** migrated from Ascend NPU to NVIDIA GPU. It performs real-time video analysis on RTSP streams with three detection scenarios:
- **Intrusion Detection** (Algorithm 1): Person enters a defined fence region
- **Helmet Detection** (Algorithm 2): Detects if workers are wearing safety helmets
- **Overcrowd Detection** (Algorithm 3): Detects when person count exceeds threshold

The system uses a modular architecture with pluggable inference backends (Ultralytics YOLO, TensorRT, ONNX Runtime, PyTorch).

## Common Commands

### Running the System

```bash
# Run the main application (production)
python main.py

# Run with specific config
python main.py  # Uses cfg/config.json

# Test individual components
python test_ultralytics.py      # Test inference engine
python test_detection.py        # Test detection algorithms
```

### Development & Testing

```bash
# Test engine with different backends
python -c "
from video_analytics.engines.factory import create_infer_engine
engine = create_infer_engine('models/yolov8n.pt', backend='ultralytics')
print(engine.get_stats())
"

# Run detector test with real image
python test_detection.py
# Then select: 1 (intrusion), 2 (helmet), or 3 (overcrowd)
```

### Model Management

```bash
# Models are stored in models/
# Required models:
# - models/yolov8n.pt        # Person detection (official YOLOv8)
# - models/safehat.pt        # Helmet detection (custom trained, classes: helmet, head)

# Convert models (if needed)
python -c "
from video_analytics.engines.factory import convert_model
convert_model('yolov8n.onnx', 'models/yolov8n.engine')
"
```

## High-Level Architecture

### Core Flow

```
RTSP Stream → StreamProcessor → Detector → EventStateMachine → Services
                                              ↓
                                   [Storage, Alarm, Video]
```

### Key Components

**1. Inference Engines** (`video_analytics/engines/`)
- `BaseInferEngine`: Abstract interface - all engines return `List[DetectionResult]`
- `UltralyticsEngine`: Primary backend using `YOLO.predict()`
- `TensorRTInferEngine`: Production backend for `.engine` files
- Factory pattern: `create_infer_engine(path, backend='ultralytics')` auto-detects backend from file extension

**2. Detectors** (`video_analytics/detectors/`)
- `BaseDetector`: Wraps an engine, processes `DetectionContext` → `DetectionResultBundle`
- Each detector has its own `EventStateMachine` per camera
- Detectors are registered to `StreamManager` by algorithm type ("1", "2", "3")

**3. State Machine** (`video_analytics/core/state_machine.py`)
- `EventStateMachine`: Manages event lifecycle (IDLE → COUNTING → TRIGGERED → ONGOING → COOLDOWN)
- Configurable: `min_trigger_frames`, `min_end_frames`, `cooldown_seconds`
- Replaces the old `defaultdict` global variables approach

**4. Stream Processing** (`video_analytics/core/stream_processor.py`)
- `StreamProcessor`: One per camera, manages RTSP connection, frame buffer, detector pipeline
- `StreamManager`: Manages multiple streams, handles camera polling
- Frame buffer: Keeps pre-event frames for video evidence generation

**5. Services** (`video_analytics/services/`)
- `StorageService`: MinIO or local file storage for images/videos
- `AlarmService`: HTTP POST or console output for alerts
- `VideoService`: Async video generation with pre/post frames

### Data Flow

```python
# 1. StreamProcessor reads frame from RTSP
frame = cap.read()

# 2. Creates context with frame buffer
context = DetectionContext(
    camera_id=camera_id,
    frame=frame,
    frame_buffer=deque([...]),  # Pre-event frames
    ...
)

# 3. Detector processes frame
result = detector.process(context)  # Returns DetectionResultBundle
# - Runs inference: engine.infer(frame)
# - Filters by class (e.g., only person class 0)
# - Checks fence regions (for intrusion)
# - Updates EventStateMachine
# - Returns triggered=True/False

# 4. If triggered, handle event
if result.triggered:
    storage.upload_image(result.visualized_frame)
    alarm.send_alarm(result.event)
    video.async_generate_and_upload(context.frame_buffer)
```

### Algorithm Types

| Type | Detector | Model | Classes | Notes |
|------|----------|-------|---------|-------|
| "1" | IntrusionDetector | yolov8n.pt | person(0) | Requires fence region setup |
| "2" | HelmetDetector | yolov8n.pt + safehat.pt | person(0), helmet(0), head(1) | Two-stage: detect person → crop → classify helmet |
| "3" | OvercrowdDetector | yolov8n.pt | person(0) | Counts persons, triggers if > max_people |

## Important Implementation Details

### Fence Configuration (Intrusion Detection)

If no fence is set for a camera, intrusion detector defaults to full-frame detection:

```python
# In main.py or initialization:
intrusion_detector = detectors["1"]
intrusion_detector.set_fence_from_points(
    camera_id="1997855044199911425",
    points=[(100, 100), (500, 100), (500, 400), (100, 400)]  # Clockwise from top-left
)
```

### Helmet Detection Two-Stage Process

1. Primary engine detects persons
2. For each person, crop region with padding
3. Secondary engine (safehat.pt) classifies helmet/head
4. "head" class without "helmet" = violation

### Configuration File (cfg/config.json)

```json
{
  "model": {
    "person_model_path": "models/yolov8n.pt",
    "helmet_model_path": "models/safehat.pt",
    "backend": "ultralytics",  // ultralytics | tensorrt | onnx | torch
    "confidence": 0.4
  },
  "detection": {
    "intrusion_min_frames": 25,      // Debounce frames
    "intrusion_cooldown": 60,        // Seconds between alerts
    "helmet_min_frames": 25,
    "overcrowd_max_people": 15
  },
  "storage": {
    "type": "local",  // local | minio
    "local_path": "./output"
  },
  "alarm": {
    "type": "console"  // console | http | async
  }
}
```

### Camera Data Format

Cameras are loaded from API or hardcoded in `main.py`:
```python
cameras = [(
    camera_id: str,      # e.g., "1997855044199911425"
    rtsp_url: str,       # e.g., "rtsp://admin:pass@ip:554/..."
    ip_address: str,     # Camera IP
    algorithm_types: str # "1" or "1,2" or "1,2,3"
)]
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point: initializes engines/detectors/services, camera poll loop |
| `video_analytics/engines/ultralytics_engine.py` | Primary inference backend |
| `video_analytics/detectors/intrusion_detector.py` | Algorithm 1 implementation |
| `video_analytics/detectors/helmet_detector.py` | Algorithm 2 (two-stage) |
| `video_analytics/core/state_machine.py` | Event lifecycle management |
| `video_analytics/core/stream_processor.py` | RTSP stream handling |
| `cfg/config.json` | Runtime configuration |

## Debugging Tips

- Enable debug output: Check `StreamProcessor._process_frame()` for `[Debug]` logs every 100 frames
- Test without RTSP: Use `test_detection.py` with a static image
- Check fence setup: If no detections for intrusion, verify fence is set or default full-frame is active
- Model classes: Helmet model uses `{0: 'helmet', 1: 'head'}`, not standard COCO classes
