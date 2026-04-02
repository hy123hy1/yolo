# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Video Analytics System** migrated from Ascend NPU to NVIDIA GPU. It performs real-time video analysis on RTSP streams with three detection scenarios:
- **Intrusion Detection** (Algorithm 1): Person enters a defined fence region
- **Helmet Detection** (Algorithm 2): Detects if workers are wearing safety helmets
- **Overcrowd Detection** (Algorithm 3): Detects when person count exceeds threshold

The system uses a modular architecture with pluggable inference backends (Ultralytics YOLO, TensorRT, ONNX Runtime, PyTorch).

## Build And Test

- Install: `pip install -r requirements.txt`
- Dev (API mode): `python main_api.py`
- Dev (legacy poll mode): `python main.py`
- Test inference: `python test_ultralytics.py`
- Test detection: `python test_detection.py`

## Architecture Boundaries

- Inference engines live in `video_analytics/engines/`
- Detectors live in `video_analytics/detectors/`
- Stream processing lives in `video_analytics/core/`
- Services (storage/alarm/video) live in `video_analytics/services/`
- Do not put business logic in API handlers (keep handlers thin)
- Shared types live in `video_analytics/detectors/base_detector.py`

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

**4. Stream Processing** (`video_analytics/core/stream_processor.py`)
- `StreamProcessor`: One per camera, manages RTSP connection, frame buffer, detector pipeline
- `StreamManager`: Manages multiple streams, handles camera polling
- Frame buffer: Keeps pre-event frames for video evidence generation

**5. Services** (`video_analytics/services/`)
- `StorageService`: MinIO or local file storage for images/videos
- `AlarmService`: HTTP POST or console output for alerts
- `VideoService`: Async video generation with pre/post frames

## Coding Conventions

- Prefer dependency injection (engines/services passed to constructors)
- Do not introduce new global state without explicit justification
- Reuse existing detector base class from `video_analytics/detectors/base_detector.py`
- Follow existing patterns: `DetectionContext` in, `DetectionResultBundle` out
- Use type hints for public methods

### Algorithm Types

| Type | Detector | Model | Classes | Notes |
|------|----------|-------|---------|-------|
| "1" | IntrusionDetector | yolov8n.pt | person(0) | Requires fence region setup |
| "2" | HelmetDetector | yolov8n.pt + safehat.pt | person(0), helmet(0), head(1) | Two-stage detection |
| "3" | OvercrowdDetector | yolov8n.pt | person(0) | Counts persons |

### Configuration

Configuration lives in `cfg/config.json`:
```json
{
  "model": {
    "person_model_path": "models/yolov8n.pt",
    "helmet_model_path": "models/safehat.pt",
    "backend": "ultralytics"
  },
  "detection": {
    "intrusion_min_frames": 25,
    "intrusion_cooldown": 60
  }
}
```

## Safety Rails

### NEVER

- Modify `cfg/config.json` or model paths without checking existing camera configs
- Remove feature flags (like `use_reloader`) without searching all call sites
- Commit without testing the API endpoint (`/set_fence`)
- Use blocking operations in detector `process()` method (it runs per-frame)
- Modify fence configuration without proper thread locking

### ALWAYS

- Show diff before committing
- Update fence coordinates when camera resolution changes
- Handle `KeyboardInterrupt` with proper cleanup in main entry points
- Test with `algorithmType` 1, 2, and 3 if modifying base detector logic
- Verify FFmpeg processes are cleaned up on shutdown

## Verification

- Backend changes: `python test_detection.py` + `python test_ultralytics.py`
- API changes: test with curl:
  ```bash
  curl -X POST "http://localhost:5005/set_fence" \
    -H "Content-Type: application/json" \
    -d '{"cam_id": "test", "url": "rtsp://...", "algorithmType": 1, "fence_area": {...}}'
  ```
- Shutdown test: Ctrl+C should kill all processes immediately without hanging
- Stream restart test: `/set_fence` with existing `cam_id` should restart cleanly

## Key Files

| File | Purpose |
|------|---------|
| `main_api.py` | API entry point: Flask server with event-driven stream management |
| `main.py` | Legacy entry point: polling-based camera discovery |
| `video_analytics/engines/ultralytics_engine.py` | Primary inference backend |
| `video_analytics/detectors/intrusion_detector.py` | Algorithm 1 with fence support |
| `video_analytics/core/stream_processor.py` | RTSP stream handling |
| `video_analytics/core/state_machine.py` | Event lifecycle management |
| `cfg/config.json` | Runtime configuration |

## Compact Instructions

Preserve:

1. Architecture decisions (event-driven API vs polling, pluggable inference backends)
2. Modified files and key changes:
   - `main_api.py`: Flask API + fence_worker process management
   - `video_analytics/core/stream_processor.py`: StreamProcessor stop logic
   - `video_analytics/detectors/intrusion_detector.py`: Fence configuration
3. Current verification status:
   - API mode: `python main_api.py` → test with curl
   - Shutdown: Ctrl+C should exit immediately
   - Stream restart: `/set_fence` with existing cam_id works
4. Open risks, TODOs, rollback notes:
   - Fence coordinate scaling depends on frontend `default_area`
   - FFmpeg subprocess cleanup relies on `psutil` + `os._exit()`
   - Detection and fence drawing run on same RTSP source (decoupled)
