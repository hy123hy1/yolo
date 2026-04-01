"""
Ultralytics YOLO 引擎测试脚本
用于验证人员检测和安全帽检测模型
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from video_analytics.engines.factory import create_infer_engine
from video_analytics.engines.ultralytics_engine import YOLOV8_CLASSES, SAFETY_HELMET_CLASSES


def test_person_detection():
    """测试人员检测模型 (yolov8n.pt)"""
    print("=" * 60)
    print("Testing Person Detection Model (YOLOv8n)")
    print("=" * 60)

    # 创建引擎
    engine = create_infer_engine(
        model_path="models/yolov8n.pt",
        backend="ultralytics",
        confidence=0.4,
        iou=0.45,
        classes=YOLOV8_CLASSES,
        device_id=0,
        verbose=False
    )

    print(f"[Test] Engine created: {engine}")
    print(f"[Test] Model classes: {engine.model_names}")

    # 预热
    print("[Test] Warming up...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.warmup(num_runs=3)

    # 测试图片 (使用随机生成的测试图像)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("[Test] Running inference...")
    detections, context = engine.infer(test_frame)

    print(f"[Test] Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det.class_name} (conf: {det.conf:.2f}) at {det.to_xyxy()}")

    # 查看统计
    stats = engine.get_stats()
    print(f"[Test] Inference time: {stats['avg_inference_time']*1000:.2f}ms")
    print(f"[Test] FPS: {stats['fps']:.1f}")

    engine.release()
    print("[Test] Person detection test passed!\n")
    return True


def test_helmet_detection():
    """测试安全帽检测模型 (safehat.pt)"""
    print("=" * 60)
    print("Testing Helmet Detection Model (Safehat)")
    print("=" * 60)

    model_path = "models/safehat.pt"
    if not os.path.exists(model_path):
        print(f"[Test] Model not found: {model_path}")
        print("[Test] Skipping helmet detection test\n")
        return False

    # 创建引擎
    engine = create_infer_engine(
        model_path=model_path,
        backend="ultralytics",
        confidence=0.3,
        iou=0.45,
        classes=SAFETY_HELMET_CLASSES,  # {0: 'helmet', 1: 'head'}
        device_id=0,
        verbose=False
    )

    print(f"[Test] Engine created: {engine}")
    print(f"[Test] Model classes: {engine.model_names}")

    # 预热
    print("[Test] Warming up...")
    engine.warmup(num_runs=3)

    # 测试
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("[Test] Running inference...")
    detections, context = engine.infer(test_frame)

    print(f"[Test] Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det.class_name} (conf: {det.conf:.2f}) at {det.to_xyxy()}")

    stats = engine.get_stats()
    print(f"[Test] Inference time: {stats['avg_inference_time']*1000:.2f}ms")

    engine.release()
    print("[Test] Helmet detection test passed!\n")
    return True


def test_with_real_image(image_path: str = None):
    """使用真实图片测试"""
    if image_path is None or not os.path.exists(image_path):
        print("[Test] No real image provided, skipping real image test")
        return

    print("=" * 60)
    print(f"Testing with real image: {image_path}")
    print("=" * 60)

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Test] Failed to load image: {image_path}")
        return

    # 创建引擎
    engine = create_infer_engine(
        model_path="models/yolov8n.pt",
        backend="ultralytics",
        confidence=0.4
    )

    # 推理
    detections, context = engine.infer(frame)

    print(f"[Test] Detected {len(detections)} objects")
    for det in detections:
        print(f"  - {det.class_name}: {det.conf:.2f}")
        # 绘制框
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det.class_name} {det.conf:.2f}",
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    output_path = "test_output.jpg"
    cv2.imwrite(output_path, frame)
    print(f"[Test] Output saved to: {output_path}")

    engine.release()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Ultralytics YOLO Engine Test Suite")
    print("=" * 60 + "\n")

    # 检查依赖
    try:
        from ultralytics import YOLO
        print("[Setup] Ultralytics is installed")
    except ImportError:
        print("[Setup] ERROR: Ultralytics not installed!")
        print("[Setup] Please run: pip install ultralytics")
        return

    try:
        import torch
        print(f"[Setup] PyTorch version: {torch.__version__}")
        print(f"[Setup] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Setup] CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[Setup] WARNING: PyTorch not installed")

    # 创建模型目录
    os.makedirs("models", exist_ok=True)

    # 运行测试
    try:
        # 测试人员检测
        if os.path.exists("models/yolov8n.pt"):
            test_person_detection()
        else:
            print("[Test] yolov8n.pt not found in models/")
            print("[Test] Please download it from: https://github.com/ultralytics/assets/releases/")

        # 测试安全帽检测
        test_helmet_detection()

        # 真实图片测试 (可选)
        # test_with_real_image("test_image.jpg")

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"[Test] ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
