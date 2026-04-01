"""
检测功能测试脚本
用于验证检测器是否正常工作
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from datetime import datetime

from video_analytics.engines.factory import create_infer_engine
from video_analytics.engines.ultralytics_engine import YOLOV8_CLASSES, SAFETY_HELMET_CLASSES
from video_analytics.detectors.intrusion_detector import IntrusionDetector
from video_analytics.detectors.helmet_detector import HelmetDetector
from video_analytics.detectors.overcrowd_detector import OvercrowdDetector
from video_analytics.detectors.base_detector import DetectionContext


def test_person_detection():
    """测试人员检测"""
    print("=" * 60)
    print("测试人员检测")
    print("=" * 60)

    # 创建引擎
    engine = create_infer_engine(
        model_path="models/yolov8n.pt",
        backend="ultralytics",
        confidence=0.4
    )

    # 创建检测器
    detector = IntrusionDetector(
        engine=engine,
        config={
            "min_frames": 1,  # 测试时降低阈值
            "confidence": 0.4
        }
    )

    # 设置全图围栏 (用于测试)
    detector.set_fence_from_points(
        "test_cam",
        [(0, 0), (640, 0), (640, 480), (0, 480)]
    )

    # 使用真实图片测试
    test_image_path = input("请输入测试图片路径 (或按Enter使用随机图片): ").strip()

    if test_image_path and os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
    else:
        print("使用随机生成的测试图片")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 创建上下文
    context = DetectionContext(
        camera_id="test_cam",
        rtsp_url="",
        ip_address="127.0.0.1",
        frame=frame,
        timestamp=datetime.now(),
        frame_buffer=[frame],
        fps=25.0
    )

    # 执行检测
    print("\n执行检测...")
    result = detector.process(context)

    print(f"\n检测结果:")
    print(f"  - 是否触发: {result.triggered}")
    print(f"  - 检测人数: {len(result.detections)}")
    print(f"  - 调试信息: {result.debug_info}")

    if result.detections:
        print(f"\n检测到的人员:")
        for i, det in enumerate(result.detections):
            print(f"  [{i+1}] {det.class_name} (conf: {det.conf:.2f}) "
                  f"位置: ({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})")

    # 保存可视化结果
    if result.visualized_frame is not None:
        output_path = "test_intrusion_output.jpg"
        cv2.imwrite(output_path, result.visualized_frame)
        print(f"\n可视化结果已保存: {output_path}")

    engine.release()
    return result


def test_helmet_detection():
    """测试安全帽检测"""
    print("\n" + "=" * 60)
    print("测试安全帽检测")
    print("=" * 60)

    # 检查模型是否存在
    if not os.path.exists("models/safehat.pt"):
        print("安全帽模型不存在: models/safehat.pt")
        print("跳过此测试")
        return None

    # 创建引擎
    person_engine = create_infer_engine(
        model_path="models/yolov8n.pt",
        backend="ultralytics",
        confidence=0.4
    )

    helmet_engine = create_infer_engine(
        model_path="models/safehat.pt",
        backend="ultralytics",
        confidence=0.3,
        classes=SAFETY_HELMET_CLASSES
    )

    # 创建检测器
    detector = HelmetDetector(
        person_engine=person_engine,
        helmet_engine=helmet_engine,
        config={
            "min_frames": 1,
            "person_confidence": 0.4,
            "helmet_confidence": 0.3
        }
    )

    # 使用真实图片测试
    test_image_path = input("请输入测试图片路径 (或按Enter使用随机图片): ").strip()

    if test_image_path and os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
    else:
        print("使用随机生成的测试图片")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 创建上下文
    context = DetectionContext(
        camera_id="test_cam",
        rtsp_url="",
        ip_address="127.0.0.1",
        frame=frame,
        timestamp=datetime.now(),
        frame_buffer=[frame],
        fps=25.0
    )

    # 执行检测
    print("\n执行检测...")
    result = detector.process(context)

    print(f"\n检测结果:")
    print(f"  - 是否触发: {result.triggered}")
    print(f"  - 检测人数: {len(result.detections)}")
    print(f"  - 调试信息: {result.debug_info}")

    # 保存可视化结果
    if result.visualized_frame is not None:
        output_path = "test_helmet_output.jpg"
        cv2.imwrite(output_path, result.visualized_frame)
        print(f"\n可视化结果已保存: {output_path}")

    person_engine.release()
    helmet_engine.release()
    return result


def test_overcrowd_detection():
    """测试超员检测"""
    print("\n" + "=" * 60)
    print("测试超员检测")
    print("=" * 60)

    # 创建引擎
    engine = create_infer_engine(
        model_path="models/yolov8n.pt",
        backend="ultralytics",
        confidence=0.4
    )

    # 创建检测器
    detector = OvercrowdDetector(
        engine=engine,
        config={
            "max_people": 2,  # 测试时设为2人
            "duration_threshold": 0.1,  # 快速触发
            "confidence": 0.4
        }
    )

    # 使用真实图片测试
    test_image_path = input("请输入测试图片路径 (或按Enter使用随机图片): ").strip()

    if test_image_path and os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
    else:
        print("使用随机生成的测试图片")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 创建上下文
    context = DetectionContext(
        camera_id="test_cam",
        rtsp_url="",
        ip_address="127.0.0.1",
        frame=frame,
        timestamp=datetime.now(),
        frame_buffer=[frame],
        fps=25.0
    )

    # 执行检测
    print("\n执行检测...")
    result = detector.process(context)

    print(f"\n检测结果:")
    print(f"  - 是否触发: {result.triggered}")
    print(f"  - 检测人数: {len(result.detections)}")
    print(f"  - 调试信息: {result.debug_info}")

    # 保存可视化结果
    if result.visualized_frame is not None:
        output_path = "test_overcrowd_output.jpg"
        cv2.imwrite(output_path, result.visualized_frame)
        print(f"\n可视化结果已保存: {output_path}")

    engine.release()
    return result


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("视频分析检测功能测试")
    print("=" * 60)

    # 检查依赖
    try:
        from ultralytics import YOLO
        print("[Setup] Ultralytics OK")
    except ImportError:
        print("[Setup] 错误: 请先安装 ultralytics")
        print("        pip install ultralytics")
        return

    # 检查模型
    if not os.path.exists("models/yolov8n.pt"):
        print("\n[Setup] 错误: 人员检测模型不存在: models/yolov8n.pt")
        print("        请下载YOLOv8n模型并放入models/目录")
        return

    print("\n选择测试项目:")
    print("1. 人员闯入检测")
    print("2. 安全帽检测")
    print("3. 人员超员检测")
    print("4. 全部测试")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == "1":
        test_person_detection()
    elif choice == "2":
        test_helmet_detection()
    elif choice == "3":
        test_overcrowd_detection()
    elif choice == "4":
        test_person_detection()
        test_helmet_detection()
        test_overcrowd_detection()
    else:
        print("无效选项")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
