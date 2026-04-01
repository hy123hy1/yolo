"""
Intrusion Detector - 人员闯入检测
电子围栏场景：检测人员进入指定区域
"""
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass

from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle,
    DetectionEvent, DetectionEventType
)
from video_analytics.engines.base_engine import BaseInferEngine, DetectionResult
from video_analytics.core.state_machine import EventStateMachine, EventState


@dataclass
class FenceRegion:
    """电子围栏区域"""
    points: List[tuple]  # 多边形顶点 [(x1,y1), (x2,y2), ...]
    name: str = "fence"

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在区域内"""
        return cv2.pointPolygonTest(
            np.array(self.points, dtype=np.int32),
            (float(x), float(y)),
            False
        ) >= 0

    def draw(self, frame: np.ndarray, color: tuple = (0, 0, 255), thickness: int = 2):
        """在图像上绘制围栏"""
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)


class IntrusionDetector(BaseDetector):
    """
    人员闯入检测器

    检测逻辑:
    1. 使用YOLO检测所有人员
    2. 计算人员中心点
    3. 判断中心点是否在电子围栏区域内
    4. 应用状态机防抖 (连续N帧检测到才触发)
    5. 事件冷却 (避免重复报警)

    配置参数:
        min_frames: 连续检测帧数阈值 (默认25帧)
        confidence: 人员检测置信度 (默认0.5)
        cooldown_seconds: 事件冷却时间 (默认60秒)
        target_classes: 目标类别ID列表 (默认[0] - person)
    """

    def __init__(
        self,
        engine: BaseInferEngine,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            engine=engine,
            config=config,
            event_type=DetectionEventType.INTRUSION
        )

        # 配置
        self.min_frames = self.config.get("min_frames", 25)
        self.confidence = self.config.get("confidence", 0.5)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 60)
        self.target_classes = self.config.get("target_classes", [0])  # person

        # 状态机管理 {camera_id: EventStateMachine}
        self._state_machines: Dict[str, EventStateMachine] = {}

        # 围栏配置 {camera_id: FenceRegion}
        self._fences: Dict[str, FenceRegion] = {}

    def set_fence(self, camera_id: str, fence: FenceRegion):
        """设置电子围栏"""
        self._fences[camera_id] = fence

    def set_fence_from_points(self, camera_id: str, points: List[tuple]):
        """从点列表设置围栏"""
        self._fences[camera_id] = FenceRegion(points=points)

    def set_fence_full_frame(self, camera_id: str, margin: float = 0.0):
        """
        设置全图围栏 (用于全画面检测)

        Args:
            camera_id: 摄像头ID
            margin: 边距比例 (0.0 = 全图, 0.1 = 留10%边距)
        """
        # 使用一个占位尺寸，实际会在第一次检测时根据帧尺寸调整
        # 或者在知道分辨率的情况下直接设置
        self._fences[camera_id] = FenceRegion(
            points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            name="full_frame_auto"
        )
        # 标记为需要自动调整
        self._fences[camera_id]._auto_resize = True
        self._fences[camera_id]._margin = margin

    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """
        处理单帧并检测闯入事件

        Args:
            context: 检测上下文

        Returns:
            DetectionResultBundle: 检测结果
        """
        import time
        start_time = time.perf_counter()

        camera_id = context.camera_id
        frame = context.frame

        # 检查围栏配置
        if camera_id not in self._fences:
            # 如果没有设置围栏，默认检测整个画面 (只要有人就算闯入)
            # 或者设置为None表示全图检测
            print(f"[IntrusionDetector] Warning: No fence configured for {camera_id}, "
                  f"using full frame detection")
            # 创建一个覆盖全图的围栏
            h, w = frame.shape[:2]
            self._fences[camera_id] = FenceRegion(
                points=[(0, 0), (w, 0), (w, h), (0, h)],
                name="full_frame"
            )

        fence = self._fences[camera_id]

        # 1. 执行推理
        detections, infer_context = self._infer(frame)

        # 2. 过滤目标类别
        person_dets = self._filter_by_class(
            detections,
            self.target_classes,
            self.confidence
        )

        # 3. 检测围栏内人员
        intruders = self._detect_intruders(person_dets, fence)

        # 4. 获取/创建状态机
        if camera_id not in self._state_machines:
            self._state_machines[camera_id] = EventStateMachine(
                min_trigger_frames=self.min_frames,
                min_end_frames=25,
                cooldown_seconds=self.cooldown_seconds
            )

        state_machine = self._state_machines[camera_id]

        # 5. 更新状态机
        has_intrusion = len(intruders) > 0
        state = state_machine.update(has_intrusion)

        # 6. 可视化
        visualized = self._visualize(frame, person_dets, intruders, fence, state)

        # 7. 构建结果
        result = DetectionResultBundle(
            triggered=state in [EventState.TRIGGERED, EventState.ONGOING],
            detections=person_dets,
            visualized_frame=visualized,
            debug_info={
                "state": state.value,
                "intruder_count": len(intruders),
                "total_persons": len(person_dets),
                "infer_time": infer_context.inference_time
            }
        )

        # 8. 事件触发时创建事件对象
        if state == EventState.TRIGGERED:
            result.event = self._create_event(context, intruders)
            self._trigger_count += 1

        self._process_count += 1
        self._total_process_time += time.perf_counter() - start_time

        return result

    def _detect_intruders(
        self,
        detections: List[DetectionResult],
        fence: FenceRegion
    ) -> List[DetectionResult]:
        """
        检测围栏内的人员

        Args:
            detections: 检测结果
            fence: 围栏区域

        Returns:
            围栏内的人员列表
        """
        intruders = []

        for det in detections:
            # 计算中心点
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2

            # 检查是否在围栏内
            if fence.contains_point(cx, cy):
                intruders.append(det)

        return intruders

    def _visualize(
        self,
        frame: np.ndarray,
        all_persons: List[DetectionResult],
        intruders: List[DetectionResult],
        fence: FenceRegion,
        state: EventState
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            frame: 原始帧
            all_persons: 所有检测到的人员
            intruders: 闯入人员
            fence: 围栏区域
            state: 当前状态

        Returns:
            可视化后的帧
        """
        img = frame.copy()

        # 绘制围栏 (根据状态改变颜色)
        fence_color = {
            EventState.IDLE: (0, 255, 0),      # 绿色 - 正常
            EventState.COOLDOWN: (255, 255, 0), # 青色 - 冷却
            EventState.TRIGGERED: (0, 0, 255),  # 红色 - 触发
            EventState.ONGOING: (0, 0, 255),    # 红色 - 持续
            EventState.ENDING: (0, 165, 255),   # 橙色 - 结束中
        }.get(state, (128, 128, 128))

        fence.draw(img, color=fence_color, thickness=2)

        # 绘制所有人员 (绿色)
        for det in all_persons:
            if det not in intruders:
                cv2.rectangle(
                    img,
                    (int(det.x1), int(det.y1)),
                    (int(det.x2), int(det.y2)),
                    (0, 255, 0), 2
                )

        # 绘制闯入人员 (红色)
        for det in intruders:
            cv2.rectangle(
                img,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                (0, 0, 255), 3
            )

            # 中心点
            cx = int((det.x1 + det.x2) / 2)
            cy = int((det.y1 + det.y2) / 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # 状态文字
        status_text = f"State: {state.value} | Intruders: {len(intruders)}"
        cv2.putText(
            img, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fence_color, 2
        )

        return img

    def _create_event(
        self,
        context: DetectionContext,
        intruders: List[DetectionResult]
    ) -> DetectionEvent:
        """创建闯入事件"""
        objects = []
        for det in intruders:
            objects.append({
                "label": det.class_name or "person",
                "confidence": round(det.conf, 2),
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "status": "intrusion"
            })

        avg_conf = sum(d.conf for d in intruders) / len(intruders) if intruders else 0

        return DetectionEvent(
            event_type=self.event_type,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            objects=objects,
            confidence=avg_conf,
            metadata={
                "intruder_count": len(intruders),
                "fence_name": self._fences.get(context.camera_id, FenceRegion([])).name
            }
        )

    def clear_fence(self, camera_id: str):
        """清除围栏配置"""
        self._fences.pop(camera_id, None)
        self._state_machines.pop(camera_id, None)
