"""
Overcrowding Detector - 人员超员/聚集检测
检测区域内人员数量是否超过阈值
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
class OvercrowdConfig:
    """超员检测配置"""
    max_people: int = 15              # 最大允许人数
    duration_threshold: float = 2.0   # 持续秒数 (超过此时间才报警)
    min_end_seconds: float = 3.0      # 结束持续秒数
    cooldown_seconds: float = 60.0    # 冷却时间
    confidence: float = 0.4           # 检测置信度


class OvercrowdDetector(BaseDetector):
    """
    人员超员检测器

    检测逻辑:
    1. 使用YOLO检测所有人员
    2. 统计人员数量
    3. 判断人数是否超过阈值
    4. 应用状态机防抖 (持续N秒才触发)
    5. 触发报警

    配置参数:
        max_people: 最大允许人数 (默认15)
        duration_threshold: 持续秒数阈值 (默认2秒)
        cooldown_seconds: 事件冷却时间 (默认60秒)
        confidence: 人员检测置信度 (默认0.4)
    """

    def __init__(
        self,
        engine: BaseInferEngine,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            engine=engine,
            config=config,
            event_type=DetectionEventType.OVERCROWD
        )

        # 配置
        self.max_people = self.config.get("max_people", 15)
        self.duration_threshold = self.config.get("duration_threshold", 2.0)
        self.min_end_seconds = self.config.get("min_end_seconds", 3.0)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 60)
        self.confidence = self.config.get("confidence", 0.4)

        # 状态机管理
        self._state_machines: Dict[str, EventStateMachine] = {}

    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """
        处理单帧并检测超员

        Args:
            context: 检测上下文

        Returns:
            DetectionResultBundle: 检测结果
        """
        import time
        start_time = time.perf_counter()

        camera_id = context.camera_id
        frame = context.frame
        fps = context.fps

        # 1. 执行推理
        detections, infer_context = self._infer(frame)

        # 2. 过滤人员
        person_dets = self._filter_by_class(detections, [0], self.confidence)
        people_count = len(person_dets)

        # 3. 判断是否超员
        is_overcrowd = people_count > self.max_people

        # 4. 计算所需帧数
        trigger_frames = int(self.duration_threshold * fps)
        end_frames = int(self.min_end_seconds * fps)

        # 5. 状态机处理
        if camera_id not in self._state_machines:
            self._state_machines[camera_id] = EventStateMachine(
                min_trigger_frames=trigger_frames,
                min_end_frames=end_frames,
                cooldown_seconds=self.cooldown_seconds
            )

        state_machine = self._state_machines[camera_id]
        state = state_machine.update(is_overcrowd)

        # 6. 可视化
        visualized = self._visualize(frame, person_dets, people_count, state)

        # 7. 构建结果
        result = DetectionResultBundle(
            triggered=state in [EventState.TRIGGERED, EventState.ONGOING],
            detections=person_dets,
            visualized_frame=visualized,
            debug_info={
                "state": state.value,
                "people_count": people_count,
                "max_people": self.max_people,
                "is_overcrowd": is_overcrowd,
                "frames_needed": trigger_frames,
                "current_frames": state_machine._consecutive_count
            }
        )

        # 8. 事件触发
        if state == EventState.TRIGGERED:
            result.event = self._create_event(context, person_dets, people_count)
            self._trigger_count += 1

        self._process_count += 1
        self._total_process_time += time.perf_counter() - start_time

        return result

    def _visualize(
        self,
        frame: np.ndarray,
        person_dets: List[DetectionResult],
        people_count: int,
        state: EventState
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            frame: 原始帧
            person_dets: 人员检测结果
            people_count: 人数
            state: 当前状态

        Returns:
            可视化后的帧
        """
        img = frame.copy()

        # 状态颜色
        state_color = {
            EventState.IDLE: (0, 255, 0),
            EventState.COOLDOWN: (255, 255, 0),
            EventState.TRIGGERED: (0, 0, 255),
            EventState.ONGOING: (0, 0, 255),
            EventState.ENDING: (0, 165, 255),
        }.get(state, (128, 128, 128))

        # 是否超员决定框颜色
        if people_count > self.max_people:
            box_color = (0, 0, 255)  # 红色 - 超员
        else:
            box_color = (0, 255, 0)  # 绿色 - 正常

        # 绘制所有人员
        for det in person_dets:
            cv2.rectangle(
                img,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                box_color, 2
            )

        # 人数统计背景
        text = f"Count: {people_count}/{self.max_people}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )

        cv2.rectangle(
            img,
            (10, 10),
            (20 + text_w, 20 + text_h + 10),
            state_color,
            -1
        )

        # 人数文字
        cv2.putText(
            img, text,
            (15, 15 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2
        )

        # 状态文字
        status_text = f"State: {state.value}"
        cv2.putText(
            img, status_text,
            (10, 50 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, state_color, 2
        )

        # 超员警告
        if people_count > self.max_people:
            warning_text = "OVERCROWDING WARNING!"
            (warn_w, warn_h), _ = cv2.getTextSize(
                warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            cv2.putText(
                img, warning_text,
                (img.shape[1] // 2 - warn_w // 2, img.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3
            )

        return img

    def _create_event(
        self,
        context: DetectionContext,
        person_dets: List[DetectionResult],
        people_count: int
    ) -> DetectionEvent:
        """创建超员事件"""
        objects = []
        for det in person_dets:
            objects.append({
                "label": "person",
                "confidence": round(det.conf, 2),
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "status": "detected"
            })

        avg_conf = sum(d.conf for d in person_dets) / len(person_dets) if person_dets else 0

        return DetectionEvent(
            event_type=self.event_type,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            objects=objects,
            confidence=avg_conf,
            metadata={
                "people_count": people_count,
                "max_people": self.max_people,
                "excess": people_count - self.max_people
            }
        )

    def update_config(self, max_people: Optional[int] = None):
        """动态更新配置"""
        if max_people is not None:
            self.max_people = max_people
