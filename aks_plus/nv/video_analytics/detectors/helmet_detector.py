"""
Helmet Detector - 安全帽检测
检测人员是否佩戴安全帽
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from datetime import datetime

from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle,
    DetectionEvent, DetectionEventType, MultiStageDetector
)
from video_analytics.engines.base_engine import BaseInferEngine, DetectionResult
from video_analytics.core.state_machine import EventStateMachine, EventState


class HelmetDetector(MultiStageDetector):
    """
    安全帽检测器

    检测逻辑:
    1. 使用YOLO检测所有人员 (primary_engine)
    2. 裁剪人员区域
    3. 使用安全帽检测模型检测是否佩戴 (secondary_engine)
    4. 应用状态机防抖
    5. 触发报警

    配置参数:
        min_frames: 连续检测帧数阈值 (默认25帧)
        person_confidence: 人员检测置信度 (默认0.4)
        helmet_confidence: 安全帽检测置信度 (默认0.5)
        cooldown_seconds: 事件冷却时间 (默认60秒)
        crop_padding: 裁剪外扩比例 (默认0.2)
    """

    def __init__(
        self,
        person_engine: BaseInferEngine,      # 人员检测模型
        helmet_engine: BaseInferEngine,      # 安全帽检测模型
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            primary_engine=person_engine,
            secondary_engine=helmet_engine,
            config=config,
            event_type=DetectionEventType.NO_HELMET
        )

        # 配置
        self.min_frames = self.config.get("min_frames", 25)
        self.person_confidence = self.config.get("person_confidence", 0.4)
        self.helmet_confidence = self.config.get("helmet_confidence", 0.5)
        self.cooldown_seconds = self.config.get("cooldown_seconds", 60)
        self.crop_padding = self.config.get("crop_padding", 0.2)

        # 类别定义 (安全帽模型)
        self.HELMET_CLASS = 0  # helmet
        self.HEAD_CLASS = 1    # head (未戴安全帽)

        # 状态机管理
        self._state_machines: Dict[str, EventStateMachine] = {}

    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """
        处理单帧并检测安全帽

        Args:
            context: 检测上下文

        Returns:
            DetectionResultBundle: 检测结果
        """
        import time
        start_time = time.perf_counter()

        camera_id = context.camera_id
        frame = context.frame

        # 1. 检测人员
        person_dets, _ = self._infer(frame)
        person_dets = self._filter_by_class(person_dets, [0], self.person_confidence)

        if not person_dets:
            return DetectionResultBundle(
                triggered=False,
                detections=[],
                debug_info={"no_persons": True}
            )

        # 2. 检测安全帽
        no_helmet_list = []
        helmet_list = []

        for person_det in person_dets:
            # 裁剪人员区域
            region = self._crop_region(
                frame,
                (person_det.x1, person_det.y1, person_det.x2, person_det.y2),
                self.crop_padding
            )

            if region.size == 0:
                continue

            # 安全帽检测
            helmet_dets, _ = self._secondary_infer(region)

            # 分析检测结果
            has_helmet, has_head = self._analyze_helmet(helmet_dets)

            if has_head and not has_helmet:
                no_helmet_list.append(person_det)
            elif has_helmet:
                helmet_list.append(person_det)

        # 3. 状态机处理
        if camera_id not in self._state_machines:
            self._state_machines[camera_id] = EventStateMachine(
                min_trigger_frames=self.min_frames,
                min_end_frames=25,
                cooldown_seconds=self.cooldown_seconds
            )

        state_machine = self._state_machines[camera_id]
        has_violation = len(no_helmet_list) > 0
        state = state_machine.update(has_violation)

        # 4. 可视化
        visualized = self._visualize(
            frame, person_dets, no_helmet_list, helmet_list, state
        )

        # 5. 构建结果
        result = DetectionResultBundle(
            triggered=state in [EventState.TRIGGERED, EventState.ONGOING],
            detections=person_dets,
            visualized_frame=visualized,
            debug_info={
                "state": state.value,
                "no_helmet_count": len(no_helmet_list),
                "with_helmet_count": len(helmet_list),
                "total_persons": len(person_dets)
            }
        )

        # 6. 事件触发
        if state == EventState.TRIGGERED:
            result.event = self._create_event(context, no_helmet_list)
            self._trigger_count += 1

        self._process_count += 1
        self._total_process_time += time.perf_counter() - start_time

        return result

    def _analyze_helmet(self, detections: List[DetectionResult]) -> Tuple[bool, bool]:
        """
        分析安全帽检测结果

        Args:
            detections: 安全帽检测结果

        Returns:
            (has_helmet, has_head): 是否检测到安全帽/头部
        """
        has_helmet = False
        has_head = False

        for det in detections:
            if det.conf < self.helmet_confidence:
                continue

            if det.class_id == self.HELMET_CLASS:
                has_helmet = True
            elif det.class_id == self.HEAD_CLASS:
                has_head = True

        return has_helmet, has_head

    def _visualize(
        self,
        frame: np.ndarray,
        all_persons: List[DetectionResult],
        no_helmet_list: List[DetectionResult],
        helmet_list: List[DetectionResult],
        state: EventState
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            frame: 原始帧
            all_persons: 所有人员
            no_helmet_list: 未戴安全帽人员
            helmet_list: 戴安全帽人员
            state: 当前状态

        Returns:
            可视化后的帧
        """
        img = frame.copy()

        # 绘制戴安全帽人员 (绿色)
        for det in helmet_list:
            cv2.rectangle(
                img,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                (0, 255, 0), 2
            )
            cv2.putText(
                img, "Helmet",
                (int(det.x1), int(det.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # 绘制未戴安全帽人员 (红色)
        for det in no_helmet_list:
            cv2.rectangle(
                img,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                (0, 0, 255), 3
            )
            cv2.putText(
                img, "NO HELMET",
                (int(det.x1), int(det.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        # 状态颜色
        fence_color = {
            EventState.IDLE: (0, 255, 0),
            EventState.COOLDOWN: (255, 255, 0),
            EventState.TRIGGERED: (0, 0, 255),
            EventState.ONGOING: (0, 0, 255),
            EventState.ENDING: (0, 165, 255),
        }.get(state, (128, 128, 128))

        # 状态文字
        status_text = f"State: {state.value} | No Helmet: {len(no_helmet_list)}"
        cv2.putText(
            img, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fence_color, 2
        )

        return img

    def _create_event(
        self,
        context: DetectionContext,
        no_helmet_list: List[DetectionResult]
    ) -> DetectionEvent:
        """创建安全帽事件"""
        objects = []
        for det in no_helmet_list:
            objects.append({
                "label": "no_helmet",
                "confidence": round(det.conf, 2),
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "status": "warning"
            })

        avg_conf = sum(d.conf for d in no_helmet_list) / len(no_helmet_list) if no_helmet_list else 0

        return DetectionEvent(
            event_type=self.event_type,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            objects=objects,
            confidence=avg_conf,
            metadata={
                "violation_count": len(no_helmet_list),
                "total_persons": len(objects) + len(context.metadata.get("helmet_list", []))
            }
        )
