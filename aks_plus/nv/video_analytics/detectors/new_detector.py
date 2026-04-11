from typing import Dict, List, Optional
from video_analytics.detectors.base_detector import (
    BaseDetector, DetectionContext, DetectionResultBundle,
    DetectionEventType
)


class NewDetector(BaseDetector):
    """新算法检测器示例（例如：烟雾检测、摔倒检测等）"""

    def __init__(self, engine, config: Dict = None):
        super().__init__(
            engine=engine,
            event_type=DetectionEventType.GATHERING,  # 或用新的类型
            config=config or {}
        )
        # 从配置获取参数
        self.min_frames = config.get("min_frames", 25)
        self.confidence = config.get("confidence", 0.5)
        self.cooldown_seconds = config.get("cooldown_seconds", 60)

    def process(self, context: DetectionContext) -> DetectionResultBundle:
        """核心处理逻辑"""
        camera_id = context.camera_id
        frame = context.frame

        # 1. 运行推理
        detections = self._infer(frame)

        # 2. 过滤目标类别（如需要）
        # filtered = self._filter_by_class(detections, [0], self.confidence)

        # 3. 业务逻辑判断
        condition = len(detections) > 0  # 你的判断条件

        # 4. 状态机管理
        state = self._get_state(camera_id).update(condition)

        # 5. 构造结果
        triggered = state in ["TRIGGERED", "ONGOING"]

        return DetectionResultBundle(
            triggered=triggered,
            event=self._create_event(context, detections, 0.9, {}) if triggered else None,
            detections=detections,
            visualized_frame=self._draw_detections(frame.copy(), detections)
        )
