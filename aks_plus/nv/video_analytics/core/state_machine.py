"""
Event State Machine
Unified state management for detection events
"""
from enum import Enum, auto
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time


class EventState(Enum):
    """事件状态"""
    IDLE = "idle"              # 空闲状态
    COUNTING = "counting"      # 计数中 (连续帧确认)
    TRIGGERED = "triggered"    # 事件触发 (首次)
    ONGOING = "ongoing"        # 事件持续中
    ENDING = "ending"          # 结束中 (连续帧消失)
    COOLDOWN = "cooldown"      # 冷却中


@dataclass
class StateTransition:
    """状态转换记录"""
    from_state: EventState
    to_state: EventState
    timestamp: datetime
    reason: str = ""


class EventStateMachine:
    """
    事件状态机

    统一封装事件生命周期管理:
    - 防抖 (连续N帧确认)
    - 事件开始/持续/结束
    - 冷却时间控制

    状态流转:
    IDLE -> COUNTING -> TRIGGERED -> ONGOING -> ENDING -> COOLDOWN -> IDLE
                    \-> ONGOING (持续检测中)

    使用示例:
        sm = EventStateMachine(
            min_trigger_frames=25,    # 25帧确认
            min_end_frames=25,        # 25帧消失结束
            cooldown_seconds=60       # 60秒冷却
        )

        # 每帧更新
        state = sm.update(detected=True)  # 检测到目标
        if state == EventState.TRIGGERED:
            print("事件开始!")
        elif state == EventState.ONGOING:
            print("事件持续中...")
        elif state == EventState.COOLDOWN:
            print("事件结束，进入冷却")
    """

    def __init__(
        self,
        min_trigger_frames: int = 25,
        min_end_frames: int = 25,
        cooldown_seconds: float = 60.0,
        name: str = ""
    ):
        """
        初始化状态机

        Args:
            min_trigger_frames: 触发事件所需的最小连续帧数
            min_end_frames: 结束事件所需的最小连续消失帧数
            cooldown_seconds: 事件结束后的冷却时间(秒)
            name: 状态机名称(用于日志)
        """
        self.min_trigger_frames = min_trigger_frames
        self.min_end_frames = min_end_frames
        self.cooldown_seconds = cooldown_seconds
        self.name = name

        # 当前状态
        self._state = EventState.IDLE

        # 计数器
        self._consecutive_count = 0      # 连续检测计数
        self._absent_count = 0           # 连续消失计数

        # 时间戳
        self._event_start_time: Optional[datetime] = None
        self._event_end_time: Optional[datetime] = None
        self._cooldown_start_time: Optional[datetime] = None
        self._last_trigger_time: Optional[datetime] = None

        # 历史记录
        self._transitions: List[StateTransition] = []
        self._max_history = 100

        # 回调函数
        self._on_trigger: Optional[Callable] = None
        self._on_end: Optional[Callable] = None
        self._on_cooldown_end: Optional[Callable] = None

    @property
    def state(self) -> EventState:
        """获取当前状态"""
        return self._state

    @property
    def is_active(self) -> bool:
        """事件是否处于活动状态"""
        return self._state in [EventState.TRIGGERED, EventState.ONGOING]

    @property
    def event_duration(self) -> float:
        """获取当前事件持续时间(秒)"""
        if self._event_start_time is None:
            return 0.0

        if self._event_end_time:
            return (self._event_end_time - self._event_start_time).total_seconds()

        return (datetime.now() - self._event_start_time).total_seconds()

    @property
    def cooldown_remaining(self) -> float:
        """获取冷却剩余时间(秒)"""
        if self._state != EventState.COOLDOWN or self._cooldown_start_time is None:
            return 0.0

        elapsed = (datetime.now() - self._cooldown_start_time).total_seconds()
        return max(0.0, self.cooldown_seconds - elapsed)

    def update(self, detected: bool) -> EventState:
        """
        更新状态机

        Args:
            detected: 当前帧是否检测到目标

        Returns:
            EventState: 更新后的状态
        """
        now = datetime.now()

        # 检查冷却时间是否结束
        if self._state == EventState.COOLDOWN:
            if self.cooldown_remaining <= 0:
                self._transition_to(EventState.IDLE, "Cooldown ended")
            else:
                return self._state

        # 状态流转逻辑
        if detected:
            # 检测到目标
            self._consecutive_count += 1
            self._absent_count = 0

            if self._state == EventState.IDLE:
                # 开始计数
                self._transition_to(EventState.COUNTING, "Target detected, start counting")

            elif self._state == EventState.COUNTING:
                # 检查是否达到触发阈值
                if self._consecutive_count >= self.min_trigger_frames:
                    self._event_start_time = now
                    self._last_trigger_time = now
                    self._transition_to(EventState.TRIGGERED, f"Triggered after {self._consecutive_count} frames")
                    self._trigger_callback()

            elif self._state in [EventState.TRIGGERED, EventState.ONGOING]:
                # 事件持续
                if self._state == EventState.TRIGGERED:
                    self._transition_to(EventState.ONGOING, "Event ongoing")

            elif self._state == EventState.ENDING:
                # 目标重新出现，回到ONGOING
                self._transition_to(EventState.ONGOING, "Target reappeared")

        else:
            # 未检测到目标
            self._absent_count += 1
            self._consecutive_count = 0

            if self._state == EventState.COUNTING:
                # 重置计数
                self._transition_to(EventState.IDLE, "Target lost during counting")

            elif self._state in [EventState.TRIGGERED, EventState.ONGOING]:
                # 开始结束流程
                self._transition_to(EventState.ENDING, "Target disappeared, start ending countdown")

            elif self._state == EventState.ENDING:
                # 检查是否达到结束阈值
                if self._absent_count >= self.min_end_frames:
                    self._event_end_time = now
                    self._cooldown_start_time = now
                    self._transition_to(EventState.COOLDOWN, f"Event ended after {self._absent_count} absent frames")
                    self._end_callback()

        return self._state

    def _transition_to(self, new_state: EventState, reason: str = ""):
        """状态转换"""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state

        # 记录转换
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason
        )
        self._transitions.append(transition)

        # 限制历史记录大小
        if len(self._transitions) > self._max_history:
            self._transitions.pop(0)

    def force_end(self):
        """强制结束当前事件"""
        if self._state in [EventState.TRIGGERED, EventState.ONGOING, EventState.ENDING]:
            self._event_end_time = datetime.now()
            self._cooldown_start_time = datetime.now()
            self._transition_to(EventState.COOLDOWN, "Force ended")
            self._end_callback()

    def reset(self):
        """重置状态机到初始状态"""
        self._state = EventState.IDLE
        self._consecutive_count = 0
        self._absent_count = 0
        self._event_start_time = None
        self._event_end_time = None
        self._cooldown_start_time = None
        self._last_trigger_time = None
        self._transitions.clear()

    def set_callbacks(
        self,
        on_trigger: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_cooldown_end: Optional[Callable] = None
    ):
        """
        设置回调函数

        Args:
            on_trigger: 事件触发时回调
            on_end: 事件结束时回调
            on_cooldown_end: 冷却结束时回调
        """
        self._on_trigger = on_trigger
        self._on_end = on_end
        self._on_cooldown_end = on_cooldown_end

    def _trigger_callback(self):
        """触发事件回调"""
        if self._on_trigger:
            try:
                self._on_trigger(self)
            except Exception as e:
                print(f"[StateMachine] Trigger callback error: {e}")

    def _end_callback(self):
        """结束事件回调"""
        if self._on_end:
            try:
                self._on_end(self)
            except Exception as e:
                print(f"[StateMachine] End callback error: {e}")

    def get_stats(self) -> dict:
        """获取状态统计信息"""
        return {
            "state": self._state.value,
            "consecutive_count": self._consecutive_count,
            "absent_count": self._absent_count,
            "event_duration": self.event_duration,
            "cooldown_remaining": self.cooldown_remaining,
            "is_active": self.is_active,
            "total_transitions": len(self._transitions)
        }

    def get_history(self) -> List[StateTransition]:
        """获取状态转换历史"""
        return self._transitions.copy()

    def __repr__(self) -> str:
        return f"EventStateMachine(name={self.name}, state={self._state.value}, active={self.is_active})"
