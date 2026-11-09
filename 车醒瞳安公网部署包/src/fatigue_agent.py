"""纯软件疲劳驾驶风险判断核心。

该模块不依赖摄像头、OpenCV 或任何硬件。上游视觉模块只需要提供
FrameFeatures，后续即可替换为 MediaPipe、YOLO 或其他模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple
from collections import deque
import math


class RiskLevel(str, Enum):
    NORMAL = "normal"
    ATTENTION = "attention"
    WARNING = "warning"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass
class FrameFeatures:
    """单帧视觉模块输出。

    数值范围约定：visibility、ear、mar、confidence 均为 0~1；
    pitch 为头部俯仰角，正值表示头部向下的角度（原型约定）。
    """

    timestamp: float
    eye_visibility: float
    eye_confidence: float
    ear: Optional[float] = None
    mar: Optional[float] = None
    pitch: Optional[float] = None
    gaze_available: bool = False
    gaze_away: bool = False
    sunglasses_suspected: bool = False


@dataclass
class RiskAssessment:
    level: RiskLevel
    confidence: float
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points: List[Tuple[float, float]]) -> float:
    """根据六个眼部关键点计算 EAR。

    点顺序：左角、上外、上内、右角、下内、下外。
    """
    if len(points) != 6:
        raise ValueError("EAR 需要恰好 6 个眼部关键点")
    horizontal = _distance(points[0], points[3])
    if horizontal == 0:
        return 0.0
    vertical = (_distance(points[1], points[5]) + _distance(points[2], points[4])) / 2
    return vertical / horizontal


def mouth_aspect_ratio(points: List[Tuple[float, float]]) -> float:
    """根据四个口部关键点计算简化 MAR。"""
    if len(points) != 4:
        raise ValueError("MAR 需要 4 个口部关键点")
    horizontal = _distance(points[0], points[1])
    return 0.0 if horizontal == 0 else _distance(points[2], points[3]) / horizontal


class FatigueAgent:
    """对连续帧特征进行平滑、降级和风险评估。"""

    def __init__(
        self,
        ear_closed_threshold: float = 0.20,
        closed_seconds_warning: float = 1.5,
        closed_seconds_high: float = 2.5,
        head_down_seconds_warning: float = 2.0,
        head_down_seconds_high: float = 4.0,
        head_pose_deviation_threshold: float = 18.0,
        baseline_pitch: Optional[float] = None,
        face_unavailable_seconds_attention: float = 2.0,
        face_unavailable_seconds_warning: float = 4.0,
        perclos_window_seconds: float = 60.0,
        perclos_min_observed_seconds: float = 15.0,
        perclos_attention_threshold: float = 0.15,
        perclos_warning_threshold: float = 0.28,
        escalation_hold_seconds: float = 0.8,
        recovery_hold_seconds: float = 3.0,
        fatigue_event_window_short_seconds: float = 300.0,
        fatigue_event_window_long_seconds: float = 600.0,
        fatigue_events_plan_rest: int = 2,
        fatigue_events_urgent_rest: int = 3,
        mar_yawn_threshold: float = 0.45,
        yawn_min_duration_seconds: float = 1.0,
        yawn_max_duration_seconds: float = 8.0,
        yawn_count_warning: int = 3,
        yawn_window_seconds: float = 300.0,
        visibility_reliable_threshold: float = 0.65,
        visibility_blocked_threshold: float = 0.25,
    ) -> None:
        self.ear_closed_threshold = ear_closed_threshold
        self.closed_seconds_warning = closed_seconds_warning
        self.closed_seconds_high = closed_seconds_high
        self.head_down_seconds_warning = head_down_seconds_warning
        self.head_down_seconds_high = head_down_seconds_high
        self.head_pose_deviation_threshold = head_pose_deviation_threshold
        self.baseline_pitch = baseline_pitch
        self.face_unavailable_seconds_attention = face_unavailable_seconds_attention
        self.face_unavailable_seconds_warning = face_unavailable_seconds_warning
        self.perclos_window_seconds = perclos_window_seconds
        self.perclos_min_observed_seconds = perclos_min_observed_seconds
        self.perclos_attention_threshold = perclos_attention_threshold
        self.perclos_warning_threshold = perclos_warning_threshold
        self.escalation_hold_seconds = escalation_hold_seconds
        self.recovery_hold_seconds = recovery_hold_seconds
        self.fatigue_event_window_short_seconds = fatigue_event_window_short_seconds
        self.fatigue_event_window_long_seconds = fatigue_event_window_long_seconds
        self.fatigue_events_plan_rest = fatigue_events_plan_rest
        self.fatigue_events_urgent_rest = fatigue_events_urgent_rest
        self.mar_yawn_threshold = mar_yawn_threshold
        self.yawn_min_duration_seconds = yawn_min_duration_seconds
        self.yawn_max_duration_seconds = yawn_max_duration_seconds
        self.yawn_count_warning = yawn_count_warning
        self.yawn_window_seconds = yawn_window_seconds
        self.visibility_reliable_threshold = visibility_reliable_threshold
        self.visibility_blocked_threshold = visibility_blocked_threshold
        self._last: Optional[FrameFeatures] = None
        self._closed_since: Optional[float] = None
        self._head_down_since: Optional[float] = None
        self._face_unavailable_since: Optional[float] = None
        self._yawns: Deque[float] = deque()
        self._yawn_active = False
        self._yawn_started_at: Optional[float] = None
        self._eye_intervals: Deque[Tuple[float, float, bool]] = deque()
        self._stable_level: Optional[RiskLevel] = None
        self._candidate_level: Optional[RiskLevel] = None
        self._candidate_since: Optional[float] = None
        self._fatigue_events: Deque[float] = deque()
        self._monitoring_events: Deque[float] = deque()

    @staticmethod
    def _priority(level: RiskLevel) -> int:
        return {RiskLevel.NORMAL: 0, RiskLevel.ATTENTION: 1, RiskLevel.WARNING: 2, RiskLevel.HIGH: 3}.get(level, 0)

    def _stabilize_level(self, raw: RiskLevel, timestamp: float) -> RiskLevel:
        """上升短暂确认，恢复延迟确认，避免风险标签频繁跳变。"""
        if self._stable_level is None or self._stable_level == RiskLevel.UNKNOWN:
            self._stable_level = raw
            self._candidate_level = None
            return raw
        if raw == RiskLevel.UNKNOWN:
            # 当前眼部不可用时保留上一个已确认的风险，防止瞬时丢脸直接解除预警。
            return self._stable_level
        if raw == self._stable_level:
            self._candidate_level = None
            self._candidate_since = None
            return self._stable_level
        if raw != self._candidate_level:
            self._candidate_level = raw
            self._candidate_since = timestamp
            return self._stable_level
        elapsed = timestamp - (self._candidate_since if self._candidate_since is not None else timestamp)
        hold = self.escalation_hold_seconds if self._priority(raw) > self._priority(self._stable_level) else self.recovery_hold_seconds
        if elapsed >= hold:
            self._stable_level = raw
            self._candidate_level = None
            self._candidate_since = None
        return self._stable_level

    @staticmethod
    def _trim(events: Deque[float], timestamp: float, window: float) -> None:
        while events and timestamp - events[0] > window:
            events.popleft()

    def update(self, frame: FrameFeatures) -> RiskAssessment:
        previous = self._last
        self._last = frame
        dt = 0.0 if previous is None else max(0.0, frame.timestamp - previous.timestamp)

        eye_reliable = frame.eye_visibility >= self.visibility_reliable_threshold and frame.eye_confidence >= 0.65
        eye_blocked = frame.eye_visibility <= self.visibility_blocked_threshold or frame.sunglasses_suspected

        face_unavailable_seconds = 0.0
        if self.baseline_pitch is not None and frame.eye_visibility <= self.visibility_blocked_threshold:
            if self._face_unavailable_since is None:
                self._face_unavailable_since = frame.timestamp
            face_unavailable_seconds = frame.timestamp - self._face_unavailable_since
        else:
            self._face_unavailable_since = None

        # PERCLOS：在最近窗口内可观测眼部帧的闭眼时间占比。
        if previous is not None and eye_reliable and frame.ear is not None and dt > 0:
            self._eye_intervals.append((frame.timestamp, dt, frame.ear < self.ear_closed_threshold))
        cutoff = frame.timestamp - self.perclos_window_seconds
        while self._eye_intervals and self._eye_intervals[0][0] < cutoff:
            self._eye_intervals.popleft()
        observed_eye_seconds = sum(item[1] for item in self._eye_intervals)
        closed_eye_seconds = sum(item[1] for item in self._eye_intervals if item[2])
        perclos = closed_eye_seconds / observed_eye_seconds if observed_eye_seconds else 0.0

        closed_seconds = 0.0
        if eye_reliable and frame.ear is not None and frame.ear < self.ear_closed_threshold:
            if self._closed_since is None:
                self._closed_since = frame.timestamp
            closed_seconds = frame.timestamp - self._closed_since
        else:
            self._closed_since = None

        head_down_seconds = 0.0
        # 用相对个人基线的姿态偏离代替固定角度，降低摄像头安装位置带来的误判。
        pose_deviation = None if frame.pitch is None or self.baseline_pitch is None else abs(frame.pitch - self.baseline_pitch)
        if pose_deviation is not None and pose_deviation > self.head_pose_deviation_threshold:
            if self._head_down_since is None:
                self._head_down_since = frame.timestamp
            head_down_seconds = frame.timestamp - self._head_down_since
        else:
            self._head_down_since = None

        # 哈欠需满足“张口持续一段时间，随后恢复闭合”的完整时间模式。
        # 因此短促说话、瞬时表情不会直接被记为一次哈欠。
        current_yawn_open_seconds = 0.0
        if frame.mar is not None and frame.mar > self.mar_yawn_threshold:
            if not self._yawn_active:
                self._yawn_active = True
                self._yawn_started_at = frame.timestamp
            started = self._yawn_started_at if self._yawn_started_at is not None else frame.timestamp
            current_yawn_open_seconds = frame.timestamp - started
        elif self._yawn_active:
            started = self._yawn_started_at if self._yawn_started_at is not None else frame.timestamp
            duration = frame.timestamp - started
            if self.yawn_min_duration_seconds <= duration <= self.yawn_max_duration_seconds:
                self._yawns.append(frame.timestamp)
            self._yawn_active = False
            self._yawn_started_at = None
        while self._yawns and frame.timestamp - self._yawns[0] > self.yawn_window_seconds:
            self._yawns.popleft()

        reasons: List[str] = []
        recommendations: List[str] = []
        score = 0.0

        if closed_seconds >= self.closed_seconds_high:
            score += 0.65; reasons.append(f"连续闭眼约 {closed_seconds:.1f} 秒")
        elif closed_seconds >= self.closed_seconds_warning:
            score += 0.40; reasons.append(f"闭眼持续约 {closed_seconds:.1f} 秒")

        if head_down_seconds >= self.head_down_seconds_high:
            score += 0.45; reasons.append(f"头部姿态持续偏离约 {head_down_seconds:.1f} 秒")
        elif head_down_seconds >= self.head_down_seconds_warning:
            score += 0.25; reasons.append(f"头部姿态持续偏离约 {head_down_seconds:.1f} 秒")

        if face_unavailable_seconds >= self.face_unavailable_seconds_warning:
            score += 0.45; reasons.append(f"驾驶员面部持续不可见约 {face_unavailable_seconds:.1f} 秒")
            recommendations.append("请抬头并恢复面向摄像头，系统暂时无法完成眼部监测")
        elif face_unavailable_seconds >= self.face_unavailable_seconds_attention:
            score += 0.25; reasons.append(f"驾驶员面部不可见约 {face_unavailable_seconds:.1f} 秒")
            recommendations.append("请恢复面向摄像头，以保持驾驶员状态监测")

        if len(self._yawns) >= self.yawn_count_warning:
            score += 0.25; reasons.append(f"近 {self.yawn_window_seconds/60:.0f} 分钟内打哈欠 {len(self._yawns)} 次")

        if frame.gaze_available and frame.gaze_away:
            score += 0.25; reasons.append("视线持续偏离前方")

        if observed_eye_seconds >= self.perclos_min_observed_seconds:
            if perclos >= self.perclos_warning_threshold:
                score += 0.45; reasons.append(f"最近 {self.perclos_window_seconds:.0f} 秒闭眼占比为 {perclos:.0%}")
            elif perclos >= self.perclos_attention_threshold:
                score += 0.25; reasons.append(f"最近 {self.perclos_window_seconds:.0f} 秒闭眼占比为 {perclos:.0%}")

        if eye_blocked:
            reasons.append("眼部信息受遮挡，无法可靠确认瞳孔状态")
            recommendations.append("请保持眼部可见；必要时结合头部姿态和车辆行为继续评估")

        if score >= 0.75:
            raw_level = RiskLevel.HIGH
        elif score >= 0.45:
            raw_level = RiskLevel.WARNING
        elif score >= 0.20:
            raw_level = RiskLevel.ATTENTION
        elif eye_blocked and score == 0:
            raw_level = RiskLevel.UNKNOWN
        else:
            raw_level = RiskLevel.NORMAL

        before_transition = self._stable_level
        level = self._stabilize_level(raw_level, frame.timestamp)
        if level != raw_level:
            if self._priority(raw_level) < self._priority(level):
                reasons.append("风险恢复确认中，持续正常后将自动解除提示")
            elif self._priority(raw_level) > self._priority(level):
                reasons.append("风险信号确认中")

        # 事件只在“已确认状态首次进入预警”时计一次，而不是按帧或按提示音计数。
        fatigue_confirmed = (
            closed_seconds >= self.closed_seconds_warning
            or len(self._yawns) >= self.yawn_count_warning
            or (observed_eye_seconds >= self.perclos_min_observed_seconds and perclos >= self.perclos_warning_threshold)
        )
        monitoring_confirmed = face_unavailable_seconds >= self.face_unavailable_seconds_warning
        distraction_confirmed = (
            (head_down_seconds >= self.head_down_seconds_warning)
            or (frame.gaze_available and frame.gaze_away)
        )
        just_entered_alert = level in (RiskLevel.WARNING, RiskLevel.HIGH) and level != before_transition
        if just_entered_alert:
            if fatigue_confirmed:
                self._fatigue_events.append(frame.timestamp)
            elif monitoring_confirmed:
                self._monitoring_events.append(frame.timestamp)

        self._trim(self._fatigue_events, frame.timestamp, self.fatigue_event_window_long_seconds)
        self._trim(self._monitoring_events, frame.timestamp, self.fatigue_event_window_short_seconds)
        fatigue_events_short = sum(frame.timestamp - event <= self.fatigue_event_window_short_seconds for event in self._fatigue_events)
        fatigue_events_long = len(self._fatigue_events)
        monitoring_events_short = len(self._monitoring_events)

        rest_recommendation = "none"
        if fatigue_events_long >= self.fatigue_events_urgent_rest:
            rest_recommendation = "urgent_rest"
            recommendations.append("疲劳风险已多次累积，请尽快选择安全地点停车休息")
            reasons.append(f"近 {self.fatigue_event_window_long_seconds / 60:.0f} 分钟已确认 {fatigue_events_long} 次疲劳风险事件")
        elif fatigue_events_short >= self.fatigue_events_plan_rest:
            rest_recommendation = "plan_rest"
            recommendations.append("疲劳风险正在累积，建议在下一个安全地点休息")
            reasons.append(f"近 {self.fatigue_event_window_short_seconds / 60:.0f} 分钟已确认 {fatigue_events_short} 次疲劳风险事件")

        if level in (RiskLevel.WARNING, RiskLevel.HIGH):
            recommendations.append("建议尽快驶入安全区域休息，不要继续疲劳驾驶")
        elif level == RiskLevel.ATTENTION:
            recommendations.append("请保持注意力，必要时安排短暂休息")
        elif level == RiskLevel.UNKNOWN:
            recommendations.append("当前证据不足，不输出确定的疲劳结论")

        # 眼部不可见时降低确定性，防止墨镜场景盲目报警。
        confidence = min(1.0, 0.45 + 0.45 * frame.eye_visibility + 0.10 * frame.eye_confidence)
        if eye_blocked:
            confidence = min(confidence, 0.55)
        if previous is not None and dt > 5:
            confidence *= 0.85

        return RiskAssessment(
            level=level,
            confidence=round(confidence, 3),
            reasons=reasons or ["暂未发现明显疲劳信号"],
            recommendations=recommendations or ["继续保持正常驾驶状态"],
            signals={
                "eye_reliable": eye_reliable,
                "eye_blocked": eye_blocked,
                "closed_seconds": round(closed_seconds, 2),
                "head_down_seconds": round(head_down_seconds, 2),
                "head_pose_deviation": None if pose_deviation is None else round(pose_deviation, 2),
                "face_unavailable_seconds": round(face_unavailable_seconds, 2),
                "mouth_open_seconds": round(current_yawn_open_seconds, 2),
                "yawn_count": len(self._yawns),
                "perclos": round(perclos, 3),
                "eye_observed_seconds": round(observed_eye_seconds, 2),
                "raw_level": raw_level.value,
                "fatigue_events_5min": fatigue_events_short,
                "fatigue_events_10min": fatigue_events_long,
                "monitoring_events_5min": monitoring_events_short,
                "rest_recommendation": rest_recommendation,
                "event_type": "fatigue" if fatigue_confirmed else ("monitoring_gap" if monitoring_confirmed else ("distraction_or_posture" if distraction_confirmed else "none")),
            },
        )

