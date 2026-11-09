"""本地驾驶训练复盘 Agent。

该模块不调用外部模型或上传视频，只基于实时检测产生的结构化摘要回答复盘问题。
它的目的，是把风险识别结果转换成教练可读的解释和训练建议。
"""

from __future__ import annotations

from typing import Any


LEVEL_TEXT = {
    "normal": "正常",
    "attention": "关注",
    "warning": "警告",
    "high": "高风险",
    "unknown": "证据不足",
}

EVENT_TEXT = {
    "fatigue": "疲劳事件",
    "monitoring_gap": "监测中断",
    "distraction_or_posture": "分心或姿态异常",
    "none": "无明确事件",
}


def _level_text(level: str | None) -> str:
    return LEVEL_TEXT.get(str(level), str(level or "未知"))


def _event_text(event_type: str | None) -> str:
    return EVENT_TEXT.get(str(event_type), str(event_type or "未知"))


def review_answer(summary: dict[str, Any], question: str) -> dict[str, Any]:
    """根据一次训练摘要返回可解释的本地复盘回答。"""
    question = (question or "").strip()
    if not question:
        question = "请总结本次训练"

    latest = summary.get("latest") or {}
    signals = latest.get("signals") or {}
    events = summary.get("events") or []
    counts = summary.get("event_counts") or {}
    peak_level = summary.get("peak_level", "normal")
    current_level = latest.get("level", summary.get("last_level", "unknown"))
    duration = float(summary.get("duration_seconds") or 0)
    frame_count = int(summary.get("frames") or 0)
    fatigue = int(counts.get("fatigue", 0))
    monitoring = int(counts.get("monitoring_gap", 0))
    distraction = int(counts.get("distraction_or_posture", 0))
    rest = signals.get("rest_recommendation", "none")

    reasons = []
    for event in events[-5:]:
        for reason in event.get("reasons", []):
            if reason not in reasons and reason != "暂未发现明显疲劳信号":
                reasons.append(reason)
    latest_reasons = [x for x in latest.get("reasons", []) if x != "暂未发现明显疲劳信号"]
    for reason in latest_reasons:
        if reason not in reasons:
            reasons.append(reason)

    q = question.lower()
    if any(key in question for key in ("为什么", "原因", "依据", "判定")):
        if reasons:
            answer = "本次判断依据主要是：" + "；".join(reasons[:4]) + "。"
        elif signals.get("eye_blocked") or signals.get("face_unavailable_seconds", 0) > 0:
            answer = "当前没有足够的眼部证据。系统检测到眼部或人脸可见性不足，因此优先输出证据不足或监测中断，不直接确认疲劳。"
        else:
            answer = "当前没有形成明确的疲劳或姿态事件，系统暂维持正常状态。"
        answer += " 这是基于连续时间信号的辅助判断，不是车规级安全结论。"
    elif any(key in question for key in ("休息", "停车", "继续")):
        if rest == "urgent_rest" or peak_level == "high":
            answer = "建议结束本轮训练，并在安全地点停车休息。不要把网页提示当作继续驾驶的许可。"
        elif rest == "plan_rest" or fatigue >= 2:
            answer = "疲劳事件正在累积，建议在下一个安全地点安排休息，再决定是否继续训练。"
        elif monitoring or signals.get("eye_blocked"):
            answer = "当前更突出的是监测中断或证据不足。应先恢复面部和眼部可见性，再决定是否继续；系统不能据此确认安全。"
        else:
            answer = "本次摘要没有触发累计休息建议，但仍应由教练和驾驶员结合实际状态决定是否休息。"
    elif any(key in question for key in ("说话", "哈欠", "张嘴")):
        yawn_count = signals.get("yawn_count", 0)
        mouth_open = signals.get("mouth_open_seconds", 0)
        answer = (
            f"系统当前记录哈欠 {yawn_count} 次，最近一次张嘴持续约 {mouth_open:.1f} 秒。"
            "单帧张嘴不会直接确认哈欠，必须满足持续张开并恢复闭合的时间模式，"
            "因此正常说话通常只会被记为候选状态。"
        )
    elif any(key in question for key in ("低头", "姿态", "分心")):
        pose = signals.get("head_pose_deviation")
        head_down = signals.get("head_down_seconds", 0)
        if pose is None:
            answer = "本次缺少可靠的头部姿态基线或面部不可见，系统不能确认持续低头，只能提示证据不足或监测中断。"
        else:
            answer = f"当前头部姿态相对个人基线偏离约 {pose} 度，持续约 {head_down:.1f} 秒。"
            if distraction:
                answer += " 已形成姿态异常事件，建议教练复核是否存在看手机、低头操作或坐姿变化。"
            else:
                answer += " 尚未达到事件升级条件，系统会继续观察持续时间和恢复情况。"
    else:
        answer = (
            f"本次训练约 {duration:.1f} 秒，分析 {frame_count} 帧；当前状态为 {_level_text(current_level)}，"
            f"最高状态为 {_level_text(peak_level)}。记录到疲劳事件 {fatigue} 次、监测中断 {monitoring} 次、"
            f"姿态异常 {distraction} 次。"
        )
        if reasons:
            answer += " 主要依据包括：" + "；".join(reasons[:3]) + "。"
        answer += " 建议结合标注视频和教练观察进行复盘。"

    return {
        "answer": answer,
        "evidence": {
            "current_level": _level_text(current_level),
            "peak_level": _level_text(peak_level),
            "fatigue_events": fatigue,
            "monitoring_events": monitoring,
            "distraction_events": distraction,
            "confidence": latest.get("confidence"),
        },
        "boundary": "本回答仅基于本次本地检测摘要，用于训练复盘，不作为真实道路驾驶或医疗判断。",
    }
