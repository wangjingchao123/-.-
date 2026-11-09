"""把结构化帧特征送入疲劳智能体。

第二阶段先定义统一的数据入口：无论特征来自 MediaPipe、其他视觉模型，
还是人工标注 CSV，最终都转换为 FrameFeatures。这样更换视觉算法不会影响
风险判断和报告模块。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    from .fatigue_agent import FatigueAgent, FrameFeatures, RiskAssessment
except ImportError:
    from fatigue_agent import FatigueAgent, FrameFeatures, RiskAssessment


def _float_or_none(value: str) -> Optional[float]:
    value = (value or '').strip()
    return None if value in {'', 'none', 'null', 'nan'} else float(value)


def read_feature_csv(path: str | Path) -> Iterator[FrameFeatures]:
    """读取标注/视觉模型导出的特征 CSV。

    必填列：timestamp, eye_visibility, eye_confidence
    可选列：ear, mar, pitch, gaze_available, gaze_away, sunglasses_suspected
    """
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield FrameFeatures(
                timestamp=float(row['timestamp']),
                eye_visibility=float(row['eye_visibility']),
                eye_confidence=float(row['eye_confidence']),
                ear=_float_or_none(row.get('ear', '')),
                mar=_float_or_none(row.get('mar', '')),
                pitch=_float_or_none(row.get('pitch', '')),
                gaze_available=row.get('gaze_available', '').strip().lower() in {'1', 'true', 'yes'},
                gaze_away=row.get('gaze_away', '').strip().lower() in {'1', 'true', 'yes'},
                sunglasses_suspected=row.get('sunglasses_suspected', '').strip().lower() in {'1', 'true', 'yes'},
            )


def analyze_feature_stream(features: Iterable[FrameFeatures], agent: Optional[FatigueAgent] = None) -> list[RiskAssessment]:
    """按时间顺序分析一段特征流，返回每一帧的风险结果。"""
    worker = agent or FatigueAgent()
    return [worker.update(frame) for frame in features]


def analyze_csv(path: str | Path, agent: Optional[FatigueAgent] = None) -> list[RiskAssessment]:
    return analyze_feature_stream(read_feature_csv(path), agent)
