"""离线驾驶视频分析入口。

示例：
python car_ai_agent\\src\\analyze_video.py path\\to\\driving.mp4

输出是 JSON 报告；不会打开摄像头，不会控制车辆，也不会将视频上传到网络。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fatigue_agent import FatigueAgent
from video_adapter import iter_video_features

RISK_PRIORITY = {'normal': 0, 'unknown': 0, 'attention': 1, 'warning': 2, 'high': 3}


def main() -> None:
    parser = argparse.ArgumentParser(description='离线分析驾驶视频的疲劳风险')
    parser.add_argument('video', help='本地视频文件路径')
    parser.add_argument('--max-frames', type=int, default=None, help='最多分析的帧数，便于快速测试')
    parser.add_argument('--out', default=None, help='可选：JSON报告输出路径')
    args = parser.parse_args()

    agent = FatigueAgent()
    levels = Counter()
    final = None; peak = None; peak_timestamp = None
    count = 0
    for features in iter_video_features(args.video, args.max_frames):
        final = agent.update(features)
        levels[final.level.value] += 1
        if peak is None or RISK_PRIORITY[final.level.value] > RISK_PRIORITY[peak.level.value]:
            peak = final; peak_timestamp = features.timestamp
        count += 1

    if final is None:
        raise RuntimeError('视频没有可读取的帧')

    report = {
        'input_video': str(Path(args.video).resolve()),
        'frames_analyzed': count,
        'risk_level_distribution': dict(levels),
        'latest_assessment': {
            'level': final.level.value,
            'confidence': final.confidence,
            'reasons': final.reasons,
            'recommendations': final.recommendations,
            'signals': final.signals,
        },
        'peak_assessment': {
            'timestamp_seconds': round(peak_timestamp, 2),
            'level': peak.level.value,
            'confidence': peak.confidence,
            'reasons': peak.reasons,
            'recommendations': peak.recommendations,
            'signals': peak.signals,
        },
        'notice': '结果仅用于比赛原型与研究验证，不作为驾驶安全或医疗结论。',
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        Path(args.out).write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
