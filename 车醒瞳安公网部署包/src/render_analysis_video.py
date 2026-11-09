"""生成带疲劳风险标注的本地视频与 JSON 报告。

示例：
python car_ai_agent\\src\\render_analysis_video.py input.mp4 \
  --out-video car_ai_agent\\reports\\annotated.mp4 \
  --out-report car_ai_agent\\reports\\analysis.json

只读取本地视频，不调用摄像头、不上传数据、不控制车辆。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fatigue_agent import FatigueAgent
from video_adapter import _require_dependencies, frame_to_features


RISK_LABELS = {
    'normal': 'NORMAL',
    'attention': 'ATTENTION',
    'warning': 'WARNING',
    'high': 'HIGH RISK',
    'unknown': 'INSUFFICIENT EVIDENCE',
}
RISK_COLORS = {
    'normal': (42, 142, 70),
    'attention': (0, 170, 230),
    'warning': (0, 145, 255),
    'high': (30, 30, 220),
    'unknown': (120, 120, 120),
}
RISK_PRIORITY = {'normal': 0, 'unknown': 0, 'attention': 1, 'warning': 2, 'high': 3}


def draw_panel(cv2, frame, assessment, features) -> None:
    """绘制英文状态面板，避免各环境缺少中文字体导致乱码。"""
    h, w = frame.shape[:2]
    color = RISK_COLORS[assessment.level.value]
    panel_h = min(170, h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    lines = [
        ('Driver Fatigue Risk Monitor', (230, 230, 230), 0.60),
        (f"Risk: {RISK_LABELS[assessment.level.value]}", color, 0.72),
        (f"Confidence: {assessment.confidence:.2f}  Eye visibility: {features.eye_visibility:.2f}", (235, 235, 235), 0.52),
        (f"EAR: {features.ear if features.ear is not None else 'N/A'}  MAR: {features.mar if features.mar is not None else 'N/A'}", (235, 235, 235), 0.52),
        (f"Closed: {assessment.signals['closed_seconds']:.1f}s  Yawns: {assessment.signals['yawn_count']}", (235, 235, 235), 0.52),
    ]
    y = 28
    for content, line_color, scale in lines:
        cv2.putText(frame, content, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, line_color, 2, cv2.LINE_AA)
        y += 28
    if assessment.level.value == 'unknown':
        cv2.putText(frame, 'Eye signal unavailable: no definite conclusion', (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description='生成带疲劳风险标注的驾驶视频')
    parser.add_argument('video', help='本地驾驶员正面视频')
    parser.add_argument('--out-video', required=True, help='输出MP4路径')
    parser.add_argument('--out-report', required=True, help='输出JSON报告路径')
    parser.add_argument('--max-frames', type=int, default=None, help='最多处理的帧数，用于快速测试')
    args = parser.parse_args()

    cv2, mp = _require_dependencies()
    source = cv2.VideoCapture(str(args.video))
    if not source.isOpened():
        raise FileNotFoundError(f'无法打开视频：{args.video}')
    fps = source.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    out_path = Path(args.out_video); out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.out_report); report_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('无法创建输出视频，请确认输出路径和编码器可用')

    agent = FatigueAgent()
    levels = Counter(); final = None; peak = None; peak_timestamp = None; count = 0
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as mesh:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            features = frame_to_features(result, count / fps, width, height)
            final = agent.update(features); levels[final.level.value] += 1
            if peak is None or RISK_PRIORITY[final.level.value] > RISK_PRIORITY[peak.level.value]:
                peak = final; peak_timestamp = features.timestamp
            draw_panel(cv2, frame, final, features)
            writer.write(frame)
            count += 1
            if args.max_frames is not None and count >= args.max_frames:
                break
    source.release(); writer.release()
    if final is None:
        raise RuntimeError('视频没有可读取的帧')

    report = {
        'input_video': str(Path(args.video).resolve()),
        'annotated_video': str(out_path.resolve()),
        'frames_analyzed': count,
        'risk_level_distribution': dict(levels),
        'latest_assessment': {
            'level': final.level.value, 'confidence': final.confidence,
            'reasons': final.reasons, 'recommendations': final.recommendations,
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
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
