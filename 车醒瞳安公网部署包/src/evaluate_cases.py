"""评估事件级测试表，生成比赛可用的结构化指标和简单图表。

输入表由 data/evaluation_template.csv 定义。该脚本不读取视频，只评估人工标签与系统输出，
因此可以在不暴露原始人脸视频的情况下复核结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


EVENTS = ('none', 'fatigue', 'distraction_or_posture', 'monitoring_gap')
VISIBILITY = ('visible', 'blocked', 'unavailable')


def safe_float(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except (TypeError, ValueError):
        return None


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get('ground_truth_event') and r.get('predicted_event')]
    event_confusion = {truth: {pred: 0 for pred in EVENTS} for truth in EVENTS}
    for row in labeled:
        truth, pred = row['ground_truth_event'], row['predicted_event']
        if truth not in event_confusion:
            event_confusion[truth] = {predicted: 0 for predicted in EVENTS}
        if pred not in event_confusion[truth]:
            for values in event_confusion.values():
                values[pred] = 0
        event_confusion[truth][pred] += 1

    total = len(labeled)
    event_correct = sum(event_confusion.get(label, {}).get(label, 0) for label in EVENTS)
    normal_total = sum(1 for r in labeled if r['ground_truth_event'] == 'none')
    normal_false_alarm = sum(1 for r in labeled if r['ground_truth_event'] == 'none' and r['predicted_event'] != 'none')
    fatigue_total = sum(1 for r in labeled if r['ground_truth_event'] == 'fatigue')
    fatigue_hit = sum(1 for r in labeled if r['ground_truth_event'] == 'fatigue' and r['predicted_event'] == 'fatigue')

    visibility_rows = [r for r in rows if r.get('ground_truth_visibility') and r.get('predicted_visibility')]
    vis_correct = sum(r['ground_truth_visibility'] == r['predicted_visibility'] for r in visibility_rows)
    latencies = [safe_float(r.get('response_ms', '')) for r in rows]
    latencies = sorted(x for x in latencies if x is not None)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * .95))] if latencies else None

    scenario_stats: dict[str, dict] = defaultdict(lambda: {'total': 0, 'correct': 0, 'false_alarm': 0})
    for row in labeled:
        scenario = row.get('scenario') or '未分类'
        item = scenario_stats[scenario]
        item['total'] += 1
        item['correct'] += int(row['ground_truth_event'] == row['predicted_event'])
        item['false_alarm'] += int(row['ground_truth_event'] == 'none' and row['predicted_event'] != 'none')
    for item in scenario_stats.values():
        item['accuracy'] = rate(item['correct'], item['total'])
        item['false_alarm_rate'] = rate(item['false_alarm'], item['total'])

    return {
        'notice': '指标仅对输入标注表计算，不代表车规级性能或真实道路安全结论。',
        'samples': {'rows': len(rows), 'labeled_event_rows': total, 'visibility_rows': len(visibility_rows)},
        'event_metrics': {
            'event_accuracy': rate(event_correct, total),
            'fatigue_event_recall': rate(fatigue_hit, fatigue_total),
            'normal_false_alarm_rate': rate(normal_false_alarm, normal_total),
            'confusion_matrix': event_confusion,
        },
        'monitoring_metrics': {'visibility_accuracy': rate(vis_correct, len(visibility_rows))},
        'latency_ms': {
            'mean': round(sum(latencies) / len(latencies), 2) if latencies else None,
            'p95': p95,
            'samples': len(latencies),
        },
        'scenario_metrics': dict(scenario_stats),
    }


def write_chart(metrics: dict, output: Path) -> None:
    try:
        os.environ.setdefault('MPLCONFIGDIR', str(output.parent / '.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
    except ImportError:
        return
    scenarios = list(metrics['scenario_metrics'])
    if not scenarios:
        return
    values = [metrics['scenario_metrics'][x]['accuracy'] or 0 for x in scenarios]
    font_path = Path(r'C:\Windows\Fonts\msyh.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar(scenarios, values, color='#2f75b5')
    ax.set_ylim(0, 1.05); ax.set_ylabel('事件级准确率', fontproperties=font); ax.set_title('车醒瞳安测试场景对比（示例）', fontproperties=font)
    ax.tick_params(axis='x', rotation=25); ax.grid(axis='y', alpha=.25)
    if font:
        for label in ax.get_xticklabels(): label.set_fontproperties(font)
        for label in ax.get_yticklabels(): label.set_fontproperties(font)
    for i, value in enumerate(values): ax.text(i, value + .03, f'{value:.0%}', ha='center', fontsize=9)
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='计算车醒瞳安事件级测试指标')
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/evaluation'))
    args = parser.parse_args()
    with args.csv_file.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    metrics = evaluate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    write_chart(metrics, args.out_dir / 'scenario_accuracy.png')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'已写入：{args.out_dir.resolve()}')


if __name__ == '__main__':
    main()
