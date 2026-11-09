"""用 CSV 模拟视频特征流，验证第二阶段数据入口。"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_stream import analyze_csv


def main() -> None:
    sample = Path(__file__).resolve().parents[1] / 'data' / 'sample_features.csv'
    results = analyze_csv(sample)
    print(json.dumps({
        'frames': len(results),
        'last_assessment': {
            'level': results[-1].level.value,
            'confidence': results[-1].confidence,
            'reasons': results[-1].reasons,
            'recommendations': results[-1].recommendations,
            'signals': results[-1].signals,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
