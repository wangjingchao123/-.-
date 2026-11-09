"""不接摄像头的第一阶段演示。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fatigue_agent import FatigueAgent, FrameFeatures


def show(title, assessment):
    print(f"\n[{title}]")
    print(f"风险等级: {assessment.level.value}")
    print(f"可信度: {assessment.confidence:.3f}")
    print("原因:", "；".join(assessment.reasons))
    print("建议:", "；".join(assessment.recommendations))
    print("信号:", assessment.signals)


def normal_case():
    agent = FatigueAgent()
    result = agent.update(FrameFeatures(0, 0.95, 0.95, ear=0.31, mar=0.30, pitch=3))
    show("正常驾驶", result)


def closed_eye_case():
    agent = FatigueAgent()
    agent.update(FrameFeatures(0, 0.95, 0.95, ear=0.30, mar=0.30, pitch=3))
    result = None
    for t in (0.5, 1.5, 2.5, 3.0):
        result = agent.update(FrameFeatures(t, 0.95, 0.95, ear=0.12, mar=0.30, pitch=4))
    show("持续闭眼", result)


def sunglasses_case():
    agent = FatigueAgent()
    result = agent.update(FrameFeatures(0, 0.10, 0.30, ear=None, mar=None, pitch=4, sunglasses_suspected=True))
    show("墨镜遮挡", result)


if __name__ == "__main__":
    normal_case()
    closed_eye_case()
    sunglasses_case()
