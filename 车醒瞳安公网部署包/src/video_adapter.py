"""MediaPipe/OpenCV 视频适配层。

这是可选层：核心风险判断不依赖这些库。安装依赖后可调用
``iter_video_features``，将视频转换成 FrameFeatures 流。
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterator, Optional, Tuple

try:
    from .fatigue_agent import FrameFeatures, eye_aspect_ratio, mouth_aspect_ratio
except ImportError:
    from fatigue_agent import FrameFeatures, eye_aspect_ratio, mouth_aspect_ratio


def _require_dependencies():
    # 避免第三方库将字体缓存写入无权限的用户目录。
    cache_dir = Path(__file__).resolve().parents[1] / '.runtime' / 'matplotlib'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache_dir))
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            '视频适配器需要 opencv-python 和 mediapipe；当前核心可先使用 CSV 特征流运行。'
        ) from exc
    return cv2, mp


def _xy(landmarks, indices, width: int, height: int):
    return [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]


def frame_to_features(result, timestamp: float, width: int, height: int) -> FrameFeatures:
    """把单帧 Face Mesh 结果转换为 FrameFeatures。

    这里先使用稳定、可解释的眼睛/嘴部几何特征；瞳孔跟踪和头姿态将在后续版本加入。
    """
    if not result.multi_face_landmarks:
        return FrameFeatures(timestamp, 0.0, 0.0, sunglasses_suspected=False)

    face = result.multi_face_landmarks[0].landmark
    # MediaPipe Face Mesh 常用眼部和嘴部关键点。
    right_eye = _xy(face, [33, 160, 158, 133, 153, 144], width, height)
    left_eye = _xy(face, [362, 385, 387, 263, 373, 380], width, height)
    mouth = _xy(face, [61, 291, 13, 14], width, height)
    ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2
    mar = mouth_aspect_ratio(mouth)
    pitch = estimate_head_pitch(face, width, height)
    return FrameFeatures(
        timestamp=timestamp,
        eye_visibility=0.95,
        eye_confidence=0.90,
        ear=ear,
        mar=mar,
        pitch=pitch,
        gaze_available=False,
    )


def estimate_head_pitch(landmarks, width: int, height: int) -> Optional[float]:
    """用 solvePnP 估计头部俯仰角。

    角度只用于与个人校准基线比较；不是医学或车规级姿态测量值。
    """
    try:
        cv2, _ = _require_dependencies()
        import numpy as np
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),   # 鼻尖
            (landmarks[152].x * width, landmarks[152].y * height), # 下巴
            (landmarks[33].x * width, landmarks[33].y * height),   # 左眼角
            (landmarks[263].x * width, landmarks[263].y * height), # 右眼角
            (landmarks[61].x * width, landmarks[61].y * height),   # 左嘴角
            (landmarks[291].x * width, landmarks[291].y * height), # 右嘴角
        ], dtype='double')
        # 通用人脸参考点，仅用于相对角度变化。
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
        ], dtype='double')
        focal = float(width)
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype='double')
        ok, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rotation, _ = cv2.Rodrigues(rotation_vector)
        # RQ 分解返回的第一个欧拉角对应 X 轴旋转，即俯仰方向。
        angles = cv2.RQDecomp3x3(rotation)[0]
        # solvePnP 的欧拉角可能出现接近 +/-180 度的等价表示；归一化为 [-90, 90]。
        pitch = (float(angles[0]) + 90.0) % 180.0 - 90.0
        return pitch
    except Exception:
        return None


def iter_video_features(video_path: str | Path, max_frames: Optional[int] = None) -> Iterator[FrameFeatures]:
    """读取视频并逐帧输出 FrameFeatures。"""
    cv2, mp = _require_dependencies()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f'无法打开视频：{video_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    count = 0
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            yield frame_to_features(result, count / fps, width, height)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    capture.release()

