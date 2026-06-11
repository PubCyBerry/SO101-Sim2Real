"""Isaac sim ↔ 실기기 LeRobot SO-101 단위/카메라 변환 공용 헬퍼.

`rollout_to_lerobot.py` (RL expert rollout → LeRobot v3 recorder) 와
`teleop_se3_agent.py` 의 policy-server 추론 경로가 공유한다. 이 모듈은
AppLauncher 를 부팅하지 않으므로 Isaac Sim 기동 전/후 어디서든 import 가능하다
(`rollout_to_lerobot.py` 는 import 시점에 AppLauncher 를 띄워 직접 재사용 불가).

단위 규약 (실기기 LeRobot SO-101 데이터셋 = North Star 계약):
- 팔 5축: radian(sim) ↔ degree(LeRobot)
- 그리퍼: radian(sim) ↔ [0,100] 정규화(LeRobot), scale = 31.75
"""

from __future__ import annotations

import math

import numpy as np

# ── 상수 ────────────────────────────────────────────────────────────────────
FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
CAMERA_KEYS = ("top", "wrist", "front")
CAMERA_SCENE_NAMES = {
    "top": "top_camera",
    "wrist": "wrist_camera",
    "front": "front_camera",
}
JOINT_FEATURE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
# rad → [0,100] 그리퍼 스케일. follower USD joint([-10°,100°]) 기준이 아니라
# 실기기 LeRobot 데이터셋의 [0,100] 정규화 관례에 맞춘 값.
GRIPPER_LEROBOT_SCALE = 31.75

_RAD_TO_DEG = 180.0 / math.pi
_DEG_TO_RAD = math.pi / 180.0


def to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """Isaac joint radian → 실기기 LeRobot SO-101 단위(arm deg, gripper [0,100])."""
    values = np.asarray(values_rad, dtype=np.float32).copy()
    values[:5] = values[:5] * _RAD_TO_DEG
    values[5] = values[5] * GRIPPER_LEROBOT_SCALE
    return values.astype(np.float32)


def from_lerobot_units(values_lerobot: np.ndarray) -> np.ndarray:
    """실기기 LeRobot SO-101 단위 → Isaac joint radian (to_lerobot_units 의 역변환).

    policy-server 가 돌려준 action(LeRobot 단위)을 sim joint target(rad)으로 되돌릴 때 쓴다.
    """
    values = np.asarray(values_lerobot, dtype=np.float32).copy()
    values[:5] = values[:5] * _DEG_TO_RAD
    values[5] = values[5] / GRIPPER_LEROBOT_SCALE
    return values.astype(np.float32)


def read_camera_rgb_u8(raw_env, scene_name: str) -> np.ndarray:
    """env.scene[scene_name].data.output["rgb"] 를 (H,W,3) uint8 로 읽는다.

    RGBA → RGB 드롭, float[0,1] → uint8[0,255] 변환. shape 검증 포함.
    """
    cam = raw_env.unwrapped.scene[scene_name]
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(rgb)
    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    if image.shape != expected_shape:
        raise ValueError(f"camera {scene_name} image shape {image.shape}, expected {expected_shape}")
    return image
