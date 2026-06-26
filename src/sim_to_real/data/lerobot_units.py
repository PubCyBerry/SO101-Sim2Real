"""Isaac sim ↔ 실기기 LeRobot SO-101 단위/카메라 변환 adapter.

`rollout_to_lerobot.py` (RL expert rollout → LeRobot v3 recorder) 와
`teleop_se3_agent.py` 의 policy-server 추론 경로가 공유한다. 이 모듈은
AppLauncher 를 부팅하지 않으므로 Isaac Sim 기동 전/후 어디서든 import 가능하다
(`rollout_to_lerobot.py` 는 import 시점에 AppLauncher 를 띄워 직접 재사용 불가).

실제 단위 규약과 변환은 `so101_contract.feature_codec` 한 곳에 정의한다.
"""

from __future__ import annotations

import numpy as np

from so101_contract.feature_codec import (
    CAMERA_KEYS,
    CODEC_VERSION,
    FPS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_FEATURE_NAMES,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)

CAMERA_SCENE_NAMES = {
    "top": "top_camera",
    "wrist": "wrist_camera",
    "front": "front_camera",
}


def to_lerobot_units(values_rad: np.ndarray) -> np.ndarray:
    """Isaac joint radian → canonical PolicyFeature v1."""
    return sim_joint_radians_to_policy_feature(values_rad)


def from_lerobot_units(values_lerobot: np.ndarray) -> np.ndarray:
    """Canonical PolicyFeature v1 → Isaac joint radian."""
    return policy_feature_to_sim_joint_radians(values_lerobot)


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
