"""LeRobot v3 feature 데이터클래스 및 빌드 유틸 (leisaac 에서 vendor).

이 모듈은 HDF5→LeRobot-v3 변환 시 env 구조에서 feature 정의를 자동으로 생성한다.
AppLauncher 부팅이 필요한 환경에서만 사용(host-only fallback `uv run --group isaac`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv
from isaaclab.sensors import Camera

from so101_contract.leader_calibration import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMITS,
)


@dataclass
class StateFeatureItem:
    """State 특성(관절 각도) 메타데이터."""

    dtype: str = "float32"
    shape: tuple = (6,)
    names: list[str] = field(
        default_factory=lambda: [
            "joint1.pos",
            "joint2.pos",
            "joint3.pos",
            "joint4.pos",
            "joint5.pos",
            "joint6.pos",
        ]
    )


@dataclass
class VideoFeatureItem:
    """비디오 특성(이미지) 메타데이터."""

    dtype: str = "video"
    shape: list = field(default_factory=lambda: [480, 640, 3])  # [h, w, c]
    names: list[str] = field(default_factory=lambda: ["height", "width", "channels"])
    video_info: dict = field(
        default_factory=lambda: {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        }
    )


@dataclass
class LeRobotDatasetCfg:
    """LeRobotDataset 생성 설정 (leisaac 에서 vendor)."""

    repo_id: str = None
    """LeRobot 데이터셋 저장소 ID."""

    fps: int = 30
    """LeRobot 데이터셋 프레임/초."""

    robot_type: str = "so_follower"
    """로봇 타입 (so_follower, bi_so101_follower 등)."""

    features: dict = None
    """LeRobotDataset 특성 사전."""

    action_align: bool = False
    """액션 shape 이 관절 수와 같은지 여부. True 면 lerobot 제한 범위로 변환."""


def build_feature_from_env(env: ManagerBasedEnv | DirectRLEnv, dataset_cfg: LeRobotDatasetCfg) -> dict:
    """환경 구조에서 LeRobot feature 정의를 자동으로 생성한다.

    Args:
        env: Isaac Lab 환경(ManagerBasedRLEnv 또는 DirectRLEnv).
        dataset_cfg: LeRobotDatasetCfg 객체(fps·robot_type 포함).

    Returns:
        feature 사전: action, observation.state, observation.images.<camera> 메타 정의.
    """
    features = {}

    default_feature_joint_names = env.cfg.default_feature_joint_names
    if isinstance(env, ManagerBasedEnv):
        action_dim = env.action_manager.total_action_dim
    else:
        action_dim = env.actions.shape[-1]

    # 액션 차원이 관절 수와 일치하면 정렬, 아니면 generic dim_<index> 사용
    if action_dim != len(default_feature_joint_names):
        action_joint_names = [f"dim_{index}" for index in range(action_dim)]
        dataset_cfg.action_align = False
    else:
        action_joint_names = default_feature_joint_names
        dataset_cfg.action_align = True

    features["action"] = asdict(StateFeatureItem(dtype="float32", shape=(action_dim,), names=action_joint_names))
    features["observation.state"] = asdict(
        StateFeatureItem(
            dtype="float32", shape=(len(default_feature_joint_names),), names=default_feature_joint_names
        )
    )

    # 장면의 카메라 센서에서 이미지 특성 생성
    for camera_key, camera_sensor in env.scene.sensors.items():
        if isinstance(camera_sensor, Camera):
            height, width = camera_sensor.image_shape
            video_feature_item = VideoFeatureItem(
                dtype="video", shape=[height, width, 3], names=["height", "width", "channels"]
            )
            video_feature_item.video_info["video.height"] = height
            video_feature_item.video_info["video.width"] = width
            video_feature_item.video_info["video.fps"] = dataset_cfg.fps
            features[f"observation.images.{camera_key}"] = asdict(video_feature_item)

    return features


def is_so101_at_rest_pose(joint_pos: torch.Tensor | np.ndarray, joint_names: list[str]) -> np.ndarray:
    """로봇이 rest pose 에 있는지 판정한다.

    Args:
        joint_pos: (6,) 또는 (N, 6) 라디안 배열.
        joint_names: 관절 이름 리스트 (순서는 joint_pos 와 매칭).

    Returns:
        (N,) 또는 스칼라 bool 배열.
    """
    # Rest pose 범위(degree) — leader_calibration 에서 정의된 값 참조
    rest_pose_range = {
        "shoulder_pan": (-20.0, 20.0),
        "shoulder_lift": (-30.0, 30.0),
        "elbow_flex": (-30.0, 30.0),
        "wrist_flex": (-30.0, 30.0),
        "wrist_roll": (-30.0, 30.0),
        "gripper": (-10.0, 10.0),
    }

    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.cpu().numpy()

    # 1D 인 경우 (6,) → (1, 6)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    joint_pos_deg = joint_pos / np.pi * 180.0

    is_reset = np.ones(joint_pos.shape[0], dtype=bool)
    for joint_name, (min_pos, max_pos) in rest_pose_range.items():
        joint_idx = joint_names.index(joint_name)
        is_reset &= (joint_pos_deg[:, joint_idx] > min_pos) & (joint_pos_deg[:, joint_idx] < max_pos)

    return is_reset[0] if squeeze else is_reset
