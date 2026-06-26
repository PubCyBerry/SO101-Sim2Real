"""Policy observation/action replay용 portable NPZ snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .feature_codec import CODEC_VERSION, JOINT_FEATURE_NAMES

SNAPSHOT_VERSION = "so101_policy_io_snapshot_v1"


def save_policy_io_snapshot(
    path: str | Path,
    *,
    observation: dict[str, Any],
    request_timestep: int,
    must_go: bool,
    action_timesteps: list[int],
    actions_feature: np.ndarray,
    actions_sim_rad: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """한 policy request/response를 모델·ROS 의존성 없는 NPZ로 저장한다."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image_keys = sorted(
        key for key, value in observation.items()
        if isinstance(value, np.ndarray) and value.ndim >= 2
    )
    scalar_observation = {
        key: value
        for key, value in observation.items()
        if key not in image_keys
    }
    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "codec_version": CODEC_VERSION,
        "request_timestep": int(request_timestep),
        "must_go": bool(must_go),
        "image_keys": image_keys,
        "observation": scalar_observation,
        "metadata": metadata or {},
    }
    arrays: dict[str, np.ndarray] = {
        "manifest_json": np.asarray(json.dumps(manifest, ensure_ascii=False)),
        "action_timesteps": np.asarray(action_timesteps, dtype=np.int64),
        "actions_feature": np.asarray(actions_feature, dtype=np.float32),
        "actions_sim_rad": np.asarray(actions_sim_rad, dtype=np.float32),
    }
    for index, key in enumerate(image_keys):
        arrays[f"image_{index}"] = np.asarray(observation[key])
    np.savez_compressed(destination, **arrays)
    return destination


def load_policy_io_snapshot(path: str | Path) -> dict[str, Any]:
    """NPZ snapshot을 observation과 action 배열로 복원한다."""
    with np.load(Path(path), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version: {manifest.get('snapshot_version')!r}")
        if manifest.get("codec_version") != CODEC_VERSION:
            raise ValueError(f"unsupported codec version: {manifest.get('codec_version')!r}")
        observation = dict(manifest["observation"])
        for index, key in enumerate(manifest["image_keys"]):
            observation[key] = data[f"image_{index}"].copy()

        missing_joint_keys = [key for key in JOINT_FEATURE_NAMES if key not in observation]
        if missing_joint_keys:
            raise ValueError(f"snapshot observation missing joint features: {missing_joint_keys}")

        return {
            "manifest": manifest,
            "observation": observation,
            "action_timesteps": data["action_timesteps"].copy(),
            "actions_feature": data["actions_feature"].copy(),
            "actions_sim_rad": data["actions_sim_rad"].copy(),
        }
