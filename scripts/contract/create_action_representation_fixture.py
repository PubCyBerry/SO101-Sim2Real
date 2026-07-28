#!/usr/bin/env python3
"""Phase 15 — schema v2 universal LeRobot v3 fixture 생성기.

joint absolute dataset과 EEF absolute dataset(3 pose format)을 만들고, 두 경우 모두
``meta/info.json``에 ``so101_action_representation`` 블록(group + joint topology)을 기록해
schema v2 계약이 dataset만으로 resolve되게 한다. 마지막으로 그 dataset에서 유효한
representation의 stats profile을 모두 생성해 한 artifact에 저장한다.

실제 학습 데이터 대체물이 아니라 CLI/factory/manifest 통합 계약 검증용이다.

.. code-block:: bash

    python scripts/contract/create_action_representation_fixture.py \\
        --output-dir /tmp/fx_joint --space joint
    python scripts/contract/create_action_representation_fixture.py \\
        --output-dir /tmp/fx_rot6d --space eef --pose-format xyz_rot6d_rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from so101_contract.action_dataset_contract import (  # noqa: E402
    ACTION_DATASET_CONTRACT_VERSION,
    DATASET_CONTRACT_BLOCK,
    POSE_FORMAT_METADATA_STRING,
    ROTATION_REPRESENTATION_TO_POSE_FORMAT,
    eef_feature_names,
    resolve_action_representation_contract,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    coerce_pose_format,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ActionStatsSampling,
    calculate_action_representation_stats,
    empty_stats_artifact,
    load_lerobot_v3_episodes,
    upsert_stats_profile,
    write_action_stats_artifact,
)
from so101_contract.eef_deployment_contract import sha256_file  # noqa: E402
from so101_contract.eef_kinematics import EEF_KINEMATICS_VERSION  # noqa: E402
from so101_contract.joint_topology import so101_arm_joint_topology  # noqa: E402
from so101_contract.pose_codec import encode_pose  # noqa: E402

JOINT_TOPOLOGY = so101_arm_joint_topology()
JOINT_FEATURE_NAMES = list(JOINT_TOPOLOGY.names) + ["gripper.pos"]


def _rotation(angle: float) -> np.ndarray:
    """z축 회전 + 약간의 tilt. 세 format 모두 non-trivial한 회전을 겪게 한다."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    tilt = 0.35
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    rotation_z = np.asarray(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y


def _eef_row(pose_format: PoseFormat, episode: int, frame: int) -> np.ndarray:
    angle = -0.25 + 0.02 * frame + 0.05 * episode
    translation = np.asarray(
        [0.18 + 0.002 * frame, 0.22 + 0.01 * episode, 0.16 + 0.003 * np.sin(frame * 0.2)],
        dtype=np.float64,
    )
    pose = encode_pose(translation, _rotation(angle), pose_format)
    gripper = np.asarray([20.0 + 60.0 * ((frame % 16) / 15.0)], dtype=np.float64)
    return np.concatenate([pose, gripper]).astype(np.float32)


def _joint_row(episode: int, frame: int) -> np.ndarray:
    """periodic wrap을 실제로 겪도록 ±π 경계를 가로지르는 궤적을 만든다."""
    base = np.asarray(
        [
            2.9 + 0.06 * frame + 0.1 * episode,  # +π 경계를 넘어간다
            -0.4 + 0.01 * frame,
            0.3 - 0.008 * frame,
            0.12 * np.sin(0.25 * frame),
            -2.95 - 0.05 * frame - 0.1 * episode,  # -π 경계를 넘어간다
        ],
        dtype=np.float64,
    )
    wrapped = (base + np.pi) % (2.0 * np.pi) - np.pi
    gripper = np.asarray([20.0 + 60.0 * ((frame % 16) / 15.0)], dtype=np.float64)
    return np.concatenate([wrapped, gripper]).astype(np.float32)


def _specs_for(space: str, pose_format: PoseFormat | None) -> list[ActionRepresentationSpec]:
    if space == "joint":
        return [
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
        ]
    return [
        ActionRepresentationSpec(mode=mode, pose_format=pose_format)
        for mode in (
            ActionRepresentationMode.EEF_ABSOLUTE,
            ActionRepresentationMode.EEF_RELATIVE,
        )
    ]


def create_fixture(
    output_dir: Path,
    *,
    space: str,
    pose_format: PoseFormat | None,
    repo_id: str,
    episodes: int,
    frames_per_episode: int,
    horizon: int,
    with_images: bool,
    image_size: int,
) -> dict[str, object]:
    if episodes <= 0 or horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if frames_per_episode < horizon:
        raise ValueError(
            f"frames_per_episode must be >= horizon: {frames_per_episode} < {horizon}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"fixture output must be absent or empty: {output_dir}")

    if space == "joint":
        names = list(JOINT_FEATURE_NAMES)
        transform_group = "arm_joints"
        joint_dim = JOINT_TOPOLOGY.dim
    else:
        if pose_format is None:
            raise ValueError("EEF fixtures require --pose-format")
        names = list(eef_feature_names(pose_format))
        joint_dim = len(names) - 1
        transform_group = f"eef_{joint_dim}d"
    dimension = len(names)

    feature = {"dtype": "float32", "shape": (dimension,), "names": names}
    features = {"observation.state": dict(feature), "action": dict(feature)}
    camera_keys = ("top", "wrist", "front") if with_images else ()
    if not with_images:
        # ACT는 state 외에 image 또는 environment_state 입력을 하나 요구한다.
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (2,),
            "names": ["fixture_phase", "fixture_episode"],
        }
    for camera_key in camera_keys:
        features[f"observation.images.{camera_key}"] = {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        root=output_dir,
        robot_type="so101_follower",
        use_videos=False,
        image_writer_threads=2 if with_images else 0,
    )
    for episode in range(episodes):
        for frame in range(frames_per_episode):
            if space == "joint":
                state = _joint_row(episode, frame)
                action = _joint_row(episode, frame + 1)
            else:
                state = _eef_row(pose_format, episode, frame)
                action = _eef_row(pose_format, episode, frame + 1)
            row = {
                "observation.state": state,
                "action": action,
                "task": "schema v2 action representation fixture",
            }
            if not with_images:
                row["observation.environment_state"] = np.asarray(
                    [
                        frame / max(frames_per_episode - 1, 1),
                        episode / max(episodes - 1, 1),
                    ],
                    dtype=np.float32,
                )
            for camera_index, camera_key in enumerate(camera_keys):
                image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                image[..., camera_index] = np.uint8((frame * 11 + episode * 37) % 256)
                image[:, :, (camera_index + 1) % 3] = np.arange(image_size, dtype=np.uint8)[
                    :, None
                ]
                row[f"observation.images.{camera_key}"] = image
            dataset.add_frame(row)
        dataset.save_episode()
    dataset.finalize()

    # --- schema v2 metadata -------------------------------------------------
    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    groups = {
        transform_group: {"start": 0, "end": dimension - 1},
        "gripper_position": {"start": dimension - 1, "end": dimension},
    }
    block: dict[str, object] = {
        "version": ACTION_DATASET_CONTRACT_VERSION,
        "space": space,
        "storage_reference": "absolute",
        "transform_group": transform_group,
        "groups": groups,
    }
    if space == "joint":
        block["joints"] = [joint.to_dict() for joint in JOINT_TOPOLOGY.joints]
    info[DATASET_CONTRACT_BLOCK] = block

    if space == "eef":
        representation = next(
            key
            for key, value in ROTATION_REPRESENTATION_TO_POSE_FORMAT.items()
            if value is pose_format
        )
        urdf_path = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
        robot_yaml_path = ROOT / "assets" / "robots" / "so101.yml"
        info["so101_eef_conversion"] = {
            "base_frame": "base_link",
            "eef_frame": "tcp_grasp",
            "eef_kinematics_version": EEF_KINEMATICS_VERSION,
            "rotation_representation": representation,
            "rotation_format": POSE_FORMAT_METADATA_STRING[pose_format],
            "gripper_format": "canonical_policy_feature_[0,100]",
            "keep_joints": False,
            "urdf_sha256": sha256_file(urdf_path),
            "robot_yaml_sha256": sha256_file(robot_yaml_path),
        }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    if space == "eef":
        modality = {
            section: {
                transform_group: {"start": 0, "end": dimension - 1},
                "gripper_position": {"start": dimension - 1, "end": dimension},
            }
            for section in ("state", "action")
        }
        (output_dir / "meta" / "modality.json").write_text(
            json.dumps(modality, indent=2) + "\n",
            encoding="utf-8",
        )

    # --- stats profiles -----------------------------------------------------
    sampling = ActionStatsSampling(action_delta_indices=tuple(range(horizon)))
    artifact = empty_stats_artifact()
    profiles: dict[str, str] = {}
    stats_file = None
    for spec in _specs_for(space, pose_format):
        contract = resolve_action_representation_contract(output_dir, spec)
        episodes_arrays = load_lerobot_v3_episodes(
            output_dir,
            state_key=contract.state_key,
            action_key=contract.action_key,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
        )
        result = calculate_action_representation_stats(
            episodes_arrays,
            sampling,
            contract.transform,
            dataset_fingerprint=contract.fingerprint,
        )
        artifact, _ = upsert_stats_profile(artifact, result)
        profiles[spec.stats_profile_kind] = result.profile_id
        stats_file = spec.stats_file
    stats_path = write_action_stats_artifact(output_dir, artifact, output_file=stats_file)

    return {
        "dataset_root": str(output_dir.resolve()),
        "repo_id": repo_id,
        "space": space,
        "pose_format": pose_format.value if pose_format else None,
        "state_action_dim": dimension,
        "episodes": episodes,
        "frames_per_episode": frames_per_episode,
        "horizon": horizon,
        "camera_keys": list(camera_keys),
        "stats_artifact": str(stats_path.resolve()),
        "stats_profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--space", choices=("joint", "eef"), required=True)
    parser.add_argument(
        "--pose-format",
        choices=("xyz_rot6d_rows", "xyz_quaternion_wxyz", "xyz_rpy"),
        help="EEF fixture에서 필수",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    pose_format = coerce_pose_format(args.pose_format) if args.pose_format else None
    if args.space == "eef" and pose_format is None:
        parser.error("--space eef requires --pose-format")
    if args.space == "joint" and pose_format is not None:
        parser.error("joint fixtures must not declare --pose-format")
    repo_id = args.repo_id or (
        f"local/so101_{args.space}_{args.pose_format or 'absolute'}_fixture"
    )
    result = create_fixture(
        args.output_dir,
        space=args.space,
        pose_format=pose_format,
        repo_id=repo_id,
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        horizon=args.horizon,
        with_images=args.with_images,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
