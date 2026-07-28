#!/usr/bin/env python3
"""SO-101 joint-space LeRobot v3 → absolute EEF-space LeRobot v3 파생 변환.

원본 데이터셋은 보존하고 출력 디렉터리에 복사한 뒤, 매 프레임의 6D joint
``observation.state``/``action``을 absolute EEF pose로 변환한다. 회전 표현은
``--rotation-representation {rot6d,rpy,wxyz}``로 선택하며 기본값은 ``rot6d``다.

기본 10D layout:

    [tcp_grasp xyz(3), rotation matrix 첫 두 row(6), gripper feature(1)]

``--rotation-representation rpy`` 7D layout:

    [tcp_grasp xyz(3), fixed-axis RPY radian(3), gripper feature(1)]

``--rotation-representation wxyz`` 8D layout:

    [tcp_grasp xyz(3), canonical unit quaternion wxyz(4), gripper feature(1)]

``--keep-joints``를 지정하면 위 layout 뒤에 arm joint radian 5개를 보존한다.

이 스크립트는 absolute 값만 저장한다. 학습/추론에서 true EEF-relative action을
사용하려면 ``T_rel = inv(T_state) @ T_action`` 및 ``T_abs = T_state @ T_rel``을
수행하는 SE(3) processor가 별도로 필요하다. 단순 벡터 뺄셈/덧셈은 회전에 적합하지 않다.

Real/sim 공통 좌표 계약:

- ``--source-domain sim``: policy feature(arm=sim degree, gripper=[0,100])
- ``--source-domain real``: real follower(arm=follower degree, gripper=[0,100])
- 두 입력 모두 calibration 후 동일한 URDF joint radian과 ``base_link→tcp_grasp`` FK를 사용
- real gripper도 canonical sim policy feature [0,100]으로 변환해 출력

사용 예:

    uv run python scripts/convert/joint_dataset_to_eef.py \
        --input-dir datasets/pick_cube_joint_v3 \
        --output-dir datasets/pick_cube_eef_v3 \
        --source-domain sim

    uv run python scripts/convert/joint_dataset_to_eef.py \
        --input-dir datasets/real_pick_cube_joint_v3 \
        --output-dir datasets/real_pick_cube_eef_v3 \
        --source-domain real \
        --rotation-representation wxyz

    uv run python scripts/convert/joint_dataset_to_eef.py --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from so101_contract.eef_kinematics import (  # noqa: E402
    EEF_KINEMATICS_VERSION,
    ROTATION_REPRESENTATION_DIMS,
    ROTATION_REPRESENTATIONS,
    SO101EndEffectorKinematics,
    decode_rotation_representation,
)
from so101_contract.feature_codec import (  # noqa: E402
    JOINT_FEATURE_NAMES,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from so101_contract.follower_calibration import (  # noqa: E402
    real_follower_to_sim_radians,
    sim_radians_to_real_follower,
)

DEFAULT_URDF = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
DEFAULT_ROBOT_YAML = ROOT / "assets" / "robots" / "so101.yml"
CONVERSION_VERSION = "so101_lerobot_abs_joint_to_abs_eef_v2"

XYZ_NAMES = ("tcp_grasp.x", "tcp_grasp.y", "tcp_grasp.z")
ROTATION_NAMES = {
    "rot6d": (
        "tcp_grasp.rot6d.r0c0",
        "tcp_grasp.rot6d.r0c1",
        "tcp_grasp.rot6d.r0c2",
        "tcp_grasp.rot6d.r1c0",
        "tcp_grasp.rot6d.r1c1",
        "tcp_grasp.rot6d.r1c2",
    ),
    "rpy": (
        "tcp_grasp.rpy.roll",
        "tcp_grasp.rpy.pitch",
        "tcp_grasp.rpy.yaw",
    ),
    "wxyz": (
        "tcp_grasp.quaternion.w",
        "tcp_grasp.quaternion.x",
        "tcp_grasp.quaternion.y",
        "tcp_grasp.quaternion.z",
    ),
}
ROTATION_METADATA_FORMATS = {
    "rot6d": "xyz+rot6d_rows",
    "rpy": "xyz+rpy_fixed_axis_radians",
    "wxyz": "xyz+quaternion_wxyz_unit_canonical",
}
GRIPPER_NAMES = ("gripper.pos",)
ARM_RAD_NAMES = tuple(f"{name.removesuffix('.pos')}.rad" for name in JOINT_FEATURE_NAMES[:5])
CORE_COLUMNS = ("observation.state", "action")
STAT_QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def _eef_names(rotation_representation: str) -> tuple[str, ...]:
    try:
        return XYZ_NAMES + ROTATION_NAMES[rotation_representation]
    except KeyError as exc:
        raise ValueError(
            f"unknown rotation representation {rotation_representation!r}; "
            f"expected one of {ROTATION_REPRESENTATIONS}"
        ) from exc


def _output_dim(rotation_representation: str, keep_joints: bool) -> int:
    eef_dim = 3 + ROTATION_REPRESENTATION_DIMS[rotation_representation]
    return eef_dim + 1 + (5 if keep_joints else 0)


def _numeric_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"statistics input must be non-empty (N,D), got {array.shape}")
    stats: dict[str, Any] = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    for name, quantile in STAT_QUANTILES.items():
        stats[name] = np.quantile(array, quantile, axis=0).tolist()
    return stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _safe_prepare_output(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("in-place conversion is forbidden: input-dir and output-dir are identical")
    if input_resolved in output_resolved.parents or output_resolved in input_resolved.parents:
        raise ValueError("input-dir and output-dir must not contain one another")

    if output_resolved.exists():
        if not overwrite:
            raise FileExistsError(f"output-dir already exists: {output_resolved}")
        unsafe = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve(), input_resolved}
        if output_resolved in unsafe or len(output_resolved.parts) < 4:
            raise ValueError(f"refusing to remove unsafe output-dir: {output_resolved}")
        shutil.rmtree(output_resolved)


def _copy_dataset(input_dir: Path, output_dir: Path, video_mode: str) -> dict[str, int]:
    counts = {"hardlinked_videos": 0, "copied_videos": 0}

    def copy_function(src: str, dst: str) -> str:
        source = Path(src)
        is_video = "videos" in source.parts
        if is_video and video_mode == "hardlink":
            try:
                os.link(src, dst)
                counts["hardlinked_videos"] += 1
                return dst
            except OSError:
                # 다른 filesystem/Windows 권한 제한이면 portable copy로 자동 폴백한다.
                pass
        if is_video:
            counts["copied_videos"] += 1
        return shutil.copy2(src, dst)

    shutil.copytree(input_dir, output_dir, copy_function=copy_function)
    return counts


def _validate_input_info(info: dict[str, Any]) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"LeRobot codebase_version must be 'v3.0', got {info.get('codebase_version')!r}"
        )
    features = info.get("features", {})
    for column in CORE_COLUMNS:
        feature = features.get(column)
        if feature is None:
            raise KeyError(f"meta/info.json features missing {column!r}")
        if feature.get("dtype") != "float32" or feature.get("shape") != [6]:
            raise ValueError(f"{column!r} must be float32 shape [6], got {feature}")
        names = feature.get("names")
        if names is not None and tuple(names) != tuple(JOINT_FEATURE_NAMES):
            raise ValueError(
                f"{column!r} joint order mismatch: {names}; expected {list(JOINT_FEATURE_NAMES)}"
            )


def _convert_joint_values(
    values: np.ndarray,
    *,
    source_domain: str,
    kinematics: SO101EndEffectorKinematics,
    keep_joints: bool,
    rotation_representation: str,
) -> np.ndarray:
    joint_values = np.asarray(values, dtype=np.float32)
    if joint_values.ndim != 2 or joint_values.shape[1] != 6:
        raise ValueError(f"joint values must have shape (N,6), got {joint_values.shape}")
    if not np.all(np.isfinite(joint_values)):
        raise ValueError("joint values contain NaN or infinity")

    if source_domain == "sim":
        joint_radians = policy_feature_to_sim_joint_radians(joint_values)
    elif source_domain == "real":
        joint_radians = real_follower_to_sim_radians(joint_values)
    else:
        raise ValueError(f"unknown source domain: {source_domain!r}")

    eef_pose = kinematics.forward_xyz_rotation(
        joint_radians[:, :5],
        rotation_representation,
    )
    # 실 follower의 기계적 gripper endpoint/offset까지 sim policy feature로 통일한다.
    canonical_gripper = sim_joint_radians_to_policy_feature(joint_radians)[:, 5:6]
    parts = [eef_pose, canonical_gripper]
    if keep_joints:
        parts.append(joint_radians[:, :5].astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def _replace_core_columns(
    output_dir: Path,
    *,
    source_domain: str,
    kinematics: SO101EndEffectorKinematics,
    keep_joints: bool,
    rotation_representation: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dim = _output_dim(rotation_representation, keep_joints)
    output_type = pa.list_(pa.float32(), output_dim)
    data_files = sorted((output_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no data parquet under {output_dir / 'data'}")

    global_parts: dict[str, list[np.ndarray]] = {column: [] for column in CORE_COLUMNS}
    episode_parts: dict[str, dict[int, list[np.ndarray]]] = {
        column: {} for column in CORE_COLUMNS
    }

    for data_path in data_files:
        table = pq.read_table(data_path)
        missing = [column for column in (*CORE_COLUMNS, "episode_index") if column not in table.column_names]
        if missing:
            raise KeyError(f"{data_path}: missing columns {missing}")
        episode_indices = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)

        for column in CORE_COLUMNS:
            values = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
            converted = _convert_joint_values(
                values,
                source_domain=source_domain,
                kinematics=kinematics,
                keep_joints=keep_joints,
                rotation_representation=rotation_representation,
            )
            global_parts[column].append(converted)
            for episode_index in np.unique(episode_indices):
                episode_parts[column].setdefault(int(episode_index), []).append(
                    converted[episode_indices == episode_index]
                )
            table = table.set_column(
                table.schema.get_field_index(column),
                column,
                pa.array(converted.tolist(), type=output_type),
            )

        temp_path = data_path.with_suffix(data_path.suffix + ".tmp")
        pq.write_table(table, temp_path)
        os.replace(temp_path, data_path)
        print(
            f"[data] {data_path.relative_to(output_dir)}: "
            f"6D joint → {output_dim}D EEF ({rotation_representation})"
        )

    global_values = {
        column: np.concatenate(parts, axis=0) for column, parts in global_parts.items()
    }
    episode_values = {
        column: {
            episode_index: np.concatenate(parts, axis=0)
            for episode_index, parts in per_episode.items()
        }
        for column, per_episode in episode_parts.items()
    }
    return global_values, episode_values


def _update_episode_statistics(
    output_dir: Path,
    episode_values: dict[str, dict[int, np.ndarray]],
) -> None:
    """Stock LeRobot v3의 episodes parquet에 기존 per-episode stats가 있으면 갱신."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_files = sorted((output_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        return
    cached_stats = {
        column: {
            episode_index: _numeric_stats(values)
            for episode_index, values in per_episode.items()
        }
        for column, per_episode in episode_values.items()
    }

    for path in episode_files:
        table = pq.read_table(path)
        if "episode_index" not in table.column_names:
            continue
        episode_indices = [int(v) for v in table.column("episode_index").to_pylist()]
        changed = False
        for index, field in enumerate(table.schema):
            for column in CORE_COLUMNS:
                prefix = f"stats/{column}/"
                if not field.name.startswith(prefix):
                    continue
                stat_name = field.name[len(prefix) :]
                if stat_name not in {"min", "max", "mean", "std", "count", *STAT_QUANTILES}:
                    continue
                values = [cached_stats[column][episode][stat_name] for episode in episode_indices]
                target_type = field.type
                if stat_name != "count" and pa.types.is_fixed_size_list(field.type):
                    target_type = pa.list_(field.type.value_type, len(values[0]))
                try:
                    replacement = pa.array(values, type=target_type)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    # 일부 upstream 버전은 count를 scalar로 저장한다.
                    if stat_name != "count":
                        raise
                    replacement = pa.array([int(v[0]) for v in values], type=target_type)
                table = table.set_column(index, field.name, replacement)
                changed = True
        if changed:
            temp_path = path.with_suffix(path.suffix + ".tmp")
            pq.write_table(table, temp_path)
            os.replace(temp_path, path)
            print(f"[meta] {path.relative_to(output_dir)}: per-episode EEF stats 갱신")


def _update_metadata(
    output_dir: Path,
    *,
    source_domain: str,
    keep_joints: bool,
    rotation_representation: str,
    global_values: dict[str, np.ndarray],
    urdf_path: Path,
    robot_yaml_path: Path,
    video_mode: str,
) -> None:
    info_path = output_dir / "meta" / "info.json"
    stats_path = output_dir / "meta" / "stats.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}

    eef_names = _eef_names(rotation_representation)
    output_names = list(eef_names + GRIPPER_NAMES + (ARM_RAD_NAMES if keep_joints else ()))
    output_dim = len(output_names)
    for column in CORE_COLUMNS:
        previous = info["features"][column]
        info["features"][column] = {
            **previous,
            "dtype": "float32",
            "shape": [output_dim],
            "names": output_names,
        }
        stats[column] = _numeric_stats(global_values[column])

    info["so101_eef_conversion"] = {
        "version": CONVERSION_VERSION,
        "source_domain": source_domain,
        "base_frame": "base_link",
        "eef_frame": "tcp_grasp",
        "eef_kinematics_version": EEF_KINEMATICS_VERSION,
        "rotation_representation": rotation_representation,
        "rotation_format": ROTATION_METADATA_FORMATS[rotation_representation],
        "gripper_format": "canonical_policy_feature_[0,100]",
        "keep_joints": keep_joints,
        "joint_format": "sim_urdf_radian" if keep_joints else None,
        "video_copy_mode": video_mode,
        "urdf_sha256": _sha256(urdf_path),
        "robot_yaml_sha256": _sha256(robot_yaml_path),
    }

    # schema v2 dataset action contract 블록. modality.json과 같은 group을 선언해
    # `so101_contract.action_dataset_contract`가 metadata만으로 계약을 resolve한다.
    eef_dim = len(eef_names)
    v2_groups = {
        f"eef_{eef_dim}d": {"start": 0, "end": eef_dim},
        "gripper_position": {"start": eef_dim, "end": eef_dim + 1},
    }
    if keep_joints:
        v2_groups["joint_position"] = {"start": eef_dim + 1, "end": eef_dim + 6}
    info["so101_action_representation"] = {
        "version": "so101_dataset_action_contract_v2",
        "space": "eef",
        "storage_reference": "absolute",
        "transform_group": f"eef_{eef_dim}d",
        "groups": v2_groups,
    }
    _atomic_write_json(info_path, info)
    _atomic_write_json(stats_path, stats)

    eef_dim = len(eef_names)
    split = {
        f"eef_{eef_dim}d": {"start": 0, "end": eef_dim},
        "gripper_position": {"start": eef_dim, "end": eef_dim + 1},
    }
    if keep_joints:
        split["joint_position"] = {"start": eef_dim + 1, "end": eef_dim + 6}
    video = {
        key.removeprefix("observation.images."): {"original_key": key}
        for key, feature in info["features"].items()
        if key.startswith("observation.images.") and feature.get("dtype") == "video"
    }
    modality = {
        "state": split,
        "action": split,
        "video": video,
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    _atomic_write_json(output_dir / "meta" / "modality.json", modality)


def _validate_output(
    output_dir: Path,
    expected_dim: int,
    rotation_representation: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    stats = json.loads((output_dir / "meta" / "stats.json").read_text(encoding="utf-8"))
    for column in CORE_COLUMNS:
        if info["features"][column]["shape"] != [expected_dim]:
            raise AssertionError(f"{column} info shape mismatch")
        if len(stats[column]["mean"]) != expected_dim:
            raise AssertionError(f"{column} stats shape mismatch")

    total_rows = 0
    for path in sorted((output_dir / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        total_rows += table.num_rows
        for column in CORE_COLUMNS:
            field_type = table.schema.field(column).type
            if not (
                pa.types.is_fixed_size_list(field_type)
                and field_type.list_size == expected_dim
                and pa.types.is_float32(field_type.value_type)
            ):
                raise AssertionError(f"{path}: {column} type mismatch: {field_type}")
            values = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise AssertionError(f"{path}: {column} contains non-finite values")
            rotation_dim = ROTATION_REPRESENTATION_DIMS[rotation_representation]
            encoded_rotation = values[:, 3 : 3 + rotation_dim]
            rotations = decode_rotation_representation(
                encoded_rotation,
                rotation_representation,
            )
            identity = np.eye(3)[None, :, :]
            if not np.allclose(
                rotations @ np.swapaxes(rotations, -1, -2),
                identity,
                atol=3e-5,
            ):
                raise AssertionError(f"{path}: decoded rotation is not orthonormal")
            if not np.allclose(np.linalg.det(rotations), 1.0, atol=3e-5):
                raise AssertionError(f"{path}: decoded rotation determinant is not +1")

            if rotation_representation == "rpy":
                roll, pitch, yaw = encoded_rotation.T
                if np.any(np.abs(roll) > np.pi + 1e-6) or np.any(np.abs(yaw) > np.pi + 1e-6):
                    raise AssertionError(f"{path}: RPY roll/yaw is outside [-pi, pi]")
                if np.any(np.abs(pitch) > np.pi / 2 + 1e-6):
                    raise AssertionError(f"{path}: RPY pitch is outside [-pi/2, pi/2]")
            elif rotation_representation == "wxyz":
                if not np.allclose(
                    np.linalg.norm(encoded_rotation, axis=1),
                    1.0,
                    atol=2e-5,
                ):
                    raise AssertionError(f"{path}: quaternion is not unit length")
                if np.any(encoded_rotation[:, 0] < -1e-7):
                    raise AssertionError(f"{path}: quaternion is outside canonical hemisphere")
    if total_rows <= 0:
        raise AssertionError("converted dataset has no rows")


def convert_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    source_domain: str,
    keep_joints: bool,
    overwrite: bool,
    video_mode: str,
    urdf_path: Path,
    robot_yaml_path: Path,
    rotation_representation: str = "rot6d",
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    urdf_path = urdf_path.resolve()
    robot_yaml_path = robot_yaml_path.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")
    info_path = input_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")
    input_info = json.loads(info_path.read_text(encoding="utf-8"))
    _validate_input_info(input_info)
    if rotation_representation not in ROTATION_REPRESENTATIONS:
        raise ValueError(
            f"unknown rotation representation {rotation_representation!r}; "
            f"expected one of {ROTATION_REPRESENTATIONS}"
        )
    _safe_prepare_output(input_dir, output_dir, overwrite)

    kinematics = SO101EndEffectorKinematics.from_files(urdf_path, robot_yaml_path)
    copy_counts = _copy_dataset(input_dir, output_dir, video_mode)
    global_values, episode_values = _replace_core_columns(
        output_dir,
        source_domain=source_domain,
        kinematics=kinematics,
        keep_joints=keep_joints,
        rotation_representation=rotation_representation,
    )
    _update_episode_statistics(output_dir, episode_values)
    _update_metadata(
        output_dir,
        source_domain=source_domain,
        keep_joints=keep_joints,
        rotation_representation=rotation_representation,
        global_values=global_values,
        urdf_path=urdf_path,
        robot_yaml_path=robot_yaml_path,
        video_mode=video_mode,
    )
    output_dim = _output_dim(rotation_representation, keep_joints)
    _validate_output(output_dir, output_dim, rotation_representation)
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_domain": source_domain,
        "rotation_representation": rotation_representation,
        "layout_dim": output_dim,
        "frames": int(global_values["action"].shape[0]),
        **copy_counts,
    }


def _write_fixture(root: Path, joint_values: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4").write_bytes(
        b"self-check-video"
    )

    fsl6 = pa.list_(pa.float32(), 6)
    episode_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    action = joint_values.astype(np.float32)
    state = joint_values.copy().astype(np.float32)
    table = pa.table(
        {
            "action": pa.array(action.tolist(), type=fsl6),
            "observation.state": pa.array(state.tolist(), type=fsl6),
            "timestamp": pa.array([0.0, 1 / 30, 0.0, 1 / 30], type=pa.float32()),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    episode_means = [
        action[episode_indices == episode].mean(axis=0).tolist() for episode in (0, 1)
    ]
    episodes = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "length": pa.array([2, 2], type=pa.int64()),
            "stats/action/mean": pa.array(episode_means, type=fsl6),
            "stats/observation.state/mean": pa.array(episode_means, type=fsl6),
        }
    )
    pq.write_table(episodes, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "so_follower",
        "fps": 30,
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": list(JOINT_FEATURE_NAMES)},
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": list(JOINT_FEATURE_NAMES),
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [2, 2, 3],
                "names": ["height", "width", "channels"],
            },
        },
    }
    _atomic_write_json(root / "meta" / "info.json", info)
    _atomic_write_json(
        root / "meta" / "stats.json",
        {
            "action": _numeric_stats(action),
            "observation.state": _numeric_stats(state),
        },
    )


def self_check() -> int:
    import pyarrow.parquet as pq

    joint_radians = np.asarray(
        [
            [0.0, -1.0, 1.0, 0.5, 0.0, np.deg2rad(-10.0)],
            [0.2, -0.8, 0.7, 0.3, 0.4, np.deg2rad(20.0)],
            [-0.3, -1.2, 1.1, 0.6, -0.5, np.deg2rad(65.0)],
            [0.5, -0.6, 0.4, -0.2, 0.8, np.deg2rad(100.0)],
        ],
        dtype=np.float32,
    )
    sim_values = sim_joint_radians_to_policy_feature(joint_radians)
    real_values = sim_radians_to_real_follower(joint_radians)

    with tempfile.TemporaryDirectory(prefix="so101-eef-self-check-") as temp:
        temp_path = Path(temp)
        sim_input, real_input = temp_path / "sim", temp_path / "real"
        _write_fixture(sim_input, sim_values)
        _write_fixture(real_input, real_values)

        common = {
            "overwrite": False,
            "video_mode": "hardlink",
            "urdf_path": DEFAULT_URDF,
            "robot_yaml_path": DEFAULT_ROBOT_YAML,
        }

        def column(root: Path, name: str) -> np.ndarray:
            table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
            return np.asarray(table.column(name).to_pylist(), dtype=np.float32)

        reconstructed: dict[str, np.ndarray] = {}
        for representation in ROTATION_REPRESENTATIONS:
            sim_output = temp_path / f"sim_eef_{representation}"
            real_output = temp_path / f"real_eef_{representation}"
            dual_output = temp_path / f"sim_eef_joint_{representation}"
            convert_dataset(
                input_dir=sim_input,
                output_dir=sim_output,
                source_domain="sim",
                keep_joints=False,
                rotation_representation=representation,
                **common,
            )
            convert_dataset(
                input_dir=real_input,
                output_dir=real_output,
                source_domain="real",
                keep_joints=False,
                rotation_representation=representation,
                **common,
            )
            convert_dataset(
                input_dir=sim_input,
                output_dir=dual_output,
                source_domain="sim",
                keep_joints=True,
                rotation_representation=representation,
                **common,
            )

            expected_dim = _output_dim(representation, False)
            expected_dual_dim = _output_dim(representation, True)
            for name in CORE_COLUMNS:
                sim_eef = column(sim_output, name)
                real_eef = column(real_output, name)
                if not np.allclose(sim_eef, real_eef, atol=2e-5):
                    maximum = float(np.max(np.abs(sim_eef - real_eef)))
                    raise AssertionError(
                        f"sim/real {name} {representation} EEF mismatch: max={maximum}"
                    )
                if sim_eef.shape[1] != expected_dim:
                    raise AssertionError(
                        f"{representation} EEF layout: {sim_eef.shape[1]} != {expected_dim}"
                    )
                if column(dual_output, name).shape[1] != expected_dual_dim:
                    raise AssertionError(
                        f"{representation} EEF+joint layout shape mismatch"
                    )

            action = column(sim_output, "action")
            rotation_dim = ROTATION_REPRESENTATION_DIMS[representation]
            reconstructed[representation] = decode_rotation_representation(
                action[:, 3 : 3 + rotation_dim],
                representation,
            )

            episode_table = pq.read_table(
                sim_output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
            )
            stats_type = episode_table.schema.field("stats/action/mean").type
            if stats_type.list_size != expected_dim:
                raise AssertionError(
                    f"{representation} per-episode stats: {stats_type.list_size} != {expected_dim}"
                )

            info = json.loads(
                (sim_output / "meta" / "info.json").read_text(encoding="utf-8")
            )
            conversion = info["so101_eef_conversion"]
            if conversion["rotation_representation"] != representation:
                raise AssertionError(f"{representation} metadata mismatch")

        for name in CORE_COLUMNS:
            if column(sim_input, name).shape[1] != 6:
                raise AssertionError("source dataset was modified")

        reference = reconstructed["rot6d"]
        for representation, rotations in reconstructed.items():
            if not np.allclose(reference, rotations, atol=3e-5):
                maximum = float(np.max(np.abs(reference - rotations)))
                raise AssertionError(
                    f"rotation round-trip mismatch for {representation}: max={maximum}"
                )

    print("[self-check] PASS")
    print("  - sim policy feature와 real follower calibration → 동일 absolute EEF")
    print("  - rot6d 10/15D, rpy 7/12D, wxyz 8/13D layout")
    print("  - rot6d/rpy/wxyz → 동일 rotation matrix round-trip")
    print("  - source 보존, info/stats/modality/per-episode stats 갱신")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="원본 joint-space LeRobot v3 디렉터리")
    parser.add_argument("--output-dir", type=Path, help="출력 absolute EEF LeRobot v3 디렉터리")
    parser.add_argument(
        "--source-domain",
        choices=("sim", "real"),
        help="입력 6D joint 단위계. 자동 판별하지 않으므로 반드시 명시",
    )
    parser.add_argument(
        "--keep-joints",
        action="store_true",
        help="EEF+gripper 뒤에 arm joint radian 5개를 추가",
    )
    parser.add_argument(
        "--rotation-representation",
        "--rotation-format",
        dest="rotation_representation",
        choices=ROTATION_REPRESENTATIONS,
        default="rot6d",
        help="EEF 회전 표현: rot6d(기본), rpy(radian), wxyz(unit quaternion)",
    )
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="파생셋 video 복제 방식(기본 hardlink, 실패 시 copy 폴백)",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--robot-yaml", type=Path, default=DEFAULT_ROBOT_YAML)
    parser.add_argument("--overwrite", action="store_true", help="기존 output-dir 삭제 후 재생성")
    parser.add_argument("--self-check", action="store_true", help="합성 sim/real v3로 전체 변환 검증")
    args = parser.parse_args()

    if args.self_check:
        return self_check()
    missing = [
        flag
        for flag, value in (
            ("--input-dir", args.input_dir),
            ("--output-dir", args.output_dir),
            ("--source-domain", args.source_domain),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required arguments: {', '.join(missing)}")

    summary = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_domain=args.source_domain,
        keep_joints=args.keep_joints,
        overwrite=args.overwrite,
        video_mode=args.video_mode,
        urdf_path=args.urdf,
        robot_yaml_path=args.robot_yaml,
        rotation_representation=args.rotation_representation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "[done] dataset values remain absolute; EEF-relative 학습/추론에는 "
        "별도의 SE(3) relative processor가 필요합니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
