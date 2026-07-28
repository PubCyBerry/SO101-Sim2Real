"""Absolute EEF LeRobot v3 dataset의 horizon-aware relative action stats.

episode boundary를 넘지 않는 full action window만 사용해 다음 순서로 통계를 만든다.

``absolute EEF action → current-state-relative SE(3) pose + absolute gripper``

큰 horizon에서도 전체 변환 결과를 RAM에 보관하지 않도록 임시 memmap을 사용하고,
각 horizon별 ``min/max/mean/std/q01/q10/q50/q90/q99``를 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .eef_action_contract import (
    ActionRepresentationConfig,
    ResolvedEEFActionContract,
    resolve_eef_action_contract,
)
from .eef_relative_action import EEF_RELATIVE_ACTION_VERSION, absolute_actions_to_relative

RELATIVE_ACTION_STATS_SCHEMA_VERSION = 1
RELATIVE_ACTION_STATS_GENERATOR_VERSION = "so101_eef_relative_stats_v1"
RELATIVE_ACTION_STATS_DEFAULT_PATH = "meta/relative_action_stats.json"
RELATIVE_ACTION_STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


@dataclass(frozen=True)
class RelativeActionSamplingConfig:
    """Training sampler와 동일해야 하는 relative stats window 계약."""

    observation_delta_indices: tuple[int, ...] = (0,)
    action_delta_indices: tuple[int, ...] = (0,)
    reference_observation_index: int = -1

    def __post_init__(self) -> None:
        if not self.observation_delta_indices:
            raise ValueError("observation_delta_indices must not be empty")
        if not self.action_delta_indices:
            raise ValueError("action_delta_indices must not be empty")
        if tuple(sorted(set(self.action_delta_indices))) != self.action_delta_indices:
            raise ValueError(
                "action_delta_indices must be unique and strictly increasing, got "
                f"{self.action_delta_indices}"
            )
        if not -len(self.observation_delta_indices) <= self.reference_observation_index < len(
            self.observation_delta_indices
        ):
            raise IndexError(
                "reference_observation_index is outside observation_delta_indices: "
                f"{self.reference_observation_index}"
            )
        reference_delta = self.observation_delta_indices[self.reference_observation_index]
        if reference_delta != 0:
            raise ValueError(
                f"v1 requires current observation delta 0 as reference, got {reference_delta}"
            )
        if self.action_delta_indices[0] != 0:
            raise ValueError(
                f"v1 requires action_delta_indices[0] == 0, got {self.action_delta_indices[0]}"
            )
        if any(index < 0 for index in self.action_delta_indices):
            raise ValueError(
                f"v1 action_delta_indices must be non-negative, got {self.action_delta_indices}"
            )

    @property
    def reference_delta(self) -> int:
        return self.observation_delta_indices[self.reference_observation_index]

    @property
    def horizon(self) -> int:
        return len(self.action_delta_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_delta_indices": list(self.observation_delta_indices),
            "action_delta_indices": list(self.action_delta_indices),
            "reference_observation_index": self.reference_observation_index,
            "reference_delta": self.reference_delta,
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class RelativeActionStatsResult:
    """한 sampling profile의 계산 결과와 dataset fingerprint."""

    dataset_contract: dict[str, Any]
    profile_id: str
    profile: dict[str, Any]


@dataclass(frozen=True)
class _Episode:
    episode_index: int
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return int(self.frame_indices.shape[0])


def _sha256_json(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _stats_profile_id(profile: dict[str, Any]) -> str:
    """Profile 계약과 수치 통계를 함께 묶는 content-addressed ID."""
    payload = {
        "generator_version": RELATIVE_ACTION_STATS_GENERATOR_VERSION,
        "production": profile.get("production"),
        "dataset_contract": profile.get("dataset_contract"),
        "transform": profile.get("transform"),
        "sampling": profile.get("sampling"),
        "stats": profile.get("stats"),
    }
    return f"sha256:{_sha256_json(payload)}"


def _load_episodes(
    dataset_root: Path,
    contract: ResolvedEEFActionContract,
    *,
    max_episodes: int | None,
) -> tuple[list[_Episode], str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("relative action stats generation requires pyarrow") from exc

    data_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data parquet under {dataset_root / 'data'}")

    parts: dict[int, dict[str, list[np.ndarray]]] = {}
    required_columns = (
        "episode_index",
        "frame_index",
        contract.state_key,
        contract.action_key,
    )
    for path in data_paths:
        table = pq.read_table(path, columns=list(required_columns))
        missing = [name for name in required_columns if name not in table.column_names]
        if missing:
            raise KeyError(f"{path}: missing required columns {missing}")
        episode_indices = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        states = np.asarray(table.column(contract.state_key).to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column(contract.action_key).to_pylist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != contract.state_dim:
            raise ValueError(f"{path}: state shape mismatch {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != contract.action_dim:
            raise ValueError(f"{path}: action shape mismatch {actions.shape}")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"{path}: state/action contains NaN or infinity")

        for episode_index in np.unique(episode_indices):
            mask = episode_indices == episode_index
            target = parts.setdefault(
                int(episode_index),
                {"frame": [], "state": [], "action": []},
            )
            target["frame"].append(frame_indices[mask])
            target["state"].append(states[mask])
            target["action"].append(actions[mask])

    selected_indices = sorted(parts)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {max_episodes}")
        selected_indices = selected_indices[:max_episodes]
    if not selected_indices:
        raise ValueError("dataset contains no selected episodes")

    episodes: list[_Episode] = []
    source_digest = hashlib.sha256()
    for episode_index in selected_indices:
        episode_parts = parts[episode_index]
        frames = np.concatenate(episode_parts["frame"]).astype(np.int64, copy=False)
        states = np.concatenate(episode_parts["state"]).astype(np.float32, copy=False)
        actions = np.concatenate(episode_parts["action"]).astype(np.float32, copy=False)
        order = np.argsort(frames, kind="stable")
        frames = frames[order]
        states = states[order]
        actions = actions[order]
        expected_frames = np.arange(len(frames), dtype=np.int64)
        if not np.array_equal(frames, expected_frames):
            raise ValueError(
                f"episode {episode_index} frame_index must be unique contiguous [0,N), "
                f"got {frames.tolist()[:20]}"
            )

        source_digest.update(np.asarray([episode_index, len(frames)], dtype="<i8").tobytes())
        source_digest.update(np.asarray(frames, dtype="<i8", order="C").tobytes())
        source_digest.update(np.asarray(states, dtype="<f4", order="C").tobytes())
        source_digest.update(np.asarray(actions, dtype="<f4", order="C").tobytes())
        episodes.append(
            _Episode(
                episode_index=episode_index,
                frame_indices=frames,
                states=states,
                actions=actions,
            )
        )

    return episodes, source_digest.hexdigest()


def _episode_anchor_bounds(
    episode_length: int,
    sampling: RelativeActionSamplingConfig,
) -> tuple[int, int]:
    required_deltas = (sampling.reference_delta, *sampling.action_delta_indices)
    start = max(0, -min(required_deltas))
    stop = episode_length - max(required_deltas)
    return start, max(start, stop)


def _relative_episode_windows(
    episode: _Episode,
    sampling: RelativeActionSamplingConfig,
    contract: ResolvedEEFActionContract,
) -> np.ndarray:
    start, stop = _episode_anchor_bounds(episode.length, sampling)
    anchors = np.arange(start, stop, dtype=np.int64)
    if anchors.size == 0:
        return np.empty(
            (0, sampling.horizon, contract.action_dim),
            dtype=np.float32,
        )
    state_indices = anchors + sampling.reference_delta
    action_indices = anchors[:, None] + np.asarray(
        sampling.action_delta_indices,
        dtype=np.int64,
    )[None, :]
    states = episode.states[state_indices]
    actions = episode.actions[action_indices]
    relative = absolute_actions_to_relative(
        states,
        actions,
        state_pose_indices=contract.state_pose_indices,
        action_pose_indices=contract.action_pose_indices,
    )
    return np.asarray(relative, dtype=np.float32)


def _numeric_stats_from_memmap(values: np.memmap) -> dict[str, Any]:
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError(f"relative stats values must be non-empty (N,H,D), got {values.shape}")
    _, horizon, feature_dim = values.shape
    stats_arrays = {
        "min": np.empty((horizon, feature_dim), dtype=np.float64),
        "max": np.empty((horizon, feature_dim), dtype=np.float64),
        "mean": np.empty((horizon, feature_dim), dtype=np.float64),
        "std": np.empty((horizon, feature_dim), dtype=np.float64),
        **{
            name: np.empty((horizon, feature_dim), dtype=np.float64)
            for name in RELATIVE_ACTION_STAT_QUANTILES
        },
    }
    for horizon_index in range(horizon):
        horizon_values = values[:, horizon_index, :]
        stats_arrays["min"][horizon_index] = np.min(horizon_values, axis=0)
        stats_arrays["max"][horizon_index] = np.max(horizon_values, axis=0)
        stats_arrays["mean"][horizon_index] = np.mean(
            horizon_values,
            axis=0,
            dtype=np.float64,
        )
        stats_arrays["std"][horizon_index] = np.std(
            horizon_values,
            axis=0,
            dtype=np.float64,
        )
        for name, quantile in RELATIVE_ACTION_STAT_QUANTILES.items():
            stats_arrays[name][horizon_index] = np.quantile(
                horizon_values,
                quantile,
                axis=0,
            )
    return {
        **{name: value.tolist() for name, value in stats_arrays.items()},
        "count": int(values.shape[0]),
    }


def calculate_relative_action_stats(
    dataset_root: str | Path,
    sampling: RelativeActionSamplingConfig,
    *,
    config: ActionRepresentationConfig | None = None,
    max_episodes: int | None = None,
    scratch_dir: str | Path | None = None,
) -> RelativeActionStatsResult:
    """Dataset 전체에서 한 horizon-aware relative stats profile을 계산."""
    root = Path(dataset_root).resolve()
    config = config or ActionRepresentationConfig(mode="eef_relative")
    contract = resolve_eef_action_contract(root, config)
    episodes, source_columns_sha256 = _load_episodes(
        root,
        contract,
        max_episodes=max_episodes,
    )
    window_counts = [
        _episode_anchor_bounds(episode.length, sampling)[1]
        - _episode_anchor_bounds(episode.length, sampling)[0]
        for episode in episodes
    ]
    total_windows = int(sum(window_counts))
    if total_windows <= 0:
        raise ValueError(
            "no full action windows are available for "
            f"action_delta_indices={sampling.action_delta_indices}"
        )

    temporary_parent = Path(scratch_dir).resolve() if scratch_dir is not None else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="eef-relative-stats-",
        dir=temporary_parent,
    ) as directory:
        memmap_path = Path(directory) / "relative_actions.f32"
        values = np.memmap(
            memmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, sampling.horizon, contract.action_dim),
        )
        offset = 0
        for episode, expected_count in zip(episodes, window_counts, strict=True):
            relative = _relative_episode_windows(episode, sampling, contract)
            if relative.shape[0] != expected_count:
                raise AssertionError(
                    f"episode {episode.episode_index} window count mismatch: "
                    f"{relative.shape[0]} != {expected_count}"
                )
            values[offset : offset + expected_count] = relative
            offset += expected_count
        values.flush()
        action_stats = _numeric_stats_from_memmap(values)
        del values

    total_frames = int(sum(episode.length for episode in episodes))
    production = max_episodes is None
    dataset_contract = {
        "contract_fingerprint": contract.fingerprint,
        "source_columns_sha256": source_columns_sha256,
        "info_sha256": contract.info_sha256,
        "modality_sha256": contract.modality_sha256,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "max_episodes": max_episodes,
        "production": production,
    }
    profile = {
        "production": production,
        "dataset_contract": dataset_contract,
        "transform": {
            "version": EEF_RELATIVE_ACTION_VERSION,
            "mode": config.mode,
            "pose_format": config.pose_format,
            "base_frame": contract.base_frame,
            "eef_frame": contract.eef_frame,
            "state_pose_indices": list(contract.state_pose_indices),
            "action_pose_indices": list(contract.action_pose_indices),
            "passthrough_action_indices": list(contract.passthrough_action_indices),
        },
        "sampling": sampling.to_dict(),
        "stats": {"action": action_stats},
    }
    profile_id = _stats_profile_id(profile)
    return RelativeActionStatsResult(
        dataset_contract=dataset_contract,
        profile_id=profile_id,
        profile=profile,
    )


def _read_stats_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": RELATIVE_ACTION_STATS_SCHEMA_VERSION,
            "generator_version": RELATIVE_ACTION_STATS_GENERATOR_VERSION,
            "dataset_contract": {},
            "profiles": {},
        }
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid relative action stats JSON {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise TypeError(f"relative action stats root must be an object: {path}")
    if artifact.get("schema_version") != RELATIVE_ACTION_STATS_SCHEMA_VERSION:
        raise ValueError(
            f"relative action stats schema mismatch: {artifact.get('schema_version')!r}"
        )
    if artifact.get("generator_version") != RELATIVE_ACTION_STATS_GENERATOR_VERSION:
        raise ValueError(
            f"relative action stats generator mismatch: {artifact.get('generator_version')!r}"
        )
    if not isinstance(artifact.get("profiles"), dict):
        raise TypeError("relative action stats 'profiles' must be an object")
    return artifact


def validate_relative_action_stats_artifact(
    artifact: dict[str, Any],
    *,
    require_production: bool = True,
) -> None:
    """저장 전/학습 로드 시 schema, shape, finite 값을 검증."""
    if artifact.get("schema_version") != RELATIVE_ACTION_STATS_SCHEMA_VERSION:
        raise ValueError("relative action stats schema_version mismatch")
    profiles = artifact.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("relative action stats has no profiles")
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id.startswith("sha256:"):
            raise ValueError(f"invalid relative stats profile id: {profile_id!r}")
        if not isinstance(profile, dict):
            raise TypeError(f"relative stats profile must be an object: {profile_id}")
        expected_profile_id = _stats_profile_id(profile)
        if profile_id != expected_profile_id:
            raise ValueError(
                f"relative stats profile content hash mismatch: "
                f"{profile_id} != {expected_profile_id}"
            )
        if require_production and profile.get("production") is not True:
            raise ValueError(f"non-production relative stats profile is not allowed: {profile_id}")
        dataset_contract = profile.get("dataset_contract")
        transform = profile.get("transform")
        sampling = profile.get("sampling")
        stats = profile.get("stats", {}).get("action")
        if (
            not isinstance(dataset_contract, dict)
            or not isinstance(transform, dict)
            or not isinstance(sampling, dict)
            or not isinstance(stats, dict)
        ):
            raise ValueError(f"profile contract/transform/sampling/action stats is missing: {profile_id}")
        if transform.get("version") != EEF_RELATIVE_ACTION_VERSION:
            raise ValueError(f"relative transform version mismatch in profile: {profile_id}")
        if transform.get("mode") != "eef_relative":
            raise ValueError(f"relative transform mode mismatch in profile: {profile_id}")
        horizon = sampling.get("horizon")
        action_dim = len(profile.get("transform", {}).get("action_pose_indices", [])) + len(
            profile.get("transform", {}).get("passthrough_action_indices", [])
        )
        if not isinstance(horizon, int) or horizon <= 0 or action_dim <= 0:
            raise ValueError(f"invalid horizon/action dim in profile: {profile_id}")
        for stat_name in ("min", "max", "mean", "std", *RELATIVE_ACTION_STAT_QUANTILES):
            values = np.asarray(stats.get(stat_name), dtype=np.float64)
            if values.shape != (horizon, action_dim):
                raise ValueError(
                    f"{profile_id} {stat_name} shape mismatch: "
                    f"{values.shape} != {(horizon, action_dim)}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{profile_id} {stat_name} contains NaN or infinity")
        count = stats.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{profile_id} count must be a positive integer, got {count!r}")


def load_relative_action_stats_profile(
    dataset_root: str | Path,
    contract: ResolvedEEFActionContract,
    sampling: RelativeActionSamplingConfig,
    *,
    stats_file: str | None = None,
    require_production: bool = True,
    verify_source_columns: bool = True,
) -> tuple[str, dict[str, Any]]:
    """학습 sampler와 정확히 일치하는 relative stats profile 하나를 로드.

    ``verify_source_columns=True``이면 parquet의 state/action checksum을 다시 계산해
    통계 생성 뒤 dataset이 수정되지 않았는지도 검사한다.
    """
    root = Path(dataset_root).resolve()
    relative_path = Path(stats_file or contract.config.stats_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"stats_file must be a dataset-relative safe path: {relative_path}")
    artifact = _read_stats_artifact(root / relative_path)
    validate_relative_action_stats_artifact(
        artifact,
        require_production=require_production,
    )

    expected_sampling = sampling.to_dict()
    matches: list[tuple[str, dict[str, Any]]] = []
    for profile_id, profile in artifact["profiles"].items():
        dataset_contract = profile["dataset_contract"]
        transform = profile["transform"]
        if dataset_contract.get("contract_fingerprint") != contract.fingerprint:
            continue
        if profile["sampling"] != expected_sampling:
            continue
        if (
            transform.get("version") != EEF_RELATIVE_ACTION_VERSION
            or transform.get("pose_format") != contract.config.pose_format
            or transform.get("base_frame") != contract.base_frame
            or transform.get("eef_frame") != contract.eef_frame
            or transform.get("state_pose_indices") != list(contract.state_pose_indices)
            or transform.get("action_pose_indices") != list(contract.action_pose_indices)
            or transform.get("passthrough_action_indices")
            != list(contract.passthrough_action_indices)
        ):
            continue
        if require_production and profile.get("production") is not True:
            continue
        matches.append((profile_id, profile))

    if not matches:
        raise KeyError(
            "no relative action stats profile matches the dataset contract and sampler: "
            f"contract={contract.fingerprint}, sampling={expected_sampling}"
        )
    if len(matches) != 1:
        raise ValueError(
            "multiple relative action stats profiles match the same contract and sampler: "
            f"{[profile_id for profile_id, _ in matches]}"
        )
    profile_id, profile = matches[0]
    profile_contract = profile["dataset_contract"]
    if profile_contract.get("info_sha256") != contract.info_sha256:
        raise ValueError("relative stats info.json checksum mismatch")
    if profile_contract.get("modality_sha256") != contract.modality_sha256:
        raise ValueError("relative stats modality.json checksum mismatch")
    if verify_source_columns:
        max_episodes = profile_contract.get("max_episodes")
        _, source_columns_sha256 = _load_episodes(
            root,
            contract,
            max_episodes=max_episodes,
        )
        if source_columns_sha256 != profile_contract.get("source_columns_sha256"):
            raise ValueError(
                "relative stats source state/action checksum mismatch; "
                "regenerate the stats artifact"
            )
    return profile_id, deepcopy(profile)


def inject_relative_action_stats(
    dataset_stats: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """기존 observation/image stats를 보존하고 action만 relative stats로 교체."""
    validate_relative_action_stats_artifact(
        {
            "schema_version": RELATIVE_ACTION_STATS_SCHEMA_VERSION,
            "generator_version": RELATIVE_ACTION_STATS_GENERATOR_VERSION,
            "profiles": {_stats_profile_id(profile): profile},
        },
        require_production=profile.get("production") is True,
    )
    merged = deepcopy(dataset_stats)
    merged["action"] = deepcopy(profile["stats"]["action"])
    return merged


def write_relative_action_stats_profile(
    dataset_root: str | Path,
    result: RelativeActionStatsResult,
    *,
    output_file: str = RELATIVE_ACTION_STATS_DEFAULT_PATH,
    overwrite_profile: bool = False,
) -> tuple[Path, bool]:
    """계산 결과를 multi-profile artifact에 atomic update.

    Returns:
        ``(path, changed)``. 동일 profile이 이미 있으면 ``changed=False``다.
    """
    root = Path(dataset_root).resolve()
    relative_path = Path(output_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"output_file must be a dataset-relative safe path: {output_file!r}")
    output_path = root / relative_path
    if not result.profile.get("production", False) and relative_path == Path(
        RELATIVE_ACTION_STATS_DEFAULT_PATH
    ):
        raise ValueError(
            "debug/subset stats cannot be written to the production default path; "
            "choose a separate --output-file"
        )

    artifact = _read_stats_artifact(output_path)
    existing_contract = artifact.get("dataset_contract")
    if existing_contract and existing_contract != result.dataset_contract:
        raise ValueError(
            "existing relative stats artifact belongs to a different dataset contract; "
            "remove it explicitly or choose another output file"
        )
    artifact["dataset_contract"] = result.dataset_contract
    profiles = artifact["profiles"]
    existing_profile = profiles.get(result.profile_id)
    if existing_profile is not None and not overwrite_profile:
        if existing_profile != result.profile:
            raise ValueError(
                f"profile id collision with different content: {result.profile_id}"
            )
        return output_path, False
    profiles[result.profile_id] = result.profile
    validate_relative_action_stats_artifact(
        artifact,
        require_production=result.profile.get("production", False),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, output_path)
    return output_path, True
