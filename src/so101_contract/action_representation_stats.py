"""Phase 13–14 — 8개 representation 공통 horizon-aware action stats.

v1 :mod:`so101_contract.eef_relative_stats`는 ``eef_relative`` 전용이었다. 이 모듈은
같은 window/quantile 규칙을 4 mode × 3 pose format으로 확장하고, **absolute와 relative
profile이 하나의 artifact에 공존**하게 한다.

처리 순서(MUST):

.. code-block:: text

    absolute dataset action
      → ActionRepresentationTransform.encode (relative 변환 또는 canonical 정규화)
      → horizon별 stats

즉 relative stats는 변환 **후**, absolute stats는 canonical absolute action에서 계산한다.
quaternion 부호/시간축 연속성과 RPY wrap은 ``encode``가 담당하므로 stats 계산 전에 이미
적용돼 있다.

profile ID는 다음을 모두 포함한 canonical JSON의 SHA-256이다.

- stats kind(mode + pose format)와 transform fingerprint
- action horizon과 sampling delta indices
- dataset fingerprint와 source column checksum
- generator/transform version

따라서 mode, format, horizon, dataset 중 하나만 달라져도 cache가 무효화된다.
checkpoint에는 :func:`serialize_stats_for_processor` 결과를 저장해 dataset 없이 stats를
복원한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .action_representation import ActionRepresentationSpec
from .action_transform import ACTION_TRANSFORM_VERSION, ActionRepresentationTransform

ACTION_STATS_SCHEMA_VERSION = 2
ACTION_STATS_GENERATOR_VERSION = "so101_action_representation_stats_v2"
ACTION_STATS_DEFAULT_PATH = "meta/action_representation_stats.json"
ACTION_STAT_QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
ACTION_STAT_NAMES = ("min", "max", "mean", "std", *ACTION_STAT_QUANTILES)


@dataclass(frozen=True)
class ActionStatsSampling:
    """Training sampler와 정확히 일치해야 하는 stats window 계약."""

    observation_delta_indices: tuple[int, ...] = (0,)
    action_delta_indices: tuple[int, ...] = (0,)
    reference_observation_index: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_delta_indices",
            tuple(int(value) for value in self.observation_delta_indices),
        )
        object.__setattr__(
            self,
            "action_delta_indices",
            tuple(int(value) for value in self.action_delta_indices),
        )
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
        if self.reference_delta != 0:
            raise ValueError(
                f"schema v2 requires current observation delta 0 as reference, got "
                f"{self.reference_delta}"
            )
        if self.action_delta_indices[0] != 0:
            raise ValueError(
                f"schema v2 requires action_delta_indices[0] == 0, got "
                f"{self.action_delta_indices[0]}"
            )
        if any(index < 0 for index in self.action_delta_indices):
            raise ValueError(
                f"action_delta_indices must be non-negative, got {self.action_delta_indices}"
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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionStatsSampling:
        return cls(
            observation_delta_indices=tuple(payload["observation_delta_indices"]),
            action_delta_indices=tuple(payload["action_delta_indices"]),
            reference_observation_index=int(payload.get("reference_observation_index", -1)),
        )


@dataclass(frozen=True)
class EpisodeArrays:
    """한 episode의 absolute state/action. episode boundary는 넘지 않는다."""

    episode_index: int
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"episode {self.episode_index} state/action must be 2D, got "
                f"{states.shape}/{actions.shape}"
            )
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"episode {self.episode_index} state/action length mismatch: "
                f"{states.shape[0]} != {actions.shape[0]}"
            )
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"episode {self.episode_index} contains NaN or infinity")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

    @property
    def length(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class ActionStatsResult:
    profile_id: str
    profile: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def stats_profile_id(profile: dict[str, Any]) -> str:
    """Profile 계약과 수치 통계를 함께 묶는 content-addressed ID."""
    payload = {
        "generator_version": ACTION_STATS_GENERATOR_VERSION,
        "production": profile.get("production"),
        "kind": profile.get("kind"),
        "transform": profile.get("transform"),
        "dataset": profile.get("dataset"),
        "sampling": profile.get("sampling"),
        "stats": profile.get("stats"),
    }
    return f"sha256:{_sha256_json(payload)}"


def _episode_anchor_bounds(episode_length: int, sampling: ActionStatsSampling) -> tuple[int, int]:
    required_deltas = (sampling.reference_delta, *sampling.action_delta_indices)
    start = max(0, -min(required_deltas))
    stop = episode_length - max(required_deltas)
    return start, max(start, stop)


def _episode_windows(
    episode: EpisodeArrays,
    sampling: ActionStatsSampling,
    transform: ActionRepresentationTransform,
) -> np.ndarray:
    start, stop = _episode_anchor_bounds(episode.length, sampling)
    anchors = np.arange(start, stop, dtype=np.int64)
    if anchors.size == 0:
        return np.empty((0, sampling.horizon, transform.action_dim), dtype=np.float32)
    state_indices = anchors + sampling.reference_delta
    action_indices = anchors[:, None] + np.asarray(
        sampling.action_delta_indices,
        dtype=np.int64,
    )[None, :]
    targets = transform.encode(
        episode.states[state_indices],
        episode.actions[action_indices],
    )
    return np.asarray(targets, dtype=np.float32)


def _numeric_stats(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError(f"action stats values must be non-empty (N,H,D), got {values.shape}")
    _, horizon, feature_dim = values.shape
    arrays = {
        name: np.empty((horizon, feature_dim), dtype=np.float64) for name in ACTION_STAT_NAMES
    }
    for index in range(horizon):
        window = values[:, index, :]
        arrays["min"][index] = np.min(window, axis=0)
        arrays["max"][index] = np.max(window, axis=0)
        arrays["mean"][index] = np.mean(window, axis=0, dtype=np.float64)
        arrays["std"][index] = np.std(window, axis=0, dtype=np.float64)
        for name, quantile in ACTION_STAT_QUANTILES.items():
            arrays[name][index] = np.quantile(window, quantile, axis=0)
    return {
        **{name: value.tolist() for name, value in arrays.items()},
        "count": int(values.shape[0]),
    }


def load_lerobot_v3_episodes(
    dataset_root: str | Path,
    *,
    state_key: str = "observation.state",
    action_key: str = "action",
    state_dim: int | None = None,
    action_dim: int | None = None,
    max_episodes: int | None = None,
) -> list[EpisodeArrays]:
    """LeRobot v3 parquet에서 episode별 absolute state/action을 읽는다.

    episode boundary를 보존하기 위해 ``episode_index``로 나누고 ``frame_index``로
    정렬하며, frame index가 ``[0, N)`` 연속이 아니면 거부한다. 여러 parquet 조각에
    흩어진 같은 episode도 하나로 합친다.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 런타임 의존성
        raise ImportError("action stats generation requires pyarrow") from exc

    root = Path(dataset_root)
    data_paths = sorted((root / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data parquet under {root / 'data'}")

    required = ("episode_index", "frame_index", state_key, action_key)
    parts: dict[int, dict[str, list[np.ndarray]]] = {}
    for path in data_paths:
        table = pq.read_table(path, columns=list(required))
        missing = [name for name in required if name not in table.column_names]
        if missing:
            raise KeyError(f"{path}: missing required columns {missing}")
        episode_indices = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        states = np.asarray(table.column(state_key).to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column(action_key).to_pylist(), dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"{path}: state/action columns must be 1-D vectors per frame")
        if state_dim is not None and states.shape[1] != state_dim:
            raise ValueError(f"{path}: state dim {states.shape[1]} != contract {state_dim}")
        if action_dim is not None and actions.shape[1] != action_dim:
            raise ValueError(f"{path}: action dim {actions.shape[1]} != contract {action_dim}")
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

    selected = sorted(parts)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive, got {max_episodes}")
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("dataset contains no selected episodes")

    episodes: list[EpisodeArrays] = []
    for episode_index in selected:
        chunk = parts[episode_index]
        frames = np.concatenate(chunk["frame"]).astype(np.int64, copy=False)
        states = np.concatenate(chunk["state"]).astype(np.float32, copy=False)
        actions = np.concatenate(chunk["action"]).astype(np.float32, copy=False)
        order = np.argsort(frames, kind="stable")
        frames = frames[order]
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            raise ValueError(
                f"episode {episode_index} frame_index must be unique contiguous [0,N), "
                f"got {frames.tolist()[:16]}"
            )
        episodes.append(
            EpisodeArrays(
                episode_index=episode_index,
                states=states[order],
                actions=actions[order],
            )
        )
    return episodes


def source_columns_sha256(episodes: Sequence[EpisodeArrays]) -> str:
    """Stats 생성 이후 dataset이 바뀌었는지 확인하기 위한 column checksum."""
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(
            np.asarray([episode.episode_index, episode.length], dtype="<i8").tobytes()
        )
        digest.update(np.asarray(episode.states, dtype="<f4", order="C").tobytes())
        digest.update(np.asarray(episode.actions, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def calculate_action_representation_stats(
    episodes: Sequence[EpisodeArrays],
    sampling: ActionStatsSampling,
    transform: ActionRepresentationTransform,
    *,
    dataset_fingerprint: str,
    dataset: dict[str, Any] | None = None,
    max_episodes: int | None = None,
    production: bool = True,
) -> ActionStatsResult:
    """한 representation의 horizon-aware stats profile을 계산.

    dataset 로딩은 호출자가 담당한다(Phase 15에서 LeRobot dataset loader와 연결).
    """
    if not episodes:
        raise ValueError("action stats require at least one episode")
    for episode in episodes:
        if episode.states.shape[1] != transform.state_dim:
            raise ValueError(
                f"episode {episode.episode_index} state dim {episode.states.shape[1]} != "
                f"contract {transform.state_dim}"
            )
        if episode.actions.shape[1] != transform.action_dim:
            raise ValueError(
                f"episode {episode.episode_index} action dim {episode.actions.shape[1]} != "
                f"contract {transform.action_dim}"
            )

    windows = [_episode_windows(episode, sampling, transform) for episode in episodes]
    total = int(sum(window.shape[0] for window in windows))
    if total <= 0:
        raise ValueError(
            "no full action windows are available for "
            f"action_delta_indices={sampling.action_delta_indices}"
        )
    values = np.concatenate([window for window in windows if window.shape[0]], axis=0)
    action_stats = _numeric_stats(values)

    dataset_section = {
        "fingerprint": dataset_fingerprint,
        "source_columns_sha256": source_columns_sha256(episodes),
        "total_episodes": len(episodes),
        "total_frames": int(sum(episode.length for episode in episodes)),
        "max_episodes": max_episodes,
        **(dataset or {}),
    }
    profile = {
        "production": bool(production),
        "kind": transform.spec.stats_profile_kind,
        "transform": {
            "version": ACTION_TRANSFORM_VERSION,
            "fingerprint": transform.fingerprint(),
            "mode": transform.spec.mode.value,
            "pose_format": transform.spec.pose_format.value,
            "state_indices": list(transform.state_indices),
            "action_indices": list(transform.action_indices),
            "passthrough_action_indices": list(transform.passthrough_action_indices),
            "joint_topology_fingerprint": (
                transform.joint_topology.fingerprint()
                if transform.joint_topology is not None
                else None
            ),
        },
        "dataset": dataset_section,
        "sampling": sampling.to_dict(),
        "stats": {"action": action_stats},
    }
    return ActionStatsResult(profile_id=stats_profile_id(profile), profile=profile)


# --- artifact I/O -------------------------------------------------------------


def empty_stats_artifact() -> dict[str, Any]:
    return {
        "schema_version": ACTION_STATS_SCHEMA_VERSION,
        "generator_version": ACTION_STATS_GENERATOR_VERSION,
        "profiles": {},
    }


def validate_action_stats_artifact(
    artifact: dict[str, Any],
    *,
    require_production: bool = True,
) -> None:
    """Schema, content hash, shape, finite 값을 검증."""
    if not isinstance(artifact, dict):
        raise TypeError("action stats artifact must be a JSON object")
    if artifact.get("schema_version") != ACTION_STATS_SCHEMA_VERSION:
        raise ValueError(
            f"action stats schema_version must be {ACTION_STATS_SCHEMA_VERSION}, "
            f"got {artifact.get('schema_version')!r}"
        )
    if artifact.get("generator_version") != ACTION_STATS_GENERATOR_VERSION:
        raise ValueError(
            f"action stats generator mismatch: {artifact.get('generator_version')!r}"
        )
    profiles = artifact.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("action stats artifact has no profiles")

    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id.startswith("sha256:"):
            raise ValueError(f"invalid action stats profile id: {profile_id!r}")
        if not isinstance(profile, dict):
            raise TypeError(f"action stats profile must be an object: {profile_id}")
        if stats_profile_id(profile) != profile_id:
            raise ValueError(f"action stats profile content hash mismatch: {profile_id}")
        if require_production and profile.get("production") is not True:
            raise ValueError(f"non-production action stats profile is not allowed: {profile_id}")

        transform = profile.get("transform")
        sampling = profile.get("sampling")
        dataset = profile.get("dataset")
        stats = profile.get("stats", {}).get("action")
        if not all(isinstance(value, dict) for value in (transform, sampling, dataset, stats)):
            raise ValueError(f"profile transform/sampling/dataset/stats is missing: {profile_id}")
        if transform.get("version") != ACTION_TRANSFORM_VERSION:
            raise ValueError(f"action transform version mismatch in profile: {profile_id}")
        action_dim = len(transform.get("action_indices", [])) + len(
            transform.get("passthrough_action_indices", [])
        )
        horizon = sampling.get("horizon")
        if not isinstance(horizon, int) or horizon <= 0 or action_dim <= 0:
            raise ValueError(f"invalid horizon/action dim in profile: {profile_id}")
        for name in ACTION_STAT_NAMES:
            values = np.asarray(stats.get(name), dtype=np.float64)
            if values.shape != (horizon, action_dim):
                raise ValueError(
                    f"{profile_id} {name} shape mismatch: {values.shape} != "
                    f"{(horizon, action_dim)}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{profile_id} {name} contains NaN or infinity")
        count = stats.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{profile_id} count must be a positive integer, got {count!r}")


def upsert_stats_profile(
    artifact: dict[str, Any],
    result: ActionStatsResult,
    *,
    overwrite: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Artifact에 profile을 추가. absolute/relative profile이 함께 공존한다.

    Returns:
        ``(artifact, changed)``
    """
    updated = deepcopy(artifact)
    if updated.get("schema_version") != ACTION_STATS_SCHEMA_VERSION:
        raise ValueError("cannot upsert into an artifact with a different schema version")
    profiles = updated.setdefault("profiles", {})
    existing = profiles.get(result.profile_id)
    if existing is not None and not overwrite:
        if existing != result.profile:
            raise ValueError(f"profile id collision with different content: {result.profile_id}")
        return updated, False
    profiles[result.profile_id] = deepcopy(result.profile)
    validate_action_stats_artifact(
        updated,
        require_production=all(
            profile.get("production") is True for profile in profiles.values()
        ),
    )
    return updated, True


def write_action_stats_artifact(
    dataset_root: str | Path,
    artifact: dict[str, Any],
    *,
    output_file: str = ACTION_STATS_DEFAULT_PATH,
) -> Path:
    """Artifact를 atomic하게 저장."""
    relative_path = Path(output_file)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"output_file must be a dataset-relative safe path: {output_file!r}")
    validate_action_stats_artifact(
        artifact,
        require_production=all(
            profile.get("production") is True
            for profile in artifact.get("profiles", {}).values()
        ),
    )
    output_path = Path(dataset_root) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def read_action_stats_artifact(
    dataset_root: str | Path,
    *,
    output_file: str = ACTION_STATS_DEFAULT_PATH,
) -> dict[str, Any]:
    path = Path(dataset_root) / output_file
    if not path.is_file():
        return empty_stats_artifact()
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid action stats JSON {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise TypeError(f"action stats root must be an object: {path}")
    return artifact


def select_stats_profile(
    artifact: dict[str, Any],
    transform: ActionRepresentationTransform,
    sampling: ActionStatsSampling,
    *,
    dataset_fingerprint: str,
    require_production: bool = True,
) -> tuple[str, dict[str, Any]]:
    """mode/format/horizon/dataset이 정확히 일치하는 profile 하나를 고른다.

    하나라도 다르면 cache miss이며 추정 대체를 하지 않는다.
    """
    validate_action_stats_artifact(artifact, require_production=False)
    expected_sampling = sampling.to_dict()
    expected_fingerprint = transform.fingerprint()
    matches: list[tuple[str, dict[str, Any]]] = []
    for profile_id, profile in artifact["profiles"].items():
        if profile.get("kind") != transform.spec.stats_profile_kind:
            continue
        if profile["transform"].get("fingerprint") != expected_fingerprint:
            continue
        if profile["sampling"] != expected_sampling:
            continue
        if profile["dataset"].get("fingerprint") != dataset_fingerprint:
            continue
        if require_production and profile.get("production") is not True:
            continue
        matches.append((profile_id, profile))

    if not matches:
        raise KeyError(
            "no action stats profile matches the transform/sampler/dataset: "
            f"kind={transform.spec.stats_profile_kind}, horizon={sampling.horizon}, "
            f"dataset={dataset_fingerprint}"
        )
    if len(matches) != 1:
        raise ValueError(
            "multiple action stats profiles match the same key: "
            f"{[profile_id for profile_id, _ in matches]}"
        )
    profile_id, profile = matches[0]
    return profile_id, deepcopy(profile)


def inject_action_stats(
    dataset_stats: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """observation/image stats는 보존하고 action stats만 교체."""
    if not isinstance(profile.get("stats", {}).get("action"), dict):
        raise ValueError("stats profile has no action stats")
    merged = deepcopy(dataset_stats)
    merged["action"] = deepcopy(profile["stats"]["action"])
    return merged


# --- checkpoint 복원 ----------------------------------------------------------


def serialize_stats_for_processor(profile: dict[str, Any]) -> dict[str, Any]:
    """Processor config에 넣어 dataset 없이 복원 가능한 stats payload."""
    profile_id = stats_profile_id(profile)
    return {
        "schema_version": ACTION_STATS_SCHEMA_VERSION,
        "generator_version": ACTION_STATS_GENERATOR_VERSION,
        "profile_id": profile_id,
        "content_sha256": profile_id.removeprefix("sha256:"),
        "profile": deepcopy(profile),
    }


def restore_stats_from_processor(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Checkpoint processor payload에서 stats profile을 복원하고 hash를 재검증."""
    if not isinstance(payload, dict):
        raise TypeError("serialized stats payload must be an object")
    if payload.get("schema_version") != ACTION_STATS_SCHEMA_VERSION:
        raise ValueError("serialized stats schema_version mismatch")
    if payload.get("generator_version") != ACTION_STATS_GENERATOR_VERSION:
        raise ValueError("serialized stats generator_version mismatch")
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("serialized stats payload has no profile")
    profile_id = stats_profile_id(profile)
    if payload.get("profile_id") != profile_id:
        raise ValueError("serialized stats profile hash mismatch (tampered checkpoint stats)")
    if payload.get("content_sha256") != profile_id.removeprefix("sha256:"):
        raise ValueError("serialized stats content hash mismatch")
    validate_action_stats_artifact(
        {
            "schema_version": ACTION_STATS_SCHEMA_VERSION,
            "generator_version": ACTION_STATS_GENERATOR_VERSION,
            "profiles": {profile_id: profile},
        },
        require_production=profile.get("production") is True,
    )
    return profile_id, deepcopy(profile)


def stats_profile_spec(profile: dict[str, Any]) -> ActionRepresentationSpec | None:
    """Profile이 어떤 representation의 것인지 확인용 helper."""
    transform = profile.get("transform")
    if not isinstance(transform, dict) or "mode" not in transform:
        return None
    return ActionRepresentationSpec.from_dict(
        {
            "schema_version": ACTION_STATS_SCHEMA_VERSION,
            "mode": transform["mode"],
            "pose_format": transform.get("pose_format"),
        }
    )
