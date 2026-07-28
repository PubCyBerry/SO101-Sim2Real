#!/usr/bin/env python3
"""Phase 13–14 — 8개 representation 공통 action stats 검증.

확인 항목:

- 8개 (mode, pose format) profile이 **하나의 artifact에 공존**
- mode/format/horizon/dataset fingerprint 중 하나만 달라도 cache invalidation
- relative stats는 변환 후, absolute stats는 canonical absolute action에서 계산
- quaternion 부호 정규화·시간축 연속성과 RPY wrap이 **stats 계산 전에** 적용
- episode boundary를 넘는 window 제외
- checkpoint processor payload에서 dataset 없이 stats 복원과 tamper 거부

.. code-block:: bash

    python scripts/contract/validate_action_representation_stats.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    iter_representation_specs,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ACTION_STATS_DEFAULT_PATH,
    ACTION_STATS_SCHEMA_VERSION,
    ActionStatsSampling,
    EpisodeArrays,
    calculate_action_representation_stats,
    empty_stats_artifact,
    inject_action_stats,
    read_action_stats_artifact,
    restore_stats_from_processor,
    select_stats_profile,
    serialize_stats_for_processor,
    stats_profile_id,
    upsert_stats_profile,
    validate_action_stats_artifact,
    write_action_stats_artifact,
)
from so101_contract.action_transform import ActionRepresentationTransform  # noqa: E402
from so101_contract.joint_topology import TAU, JointSpec, JointTopology, JointType  # noqa: E402
from so101_contract.pose_codec import encode_pose  # noqa: E402

_DATASET_FINGERPRINT = "c" * 64
_ARM_JOINT_NAMES = tuple(f"arm.joint_{index}" for index in range(5))


def _joint_topology() -> JointTopology:
    return JointTopology(
        joints=tuple(
            JointSpec(name, JointType.REVOLUTE, period=TAU, lower=-math.pi, upper=math.pi)
            for name in _ARM_JOINT_NAMES
        )
    )


def _transform(spec: ActionRepresentationSpec) -> ActionRepresentationTransform:
    if spec.is_eef:
        pose_dim = spec.pose_dim
        return ActionRepresentationTransform(
            spec=spec,
            state_indices=tuple(range(pose_dim)),
            action_indices=tuple(range(pose_dim)),
            passthrough_action_indices=(pose_dim,),
            state_dim=pose_dim + 1,
            action_dim=pose_dim + 1,
        )
    topology = _joint_topology()
    return ActionRepresentationTransform(
        spec=spec,
        state_indices=tuple(range(topology.dim)),
        action_indices=tuple(range(topology.dim)),
        passthrough_action_indices=(topology.dim,),
        state_dim=topology.dim + 1,
        action_dim=topology.dim + 1,
        joint_topology=topology,
    )


def _random_rotations(count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrices = np.empty((count, 3, 3), dtype=np.float64)
    for index in range(count):
        q, r = np.linalg.qr(generator.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        matrices[index] = q
    return matrices


def _episode(spec: ActionRepresentationSpec, *, length: int, seed: int) -> EpisodeArrays:
    generator = np.random.default_rng(seed)
    if spec.is_eef:
        rotations = _random_rotations(length + 1, seed + 1)
        translations = np.cumsum(generator.normal(scale=0.01, size=(length + 1, 3)), axis=0)
        poses = encode_pose(translations, rotations, spec.pose_format)
        gripper = generator.uniform(0.0, 100.0, size=(length + 1, 1))
        columns = np.concatenate([poses, gripper], axis=-1)
    else:
        joints = np.cumsum(generator.normal(scale=0.05, size=(length + 1, 5)), axis=0)
        gripper = generator.uniform(0.0, 100.0, size=(length + 1, 1))
        columns = np.concatenate([joints, gripper], axis=-1)
    # action[t]는 다음 step의 absolute target이다.
    return EpisodeArrays(
        episode_index=seed,
        states=columns[:-1].astype(np.float32),
        actions=columns[1:].astype(np.float32),
    )


def _episodes(spec: ActionRepresentationSpec) -> list[EpisodeArrays]:
    return [
        _episode(spec, length=24, seed=1),
        _episode(spec, length=17, seed=2),
    ]


def check_all_representations_coexist() -> None:
    """8개 profile이 한 artifact에 공존하고 각각 정확히 선택된다."""
    sampling = ActionStatsSampling(action_delta_indices=tuple(range(4)))
    artifact = empty_stats_artifact()
    entries: list[tuple[ActionRepresentationSpec, ActionRepresentationTransform, str]] = []
    for spec in iter_representation_specs():
        transform = _transform(spec)
        result = calculate_action_representation_stats(
            _episodes(spec),
            sampling,
            transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        )
        artifact, changed = upsert_stats_profile(artifact, result)
        if not changed:
            raise AssertionError(f"{spec.stats_profile_kind} profile was not inserted")
        entries.append((spec, transform, result.profile_id))

    if len(artifact["profiles"]) != 8:
        raise AssertionError(f"expected 8 coexisting profiles, got {len(artifact['profiles'])}")
    kinds = {profile["kind"] for profile in artifact["profiles"].values()}
    if len(kinds) != 8:
        raise AssertionError(f"profile kinds are not unique: {sorted(kinds)}")
    if not {"joint_absolute", "joint_relative"} <= kinds:
        raise AssertionError("joint absolute/relative profiles are missing")
    if not {"eef_absolute_rot6d", "eef_relative_wxyz", "eef_absolute_rpy"} <= kinds:
        raise AssertionError("EEF absolute/relative format profiles are missing")

    validate_action_stats_artifact(artifact)
    for spec, transform, profile_id in entries:
        selected_id, profile = select_stats_profile(
            artifact,
            transform,
            sampling,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        )
        if selected_id != profile_id:
            raise AssertionError(f"{spec.stats_profile_kind} selected the wrong profile")
        stats = profile["stats"]["action"]
        expected_shape = (sampling.horizon, transform.action_dim)
        if np.asarray(stats["mean"]).shape != expected_shape:
            raise AssertionError(f"{spec.stats_profile_kind} stats shape mismatch")
    print("PASS: 8 representation stats profiles coexist in one artifact and select uniquely")


def check_cache_invalidation() -> None:
    """mode/format/horizon/dataset/transform 변경이 모두 cache miss를 만든다."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_ROT6D_ROWS,
    )
    transform = _transform(spec)
    sampling = ActionStatsSampling(action_delta_indices=(0, 1, 2, 3))
    episodes = _episodes(spec)
    artifact, _ = upsert_stats_profile(
        empty_stats_artifact(),
        calculate_action_representation_stats(
            episodes,
            sampling,
            transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ),
    )

    misses = {
        "horizon": lambda: select_stats_profile(
            artifact,
            transform,
            ActionStatsSampling(action_delta_indices=(0, 1, 2)),
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ),
        "dataset": lambda: select_stats_profile(
            artifact,
            transform,
            sampling,
            dataset_fingerprint="d" * 64,
        ),
        "mode": lambda: select_stats_profile(
            artifact,
            _transform(
                ActionRepresentationSpec(
                    mode=ActionRepresentationMode.EEF_ABSOLUTE,
                    pose_format=PoseFormat.XYZ_ROT6D_ROWS,
                )
            ),
            sampling,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ),
        "pose format": lambda: select_stats_profile(
            artifact,
            _transform(
                ActionRepresentationSpec(
                    mode=ActionRepresentationMode.EEF_RELATIVE,
                    pose_format=PoseFormat.XYZ_RPY,
                )
            ),
            sampling,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ),
        "transform indices": lambda: select_stats_profile(
            artifact,
            ActionRepresentationTransform(
                spec=spec,
                state_indices=tuple(range(1, 10)),
                action_indices=tuple(range(1, 10)),
                passthrough_action_indices=(0,),
                state_dim=10,
                action_dim=10,
            ),
            sampling,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ),
    }
    for label, call in misses.items():
        try:
            call()
        except KeyError:
            continue
        raise AssertionError(f"stale stats profile was reused after a {label} change")

    # 같은 입력은 같은 profile ID를 만든다(재생성해도 중복 저장되지 않는다).
    repeated = calculate_action_representation_stats(
        episodes,
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    )
    artifact, changed = upsert_stats_profile(artifact, repeated)
    if changed or len(artifact["profiles"]) != 1:
        raise AssertionError("recomputing identical stats must be a no-op")

    # 데이터가 바뀌면 source checksum이 바뀌어 새 profile이 된다.
    mutated = list(episodes)
    mutated[0] = EpisodeArrays(
        episode_index=episodes[0].episode_index,
        states=episodes[0].states,
        actions=episodes[0].actions + np.float32(0.01),
    )
    changed_result = calculate_action_representation_stats(
        mutated,
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    )
    if changed_result.profile_id == repeated.profile_id:
        raise AssertionError("modified source columns must produce a new profile id")
    print("PASS: stats cache invalidates on mode/format/horizon/dataset/transform/source change")


def check_transform_ordering() -> None:
    """relative stats는 변환 후, absolute stats는 canonical absolute에서 계산된다."""
    sampling = ActionStatsSampling(action_delta_indices=(0, 1, 2))
    for pose_format in (PoseFormat.XYZ_ROT6D_ROWS, PoseFormat.XYZ_RPY):
        relative_spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=pose_format,
        )
        absolute_spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_ABSOLUTE,
            pose_format=pose_format,
        )
        episodes = _episodes(relative_spec)
        relative_transform = _transform(relative_spec)
        absolute_transform = _transform(absolute_spec)
        relative = calculate_action_representation_stats(
            episodes,
            sampling,
            relative_transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ).profile["stats"]["action"]
        absolute = calculate_action_representation_stats(
            episodes,
            sampling,
            absolute_transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        ).profile["stats"]["action"]

        # relative translation은 state 기준이라 absolute 좌표보다 훨씬 작다.
        relative_scale = float(np.max(np.abs(np.asarray(relative["mean"])[:, :3])))
        absolute_scale = float(np.max(np.abs(np.asarray(absolute["mean"])[:, :3])))
        if relative_scale >= absolute_scale:
            raise AssertionError(
                f"{pose_format.value} relative stats do not look state-anchored: "
                f"{relative_scale} >= {absolute_scale}"
            )
        # gripper(passthrough)는 두 mode에서 동일한 absolute 통계여야 한다.
        if not np.allclose(
            np.asarray(relative["mean"])[:, -1],
            np.asarray(absolute["mean"])[:, -1],
            atol=1e-6,
        ):
            raise AssertionError("gripper passthrough stats differ between modes")

        # stats는 transform.encode 결과를 그대로 집계한다(수동 재계산과 일치).
        windows = []
        for episode in episodes:
            for anchor in range(episode.length - sampling.horizon + 1):
                targets = relative_transform.encode(
                    episode.states[anchor],
                    episode.actions[anchor : anchor + sampling.horizon],
                )
                windows.append(np.asarray(targets, dtype=np.float32))
        manual_mean = np.mean(np.stack(windows), axis=0, dtype=np.float64)
        if not np.allclose(manual_mean, np.asarray(relative["mean"]), atol=1e-6):
            raise AssertionError(f"{pose_format.value} stats disagree with transform.encode")
        if relative["count"] != len(windows):
            raise AssertionError("episode-boundary window count mismatch")
    print("PASS: relative stats computed after the transform, absolute from canonical actions")


def check_normalization_before_stats() -> None:
    """quaternion 부호/연속성과 RPY wrap이 stats 계산 전에 적용된다."""
    sampling = ActionStatsSampling(action_delta_indices=(0, 1, 2, 3))

    quaternion_spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_ABSOLUTE,
        pose_format=PoseFormat.XYZ_QUATERNION_WXYZ,
    )
    transform = _transform(quaternion_spec)
    canonical = _episodes(quaternion_spec)
    flipped = []
    for episode in canonical:
        actions = episode.actions.copy()
        states = episode.states.copy()
        # 시간 축을 따라 부호를 번갈아 뒤집는다: 같은 회전, 다른 벡터.
        actions[::2, 3:7] *= -1.0
        states[::2, 3:7] *= -1.0
        flipped.append(
            EpisodeArrays(
                episode_index=episode.episode_index,
                states=states,
                actions=actions,
            )
        )
    canonical_stats = calculate_action_representation_stats(
        canonical,
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    ).profile["stats"]["action"]
    flipped_stats = calculate_action_representation_stats(
        flipped,
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    ).profile["stats"]["action"]
    raw_difference = max(
        float(np.max(np.abs(left.actions[:, 3:7] - right.actions[:, 3:7])))
        for left, right in zip(canonical, flipped, strict=True)
    )
    if not np.allclose(
        np.asarray(canonical_stats["mean"]),
        np.asarray(flipped_stats["mean"]),
        atol=1e-6,
    ):
        raise AssertionError(
            "quaternion sign flips changed the stats; canonicalization did not run first"
        )
    if raw_difference <= 0.5:
        raise AssertionError("quaternion fixture does not actually flip signs")

    rpy_spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_ABSOLUTE,
        pose_format=PoseFormat.XYZ_RPY,
    )
    rpy_transform = _transform(rpy_spec)
    base = _episodes(rpy_spec)
    unwrapped = [
        EpisodeArrays(
            episode_index=episode.episode_index,
            states=episode.states + np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * math.pi, 0.0],
                dtype=np.float32,
            ),
            actions=episode.actions + np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * math.pi, 0.0],
                dtype=np.float32,
            ),
        )
        for episode in base
    ]
    base_stats = calculate_action_representation_stats(
        base,
        sampling,
        rpy_transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    ).profile["stats"]["action"]
    unwrapped_stats = calculate_action_representation_stats(
        unwrapped,
        sampling,
        rpy_transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    ).profile["stats"]["action"]
    if not np.allclose(
        np.asarray(base_stats["mean"])[:, 5],
        np.asarray(unwrapped_stats["mean"])[:, 5],
        atol=1e-5,
    ):
        raise AssertionError("RPY yaw wrap did not run before stats")
    if float(np.max(np.abs(np.asarray(unwrapped_stats["max"])[:, 3:6]))) > math.pi + 1e-5:
        raise AssertionError("stats contain unwrapped RPY values")
    print("PASS: quaternion sign/continuity and RPY wrap applied before stats")


def check_artifact_io_and_restore() -> None:
    """artifact I/O, checkpoint 복원, tamper 거부."""
    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE)
    transform = _transform(spec)
    sampling = ActionStatsSampling(action_delta_indices=(0, 1))
    result = calculate_action_representation_stats(
        _episodes(spec),
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
    )
    artifact, _ = upsert_stats_profile(empty_stats_artifact(), result)

    with tempfile.TemporaryDirectory(prefix="so101-action-stats-") as directory:
        root = Path(directory)
        path = write_action_stats_artifact(root, artifact)
        if path != root / ACTION_STATS_DEFAULT_PATH:
            raise AssertionError("stats artifact path mismatch")
        reloaded = read_action_stats_artifact(root)
        if reloaded != json.loads(json.dumps(artifact)):
            raise AssertionError("stats artifact write/read round-trip failed")
        if read_action_stats_artifact(root / "empty")["schema_version"] != (
            ACTION_STATS_SCHEMA_VERSION
        ):
            raise AssertionError("missing artifact must return an empty schema v2 artifact")

    payload = serialize_stats_for_processor(result.profile)
    restored_id, restored_profile = restore_stats_from_processor(payload)
    if restored_id != result.profile_id or restored_profile != result.profile:
        raise AssertionError("checkpoint stats restore failed")

    tampered = json.loads(json.dumps(payload))
    tampered["profile"]["stats"]["action"]["mean"][0][0] += 1.0
    try:
        restore_stats_from_processor(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered checkpoint stats were accepted")

    corrupted = json.loads(json.dumps(artifact))
    profile_id = next(iter(corrupted["profiles"]))
    corrupted["profiles"][profile_id]["stats"]["action"]["count"] = 0
    try:
        validate_action_stats_artifact(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted artifact was accepted")

    debug = calculate_action_representation_stats(
        _episodes(spec)[:1],
        sampling,
        transform,
        dataset_fingerprint=_DATASET_FINGERPRINT,
        production=False,
    )
    if stats_profile_id(debug.profile) != debug.profile_id:
        raise AssertionError("profile id is not content addressed")
    try:
        validate_action_stats_artifact(
            {
                "schema_version": ACTION_STATS_SCHEMA_VERSION,
                "generator_version": artifact["generator_version"],
                "profiles": {debug.profile_id: debug.profile},
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-production profile was accepted by default")

    merged = inject_action_stats(
        {"observation.state": {"mean": [0.0]}, "action": {"mean": [99.0]}},
        result.profile,
    )
    if merged["observation.state"] != {"mean": [0.0]}:
        raise AssertionError("observation stats were not preserved")
    if merged["action"] != result.profile["stats"]["action"]:
        raise AssertionError("action stats were not replaced by the profile")
    print("PASS: stats artifact I/O, checkpoint restore, tamper rejection, injection")


def check_sampling_contract() -> None:
    """sampler 계약 위반과 window 부족을 거부."""
    rejects = [
        ("empty action deltas", lambda: ActionStatsSampling(action_delta_indices=())),
        (
            "non-monotonic deltas",
            lambda: ActionStatsSampling(action_delta_indices=(0, 2, 1)),
        ),
        ("non-zero first delta", lambda: ActionStatsSampling(action_delta_indices=(1, 2))),
        (
            "non-zero reference delta",
            lambda: ActionStatsSampling(
                observation_delta_indices=(-1,),
                action_delta_indices=(0, 1),
            ),
        ),
    ]
    for label, call in rejects:
        try:
            call()
        except (IndexError, ValueError):
            continue
        raise AssertionError(f"invalid sampling config was accepted: {label}")

    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
    transform = _transform(spec)
    short = [
        EpisodeArrays(
            episode_index=0,
            states=np.zeros((2, transform.action_dim), dtype=np.float32),
            actions=np.zeros((2, transform.action_dim), dtype=np.float32),
        )
    ]
    try:
        calculate_action_representation_stats(
            short,
            ActionStatsSampling(action_delta_indices=tuple(range(8))),
            transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("episodes shorter than the horizon must not produce windows")

    mismatched = [
        EpisodeArrays(
            episode_index=0,
            states=np.zeros((10, transform.action_dim + 1), dtype=np.float32),
            actions=np.zeros((10, transform.action_dim + 1), dtype=np.float32),
        )
    ]
    try:
        calculate_action_representation_stats(
            mismatched,
            ActionStatsSampling(action_delta_indices=(0, 1)),
            transform,
            dataset_fingerprint=_DATASET_FINGERPRINT,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dimension mismatch between episodes and transform was accepted")
    print(f"PASS: sampling contract ({len(rejects) + 2} invalid cases rejected)")


CHECKS = (
    check_all_representations_coexist,
    check_cache_invalidation,
    check_transform_ordering,
    check_normalization_before_stats,
    check_artifact_io_and_restore,
    check_sampling_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: action representation stats contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
