#!/usr/bin/env python3
"""Phase 13–14 — mode 중립 LeRobot processor step 검증.

확인 항목:

- 4개 mode의 encode/decode step ordering과 full-chunk 1회 호출
- ``joint_absolute``/``eef_absolute``는 canonical 정규화만 하고 **state cache를 만들지 않음**
- ``joint_relative``/``eef_relative``는 하나의 prediction-time state로 chunk 전체를 복원
- ``eef_relative`` decode 결과가 quaternion 연속성/RPY wrap/Rot6D 직교 canonical form 유지
- serialize → reload → pair relink 후 동일 결과, dataset 없이 stats 복원
- v1 ``eef_relative + rot6d`` processor와 **수치 parity** 및 compatibility adapter

LeRobot이 필요하므로 policy-server 이미지에서 실행한다.

.. code-block:: bash

    docker run --rm -v "$PWD:/workspace" -w /workspace --entrypoint python \\
        policy-server:0.6.0 scripts/contract/validate_action_representation_processor.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.processor.pipeline import (  # noqa: E402
    DataProcessorPipeline,
    TransitionKey,
)
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    iter_representation_specs,
)
from so101_contract.action_representation_processor import (  # noqa: E402
    ACTION_PROCESSOR_SCHEMA_VERSION,
    POSTPROCESSOR_REGISTRY_NAME,
    PREPROCESSOR_REGISTRY_NAME,
    ActionRepresentationDecodeStep,
    ActionRepresentationEncodeStep,
    has_action_representation_processor_steps,
    make_action_representation_processor_steps,
    reconnect_action_representation_processor_steps,
    transform_from_v1_eef_contract,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ActionStatsSampling,
    EpisodeArrays,
    calculate_action_representation_stats,
    serialize_stats_for_processor,
)
from so101_contract.action_transform import ActionRepresentationTransform  # noqa: E402
from so101_contract.eef_action_contract import ActionRepresentationConfig  # noqa: E402
from so101_contract.eef_relative_action import (  # noqa: E402
    absolute_actions_to_relative as v1_absolute_to_relative,
    relative_actions_to_absolute as v1_relative_to_absolute,
)
from so101_contract.joint_topology import (  # noqa: E402
    TAU,
    JointSpec,
    JointTopology,
    JointType,
    canonicalize_joint_actions,
)
from so101_contract.pose_codec import decode_pose, encode_pose  # noqa: E402

_FINGERPRINT = "e" * 64
_ARM_JOINT_NAMES = tuple(f"arm.joint_{index}" for index in range(5))
TOLERANCE = 3e-5


def _max_error(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


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


def _batch(
    spec: ActionRepresentationSpec,
    *,
    batch: int = 2,
    horizon: int = 5,
    seed: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = np.random.default_rng(seed)
    if spec.is_eef:
        state_pose = encode_pose(
            generator.normal(scale=0.2, size=(batch, 3)),
            _random_rotations(batch, seed + 1),
            spec.pose_format,
        )
        action_pose = encode_pose(
            generator.normal(scale=0.2, size=(batch, horizon, 3)),
            _random_rotations(batch * horizon, seed + 2).reshape(batch, horizon, 3, 3),
            spec.pose_format,
        )
        state = np.concatenate([state_pose, generator.uniform(0, 100, size=(batch, 1))], axis=-1)
        actions = np.concatenate(
            [action_pose, generator.uniform(0, 100, size=(batch, horizon, 1))],
            axis=-1,
        )
    else:
        state = np.concatenate(
            [
                generator.uniform(-math.pi, math.pi, size=(batch, 5)),
                generator.uniform(0, 100, size=(batch, 1)),
            ],
            axis=-1,
        )
        actions = np.concatenate(
            [
                generator.uniform(-math.pi, math.pi, size=(batch, horizon, 5)),
                generator.uniform(0, 100, size=(batch, horizon, 1)),
            ],
            axis=-1,
        )
    return (
        torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32)),
    )


def _stats_payload(spec: ActionRepresentationSpec, transform) -> dict:
    generator = np.random.default_rng(5)
    length = 20
    if spec.is_eef:
        poses = encode_pose(
            np.cumsum(generator.normal(scale=0.01, size=(length + 1, 3)), axis=0),
            _random_rotations(length + 1, 9),
            spec.pose_format,
        )
        columns = np.concatenate(
            [poses, generator.uniform(0, 100, size=(length + 1, 1))],
            axis=-1,
        )
    else:
        columns = np.concatenate(
            [
                np.cumsum(generator.normal(scale=0.05, size=(length + 1, 5)), axis=0),
                generator.uniform(0, 100, size=(length + 1, 1)),
            ],
            axis=-1,
        )
    episodes = [
        EpisodeArrays(
            episode_index=0,
            states=columns[:-1].astype(np.float32),
            actions=columns[1:].astype(np.float32),
        )
    ]
    result = calculate_action_representation_stats(
        episodes,
        ActionStatsSampling(action_delta_indices=(0, 1, 2, 3, 4)),
        transform,
        dataset_fingerprint="f" * 64,
    )
    return serialize_stats_for_processor(result.profile)


def _transition(state: torch.Tensor, action: torch.Tensor | None) -> dict:
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
    }
    return transition


def check_all_modes_round_trip() -> None:
    """4개 mode 전부 encode → decode가 absolute action을 복원한다."""
    for spec in iter_representation_specs():
        transform = _transform(spec)
        encode_step, decode_step = make_action_representation_processor_steps(
            transform,
            contract_fingerprint=_FINGERPRINT,
            stats_payload=_stats_payload(spec, transform),
        )
        state, actions = _batch(spec)
        encoded = encode_step(_transition(state, actions))[TransitionKey.ACTION]
        if encoded.shape != actions.shape:
            raise AssertionError(f"{spec.stats_profile_kind} encode changed the chunk shape")
        decoded = decode_step(_transition(state, encoded))[TransitionKey.ACTION]

        if spec.is_eef:
            expected_rotation = decode_pose(
                actions[..., : spec.pose_dim].numpy(),
                spec.pose_format,
            )[1]
            actual_rotation = decode_pose(
                decoded[..., : spec.pose_dim].numpy(),
                spec.pose_format,
            )[1]
            if _max_error(actual_rotation, expected_rotation) > TOLERANCE:
                raise AssertionError(f"{spec.stats_profile_kind} rotation round-trip failed")
            if _max_error(decoded[..., :3], actions[..., :3]) > TOLERANCE:
                raise AssertionError(f"{spec.stats_profile_kind} translation round-trip failed")
        else:
            expected = canonicalize_joint_actions(actions.numpy(), transform.joint_topology)
            if _max_error(decoded.numpy(), expected) > TOLERANCE:
                raise AssertionError(f"{spec.stats_profile_kind} joint round-trip failed")
        if _max_error(decoded[..., -1], actions[..., -1]) > 0.0:
            raise AssertionError(f"{spec.stats_profile_kind} gripper passthrough changed")
        if decoded.dtype is not torch.float32:
            raise AssertionError("processor did not preserve float32")
    print("PASS: all 4 modes encode/decode round-trip through registered processor steps")


def check_absolute_modes_skip_transform() -> None:
    """absolute mode는 target 변환을 건너뛰고 state cache를 요구하지 않는다."""
    for spec in (
        ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
        ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_ABSOLUTE,
            pose_format=PoseFormat.XYZ_ROT6D_ROWS,
        ),
    ):
        transform = _transform(spec)
        if transform.requires_state_reference:
            raise AssertionError(f"{spec.mode.value} must not require a state reference")
        encode_step, decode_step = make_action_representation_processor_steps(
            transform,
            contract_fingerprint=_FINGERPRINT,
            stats_payload=_stats_payload(spec, transform),
        )
        state, actions = _batch(spec)
        encoded = encode_step(_transition(state, actions))[TransitionKey.ACTION]
        # canonical 입력이므로 encode는 값을 바꾸지 않는다(정규화만 수행).
        if _max_error(encoded, actions) > TOLERANCE:
            raise AssertionError(f"{spec.mode.value} encode must not transform the target")

        # absolute mode는 observation을 받아도 state cache를 만들지 않는다.
        if encode_step.get_cached_state() is not None:
            raise AssertionError(
                f"{spec.mode.value} must not cache a prediction-time state"
            )
        # 이전에 남아 있던 cache도 비워야 한다(잘못된 re-anchoring 차단).
        encode_step._last_state = state.clone()
        encode_step(_transition(state, actions))
        if encode_step.get_cached_state() is not None:
            raise AssertionError(
                f"{spec.mode.value} must clear any stale cached state"
            )

        # state 없이도 동작한다.
        encode_step.reset()
        without_state = encode_step({TransitionKey.ACTION: actions})[TransitionKey.ACTION]
        if _max_error(without_state, encoded) > TOLERANCE:
            raise AssertionError(f"{spec.mode.value} encode must work without an observation")
        decoded = decode_step({TransitionKey.ACTION: encoded})[TransitionKey.ACTION]
        if _max_error(decoded, encoded) > TOLERANCE:
            raise AssertionError(f"{spec.mode.value} decode must be idempotent")

    # relative mode는 반대로 state가 없으면 실패한다.
    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE)
    transform = _transform(spec)
    encode_step, decode_step = make_action_representation_processor_steps(
        transform,
        contract_fingerprint=_FINGERPRINT,
        stats_payload=_stats_payload(spec, transform),
    )
    relative_state, actions = _batch(spec)
    try:
        decode_step({TransitionKey.ACTION: actions})
    except RuntimeError:
        pass
    else:
        raise AssertionError("relative decoder without a cached state must fail")
    encode_step(_transition(relative_state, actions))
    if encode_step.get_cached_state() is None:
        raise AssertionError("relative mode must cache the prediction-time state")
    print("PASS: absolute modes cache no state; relative modes require the cached state")


def check_single_chunk_reference() -> None:
    """Relative decode는 chunk를 만든 observation 하나만 기준으로 쓴다."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_QUATERNION_WXYZ,
    )
    transform = _transform(spec)
    encode_step, decode_step = make_action_representation_processor_steps(
        transform,
        contract_fingerprint=_FINGERPRINT,
        stats_payload=_stats_payload(spec, transform),
    )
    state, actions = _batch(spec, seed=21)
    targets = encode_step(_transition(state, actions))[TransitionKey.ACTION]
    first = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]

    # 같은 chunk를 다시 decode해도 결과가 같아야 한다(재anchor 금지).
    again = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]
    if _max_error(first, again) > 0.0:
        raise AssertionError("decoding the same chunk twice produced different results")

    # 새 observation이 들어오면 그 다음 chunk부터 기준이 바뀐다.
    other_state, _ = _batch(spec, seed=33)
    encode_step(_transition(other_state, None))
    rebased = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]
    if _max_error(rebased, first) <= TOLERANCE:
        raise AssertionError("decoder ignored the newly cached observation state")

    # step별로 잘라 호출하는 경로는 명시적으로 막혀 있다.
    try:
        decode_step({TransitionKey.ACTION: targets[:, 0]})
    except NotImplementedError:
        pass
    else:
        raise AssertionError("per-step postprocessing must be rejected")
    try:
        encode_step(_transition(state, actions[:, 0]))
    except ValueError:
        pass
    else:
        raise AssertionError("per-step preprocessing must be rejected")

    encode_step.reset()
    if encode_step.get_cached_state() is not None:
        raise AssertionError("reset() did not clear the cached state")
    print("PASS: one full-chunk reference state, no per-step re-anchoring")


def check_relative_decode_canonical_form() -> None:
    """복원된 absolute EEF chunk도 encode와 같은 canonical form을 만족한다."""
    for pose_format in (
        PoseFormat.XYZ_ROT6D_ROWS,
        PoseFormat.XYZ_QUATERNION_WXYZ,
        PoseFormat.XYZ_RPY,
    ):
        spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=pose_format,
        )
        transform = _transform(spec)
        encode_step, decode_step = make_action_representation_processor_steps(
            transform,
            contract_fingerprint=_FINGERPRINT,
            stats_payload=_stats_payload(spec, transform),
        )
        state, actions = _batch(spec, batch=2, horizon=6, seed=77)
        targets = encode_step(_transition(state, actions))[TransitionKey.ACTION]
        decoded = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]
        pose = decoded[..., : spec.pose_dim].numpy()

        # 1) 물리 SE(3)는 보존된다(canonicalization은 표현만 바꾼다).
        expected_rotation = decode_pose(actions[..., : spec.pose_dim].numpy(), pose_format)[1]
        actual_rotation = decode_pose(pose, pose_format)[1]
        if _max_error(actual_rotation, expected_rotation) > TOLERANCE:
            raise AssertionError(f"{pose_format.value} decode changed the physical rotation")
        if _max_error(decoded[..., :3], actions[..., :3]) > TOLERANCE:
            raise AssertionError(f"{pose_format.value} decode changed the translation")

        # 2) format별 canonical form.
        if pose_format is PoseFormat.XYZ_QUATERNION_WXYZ:
            quaternion = pose[..., 3:7]
            norms = np.linalg.norm(quaternion, axis=-1)
            if float(np.max(np.abs(norms - 1.0))) > TOLERANCE:
                raise AssertionError("decoded quaternions are not unit norm")
            dots = np.sum(quaternion[:, 1:] * quaternion[:, :-1], axis=-1)
            if float(np.min(dots)) < 0.0:
                raise AssertionError(
                    "decoded quaternion chunk still contains a sign discontinuity"
                )
        elif pose_format is PoseFormat.XYZ_RPY:
            angles = pose[..., 3:6]
            if float(np.max(np.abs(angles))) > math.pi + 1e-6:
                raise AssertionError("decoded RPY escaped the wrapped range")
        else:
            rows = pose[..., 3:9].reshape(*pose.shape[:-1], 2, 3)
            row_norms = np.linalg.norm(rows, axis=-1)
            if float(np.max(np.abs(row_norms - 1.0))) > TOLERANCE:
                raise AssertionError("decoded Rot6D rows are not unit length")
            orthogonality = np.sum(rows[..., 0, :] * rows[..., 1, :], axis=-1)
            if float(np.max(np.abs(orthogonality))) > TOLERANCE:
                raise AssertionError("decoded Rot6D rows are not orthogonal")

        # 3) gripper passthrough는 절대 건드리지 않는다.
        if _max_error(decoded[..., -1], actions[..., -1]) > 0.0:
            raise AssertionError(f"{pose_format.value} decode changed the gripper passthrough")

        # 4) canonical form은 고정점이다(다시 encode/decode해도 그대로).
        again = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]
        if _max_error(again, decoded) > 0.0:
            raise AssertionError("decode is not deterministic")
    print("PASS: eef_relative decode returns canonical pose form with SE(3) preserved")


def check_serialization_and_relink() -> None:
    """직렬화 → 로드 → pair relink 후에도 같은 결과와 stats 복원."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_RPY,
    )
    transform = _transform(spec)
    stats_payload = _stats_payload(spec, transform)
    encode_step, decode_step = make_action_representation_processor_steps(
        transform,
        contract_fingerprint=_FINGERPRINT,
        stats_payload=stats_payload,
        manifest_context={"note": "phase-14"},
    )
    preprocessor = DataProcessorPipeline(steps=[encode_step], name="pre")
    postprocessor = DataProcessorPipeline(steps=[decode_step], name="post")
    if not has_action_representation_processor_steps(preprocessor, postprocessor):
        raise AssertionError("processor pair detection failed")

    state, actions = _batch(spec, seed=41)
    targets = encode_step(_transition(state, actions))[TransitionKey.ACTION]
    expected = decode_step({TransitionKey.ACTION: targets})[TransitionKey.ACTION]

    with tempfile.TemporaryDirectory(prefix="so101-action-processor-") as directory:
        root = Path(directory)
        preprocessor.save_pretrained(root / "pre")
        postprocessor.save_pretrained(root / "post")
        loaded_pre = DataProcessorPipeline.from_pretrained(root / "pre", config_filename="pre.json")
        loaded_post = DataProcessorPipeline.from_pretrained(
            root / "post",
            config_filename="post.json",
        )

        loaded_encode = loaded_pre.steps[0]
        loaded_decode = loaded_post.steps[0]
        if not isinstance(loaded_encode, ActionRepresentationEncodeStep):
            raise AssertionError("encode step type was lost during serialization")
        if not isinstance(loaded_decode, ActionRepresentationDecodeStep):
            raise AssertionError("decode step type was lost during serialization")
        if loaded_encode.transform.fingerprint() != transform.fingerprint():
            raise AssertionError("transform contract changed during serialization")

        # relink 전에는 연결이 없다.
        try:
            loaded_decode({TransitionKey.ACTION: targets})
        except RuntimeError:
            pass
        else:
            raise AssertionError("unconnected decode step must fail")
        reconnect_action_representation_processor_steps(loaded_pre, loaded_post)

        reloaded_targets = loaded_encode(_transition(state, actions))[TransitionKey.ACTION]
        if _max_error(reloaded_targets, targets) > 0.0:
            raise AssertionError("reloaded encode step produced different targets")
        reloaded = loaded_decode({TransitionKey.ACTION: reloaded_targets})[TransitionKey.ACTION]
        if _max_error(reloaded, expected) > 0.0:
            raise AssertionError("reloaded processor pair produced different actions")

        config = json.loads((root / "pre" / "pre.json").read_text(encoding="utf-8"))
        names = [step.get("registry_name") or step.get("class") for step in config["steps"]]
        if PREPROCESSOR_REGISTRY_NAME not in json.dumps(config):
            raise AssertionError(f"registry name missing from the saved config: {names}")
        if loaded_encode.stats_payload != stats_payload:
            raise AssertionError("stats payload was not restored from the checkpoint")
        if loaded_encode.schema_version != ACTION_PROCESSOR_SCHEMA_VERSION:
            raise AssertionError("processor schema version was not persisted")

    post_config = decode_step.get_config()
    if POSTPROCESSOR_REGISTRY_NAME and "transform" not in post_config:
        raise AssertionError("decode step config does not carry the transform contract")
    print("PASS: processor serialization, dataset-free stats restore, pair relink")


def check_strict_guards() -> None:
    """계약 위반 step 생성/연결을 거부한다."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_ROT6D_ROWS,
    )
    transform = _transform(spec)
    stats_payload = _stats_payload(spec, transform)
    other = _transform(ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE))

    tampered_stats = json.loads(json.dumps(stats_payload))
    tampered_stats["profile"]["stats"]["action"]["mean"][0][0] += 1.0

    rejects = {
        "missing contract fingerprint": lambda: ActionRepresentationEncodeStep(
            transform=transform,
            contract_fingerprint="",
            stats_payload=stats_payload,
        ),
        "missing stats payload": lambda: ActionRepresentationEncodeStep(
            transform=transform,
            contract_fingerprint=_FINGERPRINT,
        ),
        "tampered stats payload": lambda: ActionRepresentationEncodeStep(
            transform=transform,
            contract_fingerprint=_FINGERPRINT,
            stats_payload=tampered_stats,
        ),
        "invalid transform payload": lambda: ActionRepresentationEncodeStep(
            transform="not-a-transform",
            contract_fingerprint=_FINGERPRINT,
            stats_payload=stats_payload,
        ),
    }
    for label, call in rejects.items():
        try:
            call()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid processor step was accepted: {label}")

    encode_step = ActionRepresentationEncodeStep(
        transform=transform,
        contract_fingerprint=_FINGERPRINT,
        stats_payload=stats_payload,
    )
    mismatched = ActionRepresentationDecodeStep(
        transform=other,
        contract_fingerprint=_FINGERPRINT,
    )
    try:
        mismatched.connect(encode_step)
    except ValueError:
        pass
    else:
        raise AssertionError("connecting a mismatched transform pair was accepted")

    wrong_fingerprint = ActionRepresentationDecodeStep(
        transform=transform,
        contract_fingerprint="a" * 64,
    )
    try:
        wrong_fingerprint.connect(encode_step)
    except ValueError:
        pass
    else:
        raise AssertionError("connecting mismatched fingerprints was accepted")

    incomplete_pre = DataProcessorPipeline(steps=[encode_step], name="pre")
    incomplete_post = DataProcessorPipeline(steps=[], name="post")
    try:
        has_action_representation_processor_steps(incomplete_pre, incomplete_post)
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete processor pair was accepted")
    print(f"PASS: strict processor guards ({len(rejects) + 3} invalid cases rejected)")


def check_v1_parity() -> None:
    """v1 ``eef_relative + rot6d`` processor와 수치 parity 및 compatibility adapter."""
    v1_config = ActionRepresentationConfig(mode="eef_relative")
    spec = v1_config.to_spec()
    transform = _transform(spec)
    state, actions = _batch(spec, seed=55)

    v1_relative = v1_absolute_to_relative(
        state,
        actions,
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
    )
    v2_relative = transform.encode(state, actions)
    if _max_error(v2_relative, v1_relative) > 1e-6:
        raise AssertionError(
            f"v1/v2 relative encode mismatch: {_max_error(v2_relative, v1_relative)}"
        )

    v1_absolute = v1_relative_to_absolute(
        state,
        v1_relative,
        state_pose_indices=tuple(range(9)),
        action_pose_indices=tuple(range(9)),
    )
    v2_absolute = transform.decode(state, v2_relative)
    if _max_error(v2_absolute, v1_absolute) > 1e-6:
        raise AssertionError("v1/v2 absolute decode mismatch")

    # compatibility adapter: v1 resolved contract → v2 transform
    class _V1Contract:
        config = v1_config
        state_pose_indices = tuple(range(9))
        action_pose_indices = tuple(range(9))
        passthrough_action_indices = (9,)
        state_dim = 10
        action_dim = 10

    adapted = transform_from_v1_eef_contract(_V1Contract())
    if adapted.fingerprint() != transform.fingerprint():
        raise AssertionError("v1 compatibility adapter produced a different transform")
    if _max_error(adapted.encode(state, actions), v1_relative) > 1e-6:
        raise AssertionError("adapted transform disagrees with the v1 processor")
    print("PASS: v1 eef_relative+rot6d numerical parity and compatibility adapter")


CHECKS = (
    check_all_modes_round_trip,
    check_absolute_modes_skip_transform,
    check_single_chunk_reference,
    check_relative_decode_canonical_form,
    check_serialization_and_relink,
    check_strict_guards,
    check_v1_parity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: action representation processor contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
