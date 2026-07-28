#!/usr/bin/env python3
"""Phase 11 — 4-mode action representation config와 universal manifest 검증.

검증 범위:

- ``joint_absolute``·``joint_relative``·``eef_absolute``·``eef_relative`` enum
- joint mode에서 ``pose_format`` 금지, EEF mode에서 3개 format 강제
- 모호한 legacy ``absolute``/``relative`` 거부와 v1 config 승격 경로
- dataset absolute-storage 원칙
- ``schema_version=2`` manifest 생성/hash/tamper 거부/CLI assertion/legacy opt-in
- 4 mode × pose format × 3 policy = 24 조합 matrix

.. code-block:: bash

    python scripts/contract/validate_action_representation_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from so101_contract.action_manifest import (  # noqa: E402
    ACTION_MANIFEST_SCHEMA_VERSION,
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
    ROTATION_CONVENTIONS,
    assert_manifest_matches_cli,
    build_action_representation_manifest,
    build_feature_contract,
    canonical_manifest_sha256,
    legacy_opt_in_record,
    manifest_schema_version,
    read_action_representation_manifest,
    require_action_representation_manifest,
    so101_contract_source_sha256,
    validate_action_representation_manifest,
    write_action_representation_manifest,
)
from so101_contract.action_representation import (  # noqa: E402
    DATASET_STORAGE_REFERENCE,
    EEF_POSE_FORMATS,
    SUPPORTED_POLICY_FAMILIES,
    ActionReference,
    ActionRepresentationMode,
    ActionRepresentationSpec,
    ActionSpace,
    PoseFormat,
    combination_id,
    dataset_space_for_mode,
    from_legacy_v1_config,
    iter_policy_combinations,
    iter_representation_specs,
    validate_dataset_storage_reference,
)
from so101_contract.eef_action_contract import ActionRepresentationConfig  # noqa: E402

_FAKE_SHA256 = "a" * 64
_FAKE_GIT_SHA = "b" * 40
_ARM_JOINT_DIM = 5
_ACTION_HORIZON = 16


def _feature_names(spec: ActionRepresentationSpec) -> tuple[list[str], dict[str, list[int]]]:
    """Mode에 맞는 feature names/groups fixture."""
    if spec.is_eef:
        pose_dim = spec.pose_dim
        names = [f"tcp_grasp.p{index}" for index in range(pose_dim)] + ["gripper.pos"]
        groups = {spec.action_group: [0, pose_dim], "gripper_position": [pose_dim, pose_dim + 1]}
        return names, groups
    names = [f"arm.joint_{index}" for index in range(_ARM_JOINT_DIM)] + ["gripper.pos"]
    groups = {
        spec.action_group: [0, _ARM_JOINT_DIM],
        "gripper_position": [_ARM_JOINT_DIM, _ARM_JOINT_DIM + 1],
    }
    return names, groups


def _manifest(
    spec: ActionRepresentationSpec,
    *,
    policy_type: str = "act",
    legacy: dict | None = None,
) -> dict:
    names, groups = _feature_names(spec)
    state_names = [name.replace("tcp_grasp.", "state.tcp_grasp.") for name in names]
    return build_action_representation_manifest(
        spec,
        state_feature=build_feature_contract("observation.state", state_names, groups),
        action_feature=build_feature_contract("action", names, groups),
        dataset={
            "repo_id": "so101/pick-cube",
            "revision": "0" * 40,
            "fingerprint": _FAKE_SHA256,
            "space": dataset_space_for_mode(spec.mode),
            "storage_reference": DATASET_STORAGE_REFERENCE,
        },
        stats={
            "profile_id": f"sha256:{_FAKE_SHA256}",
            "content_sha256": _FAKE_SHA256,
            "kind": spec.stats_profile_kind,
            "horizon": _ACTION_HORIZON,
        },
        policy={
            "type": policy_type,
            "model_family": "GR00T-N1.7" if policy_type == "groot" else policy_type.upper(),
            "base_model_path": None,
            "chunk_size": _ACTION_HORIZON,
            "execution_horizon": 8,
            "prediction_api": "predict_action_chunk",
            "full_chunk_postprocess_required": True,
        },
        runtime={
            "lerobot_version": "0.6.0",
            "lerobot_commit": _FAKE_GIT_SHA,
            "project_commit": _FAKE_GIT_SHA,
            "so101_contract_source_sha256": so101_contract_source_sha256(),
            "processor_source_sha256": _FAKE_SHA256,
            "compatible_clients": ["so101_eef_robot_client"],
        },
        action_horizon=_ACTION_HORIZON,
        resolved_contract_fingerprint=_FAKE_SHA256,
        kinematics=(
            {
                "version": "so101_base_tcp_grasp_fk_v2",
                "urdf_sha256": _FAKE_SHA256,
                "robot_yaml_sha256": _FAKE_SHA256,
            }
            if spec.is_eef
            else None
        ),
        legacy=legacy,
    )


def check_mode_enum() -> None:
    """4-mode enum과 space/reference 분해."""
    expected = {
        ActionRepresentationMode.JOINT_ABSOLUTE: (ActionSpace.JOINT, ActionReference.ABSOLUTE),
        ActionRepresentationMode.JOINT_RELATIVE: (ActionSpace.JOINT, ActionReference.RELATIVE),
        ActionRepresentationMode.EEF_ABSOLUTE: (ActionSpace.EEF, ActionReference.ABSOLUTE),
        ActionRepresentationMode.EEF_RELATIVE: (ActionSpace.EEF, ActionReference.RELATIVE),
    }
    if set(ActionRepresentationMode) != set(expected):
        raise AssertionError(f"mode enum drifted: {[m.value for m in ActionRepresentationMode]}")
    for mode, (space, reference) in expected.items():
        if mode.space is not space or mode.reference is not reference:
            raise AssertionError(f"{mode.value} decomposes incorrectly")

    # relative는 state가 아니라 action target의 표현이다.
    for spec in iter_representation_specs():
        semantics = spec.semantics
        if not semantics["state"].startswith("current absolute"):
            raise AssertionError(
                f"{spec.mode.value} state must stay absolute, got {semantics['state']!r}"
            )
        if spec.is_relative and "relative" not in semantics["target"]:
            raise AssertionError(f"{spec.mode.value} target semantics are wrong")
        if spec.gripper_representation != "absolute":
            raise AssertionError("gripper must be an absolute passthrough in every mode")
    print("PASS: 4-mode enum, space/reference split, state/action semantics")


def check_pose_format_rules() -> None:
    """joint mode의 pose_format 금지, EEF mode의 format 강제와 차원."""
    joint = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
    if joint.pose_format is not PoseFormat.NOT_APPLICABLE:
        raise AssertionError("joint mode must normalize pose_format to not_applicable")
    if joint.base_frame is not None or joint.eef_frame is not None:
        raise AssertionError("joint mode must not carry EEF frames")

    expected_dims = {
        PoseFormat.XYZ_ROT6D_ROWS: (9, 10),
        PoseFormat.XYZ_QUATERNION_WXYZ: (7, 8),
        PoseFormat.XYZ_RPY: (6, 7),
    }
    for pose_format, (pose_dim, action_dim) in expected_dims.items():
        spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=pose_format,
        )
        if spec.pose_dim != pose_dim or spec.expected_action_dim() != action_dim:
            raise AssertionError(
                f"{pose_format.value} dims wrong: {spec.pose_dim}/{spec.expected_action_dim()}"
            )
        if spec.action_group != f"eef_{pose_dim}d":
            raise AssertionError(f"{pose_format.value} default group wrong: {spec.action_group}")

    rejects = [
        (
            "joint mode with EEF pose format",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.JOINT_RELATIVE,
                pose_format=PoseFormat.XYZ_RPY,
            ),
        ),
        (
            "EEF mode without pose format",
            lambda: ActionRepresentationSpec(mode=ActionRepresentationMode.EEF_ABSOLUTE),
        ),
        (
            "ambiguous legacy absolute",
            lambda: ActionRepresentationSpec(mode="absolute"),
        ),
        (
            "ambiguous legacy relative",
            lambda: ActionRepresentationSpec(mode="relative"),
        ),
        (
            "unknown mode",
            lambda: ActionRepresentationSpec(mode="eef_delta"),
        ),
        (
            "unknown pose format",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_RELATIVE,
                pose_format="xyz_euler_zyx",
            ),
        ),
        (
            "joint mode with EEF frames",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.JOINT_ABSOLUTE,
                base_frame="base_link",
            ),
        ),
        (
            "relative gripper",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_RELATIVE,
                pose_format=PoseFormat.XYZ_ROT6D_ROWS,
                gripper_representation="relative",
            ),
        ),
        (
            "passthrough collides with transform group",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.JOINT_RELATIVE,
                action_group="gripper_position",
            ),
        ),
        (
            "unsafe stats path",
            lambda: ActionRepresentationSpec(
                mode=ActionRepresentationMode.JOINT_ABSOLUTE,
                stats_file="/etc/passwd",
            ),
        ),
    ]
    for label, call in rejects:
        try:
            call()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid config was accepted: {label}")

    # joint dimension은 하드코딩하지 않고 dataset metadata에서 resolve한다.
    try:
        joint.expected_action_dim()
    except ValueError:
        pass
    else:
        raise AssertionError("joint action dim must not be inferred without dataset metadata")
    if joint.expected_action_dim(joint_dim=5) != 6 or joint.expected_action_dim(joint_dim=6) != 7:
        raise AssertionError("joint action dim resolution is wrong")
    print(f"PASS: pose format rules ({len(rejects)} invalid configs rejected)")


def check_serialization_round_trip() -> None:
    """8개 spec 전부 dict/JSON round-trip과 fingerprint 안정성."""
    seen: set[str] = set()
    for spec in iter_representation_specs():
        payload = json.loads(json.dumps(spec.to_dict()))
        restored = ActionRepresentationSpec.from_dict(payload)
        if restored != spec or restored.fingerprint() != spec.fingerprint():
            raise AssertionError(f"{spec.mode.value} serialization round-trip failed")
        if spec.fingerprint() in seen:
            raise AssertionError(f"{spec.stats_profile_kind} fingerprint collides")
        seen.add(spec.fingerprint())

        tampered = dict(payload)
        tampered["space"] = "joint" if spec.is_eef else "eef"
        try:
            ActionRepresentationSpec.from_dict(tampered)
        except ValueError:
            pass
        else:
            raise AssertionError("derived field mismatch must be rejected")

    bad_version = dict(next(iter_representation_specs()).to_dict())
    bad_version["schema_version"] = 1
    try:
        ActionRepresentationSpec.from_dict(bad_version)
    except ValueError:
        pass
    else:
        raise AssertionError("schema_version=1 payload must be rejected by the v2 loader")
    if len(seen) != 8:
        raise AssertionError(f"expected 8 representation specs, got {len(seen)}")
    print("PASS: 8 representation specs serialize and fingerprint uniquely")


def check_dataset_storage_principle() -> None:
    """Dataset은 모든 mode에서 absolute를 저장한다."""
    if validate_dataset_storage_reference(None) != DATASET_STORAGE_REFERENCE:
        raise AssertionError("default dataset storage reference must be absolute")
    try:
        validate_dataset_storage_reference("relative")
    except ValueError:
        pass
    else:
        raise AssertionError("relative dataset storage must be rejected")
    for spec in iter_representation_specs():
        expected = "absolute_eef" if spec.is_eef else "absolute_joint"
        if dataset_space_for_mode(spec.mode) != expected:
            raise AssertionError(f"{spec.mode.value} maps to the wrong dataset space")
    print("PASS: dataset absolute-storage principle is a code contract")


def check_legacy_v1_promotion() -> None:
    """v1 config 호환 계층."""
    v1_relative = ActionRepresentationConfig(mode="eef_relative")
    promoted = v1_relative.to_spec()
    if promoted.mode is not ActionRepresentationMode.EEF_RELATIVE:
        raise AssertionError("v1 eef_relative did not promote to the v2 mode")
    if promoted.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
        raise AssertionError("v1 eef_relative must promote to xyz_rot6d_rows")
    if promoted.action_group != "eef_9d":
        raise AssertionError(f"v1 group was lost: {promoted.action_group}")

    v1_absolute = ActionRepresentationConfig()
    try:
        v1_absolute.to_spec()
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous v1 absolute must require an explicit opt-in")
    legacy = from_legacy_v1_config(v1_absolute, allow_legacy_absolute=True)
    if legacy.mode is not ActionRepresentationMode.JOINT_ABSOLUTE:
        raise AssertionError("legacy opt-in must resolve to joint_absolute")
    print("PASS: v1 config compatibility layer (eef_relative auto, absolute opt-in only)")


def check_manifest_build_and_validate() -> None:
    """모든 mode에서 manifest가 생성되고 자체 검증을 통과한다."""
    for spec in iter_representation_specs():
        manifest = _manifest(spec)
        if manifest_schema_version(manifest) != ACTION_MANIFEST_SCHEMA_VERSION:
            raise AssertionError("manifest schema version is not 2")
        restored = validate_action_representation_manifest(manifest, expected_spec=spec)
        if restored != spec:
            raise AssertionError(f"{spec.mode.value} manifest spec round-trip failed")
        if manifest["rotation_convention"] != ROTATION_CONVENTIONS[spec.pose_format]:
            raise AssertionError("rotation convention was not recorded")
        if manifest["state_dim"] != manifest["action_dim"]:
            raise AssertionError("state/action layout must match")
        if spec.is_eef and manifest["action_dim"] != spec.expected_action_dim():
            raise AssertionError("EEF action dim disagrees with the pose format")
        if not spec.is_eef and manifest["kinematics"] is not None:
            raise AssertionError("joint manifest must not claim an EEF kinematics contract")
        # async client가 쓰는 horizon 규칙(chunk_size / execution_horizon)이 남아 있어야 한다.
        if manifest["policy"]["chunk_size"] != _ACTION_HORIZON:
            raise AssertionError("policy.chunk_size was dropped from the v2 manifest")
        if manifest["policy"]["execution_horizon"] > manifest["policy"]["chunk_size"]:
            raise AssertionError("execution horizon exceeds the model chunk")
    print("PASS: schema v2 manifest builds and validates in all 8 representations")


def check_manifest_tamper_rejection() -> None:
    """Content hash와 절별 정합 위반은 전부 거부한다."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_QUATERNION_WXYZ,
    )
    base = _manifest(spec)
    if canonical_manifest_sha256(
        {key: value for key, value in base.items() if key != "manifest_sha256"}
    ) != base["manifest_sha256"]:
        raise AssertionError("manifest hash is not reproducible")

    def mutate(**changes) -> dict:
        payload = json.loads(json.dumps(base))
        for key, value in changes.items():
            target = payload
            *parents, leaf = key.split(".")
            for parent in parents:
                target = target[parent]
            target[leaf] = value
        return payload

    tampers = {
        "silent hash edit": mutate(mode="eef_absolute"),
        "stats hash swap": mutate(**{"stats.content_sha256": "c" * 64}),
        "policy downgrade": mutate(**{"policy.full_chunk_postprocess_required": False}),
        "unsupported policy": mutate(**{"policy.type": "diffusion"}),
        "relative dataset": mutate(**{"dataset.storage_reference": "relative"}),
        "wrong dataset space": mutate(**{"dataset.space": "absolute_joint"}),
        "missing kinematics": mutate(kinematics=None),
        "horizon mismatch": mutate(**{"stats.horizon": 4}),
        "stats kind mismatch": mutate(**{"stats.kind": "eef_relative_rot6d"}),
        "bad runtime commit": mutate(**{"runtime.project_commit": "not-a-sha"}),
        "legacy flag without opt-in": mutate(**{"legacy.flag": LEGACY_JOINT_ABSOLUTE_OPT_IN}),
    }
    for label, payload in tampers.items():
        if label != "silent hash edit":
            # hash를 다시 계산해도 절별 정합 검사에서 걸려야 한다.
            payload.pop("manifest_sha256")
            payload["manifest_sha256"] = canonical_manifest_sha256(payload)
        try:
            validate_action_representation_manifest(payload)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"tampered manifest was accepted: {label}")

    dim_mismatch = json.loads(json.dumps(base))
    dim_mismatch["features"]["action"]["names"].append("extra")
    dim_mismatch.pop("manifest_sha256")
    dim_mismatch["manifest_sha256"] = canonical_manifest_sha256(dim_mismatch)
    try:
        validate_action_representation_manifest(dim_mismatch)
    except ValueError:
        pass
    else:
        raise AssertionError("feature dim/name mismatch was accepted")
    print(f"PASS: {len(tampers) + 1} tampered manifests rejected")


def check_cli_assertion() -> None:
    """추론 CLI는 override가 아니라 assertion이다."""
    spec = ActionRepresentationSpec(
        mode=ActionRepresentationMode.EEF_RELATIVE,
        pose_format=PoseFormat.XYZ_RPY,
    )
    manifest = _manifest(spec, policy_type="smolvla")
    assert_manifest_matches_cli(
        manifest,
        mode="eef_relative",
        pose_format="xyz_rpy",
        policy_type="smolvla",
    )
    for label, kwargs in {
        "mode override": {"mode": "eef_absolute"},
        "pose format override": {"pose_format": "xyz_rot6d_rows"},
        "policy override": {"policy_type": "act"},
    }.items():
        try:
            assert_manifest_matches_cli(manifest, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"CLI/checkpoint mismatch was accepted: {label}")
    print("PASS: CLI action representation flags act as assertions")


def check_manifest_io_and_legacy_gate() -> None:
    """모든 mode에서 manifest 필수, 없으면 fail-fast."""
    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
    manifest = _manifest(spec, policy_type="groot")
    with tempfile.TemporaryDirectory(prefix="so101-manifest-") as directory:
        root = Path(directory)
        checkpoint = root / "pretrained_model"
        path = write_action_representation_manifest(checkpoint, manifest)
        if path.name != ACTION_REPRESENTATION_MANIFEST:
            raise AssertionError("manifest file name changed")
        loaded = read_action_representation_manifest(checkpoint)
        if loaded != manifest:
            raise AssertionError("manifest write/read round-trip failed")
        validate_action_representation_manifest(loaded, expected_spec=spec)

        empty = root / "legacy_checkpoint"
        empty.mkdir()
        try:
            require_action_representation_manifest(empty)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing manifest must fail fast")
        if require_action_representation_manifest(
            empty,
            allow_legacy_joint_absolute=True,
        ) is not None:
            raise AssertionError("legacy opt-in must report a missing manifest as None")

    legacy_manifest = _manifest(
        spec,
        legacy=legacy_opt_in_record("migrated joint-absolute checkpoint", source="act_joint_abs"),
    )
    validate_action_representation_manifest(legacy_manifest)
    if legacy_manifest["legacy"]["flag"] != LEGACY_JOINT_ABSOLUTE_OPT_IN:
        raise AssertionError("legacy opt-in was not recorded in the new manifest")

    try:
        _manifest(
            ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_RELATIVE,
                pose_format=PoseFormat.XYZ_ROT6D_ROWS,
            ),
            legacy=legacy_opt_in_record("invalid"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy opt-in must be limited to joint_absolute checkpoints")
    print("PASS: manifest is required in every mode; legacy needs an explicit recorded opt-in")


def check_combination_matrix() -> None:
    """4 mode × pose format × 3 policy = 24 조합."""
    combinations = list(iter_policy_combinations())
    if len(combinations) != 24:
        raise AssertionError(f"expected 24 combinations, got {len(combinations)}")
    identifiers = {combination_id(policy, spec) for policy, spec in combinations}
    if len(identifiers) != 24:
        raise AssertionError("combination identifiers are not unique")
    policies = {policy for policy, _ in combinations}
    if policies != set(SUPPORTED_POLICY_FAMILIES):
        raise AssertionError(f"policy coverage is wrong: {sorted(policies)}")
    eef_specs = [spec for _, spec in combinations if spec.is_eef]
    if len(eef_specs) != 18 or {spec.pose_format for spec in eef_specs} != set(EEF_POSE_FORMATS):
        raise AssertionError("EEF pose format coverage is wrong")

    for policy, spec in combinations:
        manifest = _manifest(spec, policy_type=policy)
        validate_action_representation_manifest(
            manifest,
            expected_spec=spec,
            expected_policy_type=policy,
        )
        if len(spec.inference_routing) < 1:
            raise AssertionError(f"{spec.mode.value} has no inference routing")
    print(f"PASS: 24-combination matrix ({len(identifiers)} manifests built and validated)")


def check_inference_routing() -> None:
    """Mode별 추론 routing 계약."""
    expected = {
        "joint_absolute": ("joint_command",),
        "joint_relative": ("restore_absolute_joint", "joint_command"),
        "eef_absolute": ("ik", "joint_command"),
        "eef_relative": ("restore_absolute_eef", "ik", "joint_command"),
    }
    for spec in iter_representation_specs():
        if spec.inference_routing != expected[spec.mode.value]:
            raise AssertionError(f"{spec.mode.value} routing drifted: {spec.inference_routing}")
        if spec.is_eef and "ik" not in spec.inference_routing:
            raise AssertionError("EEF modes must route through IK")
        if not spec.is_eef and "ik" in spec.inference_routing:
            raise AssertionError("joint modes must not route through IK")
    print("PASS: mode-specific inference routing")


CHECKS = (
    check_mode_enum,
    check_pose_format_rules,
    check_serialization_round_trip,
    check_dataset_storage_principle,
    check_legacy_v1_promotion,
    check_manifest_build_and_validate,
    check_manifest_tamper_rejection,
    check_cli_assertion,
    check_manifest_io_and_legacy_gate,
    check_combination_matrix,
    check_inference_routing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: action representation v2 contract ({len(CHECKS)} checks)")


if __name__ == "__main__":
    main()
