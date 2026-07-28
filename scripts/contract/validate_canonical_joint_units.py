#!/usr/bin/env python3
"""Phase 16 correction — canonical joint 단위와 client 경계 변환 검증(실행 기반).

옛 코드에서 실제로 있던 버그를 각각 잡는다.

1. ROS observation이 ``to_lerobot_units``로 arm을 **degree feature**로 보내던 버그
   → manifest layout(arm radian + gripper [0,100])인지 수치로 확인
2. ROS publish가 router의 canonical radian을 ``from_lerobot_units``로 **한 번 더** 변환하던 버그
   → sim publish helper 출력이 router canonical과 동일한지 확인
3. real joint 경로가 stock client(follower degree 기대)로 가던 버그
   → real observation이 canonical arm radian이고 명령이 follower 변환 **1회**인지 확인
4. non-Rot6D real EEF residual이 rot6d 슬라이스를 쓰던 버그
   → 3 pose format 모두 유한한 position/geodesic 오차
5. joint manifest 단위 계약(degree/누락) 거부
6. wrapper가 두 joint mode를 representation-aware client로 보내는지

.. code-block:: bash

    python scripts/contract/validate_canonical_joint_units.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "contract"))

from validate_action_routing import _checkpoint_dir, _manifest  # noqa: E402
from validate_representation_cli_assertions import (  # noqa: E402
    _real_kinematics_checkpoint,
)

from so101_contract.action_manifest import write_action_representation_manifest  # noqa: E402
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
)
from so101_contract.feature_codec import (  # noqa: E402
    sim_joint_radians_to_policy_feature,
)
from so101_contract.follower_calibration import (  # noqa: E402
    real_follower_to_sim_radians,
    sim_radians_to_real_follower,
)
from so101_contract.inference_startup import (  # noqa: E402
    build_router,
    eef_pose_residual,
    format_startup_log,
    observation_to_manifest_format,
    plan_inference_startup,
    startup_log_fields,
)
from so101_contract.joint_feature_codec import (  # noqa: E402
    CANONICAL_ARM_UNIT,
    build_joint_feature_contract,
    validate_joint_feature_contract,
    canonical_joint_state_to_feature,
    feature_to_canonical_joint_state,
    real_follower_state_to_feature,
    sim_publish_command,
    validate_joint_unit_contract,
)
from so101_contract.joint_topology import so101_arm_joint_topology  # noqa: E402

URDF = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
ROBOT_YAML = ROOT / "assets" / "robots" / "so101.yml"
REAL_WRAPPER = ROOT / "scripts" / "real" / "lerobot.sh"

#: 자명하지 않은 canonical state(모두 0/대칭이 아니어야 degree/radian 혼동이 드러난다).
CANONICAL_STATE = np.asarray([0.62, -0.41, 0.93, -0.18, 1.27, 0.35], dtype=np.float32)
HORIZON = 4


def _joint_plan(mode: ActionRepresentationMode, root: Path):
    spec = ActionRepresentationSpec(mode=mode)
    return plan_inference_startup(_checkpoint_dir(spec, root))


def _joint_chunk() -> np.ndarray:
    """이미 postprocess된 absolute joint chunk(arm radian + gripper feature)."""
    rows = []
    for index in range(HORIZON):
        arm = CANONICAL_STATE[:5] + np.float32(0.01 * (index + 1))
        rows.append(np.concatenate([arm, [30.0 + 10.0 * index]]))
    return np.stack(rows).astype(np.float32)


def check_joint_observation_is_radian() -> None:
    """finding D1: joint observation이 arm radian + gripper [0,100]이고 degree가 아니다."""
    with tempfile.TemporaryDirectory(prefix="unit-obs-") as directory:
        root = Path(directory)
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            plan = _joint_plan(mode, root)
            feature = plan.observation_feature_from_canonical_joint_state(CANONICAL_STATE)
            if feature.shape != (6,):
                raise AssertionError(f"{mode.value} observation dim {feature.shape} != (6,)")
            # arm은 그대로여야 한다.
            if float(np.max(np.abs(feature[:5] - CANONICAL_STATE[:5]))) > 1e-6:
                raise AssertionError(
                    f"{mode.value} arm observation was converted: {feature[:5]} != "
                    f"{CANONICAL_STATE[:5]}"
                )
            # legacy degree encoding과 명확히 달라야 한다.
            legacy = sim_joint_radians_to_policy_feature(CANONICAL_STATE)
            if float(np.max(np.abs(feature[:5] - legacy[:5]))) < 1.0:
                raise AssertionError(
                    "observation is indistinguishable from the legacy degree encoding"
                )
            expected_degrees = np.degrees(CANONICAL_STATE[:5])
            if float(np.max(np.abs(feature[:5] - expected_degrees))) < 1.0:
                raise AssertionError("observation arm looks like degrees, not radians")
            # gripper는 [0,100] policy feature.
            if not (0.0 <= float(feature[5]) <= 100.0):
                raise AssertionError(f"gripper feature out of range: {feature[5]}")
            if abs(float(feature[5]) - float(legacy[5])) > 1e-3:
                raise AssertionError("gripper feature must match the policy feature codec")
            # round-trip
            back = feature_to_canonical_joint_state(feature)
            if float(np.max(np.abs(back - CANONICAL_STATE))) > 1e-5:
                raise AssertionError("joint feature round-trip lost information")
    print(f"PASS: joint observation is arm {CANONICAL_ARM_UNIT} + gripper [0,100] (not degrees)")


def check_sim_publish_has_no_second_conversion() -> None:
    """finding D2: sim publish 값이 router canonical radian과 동일하다."""
    from so101_contract.feature_codec import policy_feature_to_sim_joint_radians

    with tempfile.TemporaryDirectory(prefix="unit-sim-") as directory:
        root = Path(directory)
        chunk = _joint_chunk()
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            plan = _joint_plan(mode, root)
            router = build_router(plan)
            routed = router.route(chunk, CANONICAL_STATE, platform="sim")
            if not routed.success or routed.ik_calls != 0:
                raise AssertionError(f"{mode.value} sim route failed: {routed.reason}")

            for index, platform_action in enumerate(routed.platform_actions):
                published = sim_publish_command(platform_action)
                if float(np.max(np.abs(published - routed.canonical_joint_radians[index]))) > 1e-6:
                    raise AssertionError(
                        f"{mode.value} sim publish differs from the router canonical command"
                    )
                # arm은 chunk의 radian 그대로여야 한다.
                if float(np.max(np.abs(published[:5] - chunk[index, :5]))) > 1e-5:
                    raise AssertionError(
                        f"{mode.value} published arm {published[:5]} != chunk {chunk[index, :5]}"
                    )
                # 옛 버그 재현: from_lerobot_units를 한 번 더 적용하면 값이 크게 달라진다.
                double = policy_feature_to_sim_joint_radians(platform_action[None, :])[0]
                if float(np.max(np.abs(double[:5] - published[:5]))) < 1e-3:
                    raise AssertionError(
                        "fixture cannot distinguish a second codec conversion; "
                        "choose a state where degree/radian differ"
                    )
    print("PASS: sim publish equals router canonical radians (no second from_lerobot_units)")


def check_real_boundary_single_conversion() -> None:
    """finding D3: real observation은 canonical radian, 명령은 follower 변환 정확히 1회."""
    with tempfile.TemporaryDirectory(prefix="unit-real-") as directory:
        root = Path(directory)
        real_state = sim_radians_to_real_follower(CANONICAL_STATE)
        if float(np.max(np.abs(real_follower_to_sim_radians(real_state) - CANONICAL_STATE))) > 1e-4:
            raise AssertionError("follower calibration round-trip is broken")

        chunk = _joint_chunk()
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            plan = _joint_plan(mode, root)
            router = build_router(plan)

            # observation: real follower → canonical arm radian + gripper feature (경계 1회)
            observation = real_follower_state_to_feature(real_state)
            if float(np.max(np.abs(observation[:5] - CANONICAL_STATE[:5]))) > 1e-4:
                raise AssertionError(
                    f"{mode.value} real observation arm is not canonical radians: {observation[:5]}"
                )
            if float(np.max(np.abs(observation - real_state))) < 1e-3:
                raise AssertionError("real observation was not converted from follower units")

            routed = router.route(chunk, real_state, platform="real")
            if not routed.success:
                raise AssertionError(f"{mode.value} real route failed: {routed.reason}")
            if routed.ik_calls != 0:
                raise AssertionError(f"{mode.value} joint route called IK {routed.ik_calls} times")

            expected = sim_radians_to_real_follower(routed.canonical_joint_radians)
            if float(np.max(np.abs(routed.platform_actions - expected))) > 1e-5:
                raise AssertionError(
                    f"{mode.value} real command is not one follower conversion of the canonical "
                    "command"
                )
            # 두 번 변환하면 크게 달라진다(경계 변환이 1회임을 증명).
            twice = sim_radians_to_real_follower(
                np.asarray(routed.platform_actions, dtype=np.float32)
            )
            if float(np.max(np.abs(twice - routed.platform_actions))) < 1e-3:
                raise AssertionError("fixture cannot distinguish a double follower conversion")
            back = real_follower_to_sim_radians(routed.platform_actions)
            if float(np.max(np.abs(back - routed.canonical_joint_radians))) > 1e-4:
                raise AssertionError("real command does not round-trip to the canonical command")
    print("PASS: real joint boundary — canonical radian observation, exactly one follower conversion")


def check_eef_residual_all_formats() -> None:
    """finding D5: 3 pose format 모두 residual이 유한하다(real client와 같은 helper)."""
    with tempfile.TemporaryDirectory(prefix="unit-resid-") as directory:
        root = Path(directory)
        joints = CANONICAL_STATE.copy()
        from so101_contract.eef_ik import IKConfig
        from so101_contract.eef_policy_io import EEFPlatformAdapterConfig, SO101EEFPolicyIO

        adapter = SO101EEFPolicyIO.from_files(
            URDF,
            ROBOT_YAML,
            ik_config=IKConfig(),
            config=EEFPlatformAdapterConfig(real_hardware_ik_validated=True),
        )
        for pose_format in (
            PoseFormat.XYZ_ROT6D_ROWS,
            PoseFormat.XYZ_QUATERNION_WXYZ,
            PoseFormat.XYZ_RPY,
        ):
            for mode in (
                ActionRepresentationMode.EEF_ABSOLUTE,
                ActionRepresentationMode.EEF_RELATIVE,
            ):
                spec = ActionRepresentationSpec(mode=mode, pose_format=pose_format)
                plan = plan_inference_startup(
                    _real_kinematics_checkpoint(spec, root),
                    urdf_path=URDF,
                    robot_yaml_path=ROBOT_YAML,
                )
                measured = observation_to_manifest_format(
                    plan,
                    adapter.observation_from_canonical_radians(joints),
                )
                shifted = joints.copy()
                shifted[0] += 0.05
                target = observation_to_manifest_format(
                    plan,
                    adapter.observation_from_canonical_radians(shifted),
                )
                if measured.shape != target.shape or measured.shape[-1] != plan.state_dim:
                    raise AssertionError(f"{pose_format.value} feature dim mismatch")

                position, orientation = eef_pose_residual(plan, measured, target)
                if not math.isfinite(position) or not math.isfinite(orientation):
                    raise AssertionError(
                        f"{mode.value}/{pose_format.value} residual is not finite"
                    )
                if position <= 0.0 or orientation <= 0.0:
                    raise AssertionError(
                        f"{mode.value}/{pose_format.value} residual should be positive for a "
                        f"shifted target, got {position}/{orientation}"
                    )
                same_position, same_orientation = eef_pose_residual(plan, measured, measured)
                if same_position > 1e-5 or same_orientation > 1e-5:
                    raise AssertionError("identical poses must give a near-zero residual")
    print("PASS: EEF residual works for rot6d/wxyz/rpy (same helper the real client uses)")


def check_joint_unit_contract_rejection() -> None:
    """finding D6: degree/누락 단위 계약은 명령 이전에 거부된다."""
    topology = so101_arm_joint_topology().to_dict()
    validate_joint_unit_contract(topology, source="fixture")

    degrees = {
        "version": topology["version"],
        "joints": [{**joint, "unit": "degree"} for joint in topology["joints"]],
    }
    missing = {
        "version": topology["version"],
        "joints": [{k: v for k, v in joint.items() if k != "unit"} for joint in topology["joints"]],
    }
    for label, payload in (("degrees", degrees), ("missing unit", missing), ("no topology", None)):
        try:
            validate_joint_unit_contract(payload, source="fixture")
        except ValueError:
            continue
        raise AssertionError(f"incompatible joint unit contract was accepted: {label}")

    # startup plan도 같은 계약을 강제한다(manifest 기반).
    with tempfile.TemporaryDirectory(prefix="unit-contract-") as directory:
        root = Path(directory)
        spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
        bad = root / "degree_units"
        bad.mkdir()
        write_action_representation_manifest(
            bad,
            _manifest(spec, joint_topology_override=degrees),
        )
        try:
            plan_inference_startup(bad)
        except ValueError as exc:
            if "unit" not in str(exc):
                raise AssertionError(f"unexpected failure: {exc}") from exc
        else:
            raise AssertionError("startup accepted a degree joint unit contract")

        absent = root / "no_topology"
        absent.mkdir()
        write_action_representation_manifest(
            absent,
            _manifest(spec, drop_joint_topology=True),
        )
        try:
            plan_inference_startup(absent)
        except ValueError:
            pass
        else:
            raise AssertionError("startup accepted a manifest without a joint unit contract")
    print("PASS: degree/missing joint unit contracts are refused before any command")


def check_startup_log_is_mode_aware() -> None:
    """finding 1: joint startup metadata가 kinematics를 읽지 않고, EEF는 version을 보고한다.

    ``eef_robot_client.__init__``이 실제로 호출하는 helper를 그대로 실행한다.
    """
    with tempfile.TemporaryDirectory(prefix="unit-log-") as directory:
        root = Path(directory)
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            plan = _joint_plan(mode, root)
            if plan.contract.kinematics is not None:
                raise AssertionError("joint manifest must not declare kinematics")

            # kinematics를 index하면 즉시 터지도록 만든 뒤 helper를 실행한다.
            class _Explode(dict):
                def __getitem__(self, key):  # noqa: ANN001
                    raise AssertionError(
                        "joint startup must not read the kinematics section"
                    )

            guarded = replace(plan.contract, kinematics=_Explode())
            guarded_plan = replace(plan, contract=guarded)
            fields = startup_log_fields(guarded_plan)
            if fields["kinematics_version"] != "not_required":
                raise AssertionError(
                    f"{mode.value} kinematics_version={fields['kinematics_version']!r}"
                )
            if fields["requires_ik"] is not False or fields["client_kind"] != "joint":
                raise AssertionError(f"{mode.value} startup fields are wrong: {fields}")
            if not isinstance(fields.get("joint_feature_contract"), dict):
                raise AssertionError("joint startup must report the explicit feature contract")
            line = format_startup_log(guarded_plan)
            if "not_required" not in line or mode.value not in line:
                raise AssertionError(f"{mode.value} startup log line is wrong: {line}")

        for pose_format in (
            PoseFormat.XYZ_ROT6D_ROWS,
            PoseFormat.XYZ_QUATERNION_WXYZ,
            PoseFormat.XYZ_RPY,
        ):
            spec = ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_ABSOLUTE,
                pose_format=pose_format,
            )
            plan = plan_inference_startup(
                _real_kinematics_checkpoint(spec, root),
                urdf_path=URDF,
                robot_yaml_path=ROBOT_YAML,
            )
            fields = startup_log_fields(plan)
            if not fields["kinematics_version"] or fields["kinematics_version"] == "not_required":
                raise AssertionError(
                    f"{pose_format.value} must report a verified kinematics version"
                )
            if fields["requires_ik"] is not True:
                raise AssertionError("EEF startup must require IK")

    # 실제 client __init__이 같은 helper를 쓰는지(다른 로그 경로 금지).
    client = (ROOT / "scripts" / "inference" / "eef_robot_client.py").read_text(encoding="utf-8")
    if "format_startup_log(self._startup_plan)" not in client:
        raise AssertionError("real client does not use the shared startup log helper")
    if "kinematics['version']" in client or 'kinematics["version"]' in client:
        raise AssertionError("real client still indexes the kinematics section directly")
    if "self._requires_ik and not config.real_hardware_ik_validated" not in client:
        raise AssertionError("dry-run warning is not limited to IK modes")
    print("PASS: startup metadata is mode-aware (joint=not_required, EEF=verified version)")


def check_explicit_joint_feature_contract() -> None:
    """finding 2: joint manifest가 명시적 feature 계약을 담고 검증된다."""
    topology = so101_arm_joint_topology().to_dict()
    good = build_joint_feature_contract(topology, gripper_index=5)
    validate_joint_feature_contract(
        good,
        joint_topology=topology,
        gripper_representation="absolute",
        action_groups={"arm_joints": [0, 5], "gripper_position": [5, 6]},
        action_dim=6,
        source="fixture",
    )

    rejects = {
        "missing payload": (None, {}),
        "wrong version": ({**good, "version": "so101_canonical_joint_feature_v1"}, {}),
        "degree arm unit": ({**good, "arm_unit": "degree"}, {}),
        "wrong gripper semantics": ({**good, "gripper_semantics": "relative"}, {}),
        "wrong arm dof": ({**good, "arm_dof": 6}, {}),
        "wrong gripper index": ({**good, "gripper_index": 4}, {}),
        "wrong gripper range": ({**good, "gripper_range": [0.0, 1.0]}, {}),
        "relative gripper representation": (good, {"gripper_representation": "relative"}),
        "gripper group with two features": (
            good,
            {"action_groups": {"arm_joints": [0, 5], "gripper_position": [5, 7]}},
        ),
        "gripper group at wrong index": (
            good,
            {"action_groups": {"arm_joints": [0, 4], "gripper_position": [4, 5]}},
        ),
        "wrong action dim": (good, {"action_dim": 7}),
    }
    for label, (payload, extra) in rejects.items():
        try:
            validate_joint_feature_contract(payload, source="fixture", **extra)
        except ValueError:
            continue
        raise AssertionError(f"invalid joint feature contract was accepted: {label}")

    # transform fingerprint / manifest hash 에 실제로 포함된다.
    from so101_contract.action_transform import ActionRepresentationTransform
    from so101_contract.joint_topology import so101_arm_joint_topology as topo

    transform = ActionRepresentationTransform(
        spec=ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
        state_indices=tuple(range(5)),
        action_indices=tuple(range(5)),
        passthrough_action_indices=(5,),
        state_dim=6,
        action_dim=6,
        joint_topology=topo(),
    )
    payload = transform.to_dict()
    if payload["joint_feature_contract"]["version"] != good["version"]:
        raise AssertionError("transform does not serialize the joint feature contract")
    if ActionRepresentationTransform.from_dict(payload) != transform:
        raise AssertionError("transform round-trip lost the joint feature contract")
    mutated = json.loads(json.dumps(payload))
    mutated["joint_feature_contract"]["arm_unit"] = "degree"
    try:
        ActionRepresentationTransform.from_dict(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered joint feature contract was accepted")
    stripped = json.loads(json.dumps(payload))
    stripped["joint_feature_contract"] = None
    try:
        ActionRepresentationTransform.from_dict(stripped)
    except ValueError:
        pass
    else:
        raise AssertionError("missing joint feature contract was silently synthesized")

    from so101_contract.action_transform import ActionRepresentationTransform as _T

    other = _T.from_dict({**payload, "joint_feature_contract": good})
    if other.fingerprint() != transform.fingerprint():
        raise AssertionError("identical contracts must give the same fingerprint")

    # startup(= manifest 경로)도 누락/degree payload 를 거부한다.
    with tempfile.TemporaryDirectory(prefix="unit-contract-payload-") as directory:
        root = Path(directory)
        spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
        for label, kwargs in {
            "missing contract": {"drop_joint_feature_contract": True},
            "degree contract": {
                "joint_feature_contract_override": {**good, "arm_unit": "degree"}
            },
            "wrong index contract": {
                "joint_feature_contract_override": {**good, "gripper_index": 3}
            },
        }.items():
            directory_path = root / label.replace(" ", "_")
            directory_path.mkdir()
            write_action_representation_manifest(directory_path, _manifest(spec, **kwargs))
            try:
                plan_inference_startup(directory_path)
            except ValueError:
                continue
            raise AssertionError(f"startup accepted a manifest with a {label}")

        # 정상 manifest 는 통과하고 계약을 그대로 노출한다.
        ok_dir = root / "ok"
        ok_dir.mkdir()
        write_action_representation_manifest(ok_dir, _manifest(spec))
        plan = plan_inference_startup(ok_dir)
        if plan.joint_unit_contract["gripper_semantics"] != good["gripper_semantics"]:
            raise AssertionError("startup did not surface the explicit gripper semantics")
    print(f"PASS: explicit joint feature contract persisted and validated "
          f"({len(rejects)} invalid payloads rejected)")


def check_wrapper_uses_representation_client() -> None:
    """finding D4: wrapper가 두 joint mode를 representation-aware client로 보낸다."""
    wrapper = REAL_WRAPPER.read_text(encoding="utf-8")
    section = wrapper[wrapper.index("  policy-client)") : wrapper.index("  *)")]
    if "lerobot.async_inference.robot_client" in section:
        raise AssertionError(
            "policy-client still dispatches to the stock robot_client, which expects "
            "follower feature units"
        )
    if "eef_robot_client.py" not in section:
        raise AssertionError("policy-client does not use the representation-aware client")
    if "--emit client_kind" not in section:
        raise AssertionError("policy-client dispatch is not manifest-driven")
    if ":-absolute}" in wrapper:
        raise AssertionError("lerobot.sh still displays the legacy 'absolute' alias")

    # 클라이언트가 joint mode에서 IK를 요구하지 않는지 실제 계획으로 확인.
    with tempfile.TemporaryDirectory(prefix="unit-wrapper-") as directory:
        root = Path(directory)
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            plan = _joint_plan(mode, root)
            if plan.client_kind != "joint" or plan.requires_ik:
                raise AssertionError(f"{mode.value} plan is wrong for the real client")
            router = build_router(plan)
            if router.policy_io is not None:
                raise AssertionError("joint real client must not hold an IK adapter")
    client = (ROOT / "scripts" / "inference" / "eef_robot_client.py").read_text(encoding="utf-8")
    if "real_follower_state_to_feature" not in client:
        raise AssertionError("real client does not build canonical joint observations")
    if "self._requires_ik and not self.config.real_hardware_ik_validated" not in client:
        raise AssertionError("the dry-run gate must apply only to IK/EEF modes")
    print("PASS: real launcher sends both joint modes through the representation-aware client")


CHECKS = (
    check_startup_log_is_mode_aware,
    check_explicit_joint_feature_contract,
    check_joint_observation_is_radian,
    check_sim_publish_has_no_second_conversion,
    check_real_boundary_single_conversion,
    check_eef_residual_all_formats,
    check_joint_unit_contract_rejection,
    check_wrapper_uses_representation_client,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: canonical joint units and client boundaries ({len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
