#!/usr/bin/env python3
"""Phase 16 — server/sim/real 진입점 startup 동작 검증(실행 기반).

grep이 아니라 **실제 startup planning 함수와 CLI를 실행**해서 다음을 증명한다.

1. valid v2 ``eef_absolute``(3 pose format 전부)가 v1 validator를 거치지 않고
   계약 + kinematics hash 검증을 통과
2. ``eef_absolute`` observation/action 이름·차원이 **manifest에서** 나온다
3. manifest 없는 대상 joint checkpoint는 migration 없이는 거부
4. assertion을 생략하면 4개 mode가 manifest에서 유도된다
5. 불일치 assertion은 publisher/robot/client 생성 marker **이전에** 멈춘다
6. joint startup은 IK adapter를 만들지 않고 URDF/YAML을 요구하지 않는다
7. real wrapper dispatch가 manifest 기반이고 두 joint mode도 preflight를 통과한다
8. v2 runtime은 legacy alias ``absolute``를 받지 않는다

정적 grep은 보조로만 쓴다.

.. code-block:: bash

    python scripts/contract/validate_representation_cli_assertions.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "contract"))

from validate_action_routing import _checkpoint_dir, _manifest  # noqa: E402

from so101_contract.action_manifest import (  # noqa: E402
    ACTION_REPRESENTATION_MANIFEST,
    LEGACY_JOINT_ABSOLUTE_OPT_IN,
    canonical_manifest_sha256,
    write_action_representation_manifest,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    iter_representation_specs,
)
from so101_contract.eef_deployment_contract import sha256_file  # noqa: E402
from so101_contract.inference_startup import (  # noqa: E402
    MissingManifestError,
    build_router,
    observation_to_manifest_format,
    plan_inference_startup,
)

ASSERT_CLI = ROOT / "scripts" / "inference" / "assert_checkpoint_representation.py"
ENTRYPOINT = ROOT / "docker" / "policy-entrypoint.sh"
ROS_NODE = ROOT / "ros2_ws" / "src" / "so101_vla_policy" / "so101_vla_policy" / "vla_policy_node.py"
REAL_CLIENT = ROOT / "scripts" / "inference" / "eef_robot_client.py"
REAL_WRAPPER = ROOT / "scripts" / "real" / "lerobot.sh"
URDF = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
ROBOT_YAML = ROOT / "assets" / "robots" / "so101.yml"


def _real_kinematics_checkpoint(spec: ActionRepresentationSpec, root: Path) -> Path:
    """실제 URDF/robot YAML hash를 담은 checkpoint manifest fixture."""
    manifest = _manifest(spec)
    if spec.is_eef:
        manifest = dict(manifest)
        manifest.pop("manifest_sha256")
        manifest["kinematics"] = {
            "version": "so101_base_tcp_grasp_fk_v2",
            "urdf_sha256": sha256_file(URDF),
            "robot_yaml_sha256": sha256_file(ROBOT_YAML),
        }
        manifest["manifest_sha256"] = canonical_manifest_sha256(manifest)
    directory = root / f"real_{spec.stats_profile_kind}"
    directory.mkdir(parents=True, exist_ok=True)
    write_action_representation_manifest(directory, manifest)
    return directory


class _RecordingPolicyIOFactory:
    """IK adapter 생성 여부를 관찰하는 주입 지점."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None]] = []

    def __call__(self, urdf, robot_yaml, ik_config=None, adapter_config=None):  # noqa: ANN001
        self.calls.append((urdf, robot_yaml))
        from so101_contract.eef_policy_io import SO101EEFPolicyIO

        return SO101EEFPolicyIO.from_files(
            urdf,
            robot_yaml,
            ik_config=ik_config,
            config=adapter_config,
        )


def check_eef_absolute_startup_without_v1() -> None:
    """finding 1: eef_absolute 3 format이 v1 validator 없이 startup을 통과한다."""
    import so101_contract.eef_deployment_contract as v1

    original = v1.validate_checkpoint_for_platform
    calls: list[str] = []

    def _tripwire(*args, **kwargs):  # noqa: ANN001
        calls.append("called")
        raise AssertionError("v1 deployment validator must not run on the schema v2 path")

    v1.validate_checkpoint_for_platform = _tripwire
    try:
        with tempfile.TemporaryDirectory(prefix="startup-eef-") as directory:
            root = Path(directory)
            for pose_format in (
                PoseFormat.XYZ_ROT6D_ROWS,
                PoseFormat.XYZ_QUATERNION_WXYZ,
                PoseFormat.XYZ_RPY,
            ):
                spec = ActionRepresentationSpec(
                    mode=ActionRepresentationMode.EEF_ABSOLUTE,
                    pose_format=pose_format,
                )
                checkpoint = _real_kinematics_checkpoint(spec, root)
                plan = plan_inference_startup(
                    checkpoint,
                    urdf_path=URDF,
                    robot_yaml_path=ROBOT_YAML,
                )
                if plan.client_kind != "eef" or not plan.requires_ik:
                    raise AssertionError(f"{pose_format.value} startup plan is wrong")
                if plan.mode != "eef_absolute" or plan.pose_format != pose_format.value:
                    raise AssertionError("resolved representation mismatch")

                # kinematics hash 불일치는 실패해야 한다(v2 manifest 절 기준).
                tampered = root / f"bad_{pose_format.value}"
                shutil.copytree(checkpoint, tampered)
                payload = json.loads(
                    (tampered / ACTION_REPRESENTATION_MANIFEST).read_text(encoding="utf-8")
                )
                payload["kinematics"]["urdf_sha256"] = "f" * 64
                payload.pop("manifest_sha256")
                payload["manifest_sha256"] = canonical_manifest_sha256(payload)
                (tampered / ACTION_REPRESENTATION_MANIFEST).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                try:
                    plan_inference_startup(
                        tampered,
                        urdf_path=URDF,
                        robot_yaml_path=ROBOT_YAML,
                    )
                except ValueError as exc:
                    if "URDF hash mismatch" not in str(exc):
                        raise AssertionError(f"unexpected failure: {exc}") from exc
                else:
                    raise AssertionError("kinematics hash mismatch was accepted")
    finally:
        v1.validate_checkpoint_for_platform = original
    if calls:
        raise AssertionError("v1 deployment validator was invoked")
    print("PASS: eef_absolute (3 formats) startup uses v2 kinematics only, never the v1 validator")


def check_features_come_from_manifest() -> None:
    """finding 3: observation/action 이름·차원이 manifest에서 온다."""
    expected_dims = {
        PoseFormat.XYZ_ROT6D_ROWS: 10,
        PoseFormat.XYZ_QUATERNION_WXYZ: 8,
        PoseFormat.XYZ_RPY: 7,
    }
    with tempfile.TemporaryDirectory(prefix="startup-feat-") as directory:
        root = Path(directory)
        for pose_format, dim in expected_dims.items():
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
                feature = plan.state_feature()
                if feature["shape"] != (dim,) or len(feature["names"]) != dim:
                    raise AssertionError(
                        f"{mode.value}/{pose_format.value} feature dim {feature['shape']} != {dim}"
                    )
                if feature["names"] != list(plan.contract.manifest["features"]["state"]["names"]):
                    raise AssertionError("feature names are not the manifest names")
                if plan.action_feature()["shape"] != (dim,):
                    raise AssertionError("action feature dim mismatch")

                # v1 FK adapter의 rot6d 10D observation → manifest pose format 변환.
                canonical = np.zeros(10, dtype=np.float32)
                canonical[3:9] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
                canonical[9] = 42.0
                converted = observation_to_manifest_format(plan, canonical)
                if converted.shape[-1] != dim:
                    raise AssertionError(
                        f"{pose_format.value} observation conversion dim {converted.shape[-1]} != {dim}"
                    )
                if abs(float(converted[-1]) - 42.0) > 1e-6:
                    raise AssertionError("gripper passthrough lost during observation conversion")

        joint_spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE)
        joint_plan = plan_inference_startup(_checkpoint_dir(joint_spec, root))
        if joint_plan.state_feature()["shape"] != (6,):
            raise AssertionError("joint feature dim must come from the manifest (6D)")
    print("PASS: observation/action names and dim come from the manifest (10D/8D/7D + joint 6D)")


def check_missing_manifest_rejected() -> None:
    """finding 2: manifest 없는 대상 checkpoint는 runtime fallback 없이 거부된다."""
    with tempfile.TemporaryDirectory(prefix="startup-missing-") as directory:
        empty = Path(directory) / "no_manifest"
        empty.mkdir()
        (empty / "config.json").write_text('{"type": "act"}', encoding="utf-8")
        for mode in (None, "joint_absolute", "eef_relative"):
            try:
                plan_inference_startup(empty, mode=mode, urdf_path=URDF, robot_yaml_path=ROBOT_YAML)
            except MissingManifestError as exc:
                if "migrate" not in str(exc):
                    raise AssertionError("refusal must point at the migration tool") from exc
            else:
                raise AssertionError(
                    f"missing manifest was silently accepted for mode={mode!r}"
                )
    print("PASS: missing manifest is refused for every mode (no silent joint_absolute fallback)")


def check_omitted_assertion_derives_all_modes() -> None:
    """finding 4: assertion 생략 시 4개 mode가 manifest에서 유도된다."""
    with tempfile.TemporaryDirectory(prefix="startup-omit-") as directory:
        root = Path(directory)
        seen = {}
        for spec in iter_representation_specs():
            checkpoint = _real_kinematics_checkpoint(spec, root)
            plan = plan_inference_startup(
                checkpoint,
                mode=None,
                pose_format=None,
                urdf_path=URDF if spec.is_eef else None,
                robot_yaml_path=ROBOT_YAML if spec.is_eef else None,
            )
            seen[spec.stats_profile_kind] = (plan.mode, plan.pose_format, plan.client_kind)
            if plan.mode != spec.mode.value:
                raise AssertionError(f"omitted assertion did not derive {spec.mode.value}")
            if plan.pose_format != spec.pose_format.value:
                raise AssertionError("omitted assertion did not derive the pose format")
            if plan.client_kind != ("eef" if spec.is_eef else "joint"):
                raise AssertionError("client dispatch was not derived from the manifest")
        modes = {value[0] for value in seen.values()}
        if modes != {
            "joint_absolute",
            "joint_relative",
            "eef_absolute",
            "eef_relative",
        }:
            raise AssertionError(f"not all 4 modes were derived: {sorted(modes)}")
    print(f"PASS: omitted assertion derives all 4 modes from the manifest ({len(seen)} specs)")


def check_mismatch_stops_before_construction() -> None:
    """finding 5: 불일치 assertion은 robot/publisher 생성 marker 이전에 멈춘다."""
    constructed: list[str] = []

    def marker_factory(urdf, robot_yaml, ik_config=None, adapter_config=None):  # noqa: ANN001
        constructed.append("policy_io")
        raise AssertionError("must not construct hardware adapters after a failed assertion")

    with tempfile.TemporaryDirectory(prefix="startup-mismatch-") as directory:
        root = Path(directory)
        spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=PoseFormat.XYZ_ROT6D_ROWS,
        )
        checkpoint = _real_kinematics_checkpoint(spec, root)
        mismatches = {
            "mode": {"mode": "eef_absolute"},
            "pose format": {"pose_format": "xyz_rpy"},
            "policy family": {"policy_type": "groot"},
        }
        for label, kwargs in mismatches.items():
            try:
                plan = plan_inference_startup(
                    checkpoint,
                    urdf_path=URDF,
                    robot_yaml_path=ROBOT_YAML,
                    **kwargs,
                )
            except ValueError:
                continue
            build_router(plan, policy_io_factory=marker_factory)
            raise AssertionError(f"{label} mismatch did not stop startup")
        if constructed:
            raise AssertionError("hardware adapter was constructed despite a failed assertion")

        # legacy alias 'absolute'는 v2 runtime에서 거부된다.
        try:
            plan_inference_startup(
                checkpoint,
                mode="absolute",
                urdf_path=URDF,
                robot_yaml_path=ROBOT_YAML,
            )
        except ValueError as exc:
            if "ambiguous" not in str(exc):
                raise AssertionError(f"unexpected error for legacy alias: {exc}") from exc
        else:
            raise AssertionError("legacy alias 'absolute' was accepted in the v2 runtime")
    print("PASS: mismatched assertions stop before any adapter/robot construction; 'absolute' rejected")


def check_joint_startup_has_no_ik() -> None:
    """finding 6: joint startup은 IK adapter를 만들지 않고 URDF/YAML을 요구하지 않는다."""
    factory = _RecordingPolicyIOFactory()
    with tempfile.TemporaryDirectory(prefix="startup-joint-") as directory:
        root = Path(directory)
        for mode in (
            ActionRepresentationMode.JOINT_ABSOLUTE,
            ActionRepresentationMode.JOINT_RELATIVE,
        ):
            spec = ActionRepresentationSpec(mode=mode)
            checkpoint = _checkpoint_dir(spec, root)
            # URDF/YAML을 주지 않아도 계획이 완성된다.
            plan = plan_inference_startup(checkpoint)
            if plan.requires_ik or plan.requires_kinematics_files:
                raise AssertionError(f"{mode.value} must not require IK/kinematics")
            if plan.urdf_path is not None or plan.robot_yaml_path is not None:
                raise AssertionError("joint plan must not carry kinematics paths")
            router = build_router(plan, policy_io_factory=factory)
            if router.policy_io is not None:
                raise AssertionError("joint router must not hold an IK adapter")
            if factory.calls:
                raise AssertionError("joint startup constructed an IK adapter")

            # joint route가 실제로 joint command까지 가고 IK 호출이 0이다.
            chunk = np.tile(
                np.asarray([0.05, -0.30, 0.55, 0.10, 0.02, 40.0], dtype=np.float32),
                (4, 1),
            )
            state = np.asarray([0.05, -0.30, 0.55, 0.10, 0.02, 0.30], dtype=np.float32)
            result = router.route(chunk, state, platform="real")
            if not result.success or result.ik_calls != 0:
                raise AssertionError(f"{mode.value} joint route failed: {result.reason}")
            if result.platform_actions is None:
                raise AssertionError("joint route produced no platform command")

            # EEF adapter를 억지로 넣으면 거부된다.
            try:
                build_router(plan, policy_io=object())
            except ValueError:
                pass
            else:
                raise AssertionError("joint router accepted an IK adapter")
    print("PASS: joint startup needs no URDF/YAML, builds no IK adapter, routes with ik_calls=0")


def check_real_wrapper_dispatch() -> None:
    """finding 5/7: wrapper dispatch가 manifest 기반이고 두 joint mode도 preflight를 통과한다."""
    with tempfile.TemporaryDirectory(prefix="startup-dispatch-") as directory:
        root = Path(directory)
        expected = {
            ActionRepresentationMode.JOINT_ABSOLUTE: "joint",
            ActionRepresentationMode.JOINT_RELATIVE: "joint",
        }
        for mode, kind in expected.items():
            spec = ActionRepresentationSpec(mode=mode)
            checkpoint = _checkpoint_dir(spec, root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ASSERT_CLI),
                    "--checkpoint",
                    str(checkpoint),
                    "--skip-kinematics",
                    "--emit",
                    "client_kind",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "ACTION_REPRESENTATION_MODE": ""},
            )
            if result.returncode != 0:
                raise AssertionError(f"{mode.value} preflight failed: {result.stderr[-300:]}")
            if result.stdout.strip() != kind:
                raise AssertionError(
                    f"{mode.value} dispatch emitted {result.stdout.strip()!r} != {kind!r}"
                )

        eef_spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=PoseFormat.XYZ_ROT6D_ROWS,
        )
        eef_checkpoint = _real_kinematics_checkpoint(eef_spec, root)
        result = subprocess.run(
            [
                sys.executable,
                str(ASSERT_CLI),
                "--checkpoint",
                str(eef_checkpoint),
                "--skip-kinematics",
                "--emit",
                "client_kind",
            ],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != "eef":
            raise AssertionError(f"EEF dispatch emitted {result.stdout.strip()!r}")

        # 사용자가 준 mode는 assertion으로만 쓰이며 dispatch를 뒤집지 못한다.
        override = subprocess.run(
            [
                sys.executable,
                str(ASSERT_CLI),
                "--checkpoint",
                str(eef_checkpoint),
                "--from-env",
                "--skip-kinematics",
                "--emit",
                "client_kind",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "ACTION_REPRESENTATION_MODE": "joint_absolute"},
        )
        if override.returncode == 0:
            raise AssertionError("user-supplied mode overrode the manifest dispatch")

        wrapper = REAL_WRAPPER.read_text(encoding="utf-8")
        if "--emit client_kind" not in wrapper:
            raise AssertionError("lerobot.sh dispatch is not manifest-driven")
        if "SO101_CLIENT_KIND" not in wrapper:
            raise AssertionError("lerobot.sh does not use the resolved client kind")
    print("PASS: wrapper dispatch is manifest-driven; both joint modes preflight; override refused")


def check_no_global_inference_defaults() -> None:
    """finding 4: entrypoint/env가 추론 assertion 기본값을 강제하지 않는다."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ACTION_REPRESENTATION_MODE=") and stripped != (
            'ACTION_REPRESENTATION_MODE="${ACTION_REPRESENTATION_MODE:-}"'
        ):
            raise AssertionError(
                f"entrypoint forces a global inference assertion default: {stripped}"
            )
    if "TRAIN_ACTION_REPRESENTATION_MODE" not in text:
        raise AssertionError("training default must live in a separate variable")

    node = ROS_NODE.read_text(encoding="utf-8")
    if '"ACTION_REPRESENTATION_MODE",\n            "absolute",' in node:
        raise AssertionError("ROS node still defaults to the legacy 'absolute' alias")

    from so101_contract.action_manifest import LEGACY_JOINT_ABSOLUTE_OPT_IN as flag

    files = [ROOT / ".env.example", *(ROOT / "env").glob("*.env"), ENTRYPOINT, REAL_WRAPPER]
    for path in files:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if flag in stripped:
                raise AssertionError(f"legacy opt-in must not be a default: {path}: {stripped}")
    print(f"PASS: no global inference assertion default; legacy opt-in absent from {len(files)} configs")


def check_ros_node_import() -> None:
    """ROS runtime이 있으면 실제 노드 모듈을 import해 배선을 확인한다.

    vla-ros 이미지에서 실행하면 동작 검증이 되고, policy-server처럼 ROS가 없는
    환경에서는 요구 사항을 명시하고 건너뛴다(정적 확인은 별도 check가 담당).
    """
    try:
        import rclpy  # noqa: F401
    except ImportError:
        print("SKIP: rclpy unavailable — run this validator in so101-vla-ros:jazzy for the "
              "ROS import check (static wiring check still ran)")
        return
    import importlib

    sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "so101_vla_policy"))
    module = importlib.import_module("so101_vla_policy.vla_policy_node")
    if not hasattr(module, "plan_inference_startup") or not hasattr(module, "build_router"):
        raise AssertionError("ROS node does not use the shared startup planner/router")
    if hasattr(module, "validate_checkpoint_for_platform"):
        raise AssertionError("ROS node still imports the v1 deployment validator")
    print("PASS: ROS node imports in the ROS runtime with the v2 planner/router wired")


def check_static_supplements() -> None:
    """보조 정적 확인(단독 증거로 쓰지 않는다)."""
    client = REAL_CLIENT.read_text(encoding="utf-8")
    if "validate_checkpoint_for_platform" in client:
        raise AssertionError("real client still references the v1 deployment validator")
    node = ROS_NODE.read_text(encoding="utf-8")
    if "validate_checkpoint_for_platform" in node or "relative_stats" in node:
        raise AssertionError("sim client still references v1 deployment fields")
    if "plan_inference_startup" not in node or "plan_inference_startup" not in client:
        raise AssertionError("clients must use the shared startup planner")
    print("PASS: static supplement — no v1 deployment calls remain in sim/real clients")


CHECKS = (
    check_eef_absolute_startup_without_v1,
    check_features_come_from_manifest,
    check_missing_manifest_rejected,
    check_omitted_assertion_derives_all_modes,
    check_mismatch_stops_before_construction,
    check_joint_startup_has_no_ik,
    check_real_wrapper_dispatch,
    check_no_global_inference_defaults,
    check_ros_node_import,
    check_static_supplements,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: representation startup/assertion behavior ({len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
