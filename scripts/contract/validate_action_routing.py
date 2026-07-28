#!/usr/bin/env python3
"""Phase 16 — mode별 추론 routing 검증 (§22.4).

확인 항목:

- 4 mode × 3 EEF pose format이 모두 **joint command**까지 도달
- joint route는 IK 호출 **0회**, EEF route는 platform IK **정확히 1회**
- 입력은 이미 postprocess된 absolute chunk이며 **relative decode를 두 번 하지 않는다**
- gripper absolute passthrough, horizon/순서/shape 보존과 **명시적 dtype 경계**
  (float32/float64 입력만 허용, platform 명령은 float32, 두 dtype을 결과에 기록)
- 잘못된 차원/group/tamper는 **command publish 이전에** 거부
- sim/real platform adapter round-trip(실제 FK/IK 사용, hardware 불필요)

.. code-block:: bash

    python scripts/contract/validate_action_routing.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.action_checkpoint_contract import (  # noqa: E402
    resolve_checkpoint_contract,
)
from so101_contract.action_manifest import (  # noqa: E402
    build_action_representation_manifest,
    build_feature_contract,
    canonical_manifest_sha256,
    so101_contract_source_sha256,
    write_action_representation_manifest,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    iter_representation_specs,
)
from so101_contract.action_routing import (  # noqa: E402
    ActionRepresentationRouter,
    make_router,
)
from so101_contract.action_transform import ActionRepresentationTransform  # noqa: E402
from so101_contract.eef_ik import IKConfig  # noqa: E402
from so101_contract.eef_policy_io import (  # noqa: E402
    EEFPlatformAdapterConfig,
    SO101EEFPolicyIO,
)
from so101_contract.feature_codec import (  # noqa: E402
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from so101_contract.follower_calibration import (  # noqa: E402
    real_follower_to_sim_radians,
    sim_radians_to_real_follower,
)
from so101_contract.joint_feature_codec import (  # noqa: E402
    build_joint_feature_contract,
)
from so101_contract.joint_topology import so101_arm_joint_topology  # noqa: E402
from so101_contract.pose_codec import convert_pose_format  # noqa: E402

URDF = ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"
ROBOT_YAML = ROOT / "assets" / "robots" / "so101.yml"
_FAKE_SHA256 = "a" * 64
_FAKE_GIT = "b" * 40
HORIZON = 4
ARM_NAMES = so101_arm_joint_topology().names


def _policy_io() -> SO101EEFPolicyIO:
    return SO101EEFPolicyIO.from_files(
        URDF,
        ROBOT_YAML,
        ik_config=IKConfig(orientation_weight=0.15, max_iterations=80),
        config=EEFPlatformAdapterConfig(
            max_arm_step_rad=1.0,
            max_gripper_step_rad=1.0,
            real_hardware_ik_validated=True,
        ),
    )


class _CountingPolicyIO:
    """IK 호출 횟수만 세는 래퍼. 운동학 정확성 검증용이 아니다."""

    def __init__(self, inner: SO101EEFPolicyIO) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def config(self):
        return self._inner.config

    def _count(self, name: str):
        def wrapper(*args, **kwargs):
            self.calls += 1
            return getattr(self._inner, name)(*args, **kwargs)

        return wrapper

    def __getattr__(self, name: str):
        if name.startswith("action_chunk_to_"):
            return self._count(name)
        return getattr(self._inner, name)


def _feature_names(spec: ActionRepresentationSpec) -> tuple[list[str], dict[str, list[int]]]:
    if spec.is_eef:
        pose_dim = spec.pose_dim
        names = [f"tcp_grasp.p{index}" for index in range(pose_dim)] + ["gripper.pos"]
        groups = {spec.action_group: [0, pose_dim], "gripper_position": [pose_dim, pose_dim + 1]}
    else:
        names = list(ARM_NAMES) + ["gripper.pos"]
        groups = {spec.action_group: [0, 5], "gripper_position": [5, 6]}
    return names, groups


def _transform_section(spec: ActionRepresentationSpec) -> dict:
    """manifest transform 절. joint mode는 arm 단위 계약(radian)을 반드시 담는다."""
    payload = {
        "version": "so101_action_transform_v2",
        "action_representation": spec.to_dict(),
        "state_indices": list(range(5 if not spec.is_eef else spec.pose_dim)),
        "action_indices": list(range(5 if not spec.is_eef else spec.pose_dim)),
        "passthrough_action_indices": [5 if not spec.is_eef else spec.pose_dim],
        "state_dim": 6 if not spec.is_eef else spec.pose_dim + 1,
        "action_dim": 6 if not spec.is_eef else spec.pose_dim + 1,
        "joint_topology": None,
        "joint_feature_contract": None,
    }
    if not spec.is_eef:
        topology = so101_arm_joint_topology().to_dict()
        payload["joint_topology"] = topology
        payload["joint_feature_contract"] = build_joint_feature_contract(
            topology,
            gripper_index=5,
            gripper_group="gripper_position",
        )
    return payload


def _manifest(
    spec: ActionRepresentationSpec,
    *,
    policy_type: str = "act",
    joint_topology_override: dict | None = None,
    drop_joint_topology: bool = False,
    drop_joint_feature_contract: bool = False,
    joint_feature_contract_override: dict | None = None,
) -> dict:
    names, groups = _feature_names(spec)
    transform = _transform_section(spec)
    if joint_topology_override is not None:
        transform["joint_topology"] = joint_topology_override
    if drop_joint_topology:
        transform["joint_topology"] = None
    if drop_joint_feature_contract:
        transform["joint_feature_contract"] = None
    if joint_feature_contract_override is not None:
        transform["joint_feature_contract"] = joint_feature_contract_override
    return build_action_representation_manifest(
        spec,
        state_feature=build_feature_contract("observation.state", names, groups),
        action_feature=build_feature_contract("action", names, groups),
        dataset={
            "repo_id": "local/routing-fixture",
            "revision": None,
            "fingerprint": _FAKE_SHA256,
            "space": "absolute_eef" if spec.is_eef else "absolute_joint",
            "storage_reference": "absolute",
        },
        stats={
            "profile_id": f"sha256:{_FAKE_SHA256}",
            "content_sha256": _FAKE_SHA256,
            "kind": spec.stats_profile_kind,
            "horizon": HORIZON,
        },
        policy={
            "type": policy_type,
            "model_family": policy_type.upper(),
            "base_model_path": None,
            "chunk_size": HORIZON,
            "execution_horizon": 2,
            "prediction_api": "predict_action_chunk",
            "full_chunk_postprocess_required": True,
        },
        runtime={
            "lerobot_version": "0.6.0",
            "lerobot_commit": _FAKE_GIT,
            "project_commit": _FAKE_GIT,
            "so101_contract_source_sha256": so101_contract_source_sha256(),
            "processor_source_sha256": _FAKE_SHA256,
            "compatible_clients": ["so101_eef_robot_client"],
        },
        action_horizon=HORIZON,
        resolved_contract_fingerprint=_FAKE_SHA256,
        transform=transform,
        kinematics=(
            {
                "version": "so101_base_tcp_grasp_fk_v2",
                "urdf_sha256": _FAKE_SHA256,
                "robot_yaml_sha256": _FAKE_SHA256,
            }
            if spec.is_eef
            else None
        ),
    )


def _checkpoint_dir(spec: ActionRepresentationSpec, root: Path) -> Path:
    directory = root / spec.stats_profile_kind
    directory.mkdir(parents=True, exist_ok=True)
    write_action_representation_manifest(directory, _manifest(spec))
    return directory


def _current_state() -> np.ndarray:
    return np.asarray([0.05, -0.30, 0.55, 0.10, 0.02, 0.30], dtype=np.float32)


def _absolute_chunk(spec: ActionRepresentationSpec, policy_io: SO101EEFPolicyIO) -> np.ndarray:
    """실제 FK로 만든 도달 가능한 absolute chunk(이미 postprocess된 상태)."""
    current = _current_state()
    gripper_features = np.asarray([30.0, 45.0, 60.0, 75.0], dtype=np.float32)
    if spec.is_eef:
        rows = []
        for index in range(HORIZON):
            joints = current.copy()
            joints[0] += 0.01 * (index + 1)
            joints[1] += 0.008 * (index + 1)
            joints[3] -= 0.006 * (index + 1)
            pose = policy_io.kinematics.forward_xyz_rot6d(joints[:5])
            if spec.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
                pose = convert_pose_format(
                    np.asarray(pose, dtype=np.float64),
                    PoseFormat.XYZ_ROT6D_ROWS,
                    spec.pose_format,
                )
            rows.append(np.concatenate([np.asarray(pose, dtype=np.float32), [gripper_features[index]]]))
        return np.stack(rows).astype(np.float32)

    rows = []
    for index in range(HORIZON):
        arm = current[:5].copy()
        arm[0] += 0.01 * (index + 1)
        arm[2] -= 0.005 * (index + 1)
        rows.append(np.concatenate([arm, [gripper_features[index]]]))
    return np.stack(rows).astype(np.float32)


def check_all_modes_reach_joint_command() -> None:
    """4 mode × 3 format 전부 joint command까지 도달하고 IK 호출 수가 정확하다."""
    policy_io = _policy_io()
    with tempfile.TemporaryDirectory(prefix="routing-ckpt-") as directory:
        root = Path(directory)
        summary = []
        for spec in iter_representation_specs():
            contract = resolve_checkpoint_contract(_checkpoint_dir(spec, root))
            counting = _CountingPolicyIO(policy_io) if spec.is_eef else None
            router = ActionRepresentationRouter.from_contract(
                contract,
                policy_io=counting or None,
                adapter_config=policy_io.config,
            )
            chunk = _absolute_chunk(spec, policy_io)
            result = router.route(chunk, _current_state(), platform="sim")
            if not result.success:
                raise AssertionError(f"{spec.stats_profile_kind} routing failed: {result.reason}")
            if result.canonical_joint_radians.shape != (HORIZON, 6):
                raise AssertionError(
                    f"{spec.stats_profile_kind} joint command shape mismatch: "
                    f"{result.canonical_joint_radians.shape}"
                )
            if result.platform_actions is None:
                raise AssertionError(f"{spec.stats_profile_kind} produced no platform command")

            expected_ik = 1 if spec.is_eef else 0
            if result.ik_calls != expected_ik:
                raise AssertionError(
                    f"{spec.stats_profile_kind} ik_calls={result.ik_calls} != {expected_ik}"
                )
            if counting is not None and counting.calls != 1:
                raise AssertionError(
                    f"{spec.stats_profile_kind} called the IK adapter {counting.calls} times"
                )
            if not spec.is_eef and result.ik is not None:
                raise AssertionError("joint route must not carry an IK result")
            if ("ik" in result.route) != spec.is_eef:
                raise AssertionError(f"{spec.stats_profile_kind} route mismatch: {result.route}")
            summary.append((spec.stats_profile_kind, result.ik_calls))
        joint_calls = [calls for kind, calls in summary if kind.startswith("joint_")]
        eef_calls = [calls for kind, calls in summary if kind.startswith("eef_")]
        if set(joint_calls) != {0} or set(eef_calls) != {1}:
            raise AssertionError(f"IK call routing is wrong: {summary}")
    print(f"PASS: 8 representations reach a joint command (joint IK=0, EEF IK=1)")


def check_no_second_decode() -> None:
    """relative chunk가 다시 decode/re-anchor되지 않는다."""
    policy_io = _policy_io()
    with tempfile.TemporaryDirectory(prefix="routing-rel-") as directory:
        root = Path(directory)
        for spec in (
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
            ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_RELATIVE,
                pose_format=PoseFormat.XYZ_ROT6D_ROWS,
            ),
        ):
            contract = resolve_checkpoint_contract(_checkpoint_dir(spec, root))
            router = ActionRepresentationRouter.from_contract(
                contract,
                policy_io=policy_io if spec.is_eef else None,
                adapter_config=policy_io.config,
            )
            absolute = _absolute_chunk(spec, policy_io)
            state = _current_state()

            # 같은 absolute chunk를 다른 두 state로 라우팅한다. 두 번째 decode가 남아 있다면
            # state가 다시 anchor로 쓰여 결과가 크게 달라진다.
            first = router.route(absolute, state, platform="canonical")
            shifted = state.copy()
            shifted[0] += 0.20
            second = router.route(absolute, shifted, platform="canonical")
            if not (first.success and second.success):
                raise AssertionError("relative routing failed on a reachable chunk")
            delta = float(
                np.max(
                    np.abs(
                        first.canonical_joint_radians[:, :5]
                        - second.canonical_joint_radians[:, :5]
                    )
                )
            )
            if delta > 0.05:
                raise AssertionError(
                    f"{spec.stats_profile_kind} re-anchored on the current state "
                    f"(max joint delta {delta:.4f}); the chunk is already absolute"
                )
            # 결정성: 같은 입력은 같은 명령.
            repeat = router.route(absolute, state, platform="canonical")
            if not np.array_equal(
                repeat.canonical_joint_radians,
                first.canonical_joint_radians,
            ):
                raise AssertionError("routing is not deterministic")
    print("PASS: relative chunks are routed as already-absolute (no second decode)")


def check_passthrough_and_shape() -> None:
    """gripper passthrough, horizon/순서, dtype 보존."""
    policy_io = _policy_io()
    with tempfile.TemporaryDirectory(prefix="routing-pass-") as directory:
        root = Path(directory)
        for spec in iter_representation_specs():
            contract = resolve_checkpoint_contract(_checkpoint_dir(spec, root))
            router = ActionRepresentationRouter.from_contract(
                contract,
                policy_io=policy_io if spec.is_eef else None,
                adapter_config=policy_io.config,
            )
            chunk = _absolute_chunk(spec, policy_io)
            result = router.route(chunk, _current_state(), platform="sim")
            if not result.success:
                raise AssertionError(f"{spec.stats_profile_kind}: {result.reason}")

            gripper_index = contract.passthrough_indices[0]
            expected_feature = chunk[:, gripper_index]
            actual_feature = sim_joint_radians_to_policy_feature(
                result.canonical_joint_radians
            )[:, 5]
            if float(np.max(np.abs(actual_feature - expected_feature))) > 1e-2:
                raise AssertionError(
                    f"{spec.stats_profile_kind} gripper passthrough drifted: "
                    f"{actual_feature} != {expected_feature}"
                )
            # 순서 보존: gripper feature가 단조 증가하는 fixture이므로 명령도 단조여야 한다.
            if np.any(np.diff(actual_feature) <= 0):
                raise AssertionError(f"{spec.stats_profile_kind} chunk order was not preserved")
            # dtype 경계는 명시적이다: float32/float64 입력만 허용하고 platform 명령은
            # 항상 float32이며, 결과가 두 dtype을 모두 기록한다(숨은 변환 금지).
            if result.canonical_joint_radians.dtype != np.float32:
                raise AssertionError("platform command must use the float32 boundary dtype")
            if result.input_dtype != "float32" or result.platform_dtype != "float32":
                raise AssertionError(
                    f"dtype bookkeeping wrong: {result.input_dtype}/{result.platform_dtype}"
                )
            wide = router.route(chunk.astype(np.float64), _current_state(), platform="sim")
            if not wide.success:
                raise AssertionError(f"float64 input rejected: {wide.reason}")
            if wide.input_dtype != "float64" or wide.platform_dtype != "float32":
                raise AssertionError(
                    f"float64 input was not recorded: {wide.input_dtype}/{wide.platform_dtype}"
                )
            if float(np.max(np.abs(wide.canonical_joint_radians - result.canonical_joint_radians))) > 1e-5:
                raise AssertionError("float64 and float32 inputs produced different commands")
            for bad in (chunk.astype(np.float16), chunk.astype(np.int32)):
                try:
                    router.route(bad, _current_state(), platform="sim")
                except TypeError:
                    continue
                raise AssertionError(f"unsupported dtype {bad.dtype} was silently converted")
            if result.horizon != HORIZON:
                raise AssertionError(f"horizon changed: {result.horizon} != {HORIZON}")
    print("PASS: gripper passthrough, chunk order, horizon; explicit float32 platform dtype boundary")


def check_rejection_before_publish() -> None:
    """차원/group/tamper 위반은 명령을 만들기 전에 거부한다."""
    policy_io = _policy_io()
    with tempfile.TemporaryDirectory(prefix="routing-reject-") as directory:
        root = Path(directory)
        spec = ActionRepresentationSpec(
            mode=ActionRepresentationMode.EEF_RELATIVE,
            pose_format=PoseFormat.XYZ_ROT6D_ROWS,
        )
        checkpoint = _checkpoint_dir(spec, root)
        contract = resolve_checkpoint_contract(checkpoint)
        router = ActionRepresentationRouter.from_contract(contract, policy_io=policy_io)
        chunk = _absolute_chunk(spec, policy_io)

        rejects = {
            "wrong action dim": lambda: router.route(
                chunk[:, :-1],
                _current_state(),
                platform="sim",
            ),
            "non-finite chunk": lambda: router.route(
                np.where(np.arange(chunk.size).reshape(chunk.shape) == 5, np.nan, chunk),
                _current_state(),
                platform="sim",
            ),
            "single step (not a chunk)": lambda: router.route(
                chunk[0],
                _current_state(),
                platform="sim",
            ),
            "empty chunk": lambda: router.route(
                chunk[:0],
                _current_state(),
                platform="sim",
            ),
            "wrong state dim": lambda: router.route(
                chunk,
                _current_state()[:5],
                platform="sim",
            ),
            "unknown platform": lambda: router.route(
                chunk,
                _current_state(),
                platform="hardware",
            ),
        }
        for label, call in rejects.items():
            try:
                call()
            except (ValueError, TypeError):
                continue
            raise AssertionError(f"invalid routing input was accepted: {label}")

        # group 불일치: EEF pose group 크기가 틀린 계약은 router 생성 자체가 실패한다.
        try:
            ActionRepresentationRouter(
                spec,
                action_dim=10,
                transform_indices=tuple(range(8)),
                passthrough_indices=(8, 9),
                policy_io=policy_io,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched pose group size was accepted")

        # joint mode는 IK adapter 없이도 만들어져야 하고, EEF mode는 거부돼야 한다.
        try:
            ActionRepresentationRouter(
                spec,
                action_dim=10,
                transform_indices=tuple(range(9)),
                passthrough_indices=(9,),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("EEF router without a platform adapter was accepted")

        # manifest tamper는 계약 resolve 단계에서 막힌다(명령 생성 전).
        manifest_path = checkpoint / "action_representation.json"
        tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
        tampered["action_dim"] = 12
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            resolve_checkpoint_contract(checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered manifest was accepted before routing")
    print(f"PASS: {len(rejects) + 3} invalid routing/contract inputs rejected before publish")


def check_platform_round_trip() -> None:
    """sim/real platform 경계가 실제 FK/IK·calibration로 왕복한다(hardware 불필요)."""
    policy_io = _policy_io()
    with tempfile.TemporaryDirectory(prefix="routing-platform-") as directory:
        root = Path(directory)
        state = _current_state()
        real_state = sim_radians_to_real_follower(state)
        if float(np.max(np.abs(real_follower_to_sim_radians(real_state) - state))) > 1e-4:
            raise AssertionError("follower calibration round-trip is broken")

        for spec in (
            ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
            ActionRepresentationSpec(
                mode=ActionRepresentationMode.EEF_ABSOLUTE,
                pose_format=PoseFormat.XYZ_ROT6D_ROWS,
            ),
        ):
            contract = resolve_checkpoint_contract(_checkpoint_dir(spec, root))
            router = ActionRepresentationRouter.from_contract(
                contract,
                policy_io=policy_io if spec.is_eef else None,
                adapter_config=policy_io.config,
            )
            chunk = _absolute_chunk(spec, policy_io)

            sim_result = router.route(chunk, state, platform="sim")
            real_result = router.route(chunk, real_state, platform="real")
            dry_result = router.route(chunk, real_state, platform="real_dry_run")
            for name, result in (
                ("sim", sim_result),
                ("real", real_result),
                ("real_dry_run", dry_result),
            ):
                if not result.success:
                    raise AssertionError(f"{spec.stats_profile_kind}/{name}: {result.reason}")

            # 두 platform은 같은 canonical 명령을 내고 frame 변환에서만 갈라진다.
            canonical_delta = float(
                np.max(
                    np.abs(
                        sim_result.canonical_joint_radians - real_result.canonical_joint_radians
                    )
                )
            )
            if canonical_delta > 1e-5:
                raise AssertionError(
                    f"{spec.stats_profile_kind}: sim/real canonical commands diverged "
                    f"({canonical_delta:.6f})"
                )
            back = real_follower_to_sim_radians(real_result.platform_actions)
            if float(np.max(np.abs(back - real_result.canonical_joint_radians))) > 1e-4:
                raise AssertionError("real platform command did not round-trip through calibration")
            if not np.array_equal(
                sim_result.platform_actions,
                sim_result.canonical_joint_radians,
            ):
                raise AssertionError("sim platform command must stay in canonical radians")

            if spec.is_eef:
                # 실제 IK 정확도: 명령 joint를 FK하면 요청한 EEF pose로 돌아온다.
                for index, joints in enumerate(sim_result.canonical_joint_radians):
                    pose = policy_io.kinematics.forward_xyz_rot6d(joints[:5])
                    target = chunk[index, : spec.pose_dim]
                    error = float(np.max(np.abs(np.asarray(pose)[:3] - target[:3])))
                    if error > 2e-3:
                        raise AssertionError(
                            f"IK/FK round-trip error at {index}: {error:.6f} m"
                        )
            else:
                expected_arm = chunk[:, :5]
                error = float(np.max(np.abs(sim_result.canonical_joint_radians[:, :5] - expected_arm)))
                if error > 1e-6:
                    raise AssertionError(f"joint route altered the arm target: {error}")
    print("PASS: sim/real platform round-trip via real FK/IK and follower calibration")


def check_transform_contract_alignment() -> None:
    """router 입력이 processor decode 출력과 같은 계약인지 확인한다."""
    policy_io = _policy_io()
    spec = ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE)
    topology = so101_arm_joint_topology()
    transform = ActionRepresentationTransform(
        spec=spec,
        state_indices=tuple(range(5)),
        action_indices=tuple(range(5)),
        passthrough_action_indices=(5,),
        state_dim=6,
        action_dim=6,
        joint_topology=topology,
    )
    state = np.asarray([0.05, -0.30, 0.55, 0.10, 0.02, 40.0], dtype=np.float32)
    absolute = np.stack(
        [
            np.concatenate([state[:5] + 0.01 * (index + 1), [40.0 + 5.0 * index]])
            for index in range(HORIZON)
        ]
    ).astype(np.float32)

    encoded = transform.encode(state, absolute)
    decoded = transform.decode(state, encoded)
    with tempfile.TemporaryDirectory(prefix="routing-align-") as directory:
        contract = resolve_checkpoint_contract(_checkpoint_dir(spec, Path(directory)))
        router = ActionRepresentationRouter.from_contract(
            contract,
            adapter_config=policy_io.config,
        )
        canonical_state = np.concatenate(
            [
                state[:5],
                policy_feature_to_sim_joint_radians(
                    np.concatenate([np.zeros(5, dtype=np.float32), state[5:6]])[None, :]
                )[0, 5:6],
            ]
        ).astype(np.float32)
        routed = router.route(np.asarray(decoded, dtype=np.float32), canonical_state, platform="sim")
        if not routed.success:
            raise AssertionError(f"decoded chunk failed routing: {routed.reason}")
        if float(np.max(np.abs(routed.canonical_joint_radians[:, :5] - absolute[:, :5]))) > 1e-5:
            raise AssertionError("processor decode → router lost the absolute joint target")
    print("PASS: processor decode output and router input share one contract")


CHECKS = (
    check_all_modes_reach_joint_command,
    check_no_second_decode,
    check_passthrough_and_shape,
    check_rejection_before_publish,
    check_platform_round_trip,
    check_transform_contract_alignment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for check in CHECKS:
        check()
    print(f"PASS: action representation routing ({len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
