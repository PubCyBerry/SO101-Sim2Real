"""Phase 16 — runtime 독립 inference startup planning.

policy-server hook, sim(ROS) client, real client가 **같은 순수 함수**로 기동 계획을 세운다.
ROS·robot·CUDA 없이 실행되므로 validator가 실제 동작을 검증할 수 있다.

계획 순서(모든 진입점 공통):

.. code-block:: text

    1. checkpoint 계약 resolve (schema v2 전용)
    2. CLI/env representation assertion (생략 = manifest 수용, override 아님)
    3. EEF mode면 manifest kinematics 절 ↔ 실제 URDF/robot YAML hash 검증
    4. manifest feature names/dim으로 observation/action schema 구성
    5. mode에 맞는 router 구성 (joint = IK 어댑터 없음, EEF = adapter 필수)

**v1 deployment 계약(:mod:`so101_contract.eef_deployment_contract`)은 이 경로에서 절대
호출하지 않는다.** v1은 migration 전용이며 runtime fallback이 아니다. manifest가 없는
checkpoint는 대상 policy에서 곧바로 실패하고, migration을 요구한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_checkpoint_contract import (
    ResolvedCheckpointContract,
    resolve_checkpoint_contract,
)
from .action_manifest import ACTION_REPRESENTATION_MANIFEST
from .action_representation import ActionRepresentationSpec, PoseFormat
from .action_routing import ActionRepresentationRouter
from .eef_deployment_contract import sha256_file
from .joint_feature_codec import (
    canonical_joint_state_to_feature,
    validate_joint_feature_contract,
    validate_joint_unit_contract,
)

INFERENCE_STARTUP_VERSION = "so101_inference_startup_v2"

#: platform 명령 경계 dtype. router 출력과 robot/sim publish는 이 dtype으로 고정한다.
PLATFORM_COMMAND_DTYPE = "float32"


class MissingManifestError(FileNotFoundError):
    """대상 policy checkpoint에 schema v2 manifest가 없을 때."""


@dataclass(frozen=True)
class InferenceStartupPlan:
    """기동 전에 확정되는 계약. robot/sim 객체를 만들기 전에 완성된다."""

    contract: ResolvedCheckpointContract
    client_kind: str  # "eef" | "joint"
    requires_ik: bool
    requires_kinematics_files: bool
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    urdf_path: str | None
    robot_yaml_path: str | None
    joint_unit_contract: dict[str, Any] | None = None

    @property
    def spec(self) -> ActionRepresentationSpec:
        return self.contract.spec

    @property
    def mode(self) -> str:
        return self.contract.spec.mode.value

    @property
    def pose_format(self) -> str:
        return self.contract.spec.pose_format.value

    @property
    def platform_command_dtype(self) -> str:
        return PLATFORM_COMMAND_DTYPE

    def state_feature(self) -> dict[str, Any]:
        """LeRobot feature dict(manifest 기준). 10D를 하드코딩하지 않는다."""
        return {
            "dtype": "float32",
            "shape": (self.state_dim,),
            "names": list(self.state_names),
        }

    def action_feature(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (self.action_dim,),
            "names": list(self.action_names),
        }

    def observation_feature_from_canonical_joint_state(self, joint_radians: Any) -> Any:
        """canonical 6D sim radian → manifest joint feature(arm radian + gripper feature).

        joint mode 전용이며 legacy degree codec을 쓰지 않는다.
        """
        if self.contract.spec.is_eef:
            raise ValueError(
                f"mode={self.mode!r} observation is an EEF pose; use "
                "observation_to_manifest_format instead"
            )
        return canonical_joint_state_to_feature(joint_radians)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pose_format": self.pose_format,
            "client_kind": self.client_kind,
            "requires_ik": self.requires_ik,
            "requires_kinematics_files": self.requires_kinematics_files,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "routing": list(self.contract.routing),
            "policy_type": self.contract.policy_type,
            "chunk_size": self.contract.chunk_size,
            "execution_horizon": self.contract.execution_horizon,
            "manifest_sha256": self.contract.manifest_sha256,
            "platform_command_dtype": self.platform_command_dtype,
            "joint_unit_contract": self.joint_unit_contract,
        }


def validate_manifest_kinematics(
    contract: ResolvedCheckpointContract,
    *,
    urdf_path: str | Path,
    robot_yaml_path: str | Path,
) -> dict[str, Any]:
    """schema v2 ``kinematics`` 절을 실제 URDF/robot YAML hash와 대조한다.

    v1 deployment validator를 쓰지 않고 v2 manifest만으로 검증한다.
    """
    kinematics = contract.kinematics
    if not isinstance(kinematics, dict):
        raise ValueError(
            f"EEF checkpoint manifest has no kinematics section: {contract.source}"
        )
    actual_urdf = sha256_file(urdf_path)
    actual_yaml = sha256_file(robot_yaml_path)
    if kinematics.get("urdf_sha256") != actual_urdf:
        raise ValueError(
            "checkpoint/client URDF hash mismatch: "
            f"{kinematics.get('urdf_sha256')} != {actual_urdf}"
        )
    if kinematics.get("robot_yaml_sha256") != actual_yaml:
        raise ValueError(
            "checkpoint/client robot YAML hash mismatch: "
            f"{kinematics.get('robot_yaml_sha256')} != {actual_yaml}"
        )
    version = kinematics.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("EEF checkpoint manifest has no kinematics version")
    return {
        "version": version,
        "urdf_sha256": actual_urdf,
        "robot_yaml_sha256": actual_yaml,
    }


def plan_inference_startup(
    source: str | Path,
    *,
    mode: str | None = None,
    pose_format: str | None = None,
    policy_type: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    urdf_path: str | Path | None = None,
    robot_yaml_path: str | Path | None = None,
    verify_kinematics: bool = True,
) -> InferenceStartupPlan:
    """robot/sim 객체 생성 **이전에** 계약을 확정한다.

    Args:
        mode/pose_format/policy_type: **선택적 assertion**. ``None``이면 manifest 값을
            그대로 쓰고 routing도 manifest에서 유도한다. 값이 다르면 여기서 실패한다.
        urdf_path/robot_yaml_path: EEF mode에서만 필요하다. joint mode는 요구하지 않는다.
    """
    try:
        contract = resolve_checkpoint_contract(
            source,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    except FileNotFoundError as exc:
        raise MissingManifestError(
            f"checkpoint has no {ACTION_REPRESENTATION_MANIFEST}: {source}. "
            "schema v2 inference never infers the action representation; migrate the "
            "checkpoint with scripts/convert/migrate_action_representation_checkpoint.py"
        ) from exc
    contract.assert_cli(mode=mode, pose_format=pose_format, policy_type=policy_type)

    is_eef = contract.spec.is_eef
    resolved_urdf = str(urdf_path) if urdf_path is not None else None
    resolved_yaml = str(robot_yaml_path) if robot_yaml_path is not None else None
    if is_eef:
        if verify_kinematics:
            if not resolved_urdf or not resolved_yaml:
                raise ValueError(
                    f"mode={contract.spec.mode.value!r} routes through IK; provide the URDF and "
                    "robot YAML paths so the manifest kinematics hashes can be verified"
                )
            validate_manifest_kinematics(
                contract,
                urdf_path=resolved_urdf,
                robot_yaml_path=resolved_yaml,
            )
    joint_unit_contract = None
    if not is_eef:
        # joint mode는 kinematics 파일을 요구하지 않고 IK를 만들지도 않는다.
        resolved_urdf = None
        resolved_yaml = None
        # arm 단위 계약(radian)을 manifest에서 확인한다. degree/누락은 명령 이전에 실패.
        transform = contract.manifest.get("transform")
        topology = transform.get("joint_topology") if isinstance(transform, dict) else None
        source_label = f"checkpoint manifest {contract.source}"
        validate_joint_unit_contract(topology, source=source_label)
        # 명시적 joint feature 계약(단위·gripper 의미·index·group)을 검증한다.
        # 기본값을 합성하지 않고, manifest feature group과 교차 검증한다.
        joint_unit_contract = validate_joint_feature_contract(
            transform.get("joint_feature_contract") if isinstance(transform, dict) else None,
            joint_topology=topology,
            gripper_representation=contract.spec.gripper_representation,
            action_groups=contract.manifest["features"]["action"]["groups"],
            action_dim=contract.action_dim,
            source=source_label,
        )

    state_names = tuple(contract.manifest["features"]["state"]["names"])
    return InferenceStartupPlan(
        contract=contract,
        client_kind="eef" if is_eef else "joint",
        requires_ik=is_eef,
        requires_kinematics_files=is_eef,
        state_names=state_names,
        action_names=contract.action_names,
        state_dim=contract.state_dim,
        action_dim=contract.action_dim,
        urdf_path=resolved_urdf,
        robot_yaml_path=resolved_yaml,
        joint_unit_contract=joint_unit_contract,
    )


def build_router(
    plan: InferenceStartupPlan,
    *,
    policy_io: Any | None = None,
    adapter_config: Any | None = None,
    ik_config: Any | None = None,
    policy_io_factory: Any | None = None,
) -> ActionRepresentationRouter:
    """계획에 맞는 router를 만든다. joint mode는 IK adapter를 만들지 않는다.

    ``policy_io_factory``는 테스트에서 IK 생성 여부를 관찰하기 위한 주입 지점이다.
    """
    if not plan.requires_ik:
        if policy_io is not None:
            raise ValueError(
                f"mode={plan.mode!r} must not receive an IK adapter; joint routing never uses IK"
            )
        return ActionRepresentationRouter.from_contract(
            plan.contract,
            policy_io=None,
            adapter_config=adapter_config,
        )

    if policy_io is None:
        factory = policy_io_factory
        if factory is None:
            from .eef_policy_io import SO101EEFPolicyIO

            def factory(urdf, yaml, ik_config=None, adapter_config=None):  # noqa: ANN001
                return SO101EEFPolicyIO.from_files(
                    urdf,
                    yaml,
                    ik_config=ik_config,
                    config=adapter_config,
                )

        policy_io = factory(
            plan.urdf_path,
            plan.robot_yaml_path,
            ik_config=ik_config,
            adapter_config=adapter_config,
        )
    return ActionRepresentationRouter.from_contract(
        plan.contract,
        policy_io=policy_io,
        adapter_config=adapter_config or getattr(policy_io, "config", None),
    )


def observation_to_manifest_format(plan: InferenceStartupPlan, canonical_observation: Any) -> Any:
    """v1 FK adapter의 canonical rot6d 10D observation을 manifest pose format으로 변환.

    ``xyz_rot6d_rows``면 그대로 두고, ``wxyz``/``rpy``면 회전 표현만 바꾼다(frame·의미 불변).
    joint mode는 변환 대상이 아니다.
    """
    import numpy as np

    values = np.asarray(canonical_observation, dtype=np.float32)
    if not plan.contract.spec.is_eef:
        return values
    if plan.contract.spec.pose_format is PoseFormat.XYZ_ROT6D_ROWS:
        if values.shape[-1] != plan.state_dim:
            raise ValueError(
                f"observation dim {values.shape[-1]} != manifest {plan.state_dim}"
            )
        return values
    from .pose_codec import convert_pose_format

    pose = convert_pose_format(
        values[..., :9].astype(np.float64),
        PoseFormat.XYZ_ROT6D_ROWS,
        plan.contract.spec.pose_format,
    )
    converted = np.concatenate(
        [np.asarray(pose, dtype=np.float32), values[..., 9:]],
        axis=-1,
    )
    if converted.shape[-1] != plan.state_dim:
        raise ValueError(
            f"converted observation dim {converted.shape[-1]} != manifest {plan.state_dim}"
        )
    return converted


def eef_pose_residual(
    plan: InferenceStartupPlan,
    measured_feature: Any,
    target_feature: Any,
) -> tuple[float, float]:
    """manifest pose format으로 표현된 measured/target에서 position·geodesic 오차를 낸다.

    rot6d(10D)·wxyz(8D)·rpy(7D) 모두 동작하며 gripper는 건드리지 않는다.
    """
    import numpy as np

    from .pose_codec import decode_pose, rotation_geodesic_angle

    if not plan.contract.spec.is_eef:
        raise ValueError(f"mode={plan.mode!r} has no EEF pose residual")
    pose_dim = plan.contract.spec.pose_dim
    measured = np.asarray(measured_feature, dtype=np.float64)[..., :pose_dim]
    target = np.asarray(target_feature, dtype=np.float64)[..., :pose_dim]
    if measured.shape[-1] != pose_dim or target.shape[-1] != pose_dim:
        raise ValueError(
            f"pose residual needs {pose_dim}D poses, got "
            f"{measured.shape[-1]}/{target.shape[-1]}"
        )
    measured_translation, measured_rotation = decode_pose(measured, plan.contract.spec.pose_format)
    target_translation, target_rotation = decode_pose(target, plan.contract.spec.pose_format)
    position = float(np.linalg.norm(measured_translation - target_translation))
    orientation = float(rotation_geodesic_angle(target_rotation, measured_rotation))
    return position, orientation


def lerobot_state_features(plan: InferenceStartupPlan) -> dict[str, Any]:
    """manifest 기준 ``observation.state`` feature 하나짜리 dict."""
    return {"observation.state": plan.state_feature()}


def startup_log_fields(plan: InferenceStartupPlan) -> dict[str, Any]:
    """진입점이 찍는 startup metadata. **mode-aware**이며 joint에서 kinematics를 읽지 않는다.

    EEF는 검증된 kinematics version을, joint는 ``not_required``를 보고한다.
    """
    fields: dict[str, Any] = {
        "mode": plan.mode,
        "pose_format": plan.pose_format,
        "client_kind": plan.client_kind,
        "state_dim": plan.state_dim,
        "action_dim": plan.action_dim,
        "manifest_sha256": plan.contract.manifest_sha256,
        "stats_profile_id": plan.contract.stats_profile_id,
        "requires_ik": plan.requires_ik,
    }
    if plan.requires_ik:
        kinematics = plan.contract.kinematics
        if not isinstance(kinematics, dict) or not kinematics.get("version"):
            raise ValueError(
                f"mode={plan.mode!r} requires a verified kinematics contract in the manifest"
            )
        fields["kinematics_version"] = str(kinematics["version"])
    else:
        # joint manifest는 kinematics=None이 정상이다. 절대 index하지 않는다.
        fields["kinematics_version"] = "not_required"
        fields["joint_feature_contract"] = plan.joint_unit_contract
    return fields


def format_startup_log(plan: InferenceStartupPlan) -> str:
    """``startup_log_fields``를 한 줄 로그로."""
    fields = startup_log_fields(plan)
    return " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if key != "joint_feature_contract"
    )


def describe_plan(plan: InferenceStartupPlan) -> str:
    return (
        f"action_representation={plan.mode} pose_format={plan.pose_format} "
        f"client={plan.client_kind} state/action={plan.state_dim}/{plan.action_dim} "
        f"routing={'->'.join(plan.contract.routing)} "
        f"manifest={plan.contract.manifest_sha256[:12]}"
    )


__all__ = [
    "INFERENCE_STARTUP_VERSION",
    "PLATFORM_COMMAND_DTYPE",
    "InferenceStartupPlan",
    "MissingManifestError",
    "build_router",
    "describe_plan",
    "format_startup_log",
    "startup_log_fields",
    "eef_pose_residual",
    "lerobot_state_features",
    "observation_to_manifest_format",
    "plan_inference_startup",
    "validate_manifest_kinematics",
]
