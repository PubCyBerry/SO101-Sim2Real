"""Phase 16 — mode별 추론 routing (§22.4).

입력은 **이미 postprocess가 끝난 full chunk**다. relative mode의 chunk는 공통
postprocessor(:mod:`so101_contract.action_representation_processor`)가 이미 canonical
absolute로 복원해 놓았다. 이 모듈은 **relative decode를 두 번 하지 않는다.**

.. code-block:: text

    joint_absolute   absolute joint            → joint command
    joint_relative   (복원된) absolute joint   → joint command
    eef_absolute     absolute EEF   → IK       → joint command
    eef_relative     (복원된) absolute EEF → IK → joint command

- joint mode는 IK를 **절대 호출하지 않는다**(``ik_calls == 0``).
- EEF mode는 platform adapter의 sequential IK를 **정확히 1회** 호출한다.
- gripper는 모든 mode에서 absolute passthrough이고, horizon/순서/shape가 보존된다.
- **dtype 경계는 명시적이다.** 입력은 ``float32``/``float64``만 받고(그 외 dtype은 거부),
  platform 명령은 robot/sim codec 계약에 맞춰 항상 ``float32``로 나간다. 결과에
  ``input_dtype``/``platform_dtype``을 기록해 변환을 숨기지 않는다.
- 차원/group 검증과 route 거부는 **command publish 이전에** 끝난다.

real/sim adapter는 같은 계약과 router를 공유하고 물리 FK/IK·joint frame 경계에서만
갈라진다. 좌표 변환은 processor(representation)와 platform adapter(FK/IK/calibration)가
이미 수행하므로 여기서 중복하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .action_checkpoint_contract import ResolvedCheckpointContract
from .action_representation import ActionRepresentationMode, ActionRepresentationSpec, PoseFormat
from .eef_policy_io import EEFJointChunkResult, SO101EEFPolicyIO
from .feature_codec import SO101_JOINT_ORDER
from .joint_feature_codec import (
    canonical_joint_state_to_real_follower,
    feature_to_canonical_joint_state,
    sim_joint_command,
)
from .follower_calibration import real_follower_to_sim_radians, sim_radians_to_real_follower
from .pose_codec import convert_pose_format

ACTION_ROUTING_VERSION = "so101_action_routing_v2"

#: platform 이름. ``real_dry_run``은 motor-off 검토용이며 명령을 만들지 않는다.
PLATFORMS = ("canonical", "sim", "real", "real_dry_run")

_ARM_DOF = 5
_CANONICAL_DOF = len(SO101_JOINT_ORDER)

#: routing 입력으로 허용하는 dtype. 그 외(float16/int/bool)는 거부한다.
SUPPORTED_INPUT_DTYPES = ("float32", "float64")
#: platform 명령 경계 dtype(robot/sim codec 계약).
PLATFORM_COMMAND_DTYPE = "float32"


@dataclass(frozen=True)
class RoutedActionChunk:
    """Routing 결과. 실패면 어떤 명령도 publish하지 않는다."""

    success: bool
    platform: str
    mode: str
    route: tuple[str, ...]
    canonical_joint_radians: np.ndarray | None
    platform_actions: np.ndarray | None
    hold_canonical_joint_radians: np.ndarray
    ik_calls: int
    ik: Any | None
    failed_index: int | None
    reason: str
    replan_required: bool
    input_dtype: str = PLATFORM_COMMAND_DTYPE
    platform_dtype: str = PLATFORM_COMMAND_DTYPE

    @property
    def horizon(self) -> int:
        return 0 if self.canonical_joint_radians is None else int(len(self.canonical_joint_radians))


class ActionRepresentationRouter:
    """검증된 checkpoint 계약으로 absolute chunk를 platform joint command로 라우팅."""

    def __init__(
        self,
        spec: ActionRepresentationSpec,
        *,
        action_dim: int,
        transform_indices: tuple[int, ...],
        passthrough_indices: tuple[int, ...],
        policy_io: SO101EEFPolicyIO | None = None,
        adapter_config: Any | None = None,
    ) -> None:
        if not isinstance(spec, ActionRepresentationSpec):
            raise TypeError(f"spec must be ActionRepresentationSpec, got {type(spec).__name__}")
        self.spec = spec
        self.action_dim = int(action_dim)
        self.transform_indices = tuple(int(index) for index in transform_indices)
        self.passthrough_indices = tuple(int(index) for index in passthrough_indices)
        self.policy_io = policy_io
        # joint mode는 IK adapter가 없어도 step-limit 계약을 쓸 수 있어야 한다.
        self.adapter_config = adapter_config or (
            policy_io.config if policy_io is not None else None
        )

        classified = set(self.transform_indices) | set(self.passthrough_indices)
        if classified != set(range(self.action_dim)):
            raise ValueError(
                "router indices do not classify the action vector: "
                f"{sorted(classified)} != {list(range(self.action_dim))}"
            )
        if len(self.passthrough_indices) != 1:
            raise ValueError(
                "SO-101 routing expects exactly one gripper passthrough index, got "
                f"{self.passthrough_indices}"
            )
        if spec.is_eef:
            if policy_io is None:
                raise ValueError(
                    f"mode={spec.mode.value!r} routes through IK and requires a platform adapter"
                )
            if len(self.transform_indices) != spec.pose_dim:
                raise ValueError(
                    f"EEF pose group must be {spec.pose_dim}D, got {len(self.transform_indices)}"
                )
        elif len(self.transform_indices) != _ARM_DOF:
            raise ValueError(
                f"SO-101 joint routing expects {_ARM_DOF} arm joints, got "
                f"{len(self.transform_indices)}"
            )

    # --- 생성 ---------------------------------------------------------------

    @classmethod
    def from_contract(
        cls,
        contract: ResolvedCheckpointContract,
        *,
        policy_io: SO101EEFPolicyIO | None = None,
        adapter_config: Any | None = None,
    ) -> "ActionRepresentationRouter":
        return cls(
            contract.spec,
            action_dim=contract.action_dim,
            transform_indices=contract.transform_indices,
            passthrough_indices=contract.passthrough_indices,
            policy_io=policy_io,
            adapter_config=adapter_config,
        )

    # --- 검증 ---------------------------------------------------------------

    def _validate_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, str]:
        """dtype/shape 계약 검증.

        허용 dtype(float32/float64)만 받고, platform 경계 dtype(float32)으로 명시적으로
        변환한 배열과 원래 dtype 이름을 함께 돌려준다.
        """
        raw = np.asarray(chunk)
        input_dtype = str(raw.dtype)
        if input_dtype not in SUPPORTED_INPUT_DTYPES:
            raise TypeError(
                f"routed chunk dtype {input_dtype!r} is not supported; expected one of "
                f"{list(SUPPORTED_INPUT_DTYPES)} (the platform command boundary is "
                f"{PLATFORM_COMMAND_DTYPE})"
            )
        actions = raw.astype(np.float32, copy=False)
        if actions.ndim != 2:
            raise ValueError(f"routed chunk must be (H,D), got shape {actions.shape}")
        if actions.shape[0] == 0:
            raise ValueError("routed chunk is empty")
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                f"routed chunk action dim {actions.shape[1]} != contract {self.action_dim}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("routed chunk contains NaN or infinity")
        return actions, input_dtype

    def _failure(
        self,
        *,
        platform: str,
        current: np.ndarray,
        reason: str,
        failed_index: int | None,
        ik_calls: int,
        ik: Any | None = None,
    ) -> RoutedActionChunk:
        return RoutedActionChunk(
            success=False,
            platform=platform,
            mode=self.spec.mode.value,
            route=self.spec.inference_routing,
            canonical_joint_radians=None,
            platform_actions=None,
            hold_canonical_joint_radians=np.asarray(current, dtype=np.float32).copy(),
            ik_calls=ik_calls,
            ik=ik,
            failed_index=failed_index,
            reason=reason,
            replan_required=True,
        )

    # --- joint route --------------------------------------------------------

    def _joint_to_canonical(
        self,
        actions: np.ndarray,
        current: np.ndarray,
        *,
        platform: str,
    ) -> RoutedActionChunk:
        """joint mode: IK 없이 canonical 6D joint radian target을 만든다.

        v2 joint feature = arm radian(변환 없음) + gripper policy feature. 공용 codec
        (:mod:`so101_contract.joint_feature_codec`)을 써서 legacy degree 변환을 배제한다.
        """
        arm = actions[:, list(self.transform_indices)].astype(np.float32)
        gripper_feature = actions[:, self.passthrough_indices[0] : self.passthrough_indices[0] + 1]
        feature_rows = np.concatenate([arm, gripper_feature.astype(np.float32)], axis=1)
        raw_targets = feature_to_canonical_joint_state(feature_rows, clamp=False)
        # joint limit clamp는 물리 한계이며, 값이 바뀌면 조용히 넘기지 않고 reason에 남긴다.
        canonical = feature_to_canonical_joint_state(feature_rows, clamp=True)

        config = self.adapter_config
        clamped = int(np.count_nonzero(canonical != raw_targets))
        if config is not None:
            arm_limits = config.arm_step_array
            previous = np.asarray(current, dtype=np.float64)
            for index, row in enumerate(canonical.astype(np.float64)):
                arm_step = np.abs(row[:_ARM_DOF] - previous[:_ARM_DOF])
                if np.any(arm_step > arm_limits + 1e-7):
                    return self._failure(
                        platform=platform,
                        current=current,
                        reason=f"arm_step_limit:{index}:{float(np.max(arm_step)):.6f}",
                        failed_index=index,
                        ik_calls=0,
                    )
                gripper_step = abs(float(row[5]) - float(previous[5]))
                if gripper_step > config.max_gripper_step_rad + 1e-7:
                    return self._failure(
                        platform=platform,
                        current=current,
                        reason=f"gripper_step_limit:{index}:{gripper_step:.6f}",
                        failed_index=index,
                        ik_calls=0,
                    )
                previous = row
        return RoutedActionChunk(
            success=True,
            platform=platform,
            mode=self.spec.mode.value,
            route=self.spec.inference_routing,
            canonical_joint_radians=canonical,
            platform_actions=None,
            hold_canonical_joint_radians=np.asarray(current, dtype=np.float32).copy(),
            ik_calls=0,
            ik=None,
            failed_index=None,
            reason="ok" if clamped == 0 else f"ok:joint_limit_clamped={clamped}",
            replan_required=False,
        )

    # --- EEF route ----------------------------------------------------------

    def _eef_chunk_as_rot6d(self, actions: np.ndarray) -> np.ndarray:
        """어떤 pose format이든 platform adapter가 쓰는 canonical 10D로 맞춘다.

        회전 표현 변환일 뿐이며 frame이나 상대/절대 의미는 바뀌지 않는다.
        """
        pose = actions[:, list(self.transform_indices)]
        if self.spec.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
            pose = convert_pose_format(pose, self.spec.pose_format, PoseFormat.XYZ_ROT6D_ROWS)
        gripper = actions[:, self.passthrough_indices[0] : self.passthrough_indices[0] + 1]
        return np.concatenate(
            [np.asarray(pose, dtype=np.float32), gripper.astype(np.float32)],
            axis=1,
        )

    # --- 공개 API -----------------------------------------------------------

    def route(
        self,
        absolute_chunk: np.ndarray,
        current_state: np.ndarray,
        *,
        platform: str = "sim",
    ) -> RoutedActionChunk:
        """이미 absolute로 복원된 full chunk를 platform joint command로 변환.

        Args:
            absolute_chunk: ``(H, action_dim)``. relative mode라도 **이미 복원된** 값이다.
            current_state: platform별 현재 상태. ``sim``/``canonical``은 6D radian,
                ``real``/``real_dry_run``은 실 follower state.
        """
        if platform not in PLATFORMS:
            raise ValueError(f"unknown platform {platform!r}; expected one of {list(PLATFORMS)}")
        actions, input_dtype = self._validate_chunk(absolute_chunk)

        if platform in ("real", "real_dry_run"):
            current_canonical = np.asarray(
                real_follower_to_sim_radians(current_state),
                dtype=np.float32,
            )
        else:
            current_canonical = np.asarray(current_state, dtype=np.float32)
        if current_canonical.shape != (_CANONICAL_DOF,) or not np.all(
            np.isfinite(current_canonical)
        ):
            raise ValueError(
                f"current canonical joint state must be finite ({_CANONICAL_DOF},), got "
                f"{current_canonical.shape}"
            )

        if self.spec.is_eef:
            result = self._route_eef(actions, current_state, current_canonical, platform)
        else:
            result = self._joint_to_canonical(actions, current_canonical, platform=platform)
            if result.success:
                result = replace(
                    result,
                    platform_actions=self._platform_actions(
                        result.canonical_joint_radians,
                        platform,
                    ),
                )
        result = replace(result, input_dtype=input_dtype, platform_dtype=PLATFORM_COMMAND_DTYPE)
        if result.success and result.canonical_joint_radians is not None:
            if result.canonical_joint_radians.dtype != np.float32:
                raise AssertionError("platform command must use the float32 boundary dtype")
            if result.canonical_joint_radians.shape != (len(actions), _CANONICAL_DOF):
                raise AssertionError(
                    "routing changed the chunk horizon: "
                    f"{result.canonical_joint_radians.shape} != {(len(actions), _CANONICAL_DOF)}"
                )
        return result

    def _route_eef(
        self,
        actions: np.ndarray,
        current_state: np.ndarray,
        current_canonical: np.ndarray,
        platform: str,
    ) -> RoutedActionChunk:
        rot6d_chunk = self._eef_chunk_as_rot6d(actions)
        if platform == "real":
            conversion = self.policy_io.action_chunk_to_real(rot6d_chunk, current_state)
        elif platform == "real_dry_run":
            conversion = self.policy_io.action_chunk_to_real_dry_run(rot6d_chunk, current_state)
        elif platform == "sim":
            conversion = self.policy_io.action_chunk_to_sim(rot6d_chunk, current_canonical)
        else:
            conversion = self.policy_io.action_chunk_to_canonical(
                rot6d_chunk,
                current_canonical,
                platform="canonical",
            )
        return self._from_eef_result(conversion, ik_calls=1)

    def _from_eef_result(
        self,
        conversion: EEFJointChunkResult,
        *,
        ik_calls: int,
    ) -> RoutedActionChunk:
        return RoutedActionChunk(
            success=conversion.success,
            platform=conversion.platform,
            mode=self.spec.mode.value,
            route=self.spec.inference_routing,
            canonical_joint_radians=conversion.canonical_joint_radians,
            platform_actions=conversion.platform_actions,
            hold_canonical_joint_radians=conversion.hold_canonical_joint_radians,
            ik_calls=ik_calls,
            ik=conversion.ik,
            failed_index=conversion.failed_index,
            reason=conversion.reason,
            replan_required=conversion.replan_required,
        )

    @staticmethod
    def _platform_actions(canonical: np.ndarray, platform: str) -> np.ndarray:
        """joint mode의 platform 경계 변환. 경계 변환은 정확히 1회다.

        sim = canonical radian 그대로, real = follower frame 1회 변환.
        """
        if platform in ("real", "real_dry_run"):
            return canonical_joint_state_to_real_follower(canonical)
        return sim_joint_command(canonical)


def make_router(
    contract: ResolvedCheckpointContract,
    *,
    urdf_path: str | None = None,
    robot_yaml_path: str | None = None,
    policy_io: SO101EEFPolicyIO | None = None,
    ik_config: Any | None = None,
    adapter_config: Any | None = None,
) -> ActionRepresentationRouter:
    """계약에 맞는 router를 만든다. EEF mode에서만 FK/IK adapter를 요구한다."""
    if contract.spec.is_eef and policy_io is None:
        if not urdf_path or not robot_yaml_path:
            raise ValueError(
                f"mode={contract.spec.mode.value!r} routes through IK; provide the URDF and "
                "robot YAML paths or a prepared SO101EEFPolicyIO"
            )
        policy_io = SO101EEFPolicyIO.from_files(
            urdf_path,
            robot_yaml_path,
            ik_config=ik_config,
            config=adapter_config,
        )
    return ActionRepresentationRouter.from_contract(
        contract,
        policy_io=policy_io,
        adapter_config=adapter_config,
    )
