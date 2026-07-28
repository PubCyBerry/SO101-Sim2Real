"""Real/sim 공통 SO-101 absolute EEF policy I/O adapter.

Policy server의 외부 계약은 canonical absolute EEF 10D다.

``[tcp_grasp xyz(3), Rot6D rows(6), absolute gripper feature(1)]``

이 모듈이 platform 차이를 경계에서만 처리한다.

- observation: platform joint state → canonical sim radian → FK → EEF 10D
- action: absolute EEF chunk → bounded sequential IK → platform joint chunk
- any-step failure: chunk 전체 폐기, hold/replan 요청
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .eef_action_contract import CANONICAL_ACTION_DIM
from .eef_ik import IKConfig, SequentialIKResult, SO101BoundedIK
from .feature_codec import (
    POLICY_GRIPPER_RANGE,
    SO101_JOINT_ORDER,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from .follower_calibration import (
    real_follower_to_sim_radians,
    sim_radians_to_real_follower,
)

EEF_POLICY_IO_VERSION = "so101_eef_platform_adapter_v1"


@dataclass(frozen=True)
class EEFPlatformAdapterConfig:
    """실행 queue에 들어가기 전 적용할 joint safety 계약."""

    max_arm_step_rad: float | tuple[float, ...] = 5.0 / 30.0
    max_gripper_step_rad: float = 2.5 / 30.0
    gripper_feature_tolerance: float = 1e-3
    real_hardware_ik_validated: bool = False

    def __post_init__(self) -> None:
        arm = np.broadcast_to(np.asarray(self.max_arm_step_rad, dtype=np.float64), (5,))
        if not np.all(np.isfinite(arm)) or np.any(arm <= 0):
            raise ValueError("max_arm_step_rad must be finite and positive")
        if self.max_gripper_step_rad <= 0 or self.gripper_feature_tolerance < 0:
            raise ValueError("gripper safety limits must be positive/non-negative")

    @property
    def arm_step_array(self) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(self.max_arm_step_rad, dtype=np.float64),
            (5,),
        ).copy()


@dataclass(frozen=True)
class EEFJointChunkResult:
    """원자적 EEF→joint chunk 변환 결과."""

    success: bool
    platform: str
    canonical_joint_radians: np.ndarray | None
    platform_actions: np.ndarray | None
    hold_canonical_joint_radians: np.ndarray
    ik: SequentialIKResult | None
    failed_index: int | None
    reason: str
    replan_required: bool


class SO101EEFPolicyIO:
    """Dataset converter, sim node, real client가 공유하는 FK/IK adapter."""

    def __init__(
        self,
        ik: SO101BoundedIK,
        *,
        config: EEFPlatformAdapterConfig | None = None,
    ) -> None:
        self.ik = ik
        self.kinematics = ik.kinematics
        self.config = config or EEFPlatformAdapterConfig()

    @classmethod
    def from_files(
        cls,
        urdf_path: str | Path,
        robot_yaml_path: str | Path,
        *,
        ik_config: IKConfig | None = None,
        config: EEFPlatformAdapterConfig | None = None,
    ) -> "SO101EEFPolicyIO":
        return cls(
            SO101BoundedIK.from_files(
                urdf_path,
                robot_yaml_path,
                config=ik_config,
            ),
            config=config,
        )

    @staticmethod
    def _canonical_joint_array(values: np.ndarray) -> np.ndarray:
        joints = np.asarray(values, dtype=np.float32)
        if joints.shape != (len(SO101_JOINT_ORDER),) or not np.all(np.isfinite(joints)):
            raise ValueError(f"canonical joint state must be finite (6,), got {joints.shape}")
        return joints

    def observation_from_canonical_radians(self, values_rad: np.ndarray) -> np.ndarray:
        joints = self._canonical_joint_array(values_rad)
        pose = self.kinematics.forward_xyz_rot6d(joints[:5])
        gripper = sim_joint_radians_to_policy_feature(joints)[5:6]
        return np.concatenate([pose, gripper]).astype(np.float32)

    def observation_from_sim(self, sim_joint_radians: np.ndarray) -> np.ndarray:
        return self.observation_from_canonical_radians(sim_joint_radians)

    def observation_from_real(self, real_follower_state: np.ndarray) -> np.ndarray:
        return self.observation_from_canonical_radians(
            real_follower_to_sim_radians(real_follower_state)
        )

    def _validate_action_chunk(self, absolute_eef_chunk: np.ndarray) -> np.ndarray:
        actions = np.asarray(absolute_eef_chunk, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != CANONICAL_ACTION_DIM:
            raise ValueError(
                f"absolute EEF action chunk must be (H,{CANONICAL_ACTION_DIM}), got {actions.shape}"
            )
        if actions.shape[0] == 0 or not np.all(np.isfinite(actions)):
            raise ValueError("absolute EEF action chunk must be non-empty and finite")
        gripper_lo, gripper_hi = POLICY_GRIPPER_RANGE
        tolerance = self.config.gripper_feature_tolerance
        if np.any(actions[:, 9] < gripper_lo - tolerance) or np.any(
            actions[:, 9] > gripper_hi + tolerance
        ):
            raise ValueError(
                f"gripper feature is outside [{gripper_lo},{gripper_hi}] beyond tolerance"
            )
        return actions

    def _failure(
        self,
        *,
        platform: str,
        current: np.ndarray,
        reason: str,
        failed_index: int | None,
        ik: SequentialIKResult | None,
    ) -> EEFJointChunkResult:
        return EEFJointChunkResult(
            success=False,
            platform=platform,
            canonical_joint_radians=None,
            platform_actions=None,
            hold_canonical_joint_radians=current.copy(),
            ik=ik,
            failed_index=failed_index,
            reason=reason,
            replan_required=True,
        )

    def action_chunk_to_canonical(
        self,
        absolute_eef_chunk: np.ndarray,
        current_canonical_joint_radians: np.ndarray,
        *,
        platform: str,
    ) -> EEFJointChunkResult:
        actions = self._validate_action_chunk(absolute_eef_chunk)
        current = self._canonical_joint_array(current_canonical_joint_radians)
        ik_result = self.ik.solve_chunk(
            actions[:, :9],
            current[:5],
            representation="rot6d",
            max_joint_step_rad=self.config.arm_step_array,
        )
        if not ik_result.success or ik_result.joint_radians is None:
            return self._failure(
                platform=platform,
                current=current,
                reason=f"ik_failed:{ik_result.reason}",
                failed_index=ik_result.failed_index,
                ik=ik_result,
            )

        gripper_features = np.clip(
            actions[:, 9],
            POLICY_GRIPPER_RANGE[0],
            POLICY_GRIPPER_RANGE[1],
        )
        dummy_features = np.zeros((len(actions), 6), dtype=np.float32)
        dummy_features[:, 5] = gripper_features
        gripper_radians = policy_feature_to_sim_joint_radians(dummy_features)[:, 5]
        previous_gripper = float(current[5])
        for index, target in enumerate(gripper_radians):
            if abs(float(target) - previous_gripper) > self.config.max_gripper_step_rad + 1e-7:
                return self._failure(
                    platform=platform,
                    current=current,
                    reason=(
                        f"gripper_step_limit:{index}:"
                        f"{abs(float(target) - previous_gripper):.6f}>"
                        f"{self.config.max_gripper_step_rad:.6f}"
                    ),
                    failed_index=index,
                    ik=ik_result,
                )
            previous_gripper = float(target)

        canonical = np.concatenate(
            [
                np.asarray(ik_result.joint_radians, dtype=np.float32),
                gripper_radians[:, None].astype(np.float32),
            ],
            axis=1,
        )
        return EEFJointChunkResult(
            success=True,
            platform=platform,
            canonical_joint_radians=canonical,
            platform_actions=None,
            hold_canonical_joint_radians=current.copy(),
            ik=ik_result,
            failed_index=None,
            reason="ok",
            replan_required=False,
        )

    def action_chunk_to_sim(
        self,
        absolute_eef_chunk: np.ndarray,
        current_sim_joint_radians: np.ndarray,
    ) -> EEFJointChunkResult:
        result = self.action_chunk_to_canonical(
            absolute_eef_chunk,
            current_sim_joint_radians,
            platform="sim",
        )
        if not result.success:
            return result
        return EEFJointChunkResult(
            **{
                **result.__dict__,
                "platform_actions": np.asarray(
                    result.canonical_joint_radians,
                    dtype=np.float32,
                ).copy(),
            }
        )

    def action_chunk_to_real(
        self,
        absolute_eef_chunk: np.ndarray,
        current_real_follower_state: np.ndarray,
    ) -> EEFJointChunkResult:
        if not self.config.real_hardware_ik_validated:
            current = real_follower_to_sim_radians(current_real_follower_state)
            return self._failure(
                platform="real",
                current=current,
                reason="real_hardware_ik_not_validated",
                failed_index=None,
                ik=None,
            )
        current = real_follower_to_sim_radians(current_real_follower_state)
        result = self.action_chunk_to_canonical(
            absolute_eef_chunk,
            current,
            platform="real",
        )
        if not result.success:
            return result
        return EEFJointChunkResult(
            **{
                **result.__dict__,
                "platform_actions": sim_radians_to_real_follower(
                    result.canonical_joint_radians
                ),
            }
        )

    def action_chunk_to_real_dry_run(
        self,
        absolute_eef_chunk: np.ndarray,
        current_real_follower_state: np.ndarray,
    ) -> EEFJointChunkResult:
        """Hardware command를 만들지 않는 motor-off/target-log용 real 변환.

        반환 target은 offline 검토에만 사용한다. 실제 queue에는
        :meth:`action_chunk_to_real`의 validation gate를 통과한 결과만 넣어야 한다.
        """
        current = real_follower_to_sim_radians(current_real_follower_state)
        result = self.action_chunk_to_canonical(
            absolute_eef_chunk,
            current,
            platform="real_dry_run",
        )
        if not result.success:
            return result
        return EEFJointChunkResult(
            **{
                **result.__dict__,
                "platform_actions": sim_radians_to_real_follower(
                    result.canonical_joint_radians
                ),
            }
        )
