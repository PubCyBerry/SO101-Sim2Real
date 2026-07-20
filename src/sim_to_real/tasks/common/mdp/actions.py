"""Custom action terms shared by SimToReal pick-and-place tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import (
    BinaryJointPositionAction,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


def _max_velocity_to_step(
    *,
    max_velocity: float | dict[str, float] | None,
    joint_names: list[str],
    num_envs: int,
    device,
    policy_dt: float,
) -> torch.Tensor | None:
    """Joint별 rad/s 상한을 policy-step rad 상한 텐서로 변환한다."""
    if max_velocity is None:
        return None

    if isinstance(max_velocity, (float, int)):
        value = float(max_velocity)
        if value <= 0.0:
            return None
        velocity = torch.full((num_envs, len(joint_names)), value, dtype=torch.float32, device=device)
    elif isinstance(max_velocity, dict):
        velocity = torch.full(
            (num_envs, len(joint_names)), float("inf"), dtype=torch.float32, device=device
        )
        index_list, _, value_list = string_utils.resolve_matching_names_values(max_velocity, joint_names)
        for index, value in zip(index_list, value_list, strict=True):
            value = float(value)
            velocity[:, index] = float("inf") if value <= 0.0 else value
    else:
        raise ValueError(
            "Unsupported max_velocity type: "
            f"{type(max_velocity)}. Supported types are float, dict, and None."
        )
    return velocity * policy_dt


class SlewLimitedJointPositionAction(JointPositionAction):
    """Joint-position action whose processed target changes at a bounded rate."""

    cfg: SlewLimitedJointPositionActionCfg

    def __init__(self, cfg: SlewLimitedJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._max_step = self._parse_max_step(cfg.max_velocity)
        self._limited_actions = self._current_joint_pos().clone()
        self._processed_actions[:] = self._limited_actions

    def _current_joint_pos(self) -> torch.Tensor:
        return self._asset.data.joint_pos[:, self._joint_ids]

    def _parse_max_step(self, max_velocity: float | dict[str, float] | None) -> torch.Tensor | None:
        policy_dt = float(self._env.cfg.sim.dt * self._env.cfg.decimation)
        return _max_velocity_to_step(
            max_velocity=max_velocity,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            device=self.device,
            policy_dt=policy_dt,
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        desired_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            desired_actions = torch.clamp(desired_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        if self._max_step is None:
            self._processed_actions = desired_actions
            return

        delta = torch.clamp(desired_actions - self._limited_actions, -self._max_step, self._max_step)
        self._limited_actions = self._limited_actions + delta
        self._processed_actions = self._limited_actions

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._limited_actions[env_ids] = self._current_joint_pos()[env_ids]
        self._processed_actions[env_ids] = self._limited_actions[env_ids]


@configclass
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`SlewLimitedJointPositionAction`."""

    class_type: type = SlewLimitedJointPositionAction

    max_velocity: float | dict[str, float] | None = None
    """Maximum processed target velocity in rad/s. Set <= 0 or None to disable."""


class SlewLimitedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """DLS IK joint target을 physics substep별 속도 상한으로 제한한다."""

    cfg: "SlewLimitedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "SlewLimitedDifferentialInverseKinematicsActionCfg", env) -> None:
        super().__init__(cfg, env)
        # ``apply_actions``는 policy step당 한 번이 아니라 매 physics substep마다 호출된다.
        # policy_dt를 쓰면 decimation 배수만큼 target slew가 빨라지므로 physics dt를 쓴다.
        physics_dt = float(env.cfg.sim.dt)
        self._max_step = _max_velocity_to_step(
            max_velocity=cfg.max_velocity,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            device=self.device,
            policy_dt=physics_dt,
        )
        self._limited_joint_targets = self._asset.data.joint_pos[:, self._joint_ids].clone()

    @property
    def applied_joint_targets(self) -> torch.Tensor:
        """이번 policy step에 실제 적용한 arm joint target."""
        return self._limited_joint_targets

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            desired = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            desired = joint_pos.clone()

        # Differential IK의 unconstrained update는 특히 wrist_flex에서 articulation
        # limit 밖 target을 낼 수 있다. 실제 적용/기록 action은 항상 soft joint limit
        # 안이어야 canonical LeRobot action 계약과 실기기 안전 범위를 만족한다.
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        desired = torch.clamp(desired, min=joint_limits[..., 0], max=joint_limits[..., 1])

        if self._max_step is None:
            self._limited_joint_targets = desired
        else:
            delta = torch.clamp(
                desired - self._limited_joint_targets,
                -self._max_step,
                self._max_step,
            )
            self._limited_joint_targets = self._limited_joint_targets + delta
        self._asset.set_joint_position_target(self._limited_joint_targets, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # env_ids와 joint_ids가 모두 list이면 PyTorch advanced-indexing이 두 축을
        # pairwise로 묶는다. 먼저 env 축을 고른 뒤 joint 축을 잘라 Cartesian slice를
        # 유지한다.
        self._limited_joint_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class SlewLimitedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """속도 제한 DLS IK action 설정."""

    class_type: type = SlewLimitedDifferentialInverseKinematicsAction
    max_velocity: float | dict[str, float] | None = None


class SlewLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary open/close target 자체를 연속적인 joint target으로 slew-limit한다."""

    cfg: "SlewLimitedBinaryJointPositionActionCfg"

    def __init__(self, cfg: "SlewLimitedBinaryJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        policy_dt = float(env.cfg.sim.dt * env.cfg.decimation)
        self._max_step = _max_velocity_to_step(
            max_velocity=cfg.max_velocity,
            joint_names=self._joint_names,
            num_envs=self.num_envs,
            device=self.device,
            policy_dt=policy_dt,
        )
        self._limited_actions = self._asset.data.joint_pos[:, self._joint_ids].clone()

    @property
    def applied_joint_targets(self) -> torch.Tensor:
        return self._limited_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        desired = self._processed_actions.clone()
        if self._max_step is None:
            self._limited_actions = desired
        else:
            delta = torch.clamp(desired - self._limited_actions, -self._max_step, self._max_step)
            self._limited_actions = self._limited_actions + delta
        self._processed_actions = self._limited_actions

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._limited_actions[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._processed_actions[env_ids] = self._limited_actions[env_ids]


@configclass
class SlewLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """속도 제한 binary joint-position action 설정."""

    class_type: type = SlewLimitedBinaryJointPositionAction
    max_velocity: float | dict[str, float] | None = None
