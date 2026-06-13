"""Custom action terms shared by SimToReal pick-and-place tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


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
        if max_velocity is None:
            return None

        if isinstance(max_velocity, (float, int)):
            max_velocity_value = float(max_velocity)
            if max_velocity_value <= 0.0:
                return None
            max_velocity_tensor = torch.full(
                (self.num_envs, self.action_dim),
                max_velocity_value,
                dtype=torch.float32,
                device=self.device,
            )
        elif isinstance(max_velocity, dict):
            max_velocity_tensor = torch.full(
                (self.num_envs, self.action_dim),
                float("inf"),
                dtype=torch.float32,
                device=self.device,
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(max_velocity, self._joint_names)
            for index, value in zip(index_list, value_list, strict=True):
                value = float(value)
                max_velocity_tensor[:, index] = float("inf") if value <= 0.0 else value
        else:
            raise ValueError(
                "Unsupported max_velocity type: "
                f"{type(max_velocity)}. Supported types are float, dict, and None."
            )

        policy_dt = float(self._env.cfg.sim.dt * self._env.cfg.decimation)
        return max_velocity_tensor * policy_dt

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
