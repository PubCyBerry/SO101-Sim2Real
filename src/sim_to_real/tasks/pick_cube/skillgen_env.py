"""SO-101 PickCube용 Isaac Lab Mimic/SkillGen 환경."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLMimicEnv

from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeSkillGenEnv(ManagerBasedRLMimicEnv):
    """단일 SO-101 end-effector의 absolute-pose Mimic 계약을 구현한다."""

    _EEF_NAME = "so101"

    def step(self, action):
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if eef_name != self._EEF_NAME:
            raise KeyError(f"unknown end effector {eef_name!r}; expected {self._EEF_NAME!r}")
        if env_ids is None:
            env_ids = slice(None)

        robot = self.scene["robot"]
        frame = self.scene["ee_frame"]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w[env_ids],
            robot.data.root_quat_w[env_ids],
            frame.data.target_pos_w[env_ids, 0, :],
            frame.data.target_quat_w[env_ids, 0, :],
        )
        return math_utils.make_pose(pos_b, math_utils.matrix_from_quat(quat_b))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict[str, torch.Tensor],
        gripper_action_dict: dict[str, torch.Tensor],
        action_noise_dict: dict[str, float] | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        target = target_eef_pose_dict[self._EEF_NAME]
        pos, rot = math_utils.unmake_pose(target)
        quat = math_utils.quat_from_matrix(rot)
        if action_noise_dict is not None and action_noise_dict.get(self._EEF_NAME, 0.0) > 0.0:
            # SkillGen v1은 noise=0이 기본이다. 활성화 시에도 quaternion을 직접 흔들지 않고
            # position에만 작은 Gaussian noise를 넣어 unit-quaternion 계약을 지킨다.
            pos = pos + float(action_noise_dict[self._EEF_NAME]) * torch.randn_like(pos)
        gripper = gripper_action_dict[self._EEF_NAME].reshape(-1)
        return torch.cat([pos.reshape(-1), quat.reshape(-1), gripper[:1]], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if action.shape[-1] != 8:
            raise ValueError(f"SkillGen action must be 8D, got {tuple(action.shape)}")
        pos = action[..., :3]
        quat = torch.nn.functional.normalize(action[..., 3:7], dim=-1)
        return {self._EEF_NAME: math_utils.make_pose(pos, math_utils.matrix_from_quat(quat))}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.shape[-1] != 8:
            raise ValueError(f"SkillGen action must be 8D, got {tuple(actions.shape)}")
        return {self._EEF_NAME: actions[..., -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # absolute-pose controller와 source eef pose는 robot base frame을 사용한다.
        # ManagerBasedRLMimicEnv 기본 구현은 env-origin frame이므로 그대로 쓰면
        # scene의 z offset(~0.55 m)과 robot root yaw가 object-centric transform에
        # 중복 반영되어 SkillGen 목표가 작업공간 밖으로 튄다.
        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]
        poses = {}
        for name, state in scene_state["rigid_object"].items():
            pos_b, quat_b = math_utils.subtract_frame_transforms(
                root_pos,
                root_quat,
                state["root_pose"][env_ids, :3],
                state["root_pose"][env_ids, 3:7],
            )
            poses[name] = math_utils.make_pose(pos_b, math_utils.matrix_from_quat(quat_b))
        required = {"Cube1", "Bowl"}
        missing = required - set(poses)
        if missing:
            raise KeyError(f"SkillGen object pose missing: {sorted(missing)}")
        return {name: poses[name] for name in sorted(required)}

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        if eef_name != self._EEF_NAME:
            return None
        return "Cube1" if subtask_index == 1 else None

    def set_skillgen_boundary_signals(
        self,
        *,
        start: dict[str, torch.Tensor],
        term: dict[str, torch.Tensor],
    ) -> None:
        """자동 SM이 알고 있는 정확한 phase boundary를 recorder에 전달한다."""
        self._skillgen_start_signals = start
        self._skillgen_term_signals = term

    def _empty_boundary_signals(self) -> dict[str, torch.Tensor]:
        return {
            "grasp_cube": torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device),
            "place_cube": torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device),
        }

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        signals = getattr(self, "_skillgen_start_signals", self._empty_boundary_signals())
        if env_ids is None:
            return signals
        return {name: value[env_ids] for name, value in signals.items()}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        signals = getattr(self, "_skillgen_term_signals", self._empty_boundary_signals())
        if env_ids is None:
            return signals
        return {name: value[env_ids] for name, value in signals.items()}
