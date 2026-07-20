"""SkillGen source demonstration용 단일 Cube1 pick-place state machine.

이 SM은 Mimic source action 계약인 ``[xyz, qwxyz, gripper]`` 8D absolute
EE target만 출력한다. Robot joint나 object pose를 직접 쓰지 않으며, 실제 접촉/배치
관측이 확인된 뒤에만 cumulative subtask boundary signal을 올린다.
"""

from __future__ import annotations

from enum import IntEnum

import torch

import isaaclab.utils.math as math_utils


class SkillGenPhase(IntEnum):
    PRE_GRASP = 0
    FINAL_GRASP_APPROACH = 1
    CLOSE = 2
    LIFT = 3
    TRANSIT = 4
    FINAL_PLACE_DESCENT = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8
    FAILED = 9


_PHASE_STEPS = {
    SkillGenPhase.PRE_GRASP: 90,
    SkillGenPhase.FINAL_GRASP_APPROACH: 60,
    SkillGenPhase.CLOSE: 45,
    SkillGenPhase.LIFT: 90,
    SkillGenPhase.TRANSIT: 90,
    SkillGenPhase.FINAL_PLACE_DESCENT: 60,
    SkillGenPhase.RELEASE: 75,
    SkillGenPhase.RETREAT: 60,
}


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _nlerp_quat(start: torch.Tensor, goal: torch.Tensor, alpha: float) -> torch.Tensor:
    """Shortest-path normalized quaternion interpolation (wxyz)."""
    goal = torch.where((start * goal).sum(dim=-1, keepdim=True) < 0.0, -goal, goal)
    return torch.nn.functional.normalize(start + alpha * (goal - start), dim=-1)


class SkillGenPickCubeStateMachine:
    """한 환경에서 Cube1을 Bowl로 옮기고 자동 boundary를 기록한다."""

    def __init__(self, grasp_offset_world: tuple[float, float, float] = (-0.03, -0.01, 0.07)) -> None:
        self.grasp_offset_world = grasp_offset_world
        self.phase = SkillGenPhase.PRE_GRASP
        self.phase_step = 0
        self.failure_category: str | None = None
        self.boundary_indices: dict[str, int] = {}
        self.frame_index = 0
        self._phase_start_pos: torch.Tensor | None = None
        self._phase_start_quat: torch.Tensor | None = None
        self._target_quat: torch.Tensor | None = None
        self._world_to_base_quat: torch.Tensor | None = None
        self._cube_pos: torch.Tensor | None = None
        self._bowl_pos: torch.Tensor | None = None
        self._grasp_stable_steps = 0
        self._grasp_lost_steps = 0
        self._place_stable_steps = 0
        self._signals: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def is_done(self) -> bool:
        return self.phase == SkillGenPhase.DONE

    @property
    def is_failed(self) -> bool:
        return self.phase == SkillGenPhase.FAILED

    def reset(self, env) -> None:
        if env.num_envs != 1:
            raise ValueError("automatic SkillGen source SM currently requires num_envs=1")

        self.phase = SkillGenPhase.PRE_GRASP
        self.phase_step = 0
        self.failure_category = None
        self.boundary_indices = {}
        self.frame_index = 0
        self._grasp_stable_steps = 0
        self._grasp_lost_steps = 0
        self._place_stable_steps = 0

        current = env.get_robot_eef_pose("so101")
        self._phase_start_pos, current_rot = math_utils.unmake_pose(current)
        self._phase_start_quat = math_utils.quat_from_matrix(current_rot)

        robot = env.scene["robot"]
        base_quat = robot.data.root_quat_w
        self._world_to_base_quat = math_utils.quat_inv(base_quat)
        world_identity = torch.zeros_like(base_quat)
        world_identity[:, 0] = 1.0
        self._target_quat = math_utils.quat_mul(self._world_to_base_quat, world_identity)

        robot_root = robot.data.root_pos_w
        self._cube_pos = env.scene["Cube1"].data.root_pos_w - robot_root
        self._bowl_pos = env.scene["Bowl"].data.root_pos_w - robot_root
        # env origin과 robot root의 yaw가 고정이므로 world delta를 robot-base 축으로 회전한다.
        self._cube_pos = math_utils.quat_apply(self._world_to_base_quat, self._cube_pos)
        self._bowl_pos = math_utils.quat_apply(self._world_to_base_quat, self._bowl_pos)

        zeros = torch.zeros((1, 1), dtype=torch.float32, device=env.device)
        self._signals = {
            "start": {"grasp_cube": zeros.clone(), "place_cube": zeros.clone()},
            "term": {"grasp_cube": zeros.clone(), "place_cube": zeros.clone()},
        }
        self._publish_signals(env)

    def _publish_signals(self, env) -> None:
        env.set_skillgen_boundary_signals(start=self._signals["start"], term=self._signals["term"])

    def _raise_signal(self, group: str, name: str) -> None:
        signal = self._signals[group][name]
        if not bool(signal[0, 0]):
            signal[0, 0] = 1.0
            self.boundary_indices[f"{name}_{group}"] = self.frame_index

    def _goal(self) -> tuple[torch.Tensor, float]:
        assert self._cube_pos is not None and self._bowl_pos is not None
        assert self._world_to_base_quat is not None

        def world_offset(x: float, y: float, z: float) -> torch.Tensor:
            """기존 검증 궤적의 world-frame offset을 robot-base frame으로 변환한다."""
            offset = self._cube_pos.new_tensor([[x, y, z]])
            return math_utils.quat_apply(self._world_to_base_quat, offset)

        grasp_x, grasp_y, grasp_z = self.grasp_offset_world
        cube_grasp = self._cube_pos + world_offset(grasp_x, grasp_y, grasp_z)
        if self.phase == SkillGenPhase.PRE_GRASP:
            return self._cube_pos + world_offset(grasp_x, grasp_y, max(grasp_z + 0.10, 0.20)), 1.0
        if self.phase == SkillGenPhase.FINAL_GRASP_APPROACH:
            return cube_grasp, 1.0
        if self.phase == SkillGenPhase.CLOSE:
            return cube_grasp, -1.0
        if self.phase == SkillGenPhase.LIFT:
            return self._cube_pos + world_offset(grasp_x, grasp_y, 0.25), -1.0
        if self.phase == SkillGenPhase.TRANSIT:
            # place subtask start는 cuRobo가 attached Cube까지 포함해 충돌 없이
            # 도달해야 한다. EEF보다 약 9 cm 아래 매달린 Cube가 Bowl rim을 완전히
            # 벗어나도록 final descent 시작 pose를 충분히 높게 둔다.
            return self._bowl_pos + world_offset(0.0, 0.0, 0.35), -1.0
        if self.phase == SkillGenPhase.FINAL_PLACE_DESCENT:
            return self._bowl_pos + world_offset(0.0, 0.0, 0.20), -1.0
        if self.phase == SkillGenPhase.RELEASE:
            return self._bowl_pos + world_offset(0.0, 0.0, 0.20), 1.0
        if self.phase == SkillGenPhase.RETREAT:
            return self._bowl_pos + world_offset(0.0, 0.0, 0.28), 1.0
        raise RuntimeError(f"phase {self.phase.name} has no action goal")

    def get_action(self, env) -> torch.Tensor:
        if self.phase in (SkillGenPhase.DONE, SkillGenPhase.FAILED):
            raise RuntimeError(f"cannot request action after {self.phase.name}")
        assert self._phase_start_pos is not None
        assert self._phase_start_quat is not None
        assert self._target_quat is not None

        goal_pos, gripper = self._goal()
        duration = _PHASE_STEPS[self.phase]
        alpha = _smoothstep((self.phase_step + 1) / duration)
        pos = self._phase_start_pos + alpha * (goal_pos - self._phase_start_pos)
        quat = _nlerp_quat(self._phase_start_quat, self._target_quat, alpha)
        gripper_tensor = torch.full((1, 1), gripper, dtype=pos.dtype, device=pos.device)
        action = torch.cat([pos, quat, gripper_tensor], dim=-1)
        if not torch.isfinite(action).all():
            self._fail("ik_tracking")
            raise FloatingPointError("non-finite SkillGen source action")
        return action

    def _enter(self, env, phase: SkillGenPhase) -> None:
        self.phase = phase
        self.phase_step = 0
        if phase not in (SkillGenPhase.DONE, SkillGenPhase.FAILED):
            pose = env.get_robot_eef_pose("so101")
            self._phase_start_pos, rot = math_utils.unmake_pose(pose)
            self._phase_start_quat = math_utils.quat_from_matrix(rot)

        if phase == SkillGenPhase.FINAL_GRASP_APPROACH:
            self._raise_signal("start", "grasp_cube")
        elif phase == SkillGenPhase.FINAL_PLACE_DESCENT:
            self._raise_signal("start", "place_cube")
        self._publish_signals(env)

    def _fail(self, category: str) -> None:
        self.failure_category = category
        self.phase = SkillGenPhase.FAILED

    def after_step(self, env) -> None:
        """실제 관측을 검사하고 phase/boundary를 한 step 진행한다."""
        if self.phase in (SkillGenPhase.DONE, SkillGenPhase.FAILED):
            return

        self.frame_index += 1
        self.phase_step += 1
        subtask_obs = env.obs_buf["subtask_terms"]
        grasped = bool(subtask_obs["cube_grasped"][0].reshape(-1)[0].item())
        placed = bool(subtask_obs["place_cube1"][0].reshape(-1)[0].item())

        if self.phase == SkillGenPhase.LIFT:
            self._grasp_stable_steps = self._grasp_stable_steps + 1 if grasped else 0
            if self._grasp_stable_steps >= 5:
                self._raise_signal("term", "grasp_cube")
        elif self.phase in (SkillGenPhase.TRANSIT, SkillGenPhase.FINAL_PLACE_DESCENT):
            self._grasp_lost_steps = 0 if grasped else self._grasp_lost_steps + 1
            if self._grasp_lost_steps >= 12:
                self._fail("grasp")
                return
        elif self.phase == SkillGenPhase.RELEASE:
            robot = env.scene["robot"]
            gripper_ids, _ = robot.find_joints(["gripper"])
            gripper_open = bool((robot.data.joint_pos[0, gripper_ids[0]] > 0.75).item())
            self._place_stable_steps = self._place_stable_steps + 1 if placed and gripper_open else 0
            if self._place_stable_steps >= 15:
                self._raise_signal("term", "place_cube")

        self._publish_signals(env)

        if self.phase_step < _PHASE_STEPS[self.phase]:
            return

        if self.phase == SkillGenPhase.PRE_GRASP:
            self._enter(env, SkillGenPhase.FINAL_GRASP_APPROACH)
        elif self.phase == SkillGenPhase.FINAL_GRASP_APPROACH:
            self._enter(env, SkillGenPhase.CLOSE)
        elif self.phase == SkillGenPhase.CLOSE:
            self._enter(env, SkillGenPhase.LIFT)
        elif self.phase == SkillGenPhase.LIFT:
            if not bool(self._signals["term"]["grasp_cube"][0, 0]):
                self._fail("grasp")
            else:
                self._enter(env, SkillGenPhase.TRANSIT)
        elif self.phase == SkillGenPhase.TRANSIT:
            self._enter(env, SkillGenPhase.FINAL_PLACE_DESCENT)
        elif self.phase == SkillGenPhase.FINAL_PLACE_DESCENT:
            self._enter(env, SkillGenPhase.RELEASE)
        elif self.phase == SkillGenPhase.RELEASE:
            if not bool(self._signals["term"]["place_cube"][0, 0]):
                self._fail("place")
            else:
                self._enter(env, SkillGenPhase.RETREAT)
        elif self.phase == SkillGenPhase.RETREAT:
            self._enter(env, SkillGenPhase.DONE)

    def boundary_signals(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            group: {name: value.clone() for name, value in signals.items()}
            for group, signals in self._signals.items()
        }
