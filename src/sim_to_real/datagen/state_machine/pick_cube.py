"""State machine for the pick-cube manipulation task."""

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from sim_to_real.tasks.common.mdp.terminations import task_done
from sim_to_real.tasks.common.mdp._geometry import JAW_GRASP_OFFSET
from sim_to_real.utils.constant import CUBE_NAMES, BOWL_NAME, CUBE_SIZES
from so101_contract.feature_codec import SO101_JOINT_ORDER

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.1  # vertical clearance for the gripper tip
_APPROACH_STEPS: int = 120  # steps to smoothly interpolate from init EE pos to hover (first cube only)

_REST_POSE_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 90.0,
    "wrist_flex": 50.0,
    "wrist_roll": 0.0,
    "gripper": -10.0,
}


def _apply_bowl_offset(pos_tensor: torch.Tensor, cube_now: int, radius: float = 0.05) -> torch.Tensor:
    """Apply an equilateral-triangle offset on the x-y plane for bowl placement.

    TODO(grasp-tuning): radius and triangle geometry tuned for orange placement.
    For cube placement in bowl, revisit placement zones and offsets.
    """
    idx = (cube_now - 1) % 3
    angle = idx * (2 * math.pi / 3)
    pos_tensor[:, 0] += radius * math.cos(angle)
    pos_tensor[:, 1] += radius * math.sin(angle)
    return pos_tensor


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PickCubeStateMachine(StateMachineBase):
    """State machine for the pick-cube manipulation task.

    The robot cycles through num_cubes cubes. For each cube it executes a fixed
    sequence of steps that moves the gripper above the cube, grasps it, lifts it,
    transports it to the bowl and places it.

    Args:
        num_cubes: Total number of cubes to pick and place. Defaults to 4.
    """

    MAX_STEPS_PER_CUBE: int = 980

    def __init__(self, num_cubes: int = 4) -> None:
        self._num_cubes = num_cubes
        self._step_count: int = 0
        self._cube_now: int = 1
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world: torch.Tensor | None = None
        self._rest_joint_pos: torch.Tensor | None = None
        self._home_start_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        """FK calibration: drive arm to rest pose and record the EE world position.

        Teleports joints to the SO-101 rest pose, steps the simulation once
        to propagate kinematics, then reads ``body_pos_w`` to get the EE world
        position that corresponds to the rest pose. This position is used as
        the home target during the return-home phase so that ``task_done()``
        can verify the arm is within the expected joint-angle tolerances.
        """
        robot = env.scene["robot"]
        joint_names = list(robot.data.joint_names)

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _REST_POSE_DEG:
                self._rest_joint_pos[:, idx] = _REST_POSE_DEG[name] * torch.pi / 180.0

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_world = robot.data.body_pos_w[:, -1, :].clone()

    def check_success(self, env) -> bool:
        """Return True if all cubes are in the bowl and the arm is at rest."""
        robot = env.scene["robot"]
        if self._rest_joint_pos is not None:
            robot.write_joint_state_to_sim(
                position=self._rest_joint_pos,
                velocity=torch.zeros_like(self._rest_joint_pos),
            )
            env.scene.update(dt=env.physics_dt)

        # Construct list of active cube configs based on num_cubes.
        cubes_cfg = [SceneEntityCfg(CUBE_NAMES[i]) for i in range(self._num_cubes)]

        success_tensor = task_done(
            env,
            objects_cfg=cubes_cfg,
            container_cfg=SceneEntityCfg(BOWL_NAME),
        )
        return bool(success_tensor.all().item())

    def pre_step(self, env) -> None:
        """Blend joint state toward rest pose during the final cube's home phase.

        Only active when ``cube_now == num_cubes`` and ``step_count >= 680``.
        Writes blended joint positions directly to the sim so the arm smoothly
        returns home while IK still receives the rest-pose EE target.
        """
        if self._cube_now == self._num_cubes and self._step_count >= 680 and self._rest_joint_pos is not None:
            robot = env.scene["robot"]
            if self._step_count == 680:
                self._home_start_pos = robot.data.joint_pos.clone()
            if self._home_start_pos is not None:
                alpha = min((self._step_count - 680) / 299.0, 1.0)
                blended = self._home_start_pos + (self._rest_joint_pos - self._home_start_pos) * alpha
                robot.write_joint_state_to_sim(position=blended, velocity=torch.zeros_like(blended))

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step (8D IK pose target)."""
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        # Get current cube and bowl positions.
        cube_name = CUBE_NAMES[self._cube_now - 1]
        cube_pos_w = env.scene[cube_name].data.root_pos_w.clone()
        bowl_pos_w = env.scene[BOWL_NAME].data.root_pos_w.clone()
        robot_base_pos_w = robot.data.root_pos_w.clone()
        robot_base_quat_w = robot.data.root_quat_w.clone()

        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        if self._cube_now == 1 and step == 0:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()

        if self._cube_now == 1 and step < _APPROACH_STEPS:
            target_pos_w, gripper_cmd = self._phase_approach_hover(cube_pos_w, num_envs, device)
        elif step < 180:
            target_pos_w, gripper_cmd = self._phase_move_above_cube(cube_pos_w, num_envs, device)
        elif step < 300:
            target_pos_w, gripper_cmd = self._phase_hover_above_cube(cube_pos_w, num_envs, device)
        elif step < 360:
            target_pos_w, gripper_cmd = self._phase_lower_to_cube(cube_pos_w, num_envs, device)
        elif step < 420:
            target_pos_w, gripper_cmd = self._phase_grasp(cube_pos_w, num_envs, device)
        elif step < 500:
            target_pos_w, gripper_cmd = self._phase_lift_cube(cube_pos_w, num_envs, device)
        elif step < 550:
            target_pos_w, gripper_cmd = self._phase_move_above_bowl(bowl_pos_w, num_envs, device)
        elif step < 600:
            target_pos_w, gripper_cmd = self._phase_lower_to_bowl(bowl_pos_w, num_envs, device)
        elif step < 640:
            target_pos_w, gripper_cmd = self._phase_release(bowl_pos_w, num_envs, device)
        elif step < 680:
            target_pos_w, gripper_cmd = self._phase_lift_gripper(bowl_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)
        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    def advance(self) -> None:
        """Advance step counter, handle cube transitions, and fast-forward home phase.

        For non-final cubes, the return-home phase (steps 680–979) is skipped
        without simulation — the arm goes straight to the next cube.
        """
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS_PER_CUBE:
            if self._cube_now >= self._num_cubes:
                self._episode_done = True
            else:
                self._cube_now += 1
                self._step_count = 0
        elif self._cube_now < self._num_cubes and self._step_count >= 680:
            # Fast-forward: skip the home phase for intermediate cubes.
            prev_cube = self._cube_now
            while self._cube_now == prev_cube and not self._episode_done:
                self._step_count += 1
                if self._step_count >= self.MAX_STEPS_PER_CUBE:
                    self._cube_now += 1
                    self._step_count = 0

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._cube_now = 1
        self._episode_done = False
        self._initial_ee_pos = None
        self._home_start_pos = None

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_approach_hover(self, cube_pos_w, num_envs, device):
        """Approach to hover height above cube on first cube only.

        TODO(grasp-tuning): Offsets and heights calibrated for orange (radius ≈ 0.03-0.04).
        Cubes are smaller (40/50mm). Review approach offset and height.
        """
        hover_target = cube_pos_w.clone()
        hover_target[:, 0] -= 0.03
        hover_target[:, 1] -= 0.01
        hover_target[:, 2] += 0.1 + _GRIPPER_OFFSET
        alpha = self._step_count / _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_move_above_cube(self, cube_pos_w, num_envs, device):
        """Move to higher hover position above cube.

        TODO(grasp-tuning): Height offset (0.15m) tuned for orange.
        Cubes in scatter may have different grasp access angle. Revisit.
        """
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.15 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_hover_above_cube(self, cube_pos_w, num_envs, device):
        """Hover directly above cube at intermediate height.

        TODO(grasp-tuning): Fine-tuned hover position for grasp approach.
        """
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lower_to_cube(self, cube_pos_w, num_envs, device):
        """Lower gripper to cube surface.

        TODO(grasp-tuning): This descent uses a static offset from cube center.
        For cube sizes 40-50mm, verify grasp offset (JAW_GRASP_OFFSET) is correct
        and descent doesn't overshoot.
        """
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_grasp(self, cube_pos_w, num_envs, device):
        """Close gripper to grasp cube.

        TODO(grasp-tuning): Grasp position and gripper closing speed.
        Contact forces and cube stability in gripper need empirical verification.
        """
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lift_cube(self, cube_pos_w, num_envs, device):
        """Lift grasped cube vertically.

        TODO(grasp-tuning): Lift height and trajectory need tuning for cube inertia.
        40-50mm cube weight (35-55g) may require different lift speed.
        """
        target_pos_w = cube_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_move_above_bowl(self, bowl_pos_w, num_envs, device):
        """Move to position above bowl.

        TODO(grasp-tuning): Bowl placement zone and approach angle.
        """
        target_pos_w = bowl_pos_w.clone()
        target_pos_w[:, 2] += 0.25
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_lower_to_bowl(self, bowl_pos_w, num_envs, device):
        """Lower cube toward bowl interior with radial offset.

        TODO(grasp-tuning): Placement offset and descent speed for soft placement.
        Current offset (0.05m triangle) was tuned for oranges. Revisit for cubes.
        """
        target_pos_w = bowl_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_bowl_offset(target_pos_w, self._cube_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)

    def _phase_release(self, bowl_pos_w, num_envs, device):
        """Open gripper to release cube into bowl.

        TODO(grasp-tuning): Release timing and residual contact handling.
        """
        target_pos_w = bowl_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_bowl_offset(target_pos_w, self._cube_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_lift_gripper(self, bowl_pos_w, num_envs, device):
        """Lift gripper away from bowl after release.

        TODO(grasp-tuning): Lift trajectory to clear bowl lip and cube.
        """
        target_pos_w = bowl_pos_w.clone()
        target_pos_w[:, 2] += 0.2
        _apply_bowl_offset(target_pos_w, self._cube_now)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    def _phase_return_home(self, num_envs, device):
        """Return arm to rest position.

        TODO(grasp-tuning): Home position and return trajectory tuning.
        """
        if self._rest_ee_pos_world is not None:
            target_pos_w = self._rest_ee_pos_world.clone()
        elif self._initial_ee_pos is not None:
            target_pos_w = self._initial_ee_pos.clone()
        else:
            target_pos_w = torch.zeros(num_envs, 3, device=device)
        return target_pos_w, torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def cube_now(self) -> int:
        return self._cube_now

    @property
    def step_count(self) -> int:
        return self._step_count
