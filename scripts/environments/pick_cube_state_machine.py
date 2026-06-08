"""PickCube rule-based state machine (Isaac Lab DifferentialIKController).

Isaac Sim/Lab 표준 pick-and-place 패턴을 따른다:
  - end-effector "pose"(위치 + 자세)를 명령한다. 자세를 top-down 으로 고정해
    SO-101 의 고정 finger 가 큐브 윗면을 찌르는 일을 막는다.
  - IK 는 Isaac Lab `DifferentialIKController`(damped least-squares). 현재 pose 에서
    점진적으로 푸므로 ikpy 처럼 해가 튀지 않는다.
  - 속도 균일화는 환경의 `SlewLimitedJointPositionAction`(max_velocity)이 담당한다.
    state machine 은 매 step 목표 joint position 만 보내면 된다.

진단(diagnostic) 출력을 풍부히 넣어 grasp 자세를 GUI 로 보며 튜닝한다.
"""

from __future__ import annotations

import argparse
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="PickCube state machine (DifferentialIK)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--output_json", type=Path, default=Path("outputs/pick_cube_state_machine.json"))

# Domain randomization
parser.add_argument("--object_radius_scale", type=float, default=1.0)
parser.add_argument("--container_angle_scale", type=float, default=1.0)
parser.add_argument("--container_radius_scale", type=float, default=1.0)

# 처리 순서
parser.add_argument("--object_order", choices=["raster", "name", "near_bowl_first", "far_bowl_first"], default="raster")
parser.add_argument("--object_cycles", type=int, default=1)
parser.add_argument("--raster_row_band", type=float, default=0.06)

# Phase 스텝 상한 (early-exit 가능)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--approach_steps", type=int, default=120)
parser.add_argument("--descend_steps", type=int, default=160)
parser.add_argument("--close_steps", type=int, default=40)
parser.add_argument("--lift_steps", type=int, default=120)
parser.add_argument("--transport_steps", type=int, default=160)
parser.add_argument("--place_steps", type=int, default=120)
parser.add_argument("--open_steps", type=int, default=30)
parser.add_argument("--retreat_steps", type=int, default=100)
parser.add_argument("--max_grasp_attempts", type=int, default=3)

# 높이/오프셋 (m)
parser.add_argument("--approach_height", type=float, default=0.12, help="큐브 중심 위 pre-pick 높이")
parser.add_argument("--grasp_z_offset", type=float, default=0.0, help="grasp 목표 z = 큐브 중심 + 이 값")
parser.add_argument("--lift_height", type=float, default=0.12)
parser.add_argument("--transport_height", type=float, default=0.12, help="그릇 위 수송 높이")
parser.add_argument("--place_height", type=float, default=0.05, help="그릇 안 release 높이")
parser.add_argument("--stack_place_height_increment", type=float, default=0.02)
parser.add_argument("--bowl_place_offset_radius", type=float, default=0.022)

# Grasp 자세 (world top-down 기준에서 보정). 진단 보고 튜닝.
parser.add_argument("--grasp_yaw_deg", type=float, default=0.0, help="top-down 자세에 world Z축 yaw 추가(도)")
parser.add_argument("--grasp_tilt_deg", type=float, default=0.0, help="수직(-Z)에서 앞으로 기울이는 각도(도)")

# IK / 허용 오차
parser.add_argument("--ik_lambda", type=float, default=0.1, help="DLS damping (클수록 안정/느림)")
parser.add_argument("--pos_tolerance", type=float, default=0.012, help="position early-exit 허용오차(m)")
parser.add_argument("--descend_tolerance", type=float, default=0.010)

# 그리퍼 (joint target, rad 또는 normalized)
parser.add_argument("--gripper_open", type=float, default=1.0)
parser.add_argument("--gripper_closed", type=float, default=0.0)
parser.add_argument("--lift_min_height", type=float, default=0.06, help="lift 성공 판정 최소 상승(m)")

# 그리퍼 effort
parser.add_argument("--disable_dynamic_gripper_effort", action="store_true")
parser.add_argument("--min_gripper_effort", type=float, default=0.5)
parser.add_argument("--carry_min_gripper_effort", type=float, default=0.5)

# 진단
parser.add_argument("--probe", action="store_true", help="settle 후 자세 진단만 출력하고 종료")
parser.add_argument("--phase_log", action="store_true", help="매 phase 진단 출력 (--verbose는 AppLauncher 예약)")

# backward compat (무시)
parser.add_argument("--continuity_weight", type=float, default=0.05)
parser.add_argument("--max_arm_step_delta", type=float, default=0.0)
parser.add_argument("--grasp_arm_step_delta", type=float, default=0.0)

AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
args.headless = not args.gui
if args.livestream < 0:
    args.livestream = 0
args.enable_cameras = False

if args.num_envs != 1:
    raise ValueError("pick_cube_state_machine.py: --num_envs=1 만 지원.")

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_matrix,
    quat_inv,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_to_real  # noqa: E402,F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_NAME,
    BOWL_SUCCESS_RADIUS,
    CUBE_NAMES,
    SO101_JOINT_ORDER,
    apply_curriculum,
)
from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim  # noqa: E402


# ── 상수 ──────────────────────────────────────────────────────────────────────

ARM_DOF = 5
CUBE_DESK_TOP_Z = 0.705
CUBE_HALF_Z = 0.0125

EE_BODY_NAME = "jaw"
# jaw body 원점 → grasp point(두 손가락 사이) 오프셋 (jaw local frame).
JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)

BOWL_PLACE_OFFSET_DIRECTIONS = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))


class FSMState(str, Enum):
    SETTLE = "SETTLE"
    OPEN = "OPEN"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"


def _grasp_R_world() -> np.ndarray:
    """Top-down grasp 자세의 world 회전행렬 (jaw body 가 가질 자세).

    1차 추정: JAW_GRASP_OFFSET(주로 -Y)가 world -Z(아래)를 향하도록.
      jaw local +Y → world +Z, +X → world +X, +Z → world -Y.
    --grasp_yaw_deg(world Z 회전), --grasp_tilt_deg(앞으로 기울임)로 보정.
    진단 출력을 보고 조정한다.
    """
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # world Z축 yaw
    yaw = math.radians(args.grasp_yaw_deg)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # world X축 tilt (수직에서 앞으로)
    tilt = math.radians(args.grasp_tilt_deg)
    cx, sx = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return Rz @ Rx @ R


# ── IK solver ─────────────────────────────────────────────────────────────────

class SO101DiffIK:
    """Isaac Lab DifferentialIKController(pose, dls) wrapper.

    Isaac Lab DifferentialInverseKinematicsAction 의 frame 변환 로직을 그대로 재현해,
    grasp point(=jaw body + JAW_GRASP_OFFSET)의 base-frame pose/jacobian 으로 IK 를 푼다.
    """

    def __init__(self, env, device: str) -> None:
        self.env = env
        self.device = device
        self.robot = env.unwrapped.scene["robot"]

        # body / joint 인덱스
        self._ee_body_idx = self.robot.data.body_names.index(EE_BODY_NAME)
        # fixed-base articulation: jacobian 에서 root body 가 빠지므로 -1
        self._jacobi_body_idx = self._ee_body_idx - 1
        self._arm_joint_ids = [self.robot.data.joint_names.index(n) for n in SO101_JOINT_ORDER[:ARM_DOF]]

        self._offset_pos = torch.tensor(JAW_GRASP_OFFSET, device=device, dtype=torch.float32).reshape(1, 3)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32).reshape(1, 4)

        cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(args.ik_lambda)},
        )
        self.controller = DifferentialIKController(cfg, num_envs=1, device=device)

    # -- frame helpers (Isaac Lab task_space_actions 재현) --

    def ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """grasp point pose in robot base frame. (pos[1,3], quat[1,4] wxyz)."""
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        pos_b, quat_b = combine_frame_transforms(pos_b, quat_b, self._offset_pos, self._offset_rot)
        return pos_b, quat_b

    def ee_pos_w(self) -> torch.Tensor:
        """grasp point world position (1,3) = jaw_pos + R(jaw_quat) @ offset."""
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos_w + quat_apply(ee_quat_w, self._offset_pos)

    def ee_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self._ee_body_idx]

    def _jacobian_b(self) -> torch.Tensor:
        jac = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._arm_joint_ids]
        jac = jac.clone()
        base_rot = self.robot.data.root_quat_w
        base_rot_m = matrix_from_quat(quat_inv(base_rot))
        jac[:, :3, :] = torch.bmm(base_rot_m, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(base_rot_m, jac[:, 3:, :])
        # offset 보정 (grasp point 가 jaw body 에서 떨어져 있으므로)
        jac[:, 0:3, :] += torch.bmm(-skew_symmetric_matrix(self._offset_pos), jac[:, 3:, :])
        jac[:, 3:, :] = torch.bmm(matrix_from_quat(self._offset_rot), jac[:, 3:, :])
        return jac

    def solve(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """world target pose → arm joint target (absolute, [5]).

        Returns (q_arm_des[5], pos_err_m, rot_err_rad).
        """
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # world target → base frame
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w.reshape(1, 3), target_quat_w.reshape(1, 4)
        )
        command = torch.cat([tgt_pos_b, tgt_quat_b], dim=-1)  # (1,7)

        ee_pos_b, ee_quat_b = self.ee_pose_b()
        jac = self._jacobian_b()
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        self.controller.set_command(command, ee_pos_b, ee_quat_b)
        q_des = self.controller.compute(ee_pos_b, ee_quat_b, jac, joint_pos)[0]  # (5,)

        # 진단용 오차 (world frame)
        pos_err = float(torch.linalg.norm(self.ee_pos_w()[0] - target_pos_w.reshape(3)).item())
        rot_err = float(quat_error_magnitude(self.ee_quat_w(), target_quat_w.reshape(1, 4))[0].item())
        return q_des, pos_err, rot_err

    @property
    def arm_joint_ids(self) -> list[int]:
        return self._arm_joint_ids


# ── env step / action ─────────────────────────────────────────────────────────

def _gripper_joint_id(robot) -> int:
    return robot.data.joint_names.index(SO101_JOINT_ORDER[ARM_DOF])


def _build_action(robot, ik: SO101DiffIK, q_arm_des: torch.Tensor, gripper_target: float, device: str) -> torch.Tensor:
    """absolute joint target → raw action (use_default_offset 보정).

    env 의 SlewLimitedJointPositionAction 은 raw*scale + default_offset 후 slew limit.
    그러므로 raw = q_des - default_joint_pos.
    action 순서 = SO101_JOINT_ORDER = [arm5, gripper].
    """
    default = robot.data.default_joint_pos[0]
    action = torch.zeros(6, device=device, dtype=torch.float32)
    # arm
    for i, jid in enumerate(ik.arm_joint_ids):
        action[i] = q_arm_des[i] - default[jid]
    # gripper
    gid = _gripper_joint_id(robot)
    action[ARM_DOF] = float(gripper_target) - default[gid]
    return action.reshape(1, 6)


def _step_env(env, action: torch.Tensor, *, min_gripper_effort: float | None = None):
    if not args.disable_dynamic_gripper_effort and getattr(env.unwrapped.cfg, "dynamic_reset_gripper_effort_limit", False):
        dynamic_reset_gripper_effort_limit_sim(
            env.unwrapped,
            "so101leader",
            min_effort=args.min_gripper_effort if min_gripper_effort is None else min_gripper_effort,
        )
    return env.step(action)


# ── state 판정 ────────────────────────────────────────────────────────────────

def _cube_lifted(env, cube_name: str) -> bool:
    cube_z = float(env.unwrapped.scene[cube_name].data.root_pos_w[0, 2].item())
    return cube_z > CUBE_DESK_TOP_Z + CUBE_HALF_Z + args.lift_min_height


def _cube_inside_bowl(env, cube_name: str) -> bool:
    bowl_pos = env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0]
    cube_pos = env.unwrapped.scene[cube_name].data.root_pos_w[0]
    xy = float(torch.linalg.norm(cube_pos[:2] - bowl_pos[:2]).item())
    dz = float((cube_pos[2] - bowl_pos[2]).item())
    return xy <= BOWL_SUCCESS_RADIUS and BOWL_HEIGHT_RANGE[0] <= dz <= BOWL_HEIGHT_RANGE[1]


# ── target suppliers (world frame pos + quat) ─────────────────────────────────

def _grasp_quat_w(device: str) -> torch.Tensor:
    R = torch.tensor(_grasp_R_world(), device=device, dtype=torch.float32).reshape(1, 3, 3)
    return quat_from_matrix(R)  # (1,4)


def _cube_target(env, cube_name: str, dz: float, device: str) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    quat = _grasp_quat_w(device)

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        pos = env.unwrapped.scene[cube_name].data.root_pos_w[0].clone()
        pos = pos.to(device=device)
        pos[2] = pos[2] + dz
        return pos, quat

    return fn


def _bowl_target(env, dz: float, device: str, xy_offset: torch.Tensor | None = None) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    quat = _grasp_quat_w(device)

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        pos = env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0].clone().to(device=device)
        if xy_offset is not None:
            pos[:2] = pos[:2] + xy_offset
        pos[2] = pos[2] + dz
        return pos, quat

    return fn


def _hold_target(pos_w: torch.Tensor, quat_w: torch.Tensor) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    p = pos_w.clone()
    q = quat_w.clone()
    return lambda: (p.clone(), q.clone())


def _bowl_xy_offset(device: str, idx: int) -> torch.Tensor:
    radius = max(0.0, float(args.bowl_place_offset_radius))
    d = BOWL_PLACE_OFFSET_DIRECTIONS[idx % len(BOWL_PLACE_OFFSET_DIRECTIONS)]
    s = radius / math.sqrt(2.0)
    return torch.tensor([d[0] * s, d[1] * s], device=device, dtype=torch.float32)


# ── 큐브 순서 ─────────────────────────────────────────────────────────────────

def _ordered_cubes(env, names: list[str]) -> list[str]:
    scene = env.unwrapped.scene
    names = list(names)
    if args.object_order == "name":
        return names
    if args.object_order in ("near_bowl_first", "far_bowl_first"):
        rev = args.object_order == "far_bowl_first"
        names.sort(key=lambda n: float(scene[n].data.root_pos_w[0, 1].item()), reverse=rev)
        return names
    xy = {n: scene[n].data.root_pos_w[0, :2].detach().cpu() for n in names}
    y_top = max(float(v[1].item()) for v in xy.values())
    band = max(1e-4, float(args.raster_row_band))
    names.sort(key=lambda n: (int((y_top - float(xy[n][1])) // band), float(xy[n][0])))
    return names


# ── phase 실행 ────────────────────────────────────────────────────────────────

def execute_phase(
    env,
    ik: SO101DiffIK,
    device: str,
    name: str,
    target_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    gripper_target: float,
    max_steps: int,
    tolerance: float,
    *,
    min_gripper_effort: float | None = None,
    require_pos: bool = True,
) -> dict[str, Any]:
    """target pose 로 DifferentialIK 이동. position 오차 ≤ tolerance 면 early-exit."""
    robot = env.unwrapped.scene["robot"]
    min_pos_err = float("inf")
    last_pos_err = float("inf")
    last_rot_err = float("inf")
    reached: int | None = None
    step = 0

    for step in range(max(1, int(max_steps))):
        tgt_pos, tgt_quat = target_fn()
        q_arm_des, pos_err, rot_err = ik.solve(tgt_pos, tgt_quat)
        action = _build_action(robot, ik, q_arm_des, gripper_target, device)
        _step_env(env, action, min_gripper_effort=min_gripper_effort)

        min_pos_err = min(min_pos_err, pos_err)
        last_pos_err, last_rot_err = pos_err, rot_err
        if pos_err <= tolerance:
            if reached is None:
                reached = step + 1
            if require_pos:
                break

    stat = {
        "phase": name,
        "steps": step + 1,
        "reached_step": reached,
        "min_pos_err_m": round(min_pos_err, 5),
        "final_pos_err_m": round(last_pos_err, 5),
        "final_rot_err_rad": round(last_rot_err, 4),
        "success": reached is not None,
    }
    if args.phase_log:
        ee = ik.ee_pos_w()[0]
        print(
            f"    [{name}] steps={stat['steps']} pos_err={stat['final_pos_err_m']:.4f} "
            f"rot_err={stat['final_rot_err_rad']:.3f} ee_w=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
        )
    return stat


def hold_pose(env, ik: SO101DiffIK, device: str, gripper_target: float, steps: int, *, min_gripper_effort: float | None = None) -> None:
    """현재 grasp point pose 를 고정한 채 그리퍼만 전환. CLOSE/RELEASE 에서 큐브를 흔들지 않음."""
    pos_w = ik.ee_pos_w()[0].clone()
    quat_w = ik.ee_quat_w()[0].clone()
    tgt = _hold_target(pos_w, quat_w)
    robot = env.unwrapped.scene["robot"]
    for _ in range(max(1, steps)):
        q_arm_des, _, _ = ik.solve(*tgt())
        action = _build_action(robot, ik, q_arm_des, gripper_target, device)
        _step_env(env, action, min_gripper_effort=min_gripper_effort)


# ── 진단 ──────────────────────────────────────────────────────────────────────

def _probe_orientation(env, ik: SO101DiffIK, device: str) -> dict[str, Any]:
    """settle 후 jaw 자세 진단: 각 local 축의 world 방향, grasp point world pos."""
    robot = env.unwrapped.scene["robot"]
    quat_w = ik.ee_quat_w()  # (1,4)
    R = matrix_from_quat(quat_w)[0].detach().cpu().numpy()  # columns = local axes in world
    ee = ik.ee_pos_w()[0].detach().cpu().numpy()
    grasp_R = _grasp_R_world()
    info = {
        "jaw_quat_w": [round(float(v), 4) for v in quat_w[0].tolist()],
        "jaw_x_axis_world": [round(float(v), 3) for v in R[:, 0].tolist()],
        "jaw_y_axis_world": [round(float(v), 3) for v in R[:, 1].tolist()],
        "jaw_z_axis_world": [round(float(v), 3) for v in R[:, 2].tolist()],
        "grasp_point_w": [round(float(v), 4) for v in ee.tolist()],
        "target_grasp_R_world_cols": {
            "x": [round(float(v), 3) for v in grasp_R[:, 0].tolist()],
            "y": [round(float(v), 3) for v in grasp_R[:, 1].tolist()],
            "z": [round(float(v), 3) for v in grasp_R[:, 2].tolist()],
        },
    }
    print("─── ORIENTATION PROBE ───")
    print(json.dumps(info, indent=2))
    return info


# ── episode ───────────────────────────────────────────────────────────────────

def run_episode(env, ik: SO101DiffIK, device: str, active_names: list[str]) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    trace: list[dict[str, Any]] = []
    cube_results: dict[str, Any] = {}

    # SETTLE: 현재 자세 유지하며 물리 안정화
    settle_pos = ik.ee_pos_w()[0].clone()
    settle_quat = ik.ee_quat_w()[0].clone()
    hold = _hold_target(settle_pos, settle_quat)
    for _ in range(args.settle_steps):
        q_arm_des, _, _ = ik.solve(*hold())
        action = _build_action(robot, ik, q_arm_des, args.gripper_open, device)
        _step_env(env, action)

    if args.probe:
        probe = _probe_orientation(env, ik, device)
        return {"success": False, "probe": probe, "cube_results": {}, "trace": []}

    ordered = _ordered_cubes(env, active_names)
    placed_count = 0

    for cube_name in ordered:
        grasped = False

        for attempt in range(1, args.max_grasp_attempts + 1):
            # OPEN
            hold_pose(env, ik, device, args.gripper_open, args.open_steps)
            # APPROACH
            trace.append(execute_phase(
                env, ik, device, f"{cube_name}.approach[{attempt}]",
                _cube_target(env, cube_name, args.approach_height, device),
                args.gripper_open, args.approach_steps, args.pos_tolerance,
            ))
            # DESCEND
            trace.append(execute_phase(
                env, ik, device, f"{cube_name}.descend[{attempt}]",
                _cube_target(env, cube_name, args.grasp_z_offset, device),
                args.gripper_open, args.descend_steps, args.descend_tolerance,
            ))
            # CLOSE (ee pose 고정)
            hold_pose(env, ik, device, args.gripper_closed, args.close_steps,
                      min_gripper_effort=args.carry_min_gripper_effort)
            # LIFT
            trace.append(execute_phase(
                env, ik, device, f"{cube_name}.lift[{attempt}]",
                _cube_target(env, cube_name, args.lift_height, device),
                args.gripper_closed, args.lift_steps, args.pos_tolerance,
                min_gripper_effort=args.carry_min_gripper_effort,
            ))

            if _cube_lifted(env, cube_name):
                grasped = True
                break
            if attempt < args.max_grasp_attempts:
                hold_pose(env, ik, device, args.gripper_open, args.open_steps)

        if not grasped:
            cube_results[cube_name] = {"grasped": False, "placed": False}
            continue

        xy_off = _bowl_xy_offset(device, placed_count)
        # TRANSPORT
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.transport",
            _bowl_target(env, args.transport_height, device, xy_off),
            args.gripper_closed, args.transport_steps, args.pos_tolerance,
            min_gripper_effort=args.carry_min_gripper_effort,
        ))
        # PLACE
        ph = args.place_height + placed_count * args.stack_place_height_increment
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.place",
            _bowl_target(env, ph, device, xy_off),
            args.gripper_closed, args.place_steps, args.pos_tolerance,
            min_gripper_effort=args.carry_min_gripper_effort,
        ))
        # RELEASE (ee 고정, 그리퍼 개방)
        hold_pose(env, ik, device, args.gripper_open, args.open_steps)
        # RETREAT
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.retreat",
            _bowl_target(env, args.transport_height, device),
            args.gripper_open, args.retreat_steps, args.pos_tolerance,
        ))

        placed = _cube_inside_bowl(env, cube_name)
        if placed:
            placed_count += 1
        cube_results[cube_name] = {"grasped": grasped, "placed": placed}

    gripper_open = float(robot.data.joint_pos[0, _gripper_joint_id(robot)].item()) > 0.6
    all_placed = all(cube_results.get(n, {}).get("placed", False) for n in active_names)
    return {
        "success": all_placed and gripper_open,
        "placed_count": placed_count,
        "cube_results": cube_results,
        "trace": trace,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    apply_curriculum(
        env_cfg,
        active_objects=args.active_objects,
        object_radius_scale=args.object_radius_scale,
        container_angle_scale=args.container_angle_scale,
        container_radius_scale=args.container_radius_scale,
    )

    env = gym.make(args.task, cfg=env_cfg)
    device = str(env.unwrapped.device)
    env.reset(seed=args.seed)

    ik = SO101DiffIK(env, device)
    active_names = list(CUBE_NAMES[: args.active_objects])
    all_results = []

    for cycle in range(max(1, args.object_cycles)):
        result = run_episode(env, ik, device, active_names)
        result["cycle"] = cycle
        all_results.append(result)
        if args.probe:
            break
        print(f"[cycle {cycle}] success={result['success']} placed={result['placed_count']}/{len(active_names)}")
        for name, r in result["cube_results"].items():
            print(f"  {name}: grasped={r['grasped']} placed={r['placed']}")
        if cycle < args.object_cycles - 1:
            env.reset(seed=args.seed + cycle + 1)

    env.close()

    out = {
        "task": args.task,
        "seed": args.seed,
        "active_objects": args.active_objects,
        "grasp_yaw_deg": args.grasp_yaw_deg,
        "grasp_tilt_deg": args.grasp_tilt_deg,
        "cycles": all_results,
        "summary": {
            "total_cycles": len(all_results),
            "success_cycles": sum(1 for r in all_results if r.get("success")),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"\n결과 저장: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
