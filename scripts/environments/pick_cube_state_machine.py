"""PickCube rule-based state machine (weighted differential IK).

Isaac Sim/Lab 표준 pick-and-place 패턴을 따른다:
  - end-effector "pose"(위치 + 자세)를 명령한다. 자세를 top-down 으로 유도해
    SO-101 의 고정 finger 가 큐브 윗면을 찌르는 일을 막는다.
  - IK 는 weighted damped-least-squares. Isaac Lab 의 frame 변환 로직을 재현하되,
    orientation 에 가중치(--rot_weight)를 두어 푼다. SO-101 은 5-DOF 라 임의 top-down
    자세+위치를 동시 만족 못 하므로, position 을 우선 도달시키고 자세는 약하게 유도한다.
    현재 pose 에서 점진적으로 푸므로 ikpy 처럼 해가 튀지 않는다.
  - 속도 균일화는 환경의 `SlewLimitedJointPositionAction`(max_velocity)이 담당한다.
    state machine 은 매 step 목표 joint position 만 보내면 된다.

진단(diagnostic) 출력을 풍부히 넣어 grasp 자세를 headless trace 로 보며 튜닝한다.
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
parser.add_argument(
    "--object_order",
    choices=["near_robot", "far_robot", "raster", "name", "near_bowl_first", "far_bowl_first"],
    default="near_robot",
    help="집는 순서. near_robot=로봇팔에서 가까운 큐브부터(기본).",
)
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
parser.add_argument(
    "--grasp_z_offset",
    type=float,
    default=0.005,
    help=(
        "grasp 목표 z = 큐브 중심 + 이 값. z 과주입(음수 깊이)은 비추 — unreachable 목표라 "
        "early-exit 안 걸려 시간↑. tilt를 z가 아니라 자세로 해결(검증 기록)."
    ),
)
parser.add_argument("--lift_height", type=float, default=0.12)
parser.add_argument("--transport_height", type=float, default=0.12, help="그릇 위 수송 높이")
parser.add_argument("--place_height", type=float, default=0.05, help="그릇 안 release 높이")
parser.add_argument("--stack_place_height_increment", type=float, default=0.02)
parser.add_argument("--bowl_place_offset_radius", type=float, default=0.022)

# Grasp 자세 (강tilt — 검증된 성공 공식). SO-101 5-DOF는 top-down 불가, 강tilt로
# 모터 jaw를 큐브 바닥 아래까지 내려 감싼다(CONTEXT.md: down_dot≈0.34, ~70°).
parser.add_argument("--grasp_yaw_deg", type=float, default=0.0, help="grasp 자세에 world Z축 yaw 추가(도)")
parser.add_argument(
    "--grasp_tilt_deg",
    type=float,
    default=60.0,
    help="수직(-Z)에서 앞으로 기울이는 각도(도). 강tilt가 모터 jaw를 큐브 옆/아래로 내림. trace의 jaw_minz로 튜닝.",
)

# descend grasp: 결정론적 random-FK 전역탐색 (모터 jaw를 큐브 아래로 감싸는 자세 선택)
parser.add_argument("--fk_samples", type=int, default=3000, help="random-FK 후보 수 (1/3 local, 2/3 global)")
parser.add_argument(
    "--grasp_tilt_weight",
    type=float,
    default=0.5,
    help=(
        "random-FK 점수에 tilt penalty 가중. 모터 jaw가 floor·고정finger 아래로 내려가는 강tilt "
        "자세를 선택해 큐브를 감싸게 한다. 0=위치만(tilt 자유). CONTEXT 검증: 모터 jaw 바닥 z<큐브 바닥."
    ),
)
parser.add_argument("--grasp_floor_z", type=float, default=0.709, help="모터 jaw가 내려갈 매트 윗면 z (tilt penalty 기준)")

# IK / 허용 오차
parser.add_argument("--ik_lambda", type=float, default=0.1, help="DLS damping (클수록 안정/느림)")
parser.add_argument(
    "--rot_weight",
    type=float,
    default=0.6,
    help=(
        "orientation 오차 가중치(0~1). 강tilt 자세는 5-DOF로 도달 가능하므로(top-down과 달리) "
        "자세를 확실히 추종하도록 높게 둔다. 너무 낮으면 자세 안 잡혀 tilt 실패."
    ),
)
parser.add_argument("--pos_tolerance", type=float, default=0.012, help="position early-exit 허용오차(m)")
parser.add_argument("--descend_tolerance", type=float, default=0.010)

# 그리퍼 (joint target, rad 또는 normalized)
parser.add_argument("--gripper_open", type=float, default=1.0)
parser.add_argument("--gripper_closed", type=float, default=0.0)
parser.add_argument("--lift_min_height", type=float, default=0.08, help="lift 성공 판정 최소 상승(m)")

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

from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
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
# 매트 윗면 world z (scene 16ff404). 놓인 큐브 바닥 = 이 값(큐브 크기 무관). grasp 판별 기준.
CUBE_DESK_TOP_Z = 0.709
# 큐브 크기: Cube1/2=30mm(half 0.015), Cube3/4=40mm(half 0.020). 단일 상수 대신 lift는
# settle 초기 중심 대비 상승으로 판정(크기 무관). grasp 바닥 판별은 CUBE_DESK_TOP_Z 사용.

EE_BODY_NAME = "jaw"
# jaw body 원점 → grasp point(두 손가락 사이) 오프셋 (jaw local frame).
JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)

BOWL_PLACE_OFFSET_DIRECTIONS = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))

# 그리퍼 finger collision 메시 bbox(메시 로컬, m) + URDF collision origin(xyz, rpy).
# STL 측정값(검증 SM). "jaw"=모터 jaw(moving_jaw_so101_v1), "gripper"=고정 finger.
# grasp 성공 판별식(CONTEXT.md): 모터 jaw 바닥 world z < 큐브 바닥 z 면 감쌈.
_FINGER_GEOM = {
    "jaw": dict(lo=(-0.0123, -0.082, -0.024), hi=(0.01, 0.01, 0.024), oxyz=(0.0, 0.0, 0.0189), orpy=(0.0, 0.0, 0.0)),
    "gripper": dict(lo=(-0.0352, -0.0242, -0.0001), hi=(0.03, 0.0278, 0.1054), oxyz=(0.0, -0.000218214, 0.000949706), orpy=(-3.14159, 0.0, 0.0)),
}


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


def _grasp_R_world(tilt_deg: float) -> np.ndarray:
    """jaw body 가 가질 world 회전행렬. tilt_deg=0 이면 top-down(level), >0 이면 tilt.

    base(top-down): JAW_GRASP_OFFSET(주로 -Y)가 world -Z(아래)를 향하도록.
      jaw local +Y → world +Z, +X → world +X, +Z → world -Y.
    tilt_deg: world X축 회전으로 수직에서 앞(+Y world, 큐브 쪽)으로 기울임.
      강tilt 면 모터 jaw 가 큐브 옆/아래로 내려가 감싼다(검증 공식).
    grasp 단계는 강tilt, transport/place 는 level(0) 사용.
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
    tilt = math.radians(tilt_deg)
    cx, sx = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return Rz @ Rx @ R


# ── IK solver ─────────────────────────────────────────────────────────────────

class SO101DiffIK:
    """Weighted damped-least-squares IK for the SO-101 grasp point.

    Isaac Lab DifferentialInverseKinematicsAction 의 frame 변환 로직(base-frame pose/jacobian,
    offset 보정)을 재현하되, position 과 orientation 에 가중치를 두어 푼다. SO-101 은 5-DOF 라
    임의 top-down 자세+위치를 동시에 만족 못 하므로, orientation 에 작은 가중치(--rot_weight)를
    주어 position 을 우선 도달시키고 자세는 "되도록" top-down 으로 유도한다.

    grasp point = jaw body + JAW_GRASP_OFFSET.
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

    def solve(
        self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, rot_weight: float | None = None
    ) -> tuple[torch.Tensor, float, float]:
        """world target pose → arm joint target (absolute, [5]).

        Weighted DLS: error 와 jacobian 의 orientation 행에 w(rot_weight)를 곱해,
        position 을 우선 만족시키고 orientation 은 약하게 유도한다.
        rot_weight=None 이면 args.rot_weight. transport/place 는 낮게(위치 우선) 준다.

        Returns (q_arm_des[5], pos_err_m, rot_err_rad).
        """
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # world target → base frame
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w.reshape(1, 3), target_quat_w.reshape(1, 4)
        )

        ee_pos_b, ee_quat_b = self.ee_pose_b()
        jac = self._jacobian_b()  # (1, 6, 5)
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        # base-frame pose error (position + axis-angle orientation)
        pos_err_b, rot_err_b = compute_pose_error(
            ee_pos_b, ee_quat_b, tgt_pos_b, tgt_quat_b, rot_error_type="axis_angle"
        )  # (1,3), (1,3)

        w = float(args.rot_weight if rot_weight is None else rot_weight)
        lam = float(args.ik_lambda)
        # orientation 행/오차에 가중치 적용 (weighted DLS)
        jac_w = jac.clone()
        jac_w[:, 3:, :] = w * jac_w[:, 3:, :]
        err6 = torch.cat([pos_err_b, w * rot_err_b], dim=1).unsqueeze(-1)  # (1,6,1)

        jt = jac_w.transpose(1, 2)  # (1,5,6)
        lam_m = (lam ** 2) * torch.eye(6, device=self.device)
        dq = (jt @ torch.inverse(jac_w @ jt + lam_m) @ err6).squeeze(-1)  # (1,5)
        q_des = (joint_pos + dq)[0]  # (5,)

        # 진단용 오차 (world frame position, base-frame rot magnitude)
        pos_err = float(torch.linalg.norm(self.ee_pos_w()[0] - target_pos_w.reshape(3)).item())
        rot_err = float(torch.linalg.norm(rot_err_b[0]).item())
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

def _cube_lifted(env, cube_name: str, rest_z: float) -> bool:
    # 큐브 크기 무관: settle 직후 중심(rest_z) 대비 lift_min_height 이상 상승하면 들림.
    cube_z = float(env.unwrapped.scene[cube_name].data.root_pos_w[0, 2].item())
    return cube_z > rest_z + args.lift_min_height


def _cube_inside_bowl(env, cube_name: str) -> bool:
    bowl_pos = env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0]
    cube_pos = env.unwrapped.scene[cube_name].data.root_pos_w[0]
    xy = float(torch.linalg.norm(cube_pos[:2] - bowl_pos[:2]).item())
    dz = float((cube_pos[2] - bowl_pos[2]).item())
    return xy <= BOWL_SUCCESS_RADIUS and BOWL_HEIGHT_RANGE[0] <= dz <= BOWL_HEIGHT_RANGE[1]


# ── target suppliers (world frame pos + quat) ─────────────────────────────────

def _grasp_quat_w(device: str, tilt_deg: float) -> torch.Tensor:
    R = torch.tensor(_grasp_R_world(tilt_deg), device=device, dtype=torch.float32).reshape(1, 3, 3)
    return quat_from_matrix(R)  # (1,4)


def _cube_target(env, cube_name: str, dz: float, device: str) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    # 큐브 grasp 단계: 강tilt (모터 jaw 가 큐브 옆/아래로 내려가 감싸도록).
    quat = _grasp_quat_w(device, args.grasp_tilt_deg)

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        pos = env.unwrapped.scene[cube_name].data.root_pos_w[0].clone()
        pos = pos.to(device=device)
        pos[2] = pos[2] + dz
        return pos, quat

    return fn


def _bowl_target(env, dz: float, device: str, xy_offset: torch.Tensor | None = None) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    # 운반/배치 단계: level(top-down, tilt=0). tilt 자세로 운반하면 그릇 들이받아 회귀(CONTEXT.md).
    quat = _grasp_quat_w(device, 0.0)

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


# ── 진단: finger 바닥 z, tilt 강도 (grasp 성공 판별식) ──────────────────────────

def _rpy_matrix(rpy: tuple[float, float, float], device: str) -> torch.Tensor:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device, dtype=torch.float32)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device, dtype=torch.float32)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return rz @ ry @ rx


def _finger_min_z(robot, body_name: str, device: str) -> float:
    """finger collision 메시 AABB 의 world z 최저점. grasp 성공 판별식용.

    "jaw"=모터 jaw 바닥 z 가 큐브 바닥(<0.717) 아래면 감쌈 = grasp 성공(CONTEXT.md).
    """
    g = _FINGER_GEOM[body_name]
    R = _rpy_matrix(g["orpy"], device)
    oxyz = torch.tensor(g["oxyz"], device=device, dtype=torch.float32)
    lo, hi = g["lo"], g["hi"]
    corners = torch.tensor(
        [[cx, cy, cz] for cx in (lo[0], hi[0]) for cy in (lo[1], hi[1]) for cz in (lo[2], hi[2])],
        device=device,
        dtype=torch.float32,
    )  # (8,3)
    p_link = (R @ corners.T).T + oxyz  # (8,3)
    bid = robot.data.body_names.index(body_name)
    bpos = robot.data.body_pos_w[0, bid]  # (3,)
    bquat = robot.data.body_quat_w[0, bid]  # (4,)
    pw = bpos + quat_apply(bquat.unsqueeze(0).expand(8, 4), p_link)  # (8,3)
    return float(pw[:, 2].min().item())


def _approach_down_dot(robot, device: str) -> float:
    """jaw 접근축(원점→grasp point, JAW_GRASP_OFFSET 방향)이 world -Z 와 이루는 dot.

    1.0=완전 수직 top-down, 0.34≈70° tilt(검증 성공값), 0=수평.
    """
    off = torch.tensor(JAW_GRASP_OFFSET, device=device, dtype=torch.float32)
    off = off / torch.linalg.norm(off)
    bid = robot.data.body_names.index(EE_BODY_NAME)
    bquat = robot.data.body_quat_w[0, bid]
    aw = quat_apply(bquat.unsqueeze(0), off.unsqueeze(0))[0]
    aw = aw / torch.linalg.norm(aw)
    return float(-aw[2].item())  # · (0,0,-1)


# ── 큐브 순서 ─────────────────────────────────────────────────────────────────

def _ordered_cubes(env, names: list[str]) -> list[str]:
    scene = env.unwrapped.scene
    names = list(names)
    if args.object_order == "name":
        return names
    if args.object_order in ("near_robot", "far_robot"):
        # 로봇 base 와의 xy 거리 순. near_robot=가까운 큐브부터(사용자 요청 기본).
        base_xy = scene["robot"].data.root_pos_w[0, :2]
        rev = args.object_order == "far_robot"
        names.sort(
            key=lambda n: float(torch.linalg.norm(scene[n].data.root_pos_w[0, :2] - base_xy).item()),
            reverse=rev,
        )
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


# ── 결정론적 random-FK 솔버 (descend grasp 자세) ──────────────────────────────

def _fk_solve_joint_target(
    env,
    ik: SO101DiffIK,
    target_pos_w: torch.Tensor,
    gripper_target: float,
    device: str,
    *,
    samples: int,
    tilt_weight: float,
    seed_offset: int,
    continuity_weight: float = 0.015,
    floor_z: float = 0.705,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """random-FK waypoint 솔버 (검증 알고리즘, 결정론화).

    robot joint state 를 잠깐 써서(write_joint_state_to_sim + scene.update) 후보 자세의
    grasp point·모터 jaw 바닥 z 를 kinematic FK 로 평가한 뒤 원복한다. 물리 step 없음.
    score = dist(grasp_point, target) + continuity·|Δq| + tilt_weight·(모터 jaw 가 floor·
    고정 finger 위에 남는 penalty). 강tilt(모터 jaw 가 큐브 아래) 자세를 결정론적으로 선택.
    RNG seed 고정(args.seed + seed_offset) → 같은 입력에 같은 출력(stochastic 제거).
    """
    scene = env.unwrapped.scene
    robot = scene["robot"]
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    saved_q = robot.data.joint_pos[:, :6].clone()
    saved_v = robot.data.joint_vel[:, :6].clone()
    current = saved_q[0, :ARM_DOF].clone()
    lo = robot.data.soft_joint_pos_limits[0, :ARM_DOF, 0].to(device)
    hi = robot.data.soft_joint_pos_limits[0, :ARM_DOF, 1].to(device)
    lo = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, -3.14))
    hi = torch.where(torch.isfinite(hi), hi, torch.full_like(hi, 3.14))
    target = target_pos_w.to(device=device, dtype=torch.float32).reshape(3)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed + seed_offset))
    zero_vel = torch.zeros((1, 6), device=device)

    best = {"score": float("inf"), "q": current.clone(), "dist": float("inf"), "jaw_minz": 999.0}

    def evaluate(q_arm: torch.Tensor) -> None:
        q = torch.zeros((1, 6), device=device)
        q[0, :ARM_DOF] = q_arm
        q[0, 5] = float(gripper_target)
        robot.write_joint_state_to_sim(q, zero_vel, env_ids=env_ids)
        scene.update(0.0)
        gp = ik.ee_pos_w()[0]
        dist = float(torch.linalg.norm(gp - target).item())
        cont = float(torch.linalg.norm(q_arm - current).item())
        score = dist + continuity_weight * cont
        jaw_z = _finger_min_z(robot, "jaw", device)
        if tilt_weight > 0.0:
            fix_z = _finger_min_z(robot, "gripper", device)
            score = score + tilt_weight * (max(0.0, jaw_z - floor_z) + max(0.0, jaw_z - fix_z))
        if score < best["score"]:
            best.update(score=score, q=q_arm.clone(), dist=dist, jaw_minz=jaw_z)

    evaluate(current)
    local_count = min(max(samples // 3, 1), samples)
    global_count = max(samples - local_count, 1)
    for _ in range(local_count):
        noise = torch.randn((ARM_DOF,), generator=gen, device=device) * 0.35
        evaluate(torch.minimum(torch.maximum(current + noise, lo), hi))
    for _ in range(global_count):
        evaluate(lo + (hi - lo) * torch.rand((ARM_DOF,), generator=gen, device=device))

    robot.write_joint_state_to_sim(saved_q, saved_v, env_ids=env_ids)
    scene.update(0.0)
    return best["q"], {"planned_dist": round(best["dist"], 5), "planned_jaw_minz": round(best["jaw_minz"], 4)}


def execute_joint_phase(
    env,
    ik: SO101DiffIK,
    device: str,
    name: str,
    q_arm_target: torch.Tensor,
    gripper_target: float,
    max_steps: int,
    joint_tol: float,
    *,
    min_gripper_effort: float | None = None,
) -> dict[str, Any]:
    """고정 arm joint target 으로 slew 이동(env SlewLimited 가 속도 제한). joint 오차로 early-exit."""
    robot = env.unwrapped.scene["robot"]
    q_arm_target = q_arm_target.to(device=device, dtype=torch.float32)
    reached: int | None = None
    step = 0
    for step in range(max(1, int(max_steps))):
        action = _build_action(robot, ik, q_arm_target, gripper_target, device)
        _step_env(env, action, min_gripper_effort=min_gripper_effort)
        cur = robot.data.joint_pos[0, ik.arm_joint_ids]
        jerr = float(torch.max(torch.abs(cur - q_arm_target)).item())
        if jerr <= joint_tol:
            reached = step + 1
            break
    jaw_mz = _finger_min_z(robot, "jaw", device)
    down_dot = _approach_down_dot(robot, device)
    ee = ik.ee_pos_w()[0]
    return {
        "phase": name,
        "steps": step + 1,
        "reached_step": reached,
        "final_joint_err_rad": round(jerr, 4),
        "down_dot": round(down_dot, 3),
        "jaw_minz": round(jaw_mz, 4),
        "ee_w": [round(float(v), 4) for v in ee.tolist()],
        "success": reached is not None,
    }


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
    rot_weight: float | None = None,
) -> dict[str, Any]:
    """target pose 로 DifferentialIK 이동(매 step 점진, grip 유지). position 오차 ≤ tolerance 면 early-exit.

    rot_weight=None 이면 args.rot_weight. transport/place 는 낮게 줘서 위치 우선 + 자세 자유.
    target_fn 이 live(매 step 큐브-grasp point offset 보정)면 큐브를 그릇에 정렬한다.
    """
    robot = env.unwrapped.scene["robot"]
    min_pos_err = float("inf")
    last_pos_err = float("inf")
    last_rot_err = float("inf")
    reached: int | None = None
    step = 0

    tgt_pos = torch.zeros(3, device=device)
    for step in range(max(1, int(max_steps))):
        tgt_pos, tgt_quat = target_fn()
        q_arm_des, pos_err, rot_err = ik.solve(tgt_pos, tgt_quat, rot_weight=rot_weight)
        action = _build_action(robot, ik, q_arm_des, gripper_target, device)
        _step_env(env, action, min_gripper_effort=min_gripper_effort)

        min_pos_err = min(min_pos_err, pos_err)
        last_pos_err, last_rot_err = pos_err, rot_err
        if pos_err <= tolerance:
            if reached is None:
                reached = step + 1
            if require_pos:
                break

    ee = ik.ee_pos_w()[0]
    tgt = tgt_pos.reshape(3)
    stat = {
        "phase": name,
        "steps": step + 1,
        "reached_step": reached,
        "min_pos_err_m": round(min_pos_err, 5),
        "final_pos_err_m": round(last_pos_err, 5),
        "final_rot_err_rad": round(last_rot_err, 4),
        "target_w": [round(float(v), 4) for v in tgt.tolist()],
        "ee_w": [round(float(v), 4) for v in ee.tolist()],
        "success": reached is not None,
    }
    jaw_mz = _finger_min_z(robot, "jaw", device)
    down_dot = _approach_down_dot(robot, device)
    stat["jaw_minz"] = round(jaw_mz, 4)
    stat["down_dot"] = round(down_dot, 3)
    if args.phase_log:
        print(
            f"    [{name}] steps={stat['steps']} pos_err={stat['final_pos_err_m']:.4f} "
            f"rot_err={stat['final_rot_err_rad']:.3f} down_dot={down_dot:.2f} jaw_minz={jaw_mz:.4f} "
            f"tgt=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) "
            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
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
    grasp_R = _grasp_R_world(args.grasp_tilt_deg)
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

    # settle 직후 각 큐브 중심 z (lift 판정 기준, 큐브 크기 무관)
    rest_cube_z = {n: float(env.unwrapped.scene[n].data.root_pos_w[0, 2].item()) for n in active_names}

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
            # DESCEND: 결정론적 random-FK로 강tilt grasp 자세(모터 jaw가 큐브 감쌈) 탐색 후 이동.
            desc_target = _cube_target(env, cube_name, args.grasp_z_offset, device)()[0]
            q_desc, fk_info = _fk_solve_joint_target(
                env, ik, desc_target, args.gripper_open, device,
                samples=args.fk_samples, tilt_weight=args.grasp_tilt_weight,
                seed_offset=0, floor_z=args.grasp_floor_z,
            )
            d_stat = execute_joint_phase(
                env, ik, device, f"{cube_name}.descend[{attempt}]",
                q_desc, args.gripper_open, args.descend_steps, joint_tol=0.03,
            )
            d_stat.update(fk_info)
            trace.append(d_stat)
            # grasp 성공 판별식: 모터 jaw 바닥 z < 큐브 바닥 z(=매트 윗면 CUBE_DESK_TOP_Z) 면 감쌈
            if args.phase_log:
                jaw_mz = _finger_min_z(robot, "jaw", device)
                fix_mz = _finger_min_z(robot, "gripper", device)
                cube_btm = CUBE_DESK_TOP_Z
                wrap = jaw_mz < cube_btm
                print(
                    f"    >> {cube_name} descend: jaw_minz={jaw_mz:.4f} fix_minz={fix_mz:.4f} "
                    f"cube_bottom={cube_btm:.4f} → {'WRAP(감쌈)' if wrap else 'ABOVE(위에 남음)'}"
                )
            # CLOSE (ee pose 고정). 그리퍼 닫기가 모터 jaw를 큐브 아래로 쓸어담음(검증 메커니즘).
            hold_pose(env, ik, device, args.gripper_closed, args.close_steps,
                      min_gripper_effort=args.carry_min_gripper_effort)
            # CLOSE 후 grasp 성공 판별식: 모터 jaw 바닥 z < 큐브 바닥 z 면 감쌈
            jaw_mz = _finger_min_z(robot, "jaw", device)
            fix_mz = _finger_min_z(robot, "gripper", device)
            cube_btm = CUBE_DESK_TOP_Z  # 놓인 큐브 바닥 = 매트 윗면 (크기 무관)
            trace.append({
                "phase": f"{cube_name}.after_close[{attempt}]",
                "jaw_minz": round(jaw_mz, 4),
                "fix_minz": round(fix_mz, 4),
                "cube_bottom": round(cube_btm, 4),
                "wrap": jaw_mz < cube_btm,
            })
            if args.phase_log:
                print(f"    >> after CLOSE: jaw_minz={jaw_mz:.4f} cube_bottom={cube_btm:.4f} "
                      f"{'WRAP(감쌈)' if jaw_mz < cube_btm else 'ABOVE(위)'}")
            # LIFT
            trace.append(execute_phase(
                env, ik, device, f"{cube_name}.lift[{attempt}]",
                _cube_target(env, cube_name, args.lift_height, device),
                args.gripper_closed, args.lift_steps, args.pos_tolerance,
                min_gripper_effort=args.carry_min_gripper_effort,
            ))

            if _cube_lifted(env, cube_name, rest_cube_z[cube_name]):
                grasped = True
                break
            if attempt < args.max_grasp_attempts:
                hold_pose(env, ik, device, args.gripper_open, args.open_steps)

        if not grasped:
            cube_results[cube_name] = {"grasped": False, "placed": False}
            continue

        # grip 품질: 큐브가 grasp point(손가락 사이) 중심에 물렸는가. 작을수록 place 정렬 정확.
        gp0 = ik.ee_pos_w()[0]
        cube0 = env.unwrapped.scene[cube_name].data.root_pos_w[0]
        trace.append({
            "phase": f"{cube_name}.grip_check",
            "cube_gp_dist": round(float(torch.linalg.norm(cube0 - gp0).item()), 4),
            "cube_gp_xy": round(float(torch.linalg.norm((cube0 - gp0)[:2]).item()), 4),
        })

        xy_off = _bowl_xy_offset(device, placed_count)

        def _bowl_cube_target(dz: float) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
            # live: 매 step 큐브-grasp point xy offset 을 보정해 grasp point 목표를 정한다.
            # → 큐브 자체가 (그릇 중심 + 배치 offset) 에 오도록. z 는 그릇 위 dz (release→낙하).
            quat = _grasp_quat_w(device, 0.0)  # level(top-down 근사)

            def fn() -> tuple[torch.Tensor, torch.Tensor]:
                bowl = env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0].clone().to(device=device)
                bowl[:2] = bowl[:2] + xy_off
                gp = ik.ee_pos_w()[0]
                cube = env.unwrapped.scene[cube_name].data.root_pos_w[0]
                off_xy = (cube[:2] - gp[:2])
                pos = bowl.clone()
                pos[:2] = pos[:2] - off_xy
                pos[2] = bowl[2] + dz
                return pos, quat

            return fn

        # TRANSPORT/PLACE: weighted-DLS 매 step 점진 이동(grip 유지) + 위치 우선(rot_weight 낮춤)
        # + live 큐브 정렬. random-FK 1회 점프는 자세 급변으로 큐브를 놓쳐서 안 쓴다.
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.transport",
            _bowl_cube_target(args.transport_height),
            args.gripper_closed, args.transport_steps, args.pos_tolerance,
            min_gripper_effort=args.carry_min_gripper_effort, rot_weight=0.1,
        ))
        ph = args.place_height + placed_count * args.stack_place_height_increment
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.place",
            _bowl_cube_target(ph),
            args.gripper_closed, args.place_steps, args.pos_tolerance,
            min_gripper_effort=args.carry_min_gripper_effort, rot_weight=0.1,
        ))
        # 큐브 상태 추적 (언제 놓치는지)
        def _cstate(tag: str) -> None:
            cube = env.unwrapped.scene[cube_name].data.root_pos_w[0]
            bowl = env.unwrapped.scene[BOWL_NAME].data.root_pos_w[0]
            trace.append({
                "phase": f"{cube_name}.{tag}",
                "cube_z": round(float(cube[2].item()), 4),
                "bowl_z": round(float(bowl[2].item()), 4),
                "xy_to_bowl": round(float(torch.linalg.norm(cube[:2] - bowl[:2]).item()), 4),
            })
        _cstate("at_place")
        # RELEASE (ee 고정, 그리퍼 개방)
        hold_pose(env, ik, device, args.gripper_open, args.open_steps)
        _cstate("after_release")
        # RETREAT
        trace.append(execute_phase(
            env, ik, device, f"{cube_name}.retreat",
            _bowl_target(env, args.transport_height, device),
            args.gripper_open, args.retreat_steps, args.pos_tolerance,
        ))
        _cstate("after_retreat")

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
