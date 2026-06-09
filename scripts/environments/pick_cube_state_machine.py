"""SO-101 pick-and-place rule-based state machine (cube_desk 씬). — **WIP: grasp 미완**

⚠ 검증된 데모는 Franka 7DOF 버전(`pick_cube_franka_state_machine.py`, DR 상태 4/4)을 쓴다.
이 SO-101(5DOF) 버전은 grasp 가 불안정해 미완이다(아래 한계 참조).

구조·인프라는 완성: scan / reached·grasped·placed / order_by_proximity / move_to / 근접순 /
도달 기반 / 파일 디버깅 로깅 / 사이드뷰 카메라 / 속도 미제한(Franka 급). IK 는 Isaac Sim 내장
**Lula `LulaKinematicsSolver` global numerical IK**(ROS2/MoveIt2 대신 — ROS2 미설치·인프라 회피).
Lula 자체는 err 0 로 수렴한다.

SO-101 grasp 가 불안정한 근본 — 세 오차원이 중첩(각 ~0.05~0.1m, 자세 의존):
  1) Lula(URDF-local frame) ↔ USD world **정합** 잔차 (RMPFLOW_BASE least-squares, ~0.1m)
  2) Lula 제어점(gripper_frame_link) ↔ 실제 손가락 갭 (자세에 따라 상대 위치 변동)
  3) **5DOF** position + full orientation 동시 만족 불가 (position-only 면 손가락 자세 미제어)
weighted DLS(local IK), random-FK, Lula(position/orientation/midpoint) 모두 이 중첩을 못 넘어
손가락이 큐브를 0.05~0.1m 빗나간다. 자세한 분석은 docs/TROUBLESHOOTING.md 참조.

실행(단일 큐브 진단):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/pick_cube_state_machine.py \
        --num_envs 1 --active_objects 1 --object_radius_scale 0 --container_angle_scale 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

_LOG_PATH = "/tmp/so101_sm_progress.txt"
open(_LOG_PATH, "w").close()


def log(msg: str) -> None:
    """Isaac Sim 이 gym.make 후 stdout/stderr 를 carb 로 재바인딩해 print 가 묻히므로,
    진행 로그를 파일에 직접 append 하고 원본 stderr fd 에도 쓴다."""
    with open(_LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg, file=sys.__stderr__, flush=True)


def _vec3(s: str) -> tuple[float, float, float]:
    p = [float(x) for x in s.split(",")]
    if len(p) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SO-101 pick-and-place state machine (cube_desk)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0)
parser.add_argument("--container_angle_scale", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
# 도달 기반 제어
parser.add_argument("--reach_tol", type=float, default=0.015, help="grasp/place 정밀 도달 거리(m)")
parser.add_argument("--coarse_tol", type=float, default=0.03, help="상공 경유 도달 거리(m)")
parser.add_argument("--max_phase_steps", type=int, default=250)
parser.add_argument("--grasp_dwell", type=int, default=45, help="그리퍼 닫힘 정착 step(SO-101 느린 그리퍼)")
parser.add_argument("--release_dwell", type=int, default=20)
parser.add_argument("--settle_steps", type=int, default=20)
# 높이/오프셋 (m)
parser.add_argument("--approach_height", type=float, default=0.12)
parser.add_argument("--grasp_z_offset", type=float, default=0.005, help="큐브 중심 기준 grasp tip z")
parser.add_argument("--lift_height", type=float, default=0.10, help="책상 윗면 기준 들어올림")
parser.add_argument("--transport_height", type=float, default=0.12, help="그릇 위 운반 높이")
parser.add_argument("--place_height", type=float, default=0.05, help="그릇 바닥 기준 release 높이")
parser.add_argument("--stack_increment", type=float, default=0.022)
parser.add_argument("--grasp_tilt_deg", type=float, default=60.0,
                    help="grasp 시 수직에서 앞으로 기울임(deg). SO-101 은 완전 top-down 불가 → 강tilt 로 감쌈")
parser.add_argument("--grasp_yaw_deg", type=float, default=0.0,
                    help="손가락 평면 yaw(world Z축, deg). SO-101 5DOF 는 yaw 가 shoulder_pan 에 "
                         "종속되므로 고정한다(동적 yaw 는 손가락 roll 을 틀어 큐브를 비껴감).")
# weighted DLS IK
parser.add_argument("--ik_lambda", type=float, default=0.1, help="DLS damping")
parser.add_argument("--rot_weight_grasp", type=float, default=0.6, help="grasp 단계 자세 가중치(자세 유도). 높으면 자세 우선·xy 정밀도↓, 낮으면 자세(tilt)부정확 — SO-101 5DOF trade-off")
parser.add_argument("--rot_weight_carry", type=float, default=0.1, help="운반/배치 자세 가중치(위치 우선)")
parser.add_argument("--max_joint_delta", type=float, default=0.08, help="step 당 arm joint 변화 상한(rad)")
parser.add_argument("--arm_max_velocity", type=float, default=0.0,
                    help="arm joint 최대 속도(rad/s). 0=env 기본(5, Franka 처럼 빠름) 유지 — 32초급 속도. "
                         "grasp 중 큐브를 쳐내면 값을 줘서(예 2~3) 약하게만 제한")
# GUI 초기 뷰 — 작업영역(큐브·그리퍼)을 가깝게 측면에서 보는 각도(사진 촬영용).
parser.add_argument("--view_eye", type=_vec3, default=(2.30, -0.72, 0.95))
parser.add_argument("--view_lookat", type=_vec3, default=(1.80, -0.43, 0.75))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
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
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# Lula motion-generation extension 은 기본 비활성 → import 전 enable.
enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver  # noqa: E402

import sim_to_real  # noqa: E402, F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
    PickCubeEnvCfg,
    apply_curriculum,
)
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import SO101_JOINT_ORDER  # noqa: E402
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

# ── 상수 ────────────────────────────────────────────────────────────────────
ARM_DOF = 5
EE_BODY_NAME = "jaw"
JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)  # jaw local → grasp point(두 손가락 사이)
CUBE_DESK_TOP_Z = 0.709  # 매트 윗면 world z (grasped 판정 기준)

# Lula IK (so_arm101.urdf 기준 global numerical IK). base/offset 은 검증된 least-squares 값.
RMPFLOW_DIR = Path("assets/robots/rmpflow").resolve()
RMPFLOW_URDF_PATH = Path("assets/robots/urdf/so_arm101.urdf").resolve()
RMPFLOW_DESCRIPTOR_PATH = RMPFLOW_DIR / "so101_robot_description.yaml"
# Lula 는 URDF-local frame 에서 FK/IK, USD articulation 은 cube_desk scene transform 아래 →
# 이 base pose 로 Lula gripper_frame_link FK 를 USD world 좌표에 매핑(02bdc71 검증값).
RMPFLOW_BASE_POS_USD = (1.81791970, -0.58952723, 0.70832908)
RMPFLOW_BASE_QUAT_USD = (0.71116823, -0.00950808, 0.01529776, 0.70279110)  # wxyz
# gripper_frame_link → grasp 접점(두 손가락 사이) 보정.
RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET = (-0.078, 0.010, -0.002)
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0

# jaw body 가 완전 top-down(아래) 일 때의 world 회전(yaw=0, tilt=0).
# jaw local +Y → world +Z, +X → world +X, +Z → world -Y.
_JAW_TOPDOWN_R = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# 동적 top-down quaternion
# ---------------------------------------------------------------------------


def _down_quat_world(yaw_rad: float, tilt_deg: float, device: str) -> torch.Tensor:
    """jaw 가 아래를 향하되 yaw 를 base→target 방향으로 맞춘 world quaternion (1,4) wxyz."""
    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = math.radians(tilt_deg)
    cx, sx = math.cos(t), math.sin(t)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    R = Rz @ Rx @ _JAW_TOPDOWN_R
    R_t = torch.tensor(R, device=device, dtype=torch.float32).reshape(1, 3, 3)
    return quat_from_matrix(R_t)


# ---------------------------------------------------------------------------
# Lula IK (SO-101 global numerical IK)
# ---------------------------------------------------------------------------


class So101LulaIK:
    """Lula ``LulaKinematicsSolver`` position-only **global** numerical IK.

    weighted DLS(local IK)가 못 찾던 SO-101 grasp 하강 자세를 global solver 로 푼다.

    **정합**: Lula 는 URDF-local frame 에서 푸므로 Lula world 와 USD world 사이에 잔차가 있다
    (RMPFLOW_BASE 만으론 ~0.1m 어긋남). 그래서 ee 판정·grasp target 은 **USD 실제 grasp point**
    (jaw body + JAW_GRASP_OFFSET)로 하고, solve 에서 매번 런타임 **shift = G_usd − F_lula**(현재
    자세의 USD grasp point − Lula gripper_frame_link FK)를 측정해 USD target → Lula frame 으로
    변환한다. → Lula 좌표 잔차가 상쇄돼 실제 손가락이 큐브에 정확히 간다.

    local 솔버라 현재 EE→target 을 `max_step` 간격으로 보간하며 warm_start 체이닝. 5DOF 라
    position-only(target_orientation=None) — 자세는 IK redundancy 가 결정.
    """

    def __init__(self, env, device: str, *, max_step: float = 0.04, tolerance: float = 0.005) -> None:
        self.robot = env.scene["robot"]
        self.device = device
        self._max_step = float(max_step)
        self._tol = float(tolerance)
        self._arm_ids = [self.robot.data.joint_names.index(n) for n in SO101_JOINT_ORDER[:ARM_DOF]]
        # ee = 두 손가락 midpoint(실제 grasp 갭 중심). jaw+offset 은 실제 갭과 어긋나므로 midpoint 사용.
        self._jaw_idx = self.robot.data.body_names.index("jaw")
        self._fix_idx = self.robot.data.body_names.index("gripper")
        self.last_err = float("nan")  # 직전 solve 의 FK→target 잔차(진단)
        self.last_ok = False
        self._kin = LulaKinematicsSolver(
            robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
            urdf_path=str(RMPFLOW_URDF_PATH),
        )
        self._sync_base_pose()

    def _sync_base_pose(self) -> None:
        self._kin.set_robot_base_pose(
            np.asarray(RMPFLOW_BASE_POS_USD, dtype=np.float32),
            np.asarray(RMPFLOW_BASE_QUAT_USD, dtype=np.float32),
        )

    def _q_arm(self) -> np.ndarray:
        return self.robot.data.joint_pos[0, self._arm_ids].detach().cpu().numpy().astype(np.float64)

    def _lula_fk(self, q: np.ndarray) -> np.ndarray:
        pos, _ = self._kin.compute_forward_kinematics("gripper_frame_link", q)
        return np.asarray(pos, dtype=np.float64)

    def ee_pos_w(self) -> torch.Tensor:
        """USD 실제 grasp 접점 (1,3) = 두 손가락(jaw, gripper) body 의 midpoint."""
        bp = self.robot.data.body_pos_w
        return 0.5 * (bp[:, self._jaw_idx] + bp[:, self._fix_idx])

    def solve(self, target_pos_w: torch.Tensor, tilt_deg: float | None = None) -> torch.Tensor:
        """USD world grasp-point target → arm joint target [5].

        tilt_deg 가 주어지면 down-tilt 자세를 orientation target 으로 함께 풀어(5DOF best-effort)
        손가락이 큐브를 아래로 감싸게 유도. None 이면 position-only.
        """
        self._sync_base_pose()
        q = self._q_arm()
        f_lula = self._lula_fk(q)  # Lula frame 의 현재 gripper_frame_link
        g_usd = self.ee_pos_w()[0].detach().cpu().numpy().astype(np.float64)  # USD 실제 grasp point
        shift = g_usd - f_lula  # Lula → USD 위치 정합(자세 의존, 매 solve 재측정)
        target_lula = target_pos_w.detach().cpu().reshape(3).numpy().astype(np.float64) - shift
        tgt_ori = None
        if tilt_deg is not None:
            quat_w = _down_quat_world(math.radians(args.grasp_yaw_deg), tilt_deg, self.device)
            tgt_ori = quat_w[0].detach().cpu().numpy().astype(np.float64)
        start = f_lula
        dist = float(np.linalg.norm(target_lula - start))
        n = max(1, int(math.ceil(dist / max(self._max_step, 1e-3))))
        ok = False
        for i in range(1, n + 1):
            sub = start + (target_lula - start) * (float(i) / n)
            q_sol, ok = self._kin.compute_inverse_kinematics(
                "gripper_frame_link",
                sub,
                target_orientation=tgt_ori,
                warm_start=q,
                position_tolerance=self._tol,
            )
            if q_sol is not None:
                q = np.asarray(q_sol, dtype=np.float64)[:ARM_DOF]
        self.last_err = float(np.linalg.norm(self._lula_fk(q) - target_lula))
        self.last_ok = bool(ok) and self.last_err <= self._tol * 3.0
        return torch.as_tensor(q[:ARM_DOF], device=self.device, dtype=torch.float32)

    @property
    def arm_ids(self) -> list[int]:
        return self._arm_ids


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------


def _apply_dr(env_cfg: PickCubeEnvCfg) -> None:
    apply_curriculum(
        env_cfg,
        active_objects=args.active_objects,
        object_radius_scale=args.object_radius_scale,
        container_angle_scale=args.container_angle_scale,
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class So101PickPlaceSM:
    def __init__(self, env) -> None:
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.ik = So101LulaIK(env, self.device)
        self._default = self.robot.data.default_joint_pos[0]
        self._gripper_id = self.robot.data.joint_names.index(SO101_JOINT_ORDER[ARM_DOF])
        # 두 손가락 body(감쌈 진단용): "jaw"=모터 손가락, "gripper"=고정 손가락.
        self._jaw_idx = self.robot.data.body_names.index("jaw")
        self._fixed_idx = self.robot.data.body_names.index("gripper")
        self.home_pos = torch.tensor([1.84, -0.40, CUBE_DESK_TOP_Z + 0.14], device=self.device)

    # --- 상태 스캔 ---
    def scan(self) -> dict[str, torch.Tensor]:
        names = list(CUBE_NAMES) + [BOWL_NAME]
        return {n: self.env.scene[n].data.root_pos_w[0, :3].clone() for n in names}

    def obj_pos(self, name: str) -> torch.Tensor:
        return self.env.scene[name].data.root_pos_w[0, :3].clone()

    def ee_pos(self) -> torch.Tensor:
        return self.ik.ee_pos_w()[0]

    # --- 조건 체크 ---
    def reached(self, target_pos_w: torch.Tensor, tol: float) -> bool:
        return torch.linalg.norm(self.ee_pos() - target_pos_w).item() < tol

    def grasped(self, cube_name: str) -> bool:
        return self.obj_pos(cube_name)[2].item() > CUBE_DESK_TOP_Z + 0.03

    def placed(self, cube_name: str) -> bool:
        p = self.obj_pos(cube_name)
        bowl_xy = self.obj_pos(BOWL_NAME)[:2]
        in_xy = torch.linalg.norm(p[:2] - bowl_xy).item() < BOWL_SUCCESS_RADIUS
        dz = p[2].item() - CUBE_DESK_TOP_Z
        in_z = BOWL_HEIGHT_RANGE[0] <= dz <= BOWL_HEIGHT_RANGE[1] + 0.10
        return in_xy and in_z

    # --- 저수준 액션 ---
    def _yaw_to(self, target_xy: torch.Tensor) -> float:
        base_xy = self.robot.data.root_pos_w[0, :2]
        return math.atan2(float(target_xy[1] - base_xy[1]), float(target_xy[0] - base_xy[0]))

    def _action(self, q_arm_des: torch.Tensor, gripper_open: bool) -> torch.Tensor:
        """absolute joint target → SlewLimitedJointPositionAction raw (q - default), [1,6]."""
        a = torch.zeros(6, device=self.device, dtype=torch.float32)
        for i, jid in enumerate(self.ik.arm_ids):
            a[i] = q_arm_des[i] - self._default[jid]
        g = GRIPPER_OPEN if gripper_open else GRIPPER_CLOSE
        a[ARM_DOF] = g - self._default[self._gripper_id]
        return a.reshape(1, 6)

    def _act(self, target_pos_w: torch.Tensor, tilt_deg: float, rot_w: float, gripper_open: bool) -> None:
        # position-only(자세 자유). ee=midpoint 라 손가락 갭이 큐브를 감싼다. orientation 강제는
        # 5DOF over-constrained 라 position 을 희생(err↑)하므로 쓰지 않는다. tilt_deg/rot_w 미사용.
        q_arm = self.ik.solve(target_pos_w, None)
        self.env.step(self._action(q_arm, gripper_open))

    def move_to(self, target_fn, tilt_deg: float, rot_w: float, gripper_open: bool, tol: float) -> bool:
        for _ in range(args.max_phase_steps):
            if not simulation_app.is_running():
                return False
            t = target_fn()
            self._act(t, tilt_deg, rot_w, gripper_open)
            if self.reached(t, tol):
                return True
        return False

    def hold(self, target_pos_w: torch.Tensor, tilt_deg: float, rot_w: float, gripper_open: bool, steps: int) -> None:
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self._act(target_pos_w, tilt_deg, rot_w, gripper_open)

    # --- target 헬퍼 ---
    @staticmethod
    def _above(pos: torch.Tensor, height: float) -> torch.Tensor:
        t = pos.clone()
        t[2] = t[2] + height
        return t

    def _xyz(self, xy: torch.Tensor, z: float) -> torch.Tensor:
        return torch.tensor([xy[0].item(), xy[1].item(), z], device=self.device)

    def order_by_proximity(self, cubes: list[str]) -> list[str]:
        base_xy = self.robot.data.root_pos_w[0, :2]
        return sorted(cubes, key=lambda c: torch.linalg.norm(self.obj_pos(c)[:2] - base_xy).item())

    # --- 오브젝트 1개 pick-and-place ---
    def pick_and_place(self, cube_name: str, n_placed: int) -> bool:
        coarse, fine = args.coarse_tol, args.reach_tol
        tilt = args.grasp_tilt_deg
        wg, wc = args.rot_weight_grasp, args.rot_weight_carry

        # 1) 큐브 상공 접근 (강tilt 자세 유도)
        self.move_to(lambda: self._above(self.obj_pos(cube_name), args.approach_height), tilt, wg, True, coarse)
        # 2) grasp 높이 하강
        r = self.move_to(lambda: self._above(self.obj_pos(cube_name), args.grasp_z_offset), tilt, wg, True, fine)
        ee = self.ee_pos()
        cp = self.obj_pos(cube_name)
        jaw = self.robot.data.body_pos_w[0, self._jaw_idx]
        fix = self.robot.data.body_pos_w[0, self._fixed_idx]
        log(f"[SM]   {cube_name} descend reached={r} lula(ok={self.ik.last_ok},err={self.ik.last_err:.4f}) "
            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) cube=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})")
        log(f"[SM]     jaw=({jaw[0]:.3f},{jaw[1]:.3f},{jaw[2]:.3f}) "
            f"gripper=({fix[0]:.3f},{fix[1]:.3f},{fix[2]:.3f})  "
            f"(두 손가락이 큐브 xy 를 사이에 둬야 감쌈)")
        grasp_xy = self.obj_pos(cube_name)[:2].clone()
        grasp_z = self.obj_pos(cube_name)[2].item()
        grasp_target = self._xyz(grasp_xy, grasp_z + args.grasp_z_offset)
        # 3) 그리퍼 닫고 정착 (자세 유지)
        self.hold(grasp_target, tilt, wg, False, args.grasp_dwell)
        # 4) 들어올림 (위치 우선)
        self.move_to(lambda: self._xyz(grasp_xy, CUBE_DESK_TOP_Z + args.lift_height), 0.0, wc, False, coarse)
        if not self.grasped(cube_name):
            log(f"[SM]   {cube_name}: grasp 실패(들리지 않음) — 건너뜀")
            return False
        # 5) 그릇 상공 운반
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), 0.0, wc, False, coarse)
        # 6) 그릇 안으로 하강
        place_h = args.place_height + n_placed * args.stack_increment
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), place_h), 0.0, wc, False, fine)
        # 7) release
        rel = self._above(self.obj_pos(BOWL_NAME), place_h)
        self.hold(rel, 0.0, wc, True, args.release_dwell)
        # 8) 후퇴
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), 0.0, wc, True, coarse)
        return True

    def run(self, active_cubes: list[str]) -> None:
        self.hold(self.home_pos, 0.0, args.rot_weight_carry, True, args.settle_steps)
        ordered = self.order_by_proximity(active_cubes)
        log(f"[SM] pick order (robot 근접순): {ordered}")
        for i, cube in enumerate(ordered):
            log(f"[SM] pick-and-place: {cube} (placed so far={i})")
            self.pick_and_place(cube, i)
        self.hold(self.home_pos, 0.0, args.rot_weight_carry, True, args.settle_steps)
        log("[SM] all cubes processed.")
        self._report(active_cubes)

    def _report(self, active_cubes: list[str]) -> None:
        n_ok = 0
        for cube in active_cubes:
            p = self.obj_pos(cube)
            bowl_xy = self.obj_pos(BOWL_NAME)[:2]
            dist = torch.linalg.norm(p[:2] - bowl_xy).item()
            ok = self.placed(cube)
            n_ok += int(ok)
            log(f"[SM] {cube}: dist_to_bowl_xy={dist:.3f}m z={p[2].item():.3f} placed={ok}")
        log(f"[SM] RESULT: {n_ok}/{len(active_cubes)} cubes in bowl.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    log("[SM] main entered.")
    env_cfg = PickCubeEnvCfg()  # 기본 actions = SlewLimitedJointPositionAction(6 joint)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.arm_max_velocity > 0:  # 0 이면 env 기본(빠름) 유지
        mv = {j: args.arm_max_velocity for j in SO101_JOINT_ORDER[:ARM_DOF]}
        mv[SO101_JOINT_ORDER[ARM_DOF]] = 5.0  # gripper 는 빠르게(닫힘 지연 방지)
        env_cfg.actions.arm.max_velocity = mv
    _apply_dr(env_cfg)
    env_cfg.viewer.eye = args.view_eye
    env_cfg.viewer.lookat = args.view_lookat
    log("[SM] env_cfg built — calling gym.make.")

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    sm = So101PickPlaceSM(env)
    active_cubes = CUBE_NAMES[: args.active_objects]
    sm.run(active_cubes)

    if not args.headless:
        while simulation_app.is_running():
            sm._act(sm.home_pos, 0.0, args.rot_weight_carry, True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        log("[SM] EXCEPTION:\n" + traceback.format_exc())
        raise
    finally:
        simulation_app.close()
