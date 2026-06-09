"""SO-101 in-sim DifferentialIK pick-and-place state machine (cube_desk 씬).

Franka 버전(`pick_cube_franka_state_machine.py`)과 **동일한 제어 경로**(Isaac Lab task-space
DifferentialIK env action + 도달-기반 FSM)를 SO-101 5DOF 에 이식한 것. 외부 Lula 솔버를 버렸다.

왜 Lula 를 버렸나 (진단)
------------------------
이전 구현은 SO-101 env 가 순수 joint-space action(`SlewLimitedJointPositionAction`)이라 in-sim
IK 가 없어서, 외부 Lula 를 *다른 URDF·다른 world frame* 에서 돌리고 결과를 런타임
``shift = g_usd - f_lula`` 로 USD 에 끼워맞췄다. 그 shift 는 (프레임 보정 잔차 +
gripper_frame_link↔손가락중점 offset)을 한 덩어리로 1차 근사한 것이라 자세가 바뀌면 무효 →
손가락이 큐브를 0.05~0.1m 빗나갔다. Lula 자체는 err 0 로 수렴했다(솔버 문제가 아니었다).

이 버전은 `SimToReal-SO101-PickCube-IK-v0`(DifferentialIKAction) 위에서 돌아, IK 가 USD
articulation Jacobian 에 대해 sim 내부에서 풀린다. 제어점(IK body+offset)과 도달 판정점이
**동일** 하므로 stale-shift 가 원천 제거된다.

Franka 와의 유일한 본질 차이 = target orientation
--------------------------------------------------
Franka(7DOF)는 고정 world-down quat 으로 충분하지만, SO-101(5DOF)은 gripper yaw 가 shoulder_pan 에
종속(=base→target radial)이라 **radial-yaw + down-tilt** 자세를 매 step 동적 구성해야 도달 가능한
pose 가 된다(full top-down 은 5DOF 로 불가 → tilt). DLS 가 over-constrained pose 를 best-effort 로
풀고, position 이 우선이라 손가락이 grasp 접점에 정확히 간다.

실행(단일 큐브 진단):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/pick_cube_state_machine.py \
        --num_envs 1 --active_objects 1 --object_radius_scale 0 --container_angle_scale 0 --headless

실행(4 큐브 full-DR):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/pick_cube_state_machine.py \
        --num_envs 1 --active_objects 4 --object_radius_scale 1 --container_angle_scale 1 --headless
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
import time

import numpy as np

from isaaclab.app import AppLauncher

_LOG_PATH = os.path.abspath("outputs/so101_sm_progress.txt")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
open(_LOG_PATH, "w").close()
# C레벨 크래시(access violation 등) 추적 — Python traceback 을 파일로 덤프.
_FH_FILE = open(os.path.abspath("outputs/so101_faulthandler.txt"), "w")
faulthandler.enable(file=_FH_FILE)


def log(msg: str) -> None:
    """진행 로그를 파일에 append. **encoding='utf-8' 필수** — Windows 기본 cp949 로 열면
    한글/em dash(—) 같은 비 cp949 문자에서 UnicodeEncodeError 가 나고, 그 예외가 except 의
    traceback 로깅에서 또 터져 연쇄로 silent exit(0) 한다. print 도 stderr 인코딩 실패 무시."""
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        print(msg, file=sys.__stderr__, flush=True)
    except (UnicodeEncodeError, OSError):
        pass


def _vec3(s: str) -> tuple[float, float, float]:
    p = [float(x) for x in s.split(",")]
    if len(p) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SO-101 DifferentialIK pick-and-place state machine (cube_desk)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-IK-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0,
                    help="큐브 scatter DR 강도 (0=고정 spawn, 1=전체 workspace)")
parser.add_argument("--container_angle_scale", type=float, default=1.0,
                    help="그릇 arc DR 강도 (0=고정, 1=기본 각도범위)")
parser.add_argument("--seed", type=int, default=0)
# 도달 기반 제어
parser.add_argument("--reach_tol", type=float, default=0.015, help="grasp/place 정밀 도달 거리(m)")
parser.add_argument("--coarse_tol", type=float, default=0.03, help="상공 경유 도달 거리(m)")
parser.add_argument("--max_phase_steps", type=int, default=300, help="한 단계 도달 못해도 넘어가는 step 상한")
parser.add_argument("--ik_scale", type=float, default=1.0,
                    help="DifferentialIKAction scale. 낮추면 step 당 ee 이동이 작아 손목 flip 억제, "
                         "너무 낮으면 정밀 수렴 안 됨")
parser.add_argument("--ik_lambda", type=float, default=0.0,
                    help="DLS lambda override(0=cfg 기본 0.1). 작을수록 position 정확(특이점 주의).")
parser.add_argument("--gripper_velocity", type=float, default=0.0,
                    help="gripper actuator velocity_limit override(rad/s, 0=cfg 기본 10). "
                         "낮추면 천천히 닫아 큐브를 쳐내지 않음(CONTEXT: 빠른 닫기 시 grasp 실패).")
parser.add_argument("--gripper_close", type=float, default=0.0,
                    help="gripper close joint target(rad) override. 기본 0.0(cfg). 큐브가 안 잡히면 "
                         "-0.1~-0.17(limit 하한)로 더 닫아 손가락이 큐브를 확실히 무름.")
parser.add_argument("--gripper_open", type=float, default=0.0,
                    help="gripper open joint target(rad) override(0=cfg 기본 1.0). 크게(1.4~1.7)하면 "
                         "갭을 넓혀 큐브가 손가락 사이로 들어옴(닫을 때 밀려나지 않음).")
parser.add_argument("--grasp_dwell", type=int, default=45, help="그리퍼 닫힘 정착 step(SO-101 느린 그리퍼)")
parser.add_argument("--release_dwell", type=int, default=20)
parser.add_argument("--settle_steps", type=int, default=20)
# 높이/오프셋 (m)
parser.add_argument("--approach_height", type=float, default=0.10, help="큐브 위 접근 높이")
parser.add_argument("--grasp_z_offset", type=float, default=0.0, help="grasp 시 큐브 중심 기준 tip z 오프셋")
parser.add_argument("--lift_height", type=float, default=0.10, help="책상 윗면 기준 들어올림")
parser.add_argument("--transport_height", type=float, default=0.12, help="그릇 위 운반 높이")
parser.add_argument("--place_height", type=float, default=0.05, help="그릇 바닥 기준 release 높이")
parser.add_argument("--stack_increment", type=float, default=0.022, help="이미 담긴 큐브당 release 높이 증가")
# 자세 (deg) — SO-101 5DOF 는 완전 top-down 불가 → tilt 로 도달 가능 자세 구성.
parser.add_argument("--grasp_tilt_deg", type=float, default=45.0,
                    help="grasp/descend 단계 수직에서 앞으로 기울임(deg). 0=완전 top-down(5DOF 도달 난).")
parser.add_argument("--carry_tilt_deg", type=float, default=20.0,
                    help="운반/배치 단계 tilt(deg). 들어올린 뒤라 작게.")
parser.add_argument("--ik_position_only", action="store_true",
                    help="orientation 없이 position(3) DiffIK 만. 5DOF over-constrained pose 회피. "
                         "in-sim 제어점 일치라 ee 가 큐브에 정확히 감(자세는 redundancy+tilt 미적용).")
parser.add_argument("--grasp_closed_loop", action="store_true",
                    help="descend 시 ee-cube xy 오차를 over-drive(target=2*cube-ee) 피드백해 ee 를 "
                         "큐브 정중앙으로 당김. 5DOF position 잔차(systematic)로 큐브가 손가락 갭 "
                         "가장자리에 걸려 닫을 때 밀려나는 것을 보정.")
# arm actuator stiffness override (0=env 기본 유지). DiffIK joint-target 추종이 느리면 상향.
parser.add_argument("--arm_stiffness", type=float, default=0.0,
                    help="arm joint PD stiffness override. 0=env 기본(soft PD 17.8) 유지. "
                         "ee 가 target 을 못 따라가면 80~300 으로 상향.")
parser.add_argument("--arm_damping", type=float, default=0.0, help="arm joint PD damping override(0=기본).")
# GUI 초기 뷰 — 작업영역(큐브·그리퍼)을 가깝게 측면에서 보는 각도.
parser.add_argument("--view_eye", type=_vec3, default=(2.30, -0.72, 0.95))
parser.add_argument("--view_lookat", type=_vec3, default=(1.80, -0.43, 0.75))
# 녹화
parser.add_argument("--step_sleep", type=float, default=0.0,
                    help="GUI 관찰용 슬로우모션: 매 env.step 후 wall-clock 지연(초). ik_scale 과 달리 "
                         "좌표를 왜곡하지 않는다. 예 0.03. headless 에선 무시.")
parser.add_argument("--video", action="store_true", help="사이드뷰를 mp4 로 docs/ 에 녹화")
parser.add_argument("--video_length", type=int, default=3000, help="녹화 최대 프레임(step) 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

# AppLauncher 에는 실제 사용하는 키만 전달한다. view_eye/view_lookat 같은 tuple 인자를
# 그대로 넘기면 Windows GUI 의 _prepare_ui(console_window→rtx.scenedb) 가 access violation
# 으로 크래시한다(docs/TROUBLESHOOTING.md "_prepare_ui access violation"). headless 는
# _prepare_ui 를 거치지 않아 영향 없지만, GUI/녹화 모드를 위해 필터링한다.
_LAUNCHER_KEYS = {"headless", "enable_cameras", "experience", "device", "cpu",
                  "disable_fabric", "offscreen_render", "kit_args"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_from_matrix,
    subtract_frame_transforms,
)

import sim_to_real  # noqa: E402, F401  (Gym 환경 등록 트리거)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
)
from sim_to_real.tasks.pick_cube.pick_cube_so101_ik_env_cfg import (  # noqa: E402
    SO101_ARM_JOINTS,
    SO101_IK_BODY_NAME,
    SO101_IK_GRASP_OFFSET,
    PickCubeSo101IkEnvCfg,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

# cube_desk 책상 윗면 world z (lift/transport 절대 높이, grasped 판정 기준).
CUBE_DESK_TOP_Z = 0.709


# ---------------------------------------------------------------------------
# 동적 down-tilt quaternion (gripper body 기준)
# ---------------------------------------------------------------------------


def _down_quat_world(yaw_rad: float, tilt_deg: float, device: str) -> torch.Tensor:
    """gripper 손가락이 아래를 향하되 yaw 를 base→target radial 로, tilt 만큼 앞으로 기울인
    world quaternion (1,4) wxyz.

    기준 R0 = identity: gripper body local frame 이 world 와 정렬일 때 손가락(local -Z)이 world
    -Z(아래)를 향한다(측정으로 gripper-local +Z ↔ world +Z 정렬 확인). 거기서 world X 로 tilt(앞
    기울임), world Z 로 yaw(radial 정렬)를 합성한다.
    """
    cz, sz = math.cos(yaw_rad), math.sin(yaw_rad)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = math.radians(tilt_deg)
    cx, sx = math.cos(t), math.sin(t)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    R = Rz @ Rx  # R0 = identity
    R_t = torch.tensor(R, device=device, dtype=torch.float32).reshape(1, 3, 3)
    return quat_from_matrix(R_t)


# ---------------------------------------------------------------------------
# Domain Randomization — events 만 강도 조정 (robot 무관 부분만; SO101Ik cfg 는 빈 reward/
# success termination 이 없어 apply_curriculum 을 못 쓴다 → Franka SM 과 동일 방식)
# ---------------------------------------------------------------------------


def _apply_dr(env_cfg: PickCubeSo101IkEnvCfg) -> None:
    from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (
        _CUBE_SCATTER_CENTER,
        _CUBE_SCATTER_X_RANGE,
        _CUBE_SCATTER_Y_RANGE,
    )

    scatter = getattr(env_cfg.events, "randomize_cubes", None)
    s = max(0.0, float(args.object_radius_scale))
    if scatter is not None and s <= 0.0:
        env_cfg.events.randomize_cubes = None
    elif scatter is not None and s != 1.0:
        cx, cy = _CUBE_SCATTER_CENTER
        x_lo, x_hi = _CUBE_SCATTER_X_RANGE
        y_lo, y_hi = _CUBE_SCATTER_Y_RANGE
        scatter.params["x_range"] = (cx - (cx - x_lo) * s, cx + (x_hi - cx) * s)
        scatter.params["y_range"] = (cy - (cy - y_lo) * s, cy + (y_hi - cy) * s)

    bowl = getattr(env_cfg.events, "randomize_bowl", None)
    a = float(args.container_angle_scale)
    if bowl is not None and a != 1.0:
        lo, hi = bowl.params["angle_range_deg"]
        bowl.params["angle_range_deg"] = (lo * a, hi * a)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class So101PickPlaceSM:
    """단일 env, 도달 조건 기반 pick-and-place 컨트롤러 (Franka SM 과 동형)."""

    def __init__(self, env) -> None:
        self.env = env  # step 용(RecordVideo wrap 시 wrapper)
        self.scene = env.unwrapped.scene
        self.device = env.unwrapped.device
        self.robot = self.scene["robot"]
        # IK 가 제어하는 작업점(gripper body + grasp offset)을 그대로 추적.
        self.ee_body_idx = self.robot.find_bodies(SO101_IK_BODY_NAME)[0][0]
        self.ee_offset = torch.tensor(SO101_IK_GRASP_OFFSET, device=self.device)
        # 두 손가락 body(감쌈 진단용).
        self._jaw_idx = self.robot.find_bodies("jaw")[0][0]
        self._gripper_jid = self.robot.data.joint_names.index("gripper")
        self.home_pos = torch.tensor([1.84, -0.40, CUBE_DESK_TOP_Z + 0.14], device=self.device)

    # --- 상태 스캔 ---
    def scan(self) -> dict[str, torch.Tensor]:
        names = list(CUBE_NAMES) + [BOWL_NAME]
        return {n: self.scene[n].data.root_pos_w[0, :3].clone() for n in names}

    def obj_pos(self, name: str) -> torch.Tensor:
        return self.scene[name].data.root_pos_w[0, :3].clone()

    def ee_pos(self) -> torch.Tensor:
        """IK 작업점(grasp 접점)의 현재 world 위치."""
        bp = self.robot.data.body_pos_w[0, self.ee_body_idx]
        bq = self.robot.data.body_quat_w[0, self.ee_body_idx]
        return bp + quat_apply(bq.unsqueeze(0), self.ee_offset.unsqueeze(0)).squeeze(0)

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

    def _act(self, target_pos_w: torch.Tensor, tilt_deg: float, gripper_open: bool) -> None:
        """world target → root frame → env.step.

        pose 모드: action = [ee pos(3), ee quat(4), gripper(1)] (radial-yaw+tilt down 자세).
        position 모드(--ik_position_only): action = [ee pos(3), gripper(1)] (orientation 자유).
        """
        root_pos = self.robot.data.root_pos_w[:, :3]
        root_quat = self.robot.data.root_quat_w
        grip = torch.tensor([[1.0 if gripper_open else -1.0]], device=self.device)
        if args.ik_position_only:
            pos_b, _ = subtract_frame_transforms(
                root_pos, root_quat, target_pos_w.view(1, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
            )
            action = torch.cat([pos_b, grip], dim=-1)
        else:
            # _down_quat_world 의 Rx(tilt) 수평성분이 gripper-local +Y 라 world 손가락 수평이
            # yaw+90° 로 돈다. base→target radial 방향으로 맞추려면 -90° 보정(측정 검증:
            # 자연 자세 fingerdir 수평이 base→cube 와 일치).
            yaw = self._yaw_to(target_pos_w[:2]) - math.pi / 2.0
            quat_w = _down_quat_world(yaw, tilt_deg, self.device)
            pos_b, quat_b = subtract_frame_transforms(
                root_pos, root_quat, target_pos_w.view(1, 3), quat_w
            )
            action = torch.cat([pos_b, quat_b, grip], dim=-1)
        self.env.step(action)
        if args.step_sleep > 0.0 and not args.headless:
            time.sleep(args.step_sleep)

    def move_to(self, target_fn, tilt_deg: float, gripper_open: bool, tol: float) -> bool:
        for _ in range(args.max_phase_steps):
            if not simulation_app.is_running():
                return False
            t = target_fn()
            self._act(t, tilt_deg, gripper_open)
            if self.reached(t, tol):
                return True
        return False

    def grasp_descend(self, cube_name: str, tilt_deg: float, tol: float) -> bool:
        """ee 를 큐브 xy 정중앙으로 당기는 over-drive 하강.

        5DOF DiffIK 가 ee 를 target 에 systematic 잔차(예: robot 쪽으로 못 미침)로 도달시키므로,
        target = 2*cube - ee 로 잔차만큼 더 멀리 명령하면 ee 가 큐브 중앙으로 수렴한다.
        (단순히 cube 를 target 으로 주면 ee 가 +x 로 2cm 치우쳐 큐브가 갭 가장자리 → 닫을 때 밀림.)
        """
        for _ in range(args.max_phase_steps):
            if not simulation_app.is_running():
                return False
            cb = self.obj_pos(cube_name)
            ee = self.ee_pos()
            tgt_z = cb[2].item() + args.grasp_z_offset
            tgt = cb.clone()
            # xy 는 강 over-drive(게인 1.0)로 정렬 우선(grasp 핵심), z 는 약(0.5)으로 xy 덜 희생.
            tgt[0] = 2.0 * cb[0] - ee[0]
            tgt[1] = 2.0 * cb[1] - ee[1]
            tgt[2] = tgt_z + 0.5 * (tgt_z - ee[2].item())
            self._act(tgt, tilt_deg, True)
            ee2 = self.ee_pos()
            cb2 = self.obj_pos(cube_name)
            if (torch.linalg.norm(ee2[:2] - cb2[:2]).item() < tol
                    and abs(ee2[2].item() - (cb2[2].item() + args.grasp_z_offset)) < tol):
                return True
        return False

    def hold(self, target_pos_w: torch.Tensor, tilt_deg: float, gripper_open: bool, steps: int) -> None:
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self._act(target_pos_w, tilt_deg, gripper_open)

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
        gtilt, ctilt = args.grasp_tilt_deg, args.carry_tilt_deg

        # 1) 큐브 상공 접근 (grasp tilt 자세, 그리퍼 열림)
        r1 = self.move_to(lambda: self._above(self.obj_pos(cube_name), args.approach_height), gtilt, True, coarse)
        # 2) grasp 높이 하강 (closed-loop 보정 옵션)
        if args.grasp_closed_loop:
            r2 = self.grasp_descend(cube_name, gtilt, fine)
        else:
            r2 = self.move_to(lambda: self._above(self.obj_pos(cube_name), args.grasp_z_offset), gtilt, True, fine)
        ee = self.ee_pos()
        cp = self.obj_pos(cube_name)
        jaw = self.robot.data.body_pos_w[0, self._jaw_idx]
        gq = self.robot.data.body_quat_w[0, self.ee_body_idx]
        # gripper-local -Z(손가락 방향) 의 world 벡터 — grasp 접근축 진단.
        fwd = quat_apply(gq.unsqueeze(0), torch.tensor([[0.0, 0.0, -1.0]], device=self.device)).squeeze(0)
        log(f"[SM]   {cube_name} descend reached={r2}(app={r1}) "
            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) cube=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f}) "
            f"jaw=({jaw[0]:.3f},{jaw[1]:.3f},{jaw[2]:.3f}) "
            f"gquat=({gq[0]:.3f},{gq[1]:.3f},{gq[2]:.3f},{gq[3]:.3f}) "
            f"fingerdir=({fwd[0]:.2f},{fwd[1]:.2f},{fwd[2]:.2f})")
        # grasp 시점 위치 캡처(이후 잡힌 큐브가 따라오므로 고정 목표 사용)
        grasp_xy = self.obj_pos(cube_name)[:2].clone()
        grasp_z = self.obj_pos(cube_name)[2].item()
        grasp_target = self._xyz(grasp_xy, grasp_z + args.grasp_z_offset)
        # 3) 그리퍼 닫고 정착 (자세 유지)
        self.hold(grasp_target, gtilt, False, args.grasp_dwell)
        ca = self.obj_pos(cube_name)
        gj = self.robot.data.joint_pos[0, self._gripper_jid].item()
        jaw2 = self.robot.data.body_pos_w[0, self._jaw_idx]
        log(f"[SM]   after-close cube=({ca[0]:.3f},{ca[1]:.3f},{ca[2]:.3f}) "
            f"gripper_joint={gj:.3f} jaw=({jaw2[0]:.3f},{jaw2[1]:.3f},{jaw2[2]:.3f})")
        # 4) 들어올림 (carry tilt)
        self.move_to(lambda: self._xyz(grasp_xy, CUBE_DESK_TOP_Z + args.lift_height), ctilt, False, coarse)
        if not self.grasped(cube_name):
            log(f"[SM]   {cube_name}: grasp 실패(들리지 않음) — 건너뜀")
            return False
        # 5) 그릇 상공 운반
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), ctilt, False, coarse)
        # 6) 그릇 안으로 하강
        place_h = args.place_height + n_placed * args.stack_increment
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), place_h), ctilt, False, fine)
        # 7) release
        rel = self._above(self.obj_pos(BOWL_NAME), place_h)
        self.hold(rel, ctilt, True, args.release_dwell)
        # 8) 후퇴
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), ctilt, True, coarse)
        return True

    def run(self, active_cubes: list[str]) -> None:
        self.hold(self.home_pos, args.carry_tilt_deg, True, args.settle_steps)
        ordered = self.order_by_proximity(active_cubes)
        log(f"[SM] pick order (robot 근접순): {ordered}")
        for i, cube in enumerate(ordered):
            log(f"[SM] pick-and-place: {cube} (placed so far={i})")
            self.pick_and_place(cube, i)
        self.hold(self.home_pos, args.carry_tilt_deg, True, args.settle_steps)
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
    env_cfg = PickCubeSo101IkEnvCfg()
    log("[SM] env_cfg constructed.")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    log("[SM] scene/seed set.")
    env_cfg.actions.arm.scale = args.ik_scale
    if args.ik_position_only:
        env_cfg.actions.arm.controller.command_type = "position"
    if args.ik_lambda > 0:
        env_cfg.actions.arm.controller.ik_params["lambda_val"] = args.ik_lambda
    if args.gripper_velocity > 0:
        env_cfg.scene.robot.actuators["gripper"].velocity_limit_sim = args.gripper_velocity
    if args.gripper_close != 0.0:
        env_cfg.actions.gripper.close_command_expr = {"gripper": args.gripper_close}
    if args.gripper_open != 0.0:
        env_cfg.actions.gripper.open_command_expr = {"gripper": args.gripper_open}
    log("[SM] arm.scale set.")
    if args.arm_stiffness > 0:
        env_cfg.scene.robot.actuators["arm_joints"].stiffness = args.arm_stiffness
    if args.arm_damping > 0:
        env_cfg.scene.robot.actuators["arm_joints"].damping = args.arm_damping
    log("[SM] applying DR...")
    _apply_dr(env_cfg)
    log(f"[SM] DR applied. headless={args.headless!r} video={args.video!r}")
    # viewer(작업영역 뷰)는 GUI/녹화에서만 의미. (과거 silent exit 은 viewer 가 아니라 로그
    # 인코딩(em dash) 문제였고 수정됨. headless 는 viewport 없어 skip.)
    if not args.headless or args.video:
        env_cfg.viewer.eye = args.view_eye
        env_cfg.viewer.lookat = args.view_lookat
    log("[SM] env_cfg built — calling gym.make.")

    if args.video:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs("docs", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="docs",
            name_prefix="so101_ik_pick_place",
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    else:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    sm = So101PickPlaceSM(env)
    active_cubes = CUBE_NAMES[: args.active_objects]
    sm.run(active_cubes)

    if not args.headless and not args.video:
        while simulation_app.is_running():
            sm._act(sm.home_pos, args.carry_tilt_deg, True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # SystemExit/KeyboardInterrupt 포함 (진단)
        import traceback

        log(f"[SM] EXIT ({type(e).__name__}): " + traceback.format_exc())
        raise
    finally:
        simulation_app.close()
