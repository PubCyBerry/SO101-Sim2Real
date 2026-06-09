"""Franka Panda pick-and-place rule-based state machine (cube_desk 씬).

Domain Randomization 으로 큐브/그릇 위치가 무작위화된 상태에서 Franka 7DOF 팔이
큐브들을 순차로 집어 그릇(Bowl)에 담는 것을 한 번의 스크립트 실행으로 보여준다.

설계
----
- ``Phase`` enum + per-env 상태 배열로 num_envs 개 SM 을 병렬 관리한다.
- 매 step 마다 모든 env 의 target/gripper 를 한꺼번에 계산해 ``env.step([N,9])`` 를
  1회 호출한다. Python 루프 안에서 ``env.step`` 을 반복하지 않는다.
- ``home_pos`` 는 env 0 의 robot root 기준 상대 오프셋으로 보존하여,
  각 env 의 robot root world 좌표가 달라도 올바른 world target 을 계산한다.
- 이동 단계(APPROACH/DESCEND/LIFT/TRANSPORT/LOWER/RETREAT)는 도달 OR
  max_phase_steps 를 초과하면 다음 단계로 전이한다(IK 미수렴 안전장치).

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \\
        scripts/environments/pick_cube_franka_state_machine.py \\
        --num_envs 4 --active_objects 4
"""

from __future__ import annotations

import argparse
import sys
from enum import IntEnum

from isaaclab.app import AppLauncher

_LOG_PATH = "/tmp/franka_sm_progress.txt"
open(_LOG_PATH, "w").close()  # 실행마다 진행 로그 초기화


def log(msg: str) -> None:
    """진행 로그. Isaac Sim 은 gym.make 로 SimulationContext 를 만들 때 stdout/stderr 를
    carb logger 로 재바인딩한다. 그 이후의 일반 print 는 (특히 출력을 파일로 리다이렉트한
    headless 실행에서) 묻히므로, 별도 파일에 직접 append 하고 원본 stderr fd 에도 쓴다."""
    with open(_LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg, file=sys.__stderr__, flush=True)


def _vec3(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (parts[0], parts[1], parts[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Franka pick-and-place state machine (cube_desk)")
parser.add_argument("--task", default="SimToReal-Franka-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0,
                    help="큐브 scatter DR 강도 (0=고정 spawn, 1=전체 workspace)")
parser.add_argument("--container_angle_scale", type=float, default=1.0,
                    help="그릇 arc DR 강도 (0=고정, 1=기본 각도범위)")
parser.add_argument("--seed", type=int, default=0)
# 도달 기반 제어 파라미터
parser.add_argument("--reach_tol", type=float, default=0.012,
                    help="ee 가 목표에 도달했다고 볼 거리(m). 작을수록 정밀하나 느림")
parser.add_argument("--coarse_tol", type=float, default=0.03,
                    help="상공 경유(approach/above-bowl/retreat) 같은 거친 단계의 도달 허용 거리(m)")
parser.add_argument("--max_phase_steps", type=int, default=300,
                    help="한 단계에서 도달 못해도 넘어가는 step 상한(IK 미수렴 안전장치)")
parser.add_argument("--ik_scale", type=float, default=1.0,
                    help="IK action scale. 낮추면 step 당 ee 이동이 작아 손목 ±180° flip(한 바퀴 회전)이 억제되나, 너무 낮으면 도달·정밀 수렴이 안 돼 grasp 실패함(0.8 이하 비권장)")
parser.add_argument("--grasp_dwell", type=int, default=30, help="그리퍼 닫힘 정착 step")
parser.add_argument("--release_dwell", type=int, default=15, help="그리퍼 열림 정착 step")
parser.add_argument("--settle_steps", type=int, default=15, help="reset 후 큐브 정착 대기 step")
# 높이/오프셋 (m)
parser.add_argument("--approach_height", type=float, default=0.13, help="큐브 위 접근 높이")
parser.add_argument("--grasp_z_offset", type=float, default=-0.003,
                    help="grasp 시 큐브 중심 기준 tip z 오프셋. 음수면 더 깊이 물어 grasp 마진↑")
parser.add_argument("--lift_height", type=float, default=0.18, help="책상 윗면 기준 들어올림 높이")
parser.add_argument("--transport_height", type=float, default=0.20, help="그릇 위 운반 높이")
parser.add_argument("--place_height", type=float, default=0.07, help="그릇 바닥 기준 release 높이")
parser.add_argument("--stack_increment", type=float, default=0.025, help="이미 담긴 큐브당 release 높이 증가")
# GUI 초기 카메라(사이드뷰) — world 좌표. headless 에선 무시됨.
parser.add_argument("--view_eye", type=_vec3, default=(3.05, -0.78, 1.02),
                    help="GUI 카메라 위치 'x,y,z' (기본: 책상 +X 측면 약간 위)")
parser.add_argument("--view_lookat", type=_vec3, default=(1.74, -0.38, 0.74),
                    help="GUI 카메라 주시점 'x,y,z' (기본: 큐브·그릇 작업영역 중심)")
parser.add_argument("--video", action="store_true",
                    help="현재 사이드뷰(viewer eye/lookat)를 mp4 로 녹화해 docs/ 에 저장")
parser.add_argument("--video_length", type=int, default=2000, help="녹화 최대 프레임(step) 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 녹화는 viewport rgb 렌더가 필요 → 카메라 활성화.
if args.video:
    args.enable_cameras = True

# AppLauncher 부팅 (isaac 모듈 import 전에).
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

import sim_to_real  # noqa: E402, F401  (Gym 환경 등록 트리거)
from sim_to_real.tasks.pick_cube_franka.pick_cube_franka_env_cfg import (  # noqa: E402
    FRANKA_EE_OFFSET,
    PickCubeFrankaEnvCfg,
)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

# cube_desk 책상 윗면 world z (lift/transport 절대 높이, grasped 판정 기준).
DESK_TOP_Z = 0.709


# ---------------------------------------------------------------------------
# Domain Randomization — events 만 강도 조정 (apply_curriculum 의 events 부분 발췌)
# ---------------------------------------------------------------------------


def _apply_dr(env_cfg: PickCubeFrankaEnvCfg) -> None:
    """object_radius_scale / container_angle_scale 을 reset 이벤트에 in-place 반영.

    apply_curriculum() 은 SO-101 reward/termination(gripper body "gripper" 가정)에
    의존해 Franka 와 호환되지 않으므로, robot 무관한 DR 이벤트 부분만 가져온다.
    """
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


class Phase(IntEnum):
    SETTLE        = 0
    APPROACH      = 1
    DESCEND       = 2
    GRASP_DWELL   = 3
    LIFT          = 4
    TRANSPORT     = 5
    LOWER         = 6
    RELEASE_DWELL = 7
    RETREAT       = 8
    HOME_FINAL    = 9
    DONE          = 10


# 이동 단계: max_phase_steps timeout 적용 대상
_MOVE_PHASES = frozenset({
    Phase.APPROACH, Phase.DESCEND, Phase.LIFT,
    Phase.TRANSPORT, Phase.LOWER, Phase.RETREAT,
})


class FrankaPickPlaceSM:
    """num_envs 병렬 env pick-and-place 컨트롤러.

    매 step 마다 모든 env 의 phase 별 target 을 계산하고
    ``env.step([N, 9])`` 를 1회 호출한다.
    """

    def __init__(self, env, active_cubes: list[str]) -> None:
        self.env      = env
        self.scene    = env.unwrapped.scene
        self.device   = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.robot    = self.scene["robot"]

        self.ee_body_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.ee_offset   = torch.tensor(FRANKA_EE_OFFSET, device=self.device)

        # world-frame 수직 아래 방향 quat [N, 4] — subtract_frame_transforms 에 직접 전달
        self.world_down_quat = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]], device=self.device
        ).expand(self.num_envs, -1)

        # home_pos: env 0 의 robot root world 좌표 기준 상대 오프셋으로 보존.
        # Isaac Lab multi-env 는 robot 을 평행 이동만 하므로 오프셋이 모든 env 에 동일하게 적용된다.
        _home_w  = torch.tensor([1.84, -0.40, DESK_TOP_Z + 0.25], device=self.device)
        _root_0  = self.robot.data.root_pos_w[0, :3].clone()
        self.home_offset = _home_w - _root_0  # [3]

        # per-env 큐브 처리 순서 (robot root 근접 순)
        self.ordered_cubes: list[list[str]] = [
            self._order_by_proximity(active_cubes, e)
            for e in range(self.num_envs)
        ]

        # per-env 상태 벡터
        self.phase       : list[Phase]                              = [Phase.SETTLE] * self.num_envs
        self.cube_idx    : list[int]                                = [0]            * self.num_envs
        self.n_placed    : list[int]                                = [0]            * self.num_envs
        self.dwell_count : list[int]                                = [0]            * self.num_envs
        self.phase_steps : list[int]                                = [0]            * self.num_envs
        self.grasp_cache : list[tuple[torch.Tensor, float] | None] = [None]         * self.num_envs

    # --- 위치 쿼리 -------------------------------------------------------

    def _home_pos_w(self, e: int) -> torch.Tensor:
        return self.robot.data.root_pos_w[e, :3] + self.home_offset

    def obj_pos(self, name: str, e: int) -> torch.Tensor:
        return self.scene[name].data.root_pos_w[e, :3].clone()

    def ee_pos(self, e: int) -> torch.Tensor:
        bp = self.robot.data.body_pos_w[e, self.ee_body_idx]
        bq = self.robot.data.body_quat_w[e, self.ee_body_idx]
        return bp + quat_apply(bq.unsqueeze(0), self.ee_offset.unsqueeze(0)).squeeze(0)

    # --- 조건 체크 -------------------------------------------------------

    def _reached(self, target: torch.Tensor, e: int, tol: float) -> bool:
        return torch.linalg.norm(self.ee_pos(e) - target).item() < tol

    def _grasped(self, cube: str, e: int) -> bool:
        return self.obj_pos(cube, e)[2].item() > DESK_TOP_Z + 0.03

    def _placed(self, cube: str, e: int) -> bool:
        p       = self.obj_pos(cube, e)
        bowl_xy = self.obj_pos(BOWL_NAME, e)[:2]
        in_xy   = torch.linalg.norm(p[:2] - bowl_xy).item() < BOWL_SUCCESS_RADIUS
        z_rel   = p[2].item() - DESK_TOP_Z
        in_z    = BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10
        return in_xy and in_z

    # --- target 헬퍼 -----------------------------------------------------

    @staticmethod
    def _above(pos: torch.Tensor, height: float) -> torch.Tensor:
        t = pos.clone()
        t[2] = t[2] + height
        return t

    def _xyz(self, xy: torch.Tensor, z: float) -> torch.Tensor:
        return torch.tensor([xy[0].item(), xy[1].item(), z], device=self.device)

    # --- 정렬 ------------------------------------------------------------

    def _order_by_proximity(self, cubes: list[str], e: int) -> list[str]:
        base_xy = self.robot.data.root_pos_w[e, :2]
        return sorted(
            cubes,
            key=lambda c: torch.linalg.norm(self.obj_pos(c, e)[:2] - base_xy).item(),
        )

    # --- per-env phase 전이 ----------------------------------------------

    def _advance_cube(self, e: int) -> None:
        """현재 큐브 처리 완료 → 다음 큐브로, 없으면 HOME_FINAL."""
        self.cube_idx[e]    += 1
        self.phase_steps[e]  = 0
        if self.cube_idx[e] >= len(self.ordered_cubes[e]):
            self.dwell_count[e] = 0
            self.phase[e]       = Phase.HOME_FINAL
        else:
            self.phase[e] = Phase.APPROACH

    def _compute_action(self, e: int) -> tuple[torch.Tensor, bool]:
        """env e 의 현재 phase 에 따라 (target_pos_w [3], gripper_open) 반환 + phase 전이."""
        ph = self.phase[e]

        if ph == Phase.DONE:
            return self._home_pos_w(e), True

        # ----- SETTLE: reset 후 큐브 정착 대기 -----
        if ph == Phase.SETTLE:
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.settle_steps:
                self.dwell_count[e] = 0
                self.phase[e]       = Phase.APPROACH
                log(f"[SM] env{e}: pick order = {self.ordered_cubes[e]}")
            return self._home_pos_w(e), True

        cube = self.ordered_cubes[e][self.cube_idx[e]]

        # 이동 단계 step 카운터 증가 (dwell 단계는 별도 dwell_count 사용)
        if ph in _MOVE_PHASES:
            self.phase_steps[e] += 1
        timeout = self.phase_steps[e] >= args.max_phase_steps

        # ----- APPROACH: 큐브 상공 접근 -----
        if ph == Phase.APPROACH:
            target = self._above(self.obj_pos(cube, e), args.approach_height)
            if self._reached(target, e, args.coarse_tol) or timeout:
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.DESCEND
            return target, True

        # ----- DESCEND: grasp 높이까지 하강 -----
        if ph == Phase.DESCEND:
            target = self._above(self.obj_pos(cube, e), args.grasp_z_offset)
            if self._reached(target, e, args.reach_tol) or timeout:
                self.grasp_cache[e] = (
                    self.obj_pos(cube, e)[:2].clone(),
                    self.obj_pos(cube, e)[2].item(),
                )
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.GRASP_DWELL
            return target, True

        # ----- GRASP_DWELL: 그리퍼 닫힘 정착 -----
        if ph == Phase.GRASP_DWELL:
            gc     = self.grasp_cache[e]
            target = self._xyz(gc[0], gc[1] + args.grasp_z_offset)
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.grasp_dwell:
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.LIFT
            return target, False

        # ----- LIFT: 책상 위로 들어올림 -----
        if ph == Phase.LIFT:
            gc     = self.grasp_cache[e]
            target = self._xyz(gc[0], DESK_TOP_Z + args.lift_height)
            if self._reached(target, e, args.coarse_tol) or timeout:
                if not self._grasped(cube, e):
                    log(f"[SM] env{e} {cube}: grasp 실패 — 다음 큐브")
                    self._advance_cube(e)
                else:
                    self.phase_steps[e] = 0
                    self.phase[e]       = Phase.TRANSPORT
            return target, False

        # ----- TRANSPORT: 그릇 상공으로 운반 -----
        if ph == Phase.TRANSPORT:
            target = self._above(self.obj_pos(BOWL_NAME, e), args.transport_height)
            if self._reached(target, e, args.coarse_tol) or timeout:
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.LOWER
            return target, False

        # ----- LOWER: 그릇 안으로 하강 -----
        if ph == Phase.LOWER:
            place_h = args.place_height + self.n_placed[e] * args.stack_increment
            target  = self._above(self.obj_pos(BOWL_NAME, e), place_h)
            if self._reached(target, e, args.reach_tol) or timeout:
                self.dwell_count[e] = 0
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.RELEASE_DWELL
            return target, False

        # ----- RELEASE_DWELL: 그리퍼 열림 정착 -----
        if ph == Phase.RELEASE_DWELL:
            place_h = args.place_height + self.n_placed[e] * args.stack_increment
            target  = self._above(self.obj_pos(BOWL_NAME, e), place_h)
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.release_dwell:
                self.dwell_count[e] = 0
                self.n_placed[e]   += 1
                self.phase_steps[e] = 0
                self.phase[e]       = Phase.RETREAT
            return target, True

        # ----- RETREAT: 그릇 위로 후퇴 -----
        if ph == Phase.RETREAT:
            target = self._above(self.obj_pos(BOWL_NAME, e), args.transport_height)
            if self._reached(target, e, args.coarse_tol) or timeout:
                self._advance_cube(e)
            return target, True

        # ----- HOME_FINAL: 홈 자세로 복귀 후 완료 -----
        if ph == Phase.HOME_FINAL:
            self.dwell_count[e] += 1
            if self.dwell_count[e] >= args.settle_steps:
                self.phase[e] = Phase.DONE
                self._report(e)
            return self._home_pos_w(e), True

        return self._home_pos_w(e), True

    # --- 배치 액션 -------------------------------------------------------

    def _act_all(self, targets_w: list[torch.Tensor], gripper_opens: list[bool]) -> None:
        """모든 env 의 target/gripper 를 배치로 env.step([N, 9]) 에 전달."""
        tgt       = torch.stack(targets_w, dim=0)          # [N, 3]
        root_pos  = self.robot.data.root_pos_w[:, :3]      # [N, 3]
        root_quat = self.robot.data.root_quat_w             # [N, 4]
        # world target → robot root frame (IK 가 root 기준 절대 pose 를 받음)
        pos_b, quat_b = subtract_frame_transforms(
            root_pos, root_quat, tgt, self.world_down_quat
        )
        grip   = torch.tensor([[1.0 if g else -1.0] for g in gripper_opens], device=self.device)
        action = torch.cat([pos_b, quat_b, grip], dim=-1)  # [N, 9]
        self.env.step(action)

    # --- 메인 루프 -------------------------------------------------------

    def run(self) -> None:
        while not all(p == Phase.DONE for p in self.phase):
            if not simulation_app.is_running():
                break
            targets, grippers = [], []
            for e in range(self.num_envs):
                t, g = self._compute_action(e)
                targets.append(t)
                grippers.append(g)
            self._act_all(targets, grippers)

    # --- 결과 리포트 -----------------------------------------------------

    def _report(self, e: int) -> None:
        cubes = self.ordered_cubes[e]
        n_ok  = sum(self._placed(c, e) for c in cubes)
        for c in cubes:
            p    = self.obj_pos(c, e)
            dist = torch.linalg.norm(p[:2] - self.obj_pos(BOWL_NAME, e)[:2]).item()
            log(f"[SM] env{e} {c}: dist_bowl={dist:.3f}m z={p[2].item():.3f} placed={self._placed(c, e)}")
        log(f"[SM] env{e} RESULT: {n_ok}/{len(cubes)} cubes in bowl.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    log("[SM] main entered.")
    env_cfg                  = PickCubeFrankaEnvCfg()
    env_cfg.scene.num_envs   = args.num_envs
    env_cfg.seed             = args.seed
    env_cfg.actions.arm.scale = args.ik_scale  # 손목 flip 억제
    _apply_dr(env_cfg)
    env_cfg.viewer.eye    = args.view_eye
    env_cfg.viewer.lookat = args.view_lookat
    log("[SM] env_cfg built — calling gym.make.")

    if args.video:
        import os

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs("docs", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="docs",
            name_prefix="franka_pick_place",
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    else:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    sm = FrankaPickPlaceSM(env, CUBE_NAMES[: args.active_objects])
    sm.run()

    if not args.headless and not args.video:
        while simulation_app.is_running():
            targets  = [sm._home_pos_w(e) for e in range(sm.num_envs)]
            grippers = [True] * sm.num_envs
            sm._act_all(targets, grippers)

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
