"""Franka Panda pick-and-place rule-based state machine (cube_desk 씬).

Domain Randomization 으로 큐브/그릇 위치가 무작위화된 상태에서 Franka 7DOF 팔이
큐브들을 순차로 집어 그릇(Bowl)에 담는 것을 한 번의 스크립트 실행으로 보여준다.

설계
----
- ``scan()``                : 모든 큐브·그릇의 현재 world 위치를 한 번에 읽는다.
- ``reached`` / ``grasped`` / ``placed`` : 단계 전이 판정(조건 체크).
- ``pick_and_place(cube)``   : 오브젝트 1개에 대한 집기→옮기기→놓기.
- ``move_to(...)``           : 목표 ee 위치에 **도달하면 즉시** 다음 단계로 넘어간다
                               (고정 step 대기를 없애 빠르다).

제어는 Isaac Lab task-space DifferentialIK. 목표 ee pose 의 **방향은 robot base
기준 수직 아래로 고정**하고(위치만 world→base 변환), 그래서 운반 중 손목(panda_joint7)이
±180° 로 휙 도는 현상을 막는다. 그리퍼는 binary(열림 +1 / 닫힘 -1) action.

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/pick_cube_franka_state_machine.py \
        --num_envs 1 --active_objects 4 --object_radius_scale 1.0 --container_angle_scale 1.0
"""

from __future__ import annotations

import argparse
import sys

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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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


class FrankaPickPlaceSM:
    """단일 env, 도달 조건 기반 pick-and-place 컨트롤러."""

    def __init__(self, env) -> None:
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        # IK 가 제어하는 작업점(panda_hand + body_offset)을 그대로 추적하기 위한 값.
        self.ee_body_idx = self.robot.find_bodies("panda_hand")[0][0]
        self.ee_offset = torch.tensor(FRANKA_EE_OFFSET, device=self.device)
        # **world frame** 수직 아래 자세. (w,x,y,z)=(0,1,0,0) = world x축 180° 회전.
        # 이 자세를 robot root frame 으로 변환해 IK 에 넘긴다(IK absolute pose 는 root 기준).
        # base 기준 고정 quat 을 쓰면 ee 가 작업영역 반대로 향해 도달하지 못한다.
        self.world_down_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device)
        self.home_pos = torch.tensor([1.84, -0.40, DESK_TOP_Z + 0.25], device=self.device)

    # --- 상태 스캔 -------------------------------------------------------

    def scan(self) -> dict[str, torch.Tensor]:
        """모든 큐브·그릇의 현재 world 위치(3,)를 dict 로 반환."""
        names = list(CUBE_NAMES) + [BOWL_NAME]
        return {n: self.env.scene[n].data.root_pos_w[0, :3].clone() for n in names}

    def obj_pos(self, name: str) -> torch.Tensor:
        return self.env.scene[name].data.root_pos_w[0, :3].clone()

    def ee_pos(self) -> torch.Tensor:
        """IK 작업점의 현재 world 위치."""
        bp = self.robot.data.body_pos_w[0, self.ee_body_idx]
        bq = self.robot.data.body_quat_w[0, self.ee_body_idx]
        return bp + quat_apply(bq.unsqueeze(0), self.ee_offset.unsqueeze(0)).squeeze(0)

    # --- 조건 체크 -------------------------------------------------------

    def reached(self, target_pos_w: torch.Tensor, tol: float) -> bool:
        return torch.linalg.norm(self.ee_pos() - target_pos_w).item() < tol

    def grasped(self, cube_name: str) -> bool:
        """큐브가 책상에서 들렸는지(grasp 성공 추정)."""
        return self.obj_pos(cube_name)[2].item() > DESK_TOP_Z + 0.03

    def placed(self, cube_name: str) -> bool:
        """큐브가 그릇 반경·높이 안에 안착했는지."""
        p = self.obj_pos(cube_name)
        bowl_xy = self.obj_pos(BOWL_NAME)[:2]
        in_xy = torch.linalg.norm(p[:2] - bowl_xy).item() < BOWL_SUCCESS_RADIUS
        z_rel = p[2].item() - DESK_TOP_Z
        in_z = BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10  # 적층 여유
        return in_xy and in_z

    # --- 저수준 액션 -----------------------------------------------------

    def _act(self, target_pos_w: torch.Tensor, gripper_open: bool) -> None:
        root_pos = self.robot.data.root_pos_w[:, :3]
        root_quat = self.robot.data.root_quat_w
        # world target pose(아래 향함) → robot root frame. 손목 ±180° flip 은 IK scale 을
        # 낮춰(env_cfg.actions.arm.scale) step 당 변화량을 제한하는 방식으로 억제한다.
        pos_b, quat_b = subtract_frame_transforms(
            root_pos, root_quat, target_pos_w.view(1, 3), self.world_down_quat
        )
        grip = torch.tensor([[1.0 if gripper_open else -1.0]], device=self.device)
        action = torch.cat([pos_b, quat_b, grip], dim=-1)
        self.env.step(action)

    def move_to(self, target_fn, gripper_open: bool, tol: float) -> bool:
        """목표(매 step 재평가)에 도달하면 즉시 종료. 미수렴 시 max_phase_steps 에서 탈출.

        Returns: 도달했으면 True.
        """
        for _ in range(args.max_phase_steps):
            if not simulation_app.is_running():
                return False
            t = target_fn()
            self._act(t, gripper_open)
            if self.reached(t, tol):
                return True
        return False

    def hold(self, target_pos_w: torch.Tensor, gripper_open: bool, steps: int) -> None:
        """한 자리에서 그리퍼 상태를 유지하며 정착(닫힘/열림 물리 시간 확보)."""
        for _ in range(steps):
            if not simulation_app.is_running():
                return
            self._act(target_pos_w, gripper_open)

    # --- target 헬퍼 -----------------------------------------------------

    @staticmethod
    def _above(pos: torch.Tensor, height: float) -> torch.Tensor:
        t = pos.clone()
        t[2] = t[2] + height
        return t

    def _xyz(self, xy: torch.Tensor, z: float) -> torch.Tensor:
        return torch.tensor([xy[0].item(), xy[1].item(), z], device=self.device)

    # --- 오브젝트 1개 pick-and-place -------------------------------------

    def pick_and_place(self, cube_name: str, n_placed: int) -> bool:
        coarse, fine = args.coarse_tol, args.reach_tol

        # 1) 큐브 상공 접근 (그리퍼 열림, 실시간 추종)
        r1 = self.move_to(lambda: self._above(self.obj_pos(cube_name), args.approach_height), True, coarse)
        # 2) grasp 높이까지 하강
        r2 = self.move_to(lambda: self._above(self.obj_pos(cube_name), args.grasp_z_offset), True, fine)
        ee = self.ee_pos()
        cp = self.obj_pos(cube_name)
        log(f"[SM]   {cube_name} descend reached={r2}(app={r1}) "
            f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) cube=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})")
        # grasp 시점 위치 캡처(이후 잡힌 큐브가 따라오므로 고정 목표 사용)
        grasp_xy = self.obj_pos(cube_name)[:2].clone()
        grasp_z = self.obj_pos(cube_name)[2].item()
        grasp_target = self._xyz(grasp_xy, grasp_z + args.grasp_z_offset)
        # 3) 그리퍼 닫고 정착
        self.hold(grasp_target, False, args.grasp_dwell)
        # 4) 들어올림
        self.move_to(lambda: self._xyz(grasp_xy, DESK_TOP_Z + args.lift_height), False, coarse)
        if not self.grasped(cube_name):
            log(f"[SM]   {cube_name}: grasp 실패(들리지 않음) — 건너뜀")
            return False
        # 5) 그릇 상공으로 운반 (실시간 그릇 추종)
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), False, coarse)
        # 6) 그릇 안으로 하강 (이미 담긴 큐브 수만큼 높이 증가)
        place_h = args.place_height + n_placed * args.stack_increment
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), place_h), False, fine)
        # 7) release
        release_target = self._above(self.obj_pos(BOWL_NAME), place_h)
        self.hold(release_target, True, args.release_dwell)
        # 8) 위로 후퇴(다음 큐브·적층 충돌 회피)
        self.move_to(lambda: self._above(self.obj_pos(BOWL_NAME), args.transport_height), True, coarse)
        return True

    # --- 전체 시퀀스 -----------------------------------------------------

    def order_by_proximity(self, cubes: list[str]) -> list[str]:
        """robot base 에서 xy 거리가 가까운 큐브부터 처리하도록 정렬."""
        base_xy = self.robot.data.root_pos_w[0, :2]
        return sorted(cubes, key=lambda c: torch.linalg.norm(self.obj_pos(c)[:2] - base_xy).item())

    def run(self, active_cubes: list[str]) -> None:
        self.hold(self.home_pos, True, args.settle_steps)
        ordered = self.order_by_proximity(active_cubes)
        log(f"[SM] pick order (robot 근접순): {ordered}")
        for i, cube in enumerate(ordered):
            log(f"[SM] pick-and-place: {cube} (placed so far={i})")
            self.pick_and_place(cube, i)
        self.hold(self.home_pos, True, args.settle_steps)
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
    env_cfg = PickCubeFrankaEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.actions.arm.scale = args.ik_scale  # 손목 flip 억제
    _apply_dr(env_cfg)
    # GUI 초기 뷰를 작업영역 사이드뷰로(기본은 하늘 높은 원점 부감).
    env_cfg.viewer.eye = args.view_eye
    env_cfg.viewer.lookat = args.view_lookat
    log("[SM] env_cfg built — calling gym.make.")

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("[SM] env created.")
    env.reset()
    log("[SM] reset done — DR applied.")

    sm = FrankaPickPlaceSM(env)
    active_cubes = CUBE_NAMES[: args.active_objects]
    sm.run(active_cubes)

    # GUI 실행이면 창을 닫을 때까지 home 자세 유지(데모 관찰용). headless 면 바로 종료.
    if not args.headless:
        while simulation_app.is_running():
            sm._act(sm.home_pos, True)

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
