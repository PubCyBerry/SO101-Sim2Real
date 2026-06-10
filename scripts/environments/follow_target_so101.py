"""SO-101 Follow Target 예제 (cube_desk 씬) — Isaac Sim Core API standalone.

Isaac Sim 5.1 Pick & Place 튜토리얼의 *Follow Target* 단계를 SO-101 로 구현한다:
움직이는 target 을 로봇 end-effector(``gripper_frame_link``)가 매 프레임 추종한다.

- 스타일: Isaac Lab ``ManagerBasedRLEnv`` 가 아니라 **Core API** (``World`` +
  ``SingleArticulation``). 튜토리얼 standalone 패턴.
- target 제어: 뷰포트에서 빨간 target 큐브를 **transform 기즈모로 드래그** (classic follow target).
- 컨트롤러 두 가지(``--controller``):
  - ``ik`` (기본): Lula ``ArticulationKinematicsSolver`` position-only IK + slew-limit
    + deadband. 가볍고 정밀(sub-cm). 충돌 회피는 없음.
  - ``rmpflow``: ``RmpFlow`` 모션 정책. jerk/accel 제한으로 부드럽고, cube_desk 의
    큐브·그릇을 obstacle 로 등록해 **반응적 충돌 회피**. (5-DOF rmpflow config 는 스캐폴드)

SO-101 은 5-DOF arm 이라 full 6-DOF pose IK 불가 → 두 모드 모두 **position-only**.
로봇은 raw 참조하면 floating base 라 넘어지므로 pick_cube env 와 동일하게
``fix_root_link=True`` + soft-PD 로 spawn 한다. Lula 자산·정합 상수는 검증된
``pick_cube_state_machine.py`` 에서 가져왔다.

실행 (GUI 인터랙티브 — 디스플레이/livestream 필요):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
        python scripts/environments/follow_target_so101.py [--controller ik|rmpflow]

실행 (헤드리스 self-test):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
        python scripts/environments/follow_target_so101.py --headless --selftest \
        [--controller ik|rmpflow]
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# C 레벨 크래시(access violation 등)의 Python traceback 을 파일로 덤프.
os.makedirs("outputs", exist_ok=True)
_FH_FILE = open(os.path.abspath("outputs/follow_target_faulthandler.txt"), "w")
faulthandler.enable(file=_FH_FILE)


def _vec3(s: str) -> tuple[float, float, float]:
    p = [float(x) for x in s.split(",")]
    if len(p) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SO-101 follow target (cube_desk, Core API)")
parser.add_argument("--controller", choices=["ik", "rmpflow"], default="ik",
                    help="ik=Lula IK+slew/deadband(정밀), rmpflow=RMPFlow(부드러움+충돌 회피)")
parser.add_argument("--no_obstacles", action="store_true",
                    help="rmpflow 모드에서 큐브/그릇 obstacle 등록을 끔(raw 추종 비교용)")
parser.add_argument("--tune", action="store_true",
                    help="GUI 로 scene + 고정된 SO-101 만 띄우고 대기 — Lula Test Widget 으로 "
                         "RMPFlow/디스크립터를 직접 튜닝할 때 사용(자체 컨트롤러는 구동 안 함)")
parser.add_argument("--selftest", action="store_true",
                    help="헤드리스 자동 검증: target 을 몇 개 reachable 지점으로 옮기며 EE 추종 거리 확인")
parser.add_argument("--selftest_tol", type=float, default=0.06,
                    help="self-test 통과 임계 EE↔target 거리(m, ik 모드)")
parser.add_argument("--settle_steps", type=int, default=120,
                    help="self-test 각 지점에서 추종 정착까지 step 수")
parser.add_argument("--target_init", type=_vec3, default=(1.80, -0.42, 0.82),
                    help="target 초기 world 위치 'x,y,z'")
# GUI 초기 뷰 — 작업영역을 측면 근접에서 본다.
parser.add_argument("--view_eye", type=_vec3, default=(2.30, -0.72, 0.95))
parser.add_argument("--view_lookat", type=_vec3, default=(1.80, -0.43, 0.78))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# self-test 는 GUI 가 필요 없으므로 headless 강제(인자로 명시 안 했으면).
if args.selftest and not args.headless:
    args.headless = True

# vars(args) 전체를 넘기면 view_eye/view_lookat 같은 tuple 커스텀 인자가
# AppLauncher → carb 설정 경로로 전달되어 Windows에서 _prepare_ui access violation 발생.
# AppLauncher가 실제로 사용하는 키만 필터링해서 전달한다.
_LAUNCHER_KEYS = {
    "headless", "livestream", "enable_cameras", "experience", "device", "cpu",
    "disable_fabric", "offscreen_render", "kit_args",
}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(_launcher_args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import isaaclab.sim as sim_utils  # noqa: E402 — pick_cube env 와 동일한 spawn(fix_root_link)
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

# Lula/RMPFlow 확장은 기본 비활성 → import 전 enable (SM 과 동일 순서).
enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import (  # noqa: E402
    ArticulationKinematicsSolver,
    ArticulationMotionPolicy,
    RmpFlow,
)
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver  # noqa: E402

# ---------------------------------------------------------------------------
# 상수 (assets + 검증된 정합값; pick_cube_state_machine.py 출처)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD_PATH = _REPO_ROOT / "assets" / "robots" / "so101_follower.usd"
SCENE_USD_PATH = _REPO_ROOT / "assets" / "scenes" / "cube_desk" / "scene.usd"
RMPFLOW_DIR = _REPO_ROOT / "assets" / "robots" / "rmpflow"
RMPFLOW_DESCRIPTOR_PATH = RMPFLOW_DIR / "so101_robot_description.yaml"
RMPFLOW_CONFIG_PATH = RMPFLOW_DIR / "so101_rmpflow_config.yaml"
RMPFLOW_URDF_PATH = _REPO_ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"

# Lula/RMPFlow end-effector frame (URDF 의 TCP 프레임).
EE_FRAME_NAME = "gripper_frame_link"

# Lula base pose (USD world). cube_desk 씬에서 least-squares 로 정합한 검증값.
RMPFLOW_BASE_POS_USD = np.array([1.81791970, -0.58952723, 0.70832908], dtype=np.float32)
RMPFLOW_BASE_QUAT_USD = np.array([0.71116823, -0.00950808, 0.01529776, 0.70279110], dtype=np.float32)  # wxyz

# Robot articulation root 배치 — pick_cube env (_ROBOT_POS/_ROBOT_ROT) 와 동일해야
# RMPFLOW_BASE 정합값이 그대로 유효하다.
ROBOT_POS = np.array([1.84, -0.565, 0.6749], dtype=np.float32)
ROBOT_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # wxyz

# Actuator soft-PD — pick_cube env(ImplicitActuatorCfg, leisaac 검증값) 과 동일.
# raw USD 드라이브 게인은 달라 IK target 으로 급스냅하며 물체를 쳐낼 수 있다.
ROBOT_STIFFNESS = 17.8
ROBOT_DAMPING = 0.6
ROBOT_MAX_EFFORT = 10.0

PHYSICS_DT = 1.0 / 60.0

# ik 모드 떨림 완화 — position-only 5-DOF 는 redundancy 로 null-space 해가 튀어 떨린다.
IK_TARGET_DEADBAND = 0.003  # target 이 이만큼(m) 안 움직이면 IK 재계산 생략 → 정지 시 떨림 제거
IK_MAX_JOINT_DELTA = 0.05   # step 당 arm joint 목표 변화 상한(rad) → 급점프 완화

# rmpflow 모드 obstacle — cube_desk 큐브/그릇을 inflated cuboid 로 근사 등록.
# (scale = 충돌 박스 전체 크기 m. 큐브 2.5cm 를 안전 마진 포함 ~6cm 로 부풀림.)
_OBSTACLE_SCALES: dict[str, tuple[float, float, float]] = {
    "Cube1": (0.06, 0.06, 0.06),
    "Cube2": (0.06, 0.06, 0.06),
    "Cube3": (0.06, 0.06, 0.06),
    "Cube4": (0.06, 0.06, 0.06),
    "Bowl": (0.16, 0.16, 0.09),
}

# self-test 에서 target 을 옮길 reachable 지점들 (책상 위 작업영역).
SELFTEST_TARGETS = [
    np.array([1.80, -0.42, 0.82], dtype=np.float32),
    np.array([1.90, -0.40, 0.80], dtype=np.float32),
    np.array([1.74, -0.45, 0.78], dtype=np.float32),
    np.array([1.86, -0.38, 0.84], dtype=np.float32),
]


_LOG_PATH = "/tmp/so101_follow_target.txt"


def log(msg: str) -> None:
    """Isaac Sim 이 stdout/stderr 를 carb 로 재바인딩해 print 가 묻히므로,
    진행 로그를 파일에 append 하고 원본 stderr fd 에도 쓴다 (SM 과 동일 패턴)."""
    with open(_LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg, file=sys.__stderr__, flush=True)


def _find_prim_path(name: str) -> str | None:
    """스테이지에서 이름이 일치하는 첫 prim 경로 (obstacle 좌표 조회용)."""
    for prim in get_current_stage().Traverse():
        if prim.GetName() == name:
            return prim.GetPath().pathString
    return None


# ---------------------------------------------------------------------------
# 컨트롤러 빌더 — (follow_step, ee_pos_fn) 반환
# ---------------------------------------------------------------------------


def _build_ik_controller(robot, target):
    """Lula position-only IK + slew-limit + deadband.

    deadband: target 이 거의 안 움직이면 IK 재계산을 생략해 정지 시 null-space 떨림을 없앤다.
    slew-limit: 해가 튈 때 joint 목표 변화율을 제한해 급점프를 완화한다.
    """
    lula = LulaKinematicsSolver(
        robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
        urdf_path=str(RMPFLOW_URDF_PATH),
    )
    lula.set_robot_base_pose(RMPFLOW_BASE_POS_USD, RMPFLOW_BASE_QUAT_USD)
    aks = ArticulationKinematicsSolver(robot, lula, EE_FRAME_NAME)

    state: dict = {"prev_target": None, "desired": None, "idx": None, "cmd": None}

    def follow_step() -> bool:
        tgt = np.asarray(target.get_world_pose()[0], dtype=np.float32).reshape(3)
        moved = state["prev_target"] is None or float(np.linalg.norm(tgt - state["prev_target"])) >= IK_TARGET_DEADBAND
        if moved:
            action, ok = aks.compute_inverse_kinematics(target_position=tgt, target_orientation=None)
            if ok and action.joint_positions is not None:
                state["desired"] = np.asarray(action.joint_positions, dtype=np.float32)
                state["idx"] = np.asarray(action.joint_indices)
                state["prev_target"] = tgt
            elif state["desired"] is None:
                return False  # 아직 한 번도 못 풀었으면 명령 없음
        if state["desired"] is None:
            return False
        if state["cmd"] is None:  # 첫 명령은 현재 관절에서 시작해 초기 급스냅 방지
            state["cmd"] = np.asarray(robot.get_joint_positions(), dtype=np.float32)[state["idx"]]
        delta = np.clip(state["desired"] - state["cmd"], -IK_MAX_JOINT_DELTA, IK_MAX_JOINT_DELTA)
        state["cmd"] = state["cmd"] + delta
        robot.apply_action(ArticulationAction(joint_positions=state["cmd"], joint_indices=state["idx"]))
        return True

    def ee_pos_fn() -> np.ndarray:
        return np.asarray(aks.compute_end_effector_pose()[0], dtype=np.float32).reshape(3)

    return follow_step, ee_pos_fn


def _register_obstacles(rmp) -> int:
    """cube_desk 의 큐브·그릇을 invisible inflated cuboid obstacle 로 등록 (static)."""
    n = 0
    for name, scale in _OBSTACLE_SCALES.items():
        path = _find_prim_path(name)
        if path is None:
            log(f"[follow_target] obstacle '{name}' prim 못 찾음 — 건너뜀")
            continue
        pos, quat = SingleXFormPrim(path).get_world_pose()
        obstacle = VisualCuboid(
            prim_path=f"/World/obstacles/{name}",
            name=f"obs_{name}",
            position=np.asarray(pos, dtype=np.float32),
            orientation=np.asarray(quat, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.float32),
            visible=False,
        )
        rmp.add_obstacle(obstacle, static=True)
        n += 1
    log(f"[follow_target] obstacle {n}개 등록")
    return n


def _build_rmpflow_controller(robot, target):
    """RMPFlow 모션 정책 — 부드러운 추종 + 큐브/그릇 반응적 회피."""
    rmp = RmpFlow(
        robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
        urdf_path=str(RMPFLOW_URDF_PATH),
        rmpflow_config_path=str(RMPFLOW_CONFIG_PATH),
        end_effector_frame_name=EE_FRAME_NAME,
        maximum_substep_size=PHYSICS_DT,
    )
    rmp.set_robot_base_pose(RMPFLOW_BASE_POS_USD, RMPFLOW_BASE_QUAT_USD)
    if args.no_obstacles:
        log("[follow_target] obstacle 등록 생략(--no_obstacles)")
    else:
        _register_obstacles(rmp)
    amp = ArticulationMotionPolicy(robot, rmp, default_physics_dt=PHYSICS_DT)

    def follow_step() -> bool:
        tgt = np.asarray(target.get_world_pose()[0], dtype=np.float32).reshape(3)
        rmp.set_end_effector_target(target_position=tgt, target_orientation=None)
        rmp.update_world()
        robot.apply_action(amp.get_next_articulation_action(PHYSICS_DT))
        return True

    def ee_pos_fn() -> np.ndarray:
        q = amp.get_active_joints_subset().get_joint_positions()
        pos, _ = rmp.get_end_effector_pose(q)
        return np.asarray(pos, dtype=np.float32).reshape(3)

    return follow_step, ee_pos_fn


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    log(f"[follow_target] main 시작 (controller={args.controller}, selftest={args.selftest})")
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

    # --tune: Lula Test Widget 의 IK/EE-viz 는 robot 이 world 원점에 있다고 가정한다
    # (set_robot_base_pose 미호출). cube_desk 배치(1.84,…)면 위젯 솔버가 원점 기준으로 풀어
    # EE 프레임이 원점에 박히고 target 이 도달 밖이 된다. → 원점 + ground plane 으로 띄운다.
    # (게인·default_q 튜닝은 배치 무관이라 결과는 cube_desk 에 그대로 적용된다.)
    if args.tune:
        world.scene.add_default_ground_plane()
        robot_pos, robot_quat = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    else:
        add_reference_to_stage(usd_path=str(SCENE_USD_PATH), prim_path="/World/Scene")
        robot_pos = tuple(float(x) for x in ROBOT_POS)
        robot_quat = tuple(float(x) for x in ROBOT_QUAT)

    # SO-101 follower: pick_cube env 와 동일하게 fix_root_link 로 spawn.
    # raw reference 만 하면 베이스가 떠 있어(floating base) 팔 반력으로 로봇이 넘어진다.
    robot_spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    )
    robot_spawn.func("/World/Robot", robot_spawn, translation=robot_pos, orientation=robot_quat)
    robot = SingleArticulation(prim_path="/World/Robot", name="so101",
                               position=np.asarray(robot_pos, dtype=np.float32),
                               orientation=np.asarray(robot_quat, dtype=np.float32))
    world.scene.add(robot)

    target = None
    if not args.tune:  # tune 모드는 위젯이 자체 target(/World/Target)을 만든다
        target = VisualCuboid(
            prim_path="/World/target",
            name="follow_target",
            position=np.asarray(args.target_init, dtype=np.float32),
            scale=np.array([0.03, 0.03, 0.03], dtype=np.float32),
            color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        world.scene.add(target)

    log("[follow_target] world.reset() ...")
    world.reset()
    log(f"[follow_target] reset 완료. robot.num_dof={robot.num_dof}")

    # env 와 동일한 soft-PD 게인·토크 상한 적용 (raw USD 드라이브 게인 무시).
    n = robot.num_dof
    controller = robot.get_articulation_controller()
    controller.set_gains(
        kps=np.full(n, ROBOT_STIFFNESS, dtype=np.float32),
        kds=np.full(n, ROBOT_DAMPING, dtype=np.float32),
    )
    try:
        controller.set_max_efforts(np.full(n, ROBOT_MAX_EFFORT, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001 — API 명칭 차이 대비 (게인만으로도 충분)
        log(f"[follow_target] set_max_efforts 생략: {exc}")

    robot.set_joint_positions(np.zeros(n, dtype=np.float32))

    if args.tune:  # 원점 로봇을 가까이서 본다
        set_camera_view(eye=np.array([1.0, -1.0, 0.8]), target=np.array([0.0, 0.0, 0.2]))
    else:
        set_camera_view(eye=np.asarray(args.view_eye), target=np.asarray(args.view_lookat))

    # --tune: 컨트롤러를 구동하지 않고 GUI 만 띄워 대기. Lula Test Widget 이 RMPFlow 로
    # 이 로봇을 잡아 튜닝한다(두 컨트롤러 충돌 방지). 베이스는 이미 fix_root_link 로 고정.
    if args.tune:
        return _run_tune(world)

    if args.controller == "rmpflow":
        follow_step, ee_pos_fn = _build_rmpflow_controller(robot, target)
    else:
        follow_step, ee_pos_fn = _build_ik_controller(robot, target)

    if args.selftest:
        return _run_selftest(world, robot, target, ee_pos_fn, follow_step)
    return _run_interactive(world, follow_step)


def _run_tune(world) -> int:
    """컨트롤러 없이 GUI 대기 (Lula Test Widget 튜닝용). headless 면 smoke 후 종료."""
    # 튜닝 UI 확장 활성화 — AppLauncher 앱엔 기본 비활성이라 Tools>Robotics 메뉴가 없다.
    for ext in ("isaacsim.robot_motion.lula_test_widget", "isaacsim.robot_setup.xrdf_editor"):
        try:
            enable_extension(ext)
        except Exception as exc:  # noqa: BLE001
            log(f"[follow_target][tune] 확장 '{ext}' 활성화 실패: {exc}")
    log("[follow_target][tune] world 원점에 고정된 SO-101 준비됨. "
        "Tools > Robotics > Lula Test Widget 으로 튜닝하세요. "
        "위젯 target(/World/Target)은 (0.5,0,0.5)에 생기니 SO-101 도달 범위인 "
        "~(0.2,0,0.25)로 옮기세요(Property>Transform).")
    if args.headless:  # smoke: 부팅·로딩만 확인하고 종료
        for _ in range(60):
            world.step(render=False)
        log("[follow_target][tune] headless smoke OK")
        return 0
    while simulation_app.is_running():
        world.step(render=True)
    return 0


def _run_interactive(world, follow_step) -> int:
    log(f"[follow_target] ({args.controller}) 재생(▶) 후 뷰포트에서 빨간 target 큐브를 기즈모로 "
        "드래그하세요. EE 가 추종합니다. (Ctrl+C / 창 닫기로 종료)")
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue
        follow_step()
    return 0


def _run_selftest(world, robot, target, ee_pos_fn, follow_step) -> int:
    """target 을 reachable 지점들로 옮기며 EE 추종 거리 + 베이스 고정을 검사한다.

    rmpflow 는 attractor+회피라 IK 보다 잔차가 크므로 tolerance·정착 step 을 완화한다.
    EE world pose 측정은 모드별 ee_pos_fn 사용 (ik=Lula FK, rmpflow=RmpFlow FK).
    """
    # ik 는 정밀 게이트(sub-cm), rmpflow 는 smoke(≤0.18m) — 5-DOF RMPFlow scaffold 는
    # IK 만큼 정밀하지 않다. rmpflow 추종 검증은 --no_obstacles 로 한다
    # (obstacle 켜면 workspace target 이 회피 영역 안이라 EE 가 일부러 거리를 둔다).
    rmpflow = args.controller == "rmpflow"
    tol = 0.18 if rmpflow else args.selftest_tol
    settle = args.settle_steps * 2 if rmpflow else args.settle_steps
    log(f"[follow_target][selftest] 시작 (controller={args.controller}, tol={tol}, settle={settle}, "
        f"obstacles={not args.no_obstacles})")

    for _ in range(10):  # 물리·렌더 워밍업
        world.step(render=False)

    ok_count = 0
    for i, tgt in enumerate(SELFTEST_TARGETS):
        target.set_world_pose(position=tgt)
        last_ok = False
        for _ in range(settle):
            world.step(render=False)
            last_ok = follow_step()
        dist = float(np.linalg.norm(ee_pos_fn() - tgt))
        passed = dist <= tol
        ok_count += int(passed)
        log(f"[selftest] #{i} target={tgt.tolist()} EE_dist={dist:.4f}m ik_ok={last_ok} pass={passed}")

    # 베이스 고정 검사 — floating base 면 팔 반력으로 root 가 이탈한다.
    base_pos = np.asarray(robot.get_world_pose()[0], dtype=np.float32).reshape(3)
    base_drift = float(np.linalg.norm(base_pos - ROBOT_POS))
    base_ok = base_drift <= 0.02
    log(f"[selftest] base_drift={base_drift:.4f}m base_fixed={base_ok}")

    all_pass = ok_count == len(SELFTEST_TARGETS) and base_ok
    log(f"[selftest] {'OK' if all_pass else 'FAIL'} — {ok_count}/{len(SELFTEST_TARGETS)} "
        f"지점 추종(≤{tol}m), base_fixed={base_ok}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    import traceback

    code = 1
    try:
        code = main()
    except Exception:  # noqa: BLE001 — close() 가 exit code 를 삼키므로 예외를 파일에 남긴다.
        log("[follow_target] 예외 발생:\n" + traceback.format_exc())
        code = 1
    finally:
        simulation_app.close()
    sys.exit(code)
