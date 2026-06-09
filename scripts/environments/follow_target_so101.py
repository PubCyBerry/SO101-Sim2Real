"""SO-101 Follow Target 예제 (cube_desk 씬) — Isaac Sim Core API standalone.

Isaac Sim 5.1 Pick & Place 튜토리얼의 *Follow Target* 단계를 SO-101 로 구현한다:
움직이는 target 을 로봇 end-effector(``gripper_frame_link``)가 매 프레임 IK 로 추종한다.

- 스타일: Isaac Lab ``ManagerBasedRLEnv`` 가 아니라 **Core API** (``World`` +
  ``SingleArticulation`` + ``ArticulationKinematicsSolver``). 튜토리얼 standalone 패턴.
- target 제어: 뷰포트에서 빨간 target 큐브를 **transform 기즈모로 드래그** (classic follow target).
- IK: SO-101 은 5-DOF arm 이라 full 6-DOF pose IK 불가 → **position-only**
  (``target_orientation=None``). 자세는 IK redundancy 가 결정. follow 데모는 grasp 접촉이
  없어 position 추종(수 cm 잔차 허용)만으로 충분하다.

Lula 자산·정합 상수는 검증된 ``pick_cube_state_machine.py`` (So101LulaIK) 에서 가져왔다.

실행 (GUI 인터랙티브 — 디스플레이/livestream 필요):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
        python scripts/environments/follow_target_so101.py

실행 (헤드리스 self-test — 사람 손 없이 EE 추종 자동 검증):
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
        python scripts/environments/follow_target_so101.py --headless --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


def _vec3(s: str) -> tuple[float, float, float]:
    p = [float(x) for x in s.split(",")]
    if len(p) != 3:
        raise argparse.ArgumentTypeError("expected 'x,y,z'")
    return (p[0], p[1], p[2])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="SO-101 follow target (cube_desk, Core API)")
parser.add_argument("--selftest", action="store_true",
                    help="헤드리스 자동 검증: target 을 몇 개 reachable 지점으로 옮기며 EE 추종 거리 확인")
parser.add_argument("--selftest_tol", type=float, default=0.06,
                    help="self-test 통과 임계 EE↔target 거리(m)")
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

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------

import isaaclab.sim as sim_utils  # noqa: E402 — pick_cube env 와 동일한 spawn(fix_root_link)
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

# Lula motion-generation 확장은 기본 비활성 → import 전 enable (SM 과 동일 순서).
enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver  # noqa: E402
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver  # noqa: E402

# ---------------------------------------------------------------------------
# 상수 (assets + 검증된 정합값; pick_cube_state_machine.py 출처)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD_PATH = _REPO_ROOT / "assets" / "robots" / "so101_follower.usd"
SCENE_USD_PATH = _REPO_ROOT / "assets" / "scenes" / "cube_desk" / "scene.usd"
RMPFLOW_DESCRIPTOR_PATH = _REPO_ROOT / "assets" / "robots" / "rmpflow" / "so101_robot_description.yaml"
RMPFLOW_URDF_PATH = _REPO_ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"

# Lula end-effector frame (URDF 의 TCP 프레임).
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


def main() -> int:
    log(f"[follow_target] main 시작 (selftest={args.selftest})")
    world = World(stage_units_in_meters=1.0)

    # cube_desk 씬(책상·큐브·그릇·조명 포함).
    add_reference_to_stage(usd_path=str(SCENE_USD_PATH), prim_path="/World/Scene")

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
    robot_spawn.func(
        "/World/Robot", robot_spawn,
        translation=tuple(float(x) for x in ROBOT_POS),
        orientation=tuple(float(x) for x in ROBOT_QUAT),
    )

    robot = SingleArticulation(
        prim_path="/World/Robot",
        name="so101",
        position=ROBOT_POS,
        orientation=ROBOT_QUAT,
    )
    world.scene.add(robot)

    target = VisualCuboid(
        prim_path="/World/target",
        name="follow_target",
        position=np.asarray(args.target_init, dtype=np.float32),
        scale=np.array([0.03, 0.03, 0.03], dtype=np.float32),
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    world.scene.add(target)

    # IK 솔버: Lula(URDF-local) + USD world 정합 base pose.
    lula = LulaKinematicsSolver(
        robot_description_path=str(RMPFLOW_DESCRIPTOR_PATH),
        urdf_path=str(RMPFLOW_URDF_PATH),
    )
    lula.set_robot_base_pose(RMPFLOW_BASE_POS_USD, RMPFLOW_BASE_QUAT_USD)

    log("[follow_target] world.reset() ...")
    world.reset()
    log(f"[follow_target] reset 완료. robot.num_dof={robot.num_dof}")

    # 물리 초기화 후 articulation 래퍼와 IK 솔버 연결.
    aks = ArticulationKinematicsSolver(robot, lula, EE_FRAME_NAME)

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

    # 초기 자세를 env 기본(전 joint 0)으로 — warm-start 안정화.
    robot.set_joint_positions(np.zeros(n, dtype=np.float32))

    set_camera_view(eye=np.asarray(args.view_eye), target=np.asarray(args.view_lookat))

    def follow_step() -> bool:
        """target world 위치로 position-only IK 1 step. 성공 시 True."""
        target_pos, _ = target.get_world_pose()
        action, ok = aks.compute_inverse_kinematics(
            target_position=np.asarray(target_pos, dtype=np.float32),
            target_orientation=None,
        )
        if ok:
            robot.apply_action(action)
        return bool(ok)

    if args.selftest:
        return _run_selftest(world, robot, target, aks, follow_step)
    return _run_interactive(world, follow_step)


def _run_interactive(world, follow_step) -> int:
    log("[follow_target] 재생(▶) 후 뷰포트에서 빨간 target 큐브를 기즈모로 드래그하세요. "
        "EE 가 추종합니다. (Ctrl+C / 창 닫기로 종료)")
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue
        follow_step()
    return 0


def _run_selftest(world, robot, target, aks, follow_step) -> int:
    """target 을 reachable 지점들로 옮기며 EE 추종 거리를 검사한다.

    EE world pose 는 ``aks.compute_end_effector_pose()`` (현재 관절 상태 FK + base pose) 로
    얻는다 — ``gripper_frame_link`` 는 URDF/Lula 전용 프레임이라 USD prim 으로 존재하지 않는다.

    주의: compute_end_effector_pose 는 **고정 가정 base** 기준 FK 라 베이스가 실제로 넘어져도
    (fix_root_link 누락 등) EE 거리는 통과로 보인다. 그래서 root world pose 이탈도 함께 검사한다.
    """
    log("[follow_target][selftest] 시작")
    # 물리·렌더가 안정되도록 워밍업.
    for _ in range(10):
        world.step(render=False)

    ok_count = 0
    for i, tgt in enumerate(SELFTEST_TARGETS):
        target.set_world_pose(position=tgt)
        last_ok = False
        for _ in range(args.settle_steps):
            world.step(render=False)
            last_ok = follow_step()
        ee_pos = np.asarray(aks.compute_end_effector_pose()[0], dtype=np.float32).reshape(3)
        dist = float(np.linalg.norm(ee_pos - tgt))
        passed = dist <= args.selftest_tol
        ok_count += int(passed)
        log(f"[selftest] #{i} target={tgt.tolist()} EE_dist={dist:.4f}m "
            f"ik_ok={last_ok} pass={passed}")

    # 베이스 고정 검사 — floating base 면 팔 반력으로 root 가 이탈한다.
    base_pos = np.asarray(robot.get_world_pose()[0], dtype=np.float32).reshape(3)
    base_drift = float(np.linalg.norm(base_pos - ROBOT_POS))
    base_ok = base_drift <= 0.02
    log(f"[selftest] base_drift={base_drift:.4f}m base_fixed={base_ok}")

    all_pass = ok_count == len(SELFTEST_TARGETS) and base_ok
    log(f"[selftest] {'OK' if all_pass else 'FAIL'} — {ok_count}/{len(SELFTEST_TARGETS)} "
        f"지점 추종(≤{args.selftest_tol}m), base_fixed={base_ok}")
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
