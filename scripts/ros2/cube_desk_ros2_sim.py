"""Isaac Sim(Windows) cube_desk 장면 + ROS 2 브릿지 (WSL2 MoveIt/cuMotion 연동).

이 앱은 cube_desk scene.usd + SO-101 follower USD 를 띄우고, OmniGraph 로 ROS 2 토픽을
주고받는다 → WSL2 의 MoveIt2(+cuMotion) 가 path planning, 이 장면이 물리 실행.

브릿지 토픽 (topic_based_ros2_control 계약):
  Pub  /isaac_joint_states     (sensor_msgs/JointState, 6축)
  Sub  /isaac_joint_commands   (sensor_msgs/JointState) → ArticulationController
  Pub  /clock                  (rosgraph_msgs/Clock, use_sim_time)
  Pub  TF base_link→{Cube1..4,Bowl}  (orchestrator 가 tf2 로 객체 pose 취득)

선행(Windows): WSL2↔Windows DDS 브릿지 설정 (docs/PATH_C_ISAAC_SIM.md 브릿지 절).
실행:
  OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python scripts/ros2/cube_desk_ros2_sim.py

주의(M1 검증): OmniGraph 노드 타입명은 Isaac Sim 5.1 기준(isaacsim.*). 설치 빌드에 따라
ArticulationController 노드명(isaacsim.core.nodes.IsaacArticulationController)·base_link
prim 경로가 다를 수 있어 M1 smoke 에서 확인/조정한다.
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cube_desk ROS2 bridge sim")
parser.add_argument("--robot_prim", default="/World/Robot")
parser.add_argument("--scene_prim", default="/World/Scene")
parser.add_argument("--gui", action="store_true", help="GUI 창 표시(기본 headless). 관찰은 보통 WSL2 RViz 로.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
# 브릿지 서버는 headless 기본 — GUI _prepare_ui 단계 access violation 회피(pick_cube_state_machine.py 동일 패턴).
# 관찰은 WSL2 RViz(MoveIt) 로. GUI 가 필요하면 --gui.
args.headless = not args.gui
if args.livestream < 0:
    args.livestream = 0
args.enable_cameras = False

# ROS2/OmniGraph 확장을 **부팅 시점**에 로드(--kit_args --enable). 런타임 enable_extension 은
# D3D12 로 뜬 headless kit 위에 viewport→RTX Hydra 를 Vulkan 으로 재초기화시켜
# rtx.scenedb access violation(Aftermath 0xbad00009)을 유발하므로 금지. 부팅 시 로드하면
# 그래픽 인터페이스가 처음부터 일관돼 재init 이 없다(headless 는 viewport 창도 미생성).
_boot_exts = " ".join(
    f"--enable {e}" for e in ("isaacsim.ros2.bridge", "isaacsim.core.nodes", "omni.graph.action")
)
args.kit_args = (getattr(args, "kit_args", "") or "") + " " + _boot_exts

launcher = AppLauncher(args)
simulation_app = launcher.app

# ── 이하 모듈은 app 기동 후 import (Isaac/Isaac Lab 규칙). 확장은 부팅 시 이미 로드됨. ──
import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

# repo 상수 (좌표 단일 출처 — CLAUDE.md 동기화 규칙)
from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH, ROBOT_USD_PATH  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import _ROBOT_POS, _ROBOT_ROT  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import SO101_JOINT_ORDER  # noqa: E402
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def build_scene():
    """cube_desk scene.usd(desk+조명+큐브+그릇) + SO-101 USD 조립."""
    stage = get_current_stage()

    # scene.usd : /Scene 하위에 desk, mat, lights, Cube1..4, Bowl 포함.
    add_reference_to_stage(usd_path=_abs(CUBE_DESK_USD_PATH), prim_path=args.scene_prim)

    # robot : /World/Robot 에 배치 (월드 좌표 _ROBOT_POS, identity rot).
    add_reference_to_stage(usd_path=_abs(ROBOT_USD_PATH), prim_path=args.robot_prim)
    robot_prim = stage.GetPrimAtPath(args.robot_prim)
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*_ROBOT_POS))
    w, x, y, z = _ROBOT_ROT
    xform.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def tune_drives():
    """관절 drive 를 position 제어용으로 튜닝 (trajectory 추종 안정).

    포럼 권고: Drive Type=Force + stiffness/damping. Isaac Lab actuator(stiffness 17.8)
    값을 시작점으로 쓰되, ROS2 trajectory 추종이 느리면 stiffness 를 높인다.
    """
    stage = get_current_stage()
    for joint in SO101_JOINT_ORDER:
        # USD 내 joint prim 경로는 로봇 USD 구조에 따름 → 와일드카드 탐색.
        for prim in stage.Traverse():
            if prim.GetName() == joint and prim.IsA(UsdPhysics.Joint):
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(800.0)   # M1/M4 튜닝
                drive.CreateDampingAttr(40.0)
                break


def build_ros2_graph():
    """OmniGraph ROS2 브릿지 (joint state pub/sub + clock + 객체 TF)."""
    cube_paths = [f"{args.scene_prim}/{n}" for n in CUBE_NAMES]
    bowl_path = f"{args.scene_prim}/{BOWL_NAME}"
    base_link = f"{args.robot_prim}/base_link"  # M1 확인: USD 내 base_link prim 경로

    og.Controller.edit(
        {"graph_path": "/CubeDeskROS2", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PubJoint", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubJoint", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArtController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "PubClock.inputs:execIn"),
                ("OnTick.outputs:tick", "PubJoint.inputs:execIn"),
                ("OnTick.outputs:tick", "SubJoint.inputs:execIn"),
                ("OnTick.outputs:tick", "ArtController.inputs:execIn"),
                ("OnTick.outputs:tick", "PubTF.inputs:execIn"),
                ("Context.outputs:context", "PubClock.inputs:context"),
                ("Context.outputs:context", "PubJoint.inputs:context"),
                ("Context.outputs:context", "SubJoint.inputs:context"),
                ("Context.outputs:context", "PubTF.inputs:context"),
                ("SimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PubJoint.inputs:timeStamp"),
                # 구독한 명령 → ArticulationController
                ("SubJoint.outputs:jointNames", "ArtController.inputs:jointNames"),
                ("SubJoint.outputs:positionCommand", "ArtController.inputs:positionCommand"),
                ("SubJoint.outputs:velocityCommand", "ArtController.inputs:velocityCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PubClock.inputs:topicName", "/clock"),
                ("PubJoint.inputs:topicName", "/isaac_joint_states"),
                ("PubJoint.inputs:targetPrim", [args.robot_prim]),
                ("SubJoint.inputs:topicName", "/isaac_joint_commands"),
                ("ArtController.inputs:targetPrim", [args.robot_prim]),
                ("PubTF.inputs:topicName", "/tf"),
                ("PubTF.inputs:parentPrim", [base_link]),
                ("PubTF.inputs:targetPrims", cube_paths + [bowl_path]),
            ],
        },
    )


def main():
    sim = SimulationContext(stage_units_in_meters=1.0)
    build_scene()
    simulation_app.update()
    tune_drives()
    build_ros2_graph()

    sim.reset()
    print("[cube_desk_ros2_sim] 브릿지 실행. /isaac_joint_states 발행, "
          "/isaac_joint_commands 구독, base_link→객체 TF 발행. Ctrl+C 종료.")
    # 물리 + OmniGraph 평가 루프 (use_sim_time → /clock 발행).
    while simulation_app.is_running():
        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
