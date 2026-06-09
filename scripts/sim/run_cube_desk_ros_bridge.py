"""cube_desk 씬을 Isaac Sim + ROS 2 bridge 로 띄우는 standalone 실행기 (PATH E, cuMotion+ROS).

NVIDIA Isaac ROS pick-and-place 튜토리얼과 같은 구조: Isaac Sim 이 로봇·물리·물체를
시뮬하고 ROS 2 bridge 로 관절 상태/명령을 주고받는다. SO-101 5DOF grasp 의 좌표 정합
문제(Lula↔USD)는 cuMotion 이 articulation frame 에서 직접 계획해 해소한다.

토폴로지(짝: ros2_ws so101_cumotion_*):
  pub  /isaac_joint_states   (sensor_msgs/JointState, 6관절)   → TopicBasedSystem state
  sub  /isaac_joint_commands (sensor_msgs/JointState, position) ← TopicBasedSystem cmd
  pub  /clock                (rosgraph_msgs/Clock)
  pub  /tf                   (base_link→Cube1..N/Bowl)          ← ground-truth 물체 포즈
  ※ /isaac_joint_states 는 controller 의 /follower/joint_states 와 분리해 피드백 루프 방지.
  ※ 물체 TF parent = base_link(=USD base 링크에 붙인 동명 Xform). SRDF virtual_joint
    (world→base_link)=identity 라 SM(ObjectPoseStore)은 lookup_transform("base_link", "Cube1")
    으로 base_link frame 포즈를 그대로 MoveIt 목표로 쓴다.

실행(Linux 서버, isaac 그룹) — 래퍼가 LD_LIBRARY_PATH(번들 ROS 2 lib)·DDS env 를 export 한다:
    scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1
  (직접 호출 시 isaacsim 번들 jazzy/lib 를 LD_LIBRARY_PATH 에 넣어야 ROS2 bridge 가 뜬다 —
   librmw_implementation.so 의 libament_index_cpp.so 의존성 해소. 래퍼 내용 참조.)

──────────────────────────────────────────────────────────────────────────────
구현 노트 (B안, 2026-06-09):
  A안(Isaac Lab InteractiveScene + GPU fabric 파이프라인)에서 OmniGraph JointState/
  ArticulationController 노드가 `omni.physx.tensors: expected device 0, received device -1`
  로 joint_states 값을 못 실었다. Isaac Lab 이 만든 GPU physics tensor view 와 OmniGraph
  노드가 만드는 view 가 충돌한 것이 원인.

  B안: scene 로드/시뮬 파이프라인을 **순수 `isaacsim.core.api.World`(CPU 백엔드) +
  `SingleArticulation`** 으로 교체. cube_desk scene.usd 와 SO-101 so101_follower.usd 를
  `add_reference_to_stage` 로 직접 stage 에 올린다. NVIDIA 공식 ROS2 예제와 동일 경로라
  OmniGraph 물리노드가 simulation view 를 단독 소유 → device 정합. OmniGraph/토픽/루프
  로직은 A안 그대로 재사용(OnPlaybackTick 으로 전환 — World 가 timeline 을 play 하므로).

  base 고정·좌표는 검증된 PickCubeEnvCfg 상수(_ROBOT_POS/_ROBOT_ROT)와 Isaac Lab
  schema 헬퍼(fix_root_link)를 재사용해 PATH C 시뮬과 동일한 로봇 배치를 보장한다.
"""

from __future__ import annotations

import argparse
import os

# DDS 전송 설정 — fastdds 가 participant 를 만들 때(첫 ROS2 노드 tick, python 시작 이후) 읽으므로
# 여기서 setdefault 해도 유효하다. UDPv4 강제는 host(브리지)↔container(ROS 스택)의 cross-UID
# SHM 공유 실패를 우회한다(브리지=일반 유저, 컨테이너=root → /dev/shm fastrtps 세그먼트 lock 충돌).
# ⚠ LD_LIBRARY_PATH(isaacsim 번들 jazzy/lib)는 동적 링커가 프로세스 시작 시 읽으므로 여기서
#   설정 불가 — 반드시 launch 전 export 한다. 래퍼 scripts/sim/run_cube_desk_ros_bridge.sh 참조.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cube_desk Isaac Sim ROS 2 bridge")
parser.add_argument("--num_cubes", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--dr", action="store_true", help="큐브 위치를 scatter 범위로 무작위화")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ⚠ 기본 headless experience(isaaclab.python.headless.kit)는 OmniGraph USD 그래프 생성을
# strip 해 "Unable to create prim for graph" 로 실패한다. enable_cameras=True 면 AppLauncher 가
# isaaclab.python.headless.rendering.kit(풀 렌더 + OmniGraph USD authoring)를 로드해 OmniGraph 가
# 동작한다. 카메라 자체는 쓰지 않는다 — 렌더 experience 만 필요.
args.enable_cameras = True
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 부팅 이후 import
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# OmniGraph + ROS 2 bridge extension 을 omni.graph.core import 전에 활성화.
enable_extension("omni.graph.action")
enable_extension("omni.graph.nodes")
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.ros2.bridge")

import omni.graph.core as og  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim.schemas as schemas  # noqa: E402 (순수 USD authoring — 시뮬 파이프라인 무관)
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH, ROBOT_USD_PATH  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    _CUBE_SCATTER_X_RANGE,
    _CUBE_SCATTER_Y_RANGE,
    _ROBOT_POS,
    _ROBOT_ROT,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

# 순수 isaacsim 경로의 stage prim 레이아웃 (env 네임스페이스 없음).
SCENE_PRIM = "/World/Scene"
ROBOT_PRIM = "/World/Robot"  # fix_root_link 후 articulation root 가 이 prim 으로 올라온다.
BASE_LINK_PRIM = f"{ROBOT_PRIM}/base/base_link"  # TF parent frame "base_link" 용 Xform.
JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"

# Isaac Lab PickCubeEnvCfg 의 검증된 actuator gain (leisaac SO101_FOLLOWER_CFG).
DRIVE_STIFFNESS = 17.8
DRIVE_DAMPING = 0.6


def _set_local_pose(prim, pos: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]) -> None:
    """prim 의 기존 translate/orient xformOp 값을 덮어쓴다.

    referenced 로봇 root(/so101_new_calib)는 translate/orient(quatd)/scale op 를 이미 가지고
    있다(translate=0, orient=identity, scale=1). 새 op 를 Add 하면 typeName/precision 충돌로
    Tf 에러가 나므로, 기존 op 를 찾아 precision 에 맞춰 값만 설정한다.
    """
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    w, x, y, z = (float(c) for c in quat_wxyz)

    translate_op = ops.get("xformOp:translate")
    if translate_op is None:
        translate_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*pos))

    orient_op = ops.get("xformOp:orient")
    if orient_op is None:
        orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    else:
        orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def build_ros_graph(object_prims: list[str]):
    """JointState pub/sub + Clock + 물체 TF OmniGraph (ROS 2 bridge, C++ only — rclpy 불필요).

    World 가 timeline 을 play 하므로 OnPlaybackTick 이 매 프레임 fire 한다(A안의 OnTick+
    수동 evaluate_sync 불필요). 물체 TF 는 parent=base_link Xform 기준으로 publish 한다.
    """
    graph, _, _, _ = og.Controller.edit(
        {"graph_path": "/ROSBridge", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishObjectTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnTick.outputs:tick", "PublishObjectTF.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishObjectTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishObjectTF.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", JOINT_STATES_TOPIC),
                ("PublishJointState.inputs:targetPrim", [ROBOT_PRIM]),
                ("SubscribeJointState.inputs:topicName", JOINT_COMMANDS_TOPIC),
                ("ArticulationController.inputs:targetPrim", [ROBOT_PRIM]),
                ("PublishObjectTF.inputs:parentPrim", [BASE_LINK_PRIM]),
                ("PublishObjectTF.inputs:targetPrims", object_prims),
            ],
        },
    )
    return graph


def main() -> None:
    np.random.seed(args.seed)

    active_cubes = CUBE_NAMES[: args.num_cubes]

    # World — 순수 isaacsim.core. backend="numpy"(CPU) → OmniGraph 물리노드가 simulation
    # view 를 단독 소유해 device 정합(A안 device -1 회피). cuMotion 제어엔 단일 로봇 +
    # 소수 큐브라 CPU 물리로 충분하다.
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # cube_desk scene.usd: SCENE_OFFSET 가 baked 돼 큐브/그릇이 이미 world 좌표에 author 됨.
    add_reference_to_stage(CUBE_DESK_USD_PATH, SCENE_PRIM)

    # SO-101 follower: defaultPrim(/so101_new_calib) 이 ROBOT_PRIM 으로 composes-in.
    robot_prim = add_reference_to_stage(ROBOT_USD_PATH, ROBOT_PRIM)
    # 로봇을 데스크 앞에 배치(PATH C 와 동일 pose). fixed joint 가 이 pose 에서 anchor 되도록
    # base 고정 전에 먼저 적용한다.
    _set_local_pose(robot_prim, _ROBOT_POS, _ROBOT_ROT)

    # base 고정. find/create fixed joint(world→base) + ArticulationRootAPI 를 부모(ROBOT_PRIM)로
    # 이동(PhysX parser 한계 회피). 이후 articulation root = ROBOT_PRIM.
    schemas.modify_articulation_root_properties(
        f"{ROBOT_PRIM}/base",
        ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    )

    # TF parent frame "base_link" 용 Xform — base 링크 자식, identity local 이라 base 와 정확히
    # 일치(USD base 링크명은 base_link 가 아니라 base 라 동명 Xform 을 새로 만든다).
    base_prim = world.stage.GetPrimAtPath(f"{ROBOT_PRIM}/base")
    UsdGeom.Xform.Define(world.stage, BASE_LINK_PRIM)

    # 물체 TF target 과 articulation wrapper.
    object_prims = [f"{SCENE_PRIM}/{n}" for n in active_cubes] + [f"{SCENE_PRIM}/{BOWL_NAME}"]
    robot = SingleArticulation(ROBOT_PRIM, name="so101_follower")
    world.scene.add(robot)

    # OmniGraph 는 reset(=play) 전 timeline 정지 상태에서 생성한다(그래프 wrap + OnPlaybackTick
    # 등록). 노드는 prim 경로만 참조하므로 prim 이 존재하면 된다.
    ros_graph = build_ros_graph(object_prims)

    # reset: 물리 뷰 초기화 + timeline play → OnPlaybackTick 시작.
    world.reset()

    # 검증된 actuator gain 적용(USD drive gain 은 micro 라 cuMotion 위치 명령 추종 불가).
    n_dof = robot.num_dof
    robot.get_articulation_controller().set_gains(
        kps=np.full(n_dof, DRIVE_STIFFNESS, dtype=np.float32),
        kds=np.full(n_dof, DRIVE_DAMPING, dtype=np.float32),
    )

    # 선택적 DR — scatter 범위에서 활성 큐브 위치 무작위화(간단 jitter).
    if args.dr:
        for name in active_cubes:
            cube = SingleRigidPrim(f"{SCENE_PRIM}/{name}")
            cube.initialize()
            pos, quat = cube.get_world_pose()
            pos = np.asarray(pos, dtype=np.float32).copy()
            pos[0] = float(np.random.uniform(*_CUBE_SCATTER_X_RANGE))
            pos[1] = float(np.random.uniform(*_CUBE_SCATTER_Y_RANGE))
            cube.set_world_pose(position=pos, orientation=quat)

    print(f"[bridge] ready. cubes={active_cubes}  dof={n_dof}", flush=True)
    print(f"[bridge] topics: {JOINT_STATES_TOPIC} / {JOINT_COMMANDS_TOPIC} / /clock / /tf"
          f"  (TF base_link→{active_cubes + [BOWL_NAME]})", flush=True)

    # 메인 루프: World.step 이 물리 step + 렌더 + OmniGraph(OnPlaybackTick) 평가를 한다.
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        with open("/tmp/bridge_diag.txt", "w") as _f:
            _f.write(traceback.format_exc())
        print("[bridge] EXCEPTION:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()
