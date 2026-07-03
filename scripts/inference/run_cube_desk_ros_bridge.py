"""cube_desk 씬을 Isaac Sim + ROS 2 bridge 로 띄우는 standalone 실행기 (VLA closed-loop 추론 bridge).

NVIDIA Isaac ROS pick-and-place 튜토리얼과 같은 구조: Isaac Sim 이 로봇·물리·물체를
시뮬하고 ROS 2 bridge 로 관절 상태/명령을 주고받는다.

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
    scripts/inference/run_cube_desk_ros_bridge.sh --num_cubes 1
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
import json
import os
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# DDS 전송 설정 — fastdds 가 participant 를 만들 때(첫 ROS2 노드 tick, python 시작 이후) 읽으므로
# 여기서 setdefault 해도 유효하다. UDPv4 강제는 host(브리지)↔container(ROS 스택)의 cross-UID
# SHM 공유 실패를 우회한다(브리지=일반 유저, 컨테이너=root → /dev/shm fastrtps 세그먼트 lock 충돌).
# ⚠ LD_LIBRARY_PATH(isaacsim 번들 jazzy/lib)는 동적 링커가 프로세스 시작 시 읽으므로 여기서
#   설정 불가 — 반드시 launch 전 export 한다. 래퍼 scripts/inference/run_cube_desk_ros_bridge.sh 참조.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="cube_desk Isaac Sim ROS 2 bridge")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0",
                    help="실행할 등록 gym env id. env_cfg 를 로드해 actuator·DR 모드를 그 단일소스에서 "
                         "읽는다(gym.make 는 안 함 — B안 pure World 유지로 ROS OmniGraph device -1 회피). "
                         "예: SimToReal-SO101-PickCube-{v0,DR-v0,DRBase-v0,Eval-v0,DR-Eval-v0}. "
                         "-DR* 는 큐브/그릇 DR 자동 on. 기본 v0=고정 실측 배치.")
parser.add_argument("--num_cubes", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--cube_name", default="",
                    help="단일 활성 큐브 직접 지정(크기별 eval: Cube1/2=40mm·Cube3/4=50mm). "
                         "빈값=CUBE_NAMES[:num_cubes]. 비활성 큐브는 z=-1 park(카메라 밖).")
parser.add_argument("--dr", action="store_true",
                    help="지정 시에만 큐브·그릇 위치를 무작위화(학습 DR 정합: scatter+arc). 미지정 시 "
                         "env_cfg 고정 기본 위치(_CUBE_INIT_STATES/_BOWL_INIT_STATE)에 배치.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_cameras",
    action="store_true",
    help="top/wrist/front 카메라 ROS publish 비활성화 (기본은 publish — VLA obs 용).",
)
# GUI 뷰포트 레이아웃 — pick_cube_curobo_demo.py 와 동일(Perspective + top/wrist/front 3-패널 dock).
parser.add_argument("--view_eye", type=float, nargs=3, default=[0.632, 0.755, 1.317],
                    help="GUI Perspective 카메라 eye(world). GUI 카메라 실측 eye/lookat 정합.")
parser.add_argument("--view_lookat", type=float, nargs=3, default=[-0.269, -0.146, 0.416],
                    help="GUI Perspective 카메라 lookat(world). set_camera_view 자동조준(up=+z).")
parser.add_argument("--layout", default="assets/layouts/pick_cube_3cam.json",
                    help="viewport docking layout JSON(ui.Workspace dump). REPO_ROOT 상대경로. "
                         "없으면 수동 dock fallback. demo 와 동일 파일·window title 공유.")
# ── VLA 성공률 eval 모드 ─────────────────────────────────────────────────────
parser.add_argument("--eval", type=int, default=0,
                    help=">0 이면 VLA 성공률 eval: N 에피소드 auto-reset + cube-in-bowl 카운트 후 종료. "
                         "VLA(policy-server+vla-ros)는 별도 실행 중이어야 함(bridge 만 reset/카운트).")
parser.add_argument("--eval_seconds", type=float, default=30.0,
                    help="eval 에피소드당 sim 시간(초). PickCubeEnvCfg episode_length_s=30 매칭.")
parser.add_argument("--eval_settle", type=float, default=1.5,
                    help="reset 후 카운트 시작 전 settle 시간(초) — 큐브 안정 + VLA RTC 재정렬.")
parser.add_argument("--eval_warmup", type=float, default=25.0,
                    help="첫 에피소드 전 1회 warmup 시간(초) — vla-ros 가 obs 받아 /isaac_joint_commands "
                         "구동 시작할 때까지 대기(미구동 시 ep1 거짓 실패 방지).")
parser.add_argument("--eval_out", default="outputs/vla_eval.json",
                    help="eval 결과 JSON 경로(REPO_ROOT 상대).")
parser.add_argument(
    "--eval_bowl_kinematic",
    action="store_true",
    help="평가 A/B: bowl을 kinematic rigid body로 고정한다. 시각·충돌·DR pose는 유지하고 "
         "로봇/큐브 접촉으로 target bowl이 밀려나는 효과만 제거한다.",
)
parser.add_argument(
    "--eval_bowl_friction",
    type=float,
    nargs=2,
    metavar=("STATIC", "DYNAMIC"),
    default=None,
    help="평가 A/B: bowl physics material의 static/dynamic friction을 런타임 override한다.",
)
parser.add_argument(
    "--eval_bowl_mass",
    type=float,
    default=0.0,
    help="평가 A/B: 0이면 USD 기본값, 양수면 bowl mass(kg)를 world reset 전에 override한다.",
)
parser.add_argument("--dump_obs", default="",
                    help="진단: 지정 시 eval ep0 에서 bridge 렌더 3캠 프레임 + arm joint 를 이 디렉터리에 저장"
                         "(학습 recorder 프레임과 시각 비교용).")
parser.add_argument(
    "--vla_action_parity",
    action="store_true",
    help="VLA 모델이 이미 slew-limited target을 출력하므로 물리 joint velocity limit은 "
         "학습 env actuator와 같은 10rad/s headroom을 사용. PATH E 기본값은 기존 근사 유지.",
)
parser.add_argument(
    "--vla_reset_file",
    default="",
    help="scene reset generation을 기록할 파일. vla_policy_node가 같은 공유 파일을 읽어 "
         "episode 경계에서 stale action queue/timestep을 초기화한다.",
)
# ── grasp sweep 검증 모드 ────────────────────────────────────────────────────
# pink IK top-down sweep(pink_ik_bridge_node.py --sweep --gen-traj)이 뽑은 궤적 JSON 을
# world 프레임서 큐브를 셀마다 teleport→replay 하며 물리로 잡히는지 검증. 잡은 시점(lift 끝)에
# perspective/top/wrist/front 4뷰를 2x2 로 캡처, 파일명=큐브 world 좌표.
parser.add_argument("--grasp_sweep", default="",
                    help="gen-traj JSON 경로. 지정 시 sweep replay+capture 모드(eval/loop 대신).")
parser.add_argument("--grasp_sweep_out", default="outputs/grasp_sweep",
                    help="2x2 캡처 PNG 출력 디렉터리(REPO_ROOT 상대).")
parser.add_argument("--grasp_settle", type=int, default=20,
                    help="셀 teleport 후 replay 전 settle step(큐브 안정).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ⚠ 기본 headless experience(isaaclab.python.headless.kit)는 OmniGraph USD 그래프 생성을
# strip 해 "Unable to create prim for graph" 로 실패한다. enable_cameras=True 면 AppLauncher 가
# isaaclab.python.headless.rendering.kit(풀 렌더 + OmniGraph USD authoring)를 로드해 OmniGraph 가
# 동작한다. 카메라 자체는 쓰지 않는다 — 렌더 experience 만 필요.
args.enable_cameras = True

# ⚠ AppLauncher 에는 **화이트리스트 키만** 전달한다(AGENTS.md). vars(args) 통째로 넘기면 커스텀
# 인자(--view_eye/--layout/--eval/--num_cubes/…)가 AppLauncher 의 UI/viewport 초기화(_prepare_ui)를
# 깨뜨려 livestream 에서 카메라 viewport docking 이 적용되지 않는다(데모 pick_cube_curobo_demo 와
# 동일 패턴으로 정합). C-레벨 크래시 추적용 faulthandler 도 부팅 전에 켠다.
import faulthandler  # noqa: E402
os.makedirs(os.path.join(REPO_ROOT, "outputs"), exist_ok=True)
faulthandler.enable(open(os.path.join(REPO_ROOT, "outputs/bridge_faulthandler.txt"), "w"))
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(_launcher_args)
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
if not args.no_cameras:
    # 카메라 render product 생성에 replicator 필요.
    enable_extension("omni.replicator.core")

import omni.graph.core as og  # noqa: E402
import torch  # noqa: E402  (world→opengl quat 변환용)
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402 (광원 spawn — curobo demo PickCubeSceneCfg 와 동일 cfg)
import isaaclab.sim.schemas as schemas  # noqa: E402 (순수 USD authoring — 시뮬 파이프라인 무관)
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

from so101_contract.leader_calibration import SO101_FOLLOWER_USD_JOINT_LIMITS  # noqa: E402
from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH, ROBOT_USD_PATH  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
    _BOWL_INIT_STATE,
    _CUBE_ARM_EXCLUDE,
    _CUBE_INIT_STATES,
    _CUBE_SCATTER_BELL,
    _CUBE_SCATTER_X_RANGE,
    _CUBE_SCATTER_Y_RANGE,
    _FRONT_CAM_LOCAL_POS,
    _FRONT_CAM_LOCAL_ROT,
    _FRONT_CAMERA_FOCAL,
    _ROBOT_POS,
    _ROBOT_ROT,
    _TOP_CAMERA_FOCAL,
    _TOP_CAMERA_POS,
    _TOP_CAMERA_ROT,
    _WRIST_CAM_LOCAL_POS,
    _WRIST_CAM_LOCAL_ROT,
    _WRIST_CAMERA_FOCAL,
)
# cube recorder/state machine이 실제 cube_desk 상판과 맞춰 사용하는 z 기준.
# common MDP의 0.760은 pen desk 기준 stale 값이라 bowl 안 cube(z≈0.73~0.76)를 실패 처리한다.
CUBE_DESK_TOP_Z = 0.705
from sim_to_real.utils.constant import (  # noqa: E402
    BOWL_NAME,
    CUBE_NAMES,
)

# 순수 isaacsim 경로의 stage prim 레이아웃 (env 네임스페이스 없음).
SCENE_PRIM = "/World/Scene"
ROBOT_PRIM = "/World/Robot"  # fix_root_link 후 articulation root 가 이 prim 으로 올라온다.
BASE_LINK_PRIM = f"{ROBOT_PRIM}/base/base_link"  # TF parent frame "base_link" 용 Xform.
JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"

# VLA obs 카메라 — gym PickCube 와 동일 prim 경로/포즈/focal(teleop 튜너 보정값 재사용).
# top=world 고정, wrist=gripper 링크 자식, front=shoulder 링크 자식.
# pick_cube_env_cfg 의 _pinhole_camera_cfg 와 동일: horizontal_aperture=20.955, 640×480.
_CAM_W, _CAM_H = 640, 480
_CAM_HORIZ_APERTURE = 20.955
# (prim_path, parent_local?, pos, rot_world_wxyz, focal, topic, frame_id)
CAMERA_SPECS = [
    ("/World/TopCamera",
     _TOP_CAMERA_POS, _TOP_CAMERA_ROT, _TOP_CAMERA_FOCAL,
     "/camera/top/image_raw", "top_camera_optical_frame"),
    (f"{ROBOT_PRIM}/gripper/WristCamera",
     _WRIST_CAM_LOCAL_POS, _WRIST_CAM_LOCAL_ROT, _WRIST_CAMERA_FOCAL,
     "/camera/wrist/image_raw", "wrist_camera_optical_frame"),
    (f"{ROBOT_PRIM}/shoulder/FrontCamera",
     _FRONT_CAM_LOCAL_POS, _FRONT_CAM_LOCAL_ROT, _FRONT_CAMERA_FOCAL,
     "/camera/front/image_raw", "front_camera_optical_frame"),
]

# ── 물리 parity: PickCubeEnvCfg(=VLA 학습 env) 와 동일 actuator/물리 ──────────────
# cuRobo demo/batch 가 데이터를 이 물리로 생성했으므로 VLA closed-loop 추론도 동일하게 맞춘다
# (sim2sim parity). 이전 GRIPPER_STIFFNESS=80 은 PATH E cuMotion position-control 용이었으나
# VLA parity 와 충돌 → 학습 env 와 동일한 soft PD 17.8 + gentle effort 로 통일. cuMotion(PATH E)
# grasp 거동 변화는 사용자 결정에 따라 감수(둘 다 이 물리 공유).
DRIVE_STIFFNESS = 17.8        # arm·gripper 공통 (PickCubeEnvCfg actuator stiffness)
DRIVE_DAMPING = 0.6           # PickCubeEnvCfg actuator damping
ARM_EFFORT_LIMIT = 10.0       # PickCubeEnvCfg arm/gripper effort_limit_sim (정적 cap)
# 그리퍼 effort = leisaac dynamic_reset_gripper_effort_limit: clamp(nearest_mass/0.15, 0.5, 10).
# 우리 큐브(35·55g): 0.035~0.055/0.15 = 0.23~0.37 < 0.5 → 전부 하한 0.5Nm 으로 클램프되므로
# static 0.5 가 동치(>0.5 는 그릇 0.25kg 근접 시뿐, 그땐 grip 불요 → grasp 거동 동일). gentle =
# 가벼운 큐브 안 으깸·sim2real.
GRIPPER_EFFORT_LIMIT = 0.5
# SlewLimitedJointPositionAction(gripper 2.5·arm 5.0 rad/s) 근사 = 물리 joint 최대속도 상한.
# bridge 는 OmniGraph 로 raw position 을 주입해 Python slew 불가 → 학습 데이터의 모션 envelope
# (빠른 close = 큐브 튕김 방지)을 joint max velocity 로 강제한다.
ARM_MAX_JOINT_VEL = 5.0
GRIPPER_MAX_JOINT_VEL = 2.5



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


def _zero_velocity(handle) -> None:
    """rigid prim 의 선/각속도를 0 으로 (텔레포트 직후 잔류 속도 제거). 핸들이 미지원이면 무시."""
    try:
        handle.set_linear_velocity(np.zeros(3, dtype=np.float32))
        handle.set_angular_velocity(np.zeros(3, dtype=np.float32))
    except Exception:  # noqa: BLE001
        pass


def _repo_path(rel: str) -> str:
    """REPO_ROOT 상대경로를 절대경로로 변환(이미 절대경로면 그대로)."""
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def _load_env_cfg(task_id: str):
    """등록 gym env id → env_cfg 인스턴스(필드 읽기용 — actuator·DR·성공 단일소스).

    ``gym.make`` 는 하지 않는다: ManagerBasedRLEnv 의 GPU fabric view 와 ROS OmniGraph 노드가
    ``device -1`` 로 충돌하는 구조적 문제(2026-06-09 B안 우회) 때문에 bridge 는 pure World 를
    유지하고, env_cfg 는 **필드만** 읽는다. ``env_cfg_entry_point``("module:Class")를 import 해
    인스턴스화한다. ``import sim_to_real`` 로 gym.register 부작용을 먼저 일으킨다.
    """
    import importlib
    import gymnasium as gym
    import sim_to_real  # noqa: F401  (gym.register 부작용)

    spec = gym.spec(task_id)
    entry = spec.kwargs["env_cfg_entry_point"]
    mod_name, cls_name = entry.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)()


def _apply_joint_limits(stage, root_path: str) -> None:
    """root_path 하위 RevoluteJoint 들의 physics:lower/upperLimit 를 leader_calibration
    단일 소스(SO101_FOLLOWER_USD_JOINT_LIMITS)로 **live stage 에 직접 set**한다.

    USD 파일 값이 stale 캐시·참조 해석으로 articulation parse 에 반영 안 되는 경우가 있어
    (관측: elbow 가 파일은 100° 인데 sim 은 90° 로 clamp), reset 전 in-memory prim 에
    직접 써 PhysX 가 파싱하는 값을 보장한다. name→limit 매핑이라 prim 경로 무관."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[bridge] joint limit: root prim 없음 {root_path}", flush=True)
        return
    applied = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        lim = SO101_FOLLOWER_USD_JOINT_LIMITS.get(prim.GetName())
        if lim is None:
            continue
        lo, hi = lim
        UsdPhysics.RevoluteJoint(prim).CreateLowerLimitAttr().Set(float(lo))
        UsdPhysics.RevoluteJoint(prim).CreateUpperLimitAttr().Set(float(hi))
        applied += 1
    print(f"[bridge] joint limit: {applied} joint 을 leader_calibration 테이블로 set "
          f"(elbow={SO101_FOLLOWER_USD_JOINT_LIMITS['elbow_flex']}, "
          f"wrist_flex={SO101_FOLLOWER_USD_JOINT_LIMITS['wrist_flex']})", flush=True)


def _create_camera_prim(stage, prim_path, pos, rot_world_wxyz, focal):
    """USD Camera prim 생성 + 포즈(world→opengl 변환)·focal·aperture 설정.

    gym TiledCamera 의 offset(convention="world") 와 동일 view 를 내려면 world-convention
    quat 을 USD(opengl) 컨벤션으로 변환해 prim local orient 로 author 한다. parent 가 link
    (wrist/front)면 그 link frame 기준 local, top 은 /World(=world) 기준.
    """
    cam = UsdGeom.Camera.Define(stage, prim_path)
    cam.GetFocalLengthAttr().Set(float(focal))
    cam.GetHorizontalApertureAttr().Set(float(_CAM_HORIZ_APERTURE))
    cam.GetVerticalApertureAttr().Set(float(_CAM_HORIZ_APERTURE) * _CAM_H / _CAM_W)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 6.0))

    q = torch.tensor([[float(c) for c in rot_world_wxyz]], dtype=torch.float32)
    w, x, y, z = convert_camera_frame_orientation_convention(q, origin="world", target="opengl")[0].tolist()

    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))


def setup_cameras(stage) -> list[tuple[str, str, str]]:
    """CAMERA_SPECS 의 카메라 prim + render product 를 만들고 (rp_path, topic, frame_id) 리스트 반환.

    parent link prim 이 없으면(이름 불일치) 경고하고 그 카메라를 건너뛴다.
    """
    import omni.replicator.core as rep

    specs: list[tuple[str, str, str]] = []
    for prim_path, pos, rot, focal, topic, frame_id in CAMERA_SPECS:
        parent_path = prim_path.rsplit("/", 1)[0]
        if not stage.GetPrimAtPath(parent_path).IsValid():
            print(f"[bridge] WARN: camera parent prim 없음 {parent_path} → {topic} 건너뜀", flush=True)
            continue
        _create_camera_prim(stage, prim_path, pos, rot, focal)
        rp = rep.create.render_product(prim_path, (_CAM_W, _CAM_H))
        specs.append((rp.path, topic, frame_id))
        print(f"[bridge] camera {prim_path} → {topic} (focal={focal})", flush=True)
    return specs


# GUI 뷰포트 docking 대상 (window title, 카메라 prim 경로). title 은 demo 와 동일하게
# "SO101 {…}" 로 만들어 assets/layouts/pick_cube_3cam.json 의 저장된 window 와 일치시킨다
# (레이아웃 복원은 title 매칭이므로 카메라 prim 경로 차이는 무관).
_BRIDGE_CAM_VIEWS = [
    ("Top Camera", "/World/TopCamera"),
    ("Wrist Camera", f"{ROBOT_PRIM}/gripper/WristCamera"),
    ("Front Camera", f"{ROBOT_PRIM}/shoulder/FrontCamera"),
]


def dock_camera_viewports() -> None:
    """top/wrist/front 카메라 viewport 3개 생성 + Perspective 와 수직 분할 docking.

    pick_cube_curobo_demo.dock_camera_viewports() 와 동일 패턴: 저장된 layout JSON
    (ui.Workspace dump)을 복원해 데모와 같은 4-패널 배치(L=Perspective, R=top/wrist/front)를
    만든다. window title 이 "SO101 Top/Wrist/Front Camera" 로 일치하므로 위치·크기 그대로
    복원되고, 실패 시 수동 dock_in fallback. GUI 모드(not headless)에서만 호출한다.
    """
    try:
        import omni.kit.app
        import omni.ui as ui
        from pxr import Sdf
        em = omni.kit.app.get_app().get_extension_manager()
        for e in ("omni.kit.viewport.window", "omni.kit.viewport.utility"):
            try:
                if not em.is_extension_enabled(e):
                    em.set_extension_enabled_immediate(e, True)
            except Exception:
                pass
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] viewport 모듈 불가: {exc}", flush=True)
        return

    created = {}
    for title, path in _BRIDGE_CAM_VIEWS:
        try:
            created[title] = create_viewport_window(name=f"SO101 {title}", camera_path=Sdf.Path(path))
            print(f"[bridge] viewport {title}: {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] viewport {title} 실패: {exc}", flush=True)

    app = omni.kit.app.get_app()
    for _ in range(3):
        app.update()

    layout_path = _repo_path(args.layout)
    if os.path.isfile(layout_path):
        try:
            with open(layout_path) as fh:
                dump = json.load(fh)
            ui.Workspace.restore_workspace(dump)
            for _ in range(3):
                app.update()
            print(f"[bridge] layout 복원: {layout_path}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] layout 복원 실패({exc}) → 수동 dock fallback", flush=True)
    else:
        print(f"[bridge] layout 파일 없음({layout_path}) → 수동 dock fallback", flush=True)

    try:
        main_vp = ui.Workspace.get_window("Viewport")
        t = created.get("Top Camera")
        w = created.get("Wrist Camera")
        f = created.get("Front Camera")
        if main_vp is not None and t is not None:
            t.dock_in(main_vp, ui.DockPosition.RIGHT, 0.5)
        if t is not None and w is not None:
            w.dock_in(t, ui.DockPosition.BOTTOM, 0.5)
        if w is not None and f is not None:
            f.dock_in(w, ui.DockPosition.BOTTOM, 0.5)
        for _ in range(3):
            app.update()
        print("[bridge] 카메라 viewport docked (L=Perspective, R=top/wrist/front)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] docking 실패: {exc}", flush=True)


def build_ros_graph(object_prims: list[str], camera_specs: list[tuple[str, str, str]] | None = None):
    """JointState pub/sub + Clock + 물체 TF OmniGraph (ROS 2 bridge, C++ only — rclpy 불필요).

    World 가 timeline 을 play 하므로 OnPlaybackTick 이 매 프레임 fire 한다(A안의 OnTick+
    수동 evaluate_sync 불필요). 물체 TF 는 parent=base_link Xform 기준으로 publish 한다.
    """
    create_nodes = [
        ("OnTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ("PublishObjectTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
    ]
    connect = [
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
        ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
    ]
    set_values = [
        ("PublishJointState.inputs:topicName", JOINT_STATES_TOPIC),
        ("PublishJointState.inputs:targetPrim", [ROBOT_PRIM]),
        ("SubscribeJointState.inputs:topicName", JOINT_COMMANDS_TOPIC),
        ("ArticulationController.inputs:targetPrim", [ROBOT_PRIM]),
        ("PublishObjectTF.inputs:parentPrim", [BASE_LINK_PRIM]),
        ("PublishObjectTF.inputs:targetPrims", object_prims),
    ]

    # 카메라 publish 노드 — render product 당 ROS2CameraHelper(type=rgb) 하나.
    for i, (rp_path, topic, frame_id) in enumerate(camera_specs or []):
        node = f"CameraHelper{i}"
        create_nodes.append((node, "isaacsim.ros2.bridge.ROS2CameraHelper"))
        connect += [
            ("OnTick.outputs:tick", f"{node}.inputs:execIn"),
            ("Context.outputs:context", f"{node}.inputs:context"),
        ]
        set_values += [
            (f"{node}.inputs:renderProductPath", rp_path),
            (f"{node}.inputs:topicName", topic),
            (f"{node}.inputs:frameId", frame_id),
            (f"{node}.inputs:type", "rgb"),
        ]

    graph, _, _, _ = og.Controller.edit(
        {"graph_path": "/ROSBridge", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connect,
            og.Controller.Keys.SET_VALUES: set_values,
        },
    )
    return graph


def main() -> None:
    np.random.seed(args.seed)

    # 실행할 env 를 --task 로 골라 그 cfg 를 로드(actuator·DR 모드 단일소스). gym.make 안 함.
    env_cfg = _load_env_cfg(args.task)
    if "-DR" in args.task and not args.dr:
        args.dr = True   # DR 변형 task → 큐브/그릇 무작위화 자동 on(명시 --dr 도 유지)
        print(f"[bridge] task={args.task} → DR on (env_cfg DR 변형)", flush=True)
    else:
        print(f"[bridge] task={args.task} · DR={'on' if args.dr else 'off(고정배치)'}", flush=True)

    active_cubes = [args.cube_name] if args.cube_name else CUBE_NAMES[: args.num_cubes]

    # World — 순수 isaacsim.core. backend="numpy"(CPU) → OmniGraph 물리노드가 simulation
    # view 를 단독 소유해 device 정합(A안 device -1 회피). cuMotion 제어엔 단일 로봇 +
    # 소수 큐브라 CPU 물리로 충분하다.
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # PhysX solver parity (PickCubeEnvCfg.__post_init__): 접촉/마찰 안정성을 학습 env 와 맞춘다.
    # (gpu_* 버퍼는 GPU multi-env scale 전용 → CPU single-env 무관. enable_external_forces_every_
    #  iteration 은 외력 미사용이라 무관 → 생략.) backend 는 CPU 유지(A안 device-1 회피, 구조적).
    try:
        pc = world.get_physics_context()
        pc.set_solver_type("TGS")
        pc.set_bounce_threshold(0.01)
        pc.set_friction_correlation_distance(0.00625)
        print("[bridge] PhysX parity: TGS · bounce 0.01 · friction_corr 0.00625", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] PhysX 설정 실패: {exc}", flush=True)

    # cube_desk scene.usd: SCENE_OFFSET 가 baked 돼 큐브/그릇이 이미 world 좌표에 author 됨.
    add_reference_to_stage(CUBE_DESK_USD_PATH, SCENE_PRIM)
    if args.eval_bowl_kinematic:
        bowl_prim = world.stage.GetPrimAtPath(f"{SCENE_PRIM}/{BOWL_NAME}")
        bowl_rb = UsdPhysics.RigidBodyAPI.Apply(bowl_prim)
        bowl_rb.CreateRigidBodyEnabledAttr().Set(True)
        bowl_rb.CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(bowl_prim).CreateEnableCCDAttr().Set(False)
        print("[bridge] eval A/B: bowl kinematic 고정", flush=True)
    if args.eval_bowl_mass > 0.0:
        bowl_prim = world.stage.GetPrimAtPath(f"{SCENE_PRIM}/{BOWL_NAME}")
        UsdPhysics.MassAPI.Apply(bowl_prim).CreateMassAttr().Set(float(args.eval_bowl_mass))
        print(f"[bridge] eval A/B: bowl mass={args.eval_bowl_mass}kg", flush=True)
    if args.eval_bowl_friction is not None:
        static_friction, dynamic_friction = args.eval_bowl_friction
        bowl_material_prim = world.stage.GetPrimAtPath(
            f"{SCENE_PRIM}/{BOWL_NAME}/Looks/BowlFriction"
        )
        bowl_material = UsdPhysics.MaterialAPI.Apply(bowl_material_prim)
        bowl_material.CreateStaticFrictionAttr().Set(float(static_friction))
        bowl_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        print(
            f"[bridge] eval A/B: bowl friction static={static_friction} "
            f"dynamic={dynamic_friction}",
            flush=True,
        )

    # 조명 parity: scene.usd 는 광원 prim 이 없어(PickCubeSceneCfg 가 /World 계층서 따로 author)
    # bridge 만 디폴트 헤드라이트로 렌더돼 curobo demo 와 노출/색이 달랐다. curobo demo(=VLA 학습
    # env)의 PickCubeSceneCfg 와 **동일 cfg**(DomeLight 2000 + KeyLight distant 1800)를 같은
    # prim 경로(/World/Light·/World/KeyLight)에 spawn 해 조명을 정확히 맞춘다.
    _dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    _dome.func("/World/Light", _dome)
    _key = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.98, 0.95), angle=1.0)
    # RotateXYZ(-50,0,-35)° 등가 quat(wxyz) — PickCubeSceneCfg.key_light init_state 와 동일.
    _key.func("/World/KeyLight", _key, orientation=(0.8644, -0.4031, -0.1271, -0.2725))
    print("[bridge] 조명 parity: DomeLight 2000 + KeyLight distant 1800 (curobo demo 동일)", flush=True)

    # SO-101 follower: defaultPrim(/so101_new_calib) 이 ROBOT_PRIM 으로 composes-in.
    robot_prim = add_reference_to_stage(ROBOT_USD_PATH, ROBOT_PRIM)
    # 로봇을 데스크 앞에 배치(PATH C 와 동일 pose). fixed joint 가 이 pose 에서 anchor 되도록
    # base 고정 전에 먼저 적용한다.
    _set_local_pose(robot_prim, _ROBOT_POS, _ROBOT_ROT)
    # joint limit 을 live stage 에 직접 set(USD 파일값이 캐시로 parse 에 반영 안 되는 케이스 보장).
    _apply_joint_limits(world.stage, ROBOT_PRIM)

    # base 고정. find/create fixed joint(world→base) + ArticulationRootAPI 를 부모(ROBOT_PRIM)로
    # 이동(PhysX parser 한계 회피). 이후 articulation root = ROBOT_PRIM.
    schemas.modify_articulation_root_properties(
        f"{ROBOT_PRIM}/base",
        ArticulationRootPropertiesCfg(
            fix_root_link=True,
            # self-collision on(기본). 과거 codec replay 에선 elbow 고굴곡(~99°)서 forearm+wrist
            # 캠홀더 convex 가 일찍 막았으나, follower calibration 이 elbow target 을 낮춰
            # plain bridge(self-collision ON)에서 grasp 정상 확인됨(2026-06-30).
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

    # VLA obs 카메라 — prim + render product 생성(robot/scene prim 이 stage 에 있은 뒤).
    camera_specs: list[tuple[str, str, str]] = []
    if not args.no_cameras:
        camera_specs = setup_cameras(world.stage)

    # OmniGraph 는 reset(=play) 전 timeline 정지 상태에서 생성한다(그래프 wrap + OnPlaybackTick
    # 등록). 노드는 prim 경로만 참조하므로 prim 이 존재하면 된다.
    # grasp_sweep(검증 batch)은 ROS 무경유 — 직접 ctrl.apply_action + annotator 캡처라 ROS graph
    # 를 만들면 IsaacArticulationController(빈 /isaac_joint_commands)가 apply_action 과 경합한다 → 스킵.
    ros_graph = None if args.grasp_sweep else build_ros_graph(object_prims, camera_specs)

    # reset: 물리 뷰 초기화 + timeline play → OnPlaybackTick 시작.
    world.reset()

    # ── actuator parity — env_cfg(SO101_FOLLOWER_CFG) per-joint gains 를 dof array 로 ──
    # USD drive gain 은 micro 라 위치 명령 추종 불가 → 학습 env 와 동일 PD 로 덮어쓴다.
    # 하드코딩 대신 env_cfg.scene.robot.actuators(Workshop per-joint 튜닝, lerobot.py 단일소스)를
    # joint 이름→dof 로 펼쳐 sim2sim parity 를 자동 보장한다(actuator 값 바뀌면 bridge 도 자동 추종).
    n_dof = robot.num_dof
    dof_names = list(robot.dof_names)
    name2dof = {n: i for i, n in enumerate(dof_names)}
    kps = np.zeros(n_dof, dtype=np.float32)
    kds = np.zeros(n_dof, dtype=np.float32)
    cfg_efforts = np.zeros(n_dof, dtype=np.float32)
    for act in env_cfg.scene.robot.actuators.values():
        for jn in act.joint_names_expr:
            di = name2dof.get(jn)
            if di is None:
                continue
            kps[di] = float(act.stiffness)
            kds[di] = float(act.damping)
            cfg_efforts[di] = float(act.effort_limit_sim)
    # 미매핑 dof 안전망(정상 경로엔 없음) — 하드코딩 fallback.
    kps[kps == 0.0] = DRIVE_STIFFNESS
    kds[kds == 0.0] = DRIVE_DAMPING
    cfg_efforts[cfg_efforts == 0.0] = ARM_EFFORT_LIMIT
    try:
        gi = robot.get_dof_index("gripper")
    except (ValueError, AttributeError, TypeError, KeyError):
        gi = n_dof - 1  # fallback: gripper 가 마지막 dof (/isaac_joint_states 순서 확인됨)
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kps=kps, kds=kds)
    print(f"[bridge] dof_names order: {dof_names}", flush=True)
    print(f"[bridge] actuator parity(env_cfg per-joint): kps={[round(float(x), 1) for x in kps]} "
          f"kds={[round(float(x), 2) for x in kds]}, gripper dof[{gi}]", flush=True)

    # 실제 articulation joint position limit 확인(USD/캐시 vs 적용값). _apply_joint_limits 가
    # 먹었으면 elbow=±100·wrist_flex/-95~105·lift±105 로 나와야. 90°로 나오면 limit 미적용.
    try:
        lims = np.asarray(robot._articulation_view.get_dof_limits()).reshape(-1, 2)
        names = list(robot.dof_names)
        deg = 180.0 / np.pi
        pairs = ", ".join(f"{names[i]}[{lims[i, 0] * deg:.0f},{lims[i, 1] * deg:.0f}]"
                          for i in range(min(len(names), lims.shape[0])))
        print(f"[bridge] 실제 dof pos limit(deg): {pairs}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] dof limit 조회 실패: {exc}", flush=True)

    # effort 상한 — env_cfg per-joint(Workshop arm 30N) 을 cfg 에서, gripper 만 dynamic clamp 동치로 override.
    efforts = cfg_efforts.copy()
    efforts[gi] = GRIPPER_EFFORT_LIMIT   # leisaac dynamic_reset_gripper_effort_limit(≤10, 큐브 gentle) 동치
    try:
        ctrl.set_max_efforts(values=efforts)
        print(f"[bridge] effort 상한(env_cfg): arm={[round(float(x), 1) for x in efforts]}Nm "
              f"· gripper override={GRIPPER_EFFORT_LIMIT}Nm", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] set_max_efforts 실패: {exc}", flush=True)

    # recorder: target 자체를 arm 5.0/gripper 2.5 rad/s로 slew하고, actuator 물리 상한은
    # 10rad/s로 둬 tracking headroom을 확보한다. VLA는 그 processed target을 학습했으므로
    # --vla_action_parity에서는 물리 상한을 10으로 맞춘다. PATH E는 기존 근사를 보존한다.
    if args.vla_action_parity:
        max_vel = np.full(n_dof, 10.0, dtype=np.float32)
        max_vel_label = "10.0 (VLA recorder actuator parity)"
    else:
        max_vel = np.full(n_dof, ARM_MAX_JOINT_VEL, dtype=np.float32)
        max_vel[gi] = GRIPPER_MAX_JOINT_VEL
        max_vel_label = f"arm={ARM_MAX_JOINT_VEL} · gripper={GRIPPER_MAX_JOINT_VEL}"
    try:
        robot._articulation_view.set_max_joint_velocities(np.array([max_vel], dtype=np.float32))
        print(f"[bridge] joint max vel: {max_vel_label} rad/s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] set_max_joint_velocities 실패: {exc}", flush=True)

    # 큐브 rigid prim 핸들 캐시 (DR·R/N·eval 리셋에서 재사용).
    cube_handles: dict[str, SingleRigidPrim] = {}
    for name in active_cubes:
        h = SingleRigidPrim(f"{SCENE_PRIM}/{name}")
        h.initialize()
        cube_handles[name] = h
    # 그릇 핸들 (eval cube-in-bowl 판정의 컨테이너 xy 기준).
    bowl_handle = SingleRigidPrim(f"{SCENE_PRIM}/{BOWL_NAME}")
    try:
        bowl_handle.initialize()
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] bowl handle init 실패: {exc}", flush=True)
    # 초기/리셋 팔 자세 = teleop_se3_agent 시작 자세(PickCubeEnvCfg robot.init_state.joint_pos).
    # 사용자 지정(2026-06-26, deg→rad): pan 0·lift -100·elbow 90(요청 +100°, USD 상한 90° 캡)·
    # wrist_flex 70·wrist_roll -100·gripper 0. name→rad 매핑이라 dof 순서 무관.
    _START_POSE_RAD = {
        "shoulder_pan": np.radians(0.0), "shoulder_lift": np.radians(-100.0),
        "elbow_flex": np.radians(90.0), "wrist_flex": np.radians(70.0),
        "wrist_roll": np.radians(-100.0), "gripper": 0.0,
    }
    home_q = np.array([_START_POSE_RAD.get(n, 0.0) for n in robot.dof_names], dtype=np.float32)
    print(f"[bridge] 초기 팔 자세(teleop init_state 정합): {[round(float(x), 3) for x in home_q]}", flush=True)

    # 비활성 큐브(active 아님)를 z=-1 로 park — 단일-큐브 학습 데이터 정합(카메라 밖).
    # 기존 bridge 는 active 만 핸들링해 비활성 큐브가 authored default 위치에 노출됐다(버그).
    inactive_handles: dict[str, SingleRigidPrim] = {}
    for name in CUBE_NAMES:
        if name in cube_handles:
            continue
        try:
            h = SingleRigidPrim(f"{SCENE_PRIM}/{name}")
            h.initialize()
            inactive_handles[name] = h
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] inactive cube {name} handle 실패: {exc}", flush=True)

    def park_inactive() -> None:
        for h in inactive_handles.values():
            try:
                h.set_world_pose(position=np.array([0.0, 0.0, -1.0], dtype=np.float32))
            except Exception:  # noqa: BLE001
                pass
            _zero_velocity(h)

    # ── DR 배치 파라미터 = env_cfg.events.randomize_cubes 단일소스에서 읽기 ──
    # bridge numpy 재현이 env_cfg 의 randomize_cubes EventTerm params 를 따라가 --task 의
    # full(종모양 bell)/base(사각형) 모드를 그대로 반영한다(하드코딩 아님). v0/Eval(고정배치)
    # 이나 --dr 수동+비-DR task 면 randomize_cubes 없음 → fallback(_CUBE_SCATTER 종모양).
    _rc = getattr(env_cfg.events, "randomize_cubes", None)
    _rcp = dict(getattr(_rc, "params", {}) or {}) if _rc is not None else {}
    _DR_X_RANGE = tuple(_rcp.get("x_range", _CUBE_SCATTER_X_RANGE))
    _DR_Y_RANGE = tuple(_rcp.get("y_range", _CUBE_SCATTER_Y_RANGE))
    _DR_BELL = _rcp.get("x_halfwidth_by_y", _CUBE_SCATTER_BELL)   # base 모드=None(사각형)
    _DR_EXCLUDE = _rcp.get("x_exclude_box", _CUBE_ARM_EXCLUDE)
    _MIN_CUBE_SEP = float(_rcp.get("min_cube_sep", 0.060))        # 큐브 볼륨 비겹침
    _MIN_BOWL_SEP = float(_rcp.get("min_bowl_sep", 0.14))         # 큐브-그릇
    _MIN_BASE_SEP = float(_rcp.get("min_base_sep", 0.135))        # 큐브-base 발치(inner-reach)
    _VOLUME_INSET = float(_rcp.get("volume_inset", 0.0))
    print(f"[bridge] DR 배치: x={_DR_X_RANGE} y={_DR_Y_RANGE} "
          f"mode={'bell(full)' if _DR_BELL else 'rect(base)'}", flush=True)

    def _bell_hw(y: float) -> float:
        """종 모양 x 반너비 |x|<=w(y) piecewise-linear (env_cfg params x_halfwidth_by_y).
        base 모드(_DR_BELL=None)면 inf → 종모양 제약 없음(사각형 스폰)."""
        if _DR_BELL is None:
            return float("inf")
        bp = sorted(_DR_BELL)
        if y <= bp[0][0]:
            return bp[0][1]
        for (y0, w0), (y1, w1) in zip(bp, bp[1:]):
            if y <= y1:
                return w0 + (w1 - w0) * (y - y0) / (y1 - y0)
        return bp[-1][1]
    _BOWL_ARC_RADIUS = 0.44
    _BOWL_ARC_DEG = (-4.0, 8.0)
    _MAX_ATTEMPTS = 50
    # 6 stable face (roll,pitch) — 학습 _randomize_cubes_scattered_fn 와 동일
    _STABLE_FACES = [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, 0.0),
                     (-np.pi / 2, 0.0), (0.0, np.pi / 2), (0.0, -np.pi / 2)]

    # authored default pose 캡처(DR 전 1회) — arc center·cube z 기준점.
    _bp, _bq = bowl_handle.get_world_pose()
    bowl_default_xy = np.asarray(_bp[:2], dtype=np.float64)
    bowl_default_z = float(_bp[2])
    bowl_default_quat = np.asarray(_bq, dtype=np.float32)
    cube_default_z = {n: float(h.get_world_pose()[0][2]) for n, h in cube_handles.items()}
    base_xy = np.array([_ROBOT_POS[0], _ROBOT_POS[1]], dtype=np.float64)

    def _face_quat(rng) -> np.ndarray:
        """6 stable face 중 1택 + random yaw → wxyz (Isaac Lab quat_from_euler_xyz 동일 공식)."""
        roll, pitch = _STABLE_FACES[int(rng.integers(0, 6))]
        yaw = float(rng.random()) * 2.0 * np.pi
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        return np.array([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                         cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp], dtype=np.float32)

    def randomize_cubes(rng) -> None:
        """학습 _randomize_cubes_scattered_fn 정합: rejection(inset rect·bowl_default·base·placed 회피)
        + 6 stable-face + random yaw. 그릇 기준=bowl_default(학습 동일, arc 이동 전)."""
        x_lo = _DR_X_RANGE[0] + _VOLUME_INSET
        x_hi = _DR_X_RANGE[1] - _VOLUME_INSET
        y_lo = _DR_Y_RANGE[0] + _VOLUME_INSET
        y_hi = _DR_Y_RANGE[1] - _VOLUME_INSET
        placed: list[tuple[float, float]] = []
        for name, h in cube_handles.items():
            fx, fy = float(bowl_default_xy[0]), float(bowl_default_xy[1])  # fallback(드묾)
            for _ in range(_MAX_ATTEMPTS):
                cx = float(rng.uniform(x_lo, x_hi))
                cy = float(rng.uniform(y_lo, y_hi))
                if abs(cx) > _bell_hw(cy):   # 종 모양 밖 배제(좌우대칭 grasp 범위)
                    continue
                _ax0, _ax1, _ay0, _ay1 = _DR_EXCLUDE   # 로봇암 주변 배제(env_cfg params)
                if _ax0 <= cx <= _ax1 and _ay0 <= cy <= _ay1:
                    continue
                if (cx - bowl_default_xy[0]) ** 2 + (cy - bowl_default_xy[1]) ** 2 < _MIN_BOWL_SEP ** 2:
                    continue
                if (cx - base_xy[0]) ** 2 + (cy - base_xy[1]) ** 2 < _MIN_BASE_SEP ** 2:
                    continue
                if any((cx - px) ** 2 + (cy - py) ** 2 < _MIN_CUBE_SEP ** 2 for px, py in placed):
                    continue
                fx, fy = cx, cy
                break
            placed.append((fx, fy))
            pos = np.array([fx, fy, cube_default_z[name]], dtype=np.float32)
            h.set_world_pose(position=pos, orientation=_face_quat(rng))
            _zero_velocity(h)

    def randomize_bowl(rng) -> None:
        """학습 _randomize_object_on_arc_fn 정합: 전방 호(radius 0.44, angle[-4,8]°) 위 xy 재배치.
        center = (default_x, default_y − radius). +angle→+x. orientation 은 default 유지."""
        ang = np.radians(float(rng.uniform(_BOWL_ARC_DEG[0], _BOWL_ARC_DEG[1])))
        cx = bowl_default_xy[0]
        cy = bowl_default_xy[1] - _BOWL_ARC_RADIUS
        pos = np.array([cx + _BOWL_ARC_RADIUS * np.sin(ang),
                        cy + _BOWL_ARC_RADIUS * np.cos(ang), bowl_default_z], dtype=np.float32)
        bowl_handle.set_world_pose(position=pos, orientation=bowl_default_quat)
        if not args.eval_bowl_kinematic:
            _zero_velocity(bowl_handle)

    def place_defaults() -> None:
        """DR off: 큐브·그릇을 env_cfg 고정 기본 위치에 둔다(teleop reset_scene_to_default 와
        동일 단일 소스 = pick_cube_env_cfg._CUBE_INIT_STATES/_BOWL_INIT_STATE).
        _CUBE_INIT_STATES 에 없는 큐브(다중-큐브 씬)는 authored 위치를 유지한다."""
        for name, h in cube_handles.items():
            init = _CUBE_INIT_STATES.get(name)
            if init is None:
                continue
            pos, rot = init
            h.set_world_pose(position=np.asarray(pos, dtype=np.float32),
                             orientation=np.asarray(rot, dtype=np.float32))
            _zero_velocity(h)
        bpos, brot = _BOWL_INIT_STATE
        bowl_handle.set_world_pose(position=np.asarray(bpos, dtype=np.float32),
                                   orientation=np.asarray(brot, dtype=np.float32))
        if not args.eval_bowl_kinematic:
            _zero_velocity(bowl_handle)

    def reset_scene(seed: int) -> None:
        """씬 리셋 + 팔 home. --dr 지정 시에만 무작위화(학습 DR 정합: 큐브 scatter+6D face →
        그릇 arc), 미지정 시 env_cfg 고정 기본 위치. 동일 seed = 동일 spawn 레이아웃
        (post-settle 은 PhysX 미세변동). np 전역 무관(default_rng)."""
        if args.dr:
            rng = np.random.default_rng(int(seed))
            randomize_cubes(rng)   # 학습 순서: 큐브(vs bowl_default) → 그릇 arc
            randomize_bowl(rng)
        else:
            place_defaults()
        park_inactive()        # 비활성 큐브 카메라 밖 유지(단일-큐브 정합)
        try:
            robot.set_joint_positions(home_q)
            robot.set_joint_velocities(np.zeros(n_dof, dtype=np.float32))   # ← home_q 아님(속도=0)
            try:   # PD target 도 home_q 로 → 첫 /isaac_joint_commands 도착 전 stale target sag 방지
                robot.set_joint_position_targets(home_q)
            except (AttributeError, TypeError):
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] arm home reset 실패: {exc}", flush=True)
        if args.vla_reset_file:
            reset_path = os.path.abspath(args.vla_reset_file)
            os.makedirs(os.path.dirname(reset_path), exist_ok=True)
            reset_generation["value"] += 1
            tmp_path = f"{reset_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(f"{reset_generation['value']}\n")
            os.replace(tmp_path, reset_path)
            print(
                f"[bridge] VLA reset token={reset_generation['value']} → {reset_path}",
                flush=True,
            )
        # 진단: 동일 레이아웃 SM 재현용 — 현 spawn(world pose, wxyz)을 SM --replay_spawn 포맷 JSON 으로
        # 덤프(gated LAYOUT_DUMP). num_envs=1 SM 은 env_origin=0 이라 world==env-local.
        _ld = os.getenv("LAYOUT_DUMP", "").strip()
        if _ld:
            _lay = {"active_objects": int(args.num_cubes), "seed": int(seed),
                    "envs": [{"env": 0, "placed": 0, "clean": False, "cubes": {}, "bowl": None}]}
            for _nm, _h in cube_handles.items():
                _pp, _qq = _h.get_world_pose()
                _lay["envs"][0]["cubes"][_nm] = [float(v) for v in (*_pp[:3], *_qq[:4])]
            _bp2, _bq2 = bowl_handle.get_world_pose()
            _lay["envs"][0]["bowl"] = [float(v) for v in (*_bp2[:3], *_bq2[:4])]
            with open(_ld, "w") as _f:
                json.dump(_lay, _f, indent=2)
            print(f"[bridge] LAYOUT DUMP → {_ld} (seed={seed})", flush=True)
        mode = f"DR (seed={seed})" if args.dr else "고정 기본 위치(DR off)"
        print(f"[bridge] scene reset · {mode}", flush=True)

    # 시작 시 1회 배치(--dr 시 무작위, 미지정 시 고정 기본). 이후 R/N 으로 재리셋.
    current_seed = {"v": int(args.seed)}
    reset_generation = {"value": 0}
    reset_scene(current_seed["v"])

    # 키보드: R = 동일 seed 리셋(재현) · N = 무작위 seed 리셋. GUI 모드만.
    reset_req = {"mode": None}   # "same" | "random" | None
    kbd_sub = None
    if not args.headless:
        try:
            import carb  # noqa: PLC0415
            import omni.appwindow  # noqa: PLC0415

            _inp = carb.input.acquire_input_interface()
            _kbd = omni.appwindow.get_default_app_window().get_keyboard()

            def _on_kbd(event, *_a):
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input.name == "R":
                        reset_req["mode"] = "same"
                    elif event.input.name == "N":
                        reset_req["mode"] = "random"
                return True

            kbd_sub = _inp.subscribe_to_keyboard_events(_kbd, _on_kbd)
            print("[bridge] keyboard: R = 동일 seed 리셋(재현) · N = 무작위 seed 리셋", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] keyboard sub 실패(리셋 키 비활성): {exc}", flush=True)

    # GUI 뷰포트 — demo 와 동일 레이아웃(Perspective 초기 부감 + top/wrist/front 3-패널 dock).
    # livestream WebRTC 가 kit 창 전체를 캡처하므로 원격에서도 카메라 피드가 보인다.
    if not args.headless:
        try:
            # world eye→lookat 자동조준(up=+z). GUI 카메라 실측 eye/lookat 정합.
            # prim 로컬 xformOp(rotateXYZ) 직접 조작은 카메라 부모 Xform 때문에 world pose 와
            # 어긋나므로(로컬 xyz≠world eye) world 공간 set_camera_view 를 쓴다.
            # roll 필요 시엔 lookat 으론 불가 — 카메라 world 4x4 matrix 를 써야 함.
            from isaacsim.core.utils.viewports import set_camera_view  # noqa: PLC0415
            set_camera_view(eye=args.view_eye, target=args.view_lookat)
            print(f"[bridge] Perspective: eye={args.view_eye} lookat={args.view_lookat}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] Perspective view 설정 실패: {exc}", flush=True)
        if not args.no_cameras:
            dock_camera_viewports()

    print(f"[bridge] ready. cubes={active_cubes}  dof={n_dof}", flush=True)
    cam_topics = " / ".join(t for _, t, _ in camera_specs) if camera_specs else "(none)"
    print(f"[bridge] topics: {JOINT_STATES_TOPIC} / {JOINT_COMMANDS_TOPIC} / /clock / /tf"
          f"  (TF base_link→{active_cubes + [BOWL_NAME]})", flush=True)
    print(f"[bridge] camera topics: {cam_topics}", flush=True)

    # ── VLA 성공률 eval 모드 ─────────────────────────────────────────────────
    # recorder/state machine success-only 선별과 동일한 cube-in-bowl 판정.
    # bridge 는 env origin=0 이라 world pose == env-local pose.
    def cube_in_bowl(cube_h) -> tuple[bool, float, float]:
        cp = np.asarray(cube_h.get_world_pose()[0], dtype=np.float64)
        bp = np.asarray(bowl_handle.get_world_pose()[0], dtype=np.float64)
        dxy = float(np.hypot(cp[0] - bp[0], cp[1] - bp[1]))
        z = float(cp[2])
        in_xy = dxy < BOWL_SUCCESS_RADIUS
        in_z = (CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[0]) < z < (
            CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[1] + 0.10
        )
        return (in_xy and in_z), dxy, z

    # ── grasp sweep 검증: 궤적 JSON replay + 잡은 시점 4뷰 2x2 캡처 ────────────
    if args.grasp_sweep:
        import omni.replicator.core as _rep
        from isaacsim.core.utils.types import ArticulationAction
        from sim_to_real.tasks.common.utils import _look_at_quat_world
        try:
            import imageio.v2 as _imageio
        except Exception:  # noqa: BLE001
            _imageio = None
        out_dir = _repo_path(args.grasp_sweep_out)
        os.makedirs(out_dir, exist_ok=True)

        # perspective 카메라 + render product (top/wrist/front 는 camera_specs 에 이미 있음).
        persp_quat = _look_at_quat_world(tuple(args.view_eye), tuple(args.view_lookat))
        _create_camera_prim(world.stage, "/World/SweepPersp", tuple(args.view_eye), persp_quat, 24.0)
        persp_rp = _rep.create.render_product("/World/SweepPersp", (_CAM_W, _CAM_H))

        rp_by_view = {"perspective": persp_rp.path}
        for rp_path, topic, _fid in camera_specs:
            for key in ("top", "wrist", "front"):
                if f"/{key}/" in topic:
                    rp_by_view[key] = rp_path
        annots = {}
        for v in ("perspective", "top", "wrist", "front"):
            if v in rp_by_view:
                a = _rep.AnnotatorRegistry.get_annotator("rgb")
                a.attach(rp_by_view[v])
                annots[v] = a
        for _ in range(5):   # annotator warmup
            world.step(render=True)

        spec = json.load(open(_repo_path(args.grasp_sweep)))
        cells = spec["cells"]
        joint_order = spec["joint_order"]
        lift_delta = float(spec["lift_delta"])
        name2dof = {n: robot.get_dof_index(n) for n in joint_order}
        cube_name = active_cubes[0]
        cube_h = cube_handles[cube_name]
        settle = max(0, int(args.grasp_settle))
        _blank = np.zeros((_CAM_H, _CAM_W, 3), np.uint8)

        def _grab_2x2():
            imgs = {}
            for v, a in annots.items():
                arr = np.asarray(a.get_data())
                if arr.size == 0:
                    return None
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[..., :3]
                imgs[v] = arr.astype(np.uint8)
            top = np.concatenate([imgs.get("perspective", _blank), imgs.get("top", _blank)], axis=1)
            bot = np.concatenate([imgs.get("wrist", _blank), imgs.get("front", _blank)], axis=1)
            return np.concatenate([top, bot], axis=0)

        # sweep 테스트는 그릇 무관(사용자 요청) → 화면 밖으로 치운다. cube_in_bowl 은 항상 False 가
        # 되지만 성공 판정은 lifted(Δz) 로 하므로 무관.
        try:
            bowl_handle.set_world_pose(position=np.array([0.0, 0.0, -1.0], np.float32))
            _zero_velocity(bowl_handle)
            print("[grasp_sweep] 그릇 park(z=-1) — sweep 테스트 화면에서 제거", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[grasp_sweep] 그릇 park 실패: {_e}", flush=True)

        n_ok = 0
        print(f"[grasp_sweep] {len(cells)} 셀 replay+capture (2x2 persp/top/wrist/front) → {out_dir}", flush=True)
        for ci, cell in enumerate(cells):
            if not simulation_app.is_running():
                break
            cx, cy, cz = cell["cube_world"]
            park_inactive()
            cube_h.set_world_pose(position=np.array([cx, cy, cz], np.float32),
                                  orientation=np.array([1.0, 0.0, 0.0, 0.0], np.float32))
            _zero_velocity(cube_h)
            robot.set_joint_positions(home_q)
            robot.set_joint_velocities(np.zeros(n_dof, np.float32))
            for _ in range(settle):
                if not simulation_app.is_running():
                    break
                ctrl.apply_action(ArticulationAction(joint_positions=home_q))
                world.step(render=True)
            spawn_z = float(cube_h.get_world_pose()[0][2])

            traj = cell["traj"]
            cap_idx = int(cell["capture_idx"])
            cap_frame = None
            max_z = spawn_z
            for ti, q in enumerate(traj):
                if not simulation_app.is_running():
                    break
                tgt = np.array(home_q, np.float32)
                for jn, val in zip(joint_order, q):
                    tgt[name2dof[jn]] = float(val)
                ctrl.apply_action(ArticulationAction(joint_positions=tgt))
                world.step(render=True)
                max_z = max(max_z, float(cube_h.get_world_pose()[0][2]))
                if ti == cap_idx:
                    cap_frame = _grab_2x2()
            lifted = max_z > spawn_z + lift_delta
            in_bowl = cube_in_bowl(cube_h)[0]
            if lifted and cap_frame is not None:
                n_ok += 1
                fn = f"grasp_wx{cx:+.3f}_wy{cy:+.3f}.png"
                p = os.path.join(out_dir, fn)
                (_imageio.imwrite(p, cap_frame) if _imageio is not None
                 else np.save(p.replace(".png", ".npy"), cap_frame))
                print(f"[grasp_sweep] {ci + 1}/{len(cells)} ({cx:+.3f},{cy:+.3f}) "
                      f"✅ Δz={max_z - spawn_z:.3f} bowl={in_bowl} → {fn}", flush=True)
            else:
                print(f"[grasp_sweep] {ci + 1}/{len(cells)} ({cx:+.3f},{cy:+.3f}) "
                      f"❌ maxΔz={max_z - spawn_z:.3f}", flush=True)
        print(f"[grasp_sweep] 완료: {n_ok}/{len(cells)} grasp 성공 캡처 → {out_dir}", flush=True)
        return

    if args.eval > 0:
        control_hz = 30
        ep_steps = max(1, int(args.eval_seconds * control_hz))
        settle_steps = max(0, int(args.eval_settle * control_hz))
        n_active = len(active_cubes)
        episodes: list[dict] = []
        print(f"[bridge] 🎯 EVAL 시작: {args.eval} 에피소드 × {args.eval_seconds}s "
              f"(settle {args.eval_settle}s) · {n_active} 큐브 · success radius "
              f"{BOWL_SUCCESS_RADIUS}m · z∈[{CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[0]:.3f},"
              f"{CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[1] + 0.10:.3f}]", flush=True)
        # 진단 dump: bridge 렌더 3캠 annotator(render_product 직결 = 정책이 받는 이미지와 동일).
        _dump_annots = []
        if args.dump_obs:
            os.makedirs(args.dump_obs, exist_ok=True)
            try:
                import omni.replicator.core as _rep
                for rp_path, topic, _fid in camera_specs:
                    a = _rep.AnnotatorRegistry.get_annotator("rgb")
                    a.attach(rp_path)
                    _dump_annots.append((topic.strip("/").split("/")[-2], a))  # top/wrist/front (image_raw 충돌 회피)
                print(f"[bridge] dump_obs: {len(_dump_annots)} cam annotator → {args.dump_obs}", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[bridge] dump_obs annotator 실패: {_e}", flush=True)

        def _save_dump(tag):
            try:
                jp = np.asarray(robot.get_joint_positions()).ravel()[:6]
                print(f"[bridge] dump[{tag}] arm joints(rad)={[round(float(x), 3) for x in jp]}", flush=True)
            except Exception:
                pass
            try:
                import imageio.v2 as _imageio
            except Exception:  # noqa: BLE001
                _imageio = None
            for cname, a in _dump_annots:
                try:
                    arr = np.asarray(a.get_data())
                    if arr.size == 0:
                        continue
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    arr = arr.astype(np.uint8)
                    p = f"{args.dump_obs}/{tag}_{cname}.png"
                    (_imageio.imwrite(p, arr) if _imageio is not None
                     else np.save(p.replace('.png', '.npy'), arr))
                except Exception as _e:  # noqa: BLE001
                    print(f"[bridge] dump {cname} 실패: {_e}", flush=True)

        # 1회 warmup — vla-ros 연결·구동 시작 대기(obs publish 하며 step). 미구동 시 ep1 거짓 실패 방지.
        warmup_steps = max(0, int(args.eval_warmup * control_hz))
        if warmup_steps:
            print(f"[bridge] eval warmup {args.eval_warmup}s (vla-ros 구동 대기)…", flush=True)
            for _ in range(warmup_steps):
                if not simulation_app.is_running():
                    break
                world.step(render=True)
        for ep in range(args.eval):
            if not simulation_app.is_running():
                break
            reset_scene(args.seed + ep)   # 에피소드별 재현 가능 seed (학습 DR)
            for _ in range(settle_steps):
                if not simulation_app.is_running():
                    break
                world.step(render=True)
            bowl_start = np.asarray(bowl_handle.get_world_pose()[0], dtype=np.float64)
            bowl_max_xy_m = 0.0
            bowl_max_z_m = 0.0
            initial_inside = {n: cube_in_bowl(h)[0] for n, h in cube_handles.items()}
            ever = dict(initial_inside)                  # 한 번이라도 그릇 안(진단용)
            prev_inside = dict(initial_inside)
            inside_streak = {n: 0 for n in active_cubes}
            cube_history = {
                n: {
                    "first_in_step": 0 if initial_inside[n] else None,
                    "last_in_step": 0 if initial_inside[n] else None,
                    "last_exit_step": None,
                    "entry_count": 1 if initial_inside[n] else 0,
                    "exit_count": 0,
                    "inside_steps": 0,
                    "max_inside_streak": 0,
                }
                for n in active_cubes
            }
            success_step: int | None = None
            for si in range(ep_steps):
                if not simulation_app.is_running():
                    break
                world.step(render=True)
                bowl_now = np.asarray(bowl_handle.get_world_pose()[0], dtype=np.float64)
                bowl_max_xy_m = max(
                    bowl_max_xy_m,
                    float(np.hypot(bowl_now[0] - bowl_start[0], bowl_now[1] - bowl_start[1])),
                )
                bowl_max_z_m = max(bowl_max_z_m, abs(float(bowl_now[2] - bowl_start[2])))
                if ep == 0 and _dump_annots and si in (1, ep_steps // 2, ep_steps - 2):
                    _save_dump(f"ep0_s{si:03d}")   # bridge 렌더 + arm joint (start/mid/end)
                current_inside = {}
                for n, h in cube_handles.items():
                    inside = cube_in_bowl(h)[0]
                    current_inside[n] = inside
                    if inside:
                        ever[n] = True
                        cube_history[n]["inside_steps"] += 1
                        inside_streak[n] += 1
                        cube_history[n]["max_inside_streak"] = max(
                            cube_history[n]["max_inside_streak"], inside_streak[n]
                        )
                        if cube_history[n]["first_in_step"] is None:
                            cube_history[n]["first_in_step"] = si
                        cube_history[n]["last_in_step"] = si
                    else:
                        inside_streak[n] = 0
                    if inside and not prev_inside[n]:
                        cube_history[n]["entry_count"] += 1
                    elif prev_inside[n] and not inside:
                        cube_history[n]["exit_count"] += 1
                        cube_history[n]["last_exit_step"] = si
                    prev_inside[n] = inside
                # 실제 task termination과 같이 all-cube 동시 성공 즉시 episode 종료.
                # 고정 horizon 끝까지 명령을 계속 보내면 이미 놓은 cube를 다시 건드려 거짓 실패가 된다.
                if current_inside and all(current_inside.values()):
                    success_step = si
                    break
            # 에피소드 종료 시점 판정(엄격 = task_done 정의)
            final = {n: cube_in_bowl(h) for n, h in cube_handles.items()}
            n_final = sum(1 for v in final.values() if v[0])
            n_ever = sum(ever.values())
            all_ok = success_step is not None or (n_final == n_active)
            bowl_final = np.asarray(bowl_handle.get_world_pose()[0], dtype=np.float64)
            recovery_layout = {"cubes": {}, "bowl": None}
            for name, handle in cube_handles.items():
                position, orientation = handle.get_world_pose()
                recovery_layout["cubes"][name] = [
                    float(value) for value in (*position[:3], *orientation[:4])
                ]
            bowl_position, bowl_orientation = bowl_handle.get_world_pose()
            recovery_layout["bowl"] = [
                float(value) for value in (*bowl_position[:3], *bowl_orientation[:4])
            ]
            episodes.append({
                "episode": ep,
                "n_final": n_final, "n_ever": n_ever, "all_ok": all_ok,
                "success_step": success_step,
                "bowl_motion": {
                    "max_xy_mm": round(bowl_max_xy_m * 1000, 1),
                    "max_z_mm": round(bowl_max_z_m * 1000, 1),
                    "final_xy_mm": round(
                        float(np.hypot(
                            bowl_final[0] - bowl_start[0],
                            bowl_final[1] - bowl_start[1],
                        )) * 1000,
                        1,
                    ),
                    "final_z_mm": round(float(bowl_final[2] - bowl_start[2]) * 1000, 1),
                },
                "recovery_layout": recovery_layout,
                "cubes": {n: {"in_bowl": bool(final[n][0]),
                              "xy_mm": round(final[n][1] * 1000, 1),
                              "z": round(final[n][2], 4),
                              "ever": bool(ever[n]),
                              **cube_history[n]} for n in active_cubes},
            })
            per_cube = " ".join(f"{n}:{'O' if final[n][0] else 'x'}"
                                f"({final[n][1] * 1000:.0f}mm,z{final[n][2]:.3f})" for n in active_cubes)
            print(f"[eval] ep {ep + 1}/{args.eval}: {n_final}/{n_active} placed "
                  f"(ever {n_ever}) all_ok={all_ok} · bowl max Δxy={bowl_max_xy_m * 1000:.1f}mm "
                  f"| {per_cube}", flush=True)

        n_ep = len(episodes)
        if n_ep:
            all_rate = sum(e["all_ok"] for e in episodes) / n_ep
            cube_rate = sum(e["n_final"] for e in episodes) / (n_ep * n_active)
            avg_placed = sum(e["n_final"] for e in episodes) / n_ep
            ever_rate = sum(e["n_ever"] for e in episodes) / (n_ep * n_active)
            summary = {
                "model": "taehunkim/so101_smolvla_sim_pick_cube",
                "n_episodes": n_ep, "n_active_cubes": n_active,
                "eval_seconds": args.eval_seconds, "eval_settle": args.eval_settle,
                "eval_bowl_kinematic": bool(args.eval_bowl_kinematic),
                "eval_bowl_mass": float(args.eval_bowl_mass),
                "eval_bowl_friction": (
                    [float(value) for value in args.eval_bowl_friction]
                    if args.eval_bowl_friction is not None
                    else None
                ),
                "all_cubes_success_rate": round(all_rate, 4),
                "per_cube_placement_rate": round(cube_rate, 4),
                "avg_cubes_placed": round(avg_placed, 3),
                "per_cube_ever_rate": round(ever_rate, 4),
                "success_radius_m": BOWL_SUCCESS_RADIUS,
                "z_window": [
                    CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[0],
                    CUBE_DESK_TOP_Z + BOWL_HEIGHT_RANGE[1] + 0.10,
                ],
                "episodes": episodes,
            }
            out_path = _repo_path(args.eval_out)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            print("=" * 64, flush=True)
            print(f"[eval] 🎬 결과 ({n_ep} ep): all-{n_active} 성공률 = {all_rate * 100:.1f}% · "
                  f"per-cube 배치율 = {cube_rate * 100:.1f}% · 평균 {avg_placed:.2f}/{n_active} placed "
                  f"(ever {ever_rate * 100:.1f}%)", flush=True)
            print(f"[eval] JSON → {out_path}", flush=True)
            print("=" * 64, flush=True)
        return

    # 메인 루프: World.step 이 물리 step + 렌더 + OmniGraph(OnPlaybackTick) 평가를 한다.
    # ── step 프로파일: world.step wall-time 누적, N step 마다 min/mean/max + 유효 step/s 출력.
    #    reset step 은 scene 재구성이라 skew → 타이밍에서 제외(reset 직후 윈도 초기화).
    _step_ms: list[float] = []
    _STEP_REPORT_EVERY = 120  # ~4s @ render_dt 30fps
    while simulation_app.is_running():
        if reset_req["mode"] is not None:
            if reset_req["mode"] == "random":
                current_seed["v"] = int.from_bytes(os.urandom(4), "little")  # 새 무작위 seed
            reset_scene(current_seed["v"])   # R=동일 seed 재현 · N=새 seed
            reset_req["mode"] = None
            _step_ms.clear()  # reset 후 윈도 초기화(첫 step skew 제외)
        _ts = time.perf_counter()
        world.step(render=True)
        _step_ms.append((time.perf_counter() - _ts) * 1e3)

        if len(_step_ms) >= _STEP_REPORT_EVERY:
            n = len(_step_ms)
            mean = sum(_step_ms) / n
            print(f"[bridge] world.step ms (n={n}): "
                  f"min/mean/max={min(_step_ms):.1f}/{mean:.1f}/{max(_step_ms):.1f} "
                  f"→ ~{1000.0 / mean:.1f} step/s (목표 33.3ms / 30fps)", flush=True)
            _step_ms.clear()


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
