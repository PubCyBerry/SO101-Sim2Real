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
import os
import threading
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
parser.add_argument("--num_cubes", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--cube_name", default="",
                    help="단일 활성 큐브 직접 지정(크기별 eval: Cube1/2=40mm·Cube3/4=50mm). "
                         "빈값=CUBE_NAMES[:num_cubes]. 비활성 큐브는 z=-1 park(카메라 밖).")
parser.add_argument("--dr", action="store_true", help="큐브 위치를 scatter 범위로 무작위화")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_cameras",
    action="store_true",
    help="top/wrist/front 카메라 ROS publish 비활성화 (기본은 publish — VLA obs 용).",
)
# GUI 뷰포트 레이아웃 — pick_cube_curobo_demo.py 와 동일(Perspective + top/wrist/front 3-패널 dock).
parser.add_argument("--view_eye", type=float, nargs=3, default=[0.9, -0.9, 1.15],
                    help="GUI Perspective 카메라 eye(world). demo 기본값과 동일.")
parser.add_argument("--view_lookat", type=float, nargs=3, default=[0.20, 0.10, 0.70],
                    help="GUI Perspective 카메라 lookat(world). demo 기본값과 동일.")
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
# ── SmolVLA cross-attention 오버레이 (SmolVLA 전용) ──────────────────────────
parser.add_argument("--attention_overlay", action="store_true",
                    help="policy-server-attn(SmolVLA)이 PUB 하는 cross-attention 히트맵을 SUB 해 "
                         "top/wrist/front 뷰에 omni.ui 오버레이 창으로 표시. GUI(not headless) 전용.")
parser.add_argument("--attn_zmq_host", default="127.0.0.1",
                    help="attention 히트맵 ZMQ PUB 호스트(policy-server). network_mode host → loopback.")
parser.add_argument("--attn_zmq_port", type=int, default=5556,
                    help="attention 히트맵 ZMQ 포트(서버 --attn_zmq_port 와 일치).")
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
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402 (광원 spawn — curobo demo PickCubeSceneCfg 와 동일 cfg)
import isaaclab.sim.schemas as schemas  # noqa: E402 (순수 USD authoring — 시뮬 파이프라인 무관)
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH, ROBOT_USD_PATH  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
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
    MAX_CUBE_FOOTPRINT_RADIUS,
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


# ─────────────────────────────────────────────────────────────────────────────
# SmolVLA cross-attention 오버레이 (policy-server-attn ZMQ SUB → omni.ui 블렌딩)
#
# policy_server_attention_bridge.py(AttentionBridgeServer)가 추론마다 카메라별 히트맵을
# ZMQ PUB(:5556). 여기서 SUB 해 라이브 렌더 프레임(rgb annotator)에 JET 히트맵을 블렌딩,
# omni.ui ByteImageProvider 창 3개(top/wrist/front)로 표시한다. 토글 체크박스는 **표시만**
# on/off(서버는 attn 모드면 항상 계산·PUB). GUI(not headless)에서만 동작.
# ─────────────────────────────────────────────────────────────────────────────
_ATTN_ENABLED = True                # 토글 상태(표시만 토글)
_ATTN_SUB = None                    # zmq SUB socket (None=오버레이 비활성)
_ATTN_HEATMAPS: dict = {}           # {cam: HxW float32 [0,1]} 최신 수신
_ATTN_LOCK: threading.Lock | None = None
_ATTN_WINDOWS: dict = {}            # {cam: (omni.ui.Window, ByteImageProvider)}
_ATTN_ANNOTS: dict = {}             # {cam: replicator rgb annotator (live 프레임)}
_ATTN_CTRL_WIN = None               # 토글 컨트롤 창 핸들 유지(GC 방지)
_ATTN_BLEND_ALPHA = 0.4             # 히트맵 가중(Stanley: 0.6 img + 0.4 heat)
_ATTN_RECV_ERR_LOGGED = False       # 수신 에러 1회만 보고용 플래그


def _attn_cam_from_topic(topic: str) -> str:
    """'/camera/<cam>/image_raw' → '<cam>'."""
    parts = [p for p in topic.split("/") if p]
    return parts[1] if len(parts) >= 2 and parts[0] == "camera" else (parts[0] if parts else topic)


def _init_attention_zmq(host: str, port: int) -> None:
    """히트맵 SUB 소켓 초기화(CONFLATE=최신만, non-block)."""
    global _ATTN_SUB, _ATTN_LOCK
    try:
        import zmq  # noqa: PLC0415
        _ATTN_LOCK = threading.Lock()
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.CONFLATE, 1)   # 최신 1개만 보관(stale 누적 방지)
        sock.connect(f"tcp://{host}:{port}")
        _ATTN_SUB = sock
        print(f"[bridge] attention ZMQ SUB → tcp://{host}:{port}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] attention ZMQ init 실패: {exc}", flush=True)
        _ATTN_SUB = None


def _poll_attention() -> None:
    """non-blocking 으로 최신 히트맵 수신 → 캐시."""
    global _ATTN_HEATMAPS
    if _ATTN_SUB is None:
        return
    try:
        import zmq  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    global _ATTN_RECV_ERR_LOGGED
    got = None
    for _ in range(4):  # CONFLATE 라 1개지만 안전 여유
        try:
            got = _ATTN_SUB.recv_pyobj(zmq.NOBLOCK)
        except zmq.Again:
            break
        except Exception as exc:  # noqa: BLE001  수신·역직렬화 에러는 첫 1회만 보고(조용히 삼키지 않음)
            if not _ATTN_RECV_ERR_LOGGED:
                print(f"[bridge] attention recv 에러(1회만 보고): {exc!r}", flush=True)
                _ATTN_RECV_ERR_LOGGED = True
            break
    if isinstance(got, dict):
        hm = got.get("heatmaps") or {}
        with _ATTN_LOCK:
            _ATTN_HEATMAPS = hm


def _setup_attention_annotators(camera_specs) -> None:
    """카메라 render product 마다 rgb annotator attach (live 오버레이 베이스 프레임)."""
    try:
        import omni.replicator.core as rep  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] attention annotator 모듈 불가: {exc}", flush=True)
        return
    for rp_path, topic, _fid in camera_specs:
        cam = _attn_cam_from_topic(topic)
        try:
            a = rep.AnnotatorRegistry.get_annotator("rgb")
            a.attach(rp_path)
            _ATTN_ANNOTS[cam] = a
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] attention annotator {cam} 실패: {exc}", flush=True)


def _attn_live_frame(cam: str):
    """rgb annotator 에서 HxWx3 uint8 RGB 추출(없으면 None)."""
    a = _ATTN_ANNOTS.get(cam)
    if a is None:
        return None
    try:
        arr = np.asarray(a.get_data())
        if arr.size == 0:
            return None
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr.astype(np.uint8))
    except Exception:  # noqa: BLE001
        return None


def _set_attention_enabled(value: bool) -> None:
    global _ATTN_ENABLED
    _ATTN_ENABLED = bool(value)
    for win, _ in _ATTN_WINDOWS.values():
        try:
            win.visible = bool(value)
        except Exception:  # noqa: BLE001
            pass
    print(f"[bridge] attention overlay: {'ON' if value else 'OFF'}", flush=True)


def _create_attention_ui(camera_specs) -> None:
    """토글 체크박스 창 + 카메라별 ByteImageProvider 오버레이 창 생성."""
    global _ATTN_CTRL_WIN
    try:
        import omni.ui as ui  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] attention UI 모듈 불가: {exc}", flush=True)
        return
    cams = [_attn_cam_from_topic(t) for _, t, _ in camera_specs]

    ctrl = ui.Window("Attention Overlay", width=240, height=84)
    with ctrl.frame:
        with ui.HStack(height=28):
            ui.Label("Show attention", width=150)
            cb = ui.CheckBox(width=24)
            cb.model.set_value(True)
            cb.model.add_value_changed_fn(lambda m: _set_attention_enabled(m.get_value_as_bool()))
    _ATTN_CTRL_WIN = ctrl

    for cam in cams:
        provider = ui.ByteImageProvider()
        win = ui.Window(f"Attention {cam.capitalize()}", width=_CAM_W // 2, height=_CAM_H // 2)
        with win.frame:
            ui.ImageWithProvider(provider)
        _ATTN_WINDOWS[cam] = (win, provider)
        print(f"[bridge] attention overlay 창: {cam}", flush=True)


def _update_attention_overlay() -> None:
    """카메라별 live 프레임 ⊕ JET(heat) 블렌딩 → ByteImageProvider 갱신."""
    try:
        import cv2  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    for cam, (win, provider) in _ATTN_WINDOWS.items():
        try:
            if not win.visible:
                continue
        except Exception:  # noqa: BLE001
            pass
        frame = _attn_live_frame(cam)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        with _ATTN_LOCK:
            heat = _ATTN_HEATMAPS.get(cam)
        if heat is None:
            rgb = frame
        else:
            hr = cv2.resize(np.asarray(heat, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            jet = cv2.applyColorMap((np.clip(hr, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
            rgb = cv2.addWeighted(frame, 1.0 - _ATTN_BLEND_ALPHA, jet, _ATTN_BLEND_ALPHA, 0.0)
        rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
        try:
            provider.set_bytes_data(rgba.tobytes(), [w, h])
        except Exception:  # noqa: BLE001  일부 빌드는 list 만 허용
            provider.set_bytes_data(rgba.reshape(-1).tolist(), [w, h])


def _attention_tick() -> None:
    """매 step 호출: 히트맵 poll + (토글 ON 시) 오버레이 갱신. SUB 없으면 no-op."""
    if _ATTN_SUB is None:
        return
    _poll_attention()
    if _ATTN_ENABLED:
        _update_attention_overlay()


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

    layout_path = (os.path.join(REPO_ROOT, args.layout)
                   if not os.path.isabs(args.layout) else args.layout)
    if os.path.isfile(layout_path):
        try:
            import json
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

    # VLA obs 카메라 — prim + render product 생성(robot/scene prim 이 stage 에 있은 뒤).
    camera_specs: list[tuple[str, str, str]] = []
    if not args.no_cameras:
        camera_specs = setup_cameras(world.stage)

    # OmniGraph 는 reset(=play) 전 timeline 정지 상태에서 생성한다(그래프 wrap + OnPlaybackTick
    # 등록). 노드는 prim 경로만 참조하므로 prim 이 존재하면 된다.
    ros_graph = build_ros_graph(object_prims, camera_specs)

    # reset: 물리 뷰 초기화 + timeline play → OnPlaybackTick 시작.
    world.reset()

    # ── actuator parity (PickCubeEnvCfg = VLA 학습 env) ──
    # USD drive gain 은 micro 라 위치 명령 추종 불가 → 학습 env 와 동일 soft PD 로 덮어쓴다.
    # arm·gripper 모두 stiffness 17.8·damping 0.6 (이전 gripper 80 은 cuMotion 용 → VLA parity 로 통일).
    n_dof = robot.num_dof
    kps = np.full(n_dof, DRIVE_STIFFNESS, dtype=np.float32)
    kds = np.full(n_dof, DRIVE_DAMPING, dtype=np.float32)
    try:
        gi = robot.get_dof_index("gripper")
    except (ValueError, AttributeError, TypeError, KeyError):
        gi = n_dof - 1  # fallback: gripper 가 마지막 dof (/isaac_joint_states 순서 확인됨)
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kps=kps, kds=kds)
    # dof_names 순서 = /isaac_joint_states publish 순서 = recorder joint_pos[:6] 순서.
    # 학습 state(joint_pos[:6])↔추론 state(vla node name-reorder→SO101_JOINT_ORDER) 정합 확인용.
    try:
        print(f"[bridge] dof_names order: {list(robot.dof_names)}", flush=True)
    except Exception:
        pass
    print(f"[bridge] actuator parity: stiffness={DRIVE_STIFFNESS} damping={DRIVE_DAMPING} "
          f"(arm·gripper 공통), gripper dof[{gi}]", flush=True)

    # effort 상한 (PickCubeEnvCfg parity): arm 10Nm, gripper gentle 0.5Nm(leisaac dynamic 동치).
    efforts = np.full(n_dof, ARM_EFFORT_LIMIT, dtype=np.float32)
    efforts[gi] = GRIPPER_EFFORT_LIMIT
    try:
        ctrl.set_max_efforts(values=efforts)
        print(f"[bridge] effort 상한: arm={ARM_EFFORT_LIMIT}Nm · gripper={GRIPPER_EFFORT_LIMIT}Nm", flush=True)
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
    # 초기/리셋 팔 자세 = 학습 데이터(cuRobo recorder) frame-0 state 정합.
    # recorder 는 episode 를 READY([0,-1.3,1.2,-20°,-90°]) settle + gripper open 에서 시작 → 녹화 첫
    # frame state(1024ep 평균, rad)가 정책이 학습한 시작 obs. 기존 home_q=zeros 는 완전 OOD 시작이라
    # 정책 첫 obs 불일치 → 즉시 drift. name→rad 매핑이라 dof 순서 무관.
    _START_POSE_RAD = {
        "shoulder_pan": 0.0, "shoulder_lift": -1.235, "elbow_flex": 1.2623,
        "wrist_flex": -0.3814, "wrist_roll": -1.2342, "gripper": 0.8483,
    }
    try:
        home_q = np.array([_START_POSE_RAD.get(n, 0.0) for n in robot.dof_names], dtype=np.float32)
    except Exception:  # noqa: BLE001  dof_names 접근 실패 시 dof 순서 가정(SO101_JOINT_ORDER)
        home_q = np.array([0.0, -1.235, 1.2623, -0.3814, -1.2342, 0.8483], dtype=np.float32)[:n_dof]
    print(f"[bridge] 초기 팔 자세(학습 frame-0 정합): {[round(float(x), 3) for x in home_q]}", flush=True)

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
                h.set_linear_velocity(np.zeros(3, dtype=np.float32))
                h.set_angular_velocity(np.zeros(3, dtype=np.float32))
            except Exception:
                pass

    # ── DR (학습 randomize_cubes_scattered / randomize_object_on_arc 정합) ──
    _MIN_CUBE_SEP = 0.060        # 큐브 볼륨 비겹침
    _MIN_BOWL_SEP = 0.14         # 큐브-그릇
    _MIN_BASE_SEP = 0.135        # 큐브-base 발치(inner-reach)
    # PickCubeEnvCfg._CUBE_VOLUME_INSET과 동일 — cube_specs 단일 진실 소스에서 파생해
    # env_cfg 와 자동 일치(하드코딩 시 크기 변경 누락→OOD eval 위험 차단).
    _VOLUME_INSET = MAX_CUBE_FOOTPRINT_RADIUS   # ≈0.0354 = max cube(50mm) face 대각 절반
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
        x_lo = _CUBE_SCATTER_X_RANGE[0] + _VOLUME_INSET
        x_hi = _CUBE_SCATTER_X_RANGE[1] - _VOLUME_INSET
        y_lo = _CUBE_SCATTER_Y_RANGE[0] + _VOLUME_INSET
        y_hi = _CUBE_SCATTER_Y_RANGE[1] - _VOLUME_INSET
        placed: list[tuple[float, float]] = []
        for name, h in cube_handles.items():
            fx, fy = float(bowl_default_xy[0]), float(bowl_default_xy[1])  # fallback(드묾)
            for _ in range(_MAX_ATTEMPTS):
                cx = float(rng.uniform(x_lo, x_hi))
                cy = float(rng.uniform(y_lo, y_hi))
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
            try:
                h.set_linear_velocity(np.zeros(3, dtype=np.float32))
                h.set_angular_velocity(np.zeros(3, dtype=np.float32))
            except Exception:
                pass

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
            try:
                bowl_handle.set_linear_velocity(np.zeros(3, dtype=np.float32))
                bowl_handle.set_angular_velocity(np.zeros(3, dtype=np.float32))
            except Exception:
                pass

    def reset_scene(seed: int) -> None:
        """seed 로 DR 재현(큐브 scatter+6D face → 그릇 arc, 학습 순서) + 팔 home.
        동일 seed = 동일 spawn 레이아웃(post-settle 은 PhysX 미세변동). np 전역 무관(default_rng)."""
        rng = np.random.default_rng(int(seed))
        randomize_cubes(rng)   # 학습 순서: 큐브(vs bowl_default) → 그릇 arc
        randomize_bowl(rng)
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
            import json as _json
            _lay = {"active_objects": int(args.num_cubes), "seed": int(seed),
                    "envs": [{"env": 0, "placed": 0, "clean": False, "cubes": {}, "bowl": None}]}
            for _nm, _h in cube_handles.items():
                _pp, _qq = _h.get_world_pose()
                _lay["envs"][0]["cubes"][_nm] = [float(v) for v in (*_pp[:3], *_qq[:4])]
            _bp2, _bq2 = bowl_handle.get_world_pose()
            _lay["envs"][0]["bowl"] = [float(v) for v in (*_bp2[:3], *_bq2[:4])]
            with open(_ld, "w") as _f:
                _json.dump(_lay, _f, indent=2)
            print(f"[bridge] LAYOUT DUMP → {_ld} (seed={seed})", flush=True)
        print(f"[bridge] scene reset + DR (seed={seed})", flush=True)

    # 시작 시 DR 1회(학습 reset=DR 정합). 이후 R/N 으로 재리셋.
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
            from isaacsim.core.utils.viewports import set_camera_view  # noqa: PLC0415
            set_camera_view(eye=args.view_eye, target=args.view_lookat)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] Perspective view 설정 실패: {exc}", flush=True)
        if not args.no_cameras:
            dock_camera_viewports()
        # SmolVLA attention 오버레이 — ZMQ SUB + rgb annotator + omni.ui 창(토글). GUI 전용.
        if args.attention_overlay and not args.no_cameras and camera_specs:
            _init_attention_zmq(args.attn_zmq_host, args.attn_zmq_port)
            _setup_attention_annotators(camera_specs)
            _create_attention_ui(camera_specs)
            print("[bridge] attention overlay 활성 (policy-server-attn PUB 대기)", flush=True)
    elif args.attention_overlay:
        print("[bridge] --attention_overlay 는 GUI(not headless) 전용 → 스킵", flush=True)

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

    if args.eval > 0:
        import json
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
            import os as _os
            _os.makedirs(args.dump_obs, exist_ok=True)
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
                _attention_tick()   # SmolVLA attention 오버레이 갱신(활성 시)
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
            out_path = os.path.join(REPO_ROOT, args.eval_out) if not os.path.isabs(args.eval_out) else args.eval_out
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
        _attention_tick()   # SmolVLA attention 오버레이 갱신(활성 시, step 타이밍 외부)
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
