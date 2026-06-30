"""Cube Pick-and-Place task configuration — pure Isaac Lab 2.3.2."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from sim_to_real.assets.robots.lerobot import SO101_FOLLOWER_CFG
from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_CFG, CUBE_DESK_USD_PATH
from so101_contract import SO101_JOINT_ORDER
from sim_to_real.tasks.common.utils import (
    SO101_JOINT_TARGET_MAX_VELOCITY,
    _look_at_quat_world,
    _pinhole_camera_cfg,
    _yaw_quat,
)
from sim_to_real.utils.constant import (
    BOWL_NAME,
    CUBE_HALF_EXTENTS,
    CUBE_NAMES,
    CUBE_SIZES,
    MAX_CUBE_FOOTPRINT_RADIUS,
)
from sim_to_real.utils.domain_randomization import (
    randomize_camera_focal,
    randomize_cubes_scattered,
    randomize_lights,
    randomize_object_mass,
    randomize_object_material,
    randomize_object_on_arc,
)

from sim_to_real.tasks.pick_cube import mdp as task_mdp


# World-frame (x, y) of the bowl = success/obs 컨테이너 중심. _BOWL_INIT_STATE 와 반드시 동기화
# (object_in_container 가 이 중심 기준 success radius 판정). 2026-06-29 사용자 DR-off 위치 변경에
# 맞춰 y 0.315→0.265 (책상 앞 모서리 env y=-0.035 에서 +30cm).
BOWL_CENTER_XY: tuple[float, float] = (-0.22, 0.265)
BOWL_SUCCESS_RADIUS: float = 0.06
BOWL_HEIGHT_RANGE: tuple[float, float] = (0.005, 0.12)

# Robot base position — recenter 로 world 원점(XY)에 배치.
# x: 0.0 (desk_left_edge=-0.44 + 440mm 장착)
# y: 0.0 (책상 앞 모서리 world y=-0.045 기준 약간 뒤)
# z: desk_top(0.705) - base_min_z(0.0301) = 0.6749 (z 불변)
_ROBOT_POS = (0.0, 0.0, 0.6749)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


# 큐브 world 좌표 = SCENE_OFFSET(0.36, 0.045, 0.705) + scene-local 위치 (+ y 0.01 shift).
# xy/yaw 는 평면 배치, z 는 cube_specs 파생(반높이) — author CUBES 와 동일 규약.
#   z중심 = 책상 상판(0.705, 매트 제거됨) + 큐브 반높이 + slack(0.001). 40mm→0.726.
# 단일 큐브 씬(2026-06-26): 40mm Cube1 1개만.
_DESK_TOP_WORLD_Z: float = 0.705
_CUBE_Z_SLACK: float = 0.001
_CUBE_LAYOUT: dict[str, tuple[float, float, float]] = {  # name -> (x, y, yaw°)
    # DR-off 기본 위치(2026-06-30 사용자 실측): 큐브 중심 = 책상 앞 모서리(env y=-0.035)에서 +29cm,
    # 책상 왼쪽 모서리(env x=-0.44)에서 +42.5cm → env (-0.015, 0.255). (실기기 실측 정합: 42.5cm·29cm)
    # yaw=0: 책상 앞 모서리(x축)와 평행한 큐브 한 면이 -y(로봇)를 향함 → face가 로봇 정면,
    #        그 면 법선이 모서리와 수직(=-y).
    "Cube1": (-0.015, 0.255, 0.0),
}
_CUBE_INIT_STATES = {
    name: (
        (x, y, _DESK_TOP_WORLD_Z + CUBE_HALF_EXTENTS[name] + _CUBE_Z_SLACK),
        _yaw_quat(yaw),
    )
    for name, (x, y, yaw) in _CUBE_LAYOUT.items()
}
# DR-off 기본 위치(2026-06-29 사용자): 그릇 중심 = 책상 앞 모서리(env y=-0.035)에서 +30cm,
# 책상 왼쪽 모서리(env x=-0.44)에서 +22cm → env (-0.22, 0.265). z=0.715 유지. BOWL_CENTER_XY 와 동기.
_BOWL_INIT_STATE = ((-0.22, 0.265, 0.715), _yaw_quat(0.0))

# ---------------------------------------------------------------------------
# 큐브 scatter workspace — randomize_cubes_scattered 기본값 및 커리큘럼 계산 기준
# ---------------------------------------------------------------------------

# 로봇 SO-101 도달(reach) 범위 안쪽으로 제한. reach 매핑 sweep(1큐브 full-scatter 12 ep,
# robot base=(0,0))으로 가장자리 실패 편향을 측정해 보수화 (recenter delta=(-1.84,+0.565)):
#   · x 극단(≤-0.23, ≥0.22)에서 grasp 실패 편향 → x_range 를 [-0.18, 0.20] 로 (±0.20 안쪽).
#   · y 매트 뒤 가장자리(≤0.10, base 에 너무 가까워 arm 이 접혀 top-down 자세 불리) 실패 →
#     y_lo 를 0.105 로. y 상한 0.22 는 그릇(y=0.305)과 이격(min_bowl_sep 추가 보장).
# (가장자리 외 실패는 reach 가 아니라 joint_fk random-FK 의 marginal grasp 분산임 — CONTEXT 참고.)
# 큐브 스폰 사각형 = 데스크 매트 위 사용자 지정 영역 (매트-local cm, 좌하단=(0,0)).
#   매트(860×400mm) env-local center=(0.09,0.245) → 좌하단 = (-0.34, 0.045).
#   매핑: env_x = -0.34 + Xcm/100, env_y = 0.045 + Ycm/100.
#   사용자 지정: X∈[16,56]cm, Y∈[11,25]cm → 아래 env-local m 범위.
#   이 범위는 **큐브 볼륨**의 사각형 경계 → DR 은 volume_inset 만큼 중심을 안쪽으로.
_MAT_BL_ENV: tuple[float, float] = (-0.34, 0.045)
_CUBE_SCATTER_X_RANGE: tuple[float, float] = (_MAT_BL_ENV[0] + 0.16, _MAT_BL_ENV[0] + 0.56)  # (-0.18, 0.22)
_CUBE_SCATTER_Y_RANGE: tuple[float, float] = (_MAT_BL_ENV[1] + 0.11, _MAT_BL_ENV[1] + 0.25)  # (0.155, 0.295)
# 큐브 한 변 (cube_specs 단일 진실 소스): Cube1/2=40mm, Cube3/4=50mm.
# DR 의 큐브간·그릇 이격이 footprint 반경(s·√2/2)에 맞춰 큐브별로 커지게 하는 기준.
_CUBE_SIZES_M: dict[str, float] = dict(CUBE_SIZES)
# 볼륨이 사각형 안에 들도록 중심 inset = max 큐브(50mm) face 대각 절반 ((s/2)·√2).
_CUBE_VOLUME_INSET: float = MAX_CUBE_FOOTPRINT_RADIUS  # ≈ 0.0354

# ---------------------------------------------------------------------------
# 카메라 리그 상수 — North Star 계약: observation.images.{top,wrist,front}
#   · 모두 640×480 (W×H) RGB, update_period=1/30
#   · 포즈/FOV 는 cube_task GUI 튜너와 실제 데이터셋 프레임 기준으로 보정.
#   · top 은 world frame 절대 좌표, wrist 는 gripper 링크 자식 prim 의 local offset.
#     num_envs=1 smoke 기준.
# ---------------------------------------------------------------------------

# 값은 GUI 카메라 튜너(teleop_se3_agent.py)로 보정한 결과. rot 은 모두
# wxyz, Isaac Lab world-convention(forward +X, up +Z).
# top: 로봇 뒤(-y)·높은 곳에서 내려보는 급경사 oblique. (2026-06-26 튜너 재보정)
_TOP_CAMERA_POS = (-0.17, 0.77, 1.05)
# _TOP_CAMERA_ROT 가 None 이 아니면 이 quat 을 직접 쓰고, None 이면 _TOP_CAMERA_TARGET
# 으로 look_at 을 계산한다(하위호환). 값은 GUI 튜너 rot_xyz_deg=(63.5, 0, -168.5) → world wxyz.
_TOP_CAMERA_ROT = (0.7538, 0.145, 0.1775, -0.6159)
_TOP_CAMERA_TARGET = (0.30, 0.425, 0.76)          # (미사용 — _TOP_CAMERA_ROT 직접 지정)
_TOP_CAMERA_FOCAL = 19.0

# wrist: gripper 위/옆에 강결합된 카메라. (2026-06-26 튜너 재보정)
# rot 은 GUI 튜너 rot_xyz_deg=(-29.5, 0, 0) → gripper-local world wxyz.
_WRIST_CAM_LOCAL_POS = (0.0, 0.045, -0.04)
_WRIST_CAM_LOCAL_ROT = (0.3562, -0.6108, 0.6108, 0.3562)
_WRIST_CAMERA_FOCAL = 19.0

# front: shoulder 링크에 장착 — shoulder_pan 회전을 따라간다.
# (USD 컨벤션: URDF `shoulder_link` → USD `shoulder`, `_link` 접미사 제거)
# pos/rot 은 --tune_cameras GUI 튜너로 실측한 shoulder local frame 값. (2026-06-26 재보정)
#   rot_xyz_deg=(-90, 0, -90), rot_quat=(0, 0, 1, 0) wxyz
_FRONT_CAMERA_POS = (-0.03, -0.005, 0.75)     # world ref (shoulder_pan=0, 기록용)
_FRONT_CAM_LOCAL_POS = (-0.045, 0.0, 0.025)      # shoulder local frame (GUI 튜너 실측)
_FRONT_CAM_LOCAL_ROT = (0.0, 0.0, 1.0, 0.0)      # wxyz shoulder local frame (fwd=local -x=world -Y)
_FRONT_CAMERA_FOCAL = 19.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


# ContactSensor 큐브 필터(모듈 상수 — scene 클래스 속성으로 두면 asset 으로 오인됨)
_CUBE_CONTACT_FILTER: list[str] = [f"{{ENV_REGEX_NS}}/Scene/{n}" for n in CUBE_NAMES]


@configclass
class PickCubeSceneCfg(InteractiveSceneCfg):
    """Scene: cube desk + SO-101 follower + 1 cube (40mm) + bowl."""

    # shared world assets (not per-env)
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # 조명: /World 계층(env 밖) 단일 배치 → InteractiveScene 이 복제하지 않음.
    # USD 광원은 scope 격리가 없어 {ENV_REGEX_NS}/Scene 안에 두면 env 수만큼 복제돼
    # N배 과노출(IsaacLab #4340/#1729). 그래서 scene.usd 에서 광원을 빼고 여기서
    # /World/Light(dome)·/World/KeyLight(distant) 1개씩만 author 한다. env=1·N-env
    # 모두 동일 노출(스케일링 불필요). 강도/색은 기존 scene.usd 값(2000/1800) 이식.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.98, 0.95), angle=1.0),
        # RotateXYZ(-50,0,-35)° 등가 quat(wxyz) — 위에서 비스듬히 내리쬐어 입체감.
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.8644, -0.4031, -0.1271, -0.2725)
        ),
    )

    # cube desk USD (contains desk, lighting, and all rigid objects; mat removed)
    # +y 0.01 shift: 책상/매트 등 정적 지오메트리를 로봇 기준 1cm 뒤로. 큐브/그릇 rigid body 는
    # 각 init_state(env-frame)로 동일 shift 반영(독립) — Scene translate 와 이중이동 없음.
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.01, 0.0)),
    )

    # SO-101 follower articulation — 단일소스 SO101_FOLLOWER_CFG(assets/robots/lerobot.py,
    # leisaac 이식)에서 검증된 actuator/solver/rigid_props 값을 가져오고, 씬 특화 필드만 덮어쓴다.
    #  - prim_path: per-env 네임스페이스
    #  - spawn: contact sensor 활성(jaw/gripper ↔ 큐브 접촉 리포트) — base spawn 의 usd_path·
    #    rigid_props·articulation_props 는 그대로 유지하고 activate_contact_sensors 만 토글.
    #  - init_state: 우리 base pose + gripper 0.20.
    #    gripper offset(=init 값). action target = raw*scale(1.0)+offset, clip 1.0 → 도달범위
    #    [offset-1, offset+1]. offset 0.20(닫힘쪽) → do-nothing target 이 잡은 큐브 유지,
    #    open 1.20 까지(30mm grasp 충분)·close -0.174 full 도달.
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_FOLLOWER_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            # 초기 자세(2026-06-26 사용자 지정, 단위=deg→rad). gripper=0 rad(중립).
            # elbow_flex 요청값 +100° 는 USD joint 상한(+90°) 초과 → 90° 로 캡(실기 한계).
            joint_pos={
                "shoulder_pan": math.radians(0.0),
                "shoulder_lift": math.radians(-100.0),
                "elbow_flex": math.radians(90.0),    # 요청 +100°, USD 상한 90° 로 캡
                "wrist_flex": math.radians(70.0),
                "wrist_roll": math.radians(-100.0),
                "gripper": 0.0,
            },
        ),
    )

    # Rigid objects inside the scene USD (spawn=None → wrap existing prims)
    Cube1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube1",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CUBE_INIT_STATES["Cube1"][0],
            rot=_CUBE_INIT_STATES["Cube1"][1],
        ),
    )
    Bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bowl",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_BOWL_INIT_STATE[0],
            rot=_BOWL_INIT_STATE[1],
        ),
    )

    # 두 손가락(jaw=가동, gripper=고정) ↔ 큐브 접촉 센서. force_matrix_w 로 큐브별 접촉력.
    # 양 손가락이 같은 큐브에 접촉 = 실제 envelop grasp 신호(기하 proxy 보다 직접적).
    # 필터 목록은 모듈 상수(_CUBE_CONTACT_FILTER) — 클래스 속성이면 InteractiveScene 이
    # asset 으로 오인하므로 클래스 밖에 둔다.
    contact_jaw: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        filter_prim_paths_expr=_CUBE_CONTACT_FILTER,
    )
    contact_gripper: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        filter_prim_paths_expr=_CUBE_CONTACT_FILTER,
    )

    # ------------------------------------------------------------------
    # 카메라는 기본 씬에 두지 않는다 — env_smoke 가 --enable_cameras 없이 돌도록.
    # 카메라 smoke/롤아웃은 gym.make() 전에 add_pick_cube_cameras(scene) 로 주입.
    # (InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 동적 주입이 센서로
    #  등록됨 — Isaac Lab 2.3.2 interactive_scene._add_entities_from_cfg 확인.)
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 카메라 리그 (선택 주입) — observation.images.{top,wrist,front}
# ---------------------------------------------------------------------------


def make_pick_cube_camera_cfgs(
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
    front_local_pos: tuple[float, float, float] | None = None,
    front_local_rot: tuple[float, float, float, float] | None = None,
    front_focal: float | None = None,
) -> dict[str, TiledCameraCfg]:
    """top/wrist/front 카메라 cfg 3개를 반환.

    기본 PickCubeSceneCfg 밖에 두어 기본 env 가 --enable_cameras 없이 돌게 한다.
    gym.make() 전에 add_pick_cube_cameras() 로 scene cfg 에 주입해서 쓴다.
    각 카메라는 480×640 RGB.

    top 회전: ``top_target`` 이 주어지면 look_at 계산, 없으면 튜닝된 기본 ``_TOP_CAMERA_ROT``.
    front: shoulder_link 하위 prim — shoulder_pan 회전을 따라간다.
           pos/rot 은 shoulder_link local frame 기준.
    """

    top_pos = _TOP_CAMERA_POS if top_pos is None else top_pos
    top_focal = _TOP_CAMERA_FOCAL if top_focal is None else top_focal
    wrist_local_pos = _WRIST_CAM_LOCAL_POS if wrist_local_pos is None else wrist_local_pos
    wrist_local_rot = _WRIST_CAM_LOCAL_ROT if wrist_local_rot is None else wrist_local_rot
    wrist_focal = _WRIST_CAMERA_FOCAL if wrist_focal is None else wrist_focal
    front_local_pos = _FRONT_CAM_LOCAL_POS if front_local_pos is None else front_local_pos
    front_local_rot = _FRONT_CAM_LOCAL_ROT if front_local_rot is None else front_local_rot
    front_focal = _FRONT_CAMERA_FOCAL if front_focal is None else front_focal

    if top_target is not None:
        top_quat = _look_at_quat_world(top_pos, top_target)
    else:
        top_quat = _TOP_CAMERA_ROT

    top = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/TopCamera",
        top_pos,
        top_quat,
        top_focal,
        focus_distance=1.3,
        clipping_range=(0.1, 6.0),
    )
    # wrist: gripper 링크 자식 prim → gripper 회전을 따라간다.
    wrist = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        wrist_local_pos,
        wrist_local_rot,
        wrist_focal,
        focus_distance=0.2,
        clipping_range=(0.02, 3.0),
    )
    # front: shoulder 링크 자식 prim → shoulder_pan 회전을 따라간다.
    # USD에서 URDF `shoulder_link` → `shoulder` (`_link` 접미사 제거 컨벤션).
    front = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/shoulder/FrontCamera",
        front_local_pos,
        front_local_rot,
        front_focal,
        focus_distance=1.0,
        clipping_range=(0.1, 6.0),
    )
    return {"top_camera": top, "wrist_camera": wrist, "front_camera": front}


def add_pick_cube_cameras(
    scene_cfg: PickCubeSceneCfg,
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
    front_local_pos: tuple[float, float, float] | None = None,
    front_local_rot: tuple[float, float, float, float] | None = None,
    front_focal: float | None = None,
) -> PickCubeSceneCfg:
    """top/wrist/front 카메라 리그를 scene cfg 인스턴스에 in-place 주입하고 반환.

    InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 여기서 추가한 속성이
    gym.make() 시 센서로 등록된다.
    """

    for name, cam_cfg in make_pick_cube_camera_cfgs(
        top_pos=top_pos,
        top_target=top_target,
        top_focal=top_focal,
        wrist_local_pos=wrist_local_pos,
        wrist_local_rot=wrist_local_rot,
        wrist_focal=wrist_focal,
        front_local_pos=front_local_pos,
        front_local_rot=front_local_rot,
        front_focal=front_focal,
    ).items():
        setattr(scene_cfg, name, cam_cfg)
    return scene_cfg


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order)
# ---------------------------------------------------------------------------


# pick_cube 전용 slew 상한: arm 5.0 유지, **그리퍼만 2.5 rad/s** 로 낮춤.
# 닫을 때 명령 속도를 줄여 큐브를 튕겨내지 않게(정렬 유지 → grasp valley 완화).
# ⚠ arm 2.5 하드캡은 **금물**: cuRobo batch 가 lock-step(고정 step수)으로 sparse plan 을 따라가는데
#   arm 을 2.5 로 묶으면 transit/descend 를 정해진 step 안에 못 끝내고 lag → grasp/place 어긋남
#   (동일 seed·DR layout 측정 all-4 90.6→59.4%). 게다가 생성 데이터는 arm 5.0 에서도 이미
#   within-task max≈2.5 rad/s(실데이터 정합) → 추가 cap 은 이득 0·컨트롤러 파손.
# 공유 상수(SO101_JOINT_TARGET_MAX_VELOCITY, pen 과 공용)는 건드리지 않는다.
_PICKCUBE_JOINT_MAX_VELOCITY: dict[str, float] = {
    **SO101_JOINT_TARGET_MAX_VELOCITY,
    "gripper": 2.5,
}


@configclass
class PickCubeActionsCfg:
    """6-dim joint position action matching North Star joint order."""

    arm: task_mdp.SlewLimitedJointPositionActionCfg = task_mdp.SlewLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        scale=1.0,
        # offset 전면 제거(VLA-only 리팩토링): action = 절대 joint target.
        # affine gripper codec + rad-space offset 0 → sim·실기기·bridge 일관(LeIsaac 동형).
        use_default_offset=False,
        max_velocity=_PICKCUBE_JOINT_MAX_VELOCITY,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class PickCubeObservationsCfg:
    """Observations: policy (6-dim joint pos) + subtask signals + rl_policy (privileged RL state).

    Groups:
      policy      — 6-dim joint pos (North Star contract, immutable).
      subtask_terms — per-cube placement signals.
      rl_policy   — TB.3 privileged state (43-dim) for RL actor/critic.
                    Does NOT contain the policy group; use obs_groups in train.py
                    to map both policy and critic to rl_policy.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """6-dim joint position in North Star order."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=SO101_JOINT_ORDER,
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Per-cube placement signals (cube-in-bowl, gripper open check)."""

        place_cube1 = ObsTerm(
            func=task_mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Cube1"),
                "container_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RlPolicyCfg(ObsGroup):
        """TB.3 privileged state for RL training (83-dim, concatenated).

        Includes joint pos, gripper body pos, all cube/bowl positions relative to env
        origin, gripper→cube relative vectors, gripper open fraction, velocities,
        그리고 grasp 정렬용 방향·크기(cube yaw, half-extent, ee quat, grasp→cup).
        No FrameTransformer dependency — resolves gripper body by name.
        """

        # 센서 노이즈(DR): 상태에 보수적 Gaussian(σ=0.005) 주입 → 추정/엔코더
        # 오차에 대한 robust 화. 단위가 섞여 있어 작은 std 로 시작(필요 시 항목별 분리).
        rl_state_obs = ObsTerm(
            func=task_mdp.rl_state,
            params={
                "object_names": CUBE_NAMES,
                "container_name": BOWL_NAME,
                "include_velocities": True,  # joint_vel+ee vel+cube vel 추가(부분관측 해소) → 43→64dim
                "include_orientation": True,  # cube yaw+half-extent+ee quat+grasp→cup → 64→83dim
                "include_container_orientation": True,  # 그릇 quat → 83→87dim(동적 그릇 tilt/엎힘 관측)
                # 큐브 크기(half-extent, m) — cube_specs 파생(40mm→0.020, 50mm→0.025).
                # 평행 jaw 벌림 폭 매칭에 필수. 하드코딩 금지(예전 drift→실측보다 작게 관측한
                # 잠복 결함 원인). CUBE_NAMES 순서와 일치.
                "object_half_extents": tuple(CUBE_HALF_EXTENTS[n] for n in CUBE_NAMES),
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.005),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True  # DR: rl_state 에 노이즈 적용
            self.concatenate_terms = True

    @configclass
    class GraspFocusCfg(ObsGroup):
        """RND novelty 전용 grasp 부분공간(~30dim). 전체 rl_state 대신 grasp 직결 차원만."""

        grasp_focus_obs = ObsTerm(
            func=task_mdp.grasp_focus_state,
            params={"cube_name": CUBE_NAMES[0]},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()
    rl_policy: RlPolicyCfg = RlPolicyCfg()
    grasp_focus: GraspFocusCfg = GraspFocusCfg()


# ---------------------------------------------------------------------------
# Rewards — Phase B 단계형 보상
# ---------------------------------------------------------------------------


@configclass
class PickCubeRewardsCfg:
    """추론/데이터 substrate — RL 보상 제거(VLA-only 리팩토링). 빈 보상 그룹."""

    pass
# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class PickCubeTerminationsCfg:
    """Episode ends on timeout or when all cubes are in the bowl."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.task_done,
        params={
            "objects_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "radius": BOWL_SUCCESS_RADIUS,
            "height_range": BOWL_HEIGHT_RANGE,
            "require_rest_pose": False,  # rest-pose check is TA.1 territory
        },
    )
    # 큐브 추락 = 회복 불가 → 실패 종료(time_out=False, success 아님). 잘못된 grasp 로
    # 큐브를 책상 밖/아래로 쳐낸 에피소드를 빠르게 컷(낭비 방지) + 안 쳐내도록 압력.
    cube_lost = DoneTerm(
        func=task_mdp.cube_lost,
        time_out=False,
        params={
            "objects_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
            "fall_z": 0.10,
        },
    )


# ---------------------------------------------------------------------------
# Events (domain randomisation)
# ---------------------------------------------------------------------------


@configclass
class PickCubeEventCfg:
    """Reset and randomisation events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 시작 포즈 jitter (DR): ±0.05 rad(~3°) — 실기 reset 편차 모사. velocity 0 유지.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 큐브(현재 1개)를 책상 위 사각형 영역에 완전 무작위 배치 (rejection sampling).
    #   · 볼륨이 사각형 안: volume_inset(최대 50mm cube face 대각 절반)
    #   · 볼륨 비겹침: cube_sizes 로 per-pair 이격 동적 계산 (r_i+r_j+margin). 50mm쌍 ≈0.071,
    #                  40mm쌍 ≈0.061. min_cube_sep=0.060 은 cube_sizes 미지정 시 fallback.
    #                  min_bowl_sep=0.14(40mm 정합)도 큐브별 +(r−r_40) 보정 → 50mm 면 ≈0.147.
    #   · full_orient: 이산 stable-face + random yaw (face 다양·drift 0·z 띄움 불요)
    randomize_cubes = randomize_cubes_scattered(
        CUBE_NAMES,
        BOWL_NAME,
        x_range=_CUBE_SCATTER_X_RANGE,
        y_range=_CUBE_SCATTER_Y_RANGE,
        full_orient=True,
        volume_inset=_CUBE_VOLUME_INSET,
        min_cube_sep=0.060,
        min_bowl_sep=0.14,
        cube_sizes=[_CUBE_SIZES_M[n] for n in CUBE_NAMES],  # 큐브 크기 대응 이격
        # base 발치(inner-reach, r<~0.13)는 안전고도 접근 IK 부재로 수행 불가.
        min_base_sep=0.135,
    )

    # 그릇 호(arc) 랜덤화 범위는 두 기하 제약으로 결정된다.
    #
    # 제약 A — 매트 왼쪽 경계(world x=-0.34):
    #   bowl_center_x - r_top(0.075) >= -0.34 + 0.01(여유)
    #   -0.22 + 0.44*sin(a) >= -0.255  →  sin(a) >= -0.0795  →  a >= -4.56°
    #   → 왼쪽 한계 -4°
    #
    # 제약 B — 그릇-Cube3 겹침:
    #   유효 충돌 반경 = r_top(0.075) + Cube3 half-diag(0.0354)
    #                   + cube_contactOffset(0.002) + bowl_contactOffset(0.004) = 0.1164m
    #   Cube3 최악 위치 (-0.05, 0.235) 기준 임계 각도 풀면 9.48°.
    #   → 안전 여유 포함 오른쪽 한계 +8°
    randomize_bowl = randomize_object_on_arc(BOWL_NAME, radius=0.44, angle_range_deg=(-4.0, 8.0))

    # 시각 DR(reset, sim2real): 라이트 밝기·색온도 + 카메라 focal. 카메라 리그 없으면 focal 은 no-op.
    # cuRobo oracle 은 큐브 world pose 만 쓰므로 grasp 성공률에 무영향(obs 시각만 변화).
    randomize_lights = randomize_lights()
    randomize_camera_focal = randomize_camera_focal()

    def __post_init__(self) -> None:
        # 물리 DR(startup): 큐브별 마찰/질량을 무작위화해 env 간 물리 다양성 확보.
        # 동적 setattr 한 EventTerm 도 EventManager 가 cfg.__dict__ 에서 수집한다.
        # grasp weld/유지력 추가가 아니라 표면/질량 분산만 주므로 reward hacking 아님.
        for name in CUBE_NAMES:
            setattr(self, f"randomize_{name.lower()}_material", randomize_object_material(name))
            setattr(self, f"randomize_{name.lower()}_mass", randomize_object_mass(name))


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class PickCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Cube Pick-and-Place environment — pure Isaac Lab 2.3.2 ManagerBased."""

    scene: PickCubeSceneCfg = PickCubeSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickCubeObservationsCfg = PickCubeObservationsCfg()
    actions: PickCubeActionsCfg = PickCubeActionsCfg()
    rewards: PickCubeRewardsCfg = PickCubeRewardsCfg()
    terminations: PickCubeTerminationsCfg = PickCubeTerminationsCfg()
    events: PickCubeEventCfg = PickCubeEventCfg()
    dynamic_reset_gripper_effort_limit: bool = True
    # 활성 teleop/datagen device 타입 (use_teleop_device 가 설정). 기본=실 leader.
    task_type: str = "so101leader"

    def use_teleop_device(self, teleop_device: str) -> None:
        """teleop/datagen device 선택. leisaac task 템플릿 등가.

        task_type 저장 + 직접 joint 제어(키보드/게임패드/state-machine) 시 중력 off
        (떨림 없는 결정적 제어). action term 구성(init_action_cfg)은 teleop/SM 드라이버가
        필요 시 별도 호출한다(우리 기본 action = VLA용 slew joint target).
        """
        self.task_type = teleop_device
        if teleop_device in ["keyboard", "gamepad", "so101_state_machine"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict, teleop_device) -> "object":
        """device 출력 → action tensor. vendored devices.action_process 에 위임.

        ``Device.advance()`` 가 호출한다. devices 는 isaac-sim 런타임서만 import 되므로
        지연 import(serial/lerobot 미설치 컨테이너서 패키지 import 안전 — Docker 정합 규칙1).
        """
        from sim_to_real.devices.action_process import preprocess_device_action

        return preprocess_device_action(action, teleop_device)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        # GPU pipeline
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # aggregate pair: 134k/4096 → ~268k/8192 → ~536k/16384. 16384 env 대비 1M 로 상향.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        # contact patch 버퍼. ~41 patch/env: 8192≈336k, 16384≈672k(>10·2^16 overflow).
        # 16384 env 대비 16·2^16=1048576 로 상향(VRAM 예산 32GB).
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**16
        # contact stack 도 대규모 env 대비 상향.
        self.sim.physx.gpu_collision_stack_size = 2**29
        # 비디오/뷰포트 카메라(RecordVideo 가 이 viewer 를 씀) — 작업공간 정면·약간
        # 낮은 각도로 두어 머리 위 KeyLight 평면에 가리지 않게 한다. world 좌표(env0).
        # robot base (0,0,0.6749), 큐브/그릇 작업공간 x~-0.24~0.26, y~0.07~0.37.
        self.viewer.eye = (0.06, 1.515, 0.98)
        self.viewer.lookat = (0.01, 0.245, 0.76)
        self.viewer.resolution = (1280, 720)

