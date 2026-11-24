"""Cube Pick-and-Place task configuration — pure Isaac Lab 2.3.2.

leisaac Workshop 3층 사다리 이식의 **leaf 층**. base(``so101_base_env_cfg.SO101TeleopEnvCfg``,
로봇+책상+조명+액션+joint 관측+sim 설정)를 상속해 태스크 고유분만 얹는다:
씬 오브젝트(큐브/그릇) · contact 센서 · subtask 관측(contact grasp + 그릇 안 배치) ·
성공/실패 종료 · DR 이벤트. env 변형 4종(default/Fixed/Eval/EvalFixed)은 Workshop 의
base/DR/Eval/DR-Eval 매트릭스 대응(우리는 DR-on 이 기본이라 축을 Fixed=DR-off 로 뒤집음).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH  # noqa: F401 (일부 스크립트 참조 대비 re-export)
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
)
from sim_to_real.utils.domain_randomization import (
    randomize_camera_focal,
    randomize_cubes_scattered,
    randomize_lights,
    randomize_object_mass,
    randomize_object_material,
    randomize_object_on_arc,
    randomize_robot_color,
)

from sim_to_real.tasks.pick_cube import mdp as task_mdp
from sim_to_real.tasks.pick_cube import spawn_area  # DR 스폰 영역 단일 기하 소스
from sim_to_real.tasks.so101_base_env_cfg import (
    SO101BaseEventCfg,
    SO101BaseSceneCfg,
    SO101PolicyObservationsCfg,
    SO101TeleopEnvCfg,
    _ROBOT_POS,  # noqa: F401  (bridge/스크립트가 pick_cube_env_cfg 에서 import → re-export)
    _ROBOT_ROT,  # noqa: F401
)


# World-frame (x, y) of the bowl = success/obs 컨테이너 중심. _BOWL_INIT_STATE 와 반드시 동기화
# (object_in_container 가 이 중심 기준 success radius 판정). 2026-06-29 사용자 DR-off 위치 변경에
# 맞춰 y 0.315→0.265 (책상 앞 모서리 env y=-0.035 에서 +30cm).
BOWL_CENTER_XY: tuple[float, float] = spawn_area.BOWL_CENTER_XY  # 단일 소스=spawn_area
BOWL_SUCCESS_RADIUS: float = 0.06
BOWL_HEIGHT_RANGE: tuple[float, float] = (0.005, 0.12)


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

# 로봇 SO-101 이 **top-down 으로 grasp 가능한 범위**를 **좌우대칭 종 모양(bell)**으로 반영.
# 측정 파이프라인(2026-07-01): pink IK sweep(kinematic map) → gen-traj → isaac 물리 replay
# 검증(46/50 셀 물리 grasp 성공, 고정 GRASP_ORIENT=face 정렬). 상세=[[grasp-sweep-topdown-validation]].
#   물리 성공 셀의 **per-y 넓은쪽 |x|** 를 좌우대칭으로 취함(사용자 지시):
#     y  6/10/14cm → |x|≤0.24  ·  18cm → 0.20  ·  22cm → 0.16  ·  26cm → 0.08
#   → 밑동 넓고 위로 갈수록 좁아지는 종 모양. 사각형 대신 이 프로파일로 스폰 제한.
#   base 발치(r<min_base_sep)·그릇 근접은 randomize_cubes 의 min_base_sep/min_bowl_sep 가 rejection.
# 데스크 매트(860×400mm, env-local center=(0.09,0.245)) 안: 매트 x∈[-0.34,0.44] y∈[0.045,0.445].
_MAT_BL_ENV: tuple[float, float] = (-0.34, 0.045)  # 매트 좌하단 env-local (참고용)
# 종 모양 스폰 프로파일·bounding box·로봇암 제외박스 = spawn_area 단일 소스(순수 python,
# isaaclab 무의존). pickplace_sm --sweep_grid·plot_sweep 가 같은 상수를 import → 경계 정합.
_CUBE_SCATTER_BELL = spawn_area.CUBE_SCATTER_BELL
_CUBE_SCATTER_X_RANGE = spawn_area.CUBE_SCATTER_X_RANGE
_CUBE_SCATTER_Y_RANGE = spawn_area.CUBE_SCATTER_Y_RANGE
_CUBE_ARM_EXCLUDE = spawn_area.CUBE_ARM_EXCLUDE
# base 모드 스폰 사각형(env-local): 책상 왼쪽끝서 X[30,50]cm·앞모서리 Y[25,35]cm →
#   env_x = -0.44 + Xcm/100, env_y = -0.045 + Ycm/100. nominal 큐브(y=0.255) 주변 좁은 영역.
_CUBE_BASE_X_RANGE: tuple[float, float] = (-0.14, 0.06)   # X 30~50cm
_CUBE_BASE_Y_RANGE: tuple[float, float] = (0.205, 0.305)  # Y 25~35cm
# 큐브 한 변 (cube_specs 단일 진실 소스): Cube1/2=40mm, Cube3/4=50mm.
# DR 의 큐브간·그릇 이격이 footprint 반경(s·√2/2)에 맞춰 큐브별로 커지게 하는 기준.
_CUBE_SIZES_M: dict[str, float] = dict(CUBE_SIZES)
_CUBE_VOLUME_INSET: float = 0.0  # 종 프로파일=중심 graspable → rect inset 불요

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
# Scene — base(SO101BaseSceneCfg) 상속 + 태스크 오브젝트(큐브/그릇/contact 센서)
# ---------------------------------------------------------------------------


# ContactSensor 큐브 필터(모듈 상수 — scene 클래스 속성으로 두면 asset 으로 오인됨)
_CUBE_CONTACT_FILTER: list[str] = [f"{{ENV_REGEX_NS}}/Scene/{n}" for n in CUBE_NAMES]


@configclass
class PickCubeSceneCfg(SO101BaseSceneCfg):
    """base 씬(책상+로봇+조명) + 1 cube(40mm) + bowl + 양 손가락 contact 센서."""

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
    # 양 손가락이 같은 큐브에 접촉 = 실제 envelop grasp 신호(any_cube_grasped 가 소비).
    # 필터 목록은 모듈 상수(_CUBE_CONTACT_FILTER) — 클래스 속성이면 InteractiveScene 이
    # asset 으로 오인하므로 클래스 밖에 둔다. (robot 의 activate_contact_sensors=True 는 base 에서.)
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
    # 카메라 리그 (static) — North Star 계약: observation.images.{top,wrist,front}, 640×480 RGB.
    # Workshop 참조처럼 scene cfg 에 **static** 정의(옛 동적주입 add_pick_cube_cameras 대체).
    # 동적주입 함수는 teleop ``--tune_cameras`` 오버라이드용으로 유지 — 같은 필드명(top/wrist/
    # front_camera)을 setattr 로 덮어써 정합(무해).
    #   top   = world 고정(급경사 부감) · wrist = gripper 링크 자식(gripper 회전 추종) ·
    #   front = shoulder 링크 자식(shoulder_pan 회전 추종).
    # ⚠ static 이므로 pick_cube env 생성은 **--enable_cameras 필요** — 무카메라 스모크는
    #   base ``Teleop-v0`` 를 쓴다(카메라 없는 substrate).
    # ------------------------------------------------------------------
    top_camera: TiledCameraCfg = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/TopCamera", _TOP_CAMERA_POS, _TOP_CAMERA_ROT, _TOP_CAMERA_FOCAL,
        focus_distance=1.3, clipping_range=(0.1, 6.0),
    )
    wrist_camera: TiledCameraCfg = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera", _WRIST_CAM_LOCAL_POS, _WRIST_CAM_LOCAL_ROT,
        _WRIST_CAMERA_FOCAL, focus_distance=0.2, clipping_range=(0.02, 3.0),
    )
    front_camera: TiledCameraCfg = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/shoulder/FrontCamera", _FRONT_CAM_LOCAL_POS, _FRONT_CAM_LOCAL_ROT,
        _FRONT_CAMERA_FOCAL, focus_distance=1.0, clipping_range=(0.1, 6.0),
    )


# ---------------------------------------------------------------------------
# 카메라 리그 튜너 오버라이드 — static 카메라(PickCubeSceneCfg)를 --tune_cameras 로 덮어쓰기
# (+ remove_pick_cube_cameras: 무카메라 실행 시 카메라·images 관측 제거)
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
    """top/wrist/front 카메라 cfg 3개를 반환(각 640×480 RGB).

    카메라는 이제 PickCubeSceneCfg 에 **static** 필드로 있다(env 기본). 이 함수는 teleop
    ``--tune_cameras`` GUI 튜너가 pos/rot/focal 을 런타임 오버라이드할 때 쓴다
    (``add_pick_cube_cameras`` 가 반환 cfg 로 static 필드를 같은 이름으로 덮어씀).

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
    """top/wrist/front 카메라 cfg 를 scene cfg 인스턴스에 in-place setattr 하고 반환.

    static 카메라 필드(top/wrist/front_camera)를 **같은 이름으로 덮어쓴다** — teleop 튜너
    오버라이드 용도(무해). InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 등록된다.
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


def remove_pick_cube_cameras(env_cfg) -> None:
    """카메라(scene static 필드 3개 + ``observations.images`` 그룹)를 env_cfg 에서 제거한다.

    카메라가 이제 PickCubeSceneCfg 에 static 이라 env 기본은 ``--enable_cameras`` 를 요구한다.
    무카메라 실행(teleop/smoke)은 gym.make 전에 이 함수로 scene 카메라와 image 관측을 **함께**
    떼어낸다 — scene 만 지우면 images obs 가 없는 sensor 를 참조해 에러가 난다.
    """
    for name in ("top_camera", "wrist_camera", "front_camera"):
        if hasattr(env_cfg.scene, name):
            delattr(env_cfg.scene, name)
    if hasattr(env_cfg.observations, "images"):
        delattr(env_cfg.observations, "images")


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order) — base 액션에 gripper cap override
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
    """6-dim joint position action matching North Star joint order (gripper cap 2.5)."""

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
# Observations — base policy(6-dim joint) 상속 + subtask(grasp + 그릇 배치)
# ---------------------------------------------------------------------------


@configclass
class PickCubeObservationsCfg(SO101PolicyObservationsCfg):
    """policy(6-dim joint, base 상속) + subtask 신호(contact grasp + cube-in-bowl)."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """서브태스크 신호: contact-sensor grasp + 그릇 안 배치."""

        # contact-sensor grasp 신호(leisaac any_vial_grasped 이식, 양 손가락 envelope).
        cube_grasped = ObsTerm(
            func=task_mdp.any_cube_grasped,
            params={
                "jaw_sensor_cfg": SceneEntityCfg("contact_jaw"),
                "gripper_sensor_cfg": SceneEntityCfg("contact_gripper"),
                "cubes": CUBE_NAMES,
                "desk_top_z": _DESK_TOP_WORLD_Z,
                "min_lift": 0.03,
                "warmup_steps": 15,
                "force_threshold": 0.5,
            },
        )

        place_cube1 = ObsTerm(
            func=task_mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Cube1"),
                "container_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )

        # EE frame pose (7D, robot-root frame) — Workshop ee_frame_state 이식. privileged 신호
        # (policy 6-dim joint 계약과 별개, datagen/eval/디버그용). idx0=gripper 기준.
        ee_pose = ObsTerm(
            func=task_mdp.ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    @configclass
    class VisualCfg(ObsGroup):
        """카메라 RGB 관측 — North Star ``observation.images.{top,wrist,front}`` (VLA 이미지 입력).

        Workshop VisualCfg 이식. 각 term = static TiledCamera(PickCubeSceneCfg.{top,wrist,
        front}_camera) 의 rgb. ``mdp.image``(isaaclab stock) 사용, normalize=False(uint8 원본).
        concatenate_terms=False → 카메라별 개별 텐서. ⚠ --enable_cameras 필요.
        """

        top = ObsTerm(
            func=task_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "data_type": "rgb", "normalize": False},
        )
        wrist = ObsTerm(
            func=task_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        front = ObsTerm(
            func=task_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    images: VisualCfg = VisualCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class PickCubeRewardsCfg:
    """추론/데이터 substrate — RL 보상 제거(VLA-only 리팩토링). 빈 보상 그룹."""

    pass


# ---------------------------------------------------------------------------
# Terminations — 기본(순간 성공) + Eval(디바운스 성공)
# ---------------------------------------------------------------------------


# success termination 공용 파라미터(순간/확정 둘이 공유)
_SUCCESS_PARAMS: dict = {
    "objects_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
    "container_center_xy": BOWL_CENTER_XY,
    "container_cfg": SceneEntityCfg(BOWL_NAME),
    "radius": BOWL_SUCCESS_RADIUS,
    "height_range": BOWL_HEIGHT_RANGE,
    "require_rest_pose": False,  # rest-pose check is TA.1 territory
}
# 큐브 추락 실패 컷 파라미터
_CUBE_LOST_PARAMS: dict = {
    "objects_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
    "fall_z": 0.10,
}


@configclass
class PickCubeTerminationsCfg:
    """Episode ends on timeout or when all cubes are in the bowl (순간 판정)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=task_mdp.task_done, params=dict(_SUCCESS_PARAMS))
    # 큐브 추락 = 회복 불가 → 실패 종료(time_out=False, success 아님). 잘못된 grasp 로
    # 큐브를 책상 밖/아래로 쳐낸 에피소드를 빠르게 컷(낭비 방지) + 안 쳐내도록 압력.
    cube_lost = DoneTerm(func=task_mdp.cube_lost, time_out=False, params=dict(_CUBE_LOST_PARAMS))


@configclass
class PickCubeEvalTerminationsCfg:
    """Eval 용 — 성공을 ``task_done_confirmed`` 로 디바운스(N step 연속 성립 시에만).

    한 프레임 떨림으로 큐브가 그릇 반경에 순간 들어왔다 나가는 가짜 성공을 걸러
    eval 성공률을 안정화(leisaac vial_placed_on_rack_termination confirm-counter 이식).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.task_done_confirmed,
        params={**_SUCCESS_PARAMS, "confirm_steps": 15},
    )
    cube_lost = DoneTerm(func=task_mdp.cube_lost, time_out=False, params=dict(_CUBE_LOST_PARAMS))


# ---------------------------------------------------------------------------
# Events — base(리셋)만 = DR-off 기본. DR 변형이 큐브/그릇/시각/물리 DR 를 얹음.
# ---------------------------------------------------------------------------


def _make_randomize_cubes(x_range, y_range, bell):
    """큐브 배치 EventTerm 팩토리. full/base 모드 공통 인자 + 모드별 영역.

    공통: full_orient · 큐브간/그릇 이격(min_cube_sep·min_bowl_sep, 겹침 금지) ·
    base 발치 이격(min_base_sep) · **로봇암 주변 제외박스**(_CUBE_ARM_EXCLUDE).
    모드별: full=좌우대칭 종모양(bell != None) · base=사각형(bell=None).
    """
    return randomize_cubes_scattered(
        CUBE_NAMES,
        BOWL_NAME,
        x_range=x_range,
        y_range=y_range,
        full_orient=True,
        volume_inset=_CUBE_VOLUME_INSET,
        min_cube_sep=0.060,
        min_bowl_sep=spawn_area.MIN_BOWL_SEP,   # 큐브-그릇 겹침 금지(spawn_area 단일 소스)
        cube_sizes=[_CUBE_SIZES_M[n] for n in CUBE_NAMES],
        min_base_sep=spawn_area.MIN_BASE_SEP,   # base 발치(inner-reach) 배제
        base_sep_offset_xy=spawn_area.PAN_AXIS_XY,  # ★min-reach 중심=pan축(마운트원점+offset). sweep 과 단일소스
        x_halfwidth_by_y=bell,             # full=종모양 · base=None(사각형)
        x_exclude_box=_CUBE_ARM_EXCLUDE,   # 로봇암 주변 배제(full·base 공통)
    )


@configclass
class PickCubeDREventCfg(SO101BaseEventCfg):
    """base 리셋(씬/포즈 jitter) + 큐브/그릇 무작위 배치 + 물리 DR + 시각 DR (DR-on 변형).

    ``randomize_cubes`` = **full 모드**(좌우대칭 종모양, grasp 물리검증 범위). base 모드는
    ``PickCubeDRBaseEventCfg`` (nominal 주변 좁은 사각형).
    """

    # full 모드: 좌우대칭 종모양 스폰 + 로봇암 제외 + 그릇/큐브/base 이격.
    randomize_cubes = _make_randomize_cubes(
        _CUBE_SCATTER_X_RANGE, _CUBE_SCATTER_Y_RANGE, _CUBE_SCATTER_BELL
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

    # 시각 DR(reset, sim2real): 라이트 밝기·색온도 + 카메라 focal + 로봇 plastic 색.
    # 카메라 리그 없으면 focal 은 no-op. shader 없으면 robot color 도 no-op.
    # cuRobo oracle 은 큐브 world pose 만 쓰므로 grasp 성공률에 무영향(obs 시각만 변화).
    randomize_lights = randomize_lights()
    randomize_camera_focal = randomize_camera_focal()
    randomize_robot_color = randomize_robot_color()  # leisaac resets.py 이식(plastic 바디만)

    def __post_init__(self) -> None:
        # 물리 DR(startup): 큐브별 마찰/질량을 무작위화해 env 간 물리 다양성 확보.
        # 동적 setattr 한 EventTerm 도 EventManager 가 cfg.__dict__ 에서 수집한다.
        # grasp weld/유지력 추가가 아니라 표면/질량 분산만 주므로 reward hacking 아님.
        for name in CUBE_NAMES:
            setattr(self, f"randomize_{name.lower()}_material", randomize_object_material(name))
            setattr(self, f"randomize_{name.lower()}_mass", randomize_object_mass(name))


@configclass
class PickCubeDRBaseEventCfg(PickCubeDREventCfg):
    """**base 모드** DR — 큐브를 nominal(y≈0.255) 주변 좁은 사각형(_CUBE_BASE_*)에만 스폰.

    full 모드(PickCubeDREventCfg=좌우대칭 종모양)와 큐브 배치 영역만 다르고, 나머지
    이벤트(그릇 arc·물리·시각 DR·로봇암 제외·그릇 이격)는 그대로 상속한다.
    """

    # base 모드: 사각형(bell=None) — 책상 왼쪽끝 X[30,50]cm·Y[25,35]cm.
    randomize_cubes = _make_randomize_cubes(
        _CUBE_BASE_X_RANGE, _CUBE_BASE_Y_RANGE, None
    )


# ---------------------------------------------------------------------------
# Environment configs — Workshop base/DR/Eval/DR-Eval 대응(우리는 DR-on 기본)
# ---------------------------------------------------------------------------


@configclass
class PickCubeEnvCfg(SO101TeleopEnvCfg):
    """Cube Pick-and-Place environment — base teleop substrate + 태스크(**DR-off 기본**).

    기본 등록 env(``SimToReal-SO101-PickCube-v0``). Workshop base 처럼 DR 없음:
    큐브/그릇 고정 실측 배치(``_CUBE_INIT_STATES``/``_BOWL_INIT_STATE``, base 리셋만) ·
    순간 성공 종료. DR 이 필요하면 ``PickCubeDREnvCfg``(-DR) 를 쓴다.
    """

    scene: PickCubeSceneCfg = PickCubeSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickCubeObservationsCfg = PickCubeObservationsCfg()
    actions: PickCubeActionsCfg = PickCubeActionsCfg()
    rewards: PickCubeRewardsCfg = PickCubeRewardsCfg()
    terminations: PickCubeTerminationsCfg = PickCubeTerminationsCfg()
    events: SO101BaseEventCfg = SO101BaseEventCfg()  # DR-off: 씬 리셋 + 소폭 포즈 jitter만
    # teleop-device 배선(use_teleop_device·preprocess_device_action·dynamic_reset_gripper_effort_limit·
    # task_type)은 base SO101TeleopEnvCfg 로 승격됨 — 여기서 상속.

    def __post_init__(self) -> None:
        super().__post_init__()  # base: sim dt/decimation/physx/episode_length
        # 비디오/뷰포트 카메라(RecordVideo 가 이 viewer 를 씀) — 작업공간 정면·약간
        # 낮은 각도로 두어 머리 위 KeyLight 평면에 가리지 않게 한다. world 좌표(env0).
        # robot base (0,0,0.6749), 큐브/그릇 작업공간 x~-0.24~0.26, y~0.07~0.37.
        self.viewer.eye = (0.06, 1.515, 0.98)
        self.viewer.lookat = (0.01, 0.245, 0.76)
        self.viewer.resolution = (1280, 720)


@configclass
class PickCubeDREnvCfg(PickCubeEnvCfg):
    """DR-on 변형(**full 모드**) — 큐브 좌우대칭 종모양 scatter + 그릇 arc + 물리·시각 DR."""

    events: PickCubeDREventCfg = PickCubeDREventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 시각 DR 의 robot color 는 Replicator 로 per-env OmniPBR 를 바인딩한다. Fabric
        # replication 이 켜지면 전 env 가 env_0 material 로 렌더돼 색이 무시되므로 끈다
        # (randomize_robot_color 가 True 면 RuntimeError). 상세=domain_randomization.
        self.scene.replicate_physics = False


@configclass
class PickCubeDRBaseEnvCfg(PickCubeEnvCfg):
    """DR-on **base 모드** 변형 — 큐브 스폰을 nominal 주변 좁은 사각형으로 제한(그 외 full 동일)."""

    events: PickCubeDRBaseEventCfg = PickCubeDRBaseEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.replicate_physics = False  # robot color DR(Replicator) 요구 — 위 참조


@configclass
class PickCubeEvalEnvCfg(PickCubeEnvCfg):
    """Eval 변형 — DR-off 고정 layout + 디바운스 성공 종료(가장 재현성 높은 평가)."""

    terminations: PickCubeEvalTerminationsCfg = PickCubeEvalTerminationsCfg()


@configclass
class PickCubeEvalDREnvCfg(PickCubeDREnvCfg):
    """Eval + DR — 무작위 layout + 디바운스 성공 종료(DR 하 성공률 평가)."""

    terminations: PickCubeEvalTerminationsCfg = PickCubeEvalTerminationsCfg()
