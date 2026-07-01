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
    MAX_CUBE_FOOTPRINT_RADIUS,
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
BOWL_CENTER_XY: tuple[float, float] = (-0.22, 0.265)
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

# 로봇 SO-101 이 **top-down 으로 grasp 가능한 범위**로 제한. pink IK top-down sweep 으로 측정
# (2026-07-01, scripts/datagen/pink_ik_bridge_node.py --sweep, 2cm grid, robot base=(0,0)):
#   각 큐브 중심에서 hover+grasp waypoint 를 azimuth 정렬 top-down 방향으로 IK 풀어
#   ①위치 도달(err<2cm) ②achieved TCP z축이 수직에서 <25° 를 만족하면 graspable.
#   측정 결과: 도달 가능=거의 곧 top-down(5-DOF 팔, 닿는 곳이면 아래로 향함) → binding=reachability.
#   graspable envelope = 초승달(x∈[-0.25,0.25], y∈[0.06,0.28], 외곽 반원 r≤0.284).
#   그 안에 맞는 **100%-graspable 최대 사각형(큐브 중심)** = x[-0.17,0.17]×y[0.08,0.22] 를 채택
#   (사용자 선택 "최대 확장"; y=0.08 은 arm-fold 경계 근처). 아래 볼륨 rect = center rect ± volume_inset.
#   base 발치(r<min_base_sep)·그릇 근접은 randomize_cubes 의 min_base_sep/min_bowl_sep 가 rejection.
# 데스크 매트(860×400mm, env-local center=(0.09,0.245)) 안: 매트 x∈[-0.34,0.44] y∈[0.045,0.445].
#   아래 범위는 매트 내부 + graspable — y_lo 는 매트 앞모서리(0.045)와 일치.
_MAT_BL_ENV: tuple[float, float] = (-0.34, 0.045)  # 매트 좌하단 env-local (참고용)
_CUBE_SCATTER_X_RANGE: tuple[float, float] = (-0.205, 0.205)          # center inset 후 x∈[-0.17,0.17]
_CUBE_SCATTER_Y_RANGE: tuple[float, float] = (_MAT_BL_ENV[1], _MAT_BL_ENV[1] + 0.21)  # (0.045,0.255) → center y∈[0.08,0.22]
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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


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


@configclass
class PickCubeDREventCfg(SO101BaseEventCfg):
    """base 리셋(씬/포즈 jitter) + 큐브/그릇 무작위 배치 + 물리 DR + 시각 DR (DR-on 변형)."""

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
    """DR-on 변형 — 큐브/그릇 scatter+arc + 물리 DR + 시각 DR(sim2real·데이터 다양성)."""

    events: PickCubeDREventCfg = PickCubeDREventCfg()


@configclass
class PickCubeEvalEnvCfg(PickCubeEnvCfg):
    """Eval 변형 — DR-off 고정 layout + 디바운스 성공 종료(가장 재현성 높은 평가)."""

    terminations: PickCubeEvalTerminationsCfg = PickCubeEvalTerminationsCfg()


@configclass
class PickCubeEvalDREnvCfg(PickCubeDREnvCfg):
    """Eval + DR — 무작위 layout + 디바운스 성공 종료(DR 하 성공률 평가)."""

    terminations: PickCubeEvalTerminationsCfg = PickCubeEvalTerminationsCfg()
