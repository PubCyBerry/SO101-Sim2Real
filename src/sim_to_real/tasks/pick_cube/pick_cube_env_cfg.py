"""Cube Pick-and-Place task configuration — pure Isaac Lab 2.3.2."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
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

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_CFG, CUBE_DESK_USD_PATH, ROBOT_USD_PATH
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (
    SO101_JOINT_ORDER,
    SO101_JOINT_TARGET_MAX_VELOCITY,
    _look_at_quat_world,
    _pinhole_camera_cfg,
    _yaw_quat,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES
from sim_to_real.utils.domain_randomization import (
    randomize_camera_focal,
    randomize_cubes_scattered,
    randomize_lights,
    randomize_object_mass,
    randomize_object_material,
    randomize_object_on_arc,
)

from sim_to_real.tasks.pick_cube import mdp as task_mdp


# World-frame (x, y) of the bowl at scene authoring time.
# BOWL_LOCAL=(-0.58, 0.26) + SCENE_OFFSET=(0.36, 0.045) = (-0.22, 0.305), +y 0.01 shift → 0.315.
BOWL_CENTER_XY: tuple[float, float] = (-0.22, 0.315)
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
# 매트 윗면 world z=0.709. z중심 = 0.709 + 반높이 + slack(0.001).
#   작은(Cube1/2, 40mm): 0.709+0.020+0.001=0.730, 큰(Cube3/4, 50mm): 0.709+0.025+0.001=0.735.
_CUBE_INIT_STATES = {
    "Cube1": ((-0.14, 0.135, 0.730), _yaw_quat(20.0)),
    "Cube2": ((0.14, 0.115, 0.730), _yaw_quat(-35.0)),
    "Cube3": ((-0.10, 0.225, 0.735), _yaw_quat(50.0)),
    "Cube4": ((0.09, 0.195, 0.735), _yaw_quat(-20.0)),
}
# BOWL_LOCAL(-0.58, 0.26, 0.010) + SCENE_OFFSET(0.36, 0.045, 0.705) = (-0.22, 0.305, 0.715), +y 0.01 → 0.315.
_BOWL_INIT_STATE = ((-0.22, 0.315, 0.715), _yaw_quat(0.0))

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
# 볼륨이 사각형 안에 들도록 중심 inset = max 큐브(50mm) face 대각 절반 ((s/2)·√2).
_CUBE_VOLUME_INSET: float = 0.050 * 0.5 * (2 ** 0.5)  # ≈ 0.0354

# 4개 기본 위치의 중심 — apply_curriculum 에서 scale=0 시 workspace 를 이 점으로 수렴시켜
# fallback(default 위치) 동작을 유도하는 데 사용한다.
_CUBE_SCATTER_CENTER: tuple[float, float] = (
    sum(v[0][0] for v in _CUBE_INIT_STATES.values()) / 4,  # ≈ 1.8375
    sum(v[0][1] for v in _CUBE_INIT_STATES.values()) / 4,  # ≈ -0.4075
)

# ---------------------------------------------------------------------------
# 카메라 리그 상수 — North Star 계약: observation.images.{top,wrist,front}
#   · 모두 640×480 (W×H) RGB, update_period=1/30
#   · 포즈/FOV 는 cube_task GUI 튜너와 실제 데이터셋 프레임 기준으로 보정.
#   · top 은 world frame 절대 좌표, wrist 는 gripper 링크 자식 prim 의 local offset.
#     num_envs=1 smoke 기준.
# ---------------------------------------------------------------------------

# 값은 GUI 카메라 튜너(teleop_se3_agent.py)로 보정한 결과. rot 은 모두
# wxyz, Isaac Lab world-convention(forward +X, up +Z).
# top: 로봇 뒤(-y)·높은 곳에서 내려보는 급경사 oblique.
_TOP_CAMERA_POS = (0.03, -0.005, 1.72)            # +y 0.01 (책상 따라 이동, top 뷰 유지)
# _TOP_CAMERA_ROT 가 None 이 아니면 이 quat 을 직접 쓰고, None 이면 _TOP_CAMERA_TARGET
# 으로 look_at 을 계산한다(하위호환).
_TOP_CAMERA_ROT = (0.5716, -0.4238, 0.4466, 0.5424)
_TOP_CAMERA_TARGET = (0.30, 0.425, 0.76)          # +y 0.01
_TOP_CAMERA_FOCAL = 18.0

# wrist: gripper 위/옆에 강결합된 카메라.
_WRIST_CAM_LOCAL_POS = (0.0, 0.045, -0.04)
_WRIST_CAM_LOCAL_ROT = (-0.3642, 0.6061, -0.6061, -0.3642)
_WRIST_CAMERA_FOCAL = 18.0

# front: shoulder 링크에 장착 — shoulder_pan 회전을 따라간다.
# (USD 컨벤션: URDF `shoulder_link` → USD `shoulder`, `_link` 접미사 제거)
# pos/rot 은 --tune_cameras GUI 튜너로 실측한 shoulder local frame 값.
#   rot_xyz_deg=(-90, 0, -90), rot_quat=(0, 0, 1, 0) wxyz
_FRONT_CAMERA_POS = (-0.03, -0.005, 0.75)     # world ref (shoulder_pan=0, 기록용)
_FRONT_CAM_LOCAL_POS = (-0.040, 0.0, 0.025)      # shoulder local frame (GUI 튜너 실측)
_FRONT_CAM_LOCAL_ROT = (0.0, 0.0, 1.0, 0.0)      # wxyz shoulder local frame (fwd=local -x=world -Y)
_FRONT_CAMERA_FOCAL = 18.0


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


# ContactSensor 큐브 필터(모듈 상수 — scene 클래스 속성으로 두면 asset 으로 오인됨)
_CUBE_CONTACT_FILTER: list[str] = [f"{{ENV_REGEX_NS}}/Scene/{n}" for n in CUBE_NAMES]


@configclass
class PickCubeSceneCfg(InteractiveSceneCfg):
    """Scene: cube desk + SO-101 follower + 4 cubes + bowl."""

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

    # cube desk USD (contains desk, lighting, mat, and all rigid objects)
    # +y 0.01 shift: 책상/매트 등 정적 지오메트리를 로봇 기준 1cm 뒤로. 큐브/그릇 rigid body 는
    # 각 init_state(env-frame)로 동일 shift 반영(독립) — Scene translate 와 이중이동 없음.
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.01, 0.0)),
    )

    # SO-101 follower articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            # ContactSensor(jaw/gripper ↔ 큐브 접촉)용 — 로봇 rigid body 접촉 리포트 활성화.
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                # leisaac SO101_FOLLOWER_CFG 검증값(enabled_self_collisions + solver 4/4).
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            # gripper offset(=이 init 값). action target = raw*scale(1.0)+offset, clip 1.0
            # → 도달범위 [offset-1, offset+1]. 트레이드오프:
            #  - offset 큼(0.80): "아무것도 안 함(action≈0)" → target 0.80 = 활짝 열림.
            #    부트스트랩으로 큐브를 잡고 시작해도 정책이 손을 벌려 곧 놓침(하류 학습 저해).
            #  - offset 0.20: do-nothing target 0.20(닫힘쪽, open 판정<0.6) → 잡은 큐브 유지.
            #    open 은 1.20 까지(30mm 큐브 grasp 충분), close 는 -0.174 full 도달.
            # pregrasp 공짜획득 우려는 pregrasp 보상 재설계(weight 0.5, diff 0.045)로 해소됨.
            joint_pos={**{j: 0.0 for j in SO101_JOINT_ORDER}, "gripper": 0.20},
        ),
        actuators={
            # leisaac SO101_FOLLOWER_CFG 검증값 이식 (ref_repos/leisaac 의
            # assets/robots/lerobot.py). Feetech STS3215 를 낮은 stiffness(soft PD)
            # + 높은 effort 상한으로 모델링한다. 그리퍼가 큐브에 막혀도 클램프
            # 토크가 최대 10 Nm 까지 올라가 grasp 가 유지된다(이전 1.5 Nm 상한은
            # stiffness 300 에서 ~0.3° 만에 포화돼 들어올릴 때 미끄러짐).
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll"],
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
                stiffness=17.8,
                damping=0.6,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
                stiffness=17.8,
                damping=0.6,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
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
    Cube2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube2",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CUBE_INIT_STATES["Cube2"][0],
            rot=_CUBE_INIT_STATES["Cube2"][1],
        ),
    )
    Cube3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube3",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CUBE_INIT_STATES["Cube3"][0],
            rot=_CUBE_INIT_STATES["Cube3"][1],
        ),
    )
    Cube4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube4",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_CUBE_INIT_STATES["Cube4"][0],
            rot=_CUBE_INIT_STATES["Cube4"][1],
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
        use_default_offset=True,
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
        place_cube2 = ObsTerm(
            func=task_mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Cube2"),
                "container_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )
        place_cube3 = ObsTerm(
            func=task_mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Cube3"),
                "container_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )
        place_cube4 = ObsTerm(
            func=task_mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Cube4"),
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
                # 큐브 크기(half-extent, m): Cube1/2=30mm→0.015, Cube3/4=40mm→0.020.
                # 평행 jaw 벌림 폭 매칭에 필수(크기 2종). CUBE_NAMES 순서와 일치.
                "object_half_extents": (0.015, 0.015, 0.020, 0.020),
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
    """단계형 보상 — reach → grasp → lift → transport → insert → release."""

    # Stage 1: EE → 가장 가까운 미배치 큐브 접근 (밀집)
    reach_cube = RewTerm(
        func=task_mdp.reach_reward,
        weight=0.0,  # v24: 1→0 (든 큐브=미배치@EE → reach 1.0 지급 = hover income. grasp_align 이 접근 대체)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 1.5: 열린 그리퍼를 큐브에 정밀 3D 정렬 (밀집, 탐색 valley 메움)
    # weight 1.0(<close 3.0): open→closed 보상 합이 단조 증가가 되도록 align 을 close 보다
    # 낮춤. hover 캠프 매력 최소화 — 닫을수록 이득이 커져 valley 가 사라진다.
    grasp_align_cube = RewTerm(
        func=task_mdp.grasp_align_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            # tolerance 확대(0.05/0.06 → 0.12/0.10): early exploration 의 arm drift(3~5cm) 가
            # tight 창을 벗어나 reward=0 → gradient 死. 완화로 "대충 정렬한 큐브도 보상" →
            # 점화 gradient 밀도↑(rl-expert). 점화 후 재축소 가능.
            "align_xy": 0.12,
            "align_z": 0.10,
        },
    )

    # Stage 1.7: 정렬된 채 닫기 (align→lift valley 메움). align(open) 의 거울(closed).
    # 그리퍼가 열림→닫힘으로 갈 때 align 은 줄지만 이 항은 늘어 연속 그래디언트 → "닫는
    # 행동"으로 넘어가게 한다(align hover 캠프 탈출). lift 게이트 없음.
    grasp_close_cube = RewTerm(
        func=task_mdp.grasp_close_reward,
        weight=3.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            # tolerance 확대(0.05/0.06 → 0.12/0.10): grasp_align 과 동일 이유 — arm drift 가
            # 보상창 벗어나 grasp_close=0.0005(死)이던 것 densify → 닫기 점화 gradient 부활.
            "align_xy": 0.12,
            "align_z": 0.10,
            # lift-gate OFF: 처음부터 켜니 desk-camp(책상서 잡고 안 듦) 유발(v26 succ→0). anneal 로 대체.
            "disable_when_lifted": False,
            # weight anneal 3.0→0.3 (iter 80→350, step=iter×48): per-step grasp_close 가 점화엔
            # 필수지만 점화 후엔 hold income→camp(v23 in-air camp·v26 desk camp). 점화 완료(~iter50)
            # 후 감쇠해 camp value↓(0.3×0.9/(1-γ)=27≪terminal 300) → camp-free spine(task_progress)
            # 가 carry 인수. grasp 는 이미 학습돼 low weight 서도 유지.
            "anneal_start_step": 3840.0,   # iter 80
            "anneal_end_step": 16800.0,    # iter 350
            "anneal_final_scale": 0.1,     # 3.0 → 0.3
        },
    )

    # Stage 1.8: 양 손가락이 같은 큐브에 물리 접촉(ContactSensor) — 직접 grasp 신호.
    # 기하 proxy 보다 직접적으로 "손가락 사이에 큐브가 끼었음"을 보상 → 점화 가속.
    grasp_contact_cube = RewTerm(
        func=task_mdp.grasp_contact_reward,
        weight=0.0,  # v24: 2→0 (잡고있으면 접촉=hover income. grasp 는 학습됨+align/close 유지)
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2: 그리퍼 닫힘 + 큐브 근접 (sparse bonus, 미배치 큐브 한정)
    # weight 0.5→0.2: grasp_align(열림+정밀)과 open/close 가 상충하므로 축소.
    # "닫은 채 근접" camping 유인을 최소화하고 align 이 접근 신호를 주도.
    pregrasp_cube = RewTerm(
        func=task_mdp.pregrasp_bonus,
        weight=0.0,  # v24: 0.2→0 (closed+near=hover income)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            "diff_threshold": 0.045,
        },
    )

    guided_lift_cube = RewTerm(
        func=task_mdp.guided_lift_reward,
        weight=0.0,  # v24: →0 (lifting=hover income; lift 는 학습됨+place_pbrs inside 가 instrumental 유도)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    grasp_cube = RewTerm(
        func=task_mdp.grasp_bonus,
        weight=0.0,  # v24: 1→0 (grasped+lifted+held=hover income; grasp 학습됨)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2.5: 닫힌 그리퍼 + 들린 큐브 + 그릇 방향 운반 (밀집 도우미)
    # weight 4→8: 부트스트랩 큐브를 "잡은 채 유지"하도록 강한 유인(놓치면 보상 급감).
    carry_cube = RewTerm(
        func=task_mdp.carry_object,
        weight=0.0,  # v23: 1.5→0 (hover income 제거; grasp+carry 는 학습됨, place_pbrs 가 운반 유도)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 3: 큐브를 책상에서 들어올린 높이 (밀집)
    lift_cube = RewTerm(
        func=task_mdp.lift_reward,
        weight=0.0,  # v23: 2→0 (height hover income 제거; lift 는 학습됨)
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
        },
    )

    # Stage 4: 들어올린 큐브의 XY → 그릇 접근 (밀집)
    transport_cube = RewTerm(
        func=task_mdp.transport_reward,
        weight=0.0,  # v23: 3→0 (transport hover income 제거; place_pbrs xy-progress 가 대체)
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )

    # Stage 4.5: 그릇 XY 근처에서 그릇 안 높이로 낮추기 (밀집)
    # v7: xy_range 0.18(느슨)→0.08 — 그릇 중심 정밀 정렬 유도(가장자리 hover 보상 차단).
    # place 정밀도 부족(v6 over_bowl 0.86→placed 0.10)의 핵심 레버.
    place_height_cube = RewTerm(
        func=task_mdp.place_height_reward,
        weight=0.0,  # v11: PBRS(place_pbrs)로 대체 — dense 유지 제거(hover 차단)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            "xy_range": 0.08,
            "require_carry": False,
        },
    )

    # Stage 5: 그릇 안 삽입 — 그리퍼 조건 없음 (밀집, 큐브 수 비례)
    insert_cube = RewTerm(
        func=task_mdp.insert_reward,
        weight=0.0,  # v11: PBRS(place_pbrs)의 inside 항으로 대체 — dense 유지 제거
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    place_pbrs_cube = RewTerm(
        func=task_mdp.place_pbrs_reward,
        weight=50.0,
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            "xy_range": 0.60,
            "gamma": 0.997,
        },
    )

    # 전체 task progress PBRS — reach→grasp→lift→transport→place 단조 Φ(telescoping).
    # hold income 0(grasp-camp 구조적 차단) + grasp=Φ점프(점화) + monotonic(grasp→lift dip 없음).
    # 기본 weight 0 (full/place 비활성). apply_skill_acquire 가 주 driver 로 켠다.
    # 전용 버퍼 _task_progress_potential_prev (place_pbrs 와 분리).
    task_progress_pbrs_cube = RewTerm(
        func=task_mdp.task_progress_pbrs_reward,
        weight=0.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            "reach_range": 0.20,
            "grasp_dist": 0.06,
            "close_threshold": 0.50,
            "lift_min": 0.02,
            "lift_ref": 0.10,
            "transport_range": 0.30,
            "gamma": 0.997,
        },
    )

    # Stage 6: 그릇 안 + 그리퍼 열림 완료 (밀집, 배치된 큐브 수)
    release_cube = RewTerm(
        func=task_mdp.release_bonus,
        weight=20.0,  # v23: 10→20 (drop/release 강화 — hover dense 제거와 함께 placed 견인)
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 5.5: 그릇 위에서 그리퍼 열기 유도 — release valley 메움 (밀집)
    # carry(8)+transport(8) 잡고-버티기 local optimum 탈출. inside 게이트 없이
    # '그릇 중심 + 들림 + open_frac' 에 연속 gradient → 그릇 위에서 손 펴 떨구기.
    # v7: close_ref 0.20→0.40, xy_range 0.10→0.06.
    # v13: xy_range 0.12, close_ref 0.35.
    # v14: dense(open_frac 직접) → PBRS화(over_bowl_drop_pbrs_reward). carry와 경쟁 안 함.
    #      gripper offset=0.20에서 정책이 안 열어 dense open_frac 보상이 무의미했음(rl-expert).
    #      PBRS φ: over_bowl위(0.6+0.2·open_frac+0.2·(1-z)) + 밖(0.3·xy+0.1·open_frac).
    over_bowl_drop_cube = RewTerm(
        func=task_mdp.over_bowl_drop_pbrs_reward,
        weight=16.0,  # v23: 12→16 (그릇 위 열기 PBRS 강화)
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "xy_range": 0.12,
            "open_threshold": 0.60,
            "close_ref": 0.30,  # v15: 0.35→0.30 (75% open으로 보상, release valley 추가 완화)
            "gamma": 0.997,
        },
    )

    # 그릇 교란 패널티 — 운반/place 중 그릇 밀치기/엎기 억제(tilt 주신호 + xy 변위).
    # v7: -5→-3 완화. 과한 패널티가 그릇 근처 정밀 접근을 위축시켜 place(안착)를 막는
    # 것으로 의심(v6: over_bowl 0.86인데 placed 0.10). 엎기 억제는 유지하되 접근 허용.
    bowl_disturb = RewTerm(
        func=task_mdp.bowl_disturb_penalty,
        weight=-3.0,
        params={"bowl_cfg": SceneEntityCfg(BOWL_NAME), "disp_coef": 4.0},
    )

    # 큐브 변위 패널티 — 잡기 전 큐브를 쳐서 밀어내기 억제(정밀 grasp proxy).
    # 안 들린 큐브의 초기 xy 대비 변위(m). 정밀 접근하면 ≈0, 거칠게 치면 패널티.
    # weight 중간: 큐브 ~10cm 밀면 -0.3/step(reach 1.0과 균형). grasp 시도 자체는 안 막게.
    cube_predisturb = RewTerm(
        func=task_mdp.cube_predisturb_penalty,
        weight=-3.0,
        params={"object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES], "lift_min": 0.02},
    )

    # 전체 성공 보너스 — 4개 큐브 전부 배치 완료
    task_success = RewTerm(
        func=task_mdp.task_success_bonus,
        weight=200.0,  # v10: 50→200 복원·강화 — 완료(terminal)가 value 최대가 되게(hover 차단)
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
            # PickCube termination은 "큐브가 그릇 안에 있음"과 일치한다.
            # release_cube가 gripper open을 별도로 보상한다.
            "require_open": False,
        },
    )

    # 행동률·관절 속도 페널티 — smoothness. v7: -1e-4→-1e-3(10×). v13: -1e-3→-1e-2(10×).
    # v12 실측: joint_vel raw 30.4/ep → -1e-3 페널티=-0.030, carry(3) 대비 100배 약해 무비용.
    # -1e-2로 올리면 -0.30/ep → carry 10% 비용, 진동이 경제적으로 불리해짐. sim2real 필수.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 속도 형상화 — 느린 정책 페널티 + 빠른 성공 보너스.
    # reward hacking 아님: 성공 반경/grasp 물리 불변, "시간"만 형상화한다.
    # v8: success 200→50·early_finish 100→30 압축에 맞춰 time_penalty 도 비례 축소
    #   (-0.02→-0.006). 시간 형상화(±)와 성공 보상의 상대 비율 유지 — 성공 압축 후
    #   "빨리 끝내기"가 성공보다 과해지는 것 방지. 행동 교정 페널티(bowl_disturb/
    #   cube_predisturb/smoothness)는 절대값 작고 상대 비율이 곧 교정 의도라 유지.
    time_penalty = RewTerm(
        func=task_mdp.time_penalty,
        weight=-0.02,  # v10: -0.006→-0.02 복원 — 버티기 시간 비용(hover 차단)
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )
    # task_done(전부 배치)가 곧 종료라 이 보너스는 완료 step 에 1회 지급되는
    # 터미널 보너스로 동작한다. 완료 시각에 따라 ~100(즉시)→~17(25s) 차등.
    early_finish_bonus = RewTerm(
        func=task_mdp.early_finish_bonus,
        weight=100.0,  # v10: 30→100 복원 — 빨리 완료 강제(hover 차단)
        params={
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "container_radius": BOWL_SUCCESS_RADIUS,
            "container_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Skill-1(acquire+transport) terminal 보너스 — '그릇 위 grasp' 도달 시 1회 지급.
    # 기본 weight 0 (full-task/skill2 에선 비활성). apply_skill_acquire 가 켠다.
    # 종료 조건(terminations.over_bowl_grasped)과 동일 판정이라 도달 step=terminal 보너스.
    over_bowl_grasped_bonus = RewTerm(
        func=task_mdp.over_bowl_grasped_bonus,
        weight=0.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "over_bowl_xy": 0.10,
            "lift_min": 0.02,
            "grasp_dist": 0.07,
            "close_threshold": 0.50,
        },
    )

    # -----------------------------------------------------------------------
    # 레퍼런스(ref_repos/pick_and_place, IsaacLab Lift-Cube-Place) 정합 보상항.
    # 기본 weight 0 (full/acquire/place/full_bc 비활성). apply_skill_ref 가 켠다.
    # target_region → 그릇(BOWL) 매핑, 높이는 DESK_TOP_Z 기준. 단일 객체 레시피라
    # active 큐브 합산(active_objects=1 이면 레퍼런스와 동일).
    # -----------------------------------------------------------------------
    ref_reaching = RewTerm(
        func=task_mdp.reaching_object_ref,
        weight=0.0,
        params={
            "std": 0.1,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
        },
    )
    ref_lifting = RewTerm(
        func=task_mdp.lifting_object_dist_limit_ref,
        weight=0.0,
        params={
            "minimal_height": 0.04,
            "minimal_dist": 0.05,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )
    ref_tracking = RewTerm(
        func=task_mdp.object_target_region_distance_ref,
        weight=0.0,
        params={
            "std": 0.3,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )
    ref_lowering = RewTerm(
        func=task_mdp.object_lowering_ref,
        weight=0.0,
        params={
            "std": 0.1,
            "minimal_dist": 0.05,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )


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

    # 큐브 4개를 매트 위 사각형 영역에 완전 무작위 배치 (rejection sampling).
    #   · 볼륨이 사각형 안: volume_inset(최대 50mm cube face 대각 절반)
    #   · 볼륨 비겹침: min_cube_sep=0.060 (40mm footprint 대각 절반 쌍 ≈0.057 + 여유),
    #                  min_bowl_sep=0.14 (그릇 반경0.06 + 큐브0.029 + arc 이동 0.05)
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
    # 초기상태 grasp 부트스트랩(backward curriculum) — PickCubeEnv 가 읽는다.
    # prob>0 이면 reset 시 해당 비율의 env 를 '큐브가 그리퍼에 잡힌 상태'로 시작.
    grasp_bootstrap_prob: float = 0.0
    grasp_bootstrap_close: float = -0.05   # 부트스트랩 시 gripper joint 닫힘 각(rad)
    grasp_bootstrap_lift: float = 0.0      # grasp point z 에 더할 들어올림(m)
    # annealing + graded backward curriculum (PickCubeEnv 가 읽는다)
    grasp_bootstrap_prob_final: float = 0.0   # prob 를 이 값으로 선형 감쇠(정상-env grasp 압력↑)
    grasp_bootstrap_anneal_steps: float = 0.0 # 감쇠 구간(common_step_counter 단위). 0=감쇠 없음
    grasp_bootstrap_pregrasp_open: float = 0.65  # pre-grasp 시 gripper open 각(rad). 0.90→0.65: 너무 벌린 시작이 "닫기 마무리" 학습을 방해 → 큐브 받아들일 최소폭으로
    grasp_bootstrap_rest_z: float = 0.726        # pre-grasp 큐브 책상 resting z(env-local, m)
    grasp_bootstrap_pregrasp_frac: float = -1.0  # pre-grasp 비율 오버라이드(>=0). -1=anneal p(학습 기본). 모니터용.

    # place 부트스트랩 — 큐브를 그릇 위에 든 채 시작(over_bowl→placed 하류 학습 가속)
    place_bootstrap_prob: float = 0.0   # place 부트스트랩 비율(고정, annealing 없음)
    place_bootstrap_z: float = 0.09    # 그릇 rim 위 큐브 높이 offset (env-local, m)

    # demo-state reset (RFCL reverse curriculum) — PickCubeEnv 가 읽는다.
    # SM 성공 궤적의 실제 scene 상태를 reset 분포로 주입(상태만 seed, 행동 클론 아님).
    demo_reset_prob: float = 0.0       # reset 시 데모 상태로 시작할 env 비율
    demo_dataset_dir: str | None = None  # demo_*.pt 디렉터리(pick_cube_state_machine --record_demos 산출)
    demo_anneal_steps: float = 0.0     # reverse curriculum 구간(common_step_counter). frac>=1-p sample. 0=전구간 uniform
    demo_subsample: int = 2            # 궤적 매 k step 만 적재(메모리)
    demo_max_files: int = 4000         # 적재할 demo 파일 상한

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


# ---------------------------------------------------------------------------
# 커리큘럼 적용 헬퍼 — gym.make() 이전에 env_cfg 에 in-place 적용
# ---------------------------------------------------------------------------

_CUBE_REWARD_TERMS = (
    "reach_cube",
    "grasp_align_cube",
    "grasp_close_cube",
    "grasp_contact_cube",
    "pregrasp_cube",
    "guided_lift_cube",
    "grasp_cube",
    "carry_cube",
    "lift_cube",
    "transport_cube",
    "place_height_cube",
    "insert_cube",
    "place_pbrs_cube",
    "task_progress_pbrs_cube",
    "release_cube",
    "over_bowl_drop_cube",
    "over_bowl_grasped_bonus",
    "cube_predisturb",
    "task_success",
    # 속도 보상도 활성 큐브 수에 맞춰 pen_cfgs 갱신
    "time_penalty",
    "early_finish_bonus",
)
_BOWL_RADIUS_REWARD_TERMS = (
    "reach_cube",
    "grasp_align_cube",
    "grasp_close_cube",
    "grasp_contact_cube",
    "pregrasp_cube",
    "guided_lift_cube",
    "grasp_cube",
    "carry_cube",
    "place_height_cube",
    "insert_cube",
    "place_pbrs_cube",
    "task_progress_pbrs_cube",
    "release_cube",
    "task_success",
    # 속도 보상의 cup_radius 도 동기화(반경 스케일 1.0 고정이라 사실상 no-op)
    "time_penalty",
    "early_finish_bonus",
)


def apply_curriculum(
    env_cfg: PickCubeEnvCfg,
    *,
    active_objects: int = 4,
    object_radius_scale: float = 1.0,
    container_angle_scale: float = 1.0,
    container_radius_scale: float = 1.0,
) -> None:
    """PickCube curriculum을 env_cfg에 in-place 적용.

    활성 큐브 수, reset 랜덤화 범위, bowl 성공 반경만 조정한다.
    """

    active_objects = max(1, min(4, active_objects))
    active_names = CUBE_NAMES[:active_objects]
    active_cfgs = [SceneEntityCfg(n) for n in active_names]
    bowl_radius = BOWL_SUCCESS_RADIUS * max(0.1, container_radius_scale)

    for term_name in _CUBE_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["object_cfgs"] = active_cfgs
    for term_name in _BOWL_RADIUS_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["container_radius"] = bowl_radius

    env_cfg.terminations.success.params["objects_cfg"] = active_cfgs
    env_cfg.terminations.success.params["radius"] = bowl_radius
    # 큐브 추락 종료도 활성 큐브만 검사(비활성 큐브는 지면 아래라 오탐 방지)
    cube_lost_term = getattr(env_cfg.terminations, "cube_lost", None)
    if cube_lost_term is not None:
        cube_lost_term.params["objects_cfg"] = active_cfgs

    # rl_state 관측의 비활성 큐브 마스킹(distractor 제거 + RND novelty 집중)
    rl_obs = getattr(env_cfg.observations.rl_policy, "rl_state_obs", None)
    if rl_obs is not None:
        rl_obs.params["num_active"] = active_objects

    # 비활성 큐브 물리 비활성화(지면 아래로) — scatter event 에 active 수 전달
    scatter = getattr(env_cfg.events, "randomize_cubes", None)
    if scatter is not None:
        scatter.params["num_active"] = active_objects

    # 큐브 scatter workspace 를 scale 에 비례해 중심으로부터 확장/축소.
    # scale=0: DR 비활성화 → authored default fixed-spawn.
    # scale=1: 전체 workspace 사용.
    scatter_term = getattr(env_cfg.events, "randomize_cubes", None)
    if scatter_term is not None and object_radius_scale <= 0.0:
        env_cfg.events.randomize_cubes = None
    elif scatter_term is not None and object_radius_scale != 1.0:
        cx, cy = _CUBE_SCATTER_CENTER
        x_lo_full, x_hi_full = _CUBE_SCATTER_X_RANGE
        y_lo_full, y_hi_full = _CUBE_SCATTER_Y_RANGE
        s = max(0.0, float(object_radius_scale))
        scatter_term.params["x_range"] = (
            cx - (cx - x_lo_full) * s,
            cx + (x_hi_full - cx) * s,
        )
        scatter_term.params["y_range"] = (
            cy - (cy - y_lo_full) * s,
            cy + (y_hi_full - cy) * s,
        )

    bowl_term = getattr(env_cfg.events, "randomize_bowl", None)
    if bowl_term is not None and container_angle_scale != 1.0:
        lo, hi = bowl_term.params["angle_range_deg"]
        bowl_term.params["angle_range_deg"] = (lo * container_angle_scale, hi * container_angle_scale)

    for cube_name in CUBE_NAMES:
        obs_name = "place_" + cube_name.lower()
        term = getattr(env_cfg.observations.subtask_terms, obs_name, None)
        if term is not None:
            term.params["radius"] = bowl_radius


# ---------------------------------------------------------------------------
# Skill chaining 프리셋 — apply_curriculum 이후 호출 (acquire | place)
# ---------------------------------------------------------------------------


def _active_cfgs_from(env_cfg: "PickCubeEnvCfg") -> list[SceneEntityCfg]:
    """apply_curriculum 이 success 종료에 주입한 활성 큐브 cfg 목록을 회수."""
    cfgs = env_cfg.terminations.success.params.get("objects_cfg")
    if cfgs:
        return cfgs
    return [SceneEntityCfg(n) for n in CUBE_NAMES]


def _set_reward_weights(env_cfg: "PickCubeEnvCfg", weights: dict) -> None:
    for name, w in weights.items():
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.weight = float(w)


def apply_skill_acquire(env_cfg: "PickCubeEnvCfg", *, episode_length_s: float = 15.0) -> None:
    """Skill-1(acquire+transport) 프리셋 — scratch 단일 run 으로 '그릇 위 grasp' 도달.

    **고정 config(재현용 source of truth).** over_bowl_grasped 종료로 끊어 skill2 에 handoff.
    **apply_curriculum 이후** 호출. scratch + sustained grasp_bootstrap 으로 한 번에.

    **scratch5 = 순수 camp-free + 그리퍼 open init** — 7회 실패 종합 결론:
    γ=0.997 에서 per-step *상태* 보상(reach/align/close/lift/carry)은 무엇이든 hold 가치를
    income×333 로 만들어 terminal 을 압도 → 그 상태서 camp(v1/scratch1/2). hold income 을
    조금이라도 남기면 camp, 다 빼면(scratch3/4) grasp shaping 소실로 ram(cube_lost 25~36%).
    → camp-free 한 것만 사용: **telescoping PBRS·terminal·penalty.**
    - **task_progress_pbrs 80(유일 driver)**: 접근(reach_prog)→grasp(Φ점프=점화)→lift→transport
      단조 Φ. telescoping 이라 hold=0(camp 구조적 불가), 진행 시에만 +.
    - **그리퍼 init 0.70 OPEN**(scratch1~4 는 0.20 닫힘 → 닫힌 채 접근=ram + grasp_align 死):
      열린 그리퍼로 부드럽게 접근→큐브 위서 닫기가 자연스러운 grasp = ram·cube_lost 동시 완화.
    - per-step 상태 보상 **전부 0**(reach/align/close 포함 — 어떤 것도 hover-camp 유발).
    - cube_predisturb −5(anti-ram), over_bowl_grasped_bonus 250(terminal).
    - 점화: sustained grasp_bootstrap(0.6, 항상 시연 → v20 PBRS-단독 blowup 회피) + RND grasp_focus
      + task_progress grasp Φ점프. (pure-telescoping 이 scratch grasp 점화 가능한지의 시험.)
    이력: v1/scratch1/2 camp(hold income)·v2/v3 blowup(resume+std reset)·scratch3/4 ram(hold
    income 0 인데 grasp shaping 도 같이 소실+닫힌 그리퍼) → scratch5 = pure PBRS + open gripper.
    """
    active_cfgs = _active_cfgs_from(env_cfg)
    env_cfg.terminations.success = DoneTerm(
        func=task_mdp.over_bowl_grasped,
        params={
            "objects_cfg": active_cfgs,
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "over_bowl_xy": 0.10,
            "lift_min": 0.02,
            "grasp_dist": 0.07,
            "close_threshold": 0.50,
        },
    )
    # 그리퍼 init OPEN(0.70) — scratch1~4 의 0.20(닫힘)이 ram + grasp_align 死의 원인.
    # 열린 그리퍼로 접근→닫기가 자연스러운 grasp(scratch5 핵심 레버). bootstrap envs 는
    # 자체적으로 gripper 를 덮어쓰므로(full=닫힘/pre=열림) 비-bootstrap scratch env 에만 적용.
    try:
        env_cfg.scene.robot.init_state.joint_pos["gripper"] = 0.70
    except Exception:
        pass
    _set_reward_weights(env_cfg, {
        # === scratch5: 순수 camp-free(telescoping PBRS·terminal·penalty 만) ===
        # per-step 상태 보상은 무엇이든 camp(γ=0.997 → hold 가치 income×333 ≫ terminal). 전부 0.
        "reach_cube": 0.0,
        "grasp_align_cube": 0.0,
        "grasp_close_cube": 0.0,
        # task_progress_pbrs = 유일 driver(80). 접근→grasp(Φ점프=점화)→lift→transport 단조 Φ.
        # telescoping → hold=0(camp 구조적 불가). 80 으로 강화(접근/grasp 신호 충분히 강하게).
        "task_progress_pbrs_cube": 80.0,
        "over_bowl_grasped_bonus": 250.0,   # terminal
        "cube_predisturb": -5.0,            # anti-ram (open gripper 와 함께 cube_lost 억제)
        # per-step 상태 보상 전부 off (camp 원천 차단)
        "grasp_contact_cube": 0.0,
        "pregrasp_cube": 0.0,
        "guided_lift_cube": 0.0,
        "grasp_cube": 0.0,
        "carry_cube": 0.0,
        "lift_cube": 0.0,
        "transport_cube": 0.0,
        "place_height_cube": 0.0,
        "insert_cube": 0.0,
        "place_pbrs_cube": 0.0,
        "release_cube": 0.0,
        "over_bowl_drop_cube": 0.0,
        "task_success": 0.0,
        "early_finish_bonus": 0.0,
    })
    env_cfg.rewards.over_bowl_grasped_bonus.params["object_cfgs"] = active_cfgs
    env_cfg.rewards.task_progress_pbrs_cube.params["object_cfgs"] = active_cfgs
    env_cfg.episode_length_s = float(episode_length_s)


def apply_skill_full_bc(env_cfg: "PickCubeEnvCfg", *, episode_length_s: float = 20.0) -> None:
    """Full pick-place 프리셋 — BC warmstart 정책의 RL finetune 용 (camp-free, 단일 end-to-end).

    전략 전환(2026-06-12): scratch reward-shaping(8회)·demo-reset-only(v16~20) 실패의 공통
    누락 = expert ACTION 미주입. 검증된 SM(해석적 IK side-approach, 1-cube ~90%) 전궤적을
    BC clone(obs→action) → 이 프리셋으로 RL finetune. BC init 이 reach→grasp→lift→transport
    →release 전체를 이미 수행 → terminal 즉시 도달로 credit assignment 해결(grasp 점화·hover
    동시 우회). reverse curriculum(demo_reset_prob, train.py CLI)로 grasped/transport 상태
    분포 유지 → erosion 방지(RFCL/IndustReal 정석, sim2real 83~99%).

    보상 = grasp 점화 dense(v4 레버) + camp-free spine. **핵심: γ=0.99** 에서 dense grasp 가
    camp-free — camp value = w/(1−γ) = grasp_close 3.0×100 = 300 < terminal ~350 → 완료가 hold
    압도(γ0.997 이면 ×333=1000 ≫ terminal 이라 camp = 8회 실패 원인). 즉 γ 가 camp/점화 분기.
    - grasp_align 1.0 + grasp_close 3.0: scratch grasp 점화(v4 가 유일하게 점화시킨 dense 레버).
    - task_progress_pbrs 80: full-task 단조 Φ(reach→grasp Φ점프→lift→transport→inside) telescoping.
    - over_bowl_drop_pbrs 16 + release 20: 그릇 위 열기·release 강조(PBRS+1회 bonus, camp-free).
    - task_success 200(require_open=True → 그릇에 든 채 success 금지, VLA clean release) + early_finish 100.
    - cube_predisturb −5 + bowl_disturb −3 + action_rate/joint_vel −1e-2(base 유지): 교정 페널티.
    **apply_curriculum 이후** 호출. terminations.success 는 base(full-place) 그대로. 그리퍼 init
    0.70 OPEN(demo/bootstrap env 는 자체 덮어씀).
    """
    active_cfgs = _active_cfgs_from(env_cfg)
    try:
        env_cfg.scene.robot.init_state.joint_pos["gripper"] = 0.70
    except Exception:
        pass
    _set_reward_weights(env_cfg, {
        # camp-free spine — full-task telescoping PBRS (BC 가 grasp 제공 → per-step 상태보상 불요)
        "task_progress_pbrs_cube": 80.0,
        # release/drop 강조 (camp-free: PBRS + 1회 bonus)
        "over_bowl_drop_cube": 16.0,
        "release_cube": 20.0,
        # terminal
        "task_success": 200.0,
        "early_finish_bonus": 100.0,
        # grasp 점화 레버(v4 가 유일하게 scratch grasp 점화) — γ=0.99 에서 dense 가 camp-free
        # (camp value = w/(1−γ) = 3.0×100 = 300 < terminal ~350 → 완료가 hold 압도). BC reach 0.32
        # 가 접근 head start, grasp_close 가 닫기 점화. reach 는 BC+task_progress 담당(per-step off).
        "reach_cube": 0.0,
        "grasp_align_cube": 1.0,   # 열린 그리퍼 정렬(pre-grasp 접근 유도)
        "grasp_close_cube": 3.0,   # 정렬된 채 닫기 dense → grasp 점화(v4 weight)
        "grasp_contact_cube": 0.0, # ContactSensor 필요 — align/close 로 충분
        "pregrasp_cube": 0.0,
        "guided_lift_cube": 0.0,
        "grasp_cube": 0.0,
        "carry_cube": 0.0,
        "lift_cube": 0.0,
        "transport_cube": 0.0,
        "place_height_cube": 0.0,
        "insert_cube": 0.0,
        "place_pbrs_cube": 0.0,             # task_progress_pbrs 가 대체(중복 progress 방지)
        "over_bowl_grasped_bonus": 0.0,
    })
    # require_open=True — 그릇에 든 채 success 금지(VLA release 품질).
    try:
        env_cfg.rewards.task_success.params["require_open"] = True
    except Exception:
        pass
    # active subset 만 보상 계산(apply_curriculum 이후)
    for _name in ("task_progress_pbrs_cube", "over_bowl_drop_cube", "release_cube",
                  "task_success", "early_finish_bonus", "grasp_align_cube", "grasp_close_cube"):
        _term = getattr(env_cfg.rewards, _name, None)
        if _term is not None and "object_cfgs" in _term.params:
            _term.params["object_cfgs"] = active_cfgs
    env_cfg.episode_length_s = float(episode_length_s)


def apply_skill_ref(env_cfg: "PickCubeEnvCfg", *, episode_length_s: float = 5.0) -> None:
    """레퍼런스(ref_repos/pick_and_place, IsaacLab Lift-Cube-Place) 정합 프리셋.

    성공이 확인된 레퍼런스의 보상·종료 구조를 SO-101+그릇 환경에 그대로 재현한다.
    우리 dense shaping(grasp_align/close·task_progress·PBRS·bootstrap·RND·terminal) 전부 끄고
    레퍼런스 6항만 사용:
      ref_reaching 1.0 · ref_lifting 30 · ref_tracking 16 · ref_lowering 7
      + action_rate −1e-4 · joint_vel −1e-4 (레퍼런스와 동일 함수·weight).
    종료도 레퍼런스와 동일: **success 종료 없음**(time_out + cube_lost(≈object_dropping)만).
    success 로 조기 종료하면 tracking/lowering 보상이 잘려 레퍼런스 MDP 와 달라지므로 끈다.
    그리퍼 init 은 우리 OPEN(0.70)로 둔다(레퍼런스 hand joint 0.0=open 과 의도 동일,
    SO-101 부호 규약이 달라 0.0 은 near-closed 라 grasp 불가). episode 5s.
    **apply_curriculum 이후** 호출. arch/gamma 는 run_expert_policy.sh ref 스테이지가 맞춘다
    (MLP[128,64,32]+obs_normalization, γ0.98, init_noise_std 1.0, entropy 0.006, lr 8e-5).
    """
    active_cfgs = _active_cfgs_from(env_cfg)
    try:
        env_cfg.scene.robot.init_state.joint_pos["gripper"] = 0.70
    except Exception:
        pass
    _set_reward_weights(env_cfg, {
        # 레퍼런스 6항 (target_region→그릇 매핑)
        "ref_reaching": 1.0,
        "ref_lifting": 30.0,
        "ref_tracking": 16.0,
        "ref_lowering": 7.0,
        "action_rate": -1e-4,
        "joint_vel": -1e-4,
        # 우리 dense/shaping/PBRS/terminal/penalty 전부 off
        "reach_cube": 0.0, "grasp_align_cube": 0.0, "grasp_close_cube": 0.0,
        "grasp_contact_cube": 0.0, "pregrasp_cube": 0.0, "guided_lift_cube": 0.0,
        "grasp_cube": 0.0, "carry_cube": 0.0, "lift_cube": 0.0, "transport_cube": 0.0,
        "place_height_cube": 0.0, "insert_cube": 0.0, "place_pbrs_cube": 0.0,
        "task_progress_pbrs_cube": 0.0, "release_cube": 0.0, "over_bowl_drop_cube": 0.0,
        "over_bowl_grasped_bonus": 0.0, "task_success": 0.0, "early_finish_bonus": 0.0,
        "bowl_disturb": 0.0, "cube_predisturb": 0.0, "time_penalty": 0.0,
    })
    # active subset 만 보상 계산(apply_curriculum 이후)
    for _name in ("ref_reaching", "ref_lifting", "ref_tracking", "ref_lowering"):
        _term = getattr(env_cfg.rewards, _name, None)
        if _term is not None and "object_cfgs" in _term.params:
            _term.params["object_cfgs"] = active_cfgs
    # 레퍼런스: success 종료 없음 — 5s 풀 에피소드(tracking/lowering 누적). cube_lost(추락) 유지.
    try:
        env_cfg.terminations.success = None
    except Exception:
        pass
    env_cfg.episode_length_s = float(episode_length_s)


def apply_skill_place(env_cfg: "PickCubeEnvCfg", *, episode_length_s: float = 5.0) -> None:
    """Skill-2(place+release) 프리셋 — over-bowl-grasped init 에서 lower+open 만.

    grasp_close/align(hold income) 제거가 핵심 — 이게 v24 hover 의 뿌리였다. require_open
    종료로 release 강제. 단기 horizon(기본 5s) → terminal 이 압도(hover 누적 불가).
    demo_reset(skill1 수집 상태)은 train.py CLI 로 주입. **apply_curriculum 이후** 호출.
    """
    active_cfgs = _active_cfgs_from(env_cfg)
    bowl_radius = env_cfg.terminations.success.params.get("radius", BOWL_SUCCESS_RADIUS)
    env_cfg.terminations.success = DoneTerm(
        func=task_mdp.cube_placed_open,
        params={
            "objects_cfg": active_cfgs,
            "robot_cfg": SceneEntityCfg("robot"),
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
            "radius": bowl_radius,
            "height_range": BOWL_HEIGHT_RANGE,
            "open_threshold": 0.60,
        },
    )
    _set_reward_weights(env_cfg, {
        # acquire/hold income 전부 제거 (grasp_close 가 hover 의 뿌리)
        "reach_cube": 0.0,
        "grasp_align_cube": 0.0,
        "grasp_close_cube": 0.0,
        "grasp_contact_cube": 0.0,
        "pregrasp_cube": 0.0,
        "guided_lift_cube": 0.0,
        "grasp_cube": 0.0,
        "carry_cube": 0.0,
        "lift_cube": 0.0,
        "transport_cube": 0.0,
        "place_height_cube": 0.0,
        "insert_cube": 0.0,
        "over_bowl_grasped_bonus": 0.0,
        # place 진행·드롭·release·완료
        "place_pbrs_cube": 50.0,
        "over_bowl_drop_cube": 24.0,
        "release_cube": 30.0,
        "task_success": 200.0,
        # early_finish 는 'placed-but-closed' farming 위험 → 끈다(단기 horizon+task_success 로 충분)
        "early_finish_bonus": 0.0,
    })
    env_cfg.rewards.task_success.params["require_open"] = True
    env_cfg.episode_length_s = float(episode_length_s)
