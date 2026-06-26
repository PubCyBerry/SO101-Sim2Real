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
from isaaclab.sensors import TiledCameraCfg
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
    """Observations: policy (6-dim joint pos, North Star/VLA 계약) + ref_policy (RL 학습용 54-dim).

    Groups:
      policy     — 6-dim joint pos (North Star contract, VLA rollout 용, immutable).
      ref_policy — 레퍼런스 정합 저차원 상태 (54-dim). RL actor/critic 입력.
                   train.py 의 obs_groups 로 policy/critic 둘 다 ref_policy 에 매핑.
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
    class RefPolicyCfg(ObsGroup):
        """ref(ref_repos/pick_and_place) 정합 저차원 상태 (54-dim).

        joint_pos6 + joint_vel6 + TCP 6d pose/vel(12) + cube 6d pose/vel(12) +
        bowl 6d pose/vel(12) + last_action6. 단일 40mm 큐브(CUBE_NAMES[0]) 전용.
        DR L0(완전고정)에선 obs 노이즈 없음(enable_corruption=False) —
        apply_dr_level(level>=2)가 Gaussian 노이즈를 켠다.
        """

        ref_state_obs = ObsTerm(
            func=task_mdp.ref_state,
            params={
                "cube_name": CUBE_NAMES[0],
                "container_name": BOWL_NAME,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    ref_policy: RefPolicyCfg = RefPolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — Phase B 단계형 보상
# ---------------------------------------------------------------------------


@configclass
class PickCubeRewardsCfg:
    """레퍼런스(ref_repos/pick_and_place, IsaacLab Lift-Cube-Place) 정합 dense 보상.

    reaching 1 · lifting 30 · tracking 16 · lowering 7 (target_region → 그릇 BOWL 매핑)
    + action_rate/joint_vel −1e-4 (레퍼런스와 동일 함수·weight). active 큐브(apply_curriculum
    이 object_cfgs 주입) 합산 — active_objects=1 이면 레퍼런스와 수치 동일.
    """

    # EE → 큐브 접근 (tanh)
    ref_reaching = RewTerm(
        func=task_mdp.reaching_object_ref,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
        },
    )
    # 그릇에서 멀고 들렸을 때 높이 비례 보상 (암묵적 grasp 신호, camp-free)
    ref_lifting = RewTerm(
        func=task_mdp.lifting_object_dist_limit_ref,
        weight=30.0,
        params={
            "minimal_height": 0.04,
            "minimal_dist": 0.05,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )
    # 큐브 → 그릇 안(3D center) 접근 (tanh) — "그릇에 넣기" 유도
    ref_tracking = RewTerm(
        func=task_mdp.object_target_region_distance_ref,
        weight=16.0,
        params={
            "std": 0.3,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )
    # 그릇 근처에서 큐브가 내려가는 동안만 (delta 보상 → camp-free)
    ref_lowering = RewTerm(
        func=task_mdp.object_lowering_ref,
        weight=7.0,
        params={
            "std": 0.1,
            "minimal_dist": 0.05,
            "object_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "container_center_xy": BOWL_CENTER_XY,
            "container_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )

    # smoothness 페널티 (레퍼런스와 동일 함수·weight)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
    #   · 볼륨이 사각형 안: volume_inset(40mm face 대각 절반)
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
    # leisaac 동적 gripper effort 배선(PickCubeEnv.step 이 읽는다). soft-PD SO-101 grasp 안정.
    dynamic_reset_gripper_effort_limit: bool = True

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

# ref dense 보상 4항 — active 큐브 subset 을 object_cfgs 로 받는다.
_CUBE_REWARD_TERMS = (
    "ref_reaching",
    "ref_lifting",
    "ref_tracking",
    "ref_lowering",
)


def apply_curriculum(
    env_cfg: PickCubeEnvCfg,
    *,
    active_objects: int = 1,
    object_radius_scale: float = 1.0,
    container_angle_scale: float = 1.0,
    container_radius_scale: float = 1.0,
) -> None:
    """PickCube curriculum을 env_cfg에 in-place 적용.

    활성 큐브 수, reset 랜덤화 범위, bowl 성공 반경만 조정한다. ref 보상은 단일 큐브
    레시피라 active subset(보통 1)만 합산한다.
    """

    active_objects = max(1, min(4, active_objects))
    active_names = CUBE_NAMES[:active_objects]
    active_cfgs = [SceneEntityCfg(n) for n in active_names]
    bowl_radius = BOWL_SUCCESS_RADIUS * max(0.1, container_radius_scale)

    for term_name in _CUBE_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["object_cfgs"] = active_cfgs

    # success(task_done) 종료 — active 큐브 + 성공 반경
    env_cfg.terminations.success.params["objects_cfg"] = active_cfgs
    env_cfg.terminations.success.params["radius"] = bowl_radius
    # 큐브 추락 종료도 활성 큐브만 검사(비활성 큐브는 지면 아래라 오탐 방지)
    cube_lost_term = getattr(env_cfg.terminations, "cube_lost", None)
    if cube_lost_term is not None:
        cube_lost_term.params["objects_cfg"] = active_cfgs

    # 비활성 큐브 물리 비활성화(지면 아래로) — scatter event 에 active 수 전달
    scatter_term = getattr(env_cfg.events, "randomize_cubes", None)
    if scatter_term is not None:
        scatter_term.params["num_active"] = active_objects

    # 큐브 scatter workspace 를 scale 에 비례해 중심으로부터 확장/축소.
    # scale=0: DR 비활성화 → authored default fixed-spawn. scale=1: 전체 workspace.
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


# ---------------------------------------------------------------------------
# DR(domain randomization) 점진 사다리 — apply_curriculum 이후 호출
# ---------------------------------------------------------------------------


def apply_dr_level(env_cfg: "PickCubeEnvCfg", level: int, *, active_objects: int = 1) -> None:
    """DR 사다리를 env_cfg.events / obs 에 in-place 적용. **apply_curriculum 이후** 호출.

    레퍼런스 정합(ref) 학습은 처음엔 DR 없이 시작해 단계적으로 추가한다. 레벨별 누적:

    | level | 추가되는 것 |
    |---|---|
    | 0 (완전고정) | 큐브 1개 고정 위치/자세(비활성 큐브 sink)·그릇 고정·jitter 0·물리/시각/obs noise off |
    | 1 (+spawn)   | 큐브 scatter + 그릇 arc 랜덤화 on |
    | 2 (+sensor)  | robot joint jitter(±0.05) + obs Gaussian noise(σ0.005) on |
    | 3 (+물리/시각)| 큐브 mass/friction + light/focal 랜덤화 on (full sim2real) |

    L0 은 authored randomize_cubes 를 degenerate range(폭 0) 변형으로 교체해 Cube1 을 고정
    위치·자세로 spawn 하면서 비활성 큐브는 num_active 로 지면 아래 sink 한다. L1+ 는 authored
    randomize_cubes(full scatter)/randomize_bowl(arc) 를 그대로 둔다.
    """
    ev = env_cfg.events
    level = int(level)
    spawn_random = level >= 1
    sensor_dr = level >= 2
    physics_visual = level >= 3

    # --- 큐브/그릇 배치 ---
    if not spawn_random:
        # L0: Cube1 고정 위치/자세. degenerate range(폭 0)+full_orient off → 결정적.
        # 비활성 큐브는 num_active 로 지면 아래 sink(_randomize_cubes_scattered_fn).
        px, py = _CUBE_INIT_STATES[CUBE_NAMES[0]][0][:2]
        env_cfg.events.randomize_cubes = randomize_cubes_scattered(
            CUBE_NAMES,
            BOWL_NAME,
            x_range=(px, px),
            y_range=(py, py),
            yaw_range_deg=(0.0, 0.0),
            full_orient=False,
            volume_inset=0.0,
            min_cube_sep=0.0,
            min_bowl_sep=0.0,
            min_base_sep=0.0,
            num_active=active_objects,
        )
        env_cfg.events.randomize_bowl = None
    # L1+: authored randomize_cubes(full scatter) + randomize_bowl(arc) 유지.

    # --- robot joint jitter (sensor DR) ---
    if getattr(ev, "reset_robot_joints", None) is not None:
        ev.reset_robot_joints.params["position_range"] = (-0.05, 0.05) if sensor_dr else (0.0, 0.0)

    # --- 물리(mass/friction) + 시각(light/focal) DR ---
    if not physics_visual:
        if hasattr(ev, "randomize_lights"):
            env_cfg.events.randomize_lights = None
        if hasattr(ev, "randomize_camera_focal"):
            env_cfg.events.randomize_camera_focal = None
        for name in CUBE_NAMES:
            for suffix in ("material", "mass"):
                attr = f"randomize_{name.lower()}_{suffix}"
                if hasattr(ev, attr):
                    setattr(env_cfg.events, attr, None)

    # --- obs Gaussian noise (sensor DR) ---
    ref_grp = getattr(env_cfg.observations, "ref_policy", None)
    if ref_grp is not None:
        ref_term = getattr(ref_grp, "ref_state_obs", None)
        if sensor_dr:
            ref_grp.enable_corruption = True
            if ref_term is not None:
                ref_term.noise = GaussianNoiseCfg(mean=0.0, std=0.005)
        else:
            ref_grp.enable_corruption = False
            if ref_term is not None:
                ref_term.noise = None
