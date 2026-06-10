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
    randomize_cubes_scattered,
    randomize_object_mass,
    randomize_object_material,
    randomize_object_on_arc,
)

from sim_to_real.tasks.pick_pen import mdp as task_mdp


# World-frame (x, y) of the bowl at scene authoring time.
# BOWL_LOCAL=(-0.58, 0.26) + SCENE_OFFSET=(2.2, -0.52) = (1.62, -0.26)
BOWL_CENTER_XY: tuple[float, float] = (1.62, -0.26)
BOWL_SUCCESS_RADIUS: float = 0.06
BOWL_HEIGHT_RANGE: tuple[float, float] = (0.005, 0.12)

# Robot base position.
# x: desk_left_edge(1.40) + 440mm = 1.84
# y: -0.565 (책상 앞 모서리 기준 10mm 뒤로 장착)
# z: desk_top(0.705) - base_min_z(0.0301) = 0.6749
_ROBOT_POS = (1.84, -0.565, 0.6749)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


# 큐브 world 좌표 = SCENE_OFFSET(2.2, -0.52, 0.705) + scene-local 위치.
# 매트 윗면 world z=0.709. z중심 = 0.709 + 반높이 + slack(0.001).
#   작은(Cube1/2, 30mm): 0.709+0.015+0.001=0.725, 큰(Cube3/4, 40mm): 0.709+0.020+0.001=0.730.
_CUBE_INIT_STATES = {
    "Cube1": ((1.70, -0.44, 0.725), _yaw_quat(20.0)),
    "Cube2": ((1.98, -0.46, 0.725), _yaw_quat(-35.0)),
    "Cube3": ((1.74, -0.35, 0.730), _yaw_quat(50.0)),
    "Cube4": ((1.93, -0.38, 0.730), _yaw_quat(-20.0)),
}
# BOWL_LOCAL(-0.58, 0.26, 0.010) + SCENE_OFFSET(2.2, -0.52, 0.705) = (1.62, -0.26, 0.715)
_BOWL_INIT_STATE = ((1.62, -0.26, 0.715), _yaw_quat(0.0))

# ---------------------------------------------------------------------------
# 큐브 scatter workspace — randomize_cubes_scattered 기본값 및 커리큘럼 계산 기준
# ---------------------------------------------------------------------------

# 로봇 도달 범위 안쪽, 매트 경계(x:[1.50,2.36], y:[-0.52,-0.12])에서 마진 확보.
# y 상한 -0.33: 그릇 기본 위치(y=-0.26)와의 충분한 이격 확보
#   (그릇-큐브 최소 거리 0.18m는 randomize_cubes_scattered 의 min_bowl_sep 로 추가 보장).
_CUBE_SCATTER_X_RANGE: tuple[float, float] = (1.60, 2.08)
# y_lo = mat_back_edge(-0.52) + 50mm 여유 = -0.47 (로봇팔 도달이 어려운 매트 아래쪽 가장자리 회피)
_CUBE_SCATTER_Y_RANGE: tuple[float, float] = (-0.47, -0.33)

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
_TOP_CAMERA_POS = (1.87, -0.58, 1.72)
# _TOP_CAMERA_ROT 가 None 이 아니면 이 quat 을 직접 쓰고, None 이면 _TOP_CAMERA_TARGET
# 으로 look_at 을 계산한다(하위호환).
_TOP_CAMERA_ROT = (0.5716, -0.4238, 0.4466, 0.5424)
_TOP_CAMERA_TARGET = (2.14, -0.15, 0.76)
_TOP_CAMERA_FOCAL = 23.0

# wrist: gripper 위/옆에 강결합된 카메라.
_WRIST_CAM_LOCAL_POS = (0.0, 0.045, -0.04)
_WRIST_CAM_LOCAL_ROT = (-0.3642, 0.6061, -0.6061, -0.3642)
_WRIST_CAMERA_FOCAL = 23.0

# front: 책상 정면에서 작업공간을 바라보는 카메라.
# 기본값은 --tune_cameras 로 튜닝 후 이 상수에 업데이트할 것.
_FRONT_CAMERA_POS = (1.87, 0.65, 1.10)
_FRONT_CAMERA_TARGET = (2.14, -0.15, 0.80)
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
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

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
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
    front_focal: float | None = None,
) -> dict[str, TiledCameraCfg]:
    """top/wrist/front 카메라 cfg 3개를 반환.

    기본 PickCubeSceneCfg 밖에 두어 기본 env 가 --enable_cameras 없이 돌게 한다.
    gym.make() 전에 add_pick_cube_cameras() 로 scene cfg 에 주입해서 쓴다.
    각 카메라는 480×640 RGB.

    top 회전: ``top_target`` 이 주어지면 look_at 계산, 없으면 튜닝된 기본 ``_TOP_CAMERA_ROT``.
    """

    top_pos = _TOP_CAMERA_POS if top_pos is None else top_pos
    top_focal = _TOP_CAMERA_FOCAL if top_focal is None else top_focal
    wrist_local_pos = _WRIST_CAM_LOCAL_POS if wrist_local_pos is None else wrist_local_pos
    wrist_local_rot = _WRIST_CAM_LOCAL_ROT if wrist_local_rot is None else wrist_local_rot
    wrist_focal = _WRIST_CAMERA_FOCAL if wrist_focal is None else wrist_focal
    front_pos = _FRONT_CAMERA_POS if front_pos is None else front_pos
    front_target = _FRONT_CAMERA_TARGET if front_target is None else front_target
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
    # wrist: robot 링크의 자식 prim → gripper 회전을 따라 이동/회전한다.
    # pos/rot 은 gripper local frame 기준. 정확한 화각은 GUI 렌더로 튜닝한다.
    wrist = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        wrist_local_pos,
        wrist_local_rot,
        wrist_focal,
        focus_distance=0.2,
        clipping_range=(0.02, 3.0),
    )
    front = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/FrontCamera",
        front_pos,
        _look_at_quat_world(front_pos, front_target),
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
    front_pos: tuple[float, float, float] | None = None,
    front_target: tuple[float, float, float] | None = None,
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
        front_pos=front_pos,
        front_target=front_target,
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
            func=task_mdp.pen_in_cup,
            params={
                "object_cfg": SceneEntityCfg("Cube1"),
                "cup_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )
        place_cube2 = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={
                "object_cfg": SceneEntityCfg("Cube2"),
                "cup_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )
        place_cube3 = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={
                "object_cfg": SceneEntityCfg("Cube3"),
                "cup_center_xy": BOWL_CENTER_XY,
                "radius": BOWL_SUCCESS_RADIUS,
                "height_range": BOWL_HEIGHT_RANGE,
            },
        )
        place_cube4 = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={
                "object_cfg": SceneEntityCfg("Cube4"),
                "cup_center_xy": BOWL_CENTER_XY,
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
                "pen_names": CUBE_NAMES,
                "cup_name": BOWL_NAME,
                "include_velocities": True,  # joint_vel+ee vel+cube vel 추가(부분관측 해소) → 43→64dim
                "include_orientation": True,  # cube yaw+half-extent+ee quat+grasp→cup → 64→83dim
                "include_container_orientation": True,  # 그릇 quat → 83→87dim(동적 그릇 tilt/엎힘 관측)
                # 큐브 크기(half-extent, m): Cube1/2=30mm→0.015, Cube3/4=40mm→0.020.
                # 평행 jaw 벌림 폭 매칭에 필수(크기 2종). CUBE_NAMES 순서와 일치.
                "pen_half_extents": (0.015, 0.015, 0.020, 0.020),
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
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
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
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
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
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 1.8: 양 손가락이 같은 큐브에 물리 접촉(ContactSensor) — 직접 grasp 신호.
    # 기하 proxy 보다 직접적으로 "손가락 사이에 큐브가 끼었음"을 보상 → 점화 가속.
    grasp_contact_cube = RewTerm(
        func=task_mdp.grasp_contact_reward,
        weight=2.0,
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2: 그리퍼 닫힘 + 큐브 근접 (sparse bonus, 미배치 큐브 한정)
    # weight 0.5→0.2: grasp_align(열림+정밀)과 open/close 가 상충하므로 축소.
    # "닫은 채 근접" camping 유인을 최소화하고 align 이 접근 신호를 주도.
    pregrasp_cube = RewTerm(
        func=task_mdp.pregrasp_bonus,
        weight=0.2,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
            "diff_threshold": 0.045,
        },
    )

    guided_lift_cube = RewTerm(
        func=task_mdp.guided_lift_reward,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    grasp_cube = RewTerm(
        func=task_mdp.grasp_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2.5: 닫힌 그리퍼 + 들린 큐브 + 그릇 방향 운반 (밀집 도우미)
    # weight 4→8: 부트스트랩 큐브를 "잡은 채 유지"하도록 강한 유인(놓치면 보상 급감).
    carry_cube = RewTerm(
        func=task_mdp.carry_pen,
        weight=8.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 3: 큐브를 책상에서 들어올린 높이 (밀집)
    lift_cube = RewTerm(
        func=task_mdp.lift_reward,
        weight=2.0,
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
        },
    )

    # Stage 4: 들어올린 큐브의 XY → 그릇 접근 (밀집)
    transport_cube = RewTerm(
        func=task_mdp.transport_reward,
        weight=8.0,
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
        },
    )

    # Stage 4.5: 그릇 XY 근처에서 그릇 안 높이로 낮추기 (밀집)
    # v7: xy_range 0.18(느슨)→0.08 — 그릇 중심 정밀 정렬 유도(가장자리 hover 보상 차단).
    # place 정밀도 부족(v6 over_bowl 0.86→placed 0.10)의 핵심 레버.
    place_height_cube = RewTerm(
        func=task_mdp.place_height_reward,
        weight=20.0,  # v8: 30→20 reward 스케일 재조정(value target 분산↓)
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
            "xy_range": 0.08,
            "require_carry": False,
        },
    )

    # Stage 5: 그릇 안 삽입 — 그리퍼 조건 없음 (밀집, 큐브 수 비례)
    insert_cube = RewTerm(
        func=task_mdp.insert_reward,
        weight=40.0,  # v8: 80→40 reward 스케일 재조정
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 6: 그릇 안 + 그리퍼 열림 완료 (밀집, 배치된 큐브 수)
    release_cube = RewTerm(
        func=task_mdp.release_bonus,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 5.5: 그릇 위에서 그리퍼 열기 유도 — release valley 메움 (밀집)
    # carry(8)+transport(8) 잡고-버티기 local optimum 탈출. inside 게이트 없이
    # '그릇 중심 + 들림 + open_frac' 에 연속 gradient → 그릇 위에서 손 펴 떨구기.
    # v7: close_ref 0.20→0.40(거의 다 열어야 보상 — '살짝 열고 hover' 캠핑 차단),
    #     xy_range 0.10→0.06(그릇 중심 정밀 정렬 유도, 가장자리 떨구기 방지).
    over_bowl_drop_cube = RewTerm(
        func=task_mdp.over_bowl_drop_reward,
        weight=12.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "xy_range": 0.06,
            "open_threshold": 0.60,
            "close_ref": 0.40,
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
        params={"pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES], "lift_min": 0.02},
    )

    # 전체 성공 보너스 — 4개 큐브 전부 배치 완료
    task_success = RewTerm(
        func=task_mdp.task_success_bonus,
        weight=50.0,  # v8: 200→50 reward 스케일 재조정(최대 value target 200→50, 안정성)
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
            # PickCube termination은 "큐브가 그릇 안에 있음"과 일치한다.
            # release_cube가 gripper open을 별도로 보상한다.
            "require_open": False,
        },
    )

    # 행동률·관절 속도 페널티 — smoothness. v7: -1e-4→-1e-3(10×). 큐브 든 채 위아래로
    # 진동하는 jittery 정책 억제(이전 -1e-4 는 사실상 0 이라 흔들기 방치). sim2real 필수.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
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
        weight=-0.006,
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )
    # task_done(전부 배치)가 곧 종료라 이 보너스는 완료 step 에 1회 지급되는
    # 터미널 보너스로 동작한다. 완료 시각에 따라 ~100(즉시)→~17(25s) 차등.
    early_finish_bonus = RewTerm(
        func=task_mdp.early_finish_bonus,
        weight=30.0,  # v8: 100→30 reward 스케일 재조정
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
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
            "pens_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_cfg": SceneEntityCfg(BOWL_NAME),
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
            "pens_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
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

    # 큐브 4개를 workspace 내에서 완전 무작위 배치 (rejection sampling)
    randomize_cubes = randomize_cubes_scattered(
        CUBE_NAMES,
        BOWL_NAME,
        x_range=_CUBE_SCATTER_X_RANGE,
        y_range=_CUBE_SCATTER_Y_RANGE,
        yaw_range_deg=(-30.0, 30.0),
        min_cube_sep=0.10,
        min_bowl_sep=0.18,
    )

    # 그릇 호(arc) 랜덤화 범위는 두 기하 제약으로 결정된다.
    #
    # 제약 A — 매트 왼쪽 경계(world x=1.50):
    #   bowl_center_x - r_top(0.075) >= 1.50 + 0.01(여유)
    #   1.62 + 0.44*sin(a) >= 1.585  →  sin(a) >= -0.0795  →  a >= -4.56°
    #   → 왼쪽 한계 -4°
    #
    # 제약 B — 그릇-Cube3 겹침:
    #   유효 충돌 반경 = r_top(0.075) + Cube3 half-diag(0.0354)
    #                   + cube_contactOffset(0.002) + bowl_contactOffset(0.004) = 0.1164m
    #   Cube3 최악 위치 (1.79, -0.33) 기준 임계 각도 풀면 9.48°.
    #   → 안전 여유 포함 오른쪽 한계 +8°
    randomize_bowl = randomize_object_on_arc(BOWL_NAME, radius=0.44, angle_range_deg=(-4.0, 8.0))

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
        # robot base (1.84,-0.565,0.6749), 큐브/그릇 작업공간 x~1.6-2.1, y~-0.5~-0.2.
        self.viewer.eye = (1.90, 0.95, 0.98)
        self.viewer.lookat = (1.85, -0.32, 0.76)
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
    "release_cube",
    "over_bowl_drop_cube",
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
            term.params["pen_cfgs"] = active_cfgs
    for term_name in _BOWL_RADIUS_REWARD_TERMS:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            term.params["cup_radius"] = bowl_radius

    env_cfg.terminations.success.params["pens_cfg"] = active_cfgs
    env_cfg.terminations.success.params["radius"] = bowl_radius
    # 큐브 추락 종료도 활성 큐브만 검사(비활성 큐브는 지면 아래라 오탐 방지)
    cube_lost_term = getattr(env_cfg.terminations, "cube_lost", None)
    if cube_lost_term is not None:
        cube_lost_term.params["pens_cfg"] = active_cfgs

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
