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

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_CFG, CUBE_DESK_USD_PATH, ROBOT_USD_PATH
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (
    SO101_JOINT_ORDER,
    _look_at_quat_world,
    _pinhole_camera_cfg,
    _yaw_quat,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES
from sim_to_real.utils.domain_randomization import (
    randomize_object_in_ellipse,
    randomize_object_on_arc,
)

from sim_to_real.tasks.pick_pen import mdp as task_mdp


# World-frame (x, y) of the bowl at scene authoring time.
# BOWL_LOCAL=(0, 0.40) + SCENE_OFFSET=(2.2, -0.57) = (2.2, -0.17)
BOWL_CENTER_XY: tuple[float, float] = (2.2, -0.17)
BOWL_SUCCESS_RADIUS: float = 0.06
BOWL_HEIGHT_RANGE: tuple[float, float] = (0.005, 0.12)

# Robot base position: SCENE_OFFSET(2.2, -0.57) + scene-local robot offset(0, -0.04).
#
# z 정합: 책상 상판(DeskTop) 윗면은 world z=0.76 (SCENE_OFFSET.z 0.76 + center
# -0.02 + half-thickness 0.02). 그러나 so101_follower.usd 의 articulation root
# 원점(z=0)은 베이스 바닥이 아니다 — 베이스 최하단 지오메트리가 local z=+0.0301
# 에 있다(USD bbox 측정). 따라서 원점을 0.76 에 두면 팔 전체가 ~3 cm 떠버린다
# (사용자가 보고한 "로봇이 책상 위에 떠 있는" 현상). 베이스 판이 상판에 닿도록
# 원점을 내린다:  robot_z = desk_top(0.76) - base_min_z(0.0301) ≈ 0.7299.
_ROBOT_POS = (2.2, -0.61, 0.7299)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


_CUBE_INIT_STATES = {
    "Cube1": ((2.05, -0.35, 0.7795), _yaw_quat(25.0)),
    "Cube2": ((2.35, -0.35, 0.7795), _yaw_quat(-30.0)),
    "Cube3": ((2.25, -0.31, 0.7795), _yaw_quat(60.0)),
    "Cube4": ((2.15, -0.31, 0.7795), _yaw_quat(-10.0)),
}
_BOWL_INIT_STATE = ((2.2, -0.17, 0.766), _yaw_quat(0.0))

# ---------------------------------------------------------------------------
# 카메라 리그 상수 — North Star 계약: observation.images.{top,front,wrist}
#   · 모두 640×480 (W×H) RGB, update_period=0.0 (render_interval 마다 갱신)
#   · 포즈/FOV 는 cube_task GUI 튜너와 실제 데이터셋 프레임 기준으로 보정.
#   · top 은 world frame 절대 좌표, front/wrist 는 각각 shoulder/gripper 링크
#     자식 prim 의 local offset. num_envs=1 smoke 기준.
# ---------------------------------------------------------------------------

# 값은 GUI 카메라 튜너(teleop_se3_agent.py)로 보정한 결과. rot 은 모두
# wxyz, Isaac Lab world-convention(forward +X, up +Z).
# top: 로봇 뒤(-y)·높은 곳에서 내려보는 급경사 oblique.
_TOP_CAMERA_POS = (2.2, -0.93, 1.70)
# _TOP_CAMERA_ROT 가 None 이 아니면 이 quat 을 직접 쓰고, None 이면 _TOP_CAMERA_TARGET
# 으로 look_at 을 계산한다(하위호환).
_TOP_CAMERA_ROT = (0.6124, -0.3536, 0.3536, 0.6124)
_TOP_CAMERA_TARGET = (2.14, -0.15, 0.76)
_TOP_CAMERA_FOCAL = 19.0

# front: shoulder_pan 전면부에 붙은 카메라.
_FRONT_CAM_LOCAL_POS = (-0.03, -0.01, 0.03)
_FRONT_CAM_LOCAL_ROT = (0.0, 0.0872, 0.9962, 0.0)
_FRONT_CAMERA_FOCAL = 19.0

# wrist: gripper 위/옆에 강결합된 카메라.
_WRIST_CAM_LOCAL_POS = (0.0, 0.05, -0.08)
_WRIST_CAM_LOCAL_ROT = (-0.183, 0.683, -0.683, -0.183)
_WRIST_CAMERA_FOCAL = 19.0


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
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # cube desk USD (contains desk, mat, and all rigid objects)
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # SO-101 follower articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
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
            joint_pos={j: 0.0 for j in SO101_JOINT_ORDER},
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
# 카메라 리그 (선택 주입) — observation.images.{top,front,wrist}
# ---------------------------------------------------------------------------


def make_pick_cube_camera_cfgs(
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_rot: tuple[float, float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    front_local_pos: tuple[float, float, float] | None = None,
    front_local_rot: tuple[float, float, float, float] | None = None,
    front_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
) -> dict[str, TiledCameraCfg]:
    """top/front/wrist 카메라 cfg 3개를 반환.

    기본 PickCubeSceneCfg 밖에 두어 기본 env 가 --enable_cameras 없이 돌게 한다.
    gym.make() 전에 add_pick_cube_cameras() 로 scene cfg 에 주입해서 쓴다.
    각 카메라는 480×640 RGB.

    top 회전: ``top_rot``(world-conv wxyz quat) 우선 → 없고 ``top_target`` 도 없으면
    보정된 기본 ``_TOP_CAMERA_ROT`` → ``top_target`` 만 주어지면 look_at 계산.
    """

    top_pos = _TOP_CAMERA_POS if top_pos is None else top_pos
    top_focal = _TOP_CAMERA_FOCAL if top_focal is None else top_focal
    front_local_pos = _FRONT_CAM_LOCAL_POS if front_local_pos is None else front_local_pos
    front_local_rot = _FRONT_CAM_LOCAL_ROT if front_local_rot is None else front_local_rot
    front_focal = _FRONT_CAMERA_FOCAL if front_focal is None else front_focal
    wrist_local_pos = _WRIST_CAM_LOCAL_POS if wrist_local_pos is None else wrist_local_pos
    wrist_local_rot = _WRIST_CAM_LOCAL_ROT if wrist_local_rot is None else wrist_local_rot
    wrist_focal = _WRIST_CAMERA_FOCAL if wrist_focal is None else wrist_focal

    if top_rot is not None:
        top_quat = top_rot
    elif top_target is not None:
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
    front = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/shoulder/FrontCamera",
        front_local_pos,
        front_local_rot,
        front_focal,
        focus_distance=0.6,
        clipping_range=(0.05, 6.0),
    )
    # front/wrist: robot 링크의 자식 prim → 각 링크 회전을 따라 이동/회전한다.
    # pos/rot 은 해당 부모 링크 local frame 기준. 정확한 화각은 GUI 렌더로 튜닝한다.
    wrist = _pinhole_camera_cfg(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        wrist_local_pos,
        wrist_local_rot,
        wrist_focal,
        focus_distance=0.2,
        clipping_range=(0.02, 3.0),
    )
    return {"top_camera": top, "front_camera": front, "wrist_camera": wrist}


def add_pick_cube_cameras(
    scene_cfg: PickCubeSceneCfg,
    *,
    top_pos: tuple[float, float, float] | None = None,
    top_rot: tuple[float, float, float, float] | None = None,
    top_target: tuple[float, float, float] | None = None,
    top_focal: float | None = None,
    front_local_pos: tuple[float, float, float] | None = None,
    front_local_rot: tuple[float, float, float, float] | None = None,
    front_focal: float | None = None,
    wrist_local_pos: tuple[float, float, float] | None = None,
    wrist_local_rot: tuple[float, float, float, float] | None = None,
    wrist_focal: float | None = None,
) -> PickCubeSceneCfg:
    """카메라 리그를 scene cfg 인스턴스에 in-place 주입하고 반환.

    InteractiveScene 이 scene_cfg.__dict__ 를 순회하므로 여기서 추가한 속성이
    gym.make() 시 센서로 등록된다.
    """

    for name, cam_cfg in make_pick_cube_camera_cfgs(
        top_pos=top_pos,
        top_rot=top_rot,
        top_target=top_target,
        top_focal=top_focal,
        front_local_pos=front_local_pos,
        front_local_rot=front_local_rot,
        front_focal=front_focal,
        wrist_local_pos=wrist_local_pos,
        wrist_local_rot=wrist_local_rot,
        wrist_focal=wrist_focal,
    ).items():
        setattr(scene_cfg, name, cam_cfg)
    return scene_cfg


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order)
# ---------------------------------------------------------------------------


@configclass
class PickCubeActionsCfg:
    """6-dim joint position action matching North Star joint order."""

    arm: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        scale=1.0,
        use_default_offset=True,
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
      rl_policy   — TB.3 privileged state (37-dim) for RL actor/critic.
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
        """TB.3 privileged state for RL training (37-dim, concatenated).

        Includes joint pos, gripper body pos, all cube/bowl positions relative to env
        origin, gripper→cube relative vectors, and gripper open fraction.
        No FrameTransformer dependency — resolves gripper body by name.
        """

        rl_state_obs = ObsTerm(
            func=task_mdp.rl_state,
            params={
                "pen_names": CUBE_NAMES,
                "cup_name": BOWL_NAME,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()
    rl_policy: RlPolicyCfg = RlPolicyCfg()


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
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2: 그리퍼 닫힘 + 큐브 근접 (sparse bonus, 미배치 큐브 한정)
    grasp_cube = RewTerm(
        func=task_mdp.grasp_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # Stage 2.5: 닫힌 그리퍼 + 들린 큐브 + 그릇 방향 운반 (밀집 도우미)
    carry_cube = RewTerm(
        func=task_mdp.carry_pen,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
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
        },
    )

    # Stage 4.5: 그릇 XY 근처에서 그릇 안 높이로 낮추기 (밀집)
    place_height_cube = RewTerm(
        func=task_mdp.place_height_reward,
        weight=30.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
            "require_carry": False,
        },
    )

    # Stage 5: 그릇 안 삽입 — 그리퍼 조건 없음 (밀집, 큐브 수 비례)
    insert_cube = RewTerm(
        func=task_mdp.insert_reward,
        weight=80.0,
        params={
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
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
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
        },
    )

    # 전체 성공 보너스 — 4개 큐브 전부 배치 완료
    task_success = RewTerm(
        func=task_mdp.task_success_bonus,
        weight=200.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "pen_cfgs": [SceneEntityCfg(n) for n in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "cup_radius": BOWL_SUCCESS_RADIUS,
            "cup_height_range": BOWL_HEIGHT_RANGE,
            # PickCube termination은 "큐브가 그릇 안에 있음"과 일치한다.
            # release_cube가 gripper open을 별도로 보상한다.
            "require_open": False,
        },
    )

    # 행동률·관절 속도 페널티
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
            "pens_cfg": [SceneEntityCfg(name) for name in CUBE_NAMES],
            "cup_center_xy": BOWL_CENTER_XY,
            "radius": BOWL_SUCCESS_RADIUS,
            "height_range": BOWL_HEIGHT_RANGE,
            "require_rest_pose": False,  # rest-pose check is TA.1 territory
        },
    )


# ---------------------------------------------------------------------------
# Events (domain randomisation)
# ---------------------------------------------------------------------------


@configclass
class PickCubeEventCfg:
    """Reset and randomisation events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Cubes scatter inside a small ellipse around their authored positions
    randomize_cube1 = randomize_object_in_ellipse("Cube1", 0.05, 0.02, (-10.0, 10.0))
    randomize_cube2 = randomize_object_in_ellipse("Cube2", 0.05, 0.02, (-10.0, 10.0))
    randomize_cube3 = randomize_object_in_ellipse("Cube3", 0.05, 0.02, (-10.0, 10.0))
    randomize_cube4 = randomize_object_in_ellipse("Cube4", 0.05, 0.02, (-10.0, 10.0))

    # Bowl swings along a forward-facing ±20° arc
    randomize_bowl = randomize_object_on_arc(BOWL_NAME, radius=0.44, angle_range_deg=(-20.0, 20.0))


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

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        # GPU pipeline
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # 2048+ env PPO에서 aggregate pair가 18k를 넘는다. 64k로 여유 확보.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024


# ---------------------------------------------------------------------------
# 커리큘럼 적용 헬퍼 — gym.make() 이전에 env_cfg 에 in-place 적용
# ---------------------------------------------------------------------------

_CUBE_REWARD_TERMS = (
    "reach_cube",
    "grasp_cube",
    "carry_cube",
    "lift_cube",
    "transport_cube",
    "place_height_cube",
    "insert_cube",
    "release_cube",
    "task_success",
)
_BOWL_RADIUS_REWARD_TERMS = (
    "reach_cube",
    "grasp_cube",
    "carry_cube",
    "place_height_cube",
    "insert_cube",
    "release_cube",
    "task_success",
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

    for cube_name in CUBE_NAMES:
        term = getattr(env_cfg.events, "randomize_" + cube_name.lower(), None)
        if term is not None and object_radius_scale != 1.0:
            p = term.params
            p["x_radius"] = p["x_radius"] * object_radius_scale
            p["y_radius"] = p["y_radius"] * object_radius_scale

    bowl_term = getattr(env_cfg.events, "randomize_bowl", None)
    if bowl_term is not None and container_angle_scale != 1.0:
        lo, hi = bowl_term.params["angle_range_deg"]
        bowl_term.params["angle_range_deg"] = (lo * container_angle_scale, hi * container_angle_scale)

    for cube_name in CUBE_NAMES:
        obs_name = "place_" + cube_name.lower()
        term = getattr(env_cfg.observations.subtask_terms, obs_name, None)
        if term is not None:
            term.params["radius"] = bowl_radius
