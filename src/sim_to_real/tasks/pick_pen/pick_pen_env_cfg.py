"""Pen Pick-and-Place task configuration — pure Isaac Lab 2.3.2."""

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
from isaaclab.utils import configclass

from sim_to_real.assets.scenes.pen_desk import PEN_DESK_CFG, PEN_DESK_USD_PATH, ROBOT_USD_PATH
from sim_to_real.utils.constant import PEN_CUP_NAME, PEN_NAMES
from sim_to_real.utils.domain_randomization import (
    randomize_object_in_ellipse,
    randomize_object_on_arc,
)

from . import mdp as task_mdp


# World-frame (x, y) of the pen cup at scene authoring time.
# PEN_CUP_LOCAL=(0, 0.40) + SCENE_OFFSET=(2.2, -0.57) = (2.2, -0.17)
PEN_CUP_CENTER_XY: tuple[float, float] = (2.2, -0.17)

# SO-101 joint order (North Star contract — must not change)
SO101_JOINT_ORDER: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Robot base position: SCENE_OFFSET(2.2, -0.57) + scene-local robot offset(0, -0.04)
# z=0.92 = desk surface (DeskTop center 0.90 + half-thickness 0.02)
_ROBOT_POS = (2.2, -0.61, 0.92)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


_PEN_INIT_STATES = {
    "PenWhite": ((2.05, -0.35, 0.9347), _yaw_quat(25.0)),
    "PenGray": ((2.35, -0.35, 0.9347), _yaw_quat(-30.0)),
    "PenBlack": ((2.25, -0.31, 0.9347), _yaw_quat(60.0)),
    "PenBlue": ((2.15, -0.31, 0.9347), _yaw_quat(-10.0)),
}
_PEN_CUP_INIT_STATE = ((2.2, -0.17, 0.926), _yaw_quat(0.0))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class PickPenSceneCfg(InteractiveSceneCfg):
    """Scene: pen desk + SO-101 follower + 4 pens + cup."""

    # shared world assets (not per-env)
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # pen desk USD (contains desk, mat, and all rigid objects)
    scene: AssetBaseCfg = PEN_DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # SO-101 follower articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            joint_pos={j: 0.0 for j in SO101_JOINT_ORDER},
        ),
        actuators={
            # Feetech STS3215 근사: 7.4V~12V variants 기준 약 1.4~2.9 Nm.
            # 시뮬 hold 안정성을 위해 3.0 Nm 상한, 속도 한계 5.5 rad/s.
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll"],
                effort_limit_sim=3.0,
                velocity_limit_sim=5.5,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=1.5,
                velocity_limit_sim=6.0,
                stiffness=300.0,
                damping=60.0,
            ),
        },
    )

    # Rigid objects inside the scene USD (spawn=None → wrap existing prims)
    PenWhite: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenWhite",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenWhite"][0],
            rot=_PEN_INIT_STATES["PenWhite"][1],
        ),
    )
    PenGray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenGray",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenGray"][0],
            rot=_PEN_INIT_STATES["PenGray"][1],
        ),
    )
    PenBlack: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenBlack",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenBlack"][0],
            rot=_PEN_INIT_STATES["PenBlack"][1],
        ),
    )
    PenBlue: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenBlue",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_INIT_STATES["PenBlue"][0],
            rot=_PEN_INIT_STATES["PenBlue"][1],
        ),
    )
    PenCup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/PenCup",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_PEN_CUP_INIT_STATE[0],
            rot=_PEN_CUP_INIT_STATE[1],
        ),
    )


# ---------------------------------------------------------------------------
# Actions  (6-dim joint position, North Star order)
# ---------------------------------------------------------------------------


@configclass
class PickPenActionsCfg:
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
class PickPenObservationsCfg:
    """Observations: policy (6-dim joint pos) + subtask signals."""

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
        """Per-pen placement signals (pen-in-cup, gripper open check)."""

        place_white = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenWhite"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_gray = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenGray"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_black = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlack"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        place_blue = ObsTerm(
            func=task_mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlue"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ---------------------------------------------------------------------------
# Rewards (stub — Phase B implements proper rewards)
# ---------------------------------------------------------------------------


@configclass
class PickPenRewardsCfg:
    """Minimal reward stub so ManagerBasedRLEnv builds. Phase B adds real terms."""

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
class PickPenTerminationsCfg:
    """Episode ends on timeout or when all pens are in the cup."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.task_done,
        params={
            "pens_cfg": [SceneEntityCfg(name) for name in PEN_NAMES],
            "cup_center_xy": PEN_CUP_CENTER_XY,
            "require_rest_pose": False,  # rest-pose check is TA.1 territory
        },
    )


# ---------------------------------------------------------------------------
# Events (domain randomisation)
# ---------------------------------------------------------------------------


@configclass
class PickPenEventCfg:
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

    # Pens scatter inside a small ellipse around their authored positions
    randomize_pen_white = randomize_object_in_ellipse("PenWhite", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_gray = randomize_object_in_ellipse("PenGray", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_black = randomize_object_in_ellipse("PenBlack", 0.05, 0.02, (-10.0, 10.0))
    randomize_pen_blue = randomize_object_in_ellipse("PenBlue", 0.05, 0.02, (-10.0, 10.0))

    # Cup swings along a forward-facing ±20° arc
    randomize_pen_cup = randomize_object_on_arc(PEN_CUP_NAME, radius=0.44, angle_range_deg=(-20.0, 20.0))


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class PickPenEnvCfg(ManagerBasedRLEnvCfg):
    """Pen Pick-and-Place environment — pure Isaac Lab 2.3.2 ManagerBased."""

    scene: PickPenSceneCfg = PickPenSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickPenObservationsCfg = PickPenObservationsCfg()
    actions: PickPenActionsCfg = PickPenActionsCfg()
    rewards: PickPenRewardsCfg = PickPenRewardsCfg()
    terminations: PickPenTerminationsCfg = PickPenTerminationsCfg()
    events: PickPenEventCfg = PickPenEventCfg()

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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
