"""Isaac Lab 3 전용 fixed-seed SO-101 parity environment.

기존 Isaac 5.1 task config를 수정하지 않고 canonical executor 검증에 필요한
robot, cube desk, 3-camera, absolute joint-position action만 구성한다.
Quaternion은 Isaac Lab 3 경계에서 모두 XYZW를 사용한다.
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_CFG, ROBOT_USD_PATH

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _yaw_xyzw(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _axis_angle_xyzw(
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    half = angle_rad * 0.5
    scale = math.sin(half)
    return (axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half))


_CUBES = {
    "Cube1": ((-0.14, 0.135, 0.730), _yaw_xyzw(20.0)),
    "Cube2": ((0.14, 0.115, 0.730), _yaw_xyzw(-35.0)),
    "Cube3": ((-0.10, 0.225, 0.735), _yaw_xyzw(50.0)),
    "Cube4": ((0.09, 0.195, 0.735), _yaw_xyzw(-20.0)),
}
_BOWL = ((-0.22, 0.315, 0.715), _yaw_xyzw(0.0))


def _camera(
    prim_path: str,
    pos: tuple[float, float, float],
    rot_xyzw: tuple[float, float, float, float],
    focal_length: float,
    focus_distance: float,
    clipping_range: tuple[float, float],
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot_xyzw, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        width=640,
        height=480,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
    )


@configclass
class PickCubeIsaac6SceneCfg(InteractiveSceneCfg):
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=1800.0,
            color=(1.0, 0.98, 0.95),
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(-0.4031, -0.1271, -0.2725, 0.8644)
        ),
    )
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.01, 0.0)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6749),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={name: 0.0 for name in JOINT_ORDER},
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=JOINT_ORDER[:5],
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

    Cube1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube1",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBES["Cube1"][0], rot=_CUBES["Cube1"][1]),
    )
    Cube2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube2",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBES["Cube2"][0], rot=_CUBES["Cube2"][1]),
    )
    Cube3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube3",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBES["Cube3"][0], rot=_CUBES["Cube3"][1]),
    )
    Cube4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Cube4",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBES["Cube4"][0], rot=_CUBES["Cube4"][1]),
    )
    Bowl = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/Bowl",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_BOWL[0], rot=_BOWL[1]),
    )

    top_camera = _camera(
        "{ENV_REGEX_NS}/TopCamera",
        (0.03, -0.005, 1.72),
        (-0.4238, 0.4466, 0.5424, 0.5716),
        18.0,
        1.3,
        (0.1, 6.0),
    )
    wrist_camera = _camera(
        "{ENV_REGEX_NS}/Robot/gripper/WristCamera",
        (0.0, 0.045, -0.04),
        (0.6061, -0.6061, -0.3642, -0.3642),
        18.0,
        0.2,
        (0.02, 3.0),
    )
    front_camera = _camera(
        "{ENV_REGEX_NS}/Robot/shoulder/FrontCamera",
        (-0.040, 0.0, 0.025),
        _axis_angle_xyzw((0.0, 1.0, 0.0), math.pi),
        18.0,
        1.0,
        (0.1, 6.0),
    )


@configclass
class ActionsCfg:
    arm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_ORDER,
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER)},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class EventsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PickCubeIsaac6ParityEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickCubeIsaac6SceneCfg = PickCubeIsaac6SceneCfg(
        num_envs=1,
        env_spacing=2.5,
        clone_in_fabric=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.seed = 0
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            enable_external_forces_every_iteration=True,
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            gpu_found_lost_aggregate_pairs_capacity=4 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_max_rigid_patch_count=16 * 2**16,
            gpu_collision_stack_size=2**29,
        )
        self.viewer.eye = (0.06, 1.515, 0.98)
        self.viewer.lookat = (0.01, 0.245, 0.76)
        self.viewer.resolution = (1280, 720)
