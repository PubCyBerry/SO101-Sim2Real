"""cube_desk 씬 위에서 Franka Panda 가 pick-and-place 하는 환경 설정.

기존 `pick_cube` 의 cube_desk 씬(`PickCubeSceneCfg`)·DR 이벤트(`PickCubeEventCfg`)를
그대로 재사용하되 **로봇만 SO-101 → Franka Panda 로 교체**하고, action 을 task-space
DifferentialIK(arm) + binary gripper 로 바꾼다.

이 환경은 scripted state machine 데모(`scripts/environments/pick_cube_franka_state_machine.py`)
전용이라 observation/termination 은 최소로 둔다(보상·privileged state 는 SO-101 RL 용이고
gripper body 이름 `gripper` 를 가정하므로 Franka 와 호환되지 않아 제외).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (
    PickCubeEventCfg,
    PickCubeSceneCfg,
)


# ---------------------------------------------------------------------------
# Franka base 배치
# ---------------------------------------------------------------------------
# 책상 윗면 높이(world z≈0.709)에 마운트한다. Franka reach ≈0.855m 인데 바닥(z=0)
# 마운트로는 책상 위 큐브(z≈0.725)까지 3D 거리가 ~1.0m 라 도달 불가 → 책상 윗면에
# 올린다(fix_root_link 로 공중 고정).
#
# **yaw 정렬이 중요**: 큐브 scatter 영역은 x 폭(≈0.48m)이 y 폭(≈0.14m)보다 훨씬 넓다.
# Franka 는 forward(USD +X) reach 는 길지만 down-facing 자세로 side 로 뻗기는 어렵다.
# 따라서 base 를 큐브 분산 주축(world +X)과 forward 가 맞도록 yaw 0° 로 두고, 큐브
# 영역 앞쪽(작은 x)에 배치해 모든 큐브가 forward reach 안에 들어오게 한다.
#   큐브 x∈[1.60,2.08] → base x=1.30 기준 forward 0.30~0.78m, side(y) ±0.10m.
#   가까운 큐브가 forward 0.22m 처럼 너무 가까우면 Franka 가 팔을 접으며 ee 가 위로
#   솟는 IK 해로 빠져 하강을 못 한다. base 를 충분히 뒤로 빼 가까운 큐브도 forward
#   0.3m 이상 확보하고, 먼 큐브(0.78m)는 reach 0.855m 안에 둔다.
_FRANKA_POS = (1.30, -0.40, 0.71)
# yaw 0° (Franka USD forward +X = world +X, 큐브 분산 방향). identity quat.
_FRANKA_ROT = (1.0, 0.0, 0.0, 0.0)

# panda_hand → 그리퍼 손가락 tip 오프셋(+Z). IK 가 제어하는 작업점.
FRANKA_EE_OFFSET = (0.0, 0.0, 0.107)
# 그리퍼 열림/닫힘 손가락 joint 목표(m).
FRANKA_GRIPPER_OPEN = 0.04
FRANKA_GRIPPER_CLOSE = 0.0


# ---------------------------------------------------------------------------
# Scene — cube_desk 그대로, robot 만 Franka 로 교체
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaSceneCfg(PickCubeSceneCfg):
    """cube_desk(책상+조명+매트+Cube1~4+Bowl) + Franka Panda."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_FRANKA_POS,
            rot=_FRANKA_ROT,
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": FRANKA_GRIPPER_OPEN,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Actions — task-space DifferentialIK(arm) + binary gripper
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaActionsCfg:
    """8-dim action: [ee pose(7) in root frame, gripper binary(1)]."""

    arm: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=FRANKA_EE_OFFSET),
        scale=1.0,
    )
    gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": FRANKA_GRIPPER_OPEN},
        close_command_expr={"panda_finger_joint.*": FRANKA_GRIPPER_CLOSE},
    )


# ---------------------------------------------------------------------------
# Observations — 최소(SM 은 scene 상태를 직접 읽음)
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — 빈 매니저 (SM 데모는 보상을 쓰지 않으나 ManagerBasedRLEnvCfg 필수 필드)
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaRewardsCfg:
    pass


# ---------------------------------------------------------------------------
# Terminations — timeout 만 (데모는 SM 이 종료를 제어)
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class PickCubeFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Franka pick-and-place on cube_desk — pure Isaac Lab 2.3.2 ManagerBased."""

    scene: PickCubeFrankaSceneCfg = PickCubeFrankaSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickCubeFrankaObservationsCfg = PickCubeFrankaObservationsCfg()
    actions: PickCubeFrankaActionsCfg = PickCubeFrankaActionsCfg()
    rewards: PickCubeFrankaRewardsCfg = PickCubeFrankaRewardsCfg()
    terminations: PickCubeFrankaTerminationsCfg = PickCubeFrankaTerminationsCfg()
    # DR 이벤트는 robot 무관(큐브/그릇 RigidObject 만 조작) → cube task 와 공유.
    events: PickCubeEventCfg = PickCubeEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # 4 큐브 순차 pick-place 데모를 한 에피소드에 끝내도록 충분히 길게.
        self.episode_length_s = 120.0
        # GPU pipeline
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024
