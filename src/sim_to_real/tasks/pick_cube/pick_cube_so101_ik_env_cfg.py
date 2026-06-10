"""cube_desk 씬에서 SO-101 follower 가 **in-sim DifferentialIK** 로 pick-and-place 하는 환경 설정.

기존 `PickCubeEnvCfg`(`pick_cube_env_cfg.py`)는 North Star 6-dim joint-position 계약을 위해
`SlewLimitedJointPositionAction`(순수 joint-space)을 쓴다. 그 환경은 in-sim IK 가 없어서
scripted state machine 이 외부 Lula 솔버를 *다른 URDF·다른 world frame* 에서 돌리고 결과를
런타임 shift 로 끼워맞춰야 했고, 그 shift 가 자세 의존이라 grasp 가 0.05~0.1m 빗나갔다.

이 환경은 Franka 데모(`pick_cube_franka_env_cfg.py`)와 동일하게 **task-space DifferentialIK(arm)
+ binary gripper** 로 바꾼다. IK 가 USD articulation Jacobian 에 대해 sim 내부에서 풀리므로
프레임 정합·제어점 불일치(stale shift)가 원천 제거된다. robot 은 SO-101 그대로 유지하므로
씬(`PickCubeSceneCfg`: cube_desk + SO-101 + Cube1~4 + Bowl)을 재사용한다.

scripted SM 데모 전용이라 observation/termination 은 최소로 둔다(보상·privileged state 는
RL 용이고 SmolVLA 학습은 기존 `PickCubeEnvCfg` 를 쓴다 — 이 환경은 SM 데모에만 쓴다).

5DOF 주의: SO-101 arm 은 5DOF 라 임의 6DOF pose 도달 불가. SM 이 보내는 target orientation 은
**radial-yaw(=base→target) + down-tilt** (5DOF 달성 가능 자세)여야 하며, DLS 가 over-constrained
pose 를 best-effort 로 푼다(lambda 를 Franka 보다 키워 정칙화).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
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

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (
    PickCubeEventCfg,
    PickCubeSceneCfg,
)

# SO-101 arm joints (gripper 제외 5DOF). North Star joint order 의 앞 5개.
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# IK 제어점 — inspect_so101_gripper_frame.py 측정으로 확정한다.
#   · gripper_frame_link 가 USD body 로 살아있으면 body_name 으로 그것, offset≈0.
#   · 아니면 "gripper" body + gripper-local grasp offset (두 손가락 사이 접점).
# 아래는 URDF gripper_frame_joint origin(gripper_link local) 초기 추정값 — 측정 후 갱신.
SO101_IK_BODY_NAME = "gripper"
SO101_IK_GRASP_OFFSET = (-0.0079, -0.000218, -0.0981)

# gripper joint 목표(rad). 현 SM 규약(open=1.0 / close=0.0). limit: [-0.1745, 1.7453].
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = 0.0


# ---------------------------------------------------------------------------
# Actions — task-space DifferentialIK(arm) + binary gripper
# ---------------------------------------------------------------------------


@configclass
class PickCubeSo101IkActionsCfg:
    """8-dim action: [ee pose(7) in root frame, gripper binary(1)]."""

    arm: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=SO101_ARM_JOINTS,
        body_name=SO101_IK_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            # 5DOF 는 6-row pose Jacobian 이 over-constrained → Franka(0.05)보다 큰 damping.
            ik_params={"lambda_val": 0.1},
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=SO101_IK_GRASP_OFFSET),
        scale=1.0,
    )
    gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": GRIPPER_OPEN},
        close_command_expr={"gripper": GRIPPER_CLOSE},
    )


# ---------------------------------------------------------------------------
# Observations — 최소(SM 은 scene 상태를 직접 읽음)
# ---------------------------------------------------------------------------


@configclass
class PickCubeSo101IkObservationsCfg:
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
class PickCubeSo101IkRewardsCfg:
    pass


# ---------------------------------------------------------------------------
# Terminations — timeout 만 (데모는 SM 이 종료를 제어)
# ---------------------------------------------------------------------------


@configclass
class PickCubeSo101IkTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------


@configclass
class PickCubeSo101IkEnvCfg(ManagerBasedRLEnvCfg):
    """SO-101 in-sim DifferentialIK pick-and-place on cube_desk — SM 데모 전용."""

    scene: PickCubeSceneCfg = PickCubeSceneCfg(num_envs=1, env_spacing=2.5)
    observations: PickCubeSo101IkObservationsCfg = PickCubeSo101IkObservationsCfg()
    actions: PickCubeSo101IkActionsCfg = PickCubeSo101IkActionsCfg()
    rewards: PickCubeSo101IkRewardsCfg = PickCubeSo101IkRewardsCfg()
    terminations: PickCubeSo101IkTerminationsCfg = PickCubeSo101IkTerminationsCfg()
    # DR 이벤트는 robot 무관(큐브/그릇 RigidObject 만 조작) → cube task 와 공유.
    events: PickCubeEventCfg = PickCubeEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4) — PickCubeEnvCfg 와 동일.
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
