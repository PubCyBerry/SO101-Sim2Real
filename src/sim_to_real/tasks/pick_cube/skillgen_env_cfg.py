"""SO-101 PickCube용 Mimic/SkillGen 전용 환경 설정."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from sim_to_real.tasks.common.mdp import (
    SlewLimitedBinaryJointPositionActionCfg,
    SlewLimitedDifferentialInverseKinematicsActionCfg,
)
from sim_to_real.tasks.common.utils import SO101_JOINT_TARGET_MAX_VELOCITY
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import PickCubeDRBaseEnvCfg, PickCubeDREnvCfg


_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@configclass
class PickCubeSkillGenActionsCfg:
    """8D absolute EE pose + binary gripper action."""

    arm_action: SlewLimitedDifferentialInverseKinematicsActionCfg = (
        SlewLimitedDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                ik_method="dls",
                use_relative_mode=False,
                ik_params={"lambda_val": 0.04},
            ),
            max_velocity={name: SO101_JOINT_TARGET_MAX_VELOCITY[name] for name in _ARM_JOINTS},
        )
    )
    gripper_action: SlewLimitedBinaryJointPositionActionCfg = SlewLimitedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 1.0},
        # 40 mm Cube1을 양쪽 finger contact envelope로 감쌀 수 있는 닫힘값.
        # 0.4 rad는 moving jaw 한쪽만 큐브를 밀고 fixed jaw에 닿지 않아 grasp가 성립하지 않는다.
        close_command_expr={"gripper": 0.01},
        max_velocity={"gripper": 2.5},
    )


def _subtasks() -> list[SubTaskConfig]:
    return [
        SubTaskConfig(
            object_ref="Cube1",
            subtask_term_signal="grasp_cube",
            subtask_start_offset_range=(0, 0),
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Close the gripper and establish a stable grasp on Cube1",
            next_subtask_description="Place Cube1 into Bowl",
        ),
        SubTaskConfig(
            object_ref="Bowl",
            subtask_term_signal="place_cube",
            subtask_start_offset_range=(0, 0),
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.0,
            num_interpolation_steps=0,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Lower, release Cube1 in Bowl, and clear the rim",
        ),
    ]


def _configure_skillgen(cfg) -> None:
    cfg.actions = PickCubeSkillGenActionsCfg()
    # 이 source SM은 cube 중심에 대해 top-down grasp를 정의한다. 일반 PickCube의
    # stable-face 6D orientation DR을 그대로 쓰면 object-centric SkillGen transform이
    # EEF 궤적까지 roll/pitch해 목표를 책상 아래로 보낸다. 또한 Cube1은 정육면체라
    # yaw가 task 의미를 바꾸지 않지만, object-centric transform은 그 yaw를 5-DOF
    # SO-101의 full orientation constraint로 전달해 도달 가능한 위치도 IK_FAIL로
    # 만든다. SkillGen은 xy layout DR을 유지하고 cube orientation만 authored pose로 고정한다.
    cfg.events.randomize_cubes.params["full_orient"] = False
    cfg.events.randomize_cubes.params["yaw_range_deg"] = (0.0, 0.0)
    cfg.datagen_config.name = "so101_pick_cube_skillgen"
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = True
    cfg.datagen_config.generation_num_trials = 10
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.generation_relative = True
    cfg.datagen_config.max_num_failures = 2500
    cfg.datagen_config.seed = 1
    cfg.subtask_configs["so101"] = _subtasks()


@configclass
class PickCubeSkillGenDRBaseEnvCfg(PickCubeDRBaseEnvCfg, MimicEnvCfg):
    """Source와 첫 smoke용 좁은 DR 분포."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_skillgen(self)


@configclass
class PickCubeSkillGenDREnvCfg(PickCubeDREnvCfg, MimicEnvCfg):
    """Production SkillGen용 full DR 분포."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_skillgen(self)
