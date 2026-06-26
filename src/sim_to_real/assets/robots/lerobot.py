"""SO-101 follower ArticulationCfg — leisaac ``assets/robots/lerobot.py`` 이식.

leisaac 의 ``SO101_FOLLOWER_CFG`` 를 우리 USD(``ROBOT_USD_PATH``)·규약으로 vendor 한 **단일
소스**다. ``pick_cube_env_cfg`` 등은 이 CFG 를 ``.replace(...)`` 로 씬 특화 필드(prim_path·
contact sensor·init pose)만 덮어쓰고, 검증된 actuator/solver 값은 여기서 가져온다.

조인트 limit/motor/rest 테이블은 ``so101_contract.leader_calibration`` 에 단일 정의되어 있고
여기서 leisaac-parity 이름으로 re-export 한다(값 중복 0). LEKIWI 는 우리 하드웨어가 아니라 제외.

원본의 ``from leisaac.utils.constant import ASSETS_ROOT`` 결합은 제거했다.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from sim_to_real.assets.scenes.cube_desk import ROBOT_USD_PATH

# leader↔sim calibration 테이블의 단일 소스에서 re-export (leisaac-parity 이름 유지).
from so101_contract.leader_calibration import (  # noqa: F401
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_REST_POSE_RANGE,
    SO101_FOLLOWER_USD_JOINT_LIMITS,
)

# rigid_props 를 명시 — teleop device 가 ``cfg.scene.robot.spawn.rigid_props.disable_gravity``
# 를 토글하므로(use_teleop_device) rigid_props 객체가 반드시 존재해야 한다.
SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        # Feetech STS3215 를 낮은 stiffness(soft PD) + 높은 effort 상한으로 모델링.
        # 그리퍼가 큐브에 막혀도 클램프 토크가 최대 10 Nm 까지 올라가 grasp 가 유지된다.
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
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
