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
        # SO-101 Feetech STS3215 액추에이터 — Workshop 참조(``assets/so101.py``)의 **per-joint
        # 튜닝을 그대로 이식**한다. joint 이름만 우리 규약(snake_case = feature_codec
        # ``SO101_JOINT_ORDER``)으로 매핑하고, stiffness/damping/effort 값은 Workshop 을 복제.
        # 주석의 Gear/Torque 는 실 하드웨어 스펙(감속비·정격 토크) — sim ``effort_limit_sim`` 은
        # 안전 위해 전 축 30 N 으로 통일(하드웨어 토크보다 낮게). base 축은 stiffness 를 높여
        # load-bearing, 손목·그리퍼는 낮춰 섬세 제어.
        # ``velocity_limit_sim=10`` = slew cap(5 rad/s) 추종 헤드룸(common/utils.py 근거).
        # ⚠ 그리퍼 effort 는 런타임에 ``gripper_effort.py`` dynamic clamp(≤10)가 override 하므로
        #    ``effort_limit_sim=30`` 은 실질적으로 arm 에만 유효(그리퍼는 gentle 유지).
        # ROTATION (Gear 1/191, Torque 34.4 N·m)
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=55, damping=0.7,
        ),
        # PITCH (Gear 1/345, Torque 62.1 N·m — HIGHEST, load-bearing)
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=30, damping=0.8,
        ),
        # ELBOW (Gear 1/191, Torque 34.4 N·m)
        "elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=25, damping=0.7,
        ),
        # WRIST PITCH (Gear 1/147, Torque 26.5 N·m)
        "wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=12, damping=0.5,
        ),
        # WRIST ROLL (Gear 1/147, Torque 26.5 N·m)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=7, damping=0.5,
        ),
        # GRIPPER / JAW (Gear 1/147, Torque 26.5 N·m)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=30, velocity_limit_sim=10.0,
            stiffness=4, damping=0.3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
