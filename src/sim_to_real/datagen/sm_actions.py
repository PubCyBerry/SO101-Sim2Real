"""State-machine 데이터생성용 IK action 설정.

SM 은 grasp 동작을 **Cartesian 웨이포인트**(8D = EE pose 7D + binary gripper 1D)로 스크립트하는
게 joint 각도보다 쉽다. 이 action cfg 가 그 8D 를 받아 IsaacLab 내장 DLS IK 로 **joint target 으로
풀어** 적용한다(cuRobo 등 외부 솔버 불요). IK 는 **모션 생성 수단**이고, 데이터셋에 기록되는 action 은
IK 가 푼 **joint target**(degrees, 우리 codec)이다 — 실기기·VLA 가 joint-space 라 호환.

leisaac ``devices/action_process.py`` 의 ``so101_state_machine`` 분기를 우리 env(`PickCubeEnvCfg`)
의 actions 로 교체하기 위한 configclass. 기본 action(`PickCubeActionsCfg.arm` = slew joint, VLA 추론용)
은 건드리지 않고, datagen 드라이버가 이 cfg 로 ``env_cfg.actions`` 를 대체한다.

이 모듈은 ``isaaclab.envs.mdp`` 를 import 하므로 **AppLauncher(SimulationApp) 부팅 후**에만 import 한다.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

# 필드 선언 순서 = action 텐서 concat 순서. SM get_action = [pos(3), quat(4), gripper(1)] = 8D.
#   arm_action(IK pose) 7D 먼저, gripper_action(binary) 1D 뒤.


@configclass
class StateMachineActionsCfg:
    """SO-101 state-machine datagen 용 IK + binary-gripper action."""

    arm_action: mdp.DifferentialInverseKinematicsActionCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        # TODO(verify): SO-101 USD articulation 의 EE rigid body 이름. leisaac 은 "gripper".
        body_name="gripper",
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.04}
        ),
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # TODO(grasp-tuning): leisaac 기본값. 큐브 grasp 에 맞춘 close target 튜닝 필요.
        open_command_expr={"gripper": 1.0},
        close_command_expr={"gripper": 0.4},
    )
