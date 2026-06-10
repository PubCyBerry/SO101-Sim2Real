"""PickCube RL 환경 — leisaac 의 동적 gripper effort 조정을 RL step 에 배선.

문제: stock ``ManagerBasedRLEnv`` 는 teleop/replay 루프와 달리
``dynamic_reset_gripper_effort_limit_sim`` 을 호출하지 않는다. 그 결과 RL 학습에서
그리퍼는 actuator cfg 의 10 Nm 풀 effort 로 닫히고, 20~35 g 의 가벼운 큐브가 강한
클램프력에 튕겨나가(convex finger 접촉) grasp 가 형성되지 못했다(guided_lift 영구 0).

leisaac 은 매 step 가장 가까운 물체 질량 기준으로 effort 를 clamp(mass/0.15, 0.5, 10)
한다 — 우리 큐브는 ~0.5 Nm. 동일 동작을 RL 의 매 control step(=env.step) 에 배선한다.
순수 물리 grasp 이며 weld/유지력 추가가 아니므로 grasp-assist 가 아니다.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeEnv(ManagerBasedRLEnv):
    """매 control step gripper effort 를 nearest-object 질량에 맞춰 동적 조정."""

    def step(self, action):
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            # teleop_device 기본값으로 호출 → 내부에서 scene["robot"] 사용.
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
