"""PickCube RL 환경 — leisaac 동적 gripper effort 배선.

stock ``ManagerBasedRLEnv`` 는 teleop/replay/추론과 달리 RL 학습 step 에서
``dynamic_reset_gripper_effort_limit_sim`` 을 호출하지 않는다. leisaac 은 모든 경로에서
물체 질량 기준 effort clamp(mass/0.15, 0.5, 10) 을 적용한다 — 그리퍼가 큐브에 막혀도
클램프 토크가 10 Nm 까지 올라가 soft-PD SO-101 의 grasp 가 유지된다. 매 step 배선한다.
``cfg.dynamic_reset_gripper_effort_limit`` 가 False 면 stock 동작(no-op)."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeEnv(ManagerBasedRLEnv):
    """동적 gripper effort 배선만 추가한 ManagerBasedRLEnv."""

    def step(self, action):
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
