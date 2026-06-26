"""PickCube 추론/데이터 substrate env — leisaac 동적 gripper effort 배선.

VLA-only 리팩토링으로 RL 학습 머신러리(grasp/place/demo 부트스트랩·reverse curriculum·
PBRS potential)는 전부 제거했다. 남은 것은 leisaac 가 모든 경로(teleop·추론 포함)에서
적용하는 **동적 gripper effort clamp**(물체 질량 기준) 하나다 — grasp 접촉 물리 정합에
필요하므로 stock ``ManagerBasedRLEnv`` 위에 step() 한 줄만 덧댄다.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeEnv(ManagerBasedRLEnv):
    """동적 gripper effort 배선(leisaac 패턴, 전 경로 적용)."""

    def step(self, action):
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
