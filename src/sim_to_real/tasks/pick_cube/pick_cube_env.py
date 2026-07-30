"""PickCube 추론/데이터 substrate env — leisaac 동적 gripper effort + 카메라 extrinsic DR 배선.

VLA-only 리팩토링으로 RL 학습 머신러리(grasp/place/demo 부트스트랩·reverse curriculum·
PBRS potential)는 전부 제거했다. 남은 것은 stock ``ManagerBasedRLEnv`` step() 앞에 붙는 두 줄:

1. leisaac 가 모든 경로(teleop·추론 포함)에서 적용하는 **동적 gripper effort clamp**(물체 질량
   기준) — grasp 접촉 물리 정합에 필요.
2. **카메라 extrinsic DR** frame-wise 갱신 — 반드시 ``super().step()`` **앞**이어야 한다.
   stock step() 은 decimation 루프 *안*에서 렌더하므로 여기서 pose 를 써야 같은 step 의 렌더가
   새 pose 를 쓴다(RGB ↔ pose 지연 0). ``mode="interval"`` EventTerm 은 렌더 뒤라 1 프레임 늦다.
   cfg 에 ``camera_extrinsic_dr`` 이 없거나 카메라가 제거된 경로는 no-op.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from sim_to_real.utils.domain_randomization import update_camera_extrinsic_dr
from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeEnv(ManagerBasedRLEnv):
    """동적 gripper effort + 카메라 extrinsic DR 배선(전 경로 적용, 둘 다 cfg 로 opt-in)."""

    def step(self, action):
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        update_camera_extrinsic_dr(self)   # ★렌더 전 — super() 뒤로 옮기면 1 프레임 늦는다
        return super().step(action)
