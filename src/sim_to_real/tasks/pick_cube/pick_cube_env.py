"""PickCube RL 환경 — (1) leisaac 동적 gripper effort 배선, (2) 초기상태 grasp 부트스트랩.

(1) 동적 effort: stock ``ManagerBasedRLEnv`` 는 teleop/replay/추론과 달리 RL 학습에서
``dynamic_reset_gripper_effort_limit_sim`` 을 호출하지 않는다. leisaac 은 모든 경로에서
물체 질량 기준 effort clamp(mass/0.15, 0.5, 10) 을 적용 → 매 step 배선한다.

(2) grasp 부트스트랩(backward curriculum): 정밀 grasp 획득은 scratch-PPO 의 탐색 벽이라
처음부터 안 풀린다(reach·pregrasp 만 받고 lift=0). NVIDIA gear_assembly 처럼 일부 env 를
**큐브가 그리퍼에 잡힌 상태**로 시작시켜 lift→carry→transport→place→release→success 하류를
먼저 학습시킨다(critic value 가 grasp 의 가치를 알게 되어 정상 시작 env 의 grasp 학습도 견인).
in-episode weld/유지력이 아니라 **초기 상태 분포만 조정**하므로 grasp-assist 금지선과 무관하다.
grasp point 는 default 자세에서 1회 캐시(고정베이스라 상수)해 FK-at-reset 불안정을 회피한다.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from sim_to_real.utils.constant import CUBE_NAMES
from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim


class PickCubeEnv(ManagerBasedRLEnv):
    """동적 gripper effort + 초기상태 grasp 부트스트랩."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._grasp_offset = None  # (3,) env-local, default 자세의 jaw·gripper 중점
        self._gripper_jid = None   # articulation 내 gripper joint index
        # cfg 로 조정: 부트스트랩 비율, 그리퍼 hold 각, 들어올림 offset
        self._bootstrap_prob = float(getattr(cfg, "grasp_bootstrap_prob", 0.0))
        self._bootstrap_close = float(getattr(cfg, "grasp_bootstrap_close", -0.05))
        self._bootstrap_lift = float(getattr(cfg, "grasp_bootstrap_lift", 0.0))

    def _cache_grasp_geom(self) -> None:
        robot = self.scene["robot"]
        bn = list(robot.data.body_names)
        if "jaw" in bn and "gripper" in bn:
            j, g = bn.index("jaw"), bn.index("gripper")
            gp = 0.5 * (robot.data.body_pos_w[:, j, :] + robot.data.body_pos_w[:, g, :])
        else:
            g = bn.index("gripper")
            gp = robot.data.body_pos_w[:, g, :]
        # env-local (고정베이스·동일로봇이라 env 간 동일) → env0 기준 (3,)
        self._grasp_offset = (gp - self.scene.env_origins)[0].clone()
        self._gripper_jid = list(robot.data.joint_names).index("gripper")

    def _bootstrap_grasp(self, env_ids: torch.Tensor) -> None:
        """env_ids 중 일부를 '큐브 잡힌 상태'로 초기화."""
        if self._grasp_offset is None or self._bootstrap_prob <= 0.0 or len(env_ids) == 0:
            return
        device = self.device
        mask = torch.rand(len(env_ids), device=device) < self._bootstrap_prob
        sel = env_ids[mask]
        if len(sel) == 0:
            return
        robot = self.scene["robot"]
        # 활성 큐브 1개(stage 별 첫 큐브)를 gr리퍼 grasp point 로 텔레포트
        cube = self.scene[CUBE_NAMES[0]]
        offset = self._grasp_offset.clone()
        offset[2] += self._bootstrap_lift  # 책상 위로 살짝 들어올린 높이에 배치
        pos = self.scene.env_origins[sel] + offset.unsqueeze(0)
        quat = torch.zeros(len(sel), 4, device=device); quat[:, 0] = 1.0
        cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=sel)
        cube.write_root_velocity_to_sim(torch.zeros(len(sel), 6, device=device), env_ids=sel)
        # 그리퍼 joint 를 닫힘 각으로 (큐브를 문 상태)
        jpos = robot.data.joint_pos[sel, self._gripper_jid].clone()
        jpos[:] = self._bootstrap_close
        robot.write_joint_position_to_sim(
            jpos.unsqueeze(-1), joint_ids=[self._gripper_jid], env_ids=sel
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # offset 은 step() 에서 1회 캐시(첫 init reset 때는 body_pos_w 미확정일 수 있어
        # 캐시 전이면 부트스트랩 skip — _bootstrap_grasp 내부 가드).
        self._bootstrap_grasp(env_ids)

    def step(self, action):
        if self._grasp_offset is None:
            self._cache_grasp_geom()
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
