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

from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES
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
        # annealing: prob 를 initial→final 로 선형 감쇠(common_step_counter 기준).
        # anneal_steps<=0 이면 감쇠 없음(prob 고정).
        self._bootstrap_prob_final = float(getattr(cfg, "grasp_bootstrap_prob_final", self._bootstrap_prob))
        self._bootstrap_anneal_steps = float(getattr(cfg, "grasp_bootstrap_anneal_steps", 0.0))
        # graded backward curriculum: 부트스트랩 env 중 pre-grasp(열린 그리퍼가 책상 위
        # 큐브 바로 위에 hover) 비율. anneal 진행도 p 로 0→1 ramp — 초반엔 full-grasp 로
        # 하류를 가르치고, 후반엔 pre-grasp 로 '실제 grasp 행동'을 가르친다.
        self._bootstrap_pregrasp_open = float(getattr(cfg, "grasp_bootstrap_pregrasp_open", 0.90))
        self._bootstrap_rest_z = float(getattr(cfg, "grasp_bootstrap_rest_z", 0.726))
        # pre-grasp 비율 오버라이드(>=0 이면 anneal p 대신 이 고정값 사용). 모니터가 세 가지
        # 시작조건(scratch/full/pre)을 동시에 측정할 때 사용. -1=anneal p(학습 기본).
        self._bootstrap_pregrasp_frac = float(getattr(cfg, "grasp_bootstrap_pregrasp_frac", -1.0))
        # place 부트스트랩(큐브를 그릇 위에 든 채 시작, over_bowl→placed 하류 학습 가속)
        self._place_bootstrap_prob = float(getattr(cfg, "place_bootstrap_prob", 0.0))
        self._place_bootstrap_z = float(getattr(cfg, "place_bootstrap_z", 0.09))
        # env별 현재 에피소드 부트스트랩 여부(모니터의 scratch/bootstrap 단계별 집계용).
        # 0=scratch(정상 시작), 1=full-grasp, 2=pre-grasp.
        self.bootstrap_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 그릇(동적 rigid body) 초기 pose — 교란 패널티(bowl_disturb)의 기준.
        # reset(randomize_bowl) 직후 값을 "교란 안 된 상태"로 저장 → reward 가 현재 pose 와
        # 비교해 tilt(엎힘)·xy 변위를 패널티화. 운반·place 중 그릇을 밀치거나 엎는 것 억제.
        self._bowl_init_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._bowl_init_quat[:, 0] = 1.0
        self._bowl_init_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # 큐브 초기 xy(에피소드 시작 = scatter/부트스트랩 적용 후) — cube_predisturb 패널티
        # 의 기준. CUBE_NAMES 순서 (num_envs, n_cubes, 2). 잡기 전 큐브를 쳐서 밀어낸
        # 거리를 이 기준으로 측정한다.
        self._cube_init_xy = torch.zeros(self.num_envs, len(CUBE_NAMES), 2, device=self.device)

        # place 단계 PBRS(place_pbrs_reward)의 이전 step potential Φ(s_{t-1}).
        # reward 함수가 매 step γΦ(s_t)-Φ(s_{t-1}) 계산 후 갱신. reset 직후 0.
        self._place_potential_prev = torch.zeros(self.num_envs, device=self.device)

        # over_bowl_drop PBRS(over_bowl_drop_pbrs_reward)의 이전 step potential.
        # over_bowl+열기 유도. reset 직후 0.
        self._over_bowl_drop_potential_prev = torch.zeros(self.num_envs, device=self.device)

    def _anneal_progress(self) -> float:
        """학습 진행도 p∈[0,1] (common_step_counter / anneal_steps). 감쇠 없으면 0."""
        if self._bootstrap_anneal_steps <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, self.common_step_counter / self._bootstrap_anneal_steps)))

    def _current_bootstrap_prob(self, p: float) -> float:
        return self._bootstrap_prob * (1.0 - p) + self._bootstrap_prob_final * p

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
        """env_ids 중 일부를 grasp 부트스트랩 상태(full-grasp 또는 pre-grasp)로 초기화.

        annealing: 진행도 p 에 따라 부트스트랩 prob 감쇠 + pre-grasp 비율 0→1 ramp.
          - full-grasp: 큐브를 grasp point(3D)에 두고 그리퍼 닫음 → 하류(lift~success) 학습.
          - pre-grasp : 큐브를 grasp point XY·책상 높이에 두고 그리퍼 open → '실제 grasp
            행동(하강+닫기)'을 align 보상과 함께 학습. grasp 갭을 단계적으로 메움.
        """
        if self._grasp_offset is None or len(env_ids) == 0:
            return
        p = self._anneal_progress()
        prob = self._current_bootstrap_prob(p)
        if prob <= 0.0:
            return
        device = self.device
        mask = torch.rand(len(env_ids), device=device) < prob
        sel = env_ids[mask]
        if len(sel) == 0:
            return
        # 선택된 env 를 pre-grasp / full-grasp 로 분할 (pre-grasp 비율 = override 또는 anneal p)
        pre_frac = self._bootstrap_pregrasp_frac if self._bootstrap_pregrasp_frac >= 0.0 else p
        is_pre = torch.rand(len(sel), device=device) < pre_frac
        robot = self.scene["robot"]
        cube = self.scene[CUBE_NAMES[0]]  # 활성 큐브 1개(stage 별 첫 큐브)
        origins = self.scene.env_origins[sel]
        quat = torch.zeros(len(sel), 4, device=device); quat[:, 0] = 1.0

        # 큐브 위치: full-grasp=grasp point(+lift), pre-grasp=grasp XY·책상 높이
        offset = self._grasp_offset.unsqueeze(0).expand(len(sel), -1).clone()  # (M,3)
        offset[:, 2] += self._bootstrap_lift
        pre_off = offset.clone()
        pre_off[:, 2] = self._bootstrap_rest_z  # 책상 위 resting 높이로 대체
        cube_off = torch.where(is_pre.unsqueeze(-1), pre_off, offset)
        pos = origins + cube_off
        cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=sel)
        cube.write_root_velocity_to_sim(torch.zeros(len(sel), 6, device=device), env_ids=sel)

        # 그리퍼 joint: full-grasp=닫힘각, pre-grasp=열림(큐브 받아들일 자세)
        jpos = torch.where(
            is_pre,
            torch.full((len(sel),), self._bootstrap_pregrasp_open, device=device),
            torch.full((len(sel),), self._bootstrap_close, device=device),
        )
        robot.write_joint_position_to_sim(
            jpos.unsqueeze(-1), joint_ids=[self._gripper_jid], env_ids=sel
        )
        # 부트스트랩 종류 기록(모니터 집계용): pre-grasp=2, full-grasp=1
        self.bootstrap_kind[sel] = torch.where(
            is_pre,
            torch.full((len(sel),), 2, device=device, dtype=torch.long),
            torch.full((len(sel),), 1, device=device, dtype=torch.long),
        )

    def _bootstrap_place(self, env_ids: torch.Tensor) -> None:
        """bootstrap_kind=0(scratch) env 중 일부를 place 부트스트랩 상태로 초기화.

        큐브를 그릇 정중앙 위(place_bootstrap_z m)에 배치 + 그리퍼 닫힘 → over_bowl→placed
        하류를 직접 학습(grasp 부트스트랩과 동형, place 단계 특화).
        """
        if self._place_bootstrap_prob <= 0.0 or len(env_ids) == 0:
            return
        # _gripper_jid 는 step() 에서 1회 캐시 — 초기 reset 에선 아직 None
        if self._grasp_offset is None:
            return
        # scratch(0)인 env 에서만 적용(grasp 부트스트랩과 중복 방지)
        scratch = env_ids[self.bootstrap_kind[env_ids] == 0]
        if len(scratch) == 0:
            return
        device = self.device
        mask = torch.rand(len(scratch), device=device) < self._place_bootstrap_prob
        sel = scratch[mask]
        if len(sel) == 0:
            return
        bowl = self.scene[BOWL_NAME]
        cube = self.scene[CUBE_NAMES[0]]
        robot = self.scene["robot"]
        origins = self.scene.env_origins[sel]
        # 그릇 env-local 위치 + z offset
        bowl_local = bowl.data.root_pos_w[sel] - origins
        cube_pos = origins.clone()
        cube_pos[:, 0] += bowl_local[:, 0]
        cube_pos[:, 1] += bowl_local[:, 1]
        cube_pos[:, 2] += bowl_local[:, 2] + self._place_bootstrap_z
        quat = torch.zeros(len(sel), 4, device=device); quat[:, 0] = 1.0
        cube.write_root_pose_to_sim(torch.cat([cube_pos, quat], dim=-1), env_ids=sel)
        cube.write_root_velocity_to_sim(torch.zeros(len(sel), 6, device=device), env_ids=sel)
        # 그리퍼 닫힘(큐브 잡은 상태)
        robot.write_joint_position_to_sim(
            torch.full((len(sel), 1), self._bootstrap_close, device=device),
            joint_ids=[self._gripper_jid], env_ids=sel,
        )
        # bootstrap_kind=3 (place 부트스트랩)
        self.bootstrap_kind[sel] = 3

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # 리셋되는 env 는 일단 scratch(0)로 초기화 — _bootstrap_grasp 가 선택된 env 만 1/2 로.
        self.bootstrap_kind[env_ids] = 0
        # offset 은 step() 에서 1회 캐시(첫 init reset 때는 body_pos_w 미확정일 수 있어
        # 캐시 전이면 부트스트랩 skip — _bootstrap_grasp 내부 가드).
        self._bootstrap_grasp(env_ids)
        self._bootstrap_place(env_ids)  # grasp 부트스트랩 후 남은 scratch env 중 일부에 place 부트스트랩
        # 그릇 초기 pose 저장(super()._reset_idx 안에서 randomize_bowl 이 적용된 직후).
        # 교란 패널티의 기준점. env-origin 무관(방향=quat, 변위=env-local xy).
        bowl = self.scene[BOWL_NAME]
        self._bowl_init_quat[env_ids] = bowl.data.root_quat_w[env_ids].clone()
        self._bowl_init_xy[env_ids] = (
            bowl.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        ).clone()
        # 큐브 초기 xy 저장(scatter + 부트스트랩 텔레포트 적용 후 = 에피소드 시작 위치).
        # cube_predisturb 패널티가 "이 위치에서 밀려난 정도"를 측정.
        origins_xy = self.scene.env_origins[env_ids, :2]
        for i, name in enumerate(CUBE_NAMES):
            cube_i = self.scene[name]
            self._cube_init_xy[env_ids, i] = (
                cube_i.data.root_pos_w[env_ids, :2] - origins_xy
            ).clone()
        # PBRS potential 초기화(reset 후 첫 step γΦ-0 jump 최소화)
        self._place_potential_prev[env_ids] = 0.0
        self._over_bowl_drop_potential_prev[env_ids] = 0.0

    def step(self, action):
        if self._grasp_offset is None:
            self._cache_grasp_geom()
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
