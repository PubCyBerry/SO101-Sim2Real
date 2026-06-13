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
from sim_to_real.tasks.common.mdp._geometry import JAW_GRASP_OFFSET, _quat_apply_wxyz


class PickCubeEnv(ManagerBasedRLEnv):
    """동적 gripper effort + 초기상태 grasp 부트스트랩."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._grasp_offset = None  # (3,) env-local, default 자세의 jaw·gripper 중점
        self._gripper_jid = None   # articulation 내 gripper joint index
        self._wrist_roll_jid = None  # wrist_roll joint index (full-grasp 부트스트랩 cube yaw 정합용)
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

        # 2-phase grasp 부트스트랩: reset 시 body_pos_w(FK)가 stale(joint write 미반영)이라
        # 큐브를 reset 자세 기준으로 놓으면 팔이 settle 하며 어긋나 grip 미성립. → reset 에선
        # pending 마킹만, FK fresh 해지는 2 step 뒤 step() 의 _apply_pending_grasp() 가 실제 배치.
        self._grasp_pending = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # countdown(2→1→0)
        self._grasp_pending_pre = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # carry 역커리큘럼: full-grasp 부트스트랩(든 큐브) env 의 그릇을 carry 경로 따라 배치.
        # f=anneal(0→1): f=0 그릇이 든 큐브 바로 밑(release만=trivial success), f=1 정상 arc(full transport).
        # achievable success 로 terminal gradient 확보 → release→short→long carry backward 학습.
        # anneal_steps<=0 이면 비활성(그릇 정상 위치 유지).
        self._carry_rc_anneal_steps = float(getattr(cfg, "carry_rc_anneal_steps", 0.0))

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

        # task_progress PBRS(task_progress_pbrs_reward)의 이전 step potential — 전용 버퍼
        # (place_pbrs 의 _place_potential_prev 와 분리). reset 직후 0.
        self._task_progress_potential_prev = torch.zeros(self.num_envs, device=self.device)

        # --- demo-state reset (RFCL reverse curriculum) ---
        # SM 성공 궤적의 실제 scene 상태를 reset 분포로 주입 → place 탐색 valley 우회.
        # 행동 클론이 아니라 상태 분포만 seed(grasp-assist 금지선과 무관). 데모 상태는
        # env-local 로 저장돼 임의 env 로 주입 가능(origin 재가산).
        self._demo_reset_prob = float(getattr(cfg, "demo_reset_prob", 0.0))
        self._demo_dataset_dir = getattr(cfg, "demo_dataset_dir", None)
        # reverse curriculum: 초반 궤적 후반부(success 근처)만 sample → 진행도 p 로 시작쪽 확장.
        # anneal_steps<=0 이면 전 구간 uniform. frac threshold = 1-p.
        self._demo_anneal_steps = float(getattr(cfg, "demo_anneal_steps", 0.0))
        self._demo_subsample = int(getattr(cfg, "demo_subsample", 2))  # 매 k step 만 적재(메모리)
        self._demo_max_files = int(getattr(cfg, "demo_max_files", 4000))
        self._demo_loaded = False
        self._demo = {}  # device 텐서 묶음 + frac
        if self._demo_reset_prob > 0.0 and self._demo_dataset_dir:
            self._load_demos()

    def _load_demos(self) -> None:
        """demo_dataset_dir 의 demo_*.pt 를 적재해 전체 상태를 device 텐서로 concat.
        각 상태에 frac(궤적 내 정규화 위치 [0,1], 1=success 종료)을 부여(reverse curriculum)."""
        import glob
        import os

        files = sorted(glob.glob(os.path.join(self._demo_dataset_dir, "demo_*.pt")))[: self._demo_max_files]
        if not files:
            print(f"[demo-reset] WARNING: no demo_*.pt in {self._demo_dataset_dir}", flush=True)
            return
        k = max(1, self._demo_subsample)
        # multi-env SM 은 모든 env DONE 까지 돌아 trailing 패딩(DONE/HOME_FINAL)이 쌓인다.
        # 추가로 RELEASE_DWELL/RETREAT(=이미 그릇에 든 success 상태)도 제외 — 거기로 reset 하면
        # 즉시 success 종료(학습 0). 데모 끝을 LOWER(그릇 위 들고 있음=완성 직전 must-complete)로
        # 둬야 reverse curriculum frac~1 이 place 완성 학습 상태가 된다(place valley 직격).
        _PAD_PHASES = {"DONE", "HOME_FINAL", "RETREAT", "RELEASE_DWELL"}
        jpos, jvel, cpos, cquat, cvel, bpos, bquat, bvel, frac = ([] for _ in range(9))
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if not d.get("meta", {}).get("success", False):
                continue
            phases = d.get("phases", [])
            T = int(d["joint_pos"].shape[0])
            # 조작 구간(패딩 제외) 인덱스만 → subsample
            keep = [t for t in range(T) if t >= len(phases) or phases[t] not in _PAD_PHASES]
            if not keep:
                continue
            keep = keep[::k]
            idx = torch.tensor(keep, dtype=torch.long)
            n = len(keep)
            jpos.append(d["joint_pos"][idx]);  jvel.append(d["joint_vel"][idx])
            cpos.append(d["cube_pos"][idx]);   cquat.append(d["cube_quat"][idx]);  cvel.append(d["cube_vel"][idx])
            bpos.append(d["bowl_pos"][idx]);   bquat.append(d["bowl_quat"][idx]);  bvel.append(d["bowl_vel"][idx])
            # frac = 조작 구간 내 정규화 위치 [0,1] (0=SETTLE 시작, 1=RELEASE/RETREAT 직후)
            frac.append(torch.arange(n, dtype=torch.float32) / max(1, n - 1))
        if not jpos:
            print(f"[demo-reset] WARNING: no success demos in {self._demo_dataset_dir}", flush=True)
            return
        dev = self.device
        self._demo = {
            "jpos": torch.cat(jpos).to(dev),   "jvel": torch.cat(jvel).to(dev),
            "cpos": torch.cat(cpos).to(dev),   "cquat": torch.cat(cquat).to(dev),  "cvel": torch.cat(cvel).to(dev),
            "bpos": torch.cat(bpos).to(dev),   "bquat": torch.cat(bquat).to(dev),  "bvel": torch.cat(bvel).to(dev),
            "frac": torch.cat(frac).to(dev),
        }
        self._demo_loaded = True
        S = self._demo["frac"].shape[0]
        print(f"[demo-reset] loaded {len(jpos)} demos → {S} states from {self._demo_dataset_dir} "
              f"(subsample {k}, anneal_steps {self._demo_anneal_steps})", flush=True)

    def _bootstrap_demo(self, env_ids: torch.Tensor) -> None:
        """env_ids 중 demo_reset_prob 비율을 SM 데모 상태로 주입(robot joint + 4 cube + bowl).
        reverse curriculum: 진행도 p 에서 frac>=1-p 인 상태만 sample(초반=success 근처)."""
        if not self._demo_loaded or len(env_ids) == 0:
            return
        device = self.device
        mask = torch.rand(len(env_ids), device=device) < self._demo_reset_prob
        sel = env_ids[mask]
        if len(sel) == 0:
            return
        # reverse curriculum 후보 풀: frac >= threshold
        if self._demo_anneal_steps > 0.0:
            p = float(min(1.0, max(0.0, self.common_step_counter / self._demo_anneal_steps)))
        else:
            p = 1.0  # anneal 없음 = 전 구간 uniform
        thresh = 1.0 - p
        pool = (self._demo["frac"] >= thresh).nonzero(as_tuple=True)[0]
        if len(pool) == 0:
            pool = torch.arange(self._demo["frac"].shape[0], device=device)
        pick = pool[torch.randint(len(pool), (len(sel),), device=device)]

        robot = self.scene["robot"]
        origins = self.scene.env_origins[sel]  # (M,3)
        # robot joint state (pos+vel, 전 관절)
        robot.write_joint_state_to_sim(
            self._demo["jpos"][pick], self._demo["jvel"][pick], env_ids=sel)
        # 큐브 4개 pose(env-local→world)+vel
        for i, name in enumerate(CUBE_NAMES):
            cube = self.scene[name]
            pos = origins + self._demo["cpos"][pick, i]            # (M,3)
            quat = self._demo["cquat"][pick, i]                    # (M,4)
            cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=sel)
            cube.write_root_velocity_to_sim(self._demo["cvel"][pick, i], env_ids=sel)
        # 그릇 pose+vel
        bowl = self.scene[BOWL_NAME]
        bpos = origins + self._demo["bpos"][pick]
        bowl.write_root_pose_to_sim(torch.cat([bpos, self._demo["bquat"][pick]], dim=-1), env_ids=sel)
        bowl.write_root_velocity_to_sim(self._demo["bvel"][pick], env_ids=sel)
        # 부트스트랩 종류 기록(모니터 집계용): 4=demo-reset
        self.bootstrap_kind[sel] = 4

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
        # _get_gripper_pos(grasp 보상 기준점)과 **동일 공식** = jaw + 회전된 JAW_GRASP_OFFSET
        # (손가락 grasp point). 옛 중점 0.5*(jaw+gripper) 은 grasp_close 기준과 ~7cm 어긋나
        # 부트스트랩 큐브가 보상 지점 밖 → grasp_close/align=0(점화 레버 死). 정합 수정.
        if "jaw" in bn:
            j = bn.index("jaw")
            off = torch.tensor(JAW_GRASP_OFFSET, device=self.device, dtype=robot.data.body_pos_w.dtype)
            off = off.unsqueeze(0).expand(robot.data.body_pos_w.shape[0], -1)
            gp = robot.data.body_pos_w[:, j, :] + _quat_apply_wxyz(robot.data.body_quat_w[:, j, :], off)
        else:
            g = bn.index("gripper")
            gp = robot.data.body_pos_w[:, g, :]
        # env-local (고정베이스·동일로봇이라 env 간 동일) → env0 기준 (3,)
        self._grasp_offset = (gp - self.scene.env_origins)[0].clone()
        self._gripper_jid = list(robot.data.joint_names).index("gripper")
        try:
            self._wrist_roll_jid = list(robot.data.joint_names).index("wrist_roll")
        except ValueError:
            self._wrist_roll_jid = None

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
        # demo-reset(kind=4) 와 비중복: scratch(kind==0) env 에서만 grasp 부트스트랩.
        scratch = env_ids[self.bootstrap_kind[env_ids] == 0]
        if len(scratch) == 0:
            return
        mask = torch.rand(len(scratch), device=device) < prob
        sel = scratch[mask]
        if len(sel) == 0:
            return
        # 선택된 env 를 pre-grasp / full-grasp 로 분할 (pre-grasp 비율 = override 또는 anneal p)
        pre_frac = self._bootstrap_pregrasp_frac if self._bootstrap_pregrasp_frac >= 0.0 else p
        is_pre = torch.rand(len(sel), device=device) < pre_frac
        # 2-phase 배치: reset 시점 body_pos_w(FK)는 joint write 미반영이라 stale.
        # 여기서 큐브를 놓으면 팔이 settle 하며 ~5cm 어긋나 그리퍼가 빈 공간을 닫아 grip 실패
        # (rollout trace 로 확정 — 큐브 낙하·grasp_close 0). 그래서 지금은 pending 마킹만 하고,
        # FK 가 fresh 해지는 2 step 뒤 _apply_pending_grasp() 가 현재(settle된) jaw grasp point 에
        # 큐브 배치 + 그리퍼 close → 실제 grip 성립. 큐브는 그때까지 scatter 위치 유지
        # (1~2 step 과도기, episode 길이 512 대비 무시 가능). yaw 다양성은 일시 제거(점화 우선).
        self._grasp_pending[sel] = 2
        self._grasp_pending_pre[sel] = is_pre
        # 부트스트랩 종류 기록(모니터 집계용): pre-grasp=2, full-grasp=1
        self.bootstrap_kind[sel] = torch.where(
            is_pre,
            torch.full((len(sel),), 2, device=device, dtype=torch.long),
            torch.full((len(sel),), 1, device=device, dtype=torch.long),
        )

    def _apply_pending_grasp(self) -> None:
        """2-phase grasp 부트스트랩 배치 — FK(body_pos_w)가 fresh 해진 시점(reset 2 step 뒤)에
        실제 수행. 현재(settle된) jaw grasp point 에 큐브를 놓고 그리퍼 close → 그리퍼가 큐브를
        실제로 감싸 grip 성립. 이후 큐브가 jaw 를 따라간다. (reset 시 stale FK 로 놓으면 grip
        실패하는 문제 해소.)"""
        pend = self._grasp_pending
        ready = pend == 1
        self._grasp_pending = torch.clamp(pend - 1, min=0)
        if not bool(ready.any()):
            return
        sel = ready.nonzero(as_tuple=False).flatten()
        device = self.device
        is_pre = self._grasp_pending_pre[sel]
        robot = self.scene["robot"]
        cube = self.scene[CUBE_NAMES[0]]
        origins = self.scene.env_origins[sel]
        # 현재(fresh) jaw grasp point = jaw + 회전된 JAW_GRASP_OFFSET (grasp_close 보상 기준과 동일)
        bn = list(robot.data.body_names)
        if "jaw" in bn:
            jid = bn.index("jaw")
            joff = torch.tensor(JAW_GRASP_OFFSET, device=device, dtype=robot.data.body_pos_w.dtype)
            joff = joff.unsqueeze(0).expand(robot.data.body_pos_w.shape[0], -1)
            gp_all = robot.data.body_pos_w[:, jid, :] + _quat_apply_wxyz(
                robot.data.body_quat_w[:, jid, :], joff
            )
        else:
            gp_all = robot.data.body_pos_w[:, bn.index("gripper"), :]
        gp = gp_all[sel]  # world (M,3)
        quat = torch.zeros(len(sel), 4, device=device)
        quat[:, 0] = 1.0   # identity (yaw 0)
        full_pos = gp.clone()
        full_pos[:, 2] += self._bootstrap_lift
        pre_pos = gp.clone()
        pre_pos[:, 2] = origins[:, 2] + self._bootstrap_rest_z  # 책상 위 resting (env-local z)
        pos = torch.where(is_pre.unsqueeze(-1), pre_pos, full_pos)
        cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=sel)
        cube.write_root_velocity_to_sim(torch.zeros(len(sel), 6, device=device), env_ids=sel)
        # 그리퍼 joint: full-grasp=닫힘각(큐브 감쌈), pre-grasp=열림(받아들일 자세)
        jpos = torch.where(
            is_pre,
            torch.full((len(sel),), self._bootstrap_pregrasp_open, device=device),
            torch.full((len(sel),), self._bootstrap_close, device=device),
        )
        robot.write_joint_position_to_sim(
            jpos.unsqueeze(-1), joint_ids=[self._gripper_jid], env_ids=sel
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

    def _carry_reverse_curriculum(self, env_ids: torch.Tensor) -> None:
        """full-grasp 부트스트랩 env(든 큐브)의 그릇을 carry 경로 따라 이동(역커리큘럼).
        bowl_xy = lerp(pickup=grasp_offset_xy, 현재 random bowl xy, f), f=clamp(step/anneal_steps).
        f=0: 그릇이 든 큐브 바로 밑 → release 만으로 success(achievable=강한 terminal gradient).
        f=1: 정상 arc 위치(full transport). step 증가로 f 0→1 = 운반 거리 점진 확대 → backward 학습.
        그릇 rigid pose 는 write 즉시 .data 갱신(articulation FK staleness 무관)."""
        if self._grasp_offset is None or self._carry_rc_anneal_steps <= 0.0 or len(env_ids) == 0:
            return
        sel = env_ids[(self._grasp_pending[env_ids] > 0) & (~self._grasp_pending_pre[env_ids])]
        if len(sel) == 0:
            return
        step = float(getattr(self, "common_step_counter", 0))
        f = max(0.0, min(1.0, step / self._carry_rc_anneal_steps))
        bowl = self.scene[BOWL_NAME]
        origins = self.scene.env_origins[sel]
        bowl_pos = bowl.data.root_pos_w[sel].clone()      # (M,3) — rigid, fresh
        bowl_quat = bowl.data.root_quat_w[sel].clone()    # (M,4)
        pickup_xy = origins[:, :2] + self._grasp_offset[:2].unsqueeze(0)  # held cube world xy
        new_xy = pickup_xy + (bowl_pos[:, :2] - pickup_xy) * f
        bowl_pos[:, 0] = new_xy[:, 0]
        bowl_pos[:, 1] = new_xy[:, 1]
        bowl.write_root_pose_to_sim(torch.cat([bowl_pos, bowl_quat], dim=-1), env_ids=sel)
        bowl.write_root_velocity_to_sim(torch.zeros(len(sel), 6, device=self.device), env_ids=sel)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # 리셋되는 env 는 일단 scratch(0)로 초기화 — _bootstrap_grasp 가 선택된 env 만 1/2 로.
        self.bootstrap_kind[env_ids] = 0
        # offset 은 step() 에서 1회 캐시(첫 init reset 때는 body_pos_w 미확정일 수 있어
        # 캐시 전이면 부트스트랩 skip — _bootstrap_grasp 내부 가드).
        # demo-reset(RFCL) 우선 — 데모 run 에선 grasp/place bootstrap prob=0 으로 두고 이것만 사용.
        self._bootstrap_demo(env_ids)
        self._bootstrap_grasp(env_ids)
        self._bootstrap_place(env_ids)  # grasp 부트스트랩 후 남은 scratch env 중 일부에 place 부트스트랩
        self._carry_reverse_curriculum(env_ids)  # full-grasp env 그릇을 carry 경로 따라(easy→hard)
        # 그릇 초기 pose 저장(super()._reset_idx 안에서 randomize_bowl 이 적용된 직후 + RC 이동 후).
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
        self._task_progress_potential_prev[env_ids] = 0.0

    def step(self, action):
        if self._grasp_offset is None:
            self._cache_grasp_geom()
        # 2-phase grasp 부트스트랩: reset 2 step 뒤(FK fresh) 큐브 배치 + 그리퍼 close.
        self._apply_pending_grasp()
        if getattr(self.cfg, "dynamic_reset_gripper_effort_limit", False):
            dynamic_reset_gripper_effort_limit_sim(self)
        return super().step(action)
