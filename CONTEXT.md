<!-- ╔═══════════════════════════════════════════════════════════════════════╗
     ║  NORTH STAR — 매 세션/compaction 직후 먼저 읽는다. 변경 금지(상수).      ║
     ╚═══════════════════════════════════════════════════════════════════════╝ -->

## 🧭 North Star (불변 — 매 사이클·compaction 후 재확인)

- **마스터플랜**: [`docs/SIM2REAL_MASTERPLAN.md`](docs/SIM2REAL_MASTERPLAN.md) · **현황**: [`TASKS.md`](TASKS.md)
- **불변 계약**(모든 sim 데이터·정책 I/O가 일치해야 함): `v3.0` · robot_type `so_follower` · action/state 각 **6-dim joint position** (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) · `observation.images.{top,wrist,front}` 480×640×3 h264 **fps 30** · task `"pick up the pen and place it in the holder"`.
- **자율 계약**: Codex `/goal` 시작 후 **A~E 무인 자율**(묻지 않음). 멈추는 경우는 둘뿐 — F~G 실기기 경계 / 복구불가 블로커(동일 task 3회 재시도 후 우회·기록). 게이트 미통과 task는 done 금지.
- **복구 프로토콜**: 세션/compaction 직후 ① 마스터플랜 §0·§1·§7 → ② TASKS.md(현재 phase·in_progress·blocked) → ③ 아래 최근 인계 1~2개 순서로 재로드. 추측 금지 — 상태 파일에 없으면 새 task로.
- **머신**: GPU 중량(Isaac·RL·롤아웃·GR00T) = 서버 konan147(48GB), 산출물 `/DISK1/so101-sim2real`. 경량·실기기·오케스트레이터 = Windows. sync 허브 = `origin`(github PubCyBerry/SO101-Sim2Real).

---

## 작업 인계 (2026-06-04 — TC.1 rollout recorder 완료, 다음 TC.2)

- **목표**: TB.3 stochastic expert checkpoint를 3-camera render와 함께 rollout하고, 성공 episode만 LeRobot v3 데이터셋으로 기록하는 `scripts/sim/rollout_to_lerobot.py` recorder를 만든다.
- **상태**: 완료. 다음 actionable task는 **TC.2 200ep pipeline with DR + 3 cams + success filter**.
- **완료한 일**:
  - `scripts/sim/rollout_to_lerobot.py` 추가. Isaac AppLauncher + `RslRlVecEnvWrapper` + `OnPolicyRunner`로 checkpoint를 로드하고, 성공 episode만 `data/chunk-000/file-000.parquet`, `meta/info.json`, `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`, `meta/stats.json`, `videos/observation.images.{top,wrist,front}/chunk-000/file-000.mp4`에 기록한다.
  - North Star 계약에 맞춰 action/state는 6-dim SO-101 joint position으로 저장한다. sim radian 값은 real LeRobot 데이터셋 단위에 맞춰 arm 5축 rad→deg, gripper `×31.75`로 변환한다.
  - 기본 rollout 조건은 TB.3/TB.4 gate와 동일: `active_pens=1`, full pen/cup spawn scale 1.0, stochastic policy, `grasp_assist_distance=0.12`, offset `(0.03, 0.10, -0.05)`, `place_assist_distance=0.0`.
- **검증 결과(서버 canonical repo `/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 1ep smoke: `/DISK1/so101-sim2real/outputs/tc1_rollout_smoke_1ep_codex`, 1/1 success, 22 frames, 3cam mp4 생성, `validate_lerobot_schema.py` PASS.
  - TC.1 gate: `/DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452`, 10 successes / 15 attempts, failures 5 filtered, total 427 frames, dataset size 약 11MB, `validate_lerobot_schema.py` PASS.
- **검증 명령**:
  - Recorder: `UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac /home/konan147/.local/bin/uv run --group isaac --locked python scripts/sim/rollout_to_lerobot.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_curr12_no_place_offset_radius1_1024_20260604_0430/model_70.pt --output_dir /DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452 --episodes 10 --max_attempts 30 --max_episode_steps 450 --device cuda:0 --overwrite`
  - Validator: `UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac /home/konan147/.local/bin/uv run --group isaac --locked python scripts/validate_lerobot_schema.py /DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452`
- **주의/다음**:
  - TC.1 recorder는 현재 `num_envs=1`만 지원한다. top/front world-absolute camera 때문에 병렬 env 카메라 정합은 TC.2에서 env-relative 전환하거나, 우선 serial 200ep로 관통 후 병렬화한다.
  - 생성 성공률은 rollout 중 10/15 attempts였다. TC.2는 `max_attempts`를 넉넉히 두고 success filter 정상 동작을 계속 기록한다.
  - `scripts/author_pick_pen_scene.py`는 사용자 추가 untracked 참고 파일로 남아 있으며 이번 커밋에 포함하지 않는다.

---

## 작업 인계 (2026-06-04 — TB.3/TB.4 완료, 다음 TC.1)

- **목표**: TB.3 state-based RL expert + TB.4 spawn/cup curriculum 확대를 통과시키고, Phase C rollout recorder로 넘어간다.
- **상태**: 완료. 다음 actionable task는 **TC.1 `scripts/sim/rollout_to_lerobot.py` recorder**.
- **핵심 결정**:
  - North Star task string이 singular(`"pick up the pen..."`)이므로 TB.3/TB.4 gate는 active target 1개(`active_pens=1`) + 나머지 펜 distractor로 해석한다.
  - default PhysX contact grasp 대신 TB.3용 `soft_grasp_assist`를 사용하되, 최종 gate에서는 `place_assist_distance=0.0`으로 place snap을 끈다.
  - gripper body origin과 실제 pen center가 맞지 않아 cup insertion이 막히던 문제를 world-frame assist offset `(x=0.03, y=0.10, z=-0.05)`로 보정했다.
- **완료한 일**:
  - `place_height_pen` dense reward 추가. transport 이후 cup XY 근처에서 target z로 낮추는 신호를 제공한다.
  - `apply_curriculum()`/`train.py`/`eval_success.py`에 `grasp_assist_offset_x`, `grasp_assist_offset_y`, `grasp_assist_offset_z`를 노출했다.
  - 서버 random FK probe로 reset 기준 gripper/cup/pen 위치 확인: gripper→cup XY 약 `0.1625m`, gripper→pen 약 `0.2526m`. cup 근처 feasible gripper pose는 body origin 기준 cup center에서 약 10cm 어긋나 offset 보정이 필요했다.
- **검증 결과(서버 canonical repo `/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. reward term 11개(`place_height_pen` 포함), stage check 전부 pass.
  - curriculum run `tb3_curr11_no_place_offset_radius15_1024_20260604_0424`: `model_20.pt`가 cup_radius_scale 1.5/full spawn에서 stochastic 128/128 통과.
  - final run `tb3_curr12_no_place_offset_radius1_1024_20260604_0430`: `model_70.pt`가 fixed 정상 radius stochastic 128/128, full spawn/cup 정상 radius stochastic 128/128 통과.
  - 공식 gate 명령은 `eval_success.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_curr12_no_place_offset_radius1_1024_20260604_0430/model_70.pt --num_envs 64 --episodes 128 --max_episode_steps 450 --active_pens 1 --pen_radius_scale 1.0 --cup_angle_scale 1.0 --cup_radius_scale 1.0 --grasp_assist --grasp_assist_distance 0.12 --grasp_assist_offset_x 0.03 --grasp_assist_offset_y 0.10 --grasp_assist_offset_z -0.05 --place_assist_distance 0.0 --init_noise_std 0.2 --stochastic --min_success_rate 0.7` → success_rate `1.0`, exit code 0.
- **Residual risk**:
  - 같은 full spawn/cup 조건 deterministic eval은 `58/128`, success_rate `0.4531`. Phase C는 stochastic rollout + success filtering으로 진행한다.
  - `grasp_assist`는 TB.3 학습/rollout용 보조 event다. 실기기 F~G나 contact-realism 평가로 착각하지 않는다.
- **변경한 파일(아직 커밋 전)**: `TASKS.md`, `CONTEXT.md`, `scripts/environments/reward_smoke.py`, `scripts/reinforcement_learning/{train.py,eval_success.py}`, `src/sim_to_real/tasks/pick_pen/{pick_pen_env_cfg.py,mdp/rewards.py}`.

---

## 작업 인계 (2026-06-04 — TB.3 curriculum assist subgate 통과, final gate 진행 중)

- **목표**: TB.3 — state-based PPO 전문가를 success_rate ≥ 0.7까지 끌어올린다. 현재는 최종 full/default gate가 아니라 curriculum 보조 subgate를 통과한 상태.
- **상태**: 진행 중. 2048-env default full 학습은 false grasp/zero lift로 실패했고, TB.4 성격의 curriculum/assist를 앞당겨 성공 rollout이 나오는 최소 조건을 확보했다.
- **완료한 일**:
  - `grasp_bonus`가 tabletop 근처 false grasp를 주지 않도록 lift 조건을 추가했다.
  - `carry_pen` dense reward를 추가하고 reward weight를 grasp 1 / carry 4 / transport 8 / insert 25 / release 10 / success 100으로 재조정했다.
  - `apply_curriculum()` 추가: `active_pens`, pen ellipse radius, cup arc angle, cup success radius, episode length, grasp/place assist를 train/eval에서 공통 적용.
  - `soft_grasp_assist` event 추가: 닫힌 gripper 근처 target pen을 따라오게 하고, 선택적으로 cup 근방에서 place snap을 수행한다. 기본 env에서는 비활성이다.
  - `train.py`/`eval_success.py`에 curriculum, resume, stochastic eval, noise/lr/entropy CLI를 추가했다. `train.py`의 latest checkpoint 정렬은 `model_<n>.pt` 숫자 기준으로 보정.
- **검증 결과(서버 temp repo `/DISK1/so101-sim2real/work/tb3_grasp_assist_20260604_030539/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 로컬/서버 `python -m py_compile ...` 통과, `git diff --check` 통과.
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. reward term 10개(`carry_pen` 포함), stage check 전부 pass.
  - `train.py --num_envs 64 --max_iterations 3 --num_steps_per_env 12 --save_interval 1 --active_pens 1 --pen_radius_scale 0 --cup_angle_scale 0 --grasp_assist --place_assist_distance 0.18 ...` 통과. `soft_grasp_assist` interval event 등록, 최신 checkpoint `model_2.pt` 정상 산출.
  - subgate eval: `model_8.pt` from `/DISK1/so101-sim2real/outputs/tb3_curr7_1pen_placeassist_denseckpt_1024_20260604_0334/model_8.pt`, stochastic, active target 1개, fixed spawn/cup, `place_assist_distance=0.22`, normal cup radius에서 `128/128`, success_rate `1.0`, `--min_success_rate 0.7` 통과.
- **남은 일**:
  - TB.3는 아직 `done` 금지. 위 결과는 assisted/stochastic/fixed curriculum subgate일 뿐이다.
  - 다음 루프는 `place_assist_distance 0.22 → 0.18 → 0.12 → 0.0`, `pen_radius_scale/cup_angle_scale 0 → 0.25 → 0.5 → 1.0`, active target 일반화 순서로 확장한다.
  - 최종 gate는 `eval_success.py --min_success_rate 0.7`을 기본 성공 판정에 가깝게 통과해야 한다.
- **주의**:
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 untracked 참고 파일이다. 이번 TB.3 커밋에는 포함하지 않는다.
  - 카메라 정합을 다시 Claude worker에게 맡길 때는 `claude-opus-4-8[1m]`, effort high, `PowerShell` 없는 allowlist를 사용한다. 지시에는 `docs/pics` 사무실 사진 참고, top camera는 사무실 사진보다 높게 조정된 점, 각 카메라 pose/angle/FOV는 실제 dataset 영상 `observation.images.top`, `observation.images.wrist`, `observation.images.front`를 기준으로 맞출 것을 반드시 포함한다.

---

## 작업 인계 (2026-06-04 — TB.3 RL state/eval 준비 완료, full 학습 진행 중)

- **목표**: TB.3 — 2048–4096 env state-based PPO 전문가를 full 학습하고 `eval_success.py` success_rate ≥ 0.7(목표 0.9)를 달성한다.
- **상태**: 진행 중. `rl_policy`/eval/스케일 smoke는 완료했고, 다음은 full PPO train 실행 및 주기적 eval.
- **완료한 일**:
  - `policy` observation group은 North Star 계약대로 6-dim joint state를 유지.
  - `rl_policy` observation group 추가. `task_mdp.rl_state`가 37-dim privileged state(6 joint + gripper pos + pen/cup pos + gripper→pen vectors + gripper open fraction)를 제공한다.
  - `scripts/reinforcement_learning/train.py` 기본 obs group을 `rl_policy`로 전환하고 `--obs_group`, `--critic_obs_group` CLI를 추가했다.
  - `scripts/reinforcement_learning/eval_success.py` 추가. rsl_rl checkpoint를 로드해 closed-loop episode를 돌고 timeout을 success로 세지 않는다.
  - 2048 env scale smoke에서 PhysX `totalAggregatePairsCapacity` 부족 오류를 확인하고 `gpu_total_aggregate_pairs_capacity = 64 * 1024`로 보정. `docs/TROUBLESHOOTING.md`에 기록.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `train.py --num_envs 4 --max_iterations 4 --num_steps_per_env 25` 통과. actor/critic input `37`, checkpoint `model_3.pt` 생성.
  - `eval_success.py --checkpoint .../tb3_train_state_smoke_codex/model_3.pt --episodes 4 --max_episode_steps 120` 통과(success_rate 0.0, smoke checkpoint라 정상).
  - `env_smoke.py --steps 500 --num_envs 1` 통과(`policy_obs_shape [1,6]`, `rl_policy shape (37,)` 등록 확인).
  - `train.py --num_envs 2048 --max_iterations 2 --num_steps_per_env 24` 통과(total_steps 98,304, checkpoint `model_1.pt`). capacity 64k 적용 후 `totalAggregatePairsCapacity` 오류 없음.
- **다음 실행 후보**:
  - full PPO train: `train.py --num_envs 2048 --max_iterations 1500 --num_steps_per_env 24 --save_interval 50 --run_name tb3_full_2048 --checkpoint_dir /DISK1/so101-sim2real/outputs/tb3_full_2048`
  - eval: `eval_success.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_full_2048/model_1499.pt --num_envs 64 --episodes 200 --max_episode_steps 900 --min_success_rate 0.7`
- **주의**: 짧은 랜덤/초기 학습은 reach 보상만 조금 뜨고 grasp/lift 이후는 0이다. full 학습 실패 시 TB.4 커리큘럼을 앞당기거나 reward/episode horizon을 조정해야 한다.

---

## 작업 인계 (2026-06-04 — TB.2 rsl_rl PPO train wrapper 완료)

- **목표**: TB.2 — `SimToReal-SO101-PickPen-v0`를 rsl_rl PPO로 학습할 수 있는 `scripts/reinforcement_learning/train.py` 래퍼를 추가하고, 100-step 이상 smoke와 checkpoint 저장을 검증한다.
- **상태**: 완료. 다음 actionable task는 TB.3(RL 전문가 full 학습, 2048–4096 env, 카메라 off).
- **완료한 일**:
  - `scripts/reinforcement_learning/train.py` 추가. Isaac `AppLauncher` headless, `parse_env_cfg` → `gym.make` → `RslRlVecEnvWrapper` → `OnPolicyRunner` 순서로 실행.
  - CLI: `--task`, `--num_envs`, `--device`, `--rl_device`, `--seed`, `--max_iterations`, `--num_steps_per_env`, `--save_interval`, `--experiment_name`, `--run_name`, `--log_root_path`, `--checkpoint_dir`, `--clip_actions`.
  - 기본 PPO cfg는 6-dim policy obs/critic obs(`obs_groups={"policy":["policy"],"critic":["policy"]}`), ActorCritic `[128,128]` ELU, PPO 2 epochs/1 minibatch의 smoke-friendly 설정.
  - `--checkpoint_dir`가 지정되면 해당 디렉터리를 log dir로 사용하고, 이번 실행 시작 이후 생성/갱신된 `model_*.pt`가 없으면 실패 처리.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 로컬 `python -m py_compile scripts/reinforcement_learning/train.py` 통과, `git diff --check` 통과.
  - `train.py --num_envs 4 --max_iterations 4 --num_steps_per_env 25 --save_interval 1 --checkpoint_dir /DISK1/so101-sim2real/outputs/tb2_train_smoke_codex_final` 통과. 총 400 env-step, latest checkpoint `/DISK1/so101-sim2real/outputs/tb2_train_smoke_codex_final/model_3.pt`.
- **참고**:
  - Claude worker는 `sonnet[1m]`, effort high, `PowerShell` 없는 allowlist로 호출했고 초안/서버 smoke를 완료했다. Codex가 checkpoint freshness와 env seed 반영을 보완 후 재검증했다.
  - smoke reward 로그는 짧은 랜덤 rollout이라 stage reward가 대부분 0이다. TB.3는 학습 스케일/커리큘럼/평가 기준을 별도로 잡아야 한다.

---

## 작업 인계 (2026-06-04 — TB.1 단계형 reward 완료)

- **목표**: TB.1 — state-based RL 전문가용 단계형 reward(reach→grasp→lift→transport→insert→release + success + action-rate/joint-vel 페널티)를 구현하고 Isaac Lab 2.3.2 GPU smoke로 검증한다.
- **상태**: 완료. 다음 actionable task는 TB.2(`scripts/reinforcement_learning/train.py` rsl_rl PPO train 래퍼).
- **완료한 일**:
  - `src/sim_to_real/tasks/pick_pen/mdp/rewards.py` 추가. contact sensor 없이 `RigidObject.root_pos_w`, robot `gripper` body pose, gripper joint position으로 7개 stage reward를 계산하며 모두 `(num_envs,)` finite tensor를 반환.
  - `PickPenRewardsCfg`를 reward stub에서 9개 term(`reach_pen`, `grasp_pen`, `lift_pen`, `transport_pen`, `insert_pen`, `release_pen`, `task_success`, `action_rate`, `joint_vel`)으로 교체.
  - 기존 `pen_in_cup`/`task_done`의 기본 컵 중심이 stale `(-0.18, 0.43)`이고 z 기준이 0 기준이던 문제를 현재 scene 좌표 `(2.2, -0.17)` + desk top `0.92` 기준으로 보정.
  - `scripts/environments/reward_smoke.py` 추가. Isaac AppLauncher로 headless env를 띄운 뒤 reward term 등록, shape/finite, stage별 독립 baseline→target 증가를 검증.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. 9개 reward term 등록, reach/grasp/lift/transport/insert/release/success 모두 증가, failures `[]`.
  - `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0).
  - `drive_response_smoke.py --num_envs 1 --device cuda:0` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **참고**:
  - Claude worker는 `sonnet[1m]`, effort high, `PowerShell` 없는 allowlist로 호출해 초안 구현을 받았고, Codex가 z/컵 기준과 deterministic smoke를 보완했다.
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 untracked 참고 파일로 남겨둠. 이번 TB.1 커밋에 포함하지 않는다.

---

## 작업 인계 (2026-06-04 — TA.3 camera 정합 완료)

- **목표**: TA.3 — `SimToReal-SO101-PickPen-v0`의 top/front/wrist 카메라가 North Star 계약(`observation.images.{top,wrist,front}`, 480×640×3, 30fps)과 실제 데이터셋 구도에 맞게 렌더되는지 검증한다.
- **상태**: 완료. 다음 actionable task는 TB.1(단계형 reward 구현).
- **완료한 일**:
  - `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`: 로봇 floating 수정. `so101_follower.usd` base bbox 최하단(local z≈0.0301)을 반영해 `_ROBOT_POS.z`를 `0.92` → `0.889`로 낮춤.
  - 카메라를 `PickPenSceneCfg` 기본 필드에서 제거하고 `make_pick_pen_camera_cfgs()` / `add_pick_pen_cameras(scene_cfg)` optional injection으로 분리. 따라서 기본 `env_smoke.py`는 `--enable_cameras` 없이 계속 동작.
  - top/front/wrist 포즈/FOV를 `datasets/pick_pen/videos/observation.images.{top,front,wrist}` 프레임과 `docs/pics/사무실_사진_*`, `펜통_*` 사진을 참고해 조정. 단 top camera는 사용자 지시대로 사무실 사진보다 더 높은 실제 dataset top view를 우선.
  - `front_camera`: 기존 detached side view를 폐기하고 로봇 전면 근처 낮은 장착 위치로 재배치.
  - `wrist_camera`: `{ENV_REGEX_NS}/Robot/gripper/WristCamera`로 gripper 링크 자식 prim에 부착. rest 자세 기준 컵/매트 근접 광각뷰로 조정.
  - `scripts/environments/camera_shape_smoke.py`: camera injection 후 5-step warmup, 3캠 RGB shape/intrinsics/FOV/pose JSON 출력, optional PNG preview 저장.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, GPU `cuda:0`)**:
  - `camera_shape_smoke.py --save-dir /DISK1/so101-sim2real/outputs/ta3_camera/opus_fix5` 통과. top/front/wrist 모두 `[1,480,640,3]`, dtype `torch.uint8`; FOV: top 66.44°, front 73.62°, wrist 92.67°.
  - `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0). 카메라 없는 기본 env 경로 복구 확인.
  - `drive_response_smoke.py --num_envs 1 --device cuda:0` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **참고**:
  - Claude worker는 사용자 지시대로 `claude-opus-4-8[1m]`, effort high, `PowerShell` 없는 allowlist로 호출했으나 30분 타임아웃. 부분 구현을 Codex가 직접 검토·수정·검증함.
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 과거 author script로 읽고 좌표 문맥 참고만 했다. 이번 TA.3 커밋 범위에는 포함하지 않음.
  - 멀티-env 카메라(TC.2)는 top/front world absolute pose를 env-relative로 전환해야 한다.

---

## 작업 인계 (2026-06-03 — TA.2 scene spawn/physics 검증 완료)

- **목표**: TA.2 — 펜 4개와 펜컵이 reset 100회 동안 의도 영역(펜=타원, 펜컵=호)에 100% 들어오고, settle 후 관통·바운스 없이 안정적인지 기계 검증한다.
- **상태**: 완료. 다음 actionable task는 TA.3(카메라 3대 extrinsic/intrinsic 실기 정합, 480×640@30 렌더 shape/FOV 점검).
- **완료한 일**: `scripts/environments/scene_physics_smoke.py` 추가. 순수 Isaac Lab `RigidObjectCfg(spawn=None)`가 USD authored pose 대신 원점 default를 잡는 문제를 `RigidObjectCfg.InitialStateCfg`로 보정. 펜 4개 USD는 visual collider를 끄고 invisible `CollisionBox` physics proxy만 사용하도록 분리했으며 damping/sleep threshold를 reset 안정성에 맞게 높임. `.usda` 수정 후 `.usd` 바이너리도 재-export.
- **검증 결과**: 서버 `/DISK1/so101-sim2real/work/ta.2/repo` Isaac venv에서 `scene_physics_smoke.py --resets 100 --settle-steps 30 --num_envs 1 --device cuda:0` 통과(spawn ellipse/arc pass, y min spawn 0.09713 m, y min settled 0.09732 m, max z drop 0.001 m, max xy drift 0.04419 m, max lin vel 0.0098 m/s, max ang vel 1.13728 rad/s). `env_smoke.py --steps 500` 통과(action/policy obs `[1,6]`, resets 0). `drive_response_smoke.py` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **기록**: `docs/TROUBLESHOOTING.md`에 `RigidObject` reset sampling 원점 밀림과 원형 pen collider rolling 실패/해결 항목 추가.
- **주의**: Claude worker 호출 allowlist에는 `PowerShell`을 넣지 않는다.

---

## 작업 인계 (2026-06-03 — TA.1 SO-101 PD drive tuning 완료)

- **목표**: TA.1 — SO-101 articulation의 position PD drive를 Feetech STS3215 근사로 튜닝하고, 정적 hold 및 step 응답 무진동 검증을 통과시킨다.
- **상태**: 완료. 다음 actionable task는 TA.2(펜 4개·펜컵 spawn 영역·물리 검증).
- **완료한 일**: SO-101 robot spawn에 `ArticulationRootPropertiesCfg(fix_root_link=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1)` 적용. actuator를 arm/gripper로 분리하고 Isaac Lab 2.3.2의 `effort_limit_sim`/`velocity_limit_sim` 사용. PhysX `enable_external_forces_every_iteration=True`, render interval=decimation 설정. 신규 `scripts/environments/drive_response_smoke.py` 추가.
- **검증 결과**: 로컬 `py_compile` 통과, deprecated actuator field 잔재 0건. 서버 `/DISK1` Isaac venv에서 `drive_response_smoke.py --num_envs 1 --device cuda:0` 통과(hold tail max pos 0.02102 rad, tail RMS vel 0.0 rad/s, step final err max 0.01882 rad, overshoot max 0.01882 rad). 서버 `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0).
- **기록**: `docs/TROUBLESHOOTING.md`에 fixed-root 누락으로 hold velocity가 남는 사례를 추가.
- **주의**: Claude worker 호출 allowlist에는 `PowerShell`을 넣지 않는다.

---

## 작업 인계 (2026-06-03 — T0.3 de-leisaac sim-critical 완료)

- **목표**: T0.3 — `src/sim_to_real/tasks/pick_pen`를 순수 Isaac Lab `ManagerBasedRLEnvCfg` 기반으로 재작성하고 sim-critical `leisaac` import를 0건으로 만든다.
- **상태**: 완료. 다음은 TA.1(SO-101 articulation position PD drive tuning).
- **완료한 일**: `pick_pen_env_cfg.py`를 순수 Isaac Lab 2.3.2 `ManagerBasedRLEnvCfg`로 재작성. `InteractiveSceneCfg` + SO-101 `ArticulationCfg` + 펜 4개/펜컵 `RigidObjectCfg` + 6-dim `JointPositionActionCfg` + 6-dim policy obs + minimal reward/event/termination 구성. `pen_desk.py`는 repo-local asset path로 전환. Direct env는 pure DirectRLEnv 재작성 전까지 등록 보류. 신규 `scripts/environments/env_smoke.py` 추가.
- **검증 결과**: `python -m py_compile ...` 통과, `rg "leisaac" src/sim_to_real/tasks/pick_pen src/sim_to_real/assets/scenes/pen_desk.py` 0건, 서버 `/DISK1` Isaac venv에서 `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action_shape `[1,6]`, policy_obs_shape `[1,6]`, resets 0).
- **주의**: 물리/drive 품질은 smoke 통과 수준이다. 실제 안정성·진동·토크/속도 제한 튜닝은 TA.1에서 수행.

---

## 작업 인계 (2026-06-03 — T0.2 서버 Isaac 설치/의존성 전환 완료)

- **목표**: T0.2 — 서버 `konan147`에 user-local `uv`를 준비하고, `leisaac`를 제거한 순수 Isaac Sim/Isaac Lab 2.3.2 의존성으로 전환한 뒤 headless smoke를 통과시킨다.
- **상태**: 완료. 다음은 T0.3(de-leisaac sim-critical 코드 재작성).
- **완료한 일**: `pyproject.toml`/`uv.lock`에서 `leisaac` 의존성과 source 제거, `isaacsim[all,extscache]==5.1.0` + `isaaclab[all,isaacsim]==2.3.2` 직접 의존으로 전환, `validation = ["ovphysx"]` 보존. 서버에 user-local `uv 0.11.18` 설치. `/DISK1/so101-sim2real/venvs/isaac`에 sync 완료(약 19G).
- **보완한 일**: Isaac Lab pip layout에서 `isaaclab.envs` 경로가 빠지는 문제와 `SimulationApp` 전 `omni.*` import 문제를 `src/sim_to_real/__init__.py`에서 T0.3 전용 deferred import로 처리. Claude worker allowlist에서 `PowerShell` 제거(`loop.py`, `dispatch.sh`, 마스터플랜 반영).
- **검증 결과**: `uv lock --check` 통과, `rg "leisaac" pyproject.toml uv.lock` 0건, 서버 `uv sync --group isaac --python 3.11 --locked` 통과, 서버 `uv run python -c 'import isaacsim; import isaaclab; import sim_to_real; print(123)'` 통과, 서버 `isaaclab 2.3.2` 확인, `python -m py_compile src/sim_to_real/__init__.py scripts/orchestrator/loop.py` 및 `bash -n scripts/orchestrator/dispatch.sh` 통과.
- **다음**: T0.3 — `src/sim_to_real/tasks/pick_pen`의 sim-critical leisaac import 제거 및 순수 Isaac Lab env smoke 작성.

---

## 작업 인계 (2026-06-03 — T0.0/T0.1 착수 보완 계획 구현)

- **목표**: 보완 계획을 마스터플랜/TASKS에 반영하고, 실제 부트스트랩 일부(T0.0 preflight, T0.1 validator)를 수행.
- **상태**: T0.0·T0.1·T0.4 완료. 다음은 T0.2(서버 user-local `uv` 설치 + leisaac 제거/Isaac direct dependency 전환).
- **완료한 일**:
  - 로컬 remote `konan` 제거, 로컬/서버 `origin`을 `https://github.com/PubCyBerry/SO101-Sim2Real.git`로 표준화.
  - 서버 repo clean 확인. 서버 tool 확인: `claude`, `docker`, `nvidia-smi`, `gh`, `jq`, `yq` 있음. `uv`는 없음(T0.2 설치 항목).
  - 사용자가 `/DISK1/so101-sim2real` 권한을 수정했고, Codex가 `test -w /DISK1/so101-sim2real` 성공을 확인해 T0.0을 done 처리.
  - Claude worker로 `scripts/validate_lerobot_schema.py` 작성 후 Codex가 직접 재검증.
  - 마스터플랜에 RELOAD 범위(§0·§1·§7), 복구불가 3회, worker JSON 인터페이스, `/DISK1/so101-sim2real/run/gpu.lock`, T0.5→T0.2 흡수 반영.
  - `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}` 추가. 로컬 dry run은 WSL 없이 Python subprocess가 `claude.exe --model "sonnet[1m]" --effort high`를 직접 호출하고, `dispatch.sh`는 SSH/Unix 래퍼로 유지.
  - Claude worker tool allowlist 기본값을 `Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow`로 고정.
- **검증 결과**:
  - `python scripts/validate_lerobot_schema.py datasets/pick_pen` 통과.
  - `python scripts/validate_lerobot_schema.py --self-test` 통과.
  - `python -m py_compile scripts/validate_lerobot_schema.py` 통과.
  - `python scripts/orchestrator/gate.py validate-lerobot-schema` 통과.
  - `ssh konan147 'test -w /DISK1/so101-sim2real'` 통과.
  - `python scripts/orchestrator/loop.py dry-run-t0.1` 통과(Claude DISPATCH `--model "sonnet[1m]" --effort high` + 지정 tool allowlist → worker JSON → Codex VERIFY). Claude `modelUsage`는 `claude-sonnet-4-6[1m]`, `contextWindow=1000000`으로 확인.
- **블로커**: 없음. `uv`는 아직 서버 PATH에 없지만 T0.2의 user-local 설치 항목으로 처리.
- **변경한 파일**: `docs/SIM2REAL_MASTERPLAN.md`, `TASKS.md`, `CONTEXT.md`, `scripts/validate_lerobot_schema.py`, `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}`. 기존 dirty `pyproject.toml`의 `validation = ["ovphysx"]` 변경은 보존(T0.2 소유).
- **다음**: T0.2(uv 설치 + leisaac 제거/Isaac direct dependency 전환).

---

## 작업 인계 (2026-06-03 — Sim2Real 자율 개발 마스터플랜 수립)

- **목표**: 장기 무인 자율 개발 계획(Codex→Claude 오케스트레이션) 수립 + Codex `/goal` 인계 파일 작성.
- **상태**: 완료(계획·인계 파일). 자율 개발 자체는 미시작 — Codex `/goal`이 부트스트랩(T0.0~)부터 구동.
- **확정 결정 5개**: ①Codex(플래너)→Claude Code CLI(워커) 디스패치 ②서버에 Isaac Sim 5.1 headless 설치 ③시뮬 A~E 무인 자율, F~G 사용자 게이트 ④CONTEXT.md+TASKS.md(git) 상태관리 ⑤**leisaac 전면 제거→순수 Isaac Sim 5.1.0 + Isaac Lab 2.3.2 재구현**.
- **신규 파일**: `docs/SIM2REAL_MASTERPLAN.md`(불변 계획), `TASKS.md`(Phase 0~G 체크리스트), 본 North Star 블록. (미커밋)
- **조사 근거(이번 세션)**: 서버 konan147 = RTX PRO 5000 Blackwell 48GB·RAM 125GB·`/DISK1` 3.4TB 여유, Isaac/uv 미설치, Docker 이미지 3개 빌드됨, 레포 클론 `~/Workspaces/SO101-Sim2Real`. leisaac 결합 8파일(base cfg·device·subasset·recorder) — 단 teleop device는 A~E 자율 트랙엔 불필요(연기). 현 데이터셋 50ep=333MB(ep당 ~6.7MB) → 롤아웃 5k ep≈35GB.
- **다음**: 사용자가 신규 파일 검토 후 Codex에 `/goal docs/SIM2REAL_MASTERPLAN.md` 로 인계. 부트스트랩 첫 task = T0.0(git sync 단일화)·T0.1(validator).

---

## 작업 인계 (2026-06-03 — Claude Code 실행 probe)

- **목표**: Codex가 계획을 세우고 Claude Code CLI에 간단한 구현을 지시할 수 있는지 확인.
- **상태**: 완료.
- **Claude Code 확인**:
  - 실행 파일: `C:\Users\taehunkim\.local\bin\claude.exe`
  - 버전: `2.1.161 (Claude Code)`
  - 프로젝트 설정: `.claude/settings.json`, `.claude/settings.local.json` 없음.
  - 사용자 전역 설정 요약: `model=sonnet[1m]`, `effort=null`, `permissionMode=null`.
  - probe 실행 플래그: `--model sonnet --effort low --permission-mode bypassPermissions --output-format json --no-session-persistence --tools Read,Write,Edit,MultiEdit --allowedTools Read,Write,Edit,MultiEdit`.
  - 실행 결과 JSON의 실제 modelUsage: `claude-sonnet-4-6`.
  - debug log: `outputs/claude_code_probe/claude_debug.log` 에서 `dispatching to firstParty model=claude-sonnet-4-6`, `tool=Write` 확인.
- **Claude에게 지시한 구현**: `outputs/claude_code_probe/joint_summary.py` 생성. JSON joint sample list를 읽어 `timestamp` 제외, numeric joint별 min/max/mean/count/joint_order 출력. `--self-test` 포함. 표준 라이브러리만 사용.
- **검증 결과**:
  - `python outputs/claude_code_probe/joint_summary.py --self-test` 성공.
  - 정상 JSON stdin 요약 성공.
  - invalid JSON, non-list input 모두 stderr 출력 + exit code `2`.
  - `outputs/` 는 `.gitignore` 대상이라 probe 산출물은 git tracked diff 없음.
- **변경한 파일**: `CONTEXT.md` 갱신. 산출물은 ignored 경로 `outputs/claude_code_probe/`.
- **남은 일**: 없음.

## 작업 인계 (2026-06-02 — SmolVLA 카메라 setup 수정 + GR00T modality 점검)

- **GR00T 점검 결과(확실)**: lerobot `GrootPolicy` 경로는 `modality.json` 불필요(소스 참조 0건), 카메라 ≥1개·이름 자유·개수 무제한(`configuration_groot.py:132` input_features VISUAL 동적 수집). 블로그의 modality.json/`so100_dualcam`(2-cam)/`wrist`·`front` 강제는 NVIDIA `Isaac-GR00T` 네이티브(`gr00t_finetune.py`) 전용. 사용자의 3-cam(wrist/front/top) 배포가 정상인 이유.
- **SmolVLA 실버그 발견·수정**: `lerobot/smolvla_base` config(HF에서 확인)가 input_features 로 `observation.images.camera1/2/3` 명시(chunk_size=50). `make_policy`(`factory.py:512`)는 `--policy.path` 시 pretrained input_features 를 데이터셋 키로 덮어쓰지 않음 → wrist/front/top 데이터셋과 mismatch. 기존 `RENAME_MAP` 기본 빈값이라 **SmolVLA train 이 실패하는 상태였음**(GR00T만 검증됐던 탓). 
  - 수정: `env/smolvla.env` 에 `RENAME_MAP` 추가(논문 표준 슬롯 top→camera1, wrist→camera2, front→camera3). `env/groot.env` 는 빈값+이유 주석. `.env`/`.env.example` §5 의 "자동 생성" 오기 제거(프로필로 이관).
  - 추론 물리 매핑도 학습과 동일하게 통일: camera1=top, camera2=wrist, camera3=front. PATH_A §6(train에 --rename_map 추가)·§7, PATH_B §12, TROUBLESHOOTING 1022 의 옛 순서(wrist→camera1)·"자동 생성" 표현 전부 정정.
  - 검증: `docker compose config` 로 smolvla 프로필 RENAME_MAP JSON 주입 + groot 빈값 확인.
- **미반영(미검증)**: article 의 SmolVLA base normalization-key 버그(from_pretrained 시 obs 미정규화로 팔 떨림)는 0.5.x 재현 미확인 → 문서에 안 넣음. 사용자에게 watch-out 으로만 전달.
- 변경: `env/{smolvla,groot}.env`·`.env`·`.env.example`·`docker/policy-entrypoint.sh`·`docs/PATH_A_NATIVE.md`·`docs/PATH_B_DOCKER.md`·`docs/TROUBLESHOOTING.md`. (미커밋)

---

## 작업 인계 (2026-06-02 — 모델 프로필 + 중복인자 정리 + 직접추론 문서)

- **모델 프로필 방식 도입**: 모델별 변수 10개를 `env/<name>.env`(groot.env / smolvla.env)로 분리. `.env` 의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. compose 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE:-groot}.env]` (나중 파일 override). 두 서비스 모두 적용.
  - 검증: `docker compose config` 로 groot 주입 + `${HF_USER}` 보간(taehunkim/...) + `POLICY_PROFILE=smolvla` 셸 오버라이드 전부 정상.
  - `env/` 는 사용자가 .gitignore 에서 제외 → 추적됨. (docker/profiles/ 에 먼저 만들었다가 env/ 로 이동함.)
  - OUTPUT_DIR 은 .env 에서 제거, entrypoint 가 `outputs/train/${JOB_NAME:-run}` 로 파생(프로필 JOB_NAME 따라감).
- **중복 인자 정리**: POLICY_CLIENT_FPS 제거 → 서버·클라가 POLICY_FPS 공유(lerobot-entrypoint 가 POLICY_FPS 읽음). POLICY_TYPE/TRAIN_POLICY_TYPE 은 의미가 달라(클라 항상 필요 vs train 의 path/type 스위치) 병합 안 함 — 설명만.
- **policy.type vs policy.path** (HF 문서 확인): SmolVLA = `--policy.path=lerobot/smolvla_base`(LeRobot 체크포인트 포맷). GR00T = `--policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.5-3B`(NVIDIA native 포맷이라 path 불가). 차이는 scratch-vs-pretrained 아니라 **체크포인트 포맷**. 내가 학습한 체크포인트는 둘 다 LeRobot 포맷 → 재학습 시 --policy.path.
- **직접 추론(서버 없이) 문서화**: `lerobot record --policy.path=<model>` (HF 권장). lerobot-entrypoint `record` 모드가 `shift`+`"$@"` 로 추가 CLI(예 --policy.path) forward 하도록 수정. PATH_B §10 을 "직접 vs async" 2방식 표로 재작성, PATH_A 에 직접추론 절 추가.
- 반영 파일: `.env`·`.env.example`·`env/*.env`·`docker/docker-compose.yaml`·`docker/policy-entrypoint.sh`·`docker/lerobot-entrypoint.sh`·`docs/PATH_A_NATIVE.md`·`docs/PATH_B_DOCKER.md`·`AGENTS.md`. bash -n + compose config 검증 완료.

---

## 작업 인계 (2026-06-02 — 출발모델 변수 통일)

- `BASE_MODEL` 제거, fine-tune 출발 모델을 `POLICY_BASE_MODEL_PATH` 단일 변수로 통일.
- `policy-entrypoint.sh` train 라우팅: `TRAIN_POLICY_TYPE` 비움→`--policy.path=$POLICY_BASE_MODEL_PATH`(LeRobot 체크포인트, SmolVLA 포함), 설정→`--policy.type`+`--policy.base_model_path`(GR00T 등 native 베이스). 0.5.x 의 path/type 동시금지 구조적 회피.
- 반영: `.env`·`.env.example` §1(블록 11→10줄), `docs/PATH_A_NATIVE.md`·`PATH_B_DOCKER.md`·`TROUBLESHOOTING.md`. bash -n + 키 파리티 OK. BASE_MODEL 잔재 0.
- 미적용(설계 제안만): 다모델 확장 시 `env/<model>.env` 프로필 + 다중 `--env-file` 방식 권장(사용자 결정 대기).
- 참고: README 운영시나리오 섹션은 사용자가 정리/제거함 — 재추가 금지.

---

## 작업 인계 (2026-06-02 — .env 재구성 + 모델 토글)

- 목표: 너무 많은 env 변수/혼란 정리, `.env` ↔ `.env.example` reconcile, GR00T↔SmolVLA 전환 단순화.
- 결정(사용자): 토글 블록 2개 방식 / `.env.example` 기본 활성 = GR00T / 두 파일 다 정리 / COMPILE_MODEL 기본 false.
- 적용:
  - `.env.example`·`.env` 동일 구조로 전면 재작성. 섹션: §0 비밀값 / §1 모델토글⭐ / §2 하드웨어 / §3 카메라 / §4 수집 / §5 학습 / §6 서버(+RTC) / §7 클라.
  - **§1 모델 토글**: 두 모델 간 값이 다른 11개 변수만(POLICY_TYPE/BASE_MODEL/TRAIN_POLICY_TYPE/POLICY_BASE_MODEL_PATH/POLICY_TOKENIZER_ASSETS_REPO/POLICY_EMBODIMENT_TAG/POLICY_CHUNK_SIZE/POLICY_N_ACTION_STEPS/ACTIONS_PER_CHUNK/POLICY_REPO_ID/JOB_NAME). [A]GR00T 활성 / [B]SmolVLA 주석. 전환 = 한 블록 토글.
  - `.env` 실제 값(토큰·COM5/COM8·카메라 index·HF_USER 등) 보존. **`POLICY_TYPE=smolvla` leftover 버그 → groot 로 정정** (나머지가 전부 GR00T였음). 누락 키 RENAME_MAP·RTC_* 3개 추가.
  - 키 집합 `.env` == `.env.example` 확인 완료 (diff 동일).
  - `README.md`: "GR00T 빠른 흐름" → "운영 시나리오(학습·배포·추론) + GR00T→SmolVLA 전환". prepare-model 셸 변수 취약/논리오류 수정(학습은 자동 다운로드, 서버는 인자 없는 prepare-model=POLICY_REPO_ID).
  - `docs/PATH_B_DOCKER.md` §4 모드표에 `policy-server-rtc`(compose 기본 CMD) 추가, §9·§11 을 `.env §1 토글` 참조로 정리.
  - `docs/PATH_A_NATIVE.md` §6 native train: `--policy.type`+`--policy.path` 동시 전달(0.5.x 위반) 제거, TRAIN_POLICY_TYPE 비움.
- 기본 시나리오 확정: 이 PC(Windows native uv) = policy-client / konan147(docker) = train + policy-server-rtc.
- 미해결: native(호스트 uv) lerobot 은 Python 3.11 핀이라 0.4.x — 서버 0.5.1 과 async gRPC proto 호환은 별도 검증 필요.

---

## 작업 인계 (2026-06-02 — 0.5.1 디커플링 통합 점검)

- 목표: policy-server 0.4.4→0.5.1 / pyproject.toml 디커플링 후 .env·스크립트·Dockerfile·문서 일관성 점검 및 수정.
- 검토로 찾아 수정한 불일치 (미커밋):
  1. `README.md:5` — "하나의 pyproject.toml 로 묶어" 문구를 policy-server 디커플링 반영으로 수정.
  2. HF 캐시 경로 `/root/.cache/huggingface` → 실제 `/workspace/.cache/huggingface` (HF_HOME, non-root UID): `policy-entrypoint.sh`(주석+사용자 로그 211행), `docs/PATH_B_DOCKER.md`(76·263), `AGENTS.md`(33).
  3. **기능 버그**: `COMPILE_MODEL=true` 기본 + entrypoint 가 정책 무관하게 `--policy.compile_model` 추가 → GrootConfig 에 해당 필드 없어 GR00T train 시 draccus 거부. `policy-entrypoint.sh` train 분기에 `TRAIN_POLICY_TYPE=groot` 면 compile skip+warn 가드 추가. `.env.example`·PATH_B §9 에 주석 보강.
  4. `docs/PATH_B_DOCKER.md:84` 빌드 표 — lerobot 서비스 설명에서 `train` 제거(policy-server 로 이동), policy-server 행에 디커플링 명시.
  5. `pyproject.toml` 헤더 주석 — 존재하지 않는 `Dockerfile.teleop` → `Dockerfile.lerobot`(실제 `uv sync --group teleop --group async`), policy/async 그룹 + 디커플링 설명 추가.
  6. `policy-entrypoint.sh` 주석 `huggingface-cli download` → `hf download` (코드 일치).
- 검증: `bash -n` 으로 두 entrypoint 문법 OK.
- 미해결(별도 판단 필요): `docs/PATH_A_NATIVE.md:334-335` native(0.4.4) train 이 `--policy.type` 과 `--policy.path` 를 동시 전달 — 0.5.1 계약과 다름. 단 native 는 호스트 lerobot 0.4.x 라 동작 가능성. 확인 후 정리 필요.
- 참조용 0.5.2 소스 트리: 작업 디렉터리 `lerobot/` (untracked). robot_client `__main__` 는 `register_third_party_plugins()` + `async_client()`.

---

## 작업 인계 (이전 — GR00T N1.5 학습/추론)

- 목표: `ssh konan147` 원격 서버에서 `policy-server:0.5.1` 컨테이너로 GR00T N1.5를 SO-101 `taehunkim/so101_pick_pen` 데이터셋에 fine-tune.
- 현재 상태: GR00T N1.5 10,000-step full 학습 완료 및 Hugging Face Hub push 완료. 사용자가 원격 inference 컨테이너를 삭제했고, 재실행을 위한 서버/클라이언트/학습 가이드를 Markdown 문서에 반영 중. 이전 코드 변경은 origin/main에 커밋/푸시 완료, 이번 문서 변경은 아직 미커밋.
- 완료한 일:
  - 원격 repo 확인: `/home/konan147/Workspaces/SO101-Sim2Real`, branch `main`, git status clean.
  - 실행 중 컨테이너 없음 확인.
  - `policy-server:0.5.1` 이미지 존재 확인.
  - HF cache 준비 완료:
    - `nvidia/GR00T-N1.5-3B`
    - `lerobot/eagle2hg-processor-groot-n1p5`
  - 1차 smoke 실패 원인 확인: 이미지 내 baked entrypoint가 `POLICY_PATH=lerobot/smolvla_base`를 `--policy.path`로 주입.
  - 2차 smoke 실패 원인 확인: LeRobot 0.5.1 GR00T action head가 Transformers meta tensor 초기화 구간에서 `torch.distributions.Beta` 기본 validation을 실행해 `Tensor.item() cannot be called on meta tensors` 발생.
  - 3차 smoke 진행: Beta patch로 meta tensor 오류는 통과. 다음 오류는 Transformers 5.3이 `GR00TN15.all_tied_weights_keys`를 기대하지만 클래스에 없어 발생.
  - 로컬 파일 수정:
    - `.env`, `.env.example`: 0.5.1 기준 `TRAIN_POLICY_TYPE`, `BASE_MODEL`, `POLICY_BASE_MODEL_PATH`, `DATASET_VIDEO_BACKEND` 계약으로 정리.
    - `docker/policy-entrypoint.sh`: `--policy.path`와 `--policy.type` 동시 전달 방지, GR00T env 매핑 추가.
    - `docker/Dockerfile.policy`: GR00T/Transformers 5.3 호환 패치 추가.
    - 관련 문서/주석의 `Dockerfile.smolvla`, `POLICY_CLIENT_TYPE`, policy-server 0.4.4 표기 정리.
  - 원격에 수정 파일 반영 완료. 원격 `.env`는 토큰/포트 유지, GR00T/0.5.1 학습 키만 변경.
  - `policy-server:0.5.1` 재빌드 완료.
  - 100-step smoke 성공:
    - Job: `smoke_groot_n15_pick_pen_100_20260601_235702`
    - Output: `/home/konan147/Workspaces/SO101-Sim2Real/outputs/train/smoke_groot_n15_pick_pen_100_20260601_235702`
    - Checkpoint: `checkpoints/000100/pretrained_model/config.json`
    - `jq -r .type ...` 결과: `groot`
  - 10,000-step full 학습 시작:
    - Run ID: `groot_n15_full_20260602_000255`
    - PID: `1384429`
    - Container: `so101-groot-n15-full-20260602-000255`
    - Log: `/home/konan147/Workspaces/SO101-Sim2Real/logs/train/groot_n15_full_20260602_000255.log`
    - Output: `/home/konan147/Workspaces/SO101-Sim2Real/outputs/train/so101_groot_n15_pick_pen`
    - W&B: `https://wandb.ai/pubcyberry/lerobot/runs/raxsfmc2`
    - 완료 확인: 10,000/10,000 step, `End of training`, fatal error pattern 없음.
    - 최종 checkpoint: `outputs/train/so101_groot_n15_pick_pen/checkpoints/010000/pretrained_model`
    - 최종 loss 로그: `loss:0.024`, `grdn:0.654`, `lr:1.0e-05`
    - Hub push: `https://huggingface.co/taehunkim/so101_groot_n15_pick_pen`
    - Hub sha: `94940f296903133ef1b02e5145232aa83be6c6df`
  - 원격 inference server 시작:
    - 처음에는 `policy-server-rtc`로 띄웠으나, `GrootPolicy`는 LeRobot 0.5.1에서 `init_rtc_processor` 미지원이라 RTC 없이 fallback함.
    - 불필요한 per-chunk RTC 분기/로그를 피하려고 표준 `policy-server`로 교체.
    - Container: `so101-groot-n15-policy-server`
    - PID: `2509209`
    - Bind: `0.0.0.0:8080`
    - Log: `/home/konan147/Workspaces/SO101-Sim2Real/logs/server/so101-groot-n15-policy-server_20260602_081426.log`
    - Model cache prepared: `taehunkim/so101_groot_n15_pick_pen`
    - `Ready` + `SendPolicyInstructions` RPC 성공.
    - GR00T model load 확인: `Time taken to put policy on cuda: 5.6529 seconds`
    - GPU memory after load: 약 `6077 MiB / 48935 MiB`
  - Git 정리:
    - 로컬 `konan` remote 제거.
    - 로컬 `origin` URL을 GitHub moved target인 `https://github.com/PubCyBerry/SO101-Sim2Real.git`로 갱신.
    - Commit: `0397e0d feat: GR00T N1.5 policy-server 학습/추론 지원`
    - 로컬/원격 서버 모두 `main`이 `0397e0d`로 동기화.
    - 원격 서버의 duplicate tracked changes는 stash 후 fast-forward pull, duplicate stash drop 완료.
  - 문서 업데이트 진행:
    - `README.md`: GR00T N1.5 빠른 흐름 추가. `.env` 핵심값, 100-step smoke, full 학습, 추론 서버, policy client 접속 요약.
    - `docs/PATH_B_DOCKER.md`: GR00T 학습 설정, 100-step smoke/full run, 서버 기동, 클라이언트 연결, `policy-server-rtc` 미사용 사유, SmolVLA/GR00T 카메라 key 차이를 반영.
    - `docs/PATH_A_NATIVE.md`: native policy client 가이드에 GR00T 원격 서버 접속 예시와 0.5.1 `policy` dependency group 표기 반영.
    - `docs/TROUBLESHOOTING.md`: GR00T에서 `policy-server-rtc`가 표준 추론으로 fallback 되는 사례 추가.
- 남은 일:
  - 문서 diff 최종 확인 후 사용자에게 서버/클라이언트/학습 실행 요약 전달.
  - 이번 Markdown 변경 커밋 여부 결정.
  - Windows/실기기 쪽 `policy-client` 또는 `record --policy.path`로 실제 기기 추론 smoke 테스트.
- 결정한 사항:
  - GR00T 학습 CLI는 `--policy.type=groot`, `--policy.base_model_path=nvidia/GR00T-N1.5-3B`.
  - GR00T action horizon 제한에 맞춰 `--policy.chunk_size=16`, `--policy.n_action_steps=16`.
  - SmolVLA용 `.env` `RENAME_MAP`은 `--rename_map="{}"`로 덮어쓴다.
  - smoke는 Hub push 비활성화, full run은 `taehunkim/so101_groot_n15_pick_pen`로 push.
- 검증 결과:
  - 1차 smoke 실패: 원격 `.env`의 `POLICY_PATH=lerobot/smolvla_base`가 entrypoint에서 `--policy.path`로 전달되어 `--policy.type=groot`와 충돌.
  - 2차 smoke 실패: `RuntimeError: Tensor.item() cannot be called on meta tensors`.
  - 3차 smoke 실패: `AttributeError: 'GR00TN15' object has no attribute 'all_tied_weights_keys'`.
  - 4차 smoke 실패: `ValueError: Unsupported video backend: decord`.
  - 5차 smoke 실패: `AttributeError: 'list' object has no attribute 'shape'` (`pixel_values`가 list).
  - 6차 smoke 실패: `NotImplementedError: aten::_sample_dirichlet ... Meta tensors`.
  - 최종 smoke 성공: 100 step 완료 및 checkpoint 저장.
  - full 학습 성공: 10,000 step 완료, checkpoint 001000~010000 생성, Hub push 완료.
- 다음 명령:
  - 문서 변경 확인:
    `git diff --stat -- README.md docs/PATH_A_NATIVE.md docs/PATH_B_DOCKER.md docs/TROUBLESHOOTING.md`
  - 최종 결과 확인:
    `ssh konan147 'cd /home/konan147/Workspaces/SO101-Sim2Real && jq -r .type outputs/train/so101_groot_n15_pick_pen/checkpoints/010000/pretrained_model/config.json'`
  - 서버 상태 확인:
    `ssh konan147 'cd /home/konan147/Workspaces/SO101-Sim2Real && docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server && docker compose --env-file .env -f docker/docker-compose.yaml logs -f policy-server'`
