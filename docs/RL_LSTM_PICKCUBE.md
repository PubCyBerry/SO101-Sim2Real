# SO-101 PickCube — LSTM(PPO) 강화학습 세션 기록

> 작업 브랜치: `worktree-lstm-ppo-pickcube` (worktree `.claude/worktrees/lstm-ppo-pickcube`)
> 작성: 2026-06-10 세션. 목적 = 진행/시행착오/조사/트러블슈팅/결과를 한 곳에 정리(다음 세션 인계용).

---

## 1. 목표와 제약

**목표**: cube_desk 씬에서 SO-101 6축 팔이 **4개 큐브를 그릇에 pick-and-place**. **LSTM(recurrent) actor-critic + PPO**. Domain Randomization 으로 일반화. **최종 합격 = DR 켠 상태 성공률 ≥ 0.90**.

**제약(엄수)**:
- 🚫 **grasp assist 금지**: 큐브를 그리퍼에 weld/부착하거나 인공 유지력 추가 금지. (기존 soft-PD + 10Nm 물리는 정당.)
- 🚫 **plate(그릇) 성공반경 스케일링 금지**: `BOWL_SUCCESS_RADIUS` 불변, `container_radius_scale=1.0` 고정.
- ✅ **속도 페널티**: 느린 정책에 페널티(빠르고 정확한 정책 우대).
- ✅ **큐브 4개 모두 사용**, DR 적용 시에도 **도달 가능 범위에서만 spawn**.
- ✅ **VRAM**: 초기 16GB soft cap → 세션 중 **32GB 까지 허용**으로 상향.
- ✅ **진행 모니터링용 에피소드 비디오** 주기 녹화.

---

## 2. 실행 환경 / 방법

- worktree 에는 별도 venv 가 없음 → **메인 `.venv` + `PYTHONPATH=$(pwd)/src`** 로 worktree 코드를 우선 적용.
- 표준 실행:
  ```bash
  cd /home/konan147/Workspaces/SO101-Sim2Real/.claude/worktrees/lstm-ppo-pickcube
  ROOT=/home/konan147/Workspaces/SO101-Sim2Real
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src $ROOT/.venv/bin/python scripts/reinforcement_learning/<script>.py ...
  ```
- 학습 호스트: Linux 서버 RTX PRO 5000 Blackwell 48GB(공유). num_envs 는 VRAM 예산 내 최대화.

---

## 3. 추가/수정한 코드 (파일별)

| 파일 | 내용 |
|---|---|
| `scripts/reinforcement_learning/train.py` | `--recurrent`(ActorCriticRecurrent LSTM/GRU)·`--rnn_*`·`--obs_normalization`·`--schedule`; `--grasp_bootstrap_prob/close`; `--video/--video_length/--video_interval`(현재 학습엔 미사용, eval로 이동); 커리큘럼·부트스트랩·비디오 배선 |
| `scripts/reinforcement_learning/eval.py` | DR-on 성공률 평가(N≥256, 부트스트랩 없이 실제 성능). `--video`(env 1개 권장) → `<run>/videos/eval/` 에피소드 녹화 |
| `scripts/reinforcement_learning/verify_reachability.py` | DR 적용 시 큐브가 SO-101 도달반경 내인지 실측(코너 거리 통계) |
| `scripts/reinforcement_learning/grasp_feasibility.py` | IK 배제, 큐브를 손가락 중점에 텔레포트→닫기→들기로 **grasp 물리 가능성** 격리 검증 |
| `scripts/reinforcement_learning/verify_bootstrap.py` | 부트스트랩 초기상태(큐브-인-그리퍼)가 유지·들림 되는지 검증(close 각 스윕) |
| `src/sim_to_real/tasks/pick_cube/pick_cube_env.py` | **신규** `PickCubeEnv(ManagerBasedRLEnv)` — (1) 매 step 동적 gripper effort 배선, (2) `_reset_idx` 에서 초기상태 grasp 부트스트랩 |
| `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` | 속도보상 2종, 물리/센서 DR, gripper init=0.80, GPU 버퍼 상향, viewer 카메라, 부트스트랩 cfg 필드, pregrasp 보상 재설계 |
| `src/sim_to_real/tasks/pick_cube/__init__.py` | entry_point → `PickCubeEnv` |
| `src/sim_to_real/tasks/pick_pen/mdp/rewards.py` | `time_penalty`, `early_finish_bonus`(속도 보상) + 헬퍼 |
| `src/sim_to_real/utils/domain_randomization.py` | `randomize_object_material`/`randomize_object_mass`(물리 DR 래퍼) |

**커밋 이력(브랜치)**:
```
08af163 LSTM(PPO) + 속도보상 + 물리/센서 DR + eval/verify 스크립트
3426a73 fix: 4096+ env PhysX patch buffer overflow
c0c1add fix: time_penalty 부호 버그
b23d965 fix: gripper open-start(offset 0.80) + 동적 effort 배선
f22db64 feat: 초기상태 grasp 부트스트랩 + pregrasp 보상 재설계
15c7c71 feat: 비디오 녹화(--video) + 8192env 버퍼 상향
0a46503 feat: 비디오 녹화 학습→eval 이동(8192 뷰포트 렌더가 학습 지연)
4e63bf2 fix: 비디오 뷰포트 카메라 정면 뷰 고정
```

---

## 4. 시행착오 & 트러블슈팅 (시간순)

### T1. `time_penalty` 부호 버그
- 현상: `Episode_Reward/time_penalty` 가 **양수(+0.02)** — 미완료를 *보상*.
- 원인: 함수가 -1.0 반환 + RewTerm weight -0.02 → 곱 +0.02.
- 해결: 함수가 +1.0 반환(weight 음수와 곱해 페널티). **교훈: RewTerm 부호는 weight×func 로 검증.**

### T2. PhysX patch buffer overflow (대규모 env)
- 현상: 4096 env 에서 `Patch buffer overflow ... at least ~166k`.
- 원인: `gpu_max_rigid_patch_count` 기본(~82k) 부족.
- 해결: `gpu_max_rigid_patch_count` 상향(4096→5·2¹⁶, 8192→10·2¹⁶), `gpu_total_aggregate_pairs_capacity` 256k→512k, `gpu_collision_stack_size` 2²⁸.

### T3. 4큐브 동시 학습 pregrasp 정체
- 현상: 400 iter(86M) success=0. reach·pregrasp 만 받고 grasp/lift=0.
- 대응: **단일 큐브 커리큘럼**(active_objects=1)으로 재시작, 성공 시 1→2→3→4 확장 전략.

### T4. gripper open/close 규약 혼동
- 사실: reward 코드 기준 **open = joint_pos > 0.6(높음), close = 낮음(→ 하한 -0.174)**. init=0.0 은 닫힘쪽.
- grasp_feasibility 초기 버전이 open/close 를 반대로 보내 큐브를 떨어뜨림 → 수정.

### T5. grasp 물리 가능성 검증 (effort 가설)
- `grasp_feasibility.py`: 큐브를 손가락 중점에 놓고 닫기+들기 → **effort 10Nm·0.5Nm 둘 다 held&lifted 100%**.
- 결론: **물리적 grasp 가능**(leisaac과 동일 actuator, weld 없음). **effort 동적조정이 grasp 가능 여부를 좌우하지 않음**(자유 포착 시 튕김 방지엔 도움).
- 한계: 이 테스트는 큐브를 정위치에 **고정(pin)** 후 닫아 hold+lift 만 검증. 자유 포착의 어려움은 별개(=학습 문제).

### T6. gripper init/offset 0.80 (action range)
- 발견: action target = `raw·1.0 + offset(=default joint)`, `clip_actions=1.0` → 도달범위 `[offset-1, offset+1]`. offset=0 이면 **최대 open 1.0rad** 뿐(40mm 큐브 폭 부족) + 닫힌 채 시작해 pregrasp 공짜 획득.
- 수정: gripper init/offset=**0.80** → open-start(>0.6) + 도달범위 `[-0.20, 1.745]`(full open↔close).
- 결과: **단독으로는 grasp 미해결**(여전히 정체). 필요조건이나 충분조건 아님.

### T7. Monitor awk 정규식 거짓 경보 ⚠
- 현상: 모니터가 "GRASP 1.9 상승" 알림 → 실제론 grasp 미발생.
- 원인: awk `/grasp_cube:/` 가 **`pregrasp_cube` 도 부분 매칭**. 본 값은 pregrasp 였음.
- 해결: `$1=="Episode_Reward/grasp_cube:"` **정확 매칭**. **교훈: grep/awk 시 grasp_cube 는 pregrasp_cube 와 구분.**

### T8. pregrasp local optimum 진단 → 보상 재설계
- 로그 확정: grasp_cube 전 구간 0. 정책이 **그리퍼를 큐브 근처(8cm)에서 닫는 것**(pregrasp 1.9)으로 보상을 챙기고 envelop+lift 를 안 함.
- 수정: `pregrasp_bonus` weight 2→0.5, diff_threshold 0.08→0.045(공짜 camping 제거).
- 결과: camping 은 줄었으나(pregrasp 1.9→0.32) **retune 만으로 grasp 미해결**(탐색 벽).

### T9. 초기상태 grasp 부트스트랩 (backward curriculum) ✅ 효과
- 근거: NVIDIA gear_assembly 가 **grasp 직전 상태에서 에피소드 시작**(어려운 grasp 획득을 RL로 안 풀고 하류만 학습).
- 구현: `PickCubeEnv._reset_idx` 가 비율 `grasp_bootstrap_prob` 만큼 env 를 **큐브-인-그리퍼**로 시작. grasp point 는 default 자세에서 **1회 캐시**(고정베이스 상수 → FK-at-reset 불안정 회피).
- `verify_bootstrap.py` close 스윕: **close=-0.15 에서 held&lifted 0.94**(0.05/-0.10 은 낮음). → 채택.
- 결과: success termination **0 → ~12%**, guided_lift 상승. (단 부트스트랩 env 포함 수치 → 실제 성능은 eval 로 판정.)

### T10. 비디오 흰 화면 (조명 과노출)
- 현상: 8192-env 학습 중 녹화 영상이 **온통 흰색**.
- 원인: scene.usd 의 **DomeLight 가 env 마다 복제** → 8192배 누적 ambient → 과노출. (코드 주석 `pick_cube_env_cfg.py:131-134` 에 예견됨.)
- 검증: env 1개 녹화 → frame mean ~105/255, 과노출 0% (정상). → **env 1개로 녹화**.

### T11. 비디오 시야 가림 (천장 KeyLight 평면)
- 현상: 머리 위 큰 흰 평면(KeyLight area light)에 데스크/로봇이 가림.
- 해결: `env_cfg.viewer.eye/lookat` 를 **작업공간 정면·약간 낮은 각도**로 고정 → 데스크·매트·큐브·그릇·팔 또렷이 보임. (학습은 렌더X라 무관.)

### T12. 학습-중 뷰포트 녹화가 너무 느림
- 8192-env 뷰포트 렌더로 iter 47s/ETA 19.6h. → **비디오를 eval(소수 env)로 이동**, 학습 런은 녹화 없이 iter 17.8s/ETA 7.4h.

---

## 5. 조사 내용 (참고 구현·MCP)

### leisaac (검증된 SO-101 Isaac Lab 구현)
- **grasp 판정 = 순수 기하/상태**(`object_grasped`: EE↔물체 < 20mm AND gripper joint < 0.26). **weld/attach 없음.**
- **동적 gripper effort**: 물체 질량 기준 `clamp(mass/0.15, 0.5, 10)` Nm → 20~35g 큐브엔 ~0.5Nm. **teleop 뿐 아니라 정책 추론(`scripts/evaluation/policy_inference.py`)·env step 레벨에도 적용**(천 접기만 off). → 우리도 RL step 에 배선(grasp 가능 여부와는 무관, 자유 포착 안정성용).
- 우리 gripper actuator(stiffness 17.8/damping 0.6/effort 10Nm/solver 4·4)는 leisaac과 동일.

### Isaac Lab gear_assembly (공식 LSTM+PPO contact-rich)
- 에피소드를 **grasp/삽입에 가까운 초기 상태**에서 시작 → 어려운 grasp 획득을 RL이 처음부터 안 풂(=우리 부트스트랩의 근거).
- **action scale 작게(0.025)**, **joint friction·actuator stiffness/damping DR**(sim2real 핵심), exteroceptive obs 에만 노이즈·proprioception clean. num_envs 256, 12~24h.

### NVIDIA isaac_ros_manipulation (`ref_repos/isaac_ros_manipulation`)
- pick-and-place = **RL 아님**. 인식(FoundationPose) → **사전정의 grasp pose(YAML)** → **cuMotion 모션플래닝** → 위치제어 그리퍼(close_position+max_effort) → (플래닝측) object attach 로 운반. UR5e/UR10e+Robotiq 기준(5DOF 미지원).
- 시사: **정밀 grasp 를 RL로 학습하지 않고 "알려진 grasp pose 로 계획"** — 우리 PATH E(cuMotion) 와 동일 철학.

### MCP (Isaac Sim / USD) — 현실적 grasp 권장
- clamp force 를 물체 질량에 맞춤(leisaac 방식)=현실적.
- **contact offset > rest offset**, 작은 물체는 rest≈0 + 약간 큰 contact offset(현 cube 0.002 다소 작음 → 0.004~0.005 검토).
- **torsional patch radius**(현재 미설정) → 잡은 물체 회전 억제(grasp 안정).
- CCD on(빠른 닫힘 tunneling 방지), friction 높게+combine mode, 손가락 collision 은 convex hull 한계 시 SDF 고려, solver velocity iter>4 경고.

---

## 6. 진단 결론

1. **grasp 물리는 가능**(feasibility 100%). 문제는 물리/effort 가 아니라 **RL 학습 설정**.
2. 막힌 핵심: (a) **pregrasp local optimum**(닫힘+근접만으로 보상), (b) **cm 단위 정밀 grasp 의 탐색 벽**(scratch-PPO 난제).
3. 해법: **초기상태 grasp 부트스트랩(backward curriculum)** 이 핵심 레버 — 하류(lift→place→success)를 먼저 학습시켜 critic value 를 전파, 정상-env grasp 학습을 견인. + 보상 재설계 + (예정) sim2real DR.

---

## 7. 현재 진행 (2026-06-10 세션 종료 시점)

- **학습 중**: stage-1(단일 큐브), LSTM(hidden 256, 1층)+PPO, **num_envs 8192**, grasp_bootstrap_prob 0.5/close -0.15, entropy 0.02, adaptive LR. 로그 `train_boot.log`, ETA ~7.4h.
- 지표(약 127 iter, 50M steps): **success termination ~0.12**(부트스트랩 포함), guided_lift 상승 중, grasp_cube 아직 낮음(정상-env 본격 grasp 대기).
- VRAM ~16-20GB(예산 32GB 내), patch overflow 0.

---

## 8. 진행 예정

1. **stage-1 성공률 ≥0.90**(eval, 부트스트랩 없이) 도달 시 **자동 커리큘럼 확장** 1→2→3→4(매시간 wakeup 이 `--resume_checkpoint` 로 재시작).
2. **(C) sim2real DR 강화**: `randomize_actuator_gains`(stiffness/damping) + joint friction 랜덤화(gear_assembly 핵심) 추가 — 일반화/sim2real.
3. 정밀도 향상이 더 필요하면: action scale 축소, grasp point↔cube dense 정렬 보상, contact/rest offset·torsional patch 튜닝.
4. **stage-4 ≥0.90 달성 = 최종 목표.** eval 영상으로 시각 확인.

---

## 9. 핵심 설정값

| 항목 | 값 |
|---|---|
| 정책 | ActorCriticRecurrent, rnn_type lstm, hidden 256, layers 1, MLP [256,128], elu, obs_normalization on |
| PPO | num_steps_per_env 48, learning_epochs 8, mini_batches 4, lr 3e-4(adaptive), entropy 0.02, gamma .99, lam .95, clip .2 |
| num_envs | 8192 (VRAM 32GB 예산) |
| 속도 보상 | `time_penalty` weight -0.02(미완료 step당), `early_finish_bonus` weight 100(완료 시각 비례, 종료 1회) |
| 부트스트랩 | prob 0.5, close -0.15(held 0.94), grasp point=default 자세 jaw·gripper 중점 캐시 |
| gripper | init/offset 0.80(open-start+full range), open>0.6 / close<0.5(<0.26 강) |
| 큐브 DR | scatter x[1.60,2.08] y[-0.47,-0.33] yaw±30°(도달 검증 max 0.333m<0.44), 마찰/질량 startup DR, rl_state GaussianNoise σ0.005 |
| GPU 버퍼 | gpu_max_rigid_patch_count 10·2¹⁶, aggregate 512k, collision_stack 2²⁸ |
| viewer(영상) | eye (1.90,0.95,0.98) lookat (1.85,-0.32,0.76) res 1280×720 |
| 금지선 | container_radius_scale 1.0 고정, grasp weld 없음 |

---

## 10. 모니터링 / 상태 파일

- 학습 로그: `train_boot.log`(또는 최신 `train_*.log`).
- 상태 파일: `/tmp/train_pid.txt`(TRAIN_PID), `/tmp/lstm_stage.txt`(현재 활성 큐브 수), `/tmp/lstm_monitor_state.txt`(마지막 평가 체크포인트 index).
- 체크포인트/영상: `outputs/rl/rsl_rl/lstm_ppo_pickcube/<run>/` (`model_*.pt`, `videos/eval/*.mp4`).
- 매시간 자동 점검: 생존확인 → 진행 grep → 새 체크포인트 eval(성공률 128env + 영상 1env) → ≥0.90 시 커리큘럼 확장.
- 실시간 모니터: success 30% 돌파 / 정상-env grasp 본격화 / 크래시 알림.

---

## 11. 미해결 / 리스크

- 정상-env 의 **처음부터 grasp 획득**이 부트스트랩 value 전파만으로 충분히 학습될지 미확정(eval 성공률로 판정 중).
- 커리큘럼 1→4 확장 시 큐브 수 증가로 난이도 급상승 가능(부트스트랩은 첫 큐브만 잡힌 채 시작).
- sim2real: 현재 DR(포즈/마찰/질량/관측노이즈)에 actuator gain·joint friction 미포함(예정).
- 정밀 grasp 가 끝내 부족하면 모방학습(SmolVLA/GR00T, 레포 본래 경로) 또는 planning(cuMotion, PATH E) 병행 검토.
</content>
