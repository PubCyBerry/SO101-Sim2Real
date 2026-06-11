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
06ce08e docs: 본 세션 기록 문서
29857af tune: 부트스트랩 큐브 놓침 방지 — gripper offset 0.80→0.20, carry/guided_lift 가중↑, bootstrap_prob 0.5→0.75
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

### T13. success ~0.13 정체 진단 → 부트스트랩 큐브 놓침 수정
- 추이(boot8192, 199 iter): success 0→0.10(50 iter 급상승) 후 **0.13 부근 평탄(log 포화)**. `grasp_cube`≈0 전 구간. ep length ~800(감소 X).
- 분석: success 0.13 ≈ 부트스트랩 env(0.5) × 하류성공 ~26% + 정상 env(0.5) × ~0. **정상-env grasp 미점화**.
  ⚠️ 0.13 은 부트스트랩 inflated — **부트스트랩 없는 eval 실성공률은 ~0% 예상**.
- 근본 원인: **gripper offset 0.80 → do-nothing(action≈0) target 0.80 = 활짝 열림** → 부트스트랩으로 잡고 시작해도 정책이 곧 손 벌려 놓침(하류 학습 저해, 하류성공 26%에 그침).
- 수정(커밋 `29857af`): **gripper offset 0.80→0.20**(do-nothing 닫힘쪽 유지, open 1.20 까지 grasp 가능), **carry_cube 4→8 / guided_lift 8→10**(hold 강화), **bootstrap_prob 0.5→0.75**.
- 판정 지표: (1) training success 가 0.13 정체 깨고 0.3~0.6 상승 = 하류 정상화. (2) `grasp_cube`>0 = 정상-env grasp 점화(eval 상승 전제).

### T14. hold-fix 효과 일부 + eval 실성공률 0% 확정 → 정책/하이퍼/입력 묶음 개입
- boot075(offset 0.20+carry/guided_lift↑+bootstrap 0.75): training success 0.13→**0.20**(hold-fix 효과 O), guided_lift 0.29. 그러나 **grasp_cube 여전히 ≈0**.
- **model_100 eval(부트스트랩 없이 128env/256ep) = success_rate 0.0** → 0.20 은 부트스트랩 inflated, **실제 실력 0% 확정**. 탐색 벽 미돌파.
- 정책/하이퍼/입력 전면 검토 후 **묶음 적용**(커밋 참조):
  - **입력**: rl_state 에 속도 추가(joint_vel 6 + ee lin vel 3 + cube lin vel 4×3) → 43→**64dim**. 부분관측 해소(LSTM 의 속도추정 부담 제거). `include_velocities=True`.
  - **하이퍼**: gamma 0.99→**0.997**(유효지평 3.3s→11s, 장기 credit). `--gamma`/`--lam` CLI.
  - **탐색**: **RND**(Random Network Distillation) 추가 — 내재 보상으로 grasp 탐색 벽 공략. `--rnd`(rnd_state=rl_policy 자동, weight 0.5 linear 감쇠, state/reward norm on).
- 정책 검토 메모: 거의 완전관측 상태라 MLP도 경쟁력 있음(LSTM은 요구사항이라 유지); student-teacher 증류는 sim2real 로드맵; symmetry 부적합.

### T15. grasp 물리 sanity — 30/40mm 양쪽 가능 확정, D 전제 반증
- `grasp_feasibility.py`로 Cube1(30mm)·Cube3(40mm) 검증: **둘 다 held_and_lifted 1.0**.
- close_target -0.15/0.0/0.10(gripper joint 0.05/0.20/0.30) **전 구간 hold 1.0** — 일단 seated 되면 do-nothing(offset 0.20)에서도 안 놓침.
- 결론: grasp 막힘은 물리/hold 아님. **gripper offset/range 변경 불필요**(D 원래 전제 반증). arm action scale 축소도 기각(scale 1.0=±1rad 범위라 줄이면 reach 불가, reach 보상 현재 작동). 정밀도 레버는 obs+정렬 보상으로 이동.

### T16. obs 충분성 — 방향·크기 누락 식별 → 64→83dim 확장
- 진단: rl_state 64dim이 전부 위치+선속도. **큐브 yaw 없음**(평행 jaw 정렬 불가), **EE 방향 없음**, **큐브 크기 없음**(30/40mm 2종인데 벌림 폭 매칭 불가). 모서리 좌표·크기 절대값은 중복(center+yaw+크기로 유도) → 제외.
- 수정: `rl_state(include_orientation=True, pen_half_extents=(.015,.015,.020,.020))` — pen yaw sin/cos(8)+half-extent(4)+ee quat(4)+grasp→cup(3) = **+19 → 83dim**. LSTM input 83 확인.

### T17. grasp 점화 직격 — 정렬 보상 + graded 부트스트랩 annealing
- **탐색 valley 진단**: reach(0.30m, 거침)→정렬→닫기→lift 사이, "열린 채 정렬됐는데 닫는 순간 align의 open조건도 깨지고 lift도 아직 0"인 무보상 구간이 grasp 탐색 벽.
- **`grasp_align_reward`(신규, weight 3.0)**: 열린 그리퍼 grasp point를 미배치 큐브에 정밀 3D 정렬(xy×z×open_frac, lift 게이트 없음). open_frac 항이 "닫은 채 근접"(camping) 제외 → pregrasp local optimum 회피. pregrasp weight 0.5→0.2 축소(open/close 상충).
- **graded 부트스트랩 annealing**(`PickCubeEnv`): prob 0.75→0 선형 감쇠(`common_step_counter` 기준) + 부트스트랩 env를 full-grasp↔pre-grasp(열린 그리퍼가 책상 위 큐브 바로 위 hover)로 분할, pre-grasp 비율 진행도 p로 0→1 ramp. 초반=하류 학습, 후반=실제 grasp 행동 학습. 정상-env grasp 학습 압력 생성(고정 0.75의 핵심 결함 해소).
- 접촉(ContactSensor) 기반 grasp는 폴백으로 보류(USD/센서 리스크) — 정렬+annealing으로 grasp 안 켜지면 도입.

### T18. RND 부분공간 명시 + 비활성 큐브 마스킹
- `obs_groups`에 `rnd_state` 명시(미지정 경고 제거, `--rnd_state_group`). 단일 큐브 스테이지에선 `rl_state(num_active=1)`로 비활성 큐브(위치·상대·속도·yaw·크기) 0 마스킹 → distractor 제거 + RND novelty 활성 큐브 집중.

### T19. align local-optimum 진단 → close-bridge 보상(개입 A, resume 적용)
- 추이(grasp_v2): iter 250~500 **scratch reach 1.0 / grasp·lift·success 0** 고착. 학습 로그 `grasp_align_cube` 0→1.78(최대 보상항). **정렬은 완벽히 학습됐으나 grasp(닫고 들기) 미점화** = align 자체가 새 local optimum("열린 그리퍼로 큐브 위 hover 캠핑"). 닫으면 align(open_frac)이 깎이는데 guided_lift는 lift 전까지 0 → **align→lift 보상 valley**를 못 넘음. pre-grasp(hover) 부트스트랩도 같은 캠프로 빠짐(monitor pre_grasp grasp 0).
- model_500 baseline(monitor): scratch reach 1.0 / grasp 0 / lift 0 / success 0 (iter450 lift 0.069는 노이즈, 안 이어짐).
- **개입 A**(grasp_v2 model_500 → grasp_v3 resume, 정책 보존):
  - **`grasp_close_reward`(신규, weight 2.5)**: grasp point가 큐브에 정밀 정렬된 채 **닫는 행동**을 보상(align의 거울: closed_frac). lift 게이트 없음 → 열림→닫힘으로 갈 때 align↓·close↑로 연속 그래디언트 = valley 제거.
  - **grasp_align weight 3→1.5**: hover 캠프 매력 축소.
  - resume: `--resume_checkpoint model_500.pt`, bootstrap 0.5/anneal 500, iteration 500→2000. 결과: align 0.78 복귀(정책 로드 확인), **`grasp_close_cube` 0.0025→0.015 성장**(close 학습 시작), grasp_cube 깜빡임.
  - 정체 시 다음 레버: close weight↑ / align 추가↓ / pre-grasp를 반쯤 닫힌 상태로 시작.

### T20. 비활성 큐브 물리 비활성화
- 단일 큐브 스테이지인데 Cube2~4가 scatter돼 작업공간에 물리·시각적으로 존재(distractor). `randomize_cubes_scattered(num_active=N)`로 앞 N개만 배치, 나머지는 **지면 아래(z=-1.0)로 치워 비활성화**(낙하해 작업공간 이탈). 커리큘럼 `active_objects`와 연동(`apply_curriculum`이 scatter·obs 양쪽에 주입).

### T21. align-camp 미돌파 → 전체 개입 묶음 + fresh 재시작(grasp_v4, claude-fable-5 검토 반영)
- grasp_v3(resume) 진단: model_500 baseline scratch grasp/lift/success=0 (align-camp 고착, iter450 lift 0.069는 노이즈). resume의 close-bridge로도 캠프 탈출 불확실 → **claude-fable-5 설계 검토** 후 **fresh 재시작**(이미 굳은 캠프를 물려받지 않는 게 더 깨끗).
- 적용 묶음(grasp_v4):
  - **보상 rebalance**: grasp_align 1.5→**1.0**, grasp_close 2.5→**3.0** → open→closed 보상 합이 단조 증가(닫을수록 이득) = valley 제거의 1차 메커니즘.
  - **`grasp_contact_reward`(신규 2.0, ContactSensor)**: jaw·gripper 두 손가락이 같은 큐브에 물리 접촉 시 보상(`force_matrix_w`). 기하 proxy보다 직접적 grasp 신호. 로봇 spawn `activate_contact_sensors=True` + jaw/gripper 센서(큐브 필터). 필터 목록은 모듈 상수(클래스 속성이면 InteractiveScene이 asset 오인).
  - **그리퍼 slew 5→2.5 rad/s**(pick_cube 전용 dict override): 닫을 때 명령속도↓로 큐브 안 튕김 → 정렬 유지(valley 실원인 완화). 공유 상수 불변.
  - **pre-grasp open 0.90→0.65**: 너무 벌린 부트스트랩 시작이 "닫기 마무리" 학습 방해 → 큐브 받아들일 최소폭.
  - **RND를 grasp_focus(30dim) 부분공간으로**: 전체 83dim novelty가 reach 후 무관 noise 추구하는 것 방지. `grasp_focus_state`(joint pos/vel+grasp point+gripper+active cube pos/vel/rel/yaw). `--rnd_state_group grasp_focus`.
  - **batch**: num_steps_per_env 48→24, epochs 8→10, mini_batches 4→8(미세탐색 density↑). bootstrap anneal 800→1000.
  - 검토 메모: transition/aperture 보상은 contact가 직접신호라 중복·충돌 위험으로 제외. arm scale↓는 reach 깨져 기각. **단일 큐브 목표 eval≥0.80**(4큐브 산술벽 대비 현실 기준).
  - 판정 leading: scratch.grasp/lift 점화 + 학습 로그 grasp_contact/grasp_close 성장. 미점화 시 다음 레버: contact weight↑·force_threshold↓·pre-grasp 더 닫기.

### T22. ✅ grasp 점화 돌파 완전 확정 (v4) — 탐색 벽 넘음
- **결과(cron 30분 점검 추이, scratch)**: grasp 0.107(model_150)→0.30→0.53→0.52→**0.84**(650)→0.71(750). lift 0.18→**0.81**. over_bowl 0.07→0.71. v1~v3 전 구간 **scratch.grasp ≡ 0**(align hover local-optimum)이던 탐색 벽을 v4가 안정적으로 넘음.
- **효과 확인된 v4 개입**: `grasp_contact_reward`(ContactSensor 양손가락) + `grasp_close_reward`(close-bridge 3.0) + 그리퍼 slew 2.5 + RND grasp_focus(30dim) + 보상 rebalance(align 1.0/close 3.0).
- **monitor_eval 집계 버그 수정**: `placed=0`인데 `success>0` 모순 발견. 원인 = `ever[stage] |= flag & ~done_mask` 게이트가 success termination(=done) step의 placed/over_bowl 누적을 제외. 성공은 정의상 모든 선행 단계 통과 → 카운트 시 함의 처리(단조성 보장: success⊆placed⊆over_bowl). `monitor_eval.py` 수정.

### T23. ⏳ place/release valley 개입 (v5, fresh) — 그릇 위 떨구기 + 그릇 교란 패널티
- **진단(v4 정체)**: grasp 해결 후 병목이 **place**로 이동. scratch `over_bowl 0.71 → placed 0.23`. 영상 관찰: 큐브를 그릇 위로 가져가도 **그리퍼를 안 연다(떨구질 못함)**. 추가로 운반/place 중 **그릇을 밀치거나 엎는** 경우 관찰.
- **원인**:
  1. **release valley**: 잡은 채 그릇 위 hover = carry(8)+transport(8)+place_height(≤30)+insert(80, 그리퍼 무관)을 계속 받음. 그리퍼를 여는 순간 carry/place_height 즉시 끊기고 큐브가 튈 리스크 → "여는 행위"의 기대값이 "잡고 버티기"보다 낮은 local optimum(grasp 때 align-hover와 동형).
  2. **gripper offset 0.20**(do-nothing=닫힘) → open(>0.6)은 능동적 큰 +action 필요, LSTM은 grasp 유지로 닫는 쪽 바이어스.
  3. **그릇이 동적 rigid body인데 obs에 그릇 center(3)만 있고 자세(quat) 없음** → 정책이 엎은 걸 관측조차 못 함(부분관측). 참고: success termination(`task_done`)은 `require_open=False` — 그리퍼 안 열어도 큐브가 그릇 안에 들어가면 성공. 즉 엄밀히 "release 실패"가 아니라 "그릇 안에 안정적으로 넣기" 병목.
- **개입(v5, fresh 재시작 — obs 차원 변경으로 v4 ckpt 비호환)**:
  - **그릇 quat obs**(`rl_state include_container_orientation`): cup quat wxyz +4 → **83→87dim**. 동적 그릇 tilt/엎힘 관측(부분관측 해소).
  - **`over_bowl_drop_reward`(신규, weight 12)**: 들린 큐브가 그릇 XY 위일 때 그리퍼 open_frac을 dense 보상(inside 게이트 없음). carry/transport 잡고-버티기 탈출 → 그릇 위에서 손 펴 떨구기 유도. 바닥까지 안 내려도 됨(그릇 내부 미끄러워 위에서 떨궈도 중앙 정착, [[bowl-interior-slippery]]).
  - **`bowl_disturb_penalty`(신규, weight -5)**: reset 직후 저장한 그릇 초기 pose(quat·xy) 기준 tilt(1-cosθ, 주신호)+xy변위(disp_coef 4) 패널티. `PickCubeEnv._reset_idx`가 randomize_bowl 직후 `_bowl_init_quat/_bowl_init_xy` 저장. weight 작게(과하면 그릇 근처 접근 회피).
  - carry(8)는 부트스트랩 하류 학습에 중요해 유지 — drop(12)>carry(8)로 여는 게 이득이게 설계.
- **나머지는 v4 그대로**: 16384env, gamma 0.997, RND grasp_focus, 부트스트랩 0.75→0(anneal 1000), batch 24/10/16, grasp 보상(align1/close3/contact2).
- v5는 iter 170(grasp 재점화 전)에서 **v6로 흡수**(아래) — 결과 미수집.

### T24. ⏳ grasp 신뢰성 — 큐브 변위 패널티 + 추락 종료 (v6, fresh)
- **진단(사용자 관찰)**: reach→grasp 과정에서 그리퍼/팔이 큐브를 **쳐서 밀어냄**. 심하면 도달 불가 영역/책상 아래로 추락 → 영영 못 집음(에피소드 실패). 이게 grasp 신뢰성(병목 ①, reach 1.0인데 grasp ~0.8)의 일부. **통찰: 제대로 집으면 큐브를 안 친다 → 큐브 변위 = 정밀 grasp의 proxy.**
- **단계 전이 분석(v4 model_750 scratch)**: reach 1.0 → grasp 0.71 → lift 0.81 → over_bowl 0.71 → placed/success 0.23. 병목 두 곳 = ① grasp 신뢰성(~0.8) ② over_bowl→placed 전이(~0.32). success ≈ 곱 0.23. 90%엔 각 전이 ~0.97+ 필요.
- **개입(v6, fresh — obs 87dim 불변, 보상/종료만 추가)**:
  - **`cube_predisturb_penalty`(신규, weight -3)**: 잡기 전(책상 위·안 들린) 큐브가 reset 초기 xy에서 밀려난 거리 패널티. `~lifted` 게이트(들어올린 변위는 의도된 거라 제외). 정밀 접근하면 변위≈0, 거칠게 치면 패널티 → "큐브 최소 이동으로 감싸 잡기" 유도. `PickCubeEnv._reset_idx`가 scatter/부트스트랩 후 `_cube_init_xy`(N,n_cubes,2) 저장.
  - **`cube_lost` termination(신규, time_out=False)**: 활성 큐브가 책상보다 0.10m 아래 추락하면 실패 종료. 회복 불가 에피소드 컷(학습 낭비 방지) + early term으로 '그 큐브 가치 0' critic 전파 → 안 쳐내도록 압력. 비활성 큐브는 active_cfgs 주입으로 제외(apply_curriculum).
- **v5 개입(T23: 그릇 quat obs 87dim + over_bowl_drop 12 + bowl_disturb -5) 포함** — v6 = v4 + T23(place) + T24(grasp 신뢰성) 통합. run `lstm256_stage1_grasp_v6`, 로그 `train_grasp_v6.log`.
- **판정**: ① grasp 신뢰성(reach→grasp 전이↑) + cube_predisturb 음수 작아짐(큐브 덜 침) + cube_lost 종료율↓ ② placed가 over_bowl 따라 상승. scratch.success→0.80 = 단일 통과 → 커리큘럼 1→2.
- **멀티 큐브 전망(분석)**: 4큐브 0.90 = 큐브당 97.4%(산술 벽). 멀티 특유 문제 = 이미 든 큐브 교란(bowl_disturb 유리)·distractor·그릇 포화·순차 선택(grasp_focus/부트스트랩이 CUBE_NAMES[0] 단일 전제 → 재설계 필요). 현실 목표 = 단일 큐브 0.90.
- **참고(모델 검토)**: obs 87dim이 위치+속도+방향 다 포함 = near-MDP(거의 완전관측) → 이론상 MLP로 충분, LSTM 이득 작음. LSTM 유지 근거 = 요구사항 + sim2real(부분관측 vision student로 갈 때 recurrent 연속성) + 약한 obs 노이즈 필터. 정체 원인은 아키텍처 아니라 보상/탐색(valley). Transformer는 현 규모/on-policy엔 오버킬.

### T25. ⚠️ v6 결과 — grasp 신뢰성 성공 / place 미해결 + jitter·hover 발견 → v7 개입
- **v6 결과(cron scratch 추이)**: grasp/lift/over_bowl이 model_300 0.39 → model_600~1250 **0.85~0.89**(v4 0.71~0.84보다↑·안정). 그러나 **placed/success 0~0.10**(over_bowl→placed 전이 ~12%, v4의 32%보다 오히려 나쁨).
  - **✅ grasp 신뢰성(T24) 성공 확정**: cube_predisturb -0.40→-0.077(큐브 거의 안 침), cube_lost 6.9%→3.4%(추락 절반). 큐브 변위 패널티+추락 종료가 의도대로 작동.
  - **❌ place valley 미해결 + 정밀도가 진짜 병목으로 판명**: over_bowl_drop 보상 2.7로 크게 받는데 placed 0.10 → "여는 행위는 늘었으나 그릇 안 안착 안 됨". release valley(안 열어서) 아니라 **정밀도(열어도 그릇 안 정확히 안 들어감)** 가 주범.
- **영상 관찰(2가지 비정상 행동)**:
  1. **큐브 든 채 위아래로 진동(jitter)**: action_rate/joint_vel 페널티 -1e-4(사실상 0) → jittery action 미억제. guided_lift(10, 로그 6.88 최대)가 높이 보상이라 진동해도 평균 보상 받음. sim2real 치명적.
  2. **그릇 위 hover(place 안 함)**: over_bowl_drop의 open_frac(0.2~0.6 선형)이 "**살짝 열고 버티기**"도 부분 보상 → 큐브 안 떨구는 새 캠핑. + bowl_disturb(-5) 위축 + place 정밀도 부족.
- **v7 개입(보상 param만, obs 87dim 불변, fresh)**:
  - **smoothness**: action_rate/joint_vel weight **-1e-4 → -1e-3**(10×) — jitter 억제. sim2real 필수.
  - **over_bowl_drop 캠핑 차단**: close_ref **0.20→0.40**(거의 다 열어야 보상, 살짝 열기 0) + xy_range **0.10→0.06**(그릇 중심 정밀).
  - **place 정밀도**: place_height xy_range **0.18→0.08**(그릇 중심 정렬, 가장자리 hover 보상 차단).
  - **bowl_disturb 완화**: **-5→-3**(그릇 근접 위축 완화 — 정밀 place 허용).
  - run `lstm256_stage1_grasp_v7`, 로그 `train_grasp_v7.log`.
- **판정**: ① jitter 감소(action_rate 페널티 절대값↓, 영상 매끄러움) ② over_bowl→placed 전이가 v6 0.12 넘어 상승(정밀도 해소). scratch.success→0.80.

### T26. 🔧 학습 속도·안정성 묶음 — reward 스케일 재조정 + epochs↓ + MLP (v8, fresh)
- **배경**: 웹 검색(2025~2026 실사례, Isaac Lab/RSL-RL/PPO 안정성)으로 속도·안정성 레버 정리 → 우리 병목은 속도 아닌 수렴이라 안정성·보상설계 우선. 사용자 지시로 3종 적용.
- **① reward 스케일 재조정(정규화)**: rsl_rl PPO는 advantage norm만 지원, **extrinsic reward/returns normalization 없음**(`ppo.py` 확인). 보상 weight 1~200 광범위 → value target 분산 큼(불안정). 큰 sparse 항 압축으로 value target 분산↓: **task_success 200→50, early_finish 100→30, insert 80→40, place_height 30→20**. 시간 형상화 쌍 일관: **time_penalty -0.02→-0.006**(success 압축에 비례, "빨리 끝내기"가 성공 압도 방지). **행동 교정 페널티(bowl_disturb/cube_predisturb/action_rate/joint_vel)는 유지** — 절대값 작아 분산 주범 아니고 상대 비율이 곧 교정 의도(흔들기/교란 억제).
- **② epochs 단축**: num_learning_epochs **10→6**. iter 속도↑(sample efficiency 약간 희생). MLP라 BPTT 없어 부담 적음.
- **③ MLP 전환**: `--recurrent` 제거 → feedforward ActorCritic. obs 87dim near-MDP(완전관측)라 LSTM 메모리 이득 작음. hidden **[256,128]** 통일(train.py, MLP도 LSTM과 동등 용량). **요구사항(LSTM)에서 벗어남 — 사용자 결정**. sim2real 부분관측 단계에선 recurrent 재도입 필요할 수 있음(현 단계는 privileged state라 MLP 적합).
- **cron 수정**: monitor_eval 호출에서 `--recurrent --rnn_*` 제거(MLP ckpt 로드). `--obs_normalization`만 유지.
- run `mlp_stage1_grasp_v8`, 로그 `train_grasp_v8.log`. 나머지 v7 동일(smoothness·place 정밀도·그릇/큐브 패널티·RND·부트스트랩·gamma 0.997).
- **판정**: ① value_function loss 안정·수렴 속도(MLP가 LSTM 대비) ② grasp 재점화·place 전이가 v6/v7 수준 이상 ③ jitter. MLP가 LSTM보다 못하면 LSTM 복귀(아키텍처 A/B 비교 겸).

### T27. ❌ v8 MLP 발산 → LSTM 복귀 (v9). monitor hidden mismatch 버그
- **v8 결과: 발산(폐기)**. iter 345 **action noise std 0.5→55.53 폭발**(정상 ~1) → 액션이 clip 끝값 포화 = 랜덤 bang-bang. grasp 점화 0, **cube_lost 종료 20.9%**(랜덤 액션이 큐브 쳐냄), value loss 6.6. LSTM(v4~v7)은 entropy 0.02에서 std 정상(~1.3)이었으나 **MLP는 entropy 0.02에 민감해 발산**(entropy 항이 policy gradient 압도). near-MDP라 이론상 MLP 가능했으나 실측 불안정.
- **부수 버그(수정)**: MLP 전환 때 train.py hidden만 [256,128]로 고치고 **monitor_eval.py는 [128,128] 방치** → cron이 v8 ckpt 로드 시 `size mismatch`(actor.0 [256,87] vs [128,87])로 **매 점검 실패**. monitor_eval hidden도 [256,128] 통일.
- **결정(사용자): LSTM 복귀**. run `lstm256_stage1_grasp_v9` — LSTM(256,1)+epochs 10(v7 동일) + **v8 reward 재조정 유지**(발산 원인은 MLP지 reward 아님, 압축은 value 안정 이득). cron도 `--recurrent` 복구. 즉 v9 = v7(검증된 LSTM) + reward 스케일 재조정.
- **교훈**: ① 아키텍처 변경 시 train/monitor/eval 등 **policy_cfg를 구성하는 모든 스크립트 동기화**. ② near-MDP라도 MLP가 entropy 설정에 민감해 발산 가능 — MLP 쓰려면 entropy_coef↓ 필요(미검증). LSTM이 이 task엔 더 안정적(요구사항과도 합치).

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

## 7. 현재 진행 (2026-06-10 세션, T21 fresh grasp_v4)

- **학습 중**: run `lstm256_stage1_grasp_v4`, 로그 `train_grasp_v4.log`. **처음부터(fresh)**, stage-1(단일 큐브), LSTM(256,1층)+PPO, **num_envs 16384**(큐브 머티리얼 통합으로 64K 한도 회피 — TROUBLESHOOTING §materials), iter 0→1500, VRAM ~23GB, ~11s/iter.
- **전체 개입 묶음(T15~T21)**: obs rl_policy **83dim**(방향·크기) + RND용 grasp_focus **30dim**, 보상 align **1.0**/close **3.0**/**contact 2.0**(ContactSensor)/guided_lift 10/carry 8/…, 그리퍼 slew **2.5**, pre-grasp open **0.65**, graded 부트스트랩 0.75→0(anneal 1000), 비활성 큐브 비활성화, batch num_steps24/epochs10/mini8, gamma 0.997.
- 판정: **scratch.grasp/lift 점화**(monitor scratch 그룹) → **scratch.success ≥0.80** = 단일 큐브 통과 → 1→2→3→4 확장. (train 로그 success는 bootstrap-inflated, 진짜 지표 아님.)
- 30분 모니터 루프(`monitor_eval.py`, 카메라=사용자 보정 고정뷰)가 scratch/full/pre 단계별 + 16-env 비디오 자동 생성. 상태: `/tmp/train_pid.txt`, `/tmp/lstm_monitor_state.txt`.

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
| PPO(v4) | num_steps_per_env **24**, learning_epochs **10**, mini_batches **16**(16384env서 ~24.5k/batch), lr 3e-4(adaptive), entropy 0.02, **gamma 0.997**, lam .95, clip .2 |
| **탐색(RND)** | `--rnd` weight 0.5(×step_dt) linear→0, num_outputs 64, predictor/target [256,128], state/reward norm on, **rnd_state=grasp_focus(30dim 부분공간)** |
| **관측 차원** | rl_policy **83dim**(속도+방향·크기): joint_pos6+target6+gripper_pos3+cube12+bowl3+rel12+gripper1 +joint_vel6+ee_vel3+cube_vel12 +cube_yaw8+half_extent4+ee_quat4+grasp→cup3. 단일 스테이지 비활성 큐브 0 마스킹(num_active) |
| 속도 보상 | `time_penalty` weight -0.02(미완료 step당), `early_finish_bonus` weight 100(완료 시각 비례, 종료 1회) |
| 단계 보상(T21/v4) | reach 1, **grasp_align 1.0**(열림+정밀), **grasp_close 3.0**(닫힘+정밀, close>align→단조), **grasp_contact 2.0**(ContactSensor 양손가락 접촉), pregrasp 0.2, **guided_lift 10**, grasp 1, **carry 8**, lift 2, transport 8, place_height 30, insert 80, release 10, task_success 200 |
| 그리퍼 slew | pick_cube 전용 dict: arm 5.0, **gripper 2.5 rad/s**(부드러운 닫기). 공유 상수 불변 |
| 부트스트랩 | prob 0.75→0 annealing(v4 anneal_iters **1000**), graded full↔pre-grasp(p ramp), **pre-grasp open 0.65**, close -0.15, grasp point=default 자세 캐시 |
| batch(v4) | num_steps_per_env **24**, learning_epochs **10**, mini_batches **8** |
| 비활성 큐브 | `num_active`=active_objects, 비활성 큐브 지면 아래(z=-1.0) 비활성화(T20) |
| gripper | **init/offset 0.20**(do-nothing 닫힘쪽 유지로 잡은 큐브 안 놓침, open 1.20 까지), open>0.6 / close<0.5(<0.26 강) |
| 큐브 DR | scatter x[1.60,2.08] y[-0.47,-0.33] yaw±30°(도달 검증 max 0.333m<0.44), 마찰/질량 startup DR, rl_state GaussianNoise σ0.005 |
| GPU 버퍼(16384) | gpu_max_rigid_patch_count 16·2¹⁶, aggregate 1M, collision_stack 2²⁹ |
| num_envs | **16384**(VRAM ~23GB). 큐브 4개 CubeFriction→공유 1개로 env당 물리 머티리얼 6→3, PhysX 64K 한도 회피(TROUBLESHOOTING). 8192 초과 불가였던 원인 |
| viewer(영상) | eye (1.90,0.95,0.98) lookat (1.85,-0.32,0.76) res 1280×720 |
| 금지선 | container_radius_scale 1.0 고정, grasp weld 없음 |

---

## 10. 모니터링 / 상태 파일

- 학습 로그: 최신 `train_grasp_v2.log`(또는 `train_*.log`).
- 상태 파일: `/tmp/train_pid.txt`(TRAIN_PID), `/tmp/lstm_stage.txt`(현재 활성 큐브 수), `/tmp/lstm_monitor_state.txt`(마지막 평가 체크포인트 index).
- 체크포인트/영상: `outputs/rl/rsl_rl/lstm_ppo_pickcube/<run>/` (`model_*.pt`, `videos/monitor/*.mp4`).

### `monitor_eval.py` — 단계별(env-type별) 성공률 + 그리드 비디오 (30분 주기)
- scratch(정상시작=진짜 실력) / full-grasp / pre-grasp 부트스트랩을 **따로 집계**. 단계: reach→grasp→lift→over_bowl→placed→success.
- 16-env 그리드를 한 화면에 담아 녹화(`video_length` step). **multi-env 과노출 근본 해결**: 광원을 scene.usd(per-env 복제)에서 빼고 `PickCubeSceneCfg`가 `/World/Light`(dome)·`/World/KeyLight`(distant)에 단일 author → env 수 무관 복제 안 됨(IsaacLab #4340). 모니터는 추가로 광원 개수 정규화(혹시 복제본 있으면 1/k) + **천장(`/Scene/Ceiling`) 비가시화**(부감 시 책상 가림 방지) + 실제 큐브 world 좌표로 oblique 카메라 자동 프레이밍.
- `--force_kind {1,2}`: 전부 full/pre-grasp 강제 시작 데모(grasp_offset 캐시 후 전체 강제 reset → step0부터 해당 상태 녹화). 0=혼합(scratch/full/pre).
- 첫 에피소드는 `_grasp_offset` 미캐시라 전부 scratch → bootstrap 그룹은 2번째+ 에피소드에서 채워짐(num_episodes 넉넉히).
- 카메라 CLI: `--cam_side`(±좌우)·`--cam_back`(거리)·`--cam_height`(높이)·`--cam_look`(고개) — 그리드 span 배수.
- **`--gui`**: Isaac Sim GUI 로 추론 실행(headless 해제, 녹화·집계 없음). 뷰포트에서 직접 카메라 이동(Alt+좌클릭 회전 / Alt+중클릭 pan / 스크롤 줌), 5초마다 현재 카메라 eye/target 을 set_camera_view 형식으로 콘솔 출력 → 원하는 뷰 값 확보용. (DISPLAY 있는 세션에서 실행.)
- 표준 호출:
  ```bash
  D=$(ls -td outputs/rl/rsl_rl/lstm_ppo_pickcube/*grasp_v2 | head -1)
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src $ROOT/.venv/bin/python \
    scripts/reinforcement_learning/monitor_eval.py --recurrent --rnn_hidden_dim 256 \
    --rnn_num_layers 1 --obs_normalization --checkpoint "$D/model_<N>.pt" \
    --num_envs 16 --num_episodes 48 --max_steps 5000 --active_objects 1 \
    --bootstrap_prob 0.6 --pregrasp_frac 0.5 --video --video_length 450 --device cuda:0
  ```
- **판정 지표**: `scratch.grasp`>0 점화(annealing 후반 핵심), `scratch.success` 추이(→0.90 = 단일 큐브 통과). full/pre_grasp 는 하류·grasp-행동 역량 진단.
- `eval.py`(부트스트랩 0, 단일 success_rate 만): 최종 합격 판정용.

---

## 11. 미해결 / 리스크 / 전망

- **정상-env grasp 점화**가 핵심 미확정. 부트스트랩 value 전파 + hold 수정(T13)으로 유도 시도 중. `grasp_cube`>0 가 leading indicator. 안 켜지면 추가 레버: grasp-point↔cube dense 정렬 보상, bootstrap_prob annealing, grasp 허용오차 점진 축소.
- **곡선 전망**: manipulation RL 은 보통 sigmoid/계단(grasp "클릭" 시 급상승). 현재는 클릭 전 log 포화(~0.13). 개입으로 클릭 유도 필요.
- **4큐브 ≥0.90 의 산술적 벽**: 한 에피소드 4개 순차 → success ≈ (큐브당 성공)⁴. 0.90 하려면 **큐브당 ~97.4%** 필요. 단일 큐브를 거의 완벽히 해야 가능 — 매우 도전적.
- sim2real: 현재 DR(포즈/마찰/질량/관측노이즈)에 actuator gain·joint friction 미포함(예정, gear_assembly 핵심).
- 정밀 grasp 가 끝내 부족하면 모방학습(SmolVLA/GR00T, 레포 본래 경로) 또는 planning(cuMotion, PATH E) 병행 검토.
</content>
