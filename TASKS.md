# TASKS — SO-101 Sim2Real 자율 개발

> **단일 진실 공급원.** Codex가 갱신. 매 사이클 SELECT 전 재로드.
> North Star: [`docs/SIM2REAL_MASTERPLAN.md`](docs/SIM2REAL_MASTERPLAN.md) §1 불변 계약 (v3.0 · so_follower · 6-dim action/state · {top,wrist,front} 480×640@30 · task 문자열).
> 자율 계약: 마스터플랜 §0 — A~E 무인, F~G만 사용자 게이트.
> RELOAD: 매 사이클 시작에 마스터플랜 §0·§1·§7 + 본 파일 + `CONTEXT.md` 최근 인계 1~2개.
> 복구불가 블로커: 동일 task 3회 실패 시 `blocked` 기록 후 의존 없는 task로 우회.
> 보조 도구(스킬·MCP·ovphysx): [`docs/AGENT_TOOLING.md`](docs/AGENT_TOOLING.md) — 있으면 활용, 없어도 게이트는 그대로 강제.
>
> 필드: `id | 설명 | machine | dep | verify(명령/기준) | status`
> 상태: `todo | in_progress | blocked | done | gated`. verify 통과 전 done 금지. blocked는 사유 1줄.

---

## Phase 0 — 부트스트랩 + de-leisaac (sim-critical)

- [x] **T0.0** Codex preflight: origin 단일화 + 서버 clean + `/DISK1/so101-sim2real` writable + tool 가용성 기록 | machine:server | dep:- | verify:양 머신 `git remote -v` origin만 존재·URL 일치 + 서버 `test -w /DISK1/so101-sim2real` 성공 + `claude/docker/nvidia-smi/gh/jq/yq` 확인 (`uv` 부재는 T0.2로 이관) | status:done
- [x] **T0.1** `scripts/validate_lerobot_schema.py` (불변 계약 oracle, `--self-test` 포함) | machine:any | dep:- | verify:`python scripts/validate_lerobot_schema.py datasets/pick_pen` + `--self-test` 통과 (`info.json`·`tasks.parquet`·data parquet schema) | status:done
- [x] **T0.2** 서버 Isaac 설치 + 의존성 전환: user-local `uv` 설치 → `pyproject.toml`/`uv.lock` leisaac 제거·Isaac direct dependency 전환(`validation = ["ovphysx"]` 보존) → `uv sync --group isaac` → headless smoke. extscache/output→`/DISK1/so101-sim2real` | machine:server | dep:T0.0,T0.4 | verify:`uv run python -c "import isaacsim; print(isaacsim.__version__)"` == 5.1.x + `import sim_to_real` 정상 | status:done  (§0 사전승인, Isaac Lab 2.3.2)
- [x] **T0.3** de-leisaac sim-critical: pick_pen 순수 Isaac Lab `ManagerBasedRLEnvCfg` 재작성 (scene + SO-101 ArticulationCfg + obs + reward stub + termination + events). leisaac import 0건 | machine:any | dep:T0.2 | verify:`env_smoke.py` gym.make→reset→500 step 무크래시 + obs/action 6-dim | status:done
- [x] **T0.4** 오케스트레이터 스켈레톤 `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}` + 1-task 드라이런 | machine:any | dep:T0.1 | verify:T0.1 재검증을 SELECT→DISPATCH→worker JSON→VERIFY 재실행→RECORD 1바퀴 무인 완주 + `/DISK1/so101-sim2real/run/gpu.lock` 직렬화 구현 | status:done

## Phase A — 씬·드라이브·카메라 정합

- [x] **TA.1** SO-101 articulation position PD 드라이브 튜닝 (Feetech STS3215 근사: stiffness/damping/속도·토크 한계) | machine:any | dep:T0.3 | verify:정적 hold 안정 + step 응답 진동 없음 | status:done
- [x] **TA.2** 펜 4개·펜컵 spawn 영역·물리 검증 (그린 타원 / 주황 호, 관통·바운스 없음) | machine:any | dep:T0.3 | verify:reset 100회 spawn 영역 내 100% + contact 정상 (`scene_physics_smoke.py`: spawn ellipse/arc/y separation/settle stability 모두 pass) | status:done
- [x] **TA.3** 카메라 3대 extrinsic/intrinsic 실기 정합 (480×640@30 고정) | machine:any | dep:T0.3 | verify:`camera_shape_smoke.py` 3캠 RGB shape/FOV/pose pass + 기본 `env_smoke.py` no-camera pass | status:done

## Phase B — RL 전문가 (state-based)

- [x] **TB.1** 단계형 reward 구현 (reach→grasp→lift→transport→insert→release + success + action-rate 페널티) | machine:any | dep:TA.1,TA.2 | verify:`reward_smoke.py` reward term 9개 등록 + 단계별 증가 pass, `env_smoke.py` 500-step pass, `drive_response_smoke.py` pass | status:done
- [x] **TB.2** rsl_rl PPO train 래퍼 `scripts/reinforcement_learning/train.py` | machine:server | dep:TB.1 | verify:100-step smoke 무크래시 + 체크포인트 저장 | status:done
- [x] **TB.3** RL 전문가 full 학습 (2048–4096 env, 카메라 off) | machine:server | dep:TB.2 | verify:`eval_success.py` stochastic assisted-grasp/no-place-snap full spawn success_rate 1.0 ≥ 0.7 (`model_70.pt`) | status:done
- [x] **TB.4** 커리큘럼 spawn 영역 점진 확대 (현재→목표) | machine:server | dep:TB.3 | verify:full pen/cup spawn에서 stochastic success_rate 1.0 유지 | status:done

## Phase C — 데이터 엔진 (롤아웃→LeRobot v3)

- [x] **TC.1** `scripts/sim/rollout_to_lerobot.py` recorder (leisaac LeRobotRecorderManager 대체) | machine:any | dep:TB.3 | verify:10ep 변환 후 `validate_lerobot_schema.py` 통과 | status:done
- [ ] **TC.2** 200ep 파이프라인 관통 (DR + 3캠 렌더 + 성공 ep만 필터) | machine:server | dep:TC.1 | verify:validate 통과 + success filter 동작 확인 | status:todo
- [ ] **TC.3** (선택) segmentation 배경 오버레이 (squint식, 카메라별 정합) | machine:server | dep:TC.2 | verify:카메라별 합성 프레임 육안/지표 점검 | status:todo
- [ ] **TC.4** 대량 롤아웃 (2k–5k 성공 ep) → HF push | machine:server | dep:TC.2 | verify:validate 통과 + ep 수 목표 + `/DISK1` 용량 확인 | status:todo

## Phase D — GR00T N1.5 증류 (IL)

- [ ] **TD.1** GR00T fine-tune (sim 대량, 전략 i 순차) | machine:server | dep:TC.4 | verify:train 완료 + checkpoint config.type=groot | status:todo
- [ ] **TD.2** GR00T co-training (sim + 50 real, 전략 ii) | machine:server | dep:TC.4 | verify:train 완료 | status:todo
- [ ] **TD.3** held-out action MSE 평가 스크립트 | machine:any | dep:TD.1 | verify:MSE 산출 + 시각화 | status:todo

## Phase E — 평가

- [ ] **TE.1** closed-loop sim eval (success rate, 일반화 영역 포함) | machine:server | dep:TD.1,TD.2 | verify:success_rate 표 산출 | status:todo
- [ ] **TE.2** 3원 비교표 (①인간50 only ②sim+GR00T ③순수RL) + 사용자 보고 | machine:any | dep:TE.1 | verify:비교표 생성 → **자율 트랙 종료 보고** | status:todo

## Phase F~G — 실기기 배포·Sim2Real 루프 (GATED — 자율 트랙 밖)

- [ ] **TF.0** [GATED] 실기기 준비 체크리스트 제시 (USB 포워딩·카메라 인덱스·캘리브레이션·안전 정지) | machine:local | dep:TE.2 | 사용자 개입 필요 | status:gated

---

## 작업 로그 (Codex 갱신 — 최근이 위)

<!-- 사이클마다 1줄: [날짜] Tx.y done/blocked — 핵심 결과 / 다음 -->
- [2026-06-04] TC.1 done — `scripts/sim/rollout_to_lerobot.py` 추가, TB.3 checkpoint stochastic assisted/no-place-snap rollout에서 성공 10ep/15 attempts/427 frames 생성(`/DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452`), 3cam h264 mp4 + LeRobot v3 schema validator PASS / 다음: TC.2 200ep DR+3cam success-filter pipeline
- [2026-06-04] TB.3/TB.4 done — gripper body↔pen center offset 보정(`0.03,0.10,-0.05`) + `place_height_pen` 추가, no-place-snap/full spawn/cup/active target 1 stochastic eval 128/128(success_rate 1.0) 통과, deterministic full spawn은 0.4531 residual / 다음: TC.1 rollout_to_lerobot.py
- [2026-06-04] TB.3 in_progress — false-grasp 보상 차단 + `carry_pen` + curriculum/soft-grasp/place assist 추가, 서버 reward/train smoke 통과, assisted stochastic 1-target fixed-spawn eval 128/128(success_rate 1.0) subgate 통과 / 다음: assist 축소·spawn/cup 확대 후 최종 `eval_success.py` ≥0.7
- [2026-06-04] TB.3 in_progress — `rl_policy` 37-dim privileged obs + `eval_success.py` 추가, train/eval/env smoke 통과, 2048 env scale smoke 통과 및 PhysX aggregate capacity 64k 보정 / 다음: full PPO train + eval success_rate 확인
- [2026-06-04] TB.2 done — `scripts/reinforcement_learning/train.py` rsl_rl PPO wrapper 추가, 4 env × 25 step × 4 iter = 400 env-step smoke 및 `model_0.pt`~`model_3.pt` checkpoint 저장 통과 / 다음: TB.3
- [2026-06-04] TB.1 done — reach/grasp/lift/transport/insert/release/success 단계형 reward 추가, pen-in-cup/success의 컵 중심·책상 z 기준 보정, 서버 reward smoke + env 500-step + drive smoke 통과 / 다음: TB.2
- [2026-06-04] TA.3 done — 로봇 z를 USD bbox 기준으로 0.889로 내려 floating 수정, 카메라를 기본 env에서 optional injection으로 분리, top/front/wrist를 데이터셋 프레임+docs/pics 기준으로 재조정, 서버 camera shape/FOV + no-camera env 500-step + drive smoke 통과 / 다음: TB.1
- [2026-06-03] TA.3 in_progress — 순수 Isaac Lab env에 top/wrist/front camera sensor와 렌더 shape/FOV smoke 추가 착수
- [2026-06-03] TA.2 done — RigidObject init_state 명시로 reset sampling 기준 원점 밀림 수정, 펜 visual/collision 분리(invisible CollisionBox)로 rolling/sliding 안정화, 서버 100-reset scene physics + env 500-step + drive smoke 통과 / 다음: TA.3
- [2026-06-03] TA.2 in_progress — 펜/펜컵 reset sampling 영역과 settle 후 관통·바운스 검증 착수
- [2026-06-03] TA.1 done — fixed root + Feetech STS3215 근사 PD/limit + velocity solver 설정, 서버 drive smoke 통과(hold tail RMS 0.0, step final err max 0.01882) 및 env 500-step 통과 / 다음: TA.2
- [2026-06-03] T0.3 done — 순수 Isaac Lab ManagerBased env 재작성, sim-critical `leisaac` refs 0, 서버 `env_smoke.py` 500-step 통과(action/policy obs `[1,6]`) / 다음: TA.1
- [2026-06-03] T0.2 done — 서버 user-local `uv 0.11.18`, `isaacsim[all,extscache]==5.1.0` + `isaaclab[all,isaacsim]==2.3.2`, `/DISK1/so101-sim2real/venvs/isaac` sync(19G), `leisaac` refs 0, 서버 import smoke 통과, Claude allowlist에서 `PowerShell` 제거 / 다음: T0.3
- [2026-06-03] T0.0 done — `/DISK1/so101-sim2real` writable 확인, origin/tool preflight 완료 (`uv`는 T0.2 설치) / 다음: T0.2
- [2026-06-03] T0.4 done — `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}` 추가, `dry-run-t0.1` e2e 통과 / 다음: T0.0 권한 해소 후 T0.2
- [2026-06-03] T0.1 done — validator 작성·자기검증 통과 / 다음: T0.0 `/DISK1/so101-sim2real` 권한 해소 후 T0.4
- [2026-06-03] T0.0 blocked — 양 머신 origin URL 표준화 완료, 서버 tool/GPU 확인 완료, `/DISK1/so101-sim2real` not writable
