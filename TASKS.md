# TASKS — SO-101 Sim2Real 자율 개발

> **단일 진실 공급원.** Codex가 갱신. 매 사이클 SELECT 전 재로드.
> North Star: [`docs/SIM2REAL_MASTERPLAN.md`](docs/SIM2REAL_MASTERPLAN.md) §1 불변 계약 (v3.0 · so_follower · 6-dim action/state · {top,wrist,front} 480×640@30 · PickCube task 문자열).
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
- [x] **TA.CUBE.PHYSICS** PickCube cube_task 물리 검증/튜닝 (큐브·그릇·데스크매트·책상·그리퍼 contact, mass/friction/offset/drive) | machine:any | dep:TA.1,TA.3 | verify:`pick_cube_physics_smoke.py` static_usd + settle + grasp_hold pass, 실패 시 USD/actuator 물성 조정 후 재검증 | status:done
- [x] **TA.CUBE.STATE_MACHINE** PickCube rule-based state machine으로 cube_desk pick-and-place 가능성 입증 + LeRobot v3 3cam episode 저장 | machine:server | dep:TA.CUBE.PHYSICS | verify:`pick_cube_state_machine.py` 1-cube fixed-spawn placed_and_released pass + `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_success_100s_retry_20260604` schema PASS | status:done

## Phase B — RL 전문가 (state-based)

- [x] **TB.1** 단계형 reward 구현 (reach→grasp→lift→transport→insert→release + success + action-rate 페널티) | machine:any | dep:TA.1,TA.2 | verify:`reward_smoke.py` reward term 9개 등록 + 단계별 증가 pass, `env_smoke.py` 500-step pass, `drive_response_smoke.py` pass | status:done
- [x] **TB.2** rsl_rl PPO train 래퍼 `scripts/reinforcement_learning/train.py` | machine:server | dep:TB.1 | verify:100-step smoke 무크래시 + 체크포인트 저장 | status:done
- [x] **TB.3** PickCube RL 전문가 full 학습 no-assist 재시작 (2048–4096 env, 카메라 off, PPO `num_learning_epochs>=20`) | machine:server | dep:TA.CUBE.PHYSICS,TA.CUBE.STATE_MACHINE,TB.2 | verify:`rg`로 grab/teleport 보조 코드 0건 + `train.py` default PickCube/no-assist/20epoch + checkpoint 산출 | status:done
- [ ] **TB.4** PickCube 커리큘럼 spawn 영역 점진 확대 (현재→목표) | machine:server | dep:TB.3 | verify:`eval_success.py` PickCube full cube/bowl spawn success_rate ≥0.7, `max_episode_steps>=900` | status:in_progress

## Phase C — 데이터 엔진 (롤아웃→LeRobot v3)

- [x] **TC.1** `scripts/sim/rollout_to_lerobot.py` recorder (leisaac LeRobotRecorderManager 대체) | machine:any | dep:TB.3 | verify:10ep 변환 후 `validate_lerobot_schema.py` 통과 | status:done
- [x] **TC.2** 200ep 파이프라인 관통 (DR + 3캠 렌더 + 성공 ep만 필터) | machine:server | dep:TC.1 | verify:validate 통과 + success filter 동작 확인 | status:done
- [x] **TC.3** (선택) segmentation 배경 오버레이 (squint식, 카메라별 정합) | machine:server | dep:TC.2 | verify:카메라별 합성 프레임 육안/지표 점검 | status:done
- [ ] **TC.4** PickCube 대량 롤아웃 (2k–5k 성공 ep) → HF push | machine:server | dep:TB.4,TC.2 | verify:새 PickCube checkpoint 명시, validate 통과 + ep 수 목표 + `/DISK1` 용량 확인 | status:todo

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
- [2026-06-05] TB.4 in_progress — speed-cap PPO 부분관측 보정으로 `rl_policy`를 37→43-dim 확장(현재 processed joint target 6개 추가), 서버 env_smoke에서 `rl_policy (43,)`와 train smoke Actor/Critic input 43 확인; cap2.0 회복 실험은 model550/model706/model715 모두 0/128이라 폐기, cap은 1.0rad/s 유지 / 다음: 43-dim fixed-spawn PickCube를 scratch PPO로 재학습
- [2026-06-05] TB.4 in_progress — 로봇팔 target 속도 제한 추가(`SlewLimitedJointPositionActionCfg`, PickPen/PickCube 정책 cap 1.00rad/s, teleop 기본 30Hz + controller 0.20rad/s cap, state-machine arm 0.006rad/step), 로컬/서버 py_compile + PickCube env_smoke 5-step 통과; model715은 0.20/0.50/1.00rad/s cap 모두 eval 0/128로 붕괴 / 다음: speed-cap 환경에서 해당 stage 재학습
- [2026-06-05] TB.4 in_progress — speed-cap 전 object0.30+Bowl0.25는 model714 deterministic 93/128 통과, object0.30+Bowl0.2625는 model715 deterministic 95/128 통과, Bowl0.275/0.30은 아직 실패 / 다음: 제한 적용 환경에서 0.2625 재확인 후 0.275 진행
- [2026-06-05] TB.4 in_progress — staged PPO curriculum으로 Bowl0.05/0.10/0.25 및 object0.10+Bowl0.25 통과, 현재 best object0.25+Bowl0.25 `/DISK1/so101-sim2real/outputs/tb4_pickcube_obj025_bowl025_std001_from698_short_4096_20260605/model_706.pt` deterministic 103/128(0.8047); object/Bowl 0.35+는 실패 / 다음: 0.30 축 분리 curriculum
- [2026-06-05] TB.4 in_progress — clean model550은 Bowl scale0.05 deterministic 91/128로 0.7 통과하지만 stochastic은 23/128, std override 0.01도 70/128로 노이즈 민감; `train/eval_success.py --override_policy_std` 추가 / 다음: 낮은 std·낮은 entropy로 scale0.05 PPO resume
- [2026-06-05] TB.4 in_progress — state-machine expert `.pt` 6 success/2 fail 생성 후 BC warm-start 2종(raw target·clipped target) 시도, 둘 다 dynamic Bowl deterministic eval 0/128로 폐기; BC target clipping(`--target_clip_actions 1.0`)은 보정으로 유지 / 다음: clean `model_550.pt`에서 Bowl scale 0.05~0.10 소 curriculum PPO 재시도
- [2026-06-04] TB.4 in_progress — 동적 Bowl 기준 PPO continuation 3종(old model749, clean model550 scale0.25, clean model550 Bowl-only scale0.25)이 모두 baseline 대비 하락; PPO-only continuation 중단 / 다음: state machine expert trajectory 기반 BC/warm-start 설계 및 구현
- [2026-06-04] TB.4 in_progress — Bowl/PenCup reset 랜덤화 후 reward/termination이 고정 cup xy를 쓰던 버그 수정(`cup_cfg`로 실제 pose 사용), 4096 env PhysX aggregate capacity 256k로 상향; 이전 TB.4 eval은 고정 좌표 기준이라 폐기, 동적 Bowl 기준 baseline model749 scale0.25 stochastic 29/128·scale1.0 11/128 / 다음: `/DISK1/so101-sim2real/outputs/tb4_pickcube_dynamic_bowl_s025_4096_from749_20260604` 학습 완료 후 재평가
- [2026-06-04] TB.4 in_progress — TB.3 best no-assist checkpoint는 `/DISK1/so101-sim2real/outputs/tb3_pickcube_noassist_1cube_fixed_placeboost_cont_2048_20260604/model_550.pt`, 1-cube fixed eval deterministic 87/128(success_rate 0.6797)·stochastic 81/128(0.6328); fine-tune model600/624는 하락 / 다음: model550에서 spawn/cup curriculum 확대
- [2026-06-04] TB.3 done — jaw-offset grasp point로 RL obs/reward 정합, PickCube success reward와 termination 조건 일치(`require_open=False`), place/insert shaping 강화, `resume_without_optimizer` 추가; no-assist PPO 20epoch+ checkpoint 산출 및 eval 완료 / 다음: TB.4 커리큘럼
- [2026-06-04] TA.CUBE.STATE_MACHINE done — `pick_cube_state_machine.py` 추가, joint command slew limit(`0.01rad/step`) + 느린 gripper close(`0.005rad/step`) + grasp retry(max 3)로 1-cube fixed-spawn pick-and-place 입증; 서버 3cam LeRobot v3 dataset `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_success_90s_slowlimit_20260604` 생성(2700 frames/90.0s/schema PASS, placed_and_released true) / 다음: no-assist PickCube PPO 재학습(TB.3)
- [2026-06-04] TA.CUBE.PHYSICS done — `pick_cube_physics_smoke.py` static_usd/settle/grasp_hold pass(`outputs/pick_cube_physics_smoke_after_actuator.json`), leisaac actuator 이식+cube/desk 물성 보정 후 fixture contact hold 검증 / 다음: rule-based state machine gate
- [2026-06-04] PickCube pivot in_progress — 목표를 PickPen에서 PickCube/cube_task로 전환, 사용자 GUI 카메라 튜닝값은 PickCube cfg에 반영된 상태, 기존 PickPen assisted RL/rollout 결과는 새 목표에서 사용하지 않음 / 다음: `pick_cube_physics_smoke.py`로 물리 gate 후 no-assist 20epoch+ PPO 재학습
- [2026-06-04] TC.4 local GUI teleop camera attach — front camera를 shoulder 링크 자식, wrist camera를 gripper 링크 자식으로 정리하고 `update_latest_camera_pose=True` 적용, shoulder_pan motion smoke에서 front/wrist camera delta 확인, GUI 실행 시 top/front/wrist camera viewport 3개 자동 생성 및 COM5 Leader 연결 확인(PID 14296 → uv 36428 → python 41332/23560) / 다음: 사용자가 viewport와 `C` metadata로 pose/FOV 튜닝
- [2026-06-04] TC.4 local GUI teleop follow-up — floating desk 원인(`SCENE_OFFSET.z=0.92`) 수정 후 USD 재생성, `pick_pen_joint_teleop.py` 삭제, `teleop_se3_agent.py`를 leisaac-free SO-101 Leader(COM5) GUI teleop/camera capture 진입점으로 교체, `C` 키 PNG+metadata 저장 및 visible GUI 실행(PID 9064 → uv 43348 → python 41052) / 다음: 사용자가 카메라 pose/FOV를 튜닝
- [2026-06-04] TC.4 local GUI teleop — `pick_pen_joint_teleop.py`에 SO-101 Leader Arm(COM5) 직접 입력 지원 추가, Gym wrapper GUI render crash 수정(`env.unwrapped.sim.render()`), visible Isaac GUI를 `--control_mode leader --leader_port COM5`로 실행(PID 37300 → python child 28820) / 다음: 사용자가 GUI에서 카메라 pose/FOV·축 방향 점검
- [2026-06-04] TC.4 quality reset — midcheck 영상 품질 문제(카메라 mismatch, 과강한 grasp assist, 짧은 horizon 의심)로 대량 rollout 재개 전 로컬 joint teleop/camera tuning 스크립트 추가, Windows `--experience isaaclab.python.rendering.kit` smoke + snapshot 생성 통과 / 다음: 사용자가 카메라 pose/FOV 튜닝 후 assist·학습 길이 재설계
- [2026-06-04] TC.4 midcheck — 사용자 요청으로 2k rollout을 `1024 successes/1514 attempts`에서 kill하고 별도 10 success ep dataset 생성(`/DISK1/so101-sim2real/outputs/tc4_rollout_10ep_midcheck_codex_20260604`), 10 successes/15 attempts/427 frames, schema PASS, 3cam mp4 valid / 다음: visual 점검 후 2k-5k 재생성+HF push
- [2026-06-04] TC.4 in_progress — 서버 `/DISK1` 여유 3.4T와 GPU 상태 확인 후 최소 목표 2,000 successful episodes로 대량 rollout/HF push 착수, target `/DISK1/so101-sim2real/outputs/tc4_rollout_2000ep_codex_20260604` / 다음: rollout 완료 후 schema validate + HF dataset push
- [2026-06-04] TC.3 done — `segmentation_overlay_preview.py` 추가, TA.3 real/sim 3cam PNG로 Squint-style foreground-mask overlay preview 생성, 서버 산출물 `/DISK1/so101-sim2real/outputs/tc3_segmentation_overlay_preview_codex_20260604_v2`, top/front=color mask·wrist=ROI fallback 기록 / 다음: TC.4 2k-5k success rollout + HF push
- [2026-06-04] TC.2 done — serial 1-env full DR + 3cam rollout 200 success ep 생성(`/DISK1/so101-sim2real/outputs/tc2_rollout_200ep_codex_20260604_0458`), 200 successes/289 attempts/89 failures filtered/10,473 frames, dataset 266MB, validator PASS / 다음: TC.3 optional segmentation overlay
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
