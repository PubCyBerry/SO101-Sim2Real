# 4-cube VLA closed-loop 성능 개선 기록

> 최종 갱신: 2026-06-23
> 대상: SmolVLA, GR00T-N1.7
> 성공 기준: 실제 배포 경로(`policy server → gRPC → ROS 2 VLA node → Isaac Sim bridge`)에서 3분 안에 4개 cube를 모두 bowl에 배치한 비율 **80~90% 이상**(최종 N≥10)

## 1. 요약

기존 4-cube 정책의 0% 성공은 단순히 모델 용량이나 학습 step 부족 때문만은 아니었다. 데이터의 목표 cube 순서가 이미지에서 관측 불가능했고, 평가 파이프라인에도 성공을 거짓 음성으로 만드는 오류와 episode 간 stale action, 동기식 추론에 의한 command refill 정지가 함께 있었다.

현재까지의 결과는 다음과 같다.

| 항목 | 기존 기준선 | 개선 후 |
|---|---:|---:|
| Expert 순서 | `Cube1 → Cube4` 고정 | 현재 관측에서 결정 가능한 `nearest` |
| 데이터 | 과거 물리·기록 규약의 4-cube 1024ep | 현재 물리·절대 gripper 규약의 256ep |
| SmolVLA all-4 성공률 | 0% (N=10) | **40% (2/5)** |
| SmolVLA final per-cube | 0% | **85% (17/20)** |
| GR00T-N1.7 | 기존 모델 0% | visual stage-3 all-4 0%, final per-cube 60%, ever 65% |

SmolVLA는 기존 0%에서 40%까지 개선됐지만 활성 목표 80~90%에는 미달한다. GR00T는 visual grounding 학습으로 최종 배치율이 25%에서 60%로 증가했지만 all-4 동시 성공은 아직 없다. 두 모델 모두 success retention과 실패 복구가 다음 병목이다.

## 2. 기존 모델이 잘 나오지 않았던 이유

### 2.1 관측 불가능한 목표 순서

기존 expert는 항상 USD prim 이름 기준으로 `Cube1 → Cube2 → Cube3 → Cube4` 순서로 행동했다. 그러나 네 cube는 외형이 같고 위치는 매 episode 무작위다. 정책 입력 이미지에는 prim 이름이 보이지 않으므로, 비슷한 관측에 서로 다른 첫 행동 label이 붙는 문제가 생긴다.

```text
이미지에서 보이는 정보: 위치·크기·로봇 상태
expert가 사용한 정보:    숨겨진 Cube1~4 identity
결과:                    label aliasing / multimodal action
```

open-loop에서는 recorded action을 평균적으로 따라가더라도, closed-loop에서 첫 target 선택이 흔들리고 짧은 pick-place 패턴을 반복했다.

### 2.2 데이터와 현재 시뮬레이션 계약 불일치

기존 4-cube 1024 데이터는 다음 변경 전 생성됐다.

- cube 크기 40/50mm 확대
- cube collider를 SDF에서 convexHull로 변경
- gripper action 기록을 pre-offset이 아닌 절대 joint target으로 통일
- 현재 카메라·물리·충돌 설정

따라서 과거 데이터로 학습한 정책은 현재 scene의 접촉과 action 의미를 정확히 학습하지 못한다.

### 2.3 평가 성공 판정 오류

bridge는 cube desk의 실제 높이 `z=0.705m` 대신 공용 pen desk 상수 `z=0.760m`를 사용했다. bowl 안에 정상 배치된 cube도 높이 조건에서 탈락해 성공을 실패로 기록했다.

또한 all-4 상태에 도달해도 episode를 즉시 종료하지 않고 고정 horizon 끝까지 실행했다. 정책이 이미 놓은 cube를 다시 건드리면 최종 결과가 실패로 바뀌었다.

### 2.4 episode reset 뒤 stale action 재사용

Isaac Sim scene만 reset되고 ROS 2 VLA node의 다음 상태가 남았다.

- action queue
- timestep
- observation cache
- 진행 중이던 inference 결과

따라서 새 episode 첫 구간에 이전 episode의 action이 적용됐다.

### 2.5 ROS timer 안의 동기식 gRPC 추론

추론 1회가 약 115~200ms인데 30Hz timer callback 안에서 동기 호출했다. 추론 중에는 command publish와 queue 소비가 함께 멈춰 recorded expert와 다른 비동기 제어가 됐다. open-loop MAE가 낮아도 실제 로봇 상태가 학습 분포에서 빠르게 벗어나는 원인이었다.

### 2.6 평가 horizon 부족

4개 cube를 순차적으로 처리하는 데 필요한 시간보다 30초와 90초가 짧았다. 동일 SmolVLA 설정에서 horizon별 결과는 다음과 같다.

| Horizon | all-4 | Final per-cube | 평균 배치 cube |
|---:|---:|---:|---:|
| 30s | 0% | 30% | 1.2/4 |
| 90s | 0% | 55% | 2.2/4 |
| 180s | **40%** | **85%** | **3.4/4 |

짧은 horizon의 0%를 정책 자체의 완전 실패로 해석하면 안 된다.

## 3. 시행착오와 판단

### 3.1 Expert order A/B

현재 물리에서 N=256으로 네 가지 순서를 비교했다.

| Order mode | Expert all-4 성공률 | 판단 |
|---|---:|---|
| `fixed` | 80.9% | identity가 관측 불가능 |
| `nearest` | 79.3% | 관측 가능, 성능 손실 작음 |
| `left_to_right` | 78.9% | 관측 가능 |
| `isolated` | 76.2% | 관측 가능, expert 성능이 가장 낮음 |

`nearest`는 현재 cube 위치와 robot state만으로 target을 다시 결정할 수 있어 채택했다.

### 3.2 데이터 기록 병목

초기에는 CPU `libx264` 인코딩과 전체 pixel의 float64 image statistics 계산 때문에 4 episode 기록에 254초가 걸렸다.

다음 변경 후 103초로 줄었다.

- 시스템 FFmpeg의 `h264_nvenc`
- preset `p4`, CQ 23
- image statistics를 8×8 stride로 표본화
- float32 통계

출력은 기존 계약인 H.264, 640×480, yuv420p, 30fps를 유지했다.

### 3.3 GR00T 중간 checkpoint 평가

GR00T nearest256 adaptation은 16,277 step까지 정상 완료됐다. 진행 중 로그를 제한된 범위로 조회했을 때 12,730 step에서 멈춘 것으로 오판했으나, 파일 timestamp와 `trainer_state.json`을 다시 확인해 다음 산출물이 모두 정상임을 확인했다.

- `checkpoint-8000`
- `checkpoint-16000`
- `checkpoint-16277` (`global_step=max_steps=16277`)
- 학습 종료 후 `trainer.save_model()`이 기록한 root final model

checkpoint-8000은 APC 16, threshold 0.25, seed 40, N=5, 180초 평가에서 all-4/final per-cube/ever-in-bowl가 모두 0%였다. 따라서 최종 checkpoint-16277로 동일 조건을 재평가한다.

### 3.4 Stage-2 하이퍼파라미터 override 함정

처음 stage-2를 시작할 때 entrypoint는 LR 3e-5와 color jitter off를 출력했지만, 컨테이너가 실행한 것은 이미지에 baked된 NVIDIA `examples/finetune.sh`였다. 실제 최종 명령은 LR 1e-4와 color jitter on이었다.

checkpoint 생성 전에 run을 중단하고, host의 `docker/gr00t-finetune.sh`를 `/host`에 bind-mount해 실제 launcher 인자를 제어하도록 수정했다. 재시작 로그에서 다음을 확인했다.

```text
--learning_rate 3e-5
--warmup_ratio 0.03
--state_dropout_prob 0.0
```

후속 검증에서 `--color_jitter_params`를 완전히 생략하면 base checkpoint 값을 상속한다는 점을 확인했다. 실제 off는 값을 비운 `--color_jitter_params`를 명시해 tyro가 `{}`로 파싱하도록 수정했다.

## 4. 구현 변경

| 파일 | 변경 |
|---|---|
| `scripts/sim/pick_cube_curobo_batch.py` | `--order_mode` 추가, nearest expert와 기록 codec 인자 추가 |
| `scripts/sim/lerobot_recorder.py` | NVENC 경로, image stats 표본화 |
| `scripts/sim/run_cube_desk_ros_bridge.py` | cube desk 높이 수정, all-4 즉시 종료, reset token, VLA action parity |
| `ros2_ws/src/so101_vla_policy/so101_vla_policy/vla_policy_node.py` | reset token 감지, queue/cache 초기화, background inference |
| `ros2_ws/src/so101_vla_policy/config/vla_policy.yaml` | reset token parameter |
| `scripts/run_nearest_256_eval.sh` | 모델별 실제 closed-loop 평가 자동화 |
| `env/smolvla_nearest256.env` | SmolVLA adaptation 전용 profile |
| `env/groot_n17_nearest256.env` | GR00T adaptation 전용 profile |

새로운 종류의 평가 오류는 `docs/TROUBLESHOOTING.md`의 **sim VLA eval 거짓 0%·episode 간 stale action·추론 refill 정지** 절에도 기록했다.

## 5. 데이터와 모델

### 5.1 nearest256 데이터

- 로컬: `outputs/so101_sim_pick_cube_current_nearest_256`
- 규모: 256 episodes, 130,214 frames, 약 1.8GB
- 생성: N=32, seed=20260622, 11 rounds, 1,389초
- 카메라: top/wrist/front 각각 130,214 frames
- action/state: 6-dim, finite 검증 통과
- Hugging Face: `taehunkim/so101_sim_pick_cube_nearest_256`
- Hub commit: `f66067542b1c2b42313a9ac165e6dc0955f3e216`

### 5.2 SmolVLA

- 초기 모델: 기존 4-cube 1024 checkpoint
- adaptation: 4,069 steps × batch 32, 데이터 1 epoch
- 학습 시간: 약 43분
- loss: 약 0.105 → 0.063
- 로컬: `outputs/train/so101_smolvla_sim_pick_cube_nearest_256_adapt/checkpoints/004069/pretrained_model`
- Hugging Face: `taehunkim/so101_smolvla_sim_pick_cube_nearest_256_adapt`
- 평가 최적값: APC 32, threshold 0.25, 180s

### 5.3 GR00T-N1.7

- 초기 모델: 기존 4-cube 1024 `checkpoint-80000`
- adaptation 데이터: nearest256의 GR00T v2.1 변환본
- 중간 모델: `outputs/train/so101_groot_n17_sim_pick_cube_nearest_256_adapt/checkpoint-8000`
- 최종 모델: `outputs/train/so101_groot_n17_sim_pick_cube_nearest_256_adapt/checkpoint-16277`
- 최종 loss tail: 0.0905 → 0.0841
- checkpoint-8000 closed-loop: all-4 0%, final per-cube 0%, ever-in-bowl 0% (N=5)
- checkpoint-16277 open-loop ep0: overall MAE 4.799 (wrist_roll 10.13°, gripper 1.06)
- checkpoint-16277 closed-loop APC16/thr0.25: all-4 0%, final per-cube 25%, avg 1.0/4, ever 30% (N=5)
- checkpoint-16277 closed-loop APC16/thr0.5: all-4 0%, final per-cube 0%, ever 10% (N=5). 더 잦은 갱신에서 cube를 책상 밖으로 밀어내는 불안정성이 증가했다.
- checkpoint-16277 closed-loop APC16/thr0.0: all-4 0%, final per-cube 10%, avg 0.4/4, ever 20% (N=5)
- checkpoint-16277 APC16/thr0.25, seed40, 300s: final 3/4, ever 4/4. 시간 내 모든 cube를 옮겼지만 동시 유지에는 실패했다.
- 같은 조건에 학습 action-term target slew(arm 5.0/gripper 2.5rad/s)를 재적용하면 final 2/4, ever 2/4로 악화됐다. 데이터가 이미 slew-limited라 이중 지연이 발생한 것으로 판단한다.
- stage-2: checkpoint-16277에서 LR 3e-5, warmup 0.03, state dropout 0.0으로 추가 1epoch 완료
- `color_jitter_params` 인자를 생략했지만 이는 off가 아니라 base checkpoint 값을 상속한다. 최종 config에 brightness0.3/contrast0.4/saturation0.5/hue0.08이 유지됐다.
- 16,277 steps, 4,770.9초, train loss 0.0513, 최종 loss tail 0.0402→0.0412, GPU 약 39GB
- stage-2 open-loop ep0 MAE 4.689, closed-loop APC16/thr0.25 seed40 N5 180s: all-4 0%, final per-cube 25%, avg1.0/4, ever25%
- visual encoder+projector smoke(`tune_visual=true`, diffusion frozen, jitter 명시 off) 100-step: trainable 29.76%, batch8 정상, open-loop MAE 4.384. seed40 N1 180s는 0/4.
- visual-only stage-3 1epoch 완료: batch8, LR1e-5, jitter off, state dropout0, 16,277 steps, 5,290.4초, train loss 0.040655
- stage-3 open-loop ep0 MAE 4.453
- stage-3 closed-loop APC16/thr0.25, seed40, N=5, 180s: all-4 0%, final per-cube 60%, avg2.4/4, ever65%
- stage-3 seed44, N=1, 300s: final3/4, ever4/4. Cube1은 한 번 bowl에 들어갔지만 종료 시 bowl 중심에서 103.6mm 떨어진 desk 위로 이탈했다.

## 6. 검증 결과와 산출물

SmolVLA 실제 closed-loop 결과:

```text
APC=32
inference threshold=0.25
seed=40
episodes=5
horizon=180s
all-4 success=2/5=40%
final per-cube=17/20=85%
average placed=3.4/4
ever-in-bowl=90%
```

주요 산출물:

- `outputs/smolvla/01_data_vs_openloop_actual_grpc_ep0.png`
- `outputs/smolvla/02_closedloop_actual_action_vs_state_seed0.png`
- `outputs/smolvla/03_data_openloop_closedloop_summary.png`
- `outputs/smolvla/04_data_openloop_closedloop_first489.png`
- `outputs/smolvla/05_nearest256_closedloop_horizon_tuning.png`
- `outputs/smolvla/metrics.json`
- `outputs/smolvla/nearest256_tuning_metrics.json`
- `outputs/vla_eval_nearest256/smolvla_apc32_thr0p25_seed40_n5_s180.json`

## 7. 다음 작업

1. eval JSON에 bowl 이동량과 cube별 bowl 진입·이탈 이력을 추가한다.
2. 동일 stage-3 checkpoint로 진단 episode를 재실행해 동적 bowl과 정책 재교란을 분리한다.
3. bowl 동역학이 주원인이면 마찰·고정 여부를 현재 expert 회귀와 함께 A/B한다.
4. 정책 재교란/미회수가 주원인이면 clean BC 반복을 중단하고 recovery/perturbation 데이터를 생성한다.
5. SmolVLA와 GR00T를 같은 개선 설정에서 N≥10, 180s로 재평가한다.
6. data/open-loop/closed-loop와 모델별 성공률을 `outputs/smolvla`의 최종 비교 plot으로 저장한다.
7. all-4 80~90%를 통과한 checkpoint만 Hugging Face에 업로드한다.

## 8. 재현 명령

SmolVLA:

```bash
bash scripts/run_nearest_256_eval.sh smolvla 32 0.25 40 5 180
```

GR00T 최종 checkpoint:

```bash
bash scripts/run_nearest_256_eval.sh groot 16 0.25 40 5 180
```

결과 확인:

```bash
jq '{
  all_success_rate,
  per_cube_success_rate,
  avg_cubes_placed,
  ever_in_bowl_rate
}' outputs/vla_eval_nearest256/*.json
```
