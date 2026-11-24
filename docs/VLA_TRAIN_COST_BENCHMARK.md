# VLA/WAM 학습 비용 벤치마크 — 시간·VRAM (200k step 기준)

LeRobot 0.6.0 이 지원하는 VLA/WAM 정책 **7종**을 이 프로젝트의 데이터 형식(2-cam 640×480@30fps,
6D joint state/action)으로 **학습(train)** 과 **파인튜닝(finetune)** 각각 돌려, batch 1→32 스윕에서
**1스텝 시간**과 **VRAM 피크**를 재고 200k step 으로 외삽한다.

> **상태**: 측정 준비 완료 · **측정 미실행**(GPU 경합 중). 아래 §6 은 소스 코드에서 유도한
> **예측**이고 실측이 아니다. §7 은 실행 후 채운다.

대상 GPU = RTX PRO 5000 Blackwell **47.8 GB** (단일 카드, 다른 작업과 공유).

---

## 1. 실행 순서 (요약)

```bash
# 0) 베이스 가중치 미리 받기 — GPU 불필요, 실측 75GB → /DISK1 (완료)
python3 scripts/benchmark/sweep_train_cost.py --prefetch

# 1) dummy 데이터셋 생성 (컨테이너 안, GPU 불필요)
docker run --rm -u $(id -u):$(id -g) \
  -v "$PWD/scripts:/workspace/scripts:ro" -v "$PWD/src:/workspace/src:ro" \
  -v /DISK1/so101-sim2real/datasets:/workspace/datasets -e HOME=/tmp \
  --entrypoint python policy-server:0.6.0 \
  /workspace/scripts/benchmark/make_dummy_dataset.py \
  --output-dir /workspace/datasets/benchmark_dummy_2cam \
  --episodes 20 --frames-per-episode 200 --horizon 100

# 2) GPU 없이 계약 검증 (config·feature·state_dict·파라미터 수)
python3 scripts/benchmark/sweep_train_cost.py --validate-cpu

# 3) 학습 스윕 — GPU 가 비었을 때. 재실행하면 끝난 조합은 건너뛴다.
python3 scripts/benchmark/sweep_train_cost.py --steps 30 --require-free-mb 40000
python3 scripts/benchmark/sweep_train_cost.py --report

# 4) 추론 비용 (full-chunk latency · VRAM) — batch 1 기본
python3 scripts/benchmark/sweep_train_cost.py --infer --batch-sizes 1
python3 scripts/benchmark/sweep_train_cost.py --infer --report

# OOM 이 GPU 경합 탓이었다면 비운 뒤 실패분만 재시도
python3 scripts/benchmark/sweep_train_cost.py --retry-failed
```

학습 조합 수 = 7 policy × 2 mode × 6 batch = **84 run**. 조합당 30 step + 모델 로드라
전체 3~6시간(대형 모델 로드가 지배적)으로 예상한다.

**선행 조건 1건 남음**: Pi05 는 `google/paligemma-3b-pt-224`(gated) tokenizer 를 하드코딩한다.
해당 모델 페이지에서 **라이선스를 1회 수락**해야 train/finetune 둘 다 돌아간다 (§8-②).

---

## 2. 대상 정책과 mode 정의

파라미터 수는 `--validate-cpu` 가 lerobot 로그에서 뽑은 **실측**이다(train mode 기준).

| policy | `--policy.type` | total / trainable | 입력 해상도 | 샘플당 프레임 | base 체크포인트 |
|---|---|---|---|---|---|
| ACT | `act` | **51.6M / 51.6M** | **리사이즈 없음(480×640)** | obs 1 / chunk 100 | 없음 → 1-step 자가 체크포인트 |
| SmolVLA | `smolvla` | **450.0M / 99.9M** | 512×512 | obs 1 / chunk 50 | `lerobot/smolvla_base` |
| Pi05 | `pi05` | ~3.62B / 전체 (미측정, §8-② 차단) | 224×224 | obs 1 / chunk 50 | `lerobot/pi05_base` |
| GR00T-N1.7 | `groot` | **3.144B / 3.144B**(full-tune) | 256×256 | obs 1 / chunk 40 | `nvidia/GR00T-N1.7-3B` |
| X-VLA | `xvla` | **880M / 880M** | 224×224 | obs 1 / chunk 32 | 호환 체크포인트 없음 → 1-step 자가 체크포인트 (§8-③) |
| VLA-JEPA | `vla_jepa` | **2.770B / 2.770B** | Qwen 프로세서 / V-JEPA2 256 | **obs 8프레임** / chunk 7 | `lerobot/VLA-JEPA-Pretrain` |
| LingBot-VA | `lingbot_va` | **5.089B / 5.089B** (+프리즈 14GB) | 128×128 / 256×320 | **obs 8~16프레임** / chunk 32 | `lerobot/lingbot_va_base` |

`train`  = `--policy.type=X` (무작위 초기화에서 출발)
`finetune` = `--policy.path=<base>` (config 까지 체크포인트 것을 쓴다)

**두 mode 의 비용 차이는 학습 알고리즘이 아니라 (a) trainable 파라미터 집합, (b) 파라미터 dtype,
(c) 입력 해상도** 에서 온다. 정책별 예외:

* **GR00T** — `GrootConfig.base_model_path` 기본값이 `nvidia/GR00T-N1.7-3B` 라 **무작위 초기화
  경로가 존재하지 않는다**. 그래서 `train` 을 **full-tune**(`tune_llm=true tune_visual=true`),
  `finetune` 을 기본 freeze(projector+diffusion+vlln)로 정의했다.
* **VLA-JEPA** — `train` 이라도 `__init__` 이 Qwen3-VL-2B·V-JEPA2 pretrained 를 강제로 내려받는다
  (action model 만 무작위).
* **ACT** — SO-101 6D 공개 base 가 없다. `finetune` 은 1-step 자가 체크포인트에서 출발하므로
  **train 과 같은 비용이 나와야 정상**이다(측정 sanity check 용 대조군).
* **SmolVLA** — `train_expert_only=True` 가 두 mode 모두 기본값이라 학습 대상은 action expert 다.
  전체 학습 비용을 보려면 `--policy.train_expert_only=false` 를 따로 재야 한다.

인자 단일 소스 = `scripts/benchmark/policy_matrix.py` (`python3 scripts/benchmark/policy_matrix.py`
로 표·caveat 출력).

### CPU 계약 검증 상태 (`--validate-cpu`, GPU 0)

"학습 루프 진입"(dataset·policy·processor·optimizer 생성 완료)까지 도달하는지 본 결과다.
GPU 스텝을 돌리기 전에 config/state_dict/feature 계약 문제를 전부 걸러낸다.

| policy | train | finetune |
|---|---|---|
| ACT | ✅ | (자가 seed 체크포인트 → GPU 스윕에서 생성·검증) |
| SmolVLA | ✅ | ✅ |
| Pi05 | ❌ gated `google/paligemma-3b-pt-224` 미승인 | ❌ 동일 |
| GR00T-N1.7 | ✅ | ✅ |
| X-VLA | ✅ (florence_config 주입 후) | (자가 seed 체크포인트 → GPU 스윕에서 생성·검증) |
| VLA-JEPA | ✅ | ✅ (reinit 3모듈 → 4 tensor 재초기화) |
| LingBot-VA | ✅ | ✅ |

→ **Pi05 를 제외한 나머지가 GPU 없이 확인됐다**(자가 seed 2건은 GPU 스윕에서 생성된다). 남은 리스크는 GPU 에서만 드러나는
것(OOM·실제 속도)과 Pi05 접근 승인이다.

---

## 3. dummy 데이터셋

`scripts/benchmark/make_dummy_dataset.py` — 프로젝트 실데이터와 같은 스키마, 카메라만 2대.

| 항목 | 값 | 근거 |
|---|---|---|
| 포맷 | LeRobot **v3.0** (`LeRobotDataset.create`) | `codebase_version: v3.0` 확인 |
| 카메라 | `observation.images.top` · `.wrist` | 요구사항 |
| 해상도·fps | 480×640×3 · 30fps | `so101_contract.feature_codec` 단일 소스 |
| 코덱 | **h264 crf23** (lerobot 기본 `libsvtav1` 대신) | 프로젝트 recorder 와 동일. AV1 decode 는 더 느려 dataloader 비용을 왜곡 |
| state/action | float32 (6,) = arm 5 radian + gripper [0,100] | `JOINT_FEATURE_NAMES` |
| 규모(기본) | 20 ep × 200 frame = 4,000 frame, 219MB | ACT chunk 100 을 덮는 최소 길이 |
| meta | schema v2 joint 블록 + `action_representation_stats.json` | act/smolvla/groot 의 `joint_absolute` 경로가 dataset 만으로 resolve |

**한계**: 219MB 는 OS page cache 에 전부 올라가 `data_s`(디코딩+I/O)가 실제 수십 GB 데이터셋보다
낙관적으로 나온다. 디코딩 CPU 비용은 그대로라 영향은 제한적이지만, 절대값을 쓸 때는 감안한다.
`--episodes` 로 키울 수 있다.

---

## 4. 무엇을 어떻게 재는가

lerobot 이 **이미 스텝마다 기록하는 값**을 쓴다 (자체 학습 루프를 만들지 않는다):

| 지표 | 출처 | 의미 |
|---|---|---|
| `updt_s` | `lerobot_train.py: update_policy()` | forward+backward+optimizer step 시간 |
| `data_s` | 학습 루프 | `next(dl_iter)` + preprocessor (video decode 포함) |
| `mem_gb` | `torch.cuda.max_memory_allocated()` (스텝마다 `reset_peak_memory_stats`) | **텐서 allocated** 피크 |
| `nvsmi_peak_mb` | 호스트에서 `nvidia-smi --query-compute-apps` 폴링 | **프로세스 실점유**(allocator reserved + CUDA context). 동거 프로세스와 분리됨 |
| `num_*_params` | 학습 시작 로그 | total / trainable 파라미터 수 |

* `--log_freq` 창 평균이라 **마지막 창**을 정상 상태로 채택한다(첫 창은 워밍업·CUDA 초기화 포함).
* 200k 외삽: `hours = 200000 × (updt_s + data_s) / 3600`.
  dataloader 를 완전히 숨겼을 때의 하한도 함께 낸다(`..._update_only`).
* **1스텝 = 1 optimizer step** 이다. batch 를 키워도 200k step 은 그대로라 총 학습 샘플 수가
  batch 에 비례해 늘어난다 — 같은 데이터량 기준 비교가 아니다(요구사항이 200k step 고정).

### 추론 (`--infer`)

`scripts/benchmark/measure_inference_cost.py` 가 컨테이너 안에서 정책을 올리고 실제 dataset 배치로
`select_action` 을 반복한다. **매 호출 전에 `policy.reset()`** 을 불러 항상 full-chunk 추론이 되게
한다 — `select_action` 은 큐가 비었을 때만 실제 계산을 하고 이후 호출은 큐 pop 이라, reset 없이는
평균이 0 에 가깝게 왜곡된다.

| 지표 | 의미 |
|---|---|
| `chunk_p50_s` / `chunk_p95_s` | full-chunk 추론 1회 latency |
| `per_step_ms` | `chunk_p50_s / n_action_steps` — chunk 를 다 소비할 때 스텝당 상환 비용 |
| `realtime_headroom_30hz` | `(n_action_steps/30) / chunk_p50_s`. **1 미만이면 30Hz 제어 미달** |
| `mem_gb_allocated` | 추론 시 tensor allocated 피크 (학습의 1/3~1/4 수준) |

참고 실측(CPU, ACT, batch 1): chunk 3.14s / 100 step = 31.4ms per step → headroom ×1.06.
CPU 로도 ACT 는 겨우 실시간이라는 뜻이고, GPU 수치는 스윕에서 채운다.

정적 VRAM 은 파라미터 수에서 결정론적으로 계산된다:

```
weights = total_params      × B
grads   = trainable_params  × B
AdamW   = trainable_params  × 2 × B      (exp_avg, exp_avg_sq)
static  = weights + grads + AdamW        (활성값 제외, batch 무관)
```

`B` = 파라미터 1개 바이트. **accelerate 의 `mixed_precision=bf16` 은 파라미터 dtype 을 바꾸지
않는다** — autocast 는 연산만 bf16 이고 가중치·grad·optimizer state 는 파라미터 dtype 그대로다.
그래서 `--policy.dtype=bfloat16` 을 지원하는 정책만 `B=2` 로 내려간다(`policy_matrix.param_bytes`).

---

## 5. GPU 경합 시 주의

* **VRAM 측정은 오염되지 않는다** — `mem_gb`(프로세스 내부)와 `nvsmi_peak_mb`(신규 PID 필터)
  둘 다 우리 프로세스만 본다.
* **시간 측정은 오염된다** — SM 을 나눠 쓰면 `updt_s` 가 부풀고, dataloader worker 가 CPU 를
  나눠 쓰면 `data_s` 도 부푼다. `--require-free-mb` / `--require-idle-util` 로 게이트한다.
* 다른 작업이 15GB 를 물고 있으면 큰 모델 조합은 **실제 용량 부족이 아니라 경합 때문에 OOM** 이
  된다. 스윕은 OOM 을 상태로 기록하므로, GPU 를 비운 뒤 `--rerun` 없이 다시 돌리면
  실패 조합만 재시도하도록 결과 파일을 손보면 된다(`results.jsonl` 에서 해당 줄 삭제).

---

## 6. 예측 (activation 제외 정적 VRAM은 실측 파라미터 기반)

정적 VRAM 만으로도 47.8GB 카드에서 **가능/불가**가 갈린다. 파라미터 수는 `--validate-cpu` 실측,
`B`(파라미터당 바이트)는 정책이 스스로 캐스팅하는 dtype 기준(`policy_matrix.param_bytes`).
`--report` 가 같은 표를 `params.json` 에서 다시 계산해 출력한다.

| policy / mode | B | total | trainable | weights | grads | AdamW | static VRAM | 47.8GB 판정 |
|---|---|---|---|---|---|---|---|---|
| ACT (양쪽) | 4 | 51.6M | 51.6M | 0.19 | 0.19 | 0.38 | **0.8 GB** | 여유. batch 32 도 문제 없음 |
| SmolVLA (양쪽) | 4 | 450.0M | 99.9M | 1.68 | 0.37 | 0.74 | **2.8 GB** | 여유 |
| X-VLA (양쪽) | 2 | 880M | 880M | 1.64 | 1.64 | 3.28 | **6.6 GB** | 여유 |
| GR00T finetune | 4 | 3.144B | **1.621B** (projector+diffusion+vlln, 52%) | 11.7 | 6.0 | 12.1 | **29.8 GB** | 가능하나 여유 작음. 활성값이 batch 를 제한 |
| GR00T train(full) | 4 | 3.144B | 3.144B | 11.7 | 11.7 | 23.4 | **46.9 GB** | **불가** — 활성값 전에 이미 카드가 찬다. LoRA(`--policy.lora_rank`) 또는 `model_params_fp32=false` 필요 |
| Pi05 (bf16, 양쪽) | 2 | ~3.62B | 전체 | 6.7 | 6.7 | 13.5 | **~27 GB** | 가능하나 batch 여유 작음. `gradient_checkpointing` 예비 |
| Pi05 (기본 fp32) | 4 | ~3.62B | 전체 | 13.5 | 13.5 | 27.0 | **~54 GB** | **불가** — 그래서 스윕이 bf16 을 강제한다 |
| VLA-JEPA (양쪽) | 2.2 | 2.770B | 2.770B | 5.7 | 5.7 | 11.4 | **22.8 GB** | 가능하나 batch 작을 것. `repeated_diffusion_steps=8` 이 활성값을 8배로 만든다 |
| LingBot-VA (양쪽) | 2 | 5.089B | 5.089B | 9.5 | 9.5 | 19.0 | **38.0 GB** | **거의 불가** — 프리즈 VAE 까지 GPU 에 올라가고 video latent 활성값이 남는다 |

핵심: **GR00T full-tune 과 Pi05 fp32 는 batch 1 도 못 올린다**(활성값 0 을 가정해도 카드 용량 초과).
GR00T 는 `model_params_fp32=True` 가 기본이라 3.1B 를 fp32 로 들고 AdamW state 까지 fp32 로 쌓는 게
원인이고, Pi05 는 dtype 기본값이 float32 다.

시간은 모델 크기·토큰 수·해상도에 좌우돼 예측 신뢰도가 낮다. 자릿수 감만 적는다
(batch 8, 200k step, 단일 카드 독점 가정):

| policy | 자릿수 예상 |
|---|---|
| ACT | 수 시간 (~3–6 h) |
| SmolVLA | 반나절~하루 (~8–20 h) |
| X-VLA | 하루~이틀 (518² 가 비싸다) |
| GR00T finetune / Pi05 | **수일** (3B 급, ~4–8 d) |
| VLA-JEPA | **수일~1주+** (8프레임 clip × diffusion 8회) |
| LingBot-VA | **1주+** (5B + VAE 인코딩 + CPU UMT5) |

→ 200k step 은 3B 급 이상 모델에서 단일 카드로는 현실적이지 않다는 게 요점이다. 실측 후 이 표를
지우고 §7 로 대체한다.

---

## 7. 측정 결과

*(미실행 — `sweep_train_cost.py --report` 출력을 여기에 붙인다)*

---

## 8. 알려진 함정 (전부 실제로 밟은 것)

1. **`policy-entrypoint.sh train` 을 쓰지 않는다.** 그 스크립트는 `${VAR:-default}` 로 env 를
   조립해 **빈 문자열도 미설정으로 취급**한다. 그래서 `-e TRAIN_ACTION_REPRESENTATION_MODE=` 로도
   `--policy.action_representation.mode` 를 끌 수 없고, 그 필드가 없는 정책(pi05·xvla·vla_jepa·
   lingbot_va — so101 patch 는 act·smolvla·groot 에만 넣었다)에서
   `The fields action_representation are not valid for PI05Config` 로 죽는다.
   → 스윕은 `--entrypoint lerobot-train` 으로 직접 부르고 `ACCELERATE_MIXED_PRECISION` 만 넣는다.
   (`action_representation.mode` 는 config 기본값이 이미 `joint_absolute` 라 명시 불필요.)
2. **Pi05 는 gated tokenizer 를 하드코딩한다** — `processor_pi05.py` 의
   `google/paligemma-3b-pt-224`(`gated: manual`). 유효한 `HF_TOKEN` 이 있어도
   `403 ... you are not in the authorized list` 다 — 계정이 모델 페이지에서 **접근을 신청·승인**
   받아야 한다(수동 심사). **현재 미승인 → Pi05 train/finetune 둘 다 차단**
   (유일한 남은 blocker. 가중치 `lerobot/pi05_base` 는 이미 받아 뒀다).
3. **X-VLA 는 lerobot 0.6.0 으로 로드되는 공개 체크포인트가 없다.** 네 후보 모두 어긋난다
   (`XVLAPolicy.from_pretrained` 은 `strict=True`):
   | 후보 | VLM 키 레이아웃 | action head | 막히는 지점 |
   |---|---|---|---|
   | `2toINF/X-VLA-Pt` (진짜 pretrained) | 구형 ✅ | 20D ✅ | upstream 포맷 — lerobot `type` 없고 키에 `model.` prefix 없음 |
   | `2toINF/X-VLA-Libero` | 구형 ✅ | 20D ✅ | 동일 |
   | `jadechoghari/X-VLA-Libero` | 구형 ✅ | **7D ❌** | 0.6.0 은 `franka_joint7`=20D (`action_hub.FrankaJoint7ActionSpace.dim_action=20`) |
   | `imstevenpmwork/xvla_super_chatton_2` | **신형 ❌**(`multi_modal_projector.*`) | 20D ✅ | 0.6.0 은 `image_projection`·`vision_tower.convs.*` 를 기대 |

   추가로 `XVLAConfig.florence_config` 기본값 `{}` 는 `ValueError: vision_config is required` 라
   scratch 학습에도 필수다 → `scripts/benchmark/xvla_florence_config.json`
   (2toINF/X-VLA-Pt 와 imstevenpmwork 것이 **diff 0** 로 동일).
   `jadechoghari` config 에는 lerobot 에 없는 `num_actions` 도 있다(draccus 거부).
   → **비용 측정 목적이므로 X-VLA fine-tune 은 ACT 처럼 1-step 자가 체크포인트에서 출발**한다.
   아키텍처·trainable 집합이 같아 비용은 실제 fine-tune 과 같다. 진짜 pretrained 가중치가 필요하면
   `2toINF/X-VLA-Pt` 를 키 prefix `model.` 추가 + lerobot config 작성으로 변환해야 한다.
4. **stats profile 은 (kind, action horizon, dataset fingerprint) 로 조회된다.** ACT 100 ·
   SmolVLA 50 · GR00T 40/16 처럼 chunk 가 달라, 한 horizon 만 만들면 다른 정책이
   `KeyError: no action stats profile matches the transform/sampler/dataset` 로 죽는다.
   → dummy 생성기가 `--horizons 100,50,40,32,16` 을 전부 만든다(`--stats-only` 로 재생성 가능).
5. **joint topology metadata 는 dataset feature 이름으로 키가 잡힌다** (`shoulder_pan.pos`),
   topology 쪽 이름(`shoulder_pan`)과 다르다 → `KeyError: joint topology metadata is missing`.
6. **lerobot 0.6.0 video 기본 코덱은 `libsvtav1`** 이고 `vcodec="libx264"` 는 거부된다
   (`h264` 로 써야 한다). 프로젝트 recorder 는 h264 라 벤치마크도 h264 로 맞춘다.
7. **`docker run` 클라이언트를 죽여도 컨테이너는 계속 돈다.** timeout·조기종료 때 좀비가 남아
   다음 조합의 시간·VRAM 을 오염시킨다 → `--name bench_<job>` 을 주고 `docker rm -f` 로 죽인다.
   `pkill -f <script>` 는 쓰지 않는다(자기 셸까지 죽는 self-kill).
8. **`hf download repo --include "a" "b"`** 는 `b` 를 위치 인자(파일명)로 먹고 `--include` 를
   무시한다. 패턴마다 `--include` 를 붙여야 한다 (LingBot 프리즈 모듈만 받을 때 필요).
9. **`--policy.path` 와 `--policy.type` 은 동시 지정 금지** (lerobot parser 제약).
   GR00T 만 예외적으로 `--policy.type=groot` + `--policy.base_model_path=...` 조합이다.
10. **VLA-JEPA base 는 action 7D·state 8D** 다. 6D 데이터셋에 붙이려면 reinit 대상이 **셋**이다 —
    `action_encoder`·`state_encoder` 만 넣으면 `action_decoder.layer2.bias (7,) vs (6,)` 에서 걸린다.
    `--policy.reinit_modules=["...action_encoder","...state_encoder","...action_decoder"]` (4 tensor
    재초기화 로그로 확인).
11. **LingBot-VA 학습은 `attn_mode=flex` 필수**이고, forward 가 매 스텝 프리즈 Wan VAE +
    UMT5-XXL 을 호출한다(`text_encoder_device=cpu` 기본 → CPU 시간이 스텝에 포함된다).
    `obs_cam_keys`·`used_action_channel_ids` 를 우리 키/6D 로 바꿔야 shape 이 맞는다.
12. **`--save_checkpoint=false` 를 반드시 넘긴다** — lerobot 은 마지막 스텝에서도 저장하므로
    3B 모델 84회 저장은 디스크·시간 낭비다.
13. **HF 캐시 snapshot 의 파일은 `blobs/` 를 가리키는 상대 symlink** 다. 다른 디렉터리로
    옮겨 쓸 때 그 경로를 그대로 `os.link` 하면 symlink 가 복제돼 링크가 깨진다
    (`FileNotFoundError: .../model.safetensors`) → `Path.resolve()` 로 blob 실경로를 쓴다.
14. **컨테이너를 호스트 uid 로 돌리면 `/root` 에 못 쓴다** — ACT 의 torchvision resnet18
    다운로드가 그리로 떨어진다 → `HOME=/tmp` + `TORCH_HOME=/workspace/hf/torch`.
15. **HF 캐시는 `/DISK1`** 로 보낸다. 루트 파티션 여유가 100GB 뿐인데 가중치가 **75GB** 다
    (`hf download` 가 onnx·tf·flax 까지 다 받아 예상보다 커진다).

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/benchmark/make_dummy_dataset.py` | 2-cam dummy LeRobot v3 데이터셋 생성 |
| `scripts/benchmark/policy_matrix.py` | 정책 7종 × mode 2종 인자·caveat 단일 소스 |
| `scripts/benchmark/sweep_train_cost.py` | prefetch · CPU 검증 · GPU 스윕 · 리포트 |
| `scripts/benchmark/measure_inference_cost.py` | 컨테이너 내부 추론 latency·VRAM 측정 (`--infer` 가 호출) |
| `scripts/benchmark/xvla_florence_config.json` | X-VLA 필수 `florence_config` (체크포인트에서 복사) |
| `/DISK1/so101-sim2real/benchmark/` | hf 캐시(75GB) · outputs · logs · `results.jsonl` · `inference_results.jsonl` · `params.json` |
| `/DISK1/so101-sim2real/datasets/benchmark_dummy_2cam/` | dummy 데이터셋 (126MB, 생성 완료) |
