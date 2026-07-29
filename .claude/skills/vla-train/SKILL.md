---
name: vla-train
description: policy-server 컨테이너로 VLA 정책(ACT·SmolVLA·GR00T-N1.7)을 학습하는 절차. 학습 데이터셋·모델·action representation(좌표계)·run 이름 4축을 한 명령에 고정하고, 학습 후 checkpoint 로 그 조합을 역추적한다. "VLA 학습 돌려줘", "ACT/SmolVLA/GR00T fine-tune", "policy-server train", "eef_relative 로 학습", "이 체크포인트 무슨 데이터·좌표계로 학습했나", "action representation stats 만들어줘" 같은 요청에 반드시 이 스킬을 쓴다.
---

# VLA 학습 runbook — 4축 고정과 사후 식별

한 run = **{데이터셋 · 모델 · representation(좌표계) · run 이름}**. 넷 다 `docker compose run` 한 줄에서
`-e` 로 고정하고 **`.env`·`env/<profile>.env` 는 건드리지 않는다** — 파일을 고쳐 학습하면 다음 run 이
같은 파일을 또 고치는 순간 이전 checkpoint 가 뭘로 학습됐는지 파일에서는 못 밝힌다(checkpoint manifest
가 정본, §4).

계약 정본 = `docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`, 변수 전체 = `docs/spec/06_RUNTIME_SPEC.md` §4.1·§6.

## 0. 실행 전 점검

GPU 1장 공유 서버 + 학습은 수 시간이다.

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
docker ps --format '{{.Names}}\t{{.Status}}' | grep -iE "policy|isaac|curobo"
docker images | grep policy-server            # policy-server:0.6.0 없으면 build 먼저
```

## 1. 4축을 무엇으로 고정하나

| 축 | 고정 수단 | 비고 |
|---|---|---|
| **데이터** | `-e DATASET_ROOT=/workspace/datasets/<이름>` + `-e HF_DATASET_REPO_ID=<라벨>` | root 가 있으면 Hub 다운로드 없이 로컬 로드. repo_id 는 manifest 에 남는 **식별 라벨**(로컬이면 `local/<이름>`). Hub 학습이면 root 를 비우고 repo_id 만 준다 |
| **모델** | `POLICY_PROFILE=<act\|smolvla\|groot_n17>` — **셸 env** | compose 가 `env/<profile>.env` 를 겹쳐 읽는 시점이 파싱 때라 `-e` 로는 못 고른다. 셸 env 가 `.env` 값을 이긴다 |
| **좌표계** | `-e TRAIN_ACTION_REPRESENTATION_MODE=` + `-e TRAIN_ACTION_REPRESENTATION_POSE_FORMAT=` | 4 mode × 3 pose format. joint mode 는 pose format 을 빈 문자열로. `TRAIN_` 없는 `ACTION_REPRESENTATION_*` 는 **추론 assertion 용**이라 학습에 쓰지 말 것 |
| **run 이름** | `-e JOB_NAME=` | `outputs/train/$JOB_NAME` 로 파생. 안 주면 프로필 기본 JOB_NAME 이 이겨서 **다른 run 이 같은 디렉터리로 간다** |

JOB_NAME 규약: `so101_<profile>_<mode>[_rot6d|_wxyz|_rpy]_<dataset>_<YYMMDD>`
(같은 날 재실행이면 `_b`, `_c` 를 붙인다 — 기존 output dir 이 있으면 lerobot 이 거부한다.)

## 2. 데이터셋 준비 — mode 가 dataset space 를 결정한다

**dataset 은 항상 그 space 의 absolute 만 저장한다.** relative 는 학습 processor 가 만든다.

| mode | 필요한 dataset | 만드는 법 |
|---|---|---|
| `eef_absolute` · `eef_relative` | absolute EEF v3 (10D/8D/7D + gripper) | joint v3 → 아래 변환기 |
| `joint_absolute` · `joint_relative` | canonical joint v3 (**arm radian**(5) + gripper [0,100]) + `meta/info.json` 의 `so101_action_representation{space:joint, joints:[topology]}` | ⚠ **as-built 로 실데이터를 그렇게 만드는 변환기가 없다** — datagen v3 는 arm degree 이고 계약 블록도 없다. `scripts/contract/create_action_representation_fixture.py` 는 합성 fixture 전용. 실데이터 joint mode 는 변환기부터 만들어야 하고, 없이 돌리면 contract resolve 에서 fail-fast |

```bash
.venv/bin/python scripts/convert/joint_dataset_to_eef.py \
  --input-dir datasets/pick_cube_v3 --output-dir datasets/pick_cube_eef_rot6d \
  --source-domain sim --rotation-representation rot6d      # [--keep-joints] [--overwrite]
```

| `--rotation-representation` | pose format | action dim |
|---|---|---|
| `rot6d` | `xyz_rot6d_rows` | 10 |
| `wxyz` | `xyz_quaternion_wxyz` | 8 |
| `rpy` | `xyz_rpy` | 7 |

- `--source-domain` 은 자동 판별하지 않는다: sim datagen = `sim`, 실기기 녹화 = `real`.
- **pose format 은 변환 시점에 굳는다.** 같은 데이터로 다른 format 을 학습하려면 다시 변환한다.
  (같은 EEF 데이터셋으로 `eef_absolute` / `eef_relative` 는 둘 다 학습 가능 — stats profile 만 다르다.)

### 2.1 stats profile 사전 생성 — 없으면 학습이 죽는다

patched 이미지는 **4 mode 전부** horizon-aware relative stats profile 을 요구하고, 학습 중에
자동 생성하지 않는다(장시간 학습이 startup 에서 멈춘 것처럼 보이는 걸 피하려는 의도적 설계).

```bash
.venv/bin/python scripts/data/generate_action_representation_stats.py \
  --dataset-root datasets/pick_cube_eef_rot6d --horizon 100 \
  --mode eef_relative --pose-format xyz_rot6d_rows        # [--all] [--overwrite]
```

**horizon = 그 정책의 학습 chunk_size**. 틀리면 학습 시작 직후
`KeyError: no action stats profile matches the transform/sampler/dataset ... horizon=<H>`.

| profile | horizon | 근거 |
|---|---|---|
| `act` | **100** | ACTConfig 기본 chunk_size |
| `smolvla` | **50** | SmolVLAConfig 기본 chunk_size |
| `groot_n17` | **40** | 프로필의 `POLICY_CHUNK_SIZE` |

profile 은 dataset fingerprint(info/stats hash)에 묶인다 → **데이터셋을 다시 만들거나 변환하면
stats 도 다시 만든다.** 산출물은 `meta/action_representation_stats.json` 하나에 누적된다.

스키마 확인: `.venv/bin/python scripts/contract/validate_lerobot_schema.py datasets/<이름>`

## 3. 학습 실행

```bash
PROFILE=act                       # act | smolvla | groot_n17
DATASET=pick_cube_eef_rot6d       # datasets/<이름> (호스트) = /workspace/datasets/<이름>
MODE=eef_relative                 # joint_absolute | joint_relative | eef_absolute | eef_relative
POSE=xyz_rot6d_rows               # EEF mode 만. joint mode 는 ""
JOB=so101_${PROFILE}_${MODE}_${DATASET}_$(date +%y%m%d)

POLICY_PROFILE=$PROFILE docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e HF_DATASET_REPO_ID=local/$DATASET \
  -e DATASET_ROOT=/workspace/datasets/$DATASET \
  -e TRAIN_ACTION_REPRESENTATION_MODE=$MODE \
  -e TRAIN_ACTION_REPRESENTATION_POSE_FORMAT=$POSE \
  -e JOB_NAME=$JOB \
  -e TRAIN_STEPS=20000 -e BATCH_SIZE=32 \
  policy-server train --policy.push_to_hub=false 2>&1 | tee logs/train_$JOB.log
```

- 뒤에 붙인 CLI 인자가 **last-wins** 로 env·`TRAIN_EXTRA_ARGS` 를 덮는다(`--save_freq=`, `--seed=` 등도 여기로).
- 시작 직후 이 줄로 4축이 실제 적용됐는지 확인한다 — 안 뜨면 stock 경로로 샌 것이다:
  `[action-representation] mode=eef_relative pose_format=xyz_rot6d_rows state/action=10/10 stats=sha256:… policy=act`
- 수 시간 run 은 tmux 또는 `nohup … > logs/train_$JOB.log 2>&1 &`.

## 4. 나중에 식별 — checkpoint 가 대장이다

별도 run 대장 파일을 만들지 말 것. `outputs/train/$JOB/checkpoints/{last,NNNNNN}/pretrained_model/` 안에:

| 파일 | 답해 주는 것 |
|---|---|
| `train_config.json` | dataset repo_id·root, policy type, steps, batch, seed, job_name, output_dir |
| `action_representation.json` (schema v2) | mode·pose_format·action_horizon·state/action dim·**dataset fingerprint**·stats profile hash·kinematics(URDF/robot yaml) hash·policy/runtime |

```bash
CK=outputs/train/$JOB/checkpoints/last/pretrained_model

jq '{job_name, steps, batch_size, seed, policy:.policy.type,
     dataset:.dataset.repo_id, root:.dataset.root}' $CK/train_config.json

jq '{mode, pose_format, action_horizon, action_dim,
     dataset:.dataset, stats:{id:.stats.profile_id, horizon:.stats.horizon}}' $CK/action_representation.json

# 계약 loader 로 읽어 routing·client_kind 까지 (추론 preflight 와 같은 코드 경로)
.venv/bin/python scripts/inference/assert_checkpoint_representation.py --checkpoint $CK --skip-kinematics --json
```

EEF mode 를 실기기로 보내기 전 preflight 는 `--skip-kinematics` 대신
`--urdf-path assets/robots/urdf/so_arm101.urdf` 로 kinematics hash 까지 대조한다.

## 5. 함정

| 증상 / 함정 | 원인·조치 |
|---|---|
| `Permission denied: /usr/local/bin/entrypoint.sh` | 호스트 `docker/policy-entrypoint.sh` 가 0600 이면 COPY+`chmod +x` = 0711 → non-root user 가 **읽지** 못한다. Dockerfile 이 `chmod 0755` 인지 확인하고 재빌드 |
| 학습 끝에 Hub push 실패 | `.env` `TRAIN_EXTRA_ARGS` 에 `--policy.push_to_hub=true` 가 있고, smolvla·groot 프로필의 `POLICY_REPO_ID` 는 **컨테이너 경로**(추론용)라 repo id 가 못 된다. 명령 끝에 `--policy.push_to_hub=false`, 또는 `-e POLICY_REPO_ID=$HF_USER/<이름>` |
| `KeyError: no action stats profile matches…` | §2.1 horizon 불일치 또는 dataset 재생성 후 stats 미갱신 |
| `no group metadata found` / `joint mode requires explicit joint topology metadata` | dataset 에 계약 블록 없음 — §2 표 |
| `dataset declares space 'eef' but config mode is 'joint_absolute'` | mode ↔ dataset space 불일치 |
| `NotImplementedError: ACT temporal ensemble` / `SmolVLA RTC` | full-chunk representation 과 공존 불가. 해당 옵션을 끈다 |
| GR00T + `COMPILE_MODEL=true` | `--policy.compile_model` 필드가 없어 자동 skip(경고만) |
| SmolVLA feature mismatch | 프로필의 `RENAME_MAP`(top/wrist/front → camera1/2/3)이 필수. `-e` 로 비우지 말 것 |
| `--policy.path` 와 `--policy.type` 동시 지정 오류 | entrypoint 가 프로필로 분기한다(`TRAIN_POLICY_TYPE` 비면 `--policy.path`). 수동으로 둘 다 주지 말 것 |
| output dir 이 이미 있음 | JOB_NAME 에 날짜/시퀀스, 이어서 할 거면 `--resume=true` |
| 경로 혼동 | 컨테이너 기준: `datasets/`→`/workspace/datasets`, `outputs/`→`/workspace/outputs`, `logs/`→`/workspace/logs` |

## 6. 학습 후 → 추론 인계

`env/<profile>.env` 의 `POLICY_REPO_ID` 를
`/workspace/outputs/train/$JOB/checkpoints/last/pretrained_model` 로 바꾸거나 `-e` 로 준다.
추론 CLI/env 의 representation 값은 override 가 아니라 **assertion** 이라 manifest 와 다르면 기동을 거부한다.
폐루프 실행 = `docs/spec/08_PIPELINES.md` §9 · `scripts/inference/demo_vla.sh`.

⚠ EEF IK 로 만든 **실기기** joint command 는 `EEF_IK_REAL_VALIDATED=true` 일 때만 나간다.
`false` 는 FK/IK·metric 만 기록하는 motor-off dry-run 이다.
