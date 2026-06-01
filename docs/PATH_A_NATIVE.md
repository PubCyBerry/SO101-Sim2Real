# 경로 A — Windows native + uv (실기기)

> [← README](../README.md) · 관련: [경로 B (Docker)](PATH_B_DOCKER.md) · [경로 C (Isaac Lab 시뮬)](PATH_C_ISAAC_SIM.md) · [트러블슈팅](TROUBLESHOOTING.md)

WSL2 / Docker / usbipd 를 거치지 않고 호스트 Windows 의 uv venv 에서 직접 `lerobot-*` CLI 를 호출한다. 직렬 포트는 `COMx`, 카메라는 OpenCV index. **빠른 반복·디버깅**에 유리한 경로.

> 사전 준비(인증, GPU/CUDA 설치 확인)는 [README §공통 준비](../README.md#공통-준비) 참고.

## 목차 <!-- omit in toc -->

- [1. 아키텍처](#1-아키텍처)
- [2. 한 번만 준비](#2-한-번만-준비)
- [3. 장치 확인과 세션 변수](#3-장치-확인과-세션-변수)
- [4. 모터 설정과 보정](#4-모터-설정과-보정)
- [5. Teleoperation 과 데이터셋](#5-teleoperation-과-데이터셋)
- [6. SmolVLA 모델 준비와 학습](#6-smolvla-모델-준비와-학습)
- [7. Async policy server / client](#7-async-policy-server--client)
- [8. 빠른 점검 순서](#8-빠른-점검-순서)
- [부록. Docker mode 대응표](#부록-docker-mode-대응표)

---

## 1. 아키텍처

```mermaid
flowchart LR
    classDef hw fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef host fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef cloud fill:#e8f5e9,stroke:#388e3c,color:#1b5e20

    LEAD["🦾 SO-101 리더 암<br/>COM5"]:::hw
    CAM["📷 카메라 1~3대<br/>OpenCV index"]:::hw
    FOLL["🦾 SO-101 팔로워 암<br/>COM6"]:::hw

    subgraph WIN["🖥️ Windows 11 + uv venv"]
        direction TB
        CLI["uv run lerobot-*<br/>(teleop / record / replay / train)"]
        ASYNC["uv run python ./docker/policy-client-shim.py<br/>+ python -m lerobot.async_inference.policy_server"]
    end

    DS["./datasets"]:::host
    OUT["./outputs"]:::host
    HF[("🤗 HuggingFace Hub")]:::cloud
    WB[("📊 W&B")]:::cloud

    LEAD -->|6 DoF, Feetech serial| CLI
    CAM -->|frames| CLI
    CLI -->|6 DoF| FOLL
    CLI -.->|record| DS
    CLI -.->|train| OUT
    DS <-->|push/pull| HF
    OUT -->|push| HF
    CLI -.-> WB
    ASYNC -.->|gRPC :8080| CLI
```

> Isaac Sim / LeIsaac 시뮬 의존성은 별도 (`isaac` 그룹). 경로 A 에는 포함하지 않는다 — [경로 C](PATH_C_ISAAC_SIM.md) 참고.

---

## 2. 한 번만 준비

### uv 와 GPU 확인

```bash
uv --version
nvidia-smi
```

### Python 3.11 환경 동기화

실기기 + async client/server + SmolVLA 학습을 한 Windows venv 에 모두 설치:

```bash
uv python install 3.11
uv sync --python 3.11 --group teleop --group async --group smolvla --no-install-project
uv run lerobot-info
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

학습을 안한다면 `smolvla` 그룹을 빼도 된다.

```bash
uv sync --python 3.11 --group teleop --group async --no-install-project
```

ABI 핀 보존을 위해 `uv lock --upgrade` 는 사용 금지.

---

## 3. 장치 확인과 세션 변수

### COM 포트와 카메라 찾기

```bash
uv run lerobot-find-port             # 인터랙티브: USB 분리 후 Enter
uv run lerobot-find-cameras opencv
```

`lerobot-find-cameras` 가 보여준 OpenCV index 는 재부팅·USB 재연결 뒤 바뀔 수 있다.

### 복사해서 바꿔 쓰는 Bash 변수

새 Git Bash 세션마다 먼저 실행하고 포트·카메라 index·HF 사용자명만 장비에 맞게 바꾼다.

```bash
mkdir -p ./datasets ./logs ./outputs

# ── Arm 직렬 포트 (Windows COM 포트) ──
TELEOP_PORT="COM5"
ROBOT_PORT="COM6"
TELEOP_ID="so101_teleop"
ROBOT_ID="so101_robot"
ROBOT_TYPE="so101_follower"
TELEOP_TYPE="so101_leader"

# ── 카메라 (OpenCV index) ──
WRIST_CAM_PORT=0
FRONT_CAM_PORT=1
TOP_CAM_PORT=2
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=25
CAM_WARMUP_S=5
CAM_FOURCC="MJPG"

CAMERAS="{
    wrist: {type: opencv, index_or_path: ${WRIST_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
    front: {type: opencv, index_or_path: ${FRONT_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
    top: {type: opencv, index_or_path: ${TOP_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
}"

# ── 태스크 / HuggingFace ──
SINGLE_TASK="pick the pen"
TASK="pick the pen"
HF_USER="your_hf_user"
HF_DATASET_REPO_ID="${HF_USER}/so101_pick_pen"
DATASET_ROOT="./datasets/so101_pick_pen"

# ── 정책 ──
POLICY_PATH="lerobot/smolvla_base"
POLICY_REPO_ID="${HF_USER}/smolvla_pick_pen"
TRAIN_POLICY_TYPE=smolvla
POLICY_CLIENT_TYPE=smolvla

# ── record ──
RECORD_FPS=30
EPISODE_TIME_S=60
RESET_TIME_S=10
NUM_EPISODES=10
PUSH_TO_HUB=true
EPISODE_INDEX=0

# ── train ──
TRAIN_STEPS=20000
BATCH_SIZE=8
JOB_NAME=smolvla_pick_pen
OUTPUT_DIR="./outputs/train/${JOB_NAME}"
NUM_WORKERS=4
WANDB_ENABLE=true
DEVICE=cuda

# ── policy server / client ──
POLICY_SERVER_HOST=127.0.0.1
POLICY_SERVER_PORT=8080
POLICY_SERVER_ADDRESS="${POLICY_SERVER_HOST}:${POLICY_SERVER_PORT}"
POLICY_FPS=30
INFERENCE_LATENCY=0.033
OBS_QUEUE_TIMEOUT=2
POLICY_DEVICE=cuda
CLIENT_DEVICE=cpu
ACTIONS_PER_CHUNK=50
CHUNK_SIZE_THRESHOLD=0.5
AGGREGATE_FN_NAME=weighted_average
POLICY_CLIENT_FPS=30

```

탑뷰 카메라가 없을 때는 `CAMERAS` 에서 `top:` 줄을 지우고 `TOP_CAM_PORT` 는 무시한다.

HF 캐시를 사용자 프로필 대신 저장소 아래에 모으려면:

```bash
export HF_HOME="$(pwd -W)/.cache/huggingface"
```

---

## 4. 모터 설정과 보정

각 arm 에 대해 필요한 시점에 한 번씩 실행.

```bash
# Follower
uv run lerobot-setup-motors \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}"

# Leader
uv run lerobot-setup-motors \
    --teleop.type="${TELEOP_TYPE}" \
    --teleop.port="${TELEOP_PORT}"
```

캘리브레이션. `id` 는 이후 teleop / record / replay 에서 동일하게 유지.

```bash
# Follower
uv run lerobot-calibrate \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}"

# Leader
uv run lerobot-calibrate \
    --teleop.type="${TELEOP_TYPE}" \
    --teleop.port="${TELEOP_PORT}" \
    --teleop.id="${TELEOP_ID}"
```

---

## 5. Teleoperation 과 데이터셋

### Teleoperation

```bash
uv run lerobot-teleoperate \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${CAMERAS}" \
    --teleop.type="${TELEOP_TYPE}" \
    --teleop.port="${TELEOP_PORT}" \
    --teleop.id="${TELEOP_ID}" \
    ${TELEOP_EXTRA_ARGS}
```

로컬 Rerun 뷰어를 띄우려면 `--display_data=true`. Docker 전용 `--display_ip=host.docker.internal` 은 native 에서 넣지 않는다.

### Record

```bash
uv run lerobot-record \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${CAMERAS}" \
    --teleop.type="${TELEOP_TYPE}" \
    --teleop.port="${TELEOP_PORT}" \
    --teleop.id="${TELEOP_ID}" \
    --dataset.repo_id="${HF_DATASET_REPO_ID}" \
    --dataset.single_task="${SINGLE_TASK}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.fps=${RECORD_FPS} \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.push_to_hub=${PUSH_TO_HUB} \
    --play_sounds=false \
    ${RECORD_EXTRA_ARGS}
```

녹화 조작:

| 키 | 기능 |
|---|---|
| → | 현재 에피소드 조기 종료 |
| ← | 현재 에피소드 취소 후 다시 녹화 |
| ESC | 세션 종료 + 인코딩·업로드 |

이어서 수집할 때는 `--resume=true` 추가.

### Replay

```bash
uv run lerobot-replay \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --dataset.repo_id="${HF_DATASET_REPO_ID}" \
    --dataset.episode=${EPISODE_INDEX} \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.fps=${RECORD_FPS} \
    --play_sounds=false
```

### Dataset viz · 편집

```bash
uv run lerobot-dataset-viz \
    --repo-id="${HF_DATASET_REPO_ID}" \
    --episode-index=${EPISODE_INDEX} \
    --root="${DATASET_ROOT}" \
    --mode=local
```

```bash
uv run lerobot-edit-dataset \
    --repo_id="${HF_DATASET_REPO_ID}" \
    --root="${DATASET_ROOT}" \
    --operation.type=delete_episodes \
    --operation.episode_indices=[${EPISODE_INDEX}]
```

### HuggingFace Hub 업로드

`lerobot-record --dataset.push_to_hub=false` 로 로컬에만 저장한 뒤 나중에 올리거나, 수동으로 재업로드할 때:

```bash
uv run hf upload "${HF_DATASET_REPO_ID}" "${DATASET_ROOT}" --repo-type=dataset
```

---

## 6. SmolVLA 모델 준비와 학습

### 모델 미리 받기

```bash
uv run hf download lerobot/smolvla_base
```

### Fine-tune

SO-101 카메라 키 (`wrist` / `front` / `top`) 가 들어간 데이터셋으로 fine-tune 해야 체크포인트의 `input_features` 가 일치한다. Windows A4000 은 작은 batch 부터:

```bash
export ACCELERATE_MIXED_PRECISION="bf16"

uv run lerobot-train \
    --dataset.repo_id="${HF_DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=${TRAIN_POLICY_TYPE} \
    --policy.path="${POLICY_PATH}" \
    --policy.repo_id="${POLICY_REPO_ID}" \
    --policy.push_to_hub=${PUSH_TO_HUB} \
    --policy.device=${DEVICE} \
    --output_dir="${OUTPUT_DIR}" \
    --steps=${TRAIN_STEPS} \
    --batch_size=${BATCH_SIZE} \
    --job_name=${JOB_NAME} \
    --num_workers=${NUM_WORKERS} \
    --wandb.enable=${WANDB_ENABLE}
```

W&B 미사용 시 `--wandb.enable=false`, Hub push 미사용 시 `--policy.push_to_hub=false`. Linux 학습 서버 (RTX PRO 5000 Blackwell 48 GB) 에서 멀티 GPU 가 필요하면 [경로 B](PATH_B_DOCKER.md) 의 docker 학습 또는 별도 Linux native 환경에서 `accelerate launch` 구성.

### Eval

```bash
uv run lerobot-eval \
    --policy.path="${POLICY_REPO_ID}" \
    --env.type=pusht \
    --eval.n_episodes=20 \
    --eval.batch_size=10
```

---

## 7. Async policy server / client

### 로컬 policy server

서버를 loopback 에 bind:

```bash
uv run python -m lerobot.async_inference.policy_server \
    --host=${POLICY_SERVER_HOST} \
    --port=${POLICY_SERVER_PORT} \
    --fps=${POLICY_FPS} \
    --inference_latency=${INFERENCE_LATENCY} \
    --obs_queue_timeout=${OBS_QUEUE_TIMEOUT}
```

모델 종류와 체크포인트는 서버가 아니라 client 가 넘긴다.

### SO-101 policy client (shim 경유)

LeRobot 0.4.4 의 async `robot_client` 는 built-in SO follower config 등록 회귀가 있어 본 저장소의 shim 을 먼저 거친다.

```bash
uv run python ./docker/policy-client-shim.py \
    --server_address=${POLICY_SERVER_ADDRESS} \
    --policy_type=${POLICY_CLIENT_TYPE} \
    --pretrained_name_or_path="${POLICY_REPO_ID}" \
    --policy_device=${POLICY_DEVICE} \
    --client_device=${CLIENT_DEVICE} \
    --task="${TASK}" \
    --actions_per_chunk=${ACTIONS_PER_CHUNK} \
    --chunk_size_threshold=${CHUNK_SIZE_THRESHOLD} \
    --aggregate_fn_name=${AGGREGATE_FN_NAME} \
    --fps=${POLICY_CLIENT_FPS} \
    --robot.type="${ROBOT_TYPE}" \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${CAMERAS}"
```

원격 정책 서버는 `--server_address=<server-ip>:8080` 으로 변경. async server 는 pickle deserialization RCE 위험 (CVE-2026-25874) 이 있으니 SSH 터널·방화벽·mTLS 래퍼 등으로 신뢰 범위를 제한할 것.

---

## 8. 빠른 점검 순서

처음 세팅한 PC 에서 실패 지점을 줄이는 권장 순서:

1. `uv run lerobot-info` + Torch CUDA 확인
2. `uv run lerobot-find-port`
3. `uv run lerobot-find-cameras opencv`
4. follower / leader `setup-motors`
5. follower / leader `calibrate`
6. `lerobot-teleoperate`
7. 1 episode `lerobot-record`
8. `lerobot-dataset-viz`
9. 필요 시 `lerobot-train` / policy server / policy client

---

## 부록. Docker mode 대응표

[경로 B](PATH_B_DOCKER.md) 의 Docker mode 와 본 경로의 native 명령 대응:

| Docker mode (경로 B) | Windows native (경로 A) |
|---|---|
| `lerobot find-port` | `uv run lerobot-find-port` |
| `lerobot find-cameras` | `uv run lerobot-find-cameras opencv` |
| `lerobot setup-motors` | `uv run lerobot-setup-motors ...` |
| `lerobot calibrate` | `uv run lerobot-calibrate ...` |
| `lerobot teleop` | `uv run lerobot-teleoperate ...` |
| `lerobot record` | `uv run lerobot-record ...` |
| `lerobot replay` | `uv run lerobot-replay ...` |
| `lerobot dataset-viz` | `uv run lerobot-dataset-viz ...` |
| `lerobot edit-dataset` | `uv run lerobot-edit-dataset ...` |
| `lerobot info` | `uv run lerobot-info` |
| `policy-server prepare-model` | `uv run hf download <repo_id>` |
| `policy-server train` | `uv run lerobot-train ...` |
| `policy-server eval` | `uv run lerobot-eval ...` |
| `policy-server policy-server` | `uv run python -m lerobot.async_inference.policy_server ...` |
| `lerobot policy-client` | `uv run python ./docker/policy-client-shim.py ...` |
