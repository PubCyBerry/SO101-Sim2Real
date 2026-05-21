# Windows Native + uv LeRobot 실행 가이드 (Git Bash)

이 문서는 Windows 11 호스트에서 WSL2 / Docker USB 포워딩을 거치지 않고
SO-101 실기기 LeRobot 워크플로를 직접 실행하는 경로를 정리한다.
현재 활성 실기기 경로인 teleop, record, replay, dataset-viz, SmolVLA train,
async policy server/client 를 대상으로 한다.

Isaac Sim / LeIsaac 시뮬레이션은 의존성과 GPU 제약이 별도이므로 이 문서의
기본 설치 대상에서 제외한다. 아래 명령은 저장소 루트에서 Git Bash 로 실행한다.

## Docker 경로와 달라지는 점

| Docker 가이드 | Windows native + uv |
|---|---|
| `docker compose ... lerobot <mode>` | `uv run lerobot-*` CLI 를 직접 호출 |
| `docker/lerobot-entrypoint.sh` 가 env var 로 CLI 인자를 조립 | Bash 변수와 CLI 인자를 명시 |
| 직렬 포트 `/dev/ttyACM*` | 직렬 포트 `COM*` |
| 카메라 `/dev/video*` 와 meta 노드 | OpenCV 카메라 index, 예: `0`, `1`, `2` |
| 컨테이너 경로 `/workspace/datasets`, `/workspace/outputs` | 로컬 경로 `./datasets`, `./outputs` |
| HF 캐시 Docker named volume | Windows 사용자 캐시 또는 직접 지정한 `HF_HOME` |
| `host.docker.internal` 로 Rerun 송출 | 로컬 실행이면 보통 생략 |

`.env.example` 은 Docker entrypoint 와 compose 보간을 기준으로 작성되어 있다.
Git Bash 는 `.env` 를 자동으로 읽지 않고, `/dev/...`, `/workspace/...`,
multiline `*_EXTRA_ARGS` 값도 그대로 재사용하지 않는다. 토큰과 기본값
참고용으로는 유지하되 native 실행은 아래 Bash 변수 블록을 기준으로 잡는 편이
덜 헷갈린다.

## 1. 한 번만 준비

### 1.1 uv 와 GPU 확인

`uv` 가 없다면 Git Bash 에서 설치한다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

installer 가 안내한 PATH 설정을 반영하거나 Git Bash 를 새로 연 뒤 버전과 GPU
가시성을 확인한다.

```bash
uv --version
nvidia-smi
```

### 1.2 Python 3.11 환경 동기화

이 저장소는 현재 `pyproject.toml` 의 Python 3.11 범위와 CUDA 12.8 Torch
설정을 기준으로 운영한다. 의존성 ABI 핀을 보존해야 하므로 임의로
`uv lock --upgrade` 하지 않는다.

실기기, async client/server, SmolVLA 학습을 한 Windows venv 에서 모두 쓸
기본 설치:

```bash
uv python install 3.11
uv sync --python 3.11 --group teleop --group async --group smolvla --no-install-project
uv run lerobot-info
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

로봇 bring-up 만 먼저 할 때는 `smolvla` 그룹을 빼도 된다.

```bash
uv sync --python 3.11 --group teleop --group async --no-install-project
```

### 1.3 Hub / W&B 인증

Hub push, private dataset/model pull, W&B 로깅을 쓸 때만 인증한다.

```bash
uv run hf auth login
uv run wandb login
```

토큰을 현재 Git Bash 세션에만 넣고 싶으면 다음처럼 설정해도 된다.

```bash
export HF_TOKEN="hf_xxx"
export WANDB_API_KEY="xxx"
```

## 2. 장치 확인과 세션 변수

### 2.1 COM 포트와 카메라 찾기

Leader / Follower Arm 을 Windows 에 직접 연결한 뒤 포트를 찾는다.

```bash
uv run lerobot-find-port
```

카메라는 OpenCV index 를 찾는다.

```bash
uv run lerobot-find-cameras opencv
```

`lerobot-find-cameras` 가 보여준 index 는 재부팅이나 USB 재연결 뒤 바뀔 수
있다. Docker `.env` 의 `BELLY_CAM_META_PORT`, `WRIST_CAM_META_PORT`,
`TOP_CAM_META_PORT` 는 native 실행에서 쓰지 않는다.

### 2.2 복사해서 바꿔 쓰는 Bash 변수

아래 블록을 새 Git Bash 세션에서 먼저 실행하고 포트, 카메라 index,
Hub 사용자명만 장비에 맞게 바꾼다.

```bash
mkdir -p ./datasets ./logs ./outputs

LEADER_PORT="COM5"
FOLLOWER_PORT="COM6"
LEADER_ID="so101_teleop"
FOLLOWER_ID="so101_robot"

WRIST_CAMERA=0
BELLY_CAMERA=1
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=25
CAMERA_WARMUP_S=5
CAMERA_FOURCC="MJPG"

TASK="pick the pen"
DATASET_NAME="so101_pick_pen"
HF_USER="your_hf_user"
DATASET_REPO="${HF_USER}/${DATASET_NAME}"
DATASET_ROOT="./datasets/${DATASET_NAME}"
POLICY_PATH="lerobot/smolvla_base"
POLICY_REPO="${HF_USER}/smolvla_pick_pen"
OUTPUT_DIR="./outputs/train/smolvla_pick_pen"

CAMERAS="{wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, warmup_s: ${CAMERA_WARMUP_S}, fourcc: ${CAMERA_FOURCC}}, belly: {type: opencv, index_or_path: ${BELLY_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, warmup_s: ${CAMERA_WARMUP_S}, fourcc: ${CAMERA_FOURCC}}}"
```

탑뷰 카메라까지 쓰면 `lerobot-find-cameras opencv` 로 찾은 index 를 같은
`CAMERAS` dict 에 `top: {...}` 항목으로 추가한다.

HF 캐시를 사용자 프로필 대신 저장소 아래에 모으고 싶으면 동기화나 다운로드
전에 한 번 지정한다.

```bash
export HF_HOME="$(pwd -W)/.cache/huggingface"
```

## 3. Docker mode 대응표

| 기존 Docker mode | Windows native 명령 |
|---|---|
| `lerobot find-port` | `uv run lerobot-find-port` |
| `lerobot find-cameras` | `uv run lerobot-find-cameras opencv` |
| `lerobot setup-motors` | `uv run lerobot-setup-motors ...` |
| `lerobot calibrate` | `uv run lerobot-calibrate ...` |
| `lerobot teleop` | `uv run lerobot-teleoperate ...` |
| `lerobot record` | `uv run lerobot-record ...` |
| `lerobot replay` | `uv run lerobot-replay ...` |
| `lerobot find-joint-limits` | `uv run lerobot-find-joint-limits ...` |
| `lerobot dataset-viz` | `uv run lerobot-dataset-viz ...` |
| `lerobot edit-dataset` | `uv run lerobot-edit-dataset ...` |
| `lerobot info` | `uv run lerobot-info` |
| `lerobot-policy-server prepare-model` | `uv run hf download <repo_id>` |
| `lerobot-policy-server train` | `uv run lerobot-train ...` |
| `lerobot-policy-server eval` | `uv run lerobot-eval ...` |
| `lerobot-policy-server policy-server` | `uv run python -m lerobot.async_inference.policy_server ...` |
| `lerobot policy-client` | `uv run python ./docker/policy-client-shim.py ...` |

## 4. 모터 설정과 보정

모터 ID / baud 설정은 각 arm 에 대해 필요한 시점에 한 번씩 실행한다.

```bash
# Follower
uv run lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}"

# Leader
uv run lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}"
```

캘리브레이션도 leader 와 follower 를 각각 수행한다. `id` 는 이후 teleop,
record, replay 에서 같은 값을 유지한다.

```bash
# Follower
uv run lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}"

# Leader
uv run lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}"
```

## 5. Teleoperation 과 데이터셋

### 5.1 Teleoperation

```bash
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="${CAMERAS}" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --display_data=false
```

native 로컬 뷰어를 띄우려면 `--display_data=true` 로 바꾼다.
Docker 전용 `--display_ip=host.docker.internal` 값은 넣지 않는다.

### 5.2 Record

```bash
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="${CAMERAS}" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.single_task="${TASK}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.fps=30 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

기존 Docker 가이드의 record 조작 키는 그대로 쓴다.

| 키 | 기능 |
|---|---|
| `Right Arrow` | 현재 에피소드 조기 종료 |
| `Left Arrow` | 현재 에피소드 취소 후 다시 녹화 |
| `Esc` | 세션 종료 후 인코딩 / 업로드 정리 |

기존 데이터셋에 이어서 수집할 때는 record 명령 끝에 `--resume=true` 를
추가한다.

### 5.3 Replay

```bash
uv run lerobot-replay \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.episode=0 \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.fps=30 \
    --play_sounds=false
```

### 5.4 Dataset viz 와 편집

```bash
uv run lerobot-dataset-viz \
    --repo-id="${DATASET_REPO}" \
    --episode-index=0 \
    --root="${DATASET_ROOT}" \
    --mode=local
```

편집 CLI 는 native 에서도 인자를 그대로 넘긴다.

```bash
uv run lerobot-edit-dataset \
    --repo_id="${DATASET_REPO}" \
    --root="${DATASET_ROOT}" \
    --operation.type=delete_episodes \
    --operation.episode_indices=[0]
```

### 5.5 Joint limit 점검

```bash
uv run lerobot-find-joint-limits \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --teleop_time_s=30
```

## 6. SmolVLA 모델 준비와 학습

### 6.1 모델 미리 받기

Docker `prepare-model` 은 named volume 을 채우는 래퍼였다. native 에서는 Hub
CLI 로 바로 받는다.

```bash
uv run hf download lerobot/smolvla_base
```

### 6.2 Fine-tune

SO-101 카메라 키가 들어간 데이터셋으로 fine-tune 해야 체크포인트의
`input_features` 가 `wrist`, `belly`, `top` 키와 맞는다.
Windows A4000 에서는 먼저 작은 batch 로 시작한다.

```bash
export ACCELERATE_MIXED_PRECISION="bf16"

uv run lerobot-train \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=smolvla \
    --policy.path="${POLICY_PATH}" \
    --policy.repo_id="${POLICY_REPO}" \
    --policy.push_to_hub=true \
    --policy.device=cuda \
    --output_dir="${OUTPUT_DIR}" \
    --steps=20000 \
    --batch_size=8 \
    --job_name=smolvla_pick_pen \
    --num_workers=4 \
    --wandb.enable=true
```

W&B 를 쓰지 않으면 `--wandb.enable=false` 로 바꾸고, Hub 에 올리지 않으면
`--policy.push_to_hub=false` 로 바꾼다. Linux H100 서버에서 멀티 GPU 로
학습할 때는 기존 `lerobot-policy-server` Docker 경로나 별도 Linux native
환경에서 `accelerate launch` 구성을 잡는다.

### 6.3 Eval

`lerobot-eval` 은 평가 환경 인자를 직접 받는다. 예를 들어 PushT 평가
환경 의존성까지 준비된 venv 에서 smoke test 할 때는 다음처럼 호출한다.

```bash
uv run lerobot-eval \
    --policy.path="${POLICY_REPO}" \
    --env.type=pusht \
    --eval.n_episodes=20 \
    --eval.batch_size=10
```

## 7. Async policy server / client

### 7.1 로컬 policy server

같은 Windows 호스트에서 follower client 와 서버를 같이 쓸 때는 서버를
loopback 에 bind 한다.

```bash
uv run python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2
```

모델 종류와 체크포인트는 서버가 아니라 client 가 넘긴다.

### 7.2 SO-101 policy client

LeRobot 0.4.4 의 async `robot_client` 는 built-in SO follower config 등록
회귀가 있어 이 저장소의 shim 을 먼저 거친다. Docker entrypoint 의
`policy-client` 모드도 같은 shim 을 사용한다.

```bash
uv run python ./docker/policy-client-shim.py \
    --server_address=127.0.0.1:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path="${POLICY_REPO}" \
    --policy_device=cuda \
    --client_device=cpu \
    --task="${TASK}" \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --fps=30 \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="${CAMERAS}"
```

원격 정책 서버를 쓸 때는 `--server_address=<server-ip>:8080` 만 바꾼다.
async server 는 신뢰하지 않는 네트워크에 그대로 노출하지 말고 방화벽,
SSH 터널, 별도 보안 래퍼 중 하나로 범위를 제한한다.

## 8. 빠른 점검 순서

처음 세팅한 PC 에서는 아래 순서로 실패 지점을 줄인다.

1. `uv run lerobot-info` 와 Torch CUDA 확인
2. `uv run lerobot-find-port`
3. `uv run lerobot-find-cameras opencv`
4. follower / leader `setup-motors`
5. follower / leader `calibrate`
6. `lerobot-teleoperate`
7. 1 episode `lerobot-record`
8. `lerobot-dataset-viz`
9. 필요 시 `lerobot-train`, policy server, policy client

## 참고

- [LeRobot Installation](https://huggingface.co/docs/lerobot/main/installation)
- [LeRobot Cameras](https://huggingface.co/docs/lerobot/main/en/cameras)
- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv Python management](https://docs.astral.sh/uv/guides/install-python/)
