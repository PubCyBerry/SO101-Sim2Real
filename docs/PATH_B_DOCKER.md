# 경로 B — Docker 컨테이너 (실기기)

> [← README](../README.md) · 관련: [경로 A (Windows native)](PATH_A_NATIVE.md) · [경로 C (Isaac Lab 시뮬)](PATH_C_ISAAC_SIM.md) · [원격 텔레옵·수집](REMOTE_TELEOP_RECORD.md) · [트러블슈팅](TROUBLESHOOTING.md)

격리된 Docker 환경에서 실기기 워크플로를 실행한다. Linux 학습 서버 배포와 async inference policy server 분리에 적합한 경로. Windows host → usbipd-win → WSL2 → Docker Desktop 경유로 USB 장치를 전달한다.

> 사전 준비(인증, `.env` 작성, Docker/usbipd 설치 확인)는 [README §공통 준비](../README.md#공통-준비) 참고. native 명령과의 대응은 [경로 A 부록](PATH_A_NATIVE.md#부록-docker-mode-대응표).

## 목차 <!-- omit in toc -->

- [1. 아키텍처](#1-아키텍처)
- [2. Docker 이미지 빌드](#2-docker-이미지-빌드)
- [3. (WSL) USB 포트 연결](#3-wsl-usb-포트-연결)
- [4. Entrypoint 모드 일람](#4-entrypoint-모드-일람)
- [5. SO-101 Motor Setup / Calibration](#5-so-101-motor-setup--calibration)
- [6. Teleoperation](#6-teleoperation)
- [7. 데이터셋 녹화 / 재실행 / 시각화](#7-데이터셋-녹화--재실행--시각화)
- [8. 모델 가중치 준비](#8-모델-가중치-준비)
- [9. Policy 학습](#9-policy-학습)
- [10. Policy 평가 및 추론](#10-policy-평가-및-추론)
- [11. Async Inference Policy Server (SmolVLA)](#11-async-inference-policy-server-smolvla)
- [12. Fine-tune 워크플로 (pick_pen)](#12-fine-tune-워크플로-pick_pen)

---

## 1. 아키텍처

```mermaid
flowchart LR
    classDef hw fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef host fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef cloud fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef teleopNode fill:#bbdefb,stroke:#1565c0,color:#0d47a1
    classDef policyNode fill:#ffcdd2,stroke:#c62828,color:#b71c1c

    LEAD["🦾 SO-101 리더 암"]:::hw
    CAM["📷 카메라 1~3대<br/>640×480@25fps MJPG"]:::hw
    FOLL["🦾 SO-101 팔로워 암"]:::hw

    subgraph WSL["🪟 Windows host → usbipd-win → WSL2 → Docker Desktop"]
        direction TB
        subgraph LERO["📦 lerobot 컨테이너 (Dockerfile.lerobot)"]
            T["🔵 teleop / record / replay / dataset-viz"]:::teleopNode
            PC["🟣 policy-client (gRPC)"]:::teleopNode
        end
        subgraph SRV["📦 policy-server 컨테이너 (Dockerfile.smolvla)"]
            PS["🔴 policy-server (gRPC :8080)<br/>train / eval"]:::policyNode
        end
    end

    DS["./datasets<br/>(bind mount)"]:::host
    OUT["./outputs<br/>(bind mount)"]:::host
    CACHE["lerobot_hf_cache<br/>(named volume, 공유)"]:::host
    HF[("🤗 HuggingFace Hub")]:::cloud
    WB[("📊 W&B")]:::cloud

    LEAD -->|6 DoF| T
    CAM -->|frames| T
    CAM -->|frames| PC
    T -->|6 DoF| FOLL
    PC -->|6 DoF| FOLL
    PC <-->|gRPC| PS
    T -.->|record| DS
    PS -.->|train| OUT
    DS <-->|push/pull| HF
    OUT -->|push| HF
    PS -.-> WB
    PS <--> CACHE
    PC <--> CACHE
```

핵심:

- **서비스별 진입점 분리**: `lerobot-entrypoint.sh` 는 `lerobot` 서비스(로봇 직결 워크플로) 의 모드 디스패처, `policy-entrypoint.sh` 는 `policy-server` 서비스(추론 서버) 의 모드 디스패처. 각 스크립트의 첫 인자가 모드를 결정한다.
- **이미지 분리**: SmolVLA / GR00T 추론과 학습 관련 의존성은 정책 서버 이미지에 격리한다.
- **HF 캐시 공유**: 명명 볼륨 `lerobot_hf_cache` 가 두 컨테이너의 `/root/.cache/huggingface` 에 마운트되어 한 번 받은 모델을 양쪽이 모두 사용.

---

## 2. Docker 이미지 빌드

| 이미지 | Dockerfile | 의존성 그룹 | 사용 서비스 |
|---|---|---|---|
| `lerobot-so101:0.4.4` | `docker/Dockerfile.lerobot` | `teleop` (lerobot[feetech] + evdev) | `lerobot` (teleop / record / replay / train / ...) |
| `policy-server:0.4.4` | `docker/Dockerfile.smolvla` | `smolvla` + `async` (lerobot[smolvla] + grpcio) | `policy-server` (async inference) |

```bash
# teleop / record / replay 용 이미지
docker compose -f docker/docker-compose.yaml build lerobot

# Async inference policy server 용 이미지
docker compose -f docker/docker-compose.yaml build policy-server
```

---

## 3. (WSL) USB 포트 연결

SO-101 Leader Arm / Follower Arm / 카메라를 PC 에 연결한 뒤, 관리자 PowerShell 에서 usbipd 로 포트 바인딩.

```powershell
# 포트 목록 조회
usbipd list
# 최초 1회만 실행
usbipd bind --busid <leader-port>
usbipd bind --busid <follower-port>
usbipd bind --busid <wrist-cam-port>
usbipd bind --busid <front-cam-port>
# usb 재연결할 때마다 / WSL 리부트할 때마다 실행
usbipd attach --wsl --busid <leader-port>
usbipd attach --wsl --busid <follower-port>
usbipd attach --wsl --busid <wrist-cam-port>
usbipd attach --wsl --busid <front-cam-port>
# Windows로 포트를 되돌릴 경우
usbipd detach --busid <port>
```

WSL 안에서 디바이스 권한 설정:

```bash
# Leader Arm, Follower Arm USB
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# Wrist Cam, Front Cam
sudo chmod 666 /dev/video0 /dev/video2
sudo usermod -aG dialout $USER
```

---

## 4. Entrypoint 모드 일람

각 서비스는 별도 진입점을 사용한다.

### `lerobot` 서비스 — 로봇 직결 워크플로 (`lerobot-entrypoint.sh`)

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot <mode> [args...]
```

| 모드 | 설명 | 필요 하드웨어 | 핵심 env var |
|---|---|---|---|
| `teleop` | 리더→팔로워 실시간 원격 조작 | Leader + Follower + 카메라 | `TELEOP_PORT`, `ROBOT_PORT`, `*_CAM_PORT`, `CAM_*` |
| `record` | 텔레옵 기반 데이터셋 수집 | Leader + Follower + 카메라 | `HF_DATASET_REPO_ID`, `SINGLE_TASK`, `NUM_EPISODES`, `EPISODE_TIME_S`, `RESET_TIME_S`, `RECORD_FPS`, `PUSH_TO_HUB` |
| `replay` | 녹화 에피소드를 팔로워에 재실행 | Follower only | `HF_DATASET_REPO_ID`, `EPISODE_INDEX` |
| `calibrate` | 리더 또는 팔로워 영점 보정 | 한쪽만 | `CALIBRATE_TARGET` (`robot` \| `teleop`) |
| `setup-motors` | Feetech 모터 ID/Baud 초기 설정 | 한쪽만 | `CALIBRATE_TARGET` |
| `find-cameras` | 시스템 카메라 자동 검출 | - | 위치 인자: `opencv` \| `realsense` |
| `find-port` | 직렬 포트 자동 감지 (인터랙티브) | - | - |
| `dataset-viz` | Rerun 기반 데이터셋 시각화 | - | `HF_DATASET_REPO_ID`, `EPISODE_INDEX`, `VIZ_MODE`, `VIZ_WS_PORT` |
| `policy-client` | 정책 서버에 gRPC 로 붙어 follower arm 구동 | Follower + 카메라 | `POLICY_SERVER_ADDRESS`, `POLICY_TYPE`, `POLICY_PATH`, `POLICY_DEVICE`, `TASK`, `ACTIONS_PER_CHUNK`, `CHUNK_SIZE_THRESHOLD`, `POLICY_CLIENT_FPS` |
| `edit-dataset` | 데이터셋 편집 (인자 완전 위임) | - | CLI 인자로 직접 전달 |
| `info` | LeRobot / Python / 시스템 정보 | - | - |
| `bash` \| `shell` | 컨테이너 인터랙티브 쉘 | - | - |
| `python <args>` | 컨테이너 내 Python 실행 | - | - |

### `policy-server` 서비스 — Async inference 서버 (`policy-entrypoint.sh`)

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server <mode> [args...]
# 또는 (CMD 기본값 = policy-server 로 즉시 서버 기동):
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
```

| 모드 | 설명 | 필요 하드웨어 | 핵심 env var |
|---|---|---|---|
| `prepare-model` | 호스트 HF 캐시에 모델 가중치 다운로드 | - | `MODEL_REPO_ID`, `MODEL_REVISION`, `PREPARE_MODEL_EXTRA_ARGS` |
| `policy-server` | SmolVLA 등 Async inference gRPC 서버 (기본 CMD) | GPU 권장 | `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT`, `POLICY_FPS`, `INFERENCE_LATENCY`, `OBS_QUEUE_TIMEOUT` |
| `train` | Policy 학습 (SmolVLA 등 — 인자 완전 위임) | GPU 권장 | CLI 인자로 직접 전달 |
| `eval` | Policy 평가/롤아웃 (인자 완전 위임) | GPU 권장 | CLI 인자로 직접 전달 |
| `info` | LeRobot / Python / 시스템 정보 | - | - |
| `bash` \| `shell` | 컨테이너 인터랙티브 쉘 | - | - |
| `python <args>` | 컨테이너 내 Python 실행 | - | - |

---

## 5. SO-101 Motor Setup / Calibration

`.env` 의 다음 값을 채우고 실행:

| 이름 | 설명 |
|---|-----|
| CALIBRATE_TARGET | `robot`: 팔로워 모터/보정, `teleop`: 리더 모터/보정 |
| TELEOP_PORT / TELEOP_ID | 리더 암 포트 + ID |
| ROBOT_PORT / ROBOT_ID | 팔로워 암 포트 + ID |

```bash
# CALIBRATE_TARGET=robot / teleop 각각 1회
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot setup-motors
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot calibrate
```

---

## 6. Teleoperation

`--display_data=true` 사용 시 호스트에서 사전 실행: `pip install rerun-sdk==0.26.2; rerun`.

| 이름 | 설명 |
|-----|------|
| ENABLED_CAMERAS | 활성 카메라 부분집합. 기본 `wrist,front`. 3개 운영 시 `wrist,front,top` |
| FRONT_CAM_PORT / WRIST_CAM_PORT / TOP_CAM_PORT | 카메라 포트 (`/dev/video*`) |
| CAM_WIDTH / CAM_HEIGHT / CAM_FPS / CAM_FOURCC | 카메라 해상도·FPS·fourcc |
| DISPLAY_DATA | 데이터 시각화 여부 |
| DISPLAY_IP / DISPLAY_PORT | Docker 송출 시 `host.docker.internal:9876` |
| TELEOP_EXTRA_ARGS | 기타 인자 |

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot teleop
```

---

## 7. 데이터셋 녹화 / 재실행 / 시각화

### 녹화 (`record`)

| 이름 | 설명 |
|-----|------|
| SINGLE_TASK | 에피소드 작업 설명 (snake_case) |
| HF_DATASET_REPO_ID | HF Hub 데이터셋 ID, 기본 `${HF_USER}/${SINGLE_TASK}` |
| NUM_EPISODES / EPISODE_TIME_S / RESET_TIME_S | 수집 파라미터 |
| RECORD_FPS | 데이터셋 저장 FPS |
| PUSH_TO_HUB | HF 업로드 여부 |
| DATASET_ROOT | 컨테이너 내 저장 경로 (`/workspace/datasets` 마운트) |
| RECORD_EXTRA_ARGS | 기타 인자 |

키보드 조작 (→ 조기 종료 / ← 재녹화 / ESC 세션 종료 + 인코딩·업로드).

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot record
```

### 재실행 (`replay`)

팔로워 직렬 포트만 있으면 동작.

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot replay
```

### 시각화 (`dataset-viz`)

`VIZ_MODE=local` 컨테이너 내부 뷰어 / `VIZ_MODE=distant` WebSocket 서버 (호스트에서 `rerun ws://localhost:${VIZ_WS_PORT}` 접속).

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot dataset-viz
```

### 진단 유틸리티

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot find-port
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot find-cameras opencv
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot find-joint-limits
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot info
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot bash
```

---

## 8. 모델 가중치 준비

HF 캐시는 명명 볼륨 `lerobot_hf_cache` 가 두 컨테이너의 `/root/.cache/huggingface` 에 마운트된다. 두 서비스가 동일 볼륨을 공유하므로 한 번 받은 모델을 양쪽이 모두 사용.

```bash
# .env 의 MODEL_REPO_ID 로 다운로드 (기본 lerobot/smolvla_base)
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server prepare-model

# 위치 인자로 다른 모델 받기
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server prepare-model nvidia/GR00T-N1.5-3B
```

다른 머신으로 캐시를 옮기려면 (명명 볼륨 특성상 rsync 불가, tarball 경유):

```bash
# 출발지: 명명 볼륨 tarball 추출
docker run --rm -v lerobot_hf_cache:/cache -v "$(pwd)":/out alpine \
    tar czf /out/hf_cache.tar.gz -C /cache .
rsync -av --progress hf_cache.tar.gz user@target-server:/tmp/

# 도착지: 빈 볼륨에 import
docker volume create lerobot_hf_cache
docker run --rm -v lerobot_hf_cache:/cache -v /tmp:/in alpine \
    tar xzf /in/hf_cache.tar.gz -C /cache
```

---

## 9. Policy 학습

`.env` 의 학습 파라미터:

| 이름 | 설명 |
|-----|------|
| HF_DATASET_REPO_ID / DATASET_ROOT | 학습 데이터셋 위치 |
| POLICY_TYPE / POLICY_PATH / POLICY_REPO_ID | 정책 종류·베이스 체크포인트·결과 push 경로 |
| JOB_NAME / BATCH_SIZE / TRAIN_STEPS / OUTPUT_DIR / DEVICE | 일반 학습 인자 |
| WANDB_ENABLE | W&B 연동 |
| TRAIN_EXTRA_ARGS | 추가 `lerobot-train` 인자 |
| **학습 속도 최적화** | |
| NUM_WORKERS | 데이터로더 워커 수 (기본 8) |
| COMPILE_MODEL / COMPILE_MODE | `torch.compile` 활성화 (10K+ steps 에서 ~20–30% 향상, 첫 스텝 컴파일 비용) |
| NUM_PROCESSES | 사용 GPU 수. 2+ 지정 시 `accelerate launch --num_processes` DDP 자동 전환 |
| MIXED_PRECISION | 혼합 정밀도 (기본 `bf16`, Ampere+ 권장. 구형 GPU `fp16`) |

**호출 컨테이너는 `policy-server`** — Dockerfile.smolvla 에만 transformers / accelerate / num2words 가 설치됨 (lerobot 이미지에서 SmolVLA 학습 불가).

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server train
```

추가 인자는 env var 빌드 값 뒤에 붙어 last-wins 로 덮어쓴다:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server train --resume=true
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server train --steps=5000
```

---

## 10. Policy 평가 및 추론

**실기기 추론** — `record` 모드에 `--policy.path=` 를 전달하면 학습된 정책으로 팔로워를 구동하면서 에피소드 기록:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot record \
    --robot.type=so101_follower \
    --robot.port=${ROBOT_PORT} \
    --robot.cameras="{
        wrist: {type: opencv, index_or_path: ${WRIST_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, warmup_s: ${CAM_WARMUP_S}, fourcc: ${CAM_FOURCC}},
        front: {type: opencv, index_or_path: ${FRONT_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, warmup_s: ${CAM_WARMUP_S}, fourcc: ${CAM_FOURCC}},
        }" \
    --robot.id=${ROBOT_ID} \
    --teleop.type=so101_leader \
    --teleop.port=${TELEOP_PORT} \
    --teleop.id=${TELEOP_ID} \
    --dataset.single_task=${SINGLE_TASK} \
    --dataset.repo_id=${HF_USER}/${SINGLE_TASK} \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.episode_time_s=${EPISODE_TIME_S} \
    --dataset.reset_time_s=${RESET_TIME_S} \
    --dataset.push_to_hub=${PUSH_TO_HUB} \
    --dataset.fps=${RECORD_FPS} \
    --dataset.root=${DATASET_ROOT} \
    ${RECORD_EXTRA_ARGS} \
    --policy.path=${POLICY_PATH}
```

**시뮬 평가** — `eval` 모드는 `lerobot-eval` 에 인자 완전 위임:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server eval \
    --policy.path=${POLICY_PATH} \
    --env.type=pusht \
    --eval.n_episodes=20 \
    --eval.batch_size=10
```

---

## 11. Async Inference Policy Server (SmolVLA)

`lerobot.async_inference.policy_server` 를 gRPC :8080 으로 띄워 SmolVLA 정책을 원격 추론. 동일 호스트의 `record` / `robot_client` 가 관측을 보내면 서버가 액션 청크를 비동기로 반환. 서버는 policy-agnostic 이므로 모델 종류/체크포인트/디바이스는 **클라이언트** 가 `--policy_type=smolvla --pretrained_name_or_path=...` 로 주입.

> 본 레포 기준 권장 체크포인트: **`lerobot/smolvla_base`** (공식 베이스, ~450M params, ~2 GB VRAM). `pick_pen` task 의 SO-101 fine-tune 공개 체크포인트는 없으므로 fine-tune 전에는 베이스 모델로 파이프라인 검증만 가능.

| 이름 | 설명 |
|---|---|
| POLICY_SERVER_HOST | bind 주소 (기본 `0.0.0.0`) |
| POLICY_SERVER_PORT | gRPC 포트 (기본 `8080`) |
| POLICY_FPS | 컨트롤 루프 FPS (기본 `30`) |
| INFERENCE_LATENCY | 목표 추론 latency 초 (기본 `0.033`) |
| OBS_QUEUE_TIMEOUT | 관측 큐 timeout 초 (기본 `2`) |
| POLICY_SERVER_EXTRA_ARGS | 추가 인자 |

서버 기동:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
docker compose logs -f policy-server   # gRPC bind 로그
```

클라이언트는 `lerobot` 서비스의 `policy-client` 모드. `.env` 의 `POLICY_*` / `TASK` / `ROBOT_*` / `*_CAM_*` 변수가 robot_client CLI 인자로 자동 매핑된다.

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot policy-client
```

원격 학습 서버의 정책 서버에 붙으려면:

```bash
POLICY_SERVER_ADDRESS=10.0.0.5:8080 \
    docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot policy-client
```

**주의사항**

- **액션 품질**: `lerobot/smolvla_base` 는 SO-101 에 미학습 → 액션 품질 무작위에 가까움. 일차 목적은 **파이프라인 검증** (gRPC 송수신, 카메라/state 매핑, action chunk 적용).
- **초기 로딩**: 첫 호출 시 VLM 백본 (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) 자동 다운로드로 30–60 초 추가 대기. `prepare-model HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 로 미리 받아두면 즉시 로드.
- **보안**: pickle deserialization RCE 위험 (CVE-2026-25874). 본 구성은 같은 호스트 loopback 한정. 외부 노출 시 SSH 터널 / mTLS 래퍼 / 방화벽 추가.
- **카메라 키 매핑**: 체크포인트 `input_features` 키와 클라이언트가 보내는 카메라 키가 정확히 일치해야 함. base 모델 직결 시 `KeyError: 'observation.images.wrist'` — fine-tune 정공법으로 해결 ([§12](#12-fine-tune-워크플로-pick_pen)).

---

## 12. Fine-tune 워크플로 (pick_pen)

앞 단계(2~11)를 엮은 end-to-end 시나리오. `lerobot/smolvla_base` 는 `camera1/2/3` 키로 학습된 베이스라 SO-101 (`wrist`/`front`) 클라이언트와 키 불일치 (`KeyError: 'observation.images.wrist'`) 가 발생한다. 정공법은 SO-101 데이터셋으로 SmolVLA 를 fine-tune 해 새 체크포인트의 `input_features` 가 자연스럽게 `wrist/front` 가 되도록 하는 것.

**1) 데이터셋 수집** (Windows 워크스테이션 `lerobot` 컨테이너):

```bash
# .env: SINGLE_TASK="pick the pen", HF_DATASET_REPO_ID=${HF_USER}/so101_pick_pen,
#       NUM_EPISODES=50, EPISODE_TIME_S=30, PUSH_TO_HUB=true
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot record
```

- 카메라 키 `wrist`, `front`, `top` 으로 저장 (변경 불필요)
- `PUSH_TO_HUB=true` 면 학습 머신에서 HF Hub pull 가능
- 50+ 에피소드, 다양한 grasp pose / pen 위치로 시연

**2) 데이터셋 검증** (선택):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot dataset-viz
```

**3) 학습 머신 선택**:

- **Linux 학습 서버** (RTX PRO 5000 Blackwell 48 GB, 권장 — 빠른 회전): 정책 서버 이미지만 빌드. 데이터셋은 HF Hub pull 또는 `rsync -av datasets/ user@<server>:/path/datasets/`
- **Windows A4000 (16 GB)** (느림, batch_size 4–8): 데이터셋 로컬

**4) Fine-tune 실행** (`policy-server` 컨테이너):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server train \
    --policy.path=lerobot/smolvla_base \
    --policy.repo_id=${HF_USER}/smolvla_pick_pen \
    --policy.push_to_hub=true \
    --dataset.repo_id=${HF_DATASET_REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    --output_dir=${OUTPUT_DIR} \
    --steps=20000 \
    --batch_size=64 \
    --job_name=smolvla_pick_pen \
    --wandb.enable=true
```

**5) 체크포인트 배포** (Windows 워크스테이션):

```bash
# .env 의 POLICY_PATH=${HF_USER}/smolvla_pick_pen
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server prepare-model ${HF_USER}/smolvla_pick_pen
```

**6) 정책 서버 재기동 + 실기기 추론**:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot policy-client
```

fine-tuned 체크포인트의 `input_features` 가 `wrist/front/top` 이므로 카메라 키 매핑 자동 일치.
