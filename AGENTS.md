# AGENTS.md

## 프로젝트 개요

SO-ARM101 6축 로봇 팔용 **실기기 LeRobot 파이프라인 + Isaac Lab Sim-to-Real 시뮬레이션**.

- **실기기 경로 (Docker)**: `docker/docker-compose.yaml` 의 두 서비스가 각자의 진입점을 사용한다. `lerobot` (Dockerfile.lerobot + `lerobot-entrypoint.sh`, teleop deps) 가 텔레오퍼레이션·데이터 수집·정책 학습·시각화를, `policy-server` (Dockerfile.policy + `policy-entrypoint.sh`, policy+async deps) 가 async inference gRPC 서버를 담당. SmolVLA 가 기본 정책이며 GR00T N1.5(flash-attn 필요) 등은 정책 서버 이미지에만 의존성을 추가한다.
- **시뮬 경로 (Host uv)**: `src/sim_to_real/` 가 `SimToReal-SO101-PickPen-v0` Gym 환경을 등록하고, `scripts/environments/teleoperation/teleop_se3_agent.py` 등이 Isaac Lab `ManagerBasedRLEnv` 로 진입한다. 호스트의 `uv run scripts/...` 로 실행하며, Isaac Sim 5.1 / IsaacLab 2.3 / leisaac 0.4.0 (git tag) 가 `isaac` 의존성 그룹으로 설치된다. Docker 시뮬 이미지는 현재 미연결 (`Dockerfile.leisaac` 부활 시 별도 진입점 신설 예정).

운영 환경: Windows 워크스테이션과 Linux 원격 서버. 자세한 사양은 §환경 사양 참조.

자세한 사용법은 `README.md`(허브)와 경로별 가이드(`docs/PATH_A_NATIVE.md` / `docs/PATH_B_DOCKER.md` / `docs/PATH_C_ISAAC_SIM.md`)에, 트러블슈팅은 `docs/TROUBLESHOOTING.md`에 정리되어 있다. 본 문서는 **README에 없는** 내부 구조·규칙과 자주 쓰는 명령만 다룬다.

## 환경 사양

| | Windows 워크스테이션 | Linux 학습 서버  |
|---|---|---|
| **OS** | Windows 11 Pro | Ubuntu 24.04.3 LTS (kernel 6.14.0-1014-oem), Dell Pro Max Tower T2 FCT2250, bare-metal |
| **CPU** | Intel Xeon W-2245 @ 3.90GHz (8 cores / 16 threads, L3 16.5 MB) | Intel Core Ultra 5 245K (Arrow Lake, 14 cores / 14 threads, L3 24 MB) |
| **RAM** | 64 GB | 128 GB DDR5 (+ swap 8 GB) |
| **Storage** | NVMe SSD 512 GB + SATA HDD 1 TB | NVMe SSD 477 GB (ext4 `/`) + SATA HDD 3.6 TB (`/DISK1`) |
| **GPU** | NVIDIA RTX A4000 16 GB (driver 596.36, CUDA 13.2, compute_cap 8.6 Ampere) | NVIDIA RTX PRO 5000 Blackwell 48 GB GDDR7 (driver 580.95.05-open, compute_cap 12.0 Blackwell) |

테스트 스위트나 lint config는 현재 정의되어 있지 않다 (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 없음). 변경 검증은 컨테이너 빌드 + 실기기 실행 + `uv run` 시뮬 1회 실행으로 수행한다.

## Docker 컨테이너 구조 (실기기 경로)

- **활성 서비스**:
  - `lerobot` (이미지 `lerobot-so101:0.4.4`, `docker/Dockerfile.lerobot`) — teleop / record / replay / train / eval / dataset-viz. `docker compose -f docker/docker-compose.yaml build lerobot`.
  - `policy-server` (이미지 `policy-server:0.4.4`, `docker/Dockerfile.policy`) — async inference gRPC 서버 (`policy-entrypoint.sh policy-server`). `docker compose -f docker/docker-compose.yaml build policy-server`. teleop 이미지와 의존성 격리: GR00T 의 flash-attn / 원격 inference(H100 ↔ Windows) 확장 대비.
- **빌드 스테이지** (`Dockerfile.lerobot` / `Dockerfile.policy` 가 Stage 1–4 동일 → BuildKit 캐시 공유): base(`nvidia/cuda:12.8.0-runtime-ubuntu24.04` + apt) → uv → python 3.11 venv → torch 2.7.0/torchvision 0.22.0 (cu128) → `uv sync --group <teleop|policy async> --no-install-project` → app(entrypoint, teleop 만 udev rules).
- **디바이스 마운트**: `${TELEOP_PORT}` `${ROBOT_PORT}` (직렬 암), `${FRONT_CAM_PORT}` `${FRONT_CAM_META_PORT}` `${WRIST_CAM_PORT}` `${WRIST_CAM_META_PORT}` `${TOP_CAM_PORT}` `{$TOP_CAM_META_PORT}`(UVC 캡처/메타 노드 쌍).
- **호스트 볼륨**: `./datasets`, `./logs`, `./outputs` → 컨테이너 `/workspace/*`. 명명 볼륨 `lerobot_hf_cache` → `/root/.cache/huggingface` (두 서비스 공유). 다른 머신으로 옮길 때는 `docker run -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export 후 전송.
- **권한·네트워크**: `privileged: true` (udev/USB 접근), `network_mode: host` (rerun 뷰어·ROS 브릿지), `ipc: host`. GPU 1장 예약 (`deploy.resources.reservations.devices`).
- **서비스별 진입점**:
  - `docker/lerobot-entrypoint.sh` (lerobot 서비스): `teleop` / `record` / `replay` / `calibrate` / `setup-motors` / `find-port` / `find-cameras` / `find-joint-limits` / `dataset-viz` / `policy-client` / `edit-dataset` / `info` / `bash` / `python`. `policy-client` 는 `lerobot.async_inference.robot_client` 로 정책 서버에 gRPC 접속해 SO-101 follower 를 구동 — `async` 의존성 그룹(grpcio + protobuf) 이 teleop 이미지에도 함께 설치된다.
  - `docker/policy-entrypoint.sh` (policy-server 서비스): `prepare-model` / `policy-server` / `train` / `eval` / `info` / `bash` / `python`. CMD 기본값 `policy-server`. **train/eval 은 이쪽**: SmolVLA 학습이 필요로 하는 transformers / accelerate / num2words 가 `policy` 그룹에만 있고 lerobot 이미지에 미설치이기 때문.
  - 모드별 env var 매핑은 각 스크립트 상단 `${VAR:-default}` 블록과 case 분기 주석에 정리되어 있다.
- **`.env` 주입 경로**: `docker compose --env-file .env` 가 컨테이너에 환경변수로 주입하고, `entrypoint.sh` 가 기본값을 채워 `lerobot-*` CLI 인자로 매핑.

## 시뮬레이션 구조 (Isaac Lab 경로)

호스트 uv 환경에서 직접 실행한다. Docker 미연결 — RT 코어 GPU 가 있는 Windows 워크스테이션에서 `uv sync --group isaac` 후 사용.

- **Python 패키지 `sim_to_real`** (`src/sim_to_real/`): `import sim_to_real` 한 번에 monkey patch + Gym 환경 등록이 모두 일어난다. 내부 구조:
  - `assets/scenes/pen_desk.py` — `PEN_DESK_CFG` (UsdFileCfg 래퍼)
  - `tasks/__init__.py` — `isaaclab_tasks.utils.import_packages` 로 하위 task config 자동 등록 (블랙리스트: `utils`, `.mdp`)
  - `tasks/pick_pen/` — `SimToReal-SO101-PickPen-v0` (entry point `pick_pen_env_cfg:PickPenEnvCfg`, env class `ManagerBasedRLEnv`), `mdp/{observations,terminations}.py`
  - `utils/{constant,domain_randomization}.py` — `randomize_object_in_ellipse` / `randomize_object_on_arc` 두 `EventTermCfg` wrapper (leisaac 의 사각형 분포로는 표현 불가능한 타원 면적 균등 / 호 1D sampling)
- **USD 에셋** (`assets/`):
  - `scenes/pen_desk/scene.usd` + `objects/<Pen*,PenCup>/<Name>.usd` — kitchen_with_orange 패턴 (객체별 self-contained USD + `prepend payload` 참조)
  - `robots/` — SO-101 follower USD + 편집용 URDF
  - 좌표 정합: `SCENE_OFFSET` 상수로 top-level translate 일괄 시프트. 펜 4개 + 펜컵 collider 는 `PhysicsCollisionAPI` 직접 부여 (Cube proxy 미사용)
  - 펜/펜컵 영역 분리: 펜은 그린 타원 (`y ∈ [0.22, 0.26]`), 펜컵은 주황 호 (`y ∈ [0.34, 0.40]`). y 마진 ≥ 0.08 m 라 펜이 펜컵 안에 spawn 불가
- **진입 스크립트** (`scripts/`):
  - `environments/list_envs.py` — leisaac 등록 환경 일람
  - `environments/teleoperation/teleop_se3_agent.py` — `gym.make("SimToReal-SO101-PickPen-v0")` + leisaac 디바이스 레이어 (`keyboard` / `gamepad` / `so101leader` / `so101leader` remote ZMQ / `bi-so101leader` / `lekiwi-*`)
  - `environments/teleoperation/replay.py` — 녹화된 시퀀스 재실행
  - `environments/teleoperation/so101_joint_state_server.py` — ZMQ PUB 으로 실제 SO-101 leader 상태를 원격에 송출 (`SO101LeaderRemote` 의 카운터파트)
  - `environments/utils/{inspect_robot_materials,patch_robot_colors}.py` — USD 머티리얼 진단/패치
- **씬 재생성**: USD 6개 (`scene.usd` + 펜 4개 + PenCup) 는 author 스크립트로 일괄 재생성. 좌표 변경 시 `pyproject.toml` 의 `[tool.setuptools]` 가 `src/` 를 기준으로 패키지를 찾으므로 import 경로는 무관하나, `pen_cup_center_xy` 같은 world-frame 상수는 `tasks/pick_pen/pick_pen_env_cfg.py` 와 오라클 state machine 양쪽을 같이 갱신해야 한다.

## Python 패키지 / 의존성 호환성 규칙

- **패키지 이름** `sim_to_real` (`pyproject.toml`). `[build-system] requires=["setuptools<82"]`, `[tool.setuptools.packages.find] where=["src"]` 로 `src/sim_to_real/` 를 editable 설치.
- **공용 deps**: `h5py<3.16`, `hf-xet>=1.4.3`, `pyzmq>=27.1.0`, `lerobot[feetech]>=0.4.4`, `torch>=2.7`, `torchvision>=0.22`, **`usd-core>=26.5`** (순수 Python — isaac 그룹 없이도 씬 author/검증 가능하도록 공용).
- **의존성 그룹**: `teleop` (실기기, ffmpeg + evdev Linux 한정), `policy` (lerobot[smolvla] + accelerate + num2words), `async` (grpcio + protobuf), `isaac` (isaacsim[all,extscache]==5.1.0 + leisaac[isaaclab,gr00t]), `dev` (ipykernel).
- **`leisaac`** 는 `[tool.uv.sources]` 의 git tag `v0.4.0` 에서 설치 (vendored 사본 없음).

ABI 호환성 핀 (`pyproject.toml`). 임의 업그레이드 / `uv lock --upgrade` 금지.

| 핀 | 이유 | 어기면 |
|---|---|---|
| `numpy==1.26.0` (override) | Isaac Sim 5.1.0의 `isaacsim_kernel`이 강제 | uv 설치 자체가 실패 |
| `pyarrow<19` (override) | numpy 1.x C-API 호환 마지막 메이저. PyArrow 19+는 numpy 2.x ABI 전용이라 numpy 1.26과 segfault | Isaac Sim 시작 후 ~30초 silent crash (`arrow.dll!arrow_vendored::date::current_zone` 백트레이스) |
| `datasets>=4.0.0,<4.7.0` (override) | datasets 4.7.0+ 가 `pa.json_()` 사용 (PyArrow 19+ 필요) — pyarrow<19 와 충돌 회피 | uv resolve 실패 |
| `h5py<3.16` | Isaac Sim 번들 HDF5 1.14.x와 ABI 일치. h5py 3.16+는 HDF5 2.0 번들 | `Windows fatal exception: code 0xc0000139` |
| `torch==2.7.0+cu128` | Isaac Sim 5.1 번들 CUDA 12.8과 일치 | 기동 시 CUDA 호출 실패 |
| `packaging>=24.2,<26.0` (override) | 다른 패키지 메타데이터 검증 충돌 회피 | `uv sync` resolve 실패 |
| `setuptools<82` (build-constraint) | 일부 의존성의 `pkg_resources` 호환 | sdist 빌드 실패 |

`override-dependencies`는 transitive 제약을 강제로 무시한다. 예: `datasets 4.x`가 `pyarrow>=21`을 요구하지만 override로 `pyarrow<19` 설치 가능 — 본 레포의 검증된 워크플로(HDF5 → isaaclab2lerobot 변환)에서는 런타임에도 정상 동작.

## 시뮬레이션 환경 제약

**RT 코어 없는 GPU(H100/A100)는 NVIDIA가 Isaac Sim 5.1 공식 미지원으로 명시.** 시스템 요구사항 문서가 *"GPUs without RT Cores (A100, H100) are not supported."*라고 못박음. 카메라 sensor가 raytracing pipeline 생성 실패 → CUDA illegal memory access. 시뮬 데이터 수집·렌더링은 NVIDIA 권장(RTX 4080+) 또는 RT 코어·16 GB VRAM 충족 GPU(RTX A4000/A5000/A6000, L40/L40S, RTX 6000 Ada, RTX PRO 5000/6000 Blackwell, GeForce RTX 40/50)에서만. Windows 워크스테이션(RTX A4000 16 GB)·Linux 서버(RTX PRO 5000 Blackwell 48 GB) 모두 조건 충족 — 시뮬·학습·추론 양쪽에서 모두 사용 가능.

## 사용자 환경 컨벤션

- Windows USB 포워딩은 PowerShell(`usbipd bind/attach`), 컨테이너 내부 실행과 호스트 측 보조 명령은 bash 사용
- HF/W&B 토큰은 `.env`에서 읽음. `.env.example`이 템플릿. 실기기는 `docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot <mode>`, 시뮬은 `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickPen-v0 ...` 가 표준 실행 패턴

## 운영 규칙

### 에러 수정 후 docs/TROUBLESHOOTING.md에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때, 그 경험을 다음 세션·다른 작업자가 활용할 수 있도록 `docs/TROUBLESHOOTING.md`에 항목을 추가한다.

- 양식은 기존 항목과 동일: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류의 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌, USD/씬 물리 등)는 인접 섹션에 배치해 흐름을 맞출 것
- 필요하면 §핵심 의존성 표나 §실제로 주의해야 할 로그 리스트도 함께 갱신
- 수정이 실패한 경우도 README에는 올리지 않는다

### USD 씬 좌표·물리 상수 동기화

`assets/scenes/pen_desk/` USD 의 좌표·치수를 변경하면 다음 위치도 같이 갱신한다.

- `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py::PEN_CUP_CENTER_XY` — `mdp.pen_in_cup` 종료 조건의 컵 기준점 (world frame)
- `SCENE_OFFSET` 일괄 시프트로 해결 가능한 경우 author 스크립트 한 번 재실행으로 USD 6개를 모두 갱신
