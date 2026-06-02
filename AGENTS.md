# AGENTS.md

SO-ARM101 6축 로봇 팔용 **실기기 LeRobot 파이프라인 + Isaac Lab Sim-to-Real 시뮬레이션**.

본 문서는 **README에 없는** 내부 구조·규칙·자주 쓰는 명령만 다룬다. 사용법은 아래 문서를 참조한다.

| 문서 | 내용 |
|---|---|
| `README.md` | 허브 |
| `docs/PATH_A_NATIVE.md` | Windows 네이티브 실행 |
| `docs/PATH_B_DOCKER.md` | Docker 실기기 경로 |
| `docs/PATH_C_ISAAC_SIM.md` | Isaac Sim 시뮬 경로 |
| `docs/TROUBLESHOOTING.md` | 트러블슈팅 |

## 두 실행 경로

| | 실기기 경로 | 시뮬 경로 |
|---|---|---|
| **실행 방식** | Docker (`docker/docker-compose.yaml`) | Host uv (`uv run scripts/...`) |
| **진입점** | `lerobot` / `policy-server` 두 서비스 | `SimToReal-SO101-PickPen-v0` Gym 환경 |
| **담당** | 텔레옵·데이터 수집·학습·추론·시각화 | Isaac Lab `ManagerBasedRLEnv` |
| **스택** | LeRobot 0.4.4 / 0.5.1 | Isaac Sim 5.1 / IsaacLab 2.3 / leisaac 0.4.0 (git tag) |
| **기본 정책** | SmolVLA (GR00T N1.5 는 policy-server 이미지에만) | — |

- 시뮬 경로는 `src/sim_to_real/` 가 Gym 환경을 등록하고 `scripts/environments/teleoperation/teleop_se3_agent.py` 등이 진입한다. `isaac` 의존성 그룹으로 설치.
- Docker 시뮬 이미지는 현재 미연결 (`Dockerfile.leisaac` 부활 시 별도 진입점 신설 예정).

## 환경 사양

| | Windows 워크스테이션 | Linux 학습 서버 |
|---|---|---|
| **OS** | Windows 11 Pro | Ubuntu 24.04.3 LTS (kernel 6.14.0-1014-oem), Dell Pro Max Tower T2 FCT2250, bare-metal |
| **CPU** | Intel Xeon W-2245 @ 3.90GHz (8C/16T, L3 16.5 MB) | Intel Core Ultra 5 245K (Arrow Lake, 14C/14T, L3 24 MB) |
| **RAM** | 64 GB | 128 GB DDR5 (+ swap 8 GB) |
| **Storage** | NVMe SSD 512 GB + SATA HDD 1 TB | NVMe SSD 477 GB (ext4 `/`) + SATA HDD 3.6 TB (`/DISK1`) |
| **GPU** | RTX A4000 16 GB (driver 596.36, CUDA 13.2, cc 8.6 Ampere) | RTX PRO 5000 Blackwell 48 GB GDDR7 (driver 580.95.05-open, cc 12.0 Blackwell) |

테스트 스위트·lint config 없음 (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 미정의). 변경 검증 = 컨테이너 빌드 + 실기기 실행 + `uv run` 시뮬 1회 실행.

## Docker 컨테이너 구조 (실기기 경로)

### 서비스

| 서비스 | 이미지 / Dockerfile | 스택 | 역할 |
|---|---|---|---|
| `lerobot` | `lerobot-so101:0.4.4` / `Dockerfile.lerobot` | Python 3.11 + LeRobot 0.4.4 (teleop deps) | teleop / record / replay / train / eval / dataset-viz |
| `policy-server` | `policy-server:0.5.1` / `Dockerfile.policy` | Python 3.12 + LeRobot 0.5.1 (policy+async deps) | async inference gRPC 서버 |

- 빌드: `docker compose -f docker/docker-compose.yaml build <서비스>`. 두 이미지는 torch/CUDA 계층 일부만 BuildKit 캐시로 공유.
- 의존성 격리 이유: GR00T 의 flash-attn / 원격 inference(H100 ↔ Windows) 확장 대비.

### compose 설정

- **디바이스 마운트**: `${TELEOP_PORT}` `${ROBOT_PORT}` (직렬 암), `${FRONT_CAM_PORT}` `${FRONT_CAM_META_PORT}` `${WRIST_CAM_PORT}` `${WRIST_CAM_META_PORT}` `${TOP_CAM_PORT}` `${TOP_CAM_META_PORT}` (UVC 캡처/메타 노드 쌍).
- **권한·네트워크**: `privileged: true` (udev/USB), `network_mode: host` (rerun 뷰어·ROS 브릿지), `ipc: host`. GPU 1장 예약.
- **호스트 볼륨**:
  - `./datasets`, `./logs`, `./outputs` → `/workspace/*` (두 서비스)
  - `./scripts` → `/workspace/scripts` (policy-server 만 — `policy-server-rtc` 모드가 `policy_server_rtc.py` 참조)
  - 명명 볼륨 `lerobot_hf_cache` (compose key `hf_cache`) → `/workspace/.cache/huggingface` (HF_HOME, 두 서비스 공유). non-root UID 실행 때문에 `/root` 가 아닌 `/workspace` 하위. 머신 이전 시 `docker run -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export.

### 진입점 모드

**`lerobot-entrypoint.sh`** (lerobot 서비스):
`teleop` · `record` · `replay` · `calibrate` · `setup-motors` · `find-port` · `find-cameras` · `find-joint-limits` · `dataset-viz` · `policy-client` · `edit-dataset` · `info` · `bash` · `python`

- `policy-client` 는 `lerobot.async_inference.robot_client` 로 정책 서버에 gRPC 접속해 SO-101 follower 구동. `async` 그룹(grpcio + protobuf)이 teleop 이미지에도 설치됨. 실제로는 `policy-client-shim.py` 경유 (아래 참조).

**`policy-entrypoint.sh`** (policy-server 서비스, CMD 기본값 `policy-server`):
`prepare-model` · `policy-server` · `policy-server-rtc` · `train` · `eval` · `info` · `bash` · `python`

- `policy-server-rtc` 는 `policy_server_rtc.py` 로 서버 측 Real-Time Chunking(RTC) 가이던스를 주입한 async 서버 (gRPC 프로토콜·클라이언트 변경 없음, `RTC_*` env 튜닝).
- **train/eval 은 이쪽 서비스**: SmolVLA 학습이 쓰는 transformers / accelerate / num2words 가 `policy` 그룹에만 있고 lerobot 이미지엔 미설치.

모드별 env var 매핑은 각 스크립트 상단 `${VAR:-default}` 블록과 case 분기 주석에 정리됨.

### 빌드·런타임 보조 스크립트

| 파일 | 용도 |
|---|---|
| `docker/groot_compat_patch.py` | `Dockerfile.policy` 빌드 시 `lerobot[smolvla,async]==0.5.1` 설치 직후 1회 실행. transformers 5.3 + torch 2.10 에서 LeRobot 0.5.1 GR00T wrapper 가 깨지는 4지점을 site-packages 에서 멱등 패치. 형태가 다르면 `RuntimeError` 로 빌드 중단(버전 트립와이어) — lerobot/transformers 업그레이드 시 이 패치부터 점검. |
| `docker/policy-client-shim.py` | `policy-client` 모드 실제 진입점. lerobot 0.4.4 `robot_client.py` 가 built-in robot config 모듈을 import 안 해 `--robot.type` 이 거부되는 회귀(huggingface/lerobot#3078)를 선행 import 로 보강. rerun viewer(`DISPLAY_DATA=true`) monkey patch 유효를 위해 `async_client()` 직접 호출(runpy 미사용). |
| `docker/lerobot_keyboard_stdin.py` + `.pth` | WSLg X 서버가 Windows Terminal 키 입력을 못 보는 환경에서 pynput 리스너 대신 `/dev/tty` + termios cbreak 리더로 `init_keyboard_listener` 대체. `.pth` 가 Python 시작 시 hook 설치 (lerobot 이미지 site-packages 에 COPY). |

### `.env` / 모델 프로필

- **주입 경로**: 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE}.env]` 가 컨테이너에 주입(나중 파일이 override). `entrypoint.sh` 가 기본값을 채워 `lerobot-*` CLI 인자로 매핑. `--env-file .env` 는 compose 보간용.
- **모델 프로필**: 모델 간 값이 다른 변수는 `env/<name>.env` 로 분리하고, `.env` 의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. 새 모델 = 프로필 파일 추가.
  - 분리 변수: `POLICY_TYPE` / `TRAIN_POLICY_TYPE` / `POLICY_BASE_MODEL_PATH` / tokenizer·embodiment·chunk·n_action_steps / `ACTIONS_PER_CHUNK` / `POLICY_REPO_ID` / `JOB_NAME`
  - train 출발 모델 라우팅: `POLICY_BASE_MODEL_PATH` 단일 변수 + `TRAIN_POLICY_TYPE` 유무로 `--policy.path`(체크포인트) ↔ `--policy.type` + `--policy.base_model_path`(native 베이스).

## 시뮬레이션 구조 (Isaac Lab 경로)

호스트 uv 환경에서 직접 실행. Docker 미연결 — RT 코어 GPU 가 있는 Windows 워크스테이션에서 `uv sync --group isaac` 후 사용.

### Python 패키지 `sim_to_real` (`src/sim_to_real/`)

`import sim_to_real` 한 번에 monkey patch + Gym 환경 등록이 모두 일어난다.

| 경로 | 내용 |
|---|---|
| `assets/scenes/pen_desk.py` | `PEN_DESK_CFG` (UsdFileCfg 래퍼) |
| `tasks/__init__.py` | `isaaclab_tasks.utils.import_packages` 로 하위 task config 자동 등록 (블랙리스트: `utils`, `.mdp`) |
| `tasks/pick_pen/` | `SimToReal-SO101-PickPen-v0` (entry point `pick_pen_env_cfg:PickPenEnvCfg`, env class `ManagerBasedRLEnv`), `mdp/{observations,terminations}.py` |
| `utils/{constant,domain_randomization}.py` | `randomize_object_in_ellipse` / `randomize_object_on_arc` 두 `EventTermCfg` wrapper — leisaac 사각형 분포로 불가능한 타원 면적 균등 / 호 1D sampling |

### USD 에셋 (`assets/`)

- `scenes/pen_desk/scene.usd` + `objects/<Pen*,PenCup>/<Name>.usd` — kitchen_with_orange 패턴 (객체별 self-contained USD + `prepend payload` 참조)
- `robots/` — SO-101 follower USD + 편집용 URDF
- **좌표 정합**: `SCENE_OFFSET` 상수로 top-level translate 일괄 시프트. 펜 4개 + 펜컵 collider 는 `PhysicsCollisionAPI` 직접 부여 (Cube proxy 미사용)
- **영역 분리**: 펜 = 그린 타원 (`y ∈ [0.22, 0.26]`), 펜컵 = 주황 호 (`y ∈ [0.34, 0.40]`). y 마진 ≥ 0.08 m 라 펜이 펜컵 안에 spawn 불가.

### 진입 스크립트 (`scripts/`)

| 스크립트 | 내용 |
|---|---|
| `environments/list_envs.py` | leisaac 등록 환경 일람 |
| `environments/teleoperation/teleop_se3_agent.py` | `gym.make("SimToReal-SO101-PickPen-v0")` + leisaac 디바이스 레이어 (`keyboard` / `gamepad` / `so101leader` / `so101leader` remote ZMQ / `bi-so101leader` / `lekiwi-*`) |
| `environments/teleoperation/replay.py` | 녹화 시퀀스 재실행 |
| `environments/teleoperation/so101_joint_state_server.py` | ZMQ PUB 으로 실제 SO-101 leader 상태를 원격 송출 (`SO101LeaderRemote` 카운터파트) |
| `environments/utils/{inspect_robot_materials,patch_robot_colors}.py` | USD 머티리얼 진단/패치 |

### 씬 재생성

USD 6개 (`scene.usd` + 펜 4개 + PenCup) 는 author 스크립트로 일괄 재생성. 좌표 변경 시 import 경로는 무관(`pyproject.toml` 의 `[tool.setuptools]` 가 `src/` 기준 패키지 탐색)하나, `pen_cup_center_xy` 같은 world-frame 상수는 `tasks/pick_pen/pick_pen_env_cfg.py` 와 오라클 state machine 양쪽을 같이 갱신해야 한다.

## Python 패키지 / 의존성

- **패키지 이름** `sim_to_real` (`pyproject.toml`). `[build-system] requires=["setuptools<82"]`, `[tool.setuptools.packages.find] where=["src"]` 로 `src/sim_to_real/` editable 설치.
- **공용 deps**: `h5py<3.16`, `hf-xet>=1.4.3`, `pyzmq>=27.1.0`, `lerobot[feetech]>=0.4.4`, `torch>=2.7`, `torchvision>=0.22`, `usd-core>=26.5` (순수 Python — isaac 그룹 없이도 씬 author/검증 가능하도록 공용).
- **`leisaac`** 는 `[tool.uv.sources]` 의 git tag `v0.4.0` 에서 설치 (vendored 사본 없음).

| 의존성 그룹 | 내용 |
|---|---|
| `teleop` | 실기기 (ffmpeg + evdev Linux 한정) |
| `policy` | lerobot[smolvla] + accelerate + num2words |
| `async` | grpcio + protobuf |
| `isaac` | isaacsim[all,extscache]==5.1.0 + leisaac[isaaclab,gr00t] |
| `dev` | ipykernel |

### ABI 호환성 핀

`pyproject.toml`. **임의 업그레이드 / `uv lock --upgrade` 금지.**

| 핀 | 이유 | 어기면 |
|---|---|---|
| `numpy==1.26.0` (override) | Isaac Sim 5.1.0 `isaacsim_kernel` 강제 | uv 설치 자체 실패 |
| `pyarrow<19` (override) | numpy 1.x C-API 호환 마지막 메이저. 19+ 는 numpy 2.x ABI 전용 | Isaac Sim 시작 후 ~30초 silent crash (`arrow.dll!...current_zone`) |
| `datasets>=4.0.0,<4.7.0` (override) | 4.7.0+ 가 `pa.json_()` 사용 (PyArrow 19+ 필요) — pyarrow<19 충돌 회피 | uv resolve 실패 |
| `h5py<3.16` | Isaac Sim 번들 HDF5 1.14.x ABI 일치. 3.16+ 는 HDF5 2.0 번들 | `Windows fatal exception: code 0xc0000139` |
| `torch==2.7.0+cu128` | Isaac Sim 5.1 번들 CUDA 12.8 일치 | 기동 시 CUDA 호출 실패 |
| `torchcodec>=0.5,<0.6` (override) | torch 2.7 호환 마지막 마이너. 0.10+ 는 PyTorch 2.11+ ABI | DataLoader worker 0 즉시 크래시, 학습 불가 |
| `packaging>=24.2,<26.0` (override) | 메타데이터 검증 충돌 회피 | `uv sync` resolve 실패 |
| `setuptools<82` (build-constraint) | 일부 의존성 `pkg_resources` 호환 | sdist 빌드 실패 |

`override-dependencies` 는 transitive 제약을 강제 무시한다. 예: `datasets 4.x` 가 `pyarrow>=21` 을 요구해도 override 로 `pyarrow<19` 설치 가능 — 검증된 워크플로(HDF5 → isaaclab2lerobot 변환)에서 런타임도 정상.

## 시뮬레이션 환경 제약 (GPU)

**RT 코어 없는 GPU(H100/A100)는 NVIDIA 가 Isaac Sim 5.1 공식 미지원.** 시스템 요구사항 문서가 *"GPUs without RT Cores (A100, H100) are not supported."* 라고 명시. 카메라 sensor 가 raytracing pipeline 생성 실패 → CUDA illegal memory access.

- **사용 가능**: NVIDIA 권장(RTX 4080+) 또는 RT 코어·16 GB VRAM 충족 GPU — RTX A4000/A5000/A6000, L40/L40S, RTX 6000 Ada, RTX PRO 5000/6000 Blackwell, GeForce RTX 40/50.
- Windows 워크스테이션(RTX A4000 16 GB)·Linux 서버(RTX PRO 5000 Blackwell 48 GB) 모두 조건 충족 — 시뮬·학습·추론 양쪽 사용 가능.

## 사용자 환경 컨벤션

- Windows USB 포워딩 = PowerShell (`usbipd bind/attach`). 컨테이너 내부 실행·호스트 보조 명령 = bash.
- HF/W&B 토큰은 `.env` 에서 읽음 (`.env.example` 템플릿).
- 표준 실행 패턴:
  - 실기기: `docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot <mode>`
  - 시뮬: `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickPen-v0 ...`

## 운영 규칙

### 에러 수정 후 docs/TROUBLESHOOTING.md 에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때 `docs/TROUBLESHOOTING.md` 에 항목 추가 (다음 세션·다른 작업자용).

- 양식: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌, USD/씬 물리 등)는 인접 섹션에 배치해 흐름 유지
- 필요 시 §핵심 의존성 표·§주의 로그 리스트도 함께 갱신
- 수정 실패한 경우도 README 에는 올리지 않음

### USD 씬 좌표·물리 상수 동기화

`assets/scenes/pen_desk/` USD 의 좌표·치수를 변경하면 다음도 같이 갱신:

- `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py::PEN_CUP_CENTER_XY` — `mdp.pen_in_cup` 종료 조건의 컵 기준점 (world frame)
- `SCENE_OFFSET` 일괄 시프트로 해결 가능하면 author 스크립트 1회 재실행으로 USD 6개 모두 갱신
