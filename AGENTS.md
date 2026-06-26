# AGENTS.md

SO-ARM101 6축 로봇 팔용 **실기기 LeRobot 파이프라인 + Isaac Lab Sim-to-Real 시뮬레이션**.

본 문서는 **README에 없는** 내부 구조·규칙·자주 쓰는 명령만 다룬다. 사용법은 아래 문서를 참조한다.

| 문서 | 내용 |
|---|---|
| `README.md` | 허브 |
| `docs/PATH_B_DOCKER.md` | Docker 실기기 경로 (LeRobot CLI teleop·record·replay) |
| `docs/PATH_GROOT_N17.md` | GR00T-N1.7 네이티브 정책 (별도 gr00t 이미지 finetune + ZMQ 서버 + gRPC↔ZMQ bridge → 기존 sim 폐루프) |
| `docs/SO101_POLICY_IO_CONTRACT.md` | SO-101 정책 입출력 규약 (feature codec 0-100·affine 그리퍼·절대 joint target) |
| `docs/LEISAAC_VLA_INFERENCE_COMPARISON.md` | LeIsaac 식 in-process VLA 추론 vs ROS2 폐루프 비교·검증 (demo_vla_inproc.sh) |
| `docs/NVIDIA_WORKSHOP_PICKCUBE_COMPARISON.md` | VLA 정책 비교·벤치마크 (ACT·SmolVLA·GR00T-N1.7) |
| `docs/NVIDIA_SO101_SIM2REAL_INDEX.md` | 시뮬-리얼 파이프라인 인덱스 + 이미지 준비 가이드 |
| `docs/TROUBLESHOOTING.md` | 트러블슈팅 |
| `docs/GRASP_PHYSICS.md` | SO-101 grasp 물리·충돌 튜닝 (큐브 collider convexHull, jaw/gripper SDF) |

## 실행 경로 — VLA 추론 중심

**VLA-only 아키텍처**: 3개 Docker 서비스로 분산된 추론 + 실기기 LeRobot CLI.

| | 실기기 경로 | VLA 시뮬 폐루프 |
|---|---|---|
| **실행 방식** | Docker (`docker/docker-compose.yaml run lerobot <mode>`) | Docker (`docker compose up` policy-server·isaac-sim·vla-ros) |
| **진입점** | LeRobot CLI (teleop·record·replay·calibrate·policy-client) | Isaac Lab `SimToReal-SO101-PickCube-v0` Gym env (ROS2 bridge via isaac-sim) |
| **정책 종류** | ACT · SmolVLA · **GR00T-N1.7** | 동일 (env=추론/데이터 substrate, RL 제거) |
| **스택** | LeRobot 0.4.4 (lerobot) / 0.5.1 (policy-server) | Isaac Sim 5.1 / IsaacLab 2.3.2 / leisaac 0.4.0 (git tag v0.4.0) |
| **서비스 3종** | `lerobot`(실기기 + 데이터) / `policy-server`(async gRPC) / `gr00t`(GR00T-N1.7 ZMQ) | `isaac-sim`(official nvcr 5.1, ROS2 bridge·WebRTC livestream) / `policy-server` / `vla-ros`(폐루프 노드) |

- 시뮬 env는 `src/sim_to_real/tasks/pick_cube/` 에서 `SimToReal-SO101-PickCube-v0` 만 등록 (RL 커리큘럼·보상기 제거, inference/teleop/데이터만 남음).
- 그리퍼 codec = **affine 전용** (feature [0,100] ↔ sim joint [-10°,100°]); offset 제거됨 (절대 joint target, `use_default_offset=False`). 단일 소스 = `src/so101_contract/feature_codec.py`.
- 데이터 생성 = TBD (현재는 실기기 LeRobot record / sim teleop 수동; cuRobo 제거됨).

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
| `lerobot` | `lerobot-so101:0.4.4` / `Dockerfile.lerobot` | Python 3.11 + LeRobot 0.4.4 (teleop deps) | 실기기: teleop / record / replay / calibrate / policy-client (async inference) |
| `policy-server` | `policy-server:0.5.1` / `Dockerfile.policy` | Python 3.12 + LeRobot 0.5.1 (policy+async deps) | async inference gRPC 서버 (ACT·SmolVLA·GR00T-N1.7 전부; groot bridge 별도) |
| `isaac-sim` | `nvcr.io/nvidia/isaac-sim:5.1.0` (공식, Dockerfile.isaac_sim 래퍼) | Ubuntu 22.04 + Isaac Sim 5.1 + ROS2 Jazzy + isaacsim.ros2_bridge | sim 폐루프: `SimToReal-SO101-PickCube-v0` 실행 + `/isaac_joint_states` PUB + `/isaac_joint_commands` SUB + WebRTC livestream :49100 |
| `vla-ros` | `so101-vla-ros:jazzy` / `Dockerfile.vla_ros` | ROS 2 Jazzy + vendored mini-lerobot | sim 폐루프 VLA 추론 노드 (`/vla_policy_action` publish) |
| `gr00t` | `gr00t-n17:ea` / `ref_repos/Isaac-GR00T/docker/Dockerfile`(무수정) | Python 3.10 + transformers 4.57 + gr00t | GR00T-N1.7 convert / finetune / ZMQ 추론 서버(:5555, policy-server-groot bridge 수신) |

- 빌드: `docker compose -f docker/docker-compose.yaml build <서비스>`. torch/CUDA 계층 일부만 BuildKit 캐시로 공유.
- 의존성 격리 이유: GR00T-N1.7 은 transformers 4.57/py3.10 으로 policy-server(5.3/3.12)와 공존 불가 → `gr00t` 별도 이미지(NVIDIA 네이티브, bind-mount+entrypoint override). 추론은 `policy-server-groot` gRPC↔ZMQ bridge 가 잇는다(`docs/PATH_GROOT_N17.md`).

### compose 설정

- **디바이스 마운트**: `${TELEOP_PORT}` `${ROBOT_PORT}` (직렬 암), `${TOP_CAM_PORT}` `${TOP_CAM_META_PORT}` `${WRIST_CAM_PORT}` `${WRIST_CAM_META_PORT}` `${FRONT_CAM_PORT}` `${FRONT_CAM_META_PORT}` (UVC 캡처/메타 노드 쌍, top/wrist/front 3개 기본 활성; `/-/dev/null` 폴백으로 미사용 시 안전 처리).
- **권한·네트워크**: `privileged: true` (udev/USB), `network_mode: host` (rerun 뷰어·ROS 브릿지), `ipc: host`. GPU 1장 예약.
- **호스트 볼륨**:
  - `./datasets`, `./logs`, `./outputs` → `/workspace/*` (두 서비스)
  - `./scripts` → `/workspace/scripts` (policy-server 만 — `policy-server-rtc` 모드가 `policy_server_rtc.py` 참조)
  - 명명 볼륨 `lerobot_hf_cache` (compose key `hf_cache`) → `/workspace/.cache/huggingface` (HF_HOME, 두 서비스 공유). non-root UID 실행 때문에 `/root` 가 아닌 `/workspace` 하위. 머신 이전 시 `docker run -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export.

### 진입점 모드

**`lerobot-entrypoint.sh`** (lerobot 서비스):
`teleop` · `record` · `replay` · `calibrate` · `setup-motors` · `find-port` · `find-cameras` · `find-joint-limits` · `dataset-viz` · `policy-client` · `edit-dataset` · `info` · `bash` · `python`

- `policy-client` 는 `lerobot.async_inference.robot_client` 로 정책 서버에 gRPC 접속해 SO-101 follower 구동. `async` 그룹(grpcio + protobuf)이 teleop 이미지에도 설치됨. 실제로는 `docker/policy-client-shim.py` 경유 (아래 참조).

**`policy-entrypoint.sh`** (policy-server 서비스, CMD 기본값 `policy-server`):
`policy-server` · `policy-server-rtc` · `policy-server-groot` · `policy-server-attn` · `info` · `bash` · `python`

> **RL 학습 삭제됨**: train/eval 모드 제거 (SmolVLA/ACT 학습은 호스트 uv 경로에서 별도 진행, 현재 스코프 아님). ✍ Policy-server는 추론 서버만 담당.

- `policy-server-rtc` 는 `policy_server_rtc.py` 로 서버 측 Real-Time Chunking(RTC) 가이던스를 주입한 async 서버 (gRPC 프로토콜·클라이언트 변경 없음, `RTC_*` env 튜닝).
- `policy-server-attn` 은 `policy_server_attention_bridge.py` (`AttentionBridgeServer`, PolicyServer 서브클래스)로 **SmolVLA 전용** cross-attention 시각화 브리지. gRPC 추론은 표준과 동일(`vla_policy_node` 무수정)하면서 매 추론마다 SmolVLA expert cross-attention(마지막 cross 레이어, head·action-step 평균)을 instance-level monkey-patch(`eager_attention_forward`·`embed_prefix`·`embed_image`)로 캡처해 카메라별 히트맵을 ZMQ PUB(`ATTN_ZMQ_*`, 기본 :5556). isaac-sim bridge(`scripts/inference/run_cube_desk_ros_bridge.py --attention_overlay`)가 SUB 해 top/wrist/front omni.ui 창에 오버레이(토글=표시만). SmolVLA 가 아니면 캡처 스킵 → groot/act 무영향. cam 매핑=입력순서(camera1=top/2=wrist/3=front).
- `policy-server-groot` 는 `scripts/inference/policy_server_groot_bridge.py` (`GrootBridgeServer`, PolicyServer 서브클래스)로 gRPC 컨트랙트를 유지한 채 추론만 `gr00t` 컨테이너의 ZMQ 서버(Gr00tPolicy N1.7)에 위임한다. `GROOT_ZMQ_*` env, `vla_policy_node` 무수정.

**`isaac-sim-entrypoint.sh`** (isaac-sim 서비스, CMD 기본값 `bridge`):
`bridge`(run_cube_desk_ros_bridge.py 래퍼) · `bash` · `python`

**`vla-ros-entrypoint.sh`** (vla-ros 서비스):
VLA 추론 노드 (`vla_policy_node`) 직접 실행, ROS 서비스 호출 불가.

**`gr00t-entrypoint.sh`** (gr00t 서비스, CMD 기본값 `zmq-server`):
`convert`(v3→v2.1+modality.json) · `finetune`(examples/finetune.sh) · `zmq-server`(run_gr00t_server.py) · `bash` · `python`

모드별 env var 매핑은 각 스크립트 상단 `${VAR:-default}` 블록과 case 분기 주석에 정리됨.

### 빌드·런타임 보조 스크립트

| 파일 | 용도 |
|---|---|
| `docker/groot_compat_patch.py` | `Dockerfile.policy` 빌드 시 `lerobot[smolvla,async]==0.5.1` 설치 직후 1회 실행. transformers 5.3 + torch 2.10 에서 LeRobot 0.5.1 GR00T wrapper 가 깨지는 4지점을 site-packages 에서 멱등 패치. 형태가 다르면 `RuntimeError` 로 빌드 중단(버전 트립와이어) — lerobot/transformers 업그레이드 시 이 패치부터 점검. |
| `docker/policy-client-shim.py` | `policy-client` 모드 실제 진입점. lerobot 0.4.4 `robot_client.py` 가 built-in robot config 모듈을 import 안 해 `--robot.type` 이 거부되는 회귀(huggingface/lerobot#3078)를 선행 import 로 보강. rerun viewer(`DISPLAY_DATA=true`) monkey patch 유효를 위해 `async_client()` 직접 호출(runpy 미사용). |
| `docker/lerobot_keyboard_stdin.py` + `.pth` | WSLg X 서버가 Windows Terminal 키 입력을 못 보는 환경에서 pynput 리스너 대신 `/dev/tty` + termios cbreak 리더로 `init_keyboard_listener` 대체. `.pth` 가 Python 시작 시 hook 설치 (lerobot 이미지 site-packages 에 COPY). |

### `.env` / 모델 프로필

- **주입 경로**: 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE}.env]` 가 컨테이너에 주입(나중 파일이 override). `entrypoint.sh` 가 기본값을 채워 `lerobot-*` CLI 인자로 매핑. `--env-file .env` 는 compose 보간용.
- **모델 프로필**: 모델 간 값이 다른 변수는 `env/<name>.env` 로 분리하고, `.env` 의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. 새 모델 = 프로필 파일 추가. 현재: `smolvla` · `groot_n17` · `act`.
  - 분리 변수: `POLICY_TYPE` / `TRAIN_POLICY_TYPE` / `POLICY_BASE_MODEL_PATH` / tokenizer·embodiment·chunk·n_action_steps / `ACTIONS_PER_CHUNK` / `POLICY_REPO_ID` / `JOB_NAME`
  - train 출발 모델 라우팅: `POLICY_BASE_MODEL_PATH` 단일 변수 + `TRAIN_POLICY_TYPE` 유무로 `--policy.path`(체크포인트) ↔ `--policy.type` + `--policy.base_model_path`(native 베이스).
  - **`groot_n17`(GR00T-N1.7)은 lerobot-train/policy-server 경로가 아니다**: `GROOT_*` 변수(`GROOT_BASE_MODEL`/`GROOT_CHECKPOINT`/`GROOT_MODALITY_CONFIG`/`GROOT_ZMQ_*`)로 `gr00t` 이미지(convert/finetune/zmq-server) + `policy-server-groot` bridge 를 구동한다. `RENAME_MAP` 비움(raw top/wrist/front). 상세 `docs/PATH_GROOT_N17.md`.

## 시뮬레이션 환경 — VLA 추론·데이터 기판

`SimToReal-SO101-PickCube-v0` Gym 환경은 **추론·데이터 기판**이다 (RL 제거됨). 온호스트 uv 실행(`uv sync --group isaac`) 또는 Docker isaac-sim 서비스 내에서 실행되며, ROS2 bridge를 통해 VLA 정책·데이터 기록을 지원한다.

### Python 패키지 `sim_to_real` (`src/sim_to_real/`)

`import sim_to_real` 한 번에 monkey patch + Gym 환경 등록이 모두 일어난다.

| 경로 | 내용 |
|---|---|
| `assets/scenes/cube_desk.py` | `CUBE_DESK_CFG` (UsdFileCfg 래퍼, pen_desk 삭제됨) |
| `tasks/__init__.py` | `isaaclab_tasks.utils.import_packages` 로 하위 task config 자동 등록 (블랙리스트: `utils`, `.mdp`) |
| `tasks/pick_cube/` | `SimToReal-SO101-PickCube-v0` (entry point `pick_cube_env_cfg:PickCubeEnvCfg`, env class `ManagerBasedRLEnv`). **RL 제거됨**: 커리큘럼 없음, 보상 없음 (PickCubeRewardsCfg=empty stub), obs/action/termination/events 만 유지. Action term = `SlewLimitedJointPositionActionCfg(use_default_offset=False)` 절대 joint target |
| `tasks/common/` | `utils.py` (수학·카메라 헬퍼) + `mdp/` (공용 obs/termination 컴포넌트) — pen/cube 도메인 중립 |
| `data/` | `lerobot_recorder.py` (LeRobot v3 writer 라이브러리) + `lerobot_units.py` (단위 변환, 단일 소스 = `src/so101_contract/feature_codec.py`) |
| `utils/{constant,domain_randomization,cube_specs,gripper_effort}.py` | `CUBE_NAMES`·`BOWL_NAME` + `randomize_object_in_ellipse` / `randomize_object_on_arc` DR 헬퍼, 큐브 크기/질량 단일 소스 (`cube_specs.py`), 그리퍼 effort clamp (env.step() 적용) |

### USD 에셋 (`assets/`)

- `scenes/cube_desk/scene.usd` + `objects/<Cube1~4,Bowl>/<Name>.usd` — kitchen_with_orange 패턴 (객체별 self-contained USD + `prepend payload` 참조). 펜 씬 삭제됨.
- 큐브 = **Cube1/2 40mm·Cube3/4 50mm** (단일 진실 소스 = `src/sim_to_real/utils/cube_specs.py`, 변경은 거기 한 곳만). 회색 펠트(라운드 visual + grasp 물리 mass 35/55g, contactOffset 0.004, solverPos 32, friction 1.8/1.5).
- 그릇 = 반구 곡면 벽(8밴드×24 panel) 동적 rigid body.
- `robots/` — SO-101 follower USD + 편집용 URDF.
- **충돌 근사**: grasp 관여 mesh 를 형상별로 — 오목(jaw/gripper, bowl)은 SDF/convexDecomposition, **볼록(큐브)은 convexHull**. jaw/gripper collider = `/so101_new_calib/{jaw,gripper}/collisions`(convexDecomposition→sdf, `scripts/assets/set_gripper_jaw_sdf_collision.py`). 팔 링크는 convexDecomposition 유지(grasp 무관·저비용). **큐브 collider는 convexHull**(2026-06-22 정정): SDF로 인한 jitter(~2.9 rad/s 회전 버즈)를 50배 감소(0.056 rad/s), grasp 성공 13/16=81% 달성. SDF는 오목 형상(bowl)에만 쓴다.
- **좌표 정합**: `SCENE_OFFSET` 상수로 top-level translate 일괄 시프트. 큐브 collider 는 self-contained USD 에 `PhysicsCollisionAPI`+convexHull 직접 부여 (별도 proxy 미사용).
- **영역 분리**: 조작 대상(큐브) = 그린 타원 (`y ∈ [0.22, 0.26]`), 컨테이너(그릇) = 주황 호 (`y ∈ [0.34, 0.40]`). y 마진 ≥ 0.08 m.

### 진입 스크립트 (`scripts/`)

| 카테고리 | 스크립트 | 내용 |
|---|---|---|
| **환경 관리** | `environments/list_envs.py` | leisaac 등록 환경 일람 |
| | `environments/author_pick_cube_scene.py` | **큐브** 씬 USD 6쌍(scene + 객체 5개) 일괄 author. 공식 pxr/PhysxSchema 스키마 API 사용. `OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python ...` 필요. 큐브 조명, 그릇 충돌(watertight + SDF) 직접 author. 펜 씬 삭제됨. |
| **에셋·충돌** | `assets/set_gripper_jaw_sdf_collision.py` | so101_follower.usd jaw/gripper collision을 convexDecomposition→**SDF** (usd-core raw, isaac 불요, backup 유지). GPU 불요. |
| | `assets/viz_collision_overlay.py` | visual vs collision(SDF/convexHull) 오버레이 PNG (matplotlib+trimesh, GPU 불요) |
| **teleop·데이터** | `environments/teleoperation/teleop_se3_agent.py` | PickCube 로컬 GUI teleop. `--task SimToReal-SO101-PickCube-v0`, `keyboard`/`so101leader`, `--tune_cameras` docking viewport, reset 시 부감 뷰. |
| | `environments/teleoperation/replay.py` | 녹화 시퀀스 재실행 |
| | `environments/teleoperation/so101_joint_state_server.py` | ZMQ PUB로 실제 SO-101 leader 상태 원격 송출 |
| | `environments/utils/{inspect_robot_materials,patch_robot_colors}.py` | USD 머티리얼 진단/패치 |
| **VLA 추론** | `inference/run_cube_desk_ros_bridge.py` (+ `.sh` wrapper) | Isaac Sim standalone + `isaacsim.ros2_bridge`로 cube_desk 실행. `/isaac_joint_states`, `/isaac_joint_commands`, `/clock`, `/cube_poses`, `/bowl_pose`(base_link) publish. **`--eval`** 모드 = closed-loop 평가. **`--attention_overlay`** = SmolVLA cross-attn 히트맵 ZMQ SUB → top/wrist/front omni.ui JET 오버레이. |
| | `inference/demo_vla.sh` | **VLA 라이브 데모 런처** — `start <act\|smolvla\|groot> [--ckpt\|--cubes\|--ip\|--gui\|--headless]` / `stop` / `status`. 정책 서버+bridge+vla-ros 자동 배선, 임시 env 생성·정리, livestream :49100. |
| | `inference/policy_server_groot_bridge.py` | `GrootBridgeServer` (PolicyServer 서브클래스). gRPC 컨트랙트 유지, 추론만 gr00t ZMQ에 위임. |
| **계약·검증** | `contract/validate_so101_io_contract.py` | SO-101 feature codec 정책 입출력 검증 (affine 그리퍼 [0,100]). |
| | `contract/replay_so101_policy_snapshot.py` | 정책 snapshot 재실행 (기록된 입력→정책 출력 비교) |
| | `contract/validate_lerobot_schema.py` | LeRobot v3 데이터셋 schema 검증 |
| **진단** | `diagnostics/compare_train_vs_rtc.py` | recorded ep를 학습·배포(RTC) 통과, SmolVLA 비교 |
| | `diagnostics/compare_train_vs_async_groot.py` | recorded ep를 학습·배포(gRPC bridge) 통과, GR00T-N1.7 비교 |
| | `diagnostics/replay_infer_overlay.py` | 추론 진단: 정책출력 오버레이+timestamp+per-joint MAE. policy-server 컨테이너서 실행. |
| | `diagnostics/plot_vla_pipeline_comparison.py` | VLA 파이프라인 비교 plot 생성 |
| **데이터** | `data/upload_to_huggingface.py` | LeRobot v3 dataset HF 업로드 + codebase_version 태그 자동 생성/이동. `.env` HF_TOKEN/HF_USER. |
| **기타** | `environments/follow_target_so101.py` | SO-101 position-only Lula IK 테스트 (5-DOF 선형 reaching) |
| | `environments/inspect_so101_gripper_frame.py` | gripper frame 진단 |
| | `ece_4560/` | 과정 프로젝트 (보유) |

### 씬 재생성

USD 6개 (`scene.usd` + 객체 5개) 는 author 스크립트로 일괄 재생성 (`author_pick_cube_scene.py`, 펜 삭제됨). 좌표 변경 시 `SCENE_OFFSET` 상수를 `assets/scenes/cube_desk.py`에서 갱신하고 스크립트 재실행하면 된다. `BOWL_CENTER_XY` 같은 world-frame 상수는 `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` 와 동기화해야 한다.

## Python 패키지 / 의존성

- **패키지 이름** `sim_to_real` (`pyproject.toml`). `[build-system] requires=["setuptools<82"]`, `[tool.setuptools.packages.find] where=["src"]` 로 `src/sim_to_real/` editable 설치.
- **공용 deps**: `h5py<3.16`, `hf-xet>=1.4.3`, `pyzmq>=27.1.0`, `lerobot[feetech]>=0.4.4`, `torch>=2.7`, `torchvision>=0.22`, `usd-core>=26.5` (순수 Python USD 작성·검증용 공용. 펜 author·USD 구조 검증은 usd-core 만으로 가능. 단 `author_pick_cube_scene.py` 는 PhysxSchema 정식 API 를 써 isaac 그룹(`uv run --group isaac`) 필요).
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
  - 시뮬 teleop: `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickCube-v0 ...`

## 운영 규칙

### 에러 수정 후 docs/TROUBLESHOOTING.md 에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때 `docs/TROUBLESHOOTING.md` 에 항목 추가 (다음 세션·다른 작업자용).

- 양식: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌, USD/씬 물리 등)는 인접 섹션에 배치해 흐름 유지
- 필요 시 §핵심 의존성 표·§주의 로그 리스트도 함께 갱신
- 수정 실패한 경우도 README 에는 올리지 않음

### 단위 및 그리퍼 codec 규약

**VLA 추론·데이터에서 통일된 계약**:

- **그리퍼 codec = affine only**: feature [0, 100] (정책 출력) ↔ sim joint [-10°, 100°] (환경). 공식: `deg = feature / 100 * 110 - 10`. 단일 소스 = `src/so101_contract/feature_codec.py`.
- **그리퍼 offset 제거**: `use_default_offset=False` (action term 설정). 모든 action = 절대 joint target (post-offset 스타일, 31.75 배수 제거됨). sim·real·bridge 공통 규약.
- 데이터 기록/재생: `src/sim_to_real/data/lerobot_units.py` 가 codec 참조, LeRobot v3 [0,100] ↔ sim [-10°,100°] 변환.

### 5-DOF IK 공통 원칙 (sim)

SO-101 은 팔 5축(+그리퍼)이라 임의 6-DOF pose(위치+방향 동시)를 만족 못 한다. **position 우선·orientation best-effort** 가 sim 규약:

- sim `follow_target_so101.py`: Lula IK·RMPFlow 모두 `target_orientation=None`(position-only). 새 IK 경로를 추가할 때도 orientation 을 hard constraint 로 넣지 말 것.
- MoveIt·cuMotion 제거됨 (PATH E 삭제).

### sim 진입 스크립트 AppLauncher 인자 필터

GUI 부팅하는 진입 스크립트는 `view_eye`/`view_lookat` 같은 **커스텀 인자**를 통째(`AppLauncher(vars(args))`)로 넘기면 Windows 에서 `_prepare_ui` access violation 이 난다. AppLauncher 가 실제 쓰는 키만 화이트리스트(`_LAUNCHER_KEYS`)로 필터해 전달하고, C-레벨 크래시 추적용 `faulthandler.enable(file=outputs/*.txt)` 을 부팅 전에 켠다. 적용 예: `follow_target_so101.py`, `run_cube_desk_ros_bridge.py`. ⚠ Linux 에선 access violation 대신 **livestream viewport docking 이 조용히 실패**(커스텀 인자가 UI 초기화 방해)하는 형태로도 나타난다 — 3-cam 레이아웃 미적용(2026-06-15 수정).
