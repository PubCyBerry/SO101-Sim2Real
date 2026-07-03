# AGENTS.md

SO-ARM101 6축 로봇 팔 **Sim-to-Real 파이프라인**. Isaac Sim 5.1 시뮬에서 VLA 정책을 학습·검증하고 실기기 SO-101 에 배포한다.

**2-머신 구조**: 실기기는 **Windows 워크스테이션(native uv)**, 시뮬·학습·추론은 **Linux 서버(Docker)**.

본 문서는 **README 에 없는** 내부 구조·규칙·자주 쓰는 명령만 다룬다.

| 문서 | 내용 |
|---|---|
| `README.md` | 허브 (설치·경로별 가이드·2-머신 개요) |
| `docs/TROUBLESHOOTING.md` | 트러블슈팅 (ABI·GPU·의존성 핀·USD/씬 물리) |

> 단위 codec·grasp 물리·GR00T 경로 등 세부는 별도 문서가 아니라 본 문서 본문에 인라인돼 있다.

## 실행 경로 — 2-머신

| | Windows 워크스테이션 | Linux 서버 |
|---|---|---|
| **실행 방식** | native uv + `pyproject.toml` (WSL·Docker 없음) | Docker (`docker compose ...`) |
| **진입점** | LeRobot CLI (`uv run lerobot-<mode>` · `robot_client`) | policy-server · isaac-sim · vla-ros |
| **작업** | teleop · record · replay · calibrate · setup-motors · find-port · **실기기 policy-client** | sim VLA 폐루프 · VLA 학습 · 추론 서버 · sim policy-client(vla-ros) |
| **정책** | ACT · SmolVLA · GR00T-N1.5 (서버에서 추론, gRPC) | 동일 (env=추론/데이터 기판, RL 제거) |
| **스택** | LeRobot 0.4.4 (pyproject `teleop`+`async`) | Isaac Sim 5.1 / IsaacLab 2.3.2 (직접 의존) / LeRobot 0.5.1 |

- 시뮬 env 는 **2층 상속 사다리**(leisaac Workshop 이식): base `so101_base_env_cfg.py`(`SO101TeleopEnvCfg` — 로봇+cube_desk USD+조명+slew 액션+6D joint obs+sim/physx+teleop-device 배선, 태스크 중립 substrate) → leaf `tasks/pick_cube/`(`PickCubeEnvCfg(SO101TeleopEnvCfg)` — 큐브/그릇/contact 센서/subtask obs/종료/DR). 등록 env 6개: `SimToReal-SO101-Teleop-v0`(base substrate) + `SimToReal-SO101-PickCube-{v0,DR-v0,DRBase-v0,Eval-v0,DR-Eval-v0}`. **DR-off 가 기본**(v0=고정 실측배치·순간 성공; `-DR`=큐브 scatter+arc+물리·시각 DR; `-Eval`=디바운스 성공 종료). datagen 은 `-DR`, closed-loop eval 은 `-Eval` 사용. RL 커리큘럼·보상기 제거(inference/teleop/데이터만). **큐브 배치 DR = 2모드**(pink IK sweep+isaac 물리검증으로 확정, `_make_randomize_cubes` 팩토리): **full**(`-DR`)=좌우대칭 **종모양**(grasp 물리검증 넓은쪽 대칭, `_CUBE_SCATTER_BELL`) · **base**(`-DRBase`)=nominal 주변 좁은 사각형(`_CUBE_BASE_*`). 양모드 공통: 로봇암 제외박스(`_CUBE_ARM_EXCLUDE`) + 그릇 겹침금지(min_bowl_sep) + base발치 제외(min_base_sep). 상세=`docs/PINK_IK_PICKPLACE.md`.
- 그리퍼 codec = **affine 전용** (feature [0,100] ↔ sim joint [-10°,100°]); offset 제거됨 (절대 joint target, `use_default_offset=False`). 단일 소스 = `src/so101_contract/feature_codec.py`.
- **데이터 생성**: ① 실기기 LeRobot `record`(Windows) + sim teleop(`teleop_se3_agent.py`, `--record_format lerobot_v3`; cross-machine = `so101leader_remote` 로 Windows leader→ZMQ→Linux sim). ② **State Machine datagen**(`scripts/datagen/record_state_machine.py`, isaac-sim `datagen` 모드) — SM 이 8D IK pose 생성 → IsaacLab DLS IK 풀이 → solved joint target(degree, joint-space)을 LeRobot v3 로 기록(VLA/real 호환). leisaac 에서 vendor(아래 `datagen/`). GPU isaac-sim 런타임 검증 진행 중(grasp waypoint·IK body_name·dof order). cuRobo batch 생성기는 제거됨.

## 환경 사양

| | Windows 워크스테이션 | Linux 학습 서버 |
|---|---|---|
| **OS** | Windows 11 Pro | Ubuntu 24.04.3 LTS (kernel 6.14.0-1014-oem), Dell Pro Max Tower T2 FCT2250, bare-metal |
| **CPU** | Intel Xeon W-2245 @ 3.90GHz (8C/16T, L3 16.5 MB) | Intel Core Ultra 5 245K (Arrow Lake, 14C/14T, L3 24 MB) |
| **RAM** | 64 GB | 128 GB DDR5 (+ swap 8 GB) |
| **Storage** | NVMe SSD 512 GB + SATA HDD 1 TB | NVMe SSD 477 GB (ext4 `/`) + SATA HDD 3.6 TB (`/DISK1`) |
| **GPU** | RTX A4000 16 GB (driver 596.36, CUDA 13.2, cc 8.6 Ampere) | RTX PRO 5000 Blackwell 48 GB GDDR7 (driver 580.95.05-open, cc 12.0 Blackwell) |
| **역할** | 실기기 SO-101 직결 (record/calibrate/policy-client). 실기기 CLI 는 GPU 불요 (추론은 서버 위임). Isaac Sim 로컬 실행도 RT 코어 충족 | 시뮬·학습·추론 (Docker) |

테스트 스위트·lint config 없음 (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 미정의). 변경 검증 = 컨테이너 빌드(`docker compose config`/`build`) + 실기기 실행 + `uv run` 시뮬 1회 실행.

## 실기기 native uv (Windows)

실기기 SO-101 제어는 WSL·Docker·usbipd 없이 Windows 호스트의 native uv 로 한다. (옛 Docker `lerobot` 서비스 + WSL ROS 스택은 제거됨.)

- **의존성**: pyproject `teleop`(ffmpeg + evdev[linux 전용] + packaging) + `async`(grpcio + protobuf) 그룹 = lerobot 0.4.4 CLI. `uv sync --group teleop --group async`.
- **`.env` 로드**: 자동 안 됨 → Git Bash `set -a; source .env; set +a` 후 `uv run lerobot-<mode>`.
- **포트**: COM 포트 직결 (`ROBOT_PORT`/`TELEOP_PORT`). usbipd 불필요.
- **CLI 모드 → 명령**:

| 모드 | 명령 | 핵심 인자 |
|---|---|---|
| find-port | `uv run lerobot-find-port` | — |
| setup-motors | `uv run lerobot-setup-motors` | `--robot.type=so101_follower --robot.port=$ROBOT_PORT` |
| calibrate | `uv run lerobot-calibrate` | `--robot.type/.port/.id` (또는 `--teleop.*`) |
| teleop | `uv run lerobot-teleoperate` | `--robot.* --teleop.* --robot.cameras=<json>` |
| record | `uv run lerobot-record` | `--robot.* --teleop.* --dataset.repo_id/.single_task/.num_episodes/.fps` |
| replay | `uv run lerobot-replay` | `--robot.* --dataset.repo_id --dataset.episode` |
| policy-client | `uv run python -m lerobot.async_inference.robot_client` | `--server_address=$POLICY_SERVER_ADDRESS --policy_type --task --actions_per_chunk --chunk_size_threshold --robot.*` |

- 변수 → CLI 매핑 패턴은 README §경로별 가이드 + `lerobot-<mode> --help`.
- **주의**: `evdev` 는 linux 전용(Windows 미설치, `sys_platform` 게이트). `--robot.type` 거부 시(huggingface/lerobot#3078) robot config 선import 또는 lerobot 0.4.5+.

## Docker 컨테이너 구조 (Linux 서버)

### 서비스 (4종)

| 서비스 | 이미지 / Dockerfile | 스택 | 역할 |
|---|---|---|---|
| `policy-server` | `policy-server:0.5.1` / `Dockerfile.policy` | Python 3.12 + LeRobot 0.5.1 (policy+async) | async inference gRPC 서버 + VLA 학습(train/eval). ACT·SmolVLA·GR00T-N1.5 (모두 lerobot 네이티브 `groot`) |
| `isaac-sim` | `nvcr.io/nvidia/isaac-sim:5.1.0` (공식, Dockerfile.isaac_sim 래퍼) | Ubuntu 22.04 + Isaac Sim 5.1 + ROS2 Jazzy + isaacsim.ros2_bridge | sim 폐루프: `SimToReal-SO101-PickCube-*`(eval=`-Eval-v0`) 실행 + `/isaac_joint_states` PUB + `/isaac_joint_commands` SUB + WebRTC livestream :49100 |
| `vla-ros` | `so101-vla-ros:jazzy` / `Dockerfile.vla_ros` | ROS 2 Jazzy + vendored mini-lerobot | sim 폐루프 VLA 추론 노드 (`vla_policy_node`, gRPC 클라이언트) |
| `pink-ik` | `so101-pink-ik:jazzy` / `Dockerfile.pink` | ROS 2 Jazzy + pin-pink(Pinocchio)·quadprog (CPU only) | **결정적 pick-place SM**(VLA 비경유). pink 미분 IK 로 큐브집기 궤적 생성 → bridge 직접 구동. `/isaac_joint_states`+`/tf` SUB → `/isaac_joint_commands` PUB. 상세=`docs/PINK_IK_PICKPLACE.md` |

- 빌드: `docker compose -f docker/docker-compose.yaml build <서비스>`. torch/CUDA 계층 일부만 BuildKit 캐시로 공유.
- GR00T-N1.5 는 LeRobot 0.5.1 내장 `groot` policy(Eagle-2.5 backbone)라 policy-server 안에서 직접 추론·학습한다(ACT/SmolVLA 와 동일, 별도 이미지·ZMQ 불필요).

### compose 설정

- **디바이스 마운트 없음**: 실기기 직렬/카메라 디바이스 마운트는 `lerobot` 서비스 삭제와 함께 제거됨. 남은 3개 서비스는 로봇 직결이 없다(isaac-sim 은 카메라를 sim 내부 렌더, 실기기 카메라는 Windows native uv OpenCV index).
- **권한·네트워크**: `network_mode: host` (ROS 브릿지·gRPC·WebRTC), `ipc: host`. GPU 1장 예약.
- **호스트 볼륨**:
  - `./datasets`, `./logs`, `./outputs` → `/workspace/*`
  - `./scripts` → `/workspace/scripts` (policy-server·isaac-sim — demo·inference 스크립트 참조)
  - `./src` → `/workspace/src:ro` (policy-server — `policy-server-affine` 모드의 `so101_contract` import용, read-only)
  - 명명 볼륨 `lerobot_hf_cache` (compose key `hf_cache`) → `/workspace/.cache/huggingface` (HF_HOME). **policy-server 전용**. non-root UID 실행 때문에 `/root` 가 아닌 `/workspace` 하위. 머신 이전 시 `docker run -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export.
  - `isaac_lab_cache_*` (kit/ov/pip/gl/compute/logs/data) → isaac-sim 전용.

### 진입점 모드

**`policy-entrypoint.sh`** (policy-server, CMD 기본값 `policy-server`):
`prepare-model` · `policy-server` · `policy-server-affine` · `train` · `eval` · `info` · `bash` · `python`

> **RL(PPO/강화학습) 제거됨**. `train`/`eval` 은 VLA 지도학습(SmolVLA/ACT) 용으로 유지 — policy-server 가 추론+학습 담당.

- **`policy-server-affine`** = stock policy-server + **real↔sim joint frame affine 어댑터**(`scripts/inference/policy_server_affine.py`, `AffineAdapterServer(PolicyServer)`). `JOINT_FRAME_MODE` ∈ `{sim-to-sim, real-to-real, sim-to-real, real-to-sim}`(학습데이터 도메인→추론 플랫폼)에 따라 `observation.state`(수신)·`action`(반환)을 변환(같은 도메인=passthrough, **이미지 무변환**). 정책 normalize 바깥 래핑 → 정규화 통계 불변, **양쪽 client(vla_policy_node·robot_client) 무변경**. cross-domain zero-shot 추론용. `so101_contract`(../src) 마운트 필요. 상세=`docs/SIM_REAL_REPLAY_CALIBRATION.md` §10.

- GR00T-N1.5 추론은 표준 `policy-server` 모드 그대로다 — `vla_policy_node` 가 `policy_type=groot` 로 SendPolicyInstructions 하면 policy-server 가 lerobot 네이티브 `GrootPolicy` 를 로드한다(별도 모드·ZMQ 불필요).
- `policy-server-rtc`(서버 측 Real-Time Chunking)는 백엔드 스크립트(`policy_server_rtc.py`)가 이 branch 에 없어 **entrypoint 에서 제거됨**. 재도입 시 스크립트 + entrypoint 모드를 함께 복원.

**`isaac-sim-entrypoint.sh`** (isaac-sim, CMD 기본값 `bridge`):
`bridge`(run_cube_desk_ros_bridge.py 래퍼) · `datagen`(record_state_machine.py — SM 데이터 생성, `DATAGEN_TASK`/`NUM_DEMOS`/`DATAGEN_EXTRA_ARGS`) · `teleop`(teleop_se3_agent.py — **cross-machine teleop + LeRobot v3 record**, `LEADER_ENDPOINT`(Windows leader ZMQ)/`DATASET_DIR`/`TASK_DESCRIPTION`/`NUM_DEMOS`/`SIM_TELEOP_EXTRA_ARGS`, livestream 관전+키보드 제어) · `bash` · `python`
- teleop 모드 deps: `Dockerfile.isaac_sim` 이 `pyzmq` 추가 설치(원격 leader SUB). `datasets` 볼륨(`../datasets:/workspace/datasets`)에 v3 출력 영속(datagen 도 공유).

**`vla-ros-entrypoint.sh`** (vla-ros):
컨테이너 안에서 `colcon build --packages-select so101_vla_policy` 후 `vla_policy_node` 직접 실행 (호스트 빌드 아님; `..:/workspace` bind-mount).

모드별 env var 매핑은 각 스크립트 상단 `${VAR:-default}` 블록과 case 분기 주석에 정리됨.

### 빌드·런타임 보조 스크립트

| 파일 | 용도 |
|---|---|
| `docker/groot_compat_patch.py` | `Dockerfile.policy` 빌드 시 `lerobot[smolvla,async]==0.5.1` 설치 직후 1회 실행. transformers 5.3 + torch 2.10 에서 LeRobot 0.5.1 GR00T-N1.5 네이티브 wrapper(`policies/groot`)가 깨지는 4지점을 site-packages 에서 멱등 패치. **N1.5 추론·학습 필수 — 삭제 금지.** 형태가 다르면 `RuntimeError` 로 빌드 중단(버전 트립와이어) — lerobot/transformers 업그레이드 시 이 패치부터 점검. |

### `.env` / 모델 프로필

- **주입 경로(Docker)**: 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE}.env]` 가 컨테이너에 주입(나중 파일이 override). `entrypoint.sh` 가 기본값을 채워 CLI 인자로 매핑. (native uv 에선 `source .env` 로 직접 로드.)
- **모델 프로필**: 모델 간 값이 다른 변수는 `env/<name>.env` 로 분리하고, `.env` 의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. 새 모델 = 프로필 파일 추가. 현재: `smolvla` · `groot_n15` · `act`.
  - 분리 변수: `POLICY_TYPE` / `TRAIN_POLICY_TYPE` / `POLICY_BASE_MODEL_PATH` / tokenizer·embodiment·chunk·n_action_steps / `ACTIONS_PER_CHUNK` / `POLICY_REPO_ID` / `JOB_NAME`
  - train 출발 모델 라우팅: `POLICY_BASE_MODEL_PATH` 단일 변수 + `TRAIN_POLICY_TYPE` 유무로 `--policy.path`(체크포인트) ↔ `--policy.type` + `--policy.base_model_path`(native 베이스).
  - **`groot_n15`(GR00T-N1.5)은 policy-server train/추론 경로 그대로**: `TRAIN_POLICY_TYPE=groot` + `POLICY_BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B` 로 `--policy.type=groot` 학습, 추론은 표준 `policy-server`(vla_policy_node 가 `policy_type=groot` 주입). `POLICY_CHUNK_SIZE`/`POLICY_N_ACTION_STEPS`=16, `RENAME_MAP` 비움(raw top/wrist/front).

## 시뮬레이션 환경 — VLA 추론·데이터 기판

`SimToReal-SO101-PickCube-*` Gym 환경은 **추론·데이터 기판**이다 (RL 제거됨). Docker isaac-sim 서비스(폐루프) 또는 온호스트 uv 실행(`uv sync --group isaac`, 수동 teleop·씬 author)에서 쓰이며, ROS2 bridge 를 통해 VLA 정책·데이터 기록을 지원한다. base substrate(`SimToReal-SO101-Teleop-v0`, 태스크 오브젝트/성공 없음)를 leaf 가 상속하는 2층 사다리(위 §실행 경로 참조).

### Python 패키지 `sim_to_real` (`src/sim_to_real/`)

`import sim_to_real` 한 번에 monkey patch + Gym 환경 등록이 모두 일어난다.

| 경로 | 내용 |
|---|---|
| `assets/scenes/cube_desk.py` | `CUBE_DESK_CFG` (UsdFileCfg 래퍼) |
| `tasks/__init__.py` | `isaaclab_tasks.utils.import_packages` 로 하위 task config 자동 등록 (블랙리스트: `utils`, `.mdp`) |
| `tasks/so101_base_env_cfg.py` | **base 층**(leisaac Workshop 이식). `SO101BaseSceneCfg`(로봇+cube_desk USD+조명, contact 리포트 on, FrameTransformer ee_frame 신규)·`SO101ActionsCfg`(slew joint)·`SO101PolicyObservationsCfg`(6D joint)·`SO101BaseEventCfg`(씬 리셋+포즈 jitter)·`SO101TeleopEnvCfg`(무종료 substrate + teleop-device 배선 `use_teleop_device`/`preprocess_device_action`). 신규 태스크는 이걸 상속 |
| `tasks/pick_cube/` | **leaf 층**. `PickCubeEnvCfg(SO101TeleopEnvCfg)` + 변형 3종(`PickCubeDREnvCfg`·`PickCubeEvalEnvCfg`·`PickCubeEvalDREnvCfg`). env class = `PickCubeEnv`(ManagerBasedRLEnv + 동적 gripper effort). base 씬에 큐브/그릇/contact 센서(`contact_jaw`+`contact_gripper`), subtask obs(`any_cube_grasped`+`object_in_container`), 종료(`task_done`/Eval=`task_done_confirmed`/`cube_lost`), DR 이벤트 추가. **RL 제거됨**(PickCubeRewardsCfg=empty stub). Action = `SlewLimitedJointPositionActionCfg(use_default_offset=False)` 절대 joint target. **신규**: per-joint actuator stiffness/damping(Workshop 값), ee_frame_state(privileged obs), image_raw(uint8 원본), VisualCfg로 static 카메라 3개(top/wrist/front)·--enable_cameras 필수 |
| `tasks/pick_cube/mdp/` | pick_cube 전용 MDP term. `observations.py:any_cube_grasped`(contact-sensor 양 손가락 envelope grasp 신호, env-state hysteresis, leisaac `any_vial_grasped` 이식) + `terminations.py:task_done_confirmed`(N-step 디바운스 성공)·`cube_lost` |
| `tasks/common/` | `utils.py` (수학·카메라 헬퍼) + `mdp/` (공용 obs/termination 컴포넌트) — 도메인 중립 |
| `data/` | `lerobot_recorder.py` (LeRobot v3 writer 라이브러리) + `lerobot_units.py` (단위 변환, 단일 소스 = `src/so101_contract/feature_codec.py`) |
| `utils/{constant,domain_randomization,cube_specs,gripper_effort,env_utils,general_assets,math_utils,monkey_patch}.py` | `CUBE_NAMES`·`BOWL_NAME` + DR 헬퍼, 큐브 크기/질량 단일 소스 (`cube_specs.py`), 그리퍼 effort clamp + vendored leisaac 유틸(`env_utils`·`general_assets`·`math_utils`·`monkey_patch`=IsaacLab 2.3.2 TerminationManager 버그 게이트 패치) |
| `datagen/` | **SM 데이터 생성 scaffold**(leisaac식): `state_machine/{base,pick_cube}.py`(StateMachineBase + PickCube SM) + `sm_actions.py`. 드라이버 = `scripts/datagen/record_state_machine.py` |
| `devices/` | **vendored leisaac teleop 스택**: `device_base`·`action_process`·`keyboard`(SO101Keyboard)·`gamepad`(SO101Gamepad)·`lerobot`(SO101Leader/Remote). lazy `__init__`(serial/lerobot 없이 import 가능). lekiwi/bi-arm 제외 |
| `assets/robots/lerobot.py` | `SO101_FOLLOWER_CFG` 단일 소스(leisaac vendor). `pick_cube_env_cfg` 가 `.replace()` 로 씬 특화만 override. limit/motor/rest 테이블은 `so101_contract.leader_calibration` 에서 가져옴(값 중복 0) |

> `tasks/pick_pen/`·`tasks/pick_cube_franka/` 는 미등록 잔재(env config 없음, RL 리팩토링 때 prune). 신규 task 추가 시 참고만.

### USD 에셋 (`assets/`)

- `scenes/cube_desk/scene.usd` + `objects/<Cube1~4,Bowl>/<Name>.usd` — kitchen_with_orange 패턴 (객체별 self-contained USD + `prepend payload` 참조).
- 큐브 = **Cube1/2 40mm·Cube3/4 50mm** (단일 진실 소스 = `src/sim_to_real/utils/cube_specs.py`, 변경은 거기 한 곳만). 회색 펠트(라운드 visual + grasp 물리 mass 35/55g, contactOffset 0.004, solverPos 32, friction 1.8/1.5).
- 그릇 = 반구 곡면 벽(8밴드×24 panel) 동적 rigid body.
- `robots/` — SO-101 follower USD + 편집용 URDF.
- **충돌 근사**: grasp 관여 mesh 를 형상별로 — 오목(jaw/gripper, bowl)은 SDF/convexDecomposition, **볼록(큐브)은 convexHull**. jaw/gripper collider = `/so101_new_calib/{jaw,gripper}/collisions`(convexDecomposition→sdf, `scripts/assets/set_gripper_jaw_sdf_collision.py`). 팔 링크는 convexDecomposition 유지. **큐브 collider는 convexHull**(2026-06-22 정정): SDF jitter(~2.9 rad/s 회전 버즈)를 50배 감소(0.056 rad/s), grasp 성공 13/16=81%. SDF는 오목 형상(bowl)에만.
- **좌표 정합**: `SCENE_OFFSET` 상수로 top-level translate 일괄 시프트. 큐브 collider 는 self-contained USD 에 `PhysicsCollisionAPI`+convexHull 직접 부여.
- **영역 분리**: 조작 대상(큐브) = 그린 타원 (`y ∈ [0.22, 0.26]`), 컨테이너(그릇) = 주황 호 (`y ∈ [0.34, 0.40]`). y 마진 ≥ 0.08 m.

### 진입 스크립트 (`scripts/`)

**배치 규약 (charter)** — 폴더 = 동사/단계 1개. "이 스크립트 어디?"는 *하는 일* 로 1:1 결정.

| 폴더 | 배치 규칙 (한 줄) |
|---|---|
| `assets/` | 기존 로봇 USD 물리 편집 (collision·joint limit) |
| `environments/` | sim 환경 구성·상호작용: 씬 author, env 조회, teleop, 머티리얼 |
| `datagen/` | **새 데이터 생성**: SM·실기기·pink 궤적 record + SM replay |
| `convert/` | 기존 데이터셋 **포맷·프레임 변환** |
| `data/` | 데이터셋 **사후 ops**(변환 아님): 병합·업로드 |
| `inference/` | **VLA 폐루프**(정책 경유) |
| `contract/` | I/O 계약·스키마 검증 |
| `real/` | Windows 실기기 native uv CLI |
| `ece_4560/` | 보관용 과정 프로젝트(격리, 파이프라인 무관) |

> 3중 모호 해소: **새 에피소드 만든다→datagen · 포맷/좌표 바꾼다→convert · 병합/업로드→data.**

실재 스크립트만 기재 (진단/inproc 등 제거된 항목은 표에 없음).

| 카테고리 | 스크립트 | 내용 |
|---|---|---|
| **환경 관리** | `environments/list_envs.py` | 등록 Gym 환경 일람 |
| | `environments/author_pick_cube_scene.py` | 큐브 씬 USD 6쌍(scene + 객체 5개) 일괄 author. 공식 pxr/PhysxSchema API. `OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python ...` 필요 |
| **에셋·충돌** | `assets/set_gripper_jaw_sdf_collision.py` | so101_follower.usd jaw/gripper collision → **SDF** (usd-core raw, isaac 불요, backup 유지) |
| | `assets/set_arm_joint_limits.py` | so101_follower.usd 팔 관절 position limit → `leader_calibration.SO101_FOLLOWER_USD_JOINT_LIMITS` 단일소스로 동기화(usd-core, 실기기 가동범위 매칭) |
| **teleop·데이터** | `environments/teleoperation/teleop_se3_agent.py` | PickCube GUI teleop + 데이터 기록. device=`keyboard`/`so101leader`/**`so101leader_remote`**(Windows leader→ZMQ `:5556`→Linux sim, `--leader_endpoint`), `--tune_cameras` docking viewport. **`--record --record_format lerobot_v3 --dataset_dir`** = LeRobot v3 기록(`record_state_machine` 와 동일 계약·기존 데이터셋 호환, --enable_cameras 미지정 시 `remove_pick_cube_cameras(env_cfg)` 로 static 카메라+images obs 제거(PickCubeSceneCfg 스폰 실패 회피). B=시작·N=성공저장·R=폐기). `hdf5` = 경량 action/state(카메라無). cross-machine 관전=`--public_ip` WebRTC livestream |
| | `environments/teleoperation/replay.py` | 녹화 시퀀스 재실행 |
| | `environments/teleoperation/so101_joint_state_server.py` | ZMQ PUB 로 실제 SO-101 leader 상태 원격 송출 |
| | `environments/utils/patch_robot_colors.py` | USD 머티리얼 패치 |
| **VLA 추론** | `inference/run_cube_desk_ros_bridge.py` (+ `.sh` wrapper) | Isaac Sim standalone + `isaacsim.ros2_bridge`로 cube_desk 실행. **`--task`** 옵션으로 env_cfg 동적 로드(gym.make 우회) — actuator per-joint stiffness/damping(env_cfg.scene.robot.actuators에서 단일소스), DR 배치(env_cfg.events.randomize_cubes params). GUI Perspective 카메라 eye=[0.632,0.755,1.317], lookat=[-0.269,-0.146,0.416]. `/isaac_joint_states`·`/isaac_joint_commands`·`/clock`·`/cube_poses`·`/bowl_pose` publish. **`--eval`** = closed-loop 평가. **`--grasp_sweep TRAJ.json`** = pink IK 궤적(gen-traj) 셀별 world teleport→물리 replay→잡은순간(gripper close) perspective/top/wrist/front 2x2 캡처(파일명=world 좌표). ROS 무경유(직접 apply_action), 그릇 park. |
| | `inference/replay_dataset_to_bridge.py` | LeRobot 데이터셋·npz·시퀀스JSON 1 에피소드를 bridge 로 replay(`/isaac_joint_commands`). `--arm_mapping {codec,calibration,follower}`(follower=실기기 녹화 replay) · `--sequence`(so101_gui 시퀀스 재현·move보간+hold) · `--ramp_in`(현재자세→첫frame, teleport 방지) · `--probe_tracking`. vla-ros 에서 실행 |
| | `inference/policy_server_affine.py` | `AffineAdapterServer(PolicyServer)` — `JOINT_FRAME_MODE`(4-case) 별 정책 I/O(state·action) real↔sim affine 변환, 이미지 무변환. `policy-server-affine` 모드가 실행. cross-domain zero-shot 추론 |
| | `inference/demo_vla.sh` | **VLA 라이브 데모 런처** — `start <act\|smolvla\|groot> [--ckpt\|--cubes\|--ip\|--gui\|--headless]` / `stop` / `status`. 정책 서버+vla-ros 자동 배선, livestream :49100 |
| **pink IK SM** | `datagen/pink_ik_bridge_node.py` | **결정적 pick-place SM**(pink 미분 IK, VLA 비경유). `/isaac_joint_states`(seed q_start)+`/tf`(cube/bowl) SUB → waypoint(hover_cube·descend·grasp·lift·over_bowl·release) 6개 각 1회 IK → smoothstep joint-space 보간 → `/isaac_joint_commands` PUB. **핵심**: URDF↔USD base z 90° 어긋남 `--base-yaw-deg 90` 보정 + grasp 방향/깊이/그리퍼는 실 trajectory(`ece_4560/.../pick_place_demo.json`) 역산. `pink-ik` 서비스가 실행. `--self-check`(offline). **`--record`** = 폐루프 궤적(state=`/isaac_joint_states`·action=publish command·3-cam)을 LeRobot v3 로 1 에피소드 기록(datagen 겸용). **`--sweep`** = 오프라인 kinematic: 큐브 (x,y) grid top-down grasp reachability 측정(고정 GRASP_ORIENT=face 정렬) → ASCII map+CSV. **`--gen-traj OUT.json`** = graspable 셀 dense 궤적(home→approach→hover→descend→grasp→lift) 출력 → bridge `--grasp_sweep`로 물리검증. `--ex-*`=로봇암 제외박스. 상세=`docs/PINK_IK_PICKPLACE.md` |
| **계약·검증** | `contract/validate_so101_io_contract.py` | SO-101 feature codec 정책 입출력 검증 (affine 그리퍼 [0,100]) |
| | `contract/replay_so101_policy_snapshot.py` | 정책 snapshot 재실행 (기록된 입력→정책 출력 비교) |
| | `contract/validate_lerobot_schema.py` | LeRobot v3 데이터셋 schema 검증 |
| **데이터 생성** | `datagen/record_state_machine.py` | SM 데이터 생성 드라이버. SM action → env step → LeRobot v3 writer 기록. isaac-sim `datagen` 모드가 실행 (`--task`·`--num_demos`·`--dataset_dir`) |
| | `datagen/replay_state_machine.py` | 기록된 SM 데모 재생 |
| | `datagen/record_real_sequence.py` | 실 follower joint 시퀀스(JSON)를 Gym env 에서 재생→LeRobot v3 기록. follower calibration 변환, action=실 follower 단위(실기기 `lerobot-replay` 호환). `--sequence`·`--self_check` |
| **데이터 변환** | `convert/isaaclab2lerobotv3.py` (+`_lerobot_features.py`) | Isaac Lab HDF5 → LeRobot v3 변환 (**host-only fallback**; in-container recorder 우선, end-to-end 미검증) |
| | `convert/sim_dataset_to_real_follower.py` | sim-프레임 LeRobot v3(arm=sim degree·gripper feature[0,100])를 실 follower 프레임으로 **in-place** 변환(`follower_calibration.policy_feature_to_real_follower`, `record_real_sequence` 의 역방향). 실기기 `lerobot-replay` 재생용, Isaac 무의존. `--self-check`. ⚠ 충돌 위험, e-stop 준비 |
| **데이터** | `data/append_sim_episode.py` | 실 follower replay 로 sim 에서 얻은 achieved 에피소드를 기존 LeRobot v3 데이터셋에 append(sim↔real replay 캘리브레이션 워크플로) |
| | `data/upload_to_huggingface.py` | LeRobot v3 dataset HF 업로드 + codebase_version 태그 자동 생성/이동. `.env` HF_TOKEN/HF_USER |
| **실기기(Windows)** | `real/lerobot.sh` | `.env`(+`POLICY_PROFILE` 프로필) 자동 로드 후 LeRobot CLI 를 변수→인자 매핑 실행(find-port·setup-motors·calibrate·teleop·record·replay·policy-client·env·raw). native uv 래퍼(Git Bash) |
| **기타** | `ece_4560/` | 과정 프로젝트 (보유) |

### 씬 재생성

USD 6개 (`scene.usd` + 객체 5개) 는 `author_pick_cube_scene.py` 로 일괄 재생성. 좌표 변경 시 `SCENE_OFFSET` 상수를 `assets/scenes/cube_desk.py` 에서 갱신하고 스크립트 재실행. `BOWL_CENTER_XY` 같은 world-frame 상수는 `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` 와 동기화.

## Python 패키지 / 의존성

- **패키지 이름** `sim_to_real` (`pyproject.toml`). `[build-system] requires=["setuptools<82"]`, `[tool.setuptools.packages.find] where=["src"]` 로 `src/sim_to_real/` editable 설치.
- **공용 deps**: `h5py<3.16`, `hf-xet>=1.4.3`, `pyzmq>=27.1.0`, `lerobot[feetech]>=0.4.4`, `torch>=2.7`, `torchvision>=0.22`, `usd-core>=26.5` (순수 Python USD 작성·검증 공용. 단 `author_pick_cube_scene.py` 는 PhysxSchema 정식 API 라 isaac 그룹 필요).
- **isaaclab** 은 직접 의존(`isaaclab[all,isaacsim]==2.3.2`, 외부 래퍼/leisaac 제거됨). PickCube SM IK 백엔드 = IsaacLab 내장 DLS(`DifferentialInverseKinematicsActionCfg`, `ik_method="dls"`). **leisaac 은 런타임 의존성이 아니다** — 유용한 코드(`devices`·`datagen`·`assets/robots`·`utils`)는 `src/sim_to_real/` 와 `src/so101_contract/leader_calibration.py` 로 vendor, leisaac 내부 import 0(IsaacLab/lerobot/sim_to_real 로 대체).

| 의존성 그룹 | 내용 | 사용처 |
|---|---|---|
| `teleop` | ffmpeg + evdev[linux] + packaging | 실기기 native uv (Windows) |
| `async` | grpcio + protobuf | 실기기 policy-client (Windows) |
| `policy` | lerobot[smolvla] + accelerate + num2words | (호스트 참조용; policy-server 는 Dockerfile.policy 가 독립 핀) |
| `isaac` | isaacsim[all,extscache]==5.1.0 + isaaclab[all,isaacsim]==2.3.2 | Linux host uv sim (teleop·author) |
| `dev` | ipykernel | — |

### ABI 호환성 핀

`pyproject.toml`. **임의 업그레이드 / `uv lock --upgrade` 금지.**

| 핀 | 이유 | 어기면 |
|---|---|---|
| `numpy==1.26.0` (override) | Isaac Sim 5.1.0 `isaacsim_kernel` 강제 | uv 설치 자체 실패 |
| `pyarrow<19` (override) | numpy 1.x C-API 호환 마지막 메이저. 19+ 는 numpy 2.x ABI 전용 | Isaac Sim 시작 후 ~30초 silent crash (`arrow.dll!...current_zone`) |
| `datasets>=4.0.0,<4.7.0` (override) | 4.7.0+ 가 `pa.json_()` 사용 (PyArrow 19+ 필요) | uv resolve 실패 |
| `h5py<3.16` | Isaac Sim 번들 HDF5 1.14.x ABI 일치. 3.16+ 는 HDF5 2.0 번들 | `Windows fatal exception: code 0xc0000139` |
| `torch==2.7.0+cu128` | Isaac Sim 5.1 번들 CUDA 12.8 일치 | 기동 시 CUDA 호출 실패 |
| `torchcodec>=0.5,<0.6` (override) | torch 2.7 호환 마지막 마이너. 0.10+ 는 PyTorch 2.11+ ABI | DataLoader worker 0 즉시 크래시, 학습 불가 |
| `packaging>=24.2,<26.0` (override) | 메타데이터 검증 충돌 회피 | `uv sync` resolve 실패 |
| `setuptools<82` (build-constraint) | 일부 의존성 `pkg_resources` 호환 | sdist 빌드 실패 |

`override-dependencies` 는 transitive 제약을 강제 무시한다. 예: `datasets 4.x` 가 `pyarrow>=21` 을 요구해도 override 로 `pyarrow<19` 설치 가능 — 검증된 워크플로에서 런타임도 정상.

## 시뮬레이션 환경 제약 (GPU)

**RT 코어 없는 GPU(H100/A100)는 NVIDIA 가 Isaac Sim 5.1 공식 미지원.** 시스템 요구사항 문서가 *"GPUs without RT Cores (A100, H100) are not supported."* 라고 명시. 카메라 sensor 가 raytracing pipeline 생성 실패 → CUDA illegal memory access.

- **사용 가능**: RTX A4000/A5000/A6000, L40/L40S, RTX 6000 Ada, RTX PRO 5000/6000 Blackwell, GeForce RTX 40/50.
- Windows 워크스테이션(RTX A4000 16 GB)·Linux 서버(RTX PRO 5000 Blackwell 48 GB) 모두 조건 충족.

## 사용자 환경 컨벤션

- 사용자에게 CLI 안내 시 Windows=Git Bash, Linux=Bash 기준.
- HF/W&B 토큰은 `.env` 에서 읽음 (`.env.example` 템플릿).
- 표준 실행 패턴:
  - 실기기(Windows): `set -a; source .env; set +a` 후 `uv run lerobot-<mode> ...`
  - 시뮬 폐루프(Linux): `docker compose --env-file .env -f docker/docker-compose.yaml up policy-server isaac-sim vla-ros`
  - 시뮬 teleop(Linux host uv): `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickCube-v0 ...`

## 운영 규칙

### 에러 수정 후 docs/TROUBLESHOOTING.md 에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때 `docs/TROUBLESHOOTING.md` 에 항목 추가 (다음 세션·다른 작업자용).

- 양식: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌, USD/씬 물리 등)는 인접 섹션에 배치
- 수정 실패한 경우도 README 에는 올리지 않음

### scratch/ — 임시물 관리

smoke test·일회성 검증·debug dump 가 레포 여기저기 흩어지지 않도록 **임시물은 전부 루트 `scratch/`** 에 둔다.

- `scratch/` 는 `.gitignore` 추적 제외(`scratch/README.md` 만 예외). 커밋 금지.
- 산출물 예: smoke test 스크립트, 디버그 로그, faulthandler 크래시 txt, 한 번 쓰고 버릴 plot/csv.
- 작업 종료 시 정리: 영구 가치 있으면 `scripts/<범주>/` 로 promote(+아래 스크립트 표 등재), 아니면 삭제.
- 하위 구조 자유 (예: `scratch/2026-06-26-<주제>/`).

### anti-fragmentation — 코드 파편화 방지

ad-hoc 작업으로 코드가 산발하지 않도록:

- **영구 스크립트는 반드시 `scripts/<범주>/`** 아래. 루트·임의 위치 금지. 범주: `assets`·`contract`·`data`·`environments`·`inference` (+신규 범주 추가 가능).
- **새 파일 만들기 전 기존 모듈·엔트리포인트 확장 우선.** 단일 소스 재사용: `src/so101_contract/feature_codec.py`(codec), `src/sim_to_real/utils/cube_specs.py`(큐브 스펙) 등.
- 한 작업 = 한 곳. 헬퍼 산발 금지. 탐색 코드는 `scratch/`, 끝나면 promote-or-delete.

### 단위 및 그리퍼 codec 규약

**VLA 추론·데이터 통일 계약**:

- **그리퍼 codec = affine only**: feature [0, 100] (정책 출력) ↔ sim joint [-10°, 100°] (환경). 공식: `deg = feature / 100 * 110 - 10`. 단일 소스 = `src/so101_contract/feature_codec.py`.
- **그리퍼 offset 제거**: `use_default_offset=False`. 모든 action = 절대 joint target (31.75 배수 제거). sim·real·bridge 공통.
- 데이터 기록/재생: `src/sim_to_real/data/lerobot_units.py` 가 codec 참조, LeRobot v3 [0,100] ↔ sim [-10°,100°] 변환.
- **실 leader ↔ sim 은 별도 계약**: `src/so101_contract/leader_calibration.py`. feature_codec 이 policy-feature ↔ sim radian 이라면, 이건 **실 leader 모터 정규화값 ↔ sim radian** 양방향. arm 은 leader [-100,100] → USD joint(관절별 비대칭) per-joint scale+offset remap(codec 의 arm 1:1 degree 로는 재현 불가), gripper affine 은 feature_codec 과 수식 동일. teleop·datagen 에서 사용.
- **실 follower ↔ sim 은 또 다른 계약**: `src/so101_contract/follower_calibration.py`. 실기기 녹화 데이터 replay 용 — 실 follower 영점/스케일이 sim URDF 영점과 어긋나(grasp 시 EE ~2.4cm 뜸) 생긴 **6축 per-joint affine** `sim_deg = A·real + B`. forward(`real_follower_to_sim_radians`)+inverse(`sim_radians_to_real_follower`) 양방향(affine 1개 역산), `fit_follower_affine`+self-check. `FOLLOWER_AFFINE_A/B` = device-specific 단일 소스(재캘리브레이션 시 이 둘만 갱신). replay `--arm_mapping follower`·`record_real_sequence.py` 사용. 상세=`docs/SIM_REAL_REPLAY_CALIBRATION.md`.

### 5-DOF IK 공통 원칙 (sim)

SO-101 은 팔 5축(+그리퍼)이라 임의 6-DOF pose 를 만족 못 한다. **position 우선·orientation best-effort**:

- 새 IK 경로 추가 시 orientation 을 hard constraint 로 넣지 말 것 (position-only).
- MoveIt·cuMotion·cuRobo·Lula·RMPFlow·follow-target IK 테스트 스크립트 제거됨.

### sim 진입 스크립트 AppLauncher 인자 필터

GUI 부팅 진입 스크립트는 `view_eye`/`view_lookat` 같은 **커스텀 인자**를 통째(`AppLauncher(vars(args))`)로 넘기면 Windows 에서 `_prepare_ui` access violation 이 난다. AppLauncher 가 실제 쓰는 키만 화이트리스트(`_LAUNCHER_KEYS`)로 필터해 전달하고, C-레벨 크래시 추적용 `faulthandler.enable(file=...)` 을 부팅 전에 켠다. 적용: `run_cube_desk_ros_bridge.py`. ⚠ Linux 에선 access violation 대신 **livestream viewport docking 이 조용히 실패**하는 형태로도 나타난다 (3-cam 레이아웃 미적용).
