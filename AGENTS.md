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
| **실행 방식** | native uv + `scripts/real/pyproject.toml` (WSL·Docker 없음) | Docker (`docker compose ...`) |
| **진입점** | LeRobot CLI (`uv run lerobot-<mode>` · `robot_client`) | policy-server · isaac-sim · vla-ros |
| **작업** | teleop · record · replay · calibrate · setup-motors · find-port · **실기기 policy-client** | sim VLA 폐루프 · VLA 학습 · 추론 서버 · sim policy-client(vla-ros) |
| **정책** | ACT · SmolVLA · GR00T-N1.7 (서버에서 추론, gRPC) | 동일 (env=추론/데이터 기판, RL 제거) |
| **스택** | LeRobot 0.6.0 (Python 3.12 전용 uv project) | Isaac Sim 5.1 / IsaacLab 2.3.2 / LeRobot 0.6.0 |

- 시뮬 env 는 **2층 상속 사다리**(leisaac Workshop 이식): base `so101_base_env_cfg.py`(`SO101TeleopEnvCfg` — 로봇+cube_desk USD+조명+slew 액션+6D joint obs+sim/physx+teleop-device 배선, 태스크 중립 substrate) → leaf `tasks/pick_cube/`(`PickCubeEnvCfg(SO101TeleopEnvCfg)` — 큐브/그릇/contact 센서/subtask obs/종료/DR). 등록 env 6개: `SimToReal-SO101-Teleop-v0`(base substrate) + `SimToReal-SO101-PickCube-{v0,DR-v0,DRBase-v0,Eval-v0,DR-Eval-v0}`. **DR-off 가 기본**(v0=고정 실측배치·순간 성공; `-DR`=큐브 scatter+arc+물리·시각 DR; `-Eval`=디바운스 성공 종료). datagen 은 `-DR`, closed-loop eval 은 `-Eval` 사용. RL 커리큘럼·보상기 제거(inference/teleop/데이터만). **큐브 배치 DR = 2모드**(pink IK sweep+isaac 물리검증으로 확정, `_make_randomize_cubes` 팩토리): **full**(`-DR`)=좌우대칭 **종모양**(grasp 물리검증 넓은쪽 대칭, `_CUBE_SCATTER_BELL`) · **base**(`-DRBase`)=nominal 주변 좁은 사각형(`_CUBE_BASE_*`). 양모드 공통: 로봇암 제외박스(`_CUBE_ARM_EXCLUDE`) + 그릇 겹침금지(min_bowl_sep) + base발치 제외(min_base_sep). 상세=`docs/PINK_IK_PICKPLACE.md`.
- 그리퍼 codec = **affine 전용** (feature [0,100] ↔ sim joint [-10°,100°]); offset 제거됨 (절대 joint target, `use_default_offset=False`). 단일 소스 = `src/so101_contract/feature_codec.py`.
- **데이터 생성**: ① 실기기 LeRobot `record`(Windows) + sim teleop(`teleop_se3_agent.py`, `--record_format lerobot_v3`; cross-machine = `so101leader_remote` 로 Windows leader→ZMQ→Linux sim). ② **State Machine datagen**(`scripts/datagen/record_state_machine.py`, isaac-sim `datagen` 모드) — SM 이 8D IK pose 생성 → IsaacLab DLS IK 풀이 → solved joint target(degree, joint-space)을 LeRobot v3 로 기록(VLA/real 호환). leisaac 에서 vendor(아래 `datagen/`). GPU isaac-sim 런타임 검증 진행 중(grasp waypoint·IK body_name·dof order). ③ **cuRobo pick-place SM**(`scripts/cuRobo/`, 2-proc ZMQ: `curobo_batch_planner.py` planner ↔ `pickplace_sm.py` executor) — collision-free 궤적 batch 생성 + DR 스폰영역 정량 sweep(현행 54-sphere·64-env·실패 셀 무재시도: yaw-zero 183/183=100%, yaw-random 1305/1305=100%). **`--record_hdf5`** = leisaac 방식 IsaacLab RecorderManager 녹화(multi-env env당 1 demo, 에피소드=정지2s→pick-place→init복귀→정지1s termination 자동 종료, 플래닝 대기 미기록) → `convert/isaaclab2lerobotv3.py` 로 LeRobot v3 사후 변환. **`--record_lerobot`** = LeRobot v3 직기록(leisaac `--use_lerobot_recorder` 동형 `SO101LeRobotRecorderManager`, single-env·성공만·CPU 스트리밍, 백엔드=기존 `LeRobotV3DatasetWriter`). 상세=`scripts/cuRobo/README.md`.

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

- **의존성**: `scripts/real/pyproject.toml`이 `lerobot[async,core_scripts,feetech]==0.6.0`을 고정한다. 최초 1회 `uv sync --project scripts/real`.
- **실행**: `scripts/real/lerobot.sh <mode>`가 루트 `.env`/model profile을 로드하고 항상 `uv run --project scripts/real`로 실행한다.
- **포트**: COM 포트 직결 (`ROBOT_PORT`/`TELEOP_PORT`). usbipd 불필요.
- **CLI 모드 → 명령**:

| 모드 | 명령 | 핵심 인자 |
|---|---|---|
| find-port | `scripts/real/lerobot.sh find-port` | — |
| setup-motors | `scripts/real/lerobot.sh setup-motors` | `ROBOT_PORT` |
| calibrate | `scripts/real/lerobot.sh calibrate` | `ROBOT_PORT`/`ROBOT_ID` 또는 leader 변수 |
| teleop | `scripts/real/lerobot.sh teleop` | follower/leader/camera를 `.env`에서 구성 |
| record | `scripts/real/lerobot.sh record` | dataset/task/episode/fps를 `.env`에서 구성 |
| replay | `scripts/real/lerobot.sh replay` | dataset repo와 episode 지정 |
| policy-client | `scripts/real/lerobot.sh policy-client` | manifest가 dispatch를 결정(preflight `--emit client_kind`). **4 mode 모두** `eef_robot_client.py`(EEF=FK/IK, joint=canonical joint 경계, stock client 아님) |

- 변수 → CLI 매핑 패턴은 README §경로별 가이드 + `lerobot-<mode> --help`.
- **주의**: 루트 `pyproject.toml`/`uv.lock`은 Isaac Sim 5.1의 Python 3.11·NumPy 1.26 호환 환경이다. Windows 실기기에서 이 환경을 재사용하지 않는다.

## Docker 컨테이너 구조 (Linux 서버)

### 서비스 (4종)

| 서비스 | 이미지 / Dockerfile | 스택 | 역할 |
|---|---|---|---|
| `policy-server` | `policy-server:0.6.0` / `Dockerfile.policy` | Python 3.12 + LeRobot 0.6.0 + EEF patch | async full-chunk gRPC 서버 + VLA 학습(train/eval). ACT·SmolVLA·GR00T-N1.7 |
| `isaac-sim` | `nvcr.io/nvidia/isaac-sim:5.1.0` (공식, Dockerfile.isaac_sim 래퍼) | Ubuntu 22.04 + Isaac Sim 5.1 + ROS2 Jazzy + isaacsim.ros2_bridge | sim 폐루프: `SimToReal-SO101-PickCube-*`(eval=`-Eval-v0`) 실행 + `/isaac_joint_states` PUB + `/isaac_joint_commands` SUB + WebRTC livestream :49100 |
| `vla-ros` | `so101-vla-ros:jazzy` / `Dockerfile.vla_ros` | ROS 2 Jazzy + vendored mini-lerobot | sim 폐루프 VLA 추론 노드 (`vla_policy_node`, gRPC 클라이언트) |
| `pink-ik` | `so101-pink-ik:jazzy` / `Dockerfile.pink` | ROS 2 Jazzy + pin-pink(Pinocchio)·quadprog (CPU only) | **결정적 pick-place SM**(VLA 비경유). pink 미분 IK 로 큐브집기 궤적 생성 → bridge 직접 구동. `/isaac_joint_states`+`/tf` SUB → `/isaac_joint_commands` PUB. 상세=`docs/PINK_IK_PICKPLACE.md` |

- 빌드: `docker compose -f docker/docker-compose.yaml build <서비스>`. torch/CUDA 계층 일부만 BuildKit 캐시로 공유.
- GR00T-N1.7은 LeRobot 0.6.0 네이티브 `groot` policy이며 ACT/SmolVLA와 같은 policy-server에서 학습·추론한다. N1.5 checkpoint는 v0.6.0에서 지원하지 않는다.

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

> **RL(PPO/강화학습) 제거됨**. `train`/`eval`은 ACT·SmolVLA·GR00T-N1.7 지도학습/평가용이며 policy-server가 추론+학습을 담당한다.

- **`policy-server-affine`** = stock policy-server + **real↔sim joint frame affine 어댑터**(`scripts/inference/policy_server_affine.py`, `AffineAdapterServer(PolicyServer)`). `JOINT_FRAME_MODE` ∈ `{sim-to-sim, real-to-real, sim-to-real, real-to-sim}`(학습데이터 도메인→추론 플랫폼)에 따라 `observation.state`(수신)·`action`(반환)을 변환(같은 도메인=passthrough, **이미지 무변환**). 정책 normalize 바깥 래핑 → 정규화 통계 불변, **양쪽 client(vla_policy_node·robot_client) 무변경**. cross-domain zero-shot 추론용. `so101_contract`(../src) 마운트 필요. 상세=`docs/SIM_REAL_REPLAY_CALIBRATION.md` §10.

- 세 EEF-relative policy 모두 표준 `policy-server` 모드를 사용한다. checkpoint processor가 full chunk를 absolute EEF로 복원하고, sim/real client가 IK 후 joint command를 생성한다.
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
| `docker/lerobot_v060_eef_relative_patch.py` | PyPI LeRobot 0.6.0 source에 공통 SE(3) processor, train/checkpoint manifest, full-chunk sync/async hook을 멱등 적용한다. 예상 upstream source가 다르면 build를 중단한다. |
| `docker/groot_compat_patch.py` | v0.5.1/N1.5 재현용 legacy 자료. **삭제 금지.** 현재 Dockerfile에서는 실행하지 않는다. |

### `.env` / 모델 프로필

- **주입 경로(Docker)**: 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE}.env]` 가 컨테이너에 주입(나중 파일이 override). `entrypoint.sh` 가 기본값을 채워 CLI 인자로 매핑. (native uv 에선 `source .env` 로 직접 로드.)
- **모델 프로필**: 모델 간 값이 다른 변수는 `env/<name>.env` 로 분리하고, `.env`의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. 현재: `act` · `smolvla` · `groot_n17`.
  - 분리 변수: `POLICY_TYPE` / `TRAIN_POLICY_TYPE` / `POLICY_BASE_MODEL_PATH` / tokenizer·embodiment·chunk·n_action_steps / `ACTIONS_PER_CHUNK` / `POLICY_REPO_ID` / `JOB_NAME`
  - train 출발 모델 라우팅: `POLICY_BASE_MODEL_PATH` 단일 변수 + `TRAIN_POLICY_TYPE` 유무로 `--policy.path`(체크포인트) ↔ `--policy.type` + `--policy.base_model_path`(native 베이스).
  - **`groot_n17`**: `TRAIN_POLICY_TYPE=groot` + `POLICY_BASE_MODEL_PATH=nvidia/GR00T-N1.7-3B`; model horizon 40, 실행 horizon 16, raw top/wrist/front key를 유지한다.

### EEF-relative action 계약

- LeRobot v3 영속 데이터는 `observation.state`와 `action` 모두 absolute EEF 10D(`[xyz, Rot6D rows, absolute gripper]`)다.
- 학습 preprocessor가 `T_rel = inv(T_state) @ T_action`을 chunk 전체에 적용하고, 추론 postprocessor가 같은 기준 state로 full chunk를 absolute EEF로 복원한다.
- checkpoint의 `action_representation.json`은 dataset/stats/processor/LeRobot/project/URDF/YAML hash를 담는다. 학습 재개·server·sim/real client 모두 누락이나 불일치를 fail-fast한다.
- EEF mode에서는 `policy-server-affine`을 사용하지 않는다. real/sim 차이는 공통 FK/IK platform adapter가 담당한다.
- EEF/Rot6D action을 elementwise 평균하지 않는다. async overlap은 IK 이후 joint queue에서 `latest_only`를 사용한다.
- **EEF IK로 산출한** 실기기 joint command는 `EEF_IK_REAL_VALIDATED=true`일 때만 허용한다. `false`는 FK/IK와 target/metric만 기록하는 motor-off dry-run이다. 이 gate는 **EEF-specific**이다 — joint-space fallback(`joint_absolute`/`joint_relative`)은 IK를 거치지 않아 이 gate의 대상이 아니고(그래서 즉시 선택 가능한 fallback이다), 대신 **일반 하드웨어 안전 절차**(작업자 입회·e-stop·감속·workspace 확인)로 별도 통제한다.
- 상세 계약과 단계별 acceptance criteria는 `docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`가 단일 기준이다.

**v2 확장(Phase 11–18)**: action representation 축을 4-mode enum(`joint_absolute`·`joint_relative`·`eef_absolute`·`eef_relative`) × EEF pose format 3종(`xyz_rot6d_rows` 10D·`xyz_quaternion_wxyz` 8D·`xyz_rpy` 7D) × 3 policy = **24 조합**으로 넓힌다. 위 v1 항목은 그중 `eef_relative + rot6d` 조합의 기준선이다.

- 모호한 `mode=absolute`는 신규 config에서 **금지**. joint mode의 `pose_format`은 `null`/`not_applicable`만 허용한다. 단일 소스 = `src/so101_contract/action_representation.py`(`ActionRepresentationSpec`), v1 `eef_action_contract.ActionRepresentationConfig`는 legacy shim(`to_spec()`로만 승격).
- dataset은 **모든 mode에서 absolute 저장**. relative는 training processor가 만들고 postprocessor가 되돌린다(별도 relative 영속 포맷 없음).
- 모든 relative 변환은 rotation matrix/SE(3) 경유. Rot6D/quaternion/RPY vector를 직접 빼거나 더하지 않는다. 단일 소스 = `src/so101_contract/pose_codec.py`.
- joint-relative는 단순 subtraction이 아니라 **topology-aware** difference/add다(revolute wrap·continuous 주기성·prismatic 선형). joint dim은 5/6/7 하드코딩 금지, feature metadata에서 resolve. 단일 소스 = `src/so101_contract/joint_topology.py`.
- 4-mode encode/decode 단일 소스 = `src/so101_contract/action_transform.py`. stats 생성기와 processor가 같은 구현을 공유한다. absolute mode는 canonical 정규화만 하고 **state cache를 만들지 않으며**(항상 `None`), relative mode만 chunk 기준 state를 cache한다. `eef_relative` decode 결과도 encode와 같은 canonical form(quaternion 연속성·RPY wrap·Rot6D 직교)을 만족한다.
- `topology_aware_add(q_state, Δq)`는 기본값으로 canonical absolute target을 반환한다(periodic joint wrap 포함). `canonicalize=False`는 디버그 전용.
- stats profile은 mode/pose format/horizon/dataset fingerprint별로 분리되며 absolute·relative가 한 artifact(`meta/action_representation_stats.json`)에 공존한다. quaternion 부호·연속성과 RPY wrap은 stats 계산 **전에** 적용된다. 단일 소스 = `src/so101_contract/action_representation_stats.py`.
- 신규 checkpoint는 mode와 무관하게 `action_representation.json`(**schema_version=2**)을 포함한다. 누락 시 추정하지 않고 fail-fast하며 `--allow-legacy-joint-absolute-checkpoint` opt-in 사실도 manifest에 기록한다. 단일 소스 = `src/so101_contract/action_manifest.py`.
- 추론 CLI의 representation 인자는 override가 아니라 **assertion**이다(불일치 시 시작 거부).
- **CLI**: `--policy.action_representation.mode={joint_absolute|joint_relative|eef_absolute|eef_relative}` + EEF에서 `--policy.action_representation.pose_format={xyz_rot6d_rows|xyz_quaternion_wxyz|xyz_rpy}`. entrypoint env = `ACTION_REPRESENTATION_MODE`/`ACTION_REPRESENTATION_POSE_FORMAT`/`ACTION_REPRESENTATION_STATS_FILE`. 3개 profile은 `eef_relative + xyz_rot6d_rows`.
- **엄격성**: 대상 policy(ACT·SmolVLA·GR00T)는 mode와 무관하게 dataset v2 계약 + stats profile + v2 processor + manifest를 **필수**로 갖는다. 없으면 fail-fast(추정 금지). 대상이 아닌 policy는 `joint_absolute` 기본값에서만 stock 경로 유지.
- **canonical joint 단위**: v2 joint feature = arm radian(5) + gripper policy feature [0,100]. platform canonical = 6D sim radian. real 경계 변환 1회(`follower_calibration`), sim은 codec 변환 없음. legacy `to_lerobot_units`/`from_lerobot_units`는 v2 joint 경로에서 금지. manifest `transform`에 arm 단위(`joint_topology`)와 **명시적 `joint_feature_contract`**(version·arm_unit·gripper_semantics·arm_dof·gripper_index)가 함께 실리고 fingerprint/manifest hash에 포함된다. 누락·degree·불일치는 명령 전 실패(합성 금지). startup 로그는 mode-aware(EEF=kinematics version, joint=not_required). 단일 소스 = `src/so101_contract/joint_feature_codec.py`.
- **추론 startup**: `src/so101_contract/inference_startup.py`가 server/sim/real 공통 계획을 만든다(계약 resolve → 선택적 assertion → EEF kinematics hash 검증 → manifest 기반 feature schema → router). 추론 representation env는 **선택적**이며 비어 있으면 manifest에서 유도한다(학습 기본값은 `TRAIN_ACTION_REPRESENTATION_*`). v1 deployment validator는 runtime에서 호출하지 않는다.
- **추론 routing**(§22.4): router 입력은 **이미 postprocess된 absolute chunk**다(relative decode 2회 금지). `joint_*`는 IK 호출 0회, `eef_*`는 platform adapter IK 정확히 1회. 단일 소스 = `src/so101_contract/action_routing.py`, checkpoint 계약 loader = `action_checkpoint_contract.py`(local dir + HF revision 동일 API).
- **legacy migration**: `action_migration.py` + `scripts/convert/migrate_action_representation_checkpoint.py`. 원본 byte 불변, 출력은 별도 디렉터리, manifest에 opt-in flag·원본 tree hash·migration version 기록. **legacy opt-in은 env profile 기본값이 될 수 없다.**
- **현재 상태**: Phase 11–17 완료. LeRobot patcher/factory가 v2로 전환됐고(11 파일), 24 조합이 실제 policy class로 검증됐다. **v1 manifest checkpoint는 이제 load fail-fast** — 승격은 Phase 16 migration/명시적 opt-in 전용. Phase 16(migration/routing/CLI assertion) 완료 — 단 **실기기 rollout과 실제 Hub 업로드는 미실행**(Phase 18/운영). Phase 17(24-combination offline matrix) 완료 — 24/24 조합 × 13 필수 check = **312/312 PASS**(`scripts/contract/validate_action_representation_matrix.py`, artifact `scratch/p17-matrix/phase17_24combo.json`). Phase 18은 **진행 중** — 전 조합 contract-level dry-run 24/24 PASS(`validate_action_representation_rollout_dry_run.py`, artifact `scratch/p18-dry-run/phase18_24combo_dry_run.json`)이나 **대표 조합 sim closed-loop은 NOT_RUN**(학습된 EEF checkpoint 없음), **real guarded rollout은 BLOCKED_EXTERNAL**(실기기 승인·e-stop gate 없음). `phase18_complete=false` 유지.

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
| | `inference/eef_robot_client.py` | LeRobot 0.6 `RobotClient` 확장. 실 follower observation을 canonical EEF로 FK하고 returned EEF full chunk를 bounded sequential IK로 joint target으로 변환. checkpoint manifest/kinematics hash 검증, motor-off dry-run, JSONL metric 포함 |
| | `inference/evaluate_eef_open_loop.py` | 학습 checkpoint를 recorded EEF dataset에 full-chunk로 적용해 translation/rotation/gripper 오차와 trajectory overlay JSON 생성 |
| | `inference/assert_checkpoint_representation.py` | **v2 Phase 16** checkpoint 계약 startup assertion. policy-entrypoint `policy-server` 모드가 `CHECKPOINT_PATH` 있을 때 호출. representation 인자는 override가 아니라 assertion |
| | `inference/demo_vla.sh` | **VLA 라이브 데모 런처** — `start <act\|smolvla\|groot> [--ckpt\|--cubes\|--ip\|--gui\|--headless]` / `stop` / `status`. 정책 서버+vla-ros 자동 배선, livestream :49100 |
| **pink IK SM** | `datagen/pink_ik_bridge_node.py` | **결정적 pick-place SM**(pink 미분 IK, VLA 비경유, 2026-07-03 motion-planning 설계 구현). `/isaac_joint_states`+`/tf`(cube xyz·yaw, bowl) SUB → **(pan,±α,ρ) 파라미터화 grasp 선택**(pan 수직평면 안 pose 만 생성=IK 항상 풀림; α 양방향 스캔 후 grasp+hover+pre corridor err score 로 최적 채택 — +α=원거리 2R 해소·-α=근거리 최소반경 해소, top-down 선호 tie-break; ρ=-Δψ/cosα 큐브 yaw face 정렬, \|Δψ·tanα\|≤τ_max; grasp z 하한=책상+jaw_floor_clear+jaw_tip_drop) → SM(pan_align→pre_grasp→**Cartesian 직선 approach**(tool축, step IK seed 연쇄)→grasp→수직 lift(H_SAFE)→등고 transit(그릇 중심서 base 쪽 --bowl-pull 당김)→release(0.5×leg)→home(REST 자세); drop·retreat 없음) → 인접점 lerp+smoothstep 재생 → `/isaac_joint_commands` PUB. IK 타겟 = **정적 TCP OP_FRAME `tcp_grasp`**(검증 grasp 역산 EE-local 오프셋, `--tcp-d*`). 그릇 회피 = 절두 원뿔 keep-out(`cone_radius(z)`, BOWL_R 실측) 스윕 체크. **핵심**: URDF↔USD base z 90° 어긋남 `--base-yaw-deg 90` 보정. `pink-ik` 서비스가 실행. 게이트는 grasp 점만(pre-grasp via 는 top-down↔도달반경 상충으로 err 큰 게 정상 — pre 까지 게이트하면 α 가 누워 양단 lateral miss). pan_align=IK 해의 shoulder_pan(URDF pan 부호가 azimuth 와 반대). `--self-check`(offline: yaw-comp 기하+plan 7케이스 — DR bell 양단·min_bowl_sep 근접·근거리·원거리). **`--record`** = 폐루프 궤적(state=`/isaac_joint_states`·action=publish command·3-cam)을 LeRobot v3 로 1 에피소드 기록(datagen 겸용). **`--sweep`** = 오프라인 kinematic: 큐브 (x,y) grid top-down grasp reachability 측정(고정 GRASP_ORIENT=face 정렬) → ASCII map+CSV. **`--gen-traj OUT.json`** = graspable 셀 dense 궤적(home→approach→hover→descend→grasp→lift) 출력 → bridge `--grasp_sweep`로 물리검증. `--ex-*`=로봇암 제외박스. 상세=`docs/PINK_IK_PICKPLACE.md` |
| **계약·검증** | `contract/validate_so101_io_contract.py` | SO-101 feature codec 정책 입출력 검증 (affine 그리퍼 [0,100]) |
| | `contract/replay_so101_policy_snapshot.py` | 정책 snapshot 재실행 (기록된 입력→정책 출력 비교) |
| | `contract/validate_lerobot_schema.py` | LeRobot v3 데이터셋 schema 검증 |
| | `contract/validate_action_representation_v2.py` | **v2** 4-mode enum(joint/eef × absolute/relative)·pose_format 규칙·schema v2 universal manifest(hash/tamper/CLI assertion/legacy opt-in)·24-combination matrix 검증 |
| | `contract/validate_pose_codec.py` | **v2** format-neutral pose codec — rot6d/quaternion wxyz/RPY encode-decode, quaternion 부호·연속성, RPY wrap·gimbal, float32/64·torch parity |
| | `contract/validate_joint_representation.py` | **v2** joint topology contract(revolute/continuous/prismatic·period·limits) + topology-aware difference/add. wrap이 단순 subtraction과 다름·canonical 복원·dimension metadata resolve 검증 |
| | `contract/validate_action_dataset_contract.py` | **v2** dataset 계약 — 3 pose format feature names/group/frame/format/dim 엄격 resolve, joint topology 선언, converter 상수 대조, fingerprint 무효화 |
| | `contract/validate_action_representation_stats.py` | **v2** 8 representation stats — 한 artifact 공존, mode/format/horizon/dataset cache invalidation, 정규화 선행, checkpoint 복원·tamper |
| | `contract/validate_action_representation_processor.py` | **v2** LeRobot encode/decode step — 4-mode round-trip, absolute canonicalize-only, full-chunk 단일 기준 state, 직렬화·relink, v1 rot6d parity (LeRobot 필요) |
| | `contract/validate_action_representation_policies.py` | **v2 Phase 15** 3 policy × 8 representation = **24 조합**. 실제 ACT/SmolVLA/GR00T class로 processor ordering·capacity(32/132)·encode 계약·1-batch forward/backward·full-chunk·manifest 검증 |
| | `contract/validate_action_representation_checkpoint_cli.py` | **v2 Phase 15** 실제 `lerobot-train` CLI checkpoint의 periodic/final manifest + reload + processor fingerprint 검증 |
| | `contract/validate_action_representation_matrix.py` | **v2 Phase 17** 24 조합 × §25.2 **13개 필수 check** 통합 runner. 조합마다 config/manifest/dataset/stats/1-batch/reload/full-chunk/**실제 SyncInferenceEngine+PolicyServer async**/processor ordering/routing/FK-IK(joint=command path)/migration/fail-fast를 새로 실행하고 machine-readable JSON(`scratch/p17-matrix/phase17_24combo.json`)으로 남긴다. 대표 조합 결과 복제 금지. weight는 SmolVLA/GR00T만 family당 1회 저장 후 조합 디렉터리에 hardlink dedup(ACT는 dimension 종속이라 조합별 실제 save/load). Docker `policy-server:0.6.0` 안에서 실행 |
| | `contract/validate_action_representation_rollout_dry_run.py` | **v2 Phase 18(부분)** 24 조합 contract-level rollout dry-run. 조합마다 checkpoint 계약 resolve→실제 router→sim boundary(EEF IK 1회/joint 0회)→`real_dry_run` motor publish 0(sink counter)→실제 `ActionChunkQueue` latest_only/overlap/stale/empty/refill→NaN·차원·rank·도달불가 chunk의 publish 이전 거부→실제 `evaluate_eef_rollout_metrics.py --mode real-dry-run` gate. Phase 17 artifact를 재검증+SHA256 기록, evaluator acceptance-gate self-test 7케이스 포함. **GPU·정책 weight 불요(CPU)**. `status=DRY_RUN_PASS`이며 `phase18_complete=false` **고정** — **non-promoting** runner라 외부 sim/real report가 둘 다 `REPORT_VERIFIED`여도 승격하지 않고, Phase 18 승격은 별도 closure 절차다. `aborts`/`invalid_chunks`/`starvation`/`empty`/`stale` 0은 장시간 측정치가 아니라 정상 경로에서 구조적으로 0이며 fail-closed 동작만 검증한다(실제 rate acceptance는 외부 evaluator report). `phase17_artifact`는 historical 입력의 형식/SHA 검증이고 현재 실행 provenance는 top-level에 따로 있다. artifact `scratch/p18-dry-run/phase18_24combo_dry_run.json` |
| | `contract/validate_action_migration.py` | **v2 Phase 16** legacy migration — manifest-less(opt-in flag 필수)·v1 eef_relative dispatch, 원본 byte 불변, migrated checkpoint reload, local vs offline Hub snapshot parity |
| | `contract/validate_action_routing.py` | **v2 Phase 16** 4-mode routing — 8 representation이 joint command 도달, joint IK 0회/EEF IK 1회, 2차 decode 없음, gripper/horizon/dtype 보존, publish 이전 거부, 실제 FK/IK·follower calibration round-trip |
| | `contract/validate_canonical_joint_units.py` | **v2 Phase 16** canonical joint 단위 경계 — joint observation이 arm radian(degree 아님), sim publish=router canonical(2차 변환 없음), real 경계 1회, pose format 중립 EEF residual, degree/누락 단위 계약 거부 |
| | `contract/validate_representation_cli_assertions.py` | **v2 Phase 16** server/sim/real assertion 배선 — entrypoint hook 순서, 불일치 시 기동 중단, legacy opt-in이 기본값 아님 |
| | `contract/create_action_representation_fixture.py` | **v2** joint/EEF(3 format) absolute fixture 생성 + `so101_action_representation` metadata/topology + 해당 dataset의 전 stats profile |
| | `contract/validate_eef_relative_contract.py` | SE(3) encode/decode, Rot6D, absolute/relative chunk 기준, Isaac-GR00T parity 검증 |
| | `contract/validate_lerobot_eef_processor.py` | LeRobot processor 순서·직렬화·stats·pair relink 검증 |
| | `contract/validate_eef_full_chunk.py` | sync/async full-chunk postprocess 호출·queue 기준 state 회귀 검증 |
| | `contract/validate_eef_checkpoint_manifest.py` | self-contained manifest 저장/복원·tamper/policy/kinematics fail-fast 검증 |
| | `contract/validate_eef_platform_adapter.py` | real/sim FK, bounded IK, command gate, dry-run 검증 |
| | `contract/validate_eef_ik_workspace.py` | DR workspace 32 chunk×8 horizon FK→IK sweep와 residual/iteration 집계 |
| | `contract/evaluate_eef_rollout_metrics.py` | sim eval JSON+runtime JSONL 또는 real dry-run/rollout JSONL을 DoD threshold로 판정 |
| | `contract/validate_{act,smolvla,groot_n17}_eef_training.py` | 각 실제 policy의 EEF 1-batch forward/backward smoke |
| **데이터 생성** | `datagen/record_state_machine.py` | SM 데이터 생성 드라이버. SM action → env step → LeRobot v3 writer 기록. isaac-sim `datagen` 모드가 실행 (`--task`·`--num_demos`·`--dataset_dir`) |
| | `datagen/replay_state_machine.py` | 기록된 SM 데모 재생 |
| | `datagen/record_real_sequence.py` | 실 follower joint 시퀀스(JSON)를 Gym env 에서 재생→LeRobot v3 기록. follower calibration 변환, action=실 follower 단위(실기기 `lerobot-replay` 호환). `--sequence`·`--self_check` |
| | `convert/migrate_action_representation_checkpoint.py` | **v2 Phase 16** legacy checkpoint → schema v2 migration CLI. 원본 불변·새 디렉터리·atomic publish. manifest 없는 checkpoint는 `--allow-legacy-joint-absolute-checkpoint`가 있을 때만 joint_absolute로 선언, v1 `eef_relative+rot6d`는 dataset/stats 명시 제공 시 dispatch. 차원·이름 기반 추정 없음 |
| **데이터 변환** | `convert/isaaclab2lerobotv3.py` | Isaac Lab HDF5(`pickplace_sm --record_hdf5` 산출) → LeRobot v3 변환. **env-free**(Isaac·lerobot 패키지 불요, h5py lazy) — `obs_x/joint_pos`·`obs_x/images/{top,wrist,front}`·`applied_target` 캐노니컬 키를 feature_codec 단위로 변환해 `LeRobotV3DatasetWriter`(직기록과 동일 writer=동일 스키마)로 기록. success demo 만(`--include_failed` 로 해제). GPU end-to-end(record→변환→`validate_lerobot_schema` PASS) 검증됨 |
| | `convert/joint_dataset_to_eef.py` | SO-101 joint-space LeRobot v3 원본을 보존하고 absolute EEF 파생 v3 생성. `--source-domain {sim,real}`로 policy-feature/follower calibration을 명시해 공통 URDF `base_link→tcp_grasp` FK 적용. `--rotation-representation {rot6d,rpy,wxyz}`(기본 rot6d=R 첫 두 row)에 따라 state/action을 EEF+gripper 10/7/8D로 변환하며, `--keep-joints`는 arm joint radian 5D를 추가(15/12/13D). `meta/{info,stats,modality}`+기존 episode stats 갱신. `--self-check`는 sim/real 정합과 세 회전 표현 round-trip 검증 |
| | `convert/sim_dataset_to_real_follower.py` | sim-프레임 LeRobot v3(arm=sim degree·gripper feature[0,100])를 실 follower 프레임으로 **in-place** 변환(`follower_calibration.policy_feature_to_real_follower`, `record_real_sequence` 의 역방향). 실기기 `lerobot-replay` 재생용, Isaac 무의존. `--self-check`. ⚠ 충돌 위험, e-stop 준비 |
| **데이터** | `data/append_sim_episode.py` | 실 follower replay 로 sim 에서 얻은 achieved 에피소드를 기존 LeRobot v3 데이터셋에 append(sim↔real replay 캘리브레이션 워크플로) |
| | `data/upload_to_huggingface.py` | LeRobot v3 dataset HF 업로드 + codebase_version 태그 자동 생성/이동. `.env` HF_TOKEN/HF_USER |
| | `data/generate_relative_action_stats.py` | (v1) absolute EEF v3에서 episode boundary를 보존한 horizon-aware relative stats profile을 생성·캐시 |
| | `data/generate_action_representation_stats.py` | **v2 universal** stats CLI. LeRobot v3 parquet loader로 mode/pose-format/horizon별 profile을 `meta/action_representation_stats.json` 한 파일에 누적(`--all`로 dataset이 지원하는 조합 전부) |
| **실기기(Windows)** | `real/lerobot.sh` | `.env`(+`POLICY_PROFILE`) 자동 로드 후 `scripts/real` 전용 LeRobot 0.6 uv project로 CLI를 실행. EEF/joint policy client 자동 분기 |
| | `real/pyproject.toml`, `real/uv.lock` | Python 3.12 + LeRobot 0.6.0 Windows 전용 lock. 루트 Isaac 환경과 분리 |
| **기타** | `ece_4560/` | 과정 프로젝트 (보유) |

### 씬 재생성

USD 6개 (`scene.usd` + 객체 5개) 는 `author_pick_cube_scene.py` 로 일괄 재생성. 좌표 변경 시 `SCENE_OFFSET` 상수를 `assets/scenes/cube_desk.py` 에서 갱신하고 스크립트 재실행. `BOWL_CENTER_XY` 같은 world-frame 상수는 `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` 와 동기화.

## Python 패키지 / 의존성

- **패키지 이름** `sim_to_real` (`pyproject.toml`). `[build-system] requires=["setuptools<82"]`, `[tool.setuptools.packages.find] where=["src"]` 로 `src/sim_to_real/` editable 설치.
- **루트 host deps**: Isaac Sim 5.1 호환을 위해 Python 3.11·NumPy 1.26·LeRobot 0.4.x/0.5.x marker를 유지하는 데이터/시뮬 보조 환경이다. policy-server와 Windows 실기기 런타임의 버전 기준이 아니다.
- **정책/실기기 deps**: Linux policy-server는 Dockerfile이 `lerobot[smolvla,async,groot]==0.6.0`을 직접 설치하고, Windows는 `scripts/real/pyproject.toml`이 `lerobot[async,core_scripts,feetech]==0.6.0`을 고정한다.
- **isaaclab** 은 직접 의존(`isaaclab[all,isaacsim]==2.3.2`, 외부 래퍼/leisaac 제거됨). PickCube SM IK 백엔드 = IsaacLab 내장 DLS(`DifferentialInverseKinematicsActionCfg`, `ik_method="dls"`). **leisaac 은 런타임 의존성이 아니다** — 유용한 코드(`devices`·`datagen`·`assets/robots`·`utils`)는 `src/sim_to_real/` 와 `src/so101_contract/leader_calibration.py` 로 vendor, leisaac 내부 import 0(IsaacLab/lerobot/sim_to_real 로 대체).

| 의존성 그룹 | 내용 | 사용처 |
|---|---|---|
| `teleop` | ffmpeg + evdev[linux] + packaging | legacy/root host 보조 경로 |
| `async` | grpcio + protobuf | legacy/root host 보조 경로 |
| `policy` | lerobot[smolvla] + 학습 보조 dependency | host 진단용; production policy-server는 Dockerfile 독립 핀 |
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
  - 실기기(Windows): `uv sync --project scripts/real` 후 `scripts/real/lerobot.sh <mode>`
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
- MoveIt·cuMotion·Lula·RMPFlow·follow-target IK 테스트 스크립트 제거됨. (cuRobo 는 `scripts/cuRobo/` pick-place SM 플래너로 복귀 — position-only 아닌 reachable-manifold 후보 IK.)

### sim 진입 스크립트 AppLauncher 인자 필터

GUI 부팅 진입 스크립트는 `view_eye`/`view_lookat` 같은 **커스텀 인자**를 통째(`AppLauncher(vars(args))`)로 넘기면 Windows 에서 `_prepare_ui` access violation 이 난다. AppLauncher 가 실제 쓰는 키만 화이트리스트(`_LAUNCHER_KEYS`)로 필터해 전달하고, C-레벨 크래시 추적용 `faulthandler.enable(file=...)` 을 부팅 전에 켠다. 적용: `run_cube_desk_ros_bridge.py`. ⚠ Linux 에선 access violation 대신 **livestream viewport docking 이 조용히 실패**하는 형태로도 나타난다 (3-cam 레이아웃 미적용).
