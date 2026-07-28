# 06. 런타임 명세 — Docker · 진입점 · 환경 변수

> **정본**: `docker/`, `.env.example`, `env/*.env`, `pyproject.toml`.
> 포트·프로토콜은 `07_INTERFACES.md` 를 전제한다.

**2-머신 구조**: 실기기 제어는 Windows 워크스테이션의 **native uv**(WSL·Docker 없음),
시뮬·학습·추론은 Linux 서버의 **Docker** 다.

---

## 1. Docker 서비스 5종

compose project name = `VLA-pipeline`. 앵커: `docker/docker-compose.yaml`.

| 서비스 | 이미지 | Dockerfile | container_name | GPU | 스택·역할 |
|---|---|---|---|---|---|
| `policy-server` | `policy-server:0.5.1` | `Dockerfile.policy` | `policy_server` | ✓ 1 | Python 3.12 + LeRobot 0.5.1. async inference gRPC 서버 + VLA 학습 |
| `vla-ros` | `so101-vla-ros:jazzy` | `Dockerfile.vla_ros` | `so101_vla_ros` | ✗ | ROS 2 Jazzy + vendored mini-lerobot. sim 폐루프 VLA 추론 노드 |
| `pink-ik` | `so101-pink-ik:jazzy` | `Dockerfile.pink` | `so101_pink_ik` | ✗ | ROS 2 Jazzy + pin-pink(Pinocchio)·quadprog. 결정적 pick-place SM (CPU only) |
| `isaac-sim` | `so101-isaac-sim:5.1.0` | `Dockerfile.isaac_sim` | `so101_isaac_sim` | ✓ 1 | Isaac Sim 5.1 + IsaacLab 2.3.2 + ROS2 bridge. sim 폐루프·datagen·teleop |
| `curobo-datagen` | `so101-curobo-datagen:5.1.0` | `Dockerfile.cuRobo` | `so101_curobo_cuRobo` | ✓ 1 | isaac-sim 이미지 + cuRobo v0.8. collision-free batch planner |

빌드 context 는 전부 `..`(repo 루트). `policy-server`·`isaac-sim` 만
`BUILDKT_INLINE_CACHE=1` build arg 를 쓴다.

---

## 2. compose 공통 설정

| 항목 | 값 | 이유 |
|---|---|---|
| `network_mode` | **`host`** (전 서비스) | ROS 브릿지·gRPC·ZMQ·WebRTC |
| `ipc` | **`host`** (전 서비스) | |
| `restart` | `"no"` (전 서비스) | |
| `stdin_open` / `tty` | `true` / `true` | |
| `ports` | **없음** | host 네트워크라 불필요 |
| `depends_on` | **정의 없음** | 기동 순서는 문서·`demo_vla.sh` 로만 강제 (§9) |
| `user` | `policy-server` 만 `${UID:-1000}:${GID:-1000}` | 나머지는 root |

healthcheck 는 두 서비스만:

| 서비스 | test | interval / timeout / retries / start_period |
|---|---|---|
| `policy-server` | `test -f /opt/lerobot_version.txt` | 30s / 5s / 3 / 15s |
| `isaac-sim` | `test -f /workspace/outputs/bridge_faulthandler.txt \|\| echo 'warming up'` | 10s / 5s / 60 / 30s |

**디바이스 마운트가 없다.** 실기기 직렬·카메라는 Windows native uv 가 담당한다.

---

## 3. 볼륨 매핑

### 3.1 bind mount

| 호스트 | 컨테이너 | 모드 | 서비스 |
|---|---|---|---|
| `../datasets` | `/workspace/datasets` | rw | policy-server · pink-ik · isaac-sim · curobo-datagen |
| `../logs` | `/workspace/logs` | rw | policy-server |
| `../outputs` | `/workspace/outputs` | rw | policy-server · isaac-sim · curobo-datagen |
| `../scripts` | `/workspace/scripts` | rw | policy-server · isaac-sim · curobo-datagen |
| `../assets` | `/workspace/assets` | rw | isaac-sim · curobo-datagen |
| `../src` | `/workspace/src` | **ro** | policy-server |
| `../src` | `/workspace/src` | rw | isaac-sim · curobo-datagen |
| `..` (repo 전체) | `/workspace` | rw | vla-ros · pink-ik |

> `datasets`·`outputs` 는 repo 안에서 `/DISK1` 심링크다. `..:/workspace` 로만 마운트하면
> 컨테이너 안에서 심링크가 깨진다(`/DISK1` 미마운트 → 출력이 ephemeral 로 소실).
> 그래서 `pink-ik` 는 repo 전체에 더해 `../datasets` 를 **직접** 다시 마운트한다 —
> docker 가 심링크를 realpath 로 풀어 영속된다.

> `isaac-sim`·`curobo-datagen` 은 `/workspace` 를 통째 마운트하지 **않는다** — 이미지 안의
> IsaacLab(`/workspace/isaaclab`)이 덮여버린다. 필요한 디렉터리만 개별 마운트한다.

### 3.2 named volume

| 볼륨 (compose key) | 실제 이름 | 컨테이너 경로 | 서비스 |
|---|---|---|---|
| `hf_cache` | `lerobot_hf_cache` | `/workspace/.cache/huggingface` | policy-server |
| `isaac_lab_cache_kit` | 동일 | `/isaac-sim/kit/cache` | isaac-sim · curobo-datagen |
| `isaac_lab_cache_ov` | 동일 | `/root/.cache/ov` | 동 |
| `isaac_lab_cache_pip` | 동일 | `/root/.cache/pip` | 동 |
| `isaac_lab_cache_gl` | 동일 | `/root/.cache/nvidia/GLCache` | 동 |
| `isaac_lab_cache_compute` | 동일 | `/root/.nv/ComputeCache` | 동 |
| `isaac_lab_logs` | 동일 | `/root/.nvidia-omniverse/logs` | 동 |
| `isaac_lab_data` | 동일 | `/root/.local/share/ov/data` | 동 |
| `isaac_lab_docs` | 동일 | `/root/Documents` | 동 |

HF 캐시가 `/root` 가 아닌 `/workspace` 하위인 이유 = policy-server 가 non-root UID 로 뜨기
때문이다. 머신 이전 시:
`docker run -v lerobot_hf_cache:/cache alpine tar czf - -C /cache . > hf_cache.tgz`.

---

## 4. Entrypoint 모드

### 4.1 `policy-entrypoint.sh` — `CMD="${1:-policy-server}"`

| 모드 | 실행 |
|---|---|
| `prepare-model` | `hf download "$MODEL_REPO_ID" --revision="$MODEL_REVISION" $PREPARE_MODEL_EXTRA_ARGS`. 위치인자 1개로 repo id override 가능. 비면 exit 1 |
| `policy-server` (기본) | `python -m lerobot.async_inference.policy_server --host --port --fps --inference_latency --obs_queue_timeout $POLICY_SERVER_EXTRA_ARGS` |
| `policy-server-affine` | 위와 동일 인자로 `scripts/inference/policy_server_affine.py` 실행 (`04_IO_CONTRACT.md §8`) |
| `train` | `NUM_PROCESSES > 1` → `accelerate launch --mixed_precision --num_processes -m lerobot.scripts.lerobot_train`; 아니면 `ACCELERATE_MIXED_PRECISION` export 후 `lerobot-train` |
| `eval` | `lerobot-eval "$@"` |
| `info` | `lerobot-info` (없으면 python 인라인 fallback) |
| `bash` / `shell` | `/bin/bash` |
| `python` | `python "$@"` |
| 그 외 | `exec "$@"` |

GPU·lerobot 체크를 건너뛰는 모드: `bash` `shell` `python` `info` `prepare-model`.

**train 출발 모델 라우팅**:

| 조건 | 인자 |
|---|---|
| `TRAIN_POLICY_TYPE` 있음 | `--policy.type=$TRAIN_POLICY_TYPE` + `--policy.base_model_path=$POLICY_BASE_MODEL_PATH` |
| 없고 `POLICY_BASE_MODEL_PATH` 있음 | `--policy.path=$POLICY_BASE_MODEL_PATH` |

> LeRobot 0.5.x parser 는 `--policy.path` 와 `--policy.type` **동시 지정을 금지**한다.

`COMPILE_MODEL=true` + `TRAIN_POLICY_TYPE=groot` 이면 경고 후 compile 을 건너뛴다.

기본값(`${VAR:-default}` 원문):

| 변수 | 기본 | 변수 | 기본 |
|---|---|---|---|
| `MODEL_REPO_ID` | `${POLICY_REPO_ID:-lerobot/smolvla_base}` | `TRAIN_STEPS` | `100000` |
| `MODEL_REVISION` | `main` | `BATCH_SIZE` | `8` |
| `POLICY_SERVER_HOST` | `0.0.0.0` | `OUTPUT_DIR` | `outputs/train/${JOB_NAME:-run}` |
| `POLICY_SERVER_PORT` | `8080` | `DEVICE` | `cuda` |
| `POLICY_FPS` | `30` | `WANDB_ENABLE` | `false` |
| `INFERENCE_LATENCY` | `0.033` | `NUM_WORKERS` | `8` |
| `OBS_QUEUE_TIMEOUT` | `2` | `COMPILE_MODEL` | `false` |
| `NUM_PROCESSES` | `1` | `COMPILE_MODE` | `reduce-overhead` |
| `MIXED_PRECISION` | `bf16` | `JOINT_FRAME_MODE` | `sim-to-sim` |

### 4.2 `isaac-sim-entrypoint.sh` — `MODE="${1:-bridge}"`

| 모드 | 실행 |
|---|---|
| `bridge` (기본) | `isaaclab.sh -p $SO101_BRIDGE_SCRIPT --livestream 2 --num_cubes $NUM_CUBES $BRIDGE_EXTRA_ARGS` |
| `datagen` | `isaaclab.sh -p scripts/datagen/record_state_machine.py --task $DATAGEN_TASK --num_demos $NUM_DEMOS $DATAGEN_EXTRA_ARGS` |
| `teleop` | `isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task $TELEOP_TASK --teleop_device $TELEOP_DEVICE --leader_endpoint $LEADER_ENDPOINT --enable_cameras --step_hz 30 --record --record_format $RECORD_FORMAT --dataset_dir $DATASET_DIR --task_description $TASK_DESCRIPTION --num_demos $NUM_DEMOS <livestream> $SIM_TELEOP_EXTRA_ARGS` |
| `bash` / `shell` | `/bin/bash` |
| `python` | `isaaclab.sh -p "$@"` |
| 그 외 | `exec "$@"` |

teleop livestream 분기: `PUBLIC_IP` 가 있으면 `--public_ip "$PUBLIC_IP"`, 없으면 `--livestream 2`.

기본값:

| 변수 | 기본 |
|---|---|
| `SO101_BRIDGE_SCRIPT` | `/workspace/scripts/inference/run_cube_desk_ros_bridge.py` |
| `NUM_CUBES` | `4` ⚠ compose 는 `${NUM_CUBES:-1}` — §5.3 |
| `BRIDGE_EXTRA_ARGS` | (빈값) |
| `DATAGEN_TASK` | `SimToReal-SO101-PickCube-v0` |
| `NUM_DEMOS` | `50` |
| `DATAGEN_EXTRA_ARGS` | (빈값) |
| `TELEOP_TASK` | `${DATAGEN_TASK}` |
| `TELEOP_DEVICE` | `so101leader_remote` |
| `LEADER_ENDPOINT` | `tcp://localhost:5556` |
| `RECORD_FORMAT` | `lerobot_v3` |
| `DATASET_DIR` | `/workspace/datasets/so101_teleop_sim` |
| `TASK_DESCRIPTION` | `pick up the cube and place it in the bowl` |
| `LIVESTREAM` / `PUBLIC_IP` | `1` / (빈값) |
| `ISAACLAB_SH` | `/workspace/isaaclab/isaaclab.sh` |

`LD_LIBRARY_PATH` 에 `/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib` 를 prepend 한다.
`set -euo pipefail`.

### 4.3 `pink-ik-entrypoint.sh` · `vla-ros-entrypoint.sh`

| | pink-ik | vla-ros |
|---|---|---|
| shell opt | `set -eo pipefail` (`-u` 없음) | `set -eo pipefail` |
| ROS setup | `source /opt/ros/jazzy/setup.bash` | 동일 |
| `PYTHONPATH` | `/workspace/src` | `/workspace/src` + `/workspace/ros2_ws/src/so101_vla_policy/vendor` |
| 빌드 | 없음 | `colcon build --symlink-install --packages-select so101_vla_policy` → `source install/setup.bash` |
| 인자 있음 | `exec "$@"` | `exec "$@"` |
| 인자 없음 | `python3 scripts/datagen/pink_ik_bridge_node.py $PINK_ARGS` | `ros2 launch so101_vla_policy vla_policy.launch.py` |

vla-ros 는 **컨테이너 안에서** colcon 빌드한다(호스트 빌드 아님).

### 4.4 컨테이너 CMD

| 이미지 | ENTRYPOINT | CMD |
|---|---|---|
| `policy-server` | `entrypoint.sh` | `["policy-server"]` |
| `isaac-sim` | `entrypoint.sh` | `["bridge"]` |
| `curobo-datagen` | `entrypoint.sh` | `["datagen"]` |
| `pink-ik` / `vla-ros` | 미지정 | compose entrypoint 가 지정 |

---

## 5. 환경 변수

### 5.1 주입 경로

```
compose env_file: [ ../.env, ../env/${POLICY_PROFILE:-groot_n15}.env ]   ← 나중 파일이 override
        ↓
entrypoint.sh 가 ${VAR:-default} 로 기본값 보충 → CLI 인자로 매핑
```

`vla_policy_node` 는 컨테이너 안에서 `.env` → `env/<POLICY_PROFILE>.env` 순으로 다시 로드한다
(`.env` 를 먼저 읽어야 프로필의 `${HF_USER}` 보간이 풀린다). native uv 에선 셸에서
`set -a; source .env; set +a`.

### 5.2 `.env.example` — 9섹션 69변수

| § | 대상 머신 | 변수 |
|---|---|---|
| §0 비밀값 | 공통 | `HF_TOKEN` · `HF_USER` · `WANDB_API_KEY` |
| §1 모델 선택 | 공통 | **`POLICY_PROFILE`** (기본 `groot_n15`) — 한 줄로 모델 결정 |
| §2 하드웨어 | 클라이언트 | `TELEOP_ID`=`so101_teleop` · `ROBOT_ID`=`so101_robot` · `TELEOP_PORT`=`/dev/ttyACM0` · `ROBOT_PORT`=`/dev/ttyACM1` · `ROBOT_TYPE`=`so101_follower` · `TELEOP_TYPE`=`so101_leader` · `CALIBRATE_TARGET`=`robot` |
| §3 카메라 | 클라이언트 | `ENABLED_CAMERAS`=`top,wrist,front` · `TOP/WRIST/FRONT_CAM_PORT`=`0/1/2` · `CAM_WIDTH`=`640` · `CAM_HEIGHT`=`480` · `CAM_FPS`=`25` · `CAM_FOURCC`=`MJPG` · `CAMERAS`(inline JSON) |
| §4 데이터 수집 | 클라이언트 | `SINGLE_TASK` · `HF_DATASET_REPO_ID` · `DATASET_ROOT` · `NUM_EPISODES`=`10` · `EPISODE_TIME_S`=`60` · `RESET_TIME_S`=`10` · `RECORD_FPS`=`30` · `PUSH_TO_HUB`=`true` · `EPISODE_INDEX`=`0` · `DISPLAY_DATA`·`DISPLAY_IP`·`DISPLAY_PORT` · `STREAM_DATA` · `ENCODER_THREADS`=`2` · `TELEOP/RECORD/REPLAY_EXTRA_ARGS` |
| §5 학습 | 서버 | `BATCH_SIZE`=`16` · `TRAIN_STEPS`=`10000` · `DEVICE`=`cuda` · `WANDB_ENABLE`=`true` · `DATASET_VIDEO_BACKEND`=`torchcodec` · `POLICY_VIDEO_BACKEND` · `TRAIN_EXTRA_ARGS` · `NUM_WORKERS`=`12` · `COMPILE_MODEL`=`false` · `COMPILE_MODE`=`reduce-overhead` · `NUM_PROCESSES`=`1` · `MIXED_PRECISION`=`bf16` · `MODEL_REVISION`=`main` · `PREPARE_MODEL_EXTRA_ARGS` |
| §6 추론 서버 | 서버 | `POLICY_SERVER_HOST`=`0.0.0.0` · `POLICY_SERVER_PORT`=`8080` · `POLICY_FPS`=`30` · `INFERENCE_LATENCY`=`0.033` · `OBS_QUEUE_TIMEOUT`=`2` · `POLICY_SERVER_EXTRA_ARGS` |
| §7 추론 클라이언트 | 클라이언트 | `POLICY_SERVER_ADDRESS`=`127.0.0.1:8080` · `POLICY_DEVICE`=`cuda` · `CLIENT_DEVICE`=`cpu` · `TASK` · `CHUNK_SIZE_THRESHOLD`=`0.5` · `AGGREGATE_FN_NAME`=`weighted_average` · `POLICY_CLIENT_EXTRA_ARGS` |
| §8 시뮬 teleop | cross-machine | `LEADER_ENDPOINT`=`tcp://localhost:5556` · `DATASET_DIR` · `NUM_DEMOS`=`50` · `TASK_DESCRIPTION` · `SIM_TELEOP_EXTRA_ARGS` · `PUBLIC_IP`(주석 처리) |

`OUTPUT_DIR` 은 의도적으로 미설정이다 — entrypoint 가 `outputs/train/${JOB_NAME}` 으로 파생한다.
`RENAME_MAP` 도 `.env` 가 아니라 프로필에서만 정의한다.

`AGGREGATE_FN_NAME` 허용값 = `04_IO_CONTRACT.md §6` 의 4종.

### 5.3 서비스별 하드코딩 environment

| 서비스 | 변수 | 값 |
|---|---|---|
| policy-server | `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` | `all` / `compute,utility,video` |
| vla-ros · pink-ik | `RMW_IMPLEMENTATION` / `FASTDDS_BUILTIN_TRANSPORTS` | `rmw_fastrtps_cpp` / `UDPv4` |
| pink-ik | `PINK_ARGS` | `${PINK_ARGS:-}` |
| isaac-sim | `ACCEPT_EULA` / `PRIVACY_CONSENT` | `Y` / `Y` |
| isaac-sim | `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics` |
| isaac-sim | `LIVESTREAM` | `"1"` (하드코딩) |
| isaac-sim | `NUM_CUBES` | `${NUM_CUBES:-1}` |
| curobo-datagen | `DATAGEN_TASK` / `NUM_DEMOS` / `NUM_CUBES` / `DATAGEN_EXTRA_ARGS` | `SimToReal-SO101-PickCube-v0` / `50` / `1` / `--headless` |

> ⚠ **`NUM_CUBES` 기본값이 세 곳에서 다르다**: compose `1` · entrypoint `4` · bridge argparse `4`.
> compose 를 통해 뜨면 `1` 이 이긴다. `09_TACIT_KNOWLEDGE.md §9` 참조.

---

## 6. 모델 프로필

모델 간 값이 다른 변수만 `env/<name>.env` 로 분리하고, `.env` 의 `POLICY_PROFILE` 한 줄로
활성 모델을 고른다. 새 모델 = 프로필 파일 추가.

| 변수 | `act.env` | `groot_n15.env` | `smolvla.env` |
|---|---|---|---|
| `POLICY_TYPE` | `act` | `groot` | `smolvla` |
| `TRAIN_POLICY_TYPE` | `act` | `groot` | **(빈값)** |
| `POLICY_BASE_MODEL_PATH` | (빈값) | `nvidia/GR00T-N1.5-3B` | `lerobot/smolvla_base` |
| `POLICY_TOKENIZER_ASSETS_REPO` | (빈값) | (빈값) | (빈값) |
| `POLICY_EMBODIMENT_TAG` | (빈값) | `new_embodiment` | (빈값) |
| `POLICY_CHUNK_SIZE` / `POLICY_N_ACTION_STEPS` | (빈값) | `16` / `16` | (빈값) |
| `ACTIONS_PER_CHUNK` | `100` | `16` | `20` |
| `POLICY_REPO_ID` | `${HF_USER}/so101_act_pick_cube` | `/workspace/outputs/train/so101_groot_n15_pick_cube/checkpoints/last/pretrained_model` | `/workspace/outputs/train/so101_smolvla_pick_cube_v2/checkpoints/last/pretrained_model` |
| `JOB_NAME` | `so101_act_pick_cube` | `so101_groot_n15_pick_cube` | `so101_smolvla_pick_cube` |
| `RENAME_MAP` | (빈값) | (빈값) | `{"observation.images.top":"...camera1", "...wrist":"...camera2", "...front":"...camera3"}` |
| `DATASET_VIDEO_BACKEND` | — | (빈값) | — |
| `HF_DATASET_REPO_ID` | — | `${HF_USER}/so101_sim_pick_cube` | — |
| `GRIPPER_CMD_OFFSET` | — | `0.0` | `0.0` |

**GR00T-N1.5 는 별도 서비스가 아니다.** LeRobot 0.5.1 내장 `groot` policy(Eagle-2.5 backbone)라
policy-server 안에서 ACT·SmolVLA 와 동일하게 학습·추론한다. 추론 시
`vla_policy_node` 가 `policy_type=groot` 로 `SendPolicyInstructions` 하면 서버가
`GrootPolicy` 를 로드한다.

`RENAME_MAP` 은 서버가 아니라 **클라이언트가** 적용한다 — 이유는 `07_INTERFACES.md §8.2`.

---

## 7. 의존성·ABI 핀

### 7.1 `pyproject.toml`

| 항목 | 값 |
|---|---|
| `requires-python` | `>=3.11,<3.13` |
| build backend | `setuptools.build_meta`, `requires = ["setuptools<82"]` |
| 패키지 탐색 | `[tool.setuptools.packages.find] where = ["src"]` → `sim_to_real`, `so101_contract` |

공용 dependencies: `h5py<3.16` · `hf-xet>=1.4.3` · `pyzmq>=27.1.0` · `lerobot[feetech]>=0.4.4` ·
`torch>=2.7` · `torchvision>=0.22` · `usd-core>=26.5`

| 그룹 | 내용 | 사용처 |
|---|---|---|
| `teleop` | `ffmpeg>=1.4` · `evdev>=1.7; sys_platform=='linux'` · `packaging>=25.0` | Windows 실기기 |
| `async` | `grpcio==1.73.1` · `protobuf>=6.31.1,<6.32.0` | Windows policy-client |
| `policy` | `lerobot[smolvla]>=0.4.2` + GR00T 의존(peft·diffusers·dm-tree·timm·decord·ninja·flash-attn 등) | **호스트 참조용** — Dockerfile.policy 는 uv sync 를 쓰지 않는다 |
| `isaac` | `isaacsim[all,extscache]==5.1.0` · `isaaclab[all,isaacsim]==2.3.2` | Linux host uv sim |
| `dev` | `ipykernel>=7.2.0` | |

index 라우팅: `torch`/`torchvision` → `pytorch-cu128`(`https://download.pytorch.org/whl/cu128`,
`explicit=true`), `isaacsim`/`isaaclab` → `nvidia`(`https://pypi.nvidia.com`).
`no-build-isolation-package = ["flash-attn"]`.

### 7.2 override-dependencies — **임의 업그레이드 금지**

`override-dependencies` 는 transitive 제약을 강제로 무시한다. 예를 들어 `datasets 4.x` 가
`pyarrow>=21` 을 요구해도 `pyarrow<19` 를 설치할 수 있다.

| 핀 | 이유 | 어기면 |
|---|---|---|
| `numpy==1.26.0` | Isaac Sim 5.1.0 `isaacsim_kernel` 강제 | uv 설치 자체 실패 |
| `pyarrow>=17,<19` | numpy 1.x C-API 호환 마지막 메이저. 19+ 는 numpy 2.x ABI 전용 | Isaac Sim 시작 후 ~30초 silent crash |
| `datasets>=4.0.0,<4.7.0` | 4.7.0+ 가 `pa.json_()` 사용(PyArrow 19+ 필요) | uv resolve 실패 |
| `torchcodec>=0.5,<0.6; sys_platform != 'win32'` | torch 2.7 호환 마지막 마이너. 0.10+ 는 PyTorch 2.11+ ABI | DataLoader worker 즉시 크래시 |
| `packaging>=24.2,<26.0` | 메타데이터 검증 충돌 회피 | `uv sync` resolve 실패 |
| `h5py<3.16` (공용 dep) | Isaac Sim 번들 HDF5 1.14.x ABI 일치. 3.16+ 는 HDF5 2.0 번들 | `Windows fatal exception: code 0xc0000139` |
| `setuptools<82` (build-constraint) | 일부 의존성 `pkg_resources` 호환 | sdist 빌드 실패 |
| `torch==2.7.0+cu128` (lock 해석값) | Isaac Sim 5.1 번들 CUDA 12.8 일치 | 기동 시 CUDA 호출 실패 |

`win32` 를 torchcodec override 에서 뺀 이유: torchcodec 0.5 는 Windows 휠이 없고 추론
클라이언트는 torchcodec 을 쓰지 않는다.

### 7.3 Dockerfile 별 핀

| 이미지 | base | 핵심 핀·설치 |
|---|---|---|
| `Dockerfile.policy` | `nvidia/cuda:12.8.0-devel-ubuntu24.04` → runtime | uv managed **CPython 3.12**, venv `/opt/venv`. `torch>=2.7.0,<2.11.0`·`torchvision>=0.22.0,<0.26.0` (cu128 index) → **`lerobot[smolvla,async]==0.5.1`** → GR00T 의존 → `flash-attn` (`TORCH_CUDA_ARCH_LIST="8.6 9.0 12.0"`, `MAX_JOBS=8`, `--no-build-isolation`) → `groot_compat_patch.py` 실행 후 삭제 |
| `Dockerfile.isaac_sim` | `nvcr.io/nvidia/isaac-lab:2.3.2` | 추가 pip: `pyarrow<19` · `pandas` · `imageio` · `imageio-ffmpeg` · `prettytable` · `pyzmq>=27.1.0`. isaaclab/isaacsim/torch/numpy/cv2 는 **베이스 보유 — 재핀 금지** |
| `Dockerfile.cuRobo` | `so101-isaac-sim:5.1.0` | `SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 pip install "/opt/curobo[cu12]" "packaging==23.0" nvidia-cuda-cccl-cu12` → `pip install --force-reinstall --no-deps "nvidia-cuda-nvrtc-cu12==12.8.93"` |
| `Dockerfile.pink` | `ros:jazzy-ros-base` | `pin-pink` · `quadprog` · `typing-extensions` · **`numpy<2`** · `pyarrow<19` · pandas · imageio |
| `Dockerfile.vla_ros` | `ros:jazzy-ros-base` | `torch`(**CPU index**) · `grpcio` · `protobuf>=6.31,<7` · `python-dotenv` · **`numpy<2`** · `huggingface_hub` · `pyarrow<19`. **lerobot 미설치** — vendored shim 사용 |

`Dockerfile.policy` 가 uv sync 를 쓰지 않는 이유: pyproject override(`numpy==1.26.0` 등)와
lerobot 0.5.1(`numpy>=2.0`)이 충돌한다. `uv pip install` 로 직접 설치한다.

**cuRobo 설치 함정 3종**(코드 주석):

1. `packaging==23.0` **정확 핀** 필수 — cuRobo 가 packaging 을 25.0 으로 올리면 `_structures`
   제거로 Isaac 번들 torch 가 즉사한다. `<26` 같은 범위 핀은 무효다.
2. nvrtc 헤더 force-reinstall — prebundle 은 lib-only 라 `nvrtc.h` 가 없다.
3. `nvidia-cuda-cccl-cu12` 별도 설치 — `[cu12]` extra 가 CCCL 헤더를 제외한다.

numpy 1.26 · warp 1.11 · torch 2.7 ABI 3핀은 불변이다. nvcc 는 불필요(NVRTC JIT).

### 7.4 `groot_compat_patch.py` — 버전 트립와이어

`Dockerfile.policy` 빌드 시 `lerobot[smolvla,async]==0.5.1` 설치 직후 **1회** 실행된다.
transformers 5.3 + torch 2.10 에서 LeRobot 0.5.1 의 GR00T-N1.5 네이티브 wrapper 가 깨지는
4지점을 site-packages 에서 **멱등** 패치한다.

> **GR00T-N1.5 추론·학습에 필수 — 삭제 금지.** 대상 코드 형태가 다르면 `RuntimeError` 로
> 빌드를 중단한다(버전 트립와이어). lerobot·transformers 업그레이드 시 이 패치부터 점검할 것.

---

## 8. 하드웨어 요구

### 8.1 2-머신 사양

| | Windows 워크스테이션 | Linux 학습 서버 |
|---|---|---|
| OS | Windows 11 Pro | Ubuntu 24.04.3 LTS (kernel 6.14.0-1014-oem) |
| CPU | Intel Xeon W-2245 @3.90 GHz (8C/16T) | Intel Core Ultra 5 245K (14C/14T) |
| RAM | 64 GB | 128 GB DDR5 (+swap 8 GB) |
| Storage | NVMe 512 GB + SATA 1 TB | NVMe 477 GB (`/`) + SATA 3.6 TB (`/DISK1`) |
| GPU | RTX A4000 16 GB (driver 596.36, CUDA 13.2, cc 8.6) | RTX PRO 5000 Blackwell 48 GB (driver 580.95.05-open, cc 12.0) |
| 역할 | 실기기 SO-101 직결 (record/calibrate/policy-client) | 시뮬·학습·추론 (Docker) |

### 8.2 ⚠ GPU 제약 — RT 코어 필수

**RT 코어 없는 GPU(A100·H100)는 NVIDIA 가 Isaac Sim 5.1 공식 미지원**이다. 시스템 요구사항
문서가 *"GPUs without RT Cores (A100, H100) are not supported."* 라고 명시한다. 카메라 sensor 가
raytracing pipeline 생성에 실패해 CUDA illegal memory access 가 난다.

사용 가능: RTX A4000/A5000/A6000 · L40/L40S · RTX 6000 Ada · RTX PRO 5000/6000 Blackwell ·
GeForce RTX 40/50. 위 2대 모두 조건을 충족한다.

---

## 9. 기동 시퀀스

`depends_on` 이 없으므로 순서를 수동으로 지켜야 한다.

**sim VLA 폐루프** (3 서비스):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml up policy-server isaac-sim vla-ros
```

`scripts/inference/demo_vla.sh` 가 이 배선을 자동화한다:

| 서브커맨드 | 동작 |
|---|---|
| `start <act\|smolvla\|groot> [옵션]` | 정책 서버 + vla-ros 기동, livestream `:49100` |
| `stop` | 종료 |
| `status` | 상태 |

주요 옵션: `--ckpt PATH` · `--cubes N`(기본 4) · `--ip ADDR` · `--gui` / `--headless` ·
`--apc N` · `--thr T` · `--seed S`(기본 0) · `--no-parity` · `--slew`(기본 on) / `--no-slew` ·
`--arm-vel V`(기본 2.3).

프로필 해석 순서: 인자 → `.env` 의 `POLICY_PROFILE` → `groot_n15`. `groot` 는 `groot_n15` 별칭.
임시 override 프로필 = `env/demo_override.env`(`POLICY_REPO_ID` 만 치환).
데모 전용 컨테이너명 = `vla_demo_ps` / `vla_demo_node`, 로그 = `outputs/p5_logs/`.

**cuRobo pick-place SM**: planner(`curobo-datagen`) 를 먼저 띄우고 SM(`isaac-sim`)을 붙인다.
상세 = `08_PIPELINES.md §5`.

---

## 10. 실기기 native uv (Windows)

WSL·Docker·usbipd 없이 Windows 호스트의 native uv 로 실행한다.

| 항목 | 내용 |
|---|---|
| 의존성 | `uv sync --group teleop --group async` (= lerobot 0.4.4 CLI) |
| `.env` 로드 | 자동 안 됨 → Git Bash `set -a; source .env; set +a` |
| 포트 | COM 포트 직결(`ROBOT_PORT`/`TELEOP_PORT`). usbipd 불필요 |
| 래퍼 | `scripts/real/lerobot.sh` — `.env`+프로필 로드 후 변수→인자 매핑 |

CLI 모드·인자 매핑 = `08_PIPELINES.md §2`.

> ⚠ `evdev` 는 linux 전용이라 Windows 에 설치되지 않는다(`sys_platform` 게이트).

---

## 참조

- 포트·프로토콜 상세 → `07_INTERFACES.md`
- 각 모드를 실제로 어떻게 쓰는가 → `08_PIPELINES.md`
- 정책 I/O 어댑터(`JOINT_FRAME_MODE`) → `04_IO_CONTRACT.md` §8
- 핀·설치 함정의 배경 → `09_TACIT_KNOWLEDGE.md` §7, `docs/TROUBLESHOOTING.md`
