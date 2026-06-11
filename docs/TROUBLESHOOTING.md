# Troubleshooting

## 목차 <!-- omit in toc -->

- [WSL2 NTFS 마운트에서 uv sync 실패 (Operation not permitted)](#wsl2-ntfs-마운트에서-uv-sync-실패-operation-not-permitted)
- [uv-compile Too many open files panic (다코어 호스트, 모든 uv RUN)](#uv-compile-too-many-open-files-panic-다코어-호스트-모든-uv-run)
- [`uv pip install torch` 단계에서 nvidia CUDA 휠 다운로드 timeout](#uv-pip-install-torch-단계에서-nvidia-cuda-휠-다운로드-timeout)
- [torchcodec `c10::MessageLogger::stream` 심볼 누락으로 학습 DataLoader 크래시](#torchcodec-c10messageloggerstream-심볼-누락으로-학습-dataloader-크래시)
- [`torch.compile` 활성화 시 `InvalidCxxCompiler: No working C++ compiler found`](#torchcompile-활성화-시-invalidcxxcompiler-no-working-c-compiler-found)
- [lerobot 0.5.x 업그레이드 후 SmolVLA import 경로 변경 (`ImportError`)](#lerobot-05x-업그레이드-후-smolvla-import-경로-변경-importerror)
- [LeRobot 0.5.1 GR00T N1.5 학습 smoke가 단계별로 실패](#lerobot-051-gr00t-n15-학습-smoke가-단계별로-실패)
- [GR00T 추론 서버에서 `policy-server-rtc`가 표준 추론으로 fallback](#gr00t-추론-서버에서-policy-server-rtc가-표준-추론으로-fallback)
- [카메라 대역폭 제한](#카메라-대역폭-제한)
- [Docker 컨테이너에서 Vulkan 초기화 실패 (Linux)](#docker-컨테이너에서-vulkan-초기화-실패-linux)
- [WSL2 + Docker 에서 Isaac Sim Vulkan/GPU 가속 불가 (회피 불가)](#wsl2--docker-에서-isaac-sim-vulkangpu-가속-불가-회피-불가)
- [Windows 네이티브 bare `isaacsim` Full App 이 app ready 직후 종료](#windows-네이티브-bare-isaacsim-full-app-이-app-ready-직후-종료)
- [Isaac Lab pip 전환 후 `import sim_to_real` 실패](#isaac-lab-pip-전환-후-import-sim_to_real-실패)
- [Isaac Lab SO-101 hold smoke에서 관절 속도 잔류](#isaac-lab-so-101-hold-smoke에서-관절-속도-잔류)
- [Isaac Lab 대규모 PPO에서 `totalAggregatePairsCapacity` 부족](#isaac-lab-대규모-ppo에서-totalaggregatepairscapacity-부족)
- [Isaac Lab `RigidObject` reset sampling이 원점 기준으로 밀림](#isaac-lab-rigidobject-reset-sampling이-원점-기준으로-밀림)
- [Isaac Lab reset 후 원형 펜 collider가 굴러 scene physics smoke 실패](#isaac-lab-reset-후-원형-펜-collider가-굴러-scene-physics-smoke-실패)
- [`lerobot record` 키보드 컨트롤이 동작하지 않음 (WSLg + Windows Terminal)](#lerobot-record-키보드-컨트롤이-동작하지-않음-wslg--windows-terminal)
- [SmolVLA fine-tune 추론 시 카메라 키 불일치 (KeyError: observation.images.camera1)](#smolvla-fine-tune-추론-시-카메라-키-불일치-keyerror-observationimagescamera1)
- [카메라 sensor 가 raytracing pipeline 생성 실패 (RT 코어 없는 GPU)](#카메라-sensor-가-raytracing-pipeline-생성-실패-rt-코어-없는-gpu)
- [Isaac Lab `RigidObject` spawn 에서 parent prim 경로 누락](#isaac-lab-rigidobject-spawn-에서-parent-prim-경로-누락)
- [Sim-to-Real 펜이 그리퍼에 잡히지 않음 (USD Cube scale + 얇은 code-spawn pen)](#sim-to-real-펜이-그리퍼에-잡히지-않음-usd-cube-scale--얇은-code-spawn-pen)
- [Sim-to-Real USD 펜이 관통하며 미끄러짐 (pen contact tuning)](#sim-to-real-usd-펜이-관통하며-미끄러짐-pen-contact-tuning)
- [Sim-to-Real SO-101 base 가 desk 위에서 떠 보임 (mat 배치)](#sim-to-real-so-101-base-가-desk-위에서-떠-보임-mat-배치)
- [Sim-to-Real 씬이 로봇 위치와 어긋남 (scene origin shift)](#sim-to-real-씬이-로봇-위치와-어긋남-scene-origin-shift)
- [Sim-to-Real 에피소드 리셋 시 펜이 한 번 튀어오름 (mat z slack)](#sim-to-real-에피소드-리셋-시-펜이-한-번-튀어오름-mat-z-slack)
- [Sim-to-Real 펜이 닿지 않았는데 그리퍼가 잡음 (pen collider 부풀림)](#sim-to-real-펜이-닿지-않았는데-그리퍼가-잡음-pen-collider-부풀림)
- [Sim-to-Real 펜 collision 형상이 visual 과 어긋남 (Cube collider → visual primitive)](#sim-to-real-펜-collision-형상이-visual-과-어긋남-cube-collider--visual-primitive)
- [Sim-to-Real B/R 리셋 후 동적 RigidBody 가 이전 위치 유지 (env subasset 등록 누락)](#sim-to-real-br-리셋-후-동적-rigidbody-가-이전-위치-유지-env-subasset-등록-누락)
- [Sim-to-Real 그리퍼·펜이 매트/책상을 관통하거나 reset 시 튀어오름 (정적 객체 contactOffset 디폴트)](#sim-to-real-그리퍼펜이-매트책상을-관통하거나-reset-시-튀어오름-정적-객체-contactoffset-디폴트)
- [Sim-to-Real 펜이 펜통 안에서 spawn 되어 겹침 (펜·펜통 sampling 영역 분리 누락)](#sim-to-real-펜이-펜통-안에서-spawn-되어-겹침-펜펜통-sampling-영역-분리-누락)
- [Sim-to-Real 펜통 호 sampling 이 매트/책상 밖으로 나감 (radius 와 default 좌표 불일치)](#sim-to-real-펜통-호-sampling-이-매트책상-밖으로-나감-radius-와-default-좌표-불일치)
- [ROS 2 (WSL2) 노드 간 토픽 통신 불가 — lo 에 MULTICAST 없어 DDS discovery 실패](#ros-2-wsl2-노드-간-토픽-통신-불가--lo-에-multicast-없어-dds-discovery-실패)
- [ROS 2 (WSL2) 카메라 image_raw 토픽이 0 fps — CycloneDDS + mirrored 네트워킹의 대용량 샘플 전달 실패](#ros-2-wsl2-카메라-image_raw-토픽이-0-fps--cyclonedds--mirrored-네트워킹의-대용량-샘플-전달-실패)
- [ROS 2 (WSL2) gscam·v4l2_camera 가 usbipd-win 가상 V4L2 디바이스에서 동작 안 함](#ros-2-wsl2-gscamv4l2_camera-가-usbipd-win-가상-v4l2-디바이스에서-동작-안-함)
- [ROS 2 colcon 빌드가 `catkin_pkg` 못 찾음 (dotfiles 의 ~/.local python 이 ament 가로챔)](#ros-2-colcon-빌드가-catkin_pkg-못-찾음-dotfiles-의-local-python-이-ament-가로챔)
- [ROS 2 빌드 스크립트 `set -u` 가 setup.bash 와 충돌 (AMENT_TRACE_SETUP_FILES unbound)](#ros-2-빌드-스크립트-set--u-가-setupbash-와-충돌-ament_trace_setup_files-unbound)
- [ROS 2 (WSL2) feetech read timeout 1회로 hardware deactivate (USB-IP 레이턴시)](#ros-2-wsl2-feetech-read-timeout-1회로-hardware-deactivate-usb-ip-레이턴시)
- [ROS 2 `libfeetech_ros2_driver.so: file too short` (빌드 캐시 손상)](#ros-2-libfeetech_ros2_driverso-file-too-short-빌드-캐시-손상)
- [ROS 2 (WSL2) `ros2 topic/node list` 가 빈 결과 (stale daemon)](#ros-2-wsl2-ros2-topicnode-list-가-빈-결과-stale-daemon)
- [Isaac Lab ManagerBasedRLEnvCfg 에 `rewards` 누락 시 `gym.make` 실패](#isaac-lab-managerbasedrlenvcfg-에-rewards-누락-시-gymmake-실패)
- [Isaac Lab `gym.make` 이후 Python `print`/로그가 사라짐 (carb stdout 재바인딩)](#isaac-lab-gymmake-이후-python-print로그가-사라짐-carb-stdout-재바인딩)
- [Isaac Lab manipulator 가 작업영역 일부만 도달 / 가까운 물체에서 ee 가 위로 솟음](#isaac-lab-manipulator-가-작업영역-일부만-도달--가까운-물체에서-ee-가-위로-솟음)
- [SO-101 5DOF grasp 가 불안정 (정합·제어점·자세 3중 오차) — Franka 권장](#so-101-5dof-grasp-가-불안정-정합제어점자세-3중-오차--franka-권장)
- [Windows Isaac Sim `_prepare_ui` access violation (tuple 인자를 AppLauncher 에 전달)](#windows-isaac-sim-_prepare_ui-access-violation-tuple-인자를-applauncher-에-전달)
- [(PATH E) Isaac Sim 헤드리스에서 OmniGraph 생성 실패 — `Unable to create prim for graph`](#path-e-isaac-sim-헤드리스에서-omnigraph-생성-실패--unable-to-create-prim-for-graph)
- [(PATH E) Isaac Sim 부팅 중 `errno=28 No space left on device` — inotify watch 고갈](#path-e-isaac-sim-부팅-중-errno28-no-space-left-on-device-가-수천-줄--inotify-watch-고갈)
- [(PATH E, 해결) Isaac Lab bridge 의 OmniGraph JointState 가 `device 0 vs -1`](#path-e-해결-isaac-lab-bridge-의-omnigraph-jointstate-가-device-0-vs--1-로-joint_states-미publish)
- [(PATH E) Isaac Sim ROS 2 bridge 가 `librmw_implementation.so` 로드 실패](#path-e-isaac-sim-ros-2-bridge-가-librmw_implementationso-로드-실패--libament_index_cppso-cannot-open)
- [(PATH E) host(bridge)↔container DDS discovery 실패 — cross-UID fastrtps SHM](#path-e-hostbridgecontainerros-스택-dds-discovery-실패--cross-uid-fastrtps-shm)
- [(PATH E) `pick_place.launch.py` ROS 스택 bringup 4대 함정](#path-e-pick_place-launchpy-ros-스택-bringup-4대-함정)
- [(PATH E) cuMotion `INVALID_INITIAL_CSPACE_POSITION` — start_state 관절 수 ≠ cspace](#path-e-cumotion-invalid_initial_cspace_position--start_state-관절-수--cspace-gripper-포함)
- [시뮬레이션 기동 시 무시해도 되는 로그](#시뮬레이션-기동-시-무시해도-되는-로그)

---

## WSL2 NTFS 마운트에서 uv sync 실패 (Operation not permitted)

**현상**: WSL2에서 Windows 드라이브(`/mnt/d/` 등)에 있는 프로젝트 폴더로 `uv sync` 실행 시 패키지 설치 실패

**오류 메시지**:

```
error: Failed to install: ipykernel-7.2.0-py3-none-any.whl (ipykernel==7.2.0)
  Caused by: Failed to copy to `/mnt/d/.../inprocess/.tmpVKxJt7/blocking.py`
  Caused by: failed to copy file ... : Operation not permitted (os error 1)
```

### 원인

uv는 파일 설치 시 임시 파일(`.tmpXXXXXX`)을 생성한 뒤 atomic rename하는 방식을 사용한다.
WSL2가 NTFS를 9P 드라이버로 마운트한 경로(`/mnt/c/`, `/mnt/d/` 등)에서는 이 오퍼레이션이 허용되지 않아 `EPERM (Operation not permitted)` 발생. `sudo`로 실행해도 파일시스템 레벨의 제약이므로 동일하게 실패한다.

### 해결 방법

두 가지 방법 중 선택:

**방법 1 — 프로젝트를 Linux 파일시스템으로 이동 (권장)**

프로젝트 폴더를 WSL 네이티브 경로(`~/`)로 옮기거나 새로 clone.

```bash
cd ~
git clone <remote-url> robotics_manipulation
cd robotics_manipulation
uv sync --group teleop
```

WSL 파일시스템은 성능과 심링크·권한 호환성 모두 우수하다.

**방법 2 — Windows 마운트에 Linux 메타데이터 활성화**

`/mnt/` 경로를 그대로 유지해야 한다면 WSL 마운트 옵션에 메타데이터를 추가한다.

```ini
# /etc/wsl.conf
[automount]
options = "metadata,umask=22,fmask=11"
```

저장 후 Windows PowerShell에서 WSL 재시작:

```powershell
wsl --shutdown
```

이후 WSL을 다시 열고 `uv sync` 재실행.

### 확인 방법

```bash
python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '/ torch', torch.__version__)"
```

---

## uv-compile Too many open files panic (다코어 호스트, 모든 uv RUN)

**현상**: `docker compose build lerobot` 에서 uv 가 bytecode 를 컴파일하는 어느 단계에서든 수십~수백 개 스레드가 동시에 panic 하며 실패. 코어 수가 많은 빌드 호스트 (예: 224 코어 Linux 서버) 에서만 재현된다. 데스크탑(16 스레드급) 에서는 무사 통과한다.

재현 단계는 두 군데 모두에서 일어난다:

1. **Stage 3 `python-setup`** — `uv python install 3.11` 이 managed CPython 의 stdlib `.pyc` 를 빌드 시점에 미리 컴파일하다 fd 소진.
2. **Stage 4 `torch-layer` / Stage 5 `teleop-deps`(또는 `policy-deps`)** — `uv pip install` / `uv sync` 가 설치 직후 venv `/opt/venv/lib/python3.11/site-packages` 안의 모든 `.py` 를 컴파일하다 fd 소진. torch + nvidia-* + numpy 등 무거운 패키지가 들어오면 더 빨리 터진다.

**오류 메시지** (둘 다 같은 line 에서 panic):

```
thread 'uv-compile' (403) panicked at crates/uv-installer/src/compile.rs:139:26:
Failed to build runtime: Os { code: 24, kind: Uncategorized, message: "Too many open files" }
...
error: Failed to bytecode-compile Python file in: /opt/venv/lib/python3.11/site-packages
  Caused by: Failed to start Python interpreter to run compile script
  Caused by: Too many open files (os error 24)
```

Stage 3 변종은 `Failed to bytecode-compile Python standard library for: cpython-...` 로 시작한다 — 메시지의 대상 디렉터리만 다르고 근본 원인은 동일.

### 원인

`UV_COMPILE_BYTECODE=1` 이 설정돼 있으면 uv 는 (a) managed CPython 설치 직후 stdlib 를, (b) 매 패키지 설치 직후 venv site-packages 를 `.pyc` 로 미리 컴파일한다 (컨테이너 기동 속도 최적화 목적). uv 의 컴파일러는 `std::thread::available_parallelism()` 만큼 워커 스레드를 띄우고 **각 워커가 자체 Tokio runtime 을 생성**한다. Tokio runtime 하나당 epoll/eventfd 등으로 fd 를 수 개 소모하므로, 호스트가 224 코어이면 224 × ~3 fd ≈ 600+ fd 가 순식간에 사용된다 (실측에서는 패키지 설치 후 컴파일 시 thread ID 가 400+ 까지 올라가 더 많은 fd 필요).

Docker 컨테이너의 기본 file descriptor soft limit 은 **1024** (hard limit 은 호스트가 1048576 이어도 무관) 이고, BuildKit 빌더도 같은 기본값을 상속한다. 호스트 셸의 `ulimit -n` 이 1048576 으로 보여도 빌드 안에서는 1024 가 적용된다.

`RAYON_NUM_THREADS` 는 uv-compile 의 자체 워커 풀에는 영향을 주지 않으므로 해결책이 못 된다 (검증 완료). `docker-compose.yaml` 의 `build:` 블록도 `ulimits` 키를 지원하지 않아 외부에서 한도를 올릴 수단이 없다.

### 해결 방법

`Dockerfile.lerobot` / `Dockerfile.policy` 의 **uv 를 호출하는 모든 RUN 명령** 안에서 `ulimit -Sn` 으로 soft 한도를 직접 끌어올린다. hard 한도가 이미 1048576 이므로 soft 만 raise 하면 된다.

> ⚠ **`ulimit` 은 RUN 경계를 넘지 못한다.** Dockerfile 의 RUN 은 매번 새 sh 프로세스를 띄우므로 직전 RUN 에서 올린 soft 한도가 다음 RUN 으로 상속되지 않는다. ENV 도 ulimit 에는 영향을 못 준다. 따라서 Stage 3 뿐 아니라 Stage 4 (`uv pip install torch ...`), Stage 5 (`uv sync ...`) **각 RUN 마다 동일 prefix 를 다시 적어줘야 한다**. 처음 발견했을 때 Stage 3 만 패치하고 Stage 4 에서 같은 panic 이 재발하는 패턴이 흔하다.

```dockerfile
# ── Stage 3 (python-setup): stdlib pyc 컴파일 ──────────────
RUN ulimit -Sn 65536 \
    && uv python install 3.11 \
    && uv venv --python 3.11 ${VIRTUAL_ENV}

# ── Stage 4 (torch-layer): site-packages pyc 컴파일 ────────
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    ulimit -Sn 65536 \
    && UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 \
       uv pip install "torch==2.7.0" "torchvision==0.22.0" \
           --index-url "https://download.pytorch.org/whl/cu128"

# ── Stage 5 (teleop-deps / policy-deps): site-packages pyc 컴파일 ──
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    ulimit -Sn 65536 \
    && UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 \
       uv sync --active --group teleop --group async --no-install-project
```

`ulimit` 은 sh builtin 이라 추가 의존성 없이 동작한다. 65536 이면 224 코어 호스트가 워커당 ~3 fd 를 쓰는 worst case (≈ 700 fd) 의 90× 여유라 안전하다.

### 확인 방법

```bash
# 빌드 — Stage 3 / 4 / 5 가 모두 통과하면 OK
docker compose --env-file .env -f docker/docker-compose.yaml build lerobot 2>&1 \
  | grep -E "(python-setup|torch-layer|teleop-deps|Bytecode compiled|Installed [0-9]+|DONE [0-9]+)"
# 정상 출력 예시:
#   #11 [python-setup 1/1] RUN ulimit -Sn 65536     && uv python install 3.11 ...
#   #11 27.06 Bytecode compiled 1448 files in 422ms
#   #11 DONE 27.2s
#   #14 [torch-layer 3/3] RUN --mount=...,target=/root/.cache/uv ... ulimit -Sn 65536 && ...
#   #14 ... Installed 28 packages in 1.87s
#   #14 DONE ...
```

빌드 컨테이너 내부의 fd 한도를 직접 확인하려면:

```bash
docker run --rm nvidia/cuda:12.8.0-runtime-ubuntu24.04 sh -c 'ulimit -Sn; ulimit -Hn'
# 1024
# 1048576
```

soft 1024 가 그대로면 위 패치가 적용되지 않은 상태다. RUN 안에 `ulimit -Sn` 라인이 빠진 곳을 찾아야 한다.

---

## `uv pip install torch` 단계에서 nvidia CUDA 휠 다운로드 timeout

**현상**: `docker compose build lerobot` 의 Stage 4 (`torch-layer`) 에서 `uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128` 이 100~130초 진행되다 nvidia-* 휠 (cublas / cudnn / cusparse / nvjitlink / cusparselt 등) 중 하나에서 timeout 으로 실패. 매번 실패하는 패키지가 달라진다 (cusparse → cublas → nvjitlink ...). 호스트에서 동일 URL 을 `curl` 로 받으면 1~35초 안에 정상 응답이 온다.

**오류 메시지**:

```
× Failed to download `nvidia-nvjitlink-cu12==12.8.61`
├─▶ Request failed after 3 retries in 126.1s
├─▶ Failed to fetch:
│   `https://pypi.nvidia.com/nvidia-nvjitlink-cu12/nvidia_nvjitlink_cu12-12.8.61-py3-none-...whl`
├─▶ error sending request for url (...) operation timed out
╰─▶ operation timed out
help: `nvidia-nvjitlink-cu12` (v12.8.61) was included because `torch` (v2.7.0+cu128) depends on `nvidia-nvjitlink-cu12`
```

### 원인

torch 2.7.0+cu128 은 transitively 28개 패키지를 끌어오는데 그중 NVIDIA CUDA 휠 합계가 ~3 GB 다 (torch 1 GB / cudnn 693 MB / nccl 192 MB / cufft 184 MB / cusparse 278 MB / cublas 581 MB / ...).

uv 는 기본적으로 **8개 이상을 동시에 다운로드**한다. `pypi.nvidia.com` (NVIDIA 가 운영하는 CDN) 은 동일 client IP 가 large file 을 다수 동시에 요청하면 일부 connection 을 throttle / silent-stall 시킨다. uv 의 기본 HTTP timeout 은 **30초** (정확히는 connect+read 별도 30s/30s) 라, stall 된 connection 이 retry 3회 안에 회복되지 못하면 빌드 전체가 실패한다.

호스트의 단발 `curl` 은 connection 1개라 throttle 대상이 아니다 — 그래서 같은 URL 이 호스트에서는 정상이고 빌드 안에서만 실패하는 현상이 나타난다. MTU 나 DNS 같은 네트워크 레이어 문제는 아니다 (busybox/alpine 컨테이너에서 wget 단발 다운로드는 35초 안에 성공함으로 확인).

추가 가중치: Stage 4 RUN 에 `--no-cache` 플래그가 걸려 있어 빌드 실패 후 재시도해도 이미 받은 휠을 못 쓰고 처음부터 ~3 GB 를 다시 받는다. 외부 네트워크가 잠시만 흔들려도 빌드 전체가 round-trip 한다.

### 해결 방법

`docker/Dockerfile.lerobot` / `docker/Dockerfile.policy` 의 Stage 4 (`torch-layer`) 와 Stage 5 (`teleop-deps` / `policy-deps`) RUN 에 세 가지를 함께 적용한다.

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=2 \
    uv pip install \
        "torch==2.7.0" \
        "torchvision==0.22.0" \
        --index-url "https://download.pytorch.org/whl/cu128"
```

- **`--mount=type=cache,target=/root/.cache/uv`** — BuildKit 영구 캐시. 한 번 받은 휠은 이미지에는 들어가지 않으면서 다음 빌드에서 재사용된다. 부분 성공 후 재시도가 거의 즉시 끝나 외부 네트워크 흔들림에 강건해진다. 동시에 기존 `--no-cache` 플래그는 제거한다 (이게 있으면 uv 가 cache 디렉터리에 쓰지 않아 캐시 마운트가 무용지물).
- **`UV_HTTP_TIMEOUT=600`** — 단일 요청 타임아웃 10분. 큰 휠 (대용량 cudnn / cublas) 의 slow connection 도 끊지 않고 끝까지 받는다.
- **`UV_CONCURRENT_DOWNLOADS=2`** — 동시 다운로드를 2개로 제한. CDN throttling 의 트리거 조건 (다수 동시 large-file) 자체를 피한다. 다운로드 총 시간은 5~10% 길어지지만 안정성이 압도적으로 향상된다.

Stage 5 (`uv sync`) 도 동일 패턴을 적용. lerobot[feetech] / lerobot[smolvla] 가 PyPI 본 인덱스를 쓰므로 throttle 가능성은 낮지만, 같은 캐시 마운트로 재빌드 시간을 단축할 수 있다.

### 확인 방법

```bash
docker compose --env-file .env -f docker/docker-compose.yaml build lerobot 2>&1 \
  | grep -E "(torch-layer|Downloaded|Installed [0-9]+ packages|DONE [0-9]+)"
# 정상 출력 예시:
#   #14 [torch-layer 3/3] RUN --mount=type=cache,target=/root/.cache/uv ...
#   ... Downloaded nvidia-cudnn-cu12 / nvidia-cublas-cu12 / ...
#   #14 Installed 28 packages in ...
#   #14 DONE 180s

# 캐시가 실제로 재사용되는지 확인 (두 번째 빌드)
docker buildx prune --filter=type=exec.cachemount=false -f >/dev/null  # 이미지 캐시만 정리, mount 캐시 유지
docker compose --env-file .env -f docker/docker-compose.yaml build lerobot --no-cache 2>&1 \
  | grep -E "torch-layer.*DONE"
# Stage 4 가 수십 초 안에 끝나면 캐시 마운트 정상 동작.
```

캐시 마운트는 BuildKit 빌더가 살아 있는 동안만 유지되므로 빌더를 재생성하면 (`docker buildx rm` / 호스트 재부팅) 다시 받아야 한다. 그래도 한 빌더 안에서는 부분 실패 → 재시도가 즉시 통과한다.

---

## torchcodec `c10::MessageLogger::stream` 심볼 누락으로 학습 DataLoader 크래시

**현상**

`policy-server train` 모드에서 DataLoader worker 0이 즉시 크래시하며 학습이 시작되지 않는다.

**오류 메시지**

```
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
...
  File ".../torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
RuntimeError: Could not load libtorchcodec. ...

FFmpeg version 6:
OSError: /opt/venv/.../torchcodec/libtorchcodec_core6.so: undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev
```

(다른 FFmpeg 버전은 `libavutil.so.5x/5y/60: cannot open shared object file`)

**원인**

torchcodec 의 버전이 고정되지 않으면 PyPI 최신 버전(0.10+)이 설치된다.  
0.10+ 부터 `libtorchcodec_core*.so` 가 PyTorch 2.11+ C++ ABI 로 빌드되어, 
`torch==2.7.0` 의 `c10::MessageLogger::stream[abi:cxx11]()` 심볼과 맞지 않는다.  
Ubuntu 24.04 apt `ffmpeg`는 libavutil.so.58 (FFmpeg 6.1) 을 제공하는데,
torchcodec 이 libavutil.so.56/57/59/60 을 순서대로 시도하기 때문에 FFmpeg 버전 불일치 오류도 함께 표시된다.

**해결 방법**

`pyproject.toml` `override-dependencies` 에 `torchcodec>=0.5,<0.6` 핀을 추가하고 Docker 이미지를 재빌드한다.

```toml
# pyproject.toml
override-dependencies = [
    ...
    "torchcodec>=0.5,<0.6",   # torch 2.7 호환 마지막 마이너 시리즈
]
```

```bash
docker compose -f docker/docker-compose.yaml build policy-server
```

이미지 재빌드 전에 즉시 우회해야 한다면 학습 명령에 `--dataset.video_backend=pyav` 추가:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  policy-server train ... --dataset.video_backend=pyav
```

**확인 방법**

컨테이너 내부에서 torchcodec import 가 정상인지 검증:

```bash
docker compose -f docker/docker-compose.yaml run --rm policy-server python \
  -c "from torchcodec.decoders import VideoDecoder; print('torchcodec OK')"
```

`torchcodec OK` 가 출력되면 정상. 학습 재실행 시 DataLoader worker 크래시 없이 Training 진행 확인.

---

## `torch.compile` 활성화 시 `InvalidCxxCompiler: No working C++ compiler found`

**현상**

`torch.compile` 이 활성화된 상태(`COMPILE_MODEL=true`)로 학습을 시작하면 첫 번째 forward pass 직후 크래시.

**오류 메시지**

```
torch._inductor.exc.InductorError: InvalidCxxCompiler:
  No working C++ compiler found in torch._inductor.config.cpp.cxx: (None, 'g++')
```

**원인**

`torch.compile` 의 inductor 백엔드는 CPU 커널을 JIT 컴파일할 때 `g++` 를 런타임에 호출한다.
GPU 학습이라도 inductor 가 CPU-side 퓨전 커널을 생성하는 경로가 존재한다.
`Dockerfile.policy` 의 `app` 스테이지(slim 런타임)에 `build-essential` / `g++` 를
제외했기 때문에 컴파일러를 찾지 못한다.

**해결 방법**

`Dockerfile.policy` 의 `app` 스테이지 apt 설치 목록에 `g++` 추가 후 이미지 재빌드:

```dockerfile
# app 스테이지 RUN apt-get install 블록에 추가
g++ \
```

```bash
docker compose -f docker/docker-compose.yaml build policy-server
```

**확인 방법**

```bash
docker compose -f docker/docker-compose.yaml run --rm policy-server python \
  -c "import subprocess, sys; r=subprocess.run(['g++','--version'],capture_output=True); print('g++ OK' if r.returncode==0 else 'MISSING')"
```

재빌드 후 `torch.compile` 활성 상태로 학습 재실행 시 첫 번째 스텝에서 수 분간 컴파일이 발생한 후 정상 진행 확인.

---

## lerobot 0.5.x 업그레이드 후 SmolVLA import 경로 변경 (`ImportError`)

**현상**

`policy-server:0.5.1` 이미지에서 SmolVLA 정책을 직접 import 하는 커스텀 스크립트 실행 시 즉시 실패.

**오류 메시지**

```
ImportError: cannot import name 'SmolVLAPolicy' from 'lerobot.policies.smolvla' (unknown location)
```

**원인**

lerobot 0.4.x 에서는 `lerobot/policies/smolvla/__init__.py` 가 `SmolVLAPolicy` 를 re-export 했으나,
0.5.x 에서 `__init__.py` 가 제거되어 namespace package 로 바뀌었다.
`lerobot.policies.smolvla` 는 더 이상 직접 import 가 불가능하고 하위 모듈을 명시해야 한다.

`policy-entrypoint.sh` 의 `policy-server` 모드(`python -m lerobot.async_inference.policy_server`)는 내부에서 올바른 경로를 사용하므로 영향 없다. 커스텀 Python 스크립트를 직접 작성할 때만 해당된다.

**해결 방법**

```python
# ❌ lerobot 0.4.x 방식 — 0.5.x 에서 ImportError
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.smolvla import SmolVLAConfig

# ✅ lerobot 0.5.x 방식 — 하위 모듈 직접 지정
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
```

같은 패턴이 다른 정책에도 적용된다.

```python
# ACT
from lerobot.policies.act.modeling_act import ACTPolicy

# Diffusion
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# GR00T
from lerobot.policies.groot.modeling_groot import GR00TPolicy
```

**확인 방법**

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLAPolicy OK:', SmolVLAPolicy)
"
```

`SmolVLAPolicy OK: <class '...SmolVLAPolicy'>` 가 출력되면 정상.

---

## LeRobot 0.5.1 GR00T N1.5 학습 smoke가 단계별로 실패

**현상**

`policy-server:0.5.1` 컨테이너에서 `TRAIN_POLICY_TYPE=groot`,
`POLICY_BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B` 설정으로 SO-101 데이터셋 fine-tune smoke를 돌리면,
모델 생성 또는 첫 batch 전처리/forward 단계에서 연쇄적으로 실패한다.

**오류 메시지**

```text
argparse.ArgumentError: Cannot specify both --policy.path and --policy.type
```

```text
RuntimeError: Tensor.item() cannot be called on meta tensors
```

```text
AttributeError: 'GR00TN15' object has no attribute 'all_tied_weights_keys'
```

```text
ValueError: Unsupported video backend: decord
```

```text
AttributeError: 'list' object has no attribute 'shape'
```

```text
NotImplementedError: aten::_sample_dirichlet: attempted to run this operator with Meta tensors
```

**원인**

여러 호환성 문제가 겹친다.

- LeRobot 0.5.1 parser는 `--policy.path`와 `--policy.type` 동시 지정을 금지한다. (과거에는 SmolVLA용 `BASE_MODEL`과 GR00T용 변수를 별도로 두어, SmolVLA 값이 남으면 `--policy.path`와 `--policy.type=groot`가 함께 나가 충돌했다. 지금은 출발 모델을 `POLICY_BASE_MODEL_PATH` 단일 변수로 통일하고 `TRAIN_POLICY_TYPE` 유무로만 둘 중 하나를 emit 하므로 구조적으로 충돌이 없다.)
- LeRobot 0.5.1 dataset decoder가 지원하는 video backend는 `torchcodec`, `pyav`, `video_reader`다. `decord`는 GR00T policy 내부 기본값으로 보이지만 `--dataset.video_backend=decord`에는 사용할 수 없다.
- Transformers 5.3 + torch 2.10 조합에서 LeRobot 0.5.1의 GR00T wrapper가 기대 속성과 tensor 초기화 흐름을 맞추지 못한다. `GR00TN15.all_tied_weights_keys`가 없고, `FlowmatchingActionHead`의 `Beta` 분포가 meta tensor 상태로 생성되어 validation/sampling에서 실패한다.
- Eagle2.5 processor 호출에서 `return_tensors="pt"`가 tokenizer 쪽으로만 전달되고 image processor에는 전달되지 않아 `pixel_values`가 tensor가 아닌 list로 남는다.

**해결 방법**

`.env` / `.env.example` / `docker/policy-entrypoint.sh`의 학습 계약을 LeRobot 0.5.1 기준으로 분리한다.

```dotenv
# SmolVLA fine-tune (LeRobot 체크포인트에서 출발)
TRAIN_POLICY_TYPE=
POLICY_BASE_MODEL_PATH=lerobot/smolvla_base

# GR00T N1.5 fine-tune (타입 wrapper + native 베이스)
TRAIN_POLICY_TYPE=groot
POLICY_BASE_MODEL_PATH=nvidia/GR00T-N1.5-3B
POLICY_TOKENIZER_ASSETS_REPO=lerobot/eagle2hg-processor-groot-n1p5
POLICY_EMBODIMENT_TAG=new_embodiment
POLICY_CHUNK_SIZE=16
POLICY_N_ACTION_STEPS=16
DATASET_VIDEO_BACKEND=torchcodec
POLICY_VIDEO_BACKEND=
```

`policy-entrypoint.sh`는 출발 모델을 `POLICY_BASE_MODEL_PATH` 단일 변수로 받아 `TRAIN_POLICY_TYPE` 유무로 라우팅한다: 비우면 `--policy.path`(LeRobot 체크포인트), 설정하면 `--policy.type` + `--policy.base_model_path`. tokenizer assets, embodiment tag, chunk/action step, dataset video backend도 명시적으로 CLI에 매핑한다.

`docker/Dockerfile.policy`에서는 upstream 패치 전까지 LeRobot site-packages에 최소 호환 패치를 적용한다.

- `FlowmatchingActionHead`: `Beta(..., validate_args=False)`로 meta tensor validation 회피.
- `FlowmatchingActionHead.sample_time`: 학습 forward 시 실제 device tensor로 `Beta` 분포를 재생성.
- `GR00TN15`: `all_tied_weights_keys = {}` 추가.
- `processor_groot.collate`: `text_kwargs={"padding": True, "return_tensors": "pt"}`와 `images_kwargs={"return_tensors": "pt", ...}`를 분리 전달.

이미지 재빌드:

```bash
docker compose --env-file .env -f docker/docker-compose.yaml build policy-server
```

**확인 방법**

이미지 패치 확인:

```bash
docker run --rm -i --entrypoint python policy-server:0.5.1 - <<'PY'
from pathlib import Path
root = Path("/opt/venv/lib/python3.12/site-packages/lerobot/policies/groot")
ah = (root / "action_head/flow_matching_action_head.py").read_text()
gn = (root / "groot_n1.py").read_text()
pg = (root / "processor_groot.py").read_text()
print("beta_init_patch", "validate_args=False" in ah)
print("sample_time_patch", "Beta(alpha, beta, validate_args=False)" in ah)
print("tied_keys_patch", "all_tied_weights_keys = {}" in gn)
print("image_tensor_patch", 'text_kwargs={"padding": True, "return_tensors": "pt"}' in pg)
PY
```

100-step smoke:

```bash
JOB="smoke_groot_n15_pick_pen_100_$(date +%Y%m%d_%H%M%S)"
docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e JOB_NAME="${JOB}" \
  -e OUTPUT_DIR="outputs/train/${JOB}" \
  -e TRAIN_STEPS=100 \
  -e BATCH_SIZE=16 \
  -e WANDB_ENABLE=false \
  policy-server train \
    --save_freq=100 \
    --policy.push_to_hub=false

jq -r .type "outputs/train/${JOB}/checkpoints/000100/pretrained_model/config.json"
```

`groot`가 출력되고 `End of training` 로그가 나오면 정상.

---

## GR00T 추론 서버에서 `policy-server-rtc`가 표준 추론으로 fallback

**현상**

GR00T N1.5 fine-tune checkpoint 로 async inference server 를 띄울 때 `policy-server-rtc` 모드를 사용하면 서버는 뜨지만 RTC guidance 가 적용되지 않는다.

**오류 메시지**

```text
[RTC] GrootPolicy 는 init_rtc_processor 를 지원하지 않습니다.
표준 추론(RTC 없음)으로 동작합니다. (SmolVLA / Pi0 / Pi0.5 만 지원)
```

**원인**

`scripts/policy_server_rtc.py` 는 `policy.init_rtc_processor()` 를 구현한 flow-matching 정책에만 RTCConfig 를 주입한다. 현재 LeRobot 0.5.1의 `GrootPolicy`는 해당 메서드를 제공하지 않는다. 따라서 `_rtc_available=False` 로 fallback 되고, 매 action chunk 마다 RTC 미지원 분기와 로그만 남는다.

**해결 방법**

GR00T 는 표준 async inference server 를 사용한다.

```bash
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
docker compose --env-file .env -f docker/docker-compose.yaml logs -f policy-server
```

직접 실행:

```bash
docker rm -f so101-groot-n15-policy-server 2>/dev/null || true
docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  --name so101-groot-n15-policy-server \
  policy-server policy-server
```

**확인 방법**

```bash
docker logs --tail 80 so101-groot-n15-policy-server
ss -ltnp | grep ':8080'
```

로그에 `PolicyServer started on 0.0.0.0:8080` 가 있고 RTC fallback warning 이 없으면 정상. 클라이언트는 `POLICY_TYPE=groot`, `POLICY_REPO_ID=taehunkim/so101_groot_n15_pick_pen`, `ACTIONS_PER_CHUNK=16` 로 접속한다.

---

## 카메라 대역폭 제한

**현상**: `lerobot-find-cameras` 실행 시 카메라가 탐지는 되지만 일부만 캡처에 성공함

**오류 메시지**:

```
Failed to connect or configure OpenCV camera 1: Failed to open OpenCVCamera(1)
Failed to connect or configure OpenCV camera 2: Failed to open OpenCVCamera(2)
```

**카메라 모델**: Microdia Integrated_Webcam_HD — USB 2.0 전용(추정)

**지원 해상도 프로파일**: `1280×720`, `640×480` 두 가지만 존재 (그 외 해상도 설정 불가)

### 원인

탐지 단계(`find_cameras`)에서는 카메라를 1대씩 열고 즉시 닫으므로 전체가 보이지만,
연결·스트리밍을 동시에 유지하면 일부 카메라가 열리지 않는다.

USB 2.0 카메라 1대의 YUY2 전송량:

```
640 × 480 × 2 bytes × 30 fps = 18.4 MB/s
```

### 테스트 결과

| 구성 | 결과 |
|------|------|
| USB 허브 + YUY2 | 1대만 성공 |
| USB 허브 + MJPEG | 1대만 성공 |
| PC 포트 직접 연결 (각각) | 2대 이상 성공 ✅ |

USB 허브 자체의 하드웨어 한계로, MJPEG로 전송량을 줄여도 허브에서는 동시에 1대만 스트리밍된다.
USB 3.2 허브도 내부적으로 USB 2.0 카메라는 HS 경로(480 Mbps 공유)를 사용하므로 허브 교체로는 해결되지 않는다.

### 해결 방법

**카메라마다 PC USB 포트에 직접 연결** (유일하게 확인된 해결책)

현재 PC(ThinkStation) 기준 사용 가능한 포트:
```
전면: 4× USB 3.2 Gen 1
후면: 4× USB 3.2 Gen 1
     2× USB 2.0
```

카메라 3대를 허브 없이 전부 직접 꽂을 수 있다.


### USB 버전 확인 방법

카메라의 USB 버전 확인

```powershell
# 1. 카메라 InstanceId 조회 (Status OK인 항목 확인)
Get-PnpDevice -Class Camera | Select-Object Status, InstanceId

# 2. ACPI 경로에서 포트 접두사 확인 (<InstanceId>에 위 결과 붙여넣기)
(Get-PnpDeviceProperty -InstanceId "USB\VID_0C45&PID_64AB&MI_00\<InstanceId>" |
  Where-Object { $_.KeyName -eq "DEVPKEY_Device_LocationPaths" }).Data |
  Where-Object { $_ -match "ACPI" }
```

출력 예시:

```markdown
ACPI(_SB_)#ACPI(PC00)#ACPI(XHCI)#ACPI(RHUB)#ACPI(HS09)#USB(2)#USBMI(0)
                                              ^^^^^^^^^^
                                              여기를 본다
```

| 접두사 | USB 버전 | 최대 속도 |
|--------|---------|---------|
| `HS##` | USB 2.0 | 480 Mbps |
| `SS##` | USB 3.0 | 5 Gbps |
| `SSP##` | USB 3.1/3.2 | 10+ Gbps |

USB 허브 버전 확인

```powershell
Get-WmiObject -Class Win32_USBHub | Select-Object DeviceID, Name
```

| 장치 이름 | USB 버전 |
|-----------|---------|
| `Generic USB Hub` | USB 2.0 |
| `Generic SuperSpeed USB Hub` | USB 3.0 |


---

## Docker 컨테이너에서 Vulkan 초기화 실패 (Linux)

**현상**: `docker compose up` 으로 컨테이너를 띄우면 Isaac Sim 이 다음 에러를 토하면서 GPU 가속을 잃고 software 로 fallback 된다. CUDA 자체는 동작하지만 (nvidia-smi 에서 컨테이너 안의 python 프로세스가 GPU 메모리를 점유) 렌더링·카메라·GPU PhysX 가 모두 죽는다.

**오류 메시지**:

```log
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed.
                Vulkan 1.1 is not supported, or your driver requires an update.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices,
                including an attempt with compatibility mode.
[Error] [omni.physx.plugin] CUDA libs are present, but no suitable CUDA GPU was found!
[Warning] [omni.physx.plugin] PhysX warning: GPU solver pipeline failed,
                switching to software
```

### 원인

호스트의 NVIDIA 드라이버가 `.run` 인스톨러로 **`--no-opengl-files`** 옵션과 함께 설치된 경우, `libGLX_nvidia.so.0` / `libnvidia-glcore.so.<ver>` / `libEGL_nvidia.so.0` 같은 그래픽스 유저 스페이스 라이브러리가 호스트에 통째로 빠져 있다. 이 상태에서는 다음이 모두 성립한다:

1. `/etc/vulkan/icd.d/nvidia_icd.json` 은 존재하지만 `library_path: libGLX_nvidia.so.0` 이 가리키는 실제 파일이 호스트에 없다 (dangling pointer).
2. `nvidia-container-cli list` 출력에 `GLX_nvidia` / `glcore` / `EGL_nvidia` 가 한 줄도 없다 → nvidia-container-runtime 이 컨테이너로 마운트할 라이브러리 자체가 호스트에 없다.
3. 컨테이너 안에서 `NVIDIA_DRIVER_CAPABILITIES=all` 을 줘도 마운트할 게 없으니 Vulkan ICD 가 동작 못 한다.

기존 설치 옵션은 `/var/log/nvidia-installer.log` 에서 확인할 수 있다:

```bash
head -15 /var/log/nvidia-installer.log
# nvidia-installer command line:
#     ./nvidia-installer
#     --no-kernel-module
#     --no-opengl-files       ← 이게 원인
#     --silent
```

추가로, docker-compose 의 `deploy.resources.reservations.devices` (`capabilities: [gpu]`) 방식은 `nvidia-container-toolkit ≥ 1.19` 의 일부 환경에서 graphics capability 를 트리거하지 않는다. 같은 호스트에서 legacy 방식 (`runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all`) 으로 띄우면 Vulkan ICD JSON 은 마운트되지만, 위 1번 이유로 라이브러리 자체가 없어서 결국 동일하게 실패한다.

### 해결 방법

같은 버전의 `.run` 인스톨러를 다시 받아서 **커널 모듈은 건드리지 않고 그래픽스 유저 스페이스만** 추가 설치한다.

```bash
# 1. 기존 컨테이너 정지 + GPU 사용 프로세스 종료 확인
docker compose down
nvidia-smi

# 2. 동일 버전 .run 다운로드 (Data Center / Tesla 경로에 호스팅됨)
cd /tmp
DRIVER_VER=$(cat /proc/driver/nvidia/version | awk '/NVRM/ {print $8}')
curl -fLO "https://us.download.nvidia.com/tesla/${DRIVER_VER}/NVIDIA-Linux-x86_64-${DRIVER_VER}.run"
chmod +x "NVIDIA-Linux-x86_64-${DRIVER_VER}.run"

# 3. --no-opengl-files 빼고 --install-libglvnd 추가, 커널 모듈은 그대로 둠
sudo sh "./NVIDIA-Linux-x86_64-${DRIVER_VER}.run" \
    --no-kernel-module \
    --install-libglvnd \
    --silent
```

`--no-kernel-module` 가 핵심이다. 커널 모듈은 이미 동작 중이므로 건드리지 않고, 빠져 있던 GL/Vulkan/EGL 유저 스페이스 라이브러리만 채워 넣는다.

또한 `docker-compose.yaml` 의 GPU 접근 방식은 legacy syntax 로 두는 편이 안정적이다:

```yaml
services:
  leisaac-debug:
    runtime: nvidia
    network_mode: host          # livestream WebRTC 동적 포트 협상에 유리
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
    volumes:
      - /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro   # ICD JSON 안전망
    # deploy: 블록은 사용하지 않음 (graphics capability 트리거 불안정)
```

### 확인 방법

설치 후 호스트에서:

```bash
ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.${DRIVER_VER}
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
nvidia-container-cli list | grep -E 'GLX_nvidia|glcore|EGL_nvidia'
```

세 파일이 모두 존재하고 `nvidia-container-cli list` 에 GLX/glcore/EGL 항목이 출력되면 호스트 측 준비 완료.

컨테이너 안에서:

```bash
docker compose run --rm leisaac-debug bash -c '
  ldconfig -p | grep -E "libGLX_nvidia|libvulkan|libnvidia-glcore" &&
  apt-get install -y vulkan-tools && vulkaninfo --summary
'
```

`vulkaninfo --summary` 가 NVIDIA GPU 의 `deviceName` 과 `apiVersion 1.4.x` 를 출력하면 컨테이너 안에서도 Vulkan 이 정상이다. 이후 `docker compose up` 시 위의 `ERROR_INCOMPATIBLE_DRIVER` / `Failed to create any GPU devices` / `no suitable CUDA GPU was found` / `switching to software` 메시지가 모두 사라진다.

#### Headless 서버에서 외부 PC 로 화면 송출

호스트에 디스플레이가 없는 경우 (서버 환경) Isaac Sim 은 `--headless --livestream=2` (사내망 WebRTC) 로 띄워 외부 PC 에서 Omniverse Streaming Client / 호환 WebRTC 클라이언트로 접속한다. 이때 컨테이너가 바인드하는 포트는 다음과 같다:

| 포트 | 프로토콜 | 용도 | 출처 |
|------|---------|------|------|
| 8011 | TCP | HTTP signaling | `omni.services.transport.server.http` |
| 48010 | TCP | livestream core | `omni.kit.livestream.core` |
| 49100 | TCP | WebRTC media | `omni.kit.livestream.webrtc` |
| 47998-48020 | UDP | 동적 미디어 범위 | `omni.services.livestream.nvcf` |

`network_mode: host` 면 별도 포트 매핑 없이 그대로 노출된다. WebRTC 동적 미디어 협상이 NAT 뒤에서 깨지는 경우가 있어 host network 가 가장 안정적이다.

---

## WSL2 + Docker 에서 Isaac Sim Vulkan/GPU 가속 불가 (회피 불가)

> ⚠️ 이 항목은 **수정 실패** 결과를 남긴다. 현재 NVIDIA 의 WSL2 GPU 노출 정책상 Docker 컨테이너 안에서 NVIDIA RTX Vulkan 가속을 동작시킬 방법이 없음이 실증으로 확인되었다. 다음 세션이 같은 시도를 반복하지 않도록 기록한다.

**현상**: WSL2 (Windows 11 + Docker Desktop + WSLg) 환경에서 `IsaacLab/docker/container.py start` → `enter base` → `python -c "from isaaclab.app import AppLauncher; AppLauncher()"` 실행 시 GPU 가속이 잡히지 않고 위 [Linux 항목과 동일한 fallback 메시지](#docker-컨테이너에서-vulkan-초기화-실패-linux) 가 출력된다.

> **LeIsaac 도 동일하게 영향받는다.** LeIsaac 의 teleop·정책 평가는 Isaac Lab → Isaac Sim → `omni.gpu_foundation_factory` 체인 위에 있어, GUI/headless 여부와 무관하게 RT-capable Vulkan device 를 요구한다. 따라서 본 항목의 진단·결론은 LeIsaac 컨테이너 경로에도 그대로 적용된다. 이번 세션에서 `AppLauncher(headless=True)` 도 동일 단계 (`No device could be created. ... Your GPUs do not support RayTracing`) 에서 실패하는 것이 확인됐다.

**오류 메시지**:

```log
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed.
                Vulkan 1.1 is not supported, or your driver requires an update.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices,
                including an attempt with compatibility mode.
[Error] [gpu.foundation.plugin] No device could be created. Some known system issues:
                - Your GPUs do not support RayTracing: DXR or Vulkan ray_tracing
                - For Linux dockers, the setup is not complete.
                  Install the latest driver, xServer and NVIDIA container runtime.
[Error] [omni.physx.plugin] CUDA libs are present, but no suitable CUDA GPU was found!
```

Carb 의 "Vulkan 1.1 is not supported" 는 잘못 라벨된 fallback 메시지다. 실제 원인은 `vkCreateInstance` 단계에서 **사용 가능한 NVIDIA ICD 가 0개** 라는 점이다.

### 원인

NVIDIA WSL CUDA driver (596.x / 595.x 시리즈) 가 WSL2 Linux 측에 노출하는 라이브러리는 **컴퓨트 전용** 이다. 호스트 `/usr/lib/wsl/lib/` 에 존재하는 것:

| 종류 | 노출됨 | 노출 안 됨 |
|---|---|---|
| Compute | `libcuda.so.1.1`, `libnvidia-ml.so.1`, `libnvidia-encode.so.1`, `libnvidia-opticalflow.so.1`, `libnvidia-ngx.so.1`, `libnvidia-gpucomp.so.<ver>` | — |
| Graphics | — | `libGLX_nvidia.so.0`, `libnvidia-glcore.so.<ver>`, `libEGL_nvidia.so.0`, `libnvoptix.so.<ver>` |

NVIDIA 는 WSL2 의 그래픽스 가속을 **D3D12 경로** (Windows 측 `nvoglv64.dll` + WSL 측 `libd3d12.so` + `/dev/dxg`) 로 설계했고, Linux native Vulkan/OpenGL ICD 는 의도적으로 제공하지 않는다. 따라서:

1. `/etc/vulkan/icd.d/nvidia_icd.json` 의 `library_path: libGLX_nvidia.so.0` 은 컨테이너 안에서 dangling pointer.
2. NVIDIA Container Toolkit 이 `NVIDIA_DRIVER_CAPABILITIES=all` 로도 가져올 graphics 라이브러리가 호스트에 없다 (위 Linux 항목의 `--no-opengl-files` 시나리오와 결과는 같지만, 호스트에 *재설치* 로 채워 넣을 라이브러리 자체가 존재하지 않는다는 점이 다르다).
3. Mesa `dzn` (D3D12-on-Vulkan) 백엔드는 ICD 로딩까지는 성공하지만 `ID3D12DeviceFactory::CreateDevice` 단계에서 `VK_ERROR_INITIALIZATION_FAILED` 로 실패 (WSL D3D12 shim + NVIDIA UMD 결합 문제).
4. Mesa `lavapipe` (CPU) 는 instance 까지는 생성되지만 Isaac Sim 의 `gpu.foundation.plugin` 이 RayTracing 가속 GPU 를 hard requirement 로 가지므로 device 단계에서 거부.

### 해결 방법

**컨테이너 우회. WSL2 또는 Windows 에 Isaac Sim 을 네이티브 설치한다.** 다음 우회로는 모두 *시도했고 실패* 했으므로 같은 함정을 반복하지 말 것:

| 시도한 우회 | 결과 |
|---|---|
| `docker/container.py` 의 `x11.yaml` 그대로 사용 | xauth 가 빈 cookie 반환 → 빈 XAUTHORITY 마운트로 X 인증 자체가 깨짐. WSLg 가 cookie 미사용이라 업스트림 경로 부적합 |
| `/usr/lib/wsl` 을 read-only 로 컨테이너에 마운트 | Toolkit 의 동적 라이브러리 주입을 덮어써 오히려 깨뜨림. 호스트에 graphics 라이브러리 자체가 없으므로 마운트해도 얻을 게 없음 |
| `/tmp/nvidia_icd.json` 수동 작성 + `VK_ICD_FILENAMES` 강제 지정 | ICD JSON 이 가리키는 `libGLX_nvidia.so.0` 자체가 없어 동일 |
| Mesa `lavapipe` ICD 강제 (`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`) | `vkCreateInstance` 통과, "Vulkan 1.1 not supported" 메시지는 사라지지만 device 단계에서 RayTracing 미지원으로 거부 |
| `kisak/kisak-mesa` PPA 로 dzn 포함 mesa 빌드 + 호스트 `libd3d12.so`/`libdxcore.so`/`libnvwgf2umx.so` 컨테이너 복사 | dzn ICD 로딩 성공, 그러나 `ID3D12DeviceFactory::CreateDevice failed → VK_ERROR_INITIALIZATION_FAILED` |
| Isaac Sim 자체 번들 Vulkan loader (`/isaac-sim/extscache/omni.gpu_foundation-*/bin/deps/libvulkan.so.1.3.239`) 우선 사용 | 시스템 loader 와 동일 결과 (ICD 자체가 없으므로 loader 가 무엇이든 무관) |
| `--/rtx/verifyDriverVersion/enabled=false` (NVIDIA 공식 문서 [`docs.omniverse.nvidia.com/.../technical-requirements.html#known-issues-and-limitations`](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html#known-issues-and-limitations) "535.256+ on Vulkan") | 이 워크어라운드는 instance 생성 *후* verify 단계용. 우리는 instance 생성 *이전* 에서 막혀서 무관 |

권장 경로:

- **WSL2 native install** — Docker 우회. WSL2 Ubuntu 에 Isaac Sim 을 직접 설치 (`pip install isaacsim` 류). NVIDIA 가 공식 지원하는 경로다.
- **Windows native install** — WSL 자체 우회. 가장 안정.
- **현 프로젝트 정책 그대로 유지** — `AGENTS.md` 가 명시한 *"시뮬레이션 경로 임시 비활성"* 상태 유지. 활성 워크플로 (`lerobot` 텔레오퍼레이션·데이터수집·SmolVLA 학습) 는 Isaac Sim 을 쓰지 않으므로 영향 없음.

### 확인 방법

이 시스템에서 이 경로가 막혀있는지 빠르게 재확인하는 명령:

```bash
# 호스트 측: WSL2 Linux 에 NVIDIA graphics 라이브러리가 정말 없는지
ls /usr/lib/wsl/lib | grep -E 'libGLX_nvidia|libnvidia-glcore|libEGL_nvidia'
# → 빈 출력이 정상. 이게 빈 출력인 한 컨테이너 우회는 불가능.

# 컨테이너 측: nvidia ICD 가 dangling 인지
docker exec isaac-lab-base bash -lc '
  cat /etc/vulkan/icd.d/nvidia_icd.json &&
  ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia* 2>&1
'
# → ICD 는 libGLX_nvidia.so.0 을 가리키지만 컨테이너 안에 그 파일이 없다고 출력.
```

NVIDIA 가 향후 WSL2 Linux 측에도 Vulkan ICD 를 노출하기로 정책을 바꾸면 (또는 mesa dzn 의 NVIDIA D3D12 호환이 개선되면) 이 항목을 재검토할 수 있다. 그 전까지 시뮬 경로는 native install 로 처리한다.

---

## Windows 네이티브 bare `isaacsim` Full App 이 app ready 직후 종료

**현상**: Windows 네이티브 uv venv 에 Isaac Sim 5.1.0 / Isaac Lab 2.3.0 이 설치된 상태에서 프로젝트 루트의 bare `isaacsim` entrypoint 만 실행하면 `Isaac-Sim Full` GUI 가 로딩 완료 직후 닫힌다.

```powershell
.\.venv\Scripts\isaacsim.exe
```

같은 환경에서 LeIsaac teleop 스크립트는 GUI 를 띄운 채 정상 동작한다.

```powershell
uv run scripts/environments/teleoperation/teleop_se3_agent.py `
  --task=LeIsaac-SO101-PickOrange-v0 `
  --teleop_device=so101leader `
  --port=COM5 `
  --num_envs=1 `
  --device=cuda `
  --enable_cameras
```

**오류 메시지**: Kit 로그는 `app ready` 까지 도달하지만 Windows Application 로그가 RTX scene DB access violation 을 기록한다.

```text
Faulting application name: python.exe
Faulting module name: rtx.scenedb.plugin.dll
Exception code: 0xc0000005
```

### 원인

bare `isaacsim` 은 기본 experience 로 `isaacsim.exp.full.kit` 를 골라 `Isaac-Sim Full` app 을 실행한다. 반면 이 레포의 Isaac Lab 스크립트는 `isaaclab.app.AppLauncher` 로 시뮬레이터를 시작한다. GUI + `--enable_cameras` 조합에서는 AppLauncher 가 Isaac Lab 의 `isaaclab.python.rendering.kit` experience 를 선택하고, 카메라가 없으면 `isaaclab.python.kit` 를 선택한다.

즉 두 명령은 같은 Isaac Sim wheel 을 쓰더라도 같은 app 을 띄우지 않는다. 이 세션에서 확인한 크래시는 Full App 이 `rtx.scenedb.plugin.dll` 을 초기화한 뒤 발생했고, LeIsaac / Isaac Lab task app 경로의 Python import 나 COM teleop 장치 연결 단계에서 발생한 것이 아니다.

같은 Windows 환경에서 신규 viewer 스크립트가 AppLauncher 기본 GUI experience 인 `isaaclab.python.kit` 를 타게 둔 경우도 같은 `rtx.scenedb.plugin.dll` access violation 이 재현됐다. viewer 기본 experience 를 `isaaclab.python.rendering.kit` 로 고정하거나 `--enable_cameras` 로 rendering experience 를 선택하게 하면 URDF import 가 진행되고 GUI 프로세스가 유지됐다.

### 해결 방법

이 레포의 시뮬레이션 GUI 는 bare Full App 대신 Isaac Lab experience 로 띄운다.

```powershell
# 카메라 sensor 를 쓰는 LeIsaac / Isaac Lab rendering GUI
.\.venv\Scripts\isaacsim.exe `
  .\.venv\Lib\site-packages\isaaclab\apps\isaaclab.python.rendering.kit

# 카메라 sensor 없는 기본 Isaac Lab GUI
.\.venv\Scripts\isaacsim.exe `
  .\.venv\Lib\site-packages\isaaclab\apps\isaaclab.python.kit
```

실제 task 를 띄울 때는 해당 스크립트를 계속 사용한다. `teleop_se3_agent.py --enable_cameras` 는 위 rendering experience 선택까지 AppLauncher 가 처리한다.

Full App UI 자체가 필요하면 먼저 사용자 설정을 초기화해 재시도한다. NVIDIA 는 Isaac Sim cache/config 충돌 시 fresh config 와 cache clear 를 점검하라고 안내한다.

```powershell
.\.venv\Scripts\isaacsim.exe --reset-user
```

`--reset-user` 뒤에도 bare Full App 이 같은 `rtx.scenedb.plugin.dll` access violation 으로 죽으면 Full App 의 cache/config 문제를 별도로 추적하고, 레포 작업은 Isaac Lab app 경로로 진행한다.

### 확인 방법

1. `.\.venv\Scripts\isaacsim.exe <isaaclab ... rendering.kit>` 실행 시 로그 폴더가 `Kit\Isaac-Sim\5.1\...` 로 잡히고 GUI 프로세스가 유지되는지 확인.
2. teleop task 는 `--enable_cameras` 를 둔 기존 명령으로 실행해 PickOrange scene 과 camera observation 이 뜨는지 확인.
3. bare Full App 재검증이 필요하면 `Get-WinEvent -LogName Application` 에 새 `rtx.scenedb.plugin.dll` / `0xc0000005` APPCRASH 가 추가되지 않았는지 확인.

---

## Isaac Lab pip 전환 후 `import sim_to_real` 실패

**현상**: `leisaac` 의존성을 제거하고 `isaacsim[all,extscache]==5.1.0` + `isaaclab[all,isaacsim]==2.3.2` 직접 의존으로 전환한 뒤, 서버 uv 환경에서 `import sim_to_real` 이 실패한다.

**오류 메시지**:

```text
ModuleNotFoundError: No module named 'isaaclab.envs'
```

또는 Isaac Sim runtime 초기화 전 직접 import 시 다음 경고 뒤에 실패한다.

```text
WARNING: Omniverse/Isaac Sim import statements must take place after the
`SimulationApp` class has been instantiated.
...
ModuleNotFoundError: No module named 'omni.physics'
# 또는
ModuleNotFoundError: No module named 'omni.timeline'
```

### 원인

Isaac Lab 2.3.2 pip 패키지는 top-level `isaaclab` launcher 패키지와 실제 core package(`isaaclab/source/isaaclab/isaaclab`)가 같은 이름을 쓴다. `isaaclab_tasks` 등 일부 extra path 는 `.pth` 로 노출되지만 core package path 는 일반 import에서 바로 잡히지 않아 `isaaclab.envs` 조회가 실패할 수 있다.

또한 Isaac Lab의 환경/asset 모듈은 `omni.physics` 같은 Kit extension이 로드된 뒤에 import해야 한다. 즉 `SimulationApp` 초기화 전의 bare `import sim_to_real` 은 환경 등록까지 끝내는 smoke로 쓰기 어렵다.

### 해결 방법

`src/sim_to_real/__init__.py` 에서 Isaac Lab pip layout을 감지해 `isaaclab.__path__` 에 core package 경로를 보강한다. T0.2처럼 의존성만 먼저 전환한 단계에서는 아직 남아 있는 `leisaac` import와 Isaac runtime 미초기화(`omni.*`)를 deferred import로 처리한다. 실제 gym 환경 등록과 500-step smoke는 T0.3의 de-leisaac 코드 재작성 후 검증한다.

### 확인 방법

서버에서 project env/cache를 `/DISK1`로 지정하고 확인한다.

```bash
cd /DISK1/so101-sim2real/work/t0.2/repo
UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac \
UV_CACHE_DIR=/DISK1/so101-sim2real/cache/uv \
  /home/konan147/.local/bin/uv run python -c 'import isaacsim; import isaaclab; import sim_to_real; print("ok")'
```

정상 출력:

```text
ok
```

`SimulationApp` 기반 headless import도 exit code 0이면 통과다. 대량의 GLFW/display warning은 headless 서버에서 흔하며, `sim_to_real-ok` 출력과 정상 종료 여부를 기준으로 판단한다.

---

## Isaac Lab SO-101 hold smoke에서 관절 속도 잔류

**현상**: TA.1 drive response smoke에서 zero-action hold 중 관절 위치 편차는 작지만 관절 속도가 계속 남아 `hold.ok=false` 가 된다. stiffness/damping/effort limit을 올려도 수치가 거의 줄지 않는다.

**오류 메시지**:

```json
{
  "status": "failed",
  "hold": {
    "tail_max_abs_pos_rad": 0.03321,
    "tail_rms_vel_rads": 0.10544,
    "final_abs_vel_rads": 0.24793,
    "ok": false
  },
  "step": {
    "final_err_max_rad": 0.03927,
    "overshoot_max_rad": 0.0394,
    "ok": true
  }
}
```

동시에 Isaac Lab이 다음 경고를 낼 수 있다.

```text
WARNING: The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False.
If you are experiencing noisy velocities, consider enabling this flag.
```

### 원인

SO-101 follower가 책상 위 고정 매니퓰레이터인데, USD spawn 설정에서 articulation root를 고정하지 않았다. base link가 미세하게 자유롭게 움직이면 zero-action hold에서도 joint velocity가 계속 남아 PD gain만 조정해서는 안정화되지 않는다.

### 해결 방법

로봇 `UsdFileCfg` 에 `ArticulationRootPropertiesCfg(fix_root_link=True)` 를 명시하고, velocity update 정확도를 위해 velocity solver iteration과 `enable_external_forces_every_iteration` 을 함께 설정한다. Deprecated actuator 필드인 `effort_limit` / `velocity_limit` 대신 Isaac Lab 2.3.2의 `effort_limit_sim` / `velocity_limit_sim` 을 사용한다.

```python
robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=3.0,
            velocity_limit_sim=5.5,
            stiffness=400.0,
            damping=80.0,
        ),
    },
)
```

환경 cfg의 `__post_init__` 에서는 다음도 설정한다.

```python
self.sim.physx.enable_external_forces_every_iteration = True
```

### 확인 방법

서버 Isaac venv에서 TA.1 smoke를 실행한다.

```bash
cd /DISK1/so101-sim2real/work/ta.1/repo
UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac \
UV_CACHE_DIR=/DISK1/so101-sim2real/cache/uv \
  /home/konan147/.local/bin/uv run python scripts/environments/drive_response_smoke.py \
    --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0
```

정상 기준 예시:

```json
{
  "status": "passed",
  "hold": {
    "tail_max_abs_pos_rad": 0.02102,
    "tail_rms_vel_rads": 0.0,
    "final_abs_vel_rads": 0.0,
    "ok": true
  },
  "step": {
    "final_err_max_rad": 0.01882,
    "overshoot_max_rad": 0.01882,
    "ok": true
  }
}
```

---

## Isaac Lab 대규모 PPO에서 `totalAggregatePairsCapacity` 부족

**현상**: TB.3/TB.4 rsl_rl PPO를 2048~4096개 env 이상으로 실행하면 학습은 진행되지만 PhysX가 aggregate pair buffer 부족을 보고한다. 이 상태에서는 일부 contact interaction이 누락될 수 있어 full training 결과를 신뢰하기 어렵다.

**오류 메시지**:

```text
[Error] [omni.physx.plugin] PhysX error: The application needs to increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 18432 , otherwise, the simulation will miss interactions
, FILE /builds/omniverse/physics/physx/source/gpubroadphase/src/PxgAABBManager.cpp, LINE 1291
[Error] [omni.physx.plugin] PhysX error: The application needs to increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to 133918 , otherwise, the simulation will miss interactions
```

### 원인

`PickPenEnvCfg.__post_init__`에서 Isaac Lab manipulation 예제 값을 따라 `gpu_total_aggregate_pairs_capacity = 16 * 1024`로 낮춰 두었다. 2048 env의 SO-101 + 펜 4개 + 펜컵 접촉 조합에서는 aggregate pair 요구량이 약 18k를 넘으므로 16k buffer가 부족하다. PickCube TB.4 4096 env에서는 요구량이 약 134k까지 올라가 128k도 부족했다.

### 해결 방법

대규모 PPO 여유를 위해 `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`와 `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py`에서 total aggregate pair capacity를 256k로 올린다.

```python
self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024
```

4096 env보다 더 키워 같은 오류가 재발하면 같은 항목을 더 크게 잡는다. Isaac Lab 2.3.2 기본값은 더 크지만, task cfg에서 직접 override하면 그 값이 적용된다.

### 확인 방법

서버 Isaac venv에서 2048/4096 env scale smoke를 실행한다.

```bash
cd /DISK1/so101-sim2real/work/ta.3/repo
UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac \
  /home/konan147/.local/bin/uv run --group isaac --locked \
  python scripts/reinforcement_learning/train.py \
    --task SimToReal-SO101-PickPen-v0 \
    --num_envs 2048 --device cuda:0 \
    --max_iterations 2 --num_steps_per_env 24 --save_interval 1 \
    --checkpoint_dir /DISK1/so101-sim2real/outputs/tb3_train_scale_2048x2_cap256k
```

정상 기준: 학습 JSON이 `status=passed`, `total_steps=98304`, checkpoint `model_1.pt`를 출력하고 위 `totalAggregatePairsCapacity` 오류가 더 이상 나오지 않는다.

---

## Isaac Lab `RigidObject` reset sampling이 원점 기준으로 밀림

**현상**: TA.2 `scene_physics_smoke.py`에서 펜과 펜컵이 USD authored 위치가 아니라 origin 주변으로 reset 된다. 펜 spawn 타원과 펜컵 호 sampling 로직은 실행되지만 기준점 자체가 `(0, 0)`이라 y 분리와 영역 검증이 실패한다.

**오류 메시지**:

```json
{
  "config": {
    "default_xy_by_object": {
      "PenWhite": [0.0, 0.0],
      "PenGray": [0.0, 0.0],
      "PenBlack": [0.0, 0.0],
      "PenBlue": [0.0, 0.0],
      "PenCup": [0.0, 0.0]
    }
  },
  "y_separation": {
    "min_spawn_observed_m": -0.04,
    "pass": false
  }
}
```

### 원인

순수 Isaac Lab `RigidObjectCfg(spawn=None)`로 기존 USD prim을 감쌀 때 `init_state`를 명시하지 않으면 `RigidObject.data.default_root_state`가 USD authored transform 대신 원점 pose로 잡힐 수 있다. reset event의 pose randomization은 `default_root_state` 기준으로 offset을 적용하므로 모든 펜과 펜컵 sampling 기준이 origin으로 밀린다.

### 해결 방법

각 펜과 펜컵 `RigidObjectCfg`에 USD authored world-frame pose와 yaw를 `RigidObjectCfg.InitialStateCfg`로 명시한다.

```python
PenWhite: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Scene/PenWhite",
    spawn=None,
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(2.05, -0.35, 0.9347),
        rot=_yaw_quat(25.0),
    ),
)
```

이 값은 scene USD의 펜·펜컵 기본 배치를 바꿀 때 함께 갱신한다.

### 확인 방법

서버 Isaac venv에서 TA.2 smoke를 실행한다.

```bash
cd /DISK1/so101-sim2real/work/ta.2/repo
UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac \
UV_CACHE_DIR=/DISK1/so101-sim2real/cache/uv \
  /home/konan147/.local/bin/uv run python scripts/environments/scene_physics_smoke.py \
    --task SimToReal-SO101-PickPen-v0 --resets 100 --settle-steps 30 --num_envs 1 --device cuda:0
```

`default_xy_by_object`가 `PenWhite=[2.05,-0.35]`, `PenCup=[2.2,-0.17]` 같은 scene 좌표로 출력되고 `spawn_ellipse.pass`, `spawn_arc.pass`, `y_separation.pass`가 모두 `true`이면 정상이다.

---

## Isaac Lab reset 후 원형 펜 collider가 굴러 scene physics smoke 실패

**현상**: 펜 spawn 영역은 맞지만 reset 후 settle 단계에서 펜이 굴러가거나 튀어 `scene_physics_smoke.py`가 실패한다. 일부 run에서는 z 하강, y 분리, 속도 조건이 동시에 깨진다.

**오류 메시지**:

```json
{
  "status": "failed",
  "y_separation": {
    "min_settled_observed_m": -0.7,
    "pass": false
  },
  "physics_stability": {
    "max_z_drop_m": 1.10865,
    "max_lin_vel_ms": 3.2,
    "vel_triggered": true,
    "pass": false
  }
}
```

### 원인

동적 펜의 collision을 visual과 같은 Capsule/Cylinder/Clip 조합으로 두면 작은 원형 물체가 reset 직후 마우스패드 접촉에서 쉽게 rolling/sliding 에너지를 얻는다. Clip 같은 비대칭 collider는 접촉 impulse를 더 불안정하게 만들 수 있다. 단순히 damping만 높이면 한 run은 통과해도 stochastic reset에서 다시 굴러 실패한다.

### 해결 방법

펜 visual과 physics proxy를 분리한다. `Barrel`, `Grip`, `BackPlug`, `Clip`의 `physics:collisionEnabled`를 끄고, root 중심에 얇은 invisible `CollisionBox` 하나만 collision으로 둔다. 동시에 pen rigid body damping/sleep threshold를 reset 안정성에 맞게 높인다.

```usda
float physxRigidBody:angularDamping = 100.0
float physxRigidBody:linearDamping = 5.0
float physxRigidBody:sleepThreshold = 0.05
float physxRigidBody:stabilizationThreshold = 0.05

def Cube "CollisionBox" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
)
{
    token visibility = "invisible"
    bool physics:collisionEnabled = 1
    float3 xformOp:scale = (0.0154, 0.118, 0.0154)
}
```

`.usda`를 수정한 뒤 scene이 참조하는 `.usd` 바이너리를 다시 export한다.

### 확인 방법

TA.2 smoke에서 `physics_stability.pass=true`이고 다음 수준의 여유가 있으면 reset 안정성은 통과다.

```json
{
  "status": "passed",
  "physics_stability": {
    "max_z_drop_m": 0.001,
    "max_xy_drift_m": 0.04419,
    "max_lin_vel_ms": 0.0098,
    "max_ang_vel_rads": 1.13728,
    "pass": true
  }
}
```

회귀 확인으로 `env_smoke.py --steps 500`와 `drive_response_smoke.py`도 이어서 실행한다.

---

## `lerobot record` 키보드 컨트롤이 동작하지 않음 (WSLg + Windows Terminal)

**현상**: `docker compose ... run --rm lerobot record` 실행 후 우측/좌측 화살표·Esc 를 눌러도 에피소드 시작/정지·재녹화·종료가 트리거되지 않는다. 증상은 두 단계로 나타난다.

**증상 ①** — DISPLAY 와 `/tmp/.X11-unix` 가 컨테이너에 노출되지 않은 경우, pynput import 자체가 실패하며 다음 트레이스 + `Switching to headless mode` 가 출력된다.

```
ImportError: this platform is not supported:
('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
```

**증상 ②** — DISPLAY/X11 소켓을 노출시켜 pynput 이 정상 import 된 뒤에도 키 입력이 묵묵부답. 콘솔에는 raw escape sequence (`^[[C` 등) 만 찍힌다.

### 원인

①: `lerobot/utils/control_utils.py` 의 `is_headless()` 는 `import pynput` 성공 여부로 헤드리스 환경을 판별한다. 컨테이너에 `DISPLAY` 가 없거나 `/tmp/.X11-unix` 가 마운트되지 않으면 import 가 실패 → `is_headless()` 가 `True` → `init_keyboard_listener()` 가 `None` 리스너를 반환.

②: WSLg 의 X 서버는 X11 윈도우로부터 들어온 키 이벤트만 본다. **Windows Terminal 은 X11 클라이언트가 아니라 Windows 네이티브 콘솔**이라, 거기서 누른 키는 X 서버를 거치지 않고 Windows 와 그 자식 (WSL → docker → 컨테이너 PTY) 으로만 흘러간다. pynput 의 X RECORD 리스너는 X 서버 측 이벤트만 듣기 때문에 이 키들을 영원히 보지 못한다.

### 해결 방법

두 단계로 나눠 적용한다.

**① docker-compose 에 X11 노출** (`docker/docker-compose.yaml`, `lerobot` 서비스):

```yaml
    volumes:
      ...
      # X11 소켓 — pynput import 시 X 연결 실패를 막기 위해 마운트
      - /tmp/.X11-unix:/tmp/.X11-unix
    environment:
      NVIDIA_VISIBLE_DEVICES:     all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
      DISPLAY: ${DISPLAY:-:0}
```

이것만으로는 ② 가 해결되지 않으니 동시에:

**② 컨테이너 안에 stdin 기반 키보드 리스너 패치 베이크 인** (`docker/Dockerfile.lerobot`):

```dockerfile
COPY docker/lerobot_keyboard_stdin.py /opt/venv/lib/python3.11/site-packages/lerobot_keyboard_stdin.py
COPY docker/lerobot_keyboard_stdin.pth /opt/venv/lib/python3.11/site-packages/lerobot_keyboard_stdin.pth
```

패치 모듈은 `/dev/tty` 를 cbreak 모드로 열어 docker PTY 로 흘러온 raw escape sequence (`\x1b[C`/`\x1b[D`/`\x1b`) 를 읽어 lerobot 이 기대하는 `{exit_early, rerecord_episode, stop_recording}` 이벤트 딕셔너리를 그대로 토글한다. `.pth` 파일이 Python 시작 시 `install_hook()` 을 호출, `lerobot.utils.control_utils` 가 import 되는 순간 `init_keyboard_listener` 를 stdin 버전으로 교체한다.

패치 적용 후 이미지를 재빌드해야 한다.

```bash
docker compose -f docker/docker-compose.yaml build lerobot
```

### 확인 방법

```bash
# 1. 패치 모듈이 이미지에 들어갔는지 확인
docker compose -f docker/docker-compose.yaml run --rm --no-deps --entrypoint python lerobot \
  -c "import lerobot.utils.control_utils as cu, lerobot_keyboard_stdin; \
      print(cu.init_keyboard_listener is lerobot_keyboard_stdin.init_keyboard_listener_stdin)"
# → True

# 2. record 실행 → 첫 에피소드 진행 중 우측 화살표 →
#    'Right arrow key pressed. Exiting loop...' 가 콘솔에 출력
docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot record
```

stdin 패치가 X 의존성을 완전히 우회하므로 WSLg 가 아닌 헤드리스 Linux 서버 (디스플레이 없음) 에서도 동일하게 동작한다. ① 의 docker-compose X11 노출은 pynput import 자체가 시작 시 트레이스를 뱉지 않게 하는 안전망 역할만 한다 (없어도 패치는 동작하지만 헤드리스 폴백 메시지가 한 번 찍힘).

---

## SmolVLA fine-tune 추론 시 카메라 키 불일치 (KeyError: observation.images.camera1)

**현상**: SmolVLA fine-tune 모델을 async policy-client 로 추론하면 follower 가 전혀 움직이지 않고, policy server 가 첫 관측부터 계속 같은 에러를 반복한다.

**오류 메시지** (policy server 로그):

```
INFO  y_server.py:226 Running inference for observation #0 (must_go: True)
ERROR y_server.py:266 Error in StreamActions: 'observation.images.camera1'
```

### 원인

`lerobot/smolvla_base` 의 config 가 입력 키로 `camera1/camera2` 을 명시한다. `--policy.path` 로 fine-tune 하면 `make_policy` 가 pretrained 의 `input_features` 를 데이터셋 키로 덮어쓰지 않으므로(자동 rename 아님), SO-101 데이터셋(`top/wrist`)을 매핑하는 `--rename_map` 을 직접 줘야 한다(`env/smolvla.env` 의 `RENAME_MAP`, 논문 표준 순서 `top→camera1, wrist→camera2`). 누락 시 학습 단계에서 feature mismatch 로 실패하고, 결과 체크포인트의 `input_features` 는 **`camera1/camera2`** 로 굳는다.

한편 async `robot_client`(0.4.4)는 `RemotePolicyConfig` 의 `rename_map` 을 채우지 않아(빈 dict) 서버 측 rename 이 무력화된다. 추론 파이프라인 1단계(`helpers.py:prepare_raw_observation`)가 클라이언트가 보낸 카메라 키로 `policy_image_features[key]` 를 직접 조회하는데, 이 조회는 preprocessor(rename 단계)보다 **먼저** 실행된다. 따라서 클라가 보낸 카메라 키(`--robot.cameras` 의 dict key)와 모델 `input_features` 키가 글자 그대로 일치하지 않으면 `KeyError`. 수집용 `top/wrist` 키를 그대로 추론에 넘기면 모델이 기대하는 `camera1` 을 못 찾는다.

추가로, 추론 모델을 가리키는 변수(`--pretrained_name_or_path`)가 fine-tune 결과가 아닌 다른 레포를 가리켜도 동일 증상이 난다.

### 해결 방법

1. 추론 클라의 `--robot.cameras` **키를 모델 `input_features` 와 일치**시킨다. SmolVLA fine-tune 이면 `camera1/camera2/camera3`, 물리 매핑은 `rename_map` 대로 `camera1=top, camera2=wrist, camera3=front`:

```bash
--robot.cameras="{
    camera1: {type: opencv, index_or_path: ${TOP_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
    camera2: {type: opencv, index_or_path: ${WRIST_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
    camera3: {type: opencv, index_or_path: ${FRONT_CAM_PORT}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, fourcc: ${CAM_FOURCC}},
}"
```

2. `--pretrained_name_or_path`(= `.env` 의 `POLICY_REPO_ID`)가 **fine-tune 결과 모델**을 가리키는지 확인한다. 베이스(`lerobot/smolvla_base`)나 다른 레포면 키가 어긋난다.

### 확인 방법

모델이 기대하는 카메라 키를 직접 출력해 클라 키와 대조:

```bash
uv run python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; p=SmolVLAPolicy.from_pretrained('<hf_user>/<model>'); print(list(p.config.image_features))"
# → ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
```

정상 동작 시 policy server 로그에 `Action chunk #N generated` 가 찍히고 follower 가 움직인다.

---

## 카메라 sensor 가 raytracing pipeline 생성 실패 (RT 코어 없는 GPU)

> ⚠ **H100/A100은 Isaac Sim 5.1 공식 미지원이다.** NVIDIA 공식 [System Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)가 다음과 같이 명시:
> > *"GPUs without RT Cores (A100, H100) are not supported."*
>
> 즉 H100/A100은 시스템 요구사항 단계부터 제외되어 있고, 아래 증상은 그 결과물이다. 워크어라운드를 찾기보다 GPU를 교체하는 게 정답.

**현상**: 위 Vulkan 문제를 해결한 뒤 (`Driver Version: ... | Graphics API: Vulkan` 가 정상 출력되고 `Streaming server started.` 까지 도달) 그 직후, 환경 초기화 단계에서 다음 트레이스로 컨테이너가 즉시 종료된다.

**오류 메시지**:

```log
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INITIALIZATION_FAILED
[Error] [carb.graphics-vulkan.plugin] vkCreateRayTracingPipelinesKHR failed.
[Error] [omni.physx.fabric.plugin] CUDA error: an illegal memory access was encountered:
                                   .../DirectGpuHelper.cpp: 563

Traceback (most recent call last):
  File ".../teleop_se3_agent.py", line 226, in main
    env = gym.make(task_name, cfg=env_cfg).unwrapped
  File ".../isaaclab/envs/mdp/observations.py", line 404, in image
    images = sensor.data.output[data_type]
  File ".../isaaclab/sensors/sensor_base.py", line 362, in _update_outdated_buffers
    self._is_outdated[outdated_env_ids] = False
RuntimeError: CUDA error: an illegal memory access was encountered
```

### 원인

NVIDIA가 시스템 요구사항 문서에서 H100/A100을 미지원으로 명시한 이유와 정확히 일치하는 메커니즘이다. 데이터센터 GPU인 **NVIDIA H100 / A100 (Hopper / Ampere-DC)** 은 **RT 코어를 탑재하지 않는다**. RT 코어는 RTX A/L 워크스테이션 시리즈와 GeForce RTX, 그리고 일부 데이터센터 GPU (L40 / L40S / A40 / RTX 6000 Ada) 에만 있다.

Isaac Sim 5.1 의 카메라 sensor (`isaaclab.sensors.camera.Camera` / `TiledCamera`) 는 무조건 RTX renderer (`RaytracedLighting` / `PathTracing`) 로 동작하도록 강제되어 있다 (`isaaclab/sensors/camera/camera_cfg.py:64`, `isaaclab/apps/isaaclab.python.rendering.kit:50-71` 에 raytracing 비활성화 옵션 부재). 그래서 RT 코어 없는 GPU 에서는 다음 흐름으로 죽는다:

1. `--enable_cameras` 로 카메라 sensor 등록
2. RTX renderer 가 `vkCreateRayTracingPipelinesKHR` 호출 → `ERROR_INITIALIZATION_FAILED`
3. `omni.physx.fabric` 가 비어 있는/유효하지 않은 GPU 버퍼를 참조 → CUDA illegal memory access
4. observation manager 가 `sensor.data.output[...]` 접근 → 이미 corrupt 된 CUDA context 라 `RuntimeError`

CUDA 자체는 정상이고 (`nvidia-smi` 에서 컨테이너의 python 프로세스가 GPU 메모리 점유), GPU 가 두 장 모두 인식되며 livestream 서버까지 정상 기동한 뒤 발생하기 때문에 위쪽 Vulkan 섹션의 증상과는 구분된다.

GPU 별 RT 코어 유무 빠른 가이드 (NVIDIA 공식 시스템 요구사항 기준):

| GPU | 아키텍처 | RT 코어 | Isaac Sim 5.1 지원 |
|------|---------|---------|------|
| H100 / H200 | Hopper | ✗ | **NVIDIA 공식 미지원** (문서 명시) |
| A100 | Ampere-DC | ✗ | **NVIDIA 공식 미지원** (문서 명시) |
| L40 / L40S / L4 | Ada-DC | ✓ | 동작 |
| A40 / A30 | Ampere-DC (visualization) | ✓ | 동작 |
| RTX A4000 / A5000 / A6000 | Ampere | ✓ | 동작 (RT 코어·16GB VRAM 충족) |
| RTX 6000 Ada / 5000 Ada | Ada | ✓ | 동작 |
| GeForce RTX 4080 (최소) / 5080 (양호) / PRO 6000 Blackwell (이상적) | 컨슈머·Pro | ✓ | NVIDIA **권장** |
| GeForce RTX 30 시리즈 | Ampere | ✓ | 권장 라인업 미만이지만 RT 코어·16GB(3080 12GB는 미달) 충족 시 동작 |


---

## Isaac Lab `RigidObject` spawn 에서 parent prim 경로 누락

**현상**: `InteractiveScene` 에 동적 물체를 추가한 뒤 scene 생성 단계에서 GUI 가 ready 로그까지 가지 못하고 `RigidObjectCfg` spawn 이 즉시 실패한다. 예를 들어 `prim_path="{ENV_REGEX_NS}/Pens/white_pen"` 처럼 아직 존재하지 않는 중간 그룹 prim 을 포함한 경로에서 재현된다.

**오류 메시지**:

```text
RuntimeError: Unable to find source prim path: '/World/envs/env_.*/Pens'.
Please create the prim before spawning.
```

### 원인

Isaac Lab shape spawner 는 leaf prim 은 만들지만 `RigidObjectCfg.prim_path` 의 미존재 parent prim 까지 자동으로 author 하지 않는다. USD scene 이 `/Pens` prim 을 먼저 만들지 않은 상태에서 regex env path 아래 자식 rigid object 를 바로 spawn 하려고 하면 source parent lookup 이 실패한다.

### 해결 방법

둘 중 하나로 경로 소유권을 명확히 한다.

1. scene USD 나 setup 코드에서 `{ENV_NS}/Pens` parent prim 을 먼저 author 한 뒤 자식 rigid object 를 spawn 한다.
2. 그룹 prim 이 꼭 필요하지 않으면 `prim_path="{ENV_REGEX_NS}/white_pen"` 처럼 이미 존재하는 env root 바로 아래에 동적 물체를 둔다.

`Sim-to-Real` 펜 task 는 이후 LeIsaac scene 방식으로 옮겨져, 펜 prim 을
`assets/scenes/so101_pick_pen/pick_pen_scene.usd` 안에 author 하고
`parse_usd_and_create_subassets()` 로 등록한다. 코드 shape spawner 로 다시
되돌릴 때는 위 parent prim 규칙을 지켜야 한다.

### 확인 방법

```powershell
uv run scripts\view_pick_pen_scene.py
```

stdout 에 `[INFO]: SO-101 pen Pick-and-Place scene is ready.` 가 찍히고 desk scene 의 펜들이 나타나면 parent prim 경로 문제는 해결된 상태다.

---

## Sim-to-Real 펜이 그리퍼에 잡히지 않음 (USD Cube scale + 얇은 code-spawn pen)

**현상**: `scripts/record_pick_pen.py` 의 초기 pen scene 에서
SO-101 그리퍼를 내려 펜을 닫아도 펜이 잡히지 않는다. 책상 면에 닿는 높이도
직관과 어긋나 보여 robot zero 가 잘못된 것처럼 보인다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서는 얇은 pen proxy 가 책상 면과 겹치거나 stable pinch contact 를 만들지 못한다.
```

### 원인

초기 authored table USD 는 `UsdGeomCube` 의 `xformOp:scale` 값을 치수처럼
썼지만 Cube 기본 size 는 2 다. 예를 들어 z scale `0.04` desk top 은 실제로
두께 `0.08` 이 되어 의도한 작업면 `z=0` 보다 위로 올라간다. 동시에 펜은 코드
`CapsuleCfg` 로 반지름 `6.5 mm`, center z `0.014` 에 따로 spawn 되어 desk/mat
collision 과 겹치기 쉬웠고 SO-101 finger mesh 가 안정적으로 집을 폭도 작았다.

SO-101 의 reach 자체가 문제였던 것은 아니다. joint-limit sample 에서
`gripper`/`jaw` body origin 은 작업면 아래(`z=-0.1325` 샘플)까지 내려간다.
다만 local runtime follower USD 는 asset root `z=0` 에서 base visual bound 의
최저점이 `z=0.030081` 이므로 table scene 에서는 별도 base-surface offset 도
맞춰야 한다.

### 해결 방법

LeIsaac `PickOrange` 방식으로 scene 소유권을 바꾼다.

1. 책상, 매트, 컵, 펜 rigid bodies 를 하나의 USD scene 에 author 한다.
2. Cube prim 은 `size = 1` 을 명시해 authored scale 과 실제 치수를 맞추고
   desk surface 를 `z=0` 으로 둔다.
3. 펜은 scene USD 의 `PhysicsRigidBodyAPI` + `PhysicsCollisionAPI` capsule
   subasset 으로 두고, 위에서 pinching 가능한 marker-size barrel 로 만든다.
4. env cfg 에서는 LeIsaac 와 같이 `parse_usd_and_create_subassets()` 로 pen
   rigid prim 을 Isaac Lab reset/recorder manager 에 등록한다.

현재 구현:

- USD scene: `assets/scenes/so101_pick_pen/pick_pen_scene.usd`
- USD load + subasset 등록: `src/sim_to_real/scenes/pick_pen_scene.py`

### 확인 방법

```powershell
uv run scripts\record_pick_pen.py `
  --teleop_device so101leader `
  --port COM5 `
  --record `
  --dataset_file outputs\datasets\so101_pick_pen_contact_check.hdf5
```

task 기동 후 env 진단에서 rigid objects 가 `PenWhite`, `PenGray`, `PenBlack`,
`PenBlue` 로 등록되고 local follower 의 base 가 desk surface 에 맞춰져 있으면
USD subasset/zero 정렬은 맞다. GUI 에서 `B` 로 control 을 시작한 뒤 pen
barrel 을 위에서 감싸도록 jaw 를 정렬해 닫아 contact 가 생기는지 확인한다.

---

## Sim-to-Real USD 펜이 관통하며 미끄러짐 (pen contact tuning)

**현상**: LeIsaac 방식의 authored USD pen scene 으로 옮긴 뒤에도 SO-101
그리퍼로 pen barrel 을 닫을 때 표면에서 바로 버티지 못하고 약간 관통하거나
고무처럼 밀렸다 튀는 느낌이 난다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 pen visual 과 jaw 가 겹쳐 보이고 pinch 중 pen 이 쉽게 밀려난다.
```

이 증상을 줄이려고 scene/robot 전체 `UsdFileCfg` 에 collision modifier 를
덮어쓴 실험 경로에서는 smoke run 에서 다음 로그도 확인됐다.

```text
[Warning] [isaaclab.sim.utils] Could not perform 'modify_collision_properties' on any prims under: '/World/envs/env_0/Robot'.
[Error] [omni.physx.plugin] PhysX error: Fetching GPU Narrowphase failed! 700
```

### 원인

pen visual 은 barrel/tip/clip 으로 세분화되어 있지만 실제 접촉은 scene USD 의
단일 capsule collider 가 담당한다. 이 collider 가 rigid body 여도 기본 contact
offset, solver iteration, friction 만 쓰면 작은 cylindrical object 를 SO-101 jaw
mesh 사이에서 집을 때 surface contact 가 늦게 풀리거나 미끄러짐이 두드러질 수
있다.

scene spawn 이나 `SO101_FOLLOWER_CFG` 의 `UsdFileCfg.collision_props` 로
collision 설정을 전체 USD 에 덮는 방식도 적절하지 않다. SO-101 runtime USD 의
jaw/gripper collision prim 은 instanced prim 이라 Isaac Lab modifier 가 적용되지
않고, desk/cup 전체 collider 까지 같은 PhysX contact 튜닝 범위에 들어가 spawn
범위만 커진다.

### 해결 방법

1. pen root 는 USD 에 `PhysicsRigidBodyAPI`, `PhysicsMassAPI`,
   `PhysxRigidBodyAPI` 를 author 하고 gravity 를 켠 dynamic rigid body 로 둔다.
2. pinch 를 담당하는 invisible capsule collider 에만 `PhysxCollisionAPI` 를
   추가해 `contactOffset=0.0015`, `restOffset=0`, torsional patch radius 를
   명시한다.
3. pen collider 에 `PenGripPhysics` physics material 을 bind 해 static/dynamic
   friction 을 높이고 restitution 은 0 으로 둔다.
4. pen rigid body 에 CCD 와 solver position/velocity iteration count 를 author
   한다. env 기본 physics material 도 같은 high-friction 방향으로 맞춘다.
5. desk/cup/robot 전체에 collision modifier 를 덮지 않고 pen contact tuning 은
   `scripts/author_pick_pen_scene.py` 의 pen collider authoring 에
   국한한다.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
uv run scripts\record_pick_pen.py --teleop_device keyboard --max_loops 1 --headless
```

smoke run 이 종료된 뒤 generated USD 에서 `PenWhite/Collision` 같은 pen
collider 만 `PhysxCollisionAPI`, `physxCollision:contactOffset`,
`material:binding:physics = </Scene/Looks/PenGripPhysics>` 를 가진다. GUI
recording 에서는 jaw 를 barrel 양옆에 맞추고 닫았을 때 pen 이 visual 중심까지
관통하지 않고 capsule surface 에서 미끄러짐이 줄어드는지 확인한다.

---

## Sim-to-Real SO-101 base 가 desk 위에서 떠 보임 (mat 배치)

**현상**: `Sim-to-Real` pen scene GUI 에서 fixed SO-101 base 아래로 그림자
간격이 도드라져 로봇이 검은 작업면 위에 떠 있는 것처럼 보인다. 실제 촬영 장면은
SO-101 base 가 desk 전면의 나무 상판에 놓이고 mat 는 pens 쪽으로 뒤에서
시작한다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 SO-101 base 지지면이 검은 DeskMat 로 보이고 base 가 떠 보인다.
```

### 원인

초기 pen scene 은 robot root 와 desk surface 를 모두 `z=0` 으로 두었지만
runtime follower USD 의 base visual bound 최저점은 asset root 기준
`z=0.030081` 이다. local URDF source 의 base bound 만 보고 USD 도 같은
surface origin 이라고 가정하면 fixed robot 이 약 3 cm 뜬 채 배치된다.
`DeskMat` 도 root 아래까지 펼쳐 둔 상태라 실제 사진과 다른 mat overlap 과 RTX
shadow 가 간격을 더 도드라지게 만든다.

### 해결 방법

1. desk surface 는 계속 `z=0` 으로 둔다.
2. `SO101_FOLLOWER_CFG` fixed root 에 `-0.0301 m` base-surface z offset 을
   적용해 authored USD base 최저점을 desk surface 에 맞춘다.
3. `DeskMat` 의 전면 edge 를 robot base 뒤로 밀어 실제 사진처럼 base 아래에
   bare desk top 이 보이게 한다.
4. 컵의 perforated render mesh 와 안정적인 collision wall, 펜의 visual detail 과
   capsule collider 를 `scripts/author_pick_pen_scene.py` 에서
   분리 author 한다.

현재 생성 USD 는
`assets/scenes/so101_pick_pen/pick_pen_scene.usd` 이다.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
uv run scripts\view_pick_pen_scene.py
```

GUI 에서 arm base 아래 지지면이 desk wood 로 보이고, pen cup wall 은 구멍이
보이는 wire mesh 이며 pens 는 barrel/tip/clip detail 을 유지하면 scene authoring
배치가 반영된 상태다.

---

## Sim-to-Real 씬이 로봇 위치와 어긋남 (scene origin shift)

**현상**: `teleop_se3_agent.py` 로 `SimToReal-SO101-PickPen-v0` 를 띄우면 책상,
마우스패드, 펜통, 펜이 origin (0, 0, 0) 부근에 모여 있고 SO-101 follower 는
2 m 떨어진 위치에서 공중에 떠 있는 것처럼 보인다. y 또는 z 축만 어긋난 경우
로봇이 책상 옆이나 책상 위 허공에 떠 있는 형태로도 나타난다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 robot 과 desk 가 서로 다른 영역에 떨어져 렌더링된다.
```

### 원인

`SO101_FOLLOWER_CFG.init_state.pos = (2.2, -0.61, 0.89)` 으로 환경 컨피그가
follower 를 절대 위치에 스폰하지만 `assets/scenes/pen_desk/scene.usd` 는
origin 기준 좌표로 author 되어 있다. 환경이 scene USD 와 robot USD 를 같은
world frame 으로 합치므로 두 좌표계가 일치하지 않으면 둘이 떨어진 채 보인다.
또한 desk top 의 z 가 robot base z 와 같으면 RTX shadow 한 픽셀 차이로 robot
이 떠 보일 수 있어 약간의 z slack 이 필요하다.

### 해결 방법

`scripts/author_pick_pen_scene.py` 에 `SCENE_OFFSET` 상수를 두고 모든 top-level
translate 를 `_shift()` 헬퍼로 한 번에 옮긴다. scene.usd 의 자식 prim 상대
좌표는 보존하고 부모만 시프트한다.

```python
# robot base = (2.2, -0.61, 0.89)
# desk front edge ≈ robot.y  → clamp 위치
# desk top z = robot.z + 0.03 ~ 0.05  → 떠 보임 방지
SCENE_OFFSET = (2.2, -0.57, 0.92)
```

값은 시각 확인을 통해 미세조정한다. 책상 정면 가장자리가 robot.y 와 같으면
로봇이 클램프 위치, robot.y 보다 +y 로 멀어지면 로봇이 책상 중앙이다.

`mdp.pen_in_cup` 같이 좌표를 직접 비교하는 task 로직이 있다면 함께 갱신한다.

- `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py` 의 `PEN_CUP_CENTER_XY`
- `src/sim_to_real/datagen/state_machine/pick_pen.py` 의 같은 상수

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
C:\OpenUSD\scripts\usdcat.bat --loadOnly assets\scenes\pen_desk\scene.usd
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

GUI 에서 robot mount 가 책상 정면 모서리에 클램프된 모습으로 보이고 펜과
펜통이 robot 의 전방 reach 안에 들어와 있으면 정합된 상태다.

---

## Sim-to-Real 에피소드 리셋 시 펜이 한 번 튀어오름 (mat z slack)

**현상**: `B` 로 에피소드를 시작하면 펜 중 하나가 한 번 펄떡 튀어 오르며 그 후
정상 안착한다. 사용자 입력 없이도 재현된다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 reset 직후 한 펜이 짧게 0.5 ~ 1 cm 솟았다 떨어진다.
```

### 원인

펜 collider box 의 z half-extent 와 마우스패드 윗면 z 가 부동소수점 오차
범위에서 정확히 같거나 살짝 겹치도록 author 되어 있으면, PhysX 는 reset 시
contact penetration 을 한 step 에 풀려고 impulse 를 가한다. 펜 4 개 중 가장
penetration 이 큰 한 개가 이 impulse 로 튀어오르고 나머지는 안 튀는 식으로
보인다.

### 해결 방법

`PENS` 튜플의 z 를 `mat_top + collider_half_thickness + 0.001 m` 로 두어
1 mm 의 slack 을 확보한다.

```python
# mat top z = 0.006, collider half-thickness = 0.0077 → 0.0137, slack 1 mm
PENS = (
    ("PenWhite", (-0.20, 0.05, 0.0147), 25.0, ...),
    ...
)
```

mat 또는 collider 두께를 바꿀 때마다 z 도 같이 갱신해야 같은 증상이 재발하지
않는다.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

`B` 를 눌러 reset 한 직후 펜 4 개가 모두 mat 표면에 안정적으로 놓이고 튀어오름
이 없으면 slack 이 충분한 상태다.

---

## Sim-to-Real 펜이 닿지 않았는데 그리퍼가 잡음 (pen collider 부풀림)

**현상**: SO-101 그리퍼가 펜 visual 에서 1 cm 가까이 떨어진 채로 jaw 를 닫아도
펜이 잡힌다. GUI 에서는 jaw 와 펜 시각 표면 사이에 명백한 공간이 있다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 그리퍼가 펜 옆/위에서 closing 했는데 펜이 finger 위로 살짝 떨어진 채
끌려간다.
```

### 원인

`scripts/author_pick_pen_scene.py` 의 펜 collider 가 visual capsule 보다
부풀려 author 되어 있었다.

| | 두께 (X/Z) | 길이 (Y) |
|---|---|---|
| Visual capsule | 0.0154 m (= 2 × radius 0.0077) | 0.1334 m (= height 0.118 + 2 × radius) |
| Collider box (이전) | 0.0184 m | 0.1504 m |

길이 축으로 약 1.7 cm, 두께 축으로 약 3 mm 더 큰 보이지 않는 box 가 펜 위로
튀어나와 있어 그리퍼가 시각 표면에 닿기 전에 contact 가 trigger 된다.

### 해결 방법

collider box 크기를 visual capsule 과 동일하게 맞춘다.

```python
PEN_BARREL_RADIUS = 0.0077
PEN_BARREL_HEIGHT = 0.118
PEN_COLLIDER_LENGTH = PEN_BARREL_HEIGHT + 2 * PEN_BARREL_RADIUS   # 0.1334
PEN_COLLIDER_THICKNESS = 2 * PEN_BARREL_RADIUS                    # 0.0154
```

`PEN_COLLIDER_THICKNESS` 가 바뀌면 펜 안착 높이의 collider half-thickness 도
같이 변하므로 `PENS` z 값을 `mat_top + thickness/2 + 0.001` 로 재계산한다.

펜의 grip / accent ring 같은 부속 부품이 capsule 보다 약간 굵어도 (예:
AccentRing radius 0.0083), 사용자가 시각 일치를 우선시했으므로 collider 는
capsule 두께에만 맞춘다. 잡는 위치가 너무 좁다고 느껴지면 thickness 를 굵은
부품 기준으로 조금 늘리는 방식으로 trade-off 한다.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

그리퍼 jaw 를 펜 barrel 옆면에 시각적으로 닿도록 정렬한 뒤 `O` 로 닫았을 때
잡히고, 1 cm 떨어진 위치에서는 닫아도 잡히지 않으면 collider 가 visual 에
일치한 상태다.

> **후속**: 위 해결은 collider Cube 의 *치수* 만 맞춘 v2. 사각형 collider 가 둥근
> capsule 끝부분을 표현 못 해 여전히 visual 과 어긋남이 남는다. v3 해결책은
> 아래 *"펜 collision 형상이 visual 과 어긋남"* 항목 참고.

---

## Sim-to-Real 펜 collision 형상이 visual 과 어긋남 (Cube collider → visual primitive)

**현상**: 펜 collider Cube 의 크기를 visual capsule 과 동일하게 맞춘 뒤에도,
펜의 둥근 끝부분에서 그리퍼가 사각 모서리를 따라 접촉하거나, capsule 본체보다
약간 굵은 Grip 부분 표면에서 시각적으로 닿는데도 잡히지 않는 경우가 남는다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 펜 capsule 둥근 끝은 부드럽게 보이지만 그리퍼 접촉은 사각 박스 모서리를
따라 발생. 굵은 grip 부분에서는 시각 접촉 대비 contact 가 늦게 trigger.
```

### 원인

`scripts/author_pick_pen_scene.py` 의 펜 author 가 visual primitive (Capsule
barrel, Cylinder grip 등) 와 별도로 invisible `Cube "Collision"` 을 두고 그
사각 박스 하나로 모든 contact 를 처리했다. Cube 는 capsule 의 둥근 끝과
Grip / BackPlug / Clip 의 굵은 부분을 모두 단일 단순 박스로 뭉뚱그린다.

SO-101 robot USD (`assets/robots/so101_follower.usda`) 는 같은 문제를 visual
mesh 를 그대로 collider 로 재사용해 해결한다:

```text
def Xform "collisions" (
    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxMeshMergeCollisionAPI"]
    prepend references = </colliders/base>      # visual mesh 와 동일
)
{
    uniform token physics:approximation = "convexDecomposition"
}
```

### 해결 방법

1. invisible `Cube "Collision"` 제거.
2. 외곽선을 만드는 각 visual primitive (Capsule barrel, Cylinder grip,
   Cylinder backplug, Cube clip) 에 직접 `PhysicsCollisionAPI` +
   `PhysxCollisionAPI` 를 부여. USD analytic primitive 는 PhysX 가 네이티브
   지원하므로 mesh approximation 불필요.
3. Cone primitive (TipSleeve, Nib) 에는 collision 부여 금지. PhysX 가 analytic
   cone 을 지원하지 않아 silently coarse convexHull 로 fallback → 형상이 어긋남.
4. 굴러감 방지는 Clip Cube 가 자연스럽게 담당 (외부로 0.0065 m 돌출 → 실제 펜
   클립과 같은 원리). 이전 invisible 박스의 stopper 효과를 자연스럽게 대체.

`scripts/author_pick_pen_scene.py::author_pen_usda` 에서:

```python
_capsule(
    lines, 1, "Barrel", radius=PEN_BARREL_RADIUS, height=PEN_BARREL_HEIGHT,
    material_path=barrel_path, collision=True,
    physics_material_path=grip_phys_path, contact_tuning=True,
)
_cylinder(lines, 1, "Grip", axis="Y", radius=0.0081, height=0.025, ...,
          collision=True, physics_material_path=grip_phys_path, contact_tuning=True)
# Clip Cube 와 BackPlug Cylinder 동일. AccentRing/TipSleeve/Nib 은 visual-only.
```

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
```

생성된 펜 USD 에서 `Cube "Collision"` 이 사라지고, 각 펜 객체마다 4 개의
`PhysicsCollisionAPI` 가 visual primitive (Barrel/Grip/BackPlug/Clip) 에
부여돼 있으면 v3 패턴이 적용된 상태:

```text
Grep "PhysicsCollisionAPI" assets/scenes/pen_desk/objects/PenWhite/PenWhite.usda  # → 4 occurrences
```

GUI 에서 그리퍼가 펜 barrel 둥근 끝에 접근할 때 사각 모서리가 아닌 곡면을 따라
접촉하고, 굵은 Grip 부분에서 시각 표면과 동시에 잡히면 정상.

---

## Sim-to-Real B/R 리셋 후 동적 RigidBody 가 이전 위치 유지 (env subasset 등록 누락)

**현상**: PenCup 처럼 동적 RigidBody 로 author 된 객체가 GUI 에서 `B`/`R` 키로
에피소드를 리셋해도 author 한 초기 위치로 돌아가지 않고 이전 에피소드 끝
지점에 그대로 머문다. 펜은 정상적으로 초기 위치로 복원된다.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 PenCup 만 매 reset 마다 이전 위치 유지. 펜은 정상 복원.
```

### 원인

`leisaac.utils.general_assets.parse_usd_and_create_subassets()` 는 인자로 받은
`specific_name_list` 와 prim path 가 매칭되는 RigidBody 만 env 의 RigidObject
슬롯으로 등록한다. 등록되지 않은 RigidBody 는 시뮬레이션 자체는 정상 동작
하지만 Isaac Lab 의 event manager 가 그 객체의 root state 를 모르기 때문에
reset 이벤트가 걸리지 않는다.

```python
# 기존 — PenCup 누락
parse_usd_and_create_subassets(SCENE_USD_PATH, self, specific_name_list=PEN_NAMES)
```

펜은 등록되어 있고 `randomize_object_uniform` reset 이벤트가 걸려 있어 매
reset 마다 `default_root_state + sampled_pose` 로 복원된다. PenCup 은 슬롯
자체가 없어 event 등록 시점에 `SceneEntityCfg("PenCup")` lookup 이 실패하거나
조용히 무시된다.

### 해결 방법

두 가지 모두 해야 한다:

1. `specific_name_list` 에 PenCup 추가 → RigidObject 슬롯 생성.
2. `randomize_object_uniform(PEN_CUP_NAME, range=(0,0))` reset 이벤트 추가 →
   매 reset 마다 author 위치로 복원 (range=(0,0) 이면 랜덤화 없이 default
   pose 그대로 복원).

```python
# src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py
parse_usd_and_create_subassets(
    PEN_DESK_USD_PATH, self,
    specific_name_list=[*PEN_NAMES, PEN_CUP_NAME],
)

domain_randomization(self, random_options=[
    *[randomize_object_uniform(name, pose_range={"x": (-0.03, 0.03), ...})
      for name in PEN_NAMES],
    randomize_object_uniform(
        PEN_CUP_NAME,
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    ),
    ...
])
```

### 확인 방법

```powershell
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

`B` 로 제어를 시작한 뒤 펜컵을 그리퍼로 옆으로 밀고 `R` 또는 `N` 으로 리셋.
펜컵이 author 한 초기 위치로 돌아오면 정상.

---

## Sim-to-Real 그리퍼·펜이 매트/책상을 관통하거나 reset 시 튀어오름 (정적 객체 contactOffset 디폴트)

**현상**: Pick 동작 중 그리퍼나 펜이 가끔 데스크매트나 책상 상판을 살짝
관통하고 빠져나오지 못한다. 또는 에피소드 리셋 직후 펜이 매트 표면에서
0.5 ~ 1 cm 튀어오른 뒤 안착하는 일이 잦다 (이전 *"펜 z slack"* 항목보다 더
강한 증상).

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 그리퍼 fingertip 이 매트 표면 아래로 들어가 멈추거나, reset 직후 펜이
매트 위로 튀어오르는 게 4 개 중 1~2 개 비율로 재현.
```

### 원인

`scripts/author_pick_pen_scene.py` 의 `_scene_desk()` 가 `DeskTop`, `DeskMat`
을 `_cube(..., collision=True)` 로만 author 하고 `contact_tuning` 파라미터를
주지 않아 `PhysxCollisionAPI` 가 부여되지 않았다. 결과:

- PhysX 디폴트 `contactOffset = 0.02 m` (2 cm) 가 적용 — 매트 두께 (6 mm) 보다
  훨씬 큰 contact margin 이 객체 표면 양쪽에 부풀어 있다.
- 매트와 책상 상판이 z 방향으로 맞닿아 있는데, 둘 다 contact margin 2 cm 가
  부풀어 있어 broadphase 에서 서로 깊이 겹쳐 보임.
- 펜 (contactOffset 0.0015) 이 매트 표면 위 1 mm slack 으로 author 됐어도, 매트
  쪽 contact margin 이 펜 위치까지 침범 → reset 첫 step 에 PhysX 가 강한 분리
  impulse 를 가함 → 펜 튀어오름.
- 빠른 그리퍼 접근 시 매트/책상 contact 가 늦게 trigger 되어 한 step 안에
  통과해버림.

### 해결 방법

정적 환경 객체 (책상, 매트) 에도 `PhysxCollisionAPI` 를 명시하고 펜과 동일한
`contactOffset = 0.0015`, `restOffset = 0` 으로 맞춘다.

```python
_cube(
    lines, 1, "DeskTop",
    translate=_shift((0.0, 0.31, -0.02)),
    scale=(1.20, 0.78, 0.04),
    material_path=desk_mat,
    collision=True,
    contact_tuning=True,        # ← 추가
)
_cube(
    lines, 1, "DeskMat",
    translate=_shift((-0.02, 0.35, 0.003)),
    scale=(1.04, 0.57, 0.006),
    material_path=mat_mat,
    collision=True,
    contact_tuning=True,        # ← 추가
)
```

`_cube` 헬퍼의 `contact_tuning=True` 가 `_collision_attrs()` 내부에서
`physxCollision:contactOffset`, `restOffset`, `torsionalPatchRadius`,
`minTorsionalPatchRadius` 4 개를 명시한다.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
```

생성된 `assets/scenes/pen_desk/scene.usd` 에서 `DeskTop`, `DeskMat` 둘 다
`["PhysicsCollisionAPI", "PhysxCollisionAPI"]` 와 `contactOffset = 0.0015` 를
가지면 적용된 상태.

```text
Grep "DeskMat" -A 4 assets/scenes/pen_desk/scene.usda
# →
# def Cube "DeskMat" (
#     prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
# )
# {
#     bool physics:collisionEnabled = 1
#     float physxCollision:contactOffset = 0.0015
```

GUI 에서 `B`/`R` 반복 리셋해도 펜 튀어오름이 없고, 빠른 그리퍼 접근에도 매트
관통이 발생하지 않으면 정상.

---

## Sim-to-Real 펜이 펜통 안에서 spawn 되어 겹침 (펜·펜통 sampling 영역 분리 누락)

**현상**: 에피소드 시작 시 펜 한 개가 펜통 안에 박힌 채로 spawn 되고, 그 펜이
펜통의 walls collider 와 contact 가 발생해 펜이나 펜통이 튀어오른다. 참고
사진: `docs/pics/펜통_펜_배치_1.jpg`.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 첫 step 직후 펜 1 개가 펜컵 wire mesh 안쪽에 박혀 있다가 contact
impulse 로 펜과 컵 둘 다 튀어 오르는 모습.
```

### 원인

펜의 author + jitter 영역과 펜통의 author + sampling 영역이 xy 평면에서
겹쳤다. 예를 들어:

- 펜 default 영역: scene-local x ∈ [0.05, 0.20], y ∈ [0.13, 0.25]
- 펜통 default: scene-local (0.0, 0.18) — 펜 default 영역과 같은 y 대역

펜 4 개 중 한 펜의 sampling 결과가 펜통 반경 (0.052 m) 안에 떨어지면 펜
collider 와 펜통 wall collider 가 동일 좌표에서 겹친 채로 reset 된다. PhysX
는 첫 step 에 penetration 을 한꺼번에 풀려고 강한 분리 impulse 를 가하므로
펜이나 펜통이 튀어 오른다.

### 해결 방법

펜 sampling 영역과 펜통 sampling 영역이 **xy 평면에서 절대 겹치지 않게**
author. 가장 단순한 방법은 둘을 y 축으로 분리:

```text
scene-local y 축 (robot scene-local y = -0.04)
  │
0.65 ┤  매트 안쪽 끝
     │
0.40 ┤  ◀── 펜통 default (호 정점)
0.34 ┤  ◀── 펜통 호 양 끝 (sampling 최저 y)
     │       ⇡
     │       y 분리 마진 ≥ 0.08 m
     │       ⇣
0.28 ┤  ◀── 펜 sampling 최고 y (default 0.26 + jitter 0.02)
0.20 ┤  ◀── 펜 sampling 최저 y
     │
0.07 ┤  매트 robot 쪽 끝
```

코드 변경:

```python
# scripts/author_pick_pen_scene.py
PEN_CUP_LOCAL = (0.0, 0.40, 0.006)   # 매트 안쪽 깊은 곳으로 이동
PENS = (
    ("PenWhite", (-0.15, 0.22, 0.0147), 25.0, ...),
    ("PenGray",  ( 0.15, 0.22, 0.0147), -30.0, ...),
    ("PenBlack", ( 0.05, 0.26, 0.0147),  60.0, ...),
    ("PenBlue",  (-0.05, 0.26, 0.0147), -10.0, ...),
)
```

각 sampling 함수의 영역도 마진 안에 들어가는지 cross-check 한다 — 펜의
`y_radius` 와 펜통 호 양 끝 y 의 차이가 충돌 안전 거리 (≥ 펜 길이 절반
0.067 + 펜통 반경 0.052 = 0.119) 보다 작으면 안 된다.

### 확인 방법

```powershell
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

`B`/`R` 로 reset 을 10 회 이상 반복해도 펜 4 개가 모두 펜통 *바깥* 매트 위에
놓이고, 펜이나 펜통이 첫 step 에 튀어오르지 않으면 영역 분리가 충분.

---

## Sim-to-Real 펜통 호 sampling 이 매트/책상 밖으로 나감 (radius 와 default 좌표 불일치)

**현상**: `randomize_object_on_arc(PEN_CUP_NAME, radius=R, angle_range_deg=(-X, X))`
의 R 만 변경하면 펜통의 sampling 호가 매트를 벗어나 책상 가장자리, 심지어
바닥으로 떨어진다. 예: `radius=1.0, angle=±30°` 일 때 양 끝이 scene-local
`(±0.5, 0.83)` 로 매트 y 범위 `[0.065, 0.635]` 밖.

**오류 메시지**:

```text
Python traceback 없음.
GUI 에서 reset 후 펜통이 매트 너머 책상 빈 공간이나 책상 가장자리 너머로
spawn 되어 떨어지는 모습.
```

### 원인

`randomize_object_on_arc` 의 호 중심은 **author 한 펜통 default 좌표에서
forward (-y) 방향으로 `radius` 만큼 떨어진 점** 이다. 즉:

```
center_y = default_y - radius
arc point (angle θ): x = radius * sin(θ),  y = center_y + radius * cos(θ)
```

`radius` 만 키우면 호 자체가 더 큰 원이 되지만 *호의 정점* (= default 위치)
은 그대로다. 결과적으로 호 양 끝이 default 보다 훨씬 더 robot 쪽 (-y) 으로
밀려나 매트 시작점 (`y = 0.065`) 보다 더 앞쪽 — 책상 위 또는 책상 밖 — 으로
나간다.

`radius` 와 `default_y` 는 **함께** 잡아야 한다.

### 해결 방법

호의 정점 (= default_y) 과 양 끝의 y 차이가 매트 안에 들어가도록 다음 조건을
같이 푼다:

```
default_y                ≤ 매트 y 끝 (0.635)
default_y - radius (1 - cos(X))   ≥ 매트 y 시작 (0.065) + 마진
radius * sin(X)           ≤ 매트 x 절반 - 마진
robot 에서 호 양 끝 거리 = radius
```

robot scene-local y = -0.04 일 때, 호 정점이 robot 정면 SO-101 reach 가장자리에
오도록 두 변수를 잡으면 다음이 자연스럽다:

```
robot scene-local y = -0.04
SO-101 reach     ≈ 0.34 ~ 0.44 m
default_y        = robot_y + radius
radius           = 0.44      (default y = 0.40, 매트 안)
angle_range_deg  = (-30, 30) (양 끝 x = ±0.22, 매트 안)
```

이전 빨간 호 (`radius=1.0`) → 새 주황 호 (`radius=0.44, default_y=0.40`).
호 양 끝 y = `0.40 - 0.44 + 0.44 * cos(30°) = 0.34` 로 매트 안 + reach
한계에 정확히 위치.

### 확인 방법

```powershell
uv run scripts\author_pick_pen_scene.py
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard
```

`B`/`R` reset 을 20 회 반복하며 펜통이 매번 매트 검은 영역 안에 떨어지고,
정면 0° 부근부터 좌우 30° 가장자리까지 골고루 sampling 되면 정상.

---

## ROS 2 (WSL2) 노드 간 토픽 통신 불가 — lo 에 MULTICAST 없어 DDS discovery 실패

**현상**: WSL2 Ubuntu 에서 ROS 2 launch 는 뜨는데 `controller_manager` 가 `robot_description` 토픽을 영영 못 받아 컨트롤러가 안 올라온다. `ros2 node list` / `ros2 topic echo` 도 빈 결과. `move_group` 은 정상(파라미터로 robot_description 을 받기 때문).

**오류 메시지**:

```
[ros2_control_node] [WARN] [follower.controller_manager]: Waiting for data on 'robot_description' topic to finish initialization
[spawner] [WARN] [...]: Could not contact service /follower/controller_manager/list_controllers
```

(talker/listener 로 격리 테스트하면 talker 는 Publishing 하는데 listener 가 `I heard` 0 회)

### 원인

WSL2 의 loopback 인터페이스 `lo` 에 **MULTICAST 플래그가 없다**.

```
$ ip link show lo
1: lo: <LOOPBACK,UP,LOWER_UP> ...   # MULTICAST 없음
```

ROS 2 기본 DDS(FastDDS/CycloneDDS)는 multicast 로 participant discovery 를 하는데, `lo` 가 multicast 를 못 하므로 같은 호스트 내 노드끼리도 서로를 발견하지 못한다. `ROS_LOCALHOST_ONLY=1`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, FastDDS UDP-only(`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`) 모두 multicast 의존이라 실패. RMW 종류(FastDDS↔CycloneDDS)와 무관하다.

### 해결 방법

CycloneDDS 를 multicast 없이 **unicast localhost peer** 로 설정한다. `ros2_ws/setup/cyclonedds_localhost.xml`:

```xml
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers><Peer address="localhost"/></Peers>
      <ParticipantIndex>auto</ParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://<repo>/ros2_ws/setup/cyclonedds_localhost.xml
```

`ros2_ws/setup/env.sh` 가 이 두 변수를 export 하며, `04_setup_bashrc.sh` 가 bashrc 에 등록한다.

> ⚠️ **이 CycloneDDS unicast 방식은 `.wslconfig` `networkingMode=mirrored` 에서 `sensor_msgs/Image` 등 대용량·복합 타입을 cross-process 로 전달하지 못한다**(discovery·작은 메시지는 정상, 카메라만 0 fps). 현재 `env.sh` 는 이 문제 때문에 **FastDDS 를 기본 RMW 로 사용**하며 위 CycloneDDS 블록은 주석으로 남아 있다. 아래 §"ROS 2 (WSL2) 카메라 image_raw 토픽이 0 fps" 참조.

### 확인 방법

```bash
source <repo>/ros2_ws/setup/env.sh
ros2 run demo_nodes_cpp talker & ros2 run demo_nodes_py listener &
# listener 에 'I heard' 가 찍히면 OK
```

mock launch 에서 컨트롤러 3개가 "Configured and activated" 되면 해결.

---

## ROS 2 (WSL2) 카메라 image_raw 토픽이 0 fps — CycloneDDS + mirrored 네트워킹의 대용량 샘플 전달 실패

**현상**: WSL2 ROS 2 에서 카메라 노드가 `Started stream` 까지 정상이고 `ros2 topic list` 에 `image_raw` 가 보이는데, 어떤 구독자도(다른 프로세스·rosbridge·`ros2 topic hz`) 이미지를 **0 개** 받는다. 반면 `joint_states` 같은 작은 토픽은 같은 graph 에서 정상 전달된다. 노드 discovery 자체는 성공(`ros2 node list` 에 보임).

**오류 메시지**: 없음(에러 없이 조용히 0 fps). 격리 진단으로만 드러난다:

```
# 같은 프로세스 안에서 두 노드로 pub→sub (동일 QoS depth10):
std_msgs/String   : 120 msgs   ← 정상
sensor_msgs/Image : 0 msgs     ← 32x32(3KB)·640x480(921KB) 모두 0, 크기 무관
```

### 원인

`.wslconfig` 의 `[experimental] networkingMode=mirrored` 환경에서 **CycloneDDS 가 `sensor_msgs/Image` 같은 복합 타입을 cross-process 로 전달하지 못한다**. discovery 와 단순 타입(String/JointState)은 정상이라 "통신은 되는데 카메라만 안 되는" 형태로 나타난다. 메시지 크기와 무관(32×32 도 0)하므로 소켓 버퍼(`rmem_max`)·`MaxMessageSize`·QoS(RELIABLE/BEST_EFFORT) 튜닝으로 해결되지 않는다. mirrored 모드가 loopback/localhost 동작을 바꾸면서 CycloneDDS 의 해당 타입 데이터 경로가 깨지는 것으로 보인다.

### 해결 방법

RMW 를 **FastDDS(`rmw_fastrtps_cpp`)** 로 전환한다. FastDDS 는 같은 호스트를 **공유메모리(SHM)** 로 전송해 깨진 mirrored loopback 경로를 우회한다. `ros2_ws/setup/env.sh`:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# (CycloneDDS 의 RMW_IMPLEMENTATION / CYCLONEDDS_URI export 는 주석 처리)
```

- **모든 노드가 같은 RMW 여야 한다** — follower·rosbridge·카메라를 전부 `source env.sh` 한 셸에서 재기동. RMW 가 섞이면 graph 가 분리돼 서로 안 보인다.
- 보조로 `net.core.rmem_max` 를 올려두면(UDP fallback 대비) 안전하다: `ros2_ws/setup/wsl_ros2_sysctl.conf` → `/etc/sysctl.d/99-ros2-wsl.conf`, `sudo sysctl --system`.

### 확인 방법

```bash
source <repo>/ros2_ws/setup/env.sh   # RMW=rmw_fastrtps_cpp
# 카메라 노드 기동 후 다른 셸(역시 env.sh)에서:
ros2 topic hz /camera/top/image_raw   # ~20fps 이상 찍히면 OK
```

3캠(`/camera/{top,wrist,front}/image_raw`)이 각 ~23fps 로 cross-process 수신되면 해결.

---

## ROS 2 (WSL2) gscam·v4l2_camera 가 usbipd-win 가상 V4L2 디바이스에서 동작 안 함

**현상**: usbipd 로 attach 한 UVC 웹캠을 `gscam` / `v4l2_camera` 로 띄우면 노드는 뜨는데 프레임이 안 나온다(0 fps). `gst-launch-1.0` 로 같은 파이프라인을 돌리면 프레임이 흐른다.

**오류 메시지**:

```
# gscam
[ERROR] [cam]: Could not get gstreamer sample.
# v4l2_camera (YUYV)
[v4l2_camera]: Starting camera        ← 여기서 멈춤(DQBUF 무한 대기)
# v4l2_camera (MJPG)
[v4l2_camera]: Current pixel format is not supported yet: MJPG
terminate called after throwing an instance of 'cv_bridge::Exception'
# 공통(IO 모드)
streaming stopped, reason not-negotiated (-4)   # io-mode=2(mmap) 사용 시
```

### 원인

usbipd-win 의 가상 V4L2 디바이스는 표준 드라이버가 기대하는 동작을 일부 못 한다:
- `gscam`: appsink 가 첫 샘플을 1초 내 pull 못 해 포기(USB-IP 지연).
- `v4l2_camera` **YUYV**: 비압축 대역폭이 커 `DQBUF` 가 무한 대기(행).
- `v4l2_camera` **MJPG**: 디바이스는 받아들이나 노드가 MJPG→rgb8 디코드를 미지원해 크래시.
- GStreamer `io-mode=2`(mmap): 가상 디바이스에서 협상 실패(`not-negotiated`).
- 추가로 카메라 기본 framerate 가 640×480 에서 25fps 인데 30fps 를 요청하면 `not-negotiated`.

### 해결 방법

OpenCV(MJPG 압축 스트림) 로 직접 캡처해 발행하는 노드를 쓴다: `ros2_ws/src/so101_bringup/scripts/cv2_camera_publisher.py`. 핵심:
- `cv2.CAP_V4L2` + `MJPG` fourcc (압축이라 USB-IP 대역폭 안정).
- **단일 스레드 라운드로빈** — USB-IP 가상 디바이스는 다중 스레드 동시 블로킹 read 와 동시 open 경합을 못 버틴다. 한 스레드에서 카메라들을 순차로 `read()` → 발행(3캠 합산 ~18fps, FastDDS 로 각 ~23fps 전달).
- `ros2 launch so101_bringup cameras_cv2.launch.py` 또는 `ros2 run so101_bringup cv2_camera_publisher.py`.

> 참고: 이미지가 발행돼도 RMW 가 CycloneDDS+mirrored 면 구독자에 0 fps 다(위 §참조). FastDDS 와 함께 써야 한다.

### 확인 방법

```bash
# 디바이스 자체 캡처 가능 여부(드라이버 무관):
gst-launch-1.0 v4l2src device=/dev/cam_top num-buffers=3 ! image/jpeg,width=640,height=480,framerate=25/1 ! jpegdec ! videoconvert ! fakesink
# → Execution ended ... (ERROR 없이) 면 캡처 OK
```

노드 로그에 `... -> /camera/<name>/image_raw 스트리밍 시작` 3줄이 뜨고 `ros2 topic hz` 가 찍히면 해결.

---

## ROS 2 colcon 빌드가 `catkin_pkg` 못 찾음 (dotfiles 의 ~/.local python 이 ament 가로챔)

**현상**: `colcon build` 가 `ament_cmake` 패키지(예: `so101_moveit_config`)에서 실패.

**오류 메시지**:

```
ModuleNotFoundError: No module named 'catkin_pkg'
CMake Error at .../ament_package_xml.cmake:95 (message):
  execute_process(/home/<user>/.local/bin/python3.11
  .../package_xml_2_cmake.py ...) returned error code 1
```

### 원인

사용자 dotfiles 가 PATH 앞쪽에 `~/.local/bin/python3.11`(또는 Windows interop 경로의 python)을 둬서, ament/cmake 의 `FindPython3` 가 시스템 python(`/usr/bin/python3`, Ubuntu 24.04=3.12, `catkin_pkg` 보유) 대신 그 python 을 고른다. 그 python 에는 ROS 의존 모듈이 없다.

### 해결 방법

colcon 빌드 시 시스템 python 을 명시한다.

```bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

`ros2_ws/setup/03_build_workspace.sh` 에 반영됨.

### 확인 방법

`/usr/bin/python3 -c "import catkin_pkg; print(catkin_pkg.__file__)"` 가 `/usr/lib/python3/dist-packages/...` 를 출력하고, 빌드가 통과하면 정상.

---

## ROS 2 빌드 스크립트 `set -u` 가 setup.bash 와 충돌 (AMENT_TRACE_SETUP_FILES unbound)

**현상**: `set -euo pipefail` 을 쓴 빌드 스크립트가 `source /opt/ros/jazzy/setup.bash` 첫 줄에서 즉시 죽는다.

**오류 메시지**:

```
/opt/ros/jazzy/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable
```

### 원인

ROS 의 `setup.bash` 계열은 미설정 변수를 참조하는 구간이 있어 `set -u`(nounset)와 호환되지 않는다.

### 해결 방법

빌드 스크립트에서 nounset 을 빼고 `set -eo pipefail` 만 쓴다. (또는 source 전후로 `set +u` / `set -u`.)

### 확인 방법

스크립트가 source 단계를 넘어 빌드까지 진행되면 정상.

---

## ROS 2 (WSL2) feetech read timeout 1회로 hardware deactivate (USB-IP 레이턴시)

**현상**: 실기기(`hardware_type:=real`) launch 시 컨트롤러가 잠깐 activate 됐다가, 한 번의 read timeout 으로 hardware 전체가 deactivate 되며 죽는다. 간헐적으로 발생.

**오류 메시지**:

```
FeetechHardwareInterface::read -> CommunicationProtocol::sync_read [... SerialPort::read_exact [Read timeout]]
[ERROR] [follower.controller_manager]: Deactivating following hardware components as their read cycle resulted in an error: [ SO101_follower_SYSTEM ]
```

### 원인

usbipd-win 의 USB/IP 는 polling 기반이라 serial round-trip 레이턴시가 수십 ms 까지 튄다. 드라이버 기본 serial timeout 이 5ms 로 매우 짧고, ros2_control 은 read 가 한 번이라도 ERROR 를 반환하면 hardware 를 deactivate 한다. 또한 `on_activate` 가 EEPROM 쓰기(`configure_joints_`) 직후 곧바로 read 를 때려 첫 read 가 실패하기 쉽다.

> 참고: 모터/배선 자체는 정상이어도 발생한다. pyserial 로 1Mbps PING(`FF FF 01 02 01 FB`)을 보내 `ffff010200fc` 응답이 오면 통신 자체는 정상.

### 해결 방법

`feetech_ros2_driver` 와 컨트롤러 설정을 WSL2 레이턴시에 맞게 조정(이 레포에 반영):

- `feetech_driver/include/feetech_driver/serial_port.hpp`: serial `timeout_` 5ms → **50ms → 250ms**
- `feetech_ros2_driver/src/feetech_ros2_driver.cpp` `on_activate`: 초기 read 전 안정화 `sleep_for(150ms)` 추가
- `so101_bringup/config/ros2_control/follower_split_controllers.yaml`: `update_rate` 100 → **50Hz**

```bash
colcon build --symlink-install --packages-select feetech_ros2_driver so101_bringup \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

네이티브 Linux(직결 USB)에서는 레이턴시가 작아 원래 값(5ms/100Hz) 복원 가능.

**(2026-06-10) timeout 만으로는 부족 — read 실패 ride-through 가 정답**: timeout 을 250ms 까지 늘려도 USB-IP 지연/desync 스파이크가 가끔 초과해 단일 실패로 죽었다. 연속실패 임계(10/100) 방식도 burst 가 임계를 넘으면 죽음. **최종 수정**: `read()` 가 sync_read 실패 시 ① `communication_protocol_->flush_input()`(=`SerialPort::flashInputBuffer()` 노출, 입력버퍼 flush 로 desync 재동기화) ② **`return_type::OK` 를 항상 반환(마지막 상태 유지, 절대 deactivate 안 함)**. 진짜 단선은 joint_states freshness(stamp 갱신 멈춤)로 외부 감지. `feetech_ros2_driver.{hpp,cpp}` + `communication_protocol.hpp(flush_input)` 반영.

**🔴 미해결 — 프레임 시프트 corruption**: 위 ride-through 로 deactivate 는 막았으나, USB-IP 링크가 수 분 내 품질 저하하면 read 가 **checksum 통과하지만 joint↔value 어긋난 값**(명령 안 한 gripper 가 -1.098 등)을 간헐 반환한다. joint_states 를 신뢰 못해 캘리브/closed-loop 가 불가. `usbipd detach 4-1 && usbipd attach --wsl --busid 4-1` 재연결로 ~1~2분 깨끗해지나 곧 재저하. 근본해결은 USB-IP/하드웨어 레벨(WSL mirrored→NAT 검토, update_rate 20Hz 로 트래픽↓, 팔을 네이티브 Linux 직결). 현재 미해결.

### 확인 방법

`hardware_type:=real` launch 후 `grep -c "Read timeout" <log>` 가 0, 컨트롤러 3개가 "Configured and activated", `ros2 topic echo /follower/joint_states --no-daemon --once` 가 실제 모터 각도를 출력하면 정상.

---

## ROS 2 `libfeetech_ros2_driver.so: file too short` (빌드 캐시 손상)

**현상**: real launch 시 hardware 플러그인 로드 실패로 controller_manager 가 예외.

**오류 메시지**:

```
Failed to load library .../libfeetech_ros2_driver.so ... 
Could not load library dlopen error: .../libfeetech_ros2_driver.so: file too short
```

### 원인

빌드가 중간에 중단되거나 incremental 빌드가 꼬여 `.so` 가 0~부분 바이트로 남았다(`ls -lh` 로 보면 비정상적으로 작음). usbipd attach 가 풀린 채 빌드/실행이 얽히면서 발생하기도 한다.

### 해결 방법

해당 패키지의 build/install 산출물을 지우고 클린 재빌드.

```bash
rm -rf ~/so101_ros2_ws/build/feetech_ros2_driver ~/so101_ros2_ws/install/feetech_ros2_driver
colcon build --symlink-install --packages-select feetech_ros2_driver \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

### 확인 방법

`ls -lh ~/so101_ros2_ws/install/feetech_ros2_driver/lib/libfeetech_ros2_driver.so` 가 수 MB(정상 ≈2.4MB) 면 OK.

---

## ROS 2 (WSL2) `ros2 topic/node list` 가 빈 결과 (stale daemon)

**현상**: launch 노드는 살아있고 RViz↔MoveIt 제어도 되는데, 별도 터미널의 `ros2 topic list` / `ros2 node list` / `ros2 topic echo` 가 빈 결과.

### 원인

`ros2` cli 의 데몬이 과거(잘못된 RMW/CYCLONEDDS_URI 환경, 디버깅 중)에 한 번 떠서 캐시된 상태로 남아, 현재 cyclonedds-localhost 환경의 노드를 보지 못한다.

### 해결 방법

데몬을 우회한다.

```bash
ros2 topic list --no-daemon
ros2 topic echo /follower/joint_states --no-daemon --once
```

또는 `ros2 daemon stop` 후 env.sh 가 적용된 셸에서 재시작. RViz↔MoveIt 의 제어 통신은 데몬과 무관하므로 영향 없다.

### 확인 방법

`ros2 topic list --no-daemon` 에 `/follower/joint_states`, `/follower/robot_description` 등이 보이면 통신 자체는 정상.

---

## Isaac Lab ManagerBasedRLEnvCfg 에 `rewards` 누락 시 `gym.make` 실패

새 task 의 env_cfg 를 `ManagerBasedRLEnvCfg` 상속으로 작성하고 `gym.make()` 하면 환경 생성 직후 죽는다. scripted state machine 데모처럼 보상이 필요 없어 `rewards` 를 정의하지 않은 경우 발생.

```text
TypeError: Missing values detected in object PickCubeFrankaEnvCfg for the following fields:
  - rewards
```

### 원인

`ManagerBasedRLEnvCfg` 는 `rewards`·`terminations` 를 `MISSING` 기본값의 필수 필드로 둔다. `gym.make` → `ManagerBasedRLEnv.__init__` → `cfg.validate()` 가 채워지지 않은 필드를 검출해 `TypeError` 를 던진다(`observations`/`actions`/`events` 만 정의하면 통과하지 못함).

### 해결 방법

보상을 안 쓰더라도 **빈 보상 매니저**를 제공한다. `RewardManager` 는 term 이 0 개여도 정상 동작(reward=0)한다.

```python
@configclass
class PickCubeFrankaRewardsCfg:
    pass

@configclass
class PickCubeFrankaEnvCfg(ManagerBasedRLEnvCfg):
    rewards: PickCubeFrankaRewardsCfg = PickCubeFrankaRewardsCfg()
    terminations: PickCubeFrankaTerminationsCfg = PickCubeFrankaTerminationsCfg()
    ...
```

`commands`·`curriculum` 은 `None` 허용이라 생략 가능하지만 `rewards`·`terminations` 는 반드시 채운다.

### 확인 방법

`gym.make(task, cfg=env_cfg)` 가 예외 없이 env 를 반환하면 정상.

---

## Isaac Lab `gym.make` 이후 Python `print`/로그가 사라짐 (carb stdout 재바인딩)

`AppLauncher` 부팅 직후의 `print` 는 보이는데, `gym.make()`(또는 `SimulationContext` 생성) 이후의 `print` 가 — 특히 출력을 파일로 리다이렉트한 headless 실행에서 — 로그에 전혀 남지 않는다. 스크립트가 멀쩡히 동작해도 진행 상황·결과를 콘솔에서 확인할 수 없어 "조용히 죽은 것"처럼 보인다.

```text
# stdout 을 파일로 받으면 부팅 로그 45줄(P2P validation)에서 끊기고,
# 그 뒤 스크립트의 print("[SM] ...") 가 한 줄도 안 보인다. exit code 는 0.
```

### 원인

Isaac Sim/omni.kit 은 `SimulationContext` 를 만들 때 `sys.stdout`/`sys.stderr` 를 carb logger 로 재바인딩한다. 출력 대상이 tty 가 아니면(파일 리다이렉트) carb 로그 자체도 대부분 억제되어, 부팅 이후 Python `print` 가 묻힌다.

### 해결 방법

진행/결과 로그를 **파일에 직접 기록**하거나, 인터프리터 원본 fd(`sys.__stderr__`) 로 쓴다. 둘 다 carb 재바인딩을 우회한다.

```python
_LOG_PATH = "/tmp/franka_sm_progress.txt"
open(_LOG_PATH, "w").close()           # 실행마다 초기화

def log(msg: str) -> None:
    with open(_LOG_PATH, "a") as f:    # 파일 IO 는 carb 재바인딩과 무관
        f.write(msg + "\n")
    print(msg, file=sys.__stderr__, flush=True)  # GUI 콘솔용 원본 fd
```

디버깅 시 예외도 `traceback.format_exc()` 를 `log()` 로 남기면 silent 종료의 실제 원인(예: 위 `rewards` 누락)을 잡을 수 있다.

### 확인 방법

`gym.make` 이후 `log("...")` 한 줄이 진행 로그 파일에 남으면 정상.

---

## Isaac Lab manipulator 가 작업영역 일부만 도달 / 가까운 물체에서 ee 가 위로 솟음

> 사례: cube_desk 씬의 **Franka Emika Panda**(`isaaclab_assets.FRANKA_PANDA_HIGH_PD_CFG`,
> 7DOF arm + parallel gripper, reach ≈0.855 m)로 큐브 4개를 그릇에 옮기는 scripted
> state machine(`scripts/environments/pick_cube_franka_state_machine.py`). SO-101 이 아닌
> Franka 를 쓴 이유는 SO-101 의 5DOF arm 으로는 full 6DOF pose IK 가 불가능하기 때문이다.

DifferentialIK + scripted state machine 으로 여러 물체를 집을 때, 일부 물체만 잡히고
나머지는 ee 가 목표에 도달하지 못한다. 멀리 있는 물체는 손이 안 닿고(`ee.x` 가 특정 값에
갇힘), 너무 가까운 물체는 ee 의 z 가 위로 솟아(예: 큐브 z≈0.73 인데 ee z≈1.07) 하강을 못 한다.

```text
# 큐브 분산축과 base yaw 가 어긋난 경우 — ee.x 가 base 부근(1.89)에 갇혀 멀리 못 감
Cube4 descend reached=False ee=(1.889,-0.327,0.746) cube=(2.064,-0.357,0.729)
# base 가 물체에 너무 가까운 경우 — ee 가 위로 솟음
Cube4 descend reached=False ee=(1.576,-0.421,1.069) cube=(1.617,-0.457,0.729)
```

### 원인

- **yaw 어긋남**: base yaw 가 물체 분산 주축과 어긋나면 그 분산이 robot 의 side(좌우)
  방향이 된다. manipulator 는 forward reach 는 길지만 down-facing 자세로 side 로 뻗기는
  어려워, 분산 양끝 물체에 손이 안 닿는다.
- **거리 부정합**: 물체 영역 폭이 manipulator 의 이상적 forward reach 환형(annulus,
  대략 0.3~0.75 m)보다 넓으면, base 에 너무 가까운 물체는 팔이 접히는 영역에 들어가
  IK 가 ee 를 위로 솟구치는 해로 풀고, 먼 물체는 reach 경계에 걸린다.

### 해결 방법

- base **yaw 를 물체 분산 주축과 forward 가 일치**하도록 둔다(예: 큐브가 world +X 로
  넓게 퍼지면 base forward 도 +X = yaw 0°).
- base **거리**를 가까운 물체도 forward ≥0.3 m, 먼 물체도 reach(예 Franka 0.855 m) 안에
  들도록 조정한다. cube_desk Franka 는 `_FRANKA_POS=(1.30, -0.40, 0.71)`, yaw 0° 로
  큐브 x∈[1.60,2.08] 전부를 forward 0.30~0.78 m 안에 두어 4/4 안정 grasp 를 얻었다.

### 확인 방법

각 물체 접근 단계에서 ee xyz 와 물체 xyz 를 로깅해 추종 여부를 본다. 모든 물체가
`reached=True` 로 잡히면 정상.

---

## SO-101 5DOF grasp 가 불안정 (정합·제어점·자세 3중 오차) — Franka 권장

cube_desk 에서 **SO-101**(arm 5DOF)로 scripted pick-and-place 를 시도하면, IK 가 수렴해도
(예: Lula `lula(ok=True,err=0.0000)`) **실제 손가락이 큐브를 0.05~0.1m 빗나가** 못 집는다.
같은 씬에서 Franka(7DOF)는 4/4 로 안정적으로 잡힌다(`pick_cube_franka_state_machine.py`).

```text
# Lula IK 는 내부적으로 수렴(err 0)하나 USD 실제 손가락이 큐브에서 벗어남
Cube1 descend reached=True lula(ok=True,err=0.0000) ee=(1.706,-0.445,0.733) cube=(1.700,-0.440,0.724)
  jaw=(1.601,-0.432,0.811) gripper=(1.630,-0.430,0.832)   # 손가락이 큐브 위/옆
```

### 원인

SO-101 5DOF 는 grasp 순간 **세 오차원이 중첩**되고, 각 오차가 자세에 따라 변해 어떤 단일
IK 로도 동시에 못 맞춘다:

1. **정합**: Lula `LulaKinematicsSolver` 는 URDF-local frame, USD articulation 은 scene
   transform 아래 → Lula world ↔ USD world 사이 ~0.1m 잔차(RMPFLOW_BASE least-squares 로도 남음).
2. **제어점**: Lula 제어 frame(`gripper_frame_link`)과 실제 두 손가락 grasp 갭(jaw/gripper
   body midpoint)이 자세에 따라 상대 위치가 변한다 → 런타임 shift 보정이 매 step 흔들려 수렴 지연.
3. **자세**: 5DOF 로 position + full orientation 동시 만족 불가. orientation 강제하면 position
   을 0.1~0.25m 희생(`lula ok=False, err=0.11` / weighted DLS pose 동일), position-only 면
   손가락이 큐브 위에서 누르는 자세(감쌈 실패).

검증한 IK 들 — weighted DLS(local), random-FK(global sampling), Lula(position-only/orientation/
midpoint ee) — 모두 이 중첩을 못 넘었다. (반면 Franka 7DOF 는 yaw 가 독립이고 full pose IK 가
가능해 한 번에 grasp.)

### 후속 1 — in-sim DifferentialIK 18회 진단 (2026-06-09): 위 오차는 풀렸으나 grip 은 불가

외부 Lula 를 버리고 Isaac Lab **in-sim `DifferentialIKAction`**(env `SimToReal-SO101-PickCube-IK-v0`)
으로 재작성해 18회 headless 진단했다. 위 "정합"·"제어점" 오차는 in-sim IK(제어점=도달점 동일)로
원천 제거됐고, 세부 실패도 데이터로 순차 해결:

- **갭 roll 정렬은 원인 아님**(gap_misalign 1~6°, 기각).
- **ee 도달**: position-only + 단계별 arm stiffness(descend 120, soft PD 정상상태 오차 제거)로
  3.2cm → **0.4cm**.
- **수평 밀림**: 닫을 때 큐브 밀림을 gripper-local 축으로 분해하니 거의 전부 **X축(jaw 회전 호
  방향)** 성분 → 그 축으로 lateral 보정해 4.4cm → **1.3cm**. (closed-loop close 는 ee 가 큐브를
  쫓아가 18cm 비산 → 역효과, 고정 hold 가 정답.)
- **z 튐**: descend z over-drive 게인 1.2~1.5 로 손가락을 큐브 측면 깊이로 내려 해소.

그러나 **grip 자체는 5DOF DiffIK 로 불가**임이 확정: **강 tilt(jaw 를 큐브 측면으로) + ee 도달을
동시에 못 푼다**. position-only=수직(jaw 가 큐브 위 8cm 에 떠 손가락이 큐브에 안 닿음→안 들림),
pose tilt=강 tilt 시 ee 가 멀어짐(tilt20→ee1.2cm·jaw위, tilt30→ee4.5cm, tilt35→자세붕괴,
`--ik_lambda`↓는 DLS 불안정). env action space 가 부팅 시 고정이라 descend(position)/close(pose)
모드 혼용도 불가.

### 후속 2 — joint_fk (in-sim FK 샘플링) 로 1큐브 grasp 해결 (2026-06-10)

IK 를 버리고 **`--controller_mode joint_fk`**(random-FK 로 joint target 직접 샘플링)로 가면 IK 가
못 만드는 강 tilt 자세를 직접 탐색해 grip 이 성립한다. **1큐브 DR-off 1/1 성공**(DiffIK 18회 0/1
대비). 복원 소스 = 커밋 `62303d9`(env `SimToReal-SO101-PickCube-v0`, joint-space
`SlewLimitedJointPositionAction`; `94780bd` DiffIK 재작성에서 제거됐던 것).

- **4큐브 full-DR 은 평균 1.5/4 (all-4 ~0%) — 별도 blocker**. reach 매핑(1큐브 12 ep, spawn 위치
  로깅) 결과 실패는 robot base 거리와 무관(먼 0.30m 성공, 가까운 0.11m 실패)해 "reach 불가
  스폰"이 아니라 **random-FK 의 marginal grasp(단일 ~67%) + 4큐브 상호작용**(나중 큐브 approach 가
  기존 큐브/그릇을 침)이 주 원인. scatter range 를 reach 안쪽으로 제한
  (`_CUBE_SCATTER_X_RANGE`=[1.66,2.04], `_CUBE_SCATTER_Y_RANGE`=[-0.46,-0.345])하고
  `--object_order far_base_first`(base 에서 먼 큐브 먼저)를 적용해도 1.5/4 로 개선되지 않았다
  (grasp 품질 근본 한계). 즉 **1큐브 grasp 는 해결, 4큐브 신뢰 expert 는 미해결 blocker**.
- 데모/비교용으로 Franka 7DOF(`pick_cube_franka_state_machine.py`, 4/4 ~30초)도 유지.

### 후속 3 — 결정적 솔버(`--grasp_config_mode deterministic`)로 random-FK 분산 제거 (2026-06-11)

random-FK 의 단일 grasp ~67% 분산을 **finite-difference DLS 결정적 솔버**로 교체해 해결.
in-sim 가상 FK(joint state 임시 기록→`scene.update(0.0)`→body pose 읽기, random-FK 와 동일
패턴) 위에서 Gauss-Newton 을 수렴시킨다 — 난수 없음, URDF/base 정합 오차 없음. grasp 단계는
**tilt ladder**(전부 양수, 중간 tilt 우선)를 결정적 순서로 시도하고 scoop(이동 jaw 바닥·고정
finger 아래)·**개방축↔큐브 면 정렬**(roll task)·수평 오차로 채점한다.

**1큐브 DR-on 20ep: random_fk 85% → deterministic 90%** (동일 seed 비교, v6 구성). 에피소드
최단 7.2초(그리퍼 2속: 이동 1.8 rad/s / close-on-cube 0.6 rad/s + dwell 단축).

검증 과정에서 확정한 **함정 4가지** (전부 코드 주석에도 기록):

1. **닫는 단계의 tilt 갈아타기 금지**: descend 와 grasp(닫기)가 각각 자유 재계산하면 닫는
   도중 반대쪽 tilt 로 갈아타며 팔이 스윙해 큐브를 쳐낸다(5.5cm 비산 실측). descend 가 고른
   tilt 를 grasp 에 잠그고, 잠긴 tilt 실패 시 자세 유지+닫기만 한다.
2. **음수 tilt(base 쪽 기울임)는 항상 scoop 이 나쁨**: 이동 jaw 기하가 비대칭이라 tilt_pen
   0.026~0.062 로 측정됨 — ladder 에서 제거.
3. **깊은 안착 오프셋의 reach 상충**: grasp 목표 z 를 큐브 중심 아래로 내리면(-8mm/-4mm)
   손가락이 깊이 물지만 reach 가장자리 실행 미달이 늘어 90%→80% 회귀. +5mm 유지가 최적.
4. **가상 FK 의 책상 침투는 정상**: 가상 FK 엔 접촉 해소가 없어 좋은 scoop 계획도 손끝이
   책상면보다 수 mm 아래다(실행 시 물리가 받침). 침투 필터를 -3mm 로 걸면 좋은 후보가 전멸해
   성공률 0 근처로 추락 — 극단(-15mm)만 거른다.
5. **수평 오차 hard gate 금지**: 꼭지점 그립 방지로 "수평 7mm 이내" hard gate 를 걸면 reach
   가장자리에서 후보가 전멸해 90%→**65%** 회귀(v7 실측). 대신 *방향 분해 soft 가중* — 면을
   따라 미끄러진 ⊥개방축 성분(모서리/꼭지점의 직접 원인) ×2.0, 무는 방향 ∥성분(패드가 감싸
   흡수) ×0.3 — 으로 채점하고, 실행 후 수평 오차가 크면 descend_fix 가 보정한다.

또한 **20ep sweep 의 ±2ep 는 GPU PhysX 비결정성 잡음**이다 — 같은 plan 으로 같은 spawn 이
run 마다 성패가 뒤집힌다(marginal 접촉). 구성 비교는 메커니즘 근거 없이 ±10%p 이내 차이로
판단하지 말 것. 남은 실패 모드는 reach 경계 spawn(base 수평 ~14cm 안쪽·x≥1.97 바깥)의
기구학적 한계 — scatter 범위 조정(env 변경, 사용자 결정 필요)으로만 해소 가능.

리뷰 영상: `--review_video_dir`(전용 뷰어 카메라, 에피소드별 `epNN_{ok,fail}.mp4`),
`--review_pose_check`(구도 PNG 확인). 4큐브는 transport **경유점**(joint-space 보간이 호를
그리며 가라앉아 그릇을 엎는 사고 차단)+`bowl_tipped` 추적 추가 — 수정판에서 첫 all-4 달성
(구버전 평균 1.625/4, all-4 0/8).

### 확인 방법

1큐브 grasp: `pick_cube_state_machine.py --controller_mode joint_fk --task
SimToReal-SO101-PickCube-v0 --active_objects 1 --object_radius_scale 0 --headless --no_videos`
→ 결과 JSON 의 `final_inside.Cube1=true`, `placed_and_released=true`. 4큐브 신뢰성은
`--active_objects 4 --object_radius_scale 1 --num_episodes N` sweep 의 per-cube/all-4 로
측정(현재 평균 1.5/4). DiffIK 진단(폐기)은 `diffik_grasp_diag.patch`(commit `12265e1` 대비)로 보존.

결정적 솔버(후속 3): 위 명령에 `--grasp_config_mode deterministic` 추가 + `--num_episodes 20
--object_radius_scale 1` sweep → JSON `all4_success_rate ≥ 0.9`, 실패 에피소드는 `fail_diag`
(det_tilt_deg/det_pos_err_h_m/det_roll_err_deg/final_error_m)로 원인 추적.

---

## Windows Isaac Sim `_prepare_ui` access violation (tuple 인자를 AppLauncher 에 전달)

### 현상

Windows에서 `scripts/environments/pick_cube_franka_state_machine.py` 실행 시 Isaac Sim 확장이 모두 로드된 직후(~11초) `Windows fatal exception: access violation` 으로 크래시. 동일 머신에서 `teleop_se3_agent.py` 는 GUI 모드로 정상 동작.

### 오류 메시지

```
Windows fatal exception: access violation

Thread 0x0000460c (most recent call first):
  File "...simulation_app.py", line 602 in _prepare_ui
  File "...simulation_app.py", line 310 in __init__
  File "...app_launcher.py", line 823 in _create_app
  File "...app_launcher.py", line 131 in __init__
  File "...pick_cube_franka_state_machine.py", line 100 in <module>
```

### 원인

`vars(args)` 전체를 `AppLauncher(vars(args))` 로 전달할 때, argparse 커스텀 인자 중 **tuple 타입** 값(`view_eye=(3.05, -0.78, 1.02)`, `view_lookat=(1.74, -0.38, 0.74)`)이 포함된다. AppLauncher 가 알 수 없는 키를 carb 설정으로 등록 시도할 때 Windows carb 가 tuple 을 처리하지 못해 access violation 이 발생한다. Linux 에서는 동일 코드가 tuple 을 무시하거나 다르게 처리해 정상 동작한다. `teleop_se3_agent.py` 는 커스텀 인자가 모두 str/int/float/bool 이라 문제가 없다.

### 해결 방법

`AppLauncher` 에 전달하는 dict 를 AppLauncher 가 실제로 사용하는 키(`headless`, `enable_cameras`, `experience`, `device`, `cpu`, `disable_fabric`, `offscreen_render`, `kit_args`)만으로 필터링한다:

```python
_LAUNCHER_KEYS = {
    "headless", "enable_cameras", "experience", "device", "cpu",
    "disable_fabric", "offscreen_render", "kit_args",
}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(_launcher_args)
```

`pick_cube_franka_state_machine.py` 에 적용 완료.

### 확인 방법

`uv run scripts/environments/pick_cube_franka_state_machine.py` 를 `--headless` 없이 실행해 Isaac Sim GUI 가 정상 기동되고 큐브 pick-and-place 씬이 렌더링되면 수정 성공.

### 플랫폼 호환성

필터링 후에도 **Linux / Windows 모두 정상 동작**한다.

| | Linux (수정 전) | Linux (수정 후) | Windows (수정 후) |
|---|---|---|---|
| tuple 인자 전달 여부 | ✓ (무시됨) | ✗ (필터링) | ✗ (필터링) |
| AppLauncher 필수 키 포함 여부 | ✓ | ✓ | ✓ |
| 크래시 여부 | 없음 | 없음 | 없음 |

Linux 에서 수정 전 코드가 정상 동작했던 이유는 tuple 을 "잘 처리해서"가 아니라 Linux carb 가 알 수 없는 키를 **무시했기 때문**이다. 무시하던 키들을 애초에 전달하지 않으므로 Linux 동작에 영향이 없다. Isaac Lab 업그레이드 시 `add_app_launcher_args` 가 새 키를 추가하면 `_LAUNCHER_KEYS` 에도 동기화해야 한다.

---

## 시뮬레이션 기동 시 무시해도 되는 로그

`teleop_se3_agent.py` 가 정상 기동한 상태에서도 수십~수백 줄의 `[Error]` / `[Warning]` 로그가 찍힌다. 대부분 **LeIsaac 제공 scene USD 에셋 자체의 품질 이슈**에서 유래하며, 시뮬레이션·텔레오퍼레이션 기능에는 영향이 없다.

기동 성공 판단 기준: 로그 하단에 다음이 출력되면 정상 동작 상태다.

```
SO101-Leader connected.
 Running calibration of SO101-Leader
...
+-------------------------------------------------+
|  Teleoperation Controls for so101_leader        |
|   B  | start control                            |
|   R  | reset simulation ...                     |
|   N  | reset simulation ...                     |
+-------------------------------------------------+
```

### 로그 카테고리별 해석

| 로그 패턴 | 의미 | 대응 |
|---------|------|------|
| `[Error] [omni.physx.plugin] PhysicsUSD: Parse collision - triangle mesh collision (approximation None/MeshSimplification) cannot be a part of a dynamic body, falling back to convexHull approximation` | 씬 속 가구(cabinet/drawer/handle 등) 의 collision geometry 가 dynamic body 에 쓸 수 없는 triangle mesh 로 authored 됨 → PhysX 가 자동으로 convex hull 근사로 대체 | 물리 근사 품질이 약간 떨어질 뿐. 무시 |
| `[Error] [omni.physx.plugin] PhysX error: Supplied PxGeometry is not valid. Shape creation method returns NULL.`<br>`PhysX Shape failed to be created on a prim: .../outlet_room/...`, `.../light_switch_room/...` | 씬 속 콘센트·전등스위치 prim 의 geometry 가 유효하지 않아 shape 생성 실패 | 단순 장식 요소 한정. pick-and-place 와 무관, 무시 |
| `[Error] [omni.physx.plugin] PhysicsUSD: CreateJoint - cannot create a joint between static bodies, joint prim: .../wall_*/world_fixed_joint` | 벽·바닥 등 static body 쌍 사이에 fixed joint 를 만들려다 실패 | static 끼리는 조인트가 불필요, 무시 |
| `[Warning] [omni.physx.plugin] ... possibly invalid inertia tensor of {1.0, 1.0, 1.0} and a negative mass, small sphere approximated inertia was used` | light_switch/outlet 등 일부 rigid body 의 mass property 가 불량 → 작은 구로 근사 | 장식요소 한정, 무시 |
| `[Warning] [omni.physx.cooking.plugin] UjitsoMeshCookingContext: cooking failure for .../cab_3_main_group/post_0_0` | cab_3 의 세로 기둥(post) 메시 쿠킹 실패 → 해당 prim 에 대해 triangle mesh collider 가 생성되지 않음 | 시각만 렌더링, 물리 충돌 없음 — 물건이 통과할 수 있으나 태스크엔 무관 |
| `[Warning] [gpu.foundation.plugin] ECC is enabled on physical device 0` | A4000 의 ECC 메모리가 켜진 상태 안내 | 정상 |
| `[Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5` | 구 API 사용 안내 | Isaac Lab 2.3 내부 호출로 사용자가 손댈 일 없음, 무시 |
| `[Warning] [pxr.Semantics] pxr.Semantics is deprecated - please use Semantics instead` | USD 모듈 deprecation 안내 | 무시 |
| `[Warning] [omni.graph.core.plugin] Found duplicate of category 'Replicator'` | OGN 카테고리 중복 등록 | 무시 |
| `[Warning] [omni.replicator.core.scripts.extension] No material configuration file, adding configuration to material settings directly.` | Replicator 의 기본 머티리얼 config 파일 부재 | 무시 |
| `[Warning] [omni.fabric.plugin] Warning: attribute overrideClipRange not found for bucket id 9` | Fabric 내부 속성 lookup 실패 | 무시 |
| `[Warning] [omni.fabric.plugin] USD->Fabric: Unhandled array type string[]`<br>`[Warning] [usdrt.population.plugin] [UsdNoticeHandler] Unhandled attribute type VtArray<std::string> (prim attribute: omni:rtx:material:db:flattener:*)` | USD 의 string 배열 속성을 Fabric/USDRT 가 처리하지 못함 (RTX material db 관련) | 렌더링엔 영향 없음, 무시 |
| `[Warning] [omni.hydra] Parameter 'diffuse_texture_enable' of shade node ... not available in the MDL representation` | OmniPBR 머티리얼의 일부 파라미터가 MDL 변환본에 없음 | 렌더링 품질엔 영향 없음, 무시 |
| `[Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions: Render resolution of (371, 278) is below minimal input resolution of 300` | 뷰포트 해상도가 DLSS 최소치 미만이라 자동 상향 | 정상 |
| `[Warning] [omni.physx.plugin] Damping attribute is unsupported for articulation joints and will be ignored (.../sink_main_group/joints/handle)` | 싱크대 articulation joint 의 damping 속성은 PhysX 에서 무시됨 | 무시 |
| `[Warning] [omni.fabric.plugin] getAttributeCount/getTypes called on non-existent path .../Robot/wrist/visuals/wrist_roll_pitch_so101_v2` | SO-101 wrist visual prim 의 attribute 조회 시점 문제 | 로봇 제어엔 영향 없음, 무시 |
| `[Warning] [carb] Client gpu.foundation.plugin has acquired [gpu::unstable::IMemoryBudgetManagerFactory v0.1] 100 times. Consider accessing this interface with carb::getCachedInterface()` | Carb 인터페이스 획득 회수가 많다는 성능 권고 | 무시 |
| `[Warning] [omni.kit.notification_manager.manager] Physics USD Load: ...` (같은 메시지가 기동 후 수십 초 지나 다시 반복) | `R`/`N` 키로 reset 하면 씬이 재로드되면서 동일 경고들이 재출력 | 정상 동작 |

### 실제로 주의해야 할 로그

위 표에 해당하지 **않는** 다음 유형이 나오면 조치가 필요하다:

- `Windows fatal exception: code 0xc0000139` → **HDF5 ABI 불일치** (앞선 섹션 참조)
- kit log 백트레이스에 `arrow.dll` / `arrow_python.dll` / `_dataset.cp311-win_amd64.pyd` → **PyArrow / NumPy ABI 불일치** (앞선 섹션 참조)
- `ConnectionError: Could not connect on port 'COMx'` → 리더 암 시리얼 연결 실패. 포트 번호 / 드라이버 확인
- `AssertionError: the dataset file already exists, please use '--resume' to resume recording` → 기존 데이터셋 파일 삭제하거나 `--resume` 플래그 추가
- `Crash detected in pid ... thread ...` + `carb.crashreporter-breakpad.plugin` → 실제 프로세스 크래시. 직전에 찍힌 Python traceback 을 분석해야 함

## PickCube lula_ik / ikpy controller_mode 통합 시 함정

`--controller_mode lula_ik`(경로 2, Lula `LulaKinematicsSolver` 직접) / `ikpy`(경로 3) 를
붙일 때 만난 함정. rmpflow driver 와 같은 phase 슬롯·base 상수를 재사용하지만 IK 호출
규약이 다르다.

### `LulaKinematicsSolver.compute_inverse_kinematics` 반환·인자 규약

- 반환은 **`(joint_positions: np.array, success: bool)`** — joint 배열을 직접 돌려준다. `ArticulationAction` 이 아니므로 `.joint_positions` 속성으로 꺼내면 항상 `None` 이라 IK 가 안 풀린 것처럼 팔이 멈춘다.
- position-only(5-DOF)는 **`target_orientation=None`**. orientation 인자를 주면 quaternion(wxyz)을 기대하며, 이 빌드는 orientation 제약 시 수렴 실패 후 warm_start 를 그대로 반환한다.
- **Lula IK 는 local 솔버** → warm_start 에서 먼 target(>~0.1 m)은 한 번에 수렴 못 한다. 현재 EE→target 을 0.04 m 간격으로 보간하며 warm-start 를 체이닝하면(`SO101LulaIkJointTarget.compute`) 멀리까지 끌고 간다. (단, `_phase` 가 매 step `compute()` 를 다시 호출하므로 slew 진행만으로도 점진 수렴하긴 한다.)

### solver(URDF base) ↔ USD/world frame 정합

- Lula 는 `set_robot_base_pose(RMPFLOW_BASE_POS_USD, RMPFLOW_BASE_QUAT_USD)` 로 USD/world 좌표 target 을 직접 받는다. **ikpy 는 base pose setter 가 없으므로** 같은 base 상수로 `base = R(base_quat)ᵀ·(target_usd − base_pos)` 변환 후 풀어야 한다(안 하면 grasp 가 0.3~0.4 m 빗나감).
- grasp 작업점 ↔ EE frame(`gripper_frame_link`) 차이는 `RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET` 상수(world-frame 근사)로 보정한다. (RmpFlow driver 와 동일.)

### ikpy 함정

- `scipy.optimize.least_squares` 는 초기 guess(=현재 joint)가 URDF bound 를 부동소수점만큼 넘으면 `ValueError: Initial guess is outside of provided bounds` → seed 를 bound 안쪽(여유 1e-4)으로 clamp.
- `Chain.from_urdf_file(..., base_elements=["base_link"], active_links_mask=[False]+[True]*5+[False])` 가 `gripper`(revolute) 가지 대신 `gripper_frame_joint`(fixed) 가지를 자동 선택 → EE = `gripper_frame_link` 로 Lula 와 일치.

> 한계(공통): 5-DOF position-only IK 는 큐브별 grasp 자세 편차가 커서 **4-cube 신뢰성은 검증된 `joint_fk` direct FSM 이 우위**다. lula_ik/ikpy 는 단일 큐브 검증·대안 백엔드 용도.

## Core API follow-target: EE frame(`gripper_frame_link`) 이 USD prim 으로 안 보임

**현상**: Core API standalone(`follow_target_so101.py`, `World`+`SingleArticulation`+`ArticulationKinematicsSolver`)에서 EE world pose 를 읽으려고 스테이지를 `Stage.Traverse()` 로 순회해 `gripper_frame_link` prim 을 찾으면 못 찾는다. `simulation_app.close()` 가 종료 코드를 덮어써 **exit code 는 0 으로 나와 성공처럼 보이지만** 실제로는 예외로 중단된다(진행 로그도 carb 재바인딩에 묻힘).

```
RuntimeError: 'gripper_frame_link' prim 을 스테이지에서 찾지 못했다.
```

### 원인

`gripper_frame_link` 는 **URDF/Lula 전용 TCP 프레임**이라 SO-101 USD 에는 같은 이름의 prim 이 없다(USD body 는 `jaw`·`gripper` 등 다른 이름). Lula 솔버는 URDF 를 읽으므로 이 프레임을 알아 IK/`ArticulationKinematicsSolver` 생성은 정상이지만, USD 스테이지 순회로는 잡히지 않는다.

### 해결 방법

- EE world pose 는 prim 검색 대신 **`ArticulationKinematicsSolver.compute_end_effector_pose()`** 로 얻는다(현재 관절 상태 FK + `set_robot_base_pose` 반영, world frame). `[0]` 이 position.
- 굳이 USD body 로 읽어야 하면 `gripper_frame_link` 가 아니라 실제 USD body 이름(`jaw`/`gripper`)을 쓴다(`pick_cube_state_machine.py` 의 grasp midpoint 패턴).
- 부팅 후 스크립트는 예외가 `close()` 에 묻히지 않도록 `main()` 을 try/except 로 감싸 traceback 을 파일(`/tmp/...`)에 남긴다(carb stdout 재바인딩 회피).

### 확인 방법

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
  python scripts/environments/follow_target_so101.py --headless --selftest
```
→ `/tmp/so101_follow_target.txt` 에 `[selftest] #0..#3 ... pass=True`, `OK — 4/4` (EE 추종 오차 ~0.005–0.008 m). 5-DOF position-only IK 는 grasp 와 달리 접촉 정밀도 부담이 없어 follow-target 은 sub-cm 로 수렴한다.

## Core API standalone 로봇이 실행 직후 넘어져 날아감 (floating base — fix_root_link 누락)

**현상**: Core API `World` + `add_reference_to_stage(robot.usd)` + `SingleArticulation` 로 SO-101 을 띄우면, Play 직후 IK 가 팔을 움직이는 순간 **로봇 전체가 반력으로 넘어져 책상 위로 날아간다**. Isaac Lab env(`gym.make`)에서는 멀쩡하던 로봇이 standalone 에서만 무너진다.

### 원인

`add_reference_to_stage` 는 USD 를 그대로 참조만 한다 → 베이스가 world 에 고정되지 않은 **floating base** 다. Isaac Lab env 는 `ArticulationCfg.spawn` 의 `ArticulationRootPropertiesCfg(fix_root_link=True)` 로 베이스를 고정하는데, raw 참조는 이 단계를 건너뛴다. 또 raw USD 드라이브 게인이 env 의 soft-PD(stiffness 17.8/damping 0.6)와 달라 IK target 으로 급스냅하며 물체를 쳐낸다.

`compute_end_effector_pose()` 기반 self-test 가 통과하는데도 무너지는 이유: 그 FK 는 **고정 가정 base**(set_robot_base_pose) 기준이라, 실제 root 가 넘어져도 joint 각도만 맞으면 EE 거리는 통과로 보인다(false pass).

### 해결 방법

- 로봇은 `add_reference_to_stage` 대신 **Isaac Lab spawn** 으로 띄워 env 와 동일하게 `fix_root_link=True` (+ `enabled_self_collisions`, solver iter 4/4) 적용:
  ```python
  import isaaclab.sim as sim_utils
  spawn = sim_utils.UsdFileCfg(usd_path=..., articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, ...))
  spawn.func("/World/Robot", spawn, translation=ROBOT_POS, orientation=ROBOT_QUAT)
  ```
- reset 후 `robot.get_articulation_controller().set_gains(kps=17.8, kds=0.6)` (+ `set_max_efforts(10)`) 로 env soft-PD 이식.
- self-test 는 EE 거리 외에 **root world pose 이탈**(`robot.get_world_pose()` ↔ ROBOT_POS, ≤0.02 m)도 검사해 floating base 를 잡는다.

### 확인 방법

`--headless --selftest` 로그에 `base_drift=0.0000m base_fixed=True` 가 찍히고 `OK — 4/4` 통과. GUI 에서 Play 해도 팔이 베이스에 붙은 채 target 을 추종한다.

## SO-101 RMPFlow(`--controller rmpflow`) 가 target 에 ~0.1-0.2m 못 미침 (untuned scaffold)

**현상**: `follow_target_so101.py --controller rmpflow` 에서 EE 가 target 을 따라가긴 하나 **일정하게 0.1-0.2 m 못 미친 채 멈춘다**. 같은 target 들을 `--controller ik` 는 sub-cm 로 도달한다.

### 원인

`so101_rmpflow_config.yaml` 은 주석대로 *"controller validation scaffold"* — 미튜닝이다. 두 요인이 겹친다:
1. **`cspace_target_rmp` (home-posture attractor)** 의 `metric_scalar` 가 크면(초기 35) default_q(home)로 끌어당기는 힘이 EE attractor(`target_rmp`)를 이겨 정지상태 오차가 남는다.
2. **`joint_velocity_cap_rmp.max_velocity`** 가 낮으면(초기 1.0) 정착 window 안에 다 못 움직인다.
3. obstacle 등록 시엔 `collision_rmp.metric_modulation_radius`(0.25 m)가 커서, cube_desk 처럼 작업영역이 좁으면 workspace target 이 회피 영역 안이라 EE 가 **일부러** 거리를 둔다(회피는 정상 동작).

### 해결 방법

- `cspace_target_rmp.metric_scalar` 를 **1.0** 으로 낮추고(자세 정규화용으로만), `joint_velocity_cap_rmp.max_velocity` 3.0, `target_rmp.accel_p_gain` 80 으로 올리면 raw 추종이 ~0.07-0.16 m 로 개선된다(1/4 → 4/4 @0.18 m). IK(sub-cm)만큼은 아니다 — 5-DOF + scaffold 한계.
- **추종 정확도 검증은 `--no_obstacles`** 로 한다(obstacle 켜면 회피 때문에 거리 측정이 무의미). 정밀이 필요하면 `--controller ik`, 부드러움·회피가 필요하면 `rmpflow`.

### 확인 방법

```bash
... follow_target_so101.py --headless --selftest --controller rmpflow --no_obstacles
```
→ `OK — 4/4 ... (≤0.18m)`, `base_fixed=True`. obstacle 켠 인터랙티브 모드에서는 큐브 근처로 target 을 끌면 EE 가 거리를 두고 우회한다.

## Lula Test Widget 에서 SO-101 EE frame 이 손끝과 ~90° 어긋남 / IK 계속 실패

**현상**: GUI 의 `Tools > Robotics > Lula Test Widget` 으로 SO-101 을 따라가게 하면, `/Lula/end_effector` 프레임이 실제 로봇 손끝과 떨어진 곳에 뜨고 `Failed to compute Inverse Kinematics` 가 도배되며 로봇이 target 을 안 따라온다.

```
[Warning][isaacsim.robot_motion.lula_test_widget.controllers] Failed to compute Inverse Kinematics
```

### 원인

URDF(Lula 가 읽는 모델)의 base 프레임이 SO-101 **USD articulation root 와 ~90° Z 회전** 어긋나게 baked 돼 있다(URDF→USD 변환 산물). 원점·zero-joint 측정: Lula FK `gripper_frame_link`=(0.39, 0, 0.23)(팔 +X) vs 실제 USD `jaw`=(0.04, −0.30, 0.29)(팔 −Y). `pick_cube_state_machine.py` 는 손으로 맞춘 `RMPFLOW_BASE`(쿼터니언 ~90°Z) + per-solve shift 로 이를 보정한다. **위젯은 base pose 를 `articulation.get_world_pose()`(보정 없음)로만 설정**해 이 90° 를 못 넣는다 — 로봇을 회전 spawn 해도 시각·Lula 가 같이 돌아 상대 오차는 불변이라 정렬 불가.

### 해결 방법

- 위젯의 live follow 는 이 에셋엔 못 쓴다. **`default_q` 편집은 Robot Description Editor 로**(관절값을 USD 에 직접 적용 — 프레임 무관), **RMPFlow 게인은 yaml 편집 + `follow_target_so101.py --controller rmpflow` 헤드리스 검증**으로 튜닝(스크립트는 보정된 `RMPFLOW_BASE` 사용). 자세한 절차는 `LULA_GUI_TUNING.md`.
- 근본 해결은 URDF↔USD base 프레임을 일치시키는 에셋 재작업(미수행).

### 확인 방법

`/tmp/lula_fk_probe.py` 류로 원점·zero-joint 에서 `lula.compute_forward_kinematics("gripper_frame_link",[0]*5)` 와 USD `jaw` body world pose 를 비교 → +X vs −Y (~90°) 차이가 보이면 동일 원인.

## Isaac Sim headless 씬 author/검증 스크립트가 부팅 후 hang (단일 GPU 경합 / app.close 좀비)

**현상**: `author_pick_cube_scene.py`(PhysxSchema 정식 API 라 `AppLauncher` headless 부팅 필요) 또는 `gym.make` 검증 스크립트를 headless 로 띄우면, GPU 배너·CUDA P2P 검증까지 로그가 찍힌 뒤 더 진행하지 않고 멈춘다. 프로세스는 살아 있고 GPU 메모리(수백 MB~수 GB)를 점유하지만 CPU 0~3% 로 idle. USD export 나 결과 파일은 멈추기 전에 기록되기도 한다.

```
... Running CUDA peer-to-peer bandwidth and latency validation.
   CPU     0
     0   1.65
(이후 진전 없음 — MAKE_OK/Authored 미출력)
Exception ignored in: <function ManagerBasedEnv.__del__ ...>
AttributeError: 'ManagerBasedRLEnv' object has no attribute '_is_closed'   # GC 부산물(2차)
```

### 원인

1. **첫 부팅 EULA**: NVIDIA Omniverse Kit 첫 실행은 EULA 동의 프롬프트에서 비대화형 입력을 기다려 멈춘다.
2. **단일 GPU 동시 isaacsim 경합**: 이 서버는 GPU 1장(RTX PRO 5000). worktree 병렬 작업 등으로 **두 isaacsim 세션이 동시에 떠 있으면** 늦게 뜬 쪽의 `gym.make`(USD 로드+물리 씬 생성)가 GPU 자원 경합으로 hang 한다. (`nvidia-smi --query-compute-apps` 에 isaacsim python 이 2개 보이면 이 경우.)
3. **app.close() 좀비**: 이 환경에서 isaacsim `simulation_app.close()` 가 자주 hang → `timeout` 으로 죽여도 좀비가 GPU lock 을 안 놓아 **다음 부팅을 막는다**(첫 부팅만 성공하고 이후 hang 하는 패턴).
4. **SDF 베이킹**: 그릇 `PhysxSDFMeshCollisionAPI` `sdfResolution=256` 은 `gym.make` 시 SDF voxel 베이킹이 분 단위라 hang 으로 오인된다.

### 해결 방법

1. `OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python ...` (EULA 자동 동의).
2. 실행 전 `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv` 로 **다른 isaacsim 세션이 GPU 를 안 쓰는지 확인** 후 단독 실행. 떠 있으면 끝나길 대기.
3. 좀비 정리는 **PID 직접** `kill -KILL <pid>` (`pkill -f author_pick_cube` 는 grep 자기 명령줄까지 매치해 셸이 죽으니 금지). author/검증 결과는 `app.close()` 전에 파일로 남기므로 timeout kill 해도 결과 파일은 유효.
4. `sdfResolution` 은 150mm 단순 곡면 그릇 기준 `128` 로 충분(256→128 로 베이킹 단축).
5. **USD 속성만 바꿀 때는 author 재실행(부팅) 불필요** — `usd-core` 의 `Usd.Stage.Open(path)` → `attr.Set(...)` → `GetRootLayer().Save()` 로 .usda/.usd 직접 패치(예: `physics:mass`, `physxSDFMeshCollision:sdfResolution`).

### 확인 방법

`nvidia-smi` 에 isaacsim python 이 1개뿐인 상태에서 위 환경변수로 띄우면 P2P 검증을 통과해 `Authored ...` / `MAKE_OK` 가 출력된다. 결과 파일(`/tmp/...`)에 `STEP_OK` 와 객체 위치(`nan=False`)가 찍히면 런타임 로드 정상.

---

## Isaac Lab 대규모 num_envs 에서 PhysX 물리 머티리얼 64K 한도 초과 (createMaterial limit)

### 현상

PickCube 를 num_envs 12288/16384 로 올리면 씬 init 중 PhysX 에러 후 학습이 진행되지 않고 멈춘다(VRAM 은 충분히 남음 — VRAM 문제 아님). 8192 는 정상.

### 오류 메시지

```
[Error] [omni.physx.plugin] PhysX error: PxPhysics::createMaterial: limit of 64K materials reached.,
  FILE .../physx/src/NpPhysics.cpp, LINE 637
```

### 원인

PhysX 는 **distinct 물리 머티리얼 64,000 개** 하드 한도가 있다(IsaacLab #941/#2494). InteractiveScene 이 env 를 복제할 때 **각 env 가 물리 머티리얼을 자기 사본으로 생성**한다(static scene + per-object 머티리얼은 physics replication 으로 공유되지 않음). PickCube 는 env 당 물리 머티리얼이 **6개**였다:

- `CubeFriction` ×4 — 큐브 4개가 **각자 자기 USD 안에** CubeFriction 보유(`/Scene/CubeN/Looks/CubeFriction`)
- `BowlFriction` ×1, `DeskFriction` ×1
- (로봇 머티리얼 3개는 **visual** 이라 `createMaterial`(물리) 카운트에 무관)
- (material DR `randomize_rigid_body_material` 은 `num_buckets` 로 풀링 → 256개 1회 생성, env 마다 복제 아님 → 주범 아님)

`8192×6=49K`(통과) → `12288×6=74K`/`16384×6=98K`(초과). 한도가 VRAM 이 아니라 **env 당 머티리얼 수 × num_envs** 라, 8192 부근이 상한이었다.

### 해결 방법

**값이 동일한 머티리얼 인스턴스를 1개로 공유**한다(maintainer 권고: per-object 머티리얼 제거/공유). 4개 큐브가 동일 `FRICTION_CUBE` 값이므로 인스턴스만 4→1 로 합쳐 env 당 6→3 개로 줄였다(`16384×3=49K` 통과, 물리값 불변).

`scripts/environments/author_pick_cube_scene.py`:

1. **큐브 USD 에서 per-cube `CubeFriction` 제거** — `author_cube()` 에서 `_physics_material(... "CubeFriction" ...)` 생성과 collider `_bind_physics` 삭제(큐브 USD 는 물리 머티리얼 0개).
2. **scene.usd 에 공유 `CubeFriction` 1개 author + over-bind** — `author_scene()` 에서 `/Scene/Looks/CubeFriction` 1개 생성, 큐브 payload 참조 직후 `stage.OverridePrim("/Scene/{name}/Box")` 로 collider 에 over-bind(`materialPurpose="physics"`).
3. **재author**: `OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src $ROOT/.venv/bin/python scripts/environments/author_pick_cube_scene.py` (scene + 큐브 USD 재생성).
4. 16384 env 대비 PhysX 버퍼도 상향: `pick_cube_env_cfg.py` `gpu_max_rigid_patch_count 16·2¹⁶`, `gpu_total_aggregate_pairs_capacity 1M`, `gpu_collision_stack_size 2²⁹`.

> 부수효과(허용): 큐브 friction DR(`randomize_object_material`)이 4큐브 공유 머티리얼을 쓰게 되어 per-cube friction 다양성이 사라진다(env 단위 다양성은 유지). 동일 조명 복제 이슈(`/World` 단일 광원)와 같은 계열의 "env 복제 시 공유" 문제다.

### 확인 방법

```bash
# 재author 후 큐브 USD 에 물리 머티리얼 0개, scene.usd 에 3개(Desk/Cube/Bowl)인지
.venv/bin/python -c "from pxr import Usd,UsdPhysics; \
  [print(p, [x.GetPath() for x in Usd.Stage.Open(p).Traverse() if x.HasAPI(UsdPhysics.MaterialAPI)]) \
   for p in ['assets/scenes/cube_desk/scene.usd','assets/scenes/cube_desk/objects/Cube1/Cube1.usd']]"
```

스모크 학습(`--num_envs 16384 --max_iterations 2`)이 `64K materials` 에러 없이 `Learning iteration` 까지 도달하면 OK. VRAM ~23GB(예산 32GB 내).

## Windows Isaac Sim 에서 ROS2/OmniGraph 확장 런타임 enable 시 RTX 그래픽 재초기화 크래시

### 현상

Windows 네이티브 Isaac Sim 5.1(standalone SimulationApp/AppLauncher)에서 부팅 후
`enable_extension("isaacsim.ros2.bridge")`(또는 `isaacsim.core.nodes`)를 호출하면
access violation 으로 즉사. GUI 모드는 `_prepare_ui`, headless 모드는 확장 enable 중
viewport window 생성 단계에서 죽는다. scene/OmniGraph 사용자 코드는 실행되기 전.

### 오류 메시지

```
[Error] [gpu.foundation.plugin] DriverShaderCacheManager::init() called with a different graphics interface, without a shutdown()!
[Warning] [carb.graphics-vulkan.plugin] Aftermath Error 0xbad00009
Windows fatal exception: access violation
  ... omni/kit/widget/viewport/impl/texture.py ... __enable_hydra_engine
  ... rtx.scenedb.plugin.dll!carbOnPluginStartup ...
  ... omni.usd.dll!omni::usd::UsdManager::createHydraEngine ...
```

### 원인

headless kit(`isaaclab.python.headless.kit`)은 그래픽 인터페이스를 **D3D12** 로 먼저 띄운다.
런타임에 `isaacsim.ros2.bridge`/`isaacsim.core.nodes` 를 enable 하면 transitive 로
`omni.kit.viewport.window` 가 끌려와 viewport window + RTX Hydra 엔진을 **Vulkan** 으로
재생성하려 한다. 이미 D3D12 로 초기화된 위에 다른 그래픽 인터페이스로 재init → `rtx.scenedb`
크래시. 즉 확장을 **런타임에** enable 하는 게 문제(부팅 시 headless 면 viewport 억제됨).

### 해결 방법

- **부팅-시점 로드도 실패(2026-06-09 검증)**: `--kit_args "--enable isaacsim.ros2.bridge ..."`
  로 부팅 시점에 로드하면 그래픽이 처음부터 Vulkan 단일이라 D3D12↔Vulkan 재init 은 사라진다.
  그러나 (1) `[Error] ROS2 Bridge startup failed` — bridge 확장이 Windows 에서 시작 자체 실패,
  (2) 부팅 중 `omni.usd!createHydraEngine` → `rtx.scenedb.plugin.dll` access violation 여전.
  GUI/headless·런타임/부팅·최소ext 등 4가지 모두 RTX/Hydra 층에서 크래시. → **이 Windows 박스
  (RTX A4000 / driver 596.36 / Isaac Sim 5.1)에서 Isaac Sim ROS2 bridge 는 불가로 결론.**
- **해결(확정)**: Isaac Sim + ROS2 bridge 경로(PATH E)는 **네이티브 Linux 서버**에서 실행.
  Linux 는 Vulkan 단일 스택·ROS2 Jazzy·cuMotion 네이티브. `docs/PATH_E_CUMOTION_PICKPLACE.md`
  §cuMotion(Linux 서버) 참조. Windows 에서 동작하는 검증된 대안 = RViz mock(OMPL) pick&place
  (`ros2_ws/setup/run_mock_pickplace_demo.sh`, physics 없는 kinematic, 4/4 planned).

### 확인 방법

Linux 서버에서 동일 `scripts/ros2/cube_desk_ros2_sim.py` 실행 시
`[cube_desk_ros2_sim] 브릿지 실행...` 까지 도달하고 `ros2 topic echo /isaac_joint_states`
가 수신되면 정상.

## MoveIt gripper action 이 `wait_for_server` 타임아웃 (GripperCommand ↔ ParallelGripperCommand 타입 불일치)

### 현상

pick&place orchestrator(`so101_pick_place_orchestrator.py`)가 그리퍼를 명령할 때마다
실패하고, 그리퍼가 RViz/실기기에서 전혀 움직이지 않는다. arm trajectory 는 정상 실행.
매 그리퍼 호출마다 5초씩 지연(`wait_for_server` 타임아웃)되어 사이클이 길어진다.

### 오류 메시지

```
[ERROR] [so101_pick_place]: gripper action server 없음
```

### 원인

action **이름**(`/follower/gripper_controller/gripper_cmd`)은 맞지만 **타입**이 다르다.
`follower_split_controllers.yaml`(및 isaac variant)의 gripper_controller 는
`parallel_gripper_action_controller/GripperActionController` 라서 action 타입이
`control_msgs/action/ParallelGripperCommand` 인데, orchestrator 의 ActionClient 는
구형 `control_msgs/action/GripperCommand` 로 생성돼 있었다. ROS 2 action 은 이름이 같아도
타입이 다르면 매칭되지 않아 `wait_for_server` 가 영영 False 를 반환한다.

`GripperCommand.Goal` = `command.position`(스칼라) + `command.max_effort` 인 반면,
`ParallelGripperCommand.Goal` = `command`(`sensor_msgs/JointState`, `name[]`/`position[]`) 로
goal 구조도 다르다.

### 해결 방법

orchestrator 의 그리퍼 클라이언트를 컨트롤러 타입에 맞춘다:

```python
from control_msgs.action import ParallelGripperCommand
# ...
self._client = ActionClient(node, ParallelGripperCommand,
                            "/follower/gripper_controller/gripper_cmd")
# ...
goal = ParallelGripperCommand.Goal()
goal.command.name = ["gripper"]        # 컨트롤러 joint 이름
goal.command.position = [float(position)]
```

### 확인 방법

재실행 시 `gripper action server 없음` 로그가 사라지고, 컨트롤러 측 demo.log 에
`[follower.gripper_controller]: Received & accepted new action goal` 이 그리퍼 명령마다
찍힌다. RViz 에서 그리퍼 jaw 가 open/close 한다. 그리퍼 호출당 5초 타임아웃이 없어져
pick&place 사이클도 빨라진다(예: 큐브당 ~16s → ~5s).

---

## (PATH E) Isaac Sim 헤드리스에서 OmniGraph 생성 실패 — `Unable to create prim for graph`

### 현상

PATH E bridge(`scripts/sim/run_cube_desk_ros_bridge.py`)가 ROS 2 bridge OmniGraph(`og.Controller.edit`)를 만들 때 종료. 빈 스테이지·최소 그래프·모든 evaluator/path 에서도 동일.

### 오류 메시지

```
[Error] [omni.graph.core.plugin] Unable to create prim for graph at /ROSBridge
omni.graph.core._impl.errors.OmniGraphError: Failed to wrap graph in node given
  {'graph_path': '/ROSBridge', 'evaluator_name': 'execution'}
```

### 원인

Isaac Lab 의 기본 헤드리스 experience(`isaaclab.python.headless.kit`)는 OmniGraph 의 USD authoring/orchestration 을 strip 한다 → 그래프 prim 을 만들 수 없다. 풀 `SimulationApp` 이나 렌더링 experience 에서는 정상(`isaacsim import SimulationApp` 단독 테스트로 확인).

### 해결 방법

AppLauncher 로 부팅하되 **렌더링 experience 를 강제**한다 — `args.enable_cameras = True` 면 `isaaclab.python.headless.rendering.kit`(풀 렌더 + OmniGraph USD authoring 포함)가 로드돼 OmniGraph 와 InteractiveScene 둘 다 동작한다.

```python
args.enable_cameras = True   # AppLauncher(vars(args)) 전에
```

### 확인 방법

bridge 가 `[bridge] ready` 까지 진행하고 `ros2 topic list`(컨테이너, --network host)에 `/clock /isaac_joint_states /isaac_joint_commands /tf` 4개가 보이면 그래프 생성·광고 정상.

---

## (PATH E) Isaac Sim 부팅 중 `errno=28 No space left on device` 가 수천 줄 — inotify watch 고갈

### 현상

디스크는 충분한데도 Isaac Sim 부팅 로그가 `Failed to create change watch ... errno=28` 로 도배되고, 이어서 OmniGraph `Unable to create prim` 등이 연쇄 발생.

### 오류 메시지

```
[Error] [carb] Failed to create change watch for `.../isaacsim/exts/...`: errno=28/No space left on device
```

### 원인

`errno=28`(ENOSPC)은 디스크가 아니라 **inotify watch 한도 초과**다. Isaac Sim 이 확장 hot-reload 용 watch 를 수천 개 만드는데, 동시 실행 프로세스(학습·isaacsim-mcp·다른 세션)와 합쳐 `fs.inotify.max_user_instances`(기본 128)/`max_user_watches`(기본 65536)를 소진. USD 프림 생성까지 실패로 번진다.

### 해결 방법

```bash
sudo sysctl -w fs.inotify.max_user_instances=1024 fs.inotify.max_user_watches=1048576
# 영구: /etc/sysctl.d/99-inotify.conf 에 같은 두 줄
```

### 확인 방법

재실행 후 로그에서 `grep -c "No space left on device"` 가 0. (inotify 한도를 올려도 별개 블로커인 device -1 은 남는다 — 아래 항목.)

---

## (PATH E, 해결) Isaac Lab bridge 의 OmniGraph JointState 가 `device 0 vs -1` 로 joint_states 미publish

### 현상

bridge 가 `[bridge] ready` 까지 가고 토픽도 광고되지만, 루프에서 JointState/ArticulationController OmniGraph 노드가 articulation 물리 텐서를 못 읽어 `/isaac_joint_states`·`/clock` 에 **값이 안 실린다**(`ros2 topic echo` 무응답, `hz` 가 not published).

### 오류 메시지

```
[Error] [omni.physx.tensors.plugin] Incompatible device of DOF position tensor in
  function getDofAttribute: expected device 0, received device -1
[Ros2JointStateMessage] Failed to get dof positions / velocities / efforts
```

### 원인

Isaac Lab `InteractiveScene`(Fabric/GPU 파이프라인)가 만든 physx tensor simulation view 와 OmniGraph 물리 노드(`IsaacArticulationController`/`ROS2PublishJointState`)가 만드는 view 가 충돌한다. graph 를 reset 전/후 생성, `OnPlaybackTick`→`OnTick`+`evaluate_sync` 강제평가, `PickCubeEnvCfg().sim`(GPU 파이프라인) 사용 — 설정으로는 미해결.

### 해결 방법 (B안 — 적용·검증 완료, 2026-06-09)

bridge 의 scene 로드/시뮬 파이프라인을 Isaac Lab `InteractiveScene`+`SimulationContext` 대신 **순수 `isaacsim.core.api.World`(CPU numpy 백엔드) + `SingleArticulation`** 으로 교체. `cube_desk/scene.usd`(SCENE_OFFSET baked → 객체 world 좌표 그대로) + `so101_follower.usd` 를 `add_reference_to_stage` 로 직접 stage 에 올린다. NVIDIA 공식 ROS2 standalone 예제와 동일 경로라 OmniGraph 물리노드가 simulation view 를 **단독 소유** → device 정합(CPU 백엔드면 양쪽 모두 -1 로 일치, GPU fabric view 와의 충돌 자체가 사라짐). 단일 로봇+소수 큐브라 CPU 물리로 cuMotion 제어에 충분.

세부:
- base 고정 = `isaaclab.sim.schemas.modify_articulation_root_properties(fix_root_link=True)` 재사용(순수 USD authoring, 시뮬 파이프라인 무관). fixed joint 생성 + ArticulationRootAPI 를 부모로 이동(PhysX parser 한계) → articulation root 가 `/World/Robot` 로 올라온다.
- 로봇 pose = `PickCubeEnvCfg._ROBOT_POS/_ROBOT_ROT` 재사용(PATH C 시뮬과 동일 배치). 단 referenced root 의 `xformOp:orient` 가 `quatd` 라 `AddOrientOp(PrecisionFloat)` 는 Tf 에러 — 기존 op precision 에 맞춰 값만 Set.
- USD drive gain 이 micro(0.05~0.85) 라 `articulation.get_articulation_controller().set_gains(kps=17.8, kds=0.6)` 로 leisaac 검증값 적용(안 하면 cuMotion 위치 명령 미추종).
- 루프 = `world.step(render=True)` 한 줄(OnPlaybackTick 으로 그래프 자동 평가 — A안의 수동 `evaluate_sync` 불요).

진입점 = `scripts/sim/run_cube_desk_ros_bridge.sh`(LD_LIBRARY_PATH·DDS env export 래퍼, 아래 두 항목 참조).

### 확인 방법

```bash
ros2 topic echo /isaac_joint_states --once   # 6관절 name/position/velocity/effort 값
ros2 topic echo /tf --once                   # base_link→Cube1/Bowl transform
```
bridge 로그에 `expected device` 에러 0건. 2026-06-09 서버 konan147 에서 `--num_cubes 1` 로 위 3토픽 모두 값 흐름 확인(검증 §5 1~3 통과). 이후 §5 4~6(RViz dry-run → 단일/4큐브 pick-and-place)은 컨테이너 ROS 스택 launch + cuMotion XRDF 검증 후.

---

## (PATH E) Isaac Sim ROS 2 bridge 가 `librmw_implementation.so` 로드 실패 — `libament_index_cpp.so: cannot open`

### 현상

호스트 uv 환경(ROS 2 미설치)에서 bridge 부팅 중 `isaacsim.ros2.bridge` extension 이 startup 실패. 토픽이 하나도 안 뜬다.

### 오류 메시지

```
[Error] [isaacsim.ros2.bridge.impl.extension] ROS2 Bridge startup failed
Could not load the dynamic library from .../isaacsim.ros2.bridge/jazzy/lib/librmw_implementation.so.
Error: libament_index_cpp.so: cannot open shared object file: No such file or directory
```

### 원인

호스트에 ROS 2 가 없으면 bridge 는 isaacsim 번들 ROS 2 lib(`exts/isaacsim.ros2.bridge/jazzy/lib`)를 dlopen 한다. 이 .so 들엔 `$ORIGIN` RPATH 가 없어, `librmw_implementation.so` 가 같은 디렉터리의 의존성(`libament_index_cpp.so` 등)을 못 찾는다. 동적 링커는 **프로세스 시작 시** `LD_LIBRARY_PATH` 를 읽으므로 python 안에서 `os.environ` 으로 늦게 넣어도 무효.

### 해결 방법

launch **전에** 번들 lib 경로를 `LD_LIBRARY_PATH` 에 export. `scripts/sim/run_cube_desk_ros_bridge.sh` 래퍼가 수행:

```bash
export LD_LIBRARY_PATH="<repo>/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib:$LD_LIBRARY_PATH"
```

### 확인 방법

bridge 로그에 `ROS2 Bridge startup failed` 가 없고 `ros2 topic list` 에 `/isaac_joint_states` 등이 뜬다.

---

## (PATH E) host(bridge)↔container(ROS 스택) DDS discovery 실패 — cross-UID fastrtps SHM

### 현상

bridge 가 `/isaac_joint_states` 등을 정상 publish 하고(`/dev/shm/fastrtps_*` 세그먼트 + UDP 7400/7410/7411 listening 확인), 컨테이너를 `--network host --ipc host` 로 띄워 RMW·DOMAIN 을 맞춰도 `ros2 topic list` 가 **빈 결과**.

### 원인

host bridge 는 일반 유저(uid 1000), 컨테이너 ROS 스택은 root(uid 0)로 실행된다. fastrtps 기본 transport 의 SHM(`/dev/shm/fastrtps_*`)은 서로의 세그먼트 lock/ring-buffer 에 **cross-UID 로 접근**해야 하는데 권한이 안 맞아 same-host 참가자 간 SHM 협상이 실패한다(metatraffic 도 SHM 우선 시 안 보임).

### 해결 방법

양쪽 모두 fastdds 를 **UDP-only** 로 강제해 SHM 협상을 우회한다:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

- bridge: 래퍼(`run_cube_desk_ros_bridge.sh`)가 export + `.py` 도 `os.environ.setdefault`(DDS init 은 python 시작 이후라 유효).
- 컨테이너: `docker run -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 --network host --ipc host …`.

> ROS 2 가 양쪽 다 fastrtps 여야 한다. isaacsim 번들은 **fastrtps 만** 포함(cyclonedds 없음)하므로 컨테이너도 fastrtps 로 맞춘다(`ros2_ws/setup/env.sh` 의 cyclonedds 는 WSL2 PATH D 전용 — PATH E 에서 source 금지).

### 확인 방법

```bash
docker run --rm --network host --ipc host \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  so101-cumotion:jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && ros2 topic echo /isaac_joint_states --once'
```
6관절 값이 찍히면 해결.

---

## (PATH E) `pick_place.launch.py` ROS 스택 bringup 4대 함정

`pick_place.launch.py`(controllers + move_group/cuMotion + SM)를 서버에서 처음 실측 기동할 때
연쇄로 막힌 4지점. 전부 소스/Dockerfile 에 반영됨(2026-06-09 서버 konan147).

### 1) controller_manager SIGSEGV — topic_based_ros2_control ABI 불일치

**현상**: `ros2_control_node` 가 hardware 'initialize' 직후 죽는다(exit -11).

```
[INFO] Loaded hardware 'SO101_follower_SYSTEM' from plugin 'topic_based_ros2_control/TopicBasedSystem'
[INFO] Initialize hardware 'SO101_follower_SYSTEM'
Stack trace ... hardware_interface::HardwareComponentInterface::get_lifecycle_id() const
Segmentation fault (Address not mapped to object [0xb0])
```

**원인**: Isaac ROS apt repo 의 `ros-jazzy-topic-based-ros2-control` 가 **99.99.1-0noble**(Isaac ROS
ros2_control 스냅샷 빌드)인데, ROS 메인 repo 가 `hardware_interface` 등을 **4.44.0**(더 최신)으로 끌어와
ABI 가 어긋난다(`HardwareComponentParams`/`get_lifecycle_id` vtable). ROS 메인 repo 엔 topic_based
**0.3.0 source 만** 있어 바이너리 다운그레이드 불가.

**해결**: PickNik 소스(`github.com/PickNikRobotics/topic_based_ros2_control`, main=0.3.0)에서 설치된
hardware_interface 4.44.0 헤더로 재빌드해 overlay 설치(`Dockerfile.cumotion_ros` 가 `/opt/tbc_overlay`
에 colcon build 후 bashrc 에서 `/opt/ros/jazzy` 다음 source). 재빌드 시 `ros_testing` 누락은
`-DBUILD_TESTING=OFF` 로 회피.

**확인**: `ros2 control list_controllers -c /follower/controller_manager` 에 broadcaster/arm/gripper 가 `active`.

### 2) kinematics 플러그인 `pick_ik/PickIkPlugin` 미설치 → set_from_ik SIGSEGV

**현상**: move_group/SM 기동 로그에 plugin load 실패, SM 의 첫 `set_from_ik` 에서 SIGSEGV.

```
The kinematics plugin (pick_ik/PickIkPlugin) failed to load. ... class ... does not exist.
Declared types are cached_ik_kinematics_plugin/... kdl_kinematics_plugin/KDLKinematicsPlugin ...
```

**원인**: `so101_moveit_config/config/kinematics.yaml` 이 `pick_ik/PickIkPlugin`(5-DOF 에 적합, rotation_scale 0.5 +
approximate)을 쓰는데 이미지에 미설치. null plugin 을 set_from_ik 가 역참조 → segfault.

**해결**: `Dockerfile.cumotion_ros` apt 에 `ros-jazzy-pick-ik` 추가(packages.ros.org 1.1.1, hardware_interface 와 동일 빌드일자).

### 3) launch `Expected … got '()' of type tuple` — 빈 리스트 파라미터

**현상**: `move_group_cumotion.launch.py`(및 이를 포함하는 pick_place) 가 노드 시작 시 즉시 예외.

```
TypeError: Expected 'value' to be one of [float, int, str, bool, bytes], but got '()' of type 'tuple'
  (launch_ros/utilities/evaluate_parameters.py: evaluate_parameter_dict)
```

**원인**: cuMotion planning pipeline yaml 의 `request_adapters: []`(빈 리스트)가 `moveit_config.to_dict()`
→ launch_ros 에서 빈 튜플 `()` 로 평가돼 Node 파라미터 타입검증 실패.

**해결**: `so101_moveit_config/config/isaac_ros_cumotion_planning.yaml` 에서 `request_adapters: []` 줄 제거
(키 생략 시 MoveIt 이 "request adapter 없음" 으로 처리). 일반화: launch Node 파라미터에 빈 리스트/딕트 금지.

### 4) SM `NameError: name 'PoseStamped' is not defined`

**원인**: `pick_place_sm.py` 가 `_pose()` 에서 `PoseStamped()` 를 쓰는데 import 누락.

**해결**: `from geometry_msgs.msg import PoseStamped` 추가.

### 확인 (통합)

`scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1`(host) + 컨테이너에서
`ros2 launch so101_cumotion_pick_place pick_place.launch.py use_rviz:=false`(fastrtps/UDPv4, `/build`·`/workspace` 마운트,
overlay source) → cuMotion `CumotionPlanner` 로드 + URDF/XRDF 로 로봇 로드 성공, 컨트롤러 3종 active,
SM 이 큐브 포즈 수신→`pick-and-place cube[0]` 까지 진행. (이후 5-DOF grasp IK 튜닝은 별개 — §PATH_E 6.)

---

## (PATH E) cuMotion `INVALID_INITIAL_CSPACE_POSITION` — start_state 관절 수 ≠ cspace (gripper 포함)

### 현상

cuMotion 이 task-space goal 을 받자마자 모든 계획 실패. 팔이 전혀 안 움직인다.

### 오류 메시지

```
[cumotion_planner] Trajectory optimization to pose failed (trajopt: INVALID_INITIAL_CSPACE_POSITION)
Invalid c-space position: Number of c-space coordinates in 'cspace_position' [6] must equal
  the number of c-space coordinates of the robot [5].
Failed call to 'planToTaskSpaceTarget()': 'initial_cspace_position' [[0 0.07 0.07 0.01 0 1.4999]] is invalid.
```

### 원인

MoveIt `MotionPlanRequest.start_state` 는 **전체 로봇 관절**(SO-101 = arm 5 + gripper, 마지막 1.4999=gripper)을
담는다. cuMotion cspace 는 tool_frame(gripper_frame_link) 으로 가는 **kinematic chain 위 관절(5축)뿐**이다
(gripper 는 분기 관절이라 XRDF cspace 에 넣어도 cuMotion 이 무시 — 구조적). cuMotion MoveIt 플러그인
(`CumotionMoveGroupClient::updateGoal`)이 request 를 **무필터 전달**해 6관절 start_state 가 5축 cspace 와 어긋난다.
**알려진 upstream 미해결 버그**([isaac_ros_cumotion#10](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/issues/10),
#53). XRDF/URDF config 로는 해결 불가(비-cspace joint 선언 메커니즘 없음, cspace 추가도 무시됨 — 3가지 시도 모두 실패).

### 해결 방법

플러그인을 패치해 updateGoal 이 start_state 를 **planning group(manipulator, 5축) active 관절로 필터링**하게 한다.
`docker/patches/cumotion_moveit_filter_start_state.patch` (Dockerfile 이 isaac_ros_cumotion clone → patch apply →
`isaac_ros_cumotion_moveit` 만 colcon build → `/opt/cumotion_overlay`, bashrc 에서 /opt/ros/jazzy 다음 source).
핵심 로직: `getRobotModel()->getJointModelGroup(req.group_name)->getActiveJointModelNames()` 로 group 관절을 얻어
`req.start_state.joint_state` 를 그 관절만 남기고 재구성. (gripper 는 cuMotion 계획에서 빠지고 `gripper_controller`
action 으로 별개 제어 — 정합.)

### 확인 방법

clean run 에서 `INVALID_INITIAL_CSPACE_POSITION` 0건, cuMotion 로그의 `initial_cspace_position` 이 5개 값.
※ stale cumotion 프로세스(과거 cspace=6 실험본)가 남으면 `[5] vs robot[6]` 역방향 에러가 섞이니 pkill 로 정리.
※ 남은 과제(별개): cspace count 해결 후에도 5-DOF IK 미도달(`INVERSE_KINEMATICS_FAILURE`)은 grasp 자세 튜닝 영역(§PATH_E 6).

## (PATH E) 5-DOF 팔에서 cuMotion/OMPL 이 grasp pose-goal 을 못 푼다 — joint-goal 로 전환

### 현상

cuMotion + ROS 로 SO-101(5축) pick-and-place 시, c-space 6vs5 패치 후에도 grasp 접근 pose 계획이
모든 시도에서 실패. orientation 제약을 풀어도(`yaw_free_tol=π`), tolerance 를 키워도, **orientation 을
완전히 제거(position-only)해도** 실패. 팔이 큐브 근처로 전혀 안 간다.

### 오류 메시지

```
[ompl] manipulator[RRTConnect]: Unable to sample any valid states for goal tree
[cumotion_planner] Trajectory optimization to pose failed (trajopt: INVERSE_KINEMATICS_FAILURE)
# /compute_ik 직접 호출 시: error_code.val = -31 (NO_IK_SOLUTION) — 거의 모든 pose 에서
```

### 원인

**MoveIt(OMPL constraint sampler)·cuMotion 의 goal 샘플러는 task-space(pose/position) goal 을
"orientation 을 정하고 IK 로 config 를 찾는" 방식으로 푼다.** 5-DOF 팔은 임의의 6-DOF orientation 을
정확히 만들 수 없어(achievable orientation 이 위치마다 thin 한 2-manifold), 샘플러가 고르는 거의 모든
orientation 에서 IK 가 실패 → goal state 를 못 만든다. position-only 도 내부적으로 랜덤 orientation+IK
라 동일하게 실패. 즉 **pose/position goal 방식 자체가 5-DOF 에 비가능**(planner/tolerance 문제 아님).
`/compute_ik` 는 exact 6-DOF pose 를 요구해 5-DOF 에선 거의 `-31` — 이걸로 reachability 판단하면 오해.

### 해결 방법

goal 을 **JOINT config** 로 준다(5-DOF-aware). `scripts/sim/probe_ik.py`(`/compute_fk` 랜덤 FK 샘플링)
로 워크스페이스를 매핑해 위치 도달성·achievable tilt 를 확인하고, SM `pick_place_sm.py::_move_to` 를
다음으로 전환:
1. `RobotState.set_to_random_positions()` 로 in-process FK 랜덤 샘플링(joint bounds 자동 준수) →
   target(x,y,z) 근처(`fk_pos_gate`)에 down-ish(tool z tilt≤max) tip 을 두는 manipulator config 탐색.
2. 그 config 의 (도달 가능) orientation + 목표 위치로 `set_from_ik` 정밀화(seed=coarse config).
3. `arm.set_goal_state(robot_state=goal_rs)` → planner(cuMotion/OMPL)는 joint→joint collision-free 만 푼다.

(과거 in-process SM 이 `joint_fk` 를 쓴 것과 동일 원리. `/compute_ik` 의존을 버리는 게 핵심.)

### 확인 방법

`OMPL OK → (x,y,z) q=[...]` 로 approach→grasp→lift 가 전부 plan+exec. `Unable to sample`/
`INVERSE_KINEMATICS_FAILURE` 0건. ※ grasp 가 큐브를 실제로 쥐는지(grip 물리)는 별개 과제 —
moving_jaw 가 큐브를 감싸도록 강tilt·그리퍼 close·위치정확도 튜닝 필요(§PATH_E 6).

## Isaac Lab SO-101 해석적 FK 가 시뮬 실측과 ~50cm 어긋남 (USD root↔URDF base frame 불일치)

### 현상

URDF(so_arm101.urdf) joint origin 으로 유도한 closed-form FK 의 TCP 예측이 시뮬 실측
(`robot.data.body_pos_w["gripper"]` + grasp offset)과 위치·방향 모두 크게 어긋남.

### 오류 메시지

```
[CALIB] zero  pred=(+0.3941,-0.0000,+0.2071)  meas=(-0.0204,+0.3784,+0.2396)  err=562.2mm
[CALIB] max err = 579.6mm  (FAIL — 상수/부호 재보정 필요)
```

### 원인

두 frame 변환이 누락됨:
1. `PickCubeEnvCfg` robot root rot `(0,0,0,1)`(wxyz) = **yaw 180°** — world→base 변환에서
   위치 차만 빼면 회전이 무시된다.
2. `so101_follower.usd` 의 root body frame 자체가 URDF base_link 와 다름 —
   **yaw −90° 회전 + 원점 시프트 (0.0204, 0.0157, 0.0325 m)** (USD 변환 시 발생).

### 해결 방법

`pick_cube_state_machine.py` 의 `_world_to_base()`: root quat yaw 역회전 후
실측 fit 상수(BASE_XY_OFFSET/BASE_Z_OFFSET/BASE_YAW_OFFSET=+90°)로 URDF base frame 으로
환산. fit 은 `--calibrate` 모드의 8개 자세 FK 비교에서 잔차 최소화로 결정.

### 확인 방법

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
    scripts/environments/pick_cube_state_machine.py --calibrate --headless
# [CALIB] max err = 1.5mm  (OK (<5mm))
```

## Isaac Lab SO-101 SM 하강 grasp 에서 큐브를 찌르고 밀어냄 (joint 보간 호 운동 + fixed finger 간섭)

### 현상

top-down grasp 하강 중 손끝이 큐브 윗면/측면을 찍어 큐브가 밀리고(drift retry 연쇄),
어쩌다 잡혀도 비비다 들어가는 식. TCP 가 목표보다 8~26mm 위에서 정지.

### 오류 메시지

```
[SM] env0 Cube1: 하강 중 큐브 밀림 — retry 3/3
[SM] env0 Cube4: grasp 진입 — ... err_z=25.7mm pitch=-90° timeout=True
```

### 원인

복합 3건 (캘리브레이션·영상·"밀린 방향 vs 닫힘축 각도" 진단으로 분리):
1. **joint 공간 보간**: IK 1회 + slew 이동은 TCP 가 호를 그려 수평 성분으로 큐브를 침.
2. **PD 중력 처짐**: stiffness 17.8 의 정적 오차(lift ~0.14rad)로 TCP 가 목표 위에서 평형.
3. **fixed finger 간섭**: TCP(gripper_frame=닫힘 중점)에서 닫힘축 base-반대쪽 ~15mm 에
   fixed finger 가 있어, 30mm 큐브 면 위치와 정확히 겹침 — 수직 하강은 구조적으로 긁음.

### 해결 방법

1. z-ramp: 하강/상승/운반을 매 step Cartesian 직선 ramp 로 (joint 보간 호 제거).
2. 적분 보상: `q_bias += KI*(q_cmd - q_now)` (KI 0.06, clip ±0.35) 를 action 에 가산.
   ※ 목표 z 를 overshoot 으로 더 내리는 방식은 접촉 후에도 밀어붙여 큐브를 밀어냄 — 금지.
3. **side-approach**: 닫힘축의 **base 쪽**으로 3.5cm 비켜 수직 하강 후 수평 SLIDE 로
   큐브를 손가락 사이에 진입시키고 close. 비킴 방향이 base-반대쪽이면 slide 중
   fixed finger 가 선두로 면을 밀고 다님(err_xy ≈ slide_stop+15mm 로 판별 가능).
   비킨 지점이 그릇 테두리(중심 0.12m 이내)와 겹치면 그릇이 더 먼 부호로 전환.

### 확인 방법

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
    scripts/environments/pick_cube_state_machine.py \
    --num_envs 1 --active_objects 4 --object_radius_scale 0 --headless --seed 42
# [SM] env0 RESULT: 4/4 cubes in bowl.  (DR full 2env 는 6/8 — reach 경계 spawn 한계)
```

## Isaac Lab SO-101 SM release 직후 jaw 가 큐브를 퍼올려 날림 (top-down release + 상승)

### 현상

큐브를 그릇 상공/안에서 release 한 직후 RETREAT 상승에서 회전형 jaw 가 큐브 밑을
스쿱해 그릇 밖으로 날아감. release 게이트(그릇 중심 정렬)를 통과해도 placed=False.

### 오류 메시지

```
[SM] env2 Cube2: dist_bowl=0.197m z=0.724 placed=False   (release 게이트는 통과)
```

### 원인

복합 3건:
1. top-down 자세의 회전형 jaw 는 **수직 평면에서 호를 그리며** 여닫음 — open 직후
   상승하면 jaw 호가 낙하 중인 큐브 밑을 퍼올린다.
2. 낙하점 기준 오류: 쥔 큐브는 TCP(gripper_frame)에서 닫힘축 방향 2~3cm 오프셋 —
   TCP 를 그릇 중심에 맞추면 큐브가 테두리 빗면에 떨어져 튕겨 나감.
3. release 자세 재배향(pitch/roll 동시 90°)을 slew 풀속도로 하면 원심력으로
   쥔 큐브가 회전 중에 빠진다.

### 해결 방법

`pick_cube_state_machine.py` TRANSPORT/RELEASE 재설계:
1. **운반 중 pitch·wrist roll 동시 점진 보간**: xy ramp 진행률(frac)에
   pitch(현재→그릇 목표)와 roll(0→90°, release 자세)을 함께 실음. roll 90° 로
   jaw 개폐 평면이 접근축 주위로 돌아 옆으로 열림 → 퍼올림 불가.
   ※ IK 로 pitch 0° 재배향을 시도하면 그릇(r≈0.35)이 top-down 한계 밖이라
   ramp 시작 pitch -90° 가정이 즉시 IK 실패 → 폴백으로 무력화된다 (실측).
2. 하강 없이 안전고도에서 그대로 떨굼. 낙하점은 TRANSPORT 목표에 TCP-큐브
   실측 오프셋을 보정해 **큐브**가 그릇 중심 위에 오게.
3. open 후 0.4 s(12 step) 정지한 다음 RETREAT (z-ramp 시작점은 실측 TCP —
   고정 가정은 release 자세가 더 높을 때 '내려갔다 올라오며' 그릇을 친다).

### 확인 방법

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
    scripts/environments/pick_cube_state_machine.py \
    --num_envs 4 --active_objects 4 --headless --seed 0
# [SM] TOTAL: 16/16 cubes in bowl across 4 envs (100%).
```

## Isaac Lab cube DR 이 robot base 발치에 spawn — 접근 IK 부재로 수행 불가

### 현상

DR full 에서 일부 큐브가 base 에서 r≈0.11 에 spawn — 안전고도(desk+0.12) 접근
IK 해가 없어 APPROACH 부터 실패(재시도 소진). SO-101 은 base 에 가까울수록
위쪽 reach 가 급감한다.

### 오류 메시지

```
[SM] env3 Cube2: IK 실패(approach (...)) — 재시도 소진
```

### 원인

`_CUBE_SCATTER_Y_RANGE` 하한(-0.46)과 x 중앙대 조합이 base(1.84,-0.565)와
거리 0.105 까지 허용 — inner-reach(r<0.13) 영역.

### 해결 방법

`randomize_cubes_scattered` 에 `min_base_sep`(기본 0 비활성) 파라미터 추가,
pick_cube cfg 에서 0.135 지정 — rejection sampling 이 base 발치 후보를 기각
(직사각형 범위 축소와 달리 workspace 면적 보존). r 0.135~0.20 의 "잡을 수 있는데
못 드는" 영역은 SM 의 DRAG phase(낮게 쥔 채 r≈0.20 으로 끌기)가 처리.

### 확인 방법

위와 동일 명령 — `IK 실패(approach ...)` 0건, DRAG 발동 로그로 끌기 확인.


## Isaac Lab SO-101 SM 운반 시 팔이 위로 휘둘렸다 그릇에 내리꽂힘 (pitch 불연속)

### 현상

큐브를 들고 그릇으로 운반할 때 팔이 위로 크게 솟았다가 그릇 위로 떨어지며
그릇을 덜컹거림("슬램덩크"). TCP xy 는 Cartesian 직선 ramp 인데도 발생.

### 오류 메시지

(로그 없음 — 영상 3~5s 구간 프레임으로 확인)

### 원인

TRANSPORT 의 ik_reach 가 그릇 앞에서 pitch 를 −90°→−40° 로 **단번에** 완화
(그릇 r≈0.35 는 top-down 한계 밖) → slew 풀속도 재배향으로 팔 전체가 호를
그리며 재배치. **위치만 ramp 하고 자세(pitch)를 ramp 하지 않은 것**이 원인.

### 해결 방법

TRANSPORT 진입 시 목표 pitch(그릇 위치의 ik_reach 채택값)를 1회 계산해 두고,
xy ramp 진행률에 맞춰 pitch 를 시작값→목표값으로 점진 보간 (wrist roll 90°
release 자세 전환도 같은 frac 에 실음). `_solve_fixed_pitch(pitch_now,
roll_offset=roll_now)` 로 매 step 연속 자세 명령.

### 확인 방법

1env DR 영상 3~5s 구간 프레임 — 솟구침/내리꽂힘 소멸, 부드러운 호.
4env DR `[SM] TOTAL: 16/16 (100%)`.
