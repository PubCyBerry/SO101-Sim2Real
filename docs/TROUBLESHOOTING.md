# Troubleshooting

## 목차 <!-- omit in toc -->

- [WSL2 NTFS 마운트에서 uv sync 실패 (Operation not permitted)](#wsl2-ntfs-마운트에서-uv-sync-실패-operation-not-permitted)
- [uv-compile Too many open files panic (다코어 호스트, 모든 uv RUN)](#uv-compile-too-many-open-files-panic-다코어-호스트-모든-uv-run)
- [`uv pip install torch` 단계에서 nvidia CUDA 휠 다운로드 timeout](#uv-pip-install-torch-단계에서-nvidia-cuda-휠-다운로드-timeout)
- [torchcodec `c10::MessageLogger::stream` 심볼 누락으로 학습 DataLoader 크래시](#torchcodec-c10messageloggerstream-심볼-누락으로-학습-dataloader-크래시)
- [`torch.compile` 활성화 시 `InvalidCxxCompiler: No working C++ compiler found`](#torchcompile-활성화-시-invalidcxxcompiler-no-working-c-compiler-found)
- [lerobot 0.5.x 업그레이드 후 SmolVLA import 경로 변경 (`ImportError`)](#lerobot-05x-업그레이드-후-smolvla-import-경로-변경-importerror)
- [카메라 대역폭 제한](#카메라-대역폭-제한)
- [Docker 컨테이너에서 Vulkan 초기화 실패 (Linux)](#docker-컨테이너에서-vulkan-초기화-실패-linux)
- [WSL2 + Docker 에서 Isaac Sim Vulkan/GPU 가속 불가 (회피 불가)](#wsl2--docker-에서-isaac-sim-vulkangpu-가속-불가-회피-불가)
- [Windows 네이티브 bare `isaacsim` Full App 이 app ready 직후 종료](#windows-네이티브-bare-isaacsim-full-app-이-app-ready-직후-종료)
- [`lerobot record` 키보드 컨트롤이 동작하지 않음 (WSLg + Windows Terminal)](#lerobot-record-키보드-컨트롤이-동작하지-않음-wslg--windows-terminal)
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

`Dockerfile.lerobot` / `Dockerfile.smolvla` 의 **uv 를 호출하는 모든 RUN 명령** 안에서 `ulimit -Sn` 으로 soft 한도를 직접 끌어올린다. hard 한도가 이미 1048576 이므로 soft 만 raise 하면 된다.

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

`docker/Dockerfile.lerobot` / `docker/Dockerfile.smolvla` 의 Stage 4 (`torch-layer`) 와 Stage 5 (`teleop-deps` / `policy-deps`) RUN 에 세 가지를 함께 적용한다.

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
