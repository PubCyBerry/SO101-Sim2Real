#!/usr/bin/env bash
# =============================================================================
# server-entrypoint.sh — `lerobot-policy-server` 서비스 (Dockerfile.smolvla) 진입점
#
# Async inference 정책 서버 전용 진입점. 로봇 직결 워크플로(teleop / record /
# replay / calibrate / train 등)는 `docker/lerobot-entrypoint.sh` 에 분리되어
# 있으며 본 이미지에서는 호출되지 않는다 (`lerobot[feetech]` 미설치).
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   prepare-model : huggingface-cli download — 호스트 HF 캐시에 모델 받기
#   policy-server : lerobot.async_inference.policy_server — gRPC 추론 서버
#   train         : lerobot-train — Policy 학습 (인자 완전 위임, SmolVLA 등)
#   eval          : lerobot-eval  — Policy 평가 (인자 완전 위임)
#   info          : lerobot-info — LeRobot / Python / 시스템 정보 출력
#   bash | shell  : 인터랙티브 Bash 쉘
#   python <args> : python 직접 실행
#   <기타>         : 명령 그대로 exec (디버깅용)
#
# ■ 환경 변수 요약 (docker-compose.yaml ↔ .env 에서 주입)
#   prepare-model : MODEL_REPO_ID  MODEL_REVISION  PREPARE_MODEL_EXTRA_ARGS
#   policy-server : POLICY_SERVER_HOST  POLICY_SERVER_PORT  POLICY_FPS
#                   INFERENCE_LATENCY   OBS_QUEUE_TIMEOUT   POLICY_SERVER_EXTRA_ARGS
#   train / eval  : 인자 완전 위임. .env 의 POLICY_TYPE / POLICY_PATH / DATASET / WANDB
#                   변수를 셸 보간으로 채워 호출한다 (README §Policy 학습 참조).
#                   NUM_WORKERS(기본 8) / COMPILE_MODEL / MIXED_PRECISION(bf16) /
#                   NUM_PROCESSES(2 이상이면 accelerate launch DDP 전환) 로 속도 최적화.
#   공통           : (HF 캐시는 명명 볼륨 lerobot_hf_cache → /root/.cache/huggingface)
# =============================================================================
set -euo pipefail

# ── prepare-model 환경 변수 ─────────────────────────────────────────────────
# 명명 볼륨 `lerobot_hf_cache` (= /root/.cache/huggingface) 에 모델 가중치를
# 미리 받아 두는 모드. 같은 볼륨을 lerobot 과 lerobot-policy-server 가
# 공유하므로 한 번만 받으면 양쪽이 모두 사용한다.
MODEL_REPO_ID="${MODEL_REPO_ID:-lerobot/smolvla_base}"
MODEL_REVISION="${MODEL_REVISION:-main}"
PREPARE_MODEL_EXTRA_ARGS="${PREPARE_MODEL_EXTRA_ARGS:-}"

# ── policy-server 환경 변수 ─────────────────────────────────────────────────
# 모델/디바이스는 클라이언트가 SendPolicyInstructions RPC 로 주입하므로 서버
# 자체는 policy-agnostic. 컨테이너 내부 bind 주소 (network_mode=host 면 호스트
# 인터페이스에 그대로 노출).
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8080}"
# 컨트롤 루프 FPS (RECORD_FPS 와 독립)
POLICY_FPS="${POLICY_FPS:-30}"
# 목표 inference latency (초). 클라이언트 chunk_size_threshold 와 함께 동작.
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
# 관측 큐 timeout (초)
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
POLICY_SERVER_EXTRA_ARGS="${POLICY_SERVER_EXTRA_ARGS:-}"

# ── train 환경 변수 ──────────────────────────────────────────────────────────
HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
POLICY_TYPE="${POLICY_TYPE:-}"
POLICY_PATH="${POLICY_PATH:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
JOB_NAME="${JOB_NAME:-}"
DEVICE="${DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
# ── 학습 속도 최적화 환경 변수 ────────────────────────────────────────────────
# NUM_WORKERS: 데이터로더 병렬 워커 수. 224코어 서버 기준 8이 기본값.
#   → lerobot 기본값(4)보다 높여 데이터 로딩 병목 완화.
#   → GPU 메모리 부족 시 낮추고, 빠른 NVMe + 고코어 환경에서는 16까지 늘릴 수 있음.
NUM_WORKERS="${NUM_WORKERS:-8}"
# COMPILE_MODEL: torch.compile 활성화 (true/false).
#   → 첫 번째 스텝에 컴파일 비용(수 분)이 발생하나 이후 스텝이 ~20-30% 빨라짐.
#   → 장기 학습(10K+ steps)에서 손익분기점 도달. 단기 디버깅에는 false 권장.
COMPILE_MODEL="${COMPILE_MODEL:-false}"
# COMPILE_MODE: torch.compile 모드.
#   reduce-overhead  — 재컴파일 최소화, 안정적 (기본)
#   max-autotune     — 커널 자동 탐색으로 최대 속도, 컴파일 시간 더 김
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
# NUM_PROCESSES: accelerate launch 프로세스 수 (= 사용 GPU 수).
#   1   — 단일 GPU (기본, Windows RTX A4000)
#   2   — 멀티 GPU (Linux H100 × 2)
#   → 2 이상이면 'accelerate launch --num_processes' 로 자동 전환.
NUM_PROCESSES="${NUM_PROCESSES:-1}"
# MIXED_PRECISION: accelerate 혼합 정밀도 모드.
#   bf16 — H100/A100 등 Ampere+ 에서 권장 (수치 안정성 우수, 속도↑)
#   fp16 — 구형 Volta/Turing GPU 호환
#   no   — 혼합 정밀도 미사용 (디버깅용)
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# ── 색상 출력 유틸 ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── GPU 확인 ──────────────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        info "NVIDIA GPU 감지됨:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
            while IFS= read -r line; do info "  GPU: $line"; done
    else
        warn "nvidia-smi 를 찾을 수 없습니다. CPU 전용으로 실행됩니다."
    fi
}

# ── LeRobot import 가능 여부 확인 (정책 서버 이미지는 lerobot[smolvla]만 설치) ──
check_lerobot() {
    local ver
    # 빌드 시점에 기록한 버전 파일을 우선 읽어 Python 기동 오버헤드를 제거.
    if [[ -f /opt/lerobot_version.txt ]]; then
        ver=$(cat /opt/lerobot_version.txt)
    else
        ver=$(python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))" 2>/dev/null || echo "unknown")
    fi
    info "LeRobot 버전: ${ver}"
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  LeRobot 0.4.4 Policy Server"
echo "========================================================"

CMD="${1:-policy-server}"

# GPU 가 필요 없는 모드(bash/python/info/prepare-model)는 체크를 건너뜀.
# prepare-model 은 huggingface-cli download 만 실행하므로 GPU 불필요.
case "$CMD" in
  bash|shell|python|info|prepare-model) ;;
  *) check_gpu; check_lerobot ;;
esac

case "$CMD" in

  # ────────────────────────────────────────────────────────────────────────────
  # prepare-model — HF 캐시 명명 볼륨에 모델 가중치 사전 다운로드
  #
  # 명명 볼륨 `lerobot_hf_cache` 가 컨테이너의 `/root/.cache/huggingface` 로
  # 마운트되어 있어, 두 서비스(lerobot, lerobot-policy-server) 가 동일 볼륨을
  # 공유한다. 한 번만 받으면 양쪽이 모두 사용한다. 다른 머신으로 옮기려면
  # `docker run ... -v lerobot_hf_cache:/cache alpine tar czf ...` 로 export.
  #
  # [env var → CLI arg 매핑]
  #   MODEL_REPO_ID            → huggingface-cli download <repo_id>
  #   MODEL_REVISION           → --revision (기본 main)
  #   PREPARE_MODEL_EXTRA_ARGS → 추가 인자 (예: --include "*.safetensors")
  #
  # [위치 인자 사용]
  #   첫 번째 인자가 있으면 MODEL_REPO_ID 를 덮어쓴다 — 여러 모델을 빠르게 받을 때 유용.
  #
  # 예시:
  #   # 기본(env): SmolVLA 베이스 받기
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     lerobot-policy-server prepare-model
  #
  #   # 위치 인자로 다른 모델 받기
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     lerobot-policy-server prepare-model nvidia/GR00T-N1.5-3B
  #
  #   # 추가 인자 전달 (특정 파일 패턴만)
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     -e PREPARE_MODEL_EXTRA_ARGS='--include *.safetensors *.json' \
  #     lerobot-policy-server prepare-model
  # ────────────────────────────────────────────────────────────────────────────
  prepare-model)
    shift || true
    # 위치 인자가 있으면 env 보다 우선
    if [[ $# -gt 0 ]]; then
        MODEL_REPO_ID="$1"
        shift
    fi
    if [[ -z "${MODEL_REPO_ID}" ]]; then
        error "MODEL_REPO_ID 가 비어 있습니다."
        error "  → .env: MODEL_REPO_ID=lerobot/smolvla_base"
        error "  → 또는: prepare-model <repo_id>"
        exit 1
    fi
    info "── Model Download 시작 ───────────────────────────"
    info "  Repo     → ${MODEL_REPO_ID}"
    info "  Revision → ${MODEL_REVISION}"
    info "  Cache    → /root/.cache/huggingface  (명명 볼륨 lerobot_hf_cache)"
    exec hf download \
        "${MODEL_REPO_ID}" \
        --revision="${MODEL_REVISION}" \
        ${PREPARE_MODEL_EXTRA_ARGS} \
        "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # policy-server — Async inference policy server (gRPC)
  #
  # lerobot.async_inference.policy_server 를 gRPC :PORT 에 띄운다.
  # 서버는 policy-agnostic: 모델 종류·체크포인트·디바이스는 클라이언트
  # (`lerobot.async_inference.robot_client`) 가 SendPolicyInstructions RPC 로
  # 주입한다. SmolVLA 의 경우 클라이언트가
  #   --policy_type=smolvla
  #   --pretrained_name_or_path=lerobot/smolvla_base
  #   --policy_device=cuda
  # 같은 인자를 전달.
  #
  # [env var → CLI arg 매핑]
  #   POLICY_SERVER_HOST  → --host                 (기본 0.0.0.0)
  #   POLICY_SERVER_PORT  → --port                 (기본 8080)
  #   POLICY_FPS          → --fps                  (기본 30)
  #   INFERENCE_LATENCY   → --inference_latency    (기본 0.033)
  #   OBS_QUEUE_TIMEOUT   → --obs_queue_timeout    (기본 2)
  #   POLICY_SERVER_EXTRA_ARGS → 추가 인자 그대로 전달
  #
  # 예시:
  #   docker compose --env-file .env -f docker/docker-compose.yaml \
  #     up -d lerobot-policy-server
  #
  # 클라이언트 예시 (같은 호스트의 lerobot 컨테이너 안에서):
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot \
  #     python -m lerobot.async_inference.robot_client \
  #       --server_address=127.0.0.1:8080 \
  #       --policy_type=smolvla \
  #       --pretrained_name_or_path=lerobot/smolvla_base \
  #       --policy_device=cuda \
  #       --robot.type=so101_follower --robot.port=/dev/ttyACM1 \
  #       --task='pick the pen' --actions_per_chunk=50 \
  #       --chunk_size_threshold=0.5
  # ────────────────────────────────────────────────────────────────────────────
  policy-server)
    info "── Policy Server 시작 (gRPC) ─────────────────────"
    info "  Bind           → ${POLICY_SERVER_HOST}:${POLICY_SERVER_PORT}"
    info "  FPS            → ${POLICY_FPS}"
    info "  Inference Lat  → ${INFERENCE_LATENCY} s"
    info "  Obs Queue TO   → ${OBS_QUEUE_TIMEOUT} s"
    info "  ※ 모델·디바이스는 클라이언트 SendPolicyInstructions 로 주입"
    shift || true
    exec python -m lerobot.async_inference.policy_server \
      --host=${POLICY_SERVER_HOST} \
      --port=${POLICY_SERVER_PORT} \
      --fps=${POLICY_FPS} \
      --inference_latency=${INFERENCE_LATENCY} \
      --obs_queue_timeout=${OBS_QUEUE_TIMEOUT} \
      ${POLICY_SERVER_EXTRA_ARGS} \
      "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # train — Policy 학습 (모든 인자를 lerobot-train 에 완전 위임)
  #
  # 본 이미지는 smolvla + async 의존성을 모두 포함하므로 SmolVLA / 기타 정책 학습
  # 가능. lerobot 이미지에는 이 의존성이 없어 이 모드를 옮겨 두었다.
  # datasets 디렉터리(/workspace/datasets)와 outputs(/workspace/outputs)가 호스트
  # bind mount 되어 학습 결과·데이터셋이 호스트와 공유된다.
  #
  # [주요 CLI 인자]
  #   --dataset.repo_id=<str>         : 학습 데이터셋 HF Hub ID (필수)
  #   --dataset.root=<path>           : 로컬 저장 루트 (기본 /workspace/datasets/...)
  #   --policy.type=<str>             : 모델 타입 (act / diffusion / smolvla / ...)
  #   --policy.path=<str>             : 사전학습 체크포인트 (e.g. lerobot/smolvla_base)
  #   --policy.repo_id=<str>          : 결과 체크포인트 push 대상
  #   --policy.push_to_hub=true|false : HF Hub 자동 푸시 (기본 false)
  #   --output_dir=<path>             : 체크포인트·로그 출력
  #   --job_name=<str>                : 실행 이름 (WandB 표시)
  #   --batch_size=<int>              : 배치 크기 (기본 8)
  #   --steps=<int>                   : 총 학습 스텝 수 (기본 100000)
  #   --wandb.enable=true|false       : WandB 로깅 (기본 false)
  #
  # 예시 (SmolVLA fine-tune):
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     lerobot-policy-server train \
  #       --policy.path=lerobot/smolvla_base \
  #       --policy.repo_id=${HF_USER}/smolvla_pick_pen \
  #       --policy.push_to_hub=true \
  #       --dataset.repo_id=${HF_DATASET_REPO_ID} \
  #       --output_dir=${OUTPUT_DIR} \
  #       --steps=20000 --batch_size=64 \
  #       --job_name=smolvla_pick_pen --wandb.enable=true
  # ────────────────────────────────────────────────────────────────────────────
  train)
    info "── Train 시작 ────────────────────────────────────"
    shift
    TRAIN_ARGS=()
    [[ -n "${HF_DATASET_REPO_ID}" ]] && TRAIN_ARGS+=("--dataset.repo_id=${HF_DATASET_REPO_ID}")
    [[ -n "${DATASET_ROOT}" ]]       && TRAIN_ARGS+=("--dataset.root=${DATASET_ROOT}")
    [[ -n "${POLICY_TYPE}" ]]        && TRAIN_ARGS+=("--policy.type=${POLICY_TYPE}")
    [[ -n "${POLICY_PATH}" ]]        && TRAIN_ARGS+=("--policy.path=${POLICY_PATH}")
    [[ -n "${POLICY_REPO_ID}" ]]     && TRAIN_ARGS+=("--policy.repo_id=${POLICY_REPO_ID}")
    [[ -n "${OUTPUT_DIR}" ]]         && TRAIN_ARGS+=("--output_dir=${OUTPUT_DIR}")
    [[ -n "${TRAIN_STEPS}" ]]        && TRAIN_ARGS+=("--steps=${TRAIN_STEPS}")
    [[ -n "${BATCH_SIZE}" ]]         && TRAIN_ARGS+=("--batch_size=${BATCH_SIZE}")
    [[ -n "${JOB_NAME}" ]]           && TRAIN_ARGS+=("--job_name=${JOB_NAME}")
    [[ -n "${WANDB_ENABLE}" ]]       && TRAIN_ARGS+=("--wandb.enable=${WANDB_ENABLE}")
    [[ -n "${DEVICE}" ]]             && TRAIN_ARGS+=("--policy.device=${DEVICE}")
    # ── 속도 최적화 인자 ──────────────────────────────────────────────────────
    TRAIN_ARGS+=("--num_workers=${NUM_WORKERS}")
    [[ "${COMPILE_MODEL}" == "true" ]] && TRAIN_ARGS+=(
        "--policy.compile_model=true"
        "--policy.compile_mode=${COMPILE_MODE}"
    )
    info "  Dataset      → ${HF_DATASET_REPO_ID:-<미설정>}"
    info "  Policy       → type=${POLICY_TYPE:-<미설정>}  path=${POLICY_PATH:-none}"
    info "  Output       → ${OUTPUT_DIR:-<미설정>}"
    info "  Steps        → ${TRAIN_STEPS}  Batch → ${BATCH_SIZE}  Device → ${DEVICE}"
    info "  Workers      → ${NUM_WORKERS}"
    info "  Compile      → ${COMPILE_MODEL}  mode=${COMPILE_MODE}"
    info "  Precision    → ${MIXED_PRECISION}  Processes → ${NUM_PROCESSES}"
    # ── 멀티-GPU / 혼합 정밀도 실행 분기 ────────────────────────────────────────
    # NUM_PROCESSES > 1: accelerate launch 로 DDP 멀티-GPU 학습
    # NUM_PROCESSES = 1: 환경변수 ACCELERATE_MIXED_PRECISION 만 주입해 단일 GPU 실행
    #   accelerate 는 Accelerator() 초기화 시 이 env var 를 자동으로 읽는다.
    # TRAIN_EXTRA_ARGS: word-split 의도적 (복수 플래그 지원)
    # "$@": 추가 CLI 인자. env var 빌드 값보다 뒤에 위치해 last-wins 로 덮어씀
    if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
        info "  → accelerate launch (DDP, ${NUM_PROCESSES} GPU)"
        exec accelerate launch \
            --mixed_precision="${MIXED_PRECISION}" \
            --num_processes="${NUM_PROCESSES}" \
            -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}" ${TRAIN_EXTRA_ARGS} "$@"
    else
        info "  → lerobot-train (단일 GPU)"
        export ACCELERATE_MIXED_PRECISION="${MIXED_PRECISION}"
        exec lerobot-train "${TRAIN_ARGS[@]}" ${TRAIN_EXTRA_ARGS} "$@"
    fi
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # eval — Policy 평가 및 롤아웃 (모든 인자를 lerobot-eval 에 완전 위임)
  #
  # [주요 CLI 인자]
  #   --policy.path=<str>           : Hub ID 또는 로컬 체크포인트 경로 (필수)
  #   --env.type=<str>              : 평가 환경 타입 (pusht / aloha / xarm / ...)
  #   --eval.n_episodes=<int>       : 평가 에피소드 수 (기본 50)
  #   --eval.batch_size=<int>       : 동시 병렬 롤아웃 수 (기본 50)
  #   --output_dir=<path>           : 결과 저장 경로
  #   --job_name=<str>              : 실행 이름
  #
  # 예시:
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     lerobot-policy-server eval \
  #       --policy.path=${HF_USER}/smolvla_pick_pen \
  #       --env.type=pusht --eval.n_episodes=20
  # ────────────────────────────────────────────────────────────────────────────
  eval)
    info "── Eval 시작 ─────────────────────────────────────"
    shift
    exec lerobot-eval "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # info — LeRobot / Python / 시스템 정보 출력 (CLI 인자 없음)
  # ────────────────────────────────────────────────────────────────────────────
  info)
    if command -v lerobot-info &>/dev/null; then
        exec lerobot-info
    else
        python -c "import sys, lerobot, torch; print(f'lerobot={lerobot.__version__}, python={sys.version.split()[0]}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
    fi
    ;;

  # ── 인터랙티브 쉘 / 직접 실행 ────────────────────────────────────────────────
  bash|shell)
    info "인터랙티브 쉘로 진입합니다."
    exec /bin/bash
    ;;

  python)
    shift
    exec python "$@"
    ;;

  *)
    exec "$@"
    ;;

esac
