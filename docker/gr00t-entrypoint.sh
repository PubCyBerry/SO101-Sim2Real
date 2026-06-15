#!/usr/bin/env bash
# =============================================================================
# gr00t-entrypoint.sh — `gr00t` 서비스(NVIDIA Isaac-GR00T 네이티브 이미지) 진입점
#
# NVIDIA Isaac-GR00T repo(ref_repos/Isaac-GR00T/docker/Dockerfile)를 **무수정**으로
# 빌드한 이미지에 bind-mount + entrypoint override 로만 우리 워크플로를 얹는다.
# repo·venv 는 /workspace, 우리 자산(datasets/outputs/configs/이 스크립트)은 /host.
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   convert    : HF v3 데이터셋 → LeRobot v2.1 변환 + meta/modality.json 주입
#   finetune   : nvidia/GR00T-N1.7-3B → SO-101 데이터셋 finetune (examples/finetune.sh)
#   zmq-server : gr00t/eval/run_gr00t_server.py — ZMQ(:5555) 추론 서버 (Gr00tPolicy)
#   bash|shell : 인터랙티브 쉘
#   python     : python 직접 실행
#   <기타>      : 명령 그대로 exec
#
# ■ 환경 변수 (.env + env/<POLICY_PROFILE>.env 에서 주입)
#   공통       : HF_DATASET_REPO_ID  GROOT_EMBODIMENT_TAG
#   convert    : GROOT_MODALITY_JSON(기본 /host/configs/so101_modality.json)
#   finetune   : GROOT_BASE_MODEL  GROOT_MODALITY_CONFIG  JOB_NAME
#                NUM_GPUS  TRAIN_STEPS  BATCH_SIZE  NUM_WORKERS  SAVE_STEPS  WANDB_ENABLE
#   zmq-server : GROOT_CHECKPOINT  GROOT_ZMQ_PORT
# =============================================================================
set -euo pipefail

# ── 색상 출력 ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── non-root HOME/캐시 (UID 1000 실행 — / 에 못 씀) ──────────────────────────
# triton(torch.compile)·wandb·matplotlib 등이 HOME/XDG 캐시를 / 에 만들려다
# PermissionError 가 난다. host-writable bind-mount(/host/outputs 하위)로 강제.
export HOME=/host/outputs/.gr00t-home
export XDG_CACHE_HOME="${HOME}/.cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export WANDB_DIR="${HOME}/wandb"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

# ── 공통 경로/기본값 ──────────────────────────────────────────────────────────
DATASET_ROOT_DIR="/host/datasets/groot"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-new_embodiment}"
GROOT_BASE_MODEL="${GROOT_BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
GROOT_MODALITY_CONFIG="${GROOT_MODALITY_CONFIG:-/host/configs/so101_config.py}"
GROOT_MODALITY_JSON="${GROOT_MODALITY_JSON:-/host/configs/so101_modality.json}"
GROOT_ZMQ_PORT="${GROOT_ZMQ_PORT:-5555}"

CMD="${1:-zmq-server}"
shift || true   # 모드 인자 제거 → 이후 "$@" 는 모드별 추가 인자만 (finetune.sh 등에 전달)

echo "========================================================"
echo "  GR00T-N1.7  (mode: ${CMD})"
echo "========================================================"
if command -v nvidia-smi &>/dev/null && [[ "$CMD" != "bash" && "$CMD" != "shell" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while IFS= read -r l; do info "GPU: $l"; done
fi

case "$CMD" in

  # ──────────────────────────────────────────────────────────────────────────
  # convert — HF v3.0 데이터셋 → LeRobot v2.1 + meta/modality.json 주입
  #
  # gr00t main venv 에는 lerobot 이 없다. 변환 스크립트는 repo 의 별도 uv 프로젝트
  # (scripts/lerobot_conversion, lerobot 커밋 핀)에서 돌린다. 비-root(host UID) 라
  # /workspace 하위에 venv 를 못 만들므로 UV_* 를 /host(쓰기가능·gr00t_uv_cache 볼륨)로.
  #
  # ⚠ 1회성: 변환 후 데이터셋은 v2.1 이 되어 재실행 시 검증 실패. 재변환하려면
  #   convert_v3_to_v2.py --force-conversion 으로 재다운로드.
  # ──────────────────────────────────────────────────────────────────────────
  convert)
    : "${HF_DATASET_REPO_ID:?HF_DATASET_REPO_ID 필요 (예: <hf_user>/so101_sim_pick_cube)}"
    DATASET_DIR="${DATASET_ROOT_DIR}/${HF_DATASET_REPO_ID}"
    # 비-root(UID 1000) 실행: root 소유 named volume·/workspace 에 못 씀 →
    # host bind-mount(/host/outputs, host UID 소유)에 conv 전용 venv·uv 캐시를 둔다.
    CONV_ENV="/host/outputs/.uv-cache/conv-venv"
    export UV_CACHE_DIR="/host/outputs/.uv-cache/cache"
    mkdir -p "${DATASET_ROOT_DIR}" "${UV_CACHE_DIR}"

    info "── v3→v2 변환 ────────────────────────────────────"
    info "  Repo    → ${HF_DATASET_REPO_ID}"
    info "  Root    → ${DATASET_ROOT_DIR}"
    info "  Target  → ${DATASET_DIR}"
    # conv 전용 venv. 로컬 프로젝트(scripts/lerobot_conversion)를 빌드하면 setuptools 가
    # root 소유 소스 디렉터리에 egg-info 를 쓰려다 실패 → 프로젝트 빌드 없이 pyproject 의
    # deps 만 직접 설치한다(lerobot git 은 uv 캐시 temp 에서 빌드되어 쓰기 OK). 스크립트는
    # 직접 실행. uv pip install 은 멱등이라 재실행 시 빠르게 확인만 한다.
    #   ⚠ lerobot 핀(c75455a6)은 scripts/lerobot_conversion/pyproject.toml 과 동기 유지.
    [[ -x "${CONV_ENV}/bin/python" ]] || uv venv "${CONV_ENV}" --python 3.10
    info "  conv venv 의존성 설치/확인 (lerobot 핀)…"
    GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "${CONV_ENV}/bin/python" \
        huggingface_hub jsonlines numpy pyarrow tqdm \
        "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
    "${CONV_ENV}/bin/python" /workspace/scripts/lerobot_conversion/convert_v3_to_v2.py \
        --repo-id "${HF_DATASET_REPO_ID}" \
        --root "${DATASET_ROOT_DIR}"

    if [[ ! -d "${DATASET_DIR}/meta" ]]; then
        error "변환 후 ${DATASET_DIR}/meta 없음 — 변환 실패."
        exit 1
    fi
    info "── modality.json 주입 ───────────────────────────"
    info "  ${GROOT_MODALITY_JSON} → ${DATASET_DIR}/meta/modality.json"
    cp "${GROOT_MODALITY_JSON}" "${DATASET_DIR}/meta/modality.json"

    info "변환 완료. meta/ 내용:"
    ls -1 "${DATASET_DIR}/meta"
    ;;

  # ──────────────────────────────────────────────────────────────────────────
  # finetune — examples/finetune.sh (num-gpus=1 → python, >1 → torchrun 자동 분기)
  #
  # finetune.sh 는 튜닝 파라미터를 **환경변수**로 받고(NUM_GPUS/MAX_STEPS/...),
  # model/dataset/embodiment/modality/output 만 플래그로 받는다(미인식 플래그는 거부).
  # ──────────────────────────────────────────────────────────────────────────
  finetune)
    : "${HF_DATASET_REPO_ID:?HF_DATASET_REPO_ID 필요}"
    DATASET_DIR="${DATASET_ROOT_DIR}/${HF_DATASET_REPO_ID}"
    if [[ ! -d "${DATASET_DIR}/meta" ]]; then
        error "변환 데이터셋 없음: ${DATASET_DIR} — 먼저 'convert' 모드 실행."
        exit 1
    fi
    if [[ ! -f "${DATASET_DIR}/meta/modality.json" ]]; then
        error "${DATASET_DIR}/meta/modality.json 없음 — convert 시 주입 누락."
        exit 1
    fi
    OUT="/host/outputs/train/${JOB_NAME:-so101_groot_n17_pick_cube}"

    # finetune.sh 가 읽는 환경변수
    export NUM_GPUS="${NUM_GPUS:-1}"
    export MAX_STEPS="${TRAIN_STEPS:-20000}"
    export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-32}"
    export DATALOADER_NUM_WORKERS="${NUM_WORKERS:-4}"
    export SAVE_STEPS="${SAVE_STEPS:-1000}"
    if [[ "${WANDB_ENABLE:-false}" == "true" ]]; then export USE_WANDB=1; else export USE_WANDB=0; fi

    info "── GR00T finetune ───────────────────────────────"
    info "  Base     → ${GROOT_BASE_MODEL}"
    info "  Dataset  → ${DATASET_DIR}"
    info "  Modality → ${GROOT_MODALITY_CONFIG}  (tag=${GROOT_EMBODIMENT_TAG})"
    info "  Output   → ${OUT}"
    info "  GPUs=${NUM_GPUS}  Steps=${MAX_STEPS}  GlobalBatch=${GLOBAL_BATCH_SIZE}  Workers=${DATALOADER_NUM_WORKERS}  WandB=${USE_WANDB}"
    cd /workspace
    exec bash examples/finetune.sh \
        --base-model-path "${GROOT_BASE_MODEL}" \
        --dataset-path "${DATASET_DIR}" \
        --embodiment-tag "${GROOT_EMBODIMENT_TAG}" \
        --modality-config-path "${GROOT_MODALITY_CONFIG}" \
        --output-dir "${OUT}" \
        "$@"
    ;;

  # ──────────────────────────────────────────────────────────────────────────
  # zmq-server — Gr00tPolicy ZMQ 추론 서버 (policy-server-groot bridge 가 접속)
  #
  # NEW_EMBODIMENT modality config 는 finetune 체크포인트에 baked → 별도 전달 불요.
  # ──────────────────────────────────────────────────────────────────────────
  zmq-server)
    : "${GROOT_CHECKPOINT:?GROOT_CHECKPOINT 필요 (finetune 체크포인트 경로 또는 HF id)}"
    if [[ "${GROOT_CHECKPOINT}" == /* && ! -e "${GROOT_CHECKPOINT}" ]]; then
        error "체크포인트 경로 없음: ${GROOT_CHECKPOINT} — finetune 완료/경로 확인."
        exit 1
    fi
    info "── GR00T ZMQ 추론 서버 ──────────────────────────"
    info "  Model → ${GROOT_CHECKPOINT}  (tag=${GROOT_EMBODIMENT_TAG})"
    info "  Bind  → 0.0.0.0:${GROOT_ZMQ_PORT}"
    cd /workspace
    exec python gr00t/eval/run_gr00t_server.py \
        --model-path "${GROOT_CHECKPOINT}" \
        --embodiment-tag "${GROOT_EMBODIMENT_TAG}" \
        --device cuda \
        --host 0.0.0.0 \
        --port "${GROOT_ZMQ_PORT}" \
        "$@"
    ;;

  bash|shell)
    info "인터랙티브 쉘로 진입합니다."
    exec /bin/bash
    ;;

  python)
    exec python "$@"
    ;;

  *)
    exec "$@"
    ;;

esac
