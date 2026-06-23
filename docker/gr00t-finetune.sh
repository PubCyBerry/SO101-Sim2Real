#!/usr/bin/env bash
# NVIDIA examples/finetune.sh 호환 wrapper.
# 이미지 재빌드 없이 학습 하이퍼파라미터를 env로 조절할 수 있도록 /host에 bind-mount한다.

set -x -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-10000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
COLOR_JITTER_ENABLE="${COLOR_JITTER_ENABLE:-true}"

BASE_MODEL_PATH=""
DATASET_PATH=""
MODALITY_CONFIG_PATH=""
EMBODIMENT_TAG=""
OUTPUT_DIR=""
EXPERIMENT_NAME=""
WANDB_PROJECT=""
STATE_DROPOUT_PROB=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash gr00t-finetune.sh \
  --base-model-path <path> \
  --dataset-path <path> \
  --embodiment-tag <tag> \
  --output-dir <path> \
  [--modality-config-path <path>] \
  [--state-dropout-prob <value>] \
  [--save-only-model] \
  [-- <extra launch_finetune.py args>...]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-model-path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --modality-config-path)
            MODALITY_CONFIG_PATH="$2"
            shift 2
            ;;
        --embodiment-tag)
            EMBODIMENT_TAG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --state-dropout-prob)
            STATE_DROPOUT_PROB="$2"
            shift 2
            ;;
        --save-only-model)
            SAVE_ONLY_MODEL=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

for required_var in BASE_MODEL_PATH DATASET_PATH EMBODIMENT_TAG OUTPUT_DIR; do
    if [ -z "${!required_var}" ]; then
        echo "Missing required argument: ${required_var}" >&2
        usage >&2
        exit 1
    fi
done

WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG+=(--use_wandb)
fi

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag "$EMBODIMENT_TAG"
    --num_gpus "$NUM_GPUS"
    --output_dir "$OUTPUT_DIR"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 5
    --max_steps "$MAX_STEPS"
    --warmup_ratio "$WARMUP_RATIO"
    --weight_decay "$WEIGHT_DECAY"
    --learning_rate "$LEARNING_RATE"
    "${WANDB_FLAG[@]}"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --shard_size "$SHARD_SIZE"
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH"
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE"
)

if [ "$COLOR_JITTER_ENABLE" = "true" ]; then
    LAUNCH_CMD+=(--color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08)
else
    # tyro dict option을 값 없이 전달하면 {}가 되어 base checkpoint의 jitter 설정 상속을 끊는다.
    LAUNCH_CMD+=(--color_jitter_params)
fi

if [ -n "$MODALITY_CONFIG_PATH" ]; then
    LAUNCH_CMD+=(--modality_config_path "$MODALITY_CONFIG_PATH")
fi
if [ -n "$EXPERIMENT_NAME" ]; then
    LAUNCH_CMD+=(--experiment_name "$EXPERIMENT_NAME")
fi
if [ -n "$WANDB_PROJECT" ]; then
    LAUNCH_CMD+=(--wandb_project "$WANDB_PROJECT")
fi
if [ -n "$STATE_DROPOUT_PROB" ]; then
    LAUNCH_CMD+=(--state_dropout_prob "$STATE_DROPOUT_PROB")
fi
if [ -n "${SAVE_ONLY_MODEL:-}" ]; then
    LAUNCH_CMD+=(--save_only_model)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec python "${LAUNCH_CMD[@]}"
fi

exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_CMD[@]}"
