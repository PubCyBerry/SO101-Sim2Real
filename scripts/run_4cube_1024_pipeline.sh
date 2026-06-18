#!/usr/bin/env bash
# =============================================================================
# 4-cube 1024 데이터 생성 + ACT/SmolVLA/GR00T-N1.7 등량 학습 무중단 파이프라인
#
# 등량 연산 기준 = 640k samples:
#   ACT     batch 32 × 20k step
#   SmolVLA batch 32 × 20k step
#   GR00T   batch  8 × 80k step
#
# 단일 GPU 공유 → 전 스테이지 직렬. 학습 스테이지는 VRAM 확보될 때까지 대기
# (사용자 라이브 eval 과 coexist; GR00T 는 eval 정지 후 자동 시작).
#
# 구동:
#   cd /home/konan147/Workspaces/SO101-Sim2Real
#   nohup setsid bash scripts/run_4cube_1024_pipeline.sh \
#     > outputs/p5_logs/4cube1024_pipeline.log 2>&1 &
# =============================================================================
set -uo pipefail

REPO=/home/konan147/Workspaces/SO101-Sim2Real
cd "$REPO" || exit 1

LOGDIR="$REPO/outputs/p5_logs"
mkdir -p "$LOGDIR"

DATASET_REPO="taehunkim/so101_sim_pick_cube_4cube_1024"
DATASET_DIR="outputs/so101_sim_pick_cube_4cube_1024"   # → /DISK1 (symlink)
PLANNER_PORT=5599
NUM_ENVS=16
EPISODES=1024

ACT_REPO="taehunkim/so101_act_sim_pick_cube_4cube_1024"
SMOLVLA_REPO="taehunkim/so101_smolvla_sim_pick_cube_4cube_1024"
GROOT_REPO="taehunkim/so101_groot_n17_sim_pick_cube_4cube_1024"
ACT_JOB="so101_act_sim_pick_cube_4cube_1024"
SMOLVLA_JOB="so101_smolvla_sim_pick_cube_4cube_1024"
GROOT_JOB="so101_groot_n17_sim_pick_cube_4cube_1024"
GROOT_STEPS=80000
GROOT_SAVE_STEPS=20000   # 24GB/checkpoint → 4 saves (20/40/60/80k)

DC=(docker compose --env-file .env -f docker/docker-compose.yaml run --rm)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
section() { echo; echo "================================================================"; log "$*"; echo "================================================================"; }

# free VRAM(MiB) 조회 (단일 GPU)
free_vram() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

# VRAM 확보 대기 — need MiB 이상 free 될 때까지 (사용자 eval 정지 대기)
wait_for_vram() {
  local need=$1 label=$2 f
  f=$(free_vram)
  if [ "${f:-0}" -ge "$need" ]; then log "[$label] VRAM OK (${f}MiB free ≥ ${need})"; return 0; fi
  log "[$label] VRAM 대기: ${f}MiB free < ${need}MiB 필요 — eval/viewer 정지 시 자동 진행 (60s 간격 폴링)"
  while :; do
    sleep 60
    f=$(free_vram)
    if [ "${f:-0}" -ge "$need" ]; then log "[$label] VRAM 확보 (${f}MiB free) → 진행"; return 0; fi
  done
}

PIPE_START=$(ts)
section "PIPELINE START — 4-cube 1024 → ACT/SmolVLA/GR00T (640k 등량)"
log "GPU free now: $(free_vram)MiB | dataset repo: $DATASET_REPO"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — 데이터 생성 (4-cube 1024 ep, 현재 DR)
# ─────────────────────────────────────────────────────────────────────────────
section "Stage 1 — 데이터 생성 (N=${NUM_ENVS}, ${EPISODES} all-4 ep)"
EXIST_EP=0
if [ -f "$DATASET_DIR/meta/info.json" ]; then
  EXIST_EP=$(grep -o '"total_episodes"[: ]*[0-9]*' "$DATASET_DIR/meta/info.json" | grep -o '[0-9]*' | head -1)
fi
if [ "${EXIST_EP:-0}" -ge "$EPISODES" ]; then
  log "Stage 1 SKIP — 데이터셋 이미 완료 (${EXIST_EP} ep) @ $DATASET_DIR (Isaac teardown 좀비로 재생성 불필요)"
else
GEN_LOG="$LOGDIR/4cube1024_gen.log"
PLANNER_LOG="$LOGDIR/4cube1024_planner.log"

log "cuRobo planner 기동 (port ${PLANNER_PORT})"
nohup env OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
  scripts/planning/curobo_planner_server.py --port "$PLANNER_PORT" \
  > "$PLANNER_LOG" 2>&1 &
PLANNER_PID=$!
log "planner PID=$PLANNER_PID — ZMQ REP bind 대기 (max 300s)"

bound=0
for _ in $(seq 1 100); do
  if ! kill -0 "$PLANNER_PID" 2>/dev/null; then log "FATAL: planner 프로세스 조기 종료 — $PLANNER_LOG 확인"; exit 1; fi
  if grep -q "ZMQ REP bind" "$PLANNER_LOG" 2>/dev/null; then bound=1; break; fi
  sleep 3
done
[ "$bound" = 1 ] || { log "FATAL: planner bind 안 됨 (300s) — $PLANNER_LOG 확인"; kill "$PLANNER_PID" 2>/dev/null; exit 1; }
log "planner ready. 배치 생성 시작 → $GEN_LOG"

OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
  scripts/sim/pick_cube_curobo_batch.py \
  --headless --num_envs "$NUM_ENVS" --chunk 64 --planner_port "$PLANNER_PORT" \
  --active_objects 4 \
  --record_dir "$DATASET_DIR" --record_episodes "$EPISODES" --record_overwrite \
  > "$GEN_LOG" 2>&1
GEN_RC=$?

log "gen 종료 (rc=$GEN_RC). planner(PID=$PLANNER_PID) 정지"
kill "$PLANNER_PID" 2>/dev/null; sleep 3; kill -9 "$PLANNER_PID" 2>/dev/null

if [ "$GEN_RC" -ne 0 ]; then log "FATAL: 데이터 생성 실패 (rc=$GEN_RC) — $GEN_LOG 확인. 중단."; exit 1; fi
if [ ! -f "$DATASET_DIR/meta/info.json" ]; then log "FATAL: $DATASET_DIR/meta/info.json 없음 — 생성 미완. 중단."; exit 1; fi
NEP=$(grep -o '"total_episodes"[: ]*[0-9]*' "$DATASET_DIR/meta/info.json" | grep -o '[0-9]*' | head -1)
log "데이터셋 생성 완료: ${NEP:-?} episodes @ $DATASET_DIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — HF Hub push (데이터셋) — 학습의 linchpin
# ─────────────────────────────────────────────────────────────────────────────
section "Stage 2 — HF push 데이터셋 → $DATASET_REPO"
PUSH_LOG="$LOGDIR/4cube1024_push_dataset.log"
uv run --no-sync python scripts/sim/upload_to_huggingface.py \
  --dataset_dir "$DATASET_DIR" --repo_id "$DATASET_REPO" \
  --commit_message "4-cube 1024-episode cuRobo sim dataset (current DR)" \
  > "$PUSH_LOG" 2>&1
PUSH_RC=$?
if [ "$PUSH_RC" -ne 0 ]; then log "FATAL: 데이터셋 push 실패 (rc=$PUSH_RC) — $PUSH_LOG. 학습 불가, 중단."; exit 1; fi
log "데이터셋 push 완료 (v3.0 태그 포함) → $DATASET_REPO"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — ACT 학습 (batch 32 × 20k = 640k)
# ─────────────────────────────────────────────────────────────────────────────
section "Stage 3 — ACT 학습 (batch 32 × 20k)"
wait_for_vram 16000 "ACT"
ACT_LOG="$LOGDIR/4cube1024_train_act.log"
POLICY_PROFILE=act "${DC[@]}" \
  -e POLICY_PROFILE=act \
  -e HF_DATASET_REPO_ID="$DATASET_REPO" \
  -e BATCH_SIZE=32 -e TRAIN_STEPS=20000 \
  -e JOB_NAME="$ACT_JOB" \
  -e POLICY_REPO_ID="$ACT_REPO" \
  policy-server train --policy.push_to_hub=true \
  > "$ACT_LOG" 2>&1
ACT_RC=$?
log "ACT 학습 종료 (rc=$ACT_RC) — push→$ACT_REPO, 로컬 outputs/train/$ACT_JOB"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — SmolVLA 학습 (batch 32 × 20k = 640k)
# ─────────────────────────────────────────────────────────────────────────────
section "Stage 4 — SmolVLA 학습 (batch 32 × 20k)"
wait_for_vram 22000 "SmolVLA"
SMOLVLA_LOG="$LOGDIR/4cube1024_train_smolvla.log"
POLICY_PROFILE=smolvla "${DC[@]}" \
  -e POLICY_PROFILE=smolvla \
  -e HF_DATASET_REPO_ID="$DATASET_REPO" \
  -e BATCH_SIZE=32 -e TRAIN_STEPS=20000 \
  -e JOB_NAME="$SMOLVLA_JOB" \
  -e POLICY_REPO_ID="$SMOLVLA_REPO" \
  policy-server train --policy.push_to_hub=true \
  > "$SMOLVLA_LOG" 2>&1
SMOLVLA_RC=$?
log "SmolVLA 학습 종료 (rc=$SMOLVLA_RC) — push→$SMOLVLA_REPO, 로컬 outputs/train/$SMOLVLA_JOB"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — GR00T-N1.7 (convert + finetune batch 8 × 80k = 640k)
# ─────────────────────────────────────────────────────────────────────────────
section "Stage 5a — GR00T convert (HF v3 → v2.1 + modality.json)"
GROOT_CONV_LOG="$LOGDIR/4cube1024_groot_convert.log"
POLICY_PROFILE=groot_n17 "${DC[@]}" \
  -e POLICY_PROFILE=groot_n17 \
  -e HF_DATASET_REPO_ID="$DATASET_REPO" \
  gr00t convert \
  > "$GROOT_CONV_LOG" 2>&1
CONV_RC=$?
GROOT_RC=1
if [ "$CONV_RC" -ne 0 ]; then
  log "ERROR: GR00T convert 실패 (rc=$CONV_RC) — $GROOT_CONV_LOG. finetune 건너뜀."
else
  log "GR00T convert 완료. finetune 대기 (VRAM)"
  section "Stage 5b — GR00T finetune (batch 8 × ${GROOT_STEPS})"
  wait_for_vram 42000 "GR00T"
  GROOT_LOG="$LOGDIR/4cube1024_train_groot.log"
  POLICY_PROFILE=groot_n17 "${DC[@]}" \
    -e POLICY_PROFILE=groot_n17 \
    -e HF_DATASET_REPO_ID="$DATASET_REPO" \
    -e BATCH_SIZE=8 -e TRAIN_STEPS="$GROOT_STEPS" -e SAVE_STEPS="$GROOT_SAVE_STEPS" \
    -e JOB_NAME="$GROOT_JOB" \
    gr00t finetune \
    > "$GROOT_LOG" 2>&1
  GROOT_RC=$?
  log "GR00T finetune 종료 (rc=$GROOT_RC) — 로컬 outputs/train/$GROOT_JOB"
  if [ "$GROOT_RC" -eq 0 ]; then
    section "Stage 5c — GR00T 모델 push → $GROOT_REPO"
    GROOT_PUSH_LOG="$LOGDIR/4cube1024_push_groot.log"
    CKPT="outputs/train/$GROOT_JOB/checkpoint-$GROOT_STEPS"
    if [ -d "$CKPT" ]; then
      # host huggingface-cli 는 .env 를 자동으로 안 읽음 → HF_TOKEN 명시 주입(아니면 401)
      HF_TOKEN_VAL=$(grep '^HF_TOKEN=' "$REPO/.env" | cut -d= -f2- | tr -d '"'\'' ')
      HF_TOKEN="$HF_TOKEN_VAL" uv run --no-sync huggingface-cli upload "$GROOT_REPO" "$CKPT" . --repo-type model --token "$HF_TOKEN_VAL" \
        > "$GROOT_PUSH_LOG" 2>&1
      log "GR00T push rc=$? → $GROOT_REPO ($CKPT)"
    else
      log "WARN: $CKPT 없음 — push 생략 (최종 checkpoint 경로 확인)"
    fi
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
section "PIPELINE DONE"
log "시작 $PIPE_START / 종료 $(ts)"
log "데이터셋: $DATASET_REPO ($DATASET_DIR)"
log "ACT     rc=${ACT_RC}      → $ACT_REPO"
log "SmolVLA rc=${SMOLVLA_RC}  → $SMOLVLA_REPO"
log "GR00T   rc=${GROOT_RC}    → $GROOT_REPO"
log "로그: $LOGDIR/4cube1024_*.log"
