#!/usr/bin/env bash
# grasp_v14 LSTM+PPO 학습 자동 점검 (cron, 세션 독립).
# 30분 주기로 최신 체크포인트에 monitor_eval 을 돌려 scratch/full/pre 단계별
# 성공률을 집계하고, 16-env 그리드 추론 비디오를 녹화·보존한다.
#
# 등록 예 (crontab -e): */30 * * * * /…/cron_monitor_v14.sh >> /…/logs/cron_monitor_v14.cron.log 2>&1
#
# - flock 으로 중복 실행 방지(점검이 30분 넘게 걸려도 안전).
# - 마지막 점검한 체크포인트를 기억해, 새 ckpt 가 없으면 GPU 낭비 없이 skip.
# - 결과 JSON 은 monitor_history/history.jsonl 에 한 줄씩 append.
# - 비디오는 timestamp+ckpt 이름으로 monitor_history/ 에 복사(덮어쓰기 방지).
set -euo pipefail

ROOT=/home/konan147/Workspaces/SO101-Sim2Real
WORKTREE="$ROOT/.claude/worktrees/lstm-ppo-pickcube"
PY="$ROOT/.venv/bin/python"
RUN_GLOB="$ROOT/outputs/rl/rsl_rl/lstm_ppo_pickcube"  # main 레포 outputs 절대경로(worktree 독립)
RUN_NAME="lstm256_stage1_grasp_v14"                    # v12 run 고정
LOCK=/tmp/cron_monitor_v14.lock

# ── 중복 실행 방지 ──────────────────────────────────────────────
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] 이전 점검이 아직 실행 중 — skip"
  exit 0
fi

cd "$WORKTREE"

# ── v12 run 디렉 탐색 (없으면 최신 run) ────────────────────────
RUN=$(ls -td "$RUN_GLOB"/*"$RUN_NAME"* 2>/dev/null | head -1 || true)
if [[ -z "${RUN:-}" || ! -d "$RUN" ]]; then
  RUN=$(ls -td "$RUN_GLOB"/* 2>/dev/null | head -1 || true)
fi
if [[ -z "${RUN:-}" || ! -d "$RUN" ]]; then
  echo "[$(date -Is)] run 디렉 없음 — skip"
  exit 0
fi

CKPT=$(ls -t "$RUN"/model_*.pt 2>/dev/null | head -1 || true)
if [[ -z "${CKPT:-}" ]]; then
  echo "[$(date -Is)] 체크포인트 없음 — skip"
  exit 0
fi

HIST_DIR="$RUN/monitor_history"
mkdir -p "$HIST_DIR"
LAST_FILE="$HIST_DIR/.last_ckpt"
LAST=$(cat "$LAST_FILE" 2>/dev/null || echo "")

if [[ "$CKPT" == "$LAST" ]]; then
  echo "[$(date -Is)] 새 체크포인트 없음 ($(basename "$CKPT")) — skip"
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
MODEL_TAG=$(basename "$CKPT" .pt)   # model_<N>
EVAL_LOG="$HIST_DIR/eval_${TS}_${MODEL_TAG}.log"

echo "[$(date -Is)] 점검 시작: $MODEL_TAG (run=$(basename "$RUN"))"

# ── monitor_eval 실행 (비디오 녹화 포함) ───────────────────────
set +e
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$WORKTREE/src" "$PY" \
  scripts/reinforcement_learning/monitor_eval.py --recurrent --rnn_hidden_dim 256 \
  --rnn_num_layers 1 --obs_normalization --checkpoint "$CKPT" \
  --num_envs 16 --num_episodes 48 --max_steps 5000 --active_objects 1 \
  --bootstrap_prob 0.6 --pregrasp_frac 0.5 --video --video_length 450 --device cuda:0 \
  > "$EVAL_LOG" 2>&1
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  echo "[$(date -Is)] monitor_eval 실패 (rc=$RC) — 로그: $EVAL_LOG"
  exit 0   # cron 은 계속 살려둔다(다음 주기 재시도)
fi

# ── 결과 JSON 추출 → history.jsonl append ──────────────────────
RESULT=$(grep -oE '\{"status": "ok".*\}' "$EVAL_LOG" | tail -1 || true)
if [[ -n "$RESULT" ]]; then
  echo "{\"ts\":\"$(date -Is)\",\"ckpt\":\"$MODEL_TAG\",\"result\":$RESULT}" \
    >> "$HIST_DIR/history.jsonl"
  echo "[$(date -Is)] 결과 기록: $HIST_DIR/history.jsonl"
  echo "  $RESULT"
else
  echo "[$(date -Is)] 결과 JSON 파싱 실패 — 로그: $EVAL_LOG"
fi

# ── 비디오 보존(덮어쓰기 방지) ─────────────────────────────────
VID="$RUN/videos/monitor/rl-video-step-0.mp4"
if [[ -f "$VID" ]]; then
  cp "$VID" "$HIST_DIR/video_${TS}_${MODEL_TAG}.mp4"
  echo "[$(date -Is)] 비디오 보존: $HIST_DIR/video_${TS}_${MODEL_TAG}.mp4"
fi

echo "$CKPT" > "$LAST_FILE"
echo "[$(date -Is)] 점검 완료: $MODEL_TAG"
