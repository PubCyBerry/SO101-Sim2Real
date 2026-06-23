#!/usr/bin/env bash
# 신규 nearest-order 256ep adaptation 모델의 단일 closed-loop 평가.
# 사용: bash scripts/run_nearest_256_eval.sh <smolvla|groot> [actions_per_chunk] [threshold] [seed] [episodes] [seconds] [command_slew] [label]
set -uo pipefail

REPO=/home/konan147/Workspaces/SO101-Sim2Real
cd "$REPO" || exit 1

MODEL="${1:-smolvla}"
APC="${2:-16}"
THRESHOLD="${3:-0.5}"
SEED="${4:-0}"
N="${5:-10}"
EVAL_SECONDS="${6:-30}"
COMMAND_SLEW="${7:-false}"
RUN_LABEL="${8:-}"
case "$COMMAND_SLEW" in
  true|false) ;;
  *) echo "command_slew는 true 또는 false여야 함: $COMMAND_SLEW" >&2; exit 2 ;;
esac
CUBES=4
PROFILE=
case "$MODEL" in
  smolvla) PROFILE=smolvla_nearest256 ;;
  groot) PROFILE=groot_n17_nearest256 ;;
  *) echo "model은 smolvla 또는 groot여야 함: $MODEL" >&2; exit 2 ;;
esac

TAG="${MODEL}_apc${APC}_thr${THRESHOLD}_seed${SEED}_n${N}_s${EVAL_SECONDS}"
if [[ "$COMMAND_SLEW" == "true" ]]; then
  TAG="${TAG}_slew"
fi
if [[ -n "$RUN_LABEL" ]]; then
  TAG="${TAG}_${RUN_LABEL}"
fi
TAG="${TAG//./p}"
LOGDIR="$REPO/outputs/vla_eval_nearest256"
RESETDIR="$REPO/logs/vla_eval_nearest256"
mkdir -p "$LOGDIR" "$RESETDIR"
OUT="$LOGDIR/${TAG}.json"
BRIDGE_LOG="$LOGDIR/${TAG}_bridge.log"
RESET_HOST="$RESETDIR/${TAG}_reset.token"
RESET_CONTAINER="/workspace/logs/vla_eval_nearest256/${TAG}_reset.token"
TRAJ_HOST="$RESETDIR/${TAG}_traj.jsonl"
TRAJ_CONTAINER="/workspace/logs/vla_eval_nearest256/${TAG}_traj.jsonl"
DC=(docker compose --env-file .env -f docker/docker-compose.yaml)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

cleanup() {
  if [[ "${1:-}" == "save" ]]; then
    for name in nearest_ps nearest_vla nearest_pg nearest_gr; do
      if docker inspect "$name" >/dev/null 2>&1; then
        docker logs "$name" > "$LOGDIR/${TAG}_${name}.runtime.log" 2>&1 || true
      fi
    done
  fi
  docker rm -f nearest_ps nearest_vla nearest_pg nearest_gr >/dev/null 2>&1 || true
}
trap 'cleanup save' EXIT
cleanup

wait_log() {
  local name=$1 pattern=$2 timeout_s=$3 elapsed=0
  while (( elapsed < timeout_s )); do
    if docker logs "$name" 2>&1 | grep -qiE "$pattern"; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      log "$name 조기 종료"
      docker logs "$name" 2>&1 | tail -n 80
      return 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done
  log "$name ready 로그 timeout(${timeout_s}s)"
  return 1
}

log "eval start model=$MODEL apc=$APC threshold=$THRESHOLD seed=$SEED n=$N seconds=$EVAL_SECONDS command_slew=$COMMAND_SLEW label=${RUN_LABEL:-none}"
if [[ "$MODEL" == "groot" ]]; then
  POLICY_PROFILE="$PROFILE" "${DC[@]}" run -d --name nearest_gr \
    gr00t zmq-server > "$LOGDIR/${TAG}_groot.log" 2>&1
  wait_log nearest_gr "ready|listening|5555|Server" 300 || exit 1
  POLICY_PROFILE="$PROFILE" "${DC[@]}" run -d --name nearest_pg \
    policy-server policy-server-groot > "$LOGDIR/${TAG}_policy.log" 2>&1
  wait_log nearest_pg "listening|ready|GrootBridge|8080" 120 || exit 1
else
  POLICY_PROFILE="$PROFILE" "${DC[@]}" run -d --name nearest_ps \
    policy-server policy-server > "$LOGDIR/${TAG}_policy.log" 2>&1
  sleep 8
fi

rm -f "$TRAJ_HOST"
POLICY_PROFILE="$PROFILE" "${DC[@]}" run -d --name nearest_vla \
  -e POLICY_PROFILE="$PROFILE" \
  -e VLA_TRAJ_LOG="$TRAJ_CONTAINER" \
  vla-ros \
  ros2 run so101_vla_policy vla_policy_node \
    --ros-args \
    --params-file /workspace/ros2_ws/src/so101_vla_policy/config/vla_policy.yaml \
    -p "actions_per_chunk:=${APC}" \
    -p "chunk_size_threshold:=${THRESHOLD}" \
    -p "vla_reset_file:=${RESET_CONTAINER}" \
    -p "command_slew_limit:=${COMMAND_SLEW}" \
  > "$LOGDIR/${TAG}_vla.log" 2>&1
wait_log nearest_vla "sent instructions|instructions sent|VLA node up" 180 || exit 1

rm -f "$OUT" "$RESET_HOST"
set +e
OMNI_KIT_ACCEPT_EULA=YES timeout --kill-after=20s 900s \
  scripts/sim/run_cube_desk_ros_bridge.sh \
    --headless --enable_cameras --num_cubes "$CUBES" --seed "$SEED" \
    --vla_action_parity \
    --vla_reset_file "$RESET_HOST" \
    --eval "$N" --eval_seconds "$EVAL_SECONDS" --eval_settle 1.5 --eval_warmup 30 \
    --eval_out "$OUT" > "$BRIDGE_LOG" 2>&1
BRIDGE_RC=$?
set -e

if [[ ! -f "$OUT" ]]; then
  log "평가 JSON 없음(rc=$BRIDGE_RC): $OUT"
  tail -n 100 "$BRIDGE_LOG"
  exit 1
fi

TMP_JSON="${OUT}.tmp"
jq \
  --arg model "$MODEL" \
  --arg profile "$PROFILE" \
  --arg run_label "$RUN_LABEL" \
  --argjson actions_per_chunk "$APC" \
  --argjson chunk_size_threshold "$THRESHOLD" \
  --argjson seed "$SEED" \
  --argjson command_slew_limit "$COMMAND_SLEW" \
  --arg trajectory_log "$TRAJ_HOST" \
  '.model = $model
   | .profile = $profile
   | .run_label = $run_label
   | .actions_per_chunk = $actions_per_chunk
   | .chunk_size_threshold = $chunk_size_threshold
   | .seed = $seed
   | .command_slew_limit = $command_slew_limit
   | .trajectory_log = $trajectory_log' \
  "$OUT" > "$TMP_JSON"
mv "$TMP_JSON" "$OUT"

jq '{
  model,
  actions_per_chunk,
  chunk_size_threshold,
  seed,
  n_episodes,
  all_cubes_success_rate,
  per_cube_placement_rate,
  per_cube_ever_rate,
  avg_cubes_placed
}' "$OUT"
log "eval done rc=$BRIDGE_RC → $OUT"
