#!/usr/bin/env bash
# =============================================================================
# 4-cube 1024 모델 3종 closed-loop sim eval (Isaac bridge + policy-server + vla-ros)
#   각 모델: 서비스 기동 → bridge --eval N --num_cubes 4 → JSON → teardown
#   동일 N·동일 num_cubes = 공정 비교. 직렬(GPU 1장).
#
# 구동:
#   cd /home/konan147/Workspaces/SO101-Sim2Real
#   nohup setsid bash scripts/run_4cube_1024_eval.sh \
#     > outputs/p5_logs/4cube1024_eval.log 2>&1 &
# =============================================================================
set -uo pipefail
REPO=/home/konan147/Workspaces/SO101-Sim2Real
cd "$REPO" || exit 1
LOGDIR="$REPO/outputs/p5_logs"; mkdir -p "$LOGDIR"

N=10
CUBES=4
DC="docker compose --env-file .env -f docker/docker-compose.yaml"
BRIDGE="scripts/sim/run_cube_desk_ros_bridge.sh"

ACT_MODEL=/workspace/outputs/train/so101_act_sim_pick_cube_4cube_1024/checkpoints/last/pretrained_model
SMOLVLA_MODEL=/workspace/outputs/train/so101_smolvla_sim_pick_cube_4cube_1024/checkpoints/last/pretrained_model
GROOT_CKPT=/host/outputs/train/so101_groot_n17_sim_pick_cube_4cube_1024/checkpoint-80000

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
sec(){ echo; echo "============================================================"; log "$*"; echo "============================================================"; }

# 컨테이너 일괄 정리(우리 eval 이름만)
teardown(){
  docker rm -f ps_eval vla_eval pg_eval gr_eval >/dev/null 2>&1 || true
  # Isaac bridge child 좀비 방지 (eval 종료 후에도 남으면)
  pkill -9 -f "run_cube_desk_ros_bridge.py" >/dev/null 2>&1 || true
  sleep 3
}

# 컨테이너 로그에서 패턴 대기(bounded). $1=name $2=pattern $3=timeout_s $4=fallback_sleep
wait_log(){
  local name=$1 pat=$2 to=$3 fb=$4 i=0
  while [ $i -lt $to ]; do
    if docker logs "$name" 2>&1 | grep -qiE "$pat"; then log "  [$name] ready ($pat)"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
      log "  [$name] 컨테이너 조기 종료 — 로그: docker logs $name"; return 1; fi
    sleep 3; i=$((i+3))
  done
  log "  [$name] 패턴 '$pat' 미확인(${to}s) — fallback ${fb}s 후 진행"; sleep "$fb"; return 0
}

# bridge eval 1회. $1=eval_out
run_bridge(){
  local out=$1
  log "  bridge --eval $N --num_cubes $CUBES → $out"
  OMNI_KIT_ACCEPT_EULA=YES "$BRIDGE" \
    --headless --enable_cameras --num_cubes "$CUBES" --seed 0 \
    --eval "$N" --eval_seconds 30 --eval_settle 1.5 --eval_warmup 30 \
    --eval_out "$out" >> "$BRIDGE_LOG" 2>&1
  local rc=$?
  log "  bridge 종료 rc=$rc"
  return $rc
}

result_line(){  # $1=label $2=json
  local j=$2
  if [ -f "$REPO/$j" ]; then
    python3 -c "import json;d=json.load(open('$REPO/$j'));print('  %-9s all-%d %.1f%% · per-cube %.1f%% · ever %.1f%% · avg %.2f/%d (n=%d)'%('$1',d.get('n_active_cubes',$CUBES),100*d.get('all_cubes_success_rate',0),100*d.get('per_cube_placement_rate',0),100*d.get('per_cube_ever_rate',0),d.get('avg_cubes_placed',0),d.get('n_active_cubes',$CUBES),d.get('n_episodes',0)))" 2>/dev/null \
      || echo "  $1: JSON 파싱 실패 ($j)"
  else echo "  $1: 결과 JSON 없음 ($j)"; fi
}

EVAL_START=$(ts)
sec "EVAL START — 3모델 closed-loop (N=$N, ${CUBES}-cube, seed 0)"
teardown

# ─── MODEL 1: ACT ─────────────────────────────────────────────────────────
sec "[1/3] ACT eval"
BRIDGE_LOG="$LOGDIR/4cube1024_eval_act_bridge.log"
POLICY_PROFILE=act_4ceval $DC run -d --name ps_eval \
  policy-server policy-server > "$LOGDIR/4cube1024_eval_act_ps.log" 2>&1
sleep 10
POLICY_PROFILE=act_4ceval $DC run -d --name vla_eval -e POLICY_PROFILE=act_4ceval \
  vla-ros > "$LOGDIR/4cube1024_eval_act_vla.log" 2>&1
wait_log vla_eval "sent instructions|instructions sent|VLA.*up" 120 20
run_bridge outputs/vla_eval_act_4cube_1024.json
teardown

# ─── MODEL 2: SmolVLA ─────────────────────────────────────────────────────
sec "[2/3] SmolVLA eval"
BRIDGE_LOG="$LOGDIR/4cube1024_eval_smolvla_bridge.log"
POLICY_PROFILE=smolvla_4ceval $DC run -d --name ps_eval \
  policy-server policy-server > "$LOGDIR/4cube1024_eval_smolvla_ps.log" 2>&1
sleep 10
POLICY_PROFILE=smolvla_4ceval $DC run -d --name vla_eval -e POLICY_PROFILE=smolvla_4ceval \
  vla-ros > "$LOGDIR/4cube1024_eval_smolvla_vla.log" 2>&1
wait_log vla_eval "sent instructions|instructions sent|VLA.*up" 120 20
run_bridge outputs/vla_eval_smolvla_4cube_1024.json
teardown

# ─── MODEL 3: GR00T-N1.7 ──────────────────────────────────────────────────
sec "[3/3] GR00T-N1.7 eval"
BRIDGE_LOG="$LOGDIR/4cube1024_eval_groot_bridge.log"
POLICY_PROFILE=groot_n17_4ceval $DC run -d --name gr_eval -e GROOT_CHECKPOINT="$GROOT_CKPT" \
  gr00t zmq-server > "$LOGDIR/4cube1024_eval_groot_zmq.log" 2>&1
wait_log gr_eval "ready|listening|5555|Server" 240 30
POLICY_PROFILE=groot_n17_4ceval $DC run -d --name pg_eval \
  policy-server policy-server-groot > "$LOGDIR/4cube1024_eval_groot_bridge_grpc.log" 2>&1
wait_log pg_eval "listening|ready|GrootBridge|8080" 90 15
POLICY_PROFILE=groot_n17_4ceval $DC run -d --name vla_eval -e POLICY_PROFILE=groot_n17_4ceval \
  vla-ros > "$LOGDIR/4cube1024_eval_groot_vla.log" 2>&1
wait_log vla_eval "sent instructions|instructions sent|VLA.*up" 120 20
run_bridge outputs/vla_eval_groot_n17_4cube_1024.json
teardown

# ─── 요약 ─────────────────────────────────────────────────────────────────
sec "EVAL DONE — 시작 $EVAL_START / 종료 $(ts)"
log "결과 (N=$N, ${CUBES}-cube, seed 0):"
result_line "ACT"     outputs/vla_eval_act_4cube_1024.json
result_line "SmolVLA" outputs/vla_eval_smolvla_4cube_1024.json
result_line "GR00T"   outputs/vla_eval_groot_n17_4cube_1024.json
log "JSON: outputs/vla_eval_*_4cube_1024.json | bridge 로그: $LOGDIR/4cube1024_eval_*_bridge.log"
