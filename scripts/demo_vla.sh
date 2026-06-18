#!/usr/bin/env bash
# =============================================================================
# VLA closed-loop 데모 런처 — Isaac Sim bridge + vla-ros + policy-server 를 띄워
#   ACT / SmolVLA / GR00T-N1.7 추론을 라이브로 관전(livestream/GUI).  eval 아님(연속 1씬).
#
# 사용:
#   scripts/demo_vla.sh start <act|smolvla|groot> [옵션]
#   scripts/demo_vla.sh stop
#   scripts/demo_vla.sh status
#
# start 옵션:
#   --ckpt PATH   모델 경로 override (기본 = 4cube_1024 로컬 체크포인트)
#                 act/smolvla = 컨테이너경로 /workspace/... 또는 HF repo
#                 groot       = 컨테이너경로 /host/outputs/.../checkpoint-XXXXX
#   --cubes N     큐브 수 1~4 (기본 4)
#   --ip ADDR     원격 WebRTC 관전 IP (tailscale/LAN). 주면 livestream mode 2 + PUBLIC_IP.
#                 안 주면 mode 1(로컬 LAN IP 광고).
#   --gui         로컬 디스플레이 GUI (DISPLAY 필요, livestream 대신).
#   --headless    화면 없이 실행(로그만, 관전 X).
#
# 예:
#   scripts/demo_vla.sh start groot --ip 10.10.16.147
#   scripts/demo_vla.sh start smolvla --cubes 1
#   scripts/demo_vla.sh stop
# =============================================================================
set -uo pipefail
REPO=/home/konan147/Workspaces/SO101-Sim2Real
cd "$REPO" || exit 1
LOGDIR="$REPO/outputs/p5_logs"; mkdir -p "$LOGDIR"
DC="docker compose --env-file .env -f docker/docker-compose.yaml"
BRIDGE="scripts/sim/run_cube_desk_ros_bridge.sh"
PIDFILE="$LOGDIR/demo_vla.pid"
BRIDGE_LOG="$LOGDIR/demo_vla_bridge.log"
PROF_FILES=(env/act_demo.env env/smolvla_demo.env env/groot_n17_demo.env)
NAMES=(vla_demo_ps vla_demo_node vla_demo_gr vla_demo_pg)

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }

# 기본 4cube_1024 모델 경로 (컨테이너 기준)
ACT_DEF=/workspace/outputs/train/so101_act_sim_pick_cube_4cube_1024/checkpoints/last/pretrained_model
SMOL_DEF=/workspace/outputs/train/so101_smolvla_sim_pick_cube_4cube_1024/checkpoints/last/pretrained_model
GROOT_DEF=/host/outputs/train/so101_groot_n17_sim_pick_cube_4cube_1024/checkpoint-80000

stop_all(){
  log "데모 정지 — 컨테이너·bridge 정리"
  if [ -f "$PIDFILE" ]; then
    local p; p=$(cat "$PIDFILE" 2>/dev/null)
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
    rm -f "$PIDFILE"
  fi
  pkill -9 -f "run_cube_desk_ros_bridge.py" 2>/dev/null || true
  docker rm -f "${NAMES[@]}" >/dev/null 2>&1 || true
  rm -f "${PROF_FILES[@]}" 2>/dev/null || true
  sleep 2
  log "정지 완료."
}

status(){
  echo "=== 데모 컨테이너 ==="; docker ps --format '{{.Names}}\t{{.Status}}' | grep -E "vla_demo_" || echo "  (없음)"
  echo "=== bridge ==="; if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "  RUNNING pid=$(cat "$PIDFILE") · log: $BRIDGE_LOG"; else echo "  (정지)"; fi
  echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null
}

# 컨테이너 로그 패턴 대기. $1=name $2=pattern $3=timeout_s
wait_log(){
  local name=$1 pat=$2 to=$3 i=0
  while [ $i -lt "$to" ]; do
    docker logs "$name" 2>&1 | grep -qiE "$pat" && { log "  [$name] ready"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^${name}$" || { log "  [$name] 조기 종료 → docker logs $name"; return 1; }
    sleep 3; i=$((i+3))
  done
  log "  [$name] '$pat' 미확인(${to}s) — 계속 진행(bridge warmup 이 버팀)"; return 0
}

# 임시 데모 프로필 생성: base 복사 + 모델 경로만 교체 (vla node _load_env override 회피)
make_profile(){  # $1=base(act|smolvla|groot_n17) $2=demo_name $3=model_path
  cp "env/$1.env" "env/$2.env"
  if [ "$1" = groot_n17 ]; then
    sed -i "s#^GROOT_CHECKPOINT=.*#GROOT_CHECKPOINT=$3#" "env/$2.env"
  else
    sed -i "s#^POLICY_REPO_ID=.*#POLICY_REPO_ID=$3#" "env/$2.env"
  fi
}

start(){
  local model=$1; shift
  local ckpt="" cubes=4 ip="" disp="stream"
  while [ $# -gt 0 ]; do case "$1" in
    --ckpt) ckpt=$2; shift 2;;
    --cubes) cubes=$2; shift 2;;
    --ip) ip=$2; shift 2;;
    --gui) disp="gui"; shift;;
    --headless) disp="headless"; shift;;
    *) log "알 수 없는 옵션: $1"; exit 1;;
  esac; done

  stop_all   # 멱등: 기존 데모 정리 후 시작
  log "데모 시작 — model=$model cubes=$cubes display=$disp"

  # ── 디스플레이 인자 ──
  local disp_args=()
  case "$disp" in
    stream) if [ -n "$ip" ]; then export LIVESTREAM=1 PUBLIC_IP="$ip"; disp_args=(--livestream 2); log "  WebRTC remote: $ip:49100 (mode2)"; \
            else disp_args=(--livestream 1); log "  WebRTC LAN: <server-ip>:49100 (mode1)"; fi;;
    gui)    disp_args=(); log "  로컬 GUI (DISPLAY=${DISPLAY:-unset})";;
    headless) disp_args=(--headless); log "  headless (관전 X, 로그만)";;
  esac

  # ── 모델별 서비스 기동 ──
  case "$model" in
    act|smolvla)
      local base=$model demo="${model}_demo" def
      [ "$model" = act ] && def="$ACT_DEF" || def="$SMOL_DEF"
      make_profile "$base" "$demo" "${ckpt:-$def}"
      log "  policy-server 기동"
      POLICY_PROFILE=$demo $DC run -d --name vla_demo_ps policy-server policy-server \
        > "$LOGDIR/demo_vla_ps.log" 2>&1
      sleep 8
      log "  vla-ros 기동 (model=${ckpt:-$def})"
      POLICY_PROFILE=$demo $DC run -d --name vla_demo_node -e POLICY_PROFILE=$demo vla-ros \
        > "$LOGDIR/demo_vla_node.log" 2>&1
      wait_log vla_demo_node "sent instructions" 150
      ;;
    groot|groot_n17)
      local demo="groot_n17_demo"
      make_profile groot_n17 "$demo" "${ckpt:-$GROOT_DEF}"
      log "  gr00t zmq-server 기동 (3B 로드 ~30-60s, ckpt=${ckpt:-$GROOT_DEF})"
      POLICY_PROFILE=$demo $DC run -d --name vla_demo_gr -e GROOT_CHECKPOINT="${ckpt:-$GROOT_DEF}" \
        gr00t zmq-server > "$LOGDIR/demo_vla_gr.log" 2>&1
      wait_log vla_demo_gr "ready|listening|5555|Server|dit\.py" 240
      log "  policy-server-groot bridge 기동"
      POLICY_PROFILE=$demo $DC run -d --name vla_demo_pg policy-server policy-server-groot \
        > "$LOGDIR/demo_vla_pg.log" 2>&1
      wait_log vla_demo_pg "listening|ready|GrootBridge|8080" 90
      log "  vla-ros 기동"
      POLICY_PROFILE=$demo $DC run -d --name vla_demo_node -e POLICY_PROFILE=$demo vla-ros \
        > "$LOGDIR/demo_vla_node.log" 2>&1
      wait_log vla_demo_node "sent instructions" 150
      ;;
    *) log "model 은 act|smolvla|groot 중 하나"; exit 1;;
  esac

  # ── Isaac bridge (연속 추론, eval 아님) — detached ──
  log "  Isaac bridge 기동 → $BRIDGE_LOG"
  nohup setsid env OMNI_KIT_ACCEPT_EULA=YES "$BRIDGE" \
    --num_cubes "$cubes" --seed 0 "${disp_args[@]}" \
    > "$BRIDGE_LOG" 2>&1 &
  echo $! > "$PIDFILE"
  log "데모 가동. bridge pid=$(cat "$PIDFILE")"
  echo
  echo "  관전:   tail -f $BRIDGE_LOG"
  [ "$disp" = stream ] && echo "  WebRTC: Omniverse Streaming Client → ${ip:-<server-LAN-ip>}:49100"
  echo "  정지:   scripts/demo_vla.sh stop"
}

# ── dispatch ──
case "${1:-}" in
  start)  shift; [ $# -ge 1 ] || { echo "사용: demo_vla.sh start <act|smolvla|groot> [옵션]"; exit 1; }; start "$@";;
  stop)   stop_all;;
  status) status;;
  *) echo "사용: scripts/demo_vla.sh {start <act|smolvla|groot> [--ckpt P] [--cubes N] [--ip A] [--gui|--headless] | stop | status}";;
esac
