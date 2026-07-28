#!/usr/bin/env bash
# =============================================================================
# VLA closed-loop 런처 — Isaac Sim bridge + vla-ros + policy-server 를 띄워
#   ACT / SmolVLA / GR00T-N1.7 추론을 라이브로 관전하거나 정량 eval한다.
#
# 모델·추론 파라미터는 Docker compose 와 동일하게 `.env` + `env/<POLICY_PROFILE>.env`
# 에서 읽는다. 더 이상 스크립트 안에 모델 경로를 하드코딩하지 않는다.
#   - 활성 프로필 = `.env` 의 POLICY_PROFILE (positional 인자로 override 가능)
#   - 모델 타입   = 프로필의 POLICY_TYPE (act|smolvla|groot) → 서비스 라우팅
#   - 모델 경로   = 프로필의 POLICY_REPO_ID (학습 산출 lerobot 체크포인트)
#
# 사용:
#   scripts/inference/demo_vla.sh start [profile] [옵션]
#   scripts/inference/demo_vla.sh stop
#   scripts/inference/demo_vla.sh status
#
# start 인자:
#   profile       env/<profile>.env 이름 (예: smolvla, groot_n17, act).
#                 생략 시 `.env` 의 POLICY_PROFILE 사용. `groot` 는 `groot_n17` 별칭.
#
# start 옵션:
#   --ckpt PATH   모델 경로 override (기본 = 프로필의 POLICY_REPO_ID)
#                 컨테이너경로 /workspace/outputs/.../pretrained_model 또는 HF repo
#   --cubes N     큐브 수 1~4 (기본 1)
#   --task ID     bridge Gym config. eval 기본은 PickCube-Eval-v0, 연속 기본은 PickCube-v0.
#   --ip ADDR     원격 WebRTC 관전 IP (tailscale/LAN). 주면 livestream mode 2 + PUBLIC_IP.
#                 안 주면 mode 1(로컬 LAN IP 광고).
#   --gui         로컬 디스플레이 GUI (DISPLAY 필요, livestream 대신).
#   --headless    화면 없이 실행(로그만, 관전 X).
#   --eval N      N 에피소드 정량 eval 후 bridge/컨테이너를 종료한다.
#   --eval-out P  eval JSON 경로(기본 outputs/vla_eval_<profile>.json).
#
# eval 거동 정합 옵션 (run_nearest_256_eval.sh 와 같은 추론·물리로 맞춤):
#   --apc N       actions_per_chunk override (기본 = 프로필 값). eval best=32.
#   --thr T       chunk_size_threshold override (기본 = 프로필 값/0.5). eval best=0.25.
#   --seed S      bridge DR seed (기본 0). eval 은 40 사용.
#   --no-parity   --vla_action_parity 끔 (기본 켜짐 — VLA recorder actuator 상한 10 정합).
#   --slew        node command target slew 적용 (기본 ON). raw 모델 출력 점프(최대 116 rad/s teleport)를
#                 arm 5.0 / gripper 2.5 rad/s 로 제한 → actuator cap 포화 폭주 방지. 학습데이터는 ≤2.5 rad/s.
#   --no-slew     slew 끔 (옛 기본 동작; raw 모델 target 직접 publish — 팔이 actuator 상한속도로 휙휙).
#   --arm-vel V   slew arm 상한 rad/s (기본 2.3 = 데이터생성 pick_cube_curobo_batch --max_cmd_vel 정합).
#                 학습데이터 arm 속도가 p99≈2.3·≤2.5 라 같은 값으로 맞춤. gripper 상한은 node 기본 2.5(=데이터).
#   `--eval`이 없는 연속 데모는 success 판정·종료가 없어 eval JSON이 나오지 않는다.
#
# 예:
#   scripts/inference/demo_vla.sh start                    # .env POLICY_PROFILE 그대로
#   scripts/inference/demo_vla.sh start smolvla --ip 10.10.16.147
#   scripts/inference/demo_vla.sh start smolvla --apc 32 --thr 0.25 --seed 40 --ip 10.10.16.147
#   scripts/inference/demo_vla.sh start groot --ip 10.10.16.147
#   scripts/inference/demo_vla.sh start act --eval 1 --headless
#   scripts/inference/demo_vla.sh start smolvla --cubes 1
#   scripts/inference/demo_vla.sh stop
# =============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$REPO" || exit 1
LOGDIR="$REPO/outputs/p5_logs"; mkdir -p "$LOGDIR"
DC="docker compose --env-file .env -f docker/docker-compose.yaml"
BRIDGE="scripts/inference/run_cube_desk_ros_bridge.sh"
PIDFILE="$LOGDIR/demo_vla.pid"
BRIDGE_LOG="$LOGDIR/demo_vla_bridge.log"
# bridge↔node 공유 reset token (수동 R/N reset 시 stale queue 초기화 — eval 정합). 같은 물리 파일.
RESET_HOST="$REPO/logs/demo_vla_reset.token"
RESET_CONTAINER="/workspace/logs/demo_vla_reset.token"
VLA_PARAMS=/workspace/ros2_ws/src/so101_vla_policy/config/vla_policy.yaml
# --ckpt override 시에만 쓰는 임시 프로필 (cleanup 대상). 평소엔 실 프로필을 그대로 쓴다.
OVERRIDE_PROFILE=demo_override
# 정리 대상 임시 프로필 파일 (옛 *_demo.env 도 호환 정리)
PROF_FILES=(env/"$OVERRIDE_PROFILE".env env/act_demo.env env/smolvla_demo.env env/groot_n17_demo.env)
NAMES=(vla_demo_ps vla_demo_node)

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }

# `.env` / 프로필 파일에서 KEY 값 읽기 (마지막 정의·`KEY=` prefix 제거)
env_get(){  grep -E "^$1=" .env 2>/dev/null         | tail -1 | cut -d= -f2-; }
prof_get(){ grep -E "^$2=" "env/$1.env" 2>/dev/null | tail -1 | cut -d= -f2-; }

# 활성 프로필 결정: 인자 없으면 .env POLICY_PROFILE, `groot` 는 groot_n17 별칭
resolve_profile(){  # $1=arg(빈문자 가능) -> stdout=profile name
  local a="$1"
  [ -z "$a" ] && a=$(env_get POLICY_PROFILE)
  [ -z "$a" ] && a=groot_n17   # compose 기본값과 정합
  [ "$a" = groot ] && a=groot_n17
  echo "$a"
}

stop_all(){
  log "데모 정지 — 컨테이너·bridge 정리"
  if [ -f "$PIDFILE" ]; then
    local p; p=$(cat "$PIDFILE" 2>/dev/null)
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
    rm -f "$PIDFILE"
  fi
  pkill -9 -f "run_cube_desk_ros_bridge.py" 2>/dev/null || true
  docker rm -f "${NAMES[@]}" >/dev/null 2>&1 || true
  rm -f "${PROF_FILES[@]}" "$RESET_HOST" 2>/dev/null || true
  sleep 2
  log "정지 완료."
}

# vla-ros 기동 — eval(run_nearest_256_eval.sh) 과 동일하게 params-file + ros param 으로
# APC/thr/reset/slew 를 명시 주입한다(연속 데모도 같은 추론 설정). 전역 active/g_apc/g_thr/g_slew 사용.
run_vla_node(){
  POLICY_PROFILE=$active $DC run -d --name vla_demo_node \
    -e POLICY_PROFILE=$active -e VLA_TRAJ_LOG="${VLA_TRAJ_LOG:-}" \
    -e VLA_EEF_METRICS_LOG="${VLA_EEF_METRICS_LOG:-/workspace/logs/eef_sim_rollout.jsonl}" \
    vla-ros \
    ros2 run so101_vla_policy vla_policy_node --ros-args \
      --params-file "$VLA_PARAMS" \
      -p "actions_per_chunk:=$g_apc" \
      -p "chunk_size_threshold:=$g_thr" \
      -p "vla_reset_file:=$RESET_CONTAINER" \
      -p "command_slew_limit:=$g_slew" \
      -p "arm_target_max_velocity:=$g_armvel" \
    > "$LOGDIR/demo_vla_node.log" 2>&1
  wait_log vla_demo_node "sent instructions|instructions sent|VLA node up" 150
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

# 임시 override 프로필 생성: 활성 프로필 복사 + 모델 경로만 교체 (vla node _load_env override 회피)
make_profile(){  # $1=src_profile $2=dst_name $3=model_path $4=ptype(unused)
  cp "env/$1.env" "env/$2.env"
  sed -i "s#^POLICY_REPO_ID=.*#POLICY_REPO_ID=$3#" "env/$2.env"
}

start(){
  # 첫 인자가 옵션(--)이 아니면 프로필 이름으로 소비
  local prof_arg=""
  if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then prof_arg=$1; shift; fi
  local ckpt="" cubes=1 ip="" disp="stream" seed=0 parity=1 slew=true task=""
  local apc_ov="" thr_ov="" armvel=2.3 eval_count=0 eval_out=""
  while [ $# -gt 0 ]; do case "$1" in
    --ckpt) ckpt=$2; shift 2;;
    --cubes) cubes=$2; shift 2;;
    --task) task=$2; shift 2;;
    --ip) ip=$2; shift 2;;
    --gui) disp="gui"; shift;;
    --headless) disp="headless"; shift;;
    --eval) eval_count=$2; shift 2;;
    --eval-out) eval_out=$2; shift 2;;
    --apc) apc_ov=$2; shift 2;;
    --thr) thr_ov=$2; shift 2;;
    --seed) seed=$2; shift 2;;
    --no-parity) parity=0; shift;;
    --slew) slew=true; shift;;
    --no-slew) slew=false; shift;;
    --arm-vel) armvel=$2; shift 2;;
    *) log "알 수 없는 옵션: $1"; exit 1;;
  esac; done

  # ── 프로필 해석 + 검증 ──
  local profile; profile=$(resolve_profile "$prof_arg")
  if [ ! -f "env/$profile.env" ]; then
    log "프로필 없음: env/$profile.env  (사용 가능: $(ls env/*.env | sed 's#env/##;s#\.env##' | tr '\n' ' '))"
    exit 1
  fi
  local ptype; ptype=$(prof_get "$profile" POLICY_TYPE)
  [ -z "$ptype" ] && { log "env/$profile.env 에 POLICY_TYPE 없음"; exit 1; }
  [[ "$eval_count" =~ ^[0-9]+$ ]] || { log "--eval은 0 이상의 정수여야 함: $eval_count"; exit 1; }
  [ -z "$eval_out" ] && eval_out="outputs/vla_eval_${profile}.json"
  if [ -z "$task" ]; then
    if [ "$eval_count" -gt 0 ]; then
      task="SimToReal-SO101-PickCube-Eval-v0"
    else
      task="SimToReal-SO101-PickCube-v0"
    fi
  fi

  stop_all   # 멱등: 기존 데모 정리 후 시작
  log "데모 시작 — profile=$profile type=$ptype cubes=$cubes display=$disp"

  # ── 디스플레이 인자 ──
  local disp_args=()
  case "$disp" in
    stream) if [ -n "$ip" ]; then export LIVESTREAM=1 PUBLIC_IP="$ip"; disp_args=(--livestream 2); log "  WebRTC remote: $ip:49100 (mode2)"; \
            else disp_args=(--livestream 1); log "  WebRTC LAN: <server-ip>:49100 (mode1)"; fi;;
    gui)    disp_args=(); log "  로컬 GUI (DISPLAY=${DISPLAY:-unset})";;
    headless) disp_args=(--headless); log "  headless (관전 X, 로그만)";;
  esac

  # ── 활성 프로필 결정: --ckpt override 시 임시 복사, 아니면 실 프로필 직접 사용 ──
  local active=$profile
  local model_path
  model_path=$(prof_get "$profile" POLICY_REPO_ID)
  if [ -n "$ckpt" ]; then
    active=$OVERRIDE_PROFILE
    make_profile "$profile" "$active" "$ckpt" "$ptype"
    model_path=$ckpt
    log "  --ckpt override → 임시 프로필 env/$active.env"
  fi
  # APC/thr: --apc/--thr override 우선, 없으면 프로필값(env/<profile>.env), 그다음 .env,
  #          마지막에 노드 기본(8/0.5)과 정합. compose 와 동일하게 .env 도 조회한다.
  local g_apc g_thr g_slew=$slew g_armvel=$armvel
  g_apc=${apc_ov:-$(prof_get "$active" ACTIONS_PER_CHUNK)}; [ -z "$g_apc" ] && g_apc=$(env_get ACTIONS_PER_CHUNK); [ -z "$g_apc" ] && g_apc=8
  g_thr=${thr_ov:-$(prof_get "$active" CHUNK_SIZE_THRESHOLD)}; [ -z "$g_thr" ] && g_thr=$(env_get CHUNK_SIZE_THRESHOLD); [ -z "$g_thr" ] && g_thr=0.5
  log "  모델=$model_path  APC=$g_apc  thr=$g_thr  parity=$parity  slew=$slew(arm≤${g_armvel})  seed=$seed"

  # ── 모델 타입별 서비스 기동 ──
  case "$ptype" in
    act|smolvla|groot)
      log "  policy-server 기동 (groot=N1.7 3B 로드는 첫 instruction 시점)"
      POLICY_PROFILE=$active $DC run -d --name vla_demo_ps policy-server policy-server \
        > "$LOGDIR/demo_vla_ps.log" 2>&1
      sleep 8
      log "  vla-ros 기동 (APC=$g_apc thr=$g_thr slew=$g_slew)"
      run_vla_node
      ;;
    *) log "POLICY_TYPE 은 act|smolvla|groot 중 하나여야 함: $ptype"; exit 1;;
  esac

  # ── Isaac bridge (연속 추론, eval 아님) — detached ──
  # eval 정합: --vla_action_parity(actuator max vel 10), --seed, --vla_reset_file(공유 reset token)
  local bridge_extra=(--task "$task" --seed "$seed" --vla_reset_file "$RESET_HOST")
  [ "$parity" = 1 ] && bridge_extra+=(--vla_action_parity)
  if [ "$eval_count" -gt 0 ]; then
    bridge_extra+=(--eval "$eval_count" --eval_out "$eval_out" --eval_model "$model_path")
  fi
  log "  Isaac bridge 기동 → $BRIDGE_LOG"
  nohup setsid env OMNI_KIT_ACCEPT_EULA=YES "$BRIDGE" \
    --num_cubes "$cubes" "${bridge_extra[@]}" "${disp_args[@]}" \
    > "$BRIDGE_LOG" 2>&1 &
  local bridge_pid=$!
  echo "$bridge_pid" > "$PIDFILE"
  log "bridge 가동. pid=$bridge_pid"

  if [ "$eval_count" -gt 0 ]; then
    log "정량 eval 진행: ${eval_count} episode → $eval_out"
    local bridge_rc=0
    wait "$bridge_pid" || bridge_rc=$?
    rm -f "$PIDFILE"
    # SIGINT로 ROS destroy_node를 실행해 final EEF metric을 flush한다.
    docker kill --signal=SIGINT vla_demo_node >/dev/null 2>&1 || true
    timeout 20 docker wait vla_demo_node >/dev/null 2>&1 || true
    docker rm -f vla_demo_node vla_demo_ps >/dev/null 2>&1 || true
    rm -f "${PROF_FILES[@]}" "$RESET_HOST" 2>/dev/null || true
    if [ "$bridge_rc" -ne 0 ]; then
      log "eval bridge 실패(rc=$bridge_rc). 로그: $BRIDGE_LOG"
      return "$bridge_rc"
    fi
    log "eval 완료: $REPO/$eval_out"
    log "runtime metrics: $REPO/logs/eef_sim_rollout.jsonl"
    return 0
  fi
  echo
  echo "  관전:   tail -f $BRIDGE_LOG"
  [ "$disp" = stream ] && echo "  WebRTC: Omniverse Streaming Client → ${ip:-<server-LAN-ip>}:49100"
  echo "  정지:   scripts/inference/demo_vla.sh stop"
}

# ── dispatch ──
case "${1:-}" in
  start)  shift; start "$@";;
  stop)   stop_all;;
  status) status;;
  *) echo "사용: scripts/inference/demo_vla.sh {start [profile] [--ckpt P] [--cubes N] [--task ID] [--ip A] [--gui|--headless] [--eval N] [--eval-out P] [--apc N] [--thr T] [--seed S] [--no-parity] [--slew|--no-slew] [--arm-vel V] | stop | status}";;
esac
