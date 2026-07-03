#!/usr/bin/env bash
# =============================================================================
# autoresearch 고정 harness — 수정 금지 (upstream autoresearch 의 prepare.py 역할)
#
# 1 실험 = ① pink 노드 오프라인 --self-check 게이트(수초)
#          ② bridge --eval N_EP: full-DR(bell) 단일 큐브 배치를 에피소드마다
#             고정 seed(SEED+ep)로 재현 리셋 + cube-in-bowl 물리 판정
#          ③ 에피소드 시작(bridge 로그 "scene reset" 마커)마다 pink 노드 컨테이너
#             fresh 재기동 → tf 재조회·재계획 (에피소드 간 상태 오염 차단)
#          ④ outputs/pink_eval.json 파싱 → greppable 요약 출력
#
# 성공 판정(고정 메트릭): 큐브가 그릇 안(BOWL_SUCCESS_RADIUS·z window)에 들어가면
# 에피소드 성공. success_rate = 성공 에피소드 / N_EP (높을수록 좋음).
#
# 사용:  bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1
# 결과:  grep "^success_rate:\|^harness_status:" autoresearch/run.log
# knob(환경변수): N_EP(20) SEED(42) TASK(SimToReal-SO101-PickCube-DR-v0)
#                 EVAL_SECONDS(35) CUBE_NAME("") MAX_SEC(2400)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

N_EP="${N_EP:-20}"
SEED="${SEED:-42}"
TASK="${TASK:-SimToReal-SO101-PickCube-DR-v0}"
EVAL_SECONDS="${EVAL_SECONDS:-35}"
CUBE_NAME="${CUBE_NAME:-}"          # 빈값=Cube1(40mm). 크기별 eval 시 Cube3 등
MAX_SEC="${MAX_SEC:-2400}"          # 전체 실험 watchdog (bridge 행 방지)

DC=(docker compose --env-file "$ROOT/.env" -f "$ROOT/docker/docker-compose.yaml")
NODE=/workspace/scripts/datagen/pink_ik_bridge_node.py
EVAL_JSON="$ROOT/outputs/pink_eval.json"
BRIDGE_LOG="$HERE/bridge.log"
PINK_LOG="$HERE/pink.log"
SELFCHECK_LOG="$HERE/selfcheck.log"

fail() { echo "harness_status: $1"; exit 1; }

cleanup() {
  docker ps -aq -f name='^ar_pink' | xargs -r docker rm -f >/dev/null 2>&1 || true
  docker rm -f so101_isaac_sim >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

# ── ① 오프라인 self-check 게이트 (기하·plan 7케이스, ROS/GPU 불요) ──────────
echo "[harness] self-check…"
if ! "${DC[@]}" run --rm -T --name ar_selfcheck pink-ik \
      python3 "$NODE" --self-check > "$SELFCHECK_LOG" 2>&1; then
  tail -n 30 "$SELFCHECK_LOG"
  fail "selfcheck_failed"
fi

# ── ② bridge --eval 기동 ────────────────────────────────────────────────────
rm -f "$EVAL_JSON"
: > "$BRIDGE_LOG"
: > "$PINK_LOG"
CUBE_ARG=""
[[ -n "$CUBE_NAME" ]] && CUBE_ARG="--cube_name $CUBE_NAME"
export NUM_CUBES=1
export BRIDGE_EXTRA_ARGS="--headless --task $TASK --dr --seed $SEED \
  --eval $N_EP --eval_seconds $EVAL_SECONDS --eval_settle 1.5 --eval_warmup 0 \
  --eval_out outputs/pink_eval.json $CUBE_ARG"
echo "[harness] bridge 기동: $TASK · $N_EP ep × ${EVAL_SECONDS}s · seed=$SEED"
"${DC[@]}" up -d isaac-sim || fail "bridge_up_failed"
docker logs -f so101_isaac_sim >> "$BRIDGE_LOG" 2>&1 &

# ── ③ 에피소드 동기화: "scene reset" 마커마다 pink 노드 fresh 재기동 ─────────
# bridge 는 기동 시 1회 선리셋 후 "EVAL 시작" → 에피소드마다 reset. 그래서
# eval 시작 이후의 reset 수 - 0 이 아니라, 전체 reset 수 - 1 = 현재 에피소드 번호.
start_pink() {
  local ep="$1"
  docker ps -aq -f name='^ar_pink' | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== episode $ep pink start $(date +%T) ===" >> "$PINK_LOG"
  "${DC[@]}" run --rm -T --name "ar_pink_$ep" pink-ik \
    python3 "$NODE" >> "$PINK_LOG" 2>&1 &
}

started=0
ep_launched=-1
deadline=$((SECONDS + MAX_SEC))
while :; do
  (( SECONDS > deadline )) && { tail -n 30 "$BRIDGE_LOG"; fail "timeout"; }
  if [[ $started == 0 ]] && grep -q "EVAL 시작" "$BRIDGE_LOG"; then
    started=1
    echo "[harness] eval 시작 감지"
  fi
  if [[ $started == 1 ]]; then
    n_resets=$(grep -c "scene reset" "$BRIDGE_LOG" || true)
    ep_now=$((n_resets - 1))   # 기동 시 선리셋 1회 제외
    if (( ep_now > ep_launched && ep_now < N_EP )); then
      ep_launched=$ep_now
      start_pink "$ep_launched"
      echo "[harness] episode $ep_launched → pink 재기동"
    fi
  fi
  running=$(docker inspect -f '{{.State.Running}}' so101_isaac_sim 2>/dev/null || echo false)
  [[ "$running" != "true" ]] && break
  sleep 2
done
sleep 1   # 마지막 docker logs flush 대기

# ── ④ 결과 파싱 ─────────────────────────────────────────────────────────────
if [[ ! -f "$EVAL_JSON" ]]; then
  echo "[harness] eval JSON 없음 — bridge 크래시로 판단. bridge.log tail:"
  tail -n 40 "$BRIDGE_LOG"
  fail "bridge_crashed"
fi
jq -r '
  "success_rate: \(.all_cubes_success_rate)",
  "ever_rate: \(.per_cube_ever_rate)",
  "n_episodes: \(.n_episodes)"
' "$EVAL_JSON"
echo "episodes:"
jq -r '.episodes[] | "  ep\(.episode): \(if .all_ok then "PASS" else "FAIL" end) ever=\(.n_ever) success_step=\(.success_step)"' "$EVAL_JSON"
echo "harness_status: ok"
