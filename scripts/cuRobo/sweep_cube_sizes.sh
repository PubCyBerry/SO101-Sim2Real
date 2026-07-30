#!/usr/bin/env bash
# 큐브 크기 DR 검증 — 크기를 하나로 고정한 sweep 을 사다리(25/30/35/40 mm)마다 1회.
#
# 공유 GPU 서버라 planner 컨테이너가 다른 세션의 정리 명령에 SIGKILL 당하는 일이 잦다.
# 그래서 (a) planner watchdog 를 백그라운드로 돌려 죽으면 즉시 되살리고,
#        (b) SM 로그에 `planner TIMEOUT` 이 찍힌 sweep(=planner 부재 구간이 있던 실행)은
#            결과를 버리고 재시도한다 — 그 구간은 진짜 실패가 아니라 인프라 사고다.
# 결과 = outputs/cube_size_dr_sweep/.
#   사용: ./scripts/cuRobo/sweep_cube_sizes.sh [yaw] [trials] [tag]   (기본 yaw=0, trials=1)
#   ⚠ 저장소 루트에서 실행한다(docker compose·outputs 상대경로).
set -u
YAW="${1:-0}"
TRIALS="${2:-1}"
TAG="${3:-yaw${YAW}}"
OUT_HOST=outputs/cube_size_dr_sweep
PLANNER_LOG="${OUT_HOST}/planner.log"
mkdir -p "$OUT_HOST"

start_planner() {
  docker rm -f curobo-planner >/dev/null 2>&1
  nohup docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
    --name curobo-planner curobo-datagen \
    python /workspace/scripts/cuRobo/curobo_batch_planner.py --max_batch_size 12 \
    >> "$PLANNER_LOG" 2>&1 &
}

wait_planner() {
  for _ in $(seq 1 60); do
    docker ps --format '{{.Names}}' | grep -qx curobo-planner \
      && docker logs curobo-planner 2>&1 | grep -q "ZMQ REP" && return 0
    sleep 5
  done
  return 1
}

watchdog() {  # planner 가 사라지면 되살린다(20 s 주기)
  while true; do
    docker ps --format '{{.Names}}' | grep -qx curobo-planner || {
      echo "[watchdog] planner 사망 → 재기동 $(date +%H:%M:%S)"
      start_planner
      wait_planner || echo "[watchdog] 재기동 실패"
    }
    sleep 20
  done
}

score() {  # $1 = json path
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"(no result: {e})"); raise SystemExit
n = sum(c["n"] for c in d["cells"]); ok = sum(c["n_placed"] for c in d["cells"])
pl = sum(c["n_planned"] for c in d["cells"])
print(f"placed {ok}/{n} ({100.0 * ok / max(n, 1):.1f}%)  planned {pl}/{n}")
PY
}

start_planner; wait_planner || { echo "planner 기동 실패"; exit 1; }
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

for mm in 25 30 35 40; do
  size="0.0${mm}"
  name="sweep_${mm}mm_${TAG}"
  json="${OUT_HOST}/${name}.json"
  log="${OUT_HOST}/${name}.log"
  # 이미 완주한 결과가 있으면 건너뛴다(중단 후 이어달리기).
  if [ -f "$json" ] && [ "$(grep -c 'chunk 11/11' "$log" 2>/dev/null || echo 0)" -gt 0 ] \
     && ! grep -q "planner TIMEOUT" "$log" 2>/dev/null; then
    echo "=== ${mm} mm SKIP (완료본 존재): $(score "$json")"
    continue
  fi
  for attempt in 1 2 3; do
    echo "=== ${mm} mm (yaw=${YAW} trials=${TRIALS}) attempt ${attempt} → ${name} ==="
    docker rm -f "sm-${name}" >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name "sm-${name}" isaac-sim \
      python /workspace/scripts/cuRobo/pickplace_sm.py sweep \
      --task SimToReal-SO101-PickCube-DR-v0 --num_envs 12 --headless --seed 0 \
      --yaw "$YAW" --trials "$TRIALS" --cube_sizes "$size" --plan_timeout_s 300 \
      --out "/workspace/${json}" > "$log" 2>&1
    rc=$?
    echo "=== ${mm} mm attempt ${attempt} rc=${rc}: $(score "$json")"
    if [ "$rc" -ne 0 ]; then echo "--- rc=${rc}(외부 종료) → 재시도 ---"; continue; fi
    if grep -q "planner TIMEOUT" "$log"; then
      echo "--- planner 부재 구간 있음 → 결과 폐기 후 재시도 ---"; continue
    fi
    break
  done
done
echo "ALL_SWEEPS_DONE"
