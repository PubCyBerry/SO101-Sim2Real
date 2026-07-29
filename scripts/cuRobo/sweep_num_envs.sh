#!/usr/bin/env bash
# num_envs 스윕 — 구성마다 64 에피소드를 생성하고 LeRobot v3 변환까지 마친 wall-clock 을 잰다.
#
#   ./sweep_num_envs.sh "1 2 4 8 16" [EPISODES]
#
# 구성당: planner 기동 → SM record(auto_trials = EPISODES/num_envs) → planner 종료 →
#         isaaclab2lerobotv3.py 변환. 각 단계 wall-clock 과 산출물 크기를 JSON 으로 남긴다.
# OOM 등으로 한 구성이 죽어도 다음 구성으로 넘어간다(status 에 기록).
#
# ⚠ 산출물이 구성당 24 GB 다. `scratch/` 는 보통 루트 파티션이니 여유를 먼저 확인할 것
#   (`df -h .`). 2026-07-28 측정 때 5 구성 124 GB 로 / 를 채웠다.
# ⚠ GPU 를 독점해야 한다. 다른 워크로드가 있으면 VRAM 절대값이 오염되고, 컨테이너를
#   일괄 정리하는 오케스트레이션이 있으면 rc=137 로 전멸한다(둘 다 실제로 겪었다).
#
# 2026-07-28 기준선(48.9 GB GPU 유휴, 구성당 64 ep) = docs/spec/09_TACIT_KNOWLEDGE.md §13.6
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/scratch/num-envs-sweep"      # 로그·warp 캐시 (gitignore, 실행 시 생성)
OUT="$HERE/out"                          # HDF5·v3 산출물 — ⚠ 구성당 24 GB
CONFIGS="${1:-1 2 4 8 16}"
EPISODES="${2:-64}"
RESULT="$HERE/logs/sweep_num_envs.json"

mkdir -p "$OUT" "$HERE/logs" "$HERE/warpcache"

VOL=(
  -v isaac_lab_cache_kit:/isaac-sim/kit/cache
  -v isaac_lab_cache_ov:/root/.cache/ov
  -v isaac_lab_cache_compute:/root/.nv/ComputeCache
  -v isaac_lab_cache_gl:/root/.cache/nvidia/GLCache
  -v isaac_lab_data:/root/.local/share/ov/data
  -v "$HERE/warpcache":/root/.cache/warp
  -v "$REPO/src":/workspace/src
  -v "$REPO/scripts":/workspace/scripts
  -v "$REPO/assets":/workspace/assets
  -v "$REPO/outputs":/workspace/outputs
  -v "$REPO/scratch":/workspace/scratch
)
PLANNER=sweep_planner

cleanup() { docker rm -f "$PLANNER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[" > "$RESULT"
FIRST=1

for N in $CONFIGS; do
  TRIALS=$(( (EPISODES + N - 1) / N ))
  TAG="n${N}"
  H5="$OUT/${TAG}.hdf5"
  LEROBOT="$OUT/${TAG}_lerobot"
  GENLOG="$HERE/logs/sweep_${TAG}_gen.log"
  CONVLOG="$HERE/logs/sweep_${TAG}_conv.log"
  rm -rf "$H5" "$LEROBOT"

  echo "=== num_envs=$N  trials=$TRIALS  (목표 $EPISODES ep) ==="

  # planner — max_batch_size 를 num_envs 와 맞춰 요청 시 재초기화를 피한다.
  cleanup
  docker run --rm -d --name "$PLANNER" --gpus all --ipc host --network host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y "${VOL[@]}" \
    --entrypoint /isaac-sim/python.sh so101-curobo-datagen:5.1.0 \
    /workspace/scripts/cuRobo/curobo_batch_planner.py --max_batch_size "$N" >/dev/null 2>&1
  PLANNER_OK=0
  for _ in $(seq 1 90); do
    docker logs "$PLANNER" 2>&1 | grep -q "ZMQ REP" && { PLANNER_OK=1; break; }
    docker ps --format '{{.Names}}' | grep -q "^${PLANNER}$" || break
    sleep 2
  done
  if [[ $PLANNER_OK -eq 0 ]]; then
    echo "  planner 기동 실패 — 건너뜀"
    docker logs "$PLANNER" 2>&1 | tail -5
    if [[ $FIRST -eq 0 ]]; then echo "," >> "$RESULT"; fi
  FIRST=0
    printf '{"num_envs":%d,"status":"planner_failed"}' "$N" >> "$RESULT"
    continue
  fi

  # VRAM 샘플링
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 2 \
    > "$HERE/logs/sweep_${TAG}_vram.csv" 2>/dev/null &
  VPID=$!

  T0=$(date +%s.%N)
  docker run --rm --gpus all --ipc host --network host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y "${VOL[@]}" \
    --entrypoint /isaac-sim/python.sh so101-isaac-sim:5.1.0 \
    /workspace/scripts/cuRobo/pickplace_sm.py random \
      --task SimToReal-SO101-PickCube-DR-v0 --headless --enable_cameras \
      --num_envs "$N" --auto_trials "$TRIALS" \
      --record_hdf5 "/workspace/scratch/sweep-out/${TAG}.hdf5" \
      --summary_dir "/workspace/scratch/sweep-out/${TAG}_summary" \
      --plan_timeout_s 900 > "$GENLOG" 2>&1
  GEN_RC=$?
  T1=$(date +%s.%N)
  kill $VPID 2>/dev/null
  cleanup

  GEN_S=$(echo "$T1 - $T0" | bc)
  if [[ $GEN_RC -eq 137 ]]; then
    echo "  ⚠ SM 컨테이너가 SIGKILL 로 죽었다(rc=137) — 외부 요인. 부분 결과다."
  fi
  VRAM=$(sort -n "$HERE/logs/sweep_${TAG}_vram.csv" 2>/dev/null | tail -1 | tr -dc '0-9')
  H5_MB=$(du -sm "$H5" 2>/dev/null | cut -f1)

  # 변환 (LeRobot v3) — 성공 demo 만
  T2=$(date +%s.%N)
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" \
    "$REPO/scripts/convert/isaaclab2lerobotv3.py" \
    --hdf5_files "$H5" --output_dir "$LEROBOT" > "$CONVLOG" 2>&1
  CONV_RC=$?
  T3=$(date +%s.%N)
  CONV_S=$(echo "$T3 - $T2" | bc)
  LR_MB=$(du -sm "$LEROBOT" 2>/dev/null | cut -f1)

  # 산출 집계 — HDF5 demo 수·성공 수, LeRobot 에피소드 수
  STATS=$("$REPO/.venv/bin/python" - "$H5" "$LEROBOT" <<'PY'
import json, pathlib, sys
h5p, lrp = sys.argv[1], sys.argv[2]
out = {"demos": 0, "success": 0, "lerobot_episodes": 0, "lerobot_frames": 0}
try:
    import h5py
    with h5py.File(h5p, "r") as f:
        d = f["data"]
        out["demos"] = len(d)
        out["success"] = sum(1 for n in d if bool(d[n].attrs.get("success", False)))
except Exception as e:
    out["h5_error"] = f"{type(e).__name__}: {e}"
try:
    info = json.loads((pathlib.Path(lrp) / "meta" / "info.json").read_text())
    out["lerobot_episodes"] = info.get("total_episodes", 0)
    out["lerobot_frames"] = info.get("total_frames", 0)
except Exception as e:
    out["lerobot_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
PY
)

  echo "  gen=${GEN_S}s conv=${CONV_S}s rc=$GEN_RC/$CONV_RC vram=${VRAM:-?} $STATS"

  if [[ $FIRST -eq 0 ]]; then echo "," >> "$RESULT"; fi
  FIRST=0
  "$REPO/.venv/bin/python" - "$N" "$TRIALS" "$GEN_S" "$CONV_S" "$GEN_RC" "$CONV_RC" \
      "${VRAM:-0}" "${H5_MB:-0}" "${LR_MB:-0}" "$STATS" >> "$RESULT" <<'PY'
import json, sys
n, trials, gen, conv, grc, crc, vram, h5mb, lrmb, stats = sys.argv[1:11]
row = {
    "num_envs": int(n), "trials": int(trials),
    "gen_s": round(float(gen), 1), "convert_s": round(float(conv), 1),
    "total_s": round(float(gen) + float(conv), 1),
    "gen_rc": int(grc), "convert_rc": int(crc),
    "vram_peak_mib": int(vram), "hdf5_mb": int(h5mb), "lerobot_mb": int(lrmb),
    **json.loads(stats),
}
succ = row.get("success", 0)
row["s_per_success_ep"] = round(row["total_s"] / succ, 2) if succ else None
row["status"] = "ok" if (grc == "0" and crc == "0" and succ) else "failed"
print(json.dumps(row), end="")
PY
done

echo "]" >> "$RESULT"
echo "[sweep] → $RESULT"
"$REPO/.venv/bin/python" -c "
import json
rows = json.load(open('$RESULT'))
print(f\"{'envs':>5}{'trials':>7}{'gen s':>9}{'conv s':>8}{'total s':>9}{'demos':>7}{'succ':>6}{'s/succ ep':>11}{'VRAM MiB':>10}{'hdf5 MB':>9}{'v3 MB':>8}  status\")
for r in rows:
    if r.get('status') == 'planner_failed':
        print(f\"{r['num_envs']:>5}{'':>7}{'—':>9}  planner 기동 실패\"); continue
    print(f\"{r['num_envs']:>5}{r['trials']:>7}{r['gen_s']:>9.1f}{r['convert_s']:>8.1f}{r['total_s']:>9.1f}\"
          f\"{r.get('demos',0):>7}{r.get('success',0):>6}{str(r.get('s_per_success_ep')):>11}\"
          f\"{r.get('vram_peak_mib',0):>10}{r.get('hdf5_mb',0):>9}{r.get('lerobot_mb',0):>8}  {r['status']}\")
"
