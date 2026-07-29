#!/usr/bin/env bash
# cuRobo pick-place 대량 데이터 생성 — 생성(GPU)과 LeRobot v3 변환(CPU)을 겹쳐 돌린다.
#
#   ./generate_dataset.sh [TOTAL_EP] [NUM_ENVS] [BATCH_EP] [OUT_ROOT]
#   기본:                  1000       16         64        datasets/so101_pickplace_pipelined
#
# 왜 배치로 쪼개나 (2026-07-28 실측, docs/spec/09_TACIT_KNOWLEDGE.md §13.6):
#   · 변환은 num_envs 와 무관한 상수다(64 ep 당 196 s = 3.06 s/에피소드, 편차 2%).
#   · 생성은 GPU(isaac-sim 컨테이너), 변환은 CPU(호스트 .venv + ffmpeg). 자원이 겹치지
#     않는데 직렬로 돌면 16-env 기준 686+196 = 882 s/64ep 다. 겹치면 686 s — 변환이 숨는다.
#   · HDF5 는 버리는 중간물이다(64 ep = 24 GB → v3 519 MB, 46배). 배치로 쪼개고 변환된
#     것부터 지우면 디스크 피크가 전체 375 GB → in-flight 2배치(~48 GB)로 떨어진다.
#
# 파이프라인 (배치 i 생성이 배치 i-1 변환과 겹친다):
#     생성 b0 ──▶ 생성 b1 ──▶ 생성 b2 ──▶ …
#                 변환 b0 ──▶ 변환 b1 ──▶ …
#
# ⚠ 변환 실패 시 그 배치 HDF5 는 **지우지 않고** 크게 알린다. 조용한 데이터 손실 금지.
# ⚠ 산출물은 datasets/(→ /DISK1) 에 쓴다. scratch/ 는 루트 파티션이라 대용량 금지
#   (2026-07-28 에 스윕 산출물 124 GB 로 / 가 100% 됐다).
#
# 산출: OUT_ROOT/v3/batch_NNN/  (배치당 LeRobot v3 데이터셋 1개)
#       변환기 writer 가 append 를 지원하지 않아(기존 디렉터리면 FileExistsError) 배치별로
#       나뉜다. 하나로 합치는 절차 = docs/spec/08_PIPELINES.md §5.8.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOTAL_EP="${1:-1000}"
NUM_ENVS="${2:-16}"
BATCH_EP="${3:-64}"
OUT_ROOT="${4:-$REPO/datasets/so101_pickplace_pipelined}"

TASK="${DATAGEN_TASK:-SimToReal-SO101-PickCube-DR-v0}"
TRIALS=$(( BATCH_EP / NUM_ENVS ))
[[ $TRIALS -lt 1 ]] && { echo "BATCH_EP($BATCH_EP) 가 NUM_ENVS($NUM_ENVS) 보다 작다"; exit 1; }
BATCH_EP=$(( TRIALS * NUM_ENVS ))                    # trial 경계로 내림 — 실제 배치 크기
N_BATCH=$(( (TOTAL_EP + BATCH_EP - 1) / BATCH_EP ))

H5DIR="$OUT_ROOT/hdf5"; V3DIR="$OUT_ROOT/v3"; LOGDIR="$OUT_ROOT/logs"
mkdir -p "$H5DIR" "$V3DIR" "$LOGDIR"

VOL=(
  -v isaac_lab_cache_kit:/isaac-sim/kit/cache
  -v isaac_lab_cache_ov:/root/.cache/ov
  -v isaac_lab_cache_compute:/root/.nv/ComputeCache
  -v isaac_lab_cache_gl:/root/.cache/nvidia/GLCache
  -v isaac_lab_cache_warp:/root/.cache/warp
  -v isaac_lab_data:/root/.local/share/ov/data
  -v "$REPO/src":/workspace/src
  -v "$REPO/scripts":/workspace/scripts
  -v "$REPO/assets":/workspace/assets
  -v "$REPO/outputs":/workspace/outputs
  -v "$OUT_ROOT":/workspace/out
)
PLANNER=datagen_planner

cleanup() { docker rm -f "$PLANNER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[datagen] total=$TOTAL_EP num_envs=$NUM_ENVS batch=$BATCH_EP × $N_BATCH  → $OUT_ROOT"
echo "[datagen] 디스크: $(df -h "$OUT_ROOT" | tail -1)"

# planner 는 **전 배치 공용으로 한 번만** 띄운다 — 배치마다 재기동하면 init 6 s 씩 낭비다.
cleanup
docker run --rm -d --name "$PLANNER" --gpus all --ipc host --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y "${VOL[@]}" \
  --entrypoint /isaac-sim/python.sh so101-curobo-datagen:5.1.0 \
  /workspace/scripts/cuRobo/curobo_batch_planner.py --max_batch_size "$NUM_ENVS" \
  > "$LOGDIR/planner.cid" 2>&1
for _ in $(seq 1 120); do
  docker logs "$PLANNER" 2>&1 | grep -q "ZMQ REP" && break
  docker ps --format '{{.Names}}' | grep -q "^${PLANNER}$" || { echo "[datagen] planner 사망"; docker logs "$PLANNER" 2>&1 | tail -5; exit 1; }
  sleep 2
done
echo "[datagen] planner ready"

CONV_PID=""; CONV_TAG=""; CONV_H5=""
FAILED_BATCHES=()   # 생성 실패
KEPT_H5=()          # 변환 실패로 보존한 HDF5

# 직전 배치 변환을 회수한다. 성공하면 HDF5 삭제, 실패하면 보존 + 경고.
reap_convert() {
  [[ -z "$CONV_PID" ]] && return 0
  wait "$CONV_PID"; local rc=$?
  local eps
  eps=$("$REPO/.venv/bin/python" -c "
import json,pathlib,sys
try: print(json.loads((pathlib.Path(sys.argv[1])/'meta'/'info.json').read_text()).get('total_episodes',0))
except Exception: print(0)" "$V3DIR/$CONV_TAG")
  if [[ $rc -eq 0 && ${eps:-0} -gt 0 ]]; then
    rm -f "$CONV_H5"
    echo "[datagen] ✓ 변환 $CONV_TAG: $eps ep → HDF5 삭제 (여유 $(df -h "$OUT_ROOT" | tail -1 | awk '{print $4}'))"
  else
    KEPT_H5+=("$CONV_H5")
    echo "[datagen] ✗✗✗ 변환 실패 $CONV_TAG (rc=$rc, episodes=${eps:-0})"
    echo "[datagen] ✗✗✗ HDF5 를 보존한다: $CONV_H5   로그: $LOGDIR/conv_$CONV_TAG.log"
  fi
  CONV_PID=""; CONV_TAG=""; CONV_H5=""
}

T_ALL0=$(date +%s)
for ((i=0; i<N_BATCH; i++)); do
  TAG=$(printf "batch_%03d" "$i")
  H5="$H5DIR/${TAG}.hdf5"
  echo "[datagen] === $TAG ($((i+1))/$N_BATCH) 생성 시작 $(date +%T) ==="
  T0=$(date +%s)
  docker run --rm --gpus all --ipc host --network host \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y "${VOL[@]}" \
    --entrypoint /isaac-sim/python.sh so101-isaac-sim:5.1.0 \
    /workspace/scripts/cuRobo/pickplace_sm.py random \
      --task "$TASK" --headless --enable_cameras \
      --num_envs "$NUM_ENVS" --auto_trials "$TRIALS" \
      --seed $(( 1000 * (i + 1) )) \
      --record_hdf5 "/workspace/out/hdf5/${TAG}.hdf5" \
      --summary_dir "/workspace/out/logs/${TAG}_summary" \
      --plan_timeout_s 900 > "$LOGDIR/gen_${TAG}.log" 2>&1
  GRC=$?
  echo "[datagen] $TAG 생성 $(( $(date +%s) - T0 ))s rc=$GRC"
  if [[ $GRC -ne 0 ]]; then
    echo "[datagen] ⚠ $TAG 생성 실패(rc=$GRC) — HDF5 보존하고 다음 배치로. 로그: $LOGDIR/gen_${TAG}.log"
    FAILED_BATCHES+=("$TAG"); [[ -f "$H5" ]] && KEPT_H5+=("$H5")
    continue
  fi

  reap_convert                      # 직전 배치 변환 회수 (여기까지 생성과 겹쳐 돌았다)

  # 이번 배치 변환을 백그라운드로 — 다음 배치 생성과 겹친다.
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" \
    "$REPO/scripts/convert/isaaclab2lerobotv3.py" \
    --hdf5_files "$H5" --output_dir "$V3DIR/$TAG" --overwrite \
    > "$LOGDIR/conv_${TAG}.log" 2>&1 &
  CONV_PID=$!; CONV_TAG="$TAG"; CONV_H5="$H5"
done
reap_convert                        # 마지막 배치 변환 (이건 못 숨는다)

echo "[datagen] ===== 완료 $(( $(date +%s) - T_ALL0 ))s ====="
"$REPO/.venv/bin/python" - "$V3DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); tot_ep = tot_fr = 0
for d in sorted(root.glob("batch_*")):
    try:
        info = json.loads((d / "meta" / "info.json").read_text())
    except Exception as e:
        print(f"  {d.name}: 메타 없음 ({type(e).__name__})"); continue
    ep, fr = info.get("total_episodes", 0), info.get("total_frames", 0)
    tot_ep += ep; tot_fr += fr
    print(f"  {d.name}: {ep} ep, {fr} frame")
print(f"합계: {tot_ep} 에피소드 · {tot_fr} frame · v3 디렉터리 {len(list(root.glob('batch_*')))}개")
PY
[[ ${#FAILED_BATCHES[@]} -gt 0 ]] && echo "[datagen] ⚠ 생성 실패 배치: ${FAILED_BATCHES[*]}"
[[ ${#KEPT_H5[@]} -gt 0 ]] && { echo "[datagen] ⚠ 보존된 HDF5 (수동 처리 필요):"; printf '    %s\n' "${KEPT_H5[@]}"; }
echo "[datagen] 배치 v3 를 합치는 절차 = docs/spec/08_PIPELINES.md §5.8"
