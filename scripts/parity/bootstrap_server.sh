#!/usr/bin/env bash
set -euo pipefail

# konan147의 Isaac Sim 6 / ROS 2 Jazzy runtime을 고정 버전으로 설치하고 검증한다.
# 호스트 전역 package는 변경하지 않으며 모든 산출물은 SO101_RUNTIME_ROOT 아래에 둔다.

PIXI_VERSION="0.70.2"
PIXI_ARCHIVE="pixi-x86_64-unknown-linux-musl.tar.gz"
PIXI_SHA256="9d0d72fc1fa1a8f87bfde943cf25d3575cb352cb516795d4dab21898bd98adea"
ISAACLAB_COMMIT="28a37cecdd433c22d9eabd6a5954add9f13a8951"
LOCK_SHA256="234ba771eafb1b870a97f5ffe35887d89fe12188f093963ea3fc0ebc9f14854b"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${SO101_RUNTIME_ROOT:-/DISK1/so101-sim2real/runtime/isaac6_ros}"
CHECK_ONLY=0
RUN_SMOKE=0
RUN_PROBE=0

usage() {
  cat <<'EOF'
사용법:
  bash scripts/parity/bootstrap_server.sh [--check-only] [--smoke] [--probe]

옵션:
  --check-only  다운로드와 Pixi install을 생략하고 현재 설치만 검증한다.
  --smoke       Isaac compatibility checker와 3-camera environment smoke를 실행한다.
  --probe       실행 중인 Zenoh router/VLA replay server를 대상으로 transport probe를 실행한다.
EOF
}

while (($#)); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --smoke)
      RUN_SMOKE=1
      ;;
    --probe)
      RUN_PROBE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "필수 명령을 찾을 수 없다: $1"
}

[[ "$(uname -s)" == "Linux" ]] || fail "이 스크립트는 Linux 서버 전용이다"
[[ "$(uname -m)" == "x86_64" ]] || fail "지원하지 않는 architecture: $(uname -m)"
[[ "$PROJECT_ROOT" == "$RUNTIME_ROOT" ]] || fail \
  "project root와 고정 runtime root가 다르다: project=$PROJECT_ROOT runtime=$RUNTIME_ROOT"

require_command git
require_command sha256sum
require_command tar
require_command jq

cd "$PROJECT_ROOT"
[[ -f pixi.toml && -f pixi.lock ]] || fail "tracked pixi.toml/pixi.lock이 없다"
[[ -f configs/parity/runtime_manifest.mock.json ]] || fail "runtime manifest가 없다"
[[ -f configs/parity/replay_checkpoint.json ]] || fail "replay checkpoint가 없다"

mkdir -p bin outputs/parity
PIXI="$RUNTIME_ROOT/bin/pixi"

if ((CHECK_ONLY == 0)); then
  require_command gh
  if [[ ! -x "$PIXI" ]]; then
    download_dir="$RUNTIME_ROOT/bin"
    archive_path="$download_dir/$PIXI_ARCHIVE"
    rm -f "$archive_path" "$archive_path.sha256"
    gh release download "v$PIXI_VERSION" \
      --repo prefix-dev/pixi \
      --pattern "$PIXI_ARCHIVE" \
      --pattern "$PIXI_ARCHIVE.sha256" \
      --dir "$download_dir"
    (
      cd "$download_dir"
      sha256sum --check "$PIXI_ARCHIVE.sha256"
      tar -xzf "$PIXI_ARCHIVE"
      chmod 0755 pixi
    )
  fi
fi

[[ -x "$PIXI" ]] || fail "Pixi executable이 없다: $PIXI"
actual_pixi_version="$("$PIXI" --version | awk '{print $2}')"
[[ "$actual_pixi_version" == "$PIXI_VERSION" ]] || fail \
  "Pixi version mismatch: actual=$actual_pixi_version expected=$PIXI_VERSION"

archive_path="$RUNTIME_ROOT/bin/$PIXI_ARCHIVE"
if [[ -f "$archive_path" ]]; then
  actual_archive_sha="$(sha256sum "$archive_path" | awk '{print $1}')"
  [[ "$actual_archive_sha" == "$PIXI_SHA256" ]] || fail \
    "Pixi archive hash mismatch: actual=$actual_archive_sha expected=$PIXI_SHA256"
fi

actual_lock_sha="$(sha256sum pixi.lock | awk '{print $1}')"
[[ "$actual_lock_sha" == "$LOCK_SHA256" ]] || fail \
  "pixi.lock hash mismatch: actual=$actual_lock_sha expected=$LOCK_SHA256"

ISAACLAB_ROOT="$RUNTIME_ROOT/IsaacLab"
if ((CHECK_ONLY == 0)) && [[ ! -d "$ISAACLAB_ROOT/.git" ]]; then
  gh repo clone isaac-sim/IsaacLab "$ISAACLAB_ROOT" -- --filter=blob:none
fi
[[ -d "$ISAACLAB_ROOT/.git" ]] || fail "IsaacLab source checkout이 없다: $ISAACLAB_ROOT"

if ((CHECK_ONLY == 0)); then
  if [[ -n "$(git -C "$ISAACLAB_ROOT" status --porcelain)" ]]; then
    fail "IsaacLab checkout에 미보존 변경이 있어 commit 전환을 거부한다"
  fi
  if ! git -C "$ISAACLAB_ROOT" cat-file -e "$ISAACLAB_COMMIT^{commit}" 2>/dev/null; then
    git -C "$ISAACLAB_ROOT" fetch --filter=blob:none origin "$ISAACLAB_COMMIT"
  fi
  git -C "$ISAACLAB_ROOT" checkout --detach "$ISAACLAB_COMMIT"
fi

actual_isaaclab_commit="$(git -C "$ISAACLAB_ROOT" rev-parse HEAD)"
[[ "$actual_isaaclab_commit" == "$ISAACLAB_COMMIT" ]] || fail \
  "IsaacLab commit mismatch: actual=$actual_isaaclab_commit expected=$ISAACLAB_COMMIT"

"$PIXI" lock --check
if ((CHECK_ONLY == 0)); then
  "$PIXI" install -e sim --locked
  "$PIXI" install -e real --locked
  "$PIXI" install -e ros-tools --locked
fi

"$PIXI" run stack-check-sim
"$PIXI" run stack-check-real
"$PIXI" run stack-check-ros
"$PIXI" run ros-build
"$PIXI" run core-test
"$PIXI" run dataset-test
"$PIXI" run -e ros-tools python scripts/parity/validate_checkpoint.py \
  --manifest configs/parity/runtime_manifest.mock.json \
  --checkpoint configs/parity/replay_checkpoint.json

manifest_lock_sha="$(jq -r '.pixi_lock_hash' configs/parity/runtime_manifest.mock.json)"
[[ "$manifest_lock_sha" == "$LOCK_SHA256" ]] || fail \
  "runtime manifest의 pixi_lock_hash가 현재 lock과 다르다: $manifest_lock_sha"

if ((RUN_SMOKE == 1)); then
  "$PIXI" run sim-compatibility-check-headless \
    --report outputs/parity/isaac_compatibility_server.json
  "$PIXI" run -e sim python scripts/parity/isaac6_smoke.py \
    --stage camera \
    --steps 5 \
    --report outputs/parity/isaac6_camera_smoke_server.json \
    --visualizer none
fi

if ((RUN_PROBE == 1)); then
  ZENOH_SESSION_CONFIG_URI="$PROJECT_ROOT/configs/zenoh/server-client.json5" \
    "$PIXI" run -e ros-tools python -m so101_vla_runtime.integration_probe \
      --samples 100 \
      --warmup 5 \
      --image-pattern gradient \
      --report outputs/parity/zenoh_probe_server.json
fi

jq -n \
  --arg status "passed" \
  --arg root "$RUNTIME_ROOT" \
  --arg pixi "$actual_pixi_version" \
  --arg lock "$actual_lock_sha" \
  --arg isaaclab "$actual_isaaclab_commit" \
  --arg manifest "$(jq -r '.manifest_hash' configs/parity/runtime_manifest.mock.json)" \
  --argjson smoke "$RUN_SMOKE" \
  --argjson probe "$RUN_PROBE" \
  '{
    status: $status,
    runtime_root: $root,
    pixi_version: $pixi,
    pixi_lock_sha256: $lock,
    isaaclab_commit: $isaaclab,
    runtime_manifest_hash: $manifest,
    smoke_executed: ($smoke == 1),
    transport_probe_executed: ($probe == 1)
  }'
