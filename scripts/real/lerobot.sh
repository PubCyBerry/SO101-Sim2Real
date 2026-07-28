#!/usr/bin/env bash
# =============================================================================
# lerobot.sh — Windows native uv 에서 .env 자동 로드 후 LeRobot CLI 실행
#
#   매번 `set -a; source .env; set +a` 하기 번거로워서 만든 래퍼.
#   .env + env/${POLICY_PROFILE}.env 를 읽어 변수를 CLI 인자로 매핑한다.
#
# 사용 (Git Bash, 레포 어디서든):
#   ./scripts/real/lerobot.sh <mode> [추가 인자...]
#
# mode:
#   find-port       포트 감지
#   setup-motors    모터 셋업           (대상 = CALIBRATE_TARGET)
#   calibrate       캘리브레이션         (대상 = CALIBRATE_TARGET)
#   teleop          텔레오퍼레이션       (+ 카메라)
#   record          데이터 수집          (+ 카메라 + 데이터셋)
#   replay          에피소드 재생
#   policy-client   실기기 VLA 추론 클라이언트
#   env             로드된 주요 변수 출력 (디버그)
#   raw <...>       env 로드 후 `uv run <...>` 그대로 실행 (탈출구)
#
# 추가 인자는 끝에 그대로 붙어 기본값을 덮어쓴다(LeRobot last-wins).
#   예) ./scripts/real/lerobot.sh record --dataset.num_episodes=3
#
# 카메라 끄기: LEROBOT_NO_CAMERAS=1 ./scripts/real/lerobot.sh teleop
# 프로필 1회 변경: POLICY_PROFILE=act ./scripts/real/lerobot.sh policy-client
# =============================================================================
set -euo pipefail

# ── 레포 루트로 이동 (이 스크립트 = <root>/scripts/real/lerobot.sh) ──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
UV_REAL=(uv run --project "${REPO_ROOT}/scripts/real")

# ── .env (+ 활성 프로필) 로드 ──
[ -f .env ] || { echo "[lerobot.sh] .env 없음: $REPO_ROOT/.env" >&2; exit 1; }
set -a
# shellcheck disable=SC1091
. ./.env
if [ -n "${POLICY_PROFILE:-}" ] && [ -f "env/${POLICY_PROFILE}.env" ]; then
  # shellcheck disable=SC1090
  . "env/${POLICY_PROFILE}.env"
  _PROFILE_NOTE="env/${POLICY_PROFILE}.env"
else
  _PROFILE_NOTE="(프로필 없음)"
fi
set +a

MODE="${1:-help}"
shift 2>/dev/null || true

# ── 공통 인자 묶음 ──
ROBOT_COMMON=(--robot.type="${ROBOT_TYPE}" --robot.port="${ROBOT_PORT}" --robot.id="${ROBOT_ID}")
TELEOP_COMMON=(--teleop.type="${TELEOP_TYPE}" --teleop.port="${TELEOP_PORT}" --teleop.id="${TELEOP_ID}")

# CALIBRATE_TARGET 에 따라 setup-motors / calibrate 대상 결정
if [ "${CALIBRATE_TARGET:-robot}" = "teleop" ]; then
  TARGET_ARGS=(--teleop.type="${TELEOP_TYPE}" --teleop.port="${TELEOP_PORT}" --teleop.id="${TELEOP_ID}")
else
  TARGET_ARGS=("${ROBOT_COMMON[@]}")
fi

# 카메라 인자 (ENABLED_CAMERAS 비었거나 LEROBOT_NO_CAMERAS=1 이면 생략)
CAMERA_ARGS=()
if [ -z "${LEROBOT_NO_CAMERAS:-}" ] && [ -n "${ENABLED_CAMERAS:-}" ] && [ -n "${CAMERAS:-}" ]; then
  CAMERA_ARGS=(--robot.cameras="${CAMERAS}")
fi

# 실행 직전 명령 echo. LEROBOT_DRY=1 이면 출력만 하고 실행 안 함(미리보기).
run() {
  echo "+ $*" >&2
  [ -n "${LEROBOT_DRY:-}" ] && exit 0
  exec "$@"
}

case "$MODE" in
  find-port)
    run "${UV_REAL[@]}" lerobot-find-port "$@"
    ;;

  setup-motors)
    run "${UV_REAL[@]}" lerobot-setup-motors "${TARGET_ARGS[@]}" "$@"
    ;;

  calibrate)
    run "${UV_REAL[@]}" lerobot-calibrate "${TARGET_ARGS[@]}" "$@"
    ;;

  teleop)
    # shellcheck disable=SC2086
    run "${UV_REAL[@]}" lerobot-teleoperate \
      "${ROBOT_COMMON[@]}" "${TELEOP_COMMON[@]}" "${CAMERA_ARGS[@]}" \
      ${TELEOP_EXTRA_ARGS:-} "$@"
    ;;

  record)
    # shellcheck disable=SC2086
    run "${UV_REAL[@]}" lerobot-record \
      "${ROBOT_COMMON[@]}" "${TELEOP_COMMON[@]}" "${CAMERA_ARGS[@]}" \
      --dataset.repo_id="${HF_DATASET_REPO_ID}" \
      --dataset.single_task="${SINGLE_TASK}" \
      --dataset.num_episodes="${NUM_EPISODES}" \
      --dataset.fps="${RECORD_FPS}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.push_to_hub="${PUSH_TO_HUB}" \
      ${RECORD_EXTRA_ARGS:-} "$@"
    ;;

  replay)
    # shellcheck disable=SC2086
    run "${UV_REAL[@]}" lerobot-replay \
      "${ROBOT_COMMON[@]}" \
      --dataset.repo_id="${HF_DATASET_REPO_ID}" \
      --dataset.episode="${EPISODE_INDEX}" \
      ${REPLAY_EXTRA_ARGS:-} "$@"
    ;;

  policy-client)
    # schema v2 preflight — client dispatch는 **checkpoint manifest**가 결정한다.
    # ACTION_REPRESENTATION_MODE 는 (있으면) assertion으로만 쓰이고 override가 아니다.
    # 두 joint mode도 여기서 resolve+assert를 통과해야 robot 객체가 만들어진다.
    # run() 은 exec 하므로 preflight 는 직접 호출한다.
    echo "+ preflight: assert_checkpoint_representation --emit client_kind" >&2
    if ! SO101_CLIENT_KIND="$("${UV_REAL[@]}" python scripts/inference/assert_checkpoint_representation.py \
      --checkpoint "${POLICY_REPO_ID}" --from-env --skip-kinematics --emit client_kind)"; then
      echo "checkpoint action representation preflight failed (migrate legacy checkpoints first)" >&2
      exit 1
    fi
    echo "[action-representation] resolved client_kind=${SO101_CLIENT_KIND} (manifest-driven)"
    # 4 mode 모두 representation-aware client 를 쓴다.
    #   EEF  : FK/IK adapter + router(IK 1회)
    #   joint: canonical joint feature 경계 + router(IK 0회) — stock client 가 아니다.
    # shellcheck disable=SC2086
    run "${UV_REAL[@]}" python scripts/inference/eef_robot_client.py \
      --server_address="${POLICY_SERVER_ADDRESS}" \
      --policy_type="${POLICY_TYPE}" \
      --pretrained_name_or_path="${POLICY_REPO_ID}" \
      --policy_device="${DEVICE:-cuda}" \
      --task="${TASK}" \
      --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
      --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
      --aggregate_fn_name=latest_only \
      --client_device="${CLIENT_DEVICE}" \
      --real_hardware_ik_validated="${EEF_IK_REAL_VALIDATED:-false}" \
      --eef_metrics_log="${EEF_REAL_METRICS_LOG:-}" \
      "${ROBOT_COMMON[@]}" "${CAMERA_ARGS[@]}" \
      ${POLICY_CLIENT_EXTRA_ARGS:-} "$@"
    ;;

  env)
    echo "REPO_ROOT       = $REPO_ROOT"
    echo "POLICY_PROFILE  = ${POLICY_PROFILE:-} -> ${_PROFILE_NOTE}"
    echo "ROBOT           = ${ROBOT_TYPE} @ ${ROBOT_PORT} (id=${ROBOT_ID})"
    echo "TELEOP          = ${TELEOP_TYPE} @ ${TELEOP_PORT} (id=${TELEOP_ID})"
    echo "CALIBRATE_TARGET= ${CALIBRATE_TARGET:-robot}"
    echo "ENABLED_CAMERAS = ${ENABLED_CAMERAS:-}"
    echo "POLICY_TYPE     = ${POLICY_TYPE:-}"
    echo "ACTION_REPRESENTATION_MODE = ${ACTION_REPRESENTATION_MODE:-<manifest>}"
    echo "EEF_IK_REAL_VALIDATED = ${EEF_IK_REAL_VALIDATED:-false}"
    echo "ACTIONS_PER_CHUNK = ${ACTIONS_PER_CHUNK:-}"
    echo "POLICY_SERVER_ADDRESS = ${POLICY_SERVER_ADDRESS:-}"
    echo "HF_DATASET_REPO_ID = ${HF_DATASET_REPO_ID:-}"
    echo "TASK            = ${TASK:-}"
    ;;

  raw)
    run "${UV_REAL[@]}" "$@"
    ;;

  help|-h|--help|"")
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;

  *)
    echo "[lerobot.sh] 알 수 없는 mode: '$MODE'" >&2
    echo "사용 가능: find-port setup-motors calibrate teleop record replay policy-client env raw help" >&2
    exit 2
    ;;
esac
