#!/usr/bin/env bash
# =============================================================================
# lerobot-entrypoint.sh — `lerobot` 서비스 (Dockerfile.lerobot) 진입점
#
# 본 스크립트는 로봇 직결 워크플로(teleop / record / replay / calibrate /
# find-* / dataset-viz / policy-client / edit-dataset) 만 다룬다.
# 정책 학습/평가(`train`, `eval`)와 정책 서버(`prepare-model`, `policy-server`)는
# `docker/policy-entrypoint.sh` (Dockerfile.policy 가 사용) 에 분리되어 있다 —
# 이 이미지에는 smolvla deps (transformers/accelerate 등) 가 설치되지 않기 때문.
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   teleop          : lerobot-teleoperate  — 실시간 원격 조작
#   record          : lerobot-record       — 텔레옵 기반 데이터 수집
#   replay          : lerobot-replay       — 녹화 에피소드 재실행
#   calibrate       : lerobot-calibrate    — 로봇/텔레옵 보정
#   setup-motors    : lerobot-setup-motors — 모터 ID/Baud 초기 설정
#   find-joint-limits: lerobot-find-joint-limits — 관절 가동 범위 탐색
#   find-cameras    : lerobot-find-cameras — 연결된 카메라 검색 및 캡처 확인
#   find-port       : lerobot-find-port    — MotorsBus USB 포트 자동 감지
#   dataset-viz     : lerobot-dataset-viz  — 데이터셋 시각화 (Rerun)
#   policy-client   : lerobot.async_inference.robot_client — 정책 서버에 붙어 팔로워 구동
#   edit-dataset    : lerobot-edit-dataset — 데이터셋 편집 (인자 완전 위임)
#   info            : lerobot-info         — LeRobot/시스템 정보 출력
#   bash | shell    : 인터랙티브 Bash 쉘
#   python <args>   : python 직접 실행
#   <기타>           : 명령 그대로 exec
#
# ■ 환경 변수 요약 (docker-compose.yaml ↔ .env 에서 주입)
#   하드웨어 : TELEOP_PORT  TELEOP_ID  ROBOT_PORT  ROBOT_ID
#             ENABLED_CAMERAS  TOP_CAM_PORT  WRIST_CAM_PORT  (FRONT_CAM_PORT 선택)
#             CAM_WIDTH  CAM_HEIGHT  CAM_FPS  CAM_WARMUP_S  CAM_FOURCC
#   record  : HF_DATASET_REPO_ID  SINGLE_TASK  NUM_EPISODES
#             EPISODE_TIME_S  RESET_TIME_S  RECORD_FPS  PUSH_TO_HUB
#             RECORD_EXTRA_ARGS
#   replay  : HF_DATASET_REPO_ID  EPISODE_INDEX  REPLAY_EXTRA_ARGS
#   기기유틸: ROBOT_TYPE  TELEOP_TYPE  CALIBRATE_TARGET  TELEOP_TIME_S
#   viz     : HF_DATASET_REPO_ID  EPISODE_INDEX  VIZ_MODE  VIZ_WS_PORT
#   policy-client: POLICY_SERVER_ADDRESS  POLICY_TYPE  POLICY_REPO_ID
#                  POLICY_DEVICE  CLIENT_DEVICE  TASK  ACTIONS_PER_CHUNK
#                  CHUNK_SIZE_THRESHOLD  AGGREGATE_FN_NAME  POLICY_FPS(공유)
#                  POLICY_CLIENT_EXTRA_ARGS
# =============================================================================
set -euo pipefail

# ── 하드웨어 환경 변수 기본값 ──────────────────────────────────────────────────
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM0}"
TELEOP_ID="${TELEOP_ID:-so101_teleop}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-so101_robot}"
# 활성 카메라 부분집합 (콤마 구분, 순서 보존). 예: "top,wrist" / "top,wrist,front" / "wrist"
ENABLED_CAMERAS="${ENABLED_CAMERAS:-top,wrist}"
TOP_CAM_PORT="${TOP_CAM_PORT:-/dev/video4}"
WRIST_CAM_PORT="${WRIST_CAM_PORT:-/dev/video2}"
FRONT_CAM_PORT="${FRONT_CAM_PORT:-/dev/video0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FPS="${CAM_FPS:-25}"
CAM_WARMUP_S="${CAM_WARMUP_S:-5}"
CAM_FOURCC="${CAM_FOURCC:-MJPG}"


# ── record 환경 변수 ────────────────────────────────────────────────────────
# HF_DATASET_REPO_ID: '{hf_username}/{dataset_name}' 형식 (필수 — 비어 있으면 오류)
DATASET_REPO_ID="${HF_DATASET_REPO_ID:-}"
# 에피소드당 작업 설명 (자유 문자열)
SINGLE_TASK="${SINGLE_TASK:-pick and place}"
# 수집할 에피소드 수
NUM_EPISODES="${NUM_EPISODES:-50}"
# 에피소드 1회 녹화 시간(초)
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
# 에피소드 간 환경 초기화 대기 시간(초)
RESET_TIME_S="${RESET_TIME_S:-30}"
# 데이터셋 저장 fps (teleop fps 와 독립적으로 설정 가능)
RECORD_FPS="${RECORD_FPS:-30}"
# HuggingFace Hub 업로드 여부 (true / false)
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
# 로컬 저장 루트 디렉터리 (비어 있으면 HF 캐시 폴더 사용)
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets}"
# 추가 lerobot-record 인자 (예: "--dataset.video=false --resume=true")
RECORD_EXTRA_ARGS="${RECORD_EXTRA_ARGS:-}"

# ── replay 환경 변수 ────────────────────────────────────────────────────────
# 재생할 에피소드 인덱스 (0-based)
EPISODE_INDEX="${EPISODE_INDEX:-0}"
# 추가 lerobot-replay 인자
REPLAY_EXTRA_ARGS="${REPLAY_EXTRA_ARGS:-}"

# ── 기기 유틸 공통 환경 변수 ──────────────────────────────────────────────────
# 로봇 타입 (calibrate / setup-motors / find-joint-limits 에서 사용)
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
# 텔레옵 타입 (calibrate / find-joint-limits 에서 사용)
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
# calibrate / setup-motors 대상: "robot" 또는 "teleop"
CALIBRATE_TARGET="${CALIBRATE_TARGET:-robot}"
# find-joint-limits 텔레옵 조작 시간(초)
TELEOP_TIME_S="${TELEOP_TIME_S:-30}"

# ── dataset-viz 환경 변수 ───────────────────────────────────────────────────
# 시각화 모드: local | distant (distant = WebSocket 서버 모드)
VIZ_MODE="${VIZ_MODE:-local}"
# distant 모드 WebSocket 포트
VIZ_WS_PORT="${VIZ_WS_PORT:-9087}"

# ── policy-client 환경 변수 (async inference 클라이언트) ────────────────────
# 정책 서버 (`policy-server` 서비스 또는 원격 H100 서버) 에 gRPC 로 붙어
# 관측을 보내고 액션을 받아 SO-101 follower arm 을 구동한다.
# 같은 호스트에 정책 서버가 떠 있으면 127.0.0.1:8080, 원격 inference 라면
# H100 서버 IP:포트 (예: "10.0.0.5:8080") 를 지정.
POLICY_SERVER_ADDRESS="${POLICY_SERVER_ADDRESS:-127.0.0.1:8080}"
# 정책 종류 (smolvla / pi0 / act / ...) — 서버가 클라이언트의 SendPolicyInstructions
# 를 받아 해당 정책을 로드한다.
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
# 정책 weight: HF Hub repo ID 또는 로컬 경로. 기본은 SmolVLA 베이스.
# 추론(policy-client)에서 로드할 모델 = fine-tune 결과(POLICY_REPO_ID). 미설정 시
# 베이스로 폴백(파이프라인 검증용 — SO-101 카메라 키 불일치 시 KeyError 주의).
POLICY_REPO_ID="${POLICY_REPO_ID:-lerobot/smolvla_base}"
# 정책이 실행될 디바이스 (서버 측). cuda / cpu / mps
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
# 클라이언트 측 후처리 디바이스 (보통 cpu)
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
# 자연어 task instruction (정책에 전달; 예: "pick the pen", "fold the t-shirt")
TASK="${TASK:-pick the pen}"
# 한 번에 받아오는 액션 청크 길이 (SmolVLA 기본 50)
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
# 청크 갱신 임계값 (0~1). 큐가 이 비율 미만으로 줄면 새 청크 요청.
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
# 청크 경계 부드럽게 합치는 함수 (weighted_average / latest / average ...)
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-weighted_average}"
# 컨트롤 루프 FPS. 서버와 동일 제어율이어야 하므로 POLICY_FPS 를 공유한다.
# (다르게 하려면 POLICY_CLIENT_EXTRA_ARGS=--fps=N 로 오버라이드)
POLICY_CLIENT_FPS="${POLICY_FPS:-30}"
# 추가 robot_client 인자 (예: --debug_visualize_queue_size=true)
POLICY_CLIENT_EXTRA_ARGS="${POLICY_CLIENT_EXTRA_ARGS:-}"

# ── 레거시 (teleop 전용, 하위 호환) ────────────────────────────────────────
TELEOP_EXTRA_ARGS="${TELEOP_EXTRA_ARGS:-}"

# ── 색상 출력 유틸 ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── GPU 확인 (선택적) ─────────────────────────────────────────────────────────
# lerobot 이미지의 모든 모드(teleop / record / replay / calibrate / policy-client)는
# 로컬 GPU 를 직접 사용하지 않는다. GPU 추론은 policy-server 가 담당.
# LEROBOT_SHOW_GPU_INFO=1 로 설정할 때만 nvidia-smi 를 호출해 cold start 오버헤드를 줄인다.
check_gpu_optional() {
    [[ "${LEROBOT_SHOW_GPU_INFO:-0}" != "1" ]] && return 0
    if command -v nvidia-smi &>/dev/null; then
        info "NVIDIA GPU 감지됨:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
            while IFS= read -r line; do info "  GPU: $line"; done
    else
        warn "nvidia-smi 를 찾을 수 없습니다."
    fi
}

# ── 직렬 포트 존재 확인 ───────────────────────────────────────────────────────
check_port() {
    local port="$1" label="$2"
    if [[ ! -e "$port" ]]; then
        error "${label} 포트 '${port}' 를 찾을 수 없습니다."
        error "  → PowerShell: usbipd attach --wsl --busid <BUS_ID>"
        error "  → WSL2:       ls /dev/ttyACM*"
        return 1
    fi
    if [[ ! -r "$port" || ! -w "$port" ]]; then
        warn "${label} 포트 '${port}' 읽기/쓰기 권한 없음 (--privileged 확인)"
    else
        info "${label} 직렬 포트 OK: ${port}"
    fi
}

# ── 카메라 디바이스 존재 및 권한 확인 ────────────────────────────────────────
check_camera() {
    local dev="$1" label="$2"
    if [[ ! -e "$dev" ]]; then
        warn "${label} 디바이스 '${dev}' 를 찾을 수 없습니다."
        warn "  → PowerShell: usbipd attach --wsl --busid <BUS_ID>"
        warn "  카메라 없이 텔레옵은 동작하지만 데이터 수집 시 이미지 누락됩니다."
        return 0
    fi
    if [[ ! -r "$dev" ]]; then
        warn "${label} 디바이스 '${dev}' 읽기 권한 없음 (--privileged 확인)"
    else
        info "${label} 카메라 OK: ${dev}"
    fi
}

# ── 카메라 JSON 빌더 ─────────────────────────────────────────────────────────
# ENABLED_CAMERAS 를 순회하며 lerobot --robot.cameras 인라인 dict 를 생성.
# 각 이름은 ${UPPER}_CAM_PORT 환경변수로 매핑된다 (예: wrist → WRIST_CAM_PORT).
build_cameras_json() {
    local out="" cam upper port_var port first=1
    local -a cams
    IFS=',' read -ra cams <<< "$ENABLED_CAMERAS"
    for cam in "${cams[@]}"; do
        cam="${cam// /}"; [[ -z "$cam" ]] && continue
        upper="${cam^^}"; port_var="${upper}_CAM_PORT"; port="${!port_var:-}"
        if [[ -z "$port" ]]; then
            error "ENABLED_CAMERAS 에 '${cam}' 가 있지만 ${port_var} 가 비어 있습니다."
            exit 1
        fi
        if (( first )); then
            out="{"
        else
            out+=", "
        fi
        first=0
        out+="${cam}: {type: opencv, index_or_path: ${port}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}, warmup_s: ${CAM_WARMUP_S}, fourcc: ${CAM_FOURCC}}"
    done
    if (( first )); then
        error "ENABLED_CAMERAS 가 비어 있습니다. 최소 1개 이상의 카메라를 지정하세요."
        exit 1
    fi
    out+="}"
    echo "$out"
}

# ── 활성 카메라 점검 (warn-only) ─────────────────────────────────────────────
check_enabled_cameras() {
    local cam upper port_var
    local -a cams
    IFS=',' read -ra cams <<< "$ENABLED_CAMERAS"
    for cam in "${cams[@]}"; do
        cam="${cam// /}"; [[ -z "$cam" ]] && continue
        upper="${cam^^}"; port_var="${upper}_CAM_PORT"
        check_camera "${!port_var:-}" "${upper}"
    done
}

# ── 활성 카메라 로그 ─────────────────────────────────────────────────────────
log_enabled_cameras() {
    local cam upper port_var
    local -a cams
    IFS=',' read -ra cams <<< "$ENABLED_CAMERAS"
    for cam in "${cams[@]}"; do
        cam="${cam// /}"; [[ -z "$cam" ]] && continue
        upper="${cam^^}"; port_var="${upper}_CAM_PORT"
        info "  ${cam} cam → ${!port_var:-<unset>}"
    done
}

# ── HF_DATASET_REPO_ID 존재 확인 ──────────────────────────────────────────────
check_dataset_repo_id() {
    if [[ -z "$DATASET_REPO_ID" ]]; then
        error "HF_DATASET_REPO_ID 가 설정되지 않았습니다."
        error "  → .env:           HF_DATASET_REPO_ID=your_username/dataset_name"
        error "  → docker compose: -e HF_DATASET_REPO_ID=your_username/dataset_name"
        exit 1
    fi
    info "Dataset repo ID: ${DATASET_REPO_ID}"
}

# ── LeRobot 버전 및 CLI entry point 확인 ──────────────────────────────────────
check_lerobot() {
    local ver
    # 빌드 시점에 기록한 버전 파일을 우선 읽어 Python 기동 오버헤드를 제거.
    # 파일이 없으면(구버전 이미지 등) importlib.metadata 로 폴백 — full import 보다 빠름.
    if [[ -f /opt/lerobot_version.txt ]]; then
        ver=$(cat /opt/lerobot_version.txt)
    else
        ver=$(python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))" 2>/dev/null || echo "unknown")
    fi
    info "LeRobot 버전: ${ver}"

    if ! command -v lerobot-teleoperate &>/dev/null; then
        error "lerobot CLI entry point 를 찾을 수 없습니다."
        error "  /opt/venv/bin/lerobot-* 가 존재하는지 확인하세요."
        exit 1
    fi
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  LeRobot 0.4.4 SO-101 Container"
echo "========================================================"

CMD="${1:-teleop}"

# 빠른 진입 모드(bash/python/find-*/info)는 모든 체크를 건너뜀.
# 하드웨어 모드에서는 lerobot CLI 존재만 확인(필수) + GPU 정보(선택, LEROBOT_SHOW_GPU_INFO=1).
# lerobot 이미지의 어떤 모드도 로컬 GPU 를 직접 사용하지 않으므로 GPU 체크는 기본 off.
case "$CMD" in
  bash|shell|python|find-port|find-cameras|info|edit-dataset) ;;
  *) check_gpu_optional; check_lerobot ;;
esac

case "$CMD" in

  # ────────────────────────────────────────────────────────────────────────────
  # teleop — 실시간 원격 조작
  #
  # [env var → CLI arg 매핑]
  #   ROBOT_PORT  → --robot.port
  #   ROBOT_ID    → --robot.id
  #   TELEOP_PORT    → --teleop.port
  #   TELEOP_ID      → --teleop.id
  #   WRIST_CAM_PORT  → --robot.cameras wrist index_or_path
  #   FRONT_CAM_PORT  → --robot.cameras front index_or_path
  #   TOP_CAM_PORT  → --robot.cameras top index_or_path
  #   CAM_WIDTH/HEIGHT/FPS → cameras 해상도·FPS
  #
  # [주요 CLI 인자 — TELEOP_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower   : 팔로워 로봇 타입
  #   --robot.port=<path>           : 팔로워 직렬 포트
  #   --robot.id=<str>              : 팔로워 ID (캘리브레이션 파일명)
  #   --robot.cameras=<json>        : 카메라 설정 (type/index_or_path/width/height/fps)
  #   --teleop.type=so101_leader    : 리더 텔레옵 타입
  #   --teleop.port=<path>          : 리더 직렬 포트
  #   --teleop.id=<str>             : 리더 ID
  #   --fps=<int>                   : 제어 주기 (기본 60)
  #   --teleop_time_s=<float>       : 최대 동작 시간(초, 기본 무제한)
  #   --display_data=true|false     : 카메라 영상 화면 출력 여부
  # ────────────────────────────────────────────────────────────────────────────
  teleop)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port   "$TELEOP_PORT"   "Leader Arm"
    check_port   "$ROBOT_PORT" "Follower Arm"
    check_enabled_cameras

    info "── Teleoperation 시작 ────────────────────────────"
    info "  Leader   → ID: ${TELEOP_ID}, PORT: ${TELEOP_PORT}"
    info "  Follower → ID: ${ROBOT_ID}, PORT: ${ROBOT_PORT}"
    info "  Cameras  → ${ENABLED_CAMERAS}"
    log_enabled_cameras

    lerobot-teleoperate \
      --robot.type=so101_follower \
      --robot.port=${ROBOT_PORT} \
      --robot.cameras="$(build_cameras_json)" \
      --robot.id=${ROBOT_ID} \
      --teleop.type=so101_leader \
      --teleop.port=${TELEOP_PORT} \
      --teleop.id=${TELEOP_ID} \
      ${TELEOP_EXTRA_ARGS}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # record — 텔레오퍼레이션 기반 데이터 수집
  #
  # [env var → CLI arg 매핑]
  #   ROBOT_PORT/ID        → --robot.port / --robot.id
  #   TELEOP_PORT/ID          → --teleop.port / --teleop.id
  #   WRIST/FRONT/TOP_CAM_PORT     → --robot.cameras (teleop 와 동일)
  #   CAM_WIDTH/HEIGHT/FPS    → cameras 해상도·FPS
  #   HF_DATASET_REPO_ID      → --dataset.repo_id        (필수)
  #   SINGLE_TASK             → --dataset.single_task
  #   NUM_EPISODES            → --dataset.num_episodes
  #   EPISODE_TIME_S          → --dataset.episode_time_s
  #   RESET_TIME_S            → --dataset.reset_time_s
  #   RECORD_FPS              → --dataset.fps
  #   PUSH_TO_HUB             → --dataset.push_to_hub
  #   DATASET_ROOT            → --dataset.root
  #
  # [주요 CLI 인자 전체 — RECORD_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower
  #   --robot.port=<path>
  #   --robot.id=<str>
  #   --robot.cameras=<json>
  #   --teleop.type=so101_leader
  #   --teleop.port=<path>
  #   --teleop.id=<str>
  #   --dataset.repo_id=<str>           : '{hf_user}/{name}' (필수)
  #   --dataset.single_task=<str>       : 에피소드 작업 설명
  #   --dataset.root=<path>             : 로컬 저장 루트 (기본: HF 캐시)
  #   --dataset.fps=<int>               : 저장 fps (기본 30)
  #   --dataset.episode_time_s=<int>    : 에피소드 녹화 시간(초, 기본 60)
  #   --dataset.reset_time_s=<int>      : 초기화 대기 시간(초, 기본 60)
  #   --dataset.num_episodes=<int>      : 수집할 에피소드 수 (기본 50)
  #   --dataset.video=true|false        : 비디오 인코딩 여부 (기본 true)
  #   --dataset.push_to_hub=true|false  : HF Hub 업로드 여부 (기본 true)
  #   --dataset.private=true|false      : Hub 비공개 업로드 (기본 false)
  #   --dataset.tags=<list>             : Hub 태그
  #   --dataset.num_image_writer_processes=<int>   : 이미지 저장 프로세스 수 (기본 0=스레드)
  #   --dataset.num_image_writer_threads_per_camera=<int> : 카메라당 스레드 수 (기본 4)
  #   --dataset.video_encoding_batch_size=<int>    : 배치 인코딩 단위 (기본 1)
  #   --dataset.rename_map=<dict>       : 관측 키 이름 변경 매핑
  #   --display_data=true|false         : 카메라 영상 화면 출력
  #   --play_sounds=true|false          : 음성 이벤트 알림 (기본 true)
  #   --resume=true|false               : 기존 데이터셋에 이어서 수집 (기본 false)
  # ────────────────────────────────────────────────────────────────────────────
  record)
    shift || true   # "record" 제거 → 나머지 인자("$@")를 lerobot-record 에 그대로 forward
    info "── 장치 점검 ─────────────────────────────────────"
    check_port   "$TELEOP_PORT"   "Leader Arm"
    check_port   "$ROBOT_PORT" "Follower Arm"
    check_enabled_cameras
    check_dataset_repo_id

    info "── Record 시작 ───────────────────────────────────"
    info "  Leader   → ID: ${TELEOP_ID}, PORT: ${TELEOP_PORT}"
    info "  Follower → ID: ${ROBOT_ID}, PORT: ${ROBOT_PORT}"
    info "  Cameras  → ${ENABLED_CAMERAS}"
    info "  Dataset  → ${DATASET_REPO_ID} (${NUM_EPISODES} episodes, ${EPISODE_TIME_S}s each)"
    info "  Task     → ${SINGLE_TASK}"

    lerobot-record \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${ROBOT_PORT} \
      --robot.id=${ROBOT_ID} \
      --robot.cameras="$(build_cameras_json)" \
      --teleop.type=${TELEOP_TYPE} \
      --teleop.port=${TELEOP_PORT} \
      --teleop.id=${TELEOP_ID} \
      --dataset.repo_id=${DATASET_REPO_ID} \
      --dataset.single_task="${SINGLE_TASK}" \
      --dataset.root=${DATASET_ROOT} \
      --dataset.fps=${RECORD_FPS} \
      --dataset.episode_time_s=${EPISODE_TIME_S} \
      --dataset.reset_time_s=${RESET_TIME_S} \
      --dataset.num_episodes=${NUM_EPISODES} \
      --dataset.push_to_hub=${PUSH_TO_HUB} \
      ${RECORD_EXTRA_ARGS} \
      "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # replay — 녹화된 에피소드를 로봇에서 재실행
  #
  # [env var → CLI arg 매핑]
  #   ROBOT_PORT/ID    → --robot.port / --robot.id
  #   HF_DATASET_REPO_ID  → --dataset.repo_id   (필수)
  #   EPISODE_INDEX       → --dataset.episode
  #   RECORD_FPS          → --dataset.fps
  #   DATASET_ROOT        → --dataset.root
  #
  # [주요 CLI 인자 전체 — REPLAY_EXTRA_ARGS 로 추가 전달 가능]
  #   --robot.type=so101_follower
  #   --robot.port=<path>
  #   --robot.id=<str>
  #   --dataset.repo_id=<str>    : '{hf_user}/{name}' (필수)
  #   --dataset.episode=<int>    : 재생할 에피소드 인덱스 (0-based, 기본 0)
  #   --dataset.root=<path>      : 로컬 저장 루트 (기본: HF 캐시)
  #   --dataset.fps=<int>        : 재생 fps (기본 30)
  #   --play_sounds=true|false   : 음성 이벤트 알림 (기본 true)
  # ────────────────────────────────────────────────────────────────────────────
  replay)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port "$ROBOT_PORT" "Follower Arm"
    check_dataset_repo_id

    info "── Replay 시작 ───────────────────────────────────"
    info "  Follower → ID: ${ROBOT_ID}, PORT: ${ROBOT_PORT}"
    info "  Dataset  → ${DATASET_REPO_ID}, episode=${EPISODE_INDEX}"

    lerobot-replay \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${ROBOT_PORT} \
      --robot.id=${ROBOT_ID} \
      --dataset.repo_id=${DATASET_REPO_ID} \
      --dataset.episode=${EPISODE_INDEX} \
      --dataset.root=${DATASET_ROOT} \
      --dataset.fps=${RECORD_FPS} \
      ${REPLAY_EXTRA_ARGS}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # calibrate — 로봇 또는 텔레옵 보정
  #
  # [env var → 동작]
  #   CALIBRATE_TARGET=robot  (기본) → 팔로워 암 보정
  #   CALIBRATE_TARGET=teleop         → 리더 텔레옵 보정
  #
  # [주요 CLI 인자 전체]
  #   [robot 타깃]
  #     --robot.type=<str>     : 로봇 타입 (so101_follower, so100_follower, ...)
  #     --robot.port=<path>    : 직렬 포트
  #     --robot.id=<str>       : 캘리브레이션 파일 ID
  #   [teleop 타깃]
  #     --teleop.type=<str>    : 텔레옵 타입 (so101_leader, so100_leader, ...)
  #     --teleop.port=<path>   : 직렬 포트
  #     --teleop.id=<str>      : 캘리브레이션 파일 ID
  #
  # ※ robot 과 teleop 중 하나만 지정해야 함 (동시 지정 불가)
  # ────────────────────────────────────────────────────────────────────────────
  calibrate)
    if [[ "$CALIBRATE_TARGET" == "teleop" ]]; then
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$TELEOP_PORT" "Leader Arm (teleop)"
        info "── Calibrate (teleop) 시작 ─────────────────────"
        info "  Teleop → ID: ${TELEOP_ID}, PORT: ${TELEOP_PORT}"
        lerobot-calibrate \
          --teleop.type=${TELEOP_TYPE} \
          --teleop.port=${TELEOP_PORT} \
          --teleop.id=${TELEOP_ID}
    else
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$ROBOT_PORT" "Follower Arm (robot)"
        info "── Calibrate (robot) 시작 ──────────────────────"
        info "  Robot → ID: ${ROBOT_ID}, PORT: ${ROBOT_PORT}"
        lerobot-calibrate \
          --robot.type=${ROBOT_TYPE} \
          --robot.port=${ROBOT_PORT} \
          --robot.id=${ROBOT_ID}
    fi
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # setup-motors — Feetech 모터 ID 및 Baud Rate 초기 설정
  #
  # [env var → 동작]
  #   CALIBRATE_TARGET=robot  (기본) → 팔로워 암 모터 설정
  #   CALIBRATE_TARGET=teleop         → 리더 텔레옵 모터 설정
  #
  # [주요 CLI 인자 전체]
  #   [robot 타깃]
  #     --robot.type=<str>    : 로봇 타입
  #     --robot.port=<path>   : 직렬 포트
  #   [teleop 타깃]
  #     --teleop.type=<str>   : 텔레옵 타입
  #     --teleop.port=<path>  : 직렬 포트
  #
  # ※ robot 과 teleop 중 하나만 지정해야 함 (동시 지정 불가)
  # ────────────────────────────────────────────────────────────────────────────
  setup-motors)
    if [[ "$CALIBRATE_TARGET" == "teleop" ]]; then
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$TELEOP_PORT" "Leader Arm (teleop)"
        info "── Setup Motors (teleop) 시작 ──────────────────"
        lerobot-setup-motors \
          --teleop.type=${TELEOP_TYPE} \
          --teleop.port=${TELEOP_PORT}
    else
        info "── 장치 점검 ─────────────────────────────────────"
        check_port "$ROBOT_PORT" "Follower Arm (robot)"
        info "── Setup Motors (robot) 시작 ───────────────────"
        lerobot-setup-motors \
          --robot.type=${ROBOT_TYPE} \
          --robot.port=${ROBOT_PORT}
    fi
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-joint-limits — 텔레오퍼레이션으로 관절 가동 범위 탐색
  #
  # robot + teleop 양쪽 모두 필요 (calibrate 와 달리 둘 다 필수)
  #
  # [env var → CLI arg 매핑]
  #   ROBOT_PORT/ID  → --robot.port / --robot.id
  #   TELEOP_PORT/ID    → --teleop.port / --teleop.id
  #   TELEOP_TIME_S     → --teleop_time_s
  #
  # [주요 CLI 인자 전체]
  #   --robot.type=<str>         : 로봇 타입
  #   --robot.port=<path>        : 팔로워 직렬 포트
  #   --robot.id=<str>           : 팔로워 ID
  #   --teleop.type=<str>        : 텔레옵 타입
  #   --teleop.port=<path>       : 리더 직렬 포트
  #   --teleop.id=<str>          : 리더 ID
  #   --teleop_time_s=<float>    : 탐색 제한 시간(초, 기본 30)
  #   --display_data=true|false  : 카메라 영상 화면 출력
  # ────────────────────────────────────────────────────────────────────────────
  find-joint-limits)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port "$TELEOP_PORT"   "Leader Arm"
    check_port "$ROBOT_PORT" "Follower Arm"

    info "── Find Joint Limits 시작 ────────────────────────"
    info "  Leader   → ID: ${TELEOP_ID}, PORT: ${TELEOP_PORT}"
    info "  Follower → ID: ${ROBOT_ID}, PORT: ${ROBOT_PORT}"
    info "  시간: ${TELEOP_TIME_S}s"

    lerobot-find-joint-limits \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${ROBOT_PORT} \
      --robot.id=${ROBOT_ID} \
      --teleop.type=${TELEOP_TYPE} \
      --teleop.port=${TELEOP_PORT} \
      --teleop.id=${TELEOP_ID} \
      --teleop_time_s=${TELEOP_TIME_S}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-cameras — 시스템에 연결된 카메라 검색 및 캡처 확인
  #
  # 하드웨어 체크 없음 (자동 스캔)
  #
  # [주요 CLI 인자 전체]
  #   [위치 인자]
  #   opencv|realsense            : 카메라 타입 지정 (생략 시 전체 스캔)
  #   [옵션]
  #   --output-dir=<path>         : 캡처 이미지 저장 경로 (기본: outputs/captured_images)
  #   --record-time-s=<float>     : 캡처 시도 시간(초, 기본 6.0)
  #
  # 예시:
  #   docker compose run --rm teleop find-cameras
  #   docker compose run --rm teleop find-cameras opencv
  #   docker compose run --rm teleop find-cameras realsense --output-dir=/workspace/data/cams
  # ────────────────────────────────────────────────────────────────────────────
  find-cameras)
    info "── Find Cameras 시작 ─────────────────────────────"
    shift
    exec lerobot-find-cameras "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # find-port — MotorsBus USB 포트 자동 감지
  #
  # 하드웨어 체크 없음; 사용자 인터랙션 필요
  # (USB 케이블 분리 후 Enter 키 입력 안내)
  #
  # [CLI 인자 없음 — 인터랙티브 전용]
  #
  # 예시:
  #   docker compose run --rm teleop find-port
  # ────────────────────────────────────────────────────────────────────────────
  find-port)
    info "── Find Port 시작 ────────────────────────────────"
    exec lerobot-find-port
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # dataset-viz — 데이터셋 시각화 (Rerun 기반)
  #
  # [env var → CLI arg 매핑]
  #   HF_DATASET_REPO_ID  → --repo-id        (필수)
  #   EPISODE_INDEX       → --episode-index
  #   VIZ_MODE            → --mode           (local | distant)
  #   VIZ_WS_PORT         → --ws-port
  #   DATASET_ROOT        → --root
  #
  # [주요 CLI 인자 전체]
  #   --repo-id=<str>          : HF Hub 데이터셋 ID (필수)
  #   --episode-index=<int>    : 시각화할 에피소드 인덱스 (필수)
  #   --root=<path>            : 로컬 저장 루트 (생략 시 HF 캐시)
  #   --output-dir=<path>      : .rrd 파일 저장 경로 (--save 1 시 사용)
  #   --batch-size=<int>       : DataLoader 배치 크기 (기본 32)
  #   --num-workers=<int>      : DataLoader 프로세스 수 (기본 4)
  #   --mode=local|distant     : 로컬 뷰어 or 원격 WebSocket 서버 (기본 local)
  #   --web-port=<int>         : Rerun 웹 포트 (distant 모드, 기본 9090)
  #   --ws-port=<int>          : Rerun WebSocket 포트 (distant 모드, 기본 9087)
  #   --save=0|1               : 1이면 .rrd 파일로 저장 (뷰어 미실행)
  #   --tolerance-s=<float>    : 타임스탬프 허용 오차(초, 기본 1e-4)
  #
  # distant 모드 이용법:
  #   서버 쪽 : VIZ_MODE=distant 로 실행
  #   클라이언트: rerun ws://HOST:9087
  # ────────────────────────────────────────────────────────────────────────────
  dataset-viz)
    check_dataset_repo_id

    info "── Dataset Viz 시작 ──────────────────────────────"
    info "  Dataset → ${DATASET_REPO_ID}, episode=${EPISODE_INDEX}"
    info "  Mode    → ${VIZ_MODE}"

    lerobot-dataset-viz \
      --repo-id=${DATASET_REPO_ID} \
      --episode-index=${EPISODE_INDEX} \
      --root=${DATASET_ROOT} \
      --mode=${VIZ_MODE} \
      --ws-port=${VIZ_WS_PORT}
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # policy-client — Async inference 클라이언트 (실제 SO-101 follower 구동)
  #
  # `lerobot.async_inference.robot_client` 를 띄워 정책 서버
  # (`policy-server` 또는 원격 H100) 의 gRPC :PORT 에 접속한다.
  # 클라이언트가 SendPolicyInstructions RPC 로 policy_type / pretrained_name_or_path
  # / policy_device 를 전달 → 서버가 해당 정책을 로드 → 클라이언트가 카메라/state
  # 관측을 송신 → 서버가 액션 청크를 비동기 반환 → 클라이언트가 follower 에 적용.
  #
  # [env var → CLI arg 매핑]
  #   POLICY_SERVER_ADDRESS    → --server_address          (예: 127.0.0.1:8080)
  #   POLICY_TYPE              → --policy_type             (smolvla 등)
  #   POLICY_REPO_ID           → --pretrained_name_or_path (fine-tune 결과 모델)
  #   POLICY_DEVICE            → --policy_device           (서버 측, cuda)
  #   CLIENT_DEVICE            → --client_device           (클라이언트 측, cpu)
  #   TASK                     → --task                    ("pick the pen" 등)
  #   ACTIONS_PER_CHUNK        → --actions_per_chunk       (SmolVLA 기본 50)
  #   CHUNK_SIZE_THRESHOLD     → --chunk_size_threshold    (기본 0.5)
  #   AGGREGATE_FN_NAME        → --aggregate_fn_name       (weighted_average 등)
  #   POLICY_FPS               → --fps                     (제어 FPS, 서버와 공유, 기본 30)
  #   ROBOT_TYPE/PORT/ID       → --robot.type/.port/.id
  #   WRIST_CAM_PORT/FRONT...  → --robot.cameras           (teleop 와 동일 매핑)
  #
  # 예시:
  #   # 같은 호스트에 정책 서버를 띄워 둔 뒤
  #   docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
  #   # 클라이언트로 붙어 follower 구동
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot policy-client
  #
  #   # 원격 H100 서버에 접속하려면
  #   POLICY_SERVER_ADDRESS=10.0.0.5:8080 \
  #     docker compose --env-file .env -f docker/docker-compose.yaml run --rm lerobot policy-client
  # ────────────────────────────────────────────────────────────────────────────
  policy-client)
    info "── 장치 점검 ─────────────────────────────────────"
    check_port   "$ROBOT_PORT" "Follower Arm"
    check_enabled_cameras

    info "── Policy Client 시작 (gRPC) ─────────────────────"
    info "  Server  → ${POLICY_SERVER_ADDRESS}"
    info "  Policy  → ${POLICY_TYPE} @ ${POLICY_REPO_ID} (device=${POLICY_DEVICE})"
    info "  Robot   → ${ROBOT_TYPE} ID=${ROBOT_ID} PORT=${ROBOT_PORT}"
    info "  Task    → ${TASK}"
    info "  Cameras → ${ENABLED_CAMERAS}"
    info "  Chunks  → ${ACTIONS_PER_CHUNK} actions, threshold=${CHUNK_SIZE_THRESHOLD}, fps=${POLICY_CLIENT_FPS}"

    shift || true
    # NOTE: `python -m lerobot.async_inference.robot_client` 대신 shim 을 거쳐
    # robot config 모듈을 선행 import 한다 (huggingface/lerobot#3078 워크어라운드).
    # lerobot 이 #3081 픽스를 포함한 버전으로 올라가면 다시 -m 호출로 되돌릴 것.
    exec python /usr/local/bin/policy-client-shim.py \
      --server_address=${POLICY_SERVER_ADDRESS} \
      --policy_type=${POLICY_TYPE} \
      --pretrained_name_or_path=${POLICY_REPO_ID} \
      --policy_device=${POLICY_DEVICE} \
      --client_device=${CLIENT_DEVICE} \
      --task="${TASK}" \
      --actions_per_chunk=${ACTIONS_PER_CHUNK} \
      --chunk_size_threshold=${CHUNK_SIZE_THRESHOLD} \
      --aggregate_fn_name=${AGGREGATE_FN_NAME} \
      --fps=${POLICY_CLIENT_FPS} \
      --robot.type=${ROBOT_TYPE} \
      --robot.port=${ROBOT_PORT} \
      --robot.id=${ROBOT_ID} \
      --robot.cameras="$(build_cameras_json)" \
      ${POLICY_CLIENT_EXTRA_ARGS} \
      "$@"
    ;;

  # train / eval ────────────────────────────────────────────────────────────
  # 이 두 모드는 policy-server 서비스 (Dockerfile.policy + policy-entrypoint.sh)
  # 로 이동되었다. 이유: SmolVLA 등 정책 학습 시 transformers / accelerate /
  # num2words 가 필요하나 본 이미지는 teleop + async 그룹만 설치 (smolvla 그룹 미설치).
  #
  # 사용 예 (policy-server 컨테이너에서 실행):
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     policy-server train --dataset.repo_id=... --policy.path=lerobot/smolvla_base ...
  #   docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  #     policy-server eval --policy.path=... --env.type=...
  # ────────────────────────────────────────────────────────────────────────────

  # ────────────────────────────────────────────────────────────────────────────
  # edit-dataset — 데이터셋 편집 (모든 인자를 lerobot-edit-dataset 에 완전 위임)
  #
  # 하드웨어 없음
  #
  # [주요 CLI 인자 전체]
  #   --repo_id=<str>              : 편집 대상 데이터셋 Hub ID (필수)
  #   --new_repo_id=<str>          : 결과 데이터셋 Hub ID (생략 시 원본 덮어쓰기)
  #   --root=<path>                : 로컬 저장 루트
  #   --push_to_hub=true|false     : 결과를 HF Hub 에 업로드 (기본 false)
  #   --operation.type=<str>       : 작업 종류 (아래 참조)
  #
  #   [delete_episodes 작업]
  #     --operation.type=delete_episodes
  #     --operation.episode_indices=[0,1,3]  : 삭제할 에피소드 인덱스 목록
  #
  #   [split 작업]
  #     --operation.type=split
  #     --operation.splits='{"train":0.8,"val":0.2}'  : 비율 or 인덱스 리스트로 분할
  #
  #   [merge 작업]
  #     --operation.type=merge
  #     --operation.repo_ids='["user/ds1","user/ds2"]'  : 병합할 데이터셋 ID 목록
  #
  #   [remove_feature 작업]
  #     --operation.type=remove_feature
  #     --operation.feature_names='["observation.images.wrist"]'  : 제거할 특성 키 목록
  #
  # 예시:
  #   docker compose run --rm teleop edit-dataset \
  #     --repo_id=my_user/so101_pick \
  #     --operation.type=delete_episodes \
  #     --operation.episode_indices=[0,5,9]
  # ────────────────────────────────────────────────────────────────────────────
  edit-dataset)
    info "── Edit Dataset 시작 ─────────────────────────────"
    shift
    exec lerobot-edit-dataset "$@"
    ;;

  # ────────────────────────────────────────────────────────────────────────────
  # info — LeRobot / Python / 시스템 정보 출력
  #
  # [CLI 인자 없음]
  #
  # 예시:
  #   docker compose run --rm teleop info
  # ────────────────────────────────────────────────────────────────────────────
  info)
    exec lerobot-info
    ;;

  # ── 인터랙티브 쉘 / 직접 실행 ────────────────────────────────────────────────
  bash|shell)
    info "인터랙티브 쉘로 진입합니다."
    exec /bin/bash
    ;;

  python)
    shift
    exec python "$@"
    ;;

  *)
    exec "$@"
    ;;

esac
