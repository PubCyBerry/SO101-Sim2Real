#!/usr/bin/env bash
# =============================================================================
# isaac-sim-entrypoint.sh — Isaac Sim 5.1 + ROS 2 Bridge 서비스 진입점
#
# 역할: cube_desk 시뮬레이션 + ROS 2 토픽 publish/subscribe
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   bridge   : (기본) scripts/inference/run_cube_desk_ros_bridge.py 실행
#              → Isaac Sim + ROS 2 bridge + WebRTC livestream
#   bash     : 인터랙티브 Bash 쉘 (디버깅)
#   python   : Python 직접 실행 (diagnostic script)
#   <기타>   : 명령 그대로 exec (고급)
#
# ■ 환경 변수 요약 (docker-compose.yaml ↔ .env 에서 주입)
#   공통  : LIVESTREAM (1=활성), PUBLIC_IP (원격 WebRTC relay용)
#   bridge: NUM_CUBES (1~4), BRIDGE_EXTRA_ARGS (--eval, --headless 등)
#   bridge: SO101_BRIDGE_SCRIPT (스크립트 경로, /workspace/scripts/inference/... 기본)
#
# ■ 주의사항
#   - ROS 2 bridge 는 DDS 환경변수에 영향을 받으므로 compose 에서 통일
#     (host ↔ container cross-UID /dev/shm 충돌 회피: fastrtps + UDPv4)
#   - WebRTC livestream(--livestream 2) 는 headless 모드에서도 작동 (GUI 없이 native 스트림)
#   - PUBLIC_IP 는 remote client 가 relay 서버를 통해 접속할 때 필요 (LAN IP 아님)
#
# =============================================================================
set -euo pipefail

# ── 환경 변수 기본값 ────────────────────────────────────────────────────────
# bridge 모드 변수
SO101_BRIDGE_SCRIPT="${SO101_BRIDGE_SCRIPT:-/workspace/scripts/inference/run_cube_desk_ros_bridge.py}"
NUM_CUBES="${NUM_CUBES:-4}"
BRIDGE_EXTRA_ARGS="${BRIDGE_EXTRA_ARGS:-}"

# 공통 변수 (livestream)
LIVESTREAM="${LIVESTREAM:-1}"
PUBLIC_IP="${PUBLIC_IP:-}"

# ROS 2 DDS 설정 (compose 에서 주입되었지만 명시적 재확인)
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

# ── 실행 모드 분기 ────────────────────────────────────────────────────────
MODE="${1:-bridge}"

case "${MODE}" in
  bridge)
    # ■ Isaac Sim + ROS 2 Bridge 메인 모드
    #   cube_desk scene load → 로봇·물체 시뮬 → ROS 2 토픽 publish/subscribe
    #   WebRTC livestream :49100 (native, GUI 창 필요 없음)
    #
    # 구성:
    #   - --livestream 2: native WebRTC streaming (포트 49100)
    #   - --num_cubes N: 1~4개 큐브 활성화 (선택 시나리오)
    #   - 추가 인자: ${BRIDGE_EXTRA_ARGS} (e.g., --eval 10 --headless)
    #   - PUBLIC_IP: remote client 용 relay 공개 IP (compose env 에서 설정)

    # PUBLIC_IP 가 지정되면 livestream 환경변수로 주입
    # (bridge 스크립트가 isaac.app.AppLauncher 의 livestream 확장을 통해 읽음)
    if [[ -n "${PUBLIC_IP}" ]]; then
      export PUBLIC_IP
    fi

    # /isaac-sim/python.sh: Isaac Sim 번들 Python 인터프리터
    # run_cube_desk_ros_bridge.py: 실제 시뮬+bridge 로직
    exec /isaac-sim/python.sh \
      "${SO101_BRIDGE_SCRIPT}" \
      --livestream 2 \
      --num_cubes "${NUM_CUBES}" \
      ${BRIDGE_EXTRA_ARGS}
    ;;

  bash|shell)
    # ■ 인터랙티브 Bash 쉘 (디버깅·탐색용)
    # ROS 2 + Isaac Sim 환경이 설정된 상태에서 셸 접속
    exec /bin/bash
    ;;

  python)
    # ■ Python 직접 실행 (위치 인자로 스크립트·모듈 명세)
    # 예: docker compose run isaac-sim python -c "import isaaclab; print(isaaclab.__version__)"
    # 예: docker compose run isaac-sim python diagnostic.py
    shift  # 첫 인자(python) 제거
    exec /isaac-sim/python.sh "$@"
    ;;

  *)
    # ■ 기타: 명령 그대로 exec (고급 사용)
    # 예: docker compose run isaac-sim <arbitrary-command>
    exec "$@"
    ;;
esac
