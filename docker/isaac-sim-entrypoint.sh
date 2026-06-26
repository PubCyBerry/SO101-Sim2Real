#!/usr/bin/env bash
# =============================================================================
# isaac-sim-entrypoint.sh — Isaac Sim 5.1 + ROS 2 Bridge 서비스 진입점
#
# 역할: cube_desk 시뮬레이션 + ROS 2 토픽 publish/subscribe
#
# ■ 실행 모드 (CMD 첫 번째 인자)
#   bridge   : (기본) scripts/inference/run_cube_desk_ros_bridge.py 실행
#              → Isaac Sim + ROS 2 bridge + WebRTC livestream
#   datagen  : scripts/datagen/record_state_machine.py 실행
#              → state-machine 기반 LeRobot v3 데이터 생성
#   bash     : 인터랙티브 Bash 쉘 (디버깅)
#   python   : Python 직접 실행 (diagnostic script)
#   <기타>   : 명령 그대로 exec (고급)
#
# ■ 환경 변수 요약 (docker-compose.yaml ↔ .env 에서 주입)
#   공통  : LIVESTREAM (1=활성), PUBLIC_IP (원격 WebRTC relay용)
#   bridge: NUM_CUBES (1~4), BRIDGE_EXTRA_ARGS (--eval, --headless 등)
#   bridge: SO101_BRIDGE_SCRIPT (스크립트 경로, /workspace/scripts/inference/... 기본)
#   datagen: DATAGEN_TASK (환경명, SimToReal-SO101-PickCube-v0 기본)
#   datagen: NUM_DEMOS (데모 수, 50 기본), DATAGEN_EXTRA_ARGS (추가 인자)
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

# datagen 모드 변수
DATAGEN_TASK="${DATAGEN_TASK:-SimToReal-SO101-PickCube-v0}"
NUM_DEMOS="${NUM_DEMOS:-50}"
DATAGEN_EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"

# 공통 변수 (livestream)
LIVESTREAM="${LIVESTREAM:-1}"
PUBLIC_IP="${PUBLIC_IP:-}"

# ROS 2 DDS 설정 (compose 에서 주입되었지만 명시적 재확인)
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

# isaacsim 번들 ROS 2 Jazzy lib — librmw_implementation.so 의 libament_index_cpp.so
# 의존성 해소. 동적 링커가 프로세스 시작 시 읽으므로 python 안에서 설정 불가 →
# 여기서 export 해야 isaacsim.ros2.bridge 확장이 로드된다(미설정 시 "ROS2 Bridge startup failed").
_ROS_BRIDGE_LIB="/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib"
export LD_LIBRARY_PATH="${_ROS_BRIDGE_LIB}:${LD_LIBRARY_PATH:-}"

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

    # isaaclab.sh -p: USD/pxr·kit 환경 셋업 후 번들 isaacsim python 으로 스크립트 실행
    # (사용자 검증 패턴: ./isaaclab.sh -p <script> --headless --livestream 2).
    exec "${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}" -p \
      "${SO101_BRIDGE_SCRIPT}" \
      --livestream 2 \
      --num_cubes "${NUM_CUBES}" \
      ${BRIDGE_EXTRA_ARGS}
    ;;

  datagen)
    # ■ State Machine 기반 LeRobot v3 데이터 생성 모드
    #   cube_desk 시뮬 + 결정적 state machine 정책
    #   → 성공 에피소드만 LeRobot v3 형식으로 기록
    #
    # 구성:
    #   - --task DATAGEN_TASK: 환경명 (기본: SimToReal-SO101-PickCube-v0)
    #   - --num_demos NUM_DEMOS: 생성할 데모 에피소드 수 (기본: 50)
    #   - 추가 인자: ${DATAGEN_EXTRA_ARGS} (e.g., --headless, --livestream 0)

    exec "${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}" -p \
      /workspace/scripts/datagen/record_state_machine.py \
      --task "${DATAGEN_TASK}" \
      --num_demos "${NUM_DEMOS}" \
      ${DATAGEN_EXTRA_ARGS}
    ;;

  bash|shell)
    # ■ 인터랙티브 Bash 쉘 (디버깅·탐색용)
    # ROS 2 + Isaac Sim 환경이 설정된 상태에서 셸 접속
    exec /bin/bash
    ;;

  python)
    # ■ Python 직접 실행 (isaaclab.sh -p, USD/kit 환경 셋업됨)
    # 예: docker compose run isaac-sim python -c "import isaaclab; print(isaaclab.__version__)"
    # 예: docker compose run isaac-sim python /workspace/scripts/<diagnostic>.py
    shift  # 첫 인자(python) 제거
    exec "${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}" -p "$@"
    ;;

  *)
    # ■ 기타: 명령 그대로 exec (고급 사용)
    # 예: docker compose run isaac-sim <arbitrary-command>
    exec "$@"
    ;;
esac
