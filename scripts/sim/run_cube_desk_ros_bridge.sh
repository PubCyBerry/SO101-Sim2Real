#!/usr/bin/env bash
# PATH E — cube_desk Isaac Sim ROS 2 bridge 런처 (B안, 순수 isaacsim.core).
#
# run_cube_desk_ros_bridge.py 를 띄우기 전에 필수 환경변수를 export 한다:
#   LD_LIBRARY_PATH  — isaacsim 번들 ROS 2 lib(jazzy/lib). 호스트에 ROS 2 가 없으므로 bridge 는
#                      이 번들 lib 를 dlopen 한다. 동적 링커가 프로세스 시작 시 읽으므로 python
#                      안에서는 설정 불가 → 여기서 export 해야 librmw_implementation.so 의
#                      libament_index_cpp.so 의존성이 해소된다.
#   RMW_IMPLEMENTATION / FASTDDS_BUILTIN_TRANSPORTS — UDPv4 강제. host(bridge, 일반 유저)↔
#                      container(ROS 스택, root) 의 cross-UID /dev/shm fastrtps 세그먼트 공유
#                      실패를 우회한다. (.py 도 setdefault 하지만 명시 export 로 단일 진실원 유지.)
#
# 사용: scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1
#   인자는 그대로 run_cube_desk_ros_bridge.py 로 전달된다(--num_cubes / --dr / --seed 등).
set -euo pipefail

# 레포 루트(이 스크립트는 scripts/sim/ 아래) 와 isaac venv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_SITE="${REPO_ROOT}/.venv/lib/python3.11/site-packages"
BUNDLED_ROS_LIB="${VENV_SITE}/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"

if [[ ! -d "${BUNDLED_ROS_LIB}" ]]; then
  echo "[run_bridge] 번들 ROS 2 lib 없음: ${BUNDLED_ROS_LIB}" >&2
  echo "[run_bridge] 'uv sync --group isaac' 로 isaacsim 설치 확인" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${BUNDLED_ROS_LIB}:${LD_LIBRARY_PATH:-}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

cd "${REPO_ROOT}"
exec uv run --group isaac python scripts/sim/run_cube_desk_ros_bridge.py "$@"
