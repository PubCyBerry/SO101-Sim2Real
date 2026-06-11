#!/usr/bin/env bash
# vla-ros 컨테이너 진입점 — ROS source → so101_vla_policy colcon build → launch.
#
# DDS: host(bridge, 일반 유저) ↔ container(root) cross-UID /dev/shm 충돌 회피 위해
# PATH E 와 동일하게 fastrtps + UDPv4 로 통일한다.
set -euo pipefail

source /opt/ros/jazzy/setup.bash

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

WS=/workspace/ros2_ws
cd "${WS}"

# so101_vla_policy 만 빌드(의존 메시지 패키지는 apt 제공). 이미 빌드돼 있으면 빠르게 통과.
colcon build --symlink-install --packages-select so101_vla_policy
source "${WS}/install/setup.bash"

# CMD 인자를 그대로 ros2 launch 에 넘긴다(없으면 기본 launch).
if [[ "$#" -eq 0 ]]; then
  exec ros2 launch so101_vla_policy vla_policy.launch.py
else
  exec "$@"
fi
