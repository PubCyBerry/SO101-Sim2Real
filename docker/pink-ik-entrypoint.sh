#!/usr/bin/env bash
# pink-ik 컨테이너 진입점 — ROS source → pink_ik_bridge_node 실행.
#
# DDS: bridge(host, isaac-sim) 와 동일 fastrtps + UDPv4 (cross-UID /dev/shm 충돌 회피).
# -u 제외: ROS setup.bash 가 미설정 변수를 참조해 set -u 와 충돌.
set -eo pipefail

source /opt/ros/jazzy/setup.bash

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

# so101_contract(SO101_JOINT_ORDER) 단일 소스 — repo 마운트(..:/workspace).
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"

NODE=/workspace/scripts/datagen/pink_ik_bridge_node.py

# CMD 인자가 있으면 그대로 실행(예: --self-check / bash). 없으면 노드 기본 구동.
# PINK_ARGS(.env·compose)로 노드 플래그 주입(예: "--loop --hover 0.12").
if [[ "$#" -gt 0 ]]; then
  exec "$@"
else
  exec python3 "${NODE}" ${PINK_ARGS:-}
fi
