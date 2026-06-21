#!/usr/bin/env bash
set -euo pipefail

# ROS setup scripts는 내부에서 아직 정의되지 않은 변수를 참조할 수 있다.
set +u
source /opt/ros/jazzy/setup.bash
source /opt/so101_ros/install/setup.bash
set -u

mode="${1:-vla-server}"
shift || true

case "$mode" in
  vla-server)
    exec ros2 run so101_vla_runtime vla_server \
      --manifest "${VLA_RUNTIME_MANIFEST:?VLA_RUNTIME_MANIFEST is required}" \
      --contract "${VLA_CONTRACT_PATH:?VLA_CONTRACT_PATH is required}" \
      --calibration "${VLA_CALIBRATION_PATH:?VLA_CALIBRATION_PATH is required}" \
      --runtime-config "${VLA_RUNTIME_CONFIG_PATH:?VLA_RUNTIME_CONFIG_PATH is required}" \
      --pixi-lock "${VLA_PIXI_LOCK_PATH:?VLA_PIXI_LOCK_PATH is required}" \
      "$@"
    ;;
  zenoh-router)
    exec ros2 run rmw_zenoh_cpp rmw_zenohd "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  python)
    exec python "$@"
    ;;
  *)
    exec "$mode" "$@"
    ;;
esac
