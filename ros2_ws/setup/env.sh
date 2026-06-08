#!/usr/bin/env bash
# SO-101 ROS 2 환경 설정 (source 전용)
#
# 사용:  source <repo>/ros2_ws/setup/env.sh
#
# 주의: WSL2 lo 인터페이스에 MULTICAST 플래그가 없어 기본 DDS discovery 가 실패한다.
# 이를 우회하기 위해 CycloneDDS 를 unicast localhost 로 강제한다(cyclonedds_localhost.xml).
# 또한 bash -lc(login) 로 실행하면 사용자 dotfiles 가 grep→rtk, python→~/.local 등으로
# ROS 실행을 오염시키므로, ROS 작업은 이 env.sh 만 source 한 깨끗한 셸에서 실행한다.

source /opt/ros/jazzy/setup.bash
[ -f "$HOME/so101_ros2_ws/install/setup.bash" ] && source "$HOME/so101_ros2_ws/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
_SO101_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export CYCLONEDDS_URI="file://${_SO101_SETUP_DIR}/cyclonedds_localhost.xml"
