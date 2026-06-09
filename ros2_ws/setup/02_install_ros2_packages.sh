#!/usr/bin/env bash
# ROS 2 Jazzy desktop + MoveIt 2 + ros2_control 스택 설치
# 선행: 01_add_ros2_apt_source.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[1/3] ros-jazzy-desktop + dev tools (대용량 다운로드)"
sudo apt-get install -y ros-jazzy-desktop ros-dev-tools

echo "[2/3] MoveIt 2 + ros2_control 스택"
sudo apt-get install -y \
  ros-jazzy-moveit \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-forward-command-controller \
  ros-jazzy-parallel-gripper-controller \
  ros-jazzy-xacro \
  ros-jazzy-rmw-cyclonedds-cpp \
  python3-colcon-common-extensions \
  mesa-utils

# Isaac Sim 연동 경로(PATH C↔D 브릿지)는 05_install_cumotion.sh 에서 추가 설치:
#   ros-jazzy-topic-based-ros2-control (Isaac Sim <-> MoveIt 하드웨어 브릿지)
#   CUDA toolkit + ros-jazzy-isaac-ros-cumotion (cuRobo GPU planner)

echo "[3/3] rosdep init/update"
sudo rosdep init 2>/dev/null || true
rosdep update
echo "DONE: ROS 2 Jazzy + MoveIt2 설치 완료"
