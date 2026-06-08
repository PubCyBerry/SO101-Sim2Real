#!/usr/bin/env bash
# ext4 워크스페이스 구성 + rosdep + colcon build
# 선행: 01_add_ros2_apt_source.sh, 02_install_ros2_packages.sh
# 소스는 이 레포(ros2_ws/src, /mnt/c)에 두고, 빌드 산출물은 WSL ext4(~/so101_ros2_ws)에 둔다.
# nounset(-u)은 ROS setup.bash 와 충돌(AMENT_TRACE_SETUP_FILES unbound)하므로 제외
set -eo pipefail

REPO_SRC="/mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/src"
WS="${HOME}/so101_ros2_ws"

source /opt/ros/jazzy/setup.bash

echo "[1/3] ext4 워크스페이스 + src 심볼릭 링크: ${WS}"
mkdir -p "${WS}"
ln -sfn "${REPO_SRC}" "${WS}/src"

echo "[2/3] rosdep install"
cd "${WS}"
rosdep install --from-paths src --ignore-src -r -y

echo "[3/3] colcon build"
# 사용자 dotfiles 의 ~/.local/bin/python3.11 이 ament 빌드를 가로채지 않도록 시스템 python(3.12) 강제
rm -rf "${WS}/build" "${WS}/install" "${WS}/log"
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
echo "DONE: 빌드 완료 → source ${WS}/install/setup.bash"
