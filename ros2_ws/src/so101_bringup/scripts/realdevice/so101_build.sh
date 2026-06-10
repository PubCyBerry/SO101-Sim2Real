#!/usr/bin/env bash
exec > /tmp/so101_build.log 2>&1
source /opt/ros/jazzy/setup.bash
cd "$HOME/so101_ros2_ws" || exit 2
colcon build --packages-select feetech_ros2_driver --symlink-install
echo "BUILD_EXIT=$?"
