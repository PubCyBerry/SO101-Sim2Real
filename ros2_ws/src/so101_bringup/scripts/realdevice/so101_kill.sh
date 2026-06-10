#!/usr/bin/env bash
# 전체 ROS 스택 종료 (자기 cmdline 에 패턴 없어 self-kill 안 됨)
pkill -f follower_moveit_demo
pkill -f ros2_control_node
pkill -f move_group
pkill -f robot_state_publisher
pkill -f cv2_camera_publisher
pkill -f cameras_cv2
pkill -f rosbridge
pkill -f spawner
sleep 4
echo "killed; remaining ros procs:"
pgrep -af "ros2_control_node|move_group|cv2_camera|rosbridge" | head -5
echo "---done"
