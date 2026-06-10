#!/usr/bin/env bash
# SO-101 ROS 2 환경 설정 (source 전용)
#
# 사용:  source <repo>/ros2_ws/setup/env.sh
#
# 주의: bash -lc(login) 로 실행하면 사용자 dotfiles 가 grep→rtk, python→~/.local 등으로
# ROS 실행을 오염시키므로, ROS 작업은 이 env.sh 만 source 한 깨끗한 셸에서 실행한다.
#
# ── RMW: FastDDS (rmw_fastrtps_cpp) ──────────────────────────────────────────
# WSL2 의 networkingMode=mirrored(.wslconfig [experimental]) 환경에서
# CycloneDDS 는 sensor_msgs/Image 같은 메시지를 cross-process 로 전달하지 못한다
# (String 등 단순 타입은 되지만 Image 는 크기 무관 0 fps — mirrored loopback 문제).
# FastDDS 는 localhost 를 공유메모리(SHM) 로 전송해 이 문제를 우회하므로
# 카메라 이미지 토픽이 정상 동작한다. 검증: 3캠 각 ~23fps cross-process 수신.
#
# CycloneDDS 로 되돌리려면 아래 FastDDS 블록을 주석 처리하고
# CycloneDDS 블록의 주석을 해제한다(단 카메라 이미지 토픽은 다시 0 fps 가 됨).

source /opt/ros/jazzy/setup.bash
[ -f "$HOME/so101_ros2_ws/install/setup.bash" ] && source "$HOME/so101_ros2_ws/install/setup.bash"

# ── RMW: FastDDS (rmw_fastrtps_cpp) ──────────────────────────────────────────
# 2026-06-10 검증 결론: 카메라 921KB raw Image 는 FastDDS(SHM/localhost) 만 cross-process
# 전달된다. CycloneDDS(lo unicast)는 MaxMessageSize/rmem 튜닝에도 이미지가 native·rosbridge
# 양쪽 모두 0 전달(joint_states 같은 작은 msg 만 됨). 가이드 §6 도 raw 이미지엔 FastDDS 권장.
#  ※ serial corruption(feetech checksum/desync)은 DDS 가 아니라 usbipd-win 의 USB/IP TCP 를
#    Hyper-V 방화벽이 간섭한 것이 원인 — 방화벽 해제(2026-06-10)로 해결. DDS 선택과 무관.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ── (대안) CycloneDDS — 이미지 전달 불가로 비활성 ─────────────────────────────
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# _SO101_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# export CYCLONEDDS_URI="file://${_SO101_SETUP_DIR}/cyclonedds_localhost.xml"
