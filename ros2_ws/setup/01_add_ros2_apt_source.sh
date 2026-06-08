#!/usr/bin/env bash
# ROS 2 Jazzy apt 저장소 + locale 설정 (Ubuntu 24.04 / WSL2)
# 공식 절차: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
set -euo pipefail

echo "[1/4] locale (UTF-8) 설정"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y locales curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "[2/4] universe 저장소 활성화"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
sudo add-apt-repository -y universe

echo "[3/4] ros2-apt-source 패키지 설치 (apt-key 대체 공식 방식)"
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F'"' '{print $4}')
echo "  ros-apt-source 버전: ${ROS_APT_SOURCE_VERSION}"
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/ros2-apt-source.deb

echo "[4/4] apt update"
sudo apt-get update -qq
apt-cache policy ros-jazzy-desktop | head -3 || true
echo "DONE: ROS 2 apt 저장소 등록 완료"
