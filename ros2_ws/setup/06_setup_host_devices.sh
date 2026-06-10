#!/usr/bin/env bash
# WSL2 호스트 디바이스/커널 셋업 (최초 1회) — 실기기 팔 + 카메라용.
#   - udev 규칙(99-so101.rules): /dev/{so101_follower,cam_top,cam_wrist,cam_front} 심볼릭 링크
#   - sysctl(wsl_ros2_sysctl.conf): net.core.rmem_max 상향(대용량 토픽 안정)
#   - 사용자를 dialout(시리얼)·video(카메라) 그룹에 추가
# 선행: usbipd 로 팔/카메라가 WSL 에 attach 되어 있어야 udev 가 심볼릭 링크를 만든다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

echo "[1/4] udev 규칙 설치"
sudo cp "${HERE}/99-so101.rules" /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --subsystem-match=video4linux

echo "[2/4] sysctl(rmem) 설치"
sudo cp "${HERE}/wsl_ros2_sysctl.conf" /etc/sysctl.d/99-ros2-wsl.conf
sudo sysctl --system >/dev/null

echo "[3/4] 그룹 추가(dialout, video)"
sudo usermod -aG dialout,video "$USER"

echo "[4/4] 확인"
ls -l /dev/so101_follower /dev/cam_top /dev/cam_wrist /dev/cam_front 2>/dev/null \
  || echo "  (심볼릭 링크 없음 — usbipd attach 후 재실행하거나 99-so101.rules 의 시리얼/USB PATH 확인)"
echo "DONE: 그룹 변경은 WSL 재진입(또는 'wsl --shutdown') 후 적용됨"
