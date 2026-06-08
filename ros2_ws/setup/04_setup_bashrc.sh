#!/usr/bin/env bash
# ~/.bashrc 에 SO-101 ROS 2 환경(env.sh) source 를 멱등 등록.
set -e
F="$HOME/.bashrc"
ENV_SH="/mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup/env.sh"

cp "$F" "$F.bak.$(date +%s)"
# 기존 마커 블록 제거(멱등)
sed -i '/# >>> ROS 2 Jazzy (SO-101) >>>/,/# <<< ROS 2 Jazzy (SO-101) <<</d' "$F"

cat >> "$F" <<EOF

# >>> ROS 2 Jazzy (SO-101) >>>
[ -f "${ENV_SH}" ] && source "${ENV_SH}"
# <<< ROS 2 Jazzy (SO-101) <<<
EOF

echo "bashrc updated"
tail -6 "$F"
