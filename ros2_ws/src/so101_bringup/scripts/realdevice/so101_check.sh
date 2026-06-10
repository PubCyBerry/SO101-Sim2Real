#!/usr/bin/env bash
source /mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup/env.sh
echo "=== RMW ==="; echo "$RMW_IMPLEMENTATION"
echo "=== native ros2 node list (FastDDS daemon 버그면 멈춤/빈값) ==="
timeout 15 ros2 node list 2>&1
echo "=== follower log: deactivate count ==="
grep -c "Deactivating" /tmp/follower_moveit.log 2>/dev/null
echo "=== follower log: read failure count ==="
grep -c "transient failure" /tmp/follower_moveit.log 2>/dev/null
echo "=== follower log: switched count ==="
grep -c "Successfully switched" /tmp/follower_moveit.log 2>/dev/null
echo "=== last read-failure line (if any) ==="
grep "transient failure" /tmp/follower_moveit.log 2>/dev/null | tail -1
echo "=== DONE ==="
