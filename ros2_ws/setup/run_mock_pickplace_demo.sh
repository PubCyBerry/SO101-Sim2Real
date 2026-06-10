#!/usr/bin/env bash
# SO-101 cube_desk Pick&Place — mock(OMPL) + RViz 데모. Isaac Sim 불필요.
# WSLg 가 Windows 화면에 RViz 창을 띄운다. Ctrl+C 로 전체 종료.
#
# 실행:  wsl -d Ubuntu-24.04 bash <repo>/ros2_ws/setup/run_mock_pickplace_demo.sh
set -o pipefail
source /mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup/env.sh

echo "[demo] move_group + mock ros2_control + RViz(MotionPlanning) 기동..."
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=mock use_rviz:=true &
DEMO=$!

# move_group/controllers/RViz 준비 대기
sleep 28

echo "[demo] orchestrator(mock_poses) 실행 — RViz 에서 SO-101 팔이 pick&place 모션 수행"
ros2 launch so101_moveit_config pick_place_orchestrator.launch.py mock_poses:=true

echo "[demo] 시퀀스 완료. RViz 는 유지됨 — 관찰 후 Ctrl+C 로 종료."
wait "$DEMO"
