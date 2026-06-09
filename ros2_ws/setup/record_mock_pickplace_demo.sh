#!/usr/bin/env bash
# SO-101 cube_desk Pick&Place — mock(OMPL) RViz 데모를 Xvfb 헤드리스로 녹화.
#
# 가상 X 디스플레이(:99)에 move_group + mock ros2_control + RViz(MotionPlanning)을 띄우고
# ffmpeg x11grab 으로 RViz 화면을 캡처한다. orchestrator(mock_poses)가 4-cube pick&place
# 시퀀스를 끝내면(4/4 planned) 녹화를 종료한다. Isaac Sim·실기기 불필요(kinematic 데모).
#
# WSLg 가 rootless 라 DISPLAY=:0 직접 grab 은 검게 잡히므로 Xvfb 가상 프레임버퍼를 쓴다.
# RViz Ogre 는 GPU 없는 Xvfb 에서 llvmpipe 소프트웨어 OpenGL 로 렌더한다(LIBGL_ALWAYS_SOFTWARE).
#
# 실행:  wsl -d Ubuntu-24.04 bash <repo>/ros2_ws/setup/record_mock_pickplace_demo.sh [out.mp4]
set -o pipefail

REPO=/mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA
OUT="${1:-$HOME/so101_rviz_mock_pickplace.mp4}"
DISP=:99
# 1920x1080 으로 렌더(좌측 패널 + 넓은 landscape 3D 뷰포트) → 1280x720 으로 스케일 출력.
W=1920; H=1080; OUTW=1280; OUTH=720; FPS=15
LOG=/tmp/so101_mock_demo
mkdir -p "$LOG"

source "$REPO/ros2_ws/setup/env.sh"

# Xvfb 에는 GPU 가 없으므로 소프트웨어 OpenGL(llvmpipe) 로 RViz Ogre 를 렌더.
export DISPLAY="$DISP"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export QT_QPA_PLATFORM=xcb

XVFB_PID=""; DEMO_PID=""; FFMPEG_PID=""
cleanup() {
  set +e
  [ -n "$FFMPEG_PID" ] && kill -INT "$FFMPEG_PID" 2>/dev/null
  [ -n "$DEMO_PID" ]   && kill -INT "$DEMO_PID"   2>/dev/null
  sleep 2
  pkill -f "rviz2"          2>/dev/null
  pkill -f "move_group"     2>/dev/null
  pkill -f "ros2_control_node" 2>/dev/null
  [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null
  rm -f "/tmp/.X${DISP#:}-lock" "/tmp/.X11-unix/X${DISP#:}" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "[rec] Xvfb $DISP ${W}x${H}x24 기동"
# 이전 런의 잔여 Xvfb·락 정리(없으면 "Server is already active for display 99" 로 기동 실패).
pkill -9 -x Xvfb 2>/dev/null; rm -f "/tmp/.X${DISP#:}-lock" "/tmp/.X11-unix/X${DISP#:}"; sleep 2
Xvfb "$DISP" -screen 0 ${W}x${H}x24 +extension GLX +render -noreset >"$LOG/xvfb.log" 2>&1 &
XVFB_PID=$!
sleep 3
if ! xdpyinfo -display "$DISP" >/dev/null 2>&1; then
  echo "[rec] Xvfb 기동 실패:"; cat "$LOG/xvfb.log"; exit 1
fi

echo "[rec] move_group + mock ros2_control + RViz 기동"
ros2 launch so101_bringup follower_moveit_demo.launch.py \
  hardware_type:=mock use_rviz:=true >"$LOG/demo.log" 2>&1 &
DEMO_PID=$!

# 준비 대기: demo.log 의 ready 마커를 폴링한다(로컬 파일 read — `ros2 node list` 는
# WSL2 daemon discovery 가 매 호출 행이 걸려 못 씀). move_group + arm 컨트롤러 + RViz.
echo -n "[rec] move_group/controllers/RViz ready 대기"
READY=0
for i in $(seq 1 90); do
  if grep -qa "You can start planning now" "$LOG/demo.log" 2>/dev/null \
     && grep -qa "Ready to take commands for planning group" "$LOG/demo.log" 2>/dev/null \
     && grep -qa "activated arm_trajectory_controller" "$LOG/demo.log" 2>/dev/null; then
    echo " OK(${i}s)"; READY=1; break
  fi
  echo -n "."; sleep 1
done
[ "$READY" = 1 ] || echo " (ready 마커 미검출 — 그래도 진행)"
# RViz 가 로봇 모델 로드/렌더 안정화할 시간
sleep 8

echo "[rec] ffmpeg x11grab → $OUT"
ffmpeg -y -nostdin -loglevel warning \
  -f x11grab -video_size ${W}x${H} -framerate $FPS -i "$DISP" \
  -vf "scale=${OUTW}:${OUTH}" \
  -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p "$OUT" \
  >"$LOG/ffmpeg.log" 2>&1 &
FFMPEG_PID=$!
sleep 1

echo "[rec] orchestrator(mock_poses) 실행 — RViz 에서 4-cube pick&place"
ros2 launch so101_moveit_config pick_place_orchestrator.launch.py \
  mock_poses:=true 2>&1 | tee "$LOG/orchestrator.log"

echo "[rec] 시퀀스 완료 — 마지막 자세 2s 더 캡처 후 녹화 종료"
sleep 2
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null
FFMPEG_PID=""

echo "=== 결과 ==="
grep -E "완료: [0-9]+/[0-9]+ planned|: planned=" "$LOG/orchestrator.log" || echo "(planned 로그 없음 — orchestrator.log 확인)"
ls -la "$OUT"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT" 2>/dev/null | xargs -I{} echo "duration={}s"
