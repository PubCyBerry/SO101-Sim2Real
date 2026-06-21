# PATH D — WSL2 + ROS 2 Jazzy + SO-101 (RViz + MoveIt 2)

> 이 문서는 Legacy MoveIt 실험 경로다. Canonical parity real client는 Windows native Pixi에서
> 실행하며 WSL2/usbipd를 사용하지 않는다. 현재 기본 경로는
> [`PATH_F_CANONICAL_PARITY.md`](PATH_F_CANONICAL_PARITY.md)다.

Windows 11 워크스테이션의 **WSL2(Ubuntu 24.04)** 위에서 ROS 2 Jazzy 로 SO-101 **follower** 팔을
RViz 시각화 + MoveIt 2 모션 플래닝/제어하는 경로.

- 실기기 LeRobot 경로(PATH A/B)·시뮬 경로(PATH C)와 **독립**. ROS 2 스택은 `ros2_ws/` 에 내재화.
- 1차 범위: **follower 1팔 MoveIt 2**. 추가로 **3캠(top/wrist/front) + ros-mcp 원격 제어/관측**을 [§8](#8-카메라--ros-mcp-원격-제어관측) 에서 다룬다. leader teleop·녹화는 미포함(원본 스택 `ref_repos/so101-ros-physical-ai/` 에 보존).
- 스택 출처: [legalaspro/so101-ros-physical-ai](https://github.com/legalaspro/so101-ros-physical-ai) (Apache-2.0). 자세한 내재화 범위는 [`../ros2_ws/README.md`](../ros2_ws/README.md).

---

## 1. 구성 요약

| 항목 | 값 |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu 24.04.x) |
| ROS | ROS 2 Jazzy (apt 바이너리) + MoveIt 2 |
| 패키지 | `so101_description`, `so101_moveit_config`, `so101_bringup`, `feetech_ros2_driver`(submodule) |
| 소스 위치 | 이 레포 `ros2_ws/src/` (git 관리) |
| 빌드 위치 | WSL ext4 `~/so101_ros2_ws/` (`src` → 레포로 심볼릭 링크, 산출물은 ext4) |
| USB 포워딩 | usbipd-win → WSL2 (팔=CH343 `/dev/ttyACM0`, 카메라=UVC `/dev/video*`) |
| 네트워킹 | `.wslconfig` `networkingMode=mirrored` (`wslconfig.example`) |
| DDS | **FastDDS (`rmw_fastrtps_cpp`)** — mirrored 에서 CycloneDDS 가 카메라 이미지 cross-process 전달 실패해 FastDDS(SHM) 사용. [§8](#8-카메라--ros-mcp-원격-제어관측)·TROUBLESHOOTING 참조 |
| 제어 | ros2_control: `arm_trajectory_controller`(FollowJointTrajectory) + `gripper_controller`(ParallelGripper) + `joint_state_broadcaster` |

### MoveIt planning group (SRDF)

- `manipulator` — base_link → gripper_frame_link (5축: shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll)
- `gripper` — gripper joint 단독. RViz MotionPlanning 의 **Planning Group 드롭다운**에서 전환.

---

## 2. 빠른 시작 (이미 셋업 완료된 경우)

WSL 터미널(bashrc 가 `env.sh` 를 자동 source):

```bash
# (A) mock — USB 불필요, RViz + MoveIt 모션 플래닝
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=mock
```

실기기는 먼저 Windows PowerShell 에서 USB attach:

```powershell
usbipd attach --wsl --busid 4-1     # follower (COM8). busid 는 usbipd list 로 확인
```

그 다음 WSL:

```bash
ls /dev/so101_follower   # udev 심볼릭 링크 확인 (없으면 §5.3)
ros2 launch so101_bringup follower_moveit_demo.launch.py \
  hardware_type:=real usb_port:=/dev/so101_follower
```

> ROS 2 cli(`ros2 topic`, `ros2 node`)는 stale daemon 때문에 빈 결과가 나올 수 있다 → `--no-daemon` 사용. RViz↔MoveIt 제어는 영향 없음([§6](#6-알려진-제약)).

---

## 3. 전체 셋업 — 한 번만

설치 스크립트는 `ros2_ws/setup/` 에 있다. WSL 경로는 `/mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup`.

### 3.1 (선택) sudo NOPASSWD 임시 설정

설치에 sudo 가 많이 필요하다. WSL 기본 사용자가 비밀번호를 요구하면:

```bash
# WSL 은 -u root 진입에 비밀번호가 필요 없다 → 이를 이용해 임시 NOPASSWD 등록
wsl -d Ubuntu-24.04 -u root -- bash -c \
  'echo "$(logname) ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/99-claude-temp-nopasswd && chmod 0440 /etc/sudoers.d/99-claude-temp-nopasswd'
```

작업 후 되돌리기: `sudo rm /etc/sudoers.d/99-claude-temp-nopasswd`

### 3.2 ROS 2 Jazzy + MoveIt 2 설치

```bash
cd /mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup
bash 01_add_ros2_apt_source.sh      # locale + universe + ros2-apt-source 등록
bash 02_install_ros2_packages.sh    # ros-jazzy-desktop + moveit + ros2_control + cyclonedds (대용량)
```

`02` 가 설치하는 핵심: `ros-jazzy-desktop`(FastDDS·rqt 포함), `ros-jazzy-moveit`, `ros-jazzy-ros2-control(lers)`,
`ros-jazzy-joint-trajectory-controller`, `ros-jazzy-parallel-gripper-controller`,
`ros-jazzy-forward-command-controller`, `ros-jazzy-xacro`, `ros-jazzy-rmw-cyclonedds-cpp`,
`ros-jazzy-rosbridge-suite`(ros-mcp), `v4l-utils`(카메라 진단),
`python3-colcon-common-extensions`, `mesa-utils`. (카메라 발행은 OpenCV 기반 `cv2_camera_publisher.py`
가 담당하므로 gscam/v4l2_camera 는 설치하지 않는다 — [§8](#8-카메라--ros-mcp-원격-제어관측).)

### 3.3 워크스페이스 빌드

```bash
# feetech submodule 체크아웃 (레포에서 한 번)
git -C /mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA submodule update --init ros2_ws/src/feetech_ros2_driver

cd /mnt/c/Users/taehunkim/Workspace/SO101-LeRobot-VLA/ros2_ws/setup
bash 03_build_workspace.sh           # ext4 워크스페이스 + 심볼릭 링크 + rosdep + colcon build
```

`03` 요점:
- 소스는 레포(`/mnt/c/.../ros2_ws/src`), 빌드 산출물은 ext4(`~/so101_ros2_ws`)에 두려고 `src` 를 심볼릭 링크.
- **시스템 python 강제**: `--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3`
  (사용자 dotfiles 의 `~/.local/bin/python3.11` 이 ament 빌드를 가로채 `catkin_pkg` 못 찾는 문제 회피).
- `set -eo pipefail` 만 사용(`-u` 는 ROS `setup.bash` 와 충돌).

### 3.4 환경 + bashrc 등록

```bash
bash 04_setup_bashrc.sh   # ~/.bashrc 에 env.sh source 를 멱등 등록 (백업 생성)
```

`env.sh` 가 매 셸에서:
- `/opt/ros/jazzy/setup.bash` + `~/so101_ros2_ws/install/setup.bash` source
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` (CycloneDDS 블록은 주석으로 보존 — [§6](#6-알려진-제약))

> ROS 작업은 **`bash -c`(또는 새 로그인 셸) + env.sh** 로 한다. `bash -lc` 로 임의 스크립트를 돌리면 사용자 dotfiles 가 `grep`→`rtk`, `python`→`~/.local`, RMW 등을 덮어써 ROS 실행이 깨진다.
> **모든 노드(follower·rosbridge·카메라)가 같은 RMW 여야 한다** — 전부 env.sh 를 source 한 셸에서 실행. RMW 가 섞이면 graph 가 분리된다.

### 3.5 호스트 디바이스/커널 셋업 (실기기·카메라, 최초 1회)

usbipd 로 팔/카메라를 attach([§5.1](#51-usb-attach-windows-powershell)·[§8](#8-카메라--ros-mcp-원격-제어관측))한 뒤:

```bash
bash 06_setup_host_devices.sh   # udev(99-so101.rules) + sysctl(rmem) + dialout/video 그룹
```

- udev: `/dev/{so101_follower,cam_top,cam_wrist,cam_front}` 안정 심볼릭 링크. **기기마다 시리얼/USB PATH 가 달라** `99-so101.rules` 를 본인 값으로 교체해야 할 수 있다(파일 주석 참조).
- sysctl: `net.core.rmem_max` 16MB(대용량 토픽 안정). `wsl_ros2_sysctl.conf` → `/etc/sysctl.d/99-ros2-wsl.conf`.
- 그룹 변경은 WSL 재진입 후 적용.

### 3.6 (선택) `.wslconfig` 네트워킹

이 셋업은 `networkingMode=mirrored` 를 전제로 한다(usbipd 미러링). `ros2_ws/setup/wslconfig.example` 을 `C:\Users\<user>\.wslconfig` 로 복사 후 `wsl --shutdown`. mirrored 에서 CycloneDDS 가 카메라 이미지를 전달 못 해 RMW 를 FastDDS 로 쓴다([§8](#8-카메라--ros-mcp-원격-제어관측)).

---

## 4. mock 실행 (USB 불필요)

```bash
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=mock
```

- `follower_moveit_demo.launch.py` = `follower_split.launch.py`(ros2_control + rsp + spawners) + `move_group` + `moveit_rviz`.
- RViz(WSLg) 가 Windows 화면에 뜬다. `MotionPlanning` 패널 → interactive marker 드래그 → `Plan` → `Execute`.
- GPU 렌더가 안 되면 `LIBGL_ALWAYS_SOFTWARE=1` 로 폴백.

---

## 5. 실기기 실행

### 5.1 USB attach (Windows PowerShell)

```powershell
usbipd list                         # follower 의 BUSID 확인 (CH343 / VID:PID 1a86:55d3)
usbipd bind --busid 4-1             # 최초 1회 (공유 등록)
usbipd attach --wsl --busid 4-1     # WSL 로 attach
```

- WSL 또는 PC 재시작 시 attach 가 풀린다 → `attach` 재실행. "already attached" 인데 WSL 에서 안 보이면 `usbipd detach --busid 4-1` 후 재attach.

### 5.2 장치 확인 (WSL)

```bash
ls -l /dev/ttyACM* /dev/so101_follower
```

CH343 은 cdc_acm 드라이버로 `/dev/ttyACM0` 에 잡힌다(커널 6.6 빌트인). udev 규칙이 `/dev/so101_follower` 심볼릭 링크 생성.

### 5.3 udev 규칙 (최초 1회)

규칙 파일은 레포 `ros2_ws/setup/99-so101.rules` 에 있다(팔=시리얼, 카메라=USB PATH). [§3.5](#35-호스트-디바이스커널-셋업-실기기카메라-최초-1회) 의 `06_setup_host_devices.sh` 가 이를 `/etc/udev/rules.d/` 로 설치 + reload + 그룹 추가한다. 팔만 빠르게 하려면:

```
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="1a86", ENV{ID_MODEL_ID}=="55d3", ENV{ID_SERIAL_SHORT}=="5AE6057916", SYMLINK+="so101_follower", GROUP="dialout", MODE="0660"
```

> serial 은 환경마다 다르다. `udevadm info --query=property --name=/dev/ttyACM0 | grep ID_SERIAL_SHORT` 로 확인 후 교체. 카메라 규칙·주의사항은 [§8](#8-카메라--ros-mcp-원격-제어관측).

수동 적용:
```bash
sudo cp ros2_ws/setup/99-so101.rules /etc/udev/rules.d/99-so101.rules
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty --subsystem-match=video4linux
sudo usermod -aG dialout,video $USER     # 최초 1회 (WSL 재진입 후 적용)
```

### 5.4 실기기 launch

> ⚠️ **실제 모터가 움직인다.** 팔 주변 정리, 비상 전원 차단 준비. 첫 동작은 짧고 느리게.

```bash
ros2 launch so101_bringup follower_moveit_demo.launch.py \
  hardware_type:=real usb_port:=/dev/so101_follower
```

정상이면: hardware `SO101_follower_SYSTEM` activate → 컨트롤러 3개 "Configured and activated" → RViz 에 실제 팔 자세 반영 → Plan & Execute 시 실제 팔 이동.

---

## 6. 알려진 제약

| 제약 | 내용 | 대응 |
|---|---|---|
| CycloneDDS + mirrored | mirrored 네트워킹에서 CycloneDDS 가 `sensor_msgs/Image` 를 cross-process 전달 못 함(작은 메시지·discovery 는 정상) → 카메라 0 fps | **FastDDS(`rmw_fastrtps_cpp`)** 로 전환(localhost SHM). env.sh 기본값. cyclonedds 블록은 주석 보존 |
| WSL2 DDS discovery | (참고) NAT 네트워킹에선 `lo` 에 MULTICAST 없어 기본 discovery 실패 | mirrored + FastDDS 면 무관. NAT 면 `cyclonedds_localhost.xml` 같은 unicast 우회 필요 |
| ros2 cli daemon | stale daemon 으로 `ros2 topic/node list` 가 빈 결과 | `--no-daemon` 사용. RViz↔MoveIt 제어는 무관 |
| USB-IP 레이턴시 | polling 기반이라 serial read 스파이크 큼 → read timeout 1회로 hardware deactivate | serial timeout 50ms, controller `update_rate` 50Hz, `on_activate` 안정화 sleep (드라이버에 반영). 네이티브 Linux 면 100Hz 복원 가능 |
| dotfiles 오염 | `bash -lc` 가 grep→rtk, python→~/.local, RMW 덮어씀 | ROS 작업은 `bash -c` + env.sh |

자세한 증상/원인/해결은 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) 의 "ROS 2 / WSL2 / MoveIt" 항목들 참조.

---

## 7. 검증 체크리스트

1. `ros2 pkg list --no-daemon | grep -E 'so101|feetech'` → 4 패키지
2. mock: `follower_moveit_demo ... hardware_type:=mock` → 컨트롤러 3개 activated, RViz Plan/Execute 동작
3. 실기기: `... hardware_type:=real` → `Read timeout` 0, joint_states 가 실제 모터 각도 반영
4. RViz: 실제 팔 자세 ↔ 모델 일치, Plan & Execute 시 실제 팔 이동

---

## 8. 카메라 + ros-mcp (원격 제어/관측)

3캠(top/wrist/front) 이미지를 ROS 토픽으로 발행하고, rosbridge 를 통해 ros-mcp(외부 에이전트)로 팔 제어 + 카메라 관측을 한다.

### 8.1 왜 cv2 + FastDDS 인가 (배경)

- **드라이버**: gscam·v4l2_camera 는 usbipd-win 가상 V4L2 에서 동작하지 않는다(appsink 타임아웃 / DQBUF 행 / MJPG 디코드 크래시). → OpenCV(MJPG) 캡처 노드 `so101_bringup/scripts/cv2_camera_publisher.py` 사용. USB-IP 가 다중 스레드 동시 read 를 못 버텨 **단일 스레드 라운드로빈**으로 3캠 캡처.
- **RMW**: `.wslconfig` mirrored 에서 CycloneDDS 는 `sensor_msgs/Image` 를 cross-process 전달 못 함(0 fps). → **FastDDS**(env.sh 기본, localhost SHM).
- 상세: `docs/TROUBLESHOOTING.md` 의 "카메라 image_raw 0 fps" / "gscam·v4l2_camera 동작 안 함".

### 8.2 카메라 USB attach (Windows PowerShell)

```powershell
usbipd list                              # 카메라(UVC) BUSID 확인
usbipd bind  --busid <BUSID> --force     # 카메라마다 (관리자 권한, 최초 1회)
usbipd attach --wsl --busid <BUSID>      # 카메라마다
```

> ⚠️ 3대가 동일 시리얼이라 udev 가 **USB PATH(attach 한 bus 위치)** 로 top/wrist/front 를 구분한다. attach 순서/포트가 바뀌면 배정이 뒤바뀐다 → §8.4 에서 **육안 확인** 필수.

### 8.3 디바이스 셋업

[§3.5](#35-호스트-디바이스커널-셋업-실기기카메라-최초-1회) `06_setup_host_devices.sh` 가 udev(카메라 심볼릭 링크)·sysctl·그룹을 설치한다. `/dev/cam_{top,wrist,front}` 가 생기는지 확인.

### 8.4 카메라 식별 검증 (최초 1회 / 순서 바뀔 때)

```bash
# 각 디바이스에서 프레임 1장 캡처해 육안 확인 후 99-so101.rules 의 USB PATH(N) 를 맞춘다
python3 - <<'PY'
import cv2
for dev in ["/dev/cam_top","/dev/cam_wrist","/dev/cam_front"]:
    c=cv2.VideoCapture(dev, cv2.CAP_V4L2); c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ok,f=c.read(); cv2.imwrite("/tmp/%s.jpg"%dev.split("_")[1], f) if ok else None; c.release()
    print(dev, ok)
PY
```

top=부감, wrist=그리퍼 클로즈업, front=정면 측면뷰. 어긋나면 `99-so101.rules` 의 `0:N:1.0` 값 교체 후 재설치.

### 8.5 실행 (각 셸 모두 env.sh)

```bash
# 카메라 발행
ros2 launch so101_bringup cameras_cv2.launch.py     # 또는 ros2 run so101_bringup cv2_camera_publisher.py
# rosbridge (ros-mcp 접속용)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090
# 팔 (필요 시)
ros2 launch so101_bringup follower.launch.py use_rviz:=false arm_controller:=trajectory_controller
```

토픽: `/camera/{top,wrist,front}/image_raw` + `/camera/{top,wrist,front}/camera_info` (각 ~23fps).

### 8.6 검증

```bash
ros2 topic hz /camera/top/image_raw          # ~20fps+
rqt_image_view                               # 드롭다운에서 각 카메라 영상 확인
rqt_graph                                    # 노드/토픽 그래프 시각 확인
```

ros-mcp 쪽에서 `127.0.0.1:9090` 접속 → `get_nodes` 에 follower·`cv2_camera_publisher`·rosbridge 가 한 graph 로 보이고, `/follower/joint_states` + `/camera/*/image_raw` 수신되면 완료.
