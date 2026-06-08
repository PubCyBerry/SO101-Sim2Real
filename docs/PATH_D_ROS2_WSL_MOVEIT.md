# PATH D — WSL2 + ROS 2 Jazzy + SO-101 (RViz + MoveIt 2)

Windows 11 워크스테이션의 **WSL2(Ubuntu 24.04)** 위에서 ROS 2 Jazzy 로 SO-101 **follower** 팔을
RViz 시각화 + MoveIt 2 모션 플래닝/제어하는 경로.

- 실기기 LeRobot 경로(PATH A/B)·시뮬 경로(PATH C)와 **독립**. ROS 2 스택은 `ros2_ws/` 에 내재화.
- 1차 범위: **follower 1팔 MoveIt 2**. leader teleop·카메라·녹화는 미포함(원본 스택 `ref_repos/so101-ros-physical-ai/` 에 보존).
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
| USB 포워딩 | usbipd-win → WSL2 (CH343, `/dev/ttyACM0` cdc_acm) |
| DDS | **CycloneDDS + unicast localhost** (WSL2 lo 에 MULTICAST 없어 필수) |
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

`02` 가 설치하는 핵심: `ros-jazzy-desktop`, `ros-jazzy-moveit`, `ros-jazzy-ros2-control(lers)`,
`ros-jazzy-joint-trajectory-controller`, `ros-jazzy-parallel-gripper-controller`,
`ros-jazzy-forward-command-controller`, `ros-jazzy-xacro`, `ros-jazzy-rmw-cyclonedds-cpp`,
`python3-colcon-common-extensions`, `mesa-utils`.

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
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
- `CYCLONEDDS_URI=file://.../cyclonedds_localhost.xml` (WSL2 DDS 우회, [§6](#6-알려진-제약))

> ROS 작업은 **`bash -c`(또는 새 로그인 셸) + env.sh** 로 한다. `bash -lc` 로 임의 스크립트를 돌리면 사용자 dotfiles 가 `grep`→`rtk`, `python`→`~/.local`, RMW 등을 덮어써 ROS 실행이 깨진다.

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

`/etc/udev/rules.d/99-so101.rules` — serial number 로 leader/follower 구분:

```
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="1a86", ENV{ID_MODEL_ID}=="55d3", ENV{ID_SERIAL_SHORT}=="5AE6057916", SYMLINK+="so101_follower", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="1a86", ENV{ID_MODEL_ID}=="55d3", ENV{ID_SERIAL_SHORT}=="5AE6082830", SYMLINK+="so101_leader",   GROUP="dialout", MODE="0660"
```

> serial 은 환경마다 다르다. `udevadm info --query=property --name=/dev/ttyACM0 | grep ID_SERIAL_SHORT` 로 확인 후 교체.

적용:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty
sudo usermod -aG dialout $USER     # 최초 1회 (로그인 갱신 필요)
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
| WSL2 DDS discovery | `lo` 에 MULTICAST 플래그가 없어 기본 DDS discovery 실패 | CycloneDDS + unicast localhost (`cyclonedds_localhost.xml`, env.sh 가 자동 적용) |
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
