# ros2_ws — SO-101 ROS 2 Jazzy 워크스페이스 (follower MoveIt2)

WSL2 Ubuntu 24.04 + ROS 2 Jazzy 에서 SO-101 **follower** 팔을 RViz 시각화 + MoveIt 2 모션 플래닝/제어하기 위한 워크스페이스.

전체 셋업·실행 절차는 [`../docs/PATH_D_ROS2_WSL_MOVEIT.md`](../docs/PATH_D_ROS2_WSL_MOVEIT.md) 참조.

## 출처 (내재화)

`src/` 의 ROS 2 패키지는 **[legalaspro/so101-ros-physical-ai](https://github.com/legalaspro/so101-ros-physical-ai)** (Apache-2.0) 에서 follower MoveIt2 범위로 내재화했다.

| 패키지 | 출처 | 비고 |
|---|---|---|
| `so101_description` | 위 레포 | URDF/Xacro + STL 메시 (onshape 원본 CAD 제외) |
| `so101_moveit_config` | 위 레포 | SRDF/OMPL/kinematics/controllers/RViz |
| `so101_bringup` | 위 레포 | follower 관련 launch·config 만 유지 (leader/teleop/recording 제거, 카메라 패키지 의존성 제거) |
| `feetech_ros2_driver` | [legalaspro/feetech_ros2_driver](https://github.com/legalaspro/feetech_ros2_driver) (`feat/joint-config-and-calibration`) | git submodule |

원본 전체 스택(teleop/episode_recorder/inference/policy_server 등)은 `ref_repos/so101-ros-physical-ai/` 에 보존.

## 디렉터리

- `setup/` — ROS 2 Jazzy + MoveIt2 설치 스크립트 (`01_add_ros2_apt_source.sh`, `02_install_ros2_packages.sh`)
- `src/` — 내재화 ROS 2 패키지 (git 관리)
- `build/ install/ log/` — colcon 산출물 (`.gitignore`)

## 빌드 (요약)

빌드 성능을 위해 WSL ext4 워크스페이스(`~/so101_ros2_ws`)에서 `src` 를 이 레포로 심볼릭 링크해 빌드한다.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/so101_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 실행 (요약)

```bash
# mock (USB 불필요) — RViz + MoveIt 모션 플래닝 검증
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=mock

# 실기기 (usbipd attach + udev 선행)
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=real usb_port:=/dev/so101_follower
```
