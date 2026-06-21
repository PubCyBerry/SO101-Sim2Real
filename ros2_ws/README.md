# ros2_ws — SO-101 ROS 2 Jazzy 워크스페이스

이 workspace에는 두 경로가 공존한다.

| 경로 | 패키지 | 실행 환경 | 문서 |
|---|---|---|---|
| Canonical parity | `so101_vla_interfaces`, `so101_vla_runtime` | Windows/Linux native Pixi + ROS Jazzy + `rmw_zenoh_cpp` | [`PATH_F_CANONICAL_PARITY`](../docs/PATH_F_CANONICAL_PARITY.md) |
| Legacy MoveIt | `so101_description`, `so101_moveit_config`, `so101_bringup`, `feetech_ros2_driver` | WSL2 Ubuntu 24.04 + ROS Jazzy | [`PATH_D_ROS2_WSL_MOVEIT`](../docs/PATH_D_ROS2_WSL_MOVEIT.md) |

Canonical package만 빌드:

```bash
pixi run ros-build
```

산출물은 repo `.pixi/ros2` 아래에 생성된다. Windows에서는 `.pixi` Junction을 통해
`D:\SO101\isaac6_ros\.pixi\ros2`에 저장된다.

아래 내용은 Legacy follower MoveIt2 경로다.

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
