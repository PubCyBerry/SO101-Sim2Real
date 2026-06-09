#!/usr/bin/env bash
# cuMotion(cuRobo) + topic_based_ros2_control 설치 (Isaac Sim pick&place 경로용)
#
# 선행: 01~03 (ROS 2 Jazzy + MoveIt2 + workspace 빌드) 완료.
# 플랫폼: WSL2 Ubuntu 24.04, RTX A4000 (nvidia-smi 로 GPU 인식 확인됨).
#
# 호스트 변경 주의: 아래는 시스템 apt 패키지 + CUDA 키링을 설치한다(수 GB).
# 되돌리기:  sudo apt-get remove --purge 'ros-jazzy-isaac-ros-cumotion*' \
#            ros-jazzy-topic-based-ros2-control cuda-toolkit-12-* && sudo apt-get autoremove
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[1/4] joint_state_topic_hardware_interface (Jazzy 의 topic_based, Isaac Sim<->MoveIt 브릿지)"
# Jazzy 에서 topic_based_ros2_control → joint_state_topic_hardware_interface 로 개명.
# plugin: joint_state_topic_hardware_interface/JointStateTopicSystem
sudo apt-get install -y ros-jazzy-joint-state-topic-hardware-interface

echo "[2/4] CUDA — WSL2 는 Windows 드라이버로 런타임 제공(nvidia-smi). toolkit 별도 설치 안 함."
# 사용자 지시: WSL2 CUDA toolkit 설치 금지. isaac_ros_cumotion apt 는 prebuilt(빌드 시 nvcc 불요),
# cuRobo 런타임은 torch 번들 CUDA 사용. 소스 빌드가 필요해 nvcc 가 요구되면 그때 별도 협의.
command -v nvcc >/dev/null 2>&1 && nvcc --version | grep release || echo "  (nvcc 없음 — apt prebuilt 경로로 진행)"

echo "[3/4] isaac_ros_cumotion + cumotion_moveit (apt 우선, 실패 시 소스)"
if ! sudo apt-get install -y ros-jazzy-isaac-ros-cumotion ros-jazzy-isaac-ros-cumotion-moveit ros-jazzy-isaac-ros-cumotion-examples; then
  echo "  apt 미제공 → 소스 빌드:"
  echo "    cd ~/so101_ros2_ws/src"
  echo "    git clone --recurse-submodules -b release-4.4 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion.git"
  echo "    cd ~/so101_ros2_ws && colcon build --packages-up-to isaac_ros_cumotion_moveit -DPython3_EXECUTABLE=/usr/bin/python3"
fi

echo "[4/4] cuRobo Python 라이브러리 (torch 필요). 검증."
# cuRobo 는 isaac_ros_cumotion 의존성으로 끌려오나, 없으면 pip 설치 안내.
python3 -c "import curobo; print('cuRobo OK:', curobo.__version__)" 2>/dev/null || {
  echo "  cuRobo 미검출 → pip 설치 안내(시간 소요, GPU 컴파일):"
  echo "    pip install torch  # CUDA 12.8 휠"
  echo "    git clone https://github.com/NVlabs/curobo.git && cd curobo && pip install -e . --no-build-isolation"
}

echo "DONE. 검증:"
echo "  source ~/so101_ros2_ws/install/setup.bash"
echo "  ros2 pkg list | grep -E 'cumotion|topic_based'"
echo "  python3 -c 'import curobo, torch; print(torch.cuda.is_available())'"
