# PATH E — Isaac Sim cube_desk + MoveIt2/cuMotion Pick & Place

cube_desk 장면(Isaac Sim, Windows)에서 SO-101 이 큐브 4개를 그릇에 담는 pick & place 를
**MoveIt2 path planning(cuMotion 우선, OMPL/Pilz 폴백)** 으로 구동한다. PATH C(Isaac Sim)와
PATH D(WSL2 ROS2 MoveIt)를 `topic_based_ros2_control` 브릿지로 잇는다.

## 아키텍처

```
[Windows] Isaac Sim 5.1                          [WSL2 Ubuntu 24.04 / ROS2 Jazzy]
  scripts/ros2/cube_desk_ros2_sim.py               isaac_pick_place.launch.py
   cube_desk scene.usd + SO-101 USD                  ros2_control (topic_based)
   OmniGraph:                                         ├ arm_trajectory_controller
    Pub /isaac_joint_states  ──── DDS ─────►          └ gripper_controller
    Sub /isaac_joint_commands ◄── 7400/  ──────────  move_group (ompl/pilz/isaac_ros_cumotion)
    Pub /clock                    7410/9387           cumotion_action_server (xrdf+urdf)
    Pub TF base_link→Cube/Bowl ───────►              pick_place_orchestrator.launch.py
                                                       moveit_py FSM + PlanningScene
```

- MoveIt "world" = 로봇 `base_link` (SRDF `virtual_joint fixed_base`).
- 5-DOF: SO-101 은 임의 6-DOF pose 도달 불가 → grasp 자세는 top-down tilt + pick_ik approximate / cuRobo pose-metric.

## 신규/수정 파일

| 파일 | 역할 |
|---|---|
| `ros2_ws/setup/05_install_cumotion.sh` | CUDA toolkit + topic_based + isaac_ros_cumotion 설치 |
| `ros2_ws/setup/cyclonedds_bridge.xml` | WSL2↔Windows DDS (Windows host IP peer) |
| `…/so101_description/urdf/ros2_control/so101_ros2_control.xacro` | `hardware_type:=isaac` (topic_based) |
| `…/so101_bringup/config/ros2_control/isaac_controllers.yaml` | isaac 컨트롤러 (use_sim_time) |
| `…/so101_bringup/launch/isaac_pick_place.launch.py` | 통합 bringup+move_group+cumotion |
| `…/so101_moveit_config/launch/move_group.launch.py` | `use_cumotion` arg (cuMotion pipeline) |
| `…/so101_moveit_config/config/so101_arm.xrdf` | cuMotion collision spheres (M2 정밀화) |
| `…/so101_moveit_config/config/moveit_py_config.yaml` | cumotion plan param 세트 |
| `…/so101_moveit_config/scripts/so101_pick_place_orchestrator.py` | pick&place FSM |
| `…/so101_moveit_config/launch/pick_place_orchestrator.launch.py` | FSM + moveit_py config |
| `scripts/ros2/cube_desk_ros2_sim.py` | Isaac Sim 장면 + ROS2 OmniGraph |
| `ros2_ws/setup/run_mock_pickplace_demo.sh` | RViz mock(OMPL) 데모 — WSLg 대화형 |
| `ros2_ws/setup/record_mock_pickplace_demo.sh` | RViz mock 데모 Xvfb 헤드리스 녹화(mp4) |

## 실행 (마일스톤 순서)

### M0 — WSL2 prereq (2026-06-09 완료)

**WSL2 에는 OMPL/Pilz 만** (cuMotion 은 Linux 서버 — 아래 §cuMotion 참조).
- ✅ `ros-jazzy-joint-state-topic-hardware-interface` 설치 (Jazzy 의 topic_based, plugin
  `joint_state_topic_hardware_interface/JointStateTopicSystem`).
- ✅ so101_description/moveit_config/bringup colcon 재빌드, xacro isaac variant 검증.
- ⛔ `ros-jazzy-isaac-ros-cumotion` 은 WSL2 미설치 — `cuda-toolkit-13-0` + `libnvvpi4` +
  `gxf-isaac-*` 풀스택 요구(CUDA 13 vs 프로젝트 12.8 핀 충돌). → Linux 서버에서만.

```bash
# WSL2 검증
source /opt/ros/jazzy/setup.bash && source ~/so101_ros2_ws/install/setup.bash
ros2 pkg prefix joint_state_topic_hardware_interface   # 설치 확인
```

### 네트워킹 — WSL2↔Windows DDS
1. WSL2 에서 Windows host IP 확인: `ip route show default | awk '{print $3}'`
2. `cyclonedds_bridge.xml` 의 `WINDOWS_HOST_IP` 치환 (미러드 네트워킹이면 localhost).
3. WSL2 ROS 셸: `export CYCLONEDDS_URI=file://.../cyclonedds_bridge.xml` (env.sh 대신/override).
4. Windows: Isaac Sim 은 Fast DDS(Cyclone 미지원). 방화벽 인바운드 7400/7410/7411/9387 허용,
   `ROS_DOMAIN_ID` 양쪽 일치.

### M1 — 브릿지 smoke (최대 리스크 선검증)
```bash
# Windows
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python scripts/ros2/cube_desk_ros2_sim.py
# WSL2
ros2 topic echo /isaac_joint_states --once        # Isaac → WSL2 수신 확인
ros2 topic echo /tf --once                          # 객체 TF 확인
ros2 topic pub -1 /isaac_joint_commands sensor_msgs/JointState '{name: [shoulder_pan], position: [0.3]}'
#   → Isaac 에서 관절 움직이면 양방향 OK
```
실패 시: 노드 타입명(OmniGraph)·base_link prim 경로·DDS 포트포워딩 점검.

### M2 — MoveIt 실행 (WSL2 = OMPL/Pilz)
```bash
# WSL2 (use_cumotion 기본 false = OMPL/Pilz)
ros2 launch so101_bringup isaac_pick_place.launch.py
# RViz MotionPlanning → 드래그 goal → Plan & Execute → Isaac 로봇 추종 확인.
# (Linux 서버에서 cuMotion 사용 시 use_cumotion:=true + Planning Library "isaac_ros_cumotion")
```

### RViz mock 데모 — Isaac Sim·실기기 불필요 (kinematic, 영상)

Isaac 연동 전에 OMPL planning + FSM 전이를 RViz 에서 검증·시연하는 mock 모드.
`MOCK_POSES_BASE`(SO-101 도달영역 안 큐브4+그릇1) 로 객체 TF 없이 동작한다.

```bash
# WSLg 로 RViz 창을 Windows 화면에 띄우는 대화형 실행
wsl -d Ubuntu-24.04 bash ros2_ws/setup/run_mock_pickplace_demo.sh

# Xvfb 가상 디스플레이(1920x1080→1280x720)에 띄워 화면을 mp4 로 녹화 (창 안 뜸, GPU 불필요)
wsl -d Ubuntu-24.04 bash ros2_ws/setup/record_mock_pickplace_demo.sh [out.mp4]
```

- 동작: HOME→큐브별 approach/descend/close+attach/lift/transport/place/release+detach/retreat→DONE,
  로그 `완료: 4/4 planned`. arm 은 `FollowJointTrajectory`, gripper 는 `ParallelGripperCommand`
  (`parallel_gripper_action_controller`) 로 mock 컨트롤러가 실시간 실행(RViz 추종).
- 녹화본: [`so101_rviz_mock_pickplace.mp4`](so101_rviz_mock_pickplace.mp4) — 1280×720, 4/4 planned.
  좌측 MotionPlanning 패널 + 3D 뷰에서 SO-101 팔이 4-cube pick&place 모션 수행, 그리퍼 개폐 포함.
- 한계: kinematic(물리 생략). mock 은 큐브/그릇 collision object 를 생략(잡은 큐브만 attach 로 표시)
  하고 arm plan 성공으로 `planned` 판정한다. 실 grasp·물리·정식 충돌회피 검증은 Isaac/실기기.
- 헤드리스 녹화: WSLg 는 rootless 라 `:0` 직접 grab 이 검게 잡힘 → Xvfb 가상 프레임버퍼 + llvmpipe
  소프트웨어 OpenGL + ffmpeg x11grab. `moveit.rviz` 카메라는 SO-101 워크스페이스 측면뷰로 설정.

### M3 — FSM 풀 사이클
```bash
# WSL2 (isaac_pick_place 스택 기동 상태에서 별도 셸)
ros2 launch so101_moveit_config pick_place_orchestrator.launch.py
#   HOME→큐브별(approach/descend/close+attach/lift/transport/place/release+detach/retreat)→DONE
#   로그: N/4 placed
```
런타임 검증 지점: moveit_py PlanningScene `apply_collision_object`/`process_attached_collision_object`
메서드명, `MultiPipelinePlanRequestParameters` 의 cumotion 세트 동작, cumotion planner_id.

### M4 — 튜닝
- 5-DOF grasp 자세: `so101_pick_place_orchestrator.py` 의 `GRASP_RPY`/`GRASP_TILT_RAD`.
- 추종 정확도: `cube_desk_ros2_sim.py` `tune_drives()` stiffness/damping.
- XRDF sphere, 처리 순서, 재시도.

### cuMotion — Linux 서버 (네이티브 Ubuntu 24.04, RTX PRO 5000)
WSL2 는 cuMotion 의존성(cuda-toolkit-13 + VPI + GXF) 미충족이라 cuMotion 은 Linux 서버에서.
네이티브 Linux 는 Isaac Sim+ROS2+cuMotion 한 그래프 → DDS 경계도 없음(NVIDIA 레퍼런스 동일).
```bash
# Linux 서버: Isaac ROS apt repo (release-4.4) + CUDA 13 repo 추가 후
sudo apt-get install -y ros-jazzy-isaac-ros-cumotion ros-jazzy-isaac-ros-cumotion-moveit
# isaac_ros_cumotion_planning.yaml 을 so101_moveit_config/config/ 로 복사
cp $(ros2 pkg prefix isaac_ros_cumotion_moveit)/share/isaac_ros_cumotion_moveit/config/isaac_ros_cumotion_planning.yaml \
   <repo>/ros2_ws/src/so101_moveit_config/config/
# moveit_py_config.yaml 의 pipeline_names 에 "isaac_ros_cumotion" 추가
colcon build --packages-select so101_moveit_config --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
# 실행 (cuMotion 활성)
ros2 launch so101_bringup isaac_pick_place.launch.py use_cumotion:=true
ros2 launch so101_moveit_config pick_place_orchestrator.launch.py use_cumotion:=true
```
M1/M2 의 Win↔WSL2 DDS 가 불안정해도 동일하게 Linux 서버 이전이 해법.

## 알려진 검증 포인트 (blind 작성 → 런타임 확인 필요)
- OmniGraph 노드 타입명 (Isaac Sim 5.1 `isaacsim.*`) — M1
- topic_based plugin 명 (`topic_based_ros2_control/TopicBasedSystem` vs 신버전) — M0/M1
- cumotion launch arg 이름 (`cumotion_planner.*` vs `cumotion_action_server.*`) — M2
- moveit_py collision/attach API 메서드명 — M3
- XRDF collision sphere 정밀도 — M2
```
