# PATH E — cuMotion + ROS 2 로 SO-101 cube_desk pick-and-place

Isaac Sim 이 `cube_desk` 씬을 시뮬하고, **NVIDIA cuMotion**(GPU collision-free 모션 플래너)을
**MoveIt 2** 에 붙여 ROS 2 로 SO-101 5DOF 팔을 제어해 4개 큐브를 그릇에 담는 경로.

- NVIDIA [Isaac ROS pick-and-place 튜토리얼](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/tutorials/pick_and_place/tutorial_pick_and_place.html) 구조를 따르되 **perception 은 생략**하고 시뮬 ground-truth 물체 포즈를 쓴다.
- 기존 in-process Lula IK SM(`scripts/environments/pick_cube_state_machine.py`)의 grasp 미완 원인(Lula↔USD 좌표 정합 잔차)을 **cuMotion 이 articulation frame 에서 직접 계획**해 구조적으로 제거한다.
- PATH C(Isaac Lab 시뮬)·PATH D(WSL2 MoveIt 실기기)와 독립. ROS 스택은 `ros2_ws/` 에 내재화.

> **검증 상태(2026-06-09, 서버 konan147)**: Docker cuMotion 스택 빌드·colcon 5패키지·Isaac Sim bridge 부팅·OmniGraph 생성·토픽 광고·host↔container DDS ✅. **device -1 블로커 해소(B안)**: bridge scene 로드/시뮬 파이프라인을 순수 `isaacsim.core.api.World`(CPU) + `SingleArticulation` 으로 재작성 → `/isaac_joint_states`(6관절 값)·`/clock`·`/tf`(base_link→Cube/Bowl) 모두 정상 publish 확인(**§5 검증 1~3 통과**). 4~6(RViz dry-run → 단일/4큐브)은 컨테이너 ROS 스택 launch + cuMotion XRDF(§5 1) 검증 후 진행. 경위·환경 함정(LD_LIBRARY_PATH, cross-UID SHM→UDPv4)은 `docs/TROUBLESHOOTING.md` PATH E 항목 + `CONTEXT.md` 참조.

---

## 1. 구성 요약

| 항목 | 값 |
|---|---|
| 플랫폼 | **Linux 서버 네이티브** (Ubuntu 24.04, RTX PRO 5000 Blackwell 48GB) — ROS 2 + cuMotion + Isaac Sim 한 머신 |
| ROS | ROS 2 Jazzy + MoveIt 2 + [isaac_ros_cumotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion) (Jazzy/x86_64 공식 지원) |
| 시뮬 | Isaac Sim 5.1 + `isaacsim.ros2.bridge` (host uv, `--group isaac`) |
| 물체 인지 | 시뮬 ground-truth 포즈 publish (`/cube_poses`, `/bowl_pose`, base_link frame) |
| 모션 플래너 | cuMotion MoveIt planner plugin (URDF + **XRDF**) + OMPL fallback |
| 제어 | `topic_based_ros2_control`(TopicBasedSystem) → arm_trajectory_controller(FollowJointTrajectory) + gripper_controller(ParallelGripperCommand) |
| SM | 커스텀 ROS 2 Python 노드 (MoveItPy + cuMotion), 8단계 |

### 토폴로지

```
Isaac Sim (run_cube_desk_ros_bridge.py)               ROS 2 그래프
  pub /isaac_joint_states  ───────────────► TopicBasedSystem(state)
  sub /isaac_joint_commands ◄────────────── TopicBasedSystem(cmd)
  pub /clock                                 controller_manager
  pub /cube_poses, /bowl_pose (base_link) ┐    ├ joint_state_broadcaster → /follower/joint_states
                                          │    ├ arm_trajectory_controller (FollowJointTrajectory)
                                          │    └ gripper_controller (ParallelGripperCommand)
                                          │  move_group + cuMotion plugin + cumotion_planner_node(XRDF)
                                          └► pick_place_sm (MoveItPy manipulator + cuMotion)
```

`/isaac_joint_states`(bridge↔하드웨어)와 `/follower/joint_states`(broadcaster→MoveIt)는 **분리**해 피드백 루프를 막는다.

---

## 2. 파일 맵 (이 경로로 추가/수정한 것)

| 파일 | 역할 |
|---|---|
| `scripts/sim/run_cube_desk_ros_bridge.py` | **(신규)** Isaac Sim standalone — 순수 `isaacsim.core.World`+`SingleArticulation` 로 cube_desk + SO-101 + ROS bridge OmniGraph + 물체 TF publish (B안) |
| `scripts/sim/run_cube_desk_ros_bridge.sh` | **(신규)** 위 스크립트 런처 — LD_LIBRARY_PATH(번들 ROS 2 lib)·DDS env export |
| `assets/robots/so101.xrdf` | **(신규)** cuMotion 용 collision sphere + c-space 5축 (tool_frame=gripper_frame_link) |
| `docker/patches/cumotion_moveit_filter_start_state.patch` | **(신규)** cuMotion MoveIt 플러그인 `updateGoal` 패치 — start_state 를 planning group 관절로 필터링(c-space 6vs5 해결). Dockerfile 이 적용·빌드 |
| `scripts/sim/gen_so101_xrdf.py` | **(신규)** XRDF↔URDF 정합·FK/IK 검증 하니스 (curobo) |
| `scripts/sim/probe_ik.py` | **(신규)** grasp reachability 프로브 — 실행 중 move_group `/compute_fk` 랜덤 FK 샘플링으로 워크스페이스·achievable tilt 매핑(5-DOF planning 진단). curobo 불요 |
| `ros2_ws/src/so101_cumotion_moveit_config/` | **(신규)** cuMotion MoveIt config 패키지 (planner plugin yaml, MoveItPy config, launch) |
| `ros2_ws/src/so101_cumotion_pick_place/` | **(신규)** SM 노드 패키지 (pick_place_sm.py, params, launch) |
| `so101_ros2_control.xacro` | **(수정)** `hardware_type:=isaac` 분기 (TopicBasedSystem) |
| `so101_bringup/config/ros2_control/follower_isaac_controllers.yaml` | **(신규)** isaac variant 컨트롤러 (100Hz) |

기존 재사용: `so101_description`(URDF/SRDF tip=gripper_frame_link), `so101_moveit_config`(SRDF group `manipulator`/`gripper`, kinematics, joint_limits), `so101_bringup/follower_split.launch.py`, `cube_desk` USD + GRASP_PHYSICS 튜닝.

---

## 3. 전체 셋업 (한 번만)

### 3.1 Isaac Sim 쪽 (host uv)

```bash
uv sync --group isaac          # 기존 PATH C 와 동일. RT코어 GPU 필요(서버 Blackwell OK)
```

### 3.2 ROS 2 + cuMotion 설치

```bash
# ROS 2 Jazzy + MoveIt + ros2_control (PATH D 의 02 스크립트 재사용 가능)
sudo apt install ros-jazzy-desktop ros-jazzy-moveit ros-jazzy-ros2-controllers \
  ros-jazzy-topic-based-ros2-control ros-jazzy-tf-transformations

# isaac_ros_cumotion (Jazzy). curobo 가 CUDA 빌드되므로 GPU·CUDA toolkit 필요.
# 공식 문서대로 isaac_ros_cumotion / isaac_ros_cumotion_moveit / isaac_ros_cumotion_robot_description 설치.
#   https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/
```

### 3.3 워크스페이스 빌드

```bash
cd ros2_ws            # 또는 ext4 빌드 워크스페이스(PATH D 패턴)
colcon build --symlink-install \
  --packages-select so101_description so101_moveit_config so101_bringup \
                    so101_cumotion_moveit_config so101_cumotion_pick_place
source install/setup.bash
export SO101_REPO=/path/to/SO101-Sim2Real   # XRDF/URDF 절대경로 해결용
```

---

## 4. 실행 순서

**터미널 1 — Isaac Sim bridge** (repo 루트, host). 래퍼가 LD_LIBRARY_PATH(번들 ROS 2 lib)·DDS env(fastrtps/UDPv4)를 export 한다:

```bash
scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 4      # 단일 큐브 진단은 --num_cubes 1
```

> 호스트엔 ROS 2 가 없어 bridge 는 isaacsim 번들 ROS 2 lib 를 쓴다. 직접 `uv run python …` 호출 시
> `LD_LIBRARY_PATH` 에 `…/isaacsim.ros2.bridge/jazzy/lib` 가 없으면 ROS2 bridge startup 이 실패한다.
> host↔container 는 cross-UID SHM 충돌을 피하려 **fastrtps + `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`** 로 통일.

**터미널 2 — ROS 스택**(move_group+cuMotion, 컨트롤러, SM). bridge 와 같은 RMW/transport 로 맞춘다:

```bash
# 컨테이너로 띄울 때: docker run … --network host --ipc host \
#   -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 so101-cumotion:jazzy …
source ros2_ws/install/setup.bash          # ⚠ env.sh(cyclonedds)는 source 금지 — PATH E 는 fastrtps
export SO101_REPO=/path/to/SO101-Sim2Real
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 launch so101_cumotion_pick_place pick_place.launch.py use_rviz:=true
```

SM 노드가 `/cube_poses`·`/bowl_pose` 를 받으면 근접순으로 큐브를 집어 그릇에 담는다.

---

## 5. 검증 절차

| # | 명령 | 통과 기준 | 상태 |
|---|---|---|---|
| 1 | XRDF/URDF 유효성 (`gen_so101_xrdf.py` 는 Python curobo 필요 — 미설치) | cuMotion 이 XRDF/URDF 로 로봇 로드 | ✅ 통과(2026-06-09, cuMotion C++ planner 가 로드 — `gen_so101_xrdf.py` 대체 검증) |
| 2 | 터미널1 실행 후 `ros2 topic echo /isaac_joint_states` | 6관절 name/position/velocity/effort 값 흐름 | ✅ 통과(2026-06-09) |
| 3 | `ros2 topic echo /tf --once` (또는 `/clock`) | `base_link→Cube1/Bowl` transform·sim-time clock | ✅ 통과(2026-06-09) |
| 4 | `pick_place.launch.py` → plan/execute | plan 성공, Isaac Sim 팔 동기 | ✅ 통과(2026-06-09, joint-goal — approach→grasp→lift OMPL OK) |
| 5 | gripper action `ParallelGripperCommand` | 그리퍼 닫힘 | 🟡 controller active·명령 전달, grip 물리 미완(§6) |
| 6 | bridge `--num_cubes 1` + 전체 launch | 1/1 bowl, 이후 `--num_cubes 4` | ⬜ grasp 물리(jaw·grip) 튜닝 후(§6) |

> **2026-06-09 통합 검증 결과**: `pick_place.launch.py`(bridge + controllers + move_group + cuMotion + SM)가
> 서버에서 end-to-end 기동. bringup 4대 함정·c-space 6vs5 해소 후, **5-DOF planning 블로커를 근본 해결**:
> MoveIt/cuMotion 의 goal 샘플러가 5-DOF 의 task-space(pose/position) goal 을 못 푼다(랜덤 orientation+IK
> 방식 → 거의 모든 orientation 도달 불가). SM `_move_to` 를 **joint-goal(FK 샘플링+set_from_ik)** 로 전환해
> approach→grasp→lift 가 전부 `OMPL OK`(plan+exec). **남은 블로커**: grasp 물리(grip 이 큐브를 못 쥠 —
> moving_jaw 가 큐브 위·그리퍼 미완전 닫힘) — §6 "grasp 물리". 헤드리스 서버라 RViz 대신 SM 자동 실행으로 검증.

검증 결과는 `CONTEXT.md` 작업 인계에 기록한다.

---

## 6. 튜닝·알려진 제약

| 항목 | 메모 |
|---|---|
| **XRDF sphere** | `assets/robots/so101.xrdf` 의 sphere 반경/중심은 **근사 초기값**. Isaac Sim cuMotion *Robot Description Editor* 로 메시에 맞춰 튜닝 후 갱신. self-collision 오류 시 `ignore` 쌍·반경 조정 |
| **cuMotion c-space 6vs5 (해결)** | MoveIt start_state 6관절(arm5+gripper) ≠ cuMotion cspace 5축 → `INVALID_INITIAL_CSPACE_POSITION`. upstream 버그(issue #10). `docker/patches/cumotion_moveit_filter_start_state.patch` 가 `updateGoal` 에서 start_state 를 planning group(5)으로 필터링. Dockerfile 이 `/opt/cumotion_overlay` 빌드. (TROUBLESHOOTING 참조) |
| **5DOF planning (해결)** | MoveIt/cuMotion goal 샘플러는 task-space goal 을 "랜덤 orientation+IK"로 풀어 5-DOF 비가능(orientation 제거해도 실패). SM `_move_to` 를 **joint-goal** 로 전환: `RobotState.set_to_random_positions()` in-process FK 샘플링으로 target 도달 config 탐색(`_fk_sample_goal`) + `set_from_ik` 정밀화 → `set_goal_state(robot_state=)`, planner 는 joint→joint 만. `scripts/sim/probe_ik.py`(`/compute_fk` 워크스페이스 프로브)로 진단. param: `fk_samples`/`fk_pos_gate`. (TROUBLESHOOTING 참조) |
| **grasp 물리 (남은 블로커)** | 팔이 큐브 도달은 하나 grip 이 큐브를 못 쥠 — moving_jaw 가 큐브 위(TCP만 근처)·그리퍼 미완전 닫힘(0.086 vs -0.16). 완전 top-down 불가라 강tilt 로 moving_jaw 를 큐브 옆/아래로 내려야(in-process SM 의 known-hard, ~1.4/4). 다음: FK 샘플 목적함수에 jaw-z 점수(`_finger_min_z`式)·그리퍼 close dwell/force·`fk_pos_gate` 강화·`set_from_ik` 정밀도. `grasp_tilt_deg`/height(LOW band)·`grasped_dz` 로 조정 |
| **cuMotion 실패 fallback** | SM 이 OMPL(`ompl_rrtc`)로 자동 재시도. cuMotion plugin = `isaac_ros_cumotion_moveit/CumotionPlanner`(검증됨), action server 정상 로드 |
| **그리퍼 5-DOF 무관** | cuMotion 은 5축 팔만 계획(패치도 manipulator group 필터), 그리퍼 open/close 는 `gripper_controller`(ParallelGripperCommand action)로 별개 제어. tool z 회전 자유는 대칭 큐브엔 무해 |
| **sim-time** | bridge `/clock` + 모든 ROS 노드 `use_sim_time:=true`. MoveIt 실행 timeout 여유 필요 |
| **그릇 내부 미끄러움** | 큐브가 곡면 타고 바닥 중앙으로 정착(의도된 물리). `place_height`·`stack_increment` 로 낙하 충격 완화 |
| **그리퍼 접촉 물리** | GRASP_PHYSICS 튜닝값(USD author, mass/contactOffset/friction) 유지. `gripper_dwell_s` 로 정착 |
| **프레임** | MoveIt virtual_joint(world→base_link)=identity → 모든 포즈를 base_link frame 으로 통일. bridge 가 robot base 기준으로 빼서 publish |

새 종류 에러를 해결하면 `docs/TROUBLESHOOTING.md` 에 기록(운영 규칙).
