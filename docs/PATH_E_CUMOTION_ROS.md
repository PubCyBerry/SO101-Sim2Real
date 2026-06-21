# PATH E — cuMotion + ROS 2 로 SO-101 cube_desk pick-and-place

> 이 문서는 별도 cuMotion/MoveIt 실험 경로이며 기존 Isaac Sim 5.1 bridge를 유지한다.
> Canonical parity 실행은 [`PATH_F_CANONICAL_PARITY.md`](PATH_F_CANONICAL_PARITY.md)를 따른다.

Isaac Sim 이 `cube_desk` 씬을 시뮬하고, **NVIDIA cuMotion**(GPU collision-free 모션 플래너)을
**MoveIt 2** 에 붙여 ROS 2 로 SO-101 5DOF 팔을 제어해 4개 큐브를 그릇에 담는 경로.

> **cuMotion = cuRobo + MoveIt/ROS 래퍼** (코어 플래너 동일). cuRobo-직접 트랙(PICKCUBE)과의 관계·
> 역할 분담·warp 분리 패턴(ZMQ vs ROS DDS, 둘 다 같은 D10 이유)은 `PICKCUBE_CUROBO_PROJECT.md` §15 참조.
> 요지: 데이터 생성=cuRobo 배치(직접), 실기기 제어=cuMotion(여기). 5-DOF·grasp 한계는 둘 다 동일.

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

---

## 7. VLA 추론 (ROS)

학습된 VLA(SmolVLA/ACT, Docker `policy-server`)로 sim SO-101 팔을 구동한다. cuMotion/MoveIt
은 쓰지 않고(VLA 가 joint target 을 직접 냄), §1 의 Isaac↔ROS 토픽 계약을 그대로 재사용한다.

### 7.1 아키텍처 (3 프로세스)

```
[호스트 server, isaac venv py3.11]              [vla-ros 컨테이너 py3.12]          [Docker policy-server]
 run_cube_desk_ros_bridge.py (상주)              so101_vla_policy 노드               lerobot 0.5.1 (gRPC)
  · joint_states/clock/tf (기존)        ── /isaac_joint_states ──▶  obs 수집
  · +카메라 3대 publish (신규)          ── /camera/{top,wrist,front}/image_raw ──▶  state rad→LeRobot deg
    OmniGraph ROS2CameraHelper                                       이미지 cv_bridge(rgb8)
  · ArticulationController              ◀── /isaac_joint_commands ──  VLA gRPC ──▶◀── action chunk
    (joint cmd 직접 적용)                                            action(LeRobot)→rad → publish
  (번들 jazzy lib, 호스트 ROS 불필요)    UDPv4 fastrtps, network host
```

- **부팅 분리**: Isaac Sim 은 한 번 띄워 상주(부팅 느림). 추론 클라(컨테이너)는 재시작 가벼움.
- **단위 책임**: bridge·ArticulationController 는 sim rad. LeRobot 단위 변환(arm deg/그리퍼[0,100])과
  VLA gRPC 는 `so101_vla_policy` 노드가 전담. 전처리(rename/resize/normalize/추론)는 policy-server.
- **런타임 분리**: rclpy(Jazzy=py3.12) ↔ lerobot(uv venv=py3.11, Isaac 핀). 그래서 VLA 노드는
  **별도 py3.12 컨테이너**(`Dockerfile.vla_ros`)에서 lerobot 을 pip 설치해 돈다.

### 7.2 토픽 계약

| 토픽 | 타입 | 방향 | 내용 |
|---|---|---|---|
| `/isaac_joint_states` | sensor_msgs/JointState | bridge→노드 | 6관절 rad (name 기준 SO101 순 재정렬) |
| `/camera/{top,wrist,front}/image_raw` | sensor_msgs/Image | bridge→노드 | rgb8 480×640 (gym 과 동일 포즈/focal) |
| `/isaac_joint_commands` | sensor_msgs/JointState | 노드→bridge | 6관절 target rad (name=SO101 순) |

### 7.3 설치 내역 (`docker/Dockerfile.vla_ros`)

호스트는 건드리지 않는다 — 모두 컨테이너 격리. base `ros:jazzy-ros-base`.

- **apt**: `ros-jazzy-cv-bridge`(이미지 디코드), `ros-jazzy-rmw-fastrtps-cpp`(DDS),
  `python3-pip`, `python3-colcon-common-extensions`, `git`.
- **pip(py3.12, `--break-system-packages`)**: `torch`(CPU 휠 — action chunk torch.Tensor unpickle),
  `grpcio`, `protobuf>=6.31`(vendored pb2 runtime 검사), `python-dotenv`, `numpy<2`(cv_bridge ABI 보호).
  **실 lerobot 은 설치 안 함** — import 체인이 transformers/datasets/diffusers 까지 끌어와 pip
  resolve 가 폭발(imageio backtracking). 대신 `ros2_ws/src/so101_vla_policy/vendor/lerobot/` 에
  gRPC pickle 호환 최소 shim(`async_inference/helpers.py` dataclass + `transport/` pb2 복사·utils
  재구현)을 두고 entrypoint 가 PYTHONPATH 로 잡는다. 실 lerobot 0.4.4 와 **양방향 pickle 호환 검증됨**
  (RemotePolicyConfig/TimedObservation 송신·TimedAction 수신).
- DDS: `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` + `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`(host↔container
  cross-UID SHM 회피, §1·`run_cube_desk_ros_bridge.sh` 와 동일).
- compose 서비스 `vla-ros`: `network_mode/ipc: host`, repo 를 `/workspace` 로 마운트,
  `.env`+`env/<POLICY_PROFILE>.env` env_file 주입. 진입점 `docker/vla-ros-entrypoint.sh`
  가 `so101_vla_policy` colcon build → `ros2 launch`.

### 7.4 실행 순서

```bash
# ① policy-server (메인 루트). POLICY_PROFILE=smolvla|act, POLICY_REPO_ID=fine-tuned 확인.
#   (a) 표준 Async Inference:
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server
#   (b) ★ Async Inference + RTC(Real-Time Chunking) 가이던스 — flow-matching(SmolVLA) 권장:
#       CMD 를 policy-server-rtc 로 override. scripts/policy_server_rtc.py(RTCPolicyServer) 기동.
#       RTC_EXECUTION_HORIZON=8 / RTC_MAX_GUIDANCE_WEIGHT=10.0 / RTC_PREFIX_ATTENTION_SCHEDULE=EXP (.env §6).
#       서버측만 바뀌고 클라/bridge 동일. 로그에 `[RTC] chunk #N | guidance ✅ | leftover=… (horizon=8)`.
docker compose --env-file .env -f docker/docker-compose.yaml run --rm policy-server policy-server-rtc

# ② Isaac Sim bridge 상주 (호스트, 카메라 publish 포함 — 기본 on). 시각 관전 livestream:
#   PUBLIC_IP=<ip> LIVESTREAM=1 scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 4 --livestream 1  (WebRTC :49100)
scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1

# ③ VLA policy-client (컨테이너, so101_vla_policy ROS 노드).
#   ⚠ policy-server 는 non-RTC(compose 기본 `policy-server`) 권장 — 서버측 RTC 오적용으로 sim action
#     오염(docs/TROUBLESHOOTING.md). ⚠ sim 모델은 그리퍼 GRIPPER_CMD_OFFSET=0.2 필요(use_default_offset
#     재적용) — env/smolvla.env 에 박혀 자동. 실기기 모델이면 0.
docker compose --env-file .env -f docker/docker-compose.yaml run --rm vla-ros
```

- 정책 파라미터(`POLICY_SERVER_ADDRESS`/`POLICY_TYPE`/`POLICY_REPO_ID`/`ACTIONS_PER_CHUNK`/
  `TASK`/`RENAME_MAP`/`POLICY_DEVICE`/`CHUNK_SIZE_THRESHOLD`)는 실기기 policy-client 와 **동일한**
  `.env`+프로필에서 읽음. ROS param override 가능(`config/vla_policy.yaml`).
- **실기기 전환**: 같은 노드를 토픽 remap + action sink shim 으로 재사용 —
  launch 에 `use_shim:=true joint_states_topic:=/follower/joint_states` 전달. shim
  (`joint_command_to_trajectory`)이 `/isaac_joint_commands`(JointState)→ arm
  FollowJointTrajectory + gripper GripperCommand 액션으로 변환(실기기 controller 이름은 param 으로 맞춤).

### 7.5 검증

1. **카메라 토픽**: bridge 상주 후 `ros2 topic hz /camera/top/image_raw`(~30Hz),
   `ros2 topic echo --once /camera/top/image_raw`(width 640 height 480 encoding rgb8).
2. **노드 단독**(policy-server 없이): `/isaac_joint_states`+이미지 수신, `/isaac_joint_commands` 형식.
3. **단위 라운드트립**: `so101_vla_policy/units.py` `from_lerobot_units(to_lerobot_units(x))≈x`
   (검증 완료, gripper 100°↔1.745rad).
4. **풀 파이프라인**: ①②③ → 상주 Isaac 창에서 팔이 추론 action 으로 구동.

### 7.6 알려진 함정

| 함정 | 메모 |
|---|---|
| **카메라 prim 부착** | bridge robot USD link prim 명(`gripper`/`shoulder`)이 gym 과 같다고 가정. 다르면 `[bridge] WARN: camera parent prim 없음` 후 skip → prim 경로 수정 |
| **numpy ABI** | cv_bridge(apt)=system numpy 1.26 빌드. torch 가 numpy 2.x 로 올리면 import 깨짐 → `numpy<2` 핀(Dockerfile 반영) |
| **vendored lerobot** | 실 lerobot 미설치 — `vendor/lerobot/` shim. server(0.5.1) 의 RemotePolicyConfig/TimedObservation/TimedAction 필드가 바뀌면 shim 도 갱신. pb2 는 `lerobot/transport/services.proto` 변경 시 재복사. 양방향 pickle 호환 검증됨 |
| **${HF_USER} 미보간** | host 에서 `POLICY_REPO_ID=${HF_USER}/…` 미해결 시 노드 경고 → param `pretrained_name_or_path` 지정 |
| **추론 모델이 profile 값으로 고정** | 노드가 시작 시 `.env`+`env/<profile>.env` 를 os.environ 재로드(override)해 `docker run -e POLICY_REPO_ID=<내 모델>` 을 **profile 의 `POLICY_REPO_ID` 로 덮음**. 추론 모델은 ROS param `pretrained_name_or_path`(`config/vla_policy.yaml`, 최우선·env reload 무관)로 고정. 확인=노드 로그 `sent instructions (model=...)`. (예: sim 모델 `taehunkim/so101_smolvla_sim_pick_cube`) |
| **카메라 convention** | gym TiledCamera offset(convention="world")와 동일 view 위해 bridge 가 world→opengl 변환 후 USD prim author |

### 7.7 실기기 배포

같은 `so101_vla_policy` 노드를 **코드 변경 없이** 실기기에 재사용한다. sim 과 차이는
obs 토픽 remap + **action sink shim**(joint target → controller 액션) 둘뿐이다. 실기기
카메라는 `cv2_camera_publisher.py` 가 이미 `/camera/{top,wrist,front}/image_raw` 로
publish 하므로 토픽 이름이 그대로 맞는다.

```
실기기 ROS(PATH D)                              vla-ros / VLA 노드                policy-server
 feetech_ros2_driver → /follower/joint_states ─▶  obs 수집                          (gRPC, fine-tuned)
 cv2_camera_publisher → /camera/*/image_raw ───▶  state(rad→deg)·이미지 ──gRPC──▶
 arm_trajectory_controller (FollowJointTrajectory) ◀── shim ◀── /isaac_joint_commands ◀── action chunk
 gripper_controller (GripperCommand)             ◀──┘ (JointState→액션 변환)
```

**실행 순서**

```bash
# ① 실기기 ROS 스택 (PATH D — 실기기 머신/WSL2)
ros2 launch so101_bringup follower_split.launch.py     # feetech 드라이버 + arm/gripper 컨트롤러
ros2 launch so101_bringup cameras_cv2.launch.py        # /camera/{top,wrist,front}/image_raw

# ② policy-server (동일, fine-tuned 모델)
docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server

# ③ VLA 노드 + action sink shim — obs remap + 실기기 컨트롤러 액션 이름 지정
ros2 launch so101_vla_policy vla_policy.launch.py \
    use_shim:=true \
    joint_states_topic:=/follower/joint_states
#   shim 컨트롤러 이름이 다르면 config/vla_policy.yaml 또는 -p 로:
#     arm_action:=/follower/arm_trajectory_controller/follow_joint_trajectory
#     gripper_action:=/follower/gripper_controller/gripper_cmd
```

> `vla-ros` 컨테이너로 띄울 경우 entrypoint 에 위 `ros2 launch … use_shim:=true …` 인자를
> compose `command` 로 넘긴다. 단 컨테이너가 실기기 ROS 그래프에 도달하려면 동일 DDS
> domain·네트워크여야 한다(머신이 다르면 `ROS_DOMAIN_ID`/디스커버리 정합).

**sim ↔ real 차이 한눈에**

| 항목 | sim | real |
|---|---|---|
| joint state | `/isaac_joint_states` | `/follower/joint_states` (remap) |
| 카메라 | bridge OmniGraph publish | `cv2_camera_publisher` (토픽 동일) |
| action sink | `/isaac_joint_commands`→ArticulationController 직접 | **shim**→FollowJointTrajectory + GripperCommand 액션 |
| 실행 | `up vla-ros` (기본) | `use_shim:=true` + remap |

**검증 필요 / 주의**

| 항목 | 확인 |
|---|---|
| joint 단위 | feetech 드라이버 joint_states = rad 가정(`to_lerobot_units` 전제). 그리퍼 [0,100] 매핑 재확인 |
| 이미지 encoding | 실기기 bgr8 → 노드가 `rgb8` 요청, cv_bridge 자동 변환 (OK) |
| shim 컨트롤러 이름 | 실기기 `so101_bringup` controller 설정과 정확히 일치(기본값=PATH D/E 관례) |
| 네트워크/DDS | 실기기 ROS(WSL2)와 VLA 노드 같은 domain·도달 가능. UDPv4 디스커버리 |
| 안전 | 노드 `clamp_joint_rad` + shim throttle(`max_rate`). 첫 구동 저속·근접감시 |

> 참고: 실기기엔 ROS 없는 VLA 경로(`docker policy-client`, feetech serial 직결)가 이미
> 동작한다(더 간단). 이 ROS 경로의 이점은 sim·real **동일 토픽 인터페이스** + 향후
> MoveIt/cuMotion(§1~6) 통합 가능성이다.

### 7.8 sim 렌더 뷰 차이 (bridge vs teleop) — ⚠ 미해결

**현상**: bridge 로 띄운 Isaac Sim GUI 가 teleop(gym) 화면과 다르다.

| 요소 | bridge (현재) | teleop (gym, 기준) |
|---|---|---|
| 바닥 | 파란 타일 그리드 | 회색 그리드 |
| 조명 | KeyLight 강한 그림자·고대비(풀 RTX) | 평탄·밝음·그림자 약함 |

**원인**:
- 바닥 = bridge `isaacsim.core.api.World.scene.add_default_ground_plane()`(파란 타일) vs teleop Isaac Lab `GroundPlaneCfg`(회색). ground asset 자체가 다름.
- 조명/명암 = bridge 는 순수 `isaacsim.core.World` 기본 render, teleop 은 Isaac Lab `ManagerBasedRLEnv` render 프리셋. **같은 cube_desk DomeLight/KeyLight USD** 인데 exposure/tonemap 차이로 bridge 만 KeyLight 하드 그림자가 드러남.
- bridge 가 순수 World 를 쓰는 이유 = Isaac Lab GPU fabric ↔ OmniGraph view 충돌(device -1) 회피(§구현노트 B안).

**핵심 구분**: 바뀐 건 **GUI Perspective viewport**. VLA 가 보는 **obs 카메라(top/wrist/front)는 별도 render product** 라 GUI 와 별개다. 따라서 우선순위:
1. obs 프레임이 실제로 다른지 확인 — `/camera/{top,wrist,front}/image_raw` 캡처 vs teleop `c`-키 캡처(`outputs/captured_images`) 비교.
2. obs 가 같으면 GUI 차이는 무시 가능(미관).
3. obs 가 다르면 **sim 분포 shift** → 정렬 필요.

**정렬 계획(obs 차이 확인 시, 미착수)**:
- ground plane 을 teleop 과 동일(회색 Isaac Lab `GroundPlaneCfg`)로 교체 or 제거.
- 조명/exposure/render mode 를 gym env 프리셋에 맞춤(`env_cfg.sim.render` 상당값을 bridge World render 설정에 반영).
- front 카메라 포즈는 `_FRONT_CAM_LOCAL_POS`(현재 전방 +2cm 적용) 공용 상수 — bridge·gym 동시 반영.

**상태**: 미해결. obs 프레임 비교 결과 대기 → 차이 시 정렬 작업 착수.

새 종류 에러를 해결하면 `docs/TROUBLESHOOTING.md` 에 기록(운영 규칙).
