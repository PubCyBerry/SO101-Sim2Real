# 실기기 SO-101 Scripted-Expert Grasp 파이프라인 (진행 문서)

> 작성: 2026-06-10 세션. WSL2(Ubuntu-24.04, `networkingMode=mirrored`) + ROS 2 Jazzy + usbipd-win.
> 목적: 실기기 SO-101 follower 에서 vision+IK 기반 scripted-expert 로 큐브 grasp 데모를 자율 생성 → LeRobot v3.0 녹화.
> 상위 맥락: 학습된 VLA 성공률 ~10% → 재학습용 양질 데이터 필요. 전략 = 시뮬 대량(오라클) + 실기기 소량(sim2real 보정), 소규모 검증 먼저. 이 문서는 **실기기 소량** 경로.

---

## 0. 불변 계약 (North Star)

action/state 6-dim joint position `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` · `observation.images.{top,wrist,front}` 480×640×3 fps30 · robot_type `so_follower` · codebase `v3.0` · task `"pick up the cube and place it in the bowl"`.

## 1. 시스템 아키텍처

```
Windows 11 (arm/카메라 USB 물리 연결)
  └─ usbipd-win (USB/IP over TCP) ──► WSL2 Ubuntu-24.04
        ├─ /dev/so101_follower (ttyACM0)   ← feetech serial
        ├─ /dev/cam_{top,wrist,front}      ← UVC 카메라 3대
        └─ ROS 2 Jazzy 그래프 (FastDDS, env.sh)
             ├─ follower_moveit_demo.launch (ros2_control + move_group)
             │    ├─ feetech_ros2_driver (hardware interface, 25Hz)
             │    ├─ arm_trajectory_controller (5축 FollowJointTrajectory)
             │    ├─ gripper_controller (ParallelGripperCommand)
             │    └─ move_group (pick_ik, OMPL/Pilz)
             ├─ cameras_cv2.launch (cv2_camera_publisher, 3캠 ~25fps)
             └─ rosbridge_server :9090  ← ros-mcp(Claude) 제어/관측
```

**역할 분담 (검증된 신뢰성 기반)**
| 작업 | 경로 | 비고 |
|---|---|---|
| joint_states / `/compute_ik` / `/compute_fk` / trajectory | native rclpy(`so101_cal.py`) 또는 ros-mcp | 둘 다 동작(serial 안정 시) |
| 큐브/그리퍼 검출 | native cv2(`detect_*.py`) | 카메라 캡처는 `cap.py` |
| 카메라 이미지 cross-process | **FastDDS(SHM) 필수** | CycloneDDS 는 921KB raw Image 전달 실패 |
| 에피소드 녹화 | launch-embedded **rosbag2** → 오프라인 변환 | joint_states+3캠 캡처 검증됨 |

## 2. 진행 상황 요약

| 단계 | 상태 | 내용 |
|---|---|---|
| Step 0 스택 기동 + 데이터경로 | ✅ done | follower+move_group+3캠+rosbridge, FastDDS. native/ros-mcp/rosbag2 수신 검증 |
| 플랫폼 안정화(serial/IK/카메라) | ✅ 대부분 done | 아래 §3 — 단 USB-IP **init/motion flakiness 잔존**(§5) |
| Step 1 hand-eye 호모그래피 캘리브 | 🔄 in-progress | IK·이동·검출 동작. sweep 1회는 motion-중 serial corruption 으로 FK freeze(§5) → 재시도 필요 |
| Step 2 단일 큐브 IK grasp | ⏳ planned | hover→wrist visual servo→descend→천천히 close→lift |
| Step 3 LeRobot v3.0 녹화+변환 | ⏳ planned | rosbag2 → `rollout_to_lerobot.py` 스키마 재사용 변환 |
| Step 4 스크립트 영속화+문서 | 🔄 진행(본 문서) | 스크립트 repo 승격, 문서/커밋 |

## 3. 해결한 핵심 문제 (시행착오 포함)

### 3.1 하드웨어 deactivate (feetech read 1회 실패 → 전체 죽음) — ✅ 해결
- **현상**: 컨트롤러가 잠깐 active 됐다가 한 번의 read timeout/checksum 오류로 hardware+3컨트롤러 전체 deactivate. 수 분~시간 내 1회만 발생해도 팔이 죽음(이전 세션 "deactivate" 의 진짜 원인).
- **시행착오**: serial timeout 5→50→**250ms** 확대(부족) → 연속실패 임계 10→100 ride-through(burst 가 임계 초과 시 여전히 죽음, 폐기).
- **해결**: `feetech_ros2_driver::read()` 가 실패 시 ① `communication_protocol_->flush_input()`(=`SerialPort::flashInputBuffer()` 노출, 입력버퍼 flush 로 재동기화) ② **항상 `return_type::OK`(마지막 상태 유지, 절대 deactivate 안 함, ride-through)**. 진짜 단선은 joint_states freshness 로 외부 감지.

### 3.2 5-DOF IK reach (전방 큐브 `/compute_ik` -31 거부) — ✅ 해결
- **현상**: pick_ik `/compute_ik` 가 현재 자세 XY column 외 전방 좌표를 전부 `-31`(NO_IK_SOLUTION). 원인 = 5-DOF 는 임의 6-DOF pose(위치+방향 동시) 를 thin manifold 에서만 만족.
- **시행착오**: `rotation_scale` 0.5→0.05(오히려 악화 — 근사해도 orientation tolerance 검증에 걸림) → `approximate: true`(도달 불가 타겟에 **위치 수십 cm 오차 근사해를 success 로 반환** → 검증서 achieved 가 타겟과 30cm 차이로 발각, 무용).
- **해결**(`kinematics.yaml`): `orientation_threshold: 3.15`(≈π, 방향 무시) + `rotation_scale: 0.02`(position 우선) + `position_threshold: 0.01` + **`approximate: false`**. → 전방 타겟 IK 성공(probe 전부 ok, achieved 가 타겟과 일치). orientation 은 best-effort down-ish.
- **참고**: RViz Plan&Execute(MoveGroup planning)는 `/compute_ik` 서비스보다 관대해 원래 동작했음.

### 3.3 USB-IP serial corruption (joint_states garbage) — ✅ 부분 해결 (§5 잔존)
- **현상**: feetech read 가 **프레임 시프트된 잘못된 joint 값**(checksum 통과하나 joint↔value 어긋남; 명령 안 한 gripper 가 -1.098, FK 가 엉뚱한 좌표)을 간헐 반환. joint_states 를 신뢰 못해 FK/IK seed·achieved 검증·캘리브 페어 오염.
- **근본 원인**: usbipd-win 의 USB/IP 는 **TCP 로 WSL 에 전달**되는데 ① **Hyper-V 방화벽**이 그 스트림을 간섭(인바운드 차단) ② `sync_read` 가 응답 servo ID 를 검증 안 해 바이트 드롭으로 응답이 한 servo 밀려도 per-servo checksum 만 통과.
- **해결(3겹)**:
  1. **Hyper-V 방화벽 해제**(유저): `Set-NetFirewallHyperVVMSetting -DefaultInboundAction Allow` → timeout/checksum 실패 급감.
  2. **ride-through + flush**(§3.1).
  3. **`sync_read` servo ID 검증**(`communication_protocol.hpp`): `response_buffer[0] != ids[i]` → 거부 → ride-through flush+skip 으로 재동기화. frame-shift corruption 차단.
- **결과**: 정지 상태에서 joint_states 일관 clean(3회 연속 동일·타겟 일치), serial 실패는 ride-through 흡수(deactivate 0).

### 3.4 DDS 선택: FastDDS vs CycloneDDS — 조사·결정
- 가이드(유저 제공)는 CycloneDDS 권장(ros2cli 동작·throughput). 그러나 **실측**: CycloneDDS(lo unicast, MaxMessageSize/rmem 튜닝)는 921KB raw Image 를 **native·rosbridge 양쪽 모두 0 전달**(joint_states 같은 작은 msg 만 됨) → 카메라 불가. FastDDS(SHM/localhost)만 이미지 cross-process 전달. 가이드 §6 도 raw 이미지엔 FastDDS 권장.
- **serial corruption 은 DDS 무관**(usbipd TCP/방화벽 문제)이므로 카메라 위해 **FastDDS 고정**(`env.sh`). (FastDDS 의 ros2cli daemon 버그(#934)로 native `ros2 node list` 가 빈값이나, 그래프 관측은 ros-mcp 로 우회.)

## 4. 검증된 결과
- 제어 파이프라인: ros-mcp/native `/compute_ik`(manipulator, gripper_frame_link) → `send_action_goal /follower/arm_trajectory_controller/follow_joint_trajectory`(5축 단일점+velocities) → 실기기 이동 성공("Goal successfully reached!"), 카메라·FK 로 검증.
- 큐브 검출: `detect_cubes.py` HSV(S<55,105<V<215)+면적/aspect 로 top 4/4 정확.
- 데이터 경로: rosbag2 가 joint_states 50Hz + 3캠 25fps 캡처(`ros2 bag info` 확인).
- 안정성: 방화벽+ID검증 적용 후 정지 상태 75s soak 에서 deactivate 0.

## 5. 알려진 한계 / 잔존 이슈 (다음 세션 우선)
- **USB-IP serial 의 init/motion flakiness**: 방화벽+드라이버 보강으로 **정지 상태**는 안정되나, ① 하드웨어 `configure_joints_`(init) 첫 read 가 간헐 timeout → "Failed to initialize hardware"(재시작 재시도로 회피, 간헐적) ② **motion 중(sync_write+read 동시)** half-duplex 버스 + USB-IP 레이턴시로 read 가 sustained checksum 실패 → ride-through 가 stale 유지 → 캘리브 sweep 의 per-pose FK 가 freeze(첫 sweep 9-pose 전부 FK 동일). `update_rate` 50→25Hz 로 낮췄으나 검증 미완.
- **방향**: ⓐ `update_rate` 추가 하향(20/10Hz)·motion 후 read-recovery 폴링(clean read 나올 때까지 대기 후 기록) ⓑ 팔을 **네이티브 Linux 박스에 직결**(USB-IP 제거 = 가장 확실) ⓒ 캘리브를 정지-측정-정지 방식으로(이동 완료 후 충분 settle + 다중 read median).
- **5-DOF grasp 신뢰성**: 시뮬 오라클에서도 hard(blocked 이력). 실기기는 wrist visual servo(closed-loop)로 보완 기대하나 미검증.
- **캘리브 편향**: 그리퍼 tip≠EE frame. 거친 호모그래피 + wrist servo 보정 구조라 수용.

## 6. 다음 단계 (planned)
1. serial motion-stability 확보(§5 ⓐ/ⓑ/ⓒ 중 택).
2. Step 1 캘리브: 정지-측정 방식으로 6~9 pose 의 (top tip 픽셀 ↔ FK base XY) 수집 → `cv2.findHomography` → `top_homography.json`(재투영 잔차 ≲1cm 검증).
3. Step 2 grasp: top 검출→H→base XY→hover IK→wrist visual servo XY 보정→descend→**천천히** close(빠르면 큐브 못 쥠, 시뮬 교훈)→lift.
4. Step 3 녹화: launch-embedded rosbag2 → `scripts/real/rosbag2_to_lerobot.py`(신규, `scripts/sim/rollout_to_lerobot.py` 의 스키마·단위변환 rad→deg·gripper×31.75 재사용) → `validate_lerobot_schema.py` PASS.

## 7. 실행 방법 (런북)

모든 노드는 `source ros2_ws/setup/env.sh`(FastDDS) 한 깨끗한 셸에서 기동. WSL 호출은 PowerShell 권장(Git Bash 는 `/mnt/c` MSYS 변환; PowerShell 은 inline `>`/`2>`/`$()`/`&&` 가 가로채짐 → 스크립트 파일 패턴 사용).

```bash
# 0) USB 연결 (PowerShell, wsl --shutdown 후엔 재연결 필요)
usbipd attach --wsl --busid 4-1     # arm
usbipd attach --wsl --busid 1-9     # cam (top/wrist/front: 1-9,5-1,6-4)
usbipd attach --wsl --busid 5-1
usbipd attach --wsl --busid 6-4

# 1) 노드 (각각 별도 셸, env.sh source)
ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=real use_rviz:=false use_cameras:=false
ros2 launch so101_bringup cameras_cv2.launch.py
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090
# 컨트롤러 init 실패("Failed to initialize hardware") 시 follower 재시작(간헐적, 재시도로 통과)
```

## 8. 스크립트 인덱스 (`ros2_ws/src/so101_bringup/scripts/realdevice/`, WIP)
- `so101_cal.py` — native IK/FK/move 헬퍼: `state` / `iktest X Y Z` / `move X Y Z [Q] [dur]` / `movej j0..j4` / `nudgez dz`. joint delta 안전검사 포함.
- `so101_calib.py` — top 호모그래피 캘리브(IK 격자): `probe`(도달성, 모션 없음) / `run`(이동→FK→tip 검출→페어+주석이미지).
- `so101_jcalib.py` — joint-space FK 캘리브(IK 불필요): `jplan`(FK 사전계산) / `jrun`.
- `cap.py` / `detect_cubes.py` / `detect_gripper.py` — 3캠 캡처 / 큐브·그리퍼 픽셀 검출.
- `so101_run.sh`(env source 래퍼) / `so101_kill.sh`(self-kill 회피 종료) / `so101_build.sh`(feetech 재빌드) / `so101_check.sh`(검증).
- ⚠️ WIP: 경로(`/mnt/c/.../Temp`)·repo 경로 하드코딩. 정식화 시 인자/상대경로화 필요.
