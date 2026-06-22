# SO-101 Canonical Parity / Isaac Sim 6 / ROS 2 Jazzy 마이그레이션 보고서

> 기준일: 2026-06-22
> 상태: software path 완료, real calibration·EEPROM readback·실측 dynamics gate 대기

## 1. 고정 stack

| 항목 | 고정값 | Windows | 서버 |
|---|---|---:|---:|
| Isaac Sim | `6.0.0.1` | 통과 | 통과 |
| Isaac Lab source | `28a37cecdd433c22d9eabd6a5954add9f13a8951` | 통과 | 통과 |
| ROS 2 | Jazzy | 통과 | 통과 |
| RMW | `rmw_zenoh_cpp` | 통과 | 통과 |
| Pixi | `0.70.2` | 통과 | 통과 |
| Python | `3.12.13` | 통과 | 통과 |
| PyTorch | `2.10.0+cu128` | 통과 | 통과 |
| Physics | PhysX | 통과 | 통과 |
| lock SHA256 | `234ba771eafb1b870a97f5ffe35887d89fe12188f093963ea3fc0ebc9f14854b` | 동일 | 동일 |

Isaac Lab package metadata는 source checkout 내부 package version을 `6.1.11` 등으로 표시하지만,
실행 package path와 Git `HEAD`는 두 머신 모두 위 고정 commit이다. 재현성 판단은 package metadata가
아니라 source commit과 lock hash를 함께 사용한다.

설치 위치:

- Windows: `D:\SO101\isaac6_ros`
  - Git repo의 `.pixi` Junction이 `D:\SO101\isaac6_ros\.pixi`를 가리킨다.
  - 재현: `scripts/parity/bootstrap_windows.ps1`
- 서버: `/DISK1/so101-sim2real/runtime/isaac6_ros`

Windows 재검증:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File scripts/parity/bootstrap_windows.ps1 -CheckOnly -Smoke
```

서버는 `scripts/parity/bootstrap_server.sh`로 재현한다. 이 스크립트는 Pixi archive checksum,
Isaac Lab commit, tracked lock hash를 먼저 검사한 뒤 `sim`/`real`/`ros-tools`를 locked install하고
stack check, ROS overlay, tests, checkpoint validator를 실행한다.

```bash
cd /DISK1/so101-sim2real/runtime/isaac6_ros
bash scripts/parity/bootstrap_server.sh
bash scripts/parity/bootstrap_server.sh --check-only --smoke --probe
```

Linux `real` environment는 LeRobot이 끌어오는 PyPI `evdev 1.9.3` sdist 대신
conda-forge `evdev 1.9.0` prebuilt package를 고정한다. Ubuntu 24.04 host kernel header와
최신 key-code table 불일치로 인한 native build 실패를 피하기 위한 platform-specific pin이다.

## 2. 기존 경로와 분리

기존 `SimToReal-SO101-PickCube-v0`와 Isaac Sim 5.1 / Isaac Lab 2.3.2 코드는 덮어쓰지 않았다.
새 경로는 다음 ID와 package로 분리했다.

```text
SimToReal-SO101-PickCube-Isaac6Parity-v0
src/sim_to_real/isaac6/
src/so101_parity/
scripts/parity/
```

서버 legacy checkout `4b909db`에서 Isaac Sim 5.1 `env_smoke.py`를 10 step 실행해
action/observation 6축, reset 0으로 rollback smoke를 통과했다.

## 3. Isaac Lab 3 API 전환

| 요구사항 | 구현 |
|---|---|
| quaternion WXYZ → XYZW | 새 config의 모든 Isaac-facing quaternion을 XYZW로 작성 |
| ROS quaternion 중복 변환 금지 | ROS interface는 geometry convention XYZW를 그대로 사용 |
| `ProxyArray.torch` | joint/camera/pose 접근에 명시적 torch view 사용 |
| 제거된 write API | 새 경로는 `JointPositionActionCfg`와 `env.step()`만 사용 |
| `TiledCameraCfg` 제거 | top/wrist/front 모두 `CameraCfg` |
| PhysX 새 config | `isaaclab_physx.physics.PhysxCfg` |
| actuator field 변경 | `effort_limit_sim`, `velocity_limit_sim` |
| launcher 변경 | `visualizer="none"` 기반 비대화형 경로 |
| legacy Core API 제거 | 새 client에 `World`/`SingleArticulation` 사용 0건 |
| fixed seed | parity environment `seed=0` |
| 30 Hz step | physics 120 Hz, decimation 4 |

공식 `find_quaternions.py`를 `src/sim_to_real/isaac6`와 `scripts/parity`에 실행해 review 대상 0건을
확인했다. `WARN_ON_TORCH_QUATF_ACCESS=1` 상태의 대표 camera/client smoke에서도 quatf torch
access warning은 0건이다.

## 4. Compatibility와 scene smoke

공식 `isaacsim.app.compatibility_check.Checker`를 사용하는 headless wrapper 결과:

| 머신 | 결과 | GPU | VRAM 판정 | OS |
|---|---|---|---|---|
| Windows | 통과 | RTX A4000 | GOOD, 16.1 GB | Windows 11 |
| 서버 | 통과 | RTX PRO 5000 Blackwell | IDEAL, 51.31 GB | Ubuntu 24.04 |

서버는 headless이므로 display 항목만 informational fail이며 GPU/driver/RTX/VRAM/CPU/RAM/storage/OS
필수 gate는 통과했다. 서버 CPU governor는 `powersave`라 성능 warning은 남지만 compatibility
실패 조건은 아니다.

두 머신에서 representative scene과 3-camera smoke를 통과했다.

```text
top   [1, 480, 640, 3] torch.uint8
wrist [1, 480, 640, 3] torch.uint8
front [1, 480, 640, 3] torch.uint8
joint [1, 6]           torch.float32
```

## 5. ROS 전환

새 typed interface:

- `/so101/vla/get_runtime_info`
- `/so101/vla/infer_chunk`
- `/so101/vla/status`

`policy-server:0.5.1` 기반 `so101-vla-ros:jazzy` image에 ROS Jazzy,
`rmw_zenoh_cpp`, interface/runtime package만 추가했다. checkpoint는 server manifest에서
startup 시 로드하고 client pickle instruction은 받지 않는다.

동작 중인 서버 service:

```text
so101-zenoh-router  tcp/0.0.0.0:7447
so101-vla-replay    deterministic replay backend
```

Windows는 LAN endpoint `tcp/10.10.16.147:7447`에 client mode로 직접 연결한다.

## 6. 검증 결과

| 항목 | 결과 |
|---|---|
| Windows core tests | 19/19 통과 |
| canonical dataset converter smoke | 1/1 통과 |
| Windows ROS overlay | 2 packages 통과 |
| 서버 ROS overlay | 2 packages 통과 |
| Windows Isaac6 sim client | 32/32, underrun/timeout/stale 0 |
| 서버 Isaac6 sim client | 32/32, underrun/timeout/stale 0 |
| checkpoint validator | 통과 |
| contract/manifest/checkpoint mismatch | fail-closed 통과 |
| second motion lease | 거부 통과 |
| mock chunk | bitwise 동일 |
| legacy 5.1 rollback | 10 step 통과 |

raw RGB 3장 service 왕복:

| payload | samples | p50 | p99 | gate |
|---|---:|---:|---:|---|
| gradient RGB, Windows→server 2026-06-22 | 100 | 17.63 ms | 18.89 ms | 통과 |
| gradient RGB, 서버 local 2026-06-22 | 100 | 8.97 ms | 10.55 ms | 통과 |
| incompressible random RGB | 100 | 31.34 ms | 42.53 ms | 실패 |

실제 camera frame은 random worst case가 아니므로 최종 network gate는 실기기 camera frame으로 다시
측정한다. 현재 결과를 압축 가능한 frame은 통과, incompressible worst case는 미통과로 구분한다.

## 7. 주요 파일

```text
pixi.toml
pixi.lock
src/so101_parity/
src/sim_to_real/isaac6/
ros2_ws/src/so101_vla_interfaces/
ros2_ws/src/so101_vla_runtime/
docker/Dockerfile.policy_ros
docker/policy-ros-entrypoint.sh
configs/parity/
configs/zenoh/
scripts/parity/
scripts/parity/bootstrap_windows.ps1
scripts/parity/bootstrap_server.sh
```

## 8. 남은 migration gate

- real paired arm pose 10~20개
- real gripper caliper 7~9점
- torque-off EEPROM readback
- 실제 camera payload network p99
- real no-load/cube-payload dynamics trace
- sim parameter fitting 후 trajectory parity

위 항목 전에는 `calibration.validated=false`, `motor_profile.readback_validated=false`를 유지하며
real motion은 실행 계층에서 차단한다.
