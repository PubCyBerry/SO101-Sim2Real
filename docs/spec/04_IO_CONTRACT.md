# 04. I/O 계약 — 단위·프레임 변환

> **정본**: `src/so101_contract/` 7개 모듈. 이 문서는 코드에서 역추출한 as-built 명세이며,
> 값의 단일 소스는 언제나 코드다. 앵커 표기 = `경로::심볼`.

`so101_contract` 는 **순수 Python + NumPy** 패키지다. Isaac Lab·ROS·torch 를 import 하지
않으므로 host uv · isaac-sim Docker · vla-ros Docker · Windows 실기기 네 환경에서 같은
구현을 쓴다 (`src/so101_contract/__init__.py` docstring).

---

## 1. 세 계약의 분리

SO-101 은 **네 개의 서로 다른 수 체계**를 오간다. 이름이 비슷해 혼동하기 쉬우므로 먼저 고정한다.

| 이름 | 정의 | arm 단위 | gripper 단위 |
|---|---|---|---|
| **policy-feature** | 정책 입출력. 데이터셋 `observation.state`·`action` 에 저장되는 값 | degree | `[0, 100]` |
| **sim joint** | Isaac Sim articulation joint 값 | radian | radian (`[-10°, 100°]` 범위) |
| **real leader** | 실 leader(Feetech) 모터 정규화값 | `[-100, 100]` | `[0, 100]` |
| **real follower** | 실 follower 관절 읽기값 | degree (device 영점) | `[0, 100]` |

이 넷을 잇는 계약이 세 모듈로 나뉜다. **셋은 서로 다른 변환이며 섞어 쓰면 안 된다.**

```
                    ┌──────────────────┐
                    │  policy-feature  │  ← 정책 · 데이터셋
                    └────────┬─────────┘
                             │ feature_codec  (§2)
                             ▼
        leader_calibration   ┌──────────┐   follower_calibration
   real leader ◀────(§3)────▶│ sim joint│◀────────(§4)────────▶ real follower
    [-100,100]               │  radian  │                        degree
                             └────┬─────┘
                                  │ eef_kinematics  (§5)
                                  ▼
                        base_link → tcp_grasp  (EEF pose)
```

| 쓰는 곳 | 계약 |
|---|---|
| VLA 학습·추론, LeRobot 데이터셋 기록/재생 | **feature_codec** (§2) |
| sim teleop (실 leader → sim 팔), datagen device | **leader_calibration** (§3) |
| 실기기 녹화 데이터 sim replay, cross-domain 추론 어댑터 | **follower_calibration** (§4) |
| EEF-space 데이터셋 파생, TCP pose 관측 | **eef_kinematics** (§5) |

> **왜 셋인가**: leader 는 정규화 `[-100,100]` 을 내보내는데 USD joint 범위가 관절별
> 비대칭(`elbow_flex ±100`, `wrist_flex -95/+105` …)이라 codec 의 arm 1:1 degree 로는
> 재현할 수 없다. follower 는 다시 device 영점이 sim URDF 영점과 어긋나 있다(§4).
> 근거 = `src/so101_contract/leader_calibration.py` 모듈 docstring.

---

## 2. `feature_codec` — policy-feature ↔ sim joint

앵커: `src/so101_contract/feature_codec.py`

### 2.1 상수

| 심볼 | 값 | 의미 |
|---|---|---|
| `CODEC_VERSION` | `"so101_joint_position_v1"` | 계약 버전. 스냅샷·ROS 어댑터가 일치 검사 |
| `FPS` | `30` | 데이터셋·정책 제어 주파수 |
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` / `IMAGE_CHANNELS` | `480` / `640` / `3` | 카메라 프레임 규격 |
| `CAMERA_KEYS` | `("top", "wrist", "front")` | 카메라 3대의 캐노니컬 키 |
| `SO101_JOINT_ORDER` | `("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")` | **전 시스템 관절 순서 정본** |
| `JOINT_FEATURE_NAMES` | `("shoulder_pan.pos", …, "gripper.pos")` | `f"{joint}.pos"` 파생. LeRobot feature 이름 |
| `POLICY_GRIPPER_RANGE` | `(0.0, 100.0)` | policy-feature gripper 범위 |
| `SIM_GRIPPER_RANGE_DEG` | `(-10.0, 100.0)` | sim gripper joint 범위(degree) |
| `SIM_JOINT_LIMITS_RAD` | arm 5축 `[-π, π]`, gripper `[rad(-10°), rad(100°)]` — `(6, 2)` float32 | `clamp_sim_joint_radians` 의 clip 경계 |

### 2.2 변환 수식

**arm 5축은 degree ↔ radian 1:1** (스케일·오프셋 없음).

**gripper 는 affine**:

```
sim_deg  = feature / 100 × 110 − 10          # feature → sim
feature  = (sim_deg + 10) / 110 × 100        # sim → feature
```

경계값:

| feature | 0 | 50 | 100 |
|---|---|---|---|
| sim gripper (degree) | −10 | 45 | 100 |

> **offset 없음**: 모든 action 은 **절대 joint target** 이다. 환경 action term 이
> `use_default_offset=False` 로 설정돼 있어 rad-space offset 이 0이다 (상세 = `03_ENV_SPEC.md §4`).

### 2.3 함수

| 시그니처 | 동작 |
|---|---|
| `sim_joint_radians_to_policy_feature(values_rad)` | sim radian `(..., 6)` → policy-feature `(..., 6)` float32 |
| `policy_feature_to_sim_joint_radians(values_feature)` | 역변환 |
| `clamp_sim_joint_radians(values_rad)` | `SIM_JOINT_LIMITS_RAD` 로 clip |

**입력 검증** (`feature_codec.py::_as_joint_array`): 마지막 축이 6이 아니거나 NaN/Inf 가
있으면 `ValueError`. 입력을 복사하므로 in-place 오염이 없다.

---

## 3. `leader_calibration` — real leader ↔ sim joint

앵커: `src/so101_contract/leader_calibration.py`

leisaac `devices/action_process.py::convert_action_from_so101_leader` 및
`utils/robot_utils.py` 의 torch 결합을 제거한 순수 numpy 재작성본이다. **세 테이블의 단일
진실 소스**이며 `sim_to_real.assets.robots.lerobot` 등이 여기서 import 한다.

### 3.1 테이블

`SO101_FOLLOWER_USD_JOINT_LIMITS` — USD 에 기록된 joint position limit (degree):

| joint | lo | hi | 비고 |
|---|---:|---:|---|
| `shoulder_pan` | −110 | 110 | |
| `shoulder_lift` | −105 | 105 | ±100→±105 (replay 가동범위 정합, 2026-06-29) |
| `elbow_flex` | −100 | 100 | 90→100 (실 calibration 정합, 2026-06-29) |
| `wrist_flex` | −95 | 105 | 95→100→105 (동상) |
| `wrist_roll` | −160 | 160 | |
| `gripper` | −10 | 100 | `SIM_GRIPPER_RANGE_DEG` 와 동일 |

`SO101_FOLLOWER_MOTOR_LIMITS` — 실기기 정규화 범위: arm 5축 `(-100, 100)`(`RANGE_M100_100`),
gripper `(0, 100)`(`RANGE_0_100`).

`SO101_FOLLOWER_REST_POSE_RANGE` — rest pose 판정 허용 범위 (degree, 중심 ±30°):

| joint | 범위 | 중심 |
|---|---|---:|
| `shoulder_pan` | (−30, 30) | 0 |
| `shoulder_lift` | (−130, −70) | −100 |
| `elbow_flex` | (60, 120) | 90 |
| `wrist_flex` | (20, 80) | 50 |
| `wrist_roll` | (−30, 30) | 0 |
| `gripper` | (−40, 20) | −10 |

### 3.2 변환 수식

```
deg = (v − motor_lo) / motor_range × usd_range + usd_lo     # leader → sim
v   = (deg − usd_lo) / usd_range × motor_range + motor_lo   # sim → leader
```

이 선형 remap 을 `deg = a·v + b` 로 환원하면 (본 문서 작성 시 산출·검산):

| joint | a | b | 검산 `v = −100, 0, 100` |
|---|---:|---:|---|
| `shoulder_pan` | 1.10 | 0 | −110, 0, 110 |
| `shoulder_lift` | 1.05 | 0 | −105, 0, 105 |
| `elbow_flex` | 1.00 | 0 | −100, 0, 100 |
| `wrist_flex` | 1.00 | **+5** | −95, 5, 105 |
| `wrist_roll` | 1.60 | 0 | −160, 0, 160 |
| `gripper` | 1.10 | −10 | (`v = 0, 50, 100`) −10, 45, 100 |

> **gripper 는 `feature_codec` 과 수식이 완전히 같다**(a=1.10, b=−10). arm 만 다르다.
> 즉 leader 계약이 필요한 이유는 **arm 의 관절별 비대칭 범위** 하나뿐이다.

### 3.3 함수

| 시그니처 | 동작 |
|---|---|
| `real_leader_to_sim_radians(joint_state)` | `dict{name: v}` 또는 `(..., 6)` → radian float32 |
| `sim_radians_to_real_leader(values_rad)` | 역변환 → 정규화값 float32 |
| `is_so101_at_rest_pose(values_rad)` | 전 관절이 `REST_POSE_RANGE` 안(**strict** `>`/`<`)이면 True. `(...)` bool array 또는 스칼라 |

같은 수식의 torch 사본이 `src/sim_to_real/devices/action_process.py::convert_action_from_so101_leader`
에 있다 (env 텐서 경로용).

---

## 4. `follower_calibration` — real follower ↔ sim joint

앵커: `src/so101_contract/follower_calibration.py`

**배경**: 실기기 녹화 데이터셋을 Isaac Sim 으로 replay 할 때, 실 follower 영점
(`so101_robot.json` homing_offset)이 sim URDF 영점과 어긋나 **같은 관절 숫자가 다른 물리
자세**가 된다 — grasp 순간 EE 가 약 2.4 cm 떠 큐브를 헛집는다. 진단 과정 =
`docs/SIM_REAL_REPLAY_CALIBRATION.md`.

### 4.1 수식

변환 2개가 아니라 **affine 1개 + 역산**으로 양방향을 지원한다.

```
sim_deg_j = A_j × real_j + B_j          # forward (replay,  Real→Sim)
real_j    = (sim_deg_j − B_j) / A_j     # inverse (배포,    Sim→Real)
```

- arm(0–4): `real_j` = 실 follower degree, `sim_deg_j` = Isaac URDF degree
- gripper(5): `real_j` = 실 gripper `[0,100]`, `sim_deg_j` = sim gripper degree `[-10,100]`

### 4.2 측정 상수 (device-specific 단일 소스)

`FOLLOWER_AFFINE_A` / `FOLLOWER_AFFINE_B`, 순서 = `SO101_JOINT_ORDER`:

| joint | A | B |
|---|---:|---:|
| `shoulder_pan` | 1.0 | +7.868132 |
| `shoulder_lift` | 1.0 | +4.483516 |
| `elbow_flex` | 1.0 | −4.439560 |
| `wrist_flex` | 1.0 | −4.175824 |
| `wrist_roll` | 1.0 | +5.274725 |
| `gripper` | 1.171267 | −17.126755 |

**출처** (코드 주석, 2026-06-30 실기기 `so101_robot`):

- arm — 각 joint 를 URDF-zero 자세로 두고 2회 읽어 평균 → `B[:5] = −real_home`
  (`A=1`, offset only). `real_home = [−7.868, −4.484, 4.440, 4.176, −5.275]`
- gripper — 끝점 2점: 완전닫힘 `real=6.085 → sim=−10°`, 완전열림 `real=100.0 → sim=+100°`
  ⇒ `A_g = (100 − (−10)) / (100 − 6.085) = 1.17127`, `B_g = −10 − A_g × 6.085 = −17.127`

**재캘리브레이션·로봇 교체 시 이 두 상수만 갱신한다.**

> ⚠ 코드 주석 경고: pan·wrist_roll 은 손맞춤 재현오차가 스냅샷 간 약 5–6° 다. reach probe
> 잔차가 크면 그 두 축을 먼저 재수집할 것.

**no-op 기준값**(재캘리 출발점): arm `A=1, B=0`, gripper `A=1.1, B=−10` — 이 값이면
`feature_codec` 과 완전히 같아진다. 실제로 `_self_check()` 가 이 동치를 assert 한다.

### 4.3 함수

| 시그니처 | 동작 |
|---|---|
| `real_follower_to_sim_radians(joint_state)` | forward. `dict` 또는 `(..., 6)` → radian float32 |
| `sim_radians_to_real_follower(values_rad)` | inverse |
| `real_follower_to_policy_feature(values)` | 합성: follower(fwd) ∘ feature_codec. **real 입력을 sim-trained 정책에** |
| `policy_feature_to_real_follower(values)` | 합성 역: 정책 출력을 real 로 |
| `fit_follower_affine(real, sim_deg)` | 매칭 포즈 `(N, 6)` 쌍에서 per-joint 1차 최소제곱 → `(A, B)` + 잔차 리포트. `x.max()-x.min() < 1e-6`(안 움직인 관절)이면 `a=1, b=mean(y−x)` 폴백 |

### 4.4 self-check

`python3 src/so101_contract/follower_calibration.py` — assert 4종:
① no-op 상수 ↔ `feature_codec` 동치 · ② 임의 affine round-trip 항등 · ③ `fit_follower_affine`
가 주입 affine 복원 · ④ real-follower ↔ policy-feature 합성 round-trip 항등.

---

## 5. `eef_kinematics` — base_link → tcp_grasp FK

앵커: `src/so101_contract/eef_kinematics.py`

joint-space 데이터셋을 EEF-space 로 파생할 때(`05_DATA_SPEC.md §8`)와 online TCP pose
관측이 **같은 구현**을 쓰도록, URDF 관절 체인과 cuRobo robot YAML 의 `tcp_grasp` fixed
transform 을 읽어 순수 NumPy FK 를 제공한다. Isaac Lab·cuRobo·Pinocchio 의존성 없음.

| 심볼 | 값 |
|---|---|
| `EEF_KINEMATICS_VERSION` | `"so101_base_tcp_grasp_fk_v2"` |
| `ARM_JOINT_ORDER` | `SO101_JOINT_ORDER[:5]` (gripper 제외 5축) |
| `ROTATION_REPRESENTATIONS` | `("rot6d", "rpy", "wxyz")` |
| `ROTATION_REPRESENTATION_DIMS` | `{"rot6d": 6, "rpy": 3, "wxyz": 4}` |

### 5.1 좌표 계약

- 입력: arm 5축 joint angle, **radian**, `ARM_JOINT_ORDER` 순서. `(..., 5)` 또는 `(..., 6)`
  (6이면 앞 5개만 사용)
- 출력: URDF `base_link` 기준 `tcp_grasp` **absolute** pose
- 회전 표현
  - `rot6d` — rotation matrix 첫 두 **행(row)** flatten (GR00T 구현과 동일)
  - `rpy` — URDF fixed-axis roll/pitch/yaw, radian, `Rz(yaw) @ Ry(pitch) @ Rx(roll)`
  - `wxyz` — unit quaternion, scalar-first, **canonical hemisphere `w ≥ 0`**
    (`w ≈ 0` 이면 첫 nonzero 성분 부호로 결정 — 통계·학습의 부호 불연속 제거)

real/sim joint feature → radian 변환은 **이 모듈 밖**에서 한다(§2·§4).

### 5.2 설정 소스

`SO101EndEffectorKinematics.from_files(urdf_path, robot_yaml_path, tcp_name="tcp_grasp")`

| 항목 | 값 | 앵커 |
|---|---|---|
| URDF 기본 경로 | `assets/robots/urdf/so_arm101.urdf` | `scripts/convert/joint_dataset_to_eef.py::DEFAULT_URDF` |
| robot YAML 기본 경로 | `assets/robots/so101.yml` | `scripts/convert/joint_dataset_to_eef.py::DEFAULT_ROBOT_YAML` |
| `base_link` | `base_link` | `assets/robots/so101.yml` `kinematics.base_link` |
| `tcp_grasp.parent_link_name` | `gripper_link` | `assets/robots/so101.yml` `kinematics.extra_links.tcp_grasp` |
| `tcp_grasp.fixed_transform` | `[0.012, −0.015, −0.025, 0.024337, 0.0, 0.999704, 0.0]` = `[xyz, q_wxyz]`, **길이 7 필수** | 동상 |

TCP 정의 근거(YAML 주석): 손가락 사이 pinch 지점을 collision sphere 기하로 유도 — fixed pad
`x ≈ −0.018` · jaw pad `x ≈ +0.018` 대칭, `y ≈ 0`, pad 유효 `z −0.05 ~ −0.09` 의 중심 `−0.070`.
`+z` = approach(손끝 방향), `x` = closing 축. quaternion = `Ry(π − 0.0486795)` —
USD 체인에 URDF `wrist_roll` origin 의 `Ry(0.0487)` 항이 없어 생기는 **이중 FK 2.79° 피치 차**를
TCP 회전으로 흡수한다(전 자세 상수라 정확 보정). 상세 = `09_TACIT_KNOWLEDGE.md §3`.

### 5.3 무결성 검사

- 생성자가 URDF 체인의 가동 관절 순서가 `ARM_JOINT_ORDER` 와 다르면 `ValueError`
- `tcp_grasp.fixed_transform` 이 길이 7이 아니면 `ValueError`
- 체인 순회 중 cycle 또는 부모 joint 누락이면 `ValueError`
- `revolute`/`continuous`/`fixed` 외 joint type 이면 `ValueError`

### 5.4 출력 차원

| 메서드 | 출력 | 차원 |
|---|---|---|
| `forward_matrices(q)` | 동차변환 | `(..., 4, 4)` float64 |
| `forward_xyz_rot6d(q)` | `xyz + R 첫 두 행` | 9 |
| `forward_xyz_rpy(q)` | `xyz + RPY` | 6 |
| `forward_xyz_wxyz(q)` | `xyz + quat` | 7 |
| `forward_xyz_rotation(q, representation)` | 위 3종 공통 진입점 | 9/6/7 |

`encode_rotation_matrices(R, rep)` / `decode_rotation_representation(v, rep)` 가 표현 간
왕복을 담당한다.

---

## 6. `action_queue` — action chunk 큐

앵커: `src/so101_contract/action_queue.py`. LeRobot async `RobotClient` 호환 semantics.

| 심볼 | 값 |
|---|---|
| `ACTION_AGGREGATE_NAMES` | `("weighted_average", "latest_only", "average", "conservative")` |

`aggregate_actions(old, new, name)` 가중치 `(old, new)`:

| name | old | new |
|---|---:|---:|
| `weighted_average` (기본) | 0.3 | 0.7 |
| `latest_only` | 0.0 | 1.0 |
| `average` | 0.5 | 0.5 |
| `conservative` | 0.7 | 0.3 |

shape 불일치 또는 미지 name 이면 `ValueError`.

`ActionChunkQueue` (thread-safe, `RLock`) 상태 전이:

| 메서드 | 동작 |
|---|---|
| `merge(incoming)` | `timestep <= latest_action` 인 action 은 **버림**. 기존 timestep 과 겹치면 `aggregate_actions` 로 결합. 정렬 후 교체. `action_chunk_size = max(기존, len(incoming))`, `_must_go = True` |
| `pop_next()` | FIFO 로 하나 꺼내고 `latest_action` 갱신. 빈 큐면 `IndexError` |
| `ready_to_send_observation(thr)` | `len(queue) / action_chunk_size <= thr`. `thr ∉ [0,1]` 이면 `ValueError`. chunk size 미확정(≤0)이면 항상 True |
| `observation_must_go()` | `_must_go` **그리고** 큐가 빈 경우에만 True |
| `mark_observation_sent(must_go)` | `must_go=True` 일 때만 `_must_go=False` |
| `mark_request_failed()` | `_must_go=True` 로 복구 |

---

## 7. `policy_snapshot` — 정책 I/O 스냅샷

앵커: `src/so101_contract/policy_snapshot.py`. 모델·ROS 의존성 없이 한 번의 policy
request/response 를 NPZ 로 보존해 오프라인 재현에 쓴다.

| 심볼 | 값 |
|---|---|
| `SNAPSHOT_VERSION` | `"so101_policy_io_snapshot_v1"` |

NPZ 배열 키:

| 키 | dtype | 내용 |
|---|---|---|
| `manifest_json` | str | `snapshot_version`·`codec_version`·`request_timestep`·`must_go`·`image_keys`·`observation`(스칼라만)·`metadata` |
| `action_timesteps` | int64 | |
| `actions_feature` | float32 | policy-feature 단위 action chunk |
| `actions_sim_rad` | float32 | sim radian 단위 action chunk |
| `image_{i}` | 원본 dtype | `image_keys[i]` 에 대응. `ndim >= 2` 인 observation 값이 자동 분류됨 |

**로드 시 검증**: `snapshot_version` 불일치 → `ValueError` · `codec_version` 불일치 →
`ValueError` · `JOINT_FEATURE_NAMES` 중 observation 에 없는 키가 있으면 `ValueError`.
`allow_pickle=False` 로 로드한다.

---

## 8. 프레임 변환 매트릭스 (`JOINT_FRAME_MODE`)

앵커: `scripts/inference/policy_server_affine.py`

`policy-server-affine` 모드는 stock `PolicyServer` 를 상속해 **정책 normalize 바깥에서**
`observation.state`(수신)와 `action`(반환)을 변환한다. 정규화 통계가 불변이고 현재 client
두 종(sim = `ros2_ws/.../vla_policy_node.py`, 실기기 = `scripts/inference/eef_robot_client.py`)
도 이 모드 때문에 바뀌지 않는다. **이미지는 변환하지 않는다.**

**적용 범위 — joint-space 전용.** 이 affine 매트릭스는 **joint mode**(`joint_absolute`·
`joint_relative`)의 cross-domain fallback 전용이다. **EEF mode 에서는 쓰지 않는다** — EEF
real/sim 차이는 공통 FK/IK platform adapter(`eef_policy_io`·`eef_kinematics`·`eef_ik`)가
담당하며, 그 위에 affine 을 겹치면 변환이 이중 적용된다. 상세 = §10 정본 문서.

모드 이름 = `<학습데이터 도메인>-to-<추론 플랫폼>`:

| `JOINT_FRAME_MODE` | obs (client → policy) | action (policy → client) |
|---|---|---|
| `sim-to-sim` (기본) | passthrough | passthrough |
| `real-to-real` | passthrough | passthrough |
| `sim-to-real` | `real_follower_to_policy_feature` | `policy_feature_to_real_follower` |
| `real-to-sim` | `policy_feature_to_real_follower` | `real_follower_to_policy_feature` |

미지 모드면 기동 시 `ValueError`. 변환 지점은 `_enqueue_observation` 의 `OBS_STATE` 키와
`_predict_action_chunk` 의 action 텐서 둘뿐이다. 실행 방법 = `06_RUNTIME_SPEC.md §4`.

---

## 9. 검증

| 수단 | 실행 | 보증 |
|---|---|---|
| `scripts/contract/validate_so101_io_contract.py` | `python -m` (인자 없음) | codec 경계값·round-trip · ROS 어댑터 `units` 동치 및 `CODEC_VERSION` 일치 · action queue 4-aggregate 및 upstream parity · snapshot round-trip. 4-validator 전부 통과 시 `PASS` |
| `scripts/contract/replay_so101_policy_snapshot.py` | `<snapshot.npz> [--server-address …]` | 오프라인 decode 오차 + (서버 지정 시) 실제 재추론 결과 대조. JSON 리포트 |
| `src/so101_contract/follower_calibration.py` | `python3 <파일>` | §4.4 assert 4종 |

허용 오차 = `atol 1e-5`, `rtol 0`.

---

## 10. 범위 밖 — action representation schema v2

이 문서(§1–§9)는 **joint 단위 codec 과 base_link → tcp_grasp absolute FK** 까지만 다룬다.
그 위의 action representation 계층(mode·pose format·runtime transform·manifest·routing)은
**구현 완료**되어 있으며 정본은 하나다:

> **정본**: [`docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`](../EEF_RELATIVE_ACTION_PIPELINE_SPEC.md)

4 mode × 3 pose format, dataset 이 그 space 의 **absolute 만** 저장한다는 규칙, relative 를
만드는 training processor 와 되돌리는 postprocessor, universal manifest(schema v2), mode 별
routing, legacy migration — **수치·스키마·규칙을 여기에 복제하지 않는다.** 위 정본을 볼 것.

이 문서와의 접점만 적으면:

| 이 문서의 계약 | schema v2 에서의 위치 |
|---|---|
| `feature_codec`·`follower_calibration`(§2·§4) | joint mode 의 platform 경계 변환 |
| `eef_kinematics` FK(§5) | EEF mode 의 FK, 그 역방향은 `eef_ik` |
| `action_queue`(§6) | IK 이후 joint queue merge(EEF 벡터를 평균하지 않는다) |
| `JOINT_FRAME_MODE` affine(§8) | **joint mode 전용**. EEF mode 는 FK/IK adapter 사용 |

`scripts/convert/joint_dataset_to_eef.py` 가 만드는 파생 데이터셋은 **absolute 값만**
저장한다. relative 는 runtime processor 가 만들며 단순 벡터 뺄셈은 부적합하다 —
모든 relative 변환은 rotation matrix/SE(3) 를 경유한다.

**검증 상태 — 구현 완료와 외부 acceptance 미실행을 구분한다.**

| 항목 | 상태 | 의미 |
|---|---|---|
| schema v2 구현·offline 검증 | **완료** | 24 조합 offline matrix 및 contract-level rollout dry-run 통과 |
| 대표 조합 sim closed-loop | `NOT_RUN` | 구현 미완료가 **아니다**. 학습된 EEF checkpoint·sim 평가가 아직 실행되지 않은 **외부 acceptance 미실행** 상태 |
| real guarded rollout | `BLOCKED_EXTERNAL` | 마찬가지로 구현 문제가 아니라 실기기 승인·작업자·e-stop gate 라는 **외부 조건 대기** 상태 |

단계 정의·완료 조건·현재 수치는 전부 위 정본에 있다.

---

## 참조

- 관측·액션이 이 단위를 어떻게 쓰는가 → `03_ENV_SPEC.md` §3, §4
- 데이터셋에 어떻게 저장되는가 → `05_DATA_SPEC.md` §3, §8
- ROS·gRPC 페이로드 단위 → `07_INTERFACES.md` §2, §9
- 실 follower 캘리브레이션 진단 서사 → `docs/SIM_REAL_REPLAY_CALIBRATION.md`
- 왜 이 상수인가 / 함정 → `09_TACIT_KNOWLEDGE.md` §3, §4
