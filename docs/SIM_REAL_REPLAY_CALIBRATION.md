# Real → Sim Replay 관절 Calibration

실기기 SO-101 로 녹화한 LeRobot 에피소드를 Isaac Sim bridge 에 replay 할 때, **같은 위치·같은
크기 큐브를 잡은 데이터인데도 sim gripper 가 큐브를 못 잡는** 문제의 진단과 해결 계획.

> 결론 먼저: 단위·위치·크기·joint limit·딜레이·actuator gain 전부 정상. 유일하게 안 맞은 것 =
> **실 follower 관절 영점/스케일(so101_robot.json) ↔ sim URDF 영점** 불일치. 같은 관절각이
> real/sim 에서 다른 물리 자세 → EE 가 ~2-3cm 어긋나 큐브를 못 잡는다. replay 변환(codec)이
> arm deg→rad **1:1(per-joint 보정 0)** 이라 이 차이를 못 잡는다.

---

## 1. 배경

- **목적**: 실기기 SO-101 teleop 으로 녹화한 pick-and-place 에피소드를 sim 에서 replay (lerobot-replay
  의 sim 판). 데이터 검증·sim2real 정합 확인·VLA 데이터 기판용.
- **도구**:
  - `scripts/inference/replay_dataset_to_bridge.py` — dataset `action` → `/isaac_joint_commands`
    (JointState rad) publish. bridge(`run_cube_desk_ros_bridge.py`)의 ArticulationController 가 적용.
    변환 = `src/so101_contract/feature_codec.py` (arm deg→rad 1:1, gripper [0,100] affine).
  - `scripts/data/append_sim_episode.py` — sim replay 기록을 기존 LeRobot v3 dataset 에 episode append.
- **데이터셋**: `taehunkim/so101_pick_cube_test`. ep0/ep1 = 실기기 teleop (큐브 = 책상 앞 모서리에서
  30cm = env y=0.265, 40mm Cube1).

## 2. 증상

실기기와 **같은 위치(30cm)·같은 크기(40mm)** 큐브를 놓고 녹화한 에피소드를 sim 에 replay 했는데
**gripper 가 큐브를 못 잡는다.** 큐브 바로 위 허공에서 그리퍼가 닫힌다(접촉은 스치듯 발생하나 grasp 실패).

## 3. 진단 — 배제한 것들 (전부 정상)

| 점검 | 방법 | 결과 |
|---|---|---|
| 단위/codec | feature_codec 검토 | arm deg·gripper[0,100] 1:1·affine. **정상** |
| 큐브 위치 | `_CUBE_LAYOUT` vs 실기기 30cm | env y=0.265 = 30cm. **일치** |
| 큐브 크기 | `cube_specs.py` + USD collider | 40mm convexHull. **정상** |
| joint limit | `set_arm_joint_limits.py` + bridge dof-print | elbow 90→100·wrist_flex 95→105·shoulder_lift ±105 확대·적용 확인 |
| self-collision | `--no_self_collisions` A/B | 팔/캠홀더 convex 충돌이 elbow 고굴곡(~90°)을 막음. off 시 ~94.6° 회복(부분 기여) |
| action-state 딜레이 | reach-probe·fps8 A/B·kinematic | 비동기 ROS 기록 artifact. fps·stiffness 무관. **kinematic(딜레이 0)도 못 잡음** → 딜레이 아님 |
| actuator gain/effort/stiffness | stiffness 17.8 vs 400 vs 3000 | steady 자세 무관(딜레이만 영향). grasp 깊이 불변 |

## 4. 근본 원인

**reach-probe(sim ground-truth) 측정** — grasp 순간(gripper 가 큐브 바로 위, xy 2.7cm):

```
gripper TCP 높이 = +4.4cm (책상면 기준)
real grasp 높이  = +2.0cm
→ gripper 가 큐브보다 ~2.4cm 높이 + 옆으로 어긋나 허공서 닫힘 → 못 잡음
fps30·8 동일, stiffness 17.8·3000 동일 = steady(딜레이/물리 아님)
```

**같은 관절각인데 sim EE 가 real 과 다른 위치.** 실기기서 큐브를 잡은 그 관절값을 sim 에 그대로
넣으니 gripper 가 ~2.4cm 높이 뜬다. 즉 `FK_sim(관절각) ≠ FK_real(관절각)`.

원인 = 실 follower 관절 **영점/스케일**(`so101_robot.json` 의 homing_offset·range 로 정의되는
calibration 영점)이 sim USD/URDF 의 관절 영점과 다르다. 실기기 "elbow 99°"와 sim "elbow 99°"가
**다른 물리 자세** → 관절별 오차가 누적돼 EE 에서 ~2-3cm 로 나타난다.

replay 변환 `feature_codec.policy_feature_to_sim_joint_radians` 은 arm 을 deg→rad **1:1(per-joint
offset·scale 보정 0)** 로 처리 → "real deg == sim deg" 를 가정한다. 영점이 어긋나면 이 가정이 깨진다.

## 5. 영향

- **모든 real→sim replay** 에 적용된다(이 에피소드 한정 아님).
- **VLA 학습**: 영향 없음 — 학습 데이터는 sim 에서 sim 영점으로 생성됨.
- **VLA 배포(policy→real)**: 별개 경로이며 `src/so101_contract/leader_calibration.py` 가 leader
  정규화↔sim 변환을 이미 처리. (단 그건 **leader 정규화[-100,100]** 용이고, follower replay 의
  영점 보정과는 다른 변환.)
- **미보정 경로 = real-녹화-데이터 → sim replay** (신규, 기존 검증 안 됨). 여기만 calibration 누락.
- `feature_codec` 은 **그대로 둔다**(policy↔sim 단일 계약). follower replay 용 영점 보정은 **별도
  레이어**로 추가(아래).

## 6. 해결 — Calibration 방법

관절별 affine 보정:

```
sim_deg_j = a_j · real_deg_j + b_j        # a=스케일, b=영점 offset (per joint)
```

### 6.1 a, b 산출 (매칭 포즈)

실기기와 sim 을 **같은 물리 자세**에 두고 관절값을 비교한다.

- **포즈 P1 = sim URDF-zero** (모든 관절 기구학적 0, 팔 곧게). 실기기를 동일 물리 자세로 맞추고
  teleop 값 `real@P1` 읽기. 스케일 a≈1 가정 시 **영점 `b_j = -real@P1_j`** (sim 0 기준).
- **포즈 P2 = 다른 known sim 자세** (예: 각 관절 +60° 부근). `real@P2` 읽기 → 2점으로 **a_j 까지**
  풀이: `a_j = (sim@P2 - sim@P1)/(real@P2 - real@P1)`, `b_j = sim@P1 - a_j·real@P1`.

> 스케일이 1 에 가깝다면(영점만 어긋남) P1 한 포즈로 충분. 의심되면 P2 로 검증.

### 6.2 적용

- 신규 단일 소스 `src/so101_contract/follower_calibration.py` (실 follower deg → sim deg 의 per-joint
  a,b 테이블 + 변환 함수). gripper 는 affine 그대로(feature_codec 과 동일).
- `replay_dataset_to_bridge.py` 의 변환에 **codec deg→rad 직전** 삽입: `real_deg → (a·+b) → sim_deg
  → rad`. recorder(있다면)도 동일 적용.
- **device-specific**: 이 a,b 는 `so101_robot.json` 의 그 물리 robot 에 묶임. 로봇/재캘리브레이션 시 갱신.

### 6.3 검증

replay + `--reach_probe` → grasp 순간 TCP 가 큐브(+2cm)에 도달하면 성공. self-collision off 와
병행(elbow 고굴곡 허용).

## 7. 결과 / 증거 (HF dataset 에피소드)

`taehunkim/so101_pick_cube_test` 에 진단 증거로 보존:

| ep | 내용 | 의미 |
|---|---|---|
| 0,1 | 실기기 teleop pick-and-place | 정답(real) |
| 2 | sim replay (async PD) | state 딜레이 + grasp 실패 |
| 3 | sim **kinematic** replay (state≡action, 딜레이 0) | 딜레이 0인데도 grasp 실패 → **딜레이가 원인 아님**을 입증 |
| (4) | calibration 적용 replay | 보정 후 grasp 성공하면 완료 (TODO) |

## 8. 참고

- `so101_robot.json` — 실기기 calibration (homing_offset·range_min/max, 6관절).
- `scripts/inference/replay_dataset_to_bridge.py` · `scripts/data/append_sim_episode.py` ·
  `scripts/assets/set_arm_joint_limits.py`.
- `src/so101_contract/feature_codec.py` (policy↔sim) · `leader_calibration.py` (leader↔sim).
- `docs/SIM_REAL_INFERENCE_PARITY.md`.
