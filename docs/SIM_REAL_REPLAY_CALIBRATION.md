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

## 6. 해결 — Calibration (구현 완료, 측정값 입력 대기)

**6축(arm 5 + gripper)** per-joint affine. 변환 2개가 아니라 **affine 1개 + 역산**으로
Real→Sim·Sim→Real 양방향을 모두 지원한다.

```
sim_deg_j = a_j · real_j + b_j        # forward (replay,  Real→Sim)
real_j    = (sim_deg_j − b_j) / a_j   # inverse (배포,    Sim→Real)
```

- arm(0-4): `real_j`=follower degree, `sim_deg_j`=URDF degree.
- **gripper(5): `real_j`=[0,100], `sim_deg_j`=sim gripper degree.** 실 [0,100] 의 열림/닫힘 끝점이
  sim [-10,100]° 끝점과 물리적으로 안 맞음 → 같은 명령이 다른 jaw 폭 → grasp 실패. **그리퍼도 보정 대상**
  (현 `[0,100]→[-10,100]°` 는 측정 아닌 설계 가정).

**구현**: `src/so101_contract/follower_calibration.py` — `FOLLOWER_AFFINE_A/B`(6-vector, **단일 소스**)
+ `real_follower_to_sim_radians`(forward) / `sim_radians_to_real_follower`(inverse) /
`fit_follower_affine`(측정 도우미) + `_self_check`. 기본 상수 = feature_codec no-op 재현(arm 1:1,
gripper [0,100]→[-10,100]°)이라 측정 전에도 무해. device-specific — `so101_robot.json` 의 그 물리
robot 에 묶임, 재캘리브레이션 시 이 두 상수만 갱신.

**배선**: `replay_dataset_to_bridge.py --arm_mapping=follower` (기존 `codec`/`calibration` 무변경).

### 6.1 측정 → 상수 채우기 (`scripts/ece_4560/real` 도구)

**arm — HOME offset 먼저 (a=1, b=−real_home):** 2.4cm 만 어긋난 정황상 sign·scale 정상, 영점만 문제일 공산.
1. sim 에서 arm=0(URDF-zero) 자세가 물리적으로 어떤 모양인지 확인(viewport).
2. `read_position.py`(torque off, 50Hz degree) → 실기기를 손으로 그 모양에 맞춤(직선/수직 기준에 수평기·직선자 정렬).
3. 안정된 arm 5축 = `real_home` → **`FOLLOWER_AFFINE_B[:5] = −real_home`** (sim 읽기 불요, URDF-zero=0).

**gripper — 끝점 2점으로 완전 affine (arm 보다 쉬움):** 기계 끝점이 명확해 정밀 자세 불필요.
1. `P_open`: 실 gripper 완전 열림 → `r_open`; sim gripper 완전 열림 각 → `s_open`.
2. `P_grasp`: 실 gripper 큐브(40mm) 파지 닫힘 → `r_grasp`; sim jaw 40mm 폭 각 → `s_grasp`.
3. 두 점을 `fit_follower_affine` 에 넣어 gripper 열 `(a_g,b_g)` → `FOLLOWER_AFFINE_A[5],B[5]`.

**arm escalate (HOME-only 부족 시):** `set_position.py` 로 서로 다른 3~5 자세(`real_pick_place.py`
waypoint pan=±45·lift=45/0 등) 명령 → `real`/`sim` 읽어 `fit_follower_affine` 로 scale 까지 + 잔차 확인.

### 6.2 검증

```
python -m so101_contract.follower_calibration          # self-check: round-trip 항등 OK
# bridge(self-collision ON 기본) 띄운 뒤 vla-ros 에서:
replay_dataset_to_bridge.py --dataset taehunkim/so101_pick_cube_test --episode <real ep> \
    --arm_mapping follower --probe_tracking --wait_for_subscriber
```
- `--probe_tracking`: cmd vs achieved steady err ≈ 0 → sim 이 target 도달(변환만의 문제 확인).
- livestream(:49100) 으로 관전 → gripper 가 큐브 집어 그릇에 넣으면 성공.
- 진단 flag(`--reach_probe`·`--no_self_collisions`)는 검증 후 제거됨 — **self-collision 영구 ON**(follower 가
  elbow target 낮춰 고굴곡 안 막힘). 더 정밀한 phase 비교는 §9 의 `--sequence`.

## 7. 결과 / 증거 (HF dataset 에피소드)

`taehunkim/so101_pick_cube_test` 에 진단 증거로 보존:

| ep | 내용 | 의미 |
|---|---|---|
| 0,1 | 실기기 teleop pick-and-place | 정답(real) |
| 2 | sim replay (async PD) | state 딜레이 + grasp 실패 |
| 3 | sim **kinematic** replay (state≡action, 딜레이 0) | 딜레이 0인데도 grasp 실패 → **딜레이가 원인 아님**을 입증 |
| (4) | calibration(`--arm_mapping=follower`) 적용 replay | **grasp+place 성공(2026-06-30)**. arm `real_home`=[-7.87,-4.48,4.44,4.18,-5.28], gripper a=1.171/b=-17.13(close 6.085→-10°·open 100→100°). plain bridge(self-collision ON). 잔여 정밀도=§9 |

## 8. 참고

- `so101_robot.json` — 실기기 calibration (homing_offset·range_min/max, 6관절).
- `scripts/ece_4560/real/{read_position,set_position}.py` — 실기기 degree 읽기/명령(측정 도구).
- `scripts/ece_4560/real/so101_gui.py` (`--run-sequence`) + `sequences/*.json` — 실기기 시퀀스 실행.
- `scripts/inference/replay_dataset_to_bridge.py` (`--arm_mapping follower`·`--sequence`·`--ramp_in`) ·
  `scripts/datagen/record_real_sequence.py` (시퀀스 sim 실행→LeRobot v3) · `scripts/data/append_sim_episode.py`.
- `src/so101_contract/follower_calibration.py` (**실 follower↔sim 측정 affine, 본 문서 해결책**) ·
  `feature_codec.py` (policy↔sim) · `leader_calibration.py` (leader↔sim).
- `docs/SIM_REAL_INFERENCE_PARITY.md`.

## 9. Sequence 검증 — affine 확정 + 잔여 정밀화 (2026-06-30)

so101_gui 시퀀스(`scripts/ece_4560/real/sequences/*.json`)를 real(`so101_gui.py --run-sequence`)·sim
(`replay_dataset_to_bridge.py --sequence ... --arm_mapping follower --probe_tracking`) 양쪽 실행 →
phase별 achieved 직접 비교.

**결론: affine 은 정확하다.** free-space(접촉 없는) phase 에서 sim achieved 가 real 과 **±0.6°(=노이즈)**
일치. per-joint fit 도 scale a=1.0±0.04·offset≈0 → 보정할 affine 오차 없음. (per-phase Δ 는 sim 추종오차일
뿐 affine 아님 — affine 이 forward+inverse 에 같이 들어가 상쇄. affine 판정은 물리자세/영상으로만.)

**남은 잔차(grasp 성공, 정밀도):**
- **중력 sag**(lift/elbow soft PD 1.6~5° under-shoot) + **바닥접촉 lateral 변형**(pan, 접촉 phase 만) —
  둘 다 sim soft PD(17.8) dynamics 지 calibration 아님. arm stiffness sweep 은 marginal → 보류(원복).
- **큐브 위치 실측 정합**: 왼쪽모서리 +42.5cm·앞모서리 +29cm (`_CUBE_LAYOUT["Cube1"]=(-0.015,0.255)`).
  5mm 정밀 grasp 라 1cm 차이가 corner-hit 좌우. bridge 재시작 시 `place_defaults` 가 이 값 반영.

**도구**: replay `--sequence JSON`(so101_gui 시퀀스 sim 재현·move보간+hold·per-phase achieved) ·
`--ramp_in`(현재자세→첫frame 보간 teleport 방지, 전 모드) · `record_real_sequence.py`(시퀀스 sim 실행→
LeRobot v3 기록, action=실 follower 단위 → 실기기 `lerobot-replay` 호환).
