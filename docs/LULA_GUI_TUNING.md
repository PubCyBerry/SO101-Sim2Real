# Lula / RMPFlow GUI 튜닝 가이드 (SO-101)

Isaac Sim **GUI 위젯**으로 SO-101 의 Lula RMPFlow·robot description 을 튜닝하는 방법.
대상 파일은 `scripts/environments/follow_target_so101.py` 의 `--controller rmpflow` 와
`assets/robots/rmpflow/` 의 두 yaml 이다.

> 코드(헤드리스) 검증은 `follow_target_so101.py --headless --selftest` 로, GUI 튜닝은 이 문서로.
> 관련 함정은 `TROUBLESHOOTING.md` (RMPFlow 미튜닝 scaffold / floating base) 참조.

---

## SO-101 모션 솔버 선택 (먼저 읽기)

SO-101 USD 는 **URDF(Lula 모델) ↔ USD articulation root 가 ~90° Z 회전** 어긋나게 baked 돼 있다. 이 정합 문제는 **"Isaac Sim 안에서 Lula 의 URDF-FK 와 USD-world 를 섞어 쓰는 경우"에만** 영향을 준다.

| 솔버 / 도구 | SO-101 사용 | 비고 |
|---|---|---|
| **Lula Test Widget (GUI)** | ❌ 불가 | base pose 를 `articulation.get_world_pose()` 로만 잡아 90° 보정을 못 넣음 → EE frame 어긋남·IK 실패. **위젯은 `default_q` 편집(Robot Description Editor)에만 사용** |
| **코드 내 Lula IK** (`follow_target_so101.py --controller ik`, `pick_cube_state_machine.py`) | ✅ 동작 | `RMPFLOW_BASE`(90° 보정 quat) + per-solve shift 를 직접 주입해 정렬. follow 는 sub-cm. 단 5-DOF 정밀도 한계 |
| **코드 내 RMPFlow** (`--controller rmpflow`) | △ 동작·헐렁 | config 가 미튜닝 scaffold → ~0.1 m. 부드러움·obstacle 회피 데모용 |
| **cuMotion + ROS** (Isaac ROS manipulation, PATH E) | ✅ **이 문제와 무관** | 별도 ROS 프로세스가 **`base_link` 프레임**에서 계획하고 물체 포즈도 `base_link` 기준 TF 로 받음 → world-level 회전이 식에 안 들어옴 |

**요점**:
- 정밀 task-space 모션이 필요하면 → **cuMotion+ROS**(PATH E, `.claude/worktrees/isaac-sim-ros-pickplace/`). 프레임 문제 없음. 진짜 숙제는 5-DOF 계획(joint-goal 로 해결됨)과 grasp 물리.
- Isaac Sim 안 단독 데모/오라클이면 → **코드 내 Lula IK**(보정값 주입). 위젯 GUI 튜닝은 이 에셋엔 불가.
- 근본 해결(위젯까지 쓰려면) = URDF↔USD base 프레임을 일치시키는 에셋 재작업(미수행).

---

## 0. 무엇을 어디서 튜닝하나

| 도구 (메뉴 `Tools > Robotics`) | 편집 파일 | 튜닝 대상 |
|---|---|---|
| **Lula Test Widget** | `so101_rmpflow_config.yaml` (게인) | RMPFlow 추종/회피 거동 — target 따라가며 실시간 확인 |
| **Lula Robot Description Editor** | `so101_robot_description.yaml` | `default_q`(rest 자세, "머리 박음" 해결) · collision sphere |

| 파일 | 경로 | 역할 |
|---|---|---|
| URDF | `assets/robots/urdf/so_arm101.urdf` | 링크/조인트 정의 |
| Robot Description | `assets/robots/rmpflow/so101_robot_description.yaml` | cspace(5축), `default_q`, collision sphere |
| RMPFlow Config | `assets/robots/rmpflow/so101_rmpflow_config.yaml` | `rmp_params` 게인 |

- SO-101 arm = **5-DOF** → **position-only**(orientation target 끄기). EE frame = **`gripper_frame_link`**.
- yaml 게인을 위젯에서 만족스럽게 잡으면 `follow_target_so101.py --controller rmpflow` 에 **자동 반영**된다(같은 파일).

---

## 1. GUI 여는 법

### 방법 A (권장) — `--tune` 모드
SO-101 을 **world 원점**에 베이스 고정해 띄우고(+ground plane), 위젯 확장도 자동 활성화한다.
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
  python scripts/environments/follow_target_so101.py --tune
```
- 자체 컨트롤러를 구동하지 않으므로 위젯의 RMPFlow 와 충돌하지 않는다.
- **로봇 로드·베이스 고정 단계를 건너뛴다** (이미 `fix_root_link` 적용됨).
- ⚠ **왜 원점인가**: Lula Test Widget 의 IK follow·EE-frame 시각화는 `set_robot_base_pose` 를 호출하지 않아 **robot 이 world 원점에 있다고 가정**한다(`test_scenarios.py`). cube_desk 배치(world ~1.84,…)면 위젯 솔버가 원점 기준으로 풀어 `/Lula/end_effector` 가 원점에 박히고 IK 가 `Failed to compute Inverse Kinematics` 로 실패한다. 게인·`default_q` 튜닝은 배치 무관이라 결과는 cube_desk 에 그대로 적용된다.

### 방법 B — 표준 풀 GUI
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac isaacsim
```
- `isaacsim.exp.full` 앱. 위젯 확장이 기본 활성이라 `Tools > Robotics` 가 처음부터 보인다.
- 단 로봇은 직접 로드 + 베이스 고정 필요 (아래 [부록](#부록-방법-b-에서-로봇-수동-배치)).

### 원격(모니터 없는 SSH)일 때 — livestream
```bash
... follow_target_so101.py --tune --livestream 2
```
Isaac Sim WebRTC Streaming Client 로 접속. (로컬 디스플레이가 있으면 불필요)

### `Tools > Robotics` 메뉴가 안 보이면
그 메뉴는 **위젯 확장이 켜져야** 생긴다. 안 보이면:
`Window > Extensions` → `lula test widget` 검색 → 토글 **ON** (default_q 편집은 `robot description` 검색 → `xrdf_editor` ON).
방법 A 는 `--tune` 이 자동으로 켜준다.

---

## 2. Lula Test Widget — RMPFlow 튜닝

> ⛔ **이 SO-101 에셋에선 위젯의 live follow(IK·RMPFlow)가 정렬되지 않는다.**
> URDF(Lula 모델)의 base 프레임이 USD articulation root 와 **~90° Z 회전** 어긋나게 baked 돼 있다
> (원점·zero-joint 측정: Lula FK gripper_frame_link=(0.39,0,0.23) 팔이 +X / 실제 USD jaw=(0.04,−0.30,0.29) 팔이 −Y).
> `pick_cube_state_machine.py` 는 손으로 맞춘 `RMPFLOW_BASE`(쿼터니언 ~90°Z) + per-solve shift 로 보정하지만,
> **위젯은 base pose 를 `articulation.get_world_pose()`(보정 없음)로만 잡아 이 90° 를 못 넣는다**(로봇을 회전 spawn 해도 시각·Lula 가 같이 돌아 상대 오차 불변). → EE frame 이 손끝에서 떨어지고 IK 가 `Failed to compute Inverse Kinematics`.
>
> **결론**: 위젯은 **`default_q` 편집(§3, Robot Description Editor — 프레임 무관)** 에만 쓰고,
> **RMPFlow 게인은 yaml 편집 + `follow_target_so101.py --controller rmpflow` 헤드리스 검증**으로 튜닝한다
> (그 스크립트는 보정된 `RMPFLOW_BASE` 를 써 Lula 가 USD 와 정렬된다). 아래 §2 절차는 frame 정합이 된 로봇(예: Franka)에서의 일반 흐름 참고용.

### 2.1 로드
1. **▶ Play**.
2. 메뉴 **`Tools > Robotics > Lula Test Widget`**.
3. 상단에서 스테이지의 **SO-101 articulation 선택**.
4. 파일 3개 지정 후 **`Load Selected Config`**:
   - **Robot URDF** → `assets/robots/urdf/so_arm101.urdf`
   - **Robot Description YAML** → `assets/robots/rmpflow/so101_robot_description.yaml`
   - **RmpFlow Config YAML** → `assets/robots/rmpflow/so101_rmpflow_config.yaml`
5. **Select End Effector Frame** → `gripper_frame_link`.
6. **Use Orientation Targets** = **OFF** (5-DOF position-only).
7. **Visualize End Effector Pose** = ON.

> 위젯은 스테이지의 로봇 base pose 를 자동으로 읽는다. 헤드리스 스크립트의
> `RMPFLOW_BASE_*` 상수는 위젯 튜닝엔 불필요.

### 2.2 실행·관찰 (RmpFlow 패널)
- ⚠ **버튼 주의 — "Follow Target" 이 두 곳에 있다**:
  - **`Lula Kinematics Solver` 패널의 Follow Target** = **정확(exact) IK**. 5-DOF SO-101 은 임의 자세를 못 풀어 `Failed to compute Inverse Kinematics` 를 도배하며 로봇이 안 따라온다 → **쓰지 말 것**.
  - **`RmpFlow` 패널의 Follow Target** = **RMPFlow**(우리가 튜닝하는 것). base pose 를 잡고 5-DOF 를 best-effort 로 부드럽게 추종 → **이걸 쓴다.**
  - 누르면 시나리오가 전환돼 IK 경고가 멈춘다. 단 이 버튼은 **벽 obstacle 을 ~(0.4,0,0.1) 에 하나 생성**한다(회피 테스트용, 정상).
- ⚠ **target 이 멀리 생긴다**: 위젯은 target 을 항상 **world (0.5, 0, 0.5)** 에 만든다(`test_scenarios.py` 의 `_create_target`). SO-101 은 5-DOF 라 도달 반경이 ~0.35 m 뿐이라 (0.5,0,0.5)(원점에서 0.7 m)도 멀다. → Stage 에서 **`/World/Target` 선택 → Property > Transform > Translate 를 `(0.2, 0, 0.25)`** 정도(원점 로봇 기준 도달 범위)로 옮긴 뒤 드래그한다. (`--tune` 은 로봇을 원점에 둔다)
- **`Follow Target`**(RmpFlow) → target 큐브 생성 → 위처럼 도달 범위로 옮기면 EE 가 추종.
- **`Toggle Debugging Mode`** → **collision sphere 표시**. 떨림/자기충돌/반발 지점을 눈으로 확인.
- **Sinusoidal Target** (frequency·radius·height 슬라이더) → 자동 궤적 추종으로 지연·떨림 정량 관찰.

### 2.3 튜닝 루프
위젯엔 게인 슬라이더가 없다. **yaml 을 에디터로 수정 → 위젯에서 `Load Selected Config` 재클릭 → 관찰** 을 반복.

---

## 3. Robot Description Editor — `default_q` (머리 박음 해결)

"바닥에 머리 박고 흔들" 의 핵심 원인은 `default_q: [0,0,0,0,0]`(SO-101 이 앞/아래로 처진 자세)이 RMPFlow 의 rest/null-space 자세이기 때문이다. 공식 설명:

> target 을 줘도 RmpFlow 는 **default 자세에 가까운 해로 null-space 를 해소**하고, target 이 없거나 못 닿으면 default 자세로 이동한다.

해결:
1. 메뉴 **`Tools > Robotics > Lula Robot Description Editor`**.
2. articulation 선택 → **Set Joint Properties** 패널.
3. 각 관절 **Joint Position** 을 "팔꿈치 들고 EE 가 책상 위를 향하는" ready 자세로 조정(뷰포트 실시간 반영).
4. **Export** → `so101_robot_description.yaml` 덮어쓰기 → `default_q` 갱신.
5. Lula Test Widget 에서 `Load Selected Config` 로 재확인.

> 같은 에디터에서 collision sphere(로봇 자기 형상) 추가/편집·export 도 가능(`Link Sphere Editor`).

---

## 4. `rmp_params` 치트시트 (`so101_rmpflow_config.yaml`)

| 파라미터 | 의미 | 조정 방향 |
|---|---|---|
| `cspace_target_rmp.metric_scalar` | `default_q`(home)로 당기는 힘 | 크면 target 못 닿음 / 작으면 자세 처짐. 절충 **1~10** |
| `target_rmp.accel_p_gain` · `accel_d_gain` | EE→target 당기는 P/D | 못 닿으면 P↑ / 떨리면 D↑ |
| `joint_velocity_cap_rmp.max_velocity` | 관절 속도 상한 | 느리면 ↑ / 진동하면 ↓ |
| `collision_rmp.metric_modulation_radius` | 반발이 켜지는 거리 | **좁은 책상은 0.05~0.08** (기본 0.25 는 과함) |
| `collision_rmp.metric_scalar` · `repulsion_gain` | 장애물 반발 세기 | 밀려나면 ↓ |
| `damping_rmp.accel_d_gain` · `metric_scalar` | 전역 감쇠 | 떨리면 ↑ |

### 증상별 빠른 처방
| 증상 | 조정 |
|---|---|
| 바닥에 머리 박음 | `default_q` ready 자세로(§3) + `cspace_target_rmp.metric_scalar` ↑(예 5) |
| 흔들흔들(떨림) | `damping_rmp.accel_d_gain` ↑ · `target_rmp.accel_d_gain` ↑ · `max_velocity` ↓ |
| target 에 못 닿음 | `target_rmp.accel_p_gain` ↑ · `cspace_target_rmp.metric_scalar` ↓ |
| 장애물에 밀려 내려감 | `collision_rmp.metric_modulation_radius` ↓ · `metric_scalar` ↓ |
| 너무 느림 | `joint_velocity_cap_rmp.max_velocity` ↑ |

> **비교 기준**: 공식 튜닝된 config —
> `.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/rmpflow/franka_rmpflow_common.yaml`
> (7-DOF Franka 용이나 gain 스케일 참고에 유용).

---

## 5. 튜닝이 미치는 범위

| 항목 | 위치 | GUI 로? |
|---|---|---|
| RMPFlow 게인 (`rmp_params`) | `so101_rmpflow_config.yaml` | Lula Test Widget ✓ (스크립트에도 자동 반영) |
| rest 자세 `default_q` | `so101_robot_description.yaml` | Robot Description Editor ✓ |
| collision sphere(로봇 자기 형상) | `so101_robot_description.yaml` | Robot Description Editor ✓ |
| **큐브/그릇 obstacle 크기·위치** | `follow_target_so101.py` 의 `_OBSTACLE_SCALES` | ✗ — 스크립트 편집 (위젯 obstacle 은 별도 WallObstacle) |

yaml 두 개를 잡은 뒤 obstacle 크기가 과하면 `_OBSTACLE_SCALES`(현재 큐브 0.06 m)만 줄이면 된다.

---

## 6. 코드로 디버깅 (선택)
```python
rmpflow.visualize_collision_spheres()   # collision sphere 렌더
rmpflow.set_ignore_state_updates(True)  # RMPFlow ↔ 시뮬레이터 분리(알고리즘만 관찰)
```

---

## 부록: 방법 B 에서 로봇 수동 배치
1. `File > New` → 뷰포트에 `assets/robots/so101_follower.usd` 드래그(또는 `File > Add Reference`).
2. **베이스 고정**: Stage 트리에서 `base_link` 선택 → 우클릭 `Create > Physics > Joint > Fixed Joint`. (안 하면 Play 시 쓰러짐)
3. **▶ Play** 후 §2 진행.

---

## 참고
- Lula Test Widget 확장: `isaacsim.robot_motion.lula_test_widget`
- Robot Description Editor: `isaacsim.robot_setup.xrdf_editor`
  ([docs](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_robot_description_editor.html))
- RMPFlow 공식 config: `motion_policy_configs/<robot>/rmpflow/*.yaml`
