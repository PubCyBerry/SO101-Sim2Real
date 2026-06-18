# Grasp 물리·충돌 튜닝 (PickPen / PickCube)

SO-101 그리퍼가 시뮬에서 객체(펜·큐브)를 안정적으로 집고 들어올리도록 하는
물리·충돌 설정과 그 근거를 정리한다. leisaac(`ref_repos/leisaac`) 검증값과의
비교가 핵심.

## 1. 증상

GUI teleop(`teleop_se3_agent.py --teleop_device=so101leader`) 녹화 중 관찰:

| 증상 | 설명 |
|---|---|
| **A. 큐브가 그리퍼에 꽂힘/박힘** | 큐브가 손가락 사이가 아닌 그리퍼 몸체에 관통·정지 |
| **B. 잡혀야 할 때 안 잡힘** | 손가락이 닿았는데 미끄러져 빠짐 |
| **C. 들어올릴 때 떨어짐** | 잡은 듯하다가 가속 시 낙하 |

## 2. leisaac 의 grasp 메커니즘

leisaac 은 두 축을 분리한다.

### 2.1 grasp "판정"은 물리가 아니라 기하학 프록시

`ref_repos/leisaac/.../tasks/lift_cube/mdp/observations.py::object_grasped`

```python
grasped = (‖cube_pos − ee_jaw_frame_pos‖ < diff_threshold=0.02)
          AND (gripper_joint_pos < grasp_threshold=0.26 rad)
```

- contact force·contact 이벤트를 보지 않는다. **"EE 가 객체 2 cm 이내 + 그리퍼가
  0.26 rad(~15°)보다 닫힘"** 이면 잡힌 것으로 간주.
- 판정 기준점은 jaw 링크 자식 frame 에 offset `(-0.021, -0.070, 0.02)` 을 준 지점
  (`single_arm_env_cfg.py` 의 `ee_frame` target index 1) — **손가락 사이 grasp
  존**. 거리 허용 2 cm 라 불완전 접촉에도 관대.

> 우리 task 는 grasp 자체를 종료 조건으로 쓰지 않고 "객체가 그릇/컵 안" 으로
> 성공을 판정하므로 이 프록시를 그대로 쓰진 않는다. 하지만 *물리적으로* 객체가
> 그리퍼에 붙어 있어야 한다는 점은 동일하고, 그 부분이 아래 actuator 에 달려 있다.

### 2.2 물리적으로 안 떨어지는 건 actuator 설정

`ref_repos/leisaac/.../assets/robots/lerobot.py::SO101_FOLLOWER_CFG`

```python
ImplicitActuatorCfg(effort_limit_sim=10, velocity_limit_sim=10,
                    stiffness=17.8, damping=0.60)   # arm·gripper 공통
ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                              solver_position_iteration_count=4,
                              solver_velocity_iteration_count=4,
                              fix_root_link=True)
soft_joint_pos_limit_factor=1.0
```

### 2.3 teleop/replay 중 gripper effort 는 동적으로 낮춘다

leisaac teleop/replay 루프는 매 step `dynamic_reset_gripper_effort_limit_sim()` 을
호출한다. 이 함수는 gripper 에 가장 가까운 rigid object 질량을 보고 gripper joint
effort limit 을 `object_mass / 0.15` 로 낮춘다. 즉 actuator cfg 의 `10 Nm` 은
상한이고, 작은 큐브를 다룰 때는 과한 클램프력으로 collision hull 안쪽까지 밀어
넣지 않도록 실시간으로 완화한다.

## 3. 비교 — 이전 우리 설정 vs leisaac

| 항목 | 이전 (우리) | leisaac = 현재 적용값 | 영향 |
|---|---|---|---|
| gripper effort_limit_sim | 1.5 | **10** | 클램프력 상한 6.7× |
| gripper stiffness / damping | 300 / 60 | **17.8 / 0.6** | soft PD, 응답성↑ |
| arm effort / stiff / damp | 3.0 / 400 / 80 | **10 / 17.8 / 0.6** | — |
| gripper/arm velocity_limit_sim | 6.0 / 5.5 | **10** | — |
| enabled_self_collisions | 미설정(False) | **True** | 손가락-팔 자기충돌 |
| solver pos / vel iteration | 8 / 1 | **4 / 4** | leisaac 정합 |
| soft_joint_pos_limit_factor | 미설정 | **1.0** | 전 관절 full range |

**핵심 메커니즘:** ImplicitActuator 의 정상상태 클램프 토크는
`min(stiffness × 위치오차, effort_limit)`. 이전 설정은 `stiffness=300, effort=1.5`
→ 위치오차 `1.5/300 ≈ 0.005 rad (0.3°)` 만 생겨도 토크가 1.5 Nm 에 포화. leader 를
아무리 더 닫아도 클램프력이 1.5 Nm 를 못 넘어 **들어올릴 때 미끄러짐(C)·접촉
미끄러짐(B)** 발생. leisaac 은 `stiffness=17.8, effort=10` 이라 오버클로즈할수록
최대 10 Nm 까지 그립력이 올라간다. 이 강한 클램프가 convexDecomposition 손가락의
line-contact 한계를 보완한다(leisaac 이 finger collision 특수 튜닝 없이도 잡히는
이유). 다만 teleop/replay 에서는 §2.3 의 동적 effort reset 이 함께 적용되어,
작은 물체를 계속 10 Nm 로 누르지는 않는다.

## 4. 충돌 형상 점검 결과 (참고)

- 로봇 USD `assets/robots/so101_follower.usd` 의 `jaw`/`gripper` 콜라이더는
  `convexDecomposition` (instanced prim — Isaac Lab modifier 로 런타임 덮어쓰기
  불가). 평평한 그립 패드가 hull 로 둥글려져 평면-평면 대신 line/point contact 가
  되기 쉽다. 이번 적용은 leisaac 과 동일하게 로봇 USD/URDF 를 수정하지 않고,
  동적 gripper effort 와 PhysX contact 전역값으로 과한 관통을 줄인다.
- 큐브는 해석적 Box(size 1 × scale 0.025) → 완전 평면. 큐브 쪽은 양호.

## 5. 적용한 변경

### 5.1 actuator 프로파일 이식 (PickPen + PickCube)

- `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`
- `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py`

위 §3 의 leisaac 검증값으로 두 task 모두 일치시킴.

### 5.1.1 dynamic gripper effort 포팅 (PickPen + PickCube)

- `src/sim_to_real/utils/gripper_effort.py`: leisaac 의 질량 기반 gripper effort
  reset 을 프로젝트 내부 helper 로 포팅. 기본은 leisaac 식을 따르되, cube 낙하를
  피하려고 `min_effort=0.5` 를 둔다.
- `scripts/environments/teleoperation/teleop_se3_agent.py`: action 적용 직전 매 step
  helper 호출. `--disable_dynamic_gripper_effort` 로 끌 수 있고,
  `--min_gripper_effort` 로 최소 effort 를 조절한다.
- `scripts/environments/teleoperation/replay.py`: replay 도 같은 helper 사용.
- `PickPenEnvCfg` / `PickCubeEnvCfg`: `dynamic_reset_gripper_effort_limit=True`.

이 변경은 **로봇 USD/URDF 를 수정하지 않는다.** leisaac 처럼 scene/env runtime 에서
gripper effort 를 완화한다.

### 5.1.2 PhysX contact 전역값 leisaac 정합

`PickPenEnvCfg` / `PickCubeEnvCfg`:
- `bounce_threshold_velocity = 0.01`
- `friction_correlation_distance = 0.00625`

### 5.2 큐브 contactOffset 분리 (P3)

`scripts/environments/author_pick_cube_scene.py`:
- 정적·두꺼운 면(책상/매트/그릇): `CONTACT_OFFSET_DEFAULT = 0.004` 유지.
- grasp 대상 큐브: `CUBE_CONTACT_OFFSET = 0.002`. 손가락 콜라이더 offset 과
  합산돼 큐브가 표면에서 ~1 cm 떨어진 채 잡히던 "거리 두고 잡힘" 완화. 큐브 CCD +
  `solverPositionIterationCount=32` 가 관통을 방어.

### 5.3 책상/매트 물리 머티리얼 (P4)

`scripts/environments/author_pick_cube_scene.py`:
- `DeskFriction` 물리 머티리얼(static 0.9 / dynamic 0.8 / restitution 0 /
  frictionCombineMode max) 을 `DeskTop`·`DeskMat` 에 bind. 이전엔 미지정 →
  PhysX 기본(마찰 ~0.5)이라 큐브 미끄러짐·그릇 밀림 가능.

### 5.4 그릇 미끄러짐 보정 (P5)

4-cube state machine triage에서 각 큐브를 넣는 순간에는 `inside=True`였지만,
다음 큐브로 이동하는 접촉 때문에 동적 그릇이 10 cm 이상 밀리고 먼저 넣은 큐브가
밖으로 튕겨 나갔다. 실제 데스크매트 위 150 mm 그릇은 이렇게 쉽게 움직이지 않도록
다음 값을 적용했다.

- `BOWL_MASS = 0.80 kg`
- `BowlFriction`: static 1.8 / dynamic 1.5 / combine max
- `physxRigidBody:angularDamping = 8.0`, `linearDamping = 2.0`

State machine도 release 후 바로 다음 큐브로 가는 대신 bowl 위 `transport_height`까지
후퇴하는 `retreat` phase를 추가해, 이미 넣은 큐브/그릇을 낮은 자세로 긁고 지나가는
경로를 줄였다.

### 5.5 씬 author pxr/PhysxSchema 재작성 + 그릇 SDF 충돌 + 큐브 질량 차등 (2026-06-08)

`author_pick_cube_scene.py` 를 문자열 USDA 조립에서 공식 pxr/PhysxSchema 스키마 API
(`UsdGeom`/`UsdPhysics`/`UsdLux`/`UsdShade`/`PhysxSchema`)로 전면 재작성. 출력 USD 의
물리 거동은 보존하되 다음을 개선·정정했다.

- **그릇 충돌**: 명시적 box 패널 다발(`_bowl_collision_walls`, 144개) → 단일 watertight
  mesh + SDF(`UsdPhysics.MeshCollisionAPI` `approximation="sdf"` +
  `PhysxSchema.PhysxSDFMeshCollisionAPI`). 외벽+내벽+상단 림+바닥을 닫은 두께 있는 shell
  이라 SDF 가 오목 캐비티를 정확히 표현(box ledge 에 큐브가 얹히던 문제 근절). 시각은
  기존 single-surface `Wall` mesh 유지(얇은 벽 룩).
  - `sdfResolution = 128`: 256 은 `gym.make` 시 SDF 베이킹이 분 단위로 길어져 부팅이
    hang 처럼 보인다. 150mm 단순 곡면 그릇엔 128 로 충분(큐브가 통과하면 상향).
- **그릇 질량**: `BOWL_MASS` 0.80 → 0.25 kg (≈250 g 두께 있는 플라스틱 그릇 현실화).
  마찰 1.8/1.5 는 매끄러운 플라스틱 현실값(0.3~0.5)보다 높지만, 데스크매트 위 큐브 투입
  충격에 그릇이 밀리지 않도록 의도적으로 유지(§5.4).
- **큐브 질량 크기별 차등**: 단일 35 g → `CUBE_MASSES` 로 40mm(Cube1/2) 35 g,
  50mm(Cube3/4) 55 g. 의자다리 커버 폼이 속이 약간 비어 쉘(표면적)비례에 가깝게 잡음
  (부피 완전비례 68 g 보다 보수적 — 50mm grasp 안정 고려).
- **조명**: scene.usd 에 `UsdLux.DomeLight`+`DistantLight(KeyLight)` 를 직접 author 해
  usdview 단독 검증이 가능해졌다. `pick_cube_env_cfg.py` 의 `/World/DomeLight` 는 제거
  (조명 소스 단일화). 멀티 env 학습에서 per-env 조명 복제로 과노출이 보이면 scene.usd
  intensity 하향 또는 `/World` 공용 dome 으로 복귀.

검증 상태: USD 구조 유효성(Stage.Open — 조명·mesh 정상)·물리값 보존(큐브 mass/CCD/
damping/solver/friction/contactOffset 가 origin/main 과 동등)은 확인. Isaac Lab 런타임
로드/teleop 는 단일 GPU 환경에서 다른 isaacsim 세션과 경합 시 `gym.make` 가 hang 하므로
GPU 여유 시 또는 GUI 로 별도 검증해야 한다(아래 TROUBLESHOOTING 참고).

## 6. 검증 방법

```powershell
# USD 재생성 (좌표·치수 변경 시)
uv run scripts\environments\author_pick_cube_scene.py

# GUI teleop 로 grasp 체감
uv run scripts\environments\teleoperation\teleop_se3_agent.py `
    --task=SimToReal-SO101-PickCube-v0 --teleop_device=so101leader `
    --port=COM5 --num_envs=1 --device=cuda --enable_cameras
```

손가락을 큐브 옆면에 정렬해 닫았을 때 (1) 표면 근처에서 잡히고 (2) 들어올릴 때
떨어지지 않으면 정상. 큐브가 그리퍼 몸체에 박히면 먼저
`--min_gripper_effort` 를 낮춰 release 전 과한 클램프를 줄이고, 그래도 반복되면
contactOffset·충돌 형상 보강을 별도 검토한다.

## 7. 트레이드오프 / 주의

- actuator 변경은 **TA.1 PD 튜닝·TB.3 RL 학습(`model_70.pt`) 당시 설정과
  달라진다.** 기존 체크포인트로 평가/롤아웃 시 동역학 불일치 가능 — 재평가 권장.
- `enabled_self_collisions=True` 는 자세에 따라 자기충돌로 관절이 멈출 수 있다.
  rest pose·궤적에서 self-collision 막힘이 없는지 GUI 확인.
- solver iteration 8/1 → 4/4 변경은 leisaac 정합이지만, 다객체 PPO(2048+ env)
  에서 안정성은 별도 smoke 로 확인.

## 8. 충돌 근사 = SDF Mesh (grasp 표면 sim2real, 2026-06-18)

**관찰**: sim 메쉬 시각화에서 그리퍼·큐브가 실제보다 크게 잡힘. 원인 = grasp 관여 mesh 의
충돌 근사가 **convexDecomposition/convexHull** 이라 손가락 사이 오목면을 메워 collision 부피가
visual 보다 부풀려짐 → 가짜 grip(실제 얇은 jaw 보다 큰 영역으로 잡음).

**PhysX 사실(결정적)**: 동적 rigid body 에서 **triangle mesh·meshSimplification 은 지원 안 됨 →
convexHull 로 fallback**. 동적 body 에서 오목/실제 형상을 정확히 표현하는 유일한 근사 = **SDF
(signed distance field)**. 즉 grasp 표면 충실 옵션은 SDF 하나뿐(나머지는 부풀림).

| 요소 | 충돌 근사 | author |
|---|---|---|
| 큐브 collider | **SDF**(라운드 mesh, res 256) — 옛 analytic Box 대체, visual 정합 | `author_pick_cube_scene.py` `_apply_sdf_mesh_collision` |
| jaw/gripper collider | convexDecomposition → **SDF**(res 256) | `scripts/assets/set_gripper_jaw_sdf_collision.py`(usd-core raw 스키마, `.preSDF.bak`) |
| 팔 링크(base~wrist) | convexDecomposition 유지 | (grasp 무관·저비용) |
| 그릇 collider | convexDecomposition 유지 | num_envs>1 SDF cooking crash 회피(§기존) |

- **비용**: SDF cooking + `sdfResolution`(메모리 ∝ res³). 256=비용/정밀 균형. 로봇 mesh 는 env 간
  공유/인스턴싱이라 1회 cooking(Factory/Forge 가 수천 env SDF 기어 사용 선례). ⚠ 4096-env SDF
  안정성은 smoke 검증 권장(그릇 SDF 는 과거 multi-env crash).
- **큐브 크기**: Cube1/2 30→**40mm**, Cube3/4 40→**50mm**(질량 35/55g). `_CUBE_INIT_STATES` z·
  `_CUBE_VOLUME_INSET`·SM `CUBE_SIZES`/`gripper_open`(40/50 앵커) 동기 갱신.
- ⚠ **재검증**: 큐브 collider sharp box→라운드 SDF + 크기 확대로 grasp 접촉면 변화 → SM/RL grasp
  성공률 재측정 필요. 진단 = `scripts/assets/viz_collision_overlay.py`(visual vs SDF source vs convexHull
  오버레이, `outputs/collision_overlay/`).
