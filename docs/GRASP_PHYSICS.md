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
이유).

## 4. 충돌 형상 점검 결과 (참고)

- 로봇 USD `assets/robots/so101_follower.usd` 의 `jaw`/`gripper` 콜라이더는
  `convexDecomposition` (instanced prim — Isaac Lab modifier 로 런타임 덮어쓰기
  불가). 평평한 그립 패드가 hull 로 둥글려져 평면-평면 대신 line/point contact 가
  되기 쉽다. 강한 클램프력(§3)으로 1차 보완. 추후 손가락 안쪽 면에 얇은 box
  충돌 패드를 추가하면 근본 개선 가능(미적용).
- 큐브는 해석적 Box(size 1 × scale 0.025) → 완전 평면. 큐브 쪽은 양호.

## 5. 적용한 변경

### 5.1 actuator 프로파일 이식 (PickPen + PickCube)

- `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`
- `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py`

위 §3 의 leisaac 검증값으로 두 task 모두 일치시킴.

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
떨어지지 않으면 정상. 큐브가 그리퍼 몸체에 박히면 손가락 충돌 패드(미적용 §4)
또는 contactOffset 재조정 필요.

## 7. 트레이드오프 / 주의

- actuator 변경은 **TA.1 PD 튜닝·TB.3 RL 학습(`model_70.pt`) 당시 설정과
  달라진다.** 기존 체크포인트로 평가/롤아웃 시 동역학 불일치 가능 — 재평가 권장.
- `enabled_self_collisions=True` 는 자세에 따라 자기충돌로 관절이 멈출 수 있다.
  rest pose·궤적에서 self-collision 막힘이 없는지 GUI 확인.
- solver iteration 8/1 → 4/4 변경은 leisaac 정합이지만, 다객체 PPO(2048+ env)
  에서 안정성은 별도 smoke 로 확인.
