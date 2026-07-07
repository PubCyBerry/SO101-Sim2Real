# cuRobo pick-place — 인터랙티브 SM (2-proc, 순수 ZMQ)

SO-101 pick-place를 **cuRobo collision-free planner**(planning)와 **IsaacLab env**(execution)로
분리해 돌리는 최소 구성. 두 프로세스는 ZMQ 하나로만 통신한다 (ROS·executor 없음).

```
┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
│ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py                  │
│  cuRobo v0.8 full pick-place │◀──REQ───│  pick_cube env variant           │
│  REP  tcp://*:5599           │───REP──▶│  reset(DR)→plan→env.step→check   │
└──────────────────────────────┘  :5599  └──────────────────────────────────┘
```

- 두 서비스 모두 `network_mode: host` → planner는 `tcp://*:5599`, SM은 `tcp://127.0.0.1:5599`.
- SM은 env 안에서 큐브/그릇 pose를 직접 읽고(`env.scene`) joint를 직접 구동(`env.step`)한다.
  ROS·TF·별도 executor가 필요 없다 (그래서 "isaac sim bridge"가 이 SM 하나로 접힘).

## 파일

| 파일 | 컨테이너 | 역할 |
|---|---|---|
| `curobo_batch_planner.py` | curobo-datagen | ZMQ REP planner. cube/bowl(base_link) → 5-phase pick-place 궤적(arm deg + gripper feat). 아래 §grasp/place 파이프라인. |
| `pickplace_sm.py` | isaac-sim | ZMQ REQ + IsaacLab pick_cube env. 키보드 상태머신(N/R 리셋 · B plan+manipulate), EE pose 매 step 출력. **결정적 replay 위해 `success`/`cube_lost` termination 비활성**(들린 큐브가 transit 중 그릇 상공 통과 시 `task_done` 조기 발화→place 전 auto-reset 버그. time_out 30s 만 유지, 끝에 `_cube_in_bowl` 판정). |
| `build_robot_model.py` | curobo-datagen | (기존) SO-101 cuRobo config 빌더. |

## 실행 (터미널 2개)

```bash
# 1) planner
docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
    python /workspace/scripts/cuRobo/curobo_batch_planner.py

# 2) SM (planner "ready" 로그 뜬 뒤)
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py \
    --task SimToReal-SO101-PickCube-DR-v0 --livestream 2
```

- **키보드 인터랙티브**(livestream 입력, `--livestream` 필수):
  - **N** = 새 DR layout 리셋(로봇 → init) · **R** = 이전과 같은 layout 리셋 · **B** = cuRobo plan + manipulation 실행
  - R/N 은 manipulation 중에도 = 진행 동작 취소 + 로봇 pose·scene 리셋. Ctrl-C·창 닫기로 종료.
- 관전: WebRTC `:49100` (원격은 `.env`에 `LIVESTREAM=1` + `PUBLIC_IP`).
- `--task`: env variant 선택 — `PickCube-v0`(고정) · `-DR-v0`(full DR) · `-DRBase-v0` · `-Eval-v0` · `-DR-Eval-v0`.
- `--cam_eye`/`--cam_target`: viewport 카메라(env 원점 상대).

## planner로 보내는 데이터 (`plan_pickplace` 요청)

- `cubes`: `[[x, y, grasp_z, qw,qx,qy,qz]]` — **6-DOF pose**(위치+quat). planner가 quat으로 큐브 yaw face-align.
- `bowl`: `[x, y]` — planner는 place 목표로 **xy만** 소비(그릇은 컨테이너+정적 obstacle이라 6D 불요).
- `start`: `[[6 joint rad, SO101 순서]]` — **reset·settle 후 실제 robot joint**. planner가 이 자세부터 계획(arm 5개 사용).

## grasp/place 파이프라인 (`curobo_batch_planner.py`)

`plan_pickplace` = **6-phase**: approach → grasp(descend) → lift → transit → release(그릇 상공) → retreat(home). place-descent 는 제거됨(아래).

**tool frame·grasp 기하** (`assets/robots/so101.yml` `extra_links.tcp_grasp` 단일 소스):
- **tool_frames = `tcp_grasp`** (손가락 사이 pinch 점). scaffold 커밋이 이 extra_link 를 실수로 제거해 tool 이 `moving_jaw` 로 바뀌면서 pan-plane 후보 IK 전멸했던 적 있음 → **복원 필수**. `docs/TROUBLESHOOTING.md` 참조.
- `tcp_grasp` = gripper_link 기준 `(0.012, -0.015, -0.025)+Ry(π-0.0487)`. approach 축 = tcp z(하향), closing 축 = tcp x.
- **★pad center 조준**(tcp 아님): fixed jaw **inner face center** 는 tcp 서 `FIXED_INNER_CENTER=(0.0215, 0.0147, 0.0463)`(closing·lateral·jaw-아래방향, m) 만큼 떨어짐(pad 이 tcp 아래 46mm·옆 15mm). tcp 를 cube center 에 조준하면 pad 이 face 서 크게 벗어나 **edge/corner 를 잡음** → `_cand_pose` 가 **pad center** 를 cube face center 밖 `FIXED_JAW_CLEAR_TARGET`(3mm)로 조준: `tcp_tgt = cube + (CUBE_HALF+clear)·n − R·FIXED_INNER_CENTER`.

**grasp 후보 검증 = 3D face-center error**(`_grasp_face_error`, IK 후 **FK 실측** 기준):
- IK→FK 실측 fixed jaw inner face center(`grasp_tcp + R·FIXED_INNER_CENTER`)를 cube face center(`cube + CUBE_HALF·n`)와 비교, `e = fixed_inner − face_center` 를 3축 분해: **e_normal**(closing clearance) · **e_tangent**(face 평면 lateral) · **e_height**(world-z). 옛 1D closing clearance 만 봐선 pad 이 face center 를 통과하는지 검증 못 해 edge 를 잡았다.
- 게이트: `e_normal∈[1,6]mm` · `|e_tangent|≤E_TANGENT_MAX` · `|e_height|≤E_HEIGHT_MAX` · wrist_roll 물리범위. IK 성공만으론 불충분.
- **★tilted 허용**: |α| 는 hard-reject 아님. score 순위 = ①centerline(√(e_t²+e_h²)) ②clearance ③yaw(dpsi) ④wrist branch(fingers-down 선호) ⑤tilt penalty. 전 후보(α 0~55°·sx 양쪽) 평가 후 최소 채택.
- **⚠ 40mm 큐브 하드웨어 한계**: pad center 를 face center 에 앉히려면 α≈55–60° 필요(geom 실측)인데 SO-101 5-DOF 로 unreachable(α≥55 IK 전멸). 75mm jaw + ≥2mm 책상 clearance 라 table clamp 가 descend 를 끊어 pad 이 상단에 걸림 — best α≈45°서 `e_h≈13mm`(수직은 +28mm=큐브 위). E_TANGENT/E_HEIGHT_MAX 를 **best-achievable(12/14mm)로 완화**해 centerline 랭킹이 최소 후보 선택(수직 edge-grip 회피). **진짜 face-center 는 50mm 큐브**(pad 이 더 낮게 앉음) 또는 짧은 jaw 필요.

**cube yaw**(`_cube_yaw`): 큐브 USD 는 rest 에 body 축 하나가 **수직**(eul pitch≈−84°) → naive z-yaw(atan2)는 gimbal-lock 쓰레기(face 서 ~28° 어긋남) → **가장 수평인 body 축**(min|z|=옆면 normal)의 solver-frame azimuth 사용. face-align(ρ=−Δψ/cosα)가 이 yaw 로 closing 축을 정렬.

**grasp 물리** — **bounded shallow-preload**(stall-press 대체): descend 를 물리 pad 최저점(tcp+`PAD_LOW_OFF`·ẑ, `PAD_LOW_OFF`=0.075=so101.yml 실측 fixed jaw tip drop) 이 `TABLE_TOP+TABLE_MARGIN`(4mm→실제 ≥2mm, IK/tilt 잔차 보상) 아래로 못 가게 `tstar` clamp → **책상 무접촉**. **table 은 world obstacle 로 넣지 않음**(로봇이 책상 위 장착→base 구가 상판 안=전 plan start-collision) — clamp 로 대체.

**place = release-above-bowl**: 깊은 linear 하강(bowl disable)은 pad 가 동적 bowl 을 밀어냄 → 제거. transit 이 큐브를 그릇 위(`TRANSIT_Z`=0.25, rim clearance +8.7mm)로 옮기면 `SETTLE_STEPS` hold 후 개방(그릇 안 낙하). 드롭 XY 는 그릇 중심서 base 쪽으로 `BOWL_PULL`(0.03) 당김(near-rim 착지, 사용자 "드롭 너무 멀다"). bowl obstacle 상시 on + `_place_bowl_obstacle` 로 실 그릇좌표 매요청 동기화.

**gripper**: approach 동안 init(feature0)→open ramp(접근 전 급개방 방지) · release 후 retreat 서 다시 init 복원.

**start self-collision**: 접힌 init(lift -100°) 이 cuRobo sphere 모델서 self-col(lower_arm↔shoulder·base↔upper_arm) → so101.yml `self_collision_ignore` 에 두 쌍 추가(접힘서만 겹치는 비인접쌍).

**오프라인 검증**: `--self-test`(고정 큐브 1개) · 별도 curobo-datagen 컨테이너에서 ZMQ REQ 로 임의 cube/start plan(isaac 불요, ~40s/plan). diag=`/workspace/outputs/planner_diag.log`(host `./outputs` 마운트) — `[ladder]` cand 별 wrist_roll·alpha·e_norm·e_tan·e_h·centerline·ok 기록(선택 grasp 추적).

## robot 시작 자세

`env_cfg.scene.robot.init_state.joint_pos`를 아래로 override하고 `reset_robot_joints` jitter를 0으로 →
**frame 0부터** 이 자세로 스폰(중립→init 이동 transient 제거). settle도 이 자세를 hold(zeros면 팔이 흘러내림).

```
shoulder_pan 0 · shoulder_lift -100 · elbow_flex 90 · wrist_flex 50 · wrist_roll -90 · gripper -10  (deg)
```

## 런타임 검증 포인트 (GPU 필요, 호스트에선 문법만 확인됨)

1. **action 단위** — planner row(arm deg + gripper feat)를 `policy_feature_to_sim_joint_radians`로
   sim radian 변환 후 `env.step`. `curobo_executor.py`와 동일 변환이라 정합할 것으로 보나 첫 실행에서 확인.
2. **base_link 프레임** — `subtract_frame_transforms(robot_base, cube)`로 planner 입력(base_link USD 규약)을 만든다.
   planner가 내부에서 `Rz(90)+BASE_T` 보정. 조준이 빗나가면 이 프레임부터 점검.
3. **성공 판정** — `--bowl_tol`(기본 6cm xy)은 튜닝 knob. 엄밀히는 env `object_in_container`와 맞출 것.
4. **auto-reset** — `env.step`이 termination에서 auto-reset. SM은 terminal state를 다음 step 전에 읽고 break.
5. **livestream 상호작용** — plan 대기·종료 중에도 `simulation_app.update()`로 Kit을 계속 pump해야
   zoom/drag/click·키(N/R/B)가 먹는다(안 그러면 프리즈). plan 대기·키 대기 중에도 `simulation_app.update()` pump 유지. **키 입력은 livestream 필수**(headless 는 입력 없음).
