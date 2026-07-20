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
| `curobo_batch_planner.py` | curobo-datagen | ZMQ REP planner. per-env cube/bowl(base_link) → 6-phase pick-place 궤적(arm deg + gripper feat) 리스트. **multi-env 실행은 IsaacLab lockstep**, planner는 cuRobo BatchMotionPlanner batch 차원을 env 차원으로 쓰고, grasp 후보는 priority 순서대로 pass를 나눠 검증한다. 아래 §grasp/place 파이프라인. |
| `pickplace_sm.py` | isaac-sim | ZMQ REQ + IsaacLab pick_cube env. **서브커맨드 `random`·`fail`·`sweep`** (아래 §실행). 한 번의 B 로 batch plan → per-env 궤적 replay(plan-fail env 는 init hold, 짧은 궤적 last-row 패딩) → per-env `_cubes_in_bowl` 판정. `--num_envs`(기본 1) lockstep. **결정적 replay 위해 `success`/`cube_lost` termination 비활성**(transit 중 그릇 상공 통과 시 `task_done` 조기 발화 버그. time_out 30s 만 유지). |
| `plot_sweep.py` | host `.venv` | `sweep` 결과 JSON → matplotlib 성공맵 PNG. `spawn_area.py` 만 importlib 로드(=isaac 무의존). `--demo` = 합성데이터 렌더 self-check. |
| `build_robot_model.py` | curobo-datagen | (기존) SO-101 cuRobo config 빌더. |

> DR 스폰영역 기하는 `src/sim_to_real/tasks/pick_cube/spawn_area.py`(순수 python 단일 소스) — `pickplace_sm sweep`·`plot_sweep`·env_cfg 가 공유. `python3 …/spawn_area.py` 로 self-check.

## 실행 (터미널 2개)

먼저 **planner** 를 띄우고 `[planner] ZMQ REP :5599` 로그가 뜬 뒤 **SM** 서브커맨드를 실행한다.

```bash
# 1) planner (항상 먼저) — --max_batch_size 는 SM --num_envs 와 맞추면 재초기화 회피
docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
    python /workspace/scripts/cuRobo/curobo_batch_planner.py
```

SM 은 **3 서브커맨드**로 나뉜다 (`pickplace_sm.py {random|fail|sweep} …`). 공용 인자:
`--task`(env variant) · `--num_envs`(기본 1) · `--livestream 2`(인터랙티브 키 필수) ·
`--cam_eye/--cam_target`(env-상대 카메라) · `--seed` · `--bowl_tol`.

```bash
# random — 통상 랜덤 DR 배치(인터랙티브 관전). N=새 layout · R=같은 layout · B=plan+run
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py random \
    --task SimToReal-SO101-PickCube-DR-v0 --livestream 2 --num_envs 1
#   (자동 배치+MP4 녹화: random --auto_trials N --record_viewport_dir DIR)

# fail — sweep 결과의 place/plan-fail 셀 좌표만 재현(인터랙티브). N=다음 batch · R=같은 · B=run
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py fail \
    --results /workspace/outputs/curobo_sweep/sweep_results.json --livestream 2 --num_envs 1

# sweep — DR 스폰영역 전체 grid+boundary 정량평가 → JSON(자동/headless)
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py sweep \
    --task SimToReal-SO101-PickCube-DR-v0 --num_envs 12 --headless \
    --out /workspace/outputs/curobo_sweep/sweep_results.json
#   시각화(host): uv run --no-sync python scripts/cuRobo/plot_sweep.py \
#                   --results outputs/curobo_sweep/sweep_results.json
```

- **키보드**(random·fail, `--livestream` 필수, WebRTC 클라이언트 입력): **N**·**R**·**B** — 위 각 모드 주석.
  R/N 은 manipulation 중에도 = 진행 동작 취소 + 리셋. Ctrl-C·창 닫기로 종료.
- 관전: WebRTC `:49100` (원격은 `.env`에 `LIVESTREAM=1` + `PUBLIC_IP`). viewer 는 env 0 추종.
- `--task` variant: `PickCube-v0`(고정) · `-DR-v0`(full DR) · `-DRBase-v0` · `-Eval-v0` · `-DR-Eval-v0`.
  **-DR* 는 `scene.replicate_physics=False`**(robot color DR 이 per-env OmniPBR 를 바인딩 — Fabric
  replication 켜지면 전 로봇 동색). robot color 는 env 당 팔레트 색 1개로 로봇 전체 재색칠(prestartup,
  런 내내 고정 — 렌더 있는 datagen/bridge 경로용).
  > ⚠ **SM 은 state-only(렌더 없음)** → `_build_env` 가 시각 DR(robot color·lights·focal)을 끄고
  > `replicate_physics=True` 로 되돌린다. -DR 의 `replicate_physics=False`+런타임 de-instance 가
  > **headless physx view 를 깨뜨리기**(`get_dof_velocities` 크래시) 때문. 시각 DR 은 카메라 경로 전용.

### sweep 서브커맨드 인자
`--nx`/`--ny`(interior grid 해상도) · `--boundary_n`(경계 곡선당 샘플) · `--trials`(셀당 반복,
판정노이즈 평균) · `--yaw`(도 고정 또는 `random`) · `--out`(JSON). 최외곽 bell·y edge + 최내곽
base/bowl/exclude 경계 셀을 항상 포함한다(`spawn_area.sweep_targets`).

## planner로 보내는 데이터 (`plan_pickplace` 요청, N=env 수)

- `cubes`: `[[x, y, grasp_z, qw,qx,qy,qz] ×N]` — per-env **6-DOF pose**. planner가 quat에서 cube face normal을 직접 추출한다.
- `bowl`: `[x, y]`(공용) 또는 `[[x, y] ×N]`(per-env) — place 목표로 **xy만** 소비.
- `start`: `[[6 joint rad, SO101 순서] ×N]` — per-env **reset·settle 후 실제 robot joint**(arm 5개 사용).
- `knobs`: 선택. `grasp_z_off`, `grip_open`, `grip_close`, `bowl_pull`, `tau_max_deg`, `seed`.
- 응답: `{"trajectories": [[[6]×T]|null ×N]}` — 실패 env 는 null.

## grasp/place 파이프라인 (`curobo_batch_planner.py`)

`plan_pickplace` = **6-phase**: approach → grasp(descend) → lift → transit → release(그릇 상공) → retreat(home). place-descent 는 제거됨(아래).

**tool frame·grasp 기하** (`assets/robots/so101.yml` `extra_links.tcp_grasp` 단일 소스):
- **tool_frames = `tcp_grasp`** (손가락 사이 pinch 점). scaffold 커밋이 이 extra_link 를 실수로 제거해 tool 이 `moving_jaw` 로 바뀌면서 face-plane 후보 IK 전멸했던 적 있음 → **복원 필수**. `docs/TROUBLESHOOTING.md` 참조.
- `tcp_grasp` = gripper_link 기준 `(0.012, -0.015, -0.025)+Ry(π-0.0487)`. approach 축 = tcp z(하향), closing 축 = tcp x.
- **★pad center 조준**(tcp 아님): fixed jaw **inner face center** 는 tcp 서 `FIXED_INNER_CENTER=(0.0215, 0.0147, 0.0463)`(closing·lateral·jaw-아래방향, m) 만큼 떨어짐(pad 이 tcp 아래 46mm·옆 15mm). tcp 를 cube center 에 조준하면 pad 이 face 서 크게 벗어나 **edge/corner 를 잡음** → simple pose가 **pad center proxy** 를 cube face center 밖 `FIXED_JAW_CLEAR_TARGET`(4mm 부근)으로 조준: `tcp_tgt = cube + (CUBE_HALF+clear)·n − R·FIXED_INNER_CENTER`.

**manifold grasp 후보 선택** (`cand_pose_manifold` — pink_ik_bridge_node §4·§6 이식):
- **왜 manifold-first 인가**: SO-101 은 pan(z-yaw)+3×pitch(평행축)+wrist_roll(tool-z) 라 **tool 접근축이 항상 pan 수직평면 안**에 있다. 도달 가능한 orientation 은 `R = Rz(pan)·Ry(-α)·R_TOPDOWN·Rz(ρ)·TCP_TWIST` 3-파라미터 manifold 가 전부. 이전 방식(closing x축 = world face normal 고정 + face-plane tilt)은 face 가 pan 평면과 어긋난 만큼 **구성 자체가 manifold 밖** → IK 전멸(실측 825 후보 중 26 solve·gate 통과 0)이 재설계 계기.
- **α** = 접근축 tilt(0=top-down). `ALPHA_SCAN_DEG` = `0,+5,-5,…,±50°` 21개, |α| 오름차순 ± interleave = 검사 우선순위. +α 는 wrist 를 base 쪽으로 당김(원거리 2R 해소), -α 는 반대(근거리 최소반경 해소).
- **ρ(wrist_roll) = -Δψ/cosα**, `Δψ = wrap90(ψ_face − (pan+90°))` — closing 축 수평투영을 cube yaw face normal 에 정렬(정사각 90° 대칭이라 Δψ∈[-45°,45°)). 1차 근사 잔차 + TCP twist 수평성분은 고정점 루프 안 **실측 잔차 feedback** 으로 소거(<0.1°).
- **결합 게이트 `|Δψ·tanα| ≤ τ`**(=closing 축 수평이탈각, 기본 `TAU_MAX_DEG=10°`, `knobs.tau_max_deg`): yaw 어긋남이 크면 큰 |α| 후보 자동 배제. top-down 부근(α≤10°)은 최악 Δψ=45° 서도 항상 생존.
- **TCP twist**: `tcp_grasp` 가 gripper_link·Ry(π-0.0487) 라 tcp ẑ 가 wrist_roll 축과 **2.79°** 어긋남 → trailing `Ry(-0.0486795)` 를 R 에 bake(미포함 시 후보가 2.79° off-manifold — cuRobo 회전 수렴 예산 소진 + pad 조준 ~2.4mm 편향).
- **pan 고정점 ×3**: tcp 목표 = `pad_target − R·FIXED_INNER_CENTER` 의 lateral offset 이 ρ 와 함께 돌아 pan 평면을 벗어남 → `pan ← atan2(tcp_tgt − PAN_AXIS_XY)` 재정렬.
- fixed jaw 가 놓일 face = ρ 보상 후 closing 축(R x̂) 최근접 내적으로 **자동 결정**(선별 로직 없음). pad 조준 식은 기존과 동일: `tcp_tgt = cube + (CUBE_HALF+clear)·n̂ − R·FIXED_INNER_CENTER`. position-only IK 는 사용하지 않는다.
- **미러 branch(ρ+π) 생성 안 함**: τ 게이트 하 |ρ|≤~70°<100° 라 기본 branch 가 항상 wrist gate 통과, 미러는 항상 탈락(wrist ~223° 뒤집기, 사용자 금지) — per-pass 1후보=1 plan_pose 라 latency 만 2배.
- cuRobo 성공 후보를 FK로 검증해 `e_normal/e_tangent/e_height`, wrist delta, face angle gate를 모두 통과한 첫 후보를 사용한다.

**grasp 후보 검증 = 3D face-center error**(`_grasp_face_error`, IK 후 **FK 실측** 기준):
- IK→FK 실측 fixed jaw inner face center(`grasp_tcp + R·FIXED_INNER_CENTER`)를 cube face center(`cube + CUBE_HALF·n`)와 비교, `e = fixed_inner − face_center` 를 3축 분해: **e_normal**(closing clearance) · **e_tangent**(face 평면 lateral) · **e_height**(world-z). 옛 1D closing clearance 만 봐선 pad 이 face center 를 통과하는지 검증 못 해 edge 를 잡았다.
- 게이트: `e_normal∈[3,5]mm` · `|e_tangent|≤E_TANGENT_MAX` · `|e_height|≤E_HEIGHT_MAX` · `|alpha(face_angle)|≤40°` · wrist_roll 물리범위. IK 성공만으론 불충분.
- **★tilt(α) 허용**: 후보 생성 단계에서 `±50°`(τ 게이트 내)를 모두 검사하고, 실제 도달성·grasp 기하는 cuRobo full-pose 계획과 FK gate에서 판정한다.
- **⚠ 40mm 큐브 하드웨어 한계**: pad center를 face center에 앉히려면 큰 grasp tilt가 필요(geom 실측)하고, SO-101 5-DOF에서는 `tilt≥55°` 부근부터 IK가 전멸한다. 75mm jaw + ≥2mm 책상 clearance 때문에 table clamp가 descend를 제한하며, best reachable tilt에서도 `e_h`가 남는다. E_TANGENT/E_HEIGHT_MAX는 **best-achievable** 범위에 맞춘 gate다.

**cube orientation**(`_cube_face_normals`): 큐브 USD 는 rest 에 body 축 하나가 **수직**이라 yaw/Euler로 orientation을 축약하면 gimbal-lock 성격의 불안정한 face 방향이 나온다. planner는 base_link cube quaternion에 solver-frame 회전 quaternion을 합성한 뒤 body basis를 직접 회전하고, 수평 성분이 큰 body axis 두 개의 ±normal(≤4개)을 그대로 반환한다(face 선별은 후보 생성기가 자동).

**grasp 물리** — **bounded shallow-preload**(stall-press 대체): descend 를 물리 pad 최저점(tcp+`PAD_LOW_OFF`·ẑ, `PAD_LOW_OFF`=0.075=so101.yml 실측 fixed jaw tip drop) 이 `TABLE_TOP+TABLE_MARGIN`(4mm→실제 ≥2mm, IK/tilt 잔차 보상) 아래로 못 가게 `tstar` clamp → **책상 무접촉**. **table 은 world obstacle 로 넣지 않음**(로봇이 책상 위 장착→base 구가 상판 안=전 plan start-collision) — clamp 로 대체.

**place = release-above-bowl**: 깊은 linear 하강(bowl disable)은 pad 가 동적 bowl 을 밀어냄 → 제거. transit 이 큐브를 그릇 위(`TRANSIT_Z`=0.25, rim clearance +8.7mm)로 옮기면 `SETTLE_STEPS` hold 후 개방(그릇 안 낙하). 드롭 XY 는 그릇 중심서 base 쪽으로 `BOWL_PULL`(0.03) 당김(near-rim 착지, 사용자 "드롭 너무 멀다"). bowl obstacle 상시 on + `_place_bowl_obstacle` 로 실 그릇좌표 매요청 동기화.

**gripper**: approach 동안 init(feature0)→open ramp(접근 전 급개방 방지) · release 후 retreat 서 다시 init 복원.

**start self-collision**: 접힌 init(lift -100°) 이 cuRobo sphere 모델서 self-col(lower_arm↔shoulder·base↔upper_arm) → so101.yml `self_collision_ignore` 에 두 쌍 추가(접힘서만 겹치는 비인접쌍).

**오프라인 검증**: `--self-check-geom`(후보 기하만 — manifold 위 존재·closing↔face 정렬·τ 게이트 assert, GPU plan 불요·단 torch/curobo import 라 컨테이너 안에서) · `--self-test`(고정 큐브 4 env — yaw 0/22.5/45° 포함) · 별도 curobo-datagen 컨테이너에서 ZMQ REQ 로 임의 cube/start plan(isaac 불요, ~40s/plan). diag=`/workspace/outputs/planner_diag.log`(host `./outputs` 마운트) — `[manifold]` cand 별 alpha·rho·wrist_roll·face_alpha·e_norm·e_tan·e_h·ok·selected 기록(선택 grasp 추적).

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
