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
`--cam_eye/--cam_target`(env-상대 카메라) · `--seed` · `--bowl_tol` ·
`--plan_timeout_s`(기본 900 s, planner 사망 시 무한 정지 방지 — 0=무제한) ·
`--log_every`(기본 0=끔; headless sweep 에서 켜면 step 당 3줄이 stdout 을 잡아먹는다).

> ⚠ 성공 판정 `--bowl_tol` 은 **xy 거리만** 본다(z 무시). 드롭 XY 가 rim 근처(`BOWL_PULL`)라
> 그릇 밖 책상에 튄 큐브도 성공으로 셀 수 있다 — env 의 `object_in_container` 와는 계약이 다르다.
> 과거 sweep 수치와의 비교 가능성 때문에 현행 유지 중이다.

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

# record — IsaacLab HDF5 데이터 녹화(random --auto_trials 전용, leisaac 방식 RecorderManager)
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py random \
    --task SimToReal-SO101-PickCube-DR-v0 --num_envs 4 --auto_trials 25 \
    --headless --enable_cameras --record_hdf5 /workspace/datasets/pick_cube_sm.hdf5
#   변환(Isaac·lerobot 불요): python scripts/convert/isaaclab2lerobotv3.py \
#     --hdf5_files datasets/pick_cube_sm.hdf5 --output_dir datasets/pick_cube_sm_v3

# record — LeRobot v3 직기록(leisaac --use_lerobot_recorder 동형, single-env 전용·성공만 저장)
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py random \
    --task SimToReal-SO101-PickCube-DR-v0 --num_envs 1 --auto_trials 25 \
    --headless --enable_cameras --record_lerobot /workspace/datasets/pick_cube_sm_v3
```

## record 모드 (`--record_hdf5` / `--record_lerobot`)

`random --auto_trials N` 에서만 동작. 두 백엔드는 에피소드 규격·종료 term 공유:

| | `--record_hdf5` | `--record_lerobot` |
|---|---|---|
| 포맷 | IsaacLab HDF5 (사후 `isaaclab2lerobotv3.py` 로 v3 변환) | LeRobot v3 즉시 (`LeRobotV3DatasetWriter`, 변환 불필요) |
| multi-env | ✅ env당 1 demo(`data/demo_N`) | ❌ `--num_envs 1` 전용 (leisaac 동형 제약) |
| 저장 범위 | 실패도 저장(`success` attr 로 구분) | **성공 에피소드만** (실패 버퍼 폐기) |
| 메모리 | 에피소드 동안 **VRAM** 누적 (~1 GiB/env/에피소드) | step 마다 CPU 스트리밍 |
| 압축 | `lzf` + frame-chunk (`hdf5_compression.hdf5_handler`) | LeRobot v3 비디오 인코딩 |
| 보존 정보 | frame + `initial_state`·`actions` | frame(action/state/3-cam)만 |
| 구현 | stock RecorderManager + `DatagenRecorderTerm` | `SO101LeRobotRecorderManager`(`src/sim_to_real/data/lerobot_recorder_manager.py`) |

⚠ `--record_lerobot` 은 기존 출력 디렉터리를 **덮어쓴다**(overwrite, `record_state_machine` 규약).

- **에피소드 규격**: `[정지 2 s(--preroll_s)] → 이동 → pick-place → init 복귀 → [정지 1 s(--posthold_s)] → 자동 종료`.
  종료는 termination term(`returned_home_after_motion`)이 발화 → env auto-reset 순간
  RecorderManager 가 HDF5 로 export. `success` attr = `placed_and_returned`(복귀+그릇 안).
- **플래닝/cold-start 대기 미포함**: plan ZMQ 블록 중엔 env.step 이 없어 기록 자체가 없고,
  settle·직전 트라이얼 꼬리 프레임은 pre-roll 직전 `recorder_manager.reset()` 이 폐기한다.
- **HDF5 내용**: `obs_x/joint_pos`(절대 rad·SO101 순서) · `obs_x/images/{top,wrist,front}`(uint8) ·
  `applied_target`(slew 통과 적용 target) + `initial_state`·`actions`. stock `states`·`obs`·
  `processed_actions` 는 읽는 코드가 없어 꺼져 있다(`actions` 는 `num_samples` attr 산출용으로 유지).
- **용량/메모리**: 3-cam 640×480 uint8 @30 Hz ≈ 2.64 MiB/frame/env — 379-step 에피소드 원본
  ≈ 999 MiB/env 가 auto-reset 까지 **VRAM** 에 누적된다. 48.9 GB 카드 기준 실측 피크 =
  2 env ~36 GB · **8 env 45.0 GB(92%, 사실상 상한)**. 더 키우려면 recorder term 에 `.cpu()` 를
  붙인다 — replay 가 8 env 에서 +41% 느려지는 대가다(`09_TACIT_KNOWLEDGE.md` §13.3).
- **압축**: `lzf` + 프레임 단위 청크. export 는 env 순차 blocking 이라 `--num_envs` 에 비례해
  심 루프를 세운다 — 실측 gzip(4) 10.8 s/demo → lzf 3.7 s/demo, 디스크는 2배.
  프리셋 표·선택 근거 = `docs/spec/09_TACIT_KNOWLEDGE.md` §13.
- 트라이얼 단위 seed 재현은 없음(run 전체 `--seed` 1회, 이후 연속 RNG 스트림).
- 변환은 `scripts/convert/isaaclab2lerobotv3.py`(env-free, success demo 만) → LeRobot v3.

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
- `cube_half`: 큐브 반변(m). SM 이 `cube_specs` 단일 소스에서 읽어 pose 와 **함께** 보낸다.
  없으면 planner 가 40 mm 로 폴백(구버전 호환).
- `knobs`: 선택. `grasp_z_off`, `grip_open`, `grip_close`, `bowl_pull`, `tau_max_deg`, `rho_cap_deg`,
  `chord_center_ratio`, `transit_z`. ⚠ **`seed` 는 no-op**(cuRobo `reset_seed()` 가 인자를 안 받아
  외부 seed 로 해를 흔들 수 없다 = planning 은 입력에 대해 결정적). SM 은 보내지 않는다.
- 응답: `{"ok":true, "trajectories": [[[6]×T]|null ×N], "diagnostics":[…]}` — 실패 env 는 null.
  요청 처리 중 예외가 나도 planner 는 죽지 않고 `{"ok":false,"err":…}` 로 답한다(REP 를
  빠뜨리면 REQ 클라이언트가 영원히 블록된다). SM 쪽은 `--plan_timeout_s`(기본 900 s) 초과 시
  해당 batch 를 plan-fail 로 기록하고 소켓을 재연결해 진행한다.

## grasp/place 파이프라인 (`curobo_batch_planner.py`)

`plan_pickplace` = **5개 계획 + gripper ramp**: approach → grasp(descend) → lift → transit → retreat(home).
release 는 별도 계획이 아니라 transit 끝 자세에서의 gripper 개방 ramp 다(하강 없음). place-descent 는 제거됨(아래).

**tool frame·grasp 기하** (`assets/robots/so101.yml` `extra_links.tcp_grasp` 단일 소스):
- **tool_frames = `tcp_grasp`** (손가락 사이 pinch 점). scaffold 커밋이 이 extra_link 를 실수로 제거해 tool 이 `moving_jaw` 로 바뀌면서 face-plane 후보 IK 전멸했던 적 있음 → **복원 필수**. `docs/TROUBLESHOOTING.md` 참조.
- `tcp_grasp` = gripper_link 기준 `(0.012, -0.015, -0.025)+Ry(π-0.0487)`. approach 축 = tcp z(하향), closing 축 = tcp x.
- **★pad center 조준**(tcp 아님): fixed jaw **inner face center** 는 tcp 서 `FIXED_INNER_CENTER=(0.0215, 0.0147, 0.0463)`(closing·lateral·jaw-아래방향, m) 만큼 떨어짐(pad 이 tcp 아래 46mm·옆 15mm). tcp 를 cube center 에 조준하면 pad 이 face 서 크게 벗어나 **edge/corner 를 잡음** → simple pose가 **pad center proxy** 를 cube face 밖 `FIXED_JAW_CLEAR_TARGET`(4mm 부근)으로 조준한다.

**manifold grasp 후보 선택** (`cand_pose_manifold` — pink_ik_bridge_node §4·§6 이식):
- **왜 manifold-first 인가**: SO-101 은 pan(z-yaw)+3×pitch(평행축)+wrist_roll(tool-z) 라 **tool 접근축이 항상 pan 수직평면 안**에 있다. 도달 가능한 orientation 은 `R = Rz(pan)·Ry(-α)·R_TOPDOWN·Rz(ρ)·TCP_TWIST` 3-파라미터 manifold 가 전부. 이전 방식(closing x축 = world face normal 고정 + face-plane tilt)은 face 가 pan 평면과 어긋난 만큼 **구성 자체가 manifold 밖** → IK 전멸(실측 825 후보 중 26 solve·gate 통과 0)이 재설계 계기.
- **α** = 접근축 tilt(0=top-down). `ALPHA_SCAN_DEG` = `0,+5,-5,…,±50°` 21개, |α| 오름차순 ± interleave = 검사 우선순위. +α 는 wrist 를 base 쪽으로 당김(원거리 2R 해소), -α 는 반대(근거리 최소반경 해소).
- **ρ(wrist_roll) = -Δψ/cosα**, `Δψ = wrap90(ψ_face − (pan+90°))` — closing 축 수평투영을 cube yaw face normal 에 정렬(정사각 90° 대칭이라 Δψ∈[-45°,45°)). 1차 근사 잔차 + TCP twist 수평성분은 고정점 루프 안 **실측 잔차 feedback** 으로 소거(<0.1°).
- **결합 게이트 `|Δψ·tanα| ≤ τ`**(=closing 축 수평이탈각, `TAU_MAX_DEG=25°` **전 구간 상수**, `knobs.tau_max_deg`): yaw 어긋남이 크면 큰 |α| 후보 자동 배제. top-down 부근(α≤10°)은 최악 Δψ=45° 서도 항상 생존. 예전의 도달반경 적응 램프(10°→25°)는 개별 실패셀에 맞춰 옮긴 임계값이라 상수로 대체했다 — 후보 품질은 FK 게이트가 독립 보장하므로 τ 는 후보 수만 늘린다.
- **TCP twist**: `tcp_grasp` 가 gripper_link·Ry(π-0.0487) 라 tcp ẑ 가 wrist_roll 축과 **2.79°** 어긋남 → trailing `Ry(-0.0486795)` 를 R 에 bake(미포함 시 후보가 2.79° off-manifold — cuRobo 회전 수렴 예산 소진 + pad 조준 ~2.4mm 편향).
- **pan 고정점**: tcp 목표 = `pad_target − R·FIXED_INNER_CENTER` 의 lateral offset 이 ρ 와 함께 돌아 pan 평면을 벗어남 → `pan ← atan2(tcp_tgt − PAN_AXIS_XY)` 재정렬. 반복 횟수 고정이 아니라 **잔차 수렴**(|Δpan|·closing 잔차 < 1e-4 rad, 상한 12회)으로 끝내고, 미수렴 후보는 폐기한다.
- fixed jaw 가 놓일 face = ρ 보상 후 closing 축(R x̂) 최근접 내적으로 **자동 결정**(선별 로직 없음). `rho` cap으로 closing 축이 face normal과 어긋나면 face-center에서 시작한 jaw chord가 cube center를 비껴 moving jaw가 모서리를 밀어낸다. `CHORD_CENTER_RATIO=0.5`가 face tangent 방향 `0.5·CUBE_HALF·tan(face_angle)` 보정을 더해 chord를 cube center 쪽으로 옮긴다. 최종 식은 `tcp_tgt = cube + (CUBE_HALF+clear)·n̂ + chord_shift·t̂ − R·FIXED_INNER_CENTER`. position-only IK 는 사용하지 않는다.
- **미러 branch(ρ+π) 생성 안 함**: τ 게이트 하 |ρ|≤~70°<100° 라 기본 branch 가 항상 wrist gate 통과, 미러는 항상 탈락(wrist ~223° 뒤집기, 사용자 금지) — per-pass 1후보=1 plan_pose 라 latency 만 2배.
- cuRobo 성공 후보를 FK로 검증해 `e_normal/e_tangent/e_height`, wrist delta, face angle gate를 모두 통과한 첫 후보를 사용한다.

**grasp 후보 검증 = 3D face-center error**(`_grasp_face_error`, IK 후 **FK 실측** 기준):
- IK→FK 실측 fixed jaw inner face center(`grasp_tcp + R·FIXED_INNER_CENTER`)를 cube face center(`cube + CUBE_HALF·n`)와 비교, `e = fixed_inner − face_center` 를 3축 분해: **e_normal**(closing clearance) · **e_tangent**(face 평면 lateral) · **e_height**(world-z). 옛 1D closing clearance 만 봐선 pad 이 face center 를 통과하는지 검증 못 해 edge 를 잡았다.
- 게이트: `e_normal∈[2,8]mm`(조준 타깃 4 mm) · `|e_tangent|≤22 mm` · `|e_height|≤28 mm` · `|alpha(face_angle)|≤40°` · wrist_roll 물리범위. IK 성공만으론 불충분.
- clearance **만** 어긋난 후보는 실측 오차만큼 조준을 face 쪽으로 민 보정본을 1회 더 시도한다.
  batch pass 경로는 보정본을 뒤 pass 후보로 덧붙여(추가 plan 호출 0) rescue 와 같은 규칙을 쓴다.
- **★tilt(α) 허용**: 후보 생성 단계에서 `±50°`(τ 게이트 내)를 모두 검사하고, 실제 도달성·grasp 기하는 cuRobo full-pose 계획과 FK gate에서 판정한다.
- **⚠ 40mm 큐브 하드웨어 한계**: pad center를 face center에 앉히려면 큰 grasp tilt가 필요(geom 실측)하고, SO-101 5-DOF에서는 `tilt≥55°` 부근부터 IK가 전멸한다. 75mm jaw + ≥2mm 책상 clearance 때문에 table clamp가 descend를 제한하며, best reachable tilt에서도 `e_h`가 남는다. E_TANGENT/E_HEIGHT_MAX는 **best-achievable** 범위에 맞춘 gate다.

**cube orientation**(`_cube_face_normals`): 큐브 USD 는 rest 에 body 축 하나가 **수직**이라 yaw/Euler로 orientation을 축약하면 gimbal-lock 성격의 불안정한 face 방향이 나온다. planner는 base_link cube quaternion에 solver-frame 회전 quaternion을 합성한 뒤 body basis를 직접 회전하고, 수평 성분이 큰 body axis 두 개의 ±normal(≤4개)을 그대로 반환한다(face 선별은 후보 생성기가 자동).

**grasp 물리** — **bounded shallow-preload**(stall-press 대체): descend 를 물리 pad 최저점(tcp+`PAD_LOW_OFF`·ẑ, `PAD_LOW_OFF`=0.075=so101.yml 실측 fixed jaw tip drop) 이 `TABLE_TOP+TABLE_MARGIN`(4mm→실제 ≥2mm, IK/tilt 잔차 보상) 아래로 못 가게 `tstar` clamp → **책상 무접촉**. **table 은 world obstacle 로 넣지 않음**(로봇이 책상 위 장착→base 구가 상판 안=전 plan start-collision) — clamp 로 대체.

**place = release-above-bowl**: 깊은 linear 하강(bowl disable)은 pad 가 동적 bowl 을 밀어냄 → 제거. transit 이 큐브를 그릇 위(`TRANSIT_Z`=0.21 — ring keep-out 이 rim 을 스스로 피해 0.25 서 인하)로 옮기면 `SETTLE_STEPS` hold 후 개방(그릇 안 낙하). 드롭 XY 는 그릇 중심서 base 쪽으로 `BOWL_PULL`(0.03) 당김(near-rim 착지, 사용자 "드롭 너무 멀다"). bowl obstacle 상시 on + `_place_bowl_obstacle` 로 실 그릇좌표 매요청 동기화.

**gripper**: approach 동안 init(feature0)→open ramp(접근 전 급개방 방지) · 폐합 뒤 `GRASP_HOLD_STEPS=5` 정지 hold로 접촉을 안정화한 후 lift · release 후 retreat 서 다시 init 복원.

**start self-collision**: 접힌 init(lift -100°) 이 cuRobo sphere 모델서 self-col(lower_arm↔shoulder·base↔upper_arm) → so101.yml `self_collision_ignore` 에 두 쌍 추가(접힘서만 겹치는 비인접쌍).

**오프라인 검증**: `--self-check-geom`(후보 기하만 — manifold 위 존재·closing↔face 정렬·τ 게이트 assert, GPU plan 불요·단 torch/curobo import 라 컨테이너 안에서) · `--self-test`(고정 큐브 4 env — yaw 0/22.5/45° 포함) · 별도 curobo-datagen 컨테이너에서 ZMQ REQ 로 임의 cube/start plan(isaac 불요, ~40s/plan). diag=`/workspace/outputs/planner_diag.log`(host `./outputs` 마운트) — `[manifold]` cand 별 alpha·rho·wrist_roll·face_alpha·e_norm·e_tan·e_h·ok·selected 기록(선택 grasp 추적).

## 정량 성능 & grasp 레버 (Phase D, sm-eval)

`pickplace_sm sweep`(kinematic 도달) + `fail --auto`(물리 place) 로 측정한 DR 스폰영역
성공률 사다리. 54-sphere 모델에서 `rho=12°`의 wrist 안전대는 유지하고, 대각 yaw의
face-center chord miss를 반 보정해 yaw-zero·yaw-random 모두 최초 시도 100%를 달성했다.

| phase | 성공 (kind) |
|---|---|
| baseline yaw0 (구 spawn 187셀) | 165/187 = **88%** (base_arc 68%·bell 68%) |
| +R1 R2 R3 yaw0 | 178/187 = **95%** (bell 68→100%) |
| +pan축 spawn 가드 +R2' yaw0 (183셀) | 183/183 = **100%·회귀0** (base_arc 68→100%) |
| +rho-cap yaw-random ×3 | 1300/1305 = **99.62%** |
| 54-sphere 재평가, chord 보정 전 yaw0 / random×3 | 182/183 = **99.45%** / 1298/1305 = **99.46%** |
| +chord-center 0.5× + grasp hold, yaw0 / random×3 | **183/183 = 100%** / **1305/1305 = 100%** |

- pan축 spawn 좌표수정(마운트원점→pan축)이 phase3 서 base_arc 68→100% 를 견인(도달불가
  -x corner 를 스폰영역서 배제, 187→183셀). R1R2R3 는 bell 을 100% 로.
- 54-sphere 실패 8건 targeted: rho cap 12/14/16/18° = 5/6/7/8 성공이었으나 18°는
  64-env 첫 planning만 약 17분으로 느렸다. `RHO_CAP=12°`를 유지하고 chord 0.5×를 적용하자
  reset/plan seed 0·1·2에서 24/24, full yaw-random에서 1305/1305 성공했다.
- 최종 산출물: `scratch/2026-07-22-curobo-sm-model54-final/`(summary JSON·5 PNG·timing·VRAM CSV).

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
