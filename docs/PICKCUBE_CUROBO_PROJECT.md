# SO-101 PickCube cuRobo — 프로젝트 마스터 문서

> **한 줄 요약**: cube_desk 에서 SO-101 팔이 큐브 4개를 그릇에 담는 **결정적 pick-and-place 오라클을
> cuRobo GPU 배치 충돌 플래닝으로 재구현**한다. 해석적 IK(검증됨, 유지)가 grasp config 를 주고,
> cuRobo `plan_cspace`(joint-goal)가 충돌-free 궤적을 만든다. 용도는 **VLA 학습 데이터 생성(sim→real)**
> 과 **학습된 VLA 의 시뮬 추론·평가(real→sim)** 양방향. 해석적 SM([`PICKCUBE_SM_PROJECT.md`])의 천장
> (descend-clip 55.8%)을 충돌 기하 인식으로 돌파하는 후속 트랙.
>
> **현재 상태(2026-06-14 착수)**: ⚪ **계획 확정·문서화 단계**. 승인된 플랜 = `~/.claude/plans/
> so-101-pick-place-vla-federated-shannon.md`. 아직 코드 미착수.
>
> **왜 cuRobo 인가**: 해석적 SM 은 grasp pose 를 "점+heuristic offset"으로 다뤄 비대칭 moving jaw
> (−y 82mm 늘어짐·열림 46mm swing)의 self-clip 을 못 본다. 4-cube 2048-env **55.8%** 천장의 93% 가
> target 큐브 self-clip([`PICKCUBE_SM_PROJECT.md`] §10), blind 튜닝 4연패. **cuRobo 는 그리퍼를 실제
> sphere 기하로 모델**해 clip-free grasp pose 를 선정하고, 이웃 큐브·그릇·카메라 홀더를 장애물로 회피한다.
>
> **작성 기준**: 2026-06-14. 단일 진입점 문서. 해석적 SM 이력 = [`PICKCUBE_SM_PROJECT.md`], RL 트랙 =
> [`PICKCUBE_RL_PROJECT.md`], cuMotion+ROS(실기기) = [`PATH_E_CUMOTION_ROS.md`].

---

## 0. 상태 대시보드

| Phase | 내용 | 상태 | Track |
|---|---|---|---|
| **P0** | cuRobo 설치 · ABI 무손상 게이트 | 🟢 DONE (nvcc 불요·핀 보존·API 검증; Isaac headless smoke 만 첫 sim 때 확인) | A |
| **P1** | so101 cuRobo config — 공식 RobotBuilder sphere fit (홀더 포함) | 🟢 DONE (54 spheres/9링크, so101.xrdf+so101_curobo.yml, IK 81%, Viser·MPC·MotionPlanner 검증) | A |
| **P1+** | SO-101 FK·IK 단독 검증 + joint-goal 인터랙티브 데모 + FK 일관성 진단 | 🟢 DONE (pose-goal 5-DOF 벽 확증·MPC차이 규명·**해석적FK↔cuRoboFK 6.9mm 발산 발견**·joint-goal 데모 작동) | A |
| **P2** | cuRobo 충돌 플래닝 ↔ pick-place 통합 (single-env) | 🟢 DONE (**IPC 사이드카**·4-큐브 side-approach·장애물 회피·DR·livestream+3cam, **~18–20초 4/4**) | A |
| **P3** | **multi-env 배치** 스케일 (BatchMotionPlanner·multi_env world) · throughput | 🟢 DONE (서버 배치+청크·클라 lock-step·grasp-retry. **N=256 chunk=64 retry=1: cube 64.6%·all-4 18.8%·VRAM 20GB·3 epi/s**). all-4 상향 = P4 min_cube_sep | A |
| **P4** | 큐브 간 최소거리 강화 (clustering 해소) | ⚪ TODO | B |
| **P5** | LeRobot v3 데이터(render-batch+시각DR 240-ep) + HF | 🟢 DONE (`taehunkim/so101_sim_pick_cube` v3.0, 143072 frame) | B |
| **P5T** | SmolVLA fine-tune (Docker policy-server train 20k) | 🟢 DONE (`taehunkim/so101_smolvla_sim_pick_cube`, wandb oh1gs82t) | B |
| **P6** | VLA 시뮬 closed-loop 추론 (ROS2 3-process) | 🟢 DONE (action [1,8,6]·17.7Hz arm 구동; 정성 grasp 는 livestream 관찰 권장) | C |

> **🔁 전체 루프(사용자 2026-06-14 확장)**: cuRobo 데이터 → HF → SmolVLA 학습 → sim VLA 추론까지 한 번에
> 검증. 게이트: **10-ep 생성·업로드 먼저 확인 후 80-ep + 학습 + 추론**. 상세·재개 가이드 = **§13**.
| **PD** | ovrtx/ovphysx throughput 실측 (옵션) | ⚪ TODO | D |

> 🎯 **성공 기준**: 2048-env 4-큐브 **~95–99% + clean↑**(knocking 없는 자연 궤적), 에피소드 **<20초**
> (600 step @30Hz). literal 100% 는 비목표(PhysX 비결정성·reach-edge 본질 노이즈).

---

## 1. 태그·중요도 범례

### 상태 태그 (Kanban)

| 배지 | 의미 |
|---|---|
| 🟢 **DONE** | 완료 + 검증됨 (헤드리스 실행·taxonomy·프레임 PNG 확인) |
| 🔵 **IN-PROGRESS** | 현재 진행 중 |
| ⚪ **TODO** | 예정 — 아직 착수 안 함 |
| 🔴 **BLOCKER** | 막힘 — 해결돼야 다음 진행 |
| ⚫ **DROPPED** | 시도 후 폐기 — 교훈만 남김 |

### 중요도

| 배지 | 의미 |
|---|---|
| 🔥 **CRITICAL** | 성패 직결 |
| ⭐ **HIGH** | 품질·일정 큰 영향 |

---

## 2. 목표와 용도 (양방향)

```mermaid
flowchart LR
  subgraph SIM2REAL["sim → real (데이터 생성)"]
    SM["cuRobo SM 오라클<br/>(해석적 IK config + cuRobo plan)"] -->|"고신뢰 expert 궤적"| REC["state-replay 재렌더<br/>3-cam RGB 부착"]
    REC --> DS[(LeRobot v3 데이터셋)]
    DS --> VLA["VLA 정책 학습"]
  end
  subgraph REAL2SIM["real → sim (추론·평가)"]
    VLA --> PS["policy-server<br/>(async + RTC)"]
    PS <-->|"ROS2 obs/action"| GUI["cube_desk single-env GUI<br/>(큐브 drag 인터랙티브)"]
  end
  VLA -.->|"실기기 배포"| REAL["real SO-101"]
  PS <-.->|"같은 서버 (parity)"| REAL
```

- 🔥 **최종 목표**: cube_desk **4-큐브 pick-and-place**, 2048-env 배치 검증 **~95–99% + clean↑**,
  에피소드 **<20초**, retry 발동 최소.
- **용도 ①(sim→real)**: VLA 학습용 **결정적** expert 데이터. (RL 트랙[`PICKCUBE_RL_PROJECT.md`]은
  다양성 담당, cuRobo SM 은 결정적 고신뢰 담당.)
- **용도 ②(real→sim)**: 학습된 VLA 를 cube_desk 시뮬에서 **closed-loop 추론** — 실기기 배포 전 검증.
  GUI 에서 물체를 옮겨도 추종·집기 시도하는지 정성 확인.
- **공유 계약**: 두 방향이 **렌더 env + LeRobot obs/action 계약**(`lerobot_units.py` "North Star:
  observation.images.{top,wrist,front}", joint rad↔deg, gripper 0–100)을 공유한다. **parity**: 같은
  policy-server 가 real/sim 양쪽 구동. 카메라 pose/intrinsic 은 실기기 리그(손목/전면 홀더)에 정합
  (최근 커밋) → real→sim 전이 전제 충족.

---

## 3. 제약 (엄수)

| # | 제약 | 상태 | 이유 | 중요도 |
|---|---|---|---|---|
| C1 | 🚫 grasp weld/attach/인공 유지력 금지 | ✅ 유지 | 물리적 정직 grasp 만(VLA 데이터 validity). cuRobo `attach`/`enable_obstacle` 는 **플래닝 충돌모델 전용** — 실제 grasp 는 그리퍼 close+마찰(30/40mm 물리 grasp 검증됨) | 🔥 |
| C2 | `BOWL_SUCCESS_RADIUS=0.06` 불변 | ✅ 유지 | 성공판정 부풀리기 금지 | 🔥 |
| C3 | **Part A 기구학 불변** (`SO101Kinematics`·`_world_to_base`·q_bias·`--calibrate`) | ✅ 유지 | `--calibrate` 실측 FK err 1.5mm. grasp config 계산에 재사용 — **재작성 금지** | 🔥 |
| C4 | 5-DOF **position 우선·orientation best-effort** | ✅ 유지 | SO-101 5축 → 임의 6-DOF pose 불가. cuRobo 도 **joint-goal(`plan_cspace`)로만** (pose-goal 5-DOF 비가능, PATH E 확증) | 🔥 |
| C5 | **검증은 ≥256-env** | ✅ 유지 | 16-env 76% vs 2048 51.6% — 소표본 과대평가 | ⭐ |
| C6 | cuRobo trajectory 위에 **q_bias 중력보상 유지** | ✅ 신규 | cuRobo 는 kinematic traj, env 는 compliant PD(stiffness 17.8) sag → 보상 없으면 grasp z 미달 | ⭐ |

---

## 4. 아키텍처: 해석적 IK(유지) + cuRobo 충돌 플래닝(신규)

```mermaid
flowchart LR
  IK["SO101Kinematics 해석적 IK<br/>(C3 불변, FK 1.5mm)"] -->|"q_grasp 후보<br/>(±90° roll·side-offset)"| SEL["clip-free grasp 선정<br/>(open-gripper sphere 기하 충돌검사)"]
  SEL -->|"goal_q (5축)"| PLAN["cuRobo plan_cspace<br/>(batch, multi_env)"]
  WORLD["per-env world<br/>(4큐브+그릇+홀더 cuboid)<br/>target=enable_obstacle(False)"] --> PLAN
  PLAN -->|"collision-free joint traj"| EXE["executor<br/>per-step target + q_bias(C6)"]
  EXE --> GRIP["gripper open/close 별도<br/>(정직 물리, C1)"]
```

| 요소 | 역할 | 핵심 |
|---|---|---|
| **해석적 IK (유지)** | grasp config 5축 해 제공 | `SO101Kinematics.ik/ik_reach`, roll ±90°·side-offset 후보. C3 |
| **clip-free 선정 (신규)** | self-clip 회피 | 후보를 **open-gripper sphere 기하**로 충돌검사 → clip 없는 config 우선. 해석적 SM 의 `_evaluate_all_grasps`/`--mj_outer` 역할을 cuRobo 충돌엔진이 대체 |
| **cuRobo plan_cspace (신규)** | 충돌-free joint→joint 궤적 (배치) | `BatchMotionPlanner.plan_cspace(goal_states, current_state)`. `multi_env=True` → per-env world |
| **per-env world (신규)** | 이웃 큐브·그릇·**카메라 홀더** 회피 | `Cuboid` + `update_world`, target 큐브는 grasp phase 에 `enable_obstacle(name, False, env_idx)` |
| **executor (유지·수정)** | traj → per-step joint action | per-env buffer 전진 + **q_bias(C6)**. transit/approach/lift/transport 를 cuRobo traj 로 교체 |
| **gripper (유지)** | open/close (cuRobo 밖) | 6번째 관절 별도 제어, 정직 물리 grasp (C1) |

**왜 이 구조**: dominant 천장(target self-clip)은 grasp **pose 선정**이 그리퍼 실제 기하를 봐야 풀린다.
cuRobo 의 full-gripper sphere 충돌모델이 이를 원리적으로 제공(해석적 heuristic offset 의 한계 돌파).
이웃/그릇/홀더 knocking 은 per-env world 회피로 해결. 5-DOF 는 joint-goal 로 pose-goal 문제 회피.

> **확정된 cuRobo API** (`ref_repos/curobo`, 표준 pip `nvidia-curobo` 구 `MotionGen` API 와 **다른
> 신버전** — copyright 2023–2026, `_src/` 레이아웃). 구현은 `curobo/examples/getting_started/
> motion_planning.py`(`plan_grasp_and_execute`) + 아래 실제 시그니처를 따른다. **`curobo.org` 구버전
> docs 금지.**

| 기능 | 실제 API (P0 설치 실측 검증 ✅) | 위치 |
|---|---|---|
| import (shim) | `from curobo.motion_planner import MotionPlanner, MotionPlannerCfg`<br>`from curobo.batch_motion_planner import BatchMotionPlanner`<br>`from curobo.scene import Scene, Cuboid, Sphere, Mesh` (`Scene`=`SceneCfg` alias)<br>`from curobo.kinematics import Kinematics, KinematicsCfg` · `from curobo.types import JointState` | `curobo/{motion_planner,batch_motion_planner,scene,kinematics}.py` shim → `_src/` |
| ❌ 틀린 경로 | `from curobo import MotionPlanner` — `__init__.py` 는 `__version__` 만 export | — |
| 로봇 config | `KinematicsCfg.from_robot_yaml_file("franka.yml")` / `MotionPlannerCfg.create(robot=, max_batch_size=N, multi_env=True, device_cfg=)` | `_src/motion/motion_planner_cfg.py:37` |
| FK | `Kinematics(cfg).compute_kinematics(JointState.from_position(q, joint_names=))` → `KinematicsState.tool_poses`·`get_link_spheres()` | `_src/.../kinematics` |
| per-env world | `multi_env=True` → 각 batch 문제가 독자 collision world | cfg.py:103,111 |
| joint-goal 배치 | `BatchMotionPlanner.plan_cspace(goal_states: JointState, current_state, max_attempts=)` | `_src/motion/motion_planner_batch.py:223` |
| world 갱신 | `update_world(scene_cfg: SceneCfg)` | batch.py:592 |
| per-env 충돌 토글 | `attachment_manager.enable_obstacle(name, enable=False, env_idx=)` | `_src/collision/attachment_manager.py:219` |
| sphere fit | `from curobo.sphere_fit import fit_spheres_to_mesh, estimate_sphere_count, SphereFitType` | `curobo/sphere_fit.py` |

**설치 (P0 ✅)**: `uv pip install -e "ref_repos/curobo[cu12]"` (torch 핀돼 있어 `cu12-torch` 아님).
**nvcc 불요** — `USE_PYBIND=0`(기본) → pybind CUDA 확장 안 빌드, 런타임 cuda-core(NVRTC)+warp 1.14 JIT.
**핀 보존 검증**(numpy 1.26·torch 2.7+cu128·pyarrow 18). ⚠ **`uv run` 은 sync 로 curobo 삭제**(lock 미등재)
→ cuRobo SM 은 **`uv run --no-sync --group isaac`** 규약(lock 미변경=ABI 핀 안전). `gen_so101_xrdf.py` 는
**구 API**(`CudaRobotModel`/`IKSolver`)라 신 API 재작성 필요(P1). PoseCostMetric/partial-pose 없음 → 5-DOF
는 joint-goal(`plan_cspace`)만.

---

## 5. 실행 환경

| 항목 | 값 |
|---|---|
| 서버 | Ubuntu 24.04.3, Intel Core Ultra 5 245K, 128GB RAM |
| GPU | RTX PRO 5000 Blackwell **48GB** (RT 코어 — Isaac Sim 5.1 요구) |
| 스택 | Isaac Sim 5.1 / IsaacLab 2.3 / **cuRobo(ref_repos)** / `ManagerBasedRLEnv` |
| Python | `uv run --group isaac` (cuRobo in-process 설치 — P0 ABI 게이트) |
| ABI 핀 | numpy==1.26 · torch==2.7.0+cu128 · pyarrow<19 등 (**업그레이드 금지**, AGENTS.md) |
| 격리 venv | `.venv-ovrtx`(ovrtx 0.3.0) · `.venv-ovphysx`(ovphysx 0.4.13) — Track D 옵션 |
| GPU 공유 | 1장 공유(학습·MCP 경합) → 장시간 run 은 스케줄·background |
| 부팅 | `OMNI_KIT_ACCEPT_EULA=YES`, `--headless` (P6 만 GUI) |

> `$ROOT = /home/konan147/Workspaces/SO101-Sim2Real`

---

## 6. 계획 & 칸반 보드 (Phase별 작업)

### P0 — cuRobo 설치·검증 게이트 🟢 DONE (Track A)
- [x] cuRobo 설치 `uv pip install -e "ref_repos/curobo[cu12]"` (nvidia-curobo 0.8.0.post1.dev35).
      **nvcc 불요**(USE_PYBIND=0 → cuda-core/warp 런타임). 핀 보존(numpy 1.26·torch 2.7·pyarrow 18).
- [x] 게이트: 공개 API import OK(`from curobo.motion_planner import …`) + franka FK CUDA 실행 검증.
- ⚠ **`uv run --no-sync --group isaac`** 규약(lock 미등재라 sync 가 curobo 삭제). Isaac headless smoke 는
      첫 sim 실행(P2) 때 확인.

### P1 — so101 cuRobo robot config 🟢 DONE (Track A)
**최종 방식 = 공식 `build_robot_model.py`(RobotBuilder API), 커스텀 sphere 수학 0.** 스크립트
`scripts/sim/build_so101_xrdf.py`:
- [x] **P1-a**: `RobotBuilder(urdf, asset, tool_frames=[gripper_frame_link]).fit_collision_spheres
      (sphere_density=2.0, clip_links={base:z,0})` — 전 link MorphIt + self-collision ignore 자동.
      카메라 홀더(`wrist_cam_mount`·`front_cam_mount`)는 URDF fixed link 라 **자동 포함**(로봇 한몸).
- [x] **P1-b thin/clip 링크 보정**: `refit_link_spheres(sphere_density=4.0)` for front_cam(0.6→93.9%)·
      base(clip 책상위 커버). density↑ 면 MorphIt voxel 격자 finer → 얇은 plate 도 정상.
- [x] **P1-c 퇴화 구 제거**: clip 하드클램프·과할당 디커플 작은구(r<0.005) 3개 필터(사용자 피드백).
- [x] **출력 2종**: `assets/robots/so101.xrdf`(Isaac) + `so101_curobo.yml`(cuRobo native·mesh_link_names
      포함 → MPC/Viser/planner 용). 54 spheres/9링크.
- [x] **검증**: `validate_so101_curobo.py`(로드·54 적재·FK·IK 81%) + **Viser 시각 검토**
      (`build_so101_xrdf.py --visualize`) + **reactive MPC 동적 검증**(`reactive_so101_viser.py` —
      EE 드래그 추종+장애물 회피, 공식 reactive_control 적용).
- ⚠ cspace 6축(gripper 포함, 공식 그대로) → P2 에서 planner `lock_joints` 로 5-DOF 계획.
- ❌ 폐기(교훈): 커스텀 SURFACE/centerline slab fit — "sparse/덕지덕지"(사용자), 공식 MorphIt density↑가 정답.

### P2 — 핵심 통합 ⚪ 🔥 (Track A)
- [ ] **신규** `scripts/planning/so101_curobo_planner.py`: BatchMotionPlanner wrapper(config·warmup·
      `plan_to_joint_goal`·per-env world·target 토글·clip-free 선정)
- [ ] **수정** SM(또는 신규 `pick_cube_curobo_sm.py`): transit/approach/lift/transport 를 cuRobo traj
      실행으로 교체. grasp config·q_bias·snapshot·gripper·`_placed`·`_pick_next`·안전망 **유지**
- [ ] 실행층: per-env traj buffer 전진 + q_bias(C6)
- [ ] **검증**: 4-env GUI(R/N) — 큐브 안 건드림·그릇 안 밀침 영상 확인

### P3 — 배치 스케일·청크 ⚪ 🔥 (Track A) — **계획 확정(2026-06-14)**
**구조 = lock-step 벡터화 SM**(D12): 전 env 동일 phase tape(고정 순서 Cube1→4), 매 IK/plan/step 배치,
per-env active/placed/failed 마스크. cuRobo `BatchMotionPlanner`(multi_env=True)가 "N문제 N world 1 GPU
call" → lock-step 이 설계 의도. straggler 세금 미미(고정 horizon traj 일괄 반환, retry 단위만).
- [x] **서버 additive 배치 엔드포인트**(`curobo_planner_server.py`) 🟢 — `BatchMotionPlanner`(multi_env,
      max_batch_size) + 배치 `IKSolver`(collision-free) + `init_batch`/`world_batch`/`ik_batch`/`plan_batch`.
      단일-env 엔드포인트(ik/plan/set_world) **무손상**. **N=4 GPU smoke 통과**: IK pos_err 0.05mm(D9 refine),
      plan 4/4, traj trim H=61. 함정 2건: ① per-env world buffer N개는 **scene_model 이 길이 N 리스트**일 때만
      할당(cache-only/단일 scene=1env) → 임시 YAML(N개 동일 스키마 scene 리스트) 경로 전달 ② batch
      `interpolated_trajectory`=buffer 전체(Hbuf~5000, last_tstep 이후 미정의, `get_interpolated_plan`은
      batch raise) → per-env `interpolated_last_tstep` trim + goal-pad 로 Hmax 균일화
- [x] **서버 내부 청크** 🟢 — 단일배치 256 = **OOM**(peak 42.5GB, planner build ~40GB+Isaac 공유불가).
      `init_batch(chunk)` 로 planner max_batch_size=chunk 빌드, `ik_batch`/`plan_batch` 가 n_envs 를 chunk
      단위 분할(부분청크 패딩, plan 은 per-chunk world 로드 → 전역 Hmax goal-pad). 물리배치(Isaac n_envs)↔
      planner 배치(chunk) 분리. **N=256 chunk=64 성공**: peak VRAM **20GB**, **3.04 epi/s**(wall 84s), cube
      56.8%·all-4 17.2%. **chunk⊥성공률 확증**(N=16 chunk8 71.9% ≈ 비청크 65.6%) — 청크는 VRAM/throughput 전용
- [x] **신규 클라이언트** `scripts/sim/pick_cube_curobo_batch.py` 🟢 — lock-step 벡터화. 스칼라 기하 헬퍼
      (closing_axis·to_base·yaw_base·grasp 선정)는 데모서 복제 후 **per-env 루프로 요청 빌드**, GPU(ik_batch·
      plan_batch)·env.step 만 배치. per-env active 마스크(grasp 도달/IK/plan 실패 env hold). q_bias(C6)·side-
      approach·DR cube_yaw·per-env world 회피 동일. **GPU smoke**: N=4 고정 16/16(100%, 15.2s sim)·**N=16 DR
      42/64(65.6%), all-4 5/16, wall 27.4s**. 단일-env 데모 무손상. 함정: numpy float32 JSON 직렬화 불가 →
      to_base/yaw/roll float() 코어션
- [x] **grasp-retry** 🟢 — `--max_retry`(기본 1). 큐브당 미placed env 만 bounded 재시도(성공/도달불가 env hold,
      lock-step). 큐브별 attempt_mask 축소. **N=256 chunk=64 max_retry=1**: cube **56.8%→64.6%**(+7.8pp),
      all-4 17.2%→18.8%, sim 20→40s·wall 84→173s(~2×). VRAM 무관(19.6GB). **all-4 천장은 retry 아님**(클러스터
      hard layout) → P4 min_cube_sep
- [ ] 추가 lever: `plan_attempts`(현 4 vs 단일-env 8)·`ik_seeds`(현 48 vs 64)↑, **P4 min_cube_sep**(all-4 천장)
- [ ] **배치 크기 = 256 시작→실측 climb**(D13): max_batch_size 256 으로 VRAM/wall-clock 실측, OOM·여유 보고
      512→2048 단계 상향. 물리배치(Isaac N) ↔ planner 배치(≤max_batch_size, 서버 청크) 분리. IPC 2048≈수 MB
- [ ] **검증 ≥256-env**(C5) taxonomy(success/clean/steps/reason_histogram)

### P4 — DR 영역·자세 재설계 + grasp 닫힘축 🟢 (Track B) — **2048: cube 87.9%·all-4 60.0%**(256: 95.3%·82.8%)
> **방향(사용자, 2026-06-14)**: livestream 으로 실패 관전 → unreachable 있음 확인 → **큐브 스폰을 매트 위
> 도달가능 사각형으로 한정 + 볼륨 비겹침 + 6D face 랜덤화**로 재설계.
- [x] **매트 사각형 영역** 🟢 — 데스크 매트(860×400mm) 좌하단=env-local(-0.34,0.045) 기준 사용자 지정
      매트-local cm 사각형(X[16,56]·Y[11,25]) → **env-local x[-0.18,0.22]·y[0.155,0.295]**. `volume_inset`
      (=40mm face 대각 절반 0.0283)으로 **큐브 볼륨이 사각형 안**(중심 아니라 부피 기준). 옛 y[0.115,0.23]
      대비 **+y 전진**(base 근접 unreachable 완화)
- [x] **볼륨 비겹침** 🟢 — `min_cube_sep` **0.10→0.060**(40mm footprint 대각 쌍 ≈0.057+여유, **non-overlap
      최소**·과분리 아님), `min_bowl_sep=0.14`(그릇 반경0.06+큐브0.029+arc0.05). 검증 N=64: 겹침 0쌍, OOB 0
- [x] **6D face 랜덤화** 🟢 — `full_orient`: **이산 stable-face(6면) + random yaw**(uniform SO(3)+낙하는 tumble
      drift 9% OOB → 폐기). drift 0·z 띄움 불요. 큐브는 면 동일해도 felt 텍스처 seam 이 달라 **VLA 시각 다양성**.
      검증: 6면 균등(flat 33%)·볼륨 in-rect·비겹침
- [x] **256 재측정** 🟢 — 새 DR 로 **cube 64.6%→92.8%(+28pp)·all-4 18.8%→74.2%(+55pp)**. 분포 {2:8,3:58,4:190}
      — **0/1개 실패 env 전무**(옛 0개 37·1개 31). 전진 Y 도달가능 영역이 unreachable 완전 제거. sep 0.060 클러스터
      우려 무효(reachability 이득이 압도). 남은 부족 = 1-2개 놓친 66 env(tight pack/marginal grasp)
- [ ] 95%+ push lever: `plan_attempts` 4→8·`ik_seeds` 48→64↑, 잔여 66 env 진단(클러스터면 sep 소폭↑)

### P5 — 렌더 env + 공유 obs/action 계약 ⚪ (Track B)
- [ ] **정확도 검증**: 2048-env state-only(`--headless --taxonomy`, 카메라 off)
- [ ] **데이터 기록**: state-replay 재렌더(권장) — 2048 state 기록 → 소수 env replay+RGB 렌더
- [ ] 공유 계약 모듈: obs 빌더(3-cam RGB+joint)·action 매핑을 **데이터 기록·VLA 추론 공유** import
- [ ] `rollout_to_lerobot.py`+`lerobot_units.py` 확장(SM/replay·다중 env). depth 보조 옵션(생성 전용)

### P6 — VLA 시뮬 추론 (single-env · ROS2 + async) ⚪ (Track C)
- [ ] **수정** `run_cube_desk_ros_bridge.py`: GUI 모드 + 3-cam RGB ROS2 publish
- [ ] **신규** `scripts/sim/sim_vla_ros_client.py`: ROS2 카메라/joint 구독 → LeRobot obs → policy-server
      async(RTC) 추론 → `/isaac_joint_commands` publish. 실기기 `robot_client.py` 패턴(parity)
- [ ] **검증**: GUI 큐브 drag → VLA 추종·집기 시도 정성 확인

### PD — throughput probe (옵션) ⚪ (Track D)
- [ ] ovrtx vs TiledCamera 렌더 throughput (`tiled_camera_throughput_bench.py` vs `ovrtx_*_probe.py`)
- [ ] ovphysx 2048 물리 throughput (`ovphysx_probe.py`) — IsaacLab 안정화 후 후순위

### 병렬 실행 전략
```
Track A (critical):  P0 → P1 → P2 → P3        (cuRobo 의존)
Track B (A와 병렬):   P4 ‖ P5                  (env·카메라만, cuRobo 불요)
Track C (A와 병렬):   P6                       (ROS2·VLA, cuRobo 전혀 불요)
Track D (독립):       ovrtx ‖ ovphysx 실측
```
- **A·B·C·D 동시 착수 가능**. 수렴점: 데이터 생성 = A(SM)+B(recorder), VLA 학습 = 데이터 후, P6 정성
  데모는 더미/학습 VLA 로 harness 먼저.

---

## 7. 타임라인 & 구간별 결과

| 날짜 | 구간 | 결과 |
|---|---|---|
| 2026-06-14 | 계획 확정·문서화 | 승인된 플랜 작성, 본 문서 생성. cuRobo API 실검증(신버전 `_src/` 레이아웃). xrdf stale(홀더 미포함) 확인 |
| 2026-06-14 | **P0 설치 게이트 🟢** | `uv pip install -e ref_repos/curobo[cu12]` 성공(nvidia-curobo 0.8.0.post1.dev35). nvcc 불요(USE_PYBIND=0, cuda-core/warp 런타임). 핀 보존(numpy 1.26·torch 2.7·pyarrow 18). 공개 API import + franka FK CUDA 실행 검증. import 경로 정정(`from curobo.motion_planner import …`, `from curobo import` ❌). `--no-sync` 규약 확정 |
| 2026-06-14 | **P1 xrdf 🟢 (공식 RobotBuilder)** | 최종 = 공식 `build_robot_model.py` 워크플로(`scripts/sim/build_so101_xrdf.py`, RobotBuilder API, 커스텀 sphere 수학 0). so101.xrdf **54 spheres/9링크**(카메라 홀더 포함, geometry=`collision_model`, self-collision ignore 자동). `validate_so101_curobo.py`: 로드 OK·54 적재·FK OK·IK 81%·Viser(:8085) 검토. 단위=미터. gen_so101_xrdf.py(구API/PATH E) 미변경 |
| 2026-06-14 | **P1 튜닝 루프(Viser)** | ① 기본 MorphIt(density 1.0)은 thin moving_jaw(cover 0.4%)·front_cam plate(0.6%)·clip base 붕괴 ② 커스텀 centerline slab 시도→"너무 sparse"(폐기, 사용자) ③ **공식 해법 확정: `--sphere-density 2.0`→moving_jaw 정상(91.7%), 두 thin/clip 링크는 `refit_link_spheres(sphere_density=4.0)`→front_cam 93.9%·base 책상위 커버** ④ clip 하드클램프 디커플 작은구(r<0.005) 3개 필터 제거. 사용자 OK |
| 2026-06-14 | **P1 동적 검증(MPC·MotionPlanner Viser)** | 공식 `reactive_control`(MPC)·`motion_planning`(plan_pose/grasp) 예제를 SO-101 `so101_curobo.yml` 로 Viser 구동, 사용자 정성 확인("얼추 돌아감"). 함정: ① 예제 `--robot`/`--scene` 절대경로 필수(상대=curobo content 기준) ② **collision_test.yml=franka 크기 → SO-101 start 충돌→Move/Grasp 전부 무반응**, SO-101 크기 `assets/robots/so101_scene.yml`(작은 기둥) 신규로 해결 ③ 5-DOF 라 임의 orientation pose-goal 실패 정상(P2 joint-goal) |
| 2026-06-14 | **P1 마무리** | 검증 Viser 종료·문서/CONTEXT/memory 갱신. **다음 세션 = 본 문서 읽고 진행현황 파악 후 시작**(사용자). P1 후속 TODO: 공식 forward/inverse_kinematics.py 예제로 SO-101 FK·IK 단독 검증 |
| 2026-06-14 | **P1+ pose-goal 벽 확증** | 공식 `motion_planning.py --visualize` SO-101: 첫 Move 1회 후 전부 fail. 원인=`plan_pose` 가 gizmo full 6-DOF pose 를 hard goal 로 줌 → 5축이 임의 orientation 불가. headless probe(`_probe_so101_posegoal.py`) A(현재pose)✅·B(45°회전)❌·C(위치만)✅ 로 결정적 증명. C4·D2 가 예고한 벽, 버그 아님. (forward/inverse_kinematics.py 는 franka 하드코딩이라 동등 검증을 probe 로 대체) |
| 2026-06-14 | **P1+ MPC vs plan_pose 규명** | 사용자 질문 "reactive MPC 는 왜 반응?". MPC=soft-cost 연속 재최적화 + `update_goal_tool_poses(run_ik=False)` → orientation penalty 만, 5축 best-effort 자연수렴, 실패개념 없음. plan_pose=궤적opt **전에 exact 6-DOF IK 게이트** → 5축 못 맞추면 시작도 못 함. cuRobo IK 도 position-only 불가(`orientation_tolerance` 는 수렴게이트만 풀고 optimizer cost 못 끔 — reach 안 점도 실패, `_probe_so101_reach.py` posonly==6dof 동일) |
| 2026-06-14 | **P1+ joint-goal 데모** | `scripts/sim/motion_plan_so101_viser.py` 신규: gizmo position → **해석적 IK(SO101Kinematics)** → `MotionPlanner.plan_cspace` → 실행. cuRobo IK 대신 해석적 IK(`scripts/sim/so101_kinematics.py`, SM 에서 **verbatim 추출** standalone, Isaac 불요) 써서 안정. 워크스페이스 전반 plan_cspace 0.05초 성공, 도달불가 정직거절. gizmo yaw→grasp_yaw 매핑(wrist_roll 반응). pitch 범위 arg(home 등 flat pose reach). 함정: pose-goal 무반응=`is_moving` stuck 아니라 plan fail; pkill 자기명령줄 self-kill(144) |
| 2026-06-14 | **P1+ FK 일관성 발견 🔴** | 사용자 "뭔가 안 맞음" 직감 검토. `_probe_fk_consistency.py`: 해석적 fk_tcp ↔ cuRobo FK (2000 config) **mean 6.86mm·p95 14.9mm·max 15.4mm**(zero-pose 만 sub-mm 일치). 해석적 FK=평면근사라 관절각 커지면 발산 → 해석적 IK config 직접 plan_cspace 실행 시 EE 최대 15mm 빗나감(grasp miss). **P2 수정안 검증**(`_probe_p2_graspik.py`): cuRobo IK 는 feasible grasp pose 를 **0.00mm 정확** 해결 → D9 |
| 2026-06-14 | **P2 ABI 게이트 🔴→IPC** | cuRobo+Isaac **in-process 불가 확정**: isaac-first→cuRobo import 死(`wp.func(module=)` 없음), curobo-first→isaacsim core ext 死(`warp.types.array` 없음). cuRobo **warp 1.14** ↔ isaacsim **omni.warp.core 1.8.2** 상호배타. → P0 트랩이 예고한 **subprocess IPC** 채택(D10) |
| 2026-06-14 | **P2 IPC 사이드카 + 1-큐브** | `scripts/planning/curobo_planner_server.py`(cuRobo 프로세스, ZMQ REP: ik/plan/set_world) + `scripts/sim/pick_cube_curobo_demo.py`(Isaac env, ZMQ REQ). 1-큐브 end-to-end: ready→pre→descend→slide→close→lift→bowl→release. 첫 grasp whiff(top-down center, fixed jaw 윗면 찌름) → **side-approach**(SM `_closing_axis` 이식)로 placed=True |
| 2026-06-14 | **P2 4-큐브 + 장애물 회피** | 4-큐브 순차, **per-env world**(set_world: 나머지 큐브+그릇 cuboid, target 제외) → plan_cspace 회피. **clearance-aware roll 선정**(±90/0/π). 함정·수정 6건: ① IK 충돌-aware 과다기각 → **IK 충돌-free·plan_cspace 만 충돌-aware** ② grasp 미세동작 plan_cspace 과다기각 → **직접 joint 보간** ③ bowl 먼 곳 pitch -45→-10 ④ DR cube yaw(±30°) 무시 → **cube_yaw 사용**(모서리 잡기 해소) ⑤ lift→bowl 실패 → 직접 fallback ⑥ READY self-collision 경계 sag → backoff(-1.3,1.2)+cube1 exact-start |
| 2026-06-14 | **P2 완료 — crisp·timed·cameras** | crisp: plan 궤적 subsample(stride3)+seq_exec 14~16step → **headless DR 4/4·4/4·4/4, 18~20초**(목표 <20s 달성, control 30Hz). 모션: READY 시작→bowl떨군뒤 다음큐브 직행(per-cube READY 복귀 제거)→끝 READY 복귀(droop 방지). livestream(WebRTC mode1, PUBLIC_IP, **0.88Mbps**)+top/wrist/front 3-cam docking viewport. 사용자 "완벽" |
| 2026-06-14 | **카메라 focal 14mm + layout JSON** | 3-cam focal 23→**14mm**(`pick_cube_env_cfg._TOP/_WRIST/_FRONT_CAMERA_FOCAL`, teleop 튜너 기본값 정합). viewport layout 저장본 → `assets/layouts/pick_cube_3cam.json`, `pick_cube_curobo_demo.dock_camera_viewports()`가 수동 dock_in 대신 `ui.Workspace.restore_workspace(dump)` JSON 복원(window title 일치, 실패 시 dock fallback, `--layout` override) |
| 2026-06-14 | **P3 계획 확정 🔵** | multi-env 논의 → **lock-step 벡터화 SM**(D12, async 기각: env.step 글로벌+cuRobo batch 정합), **신규 `pick_cube_curobo_batch.py`**(단일-env 데모 보존), **256→실측 climb**(D13, 서버 청크로 물리/planner 배치 분리). cuRobo batch API 소스 검증: `plan_cspace(goal_states(N,dof),current_state(N,dof))` 단일 GPU pass, per-env world=동일 스키마 N슬롯+`update_obstacle_pose`/`enable_obstacle(env_idx)`, multi_env→PRM off(cspace OK). 착수=서버 additive 배치 엔드포인트 |
| 2026-06-14 | **P3 서버 배치 엔드포인트 🟢** | `curobo_planner_server.py` additive: `init_batch`(BatchMotionPlanner multi_env + 배치 IKSolver collision-free)·`world_batch`(per-env update_obstacle_pose+enable_obstacle)·`ik_batch`(per-env 해석적 seed→배치 cuRobo IK refine)·`plan_batch`(plan_cspace N + per-env trim/goal-pad). 단일-env 무손상. **N=4 GPU smoke 통과**(IK 0.05mm·plan 4/4·H=61). 함정: ① N-env world buffer = scene_model 길이 N 리스트 필수(임시 YAML) ② batch interpolated_trajectory 미trim(Hbuf~5000)→last_tstep trim. 다음=클라 `pick_cube_curobo_batch.py` |
| 2026-06-14 | **P3 배치 클라이언트 🟢** | `pick_cube_curobo_batch.py` lock-step 벡터화(스칼라 기하 per-env 루프 + GPU·step 배치). **N=4 고정 16/16(100%, 15.2s sim)·N=16 DR 42/64(65.6%, all-4 5/16, wall 27.4s)** 크래시 없음·per-env 이질성 정상. 함정: numpy float32 JSON 불가→float() 코어션 |
| 2026-06-14 | **P3 서버 청크 + 256 측정 🟢** | 단일배치 256 OOM(42.5GB)→**서버 내부 청크**(`init_batch(chunk)`, ik/plan chunk 분할+부분청크 패딩+per-chunk world 로드+전역 Hmax pad). **N=256 chunk=64**: peak VRAM **20GB**·**3.04 epi/s**(wall 84s)·cube 56.8%·all-4 17.2%·분포{0:37,1:31,2:57,3:87,4:44}. **chunk⊥성공률**(N16 chunk8 71.9%≈비청크). 사용자 VRAM 질문 정리: 주범=IK/trajopt seed 텐서(world_model 아님, depth는 늘림)·warp 통일 무의미(D10)·RL 16384는 가벼운 step이라 별개·16→256 cliff 없음(둘 다 ~65%, DR 난이도가 원인). 다음=grasp-retry(성공률 lever, 256서 검증) |
| 2026-06-14 | **P3 grasp-retry 🟢 → P3 완료** | `--max_retry`(기본1) 큐브당 미placed env bounded 재시도(lock-step, attempt_mask 축소). **N=256 chunk=64 retry=1: cube 56.8%→64.6%(+7.8pp)·all-4 17.2%→18.8%·sim 40s·wall 173s(~2×)·VRAM 19.6GB**. N=16 seed0 retry=2 79.7%·all-4 50%. **all-4 천장은 retry 무관**(클러스터 hard layout, 37/256 env=0) → P4 min_cube_sep 가 진짜 lever. **P3 DONE** |
| 2026-06-14 | **실패 관전 livestream + P4 DR 재설계 🟢** | `pick_cube_curobo_batch.py` 에 `--dump_fail`/`--load_fail`(worst-N 실패 layout 초기 pose env-origin 상대 덤프→재현, viewer/watch 무한루프) 추가. N=4 livestream 관전 → 사용자 "진짜 unreachable 있음" 확인(worst 4 중 3개 큐브간 40-43mm 클러스터, 1개 106mm reach-edge). → **DR 재설계**: ① 매트 사각형 영역(매트-local cm→env-local x[-0.18,0.22]·y[0.155,0.295], volume_inset 0.0283 으로 볼륨 in-rect) ② 볼륨 비겹침(min_cube_sep 0.10→0.060·min_bowl_sep 0.14) ③ full_orient=이산 stable-face+yaw(6면 균등·drift0, uniform SO(3)+낙하 9% OOB 폐기). N=64 검증: OOB 0·겹침 0·6면 균등. `domain_randomization._randomize_cubes_scattered_fn`(full_orient·volume_inset 추가)·`pick_cube_env_cfg`(_MAT_BL_ENV·rect 상수) |
| 2026-06-14 | **새 DR 256 재측정 🟢** | N=256 chunk=64 retry=1 seed0: **cube 64.6%→92.8%(+28pp)·all-4 18.8%→74.2%(+55pp)**·sim 32s·wall 104s. 분포 {2:8,3:58,4:190} — **0/1개 실패 env 전무**(전진 Y 도달가능 영역이 unreachable 완전 제거). sep 0.060 클러스터 우려 무효. 성공기준(~95-99%) 근접. 잔여=1-2개 놓친 66 env. 다음 lever=plan_attempts/ik_seeds↑ |
| 2026-06-14 | **천장 제거 + 회귀 디버깅 ⚠** | 사용자 livestream 피드백 3건. ① **천장 제거**(author scene.usd, 거슬림) 🟢. ② **팝콘(그릇서 큐브 충돌 폭발)**: restitution 은 원인 아님(bowl combine=min → 큐브0 과 min=0 이미 무반발). **maxDepenetrationVelocity 1.0→0.5 시도 = grasp grip 약화로 92.8→77% 회귀**(침투 분리 cap 이 jaw grip 방해) → **원복**. 팝콘=가벼운 큐브(20-35g) 그릇 다중 적재 솔버 불안정, maxDepen 으론 못 고침(grasp 비용 큼). ③ **grasp-face fix**(동적순서 isolated+free-side score+face-yaw): 토글 ON 시 89.5%/65.6% 회귀(isolated 가 reach-edge 큐브 먼저 집음) → **기본 OFF**(`--order_mode/-free_side/--face_yaw` 플래그 보존). **bowl_z 0.08 도 회귀(rim 충돌)→0.12 원복**. 물리 원복 후 **seed0 256 = 93.3%/all-4 76.2% 재확인**(baseline 안정). 팝콘·grasp-face 는 비회귀 방법 재설계 필요 |
| 2026-06-14 | **닫힘축 clearance 버그 수정 🟢🎯** | 사용자 정밀 지적: "X-나란 큐브는 Y로 닫아야 하는데 X로 닫아 실패". 원인=select_grasp 의 finger clearance 점을 닫힘축 **수직**(fx,fy=-dy,dx)으로 잡아 → 이웃 향한 closing 이 오히려 clearance 높게 나와 우대(정반대 버그). **수정: finger 점을 닫힘축 방향(dx,dy)으로** → 이웃 향한 closing penalize → 수직 closing 선택. score=1.5×clear-0.15×flip. **seed0 256 = 95.3%/all-4 82.8%**(93.3/76.2 대비 +2/+6.6pp, retry↓로 24s 더 빠름). **성공기준 ~95% 달성**. 기본 적용(order fixed·naive yaw 유지). 남음=팝콘(maxDepen 불가→대안) |
| 2026-06-14 | **2048 정확도 평가 🟢** | N=2048 chunk=64 retry=1 seed0(닫힘축 수정·물리원복): **cube 87.9%·all-4 60.0%**·sim 34s·wall 469s(7.8min). 분포 {1:5,2:165,3:649,4:1229} — **1개 놓침 32%(649)가 지배 실패**. 256(95.3%) 대비 낮음=소표본 과대평가(C5)+**팝콘(그릇서 큐브 튕겨 나가 placement 실패)**이 정확도도 깎음. **팝콘 = 미해결 핵심 문제**(아래) |
| 2026-06-14 | **팝콘 측정(속도 스파이크) + 팝콘제외 성공률 🟢** | batch client 에 **큐브 max 선속도 추적**(`--pop_speed` 2.5, act 매 step GPU 갱신) + 분류(success/grasp-fail/popcorn-fail=미placed&spike). vmax **명확 bimodal**: p50≈1.2(정상 그릇낙하), 꼬리 p99 28~34·max **126(256)~265(2048) m/s**(폭발). **팝콘제외: 256 cube 97.0%·all-4 88.9% / 2048 cube 88.5%·all-4 56.0%**. **N-의존 시사**(단정 못함, N당 1run·변동큼): popcorn-fail 256 **2.1%**→2048 **4.8%**(~2×), raw all-4 2048 run간 60%↔45.8% 큰 변동. buffer 는 16384 대비 충분(overflow 아닐 듯) → 팝콘 비결정성+적재 chaos. 팝콘제외 256↔2048 격차(97↔88.5)는 hard-layout(C5)+**팝콘 collateral**(폭발 큐브가 이웃 침→이웃은 grasp-fail 로 계상, excl 이 collateral 미제거) |
| 2026-06-14 | **세션 마무리: focal 재적용·roll -90·grasp-effort·contactOffset 🟢** | ① focal 14mm 가 초반 변경 유실로 23.0 으로 커밋됐던 것 발견 → **재적용·grep 검증**(8d15fff). ② **roll -90° 선호**(free 큐브 clearance 동률서 bias 0.4, 클러스터는 clearance 지배). ③ **grasp-effort 확인**: leisaac `dynamic_reset_gripper_effort_limit` 우리도 보유(`utils/gripper_effort.py`), **PickCubeEnv.step 이 매 step 배선**(flag True) → batch/demo/SM 모두 자동 적용. 그리퍼 effort=`clamp(mass/0.15,0.5,10)`=큐브 **0.5Nm**(gentle). ④ **그리퍼 contactOffset 0.002·restOffset 0**(큐브와 매칭, 이전 미설정=큰 skin) → finger-큐브 빈 공간(gentle grip 이 skin 안 눌러서 생김) 축소. **256 검증: cube 95.3→96.4%·all-4 82.8→86.7%·popcorn-fail 2.1→1.1%**(회귀 없음·개선). 커밋 8d15fff·0824be3 |
| 2026-06-16 | **🎯 5-DOF native pose-goal 부활 — D2/C4/D9 정정** | 옛 "5축 cuRobo pose-goal 비가능" 은 **오진단**. cuRobo V2 소스 커널까지 검증: position-only 진짜 API = **`ToolPoseCriteria.track_position()`**(회전 weight=0 → `wp_tool_pose.compute_rotation_error_axis_angle` 에서 `angular_distance=0`·gradient=0 → orientation 게이트 무조건 통과 + optimizer 자세 무시). 옛 probe 는 `orientation_tolerance`(수렴 게이트)만 풀어봐 cost 못 끔 = 오진단 근원. 주입 = **`MotionPlanner.update_tool_pose_criteria({tool: track_position()})`**(warmup 후 안전, plan_grasp 도 동일 패턴). **정량 probe `_probe_posonly.py`**(N=200, orientation_tolerance=0.05 고정): full-pose+임의자세 **0.0%** vs track_position+임의자세 **97.5%**(Δ+97.5pp), plan_pose 9/10 궤적. **`motion_plan_so101_viser.py` 를 native `plan_pose`(position-only)로 재작성**(해석적 IK·FK 15mm 발산 우회 제거). 다음 = SM 을 plan_pose(transit)+goal-set(grasp)로 재구성(Stage 2) |
| 2026-06-16 | **viser 인터랙티브 Move/Grasp/Home 완성 + 5-DOF grasp 자세 지식** | `motion_plan_so101_viser.py` 를 공식 `--visualize` 구조(궤적 plot + Move + Grasp + status)로 재작성. 함정·지식 다수(§14 신설). 요지: ① **Move=position-only plan_pose**(자세 무시, feasible 90%·graph_attempt=1 유지·**graph 즉시(=0)는 5축서 악화** 90→62.5%) ② **Grasp=holder-up swoop 3단계**(grasp config 먼저→hover 도 같은 자세→plan_cspace transit→직접보간 descend/lift) — 단일 plan_pose "찌르기"·중간 wrist flip 방지 ③ **카메라 홀더 up = wrist_roll −90°**(측정: −90 Δz+0.044 up·+90 down, 중립 홀더=+y) ④ grasp 후보 = top-down+tilt(0-90°)×roll(6)×**uniform azimuth(12)**(centered az 는 악화), collision-sphere FK(`get_sphere_index_from_link_name`)로 **holder-up(Δz>0) 강제·없으면 거부**(충돌모델에 ground plane 없어 IK 가 홀더-바닥 허용→스코어링으로 차단) ⑤ **current_state stuck**: 도달 config 체이닝이 self-collision 경계 안착→"Start in collision" 영구 stuck→**실패 시 home 복귀 재시도**+Home 버튼 ⑥ 함정: IKSolver 두 batch 크기→CUDA graph 깨짐(`use_cuda_graph=False`)·viser websockets≥13.1(isaacsim==12.0 충돌)·pkill self-kill. reach-edge(dist>~0.42·저z)는 5축 물리한계로 거부 |

> 진행 시 여기에 Phase별 실측(taxonomy success·clean·steps·reason_histogram, throughput)을 누적 기록.

---

## 13. P5+ VLA 전체 루프 — 진행·재개 가이드 (🔵 자율 진행 중, 2026-06-14)

> **목표(사용자)**: cuRobo single-env pick-place 로 expert 데이터 생성 → HF → SmolVLA fine-tune →
> cube_desk 시뮬 closed-loop 추론. **게이트**: 10-ep 생성·업로드 확인 후 80-ep + 학습 + 추론.
> **승인 플랜**: `~/.claude/plans/curious-hopping-lampson.md`.
> **결정**: 에피소드 = full-round(4큐브, all-4 성공만 기록) · 학습 = smoke→full 20k 단계 · 추론 = 검증된
> ROS2 경로 재사용. **repo**: 데이터 `taehunkim/so101_sim_pick_cube`(신규, 실기기 `so101_pick_cube` 와 분리) ·
> 모델 `taehunkim/so101_smolvla_pick_cube`.

### 불변 계약 (North Star)
LeRobot v3.0 · robot_type `so_follower` · action/state 6-dim(arm 5축 deg + gripper [0,100], scale 31.75) ·
`observation.images.{top,wrist,front}` 480×640×3 h264 yuv420p · fps 30 · task `"pick up the cube and place
it in the bowl"`. 참고 실기기 데이터셋 = `datasets/pick_cube`.

### 구현된 코드 (P5, 완료·검증)
| 파일 | 내용 |
|---|---|
| `scripts/sim/lerobot_recorder.py` (신규) | `LeRobotV3DatasetWriter` — rollout_to_lerobot writer 추출(스키마 동일). `add_frame`·`commit_episode(success, task)`·`finalize`. pyarrow/imageio **지연 import**(ABI). 더미 데이터 end-to-end 검증 완료(v3.0·so_follower·3 mp4) |
| `scripts/sim/rollout_to_lerobot.py` (수정) | inline writer 제거 → 공유 모듈 import (RL 경로 의미 보존) |
| `scripts/sim/pick_cube_curobo_demo.py` (수정) | `--record_dir/--record_episodes/--record_overwrite/--record_max_attempts/--record_warmup`. record 모드: cameras 강제 ON, act() 에서 (state 측정+3캠+action=env.step 입력) 캡처, run_round 단위 에피소드, all-4 성공만 commit, N 성공까지 반복 |
| `scripts/sim/upload_to_huggingface.py` (신규) | `huggingface_hub.upload_folder` (Isaac 무의존, 호스트 uv). HF_TOKEN/HF_USER `.env` 파싱. dataset repo 자동 생성 |

action 기록 = **실행된 env.step 입력**(arm q_cmd+q_bias, gripper grip-offset). q_bias(C6 중력보상)·
GRIPPER_ACTION_OFFSET 은 sim 유물이나 rollout_to_lerobot 와 동일하게 실행 action 기록(BC 인과 일치).

### 실행 명령 (재현·재개)
```bash
# 1) cuRobo planner 사이드카 (ready 신호 = "[planner] ZMQ REP bind")
nohup env OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
  scripts/planning/curobo_planner_server.py --port 5599 > outputs/p5_logs/planner.log 2>&1 &
# 2) recorder (10-ep 게이트 → 80-ep 본생성). headless + cameras 강제.
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python scripts/sim/pick_cube_curobo_demo.py \
  --headless --planner_port 5599 --active_objects 4 \
  --record_dir outputs/so101_sim_pick_cube --record_episodes 10 --record_overwrite   # 80→--record_episodes 80
# 3) HF 업로드 (호스트 uv, --group isaac 불요)
uv run --no-sync python scripts/sim/upload_to_huggingface.py \
  --dataset_dir outputs/so101_sim_pick_cube --repo_id taehunkim/so101_sim_pick_cube
# 4) SmolVLA 학습 (Docker; ⚠ HF_DATASET_REPO_ID 를 sim repo 로 override — .env 는 실기기 repo)
#    smoke(100): -e TRAIN_STEPS=100 -e BATCH_SIZE=16 -e WANDB_ENABLE=false ... --policy.push_to_hub=false
docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e HF_DATASET_REPO_ID=taehunkim/so101_sim_pick_cube policy-server train   # full 20k, push_to_hub
# 5) 추론(ROS2): run_cube_desk_ros_bridge(호스트 Isaac GUI) + so101_vla_policy 노드(Docker) + policy-server gRPC
```
- 산출물: 데이터 `outputs/so101_sim_pick_cube/`, 학습 `outputs/train/so101_smolvla_pick_cube/`, 로그 `outputs/p5_logs/`.
- ⚠ 데모 재시작 = python child 직접 kill(TaskStop 만으론 Isaac 좀비). planner = ZMQ `{"cmd":"shutdown"}`.

### 진행 상태 (자율 갱신 — compaction 후 여기부터 이어서)
- [x] **P5 코드 + 검증** (writer 더미 end-to-end, py_compile 4파일, huggingface_hub 0.35.3 확인)
- [x] **🟢 10-ep 게이트 통과** — 10/10 all-4(폐기0, 5520 frame), 스키마 v3.0·so_follower·3cam OK,
      **HF 업로드 확인**(`taehunkim/so101_sim_pick_cube`, 9파일 184MB). 데이터 실경로 `/DISK1/so101-sim2real/lerobot_outputs/so101_sim_pick_cube`(outputs/ 심링크)
- [x] **⚙ 아키텍처 전환(사용자)**: single-env 느림(매 step 3캠 렌더) → **success-only + render-batch**. 80-ep single-env 중단,
      `pick_cube_curobo_batch.py` 에 카메라 N-env 배치 렌더 + per-env 기록 훅 추가(all-4 성공 env 만 commit). render-batch 검증 N=4 all-4 4/4 1라운드.
- [x] **🟢 시각 DR 확장(Workshop resets.py 참고)**: `domain_randomization.py` 에 `randomize_lights`(/World/Light·/World/KeyLight intensity+warmth, 글로벌)·`randomize_camera_focal`(top/wrist/front focalLength, IsValid 가드) 추가 → `pick_cube_env_cfg.PickCubeEventCfg` reset-mode 배선. robot color·mat 회전은 fragile/geometry 위험이라 보류. cuRobo oracle 무영향(visual only)
- [x] **🟢 240-ep 생성+재업로드 완료** — render-batch N=16, 18라운드 ~14/round, **240 ep·143072 frame**·DR 작동 확인(라운드별 밝기 132-151·warm→cool). HF 재업로드(`taehunkim/so101_sim_pick_cube` v3.0, 9파일 ~4.8GB). planner 종료(GPU 확보).
      ⚠ **함정: LeRobot train 은 dataset repo 의 codebase_version 태그(`v3.0`) 를 revision 으로 찾음** — `upload_folder` 는 태그 미생성 → `RevisionNotFoundError`. 해결: `api.create_tag(repo,"v3.0",repo_type="dataset")`(upload 스크립트에 자동화 반영). 실기기셋엔 LeRobotDataset 가 태그를 달아둠.
- [x] **🟢 SmolVLA smoke(100) 통과** — config.type==smolvla, 체크포인트 저장, ~4 step/s. **함정 2건 해결**: ① tasks.parquet 은 pandas 인덱스(task 문자열) 필요(pyarrow 직접 write=`Task cannot be None`) → writer `_write_tasks` pandas 화 ② dataset task linkage 확인.
- [ ] **🔵 full 20k 진행 중**(`outputs/p5_logs/full_train.log`, PID full_train.pid) — wandb `pubcyberry/lerobot/oh1gs82t`, batch32, push→`taehunkim/so101_smolvla_sim_pick_cube`, save_freq 4000, ETA ~1.5-2h.
      ⚠ **출력 dir 충돌**: 이전 `so101_smolvla_pick_cube` run 존재 → 별도 JOB `so101_smolvla_sim_pick_cube`(+ POLICY_REPO_ID 동명 repo)로 분리. 명령: `docker compose --env-file .env -f docker/docker-compose.yaml run --rm -e HF_DATASET_REPO_ID=taehunkim/so101_sim_pick_cube -e JOB_NAME=so101_smolvla_sim_pick_cube -e OUTPUT_DIR=outputs/train/so101_smolvla_sim_pick_cube -e POLICY_REPO_ID=taehunkim/so101_smolvla_sim_pick_cube policy-server train`
- [x] **🟢 sim VLA closed-loop 추론 검증 완료** (ROS2 3-process, PATH_E §7.4). bridge(headless, 3캠 52Hz+joint publish) + 기존 `policy_server`(재사용, 모델-agnostic) + `vla-ros`(my sim 모델). policy_server: **inference 0.12s, action [1,8,6]**. `/isaac_joint_commands` **17.7Hz·값 변화**(arm 실제 구동). 전체 sim→train→sim 루프 동작.
      ⚠ **함정: vla 노드가 profile(`POLICY_REPO_ID`)를 내부 env reload 로 -e override 위에 덮음** → 모델은 ROS param `pretrained_name_or_path`(최우선)로 고정. `config/vla_policy.yaml` 에 `taehunkim/so101_smolvla_sim_pick_cube` 설정.
      재현 — headless proxy: ① `docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server`(non-RTC 기본) ② `scripts/sim/run_cube_desk_ros_bridge.sh --headless --num_cubes 4` ③ `docker compose --env-file .env -f docker/docker-compose.yaml run --rm vla-ros`. ⚠ **그리퍼 offset 은 `env/smolvla.env` 의 `GRIPPER_CMD_OFFSET=0.2` 로 자동 적용**(sim 모델 전용; 실기기 모델이면 0). 빼면 그리퍼 0.20rad 덜 열려 grasp 실패 — 아래 ①.
      재현 — **livestream 시각 관찰**(검증됨): ② 를 `PUBLIC_IP=10.10.16.147 LIVESTREAM=1 scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 4 --livestream 1` 로(`--headless` 빼고) → WebRTC client 를 **`<PUBLIC_IP>:49100`** 접속. `--livestream 2` 만 쓰면 LAN IP 광고로 검은화면(memory `isaac-livestream-tailscale-publicip`). 모델은 `config/vla_policy.yaml::pretrained_name_or_path`(현 `actions_per_chunk:24`).
      **RTC 변형**: `docker compose ... run --rm policy-server policy-server-rtc`(RTCPolicyServer). ⚠ **2026-06-15 정량 검증 결과 RTC 는 비권장**(아래 RTC 오적용 항목) — compose 기본을 non-RTC(`policy-server`)로 변경함.
      미완: **정성 grasp 성공률**(headless proxy 만 확인 — 팔이 obs 따라 움직임 확정, 실제 집기 완성은 livestream/GUI 로 사용자 관찰 권장. 240 demo·20k step 이라 grasp 완성도는 데이터/스텝↑로 개선 여지).
- [x] **🟢 정량 성공률 eval (2026-06-15)** — `run_cube_desk_ros_bridge.py --eval N` 모드 신설(auto-reset + DR + cube-in-bowl 카운트, terminations.task_done 동일 판정: radius 0.06·z∈[DESK_TOP_Z+0.005, +0.12]). **결과 N=10: all-4 0.0%·per-cube 0.0%·평균 0.00/4·ever-in-bowl 0.0%**(`outputs/vla_eval.json`). 큐브 z 가 전 에피소드 spawn 값(0.724/0.729) 유지 = **한 번도 들리지 않음**. policy-server 추론·`/isaac_joint_commands` 구동은 정상(plumbing OK) → 팔은 움직이나 grasp 미완성 = **checkpoint 정책 능력 부족**(240 demo·20k step 과소학습), 물리/배선 문제 아님.
      ⚠ **물리 parity 동시 적용**(bridge ↔ cuRobo demo/학습 env): 그리퍼 stiffness 80→**17.8**(soft PD), gripper effort **0.5Nm** gentle(leisaac dynamic 동치), arm effort 10Nm, joint max vel arm 5·gripper 2.5(slew 근사), PhysX TGS·bounce 0.01·friction_corr 0.00625. backend 는 CPU 유지(A안 device-1 회피, 구조적 gap). cuMotion(PATH E) 도 이 물리 공유(사용자 "무조건 적용").
      ⚠ **GUI 뷰포트**: bridge 에 demo 와 동일 3-cam docking 레이아웃(`dock_camera_viewports`, `assets/layouts/pick_cube_3cam.json` 복원, `--view_eye/--view_lookat/--layout`) 추가.
      ⚠ 개선 레버: 데이터 240→대량·step 20k→↑·RL 보강. eval 재현: ① `docker compose ... up -d policy-server`(non-RTC) ② `docker compose ... run --rm vla-ros`(GRIPPER_CMD_OFFSET=0.2 는 smolvla.env 자동) ③ `scripts/sim/run_cube_desk_ros_bridge.sh --headless --num_cubes 4 --eval 10`.
- [x] **🟢 추론 경로 면밀 감사 + 단위/오프셋/RTC 진단 (2026-06-15)** — sim 배포 추론의 모든 변환점 감사. **정합 확인**: state/action 단위(rad↔deg·gripper×31.75)·joint 순서(bridge dof_names = SO101_JOINT_ORDER 확정)·정규화(STATE/ACTION MEAN_STD 학습 stats)·이미지(server `permute`+`/255`, top→camera1 rename train=infer 동일, 480→256→512 해상도 무관 <2deg)·카메라 focal/aperture(14mm/20.955). **불일치 2건**:
      ① **그리퍼 +0.20 offset** — recorder 가 action 을 `use_default_offset` pre-offset(grip_target−0.20)으로 기록(env action term 이 +0.20 재적용). bridge 배포는 raw 적용 → 그리퍼 0.20rad 덜 열림. **fix: vla 노드가 그리퍼 target 에 `+0.20` **재적용**(제거 아님!) — `gripper_cmd_offset`/`GRIPPER_CMD_OFFSET` env, `env/smolvla.env` 에 `=0.2` 박아 sim 모델 자동 적용. 실기기 모델(절대각)=0.** replay(action-vs-action)는 못 잡음(적용 단계 버그).
      ② **🔥 RTC 오적용** — `policy_server_rtc.py`(서버측 RTC)가 `prev_chunk_left_over` 를 **wall-clock 추정**하는데 공식 RTC(lerobot docs)는 ActionQueue **co-located** 로 실제 leftover 사용. 게다가 bridge **sim-time≠wall-clock(렌더 병목)** → 추정 붕괴. 같은 recorded obs open-loop: **non-RTC 3.9deg(=training 충실) vs RTC h8 15deg·h12 28deg**(execution_horizon↑일수록 악화). → **compose 기본 non-RTC 로 변경.** RTC 는 co-located ActionQueue 재설계 전까지 비권장.
      **진단 도구 신설**: `scripts/sim/replay_infer_overlay.py`(학습 입력방식 in-process predict_action_chunk vs recorded overlay, feed marker) · `scripts/sim/compare_train_vs_rtc.py`(training vs async±RTC gRPC overlay, MAE). 결과 PNG=`outputs/{vla_replay_overlay,compare_*}.png`.
      **eval ladder**(N=10, all-4 전부 0% — BC covariate shift 잔여): ever-in-bowl RTC+gripper미수정 ~0% → RTC+gripperfix 5% → **non-RTC+gripperfix 12.5%**. open-loop 충실하나 closed-loop 누적 drift 로 완주 불가 → 데이터 증설/RL 필요(보류).
- [x] **🟢 데이터 jerky 결함 진단 + slew-limited 기록 수정 (2026-06-15, 🔵 A/B 검증 중)** — 사용자가 HF
      `visualize_dataset` 의 Action Velocity Smoothness Proxy **"Jerky"**(arm 3축, |Δ|max≈0.80) 발견.
      **근본 원인**: recorder 가 env slew limiter **이전** 의 raw command(`q_cmd+q_bias`)를 action 으로 기록
      (`pick_cube_curobo_batch.py` act() → `env.step` → `common/mdp/actions.py:60-72` clamp). 즉 데이터에 든
      거대 점프 = **로봇이 실제로 한 적 없는 물리 불가능 teleport**. 정책은 즉시 clamp 되는 점프 + 관측 불가
      q_bias 적분기를 예측하라고 학습 → BC aliasing·closed-loop drift. **기존 240-ep 정량 확인**: arm 단일-step
      max Δ = sh_lift **124°**·elbow **137°**·wr_roll **239°** (slew cap **9.55°/step** 의 13~25×), >cap
      0.2~0.9%(드묾=phase 경계 teleport). episode 끝 near-idle tail 평균 17.8·max 29 frame.
      **수정(recorder-only, 세 jerk 원천 동시 차단)**: action 기록을 raw → **env action term 의 slew-limited
      `processed_actions − offset`**(달성가능 target)로. arm 점프 ≤cap 등속 ramp 화, stride3·q_bias 점프·
      gripper 28.6→5.3 모두 흡수. gripper offset 규약(pre-offset)·deploy 정합 보존(무변경). **추가: 녹화 tail
      제거** — per-env `rec_freeze`(터미널=all-placed/진행불가 시 이후 idle 미기록) + 끝 READY 복귀 settle 미기록
      → episode 가 마지막 release(cube-in-bowl) frame 에서 종료. 동일 수정 demo 미러.
      **smoke(8-ep) IsaacLab 실검증 통과**: arm max Δ **≤9.549°(=cap)·>cap 0.00%**, gripper 28.6→5.3, all-4
      16/16(오라클 무손상), 마지막 frame gripper open=release. **🔵 256-ep regen 진행 중**(`so101_sim_pick_cube_smooth`,
      `outputs/p5_logs/regen256.log`) → 업로드 `taehunkim/so101_sim_pick_cube_smooth` → SmolVLA 20k(`_smooth` JOB,
      단일 변수 A/B) → bridge `--eval 10` 로 baseline(12.5%) 대비 측정. 게이트: 개선 시 대량 재생성, 미미 시
      데이터 양·RL 이 주 lever 확정.
- [ ] **🔵 데이터 생성 속도·품질 조사 + 방법 결정 (2026-06-15)** — 사용자 "생성 너무 오래·품질 기대 이하".
      **속도 실측 정정**: render-batch N=16 도 **256-ep ≈ 2시간**(~1.9 ep/min, round당 ~7-8분). 병목 = **생성 중
      매 step 3캠 raytrace 렌더**(cuRobo plan 은 phase당 1회라 무관). render-batch 는 single-env(80-ep 2.1h) 대비
      빠르나 여전히 느림. **속도 레버(프레임워크 교체 불요, 우리 코드 내)**: ① 렌더 res 480×640→256(SmolVLA 다운샘플,
      "480→256 <2deg" 확인 → 픽셀 0.2×·~3-5× 빠름) ② N 16→32-48(48GB VRAM 여유). 둘이면 ~20-30분. ⚠ **현 regen 은
      baseline(480·N16) 정합 위해 그대로**, res/N 최적화는 **다음 대량 생성부터**.
      **방법 서베이(웹)**: MimicGen/DexMimicGen(few demo→대량 궤적 변환)·DemoGen(1 demo→수백, 3D 씬편집)·**SkillGen
      (SkillMimicGen, arXiv 2410.18907, CoRL24)**·RoboEngine(픽셀 증강)·GR00T-Dreams(Cosmos world-model)·VLA-RFT(RL).
      **결정 = SkillGen 채택 ❌**: (a) **Isaac Sim 6.0 필요**(우리 5.1·ABI 핀) (b) **state-only 생성**(VLA용 RGB 별도
      렌더 → 속도 안 풀림) (c) 핵심(cuRobo transit + skill stitch)을 **우리가 이미 구현**. 코드 위치=Isaac Lab 통합
      (`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --use_skillgen`, `isaaclab_mimic/motion_planners/curobo/`),
      독립 repo 아님. **채택 권장 = 우리 파이프라인 유지 + (속도) res↓·N↑ + (품질) RoboEngine 픽셀증강(공짜)·recovery/
      perturbation 궤적·VLA-RFT(drift 직격)**. 품질 천장은 §13 audit 대로 **closed-loop drift**(open-loop 충실) →
      BC 데이터 청결만으론 불충분, 복귀 데이터/RL 필요.

- [x] **🟢 카메라DR 정합 + smooth 256ep 재생성 + GR00T-N1.7 20k A/B + 추론감사 (2026-06-15/16)** — 사용자 지시.
      ① **카메라 DR 정합**: `randomize_camera_focal` focal_range `(12,16)→(16,20)`(보정 nominal 18mm 중심, focal 14→18 변경 d1d3fd7 반영). ② **smooth 256ep 재생성**: slew-fix 코드로 `pick_cube_curobo_batch N=16 --record 256` → **256ep·125602frame**(tail-trim ~490f/ep), 2h9m. ③ **HF 신규 repo** `taehunkim/so101_sim_pick_cube_smooth`(기존 240ep **보존**, A/B 위해). 
      **GR00T A/B**(동일 N1.7·하이퍼 20k·batch8·백본동결·bf16, 데이터만 변수): A=`so101_sim_pick_cube`(240ep jerky·focal14)·B=smooth(256ep·focal18). train_loss A 0.129/B 0.119, 각 1h43m(25.9smp/s). **eval seed0 N=20**(`run_cube_desk_ros_bridge.sh --eval 20`, profile groot_n17): **all-4/per-cube/평균 둘 다 0%**, **ever-in-bowl A 0.0% → B 6.25%(5/80)**. → **B>A 입증**(smooth+cam 수정이 큐브를 그릇까지 보냄), **단 완주 0%**(closed-loop drift 잔존). `outputs/vla_eval_{baseline,smooth}.json`.
      **추론 파이프라인 감사** 신규 문서 `docs/SIM_REAL_INFERENCE_PARITY.md`(sim/real 변환·분기인자). 🔴 **gripper offset sim+0.20/real0 = sim데이터 규약**(Isaac use_default_offset) → **sim 학습모델 real 배포 시 0 가정 깨짐**(함정). 🟠 **GR00T single_arm RELATIVE**(문서 "절대" 정정: `decode_action` 이 입력 state 기준 절대복원해 반환 → action overlay 보정불요, real 은 state 정확도 critical). 🟠 min-max정규화 sim-stats·카메라 intrinsic 갭.
      **GR00T 성능 진단도구** 신규 `scripts/sim/compare_train_vs_async_groot.py`(compare_train_vs_rtc 의 GR00T판): 무작위 ep recorded obs 를 Path A(gr00t ZMQ teacher-forced)·Path B(async gRPC bridge) 통과→recorded 오버레이+입력 timestamp 마커. **ep197: A−rec 4.67°·B−rec 3.60°·A−B 4.91°**(open-loop fidelity 양호, B 가 refill 잦아 tighter). PNG `outputs/compare_groot_ep197.png`.
      **카메라 flat-arm 진단**(사용자 관찰): wrist/front 에서 팔이 명암 없이 flat 보라 = 머티리얼 정상(`patch_robot_colors` 보라PLA diffuse rough0.6·metallic0)인데 **DomeLight(2000) 지배 균일조명 + 매트재질 + 평판 gripper 정면 클로즈업** 탓. 녹화 데이터도 동일(train↔infer 정합, 모델 버그 아님)·sim2real 시각갭. realism 원하면 dome↓·directional↑·specular.
      **SmolVLA A/B 🟢 완료**(사용자 추가지시): A=기존 `so101_smolvla_sim_pick_cube`(240ep jerky 재사용)·B=smooth 256ep 20k batch32→push `so101_smolvla_sim_pick_cube_smooth`. eval profile=smolvla·seed0 N=20·chunk20. **A ever 8.75%(7/20ep)·배치0% / B ever 18.75%(10/20ep)·배치1.25%(1/80 안착)**. `vla_eval_smolvla_{baseline,smooth}.json`.
      **🎯 전체 A/B 종합**(동일 seed0·N20·chunk):

      | 모델 | jerky(A) ever | **smooth(B) ever** | smooth 배치 |
      |---|---|---|---|
      | GR00T-N1.7-3B | 0.0% | **6.25%** | 0% |
      | SmolVLA(0.45B) | 8.75% | **18.75%** | 1.25%(1/80) |

      **결론**: ① smooth+카메라DR 수정이 **두 모델 다 ever-in-bowl 개선**(데이터 품질 효과 일관 입증). ② **SmolVLA > GR00T**(이 태스크): smooth SmolVLA 만 실제 안착 도달. GR00T 약세 가설=RELATIVE action state의존+3B 과대·소데이터. ③ **둘 다 완주(all-4)·안착률 ≤1.25%** — **closed-loop drift 가 공통 벽**(open-loop fidelity 양호, §13 compare_groot 4.67° 참조). 다음 = 데이터 대량화·recovery 궤적·VLA-RFT.

### 🎉 전체 루프 완료 요약 (2026-06-15)
cuRobo(P0-P4) → **render-batch+시각DR 240-ep**(`taehunkim/so101_sim_pick_cube` v3.0) → **SmolVLA 20k fine-tune**
(`taehunkim/so101_smolvla_sim_pick_cube`, wandb oh1gs82t) → **sim closed-loop 추론**(action [1,8,6]·17.7Hz arm 구동).
신규/수정 파일 미커밋(사용자 커밋 요청 대기): `scripts/sim/{lerobot_recorder,upload_to_huggingface}.py`(신규),
`scripts/sim/{pick_cube_curobo_demo,pick_cube_curobo_batch,rollout_to_lerobot}.py`·`src/sim_to_real/{utils/domain_randomization,tasks/pick_cube/pick_cube_env_cfg}.py`·`ros2_ws/.../config/vla_policy.yaml`·`.claude/settings.json`(수정).

### 스루풋 분석 (느림 원인·스케일 경로 — 2026-06-14 사용자 진단)
- **병목 = single-env 매 control step 3캠(480×640) raytrace 렌더**. cuRobo plan 은 phase당 1회라 거의 무관.
  80-ep ≈ ~1.6min/ep(=2.1h). 성공률 ~100%(폐기 0).
- ⚠ **state-replay "성공분만 렌더" 이득이 우리는 ≈0**: 성공률 ~100%라 실패 렌더 낭비가 없고, 재렌더도
  80 에피소드 동일 렌더량. **렌더 총량이 안 줆**.
- **진짜 레버 = 렌더 배치(TiledCamera N-env 동시)**: N 에피소드 프레임을 step당 병렬 렌더 → ~Nx wall.
  = **배치 replay-render**(N env 각자 다른 궤적 재생 + 카메라 ON). 신규 빌드: full scene state(robot q +
  Cube1~4 + Bowl root pose) per-frame dump(batch client) + 가변길이 per-env 재생 + N-env writer. **2048
  스케일 재생성 시 도입**(80-ep 엔 빌드시간>이득이라 미적용). ovrtx 렌더 probe = 그 단계서 측정 후 교체 후보.
- **현 80-ep 결정**: single-env recorder 완주(리스크 0). 배치 replay-render 는 스케일 트랙으로 분리.

> **재개 절차(compaction 후)**: ① 이 §13 진행 상태 체크박스 확인 ② `outputs/p5_logs/*.log` tail(planner.log·record80.log·train*.log) 로 현재 단계 파악 ③ `nvidia-smi` + `ps -ef|grep -E "curobo_planner_server|pick_cube_curobo_demo|lerobot-train"` 로 실행 중 작업 확인 ④ 미완 단계의 위 명령 이어서 실행. GPU 1장 공유 → 단계 **순차**(데이터→학습→추론 동시 금지). planner 사이드카(port 5599)는 데이터 단계 끝나면 ZMQ `{"cmd":"shutdown"}` 또는 PID kill.

---

## 8. 주요 결정사항 (Decision Log)

| # | 결정 | 근거 |
|---|---|---|
| D1 | **cuRobo 채택**(Isaac ROS·SDG·depth-planning 기각) | ROS=비배치, SDG=박스라 자명, depth=sim ground-truth 있음. cuRobo 만 배치 충돌 플래닝 |
| D2 | ⚠️**정정(2026-06-16)**: pose-goal **가능**. ~~joint-goal 만~~ | ~~5-DOF 는 pose-goal 비가능~~ → **오진단**. `ToolPoseCriteria.track_position()`(회전 weight=0)로 native position-only `plan_pose` 작동(probe 0%→97.5%). 옛 근거는 `orientation_tolerance`(게이트)만 풀어본 것. TROUBLESHOOTING §cuRobo 5-DOF position-only 참조. 기존 joint-goal SM(95%)은 유지, 신규 경로는 plan_pose |
| D3 | 해석적 IK·q_bias **유지**(C3), executor 만 교체 | `--calibrate` 검증 1.5mm, 재작성 위험 |
| D4 | **scatter 축소 대신 큐브 간격↑**(사용자) | 천장은 reach 아니라 clustering. 뭉치면 grasp 못 찾음 |
| D5 | 카메라 홀더를 **충돌 모델에 포함**(사용자) | fixed joint 강체 → 모르면 transit 중 큐브/그릇 침 |
| D6 | VLA sim 추론 = **single-env ROS2+async 인터랙티브**(사용자) | GUI drag 추종 확인 목적. multi-env 정량 eval 은 후순위 |
| D7 | VLA deps **격리**(policy-server gRPC), in-process 금지 | transformers/torch ↔ isaac 핀 ABI 위험. + real/sim parity |
| D8 | literal 100% 비목표, **~95–99%+clean**(사용자) | PhysX 비결정성·reach-edge 노이즈. 데이터는 실패 episode 버림 |
| D9 | **해석적 IK config 직접 사용 금지 — cuRobo IK 로 refine** (P2 선행) ⚠️**일부 정정(2026-06-16)**: 해석적 IK 자체가 불필요해짐 | 해석적 FK 는 평면근사라 cuRobo(실제 URDF) FK 와 mean 6.9mm·max 15mm 발산(`_probe_fk_consistency`). 직접 plan_cspace 면 grasp 가 큐브 빗나감. ~~recipe: 해석적 ik_reach → feasible orientation+seed → cuRobo IK refine~~. **정정**: native `plan_pose`(`track_position` position-only 또는 goal-set 자세후보)면 cuRobo IK 가 EE position 을 5mm tol 내 정확 해결 → **해석적 IK·FK 발산 우회 불요**(D2 정정 연쇄). 기존 joint-goal SM 은 D9 recipe 로 작동 중이라 유지, 신규 SM(Stage 2)은 plan_pose 경로 |
| D10 | **cuRobo+Isaac = subprocess IPC** (in-process 금지) | warp 1.14(cuRobo) ↔ omni.warp.core 1.8.2(isaacsim) 상호배타 — 한 프로세스 공존 물리적 불가(ABI 게이트 양방향 확증). cuRobo 플래너 = 별도 프로세스(warp 1.14, isaac 無), Isaac env = 메인, **ZMQ REQ/REP**(ik/plan/set_world). plan 은 phase 당 1회라 IPC 오버헤드 무시. P0 트랩이 예고. P3 배치도 이 구조 유지 |
| D11 | **IK 충돌-free, plan_cspace 만 충돌-aware** + **grasp 미세동작 직접 joint 실행** | set_world 는 planner 만 갱신(IK 는 더미씬 유지). IK 에도 장애물 주면 이웃 근처 grasp config 를 그리퍼 sphere 스침으로 과다 기각(수동 가능한데 실패). 이웃 회피는 env clearance roll-선택 + plan_cspace. grasp 미세동작(descend·slide·lift)은 plan_cspace 가 target 근처서 과다기각 → 직접 보간(짧고 clearance 확보됨). 긴 transit(→pre·lift→bowl)만 plan_cspace |
| D12 | **P3 = lock-step 벡터화 SM**(async per-env 기각) | env.step 은 글로벌 배치라 async 는 한 env 일시정지 불가 + cuRobo batch 모델(N문제 N world 1 GPU call)과 충돌 + bookkeeping 폭증. lock-step(전 env 동일 phase tape, 고정 순서 Cube1→4)이 batch 설계 의도와 정합·결정적. 세금=straggler 인데 고정 horizon traj 일괄 반환이라 waypoint 단위 straggler 없음(retry 단위뿐). per-env world = 동일 스키마 N슬롯 + `update_obstacle_pose`/`enable_obstacle(env_idx)`. **신규 `pick_cube_curobo_batch.py`**(단일-env 데모 100% 보존, 두 파일 중복 일부 수용 — 사용자) |
| D13 | **배치 크기 256 시작→실측 climb**(2048 직행 기각) | max_batch_size 는 config 생성 시 고정. 첫 실측서 OOM/디버그 회피 위해 256(C5 검증 하한 충족)서 시작, VRAM/wall-clock 보고 512→2048 단계 상향. 물리배치(Isaac N)↔planner 배치(≤max_batch_size)는 **서버 내부 청크로 분리** → 클라는 항상 N개 전송 |

---

## 9. 트러블슈팅 (예상 함정 — 발생 시 docs/TROUBLESHOOTING.md 이관)

| 함정 | 대응 |
|---|---|
| cuRobo in-process 설치가 ABI 핀 깸 | P0 게이트 조기검출 → subprocess IPC / 경량 sweep-check 폴백 |
| **target 충돌 끄면 open-jaw clip 재발** | grasp config 자체가 side-approach clip-free 여야. cuRobo 가치 = pose **선정**+이웃/홀더 transit 회피 |
| xrdf stale(홀더·sphere) | P1 전면 검증·재fit, 가정 금지 |
| 2048 batch VRAM/throughput | 청크 루프. plan 은 phase 당 1회(step 비용 아님) |
| cuRobo kinematic traj ↔ PD sag | q_bias 보상 유지(C6) |
| ROS2 환경(P6) | LD_LIBRARY_PATH·`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`·inotify (PATH E 함정 재사용) |
| **해석적 FK↔cuRobo FK 발산(15mm)** | 해석적 IK config 직접 plan_cspace 금지. cuRobo IK refine(D9). zero-pose 만 일치한다고 전 워크스페이스 가정 금물 |
| cuRobo IK position-only 안 됨 | ⚠️**정정(2026-06-16)**: `orientation_tolerance`(수렴 게이트)는 cost 못 끔(맞음) — 하지만 진짜 API 는 **`ToolPoseCriteria.track_position()`**(회전 weight=0)다. `update_tool_pose_criteria` 로 주입 → angular_distance·gradient 0 → native position-only `plan_pose` 작동(probe 0%→97.5%). 해석적 공급(D9) 불요. TROUBLESHOOTING §cuRobo 5-DOF position-only |
| pose-goal Viser 무반응 ≠ stuck flag | `plan_pose` plan 실패(5-DOF). 단, daemon-thread 예외 시 `is_moving` stuck 별도 위험 → try/finally. PYTHONUNBUFFERED=1 로 런타임 print 봐야(background pipe block-buffered) |
| pkill self-kill (exit 144) | `pkill -f "X.py"` 를 `X.py` 실행과 **한 bash 명령**에 합치면 자기 명령줄 매칭→자살. kill 과 launch 분리 |
| TaskStop 해도 Isaac 좀비 (49100 다중 LISTEN→WebRTC 검은화면) | TaskStop 은 bash 만 죽이고 Isaac python child 생존 → 좀비 누적. **python child PID 직접 kill**(`ps…|grep…|awk` 후 kill) |
| grasp 가 큐브 모서리/꼭지점 잡고 밀침 | DR `randomize_cubes` 가 **yaw ±30° 랜덤**. grasp 의 closing-axis 에 **cube 실제 yaw 사용**(snap 에서 quat→yaw). yaw=0 고정 금지 |
| set_world 후 IK 도달가능한데 plan/grasp 실패 (수동 가능) | IK 가 충돌-aware 면 이웃 근처 config 과다 기각. **IK 는 충돌-free**(D11). grasp 미세동작은 **직접 joint 보간**(plan_cspace 미사용) |
| 시작 자세서 안 움직임 (전 큐브 →pre plan 실패) | READY 가 cuRobo self-collision **경계 바로 안쪽**(-1.5,1.4)이면 q_bias sag 로 측정 config 가 경계 넘어 plan 시작 실패. **경계서 떨어진 fold(-1.3,1.2) backoff** + cube1 은 exact READY(측정값 말고) 서 plan |
| 큐브 4개 후 팔이 흘러내림 | idle 시 `act(cur_arm())`(측정값 홀딩)=droop 강화. **`act(READY)`**(목표 명령)로 유지 |
| **grasp 닫힘축이 이웃 향함(X-나란→X-close 실패)** | select_grasp clearance finger 점을 닫힘축 **수직** 잡으면 이웃 향한 closing 우대(버그). finger 점을 **닫힘축 방향(dx,dy)**으로 → 이웃 향한 closing penalize → 수직 closing 선택. 95.3%@256 |
| grasp-effort(그리퍼 힘) 어디? | leisaac `dynamic_reset_gripper_effort_limit` = 매 step 그리퍼 effort=`clamp(가장가까운물체 mass/0.15, 0.5, 10)`. 우리 `utils/gripper_effort.py`(이식) + **PickCubeEnv.step 이 배선**(flag True) → batch/demo/SM 자동 적용. 큐브 0.5Nm gentle(안 으깸·sim2real). 정적 actuator effort_limit_sim=10 은 cap, 런타임 override |
| finger-큐브 잡을 때 빈 공간 | gentle grip(0.5Nm)이 contactOffset **skin 안 눌러** finger 가 큐브서 수 mm 떨어져 마찰 hold(grasp 정상). 그리퍼 contactOffset 미설정(PhysX 기본 큰 skin)이 원인 → **그리퍼 contactOffset 0.002·restOffset 0(큐브와 매칭)** 으로 skin 축소. 큐브 visual=라운드라 면 중앙 외엔 더 떨어져 보임 |
| **🔴 팝콘(그릇서 큐브 충돌 폭발) — 미해결** | 가벼운 큐브(20-35g) 미끌 그릇 다중 적재 → 솔버 침투 해소 불안정 → 폭발(2048서 1-miss 32% 의 큰 몫). **maxDepenVel↓ = grasp 15pp 회귀(폐기). restitution 무관(combine min). friction 변경=넘침위험(금지).** 후보(미검증, 비회귀 우선): ① **damping↑**(linear/angular 1.5→4~8: 충돌 에너지 소산, free-motion만 영향, grasp 무해 기대) ② **solverPos 32→64**(velocity cap 아님) ③ **adaptive release**(그릇 채워질수록 release 높이↑, 낙하 KE↓) ④ 큐브 mass↑(grasp 재검증 필요) ⑤ compliant contact. **각 후보 256 격리검증 필수**(maxDepen 처럼 회귀 가능) |

---

## 10. 검증 방법

- **자가검증 루프(GUI 불요)**: 헤드리스 실행 → taxonomy JSON(success/clean/steps/reason_histogram) +
  `--video` → **ffmpeg 프레임 추출 → 에이전트가 PNG Read 로 직접 진단**(descend-clip/knocking). sphere
  는 오버레이 PNG 렌더 후 Read.
- **Phase 게이트**: P0 gen_so101_xrdf IK>90% / P1 collision율·오버레이 / P2 4-env 영상 / P3 2048
  taxonomy / P6 GUI drag 추종.
- **사람 체크포인트**: P1 sphere 오버레이, P2 첫 grasp 영상, P3 2048 결과, P6 정성 추종 — 그 외 자율.

---

## 11. 재현 절차

> ⚠ **cuRobo 는 `uv run --no-sync --group isaac`**(sync 가 curobo 삭제). 공식 예제 `--robot`/`--scene`
> 는 **절대경로** 필요(상대경로는 curobo content 디렉토리 기준 해석 → FileNotFound). Viser 는
> 0.0.0.0 바인딩 → tailscale `100.79.237.116:<port>` 접속.

```bash
# ── P0/P1 (DONE, 작동 확인) ──────────────────────────────────────────────
# robot config 빌드 (so101.xrdf=Isaac + so101_curobo.yml=cuRobo). --visualize 면 Viser sphere 뷰.
uv run --no-sync --group isaac python scripts/sim/build_so101_xrdf.py [--visualize --viz_port 8085]

# xrdf↔cuRobo 정합 검증 (로드·sphere·FK·IK)
uv run --no-sync --group isaac python scripts/sim/validate_so101_curobo.py

# reactive MPC Viser (EE 드래그 추종+장애물 회피, 공식 reactive_control 적용)
uv run --no-sync --group isaac python scripts/sim/reactive_so101_viser.py --port 8086

# 공식 motion_planning Viser (Move=pose계획, Grasp=3단계). ⚠ --scene 는 SO-101 크기 so101_scene.yml
#   (franka 크기 collision_test.yml 은 SO-101 start 충돌 → 전부 실패). 둘 다 절대경로.
uv run --no-sync --group isaac python \
  ref_repos/curobo/curobo/examples/getting_started/motion_planning.py --visualize \
  --robot "$(pwd)/assets/robots/so101_curobo.yml" \
  --scene "$(pwd)/assets/robots/so101_scene.yml" --port 8087
#   5-DOF 라 임의 orientation 드래그는 "Motion planning failed"(정상). position 위주로 드래그.

# ── P1 후속 TODO (다음 세션) ─────────────────────────────────────────────
# 공식 forward_kinematics.py / inverse_kinematics.py 예제로 SO-101 FK·IK 단독 검증 (사용자 지시).

# ── P2 (DONE) — IPC 사이드카 + 4-큐브 pick-place ─────────────────────────
# ① 터미널 A: cuRobo planner 사이드카 (isaac 無, warp 1.14 유지, ZMQ REP)
uv run --no-sync --group isaac python scripts/planning/curobo_planner_server.py --port 5599
# ② 터미널 B: Isaac env 클라이언트 (ZMQ REQ). headless 검증 / livestream 관전 / 카메라.
#   headless DR 검증(시간·placed):
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
  scripts/sim/pick_cube_curobo_demo.py --headless --planner_port 5599 --loop 3
#   원격 livestream 관전 + 3-cam (mode1 PUBLIC_IP):
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
  scripts/sim/pick_cube_curobo_demo.py --public_ip 100.79.237.116 --planner_port 5599 --cameras
#   주요 옵션: --no_dr(고정 spawn) --active_objects N --loop N --side_offset --grip_open/close
#   ⚠ 데모 재시작 시 python child 직접 kill(TaskStop 만으론 좀비). planner shutdown=ZMQ {"cmd":"shutdown"}.

# ── P3~ (예정) ───────────────────────────────────────────────────────────
# P3: multi-env 배치 — BatchMotionPlanner.plan_cspace(multi_env=True) per-env world. 다음 논의.
# P5: 데이터(state-replay 재렌더) / P6: ROS2+async VLA 추론
```

---

## 12. 참고 자료

- 승인 플랜: `~/.claude/plans/so-101-pick-place-vla-federated-shannon.md`
- 형제 문서: [`PICKCUBE_SM_PROJECT.md`](PICKCUBE_SM_PROJECT.md)(해석적 SM·천장 진단) ·
  [`PICKCUBE_RL_PROJECT.md`](PICKCUBE_RL_PROJECT.md)(RL) · [`PATH_E_CUMOTION_ROS.md`](PATH_E_CUMOTION_ROS.md)
  (cuMotion+ROS 실기기) · [`GRASP_PHYSICS.md`](GRASP_PHYSICS.md) · [`LULA_GUI_TUNING.md`](LULA_GUI_TUNING.md)
- cuRobo: `ref_repos/curobo/curobo/examples/getting_started/motion_planning.py`, `_src/motion/`,
  `_src/collision/attachment_manager.py`, `sphere_fit.py`
- 핵심 코드: `scripts/environments/pick_cube_state_machine.py`, `src/sim_to_real/tasks/pick_cube/
  pick_cube_env_cfg.py`, `src/sim_to_real/utils/domain_randomization.py`, `scripts/sim/{gen_so101_xrdf,
  rollout_to_lerobot,lerobot_units,run_cube_desk_ros_bridge}.py`, `assets/robots/so101.xrdf`
- 실기기 추론: `docker/policy-client-shim.py`, `docker/policy-entrypoint.sh`, `scripts/policy_server_rtc.py`
- perf probe: `scripts/perf/{tiled_camera_throughput_bench,ovrtx_probe,ovphysx_probe,isaac_env_step_throughput}.py`

---

## 14. cuRobo native pose-goal + 인터랙티브 viser 지식 (2026-06-16, Stage 1)

> `scripts/sim/motion_plan_so101_viser.py` (공식 `motion_planning.py --visualize` 5-DOF 판). Move·
> Grasp·Home 버튼 + 궤적 plot. 여기 모은 지식은 **Stage 2 cuRobo-native SM** 설계에 직접 재사용.
> 실행: `uv run --no-sync --group isaac python scripts/sim/motion_plan_so101_viser.py --port 8088`
> (tailscale `100.79.237.116:8088`).

### 14.1 핵심 API (D2/C4/D9 정정의 실체)

| 용도 | API |
|---|---|
| **position-only**(자세 무시) | `planner.update_tool_pose_criteria({tool: ToolPoseCriteria.track_position()})` → `plan_pose`. 회전 weight `[1,1,1,0,0,0]` → `wp_tool_pose` 커널서 `angular_distance=0`·gradient=0. **`orientation_tolerance`(수렴 게이트)와 다름** — 게이트만 풀면 cost 살아 5축 실패(옛 오진단). |
| **자세 후보 선택** | `GoalToolPose`(num_goalset>1) 또는 후보별 독립 batch IK. `ToolPoseCriteria`(public `curobo.types`). |
| **joint-goal transit** | `planner.plan_cspace(goal_state, current_state, max_attempts)` — 충돌회피, 자세 이슈 없음. |
| **임의 링크 FK** | `get_link_poses` 는 cam mount 미지원 → `kin.config.kinematics_config.get_sphere_index_from_link_name("wrist_cam_mount_link")` + `compute_kinematics(q).get_link_spheres()` 로 sphere 월드좌표. |

### 14.2 Move (position-only plan_pose)

- feasible 큐브존 타깃 **~90%**(probe N=40). 실패 = 그 위치가 5축이 충돌-free 로 못 가는 **내재
  infeasible**(IK collision-aware 인데 해 없음) — 튜닝 버그 아님.
- 노브: `num_trajopt_seeds=8`·`optimizer_collision_activation_distance=0.025`. **`enable_graph_attempt=1`
  유지**(기본). ⚠ **graph 즉시(=0)는 5축 cramped 에서 오히려 악화**(90%→62.5%, probe). graph 는
  retry 단계에서만.

### 14.3 Grasp (holder-up swoop 3단계) — 공식 `plan_grasp` 가 5축서 안 되는 이유 + 대체

- 공식 `plan_grasp` 는 approach/grasp/lift 내부에서 `linear_motion`·full-pose criteria 로 풀어
  **5축이 orientation 못 맞춰 실패**(+ 끝에 standard_criteria 복원해 다음 Move 도 깸). → 직접 구현.
- **3단계**: ① 후보 자세 배치 IK → reachable 중 **카메라 홀더 최상(Δz↑)** config q* 채택 ② **같은
  자세 q\*** 로 hover(큐브 +OFF) IK → `plan_cspace`(joint-goal)로 이동 ③ hover↔grasp **직접 joint 보간**
  (D11: 짧은 grasp 미세동작은 trajopt 과다기각) + gripper open→close.
- **hover 와 grasp 를 같은 q\* 로** 해야 함. 안 그러면(approach 자세 자유) optimizer 가 holder-down
  골라 **모션 중 wrist 180° flip**(사용자 지적). hover=grasp 자세 동일 → 내내 holder-up.

### 14.4 5-DOF grasp 자세 manifold (측정값)

- **낮은 큐브존(z~0.05) = top-down**(tool-z↓), **먼 타깃(dist↑) = 수평**(tilt↑). reachable 자세는
  approach 방위각이 위치에 묶인 좁은 manifold.
- **카메라 홀더 up = `wrist_roll −90°`**(측정: 수평 reach 자세서 −90→holder Δz **+0.044(up)**·+90→
  down·0→side). 중립(q=0) 홀더 = gripper **+y(왼쪽)**, holder-grip=(-0.015,+0.043,+0.014).
- 후보 goal-set = **top-down([0,1,0,0]) → roll(approach축, 홀더 up/down) → tilt(0-90° pitch) →
  uniform world-z azimuth**. tilt `[0,25,50,75,90]`·roll `[-90,-45,0,45,90,180]`·**azimuth 12 uniform**
  = 360. ⚠ **azimuth 를 타깃방향(atan2)에 centered 하면 오히려 악화** — 내 파라미터 azimuth 가 pan
  방향과 직접 대응 안 함. uniform 미세(30°)가 정답.
- **holder-up 강제**: collision 모델에 **ground/table plane 이 없어** IK 가 홀더-바닥 관통 config 허용
  → holder-gripper Δz>0.005 인 reachable 후보만 채택, **없으면 grasp 거부**(바닥 박기 금지). 거의 모든
  큐브존서 holder-up reachable(74/80 등), **reach-edge(dist>~0.42·저z)만 거부**(5축 물리한계).
  - 미래: ground plane 을 충돌모델에 추가하면 더 물리적(단 base_link 가 z=0 mount 라 충돌 — base sphere
    disable 필요). 현재는 스코어링으로 대체.

### 14.5 함정 (Stage 2 재사용 주의)

| 함정 | 대응 |
|---|---|
| **current_state stuck** — 도달 config 체이닝이 self-collision 경계 안착 → 이후 모든 plan "Start in collision" 영구 stuck(사용자: "어느 시점부터 어디든 다 안돼") | 실패 시 **home(default) 복귀 후 재시도**(home 은 항상 clean). + 수동 **Home 버튼**. headless 30회 체이닝 최대 연속실패 5+(stuck)→1(복구) |
| **IKSolver 두 batch 크기 호출 → CUDA context 깨짐** | `use_cuda_graph=False`(가변 batch). graph on 이면 batch 360→1 시 "CUDA graph reset is not available"→illegal memory cascade(viser 먹통). TROUBLESHOOTING 참조 |
| **viser 안 뜸** (`No module named 'websockets.asyncio'`) | isaacsim `websockets==12.0` 핀 vs viser≥13.1 → `uv pip install 'websockets>=13.1,<14'`(post-sync 보강). TROUBLESHOOTING 참조 |
| **pkill self-kill**(exit 144) | pkill 과 launch 분리. 또는 `ss -ltnp ':8088'` 로 PID 추출 후 kill |
| grasp 자세 IK 가 holder 방향 안 봄 | IK 는 reachability 만 최적화 → 홀더 up/down 은 후보 자세로 주고 FK 로 사후 스코어링(Δz). collision-aware 라도 ground 없으면 바닥 관통 허용 |
| reach-edge grasp 0 reachable | position-only Move 는 닿아도 orientation-constrained grasp 자세는 manifold 0 → 정직 거부. 5축 본질 |

---

## 15. cuRobo(직접) vs cuMotion(Path E) — 관계·역할 분담

> 자주 헷갈림: "cuRobo 버리고 cuMotion 으로 갈아탈까?" → **비추천**. 둘은 같은 코어, 다른 포장.

### 15.1 관계: cuMotion = cuRobo + MoveIt/ROS 래퍼

- **cuRobo** = NVIDIA GPU 모션플래닝 **라이브러리**(Python/CUDA/warp). 독립 설치(`ref_repos/curobo`, pip).
- **cuMotion**(`isaac_ros_cumotion`) = cuRobo 를 **MoveIt 2 planner plugin** 으로 감싼 ROS 2 패키지
  (빌드 시 cuRobo CUDA 빌드). **코어 플래너는 동일 cuRobo.**
- 즉 cuRobo 가 원본, cuMotion 은 add-on. "cuMotion 에서 cuRobo 추출"은 개념 오류 — 우리는 이미
  cuRobo 를 standalone 직접 사용 중(PICKCUBE 트랙). cuMotion 은 ROS 가 필요할 때의 창구.

### 15.2 두 트랙 역할 분담 (둘 다 유지)

| | **cuRobo 직접** (PICKCUBE, Track A) | **cuMotion** (PATH E) |
|---|---|---|
| 접근 | Python API 직접, ZMQ 사이드카 | ROS 2 / MoveIt 2 (MoveItPy) |
| 배치 | **`BatchMotionPlanner(multi_env)`** = N문제 1 GPU call(256~2048 env) | 불가(단일 arm·ROS 페이스) |
| 용도 | **sim 배치 데이터 생성 + sim 폐루프 추론** (throughput·커스텀) | **실기기-지향 ROS 제어**(ros2_control·trajectory) |
| 5-DOF 우회 | `ToolPoseCriteria.track_position()`·goal-set (Python 직접, §14) | MoveIt `kinematics.yaml`(rotation_scale·orientation_threshold)·joint-goal FK 샘플 |
| 현재 코드 | `curobo_planner_server.py`(single ik/plan + batch init/ik/plan), `pick_cube_curobo_{demo,batch}.py`, `motion_plan_so101_viser.py` | `ros2_ws/` cuMotion config + SM 노드 |

- **공유**: `so101.xrdf`·cube_desk 씬·GRASP_PHYSICS 물리 튜닝. (Stage 2 에서 grasp 자세 선정 로직도 공통 모듈화 후보.)
- **5-DOF 한계·grasp 어려움은 둘 다 동일**(같은 코어 + 같은 로봇 물리). 플래너 갈아타기로 안 풀림 — Path E
  블로커도 grasp 물리. → 갈아타기는 지름길 아님. cuMotion 이 이기는 영역 = **실기기 배포**(ROS 표준).

### 15.3 warp 분리 패턴 (둘 다 같은 이유)

cuRobo(warp 1.14) ↔ Isaac Sim(omni.warp.core 1.8.2)는 한 프로세스 공존 불가(D10). **두 트랙 다 동일
패턴으로 회피** — cuRobo-stack 과 Isaac-stack 을 별 프로세스로 두고 IPC:

```
cuRobo 직접 :  [cuRobo proc · warp1.14] ──ZMQ──►   [Isaac env · omni.warp]
cuMotion    :  [ROS 컨테이너: move_group+cuMotion(cuRobo·warp)] ──ROS DDS──► [Isaac Sim bridge · omni.warp]
```

차이 = IPC 전송(ZMQ vs ROS DDS)뿐. cuMotion 이 ROS 2 인 건 **본성**(MoveIt 플러그인, 실로봇 표준)이지
warp 회피용 워크어라운드가 아님 — 단, 프로세스 분리 자체는 같은 ABI 이유. Path E 는 추가로
rclpy(py3.12) ↔ lerobot/isaac(py3.11) 런타임도 분리.

---

## 16. 1큐브 고정확도 트랙 + closed-loop 0% 진단 (2026-06-16/17)

> **동기**: 4큐브 VLA closed-loop 벽(ever-in-bowl SmolVLA 18.75%·GR00T 6.25%, all-4 완주 0%). 병목 =
> 멀티객체 순서가 아니라 **단일 pick-place + closed-loop drift**(ever 낮음 = 1개도 드묾) → 큐브 1개로 줄여
> 고정확도 우선(커리큘럼 0단계).

### 16.1 전체 루프 실행 (완주)
- **데이터 `taehunkim/so101_sim_pick_cube_1cube`**(v3.0·1024ep·150905f·480): `pick_cube_curobo_batch.py
  --mix_sizes` 신설 — env별 1큐브 무작위 배정(Cube1/2=30mm·Cube3/4=40mm 4중 균등→크기~50/50)+비배정 z=-1
  park(기존 batch 는 active<4 시 비활성 큐브 미park=카메라 노출 버그). 배정 큐브 XY=**직사각형 uniform 재샘플**
  (bowl/base rejection, cube-cube 없음) — DR 4큐브 순차배치 편향(이전 큐브 회피·default 폴백 스파이크) 제거.
  수학검증: 새 방식 완전-내부 셀 CV **2.0%**·chi²≈dof(균일) vs 기존 4큐브-keep1 CV 16.3%·chi² 981. 6D pose
  (바닥 face 6면 균등+z-yaw 360°)는 `full_orient=True` 로 이미 정합. 생성 N=32 ~2.8h(1024×147f≈150k > 256×490,
  해상도 480 유지라 가속 기대 무산).
- **학습**: SmolVLA 20k batch32(`so101_smolvla_sim_pick_cube_1cube` HF) · GR00T-N1.7 convert+20k batch8
  (`outputs/train/so101_groot_n17_sim_pick_cube_1cube/checkpoint-20000`, loss 0.101).

### 16.2 eval — closed-loop 0% (🔴 벽 지속)
- `run_cube_desk_ros_bridge.py --num_cubes 1 --eval 20`(`--cube_name` 크기별 활성·비활성 park):
  **SmolVLA 30·40mm·GR00T 30mm 셋 다 0%**(all-1/per-cube/ever, 큐브 z 불변=미접촉).
- **open-loop**(`compare_train_vs_async_groot`/`_rtc`, ep0/243): MAE GR00T 2.6-2.8°·SmolVLA 1.3°=양호 → **모델·데이터 정상**.
- **시작자세 정합**(bridge `home_q` zeros→학습 frame-0 `[0,-1.235,1.262,-0.381,-1.234,0.848]`rad + velocity버그 fix +
  PD target hold): 적용 확인됐으나 **재평가 여전히 0%**.
- **arm 움직임**(`--dump_obs` annotator+joint): 정적 아님 — home_q(팔 책상향)서 즉시 위/딴데로 큼(정책이 큐브 반대로).
  **카메라/렌더 동일**: bridge 가 env_cfg 카메라 상수 import(공유)·top캠(world고정) 렌더 육안 일치(front 차이는
  shoulder장착캠+arm pose 교란, 카메라 불일치 아님).

### 16.3 sim bridge(eval) ↔ recorder(데이터생성) 환경 차이 (closed-loop OOD 후보)
data·camera·render·시작자세·물리(stiffness 17.8·damping 0.6·effort)·제어율(120/30Hz)·joint순서·단위 **다 정합**. 남은 차이(영향순):

| 차이 | recorder(`pick_cube_curobo_batch`, ManagerBasedRLEnv) | bridge(`run_cube_desk_ros_bridge`, isaacsim.core+ROS2) | 비고 |
|---|---|---|---|
| 🔴 액션 적용 | `SlewLimitedJointPositionAction` — target 을 이전 target 기준 ±(5rad/s÷30) ramp(내부 integrator)+`use_default_offset`(arm0·grip0.20) | ROS2 raw position 직주입(slew term 없음), 물리 `max_joint_velocities`(arm5·grip2.5) 근사만 | target-ramp ↔ joint-vel-cap 메커니즘 차 |
| 🔴 제어 루프/타이밍 | `env.step()` 동기 lock-step(1 action→4 physics→obs, 결정적) | ROS2 async — vla node `create_timer(1/30)` chunk 소비, bridge 3캠 raytrace 로 world.step 느림 → sim-time≠wall 이면 chunk 소비율 desync(§13 라이브 전용·offline 재현불가) | **prime suspect** |
| 🟠 이미지 | TiledCamera → h264 mp4 인코딩→디코딩(학습 입력) | Replicator render_product → raw rgb ROS2·렌더러 다름 | 픽셀 분포 차 |
| 🟡 gripper 속도 | slew 5.0 rad/s | vel cap 2.5 rad/s | grasp 타이밍 |
| 🟡 q_bias 중력보상 | 명령에 적분 보상 추가(녹화 action 에 baked) | integrator 없음(녹화 action 운반) | obs 어긋나면 재유도 불가 |

**다음 레버**: ① 제어타이밍 직격(vla node `create_timer` 의 `use_sim_time` 준수 확인 → bridge lock-step 또는
sim-time-aware 소비) ② bridge Python slew(학습 envelope 재현) ③ deploy 이미지 h264 round-trip ④ recovery 궤적·VLA-RFT.
**cleaner BC data·짧은 horizon·시작자세정합 만으론 closed-loop drift 불충분 입증**(plan gate-5 위험 적중).

### 16.4 신설/수정 코드
- `pick_cube_curobo_batch.py`: `--mix_sizes`(크기 DR + uniform 재샘플 + per-env park).
- `run_cube_desk_ros_bridge.py`: `--cube_name`·비활성 park·`home_q` 학습 frame-0 정합·`--dump_obs`.
- 산출물: `outputs/eval_1cube_posefix/`·`outputs/compare_1cube/`(open-loop overlay PNG)·HF repo 2종.
