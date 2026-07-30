# 03. 시뮬레이션 환경 명세

> **정본**: `src/sim_to_real/tasks/`, `src/sim_to_real/assets/`, `src/sim_to_real/utils/`.
> 단위·관절 순서는 `04_IO_CONTRACT.md` 를 전제한다. 앵커 표기 = `경로::심볼`.

`SimToReal-SO101-*` Gym 환경은 **VLA 추론·데이터 생성 기판**이다. RL 보상·커리큘럼은
제거됐다(`PickCubeRewardsCfg` = 빈 stub). 상세 = §10, `09_TACIT_KNOWLEDGE.md §11`.

---

## 1. 상속 사다리

leisaac Workshop 의 3층 사다리를 2층으로 접어 이식했다. 카메라를 leaf 에 static 으로 두면서
중간층이 불필요해졌다.

```
ManagerBasedRLEnvCfg  (Isaac Lab)
        │
        ▼
SO101TeleopEnvCfg              ← base substrate (태스크 중립)
  로봇 + cube_desk USD + 조명 + slew joint 액션 + 6D joint 관측
  + 리셋 이벤트 + sim/physx 설정 + teleop-device 배선
  rewards = None · terminations = None  (무종료)
        │
        ▼
PickCubeEnvCfg                 ← leaf (태스크 고유분)
  + 큐브/그릇 RigidObject · contact 센서 2개 · static 카메라 3대
  + subtask 관측 · 성공/실패 종료 · DR 이벤트
        │
        ├── PickCubeDREnvCfg        (DR full)
        ├── PickCubeDRBaseEnvCfg    (DR base)
        ├── PickCubeEvalEnvCfg      (디바운스 성공)
        └── PickCubeEvalDREnvCfg    (DR + 디바운스)
```

| 층 | 앵커 |
|---|---|
| base | `src/sim_to_real/tasks/so101_base_env_cfg.py::SO101TeleopEnvCfg` |
| leaf | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::PickCubeEnvCfg` |
| env class | `src/sim_to_real/tasks/pick_cube/pick_cube_env.py::PickCubeEnv` |

`PickCubeEnv` 가 `ManagerBasedRLEnv` 에 더하는 것은 **한 가지뿐** — `step()` 앞에서
`dynamic_reset_gripper_effort_limit_sim(self)` 호출(§6.2).

**신규 태스크**는 base 를 상속해 씬 오브젝트·성공 판정·카메라만 얹는다.

---

## 2. 등록 환경 6종

등록 지점은 `src/sim_to_real/tasks/pick_cube/__init__.py` **한 곳뿐**이다.
`import sim_to_real` 한 번에 monkey patch + 등록이 모두 일어난다
(`src/sim_to_real/__init__.py`, 하위 config 자동 수집 = `tasks/__init__.py`).

| env id | cfg 클래스 | DR | 성공 종료 | 용도 |
|---|---|---|---|---|
| `SimToReal-SO101-Teleop-v0` | `SO101TeleopEnvCfg` | 없음 | **없음(무종료)** | base substrate. teleop·씬 author·무카메라 스모크 |
| `SimToReal-SO101-PickCube-v0` | `PickCubeEnvCfg` | 없음(고정 실측 배치) | 순간 판정 | **기본**. 결정적 재현 |
| `SimToReal-SO101-PickCube-DR-v0` | `PickCubeDREnvCfg` | **full**(종모양 scatter) | 순간 판정 | datagen — 다양성 |
| `SimToReal-SO101-PickCube-DRBase-v0` | `PickCubeDRBaseEnvCfg` | **base**(좁은 사각형) | 순간 판정 | nominal 주변만 |
| `SimToReal-SO101-PickCube-Eval-v0` | `PickCubeEvalEnvCfg` | 없음 | **디바운스**(15 step) | 재현성 최고 평가 |
| `SimToReal-SO101-PickCube-DR-Eval-v0` | `PickCubeEvalDREnvCfg` | full | 디바운스 | DR 하 성공률 평가 |

- 전부 `disable_env_checker=True`
- `Teleop-v0` 만 entry point 가 stock `isaaclab.envs:ManagerBasedRLEnv`, 나머지 5종은
  `PickCubeEnv`
- **DR-off 가 기본**이다. datagen 은 `-DR`, closed-loop eval 은 `-Eval` 을 쓴다.
- `scene.num_envs = 1`, `env_spacing = 2.5` (cfg 기본값; 스크립트가 `--num_envs` 로 override)

> ⚠ **PickCube 5종은 `--enable_cameras` 가 필요하다** — 씬에 static 카메라 3대가 있다(§7).
> 무카메라 실행은 base `Teleop-v0` 를 쓰거나 leaf 에서
> `pick_cube_env_cfg.py::remove_pick_cube_cameras(env_cfg)` 로 카메라와 `images` 관측을
> **함께** 떼어낸다. scene 만 지우면 없는 sensor 를 참조해 에러가 난다.

**DR 변형은 `scene.replicate_physics = False`** 로 강제된다 — robot color DR 이 Replicator
per-env material 바인딩을 쓰는데 Fabric replication 이 켜지면 전 env 가 env_0 material 로
렌더된다(§11.5).

viewer(RecordVideo 가 쓰는 카메라): `eye = (0.06, 1.515, 0.98)`, `lookat = (0.01, 0.245, 0.76)`,
`resolution = (1280, 720)`.

---

## 3. 관측 공간

3개 그룹. 순서 = `policy`, `subtask_terms`, `images`.

| group | term | 함수 | shape | dtype·단위 | concat |
|---|---|---|---|---|---|
| `policy` | `joint_pos` | `mdp.joint_pos_rel` | `(N, 6)` | float32, **radian, `default_joint_pos` 대비 상대값** | `True` |
| `subtask_terms` | `cube_grasped` | `task_mdp.any_cube_grasped` | `(N, 1)` | float 0/1 | `False` |
| `subtask_terms` | `place_cube1` | `task_mdp.object_in_container` | `(N,)` | bool | |
| `subtask_terms` | `ee_pose` | `task_mdp.ee_frame_state` | `(N, 7)` | float32, m + quat wxyz, **robot root frame** | |
| `images` | `top`/`wrist`/`front` | `task_mdp.image`(`normalize=False`) | `(N, 480, 640, 3)` | **uint8** | `False` |

- **noise·clip term 이 없다.** 세 그룹 모두 `enable_corruption = False`.
- `policy` 만이 VLA 정책 계약이다. `subtask_terms` 의 `ee_pose` 는 **privileged**(datagen·eval·
  디버그용)이며 정책 6-dim joint 계약과 별개다.
- `joint_pos_rel` 은 절대각이 **아니다**. 절대 joint 는 recorder 가 따로 기록한다
  (`05_DATA_SPEC.md §6`).

### 3.1 subtask term 파라미터

`cube_grasped` (`pick_cube_env_cfg.py::PickCubeObservationsCfg`):

| 파라미터 | 값 |
|---|---|
| `jaw_sensor_cfg` / `gripper_sensor_cfg` | `contact_jaw` / `contact_gripper` |
| `cubes` | `CUBE_NAMES` = `['Cube1']` |
| `desk_top_z` | `0.705` (`_DESK_TOP_WORLD_Z`) |
| `min_lift` | `0.03` |
| `warmup_steps` | `15` |
| `force_threshold` | `0.5` (N) |

`place_cube1`: `container_center_xy = (-0.22, 0.265)`, `radius = 0.06`, `height_range = (0.005, 0.12)`.

### 3.2 grasp 신호 판정 로직

앵커: `src/sim_to_real/tasks/pick_cube/mdp/observations.py::any_cube_grasped`
(leisaac `any_vial_grasped` 이식)

접촉력은 contact sensor 의 `force_matrix_w` 를 body 축으로 합산해 `force_threshold` 와 비교한다.
`require_both_fingers=True`(기본)면 jaw·gripper **둘 다 같은 큐브**에 접촉해야 한다(envelope grasp).

env-state hysteresis:

| 전이 | 조건 |
|---|---|
| 개시 | 접촉 성립 **AND** `cube_z > desk_top_z + min_lift` **AND** 아직 holding 아님 |
| 유지 | holding **AND** 접촉 유지 (높이 무관) |
| 해제 | 접촉 끊김 |
| 강제 클리어 | `episode_length_buf <= 1` (리셋 직후) |
| 무조건 0 | `episode_length_buf < warmup_steps` (초기 접촉 노이즈 차단) |

상태는 `env._cube_is_holding` 에 산다. 이 term 은 자체 self-check 6케이스를 갖는다:
`python3 src/sim_to_real/tasks/pick_cube/mdp/observations.py`.

> `desk_top_z` 를 **명시 주입**하는 이유: 공유 상수 `_geometry.DESK_TOP_Z`(0.76)는 이전 태스크
> 잔재라 쓰지 않는다. 코드 주석이 그렇게 명시한다. 관련 결함 = §10.3.

---

## 4. 액션 공간

term 이름 `arm` 하나. 클래스 =
`src/sim_to_real/tasks/common/mdp/actions.py::SlewLimitedJointPositionAction`.

| 항목 | 값 |
|---|---|
| 차원 | 6 |
| joint 순서 | `SO101_JOINT_ORDER` (`04_IO_CONTRACT.md §2.1`) |
| 단위 | **radian, 절대 joint target** |
| `scale` | `1.0` |
| `use_default_offset` | **`False`** — offset 0, 절대 target |
| `max_velocity` (base) | 전 축 `5.0` rad/s |
| `max_velocity` (pick_cube) | arm `5.0`, **gripper `2.5`** rad/s |

### 4.1 slew 동작

```
desired  = raw × scale + offset          (cfg.clip 있으면 clamp — 현재 미설정)
delta    = clamp(desired − limited, ±max_step)
limited += delta
processed_actions = limited
```

`max_step = max_velocity × (sim.dt × decimation) = max_velocity / 30`:

| 축 | max_velocity | per-step 상한 |
|---|---:|---:|
| arm 5축 | 5.0 rad/s | 0.1667 rad |
| gripper | 2.5 rad/s | 0.0833 rad |

리셋 시 `limited = 현재 joint_pos` 로 재설정된다 — 리셋 직후 첫 step 이 현재 자세에서 출발한다.
`max_velocity` 는 float·dict·None 을 받으며 `<= 0` 이면 `inf`(무제한)로 해석한다.

> ⚠ **arm 을 2.5 rad/s 로 하드캡하면 안 된다.** 근거 = `09_TACIT_KNOWLEDGE.md §4`.

gripper 만 2.5 로 낮춘 이유: 닫을 때 명령 속도를 줄여 큐브를 튕겨내지 않게 하려는 것
(grasp valley 완화). 공유 상수 `SO101_JOINT_TARGET_MAX_VELOCITY` 자체는 건드리지 않고
leaf 에서 gripper 만 override 한다.

### 4.2 대체 action 구성

기본 action 은 VLA 용 6D slew joint target 이다. teleop·SM 드라이버는 필요 시 다른 구성을
주입한다 (`src/sim_to_real/devices/action_process.py::init_action_cfg`).

| device | arm | gripper | 총 dim |
|---|---|---|---:|
| `so101leader` | `JointPositionActionCfg` 5축 | `JointPositionActionCfg` | 6 |
| `keyboard` / `gamepad` | `DifferentialIK` 4축(relative) | `RelativeJointPositionActionCfg` `[shoulder_pan, gripper]` | 8 |
| `so101_state_machine` | `DifferentialIK` 5축 (`dls`, `lambda_val=0.04`) | `BinaryJointPositionActionCfg` (open 1.0 / close 0.4) | 8 |

SM datagen 전용 cfg = `src/sim_to_real/datagen/sm_actions.py::StateMachineActionsCfg`
(IK 7D + binary gripper 1D = 8D, `body_name="gripper"`).

`SO101TeleopEnvCfg.use_teleop_device(device)` 는 `task_type` 을 저장하고,
`keyboard`/`gamepad`/`so101_state_machine` 이면 `robot.spawn.rigid_props.disable_gravity = True`
로 바꾼다(떨림 없는 결정적 직접 제어). action term 구성 자체는 드라이버 책임이다.

---

## 5. 시뮬레이션 파라미터

앵커: `src/sim_to_real/tasks/so101_base_env_cfg.py::SO101TeleopEnvCfg.__post_init__`

| 항목 | 값 | 의미 |
|---|---|---|
| `sim.dt` | `1/120` | 물리 120 Hz |
| `decimation` | `4` | **정책 30 Hz** (= `FPS`) |
| `sim.render_interval` | `4` | decimation 과 동일 |
| `episode_length_s` | `30.0` | 900 step @30 Hz |

PhysX:

| 항목 | 값 |
|---|---|
| `enable_external_forces_every_iteration` | `True` |
| `bounce_threshold_velocity` | `0.01` |
| `friction_correlation_distance` | `0.00625` |
| `gpu_found_lost_aggregate_pairs_capacity` | `4194304` (`1024·1024·4`) |
| `gpu_total_aggregate_pairs_capacity` | `1048576` |
| `gpu_max_rigid_patch_count` | `1048576` (`16·2**16`) |
| `gpu_collision_stack_size` | `536870912` (`2**29`) |

---

## 6. 액추에이터

앵커: `src/sim_to_real/assets/robots/lerobot.py::SO101_FOLLOWER_CFG` (단일 소스).
전부 `ImplicitActuatorCfg`. 씬은 `.replace()` 로 prim path·contact 센서·init pose 만 override 한다.

| joint | stiffness | damping | effort_limit_sim | velocity_limit_sim | 실 HW (감속비 / 정격토크) |
|---|---:|---:|---:|---:|---|
| `shoulder_pan` | 55 | 0.7 | 30 | 10.0 | 1/191 · 34.4 N·m |
| `shoulder_lift` | 30 | 0.8 | 30 | 10.0 | 1/345 · 62.1 N·m (최대, load-bearing) |
| `elbow_flex` | 25 | 0.7 | 30 | 10.0 | 1/191 · 34.4 N·m |
| `wrist_flex` | 12 | 0.5 | 30 | 10.0 | 1/147 · 26.5 N·m |
| `wrist_roll` | 7 | 0.5 | 30 | 10.0 | 1/147 · 26.5 N·m |
| `gripper` | 4 | 0.3 | 30 | 10.0 | 1/147 · 26.5 N·m |

base 축은 stiffness 를 높여 load-bearing, 손목·그리퍼는 낮춰 섬세 제어.
`effort_limit_sim = 30` 은 안전을 위해 실 HW 토크보다 낮게 통일한 값이다.
`velocity_limit_sim = 10` 은 slew cap 5 rad/s(§4) 추종 헤드룸이다.

기타 articulation 설정: `fix_root_link=True`, `enabled_self_collisions=True`,
`solver_position_iteration_count=4`, `solver_velocity_iteration_count=4`,
`soft_joint_pos_limit_factor=1.0`.

### 6.1 초기 자세

| joint | 값 (degree) |
|---|---:|
| `shoulder_pan` | 0 |
| `shoulder_lift` | −100 |
| `elbow_flex` | 90 (요청 +100°가 USD 상한 90°로 캡됨) |
| `wrist_flex` | 70 |
| `wrist_roll` | −100 |
| `gripper` | 0 |

로봇 배치: `pos = (0.0, 0.0, 0.6749)`(world 원점 XY, z = 책상 상판 0.705 − base_min_z 0.0301),
`rot = (0, 0, 0, 1)` wxyz identity.

### 6.2 gripper effort 런타임 clamp

> ⚠ **`effort_limit_sim = 30` 은 gripper 에 실질적으로 적용되지 않는다.**

`PickCubeEnv.step()` 이 매 step
`src/sim_to_real/utils/gripper_effort.py::dynamic_reset_gripper_effort_limit_sim` 를 호출해
**가장 가까운 rigid object 의 질량**으로 gripper joint effort limit 을 덮어쓴다:

```
limit = clamp(mass / mass_scale, min_effort, max_effort)
      = clamp(mass / 0.15, 0.5, 10.0)
```

Cube1 질량 `0.035 kg` → `0.233` → 하한 `0.5` 로 클램프 ⇒ **실효 gripper effort = 0.5**.
`update_threshold = 0.05` 이상 변할 때만 sim 에 write 한다.

목적: 그리퍼가 작은 물체를 convex-decomposition 충돌 hull 안으로 과구동하는 것을 막으면서
들어올릴 힘은 남기는 것(leisaac teleop/replay 동작 이식). 근거 = `09_TACIT_KNOWLEDGE.md §4`.

`SO101TeleopEnvCfg.dynamic_reset_gripper_effort_limit = True` 로 켠다.

---

## 7. 카메라 리그

계약: `observation.images.{top, wrist, front}`, 모두 **640×480 RGB**, `update_period = 1/30`.
헬퍼 = `src/sim_to_real/tasks/common/utils.py::_pinhole_camera_cfg` — `TiledCameraCfg`,
`data_types=["rgb"]`, `horizontal_aperture=20.955`, `update_latest_camera_pose=True`,
offset convention `world`.

| 카메라 | prim path | pos | rot (wxyz) | focal | focus | clipping |
|---|---|---|---|---:|---:|---|
| `top_camera` | `{ENV}/TopCamera` (world 고정) | `(-0.17, 0.77, 1.05)` | `(0.7538, 0.145, 0.1775, -0.6159)` | 19.0 | 1.3 | `(0.1, 6.0)` |
| `wrist_camera` | `{ENV}/Robot/gripper/WristCamera` (local) | `(0.0, 0.045, -0.04)` | `(0.3562, -0.6108, 0.6108, 0.3562)` | 19.0 | 0.2 | `(0.02, 3.0)` |
| `front_camera` | `{ENV}/Robot/shoulder/FrontCamera` (local) | `(-0.045, 0.0, 0.025)` | `(0.0, 0.0, 1.0, 0.0)` | 19.0 | 1.0 | `(0.1, 6.0)` |

- `top` 은 world frame 절대 좌표(로봇 뒤 −y 높은 곳에서 내려보는 급경사 oblique).
- `wrist` 는 gripper 링크 자식 prim → gripper 회전을 따라간다.
- `front` 는 shoulder 링크 자식 prim → `shoulder_pan` 회전을 따라간다.
  (USD 컨벤션: URDF `shoulder_link` → USD `shoulder`, `_link` 접미사 제거)

값은 GUI 카메라 튜너(`teleop_se3_agent.py --tune_cameras`)로 보정한 결과다.
튜너 회전 입력 → 저장된 quat: top `rot_xyz_deg=(63.5, 0, -168.5)`,
wrist `(-29.5, 0, 0)`, front `(-90, 0, -90)`.

런타임 override API:

| 함수 | 용도 |
|---|---|
| `make_pick_cube_camera_cfgs(...)` | 3개 cfg 생성(전 인자 None 이면 기본값) |
| `add_pick_cube_cameras(scene_cfg, ...)` | 같은 필드명으로 in-place `setattr` — 튜너 override |
| `remove_pick_cube_cameras(env_cfg)` | 카메라 3개 + `observations.images` 그룹을 **함께** 제거 |

미사용 잔재 상수: `_TOP_CAMERA_TARGET`(look_at 계산 경로가 꺼져 있음), `_FRONT_CAMERA_POS`(기록용).

---

## 8. contact 센서

앵커: `pick_cube_env_cfg.py::PickCubeSceneCfg`

| 센서 | prim path | update_period | filter |
|---|---|---|---|
| `contact_jaw` | `{ENV}/Robot/jaw` (가동 손가락) | `0.0` | `{ENV}/Scene/Cube1` |
| `contact_gripper` | `{ENV}/Robot/gripper` (고정 손가락) | `0.0` | 동일 |

로봇 spawn 의 `activate_contact_sensors=True` 는 base 층에서 켠다.
필터 목록은 **모듈 상수**(`_CUBE_CONTACT_FILTER`)로 클래스 밖에 둔다 — 클래스 속성이면
`InteractiveScene` 이 asset 으로 오인한다.

`ee_frame` (`FrameTransformerCfg`, `{ENV}/Robot/base` 기준)의 target 순서는 **고정**이다:

| idx | 프레임 | 소비처 |
|---|---|---|
| 0 | `gripper` | `ee_frame_state` 의 기준 프레임 |
| 1 | `jaw` | `object_grasped`·`ee_near_object` 의 jaw 위치 |

kinematic 센서라 렌더·카메라와 무관하다 (`--enable_cameras` 불요).

---

## 9. 씬 지오메트리

### 9.1 큐브

단일 진실 소스 = `src/sim_to_real/utils/cube_specs.py::CUBE_SPECS` (`CubeSpec` dataclass).
크기·질량만 1차값이고 나머지는 파생이다. **크기 변경은 이 파일 한 곳만 고치고 USD 를 재author 한다.**

| name | size (m) | mass (kg) | half_extent | footprint_radius (`s/2·√2`) |
|---|---:|---:|---:|---:|
| `Cube1` | 0.040 | 0.035 | 0.020 | 0.028284 |
| `Cube2` | 0.040 | 0.035 | 0.020 | 0.028284 |
| `Cube3` | 0.050 | 0.055 | 0.025 | 0.035355 |
| `Cube4` | 0.050 | 0.055 | 0.025 | 0.035355 |

질량 근거(코드 주석): 의자다리 커버 폼이라 부피 완전비례보다 가볍게, **쉘(표면적 ∝ 변²) 비례** —
40 mm 35 g, 50 mm `35 × (50/40)² ≈ 54.7 → 55 g`.

`cube_specs.py` 는 **stdlib 만** 쓴다(상대 import 금지) — author 스크립트가 AppLauncher 이전에
importlib 로 직접 로드하기 때문이다.

**현재 활성 큐브는 `Cube1` 하나**다 (`src/sim_to_real/utils/constant.py::CUBE_NAMES = ['Cube1']`).
나머지는 씬 USD 에 존재하지만 env cfg 가 wrap 하지 않는다.

#### 9.1.1 크기 DR 사다리 (2026-07-29)

`CUBE_SIZE_CHOICES = (0.025, 0.030, 0.035, 0.040)` — DR-on env 가 env 마다 하나를 뽑는다
(등확률, 5 mm 간격). 질량은 `mass_for_size(s) = 0.035 · (s/0.040)²` (쉘 비례, `CUBE_SPECS`
와 같은 규칙)로 함께 따라간다.

| size (m) | half | mass (kg) | USD scale (`size/0.040`) |
|---:|---:|---:|---:|
| 0.025 | 0.0125 | 0.0137 | 0.625 |
| 0.030 | 0.0150 | 0.0197 | 0.750 |
| 0.035 | 0.0175 | 0.0268 | 0.875 |
| 0.040 | 0.0200 | 0.0350 | 1.000 |

**authored 40 mm 가 상한**이다 — scale ≤ 1 만 쓴다. 키우는 방향은 spawn z·DR 이격·planner
obstacle blob 이 전부 40 mm 기준이라 함께 손봐야 한다(줄이는 쪽은 모두 안전측).
이벤트 정의 = §11.6, grasp 조준에 미치는 영향 = `09_TACIT_KNOWLEDGE.md §14.8`.

### 9.2 고정 배치 (DR-off)

| 오브젝트 | env-local 좌표 | 산출 |
|---|---|---|
| 책상 상판 z | `0.705` | `_DESK_TOP_WORLD_Z` |
| Cube1 | `(-0.015, 0.255, 0.726)`, yaw 0° | z = `0.705 + half_extent 0.020 + slack 0.001` |
| Bowl | `(-0.22, 0.265, 0.715)`, yaw 0° | `_BOWL_INIT_STATE` |

실측 근거(코드 주석, 2026-06-30 사용자 실측): 큐브 중심 = 책상 앞 모서리(env y = −0.035)에서
+29 cm, 책상 왼쪽 모서리(env x = −0.44)에서 +42.5 cm. 그릇은 앞 모서리 +30 cm, 왼쪽 +22 cm.
`yaw = 0` 은 큐브 한 면이 −y(로봇)를 향하는 자세다.

씬 USD 는 `+y 0.01` 시프트로 배치된다(정적 지오메트리를 로봇 기준 1 cm 뒤로). 큐브·그릇
rigid body 는 leaf `init_state`(env-frame)로 같은 시프트를 독립 반영하므로 이중 이동이 없다.

### 9.3 그릇

앵커: `scripts/environments/author_pick_cube_scene.py`

| 항목 | 값 |
|---|---|
| 바닥 반경 / 상단 반경 | `0.0325` / `0.075` m (지름 65 / 150 mm) |
| 바닥 두께 / 벽 높이 | `0.005` / `0.065` m → 외형 높이 `0.070` m |
| 벽 두께 / 캐비티 바닥 높이 | `0.004` / `0.003` m |
| 시각 mesh 분할 | 위도 `20` 밴드 × 경도 `24` panel |
| 질량 | `0.25` kg |
| 충돌 | **convexDecomposition** (`maxConvexHulls=64`, `hullVertexLimit=64`, `voxelResolution=500000`) |

### 9.4 충돌 근사 규약

**형상별로 다르게 준다.** 근거·측정치 = `09_TACIT_KNOWLEDGE.md §2`.

| 대상 | 근사 | 이유 |
|---|---|---|
| 큐브 | **convexHull** | 볼록이라 SDF 불필요. SDF 는 평평한 책상 접촉에서 normal 이 매 step 뒤집혀 회전 버즈 발생 |
| 그릇 | convexDecomposition | 오목 캐비티. SDF triangle mesh 는 `num_envs>1` 에서 per-instance cooking 비용·불안정 |
| jaw / gripper | **SDF** | 오목 형상, grasp 정확도 필요 |
| 팔 링크 | convexDecomposition | |

contact offset: 정적·두꺼운 면(책상/매트/그릇) `0.004`, **grasp 대상 큐브 `0.002`**
(convexHull 접촉이 안정적이라 좁은 margin 이 가능).

### 9.5 조명

`/World` 계층에 **env 밖 단일 배치**한다. USD 광원은 scope 격리가 없어 `{ENV}/Scene` 안에 두면
env 수만큼 복제돼 N배 과노출된다(IsaacLab #4340/#1729).

| prim | 종류 | intensity | color | 기타 |
|---|---|---:|---|---|
| `/World/Light` | `DomeLight` | 2000.0 | `(0.9, 0.9, 0.9)` | |
| `/World/KeyLight` | `DistantLight` | 1800.0 | `(1.0, 0.98, 0.95)` | `angle=1.0`, `rot=(0.8644, -0.4031, -0.1271, -0.2725)` ≈ `RotateXYZ(-50, 0, -35)°` |

`ground_plane` = `/World/GroundPlane` (`GroundPlaneCfg()`).

---

## 10. 종료 조건

### 10.1 기본 (`PickCubeTerminationsCfg`)

| term | 함수 | `time_out` | 파라미터 |
|---|---|---|---|
| `time_out` | `mdp.time_out` | `True` | `episode_length_s = 30.0` → 900 step |
| `success` | `task_mdp.task_done` | | `objects_cfg=[Cube1]`, `container_center_xy=(-0.22, 0.265)`, `container_cfg=Bowl`, `radius=0.06`, `height_range=(0.005, 0.12)`, `require_rest_pose=False` |
| `cube_lost` | `task_mdp.cube_lost` | `False` | `objects_cfg=[Cube1]`, `fall_z=0.10` |

`cube_lost` 는 회복 불가한 실패를 빠르게 컷한다(큐브를 책상 밖/아래로 쳐낸 에피소드).
성공이 아니고 timeout 도 아니다.

### 10.2 Eval (`PickCubeEvalTerminationsCfg`)

`success` 만 `task_mdp.task_done_confirmed` 로 바뀌고 `confirm_steps = 15`(0.5 s @30 Hz) 가
추가된다. 한 프레임 떨림으로 큐브가 반경에 순간 들어왔다 나가는 **가짜 성공**을 걸러 평가
성공률을 안정화한다(leisaac confirm-counter 이식). 카운터는 `env._pick_success_counter` 에 살고
`episode_length_buf <= 1` 에서 리셋된다.

### 10.3 ⚠ 성공 판정 z 기준의 잠복 결함

> **`task_done`·`cube_lost`·`object_in_container` 의 z 판정이 레거시 상수를 쓴다.**
> 상세 = `09_TACIT_KNOWLEDGE.md §9 INC-10`.

`src/sim_to_real/tasks/common/mdp/_geometry.py::DESK_TOP_Z = 0.76` 은 이전 태스크(pen) 잔재이며
현 책상 상판은 `0.705` 다. `pick_cube/mdp/observations.py` 주석이 "0.76 은 pen 잔재라 쓰지
않는다"고 명시하지만, **종료 판정은 여전히 이 값을 쓴다**. `task_done` 은 `container_cfg=Bowl`
을 받아 xy 는 실제 그릇 좌표를 쓰지만 z 만 하드코딩 상수다.

산술 대조:

| 항목 | z (m) |
|---|---:|
| 성공 판정 창 = `0.76 + height_range` | `[0.765, 0.880]` |
| 책상 위 큐브 중심 | 0.726 |
| 그릇 캐비티 바닥 위 큐브 중심 (`0.715 + 0.005 + 0.003 + 0.020`) | 0.743 |
| 그릇 rim | 0.785 |
| `cube_lost` 임계 = `0.76 − 0.10` | 0.660 |

그릇 안에 안착한 큐브(0.743)가 성공 창 하한 0.765 에 못 미친다. **이 문서는 as-built 기록이며
코드를 수정하지 않는다.** GPU 실행 확인과 수정은 별건이다.

---

## 11. 도메인 랜덤화

DR-off 기본(`SO101BaseEventCfg`)은 씬 리셋과 포즈 jitter 뿐이다.

| event | mode | 내용 |
|---|---|---|
| `reset_scene` | `reset` | `mdp.reset_scene_to_default` |
| `reset_robot_joints` | `reset` | `mdp.reset_joints_by_offset`, `position_range=(-0.05, 0.05)` rad(≈±3°), `velocity_range=(0.0, 0.0)` |

DR-on(`PickCubeDREventCfg`)이 얹는 것:

| event | mode | 파라미터 |
|---|---|---|
| `randomize_bowl` | `reset` | `radius=0.44`, `angle_range_deg=(-4.0, 8.0)` |
| `randomize_cubes` | `reset` | §11.2 |
| `randomize_lights` | `reset` | dome `(1400.0, 2700.0)`, key `(1100.0, 2400.0)`, `warmth_range=(0.0, 1.0)` |
| `randomize_camera_focal` | `reset` | `(14.0, 22.0)`, per-env 독립 샘플 |
| `randomize_robot_color` | **`prestartup`** | §11.5 |
| `randomize_cube_sizes` | **`prestartup`** | §11.6 — `CUBE_SIZE_CHOICES` 이산 샘플 |
| `randomize_cube1_material` | **`startup`** | static friction `(1.4, 2.0)`, dynamic `(1.2, 1.7)`, restitution `(0.0, 0.0)`, `num_buckets=64` |
| `randomize_cube1_mass` | **`startup`** | `mass_range=(0.9, 1.1)`, `operation="scale"` |

물리 DR(startup)은 env 간 물리 다양성만 준다 — grasp weld·유지력 추가가 아니다.

### 11.1 ★ 선언 순서 = 적용 순서

`randomize_bowl` 이 `randomize_cubes` **앞에 선언**돼 있다. `EventManager` 가 `cfg.__dict__`
순서로 적용하므로, 큐브 배치의 `min_bowl_sep` rejection 이 **arc 이동 후의 실제 그릇 좌표**를
본다. 순서를 뒤집으면 큐브가 nominal 그릇 기준으로 배치된 뒤 그릇이 움직여 사후에
`min_bowl_sep` 불변식이 깨진다(64 env 중 1개 재현). 근거 = `09_TACIT_KNOWLEDGE.md §5`.

그릇 arc 범위의 기하 근거(코드 주석):

- 왼쪽 한계 −4° — 매트 왼쪽 경계(world x = −0.34): `-0.22 + 0.44·sin(a) ≥ -0.255` → `a ≥ -4.56°`
- 오른쪽 한계 +8° — 그릇-Cube3 겹침: 유효 충돌 반경
  `0.075 + 0.0354 + 0.002 + 0.004 = 0.1164 m`, Cube3 최악 위치 `(-0.05, 0.235)` 기준 임계
  `9.48°` → 여유 포함 8°

### 11.2 큐브 스폰 영역

**단일 기하 소스** = `src/sim_to_real/tasks/pick_cube/spawn_area.py` (순수 python, isaaclab 무의존).
env cfg · `pickplace_sm --sweep_grid` · `plot_sweep` 세 곳이 이 모듈을 공유하므로 경계가
어긋나지 않는다. **값을 바꿀 땐 여기 한 곳만.**

```
in_spawn_area(x, y) =  |x| ≤ bell(y)                       # 좌우대칭 종모양 (최외곽)
                     ∧ (x, y) ∉ arm_exclude_box            # 로봇암 주변 배제
                     ∧ dist((x,y), bowl) ≥ MIN_BOWL_SEP    # 그릇 겹침 금지 (내곽)
                     ∧ dist((x,y), pan_axis) ≥ MIN_BASE_SEP # base 발치 배제 (최내곽)
```

| 상수 | 값 |
|---|---|
| `CUBE_SCATTER_BELL` (y, x 반너비) | `(0.06, 0.24)`, `(0.14, 0.24)`, `(0.18, 0.20)`, `(0.22, 0.16)`, `(0.26, 0.08)` — 사이 선형보간, 밖은 clamp |
| `CUBE_SCATTER_X_RANGE` / `Y_RANGE` | `(-0.24, 0.24)` / `(0.06, 0.26)` |
| `CUBE_ARM_EXCLUDE` `(x0,x1,y0,y1)` | `(-0.09, 0.04, -0.045, 0.155)` |
| `BOWL_CENTER_XY` / `MIN_BOWL_SEP` | `(-0.22, 0.265)` / `0.14` |
| `BASE_XY` | `(0.0, 0.0)` — 마운트 원점. **plot 마커·meta 전용** |
| `PAN_AXIS_XY` / `MIN_BASE_SEP` | `(-0.021, 0.023)` / `0.123` |

> ★ **min-reach 가드의 중심은 `shoulder_pan` 축이지 마운트 원점이 아니다.** URDF
> `shoulder_pan` origin `(0.0388, 0, 0.0624)`(base_link 기준)을 env 프레임으로 옮기면 축이
> 마운트 원점에서 −x 2.1 cm · +y 2.3 cm 어긋난다. 원점 기준 가드는 도달 불가한 corner 를
> 통과시킨다(예: env `(-0.092, 0.107)` 은 원점거리 0.141 > 0.135 로 통과하나 pan축 거리
> 0.109 로 IK 도달 불가). 상세 = `09_TACIT_KNOWLEDGE.md §3`.

종 모양의 측정 근거(2026-07-01): pink IK sweep(kinematic map) → gen-traj → Isaac 물리 replay
검증(50 셀 중 46 성공). 물리 성공 셀의 per-y **넓은쪽 |x|** 를 좌우대칭으로 취했다:
`y = 6/10/14 cm → |x| ≤ 0.24`, `18 cm → 0.20`, `22 cm → 0.16`, `26 cm → 0.08`.

**base 모드**(`PickCubeDRBaseEventCfg`)는 bell 대신 사각형만 쓴다:
`x ∈ (-0.14, 0.06)`, `y ∈ (0.205, 0.305)` (책상 왼쪽끝 X 30–50 cm · 앞모서리 Y 25–35 cm).
그 외 이벤트는 full 모드를 그대로 상속한다.

### 11.3 배치 샘플러 파라미터

`_make_randomize_cubes` 가 `randomize_cubes_scattered` 에 넘기는 값:

| 인자 | full | base |
|---|---|---|
| `x_range` / `y_range` | `(-0.24, 0.24)` / `(0.06, 0.26)` | `(-0.14, 0.06)` / `(0.205, 0.305)` |
| `x_halfwidth_by_y` | `CUBE_SCATTER_BELL` | `None` (사각형) |
| `full_orient` | `True` | `True` |
| `volume_inset` | `0.0` | `0.0` |
| `min_cube_sep` | `0.060` | `0.060` |
| `min_bowl_sep` | `0.14` | `0.14` |
| `min_base_sep` | `0.123` | `0.123` |
| `base_sep_offset_xy` | `(-0.021, 0.023)` | 동일 |
| `x_exclude_box` | `CUBE_ARM_EXCLUDE` | 동일 |
| `cube_sizes` | `[0.040]` | 동일 |

샘플러 기본값(미지정 시): `yaw_range_deg=(-30.0, 30.0)`(단 `full_orient=True` 면 미사용),
`max_attempts=50`, `cube_sep_margin=0.005`, `z_range=(0.0, 0.0)`, `num_active=None`.

구현 사실(`src/sim_to_real/utils/domain_randomization.py`):

- 순차 rejection sampling
- 그릇 기준점 = **post-DR `root_pos_w`** (§11.1)
- base 기준점 = `default_root_state[:, :2] + base_sep_offset_xy`
- 큐브쌍 이격 = `r_i + r_j + margin` (`r = s·√2/2`)
- 그릇 이격 = `min_bowl_sep + (r_i − r_40mm)` — 큐브가 크면 자동으로 더 멀어진다
- `full_orient` = **6 이산 stable face × uniform yaw** (uniform SO(3) 는 폐기됨)
- `num_active` 초과 큐브는 `z = -1.0` 으로 파킹
- `max_attempts` 실패 시 default 좌표 fallback
- **크기 DR z 보정**: `final_z += (env.cube_size_m[name] − cube_sizes[i]) / 2`.
  authored z 는 nominal 반높이 기준이라, 스케일된 env 는 그만큼 내려앉혀야 한다
  (안 하면 작은 큐브가 공중에서 떨어져 튀며 DR 이 정한 xy 를 벗어난다)

### 11.4 sweep 타깃 생성

`spawn_area.py::sweep_targets(nx=15, ny=8, boundary_n=20, inset=0.006, dedup_r=0.010)` 가
`(x, y, kind)` 목록을 낸다. kind = `bell`·`yedge`·`base_arc`·`bowl_arc`·`exclude_edge`·`interior`.
경계 타깃을 먼저 넣고 interior 를 dedup 로 얹어, **최외곽/최내곽 경계면이 항상 평가에 포함**된다.

self-check: `python3 src/sim_to_real/tasks/pick_cube/spawn_area.py` — 마스크 5케이스 +
타깃 불변식 assert.

### 11.5 robot color DR

`ROBOT_PLASTIC_COLORS` 팔레트 5색:

| 이름 | RGB |
|---|---|
| `purple` (기본) | `(0.40, 0.03, 0.75)` |
| `orange` | `(0.876, 0.317, 0.132)` |
| `teal` | `(0.0, 0.8, 0.502)` |
| `white` | `(0.95, 0.95, 0.95)` |
| `black` | `(0.08, 0.08, 0.08)` |

per-env robot **root** 에 OmniPBR 1개를 `strongerThanDescendants` 로 바인딩해 서브트리 전체를
한 색으로 override 한다(mesh 단위 재바인딩은 body 에 강하게 바인딩된 원본 material 을 못 이긴다).

두 가지 강한 제약이 있다. 근거 = `09_TACIT_KNOWLEDGE.md §5`.

- `scene.replicate_physics` 가 `True` 면 **`RuntimeError`** — Fabric replication 이 per-env
  material 을 무시한다.
- `mode="prestartup"` 이어야 한다. `reset` 이면 `__call__` 이 `scene.update` 뒤로 밀려
  view 무효 상태에서 크래시한다. ⇒ **env 당 색은 런 내내 고정**이며 리셋마다 재추첨할 수 없다.
  다양성은 env 수 + `--seed` 재추첨으로 얻는다.

`randomize_camera_focal` 기본 glob 3개: `/World/envs/env_.*/TopCamera`,
`.../Robot/gripper/WristCamera`, `.../Robot/shoulder/FrontCamera`.

### 11.6 큐브 크기 DR

`randomize_cube_scale(CUBE_NAMES, CUBE_SIZE_CHOICES, base_sizes)` — mode **`prestartup`**.
사다리·질량 규칙 = §9.1.1. env 마다 크기를 하나 뽑아 그 env 의 큐브 prim 에
`xformOp:scale = size/base` 를 걸고 `physics:mass` 를 `mass_for_size(size)` 로 덮어쓴다.
뽑힌 값은 `env.cube_size_m[name]` `(num_envs,)` 텐서로 남고, 이것이 **소비자의 유일한 통로**다:

| 소비자 | 쓰임 |
|---|---|
| `_randomize_cubes_scattered_fn` | spawn z 보정(§11.3) |
| `pickplace_sm.py::_cube_halves` | grasp 조준 z (`TABLE_TOP_BASE + half`) + planner 요청 `cube_half` |

⚠ 이 표 **밖의** 소비자(`datagen/state_machine/pick_cube.py`, `pink_ik_bridge_node.py` 등
legacy 경로)는 `CUBE_SIZES` authored nominal 을 가정한다 — 크기 DR 이 켜진 env 를 그 경로로
돌리면 조준이 조용히 어긋난다. 크기 DR 은 현재 cuRobo SM 경로에서만 배선돼 있다.

스톡 `mdp.randomize_rigid_body_scale` 을 쓰지 않는 이유 3가지(구현 주석과 동일):

1. 스톡은 연속 range 만 받는다 — 사다리는 5 mm 이산이다.
2. 스톡은 `xformOp:scale` 부재 시 `xformOpOrder` 를 `[translate, orient, scale]` 로 **덮어쓴다**.
   씬 큐브의 order 는 `[translate, rotateZ]` 라 authored rotateZ 가 사라진다. 자체 구현은
   `AddScaleOp()` 로 **append** 만 한다.
3. USD scale 은 `physics:mass` 를 건드리지 않아 25 mm 큐브가 40 mm 질량으로 남는다.

제약은 robot color DR(§11.5)과 같다: `replicate_physics=False` 필수(아니면 `RuntimeError`),
**env 당 크기는 런 내내 고정**(리셋 재추첨 불가) → 다양성은 env 수 × `--seed`.
그래서 `pickplace_sm` 의 state-only 경로는 시각 DR 만 끄고 크기 DR 은 남기며,
`replicate_physics` 를 이 이벤트 유무로 정한다.

---

## 12. 상수 대장

`scripts/contract/validate_spec_constants.py` 의 대조 대상이다. 값은 코드에서 AST 로 추출한
**Python repr 원문**이며, 이 표와 코드가 갈라지면 검증기가 실패한다.

> Python 리터럴 상수만 싣는다. `CUBE_SPECS`(dataclass 호출)·`SO101_FOLLOWER_CFG`(cfg 객체)
> 같은 비-리터럴 값은 §6·§9.1 에 별도 서술한다.

> `TABLE_TOP_BASE`(base_link 프레임 상판)와 `_DESK_TOP_WORLD_Z`(world 프레임 상판)는 **같은
> 물리량의 다른 프레임 표현**이다: `0.705(world) − 0.6749(robot base z) = 0.0301` 이고,
> 등재값 `0.0298` 은 큐브 collider 접촉 침투 0.34 mm 를 반영한 **실측 정착값**이다. 한쪽을
> 고치면 다른 쪽도 검토해야 한다 — 근거 = `09_TACIT_KNOWLEDGE.md` §14.

| 심볼 | 값 | 앵커 |
|---|---|---|
| `_ROBOT_POS` | `(0.0, 0.0, 0.6749)` | `src/sim_to_real/tasks/so101_base_env_cfg.py::_ROBOT_POS` |
| `_ROBOT_ROT` | `(0.0, 0.0, 0.0, 1.0)` | `src/sim_to_real/tasks/so101_base_env_cfg.py::_ROBOT_ROT` |
| `BOWL_SUCCESS_RADIUS` | `0.06` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::BOWL_SUCCESS_RADIUS` |
| `BOWL_HEIGHT_RANGE` | `(0.005, 0.12)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::BOWL_HEIGHT_RANGE` |
| `_DESK_TOP_WORLD_Z` | `0.705` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_DESK_TOP_WORLD_Z` |
| `_CUBE_Z_SLACK` | `0.001` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_Z_SLACK` |
| `_CUBE_LAYOUT` | `{'Cube1': (-0.015, 0.255, 0.0)}` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_LAYOUT` |
| `_MAT_BL_ENV` | `(-0.34, 0.045)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_MAT_BL_ENV` |
| `_CUBE_BASE_X_RANGE` | `(-0.14, 0.06)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_BASE_X_RANGE` |
| `_CUBE_BASE_Y_RANGE` | `(0.205, 0.305)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_BASE_Y_RANGE` |
| `_CUBE_VOLUME_INSET` | `0.0` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_VOLUME_INSET` |
| `_TOP_CAMERA_POS` | `(-0.17, 0.77, 1.05)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_TOP_CAMERA_POS` |
| `_TOP_CAMERA_ROT` | `(0.7538, 0.145, 0.1775, -0.6159)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_TOP_CAMERA_ROT` |
| `_TOP_CAMERA_FOCAL` | `19.0` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_TOP_CAMERA_FOCAL` |
| `_WRIST_CAM_LOCAL_POS` | `(0.0, 0.045, -0.04)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_WRIST_CAM_LOCAL_POS` |
| `_WRIST_CAM_LOCAL_ROT` | `(0.3562, -0.6108, 0.6108, 0.3562)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_WRIST_CAM_LOCAL_ROT` |
| `_WRIST_CAMERA_FOCAL` | `19.0` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_WRIST_CAMERA_FOCAL` |
| `_FRONT_CAM_LOCAL_POS` | `(-0.045, 0.0, 0.025)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_FRONT_CAM_LOCAL_POS` |
| `_FRONT_CAM_LOCAL_ROT` | `(0.0, 0.0, 1.0, 0.0)` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_FRONT_CAM_LOCAL_ROT` |
| `_FRONT_CAMERA_FOCAL` | `19.0` | `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_FRONT_CAMERA_FOCAL` |
| `CUBE_SCATTER_BELL` | `[(0.06, 0.24), (0.14, 0.24), (0.18, 0.2), (0.22, 0.16), (0.26, 0.08)]` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::CUBE_SCATTER_BELL` |
| `CUBE_SCATTER_X_RANGE` | `(-0.24, 0.24)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::CUBE_SCATTER_X_RANGE` |
| `CUBE_SCATTER_Y_RANGE` | `(0.06, 0.26)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::CUBE_SCATTER_Y_RANGE` |
| `CUBE_ARM_EXCLUDE` | `(-0.09, 0.04, -0.045, 0.155)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::CUBE_ARM_EXCLUDE` |
| `BOWL_CENTER_XY` | `(-0.22, 0.265)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::BOWL_CENTER_XY` |
| `MIN_BOWL_SEP` | `0.14` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::MIN_BOWL_SEP` |
| `BASE_XY` | `(0.0, 0.0)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::BASE_XY` |
| `PAN_AXIS_XY` | `(-0.021, 0.023)` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::PAN_AXIS_XY` |
| `MIN_BASE_SEP` | `0.123` | `src/sim_to_real/tasks/pick_cube/spawn_area.py::MIN_BASE_SEP` |
| `CUBE_SIZE_CHOICES` | `(0.025, 0.03, 0.035, 0.04)` | `src/sim_to_real/utils/cube_specs.py::CUBE_SIZE_CHOICES` |
| `DESK_TOP_Z` | `0.76` | `src/sim_to_real/tasks/common/mdp/_geometry.py::DESK_TOP_Z` |
| `JAW_GRASP_OFFSET` | `(-0.021, -0.07, 0.02)` | `src/sim_to_real/tasks/common/mdp/_geometry.py::JAW_GRASP_OFFSET` |
| `CONTAINER_DEFAULT_CENTER_XY` | `(0.36, 0.395)` | `src/sim_to_real/tasks/common/mdp/_geometry.py::CONTAINER_DEFAULT_CENTER_XY` |
| `SO101_JOINT_TARGET_MAX_VELOCITY` | `{'shoulder_pan': 5.0, 'shoulder_lift': 5.0, 'elbow_flex': 5.0, 'wrist_flex': 5.0, 'wrist_roll': 5.0, 'gripper': 5.0}` | `src/sim_to_real/tasks/common/utils.py::SO101_JOINT_TARGET_MAX_VELOCITY` |
| `CUBE_NAMES` | `['Cube1']` | `src/sim_to_real/utils/constant.py::CUBE_NAMES` |
| `BOWL_NAME` | `'Bowl'` | `src/sim_to_real/utils/constant.py::BOWL_NAME` |
| `BOWL_LOCAL` | `(-0.58, 0.26, 0.01)` | `scripts/environments/author_pick_cube_scene.py::BOWL_LOCAL` |
| `CONTACT_OFFSET_DEFAULT` | `0.004` | `scripts/environments/author_pick_cube_scene.py::CONTACT_OFFSET_DEFAULT` |
| `CUBE_CONTACT_OFFSET` | `0.002` | `scripts/environments/author_pick_cube_scene.py::CUBE_CONTACT_OFFSET` |
| `CUBE_ROUND_RADIUS_FRAC` | `0.22` | `scripts/environments/author_pick_cube_scene.py::CUBE_ROUND_RADIUS_FRAC` |
| `CUBE_ROUND_SEGS` | `10` | `scripts/environments/author_pick_cube_scene.py::CUBE_ROUND_SEGS` |
| `CUBE_COLLISION_SEGS` | `6` | `scripts/environments/author_pick_cube_scene.py::CUBE_COLLISION_SEGS` |
| `CUBE_FELT_ROUGHNESS` | `0.95` | `scripts/environments/author_pick_cube_scene.py::CUBE_FELT_ROUGHNESS` |
| `BOWL_MASS` | `0.25` | `scripts/environments/author_pick_cube_scene.py::BOWL_MASS` |
| `BOWL_R_BOTTOM` | `0.0325` | `scripts/environments/author_pick_cube_scene.py::BOWL_R_BOTTOM` |
| `BOWL_R_TOP` | `0.075` | `scripts/environments/author_pick_cube_scene.py::BOWL_R_TOP` |
| `BOWL_Z_BASE` | `0.005` | `scripts/environments/author_pick_cube_scene.py::BOWL_Z_BASE` |
| `BOWL_DEPTH` | `0.065` | `scripts/environments/author_pick_cube_scene.py::BOWL_DEPTH` |
| `BOWL_LATS` | `20` | `scripts/environments/author_pick_cube_scene.py::BOWL_LATS` |
| `BOWL_LONS` | `24` | `scripts/environments/author_pick_cube_scene.py::BOWL_LONS` |
| `BOWL_WALL_THICKNESS` | `0.004` | `scripts/environments/author_pick_cube_scene.py::BOWL_WALL_THICKNESS` |
| `BOWL_FLOOR_THICKNESS` | `0.003` | `scripts/environments/author_pick_cube_scene.py::BOWL_FLOOR_THICKNESS` |
| `BOWL_MAX_CONVEX_HULLS` | `64` | `scripts/environments/author_pick_cube_scene.py::BOWL_MAX_CONVEX_HULLS` |
| `BOWL_HULL_VERTEX_LIMIT` | `64` | `scripts/environments/author_pick_cube_scene.py::BOWL_HULL_VERTEX_LIMIT` |
| `BOWL_VOXEL_RESOLUTION` | `500000` | `scripts/environments/author_pick_cube_scene.py::BOWL_VOXEL_RESOLUTION` |
| `TABLE_TOP_BASE` | `0.0298` | `src/so101_contract/grasp_geometry.py::TABLE_TOP_BASE` |
| `GRASP_Z_OFF` | `0.0022` | `scripts/cuRobo/curobo_batch_planner.py::GRASP_Z_OFF` |
| `TABLE_MARGIN` | `0.004` | `scripts/cuRobo/curobo_batch_planner.py::TABLE_MARGIN` |

---

## 13. 씬 재생성

USD 6개(`scene.usd` + 객체 5개)는 `scripts/environments/author_pick_cube_scene.py` 로 일괄
재생성한다. 공식 `pxr`/`PhysxSchema` API 를 쓰므로 isaac 의존성 그룹이 필요하다:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python scripts/environments/author_pick_cube_scene.py
```

좌표를 바꾸면 `src/sim_to_real/assets/scenes/cube_desk.py` 의 `SCENE_OFFSET` 을 갱신하고
스크립트를 재실행한다. `BOWL_CENTER_XY` 같은 world-frame 상수는
`pick_cube_env_cfg.py`(→ `spawn_area.py`)와 동기화해야 한다.

---

## 참조

- 단위·관절 순서 → `04_IO_CONTRACT.md`
- 관측이 데이터셋에 어떻게 저장되는가 → `05_DATA_SPEC.md` §3, §6
- 환경을 실행하는 진입점·인자 → `08_PIPELINES.md`
- 왜 이 상수인가 / 함정 / 불일치 → `09_TACIT_KNOWLEDGE.md`
