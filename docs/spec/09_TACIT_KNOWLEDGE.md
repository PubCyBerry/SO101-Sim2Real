# 09. 암묵지 — 불변식 · 매직 상수 근거 · 함정

> 이 문서는 **"왜 이렇게 돼 있는가"** 전용이다. 다른 명세 문서가 *무엇인지*를 말한다면
> 여기는 *왜 그 값이고, 바꾸면 무엇이 깨지는지*를 말한다.

## 1. 읽는 법

각 항목은 4블록이다:

- **결정** — 코드가 실제로 하는 선택
- **근거** — 그렇게 한 이유(측정치가 있으면 숫자)
- **어기면** — 바꿨을 때 관측된 실패
- **앵커** — `경로::심볼`

`src`+`scripts`+`docker` 에서 `⚠ / 주의 / 함정 / 반드시 / 금지 / 절대 / ★ / HACK / WORKAROUND`
마커가 **25개 파일 79곳** 있다(§10 인덱스). 이 문서는 그중 시스템 동작에 영향을 주는 것을
주제별로 정리한 것이다.

---

## 2. 물리 · grasp

### 2.1 큐브 collider 는 convexHull — SDF 를 쓰면 안 된다

- **결정**: 큐브 충돌 = **convexHull**. SDF 는 오목 형상(그릇, jaw/gripper)에만 쓴다.
- **근거**: 큐브는 볼록이라 convexHull 이 라운드 표면을 정확히 표현한다. SDF 는 평평한
  책상 접촉에서 normal 이 매 step 뒤집혀 큐브가 제자리 회전 버즈("덜그럭")를 낸다.
- **어기면** (2026-06-22 실측): jitter **2.9 rad/s** → convexHull 로 **0.056 rad/s**(50배 감소).
  그 불안정이 grasp 도 망가뜨렸다 — 고정 spawn SM **3/16** → convexHull 복원 후 **13/16 (81%)**.
- **앵커**: `scripts/environments/author_pick_cube_scene.py::CUBE_COLLISION_SEGS` 주변 주석.
  측정 도구는 `scripts/test/measure_cube_jitter.py`(현재 트리에 없음).

### 2.2 contact offset 은 형상별로 다르다

- **결정**: 정적·두꺼운 면(책상/매트/그릇) `0.004`, **grasp 대상 큐브 `0.002`**.
- **근거**: convexHull 접촉이 안정적이라 큐브만 좁은 margin 이 가능하다.
- **앵커**: `scripts/environments/author_pick_cube_scene.py::CONTACT_OFFSET_DEFAULT`,
  `::CUBE_CONTACT_OFFSET`

### 2.3 그릇은 convexDecomposition — SDF 불가

- **결정**: 그릇 충돌 = convexDecomposition (`maxConvexHulls=64`, `voxelResolution=500000`).
- **근거**: SDF triangle mesh 는 `num_envs > 1` 에서 per-instance cooking 비용이 크고 불안정
  (crash)하다. Isaac Lab RL 표준은 convex 계열이다. watertight 두께 shell 을 여러 convex hull 로
  분해하되 shrinkWrap + 충분한 hull 수로 오목 캐비티를 보존해 큐브가 바닥까지 가라앉게 한다.
- **앵커**: `scripts/environments/author_pick_cube_scene.py::BOWL_MAX_CONVEX_HULLS`

### 2.4 큐브 질량은 부피비례가 아니라 쉘비례

- **결정**: 40 mm = 35 g, 50 mm = 55 g.
- **근거**: 의자다리 커버 폼이라 속이 비어 있다. 표면적(변²) 비례:
  `35 × (50/40)² ≈ 54.7 → 55 g`.
- **앵커**: `src/sim_to_real/utils/cube_specs.py::CUBE_SPECS`

### 2.5 물리 상수는 임의 변경 금지

- **결정**: `author_pick_cube_scene.py` 의 물리 상수 블록은 grasp 검증을 거친 값이다.
- **어기면**: grasp 성공률이 회귀한다(§2.1 사례).
- **앵커**: `scripts/environments/author_pick_cube_scene.py` — "물리 상수 (임의 변경 금지)" 블록.

> 코드 주석은 이 근거를 `docs/GRASP_PHYSICS.md` 로 참조하지만 **그 파일은 없다**(§9 INC-11).
> 근거는 이 절이 대체한다.

---

## 3. 기구학 · 도달성

### 3.1 ★ min-reach 가드의 중심은 pan 축이지 마운트 원점이 아니다

- **결정**: 큐브 스폰의 base 이격 판정 중심 = `PAN_AXIS_XY = (-0.021, 0.023)`(env-local).
  `BASE_XY = (0.0, 0.0)`(마운트 원점)은 **plot 마커·meta 전용**이다.
- **근거**: URDF `shoulder_pan` origin 이 `base_link` 기준 `(0.0388, 0, 0.0624)` m 다.
  USD→env 프레임(`Rz90 + BASE_T`, env = `Rz180 · base_link`) 변환을 거치면 팔의 실제 회전
  중심이 마운트 원점에서 **−x 2.1 cm · +y 2.3 cm** 어긋난다.
- **어기면**: 원점 기준 가드는 도달 불가한 corner 를 통과시킨다. 예 — env `(-0.092, 0.107)` 은
  마운트 거리 `0.141 > 0.135` 로 통과하지만 pan 축 거리는 `0.109` 라 IK 가 풀리지 않는다.
  이것이 재-sweep 실패의 근본 원인이었다. 수정 후 `base_arc` 성공률 **68% → 100%**,
  평가 셀이 187 → 183 으로 줄었다(도달 불가 corner 자동 배제).
- **앵커**: `src/sim_to_real/tasks/pick_cube/spawn_area.py::PAN_AXIS_XY`,
  `::MIN_BASE_SEP`. 정량 결과 = `08_PIPELINES.md §5.7`.

> sweep map 에서 base 마커가 우측으로 치우쳐 보이는 것은 **물리적으로 정상**이다
> (마커 = 마운트 원점, 판정 박스 = pan 축).

### 3.2 URDF base 가 USD base 보다 z 축 90° 어긋나 있다

- **결정**: pink IK 는 `--base-yaw-deg 90`, cuRobo planner 는 `BASE_YAW = 90.0` +
  `BASE_T = (0.01576, -0.02079, -0.03248)` 로 보정한다.
- **어기면**: pan 이 97° 빗나가 전혀 다른 곳을 집는다.
- **앵커**: `scripts/cuRobo/curobo_batch_planner.py::BASE_YAW`, `::BASE_T` ·
  `scripts/datagen/pink_ik_bridge_node.py` `--base-yaw-deg`

### 3.3 이중 FK 2.79° 피치 차 — TCP 회전으로 흡수

- **결정**: `tcp_grasp` quaternion = `Ry(π − 0.0486795)`. 상수 `TCP_TWIST_RY = -0.0486795`.
- **근거**: USD 체인에 URDF `wrist_roll` origin 의 `Ry(0.0487)` 항이 없다. 그래서 모델 tool 이
  sim 실제 대비 **2.79° 피치** 어긋난다. 전 자세에서 상수이므로 TCP 회전에 흡수하면 정확히
  보정된다.
- **앵커**: `assets/robots/so101.yml` `kinematics.extra_links.tcp_grasp`,
  `scripts/cuRobo/curobo_batch_planner.py::TCP_TWIST_RY`

### 3.4 5-DOF 는 position 우선 · orientation best-effort

- **결정**: SO-101 은 팔 5축(+그리퍼)이라 임의 6-DOF pose 를 만족할 수 없다.
  **새 IK 경로에 orientation 을 hard constraint 로 넣지 않는다.**
- **어기면**: 해가 없어 IK 가 전면 실패한다.
- **앵커**: cuRobo planner 의 `ALPHA_SCAN_DEG` 양방향 스캔 · pink 의 `--ori-cost` soft weight.

### 3.5 attached_object 는 tcp_grasp 와 transform 이 같아야 한다

- **결정**: `so101.yml` 의 `attached_object.fixed_transform` = `tcp_grasp.fixed_transform`.
- **근거**: cuRobo `AttachmentManager.update()` 가 sphere offset 을 `tool_frames[0]`(=`tcp_grasp`)
  프레임으로 계산해 이 링크 슬롯에 기록하고, FK 는 이 링크 프레임으로 배치한다.
- **어기면**: 부착된 큐브 blob 이 `(0, −0.015, −0.025) + Ry(π)` 만큼 오배치된다.
- **앵커**: `assets/robots/so101.yml` `kinematics.extra_links.attached_object`

### 3.6 책상은 cuRobo obstacle 이 아니다

- **결정**: planner world 에 책상을 넣지 않는다.
- **근거**: base collision sphere 가 상판 안쪽에 있어 모든 plan 이 start-collision 이 된다.
- **앵커**: `scripts/cuRobo/curobo_batch_planner.py` world 구성부.

---

## 4. 제어

### 4.1 ⚠ arm slew 2.5 rad/s 하드캡은 금물

- **결정**: arm `5.0` rad/s 유지, **gripper 만 `2.5`**.
- **근거**: cuRobo batch 는 lock-step(고정 step 수)으로 sparse plan 을 따라간다. arm 을 2.5 로
  묶으면 transit·descend 를 정해진 step 안에 끝내지 못하고 lag 가 쌓인다.
- **어기면** (동일 seed·DR layout 측정): all-4 성공률 **90.6% → 59.4%**. 게다가 생성된 데이터는
  arm 5.0 에서도 이미 within-task max ≈ 2.5 rad/s 라 추가 cap 은 **이득 0 · 컨트롤러 파손**뿐이다.
- **앵커**: `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_PICKCUBE_JOINT_MAX_VELOCITY`

gripper 를 2.5 로 낮춘 이유는 별개다 — 닫을 때 명령 속도를 줄여 큐브를 튕겨내지 않게 하려는
것(grasp valley 완화). 공유 상수 `SO101_JOINT_TARGET_MAX_VELOCITY` 는 건드리지 않는다.

### 4.2 gripper effort 는 런타임이 actuator 설정을 무력화한다

- **결정**: actuator `effort_limit_sim = 30` 이지만 `PickCubeEnv.step()` 이 매 step
  `clamp(nearest_mass / 0.15, 0.5, 10.0)` 로 gripper joint 를 덮어쓴다.
- **근거**: 그리퍼가 작은 물체를 convex-decomposition 충돌 hull 안으로 과구동하는 것을 막으면서
  들어올릴 힘은 남긴다(leisaac teleop/replay 동작 이식).
- **실효값**: Cube1 `0.035 kg` → `0.233` → 하한 클램프 ⇒ **0.5**. 그래서 bridge 는 static
  `GRIPPER_EFFORT_LIMIT = 0.5` 로 같은 물리를 재현한다.
- **앵커**: `src/sim_to_real/utils/gripper_effort.py::dynamic_reset_gripper_effort_limit_sim` ·
  `src/sim_to_real/assets/robots/lerobot.py` actuator 주석

### 4.3 기록되는 action 은 slew 통과 후 target 이어야 한다

- **결정**: recorder 는 `arm.processed_actions`(slew·offset 적용 후)를 기록한다.
  pre-slew raw action 은 기록하지 않는다.
- **근거**: raw command 는 물리적으로 불가능한 teleport 를 포함한다(arm 239°/step, cap 9.55°의
  25배). 그것을 BC target 으로 쓰면 학습 데이터가 jerky 해진다.
- **앵커**: `src/sim_to_real/tasks/common/mdp/recorders.py::DatagenRecorderTerm.record_post_step` ·
  `src/sim_to_real/data/lerobot_recorder_manager.py`

### 4.4 offset 전면 제거 — action 은 절대 joint target

- **결정**: `use_default_offset=False`. rad-space offset 0. 그리퍼 codec 은 affine 단독.
- **근거**: sim·실기기·bridge 가 같은 수를 같은 뜻으로 쓰게 하려는 VLA-only 리팩토링.
  이전의 31.75 배수 offset 은 제거됐다.
- **앵커**: `src/sim_to_real/tasks/so101_base_env_cfg.py::SO101ActionsCfg` ·
  `04_IO_CONTRACT.md §2.2`

### 4.5 bridge 는 Python slew 를 걸 수 없다

- **결정**: OmniGraph 로 raw position 을 주입하므로 slew 를 **joint 최대속도 상한**
  (arm 5.0 / gripper 2.5)으로 대신 강제한다.
- **앵커**: `scripts/inference/run_cube_desk_ros_bridge.py::ARM_MAX_JOINT_VEL`, `::GRIPPER_MAX_JOINT_VEL`

`vla_policy_node` 의 `command_slew_limit` 파라미터는 배포 경로에서 학습 env 의 slew 를
재현하는 **옵션**이다(기본 off) — 데이터 action 은 slew-limited 지만 정책의 OOD 예측 점프까지
보장되지는 않기 때문이다.

---

## 5. 씬 · 도메인 랜덤화

### 5.1 ★ EventCfg 선언 순서 = 적용 순서

- **결정**: `randomize_bowl` 을 `randomize_cubes` **앞에** 선언한다.
- **근거**: `EventManager` 가 `cfg.__dict__` 순서로 적용한다. `write_root_pose_to_sim` 이
  물리 step 없이 `root_pose_w` 를 즉시 갱신하므로, 큐브 배치의 `min_bowl_sep` rejection 이
  **arc 이동 후 실제 그릇 좌표**를 본다.
- **어기면**: 큐브가 nominal 그릇 기준으로 배치된 뒤 그릇이 움직여 사후에 불변식이 깨진다 —
  cube-bowl 거리 `0.126 < 0.14` 로 스폰되어 transit 계획이 실패한다(64 env 중 1개 재현).
- **앵커**: `src/sim_to_real/utils/domain_randomization.py::randomize_cubes_scattered` docstring ·
  `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::PickCubeDREventCfg`

> `spawn_area.in_spawn_area()` 는 **nominal 그릇** 기준이라 arc DR 된 env 에서는 runtime 판정과
> 발산한다(runtime 이 더 엄격). sweep/plot 은 그릇을 nominal 고정하므로 정합한다.

### 5.2 조명은 `/World` 에 둔다 — env 안에 두면 N배 과노출

- **결정**: `/World/Light`(dome) · `/World/KeyLight`(distant) 각 1개만 author. scene USD 에서
  광원을 뺐다.
- **근거**: USD 광원은 scope 격리가 없어 `{ENV_REGEX_NS}/Scene` 안에 두면 env 수만큼 복제된다
  (IsaacLab #4340 / #1729).
- **어기면**: `num_envs` 에 비례해 과노출.
- **앵커**: `src/sim_to_real/tasks/so101_base_env_cfg.py::SO101BaseSceneCfg`

### 5.3 robot color DR — Replicator + prestartup + replicate_physics=False

세 제약이 한 묶음이다.

- **Fabric**: `scene.replicate_physics=True` 면 per-env material 이 무시돼 전 로봇이 env_0 색으로
  렌더된다 ⇒ 코드가 `RuntimeError` 로 막는다.
- **root 바인딩**: mesh 단위 재바인딩은 body 처럼 링크에 강하게 바인딩된 원본 material 을
  못 이긴다(실측: 관절만 바뀜). robot **root** 에 OmniPBR 를 `strongerThanDescendants` 로
  override 해야 전 서브트리가 바뀐다.
- **`mode="prestartup"`**: `__init__` 이 robot subtree 를 de-instance(구조 변경)해 physx tensor
  view 를 무효화한다. prestartup 은 `scene.update` **전에** apply 되며 그 Replicator op 이 view 를
  리프레시한다. `reset` 모드면 `__call__` 이 `scene.update` 뒤로 밀려 view 무효 상태에서
  `get_dof_velocities` 가 크래시한다("Simulation view invalidated").
- **귀결**: **env 당 색은 런 내내 고정**이다. 리셋마다 재추첨할 수 없다. 다양성은 env 수 +
  `--seed` 재추첨으로 얻는다.
- **앵커**: `src/sim_to_real/utils/domain_randomization.py::randomize_robot_color`,
  `::ROBOT_PLASTIC_COLORS`

> state-only headless 실행은 시각 DR 을 제거하고 `replicate_physics=True` 를 복원해야 한다 —
> 안 하면 `get_dof_velocities` 빌드 크래시가 난다. cuRobo SM 의 non-record 경로가 그렇게 한다
> (`08_PIPELINES.md §5.4`).

### 5.4 `full_orient` 는 uniform SO(3) 가 아니다

- **결정**: **6 이산 stable face × uniform yaw**.
- **근거**: uniform SO(3) 샘플은 큐브가 모서리로 서는 비현실적 자세를 만든다(폐기됨).
- **앵커**: `src/sim_to_real/utils/domain_randomization.py::_randomize_cubes_scattered_fn`

### 5.5 그릇 arc 범위는 기하 계산 결과다

- `-4°` — 매트 왼쪽 경계(world x = −0.34): `-0.22 + 0.44·sin(a) ≥ -0.255` → `a ≥ -4.56°`
- `+8°` — 그릇-Cube3 겹침: 유효 충돌 반경
  `r_top 0.075 + Cube3 half-diag 0.0354 + cube offset 0.002 + bowl offset 0.004 = 0.1164 m`,
  Cube3 최악 위치 `(-0.05, 0.235)` 기준 임계 `9.48°` → 여유 포함 8°
- **앵커**: `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::PickCubeDREventCfg` 주석

### 5.6 contact 센서 필터는 클래스 밖 상수여야 한다

- **결정**: `_CUBE_CONTACT_FILTER` 를 모듈 상수로 둔다.
- **근거**: 클래스 속성이면 `InteractiveScene` 이 asset 으로 오인한다.
- **앵커**: `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::_CUBE_CONTACT_FILTER`

---

## 6. 부팅 · 플랫폼

### 6.1 ⚠ AppLauncher 에는 화이트리스트 키만 넘긴다

- **결정**: `vars(args)` 통째가 아니라 `_LAUNCHER_KEYS` 로 필터해 전달한다.
  키 = `headless, livestream, enable_cameras, device, kit_args, experience, rendering_mode`.
- **근거**: 커스텀 인자(`--view_eye`/`--layout`/`--eval`/`--num_cubes` …)가 AppLauncher 의
  UI/viewport 초기화(`_prepare_ui`)를 깨뜨린다.
- **어기면**: Windows 에서 **access violation**, Linux 에서는 **livestream viewport docking 이
  조용히 실패**한다(3-cam 레이아웃 미적용). 후자가 더 위험하다 — 에러 없이 잘못된다.
- **부수**: C-레벨 크래시 추적용 `faulthandler.enable(file=...)` 를 부팅 전에 켠다
  (`outputs/bridge_faulthandler.txt` — isaac-sim healthcheck 도 이 파일을 본다).
- **앵커**: `scripts/inference/run_cube_desk_ros_bridge.py::_LAUNCHER_KEYS` ·
  `scripts/cuRobo/pickplace_sm.py` 동일 패턴

### 6.2 headless 에서도 `enable_cameras=True` 가 필요하다

- **결정**: bridge 는 `args.enable_cameras = True` 를 강제한다.
- **근거**: 기본 headless experience(`isaaclab.python.headless.kit`)는 OmniGraph USD 그래프
  생성을 strip 해 `"Unable to create prim for graph"` 로 실패한다. `enable_cameras=True` 면
  AppLauncher 가 `isaaclab.python.headless.rendering.kit`(풀 렌더 + OmniGraph USD authoring)을
  로드한다. **카메라 자체는 쓰지 않는다 — 렌더 experience 만 필요하다.**
- **앵커**: `scripts/inference/run_cube_desk_ros_bridge.py`

### 6.3 `--enable_cameras` 와 `remove_pick_cube_cameras` 는 짝이다

- **결정**: PickCube env 는 static 카메라가 있어 `--enable_cameras` 를 요구한다. 무카메라
  실행은 `remove_pick_cube_cameras(env_cfg)` 로 **카메라와 images 관측을 함께** 제거한다.
- **어기면**: scene 만 지우면 없는 sensor 를 참조하는 `images` obs 가 에러를 낸다.
- **앵커**: `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py::remove_pick_cube_cameras`

### 6.4 `IMAGEIO_FFMPEG_EXE` 는 import 전에 주입해야 한다

- **근거**: imageio-ffmpeg 번들 바이너리는 NVENC 없이 빌드된 경우가 많다. 시스템 FFmpeg 를
  명시하면 실제 encoder 목록을 쓸 수 있다.
- **앵커**: `src/sim_to_real/data/lerobot_recorder.py::LeRobotV3DatasetWriter._open_video_writers`

### 6.5 IsaacLab `TerminationManager` 패치는 버전 게이트가 있다

- **결정**: `import sim_to_real` 시 `monkey_patch()` 를 best-effort 적용한다.
  현재 `compute` 소스에 per-term 기록 라인(`_term_dones[:, i]`)이 있으면 **스킵**한다.
- **근거**: IsaacLab 의 `TerminationManager.compute()` 가 `_term_dones` 를 term 마다 갱신하지
  못해 한 번 True 가 된 값이 False 로 돌아가지 못하던 버그(IsaacLab commit `f498245` 에서 수정).
  redundant override 로 공식 수정과 divergence 하는 것을 막으려고 게이트를 뒀다.
- **앵커**: `src/sim_to_real/utils/monkey_patch.py`

### 6.6 USD export 는 상대 asset path 를 절대경로로 re-anchor 한다

- **결정**: export 직후 절대 asset path 를 레이어 기준 상대경로로 되돌린다. 상대화 불가한
  절대경로가 남으면 author 시점에 `RuntimeError`(트립와이어).
- **근거**: 일부 플랫폼(특히 Linux)에서 USD layer Export 가 작성한 상대경로를 author 머신의
  절대경로로 바꾼다. 그 경로는 다른 머신·컨테이너에서 깨진다.
- **어기면**: 큐브가 검게 렌더된다(텍스처 로드 실패).
- **앵커**: `scripts/environments/author_pick_cube_scene.py` layer export 후처리.

---

## 7. 의존성 · 설치

### 7.1 `packaging==23.0` 은 **정확 핀**이어야 한다

- **근거**: cuRobo 설치가 packaging 을 25.0 으로 올리면 `_structures` 제거로 Isaac 번들 torch 가
  즉사한다(`_vendor` alias).
- **어기면**: `<26` 같은 **범위 핀은 무효**다 — 25.0 이 통과한다.
- **앵커**: `docker/Dockerfile.cuRobo`

### 7.2 nvrtc 헤더·CCCL 은 별도 설치

- prebundle nvrtc 는 lib-only 라 `nvrtc.h` 가 없다 → `--force-reinstall --no-deps
  nvidia-cuda-nvrtc-cu12==12.8.93`
- `[cu12]` extra 가 CCCL 헤더를 제외한다 → `nvidia-cuda-cccl-cu12` 별도
- nvcc 는 불필요(NVRTC JIT)
- **앵커**: `docker/Dockerfile.cuRobo`

### 7.3 `lerobot_v060_eef_relative_patch.py` 는 삭제 금지 · 트립와이어

- **결정**: `lerobot[smolvla,async,groot]==0.6.0` 설치 직후 1회 실행. PyPI LeRobot 0.6.0
  source 에 공통 SE(3) processor, train/checkpoint manifest, full-chunk sync/async hook 을
  site-packages 에서 **멱등** 적용한다.
- **어기면**: 세 정책(ACT·SmolVLA·GR00T-N1.7) 어느 것도 schema v2 manifest 를 만들지 못하고
  full-chunk postprocess 계약이 깨진다. 예상 upstream source 형태가 다르면 빌드를 중단한다
  — **lerobot·transformers 업그레이드 시 이 패치부터 점검**하라는 신호다.
- **앵커**: `docker/lerobot_v060_eef_relative_patch.py` · 계약 정본
  `docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`

> **legacy**: `docker/groot_compat_patch.py` 는 LeRobot 0.5.1 + GR00T-N1.5 재현용 자료다.
> transformers 5.3 + torch 2.10 에서 0.5.1 의 N1.5 wrapper 가 깨지는 4지점을 패치했다.
> **현재 `Dockerfile.policy` 는 실행하지 않는다.** 이력 보존을 위해 삭제하지 않는다.

### 7.4 ABI 핀 — `uv lock --upgrade` 금지

핀 목록과 "어기면" 은 `06_RUNTIME_SPEC.md §7.2` 에 있다. 요약하면 numpy 1.26 · pyarrow<19 ·
h5py<3.16 · torchcodec 0.5.x · torch 2.7.0+cu128 이 Isaac Sim 5.1 번들과 ABI 로 묶여 있다.

### 7.5 ★ cuRobo v2 in-process 가 되는 이유 (2026-07-01 오진 정정)

- **결정**: Mimic/SkillGen 증강 생성(`generate_mimic_dataset.py`)은 **cuRobo v0.8 in-process** 로
  구동한다. 공식 isaaclab 은 cuRobo v0.7 까지만 지원했다.
- **진범**: 2026-07-01 에 "cuRobo v0.8 + Isaac in-process = cuda 컨텍스트 공존 불가 → 2-프로세스 확정"
  이라고 판정했는데, 이는 **오진**이었다. 실제 원인은 **warp shadowing 한 줄**이다.
- **정정**: `import warp as _warp` 를 `AppLauncher` 보다 **먼저**(파일 맨 앞)에 두면 kit 확장
  `omni.warp.core` 1.8.2 대신 이미지 site-packages 의 warp 1.11 이 `sys.modules` 를 선점한다.
  그러면 cuRobo v0.8 커널이 정상 임포트된다.
- **증상 (없음 시)**: `TypeError: func() got an unexpected keyword argument 'module'` —
  cuRobo 가 warp 1.8.2 API 를 기대하는데 warp 1.11 이 `module` 파라미터를 모른다.
- **어기면**: 2-프로세스 오버헤드 + IPC latency + 복잡한 protocol. 단일 process 로 단순화되면서
  3-cam 캡처·subtask 신호·생성기 루프가 동기화된다.
- **앵커**: `scripts/datagen/generate_mimic_dataset.py::import warp` (파일 첫 줄) ·
  `scripts/datagen/annotate_mimic_demos.py::import warp` (파일 첫 줄)

### 7.6 cuRobo v1 vs v2 결정 기록 — 공식 v1 아카이브

- **결정**: feat/skillgen-datagen 브랜치는 공식 Isaac Lab 2.3.2 SkillGen + cuRobo **v1**(commit
  `ebb71702`, 공식 검증 commit)을 별도 이미지(`Dockerfile.skillgen`)에 설치했다. 우리 정본은
  **v2 in-process** 로 택했다.
- **이유** (성공률 아님 — 백엔드·베이스·파이프라인이 달라 비등가):
  ① **IPC·별도 이미지 불필요**: in-process 로 매 trial 초기화가 없고, planner 단일 기동.
  ② **planner 백엔드 동일화**: main SM(`curobo_batch_planner.py`)과 같은 v0.8 을 써서
     `grasp_manifold.py`·`curobo_frames.py` **단일 소스 가능**. v1 은 API 가 달라 별개로 관리했다.
- **v1 사용 예약**: 공식 추천이 있으면 재검토. 현재는 아카이브(branch `feat/skillgen-datagen` 보존).
- **앵커**: `scripts/datagen/generate_mimic_dataset.py::SO101SkillGenPlanner` 클래스 주석

### 7.7 cuRobo main planner(§5) 와 Mimic planner(§5.9) 는 같은 객체

- **결정**: Mimic 생성기가 쓰는 `SO101SkillGenPlanner` 는 `scripts/cuRobo/curobo_batch_planner.py`
  의 같은 `SO101CuRoboPlanner` 로 부터 상속한다.
- **근거**: grasp 기하·manifold·계획 세팅이 둘 다 동일해야 source 와 증강이 **같은 수준의
  도달 가능성**을 갖는다. 따로 만들면 수식 드리프트 위험.
- **앵커**: `src/sim_to_real/datagen/skillgen_planner.py::SO101SkillGenPlanner`

---

## 8. 안전

### 8.1 ⚠ pickle payload — 역직렬화 RCE

policy-server gRPC 페이로드가 pickle 이고 전 서비스가 `network_mode: host` 다.
신뢰할 수 없는 네트워크에 노출하지 말 것. `.env.example`·compose 주석에 CVE 경고가 있다.
상세 = `07_INTERFACES.md §8.4`.

### 8.2 ⚠ 실기기 replay 는 충돌 위험

`scripts/convert/sim_dataset_to_real_follower.py` 로 변환한 데이터를 실기기에서 replay 할 때
잘못된 관절 타깃은 충돌을 낸다. **e-stop 준비 후 실행**한다.

### 8.3 데이터셋 디렉터리 삭제 안전장치

`LeRobotV3DatasetWriter(overwrite=True)` 는 `/`·홈·cwd·`len(parts) < 4` 인 경로 삭제를 거부한다.
`--record_lerobot` 는 기존 디렉터리를 **덮어쓴다**.

### 8.4 follower calibration 재현오차

pan·wrist_roll 은 손맞춤 스냅샷 간 약 5–6° 차이가 난다. reach probe 잔차가 크면 그 두 축부터
재수집한다. `04_IO_CONTRACT.md §4.2`.

---

## 9. 불일치 대장

원문 대조로 확인한 불일치다. **이 작업은 문서화 범위이므로 코드는 수정하지 않았다.**

| ID | 종류 | 주장 (위치) | 코드 실제 (앵커) | 영향 | 조치 |
|---|---|---|---|---|---|
| INC-01 | 문서↔코드 | `min_base_sep 0.135` — `docs/PINK_IK_PICKPLACE.md` §11 | **`0.123`** — `spawn_area.py::MIN_BASE_SEP` | 없음(설명만) | 기록. `03_ENV_SPEC.md §11.2` 가 정본 |
| INC-02 | 문서↔코드 | `TRANSIT_Z = 0.25` — `scripts/cuRobo/README.md` | **`0.21`** — `curobo_batch_planner.py::TRANSIT_Z` | 없음 | **문서 정정**(README 1줄) |
| INC-03 | 문서↔코드 | "서비스 4종" — `AGENTS.md` | **5개** — `docker/docker-compose.yaml` (`curobo-datagen` 누락) | 신규 작업자가 planner 서비스를 모른다 | **문서 정정**. `06_RUNTIME_SPEC.md §1` 이 정본 |
| INC-04 | 문서 누락 | `scripts/cuRobo/` 가 `AGENTS.md` 배치 규약·스크립트 표에 없음 | 4파일 3,257줄(현행 주력 datagen) | 배치 규약이 실제와 다름 | **문서 정정**(charter 1행 추가) |
| INC-05 | 문서↔코드 | `docs/PINK_IK_PICKPLACE.md` §5 = `7 waypoint + retreat` | 실제 `pan_align→pre_grasp→descend→approach→grasp→lift→transit→release→home`(**retreat 제거**) | 문서대로 읽으면 코드를 못 찾음 | **기록 + 스테일 배너**. 현행 = `08_PIPELINES.md §6` |
| INC-06 | 문서↔코드 | `--ori-cost` 기본 `0.5`, 인자 6개 누락 — `PINK_IK_PICKPLACE.md` §8 | **`1.0`** — `pink_ik_bridge_node.py` | 재현 실패 | **기록 + 배너**. `08_PIPELINES.md §6` 이 정본 |
| INC-07 | 문서↔코드 | 그릇 "8밴드×24 panel" — `AGENTS.md` | `BOWL_LATS = 20`, `BOWL_LONS = 24` | 없음 | **문서 정정** |
| INC-08 | 문서↔코드 | 큐브 영역 `y ∈ [0.22, 0.26]` — `AGENTS.md` | 현행 bell `y ∈ [0.06, 0.26]` | 스폰 범위 오해 | **문서 정정** |
| INC-09 | 문서 불완전 | `.env` 표 §0–§7 (8행) — `README.md` | `.env.example` §0–§8, **69변수** | §8 sim teleop 변수 누락 | **문서 정정**. `06_RUNTIME_SPEC.md §5.2` 가 전수 |
| **INC-10** | **코드↔코드** | — | `_geometry.py::DESK_TOP_Z = 0.76`(pen 잔재)을 `task_done`·`cube_lost`·`object_in_container` 가 사용. 실 상판 = `_DESK_TOP_WORLD_Z = 0.705` | **성공 종료가 발화 불가한 구조** — 아래 상세 | **코드 결함(별건)**. 기록·경고만 |
| INC-11 | 끊어진 참조 | `docs/GRASP_PHYSICS.md`(`author_pick_cube_scene.py`) · `docs/SIM_REAL_INFERENCE_PARITY.md`(`SIM_REAL_REPLAY_CALIBRATION.md`) · `docs/PATH_E_CUMOTION_PICKPLACE.md`(`TROUBLESHOOTING.md`) · `docs/pics/펜통_펜_배치_1.jpg` | 4건 모두 파일 없음 | 근거 추적 불가 | **기록만**. `GRASP_PHYSICS` 내용은 **§2 가 대체** |
| INC-12 | 코드↔코드 | `NUM_CUBES` 기본값 3곳 상이 | compose `1` · `isaac-sim-entrypoint.sh` `4` · bridge argparse `4` | compose 경유 시 1이 이김 | 기록. `06_RUNTIME_SPEC.md §5.3` |
| INC-13 | 코드↔주석 | bridge 주석 "PickCubeEnvCfg actuator stiffness = 17.8" | 현 env 는 per-joint 55/30/25/12/7/4 | 없음(해당 상수는 fallback) | 기록. `07_INTERFACES.md §2.2` |
| INC-14 | 코드↔주석 | `common/utils.py` docstring "그리퍼 = 1.0 rad/s" | 상수는 `5.0`, pick_cube 가 `2.5` 로 override | 없음 | 기록 |
| INC-15 | 코드↔코드 | `PickCubeStateMachine(num_cubes=4)` 기본 | `CUBE_NAMES = ['Cube1']` (1개) | `num_cubes > 1` 은 IndexError 경로 | 기록. `08_PIPELINES.md §4` |
| INC-16 | 잔재 | `AGENTS.md` 가 `tasks/pick_pen/`·`tasks/pick_cube_franka/` 를 "미등록 잔재"로 언급 | 두 디렉터리 모두 **없음**(정리 완료) | 없음 | **문서 정정** |

### INC-10 상세 — 성공 종료 발화 불가

`src/sim_to_real/tasks/common/mdp/_geometry.py::DESK_TOP_Z = 0.76` 은 이전 태스크(pen) 시절
상판 높이다. `src/sim_to_real/tasks/pick_cube/mdp/observations.py` 주석이 **"0.76 은 pen 잔재라
쓰지 않는다"** 고 명시하고 grasp term 에는 `0.705` 를 명시 주입하지만,
`common/mdp/terminations.py::task_done` 은 여전히 `DESK_TOP_Z` 를 쓴다. `container_cfg=Bowl` 을
받아 **xy 는 실제 그릇 좌표**를 쓰면서 **z 만 하드코딩 상수**다.

| 항목 | z (m) | 산출 |
|---|---:|---|
| 성공 판정 창 | `[0.765, 0.880]` | `0.76 + BOWL_HEIGHT_RANGE(0.005, 0.12)` |
| 책상 위 큐브 중심 | 0.726 | `0.705 + 0.020 + 0.001` |
| **그릇 안 큐브 중심** | **0.743** | `0.715 + BOWL_Z_BASE 0.005 + BOWL_FLOOR_THICKNESS 0.003 + 0.020` |
| 그릇 rim | 0.785 | `0.715 + 0.070` |
| `cube_lost` 임계 | 0.660 | `0.76 − 0.10` |

그릇 안에 안착한 큐브(0.743)가 성공 창 하한 0.765 에 못 미친다. 그릇 내부는 미끄러워 큐브가
바닥 중앙으로 가라앉으므로 이 값이 정상 안착 위치다.

**영향 범위**: `task_done`(기본 success) · `task_done_confirmed`(Eval success) ·
`cube_lost`(실패 컷) · `object_in_container`(subtask 관측 `place_cube1`).
`08_PIPELINES.md §9.2` 의 eval 수치를 해석하기 전에 반드시 확인할 것.

**수정 후보**(적용하지 않음): `_geometry.DESK_TOP_Z` 를 `0.705` 로 고치거나,
`task_done` 이 `container_cfg` 가 있을 때 그릇 root z 를 쓰도록 바꾸거나,
`pick_cube` 가 `desk_top_z` 를 grasp term 처럼 명시 주입하도록 파라미터를 추가한다.
**어느 쪽이든 GPU 실행으로 성공률 회귀를 확인해야 한다.**

관련: `_geometry.CONTAINER_DEFAULT_CENTER_XY = (0.36, 0.395)` 도 현 그릇 위치
`(-0.22, 0.265)` 와 다른 잔재값이다(단 pick_cube 는 항상 명시 주입하므로 미사용).

---

## 10. 마커 인덱스

수집 명령(작업 시점 기준, `ece_4560` 제외):

```bash
grep -rcE '(⚠|주의|함정|반드시|금지|절대|WORKAROUND|HACK|★)' \
  --include='*.py' --include='*.sh' src scripts docker | grep -v ece_4560
```

결과 = **25개 파일 79곳**.

| 파일 | 건수 | 본 문서 절 |
|---|---:|---|
| `src/sim_to_real/utils/domain_randomization.py` | 9 | §5.1 · §5.3 · §5.4 |
| `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` | 10 | §4.1 · §5.5 · §5.6 · §6.3 |
| `scripts/environments/author_pick_cube_scene.py` | 9 | §2.1 · §2.2 · §2.5 · §6.6 |
| `scripts/datagen/pink_ik_bridge_node.py` | 9 | §3.2 · §3.4 |
| `scripts/cuRobo/curobo_batch_planner.py` | 6 | §3.2 · §3.3 · §3.6 · §7.5 |
| `scripts/inference/run_cube_desk_ros_bridge.py` | 5 | §4.5 · §6.1 · §6.2 |
| `src/sim_to_real/tasks/pick_cube/spawn_area.py` | 5 | §3.1 · §5.1 |
| `src/sim_to_real/tasks/common/mdp/recorders.py` | 3 | §4.3 |
| `scripts/cuRobo/pickplace_sm.py` | 3 | §6.1 |
| `src/sim_to_real/assets/robots/lerobot.py` | 2 | §4.2 |
| `scripts/convert/isaaclab2lerobotv3.py` | 2 | §11.2 |
| `docker/policy-entrypoint.sh` | 2 | `06_RUNTIME_SPEC.md §4.1` |
| `docker/isaac-sim-entrypoint.sh` | 2 | `07_INTERFACES.md §7` |
| `src/so101_contract/follower_calibration.py` | 1 | §8.4 |
| `src/sim_to_real/utils/env_utils.py` | 1 | §11.1 |
| `src/sim_to_real/utils/cube_specs.py` | 1 | §7.6 |
| `src/sim_to_real/tasks/so101_base_env_cfg.py` | 1 | §4.4 |
| `src/sim_to_real/tasks/common/utils.py` | 1 | INC-14 |
| `scripts/inference/replay_dataset_to_bridge.py` | 1 | — (USD limit 초과 경고) |
| `scripts/inference/demo_vla.sh` | 1 | — (연속 데모엔 성공 판정 없음) |
| `scripts/environments/teleoperation/teleop_se3_agent.py` | 1 | §11.3 |
| `scripts/data/upload_to_huggingface.py` | 1 | §11.4 |
| `scripts/cuRobo/plot_sweep.py` | 1 | — |
| `scripts/convert/sim_dataset_to_real_follower.py` | 1 | §8.2 |
| `scripts/convert/joint_dataset_to_eef.py` | 1 | `05_DATA_SPEC.md §6.3` |

---

## 11. 자잘하지만 물리는 함정

### 11.1 중복 금지 — gripper effort 로직은 한 곳에만

`src/sim_to_real/utils/env_utils.py` 는 vendored leisaac 유틸이지만 gripper effort 부분은
`utils/gripper_effort.py` 로 포팅돼 있다. **두 곳에 두지 않는다.**

### 11.2 demo 정렬은 숫자 기준

HDF5 `demo_N` 을 사전순 정렬하면 `demo_10` 이 `demo_2` 앞에 온다.
`scripts/convert/isaaclab2lerobotv3.py` 는 숫자 기준으로 정렬한다.

### 11.3 카메라 focal 은 USD attr 로 못 바꾼다

`TiledCamera` 는 USD `focalLength` attr 변경을 렌더·데이터에 반영하지 않는다.
cfg 로 지정해야 한다(`teleop_se3_agent.py --tune_cameras` 가 cfg 를 override 한다).

### 11.4 HF 업로드는 `codebase_version` 태그를 직접 만들어야 한다

LeRobot 은 dataset repo 의 `codebase_version` 태그(예 `v3.0`)를 revision 으로 찾는데
`upload_folder` 는 태그를 만들지 않는다 → `RevisionNotFound`. 게다가 `exist_ok` 만으로는 옛
커밋에 고정돼 재업로드분이 안 보이므로 **delete + create 로 main HEAD 에 이동**시킨다.

### 11.5 `envs/env_10` 자연 정렬

env prim 을 lexicographic 정렬하면 `env_10 < env_2` 라 10 env 이상에서 per-env material 이
어긋난다. `re.search(r"env_(\d+)")` 로 숫자 정렬한다.

### 11.6 그릇 내부는 미끄럽다

큐브가 바닥 중앙으로 미끄러져 깔린다. 성공 판정 z 를 정할 때 이 안착 위치를 기준으로 해야
한다(INC-10 의 전제).

---

## 12. 제거된 것들과 이유

| 제거 대상 | 이유 |
|---|---|
| RL 보상·커리큘럼 (`PickCubeRewardsCfg` = 빈 stub) | VLA-only 리팩토링. env 는 추론·데이터 기판으로 축소 |
| MoveIt · cuMotion · Lula · RMPFlow · follow-target IK | 5-DOF pose-goal 한계와 유지비. cuRobo 로 수렴 |
| `policy-server-rtc` 모드 | 백엔드 스크립트(`policy_server_rtc.py`)가 이 branch 에 없어 entrypoint 에서 제거. 재도입 시 스크립트 + 모드를 함께 복원 |
| GR00T-N1.7 **전용 컨테이너·bridge** | 별도 서비스가 불필요해졌다. LeRobot 0.6.0 이 N1.7 을 네이티브 `groot` policy 로 지원해 policy-server 안에서 ACT/SmolVLA 와 같은 경로로 학습·추론한다 (`env/groot_n17.env`) |
| GR00T-**N1.5** 경로 (`env/groot_n15.env`) | LeRobot v0.6.0 이 N1.5 config/checkpoint 를 명시적으로 거부한다(N1.5 는 0.5.1 을 쓰라는 오류). 프로필을 `groot_n17` 로 대체 |
| leisaac 런타임 의존 | 유용한 코드만 `src/sim_to_real/`·`src/so101_contract/` 로 vendor. leisaac import 0건 |
| `lerobot` Docker 서비스 + WSL ROS 스택 | 실기기는 Windows native uv 로 전환 |
| 그리퍼 offset(31.75 배수) | 절대 joint target 으로 통일 (§4.4) |
| `tasks/pick_pen/` · `tasks/pick_cube_franka/` | 미등록 잔재였고 현재 트리에 없음 (INC-16) |
| uniform SO(3) 큐브 orientation | 비현실적 자세 생성 (§5.4) |
| place-descent (cuRobo SM) | 깊은 linear 하강이 동적 그릇을 밀어냄 → release-above-bowl 로 대체 |
| retreat phase (pink SM) | 불필요 (INC-05) |

---

## 13. cuRobo SM 녹화 성능 — 무엇이 병목이고 무엇이 아닌가

2026-07-28 GPU 실측(RTX PRO 5000 Blackwell 48 GB, `--num_envs 2`, 379 step/ep, 2/2 성공).
**추정이 아니라 계측값이다.** 최적화 방향을 잘못 잡지 않도록 남긴다.

### 13.1 트라이얼 예산 — planner 는 병목이 아니다

| 구간 | 시간 | 비중 |
|---|---|---|
| Isaac 부팅 13.0 s · planner init 6.0 s | — | 1회성 |
| plan | 3.0 s | 5% |
| replay + preroll + posthold | 22.8 s | 37% |
| export (gzip) | 21.6 s (10.8/demo) | 35% |

실제 DR 스폰에서 plan 은 `candidate_passes=1` 로 첫 후보가 바로 풀린다(approach 871 ms).
**합성 부하로 재면 안 된다** — 격자 좌표 + yaw 45/60/75 를 강제하면 `passes=29`, plan 50 s 가
나와 approach 가 96% 인 것처럼 보인다. 실분포에서는 5 phase 가 균등하다
(approach 871 / transit 653 / retreat 496 / grasp 489 / lift 460 ms).

replay 는 379 step / 22.8 s = **16.6 step/s = 0.55× 실시간**(카메라 3대 640×480 렌더 bound).
카메라 해상도는 계약이라 못 줄인다 — 유일한 레버는 `--num_envs` 증가(타일드 렌더 상각)다.

### 13.2 왜 lzf + frame-chunk 인가

IsaacLab 기본은 gzip(4)이고, export 는 `RecorderManager.export_episodes` 가 **env 순차**로
돌며 심 루프를 세운다. 실제 렌더 프레임(원본 999 MiB/demo) 측정:

| 설정 | MiB/s | 압축률 | s/demo | 1000 ep 디스크 |
|---|---|---|---|---|
| gzip(4) auto-chunk (IsaacLab 기본) | 123 | 6.56 | 8.13 | 152 GB |
| gzip(1) auto-chunk | 158 | 5.50 | 6.31 | 182 GB |
| lzf auto-chunk | 198 | 3.70 | 5.04 | 270 GB |
| **lzf frame-chunk (채택)** | **359** | **3.26** | **2.79** | **306 GB** |
| none frame-chunk | 2066 | 1.00 | 0.48 | 999 GB |

**청크를 프레임 단위로 바꾸면 압축률이 떨어진다**(6.56 → 4.75). h5py 자동 청크
`(24,60,80,1)` 가 24 프레임에 걸쳐 타일을 인터리브해 **정적 배경의 시간축 중복**을 잡기
때문이다. lzf 는 그 이득이 작고 속도가 압도적이라 frame-chunk 와 짝지었다. "자동 청크가
멍청하다"고 오해하지 말 것 — gzip 을 쓸 거면 auto-chunk 를 유지해야 한다.

속도 vs 디스크 트레이드오프일 뿐 **값 계약과 무관**하다(전 프리셋 왕복 배열 동일).

### 13.3 녹화 버퍼는 VRAM 에 쌓인다 — 그런데 CPU 로 내리면 더 느리다

`EpisodeData.add` 가 `value.clone()` 으로 device 를 보존해, recorder term 이 GPU 텐서를
돌려주면 이미지가 에피소드 내내 VRAM 에 쌓이고 export 직전 `torch.stack` 이 피크를 2배로
만든다(999 MiB/env/에피소드).

`DatagenRecorderTerm.record_pre_step` 에서 `.cpu()` 로 미리 내려 host RAM 으로 옮기는 안을
실측 A/B 했다 (num_envs=8, 양쪽 8/8 성공·HDF5 키 동일):

| | `.cpu()` 적용 | `.cpu()` 없음 (채택) |
|---|---|---|
| plan | 47.5 s | 45.6 s |
| replay | **76.5 s** | **25.9 s** |
| export | 28.0 s | 31.0 s |
| 트라이얼 | 165.1 s | **117.3 s** |
| 에피소드당 | 20.6 s | **14.7 s** |
| VRAM 피크 | 34.4 GB | 45.0 GB |

`.cpu()` 비용은 대역폭(8 env × 21 MiB/step @30 Hz = 634 MB/s)이 아니라 **스텝마다 렌더
파이프라인을 드레인시키는 동기화**다. env 가 늘수록 커진다 — 2-env 에선 +26%, 8-env 에선
**+41%**. 그래서 채택하지 않았다.

> ⚠ 위 표의 **VRAM 절대값은 신뢰하지 말 것** — 이 A/B 실행 중 다른 워크로드가 같은 GPU 에서
> 26 GB 를 잡고 있었다. 두 실행이 연속이라 **차이(−10.6 GB)** 는 유효하지만(8 env × 1 GiB
> 이미지 버퍼 + stack 피크와 일치), 절대값은 그만큼 부풀려져 있다. 깨끗한 값은 §13.6.

**더 높은 `--num_envs` 가 필요하거나 GPU 를 학습과 공유할 때만** `DatagenRecorderTerm` 의
반환 텐서에 `.cpu()` 를 붙인다(한 줄). VRAM −10.6 GB 를 replay +41% 로 산다.

### 13.4 `use_cuda_graph=False` 는 필수다 — 다시 켜지 말 것

`curobo_batch_planner.PickPlacePlanner` 의 `use_cuda_graph=False` 는 놓친 최적화가 아니다.
`curobo.runtime.cuda_graph_reset = True` 로 게이트를 열고 `use_cuda_graph=True` + 
`warmup(enable_graph=True)` 로 A/B 한 결과:

| 요청 | 현행 | cuda graph |
|---|---|---|
| #1 | 50.8 s · 4/4 | 9.6 s · 4/4 (궤적 동일) |
| #2 | 45.7 s · 4/4 | 8.0 s · 4/4 (궤적 동일) |
| #3 | 53.1 s · **4/4** | 6.5 s · **1/4 회귀** |
| #4 | 50.3 s · 4/4 | **`CUDA error: an illegal instruction was encountered`** |

5–7× 빠르지만 3번째 요청에서 해를 잃고 4번째에서 프로세스가 죽는다. `_plan_to_batch` 가
plan 사이에 `update_tool_pose_criteria`/`disable_link_collision` 을 토글하고
`_ensure_batch_size` 가 solver 를 destroy 하는 구조와 graph 재캡처가 맞지 않는 것으로 보인다.

### 13.6 `--num_envs` 스윕 — 16 이 최적, VRAM 상한은 아직 안 닿았다

2026-07-28, **유휴 GPU**(48.9 GB, 외부 워크로드 0)에서 구성마다 64 에피소드를 생성하고
`scripts/convert/isaaclab2lerobotv3.py` 변환까지 마친 wall-clock. 전 구성 **64/64 성공**.

| num_envs | trials | 생성 | 변환 | 합계 | **s/에피소드** | VRAM 피크 |
|---|---|---|---|---|---|---|
| 1 | 64 | 1831.2 s | 191.8 s | 2023.0 s | 31.61 | 9.7 GB |
| 2 | 32 | 1551.5 s | 195.0 s | 1746.6 s | 27.29 | 11.4 GB |
| 4 | 16 | 1351.3 s | 194.8 s | 1546.1 s | 24.16 | 14.7 GB |
| 8 | 8 | 877.5 s | 195.8 s | 1073.4 s | 16.77 | 22.1 GB |
| **16** | 4 | **686.1 s** | 196.0 s | **882.1 s** | **13.78** | 34.9 GB |

1000 에피소드 환산: 1-env 8.8 h · 8-env 4.7 h · **16-env 3.8 h**.

**§13.3 이 "8 env 45 GB = 사실상 상한, 16 env OOM 예상"이라 적었던 것은 오염된 측정이었다.**
깨끗한 환경에서 8-env 는 22.1 GB, 16-env 는 34.9 GB 로 OOM 이 나지 않는다.

**변환 196 s 는 `num_envs` 와 무관한 상수다**(191.8–196.0, 편차 2%). 에피소드 단위 CPU
작업이라 상각되지 않아 16-env 총시간의 22%를 차지한다. 생성은 GPU, 변환은 CPU 라 자원이
겹치지 않으므로 **`scripts/cuRobo/generate_dataset.sh` 가 둘을 파이프라인으로 겹친다**
(배치 N 생성 중 배치 N-1 변환) — 1000 ep 3.83 → 2.98 h, HDF5 디스크 피크 375 → 48 GB.
e2e 실측(64 ep × 3 배치)에서 변환 2회분 409 s 가 숨고 디스크 피크가 48.4 GB 로 잡혔다.
설계·실측·병합 절차 = `08_PIPELINES.md` §5.8.

32-env 는 미측정이다. 8→16 이 아직 1.22× 로 꺾이지 않았고 여유가 14 GB 남지만, VRAM 이
8-env 배증마다 +12.8 GB 라 아슬아슬하다.

재측정 하네스 = `scripts/cuRobo/sweep_num_envs.sh`
(`./sweep_num_envs.sh "1 2 4 8 16" 64` → `scratch/num-envs-sweep/logs/sweep_num_envs.json`).
하드웨어·카메라 해상도·압축 프리셋을 바꾸면 이 표를 다시 떠야 한다.

### 13.5 warp 캐시는 ComputeCache 와 다르다

`/root/.nv/ComputeCache` 는 CUDA 드라이버의 PTX→SASS JIT 캐시이고, warp 커널 캐시는
`/root/.cache/warp/<ver>` 다. 후자를 안 걸면 `run --rm` 마다 재컴파일 —
planner 첫 plan 이 58.0 s(cold) vs 50.8 s(warm). 볼륨·무효화 절차 = `06_RUNTIME_SPEC.md §3.2`.

---

## 14. 책상 높이 · grasp 조준 z — 두 상수가 서로를 가리고 있었다

### 14.1 증상

`pickplace_sm` 이 **매 실행** `⚠ cube z drift 10.3 mm` 를 띄웠다. 성공률에는 영향이 없어
(2026-07-28 스윕 320/320) 감시만 하고 미뤄둔 잠복 결함이었다.

### 14.2 실측 (2026-07-29, 96 샘플)

| 값 | 측정 | 비고 |
|---|---|---|
| 큐브 안착 중심 z (base_link) | **0.04976** ± 0.0001 | 추가 30 스텝 변화 0.09 mm = **정착값, 잔진동 아님** |
| 로봇 base z (world) | 0.67490 | `_ROBOT_POS` 와 일치 |
| 큐브 중심 z (world) | 0.72466 | 저작 0.705+0.020=0.725 대비 −0.34 mm(접촉 침투) |

### 14.3 원인 — 오차가 둘이고 서로 상쇄하고 있었다

| 상수 | 옛 값 | 실측 | 오차 |
|---|---|---|---|
| `--grasp_z`(base) | 0.060 (경험 튜닝) | 0.04976 | **+10.24 mm** |
| `TABLE_TOP`(urdf) | `0.035 + BASE_T[2]` = 0.00252 | −0.00272 | **+5.24 mm** |

`--grasp_z` 가 10.24 mm 높고 `GRASP_Z_OFF = −0.008` 이 8 mm 끌어내려, solver 조준점이
**중심보다 2.24 mm 위**라는 working point 에 우연히 안착해 있었다. 두 오류가 서로를 가려
어느 쪽도 단독으로는 못 고친다 — 그래서 "함께 재측정할 때 교체한다"고 미뤄뒀던 것이다.

### 14.4 수정 — 단일 소스에서 파생

```
so101_contract/grasp_geometry.py :: TABLE_TOP_BASE = 0.0298   (base_link, 실측 정착값)
    ├── SM      : grasp_z   = TABLE_TOP_BASE + cube_half
    └── planner : TABLE_TOP = TABLE_TOP_BASE + BASE_T[2]
GRASP_Z_OFF : −0.008 → +0.0022
```

`GRASP_Z_OFF` 는 이제 **중심 대비 조준 높이**만 뜻한다. 값 +0.0022 는 옛 조합이 만들던
solver 조준점 `0.01952` 를 **정확히 보존**하도록 잡았다 — 프레임 장부만 고치고 물리
working point 는 건드리지 않았다.

**임의 큐브 크기에 유효하다**: 씬은 큐브를 `상판 + half + slack` 에 저작하고(`author_pick_cube_scene.py`)
물리는 `상판 + half − 침투` 로 정착시킨다. 둘 다 `half` 에 선형이고 상수항이 같아 40/50 mm
어느 쪽이든 유도식이 맞는다 — 50 mm 별도 실측이 필요 없는 이유다.

`--grasp_z` 는 `default=None` 으로 격하됐다(튜닝 override). 상시 경로는 유도식이다.

### 14.5 부수 효과 — descend clamp 와 bowl ring 이 5.24 mm 내려간다

`TABLE_TOP` 은 조준뿐 아니라 pad 최저점 clamp(`TABLE_TOP + TABLE_MARGIN`)와 bowl ring
obstacle 높이(`TABLE_TOP + BOWL_RING_H/2`)에도 쓰인다. 상판을 실측값으로 내리면 이 둘이
같이 내려간다 — 즉 **유효 clamp 여유가 9.24 mm 에서 의도했던 4 mm 로 돌아온다**
(`TABLE_MARGIN` 의 "실제 ≥2 mm 무접촉" 요구는 여전히 충족).

조준점은 보존되지만 이건 실제 거동 변화라 sweep 으로 판정했다 — §14.6.

### 14.6 무회귀 검증

기준선 파라미터 = `sweep` 기본값(`nx=15 ny=8 boundary_n=20`) = **124 셀**.

| | 기준선(main) | 수정 후 | 판정 |
|---|---|---|---|
| yaw0 | 124/124 | **124/124 = 100%** | 동률 |
| yaw random ×3 | 372/372 | **372/372 = 100%** | 동률 |
| drift 경고 | 매 실행 발화 | **0회** | 해소 |

`TABLE_TOP` 하강으로 descend clamp 여유가 9.24 → 4 mm 로 좁아졌는데도 회귀가 없다 —
clamp 가 실제로 binding 하는 경우가 드물다는 뜻이다(대부분 `_descend_tstar` 의 기하 해가
먼저 지배한다). 496 시도 전부 성공.

### 14.7 곁다리로 고친 것 — headless sweep 이 부팅 중 죽던 버그

`pickplace_sm` 이 `omni.appwindow` 를 모듈 최상단에서 import 했는데, 이 확장은
**headless + 카메라 없음**(= `sweep` 의 정확한 조합)에서 로드되지 않아
`ModuleNotFoundError` 로 죽었다. record 경로가 늘 `--enable_cameras` 라 가려져 있었다.
`_key_listener()` 안으로 지연 import — interactive 모드에서만 필요한 심볼이다.

### 14.8 큐브가 작아지면 table clamp 가 조준을 밀어올린다 (크기 DR 전제조건)

크기 DR(25~40 mm, `03_ENV_SPEC.md` §9.1.1)을 넣기 전에 답해야 했던 질문:
**작은 큐브에서도 pad 조준이 FK 게이트 안에 남는가.**

`cand_pose_manifold` 는 pad center 를 **face center 높이**(= 큐브 중심 높이)에 조준한다.
그런데 descend 는 jaw tip 이 `TABLE_TOP + TABLE_MARGIN` 아래로 못 가게 clamp 된다
(`_descend_tstar`). tip 은 pad center 보다 `PAD_LOW_OFF − FIXED_INNER_CENTER[2]` = 28.7 mm
아래이므로 **pad center 가 내려갈 수 있는 하한**이 `TABLE_TOP + 4 + 28.7 mm` 로 고정된다.
큐브가 작아지면 중심은 내려가는데 이 하한은 그대로다 → 조준이 위로 벌어진다.

α = 0(top-down, 최악)에서 정리하면 오차는 **크기에만** 의존한다:

```
e_h = TABLE_MARGIN + PAD_LOW_OFF − FIXED_INNER_CENTER[2] − half − GRASP_Z_OFF
    = 0.004 + 0.075 − 0.0463 − half − 0.0022
    = 0.0305 − half
```

| 큐브 | half | e_h (해석, α=0) | 한계 `E_HEIGHT_MAX` |
|---|---:|---:|---:|
| 40 mm | 0.0200 | +10.5 mm | 28 mm |
| 35 mm | 0.0175 | +13.0 mm | 〃 |
| 30 mm | 0.0150 | +15.5 mm | 〃 |
| 25 mm | 0.0125 | +18.0 mm | 〃 |

★ **`half < 0.0305` 인 한 clamp 는 항상 binding 이다.** 즉 40 mm 에서도 pad center 는 이미
큐브 중심보다 10.5 mm 위에서 잡고 있었고, 크기 DR 은 그 편차를 키울 뿐 **새로운 실패 모드를
만들지 않는다**. (§14.6 의 "clamp 가 실제로 binding 하는 경우가 드물다"는 서술은 성공률이
안 변했다는 관찰에 대한 잘못된 해석이었다 — binding 은 상시이고, 둔감한 쪽은 `e_h` 다.)

물리적으로도 무리가 없다: pad 은 center 에서 tip 까지 28.7 mm 뻗어 있어 25 mm 큐브의 옆면
(`z ∈ [table, table+25 mm]`)을 pad 하단 구간이 통째로 덮는다.

`curobo_batch_planner.py --self-check-geom` 이 사다리 전 구간에 이 부등식을 assert 한다
(`_check_grasp_height_budget`). 크기 하한을 더 낮추거나 `TABLE_MARGIN`·pad 기하를 건드리면
sweep 을 돌리기 전에 여기서 먼저 터진다.

---

## 15. gripper ramp 가 slew cap 을 넘어 폐합이 lift 전에 안 끝나고 있었다

### 15.1 증상

크기 DR sweep(크기 4종 × yaw0/yaw-random, 총 2232 시도)에서 **3건**만 place-fail 이 났다.
전부 30 mm·yaw-random 이고, 같은 셀을 `fail --auto` 로 재현하면 성공했다(확률적 꼬리).

sweep 실패 기록에 물리 흔적(`max_cube_z`·`final_cube_xyz`)을 남기게 하고 다시 재니
정체가 드러났다:

```
cell (+0.1371,+0.1743) interior  yaw=83.6  max_z=0.7255  final=(+0.1439,+0.2063,+0.7199)
```

`max_cube_z` 가 안착면(책상 0.705 + half 0.015 = 0.720)보다 **5 mm** 높을 뿐이다 —
그릇 밖으로 튕긴 게 아니라 **애초에 큐브를 들지 못했다**. 큐브는 제자리에서 32 mm 밀렸다
(jaw 가 밀어냄 = skate 항복모드).

### 15.2 원인 — 명령 속도가 env slew cap 의 3.2 배

`_assemble` 은 gripper 폐합을 `CLOSE_STEPS = 5` 프레임 linspace 로 보냈다.

| | Δ(feature) | Δ(rad) | 프레임 | 명령 속도 | env cap |
|---|---:|---:|---:|---:|---:|
| close (75→5) | 70 | 1.344 | 5 | **8.06 rad/s** | 2.5 |
| open (5→75) | 70 | 1.344 | 10 | **4.03 rad/s** | 2.5 |

`SlewLimitedJointPositionAction` 이 명령을 cap 으로 잘라내므로, 폐합에 실제로 필요한 시간은
`1.344 / 2.5 = 0.54 s` = **16 프레임**이다. 그런데 예산은 `close(5) + grasp_hold(5) = 10`
프레임뿐 → 그리퍼가 1.344 rad 중 0.833 rad 만 닫힌 채 **lift 가 시작**됐다. 남은 29° 는
팔이 올라가는 도중에 닫힌다. 대개는 그래도 물리지만, 드물게 jaw 가 아직 벌어진 쪽으로
큐브를 밀어내고 grasp 가 실패한다. release 도 같은 구조다 — 다 열리기 전에 retreat 이
시작되면 큐브를 끌고 나온다.

같은 종류의 함정이 이미 두 번 있었다: arm slew cap 2.5 가 lock-step 을 깨뜨린 건
(`08_PIPELINES.md` 상수 표 주석), 기록된 action 이 물리적으로 불가능해 VLA 데이터가
jerky 했던 건([[vla-data-jerky-slew-record-fix]]) — **명령과 cap 이 갈리면 조용히 어긋난다.**

### 15.3 수정 — ramp 길이를 cap 에서 유도

```python
def grip_ramp_steps(g_from, g_to):
    delta_rad = abs(g_from - g_to) * _GRIP_RAD_PER_FEATURE
    return max(GRIP_RAMP_MIN_STEPS, ceil(delta_rad / (GRIPPER_SLEW_MAX_RAD_S / CONTROL_HZ)))
```

`CLOSE_STEPS`·`OPEN_STEPS` 상수를 없애고 `_assemble` 이 매번 유도한다(knob 으로 `grip_open`/
`grip_close` 를 바꿔도 따라간다). 기본값에서 close = open = **17 프레임 = 2.37 rad/s ≤ 2.5**.

★ **실제 폐합 속도는 그대로다** — 어차피 cap 이 지배했다. 바뀌는 것은 lift/retreat 타이밍이
폐합·개방 **완료 뒤로** 밀린다는 것뿐이다(순수 안정화, 궤적 기하 불변). 부수 이득으로 기록된
action 이 물리적으로 실현 가능한 값이 된다.

`--self-check-geom` 의 `_check_grip_ramp_within_cap` 이 이 부등식을 assert 한다.

### 15.4 검증

`sweep`(124 셀) × 크기 4종, `--yaw random --trials 3` = 크기당 372 시도:

| 큐브 | 수정 전 | 수정 후 |
|---|---|---|
| 25 mm | 372/372 | 372/372 |
| 30 mm | **370/372 · 371/372**(2회 측정 모두 실패) | **372/372** |
| 35 mm | 372/372 | 372/372 |
| 40 mm | 372/372 | 372/372 |

실패가 30 mm 에만 몰린 것은 통계적으로 유의하지 않다(전체 3/2232 ≈ 0.13 % 의 꼬리이고,
다른 크기 1488 시도에서 0건일 확률이 ~14 %). 즉 **크기 DR 이 만든 결함이 아니라 원래 있던
ramp 결함**이며, 40 mm 단일 크기 시절의 100 % 도 그만큼 운이었다.

### 15.5 grasp 후보를 "처음 통과한 것"으로 채택하고 있었다

ramp 를 고친 뒤에도 ~0.05 %/시도 의 꼬리가 남았고, 실패 진단(§15.1 의 기록 필드)을 모으니
전부 **큐브를 밀어낸 뒤 못 드는** 모습이었다. 채택된 후보의 FK 오차가 범인을 지목했다:

| 실패 | lift | 큐브 이동 | e_n | e_t | face_angle |
|---|---:|---:|---:|---:|---:|
| 25 mm yaw0 | +2.7 mm | 20 mm | +3.6 | +1.5 | **+32.7°** |
| 25 mm yaw0(다음 회차) | +1.7 mm | 16 mm | **+7.9** | +0.9 | −12.9° |

둘 다 게이트를 통과한다(`SIMPLE_FACE_GATE_MAX_DEG=40` · clearance 창 `2~8 mm`). 게이트가
넓은데 **채택은 게이트 통과 순간 끝났다** — `_plan_pregrasp_batch` 는 `|α|` 오름차순으로 돌며
처음 통과한 후보를 쓰고 `_candidate_score` 는 계산해 진단에 싣기만 했다. 즉 "통과했지만
조준이 나쁜" 후보가 그대로 실행됐다.

수정 3가지:

1. **최선 후보 채택** — `CANDIDATE_SCAN_PASSES = 3`. 통과한 env 도 앞 3 pass 는 batch slot 을
   계속 쓰며 후보를 비교하고, score 최소를 채택한다. 미통과 env 때문에 어차피 pass 를 더
   돌므로 plan 호출은 늘지 않는다.
2. **score 재정렬** — `(face_angle 5° 버킷, |e_n − target|, |e_t|, |e_h|, …)`.
   ① 비스듬한 closing 이 한쪽 jaw 로 face 모서리를 때려 큐브를 민다.
   ② `e_n` = pad↔face 거리 = **폐합이 이동해야 하는 거리**(멀수록 칠 확률↑).
   ④ 예전 1순위 `|e_h|` 는 table clamp 지배로 변동이 4 mm 안쪽(§14.8)이라 변별력이 없는데
      정렬·clearance 를 눌러 이겼다.
3. **접촉 정착 연장** — `GRASP_HOLD_STEPS 5 → 15`. ramp 가 17 프레임으로 늘어난 뒤 hold 5 는
   접촉력 정착 전에 lift 를 시작시켰다(큐브를 16 mm 들다 놓친 slip 1건).

**검증**(크기 4종 × `yaw0`(124) + `yaw random ×3`(372) = 1984 시도):

| 단계 | yaw0 (496) | yaw-random ×3 (1488) | 합 |
|---|---|---|---|
| 원본 | 496/496 | 1485/1488 | 1981 (99.85 %) |
| + ramp 유도(§15.3) | 495/496 | 1488/1488 | 1983 (99.95 %) |
| + 최선 후보·score·hold | **496/496** | 1485/1488 | 1981 (99.85 %) |
| + `--grasp_retries 1` | 124/124 (25 mm 만, 재시도 발동 0회) | 미측정 | **전면 검증 미완** |

### 15.6 잔여 실패의 정체 — 조준이 아니라 5-DOF 한계

실패를 하나씩 잡을 때마다 **지배 요인이 바뀌었다**(정렬 32.7° → clearance 7.9 mm → slip →
tilt 15°). 마지막 케이스는 `face_angle +2.3°` · `e_n +4.4 mm` 로 조준이 거의 완벽한데도
큐브를 25 mm 밀어냈다 — α(tilt) 15° 가 강제된 셀, 즉 α 0~10° 가 IK·게이트로 불가능해서
비스듬한 접근밖에 없는 위치였다. 조준 품질 튜닝으로는 여기를 못 넘는다.

두 가지 구조적 원인이 남는다:

- **대각 yaw(≈45°)**: 정사각 큐브는 어느 face 를 골라도 `wrap90` 후 최대 45° 어긋나고,
  `RHO_CAP_DEG=12` 가 wrist_roll 보호를 위해 12° 까지만 보정 → `face_angle` 17~25° 잔류.
  `rho_cap_deg 12→16` A/B: 72 시도 71 성공으로 **개선 신호 없음** → 12 유지.
- **α 강제 셀**: 원거리·경계에서 작은 α 가 도달 불가 → 기울어진 접근이 유일한 해.

그래서 마지막 조치는 조준이 아니라 **재시도**다(`--grasp_retries`, 기본 1):
grasp 실패는 큐브를 15~48 mm 밀어내므로 **입력 pose 가 바뀐다** → 결정적 planner 도 재계획하면
다른 후보·다른 궤적을 낸다. 성공한 env 는 init hold 로 고정해 그릇 안 큐브를 다시 건드리지
않고 판정을 누적한다. record(데이터 생성) 경로는 에피소드 규격 때문에 재시도하지 않는다
— 실패 에피소드는 `success=False` 로 저장돼 변환기가 걸러낸다.

⚠ **retry 포함 구성의 전면 sweep 은 사용자 요청으로 중단됐다**(25 mm yaw0 124/124 까지만).
남은 조합을 재려면 `scratch/chain_full_verify.sh` 를 다시 돌린다.

---

## 15. 카메라 extrinsic DR — 왜 EventTerm 이 아니고, 왜 USD write 가 되는가

frame-wise 카메라 pose DR(`03_ENV_SPEC.md §11.6`)을 만들면서 확정한 네 가지.

### 15.1 ★ frame-wise 갱신은 `mode="interval"` EventTerm 으로 하면 1 프레임 늦는다

- **결정**: pose write 는 `PickCubeEnv.step()` **최상단**(= `super().step()` 앞)에서 한다.
- **근거**: `ManagerBasedRLEnv.step()` 은 decimation 루프 **안**에서 `sim.render()` 를 부르고,
  interval 이벤트는 그 루프가 끝난 **뒤**에 `event_manager.apply(mode="interval")` 로 돈다.
  즉 interval term 이 쓴 pose 는 다음 step 의 렌더에 처음 반영된다 → RGB 와 extrinsic
  metadata 가 한 프레임 어긋난다.
- **어기면**: 조용히 어긋난다. 픽셀은 그럴듯하고 라벨만 틀린다(학습이 잘못된 pose 를 배운다).
- **앵커**: `src/sim_to_real/tasks/pick_cube/pick_cube_env.py::PickCubeEnv.step`

reset 상태 초기화는 EventTerm 으로 해도 된다(`reset_camera_extrinsic_dr`) — `_reset_idx` 안에서
돌고, 그 직후의 `num_rerenders_on_reset` 재렌더가 새 bias 를 보게 하려고 reset 핸들러가 pose 를
바로 write 한다.

### 15.2 ★ 카메라 xform 의 USD write 는 Fabric 렌더에 반영된다 (material 과 다르다)

- **측정(2026-07-30, probe)**: 정지 씬에서 `cam._view.set_local_poses()` 로 로컬 x 를 크게 밀고
  같은 step 의 RGB 를 비교 — 프레임 평균 픽셀차(0~255) `noop 0.036 / 0.20 m 이동 14.4`(top),
  `0.078 / 24.5`(wrist, 0.05 m), `0.036 / 44.3`(front, 0.10 m). **articulation 링크 자식
  (wrist=gripper, front=shoulder)도 동일하게 반영**된다.
- **왜 확인이 필요했나**: `§5.5 robot color DR` 이 정확히 반대였다 — 런타임 USD material 편집이
  Fabric 렌더에 무반영이라 Replicator 로 재작성해야 했다. 그 함정은 **material 한정**이고
  transform 에는 해당하지 않는다는 걸 실증하기 전에는 구현 경로를 고를 수 없었다.
- **부수 관측**: 큰 점프 직후 1 프레임에 잔상이 남는다(nominal 복원 후 픽셀차 1.2~2.8 vs noop
  0.036) — RTX 시간축 누적(TAA/denoise). `temporal_mode="iid"` 가 비현실적인 또 하나의 이유다.
- **앵커**: `scripts/contract/validate_camera_extrinsic_dr.py` (상태기계) ·
  `scratch/2026-07-29-camera-extrinsic-dr/` (probe·스모크, 임시물)

### 15.3 `set_world_poses` 가 아니라 `_view.set_local_poses` 를 쓴다

- **결정**: private `cam._view.set_local_poses()` (IsaacLab 2.3.2 에 local pose 공개 API 없음).
- **근거**: `Camera.set_world_poses()` 는 부모 world transform 을 USD
  `ComputeLocalToWorldTransform` 으로 재계산해 local 을 역산한다 — 우리가 원하는 건 애초에
  local mount 갱신이고, 부모가 physics 로 움직이는 링크(wrist/front)면 그 재계산이 비용이자
  stale 위험이다. local 로 쓰면 **세 카메라가 같은 코드 한 줄**이 된다(top 은 부모가 정적일 뿐).
- **검증**: world 위치를 `T_world_parent(t) ⊗ local(t)` 과 대조 — 오차 **0.000 mm**(wrist·front,
  16 env × 120 프레임). 부모-자식 관계가 유지된다.

### 15.4 nominal 은 cfg 가 아니라 prim 에서 읽는다 (convention 왕복 금지)

- **결정**: `CameraExtrinsicDR.__init__` 이 `cam._view.get_local_poses()` 로 nominal 을 1회 캡처.
- **근거**: cfg 의 `OffsetCfg.convention="world"`(fwd +X·up +Z)는 **author 시점 표기**다. prim 에
  실제 저장된 quat 은 OpenGL/USD 카메라 규약(fwd −Z·up +Y)으로 변환돼 있어 값이 다르다
  (`03_ENV_SPEC.md §7` 표 아래 실측값). prim 값을 기준으로 삼으면 변환 왕복이 0회가 되고,
  teleop `--tune_cameras` 오버라이드도 자동으로 따라온다.
- **함정**: 이 규약 때문에 delta 축 이름을 roll/pitch/yaw 로 부르면 오해가 생긴다 →
  `x=right(tilt)` · `y=up(pan)` · `z=backward(roll)` 로 부른다.
- **함정 2**: USD 는 `xformOp:orient` 를 `w>0` 정규형으로 돌려줄 수 있다(front nominal 은
  `w=-0.5`). 읽어온 quat 을 **componentwise** 로 비교하면 같은 회전인데 diff≈1 로 보인다 —
  회전 오차는 두 quat 내적의 각도(부호 무시)로 재야 한다.

---

## 참조

- 상수의 현재 값 → `03_ENV_SPEC.md §12` 상수 대장
- 계약 수식 → `04_IO_CONTRACT.md`
- 핀·설치 → `06_RUNTIME_SPEC.md §7`
- 에러 사례집 → `docs/TROUBLESHOOTING.md` (92개 항목, 별 도메인)

---

## 16. Mimic/SkillGen 증강 데이터 생성

### 16.1 ★ 5-DOF 부분 pose — 못 만드는 축은 1개뿐이다

- **결정**: 도달 가능 orientation manifold = 3-parameter 족
  ```
  R(pan, α, ρ) = Rz(pan) · Ry(−α) · R_TOPDOWN · Rz(ρ) · Ry(TCP_TWIST_RY)
  ```
  이 밖의 회전은 어떤 관절값으로도 만들 수 없다.
- **증상** (manifold 무시 시): IK 가 position 까지 어긋난다(실측: 증강 실패 5/5 가 위치 잔차
  20~181 mm · 회전 잔차 20~93°, 팔이 허공을 집었다).
- **여축 식별**: manifold 위치는 pan 으로 고정되므로 자유 파라미터는 α·ρ 둘. 생성할 수 없는
  회전 축 = 이 둘의 가중합 영역 complement 다. 캐노니컬 top-down 에서 그것은 **tool ŷ**(pitch).
  그래서 orientation 의 pitch 를 free parameter 로 쓰면 안 된다.
- **함정 반대 신호**: cuRobo 의 jacobian 부호에 따라 spatial frame vs tool frame 축이 뒤바뀐다.
  `track_position()` 으로 회전 3축을 다 풀면 `wrist_roll` 이 free 가 되어 **경계에서 슬루 상한에
  붙는다**. 증상이 조용하다 — 성공률 소폭 악화.
- **우리 방식**: tool 프레임 pitch(ŷ) 에 정확히 weight 0, 나머지는 soft cost. 자기검사 ⑤ 에서
  `(0,0,0)→[0,1,0]` 불가 검증한다.
- **앵커**: `so101_contract/grasp_manifold.py` 전체 · `_self_check` ⑤

### 16.2 main SM 이 99.9% 인 이유 — 오차 큰 후보를 버려서

- **결정**: cuRobo batch planner 의 성공률 99.9% 는 IK 오차를 줄인 것이 아니라 **오차 큰
  후보를 게이트로 버린 것**이다.
- **동작**: `_gate_candidate` 가 **IK 해의 FK 를 다시 계산**해 파지 기하를 검사한다:
  `e_n ∈ [2, 8] mm` · `|e_t| ≤ 22` · `|e_h| ≤ 28` · `|face| ≤ 40°` · `|Δwrist_roll| ≤ 100°`.
  실패하면 다음 후보, clearance 만 어긋나면 조준 되먹임한다.
- **주의**: 게이트는 **해/궤적 끝점** 기준이어야 한다. plan 목표 시점 게이트는 판별력이
  없다(목표 산포는 source 수준이고 벗어나는 건 실행이다). 그 임계로 조기 중단했으면 **성공할
  시도의 90%를 죽였을 것**.
- **앵커**: `scripts/cuRobo/curobo_batch_planner.py::_gate_candidate`

### 16.3 ★ 재시도 금지 목록 — 측정 결과 무효화된 접근법 10건

실제 데이터로 반증된 것들이다. 같은 실수를 반복하지 말 것:

| 시도 | 결과 | 근거 |
|---|---|---|
| 대칭 인지 source 선택(회전을 길이로 환산) | **81.6 → 22.2%** | offset 사다리 vs 분산 오프라인 지표 비정합 |
| `rot_weight` 0.05 (회전 경시) | **76.1 → 52~57%** | orientation 이 5-DOF 도달성 절벽 결정 |
| Δψ 인지 source 선택 | **83.3% → 70.2%** (대표본) | 소표본 오버핏 |
| source 풀 24→40→80 | **전부 70.2%** (대표본 동일) | 다양성 ceiling |
| `(α, ρ)` 구간 내 동결 | **46.7 → 16.7%** | 알파+로 제약 과강 |
| `n_repeat` 축소 | **stall 불변** + 재개 충격 악화 | 무차원 스케일 반사슷 |
| `num_interpolation_steps` 15→40 | **0.29 → 0.26%** (정체) | 명령 포화 스캔 해상도 한계 |
| replay retiming | **+11.2%p → +2.1%p** (seed 다름) | seed 간 분산 >seed 내 수정 효과 |
| tcp 위치 보존 보정 | **3 지표 전부 악화** | 목적함수 왜곡 |
| `wrist_roll` 축별 clamp 0.3× | **평균 0.037 → 0.625 rad/s 악화** | clamp 는 부드럽기 도구 아님 |

**결론**: 세 번 독립 확증 — **"위치를 가깝게"는 틀린 목적함수다**. 어느 face 를 무느냐(회전)가
5-DOF 도달성을 가르고 그게 성공률 절벽을 만든다.

### 16.4 ★ A/B 방법론 — 단일 seed 판정 금지

- **교훈**: retry retiming 이 seed 41000 에서 +11.2%p → seed 42000 에서 +2.1%p 로 무너졌다.
- **정책**: **최소 2 seed × 2 조건(4칸 matrix)**. seed 간 분산이 조건 간 차이보다 크다.
- **이유**: attempt 간 chaotic behavior(밤뮤 여역률) — 개루프 재생이 미세한 물리 변동에 극민감.

### 16.5 측정 지표가 오판을 낸 사례 — 명령 포화율이 정답

- **잘못된 지표**: 물리 joint 속도의 "재개 충격"(0→max 가속도).
  - 보간 0→15: 물리 **25.1 → 27.4 rad/s²**(악화), 명령 **1.89 → 0.29%** 포화(개선)
  - 사람 관찰: "확 튐" = slew limiter 가 cap 에 붙어 최대속도 램프를 만드는 구간.
- **정답**: **명령 포화율**(post-slew target 이 per-joint cap 에 붙은 프레임 비율).
- **교훈**: 지표 선택이 결론을 결정한다. 물리 센서도 신뢰성 보증 구간만 쓴다.

### 16.6 ★ 성공률 측정의 함정 — OR-latch 오탐

- **결정**: 공식 generator 는 성공을 에피소드 전체에 **OR-latch** 한다
  (`isaaclab_mimic/datagen/data_generator.py:886`,  `generated_success = generated_success or ...`).
- **증상**: 한 프레임만 조건을 만족하면 성공으로 굳고 종료 시점에 재확인하지 않는다. 큐브가
  그릇 반경을 잠깐 지나갔다가 밀려남 = **성공으로 집계**(50 demo 중 4건 실측, 그릇을 40.7°
  엎은 것 포함).
- **대응**: `validate_place_success.py` 로 **종료 시점 재검증** 필수(§5.9.5). 기존 81.6%
  는 오탐 미보정 값이다.
- **앵커**: `scripts/contract/validate_place_success.py`

### 16.7 큐브 mute 는 구조적으로 필요하다

- **증상**: 파지 목표는 정의상 손가락이 큐브를 감싼 자세라 cuRobo 가 `scene_collision` 으로
  계획을 거부한다(모든 실패 64건 중 **45건(70%)** = START 깨끗·GOAL `scene_collision`,
  전부 파지 전 approach 전이).
- **시도**: `max_attempts` 3→8 올림 → 68%에 머뭄 (예산 문제 아니라 **목표 불가능**).
- **해법**: `cubes.cubes[0].disable()` 으로 grasp 계획 시 mute, approach 후 enable.
- **사후 검사**: retiming 이후 큐브를 켠 상태로 waypoint 별 사후 충돌 검사. 꼬리(파지 접근)
  밖 관통을 잡는다(full post-eval 아님 — 성능상).
- **앵커**: `src/sim_to_real/datagen/skillgen_planner.py::attach_detach_cube`

### 16.8 꼭짓점 서기 수식 — 반 두 번은 1/4다

- **실수**: `s·√3/4` (정확 `s·√3/2 ÷ 2`) 는 값이 **절반**이다. 40 mm 에서 17.3 mm 로
  계산되면 들림 게이트가 무력해진다(검증기 assert 도 못 막음).
- **정답**: `s·√3/2` + margin.
- **단일 소스**: `pick_cube/mdp/observations.py::min_lift_for_cube` 하나만 수정한다.
- **앵커**: `so101_contract/grasp_manifold.py` 주석 + `src/sim_to_real/tasks/pick_cube/mdp/observations.py`

### 7.6 `cube_specs.py` 는 stdlib 전용

- **근거**: author 스크립트가 AppLauncher 부팅 **전에** importlib 로 이 파일만 직접 로드한다.
  상대 import 가 있으면 패키지 `__init__`(→ isaaclab)이 끌려온다.
- **앵커**: `src/sim_to_real/utils/cube_specs.py`

---
