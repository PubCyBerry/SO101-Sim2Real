# 05. 데이터 명세 — 데이터셋 스키마·변환

> **정본**: `src/sim_to_real/data/`, `src/sim_to_real/tasks/common/mdp/recorders.py`,
> `scripts/convert/`. 단위는 `04_IO_CONTRACT.md`, 관측 출처는 `03_ENV_SPEC.md §3` 을 전제한다.

---

## 1. 포맷 2종

| | LeRobot Dataset v3 | Isaac Lab HDF5 |
|---|---|---|
| 역할 | **정본** — 학습·업로드·재생 | 중간 산출물 |
| 생성 | `LeRobotV3DatasetWriter` (직기록) | IsaacLab `RecorderManager` |
| multi-env | ✗ single-env 전용 | ✓ env 당 1 demo |
| 저장 범위 | **성공 에피소드만** | 실패도 저장(`success` attr 로 구분) |
| 보존 정보 | frame(action/state/3-cam) | 전체 씬 state(`states`·`initial_state`) — replay·재라벨 가능 |
| 메모리 | step 마다 CPU 스트리밍 | 에피소드 동안 이미지 누적 |
| 후처리 | 없음 | `scripts/convert/isaaclab2lerobotv3.py` 로 v3 변환 |

두 경로 모두 **같은 writer 백엔드**로 수렴하므로 스키마 계약이 하나다.

---

## 2. LeRobot v3 디렉터리 레이아웃

앵커: `src/sim_to_real/data/lerobot_recorder.py::LeRobotV3DatasetWriter`

```
<root>/
├── data/chunk-000/file-000.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    ├── observation.images.wrist/chunk-000/file-000.mp4
    └── observation.images.front/chunk-000/file-000.mp4
```

경로 템플릿(`meta/info.json`에 기록):
`data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` ·
`videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`

### 2.1 생성자 파라미터

| 인자 | 기본값 | 비고 |
|---|---|---|
| `output_dir` | (필수) | |
| `overwrite` | `False` | `True` 여도 **안전장치**: `/`·홈·cwd·`len(parts) < 4` 는 삭제 거부 |
| `enable_videos` | `True` | |
| `robot_type` | `"so_follower"` | `info.json` 및 스키마 검증 기준 |
| `video_quality` | `8` | `[0, 10]` 밖이면 `ValueError` |
| `video_codec` | `"libx264"` | `{"libx264", "h264_nvenc"}` 만 허용 |
| `video_preset` | `None` | |
| `video_ffmpeg_exe` | `None` | 지정 시 `IMAGEIO_FFMPEG_EXE` 주입 — 번들 ffmpeg 는 NVENC 미포함인 경우가 많다 |
| `video_nvenc_cq` | `23` | `[0, 51]` 밖이면 `ValueError`. NVENC 는 qscale 대신 CQ 사용 |

### 2.2 API

| 메서드 | 동작 |
|---|---|
| `add_frame(action, state, images)` | 현재 에피소드 버퍼에 1 프레임. caller 가 **단위 변환·이미지 캡처를 마친 상태**로 넘긴다 |
| `commit_episode(success, task_name)` | `success and length > 0` 일 때만 flush. 버퍼는 **항상** 비운다. flush 여부 반환 |
| `finalize(task_name=None)` | 비디오 close + meta 5종 기록. 요약 dict 반환 |

기본 task 문자열 = `"pick up the cube and place it in the bowl"`.

---

## 3. `data/**.parquet` 스키마

| 컬럼 | Arrow 타입 | 값 |
|---|---|---|
| `action` | `fixed_size_list<float32>[6]` | policy-feature (arm degree, gripper `[0,100]`) |
| `observation.state` | `fixed_size_list<float32>[6]` | 동일 단위 |
| `timestamp` | `float32` | `frame_index / 30` |
| `frame_index` | `int64` | 에피소드 내 0-based |
| `episode_index` | `int64` | |
| `index` | `int64` | 데이터셋 전역 0-based |
| `task_index` | `int64` | 현재 항상 `0` (단일 task) |

---

## 4. `meta/` 스키마

### 4.1 `info.json`

| 키 | 값 |
|---|---|
| `codebase_version` | `"v3.0"` |
| `robot_type` | 생성자 인자 (기본 `"so_follower"`) |
| `total_episodes` / `total_frames` | 집계값 |
| `total_tasks` | `1` |
| `chunks_size` | `1000` |
| `data_files_size_in_mb` / `video_files_size_in_mb` | `100` / `200` |
| `fps` | `30` |
| `splits` | `{"train": "0:<total_episodes>"}` |
| `data_path` / `video_path` | §2 템플릿 |
| `features` | 아래 |

`features`:

| key | dtype | shape | names |
|---|---|---|---|
| `action` | `float32` | `[6]` | `JOINT_FEATURE_NAMES` (`shoulder_pan.pos` … `gripper.pos`) |
| `observation.state` | `float32` | `[6]` | 동일 |
| `observation.images.{top,wrist,front}` | `video` | `[480, 640, 3]` | `["height","width","channels"]` |
| `timestamp` | `float32` | `[1]` | `None` |
| `frame_index` · `episode_index` · `index` · `task_index` | `int64` | `[1]` | `None` |

각 카메라 feature 의 `info` 블록:

| 키 | 값 |
|---|---|
| `video.height` / `video.width` | `480` / `640` |
| `video.codec` | `"h264"` |
| `video.pix_fmt` | `"yuv420p"` |
| `video.fps` | `30` |
| `video.channels` | `3` |
| `video.is_depth_map` / `has_audio` | `False` / `False` |

### 4.2 `tasks.parquet`

> ⚠ **pandas 로 써야 한다.** LeRobot v3 는 이 파일을 pandas DataFrame(**인덱스 = task 문자열**,
> 컬럼 = `task_index`)으로 읽어 `task_index → 문자열` 을 룩업한다. pyarrow 로 직접 쓰면 pandas
> index 메타데이터가 없어 룩업이 깨진다(`"Task cannot be None"`).

현재 구현은 `pd.DataFrame({"task_index": [0]}, index=[task_name])` 1행이다.

### 4.3 `meta/episodes/**.parquet`

| 컬럼 | 타입 |
|---|---|
| `episode_index` · `length` | `int64` |
| `tasks` | `list<string>` |
| `data/chunk_index` · `data/file_index` | `int64` (현재 항상 0) |
| `dataset_from_index` · `dataset_to_index` | `int64` (전역 index 구간) |
| `videos/observation.images.{cam}/chunk_index` · `file_index` | `int64` |
| `videos/observation.images.{cam}/from_timestamp` · `to_timestamp` | `float64` |
| `meta/episodes/chunk_index` · `file_index` | `int64` |

비디오 타임스탬프 = `dataset_from_index / 30` ~ `dataset_to_index / 30`.

### 4.4 `stats.json`

수치 키(`action`·`observation.state`·`timestamp`·`frame_index`·`episode_index`·`index`·`task_index`):
`min`·`max`·`mean`·`std`·`count`·`q01`·`q10`·`q50`·`q90`·`q99` (컬럼별 리스트).
빈 데이터셋이면 전부 0, `count = [0]`.

이미지 키(`observation.images.{cam}`)는 `ImageStats` 가 만든다:

- **표본화**: `sample_stride = 8` 규칙 격자(1/64 표본). 전 픽셀을 float64 로 돌리면 통계 계산이
  video 인코딩보다 느려지는데, 이 표본이면 채널 mean/std 오차가 충분히 작다.
- 값 범위는 `[0, 1]`(uint8 을 255로 나눔), 중첩 리스트 `[[[v]]]` 형태
- **분위수는 근사**: `q01 = min`, `q10 = q50 = q90 = mean`, `q99 = max`
- 프레임이 없으면 `mean = 0.5`, `std = 0`, `min = 0`, `max = 1`

---

## 5. 기록 경로 2종

### 5.1 `SO101LeRobotRecorderManager` (직기록)

앵커: `src/sim_to_real/data/lerobot_recorder_manager.py`

leisaac `--use_lerobot_recorder` 동형. 부모 `RecorderManager` 를 `DatasetExportMode.EXPORT_NONE`
로 위장 초기화해 HDF5 file handler 생성을 막고, 백엔드로 `LeRobotV3DatasetWriter` 를 쓴다.

**`env.num_envs == 1` assert** — multi-env 는 `--record_hdf5` 경로를 써야 한다.

`record_post_step` 에서 프레임 1개를 조립한다:

| 항목 | 소스 | 변환 |
|---|---|---|
| `state` | `robot.data.joint_pos[0, joint_idx]` (절대 radian) | `to_lerobot_units` |
| `action` | `arm.processed_actions[0, action_idx]` (**slew 통과 후** target) | `to_lerobot_units` |
| `images` | `env.obs_buf["images"][cam][0]` | 그대로 uint8 |

> `obs_buf` 는 이 시점에 아직 `obs_t` 다(obs 갱신은 step 말미) → `obs_t` + `action_t` 정합.

**인덱스는 이름으로 매핑한다** — `JointAction` 은 `preserve_order=False` 가 기본이라 action
컬럼이 asset 순서일 수 있다. `robot.joint_names` / `arm._joint_names` 에서
`SO101_JOINT_ORDER` 순서를 역인덱싱한다.

`reset()` 은 미커밋 버퍼를 폐기한다(record 루프의 쓰레기 프레임 절삭).

### 5.2 `DatagenRecorderTerm` (HDF5)

앵커: `src/sim_to_real/tasks/common/mdp/recorders.py`

정책 obs 가 `joint_pos_rel`(+concat)이라 stock `ActionStateRecorderManagerCfg` 만으로는
**절대 joint 각과 카메라 이미지가 HDF5 에 남지 않는다.** 이 term 이 캐노니컬 형태로 직접
기록해 변환기가 Isaac 부팅 없이 동작하게 한다.

| HDF5 키 | shape | 단위·시점 |
|---|---|---|
| `obs_x/joint_pos` | `(T, 6)` | 절대 **radian**, SO101 순서, pre-step (`obs_t`) |
| `obs_x/images/{top,wrist,front}` | `(T, H, W, 3)` | uint8, pre-step |
| `applied_target` | `(T, 6)` | **slew 통과한 실제 적용** joint target radian, post-step |

`SO101DatagenRecorderManagerCfg` = stock 5종(`initial_state`/`states`/`actions`/`obs`/
`processed_actions`) + 이 term.

> **왜 `processed_actions` 인가**: pre-slew raw action 을 BC target 으로 쓰면 물리적으로 불가능한
> teleport 가 학습 타깃에 들어가 jerky 데이터가 된다. 근거 = `09_TACIT_KNOWLEDGE.md §4`.

demo attrs: `success`(bool) · `num_samples` · `seed`.

### 5.3 단위 변환 헬퍼

앵커: `src/sim_to_real/data/lerobot_units.py` — `feature_codec` 에 위임하는 얇은 층이다.

| 심볼 | 내용 |
|---|---|
| `to_lerobot_units` / `from_lerobot_units` | `sim_joint_radians_to_policy_feature` / 역 |
| `CAMERA_SCENE_NAMES` | `{"top": "top_camera", "wrist": "wrist_camera", "front": "front_camera"}` |
| `read_camera_rgb_u8(raw_env, scene_name)` | RGBA→RGB 드롭, float`[0,1]`→uint8, `(480,640,3)` shape 검증 |

---

## 6. 변환기

### 6.1 `isaaclab2lerobotv3.py` — HDF5 → LeRobot v3

**env-free**: Isaac Sim 도 lerobot 패키지도 불필요하다. h5py lazy slice 라 에피소드를 통째
로드하지 않는다. `LeRobotV3DatasetWriter` 를 importlib 파일 로드로 가져와 패키지 `__init__`
(→ isaaclab)을 우회한다.

| 인자 | 기본값 |
|---|---|
| `--hdf5_files` | (필수, 쉼표 구분 복수) |
| `--output_dir` | (필수) |
| `--task` | `pick up the cube and place it in the bowl` |
| `--skip_frames` | `0` — 에피소드 머리/꼬리가 녹화 시점에 이미 규격이라 |
| `--min_frames` | `10` |
| `--include_failed` | off — 기본은 `success=True` 만 |
| `--overwrite` | off |

동작:

- 프레임 정합: `n = min(len(applied_target), len(joint_pos), *len(images))`
- `n − skip_frames < min_frames` 면 스킵
- demo 정렬은 **숫자 기준** — 사전순이면 `demo_10` 이 `demo_2` 앞에 온다
- 단위: `sim_joint_radians_to_policy_feature` (단일 소스)

### 6.2 `sim_dataset_to_real_follower.py` — sim frame → real follower frame (**in-place**)

sim-프레임 v3 데이터셋을 실기기 `lerobot-replay` 로 재생하기 위한 변환. Isaac 무의존.

| 항목 | 내용 |
|---|---|
| 변환 | `follower_calibration.policy_feature_to_real_follower` (`04_IO_CONTRACT.md §4`) |
| 대상 | `--convert {action, state, both}` (기본 `both`) |
| 갱신 | `data/**/*.parquet` 덮어쓰기 + `meta/stats.json` |
| 불변 | `videos/`, `meta/info.json`, `tasks`, `episodes` |
| 게이트 | `info.json` 의 feature `shape != [6]` 이면 `ValueError` |
| self-check | `--self-check` (arm offset 부호 + 배치 `(N,6)` 정합) |

> ⚠ 실기기 replay 는 잘못된 관절 타깃 = **충돌 위험**이다. e-stop 준비하고 실행할 것
> (스크립트 docstring 경고).

### 6.3 `joint_dataset_to_eef.py` — joint-space → absolute EEF-space

원본 데이터셋을 **보존**하고 별 디렉터리에 파생본을 만든다. FK = `04_IO_CONTRACT.md §5`.

| `--rotation-representation` | layout | dim |
|---|---|---:|
| `rot6d` (기본) | `[tcp xyz(3), R 첫 두 row(6), gripper(1)]` | **10** |
| `rpy` | `[tcp xyz(3), fixed-axis RPY radian(3), gripper(1)]` | **7** |
| `wxyz` | `[tcp xyz(3), canonical quat wxyz(4), gripper(1)]` | **8** |

`--keep-joints` 를 주면 위 layout **뒤에** arm joint radian 5개가 붙는다 → 15 / 12 / 13D.

| 인자 | 기본값 |
|---|---|
| `--input-dir` · `--output-dir` · `--source-domain` | `--self-check` 없으면 **전부 필수** |
| `--source-domain` | `{sim, real}` — 입력 단위계를 **자동 판별하지 않는다** |
| `--rotation-representation` | `rot6d` |
| `--video-mode` | `hardlink` (실패 시 copy) |
| `--urdf` / `--robot-yaml` | `assets/robots/urdf/so_arm101.urdf` / `assets/robots/so101.yml` |
| `--keep-joints` · `--overwrite` · `--self-check` | off |

좌표 계약:

- `--source-domain sim` → 입력 = policy feature (arm sim degree, gripper `[0,100]`)
- `--source-domain real` → 입력 = real follower (arm follower degree, gripper `[0,100]`).
  **real gripper 도 canonical sim policy feature `[0,100]` 으로 변환해 출력**한다.
- 두 입력 모두 calibration 후 **같은 URDF joint radian + `base_link→tcp_grasp` FK** 를 쓴다.

갱신 항목: `meta/{info, stats, modality}` + per-episode stats.

> **absolute 값만 저장한다.** EEF-relative 학습·추론에는 `T_rel = inv(T_state) @ T_action`,
> `T_abs = T_state @ T_rel` 을 수행하는 SE(3) processor 가 별도로 필요하다 — 단순 벡터
> 뺄셈/덧셈은 회전에 부적합하다. 상세 = `04_IO_CONTRACT.md §10`.

`--self-check`: 합성 sim/real v3 로 전체 변환 + 세 회전 표현 round-trip + source 보존 +
meta 갱신 검증.

### 6.4 사후 ops (변환 아님)

| 스크립트 | 용도 |
|---|---|
| `scripts/data/append_sim_episode.py` | replay 로 얻은 achieved 에피소드를 기존 v3 에 append (`--record_dir`·`--repo_id` 필수) |
| `scripts/data/upload_to_huggingface.py` | HF 업로드 + `codebase_version` 태그 자동 생성/이동. `.env` 의 `HF_TOKEN`/`HF_USER` 사용 |

---

## 7. 스키마 검증

앵커: `scripts/contract/validate_lerobot_schema.py`

```bash
python scripts/contract/validate_lerobot_schema.py <dataset_root>
python scripts/contract/validate_lerobot_schema.py --self-test
```

exit 0 = PASS(오류 0건, 경고는 허용) · exit 1 = 오류 ≥1 또는 경로 문제.

### 7.1 합격 기준 상수

| 상수 | 값 |
|---|---|
| `EXPECTED_CODEBASE_VERSION` | `"v3.0"` |
| `EXPECTED_ROBOT_TYPE` | `"so_follower"` |
| `EXPECTED_FPS` | `30` |
| `EXPECTED_JOINT_NAMES` | `shoulder_pan.pos` · `shoulder_lift.pos` · `elbow_flex.pos` · `wrist_flex.pos` · `wrist_roll.pos` · `gripper.pos` |
| `REQUIRED_CAMERAS` | `["top", "wrist"]` — 없으면 **ERROR** |
| `OPTIONAL_CAMERAS` | `["front"]` — 없으면 **WARNING** |
| `EXPECTED_TASK` | `"pick up the cube and place it in the bowl"` (`--expected-task` 로 override) |
| `EXPECTED_CAMERA_INFO` | `video.codec=h264` · `fps=30` · `channels=3` · `height=480` · `width=640` |
| `REQUIRED_DATA_COLS` | `action` · `observation.state` · `timestamp` · `frame_index` · `episode_index` · `index` · `task_index` |

### 7.2 검사 항목

| 검사 | 내용 |
|---|---|
| `validate_info` | 파일 존재 · `codebase_version`/`robot_type`/`fps` **완전일치** · `action`·`observation.state` 의 `dtype=="float32"`, `shape==[6]`, `names` 일치 · 카메라별 `dtype=="video"`, `shape==[480,640,3]`, `info` 5키 완전일치 |
| `validate_tasks` | `meta/tasks.parquet` 에 `task_index`·`__index_level_0__` 존재 · 행 수 1 · `task_index[0]==0` · 인덱스 문자열 == 기대 task |
| `validate_data_parquet` | `data/**/file-*.parquet` **첫 파일만** — 필수 컬럼 전부 존재 · `action`/`observation.state` 타입이 `fixed_size_list<float32>[6]` |

`--self-test`: 유효 픽스처 1건 + 오류 검출 5케이스(누락 파일 / 버전 불일치 / joint names /
data 컬럼 누락 / action shape 3)를 모두 검출해야 통과. pyarrow 미설치면 SKIP + exit 1.

---

## 참조

- 단위·codec → `04_IO_CONTRACT.md` §2, §4, §5
- 기록되는 관측·액션의 출처 → `03_ENV_SPEC.md` §3, §4
- 데이터 생성 워크플로 전체 → `08_PIPELINES.md`
- 기록 관련 함정 → `09_TACIT_KNOWLEDGE.md` §4
