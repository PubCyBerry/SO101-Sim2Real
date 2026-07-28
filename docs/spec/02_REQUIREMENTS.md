# 02. 요구사항 — 코드에서 역추출

> 이 문서는 **코드가 이미 강제하는 것**을 요구사항 문장으로 옮긴 것이다. 새로 만들어낸
> 요구사항은 없다. 근거 앵커가 없는 항목은 싣지 않는다.

---

## 1. 도출 방법

| 코드 패턴 | → | 유형 |
|---|---|---|
| `raise` / `assert` / `SystemExit` / `parser.error` | 위반이 실행을 막음 | **FR** (E1) |
| 상수·시그니처·스키마로 고정된 값 | 계약 | **FR** (E2) |
| 측정 결과가 코드/문서에 기록됨 | 실측 | **NFR** (E3) |
| ⚠ 주석만 있고 강제는 없음 | 관행 | **FR/NFR** (E4) |
| 외부 요인(ABI·HW·알고리즘 한계) | 선택 불가 | **CON** |
| 검증기의 통과 기준 | 합격선 | **AC** |

### 근거 등급 (E-grade)

| 등급 | 정의 | 위반 시 |
|---|---|---|
| **E1** 강제 | 예외·빌드 중단 | 즉시 실패 |
| **E2** 계약 | 상수·스키마로 고정 | 조용히 어긋남 |
| **E3** 정량 | 측정 근거 존재 | 성능 회귀 |
| **E4** 관행 | 주석만 | 재발 |

### ID 규칙

`FR-<도메인>-<nn>` — 도메인 `IO`(→04) `ENV`(→03) `DATA`(→05) `RT`(→06) `IF`(→07) `PIPE`(→08) ·
`NFR-<범주>-<nn>` — 범주 `COMPAT` `REPRO` `PERF` `SAFE` `OBS` ·
`CON-<nn>` · `AC-<nn>`.

**`검증` 열이 빈 항목은 미검증 요구사항**이며 §9 에 모은다.

`raise`/`assert`/`SystemExit`/`parser.error` 지점은 `src`+`scripts`(ece_4560 제외) 32개 파일
**148곳**이다. 아래는 그중 시스템 계약에 해당하는 것을 정리한 것이다.

---

## 2. FR — I/O 계약

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-IO-01 | 관절 순서는 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` 로 전 시스템에서 동일해야 한다 | E2 | `src/so101_contract/feature_codec.py::SO101_JOINT_ORDER` | `validate_so101_io_contract.py` | 04 §2.1 |
| FR-IO-02 | 관절 배열은 마지막 축이 6이고 유한값이어야 하며, 아니면 `ValueError` 를 내야 한다 | E1 | `feature_codec.py::_as_joint_array` | 동상 | 04 §2.3 |
| FR-IO-03 | arm 5축은 policy-feature ↔ sim 간 degree ↔ radian 1:1 이어야 한다 | E2 | `feature_codec.py::sim_joint_radians_to_policy_feature` | 동상 | 04 §2.2 |
| FR-IO-04 | gripper 는 `sim_deg = feature/100 × 110 − 10` affine 이어야 한다(경계 0→−10°, 100→100°) | E2 | `feature_codec.py::POLICY_GRIPPER_RANGE`, `::SIM_GRIPPER_RANGE_DEG` | 동상 | 04 §2.2 |
| FR-IO-05 | action 은 **절대 joint target** 이어야 한다 (offset 0) | E2 | `so101_base_env_cfg.py::SO101ActionsCfg` (`use_default_offset=False`) | — | 04 §2.2 · 09 §4.4 |
| FR-IO-06 | 실 leader ↔ sim 변환은 **관절별 비대칭 USD limit** 을 반영한 per-joint 선형 remap 이어야 한다 | E2 | `leader_calibration.py::SO101_FOLLOWER_USD_JOINT_LIMITS` | — | 04 §3.2 |
| FR-IO-07 | 실 follower ↔ sim 변환은 affine 1개 + 역산으로 양방향을 지원해야 한다 | E2 | `follower_calibration.py::FOLLOWER_AFFINE_A`/`_B` | `python3 follower_calibration.py` | 04 §4 |
| FR-IO-08 | follower affine 의 no-op 기준값은 `feature_codec` 과 수치적으로 동일해야 한다 | E1 | `follower_calibration.py::_self_check` (assert) | 동상 | 04 §4.2 |
| FR-IO-09 | EEF FK 는 `base_link → tcp_grasp` absolute pose 를 내야 하며, URDF 가동 관절 순서가 `ARM_JOINT_ORDER` 와 다르면 `ValueError` 를 내야 한다 | E1 | `eef_kinematics.py::SO101EndEffectorKinematics.__init__` | `joint_dataset_to_eef.py --self-check` | 04 §5 |
| FR-IO-10 | `tcp_grasp.fixed_transform` 은 `[xyz, q_wxyz]` 길이 7이어야 한다 | E1 | `eef_kinematics.py::from_files` | 동상 | 04 §5.2 |
| FR-IO-11 | quaternion 출력은 canonical hemisphere(`w ≥ 0`)로 고정해야 한다 | E2 | `eef_kinematics.py::_matrix_to_quaternion_wxyz` | 동상 | 04 §5.1 |
| FR-IO-12 | action chunk 병합은 `timestep ≤ latest_action` 을 버리고 겹치는 timestep 만 aggregate 해야 한다 | E2 | `action_queue.py::ActionChunkQueue.merge` | `validate_so101_io_contract.py` | 04 §6 |
| FR-IO-13 | 미지 aggregate 이름·shape 불일치는 `ValueError` 여야 한다 | E1 | `action_queue.py::aggregate_actions` | 동상 | 04 §6 |
| FR-IO-14 | 스냅샷 로드는 snapshot·codec 버전 불일치와 joint feature 누락을 거부해야 한다 | E1 | `policy_snapshot.py::load_policy_io_snapshot` | `replay_so101_policy_snapshot.py` | 04 §7 |
| FR-IO-15 | `JOINT_FRAME_MODE` 는 4-case 만 허용하고 미지 값이면 기동 실패해야 한다 | E1 | `scripts/inference/policy_server_affine.py::_MODES` | — | 04 §8 |
| FR-IO-16 | 프레임 어댑터는 `observation.state` 와 `action` 만 변환하고 **이미지는 변환하지 않아야** 한다 | E2 | `policy_server_affine.py::_enqueue_observation`, `::_predict_action_chunk` | — | 04 §8 |

---

## 3. FR — 시뮬레이션 환경

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-ENV-01 | Gym 환경 6종을 등록해야 한다 (Teleop-v0 + PickCube 5종) | E2 | `src/sim_to_real/tasks/pick_cube/__init__.py::_PICK_CUBE_VARIANTS` | `list_envs.py` | 03 §2 |
| FR-ENV-02 | DR-off 가 기본이어야 한다(`PickCube-v0` = 고정 실측 배치) | E2 | `pick_cube_env_cfg.py::PickCubeEnvCfg` | — | 03 §2 |
| FR-ENV-03 | policy 관측은 6-dim joint position 하나여야 한다 | E2 | `so101_base_env_cfg.py::SO101PolicyObservationsCfg` | — | 03 §3 |
| FR-ENV-04 | 관측에 noise·clip 을 넣지 않아야 한다(`enable_corruption=False`) | E2 | 동상 · `pick_cube_env_cfg.py::PickCubeObservationsCfg` | — | 03 §3 |
| FR-ENV-05 | 이미지 관측은 uint8 원본 `(480, 640, 3)` 이어야 한다(`normalize=False`) | E2 | `pick_cube_env_cfg.py::VisualCfg` | — | 03 §3 |
| FR-ENV-06 | action 은 slew-limited joint position 6D 여야 하며, per-step 상한 = `max_velocity × sim.dt × decimation` 이어야 한다 | E2 | `tasks/common/mdp/actions.py::SlewLimitedJointPositionAction` | — | 03 §4.1 |
| FR-ENV-07 | 리셋 후 첫 step 은 현재 자세에서 출발해야 한다(slew 상태 재설정) | E2 | `actions.py::SlewLimitedJointPositionAction.reset` | — | 03 §4.1 |
| FR-ENV-08 | 정책 제어 주파수는 30 Hz(`sim.dt 1/120` × `decimation 4`)여야 한다 | E2 | `so101_base_env_cfg.py::SO101TeleopEnvCfg.__post_init__` · `feature_codec.py::FPS` | — | 03 §5 |
| FR-ENV-09 | grasp 신호는 양 손가락 contact + 들림 + hysteresis + warmup 을 만족해야 한다 | E2 | `tasks/pick_cube/mdp/observations.py::any_cube_grasped` | `python3 <파일>` | 03 §3.2 |
| FR-ENV-10 | Eval 변형은 성공을 N-step 디바운스해야 한다(`confirm_steps = 15`) | E2 | `tasks/pick_cube/mdp/terminations.py::task_done_confirmed` | — | 03 §10.2 |
| FR-ENV-11 | 큐브 스폰 영역은 4개 조건(bell ∧ ¬arm_exclude ∧ bowl_sep ∧ base_sep)을 모두 만족해야 한다 | E2 | `tasks/pick_cube/spawn_area.py::in_spawn_area` | `python3 <파일>` | 03 §11.2 |
| FR-ENV-12 | min-reach 판정 중심은 **shoulder_pan 축**이어야 한다(마운트 원점 아님) | E3 | `spawn_area.py::PAN_AXIS_XY` | sweep 183/183 | 09 §3.1 |
| FR-ENV-13 | 스폰 영역 기하는 env cfg · sweep · plot 이 **한 모듈**을 공유해야 한다 | E2 | `spawn_area.py` 모듈 docstring | `python3 <파일>` | 03 §11.2 |
| FR-ENV-14 | 그릇 DR 은 큐브 배치보다 **먼저** 적용돼야 한다 | E4 | `domain_randomization.py::randomize_cubes_scattered` docstring | — | 09 §5.1 |
| FR-ENV-15 | robot color DR 은 `replicate_physics=False` 를 요구하고, 아니면 `RuntimeError` 를 내야 한다 | E1 | `domain_randomization.py::_RandomizeRobotColor.__init__` | — | 09 §5.3 |
| FR-ENV-16 | 조명은 `/World` 계층에 env 당 복제 없이 단일 배치해야 한다 | E2 | `so101_base_env_cfg.py::SO101BaseSceneCfg` | — | 09 §5.2 |
| FR-ENV-17 | 큐브 크기·질량은 단일 소스에서만 정의해야 한다 | E2 | `src/sim_to_real/utils/cube_specs.py::CUBE_SPECS` | — | 03 §9.1 |
| FR-ENV-18 | gripper effort 는 인접 물체 질량으로 매 step clamp 돼야 한다 | E2 | `utils/gripper_effort.py::dynamic_reset_gripper_effort_limit_sim` | — | 03 §6.2 |
| FR-ENV-19 | 무카메라 실행은 카메라와 `images` 관측을 **함께** 제거해야 한다 | E2 | `pick_cube_env_cfg.py::remove_pick_cube_cameras` | — | 09 §6.3 |

---

## 4. FR — 데이터

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-DATA-01 | 데이터셋은 LeRobot v3(`codebase_version = "v3.0"`)여야 한다 | E2 | `data/lerobot_recorder.py::_write_info` | `validate_lerobot_schema.py` | 05 §4.1 |
| FR-DATA-02 | `action`·`observation.state` 는 `fixed_size_list<float32>[6]` 이어야 한다 | E2 | `lerobot_recorder.py::_write_data_parquet` | 동상 | 05 §3 |
| FR-DATA-03 | feature 이름은 `<joint>.pos` 6개여야 한다 | E2 | `feature_codec.py::JOINT_FEATURE_NAMES` | 동상 | 05 §4.1 |
| FR-DATA-04 | 카메라 3종은 `video` dtype `[480, 640, 3]`, h264/yuv420p/30fps 여야 한다 | E2 | `lerobot_recorder.py::_write_info` | 동상 | 05 §4.1 |
| FR-DATA-05 | `tasks.parquet` 은 pandas 인덱스(task 문자열) 형태여야 한다 | E2 | `lerobot_recorder.py::_write_tasks` | 동상 | 05 §4.2 |
| FR-DATA-06 | LeRobot 직기록은 **성공 에피소드만** 저장해야 한다 | E2 | `lerobot_recorder.py::commit_episode` | — | 05 §2.2 |
| FR-DATA-07 | LeRobot 직기록은 single-env 전용이어야 한다 | E1 | `data/lerobot_recorder_manager.py` (assert) | — | 05 §5.1 |
| FR-DATA-08 | 기록되는 action 은 **slew 통과 후** target 이어야 한다 | E2 | `tasks/common/mdp/recorders.py::record_post_step` · `lerobot_recorder_manager.py` | — | 09 §4.3 |
| FR-DATA-09 | HDF5 는 절대 joint radian·3-cam uint8·적용 target 을 캐노니컬 키로 남겨야 한다 | E2 | `recorders.py::DatagenRecorderTerm` | `isaaclab2lerobotv3.py` | 05 §5.2 |
| FR-DATA-10 | action/joint 컬럼은 **이름 기준**으로 SO101 순서에 매핑해야 한다 | E2 | `recorders.py::_resolve_indices` (assert) | — | 05 §5.1 |
| FR-DATA-11 | HDF5→v3 변환은 Isaac·lerobot 없이 동작해야 한다 | E2 | `scripts/convert/isaaclab2lerobotv3.py` (importlib 파일 로드) | `validate_lerobot_schema.py` | 05 §6.1 |
| FR-DATA-12 | 두 기록 경로는 **동일 writer = 동일 스키마**를 써야 한다 | E2 | `isaaclab2lerobotv3.py` docstring | 동상 | 05 §1 |
| FR-DATA-13 | EEF 변환은 원본을 보존하고 별 디렉터리에 파생본을 만들어야 한다 | E2 | `scripts/convert/joint_dataset_to_eef.py` | `--self-check` | 05 §6.3 |
| FR-DATA-14 | EEF 변환 입력 단위계(`--source-domain`)는 **명시**해야 한다(자동 판별 금지) | E1 | `joint_dataset_to_eef.py` (`parser.error`) | — | 05 §6.3 |
| FR-DATA-15 | 데이터셋 출력 디렉터리 삭제는 위험 경로를 거부해야 한다 | E1 | `lerobot_recorder.py::_prepare_output_dir` | — | 09 §8.3 |
| FR-DATA-16 | video codec 은 `libx264`/`h264_nvenc` 만, quality·CQ 는 범위 검증해야 한다 | E1 | `lerobot_recorder.py::__init__` | — | 05 §2.1 |

---

## 5. FR — 런타임

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-RT-01 | Docker 서비스 5종을 제공해야 한다 | E2 | `docker/docker-compose.yaml` | `docker compose config` | 06 §1 |
| FR-RT-02 | 전 서비스는 `network_mode: host`·`ipc: host` 여야 한다 | E2 | 동상 | — | 06 §2 |
| FR-RT-03 | env 주입은 `.env` → `env/<POLICY_PROFILE>.env` 순(나중이 override)이어야 한다 | E2 | compose `env_file` · `vla_policy_node.py::_load_env` | — | 06 §5.1 |
| FR-RT-04 | 모델 선택은 `POLICY_PROFILE` 한 줄로 결정돼야 한다 | E2 | `.env.example` §1 · `env/*.env` | — | 06 §6 |
| FR-RT-05 | train 은 `--policy.type` 과 `--policy.path` 를 동시 지정하지 않아야 한다 | E2 | `docker/policy-entrypoint.sh` | — | 06 §4.1 |
| FR-RT-06 | GR00T-N1.5 는 policy-server 안에서 ACT/SmolVLA 와 동일 경로로 학습·추론해야 한다 | E2 | `env/groot_n15.env` · `policy-entrypoint.sh` | — | 06 §6 |
| FR-RT-07 | GR00T 호환 패치는 빌드 시 1회 적용되고, 대상 형태가 다르면 빌드를 중단해야 한다 | E1 | `docker/groot_compat_patch.py` | 빌드 | 09 §7.3 |
| FR-RT-08 | `datasets`·`outputs` 심링크는 컨테이너에서 영속돼야 한다(호스트 경로 직접 마운트) | E2 | `docker-compose.yaml` pink-ik 볼륨 주석 | — | 06 §3.1 |
| FR-RT-09 | isaac 계열 이미지는 `/workspace` 를 통째 마운트하지 않아야 한다 | E2 | `docker-compose.yaml` 주석 | — | 06 §3.1 |
| FR-RT-10 | vla-ros 는 컨테이너 안에서 colcon 빌드해야 한다 | E2 | `docker/vla-ros-entrypoint.sh` | — | 06 §4.3 |
| FR-RT-11 | 실기기 제어는 Windows native uv 로만 하고 디바이스 마운트를 두지 않아야 한다 | E2 | `docker-compose.yaml`(디바이스 없음) · `scripts/real/lerobot.sh` | — | 06 §10 |

---

## 6. FR — 인터페이스

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-IF-01 | sim 관측·명령은 `/isaac_joint_states` · `/isaac_joint_commands`(`JointState`, radian)여야 한다 | E2 | `run_cube_desk_ros_bridge.py::JOINT_STATES_TOPIC`, `::JOINT_COMMANDS_TOPIC` | — | 07 §2 |
| FR-IF-02 | 카메라 3종은 `/camera/{top,wrist,front}/image_raw` (`rgb8` 640×480)여야 한다 | E2 | `run_cube_desk_ros_bridge.py` camera 정의 | — | 07 §2 |
| FR-IF-03 | 수신 `JointState` 는 **이름 기준**으로 재정렬해야 한다 | E2 | `vla_policy_node.py::_joint_cb` | — | 07 §3.2 |
| FR-IF-04 | 전 ROS 경로는 `rmw_fastrtps_cpp` + `UDPv4` 를 써야 한다 | E2 | compose · 4개 entrypoint · bridge | — | 07 §5 |
| FR-IF-05 | planner 프로토콜은 JSON REQ/REP 3-command(`ping`/`plan_pickplace`/`shutdown`)여야 한다 | E2 | `curobo_batch_planner.py` docstring | `--self-test` | 07 §6.1 |
| FR-IF-06 | planner 요청 좌표는 `base_link` frame 이어야 한다 | E2 | 동상 | — | 07 §6.1 |
| FR-IF-07 | planner 응답 궤적 row 는 `[arm degree ×5, gripper feature]` 여야 한다 | E2 | `pickplace_sm.py` 변환부 | — | 07 §6.1 |
| FR-IF-08 | plan 대기 중에도 Isaac Sim 을 pump 해야 한다(GUI 프리즈 방지) | E2 | `pickplace_sm.py::_recv_plan` | — | 07 §6.2 |
| FR-IF-09 | leader teleop 페이로드는 `struct("<6f")` 24 byte, CONFLATE PUB 이어야 한다 | E2 | `so101_joint_state_server.py` | — | 07 §7 |
| FR-IF-10 | gRPC rename 은 **클라이언트가** 적용하고 server `rename_map` 은 비워야 한다 | E2 | `vla_policy_node.py::PolicySessionConfig` 주석 | — | 07 §8.2 |
| FR-IF-11 | 추론은 제어 루프를 막지 않아야 한다(백그라운드 executor) | E2 | `vla_policy_node.py::_start_inference` | — | 07 §3.2 |
| FR-IF-12 | 리셋 토큰 변경 시 큐·timestep·관측 캐시를 폐기해야 한다 | E2 | `vla_policy_node.py::_check_external_reset` | — | 07 §10 |

---

## 7. FR — 파이프라인

| ID | 요구사항 | E | 근거 앵커 | 검증 | 상세 |
|---|---|---|---|---|---|
| FR-PIPE-01 | 녹화 모드 2종은 상호배타여야 한다 | E1 | `scripts/cuRobo/pickplace_sm.py` (SystemExit) | — | 08 §5.6 |
| FR-PIPE-02 | 녹화는 `--enable_cameras` 와 `--auto_trials N > 0` 을 요구해야 한다 | E1 | 동상 | — | 08 §5.6 |
| FR-PIPE-03 | LeRobot 직기록은 `--num_envs 1` 을 요구해야 한다 | E1 | 동상 | — | 08 §5.6 |
| FR-PIPE-04 | 에피소드는 정지 2 s → 동작 → init 복귀 → 정지 1 s 규격이어야 한다 | E2 | `pickplace_sm.py` `--preroll_s`/`--posthold_s` 기본값 | — | 08 §5.6 |
| FR-PIPE-05 | 플래닝 대기 구간은 기록되지 않아야 한다 | E2 | `pickplace_sm.py::_manipulate_record` | — | 08 §5.6 |
| FR-PIPE-06 | teleop 은 `--record` + `lerobot_v3` 에 `--enable_cameras` 를 요구해야 한다 | E1 | `teleop_se3_agent.py` (ValueError) | — | 08 §3 |
| FR-PIPE-07 | 미지원 teleop device 는 `NotImplementedError` 여야 한다 | E1 | `teleop_se3_agent.py` | — | 08 §3 |
| FR-PIPE-08 | sweep 결과는 증분 저장(중단 안전)해야 한다 | E2 | `pickplace_sm.py::run_sweep` | — | 08 §5.5 |
| FR-PIPE-09 | AppLauncher 에는 화이트리스트 키만 전달해야 한다 | E4 | `run_cube_desk_ros_bridge.py::_LAUNCHER_KEYS` · `pickplace_sm.py` | — | 09 §6.1 |
| FR-PIPE-10 | USD author 는 절대 asset path 를 상대로 되돌리고, 불가하면 `RuntimeError` 를 내야 한다 | E1 | `author_pick_cube_scene.py` | — | 09 §6.6 |

---

## 8. NFR

| ID | 요구사항 | E | 근거 | 상세 |
|---|---|---|---|---|
| NFR-COMPAT-01 | ABI 핀 8종을 유지해야 한다(`uv lock --upgrade` 금지) | E1 | `pyproject.toml` `override-dependencies` | 06 §7.2 |
| NFR-COMPAT-02 | Python 은 `>=3.11,<3.13` 이어야 한다 | E1 | `pyproject.toml` `requires-python` | 06 §7.1 |
| NFR-COMPAT-03 | cuRobo 이미지는 `packaging==23.0` **정확 핀**이어야 한다 | E3 | `docker/Dockerfile.cuRobo` 주석(범위 핀 무효 실측) | 09 §7.1 |
| NFR-COMPAT-04 | isaac 계열 이미지는 베이스의 torch/numpy/isaaclab 을 재핀하지 않아야 한다 | E4 | `docker/Dockerfile.isaac_sim` 주석 | 06 §7.3 |
| NFR-REPRO-01 | `PickCube-v0` 는 고정 배치로 결정적이어야 한다 | E2 | `pick_cube_env_cfg.py::_CUBE_LAYOUT` | 03 §9.2 |
| NFR-REPRO-02 | Eval 변형은 디바운스로 가짜 성공을 걸러야 한다 | E2 | `PickCubeEvalTerminationsCfg` | 03 §10.2 |
| NFR-REPRO-03 | robot color 는 env 당 런 내내 고정된다(리셋 재추첨 불가) | E3 | `domain_randomization.py` 주석(실측) | 09 §5.3 |
| NFR-PERF-01 | 제어·기록 주파수는 30 Hz 다 | E2 | `FPS` · `decimation` | 03 §5 |
| NFR-PERF-02 | cuRobo SM 스폰 영역 성공률 실측: yaw-zero **183/183**, yaw-random **1305/1305** (54-sphere, chord 0.5×) | E3 | `scripts/cuRobo/README.md` 정량표 | 08 §5.7 |
| NFR-PERF-03 | 큐브 collider convexHull 전환 실측: jitter 2.9 → 0.056 rad/s, grasp 3/16 → 13/16 | E3 | `author_pick_cube_scene.py` 주석 | 09 §2.1 |
| NFR-PERF-04 | arm slew 2.5 하드캡 시 all-4 성공률 90.6% → 59.4% (동일 seed) | E3 | `pick_cube_env_cfg.py` 주석 | 09 §4.1 |
| NFR-PERF-05 | HDF5 녹화 메모리 ~1.2 GB/env/15 s (3-cam 640×480 uint8 @30 Hz) | E3 | `scripts/cuRobo/README.md` | 08 §5.6 |
| NFR-PERF-06 | `RHO_CAP` 18° 는 64-env 첫 planning 약 17분 — 12° 유지 | E3 | `scripts/cuRobo/README.md` | 08 §5.7 |
| NFR-PERF-07 | 이미지 통계는 1/64 격자 표본으로 계산한다 | E2 | `lerobot_recorder.py::ImageStats.sample_stride` | 05 §4.4 |
| NFR-SAFE-01 | gRPC payload 가 pickle 이므로 신뢰 네트워크 밖에 노출하지 않아야 한다 | E4 | `.env.example`·compose CVE 주석 | 09 §8.1 |
| NFR-SAFE-02 | 실기기 replay 는 e-stop 준비 후 실행해야 한다 | E4 | `sim_dataset_to_real_follower.py` docstring | 09 §8.2 |
| NFR-SAFE-03 | gripper effort 는 물체를 으깨지 않는 범위로 clamp 돼야 한다 | E2 | `gripper_effort.py` | 03 §6.2 |
| NFR-OBS-01 | 부팅 전 `faulthandler` 를 켜 C-레벨 크래시를 파일로 남겨야 한다 | E2 | `run_cube_desk_ros_bridge.py` | 09 §6.1 |
| NFR-OBS-02 | gRPC 왕복 시간을 10 콜마다 분해 출력해야 한다 | E2 | `vla_policy_node.py::_record_roundtrip` | 07 §8.3 |
| NFR-OBS-03 | planner 진단을 파일로 남겨야 한다 | E2 | `curobo_batch_planner.py::DIAG_LOG` | 07 §6.2 |

---

## 9. 제약 (CON)

선택 불가한 외부 요인이다.

| ID | 제약 | 근거 | 상세 |
|---|---|---|---|
| CON-01 | **RT 코어 없는 GPU(A100·H100)는 Isaac Sim 5.1 미지원** | NVIDIA 시스템 요구사항 | 06 §8.2 |
| CON-02 | SO-101 은 팔 5-DOF 라 임의 6-DOF pose 를 만족할 수 없다 — position 우선·orientation best-effort | 기구학 | 09 §3.4 |
| CON-03 | cuRobo 와 Isaac Sim 은 in-process 공존이 불가능하다 | cuda-core ↔ physx 런타임 충돌 | 09 §7.5 |
| CON-04 | bridge 는 OmniGraph 경로라 Python slew 를 걸 수 없다 | 아키텍처 | 09 §4.5 |
| CON-05 | robot color 는 리셋마다 재추첨할 수 없다 | Replicator de-instance ↔ physx view | 09 §5.3 |
| CON-06 | 책상을 cuRobo obstacle 로 넣을 수 없다 | base sphere 가 상판 내부 | 09 §3.6 |
| CON-07 | `evdev` 는 Linux 전용이라 Windows 에 설치되지 않는다 | 플랫폼 | 06 §10 |
| CON-08 | `TiledCamera` focal 은 USD attr 로 바꿀 수 없다 | Isaac Sim 구현 | 09 §11.3 |
| CON-09 | 테스트 스위트·lint config 가 없다 — 검증은 self-check + 빌드 + 실행이다 | 프로젝트 결정 | 08 §10 |

---

## 10. 수용 기준 (AC)

검증기가 판정하는 합격선이다.

| ID | 기준 | 판정 | 상세 |
|---|---|---|---|
| AC-01 | I/O 계약 4-validator 전부 통과 (`atol 1e-5`) | `validate_so101_io_contract.py` → `PASS` | 04 §9 |
| AC-02 | 데이터셋이 v3 스키마 불변 상수를 모두 만족 | `validate_lerobot_schema.py` → exit 0 | 05 §7.1 |
| AC-03 | 스키마 검증기가 오류 5케이스를 전부 검출 | `--self-test` → exit 0 | 05 §7.2 |
| AC-04 | follower affine round-trip·fit 복원 4종 assert 통과 | `python3 follower_calibration.py` | 04 §4.4 |
| AC-05 | 스폰 마스크·타깃 불변식 assert 통과 | `python3 spawn_area.py` | 03 §11.4 |
| AC-06 | grasp hysteresis 6케이스 통과 | `python3 .../mdp/observations.py` | 03 §3.2 |
| AC-07 | EEF 변환이 세 회전 표현 round-trip + meta 갱신을 만족 | `joint_dataset_to_eef.py --self-check` | 05 §6.3 |
| AC-08 | planner 후보 생성 기하 자체 점검 통과 | `curobo_batch_planner.py --self-check-geom` → `GEOM_SELFCHECK_OK` | 08 §10 |
| AC-09 | GR00T 호환 패치가 멱등 적용되고 형태 불일치 시 빌드 중단 | 이미지 빌드 성공 | 09 §7.3 |
| AC-10 | 명세 문서 상수가 코드 상수와 일치 | `validate_spec_constants.py` → `OK: N/N` | `SPEC.md §4` |

---

## 11. 미검증 요구사항 (리스크)

`검증` 열이 빈 항목 = 자동 검증 수단이 없다. 회귀가 조용히 들어올 수 있는 지점이다.

| 미검증 항목 | 위험 | 완화 |
|---|---|---|
| FR-ENV-14 (DR event 순서) | 순서를 바꿔도 아무 에러가 안 난다. 64 env 중 1개 재현 수준이라 눈에 안 띈다 | 09 §5.1 기록 · 코드 주석 |
| FR-PIPE-09 (AppLauncher 필터) | Linux 에서는 **조용히** viewport docking 만 실패한다 | 09 §6.1 기록 |
| FR-IO-05 / FR-ENV-04·05 | 관측·액션 계약 변경이 학습 후에야 드러난다 | AC-10 상수 대조 |
| FR-RT-02·08·09 (compose 배선) | 잘못 마운트해도 기동은 된다(출력만 소실) | `docker compose config` |
| NFR-SAFE-01·02 | 강제 수단이 없다 | 문서 경고 |
| **INC-10 (성공 판정 z)** | **성공 종료가 발화하지 않는 구조** — eval 수치가 무의미해질 수 있다 | `09 §9` 상세. **GPU 실행 확인 필요** |

---

## 12. 명시적 비요구 — 의도적으로 없는 것

| 항목 | 상태 | 이유 |
|---|---|---|
| RL 보상·커리큘럼 | 제거됨(빈 stub) | VLA-only 리팩토링. env = 추론·데이터 기판 |
| MoveIt · cuMotion · Lula · RMPFlow | 제거됨 | 5-DOF pose-goal 한계 + 유지비. cuRobo 로 수렴 |
| 6-DOF orientation hard constraint | 도입 안 함 | CON-02 |
| 테스트 스위트 · lint config | 없음 | CON-09. self-check 로 대체 |
| `depends_on` 기동 순서 | 정의 안 함 | 수동·`demo_vla.sh` 로 배선 |
| EEF-**relative** action | **미구현** | absolute EEF 파생까지만 커밋됨 — `04_IO_CONTRACT.md §10` |
| GR00T-N1.7 | 제거됨 | transformers 4.57 ↔ 5.3 충돌. N1.5 로 통일 |
| `policy-server-rtc` | 제거됨 | 백엔드 스크립트 부재 |

---

## 참조

- 각 요구사항의 구현 상세 → 03~08
- 왜 그 값인가 → `09_TACIT_KNOWLEDGE.md`
- 불일치 대장 → `09_TACIT_KNOWLEDGE.md §9`
