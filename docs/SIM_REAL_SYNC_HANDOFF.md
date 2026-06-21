# Sim ↔ Real 동등 추론 동기화 — 세션 인계 & 설계 논의

> 작성: 2026-06-19. 목적: sim 학습 VLA(SmolVLA `4cube_1024`)를 실기기 추론할 때 **sim과 동일하게
> 동작**하게 만드는 작업의 현재 상태 인계 + **다음 세션에서 본격 논의할 "제대로 된 동기화 구조"** 시드.
>
> 상세 단위/변환 감사는 [`SIM_REAL_INFERENCE_PARITY.md`](SIM_REAL_INFERENCE_PARITY.md) §1·§5.2·§5.3 (중복 안 함).
>
> **상태 갱신(2026-06-22):** 본 문서는 이전 affine shim 설계의 역사 기록이다. 정식 계약은
> `so101-canonical-v1`(arm=절대 URDF rad, gripper=jaw aperture mm)로 확정했고
> `src/so101_parity/`의 codec/calibration/executor와 ROS Jazzy sim/real client로 구현했다.
> 설치·운영은 [`PATH_F_CANONICAL_PARITY.md`](PATH_F_CANONICAL_PARITY.md)를 따른다.
> 기존 `GRIPPER_AFFINE` shim은 Legacy model frame 호환용이며 canonical runtime에 중복 적용하지 않는다.

---

## 1. 발단

사용자 증상: sim 학습 SmolVLA 를 실기기 추론하니 **그리퍼가 Isaac 보다 덜 열리고, 다른 joint 값도 차이**.
→ "sim 추론값 = real 로 보내는 값인가" 추적.

## 2. 이번 세션 확정 발견

| 항목 | 결론 | 근거 |
|---|---|---|
| **정규화 구조** | SmolVLA pre/post normalizer 가 **동일 stats blob 공유 → 상쇄**. 모델 I/O = **sim(데이터셋) 프레임 그대로**, 추가 스케일 없음(VISUAL=IDENTITY) | 모델 HF 캐시 `policy_pre/postprocessor*.json`+safetensors |
| **gripper** | sim `rad×31.75` vs real 항상 `RANGE_0_100`(캘리브 full-travel %). 같은 'open'이 sim≈27 vs real≈51 → 무보정 시 **sim 37° → real ~20°, 약 절반만 열림** | baked stats(action q99=20.6) + real teleop `pick_cube_v2`(q99≈51, grasp-open 45-60°) |
| **arm 5축** | 단위는 deg 일치(`use_degrees` 설치본 기본 True). 단 **real 0°=calibration home(`set_half_turn_homings`) vs sim 0°=URDF zero → per-joint 영점 offset(+부호)**. scale 1:1 | `config_so_follower.py:43`, calibration JSON homing_offset |
| **gain(별개)** | real `P_Coefficient=16`(떨림억제)+중력 → droop. 명령 맞아도 도달각 처짐. affine 무관(동특성) | `so_follower.py configure()` |
| **서버=이 PC sim** | **byte-identical** (git HEAD `4b909db8`, `so101_follower.usd`·urdf·env_cfg·lerobot_units md5 일치, clean). 이 PC 측정 affine 이 서버-학습 모델에 유효 | `ssh konan147` md5 비교 |

> real teleop `datasets/pick_cube_v2`(100ep, so_follower) 의 arm 분포가 sim 과 거의 겹침 → arm은 **단위뿐 아니라 영점·부호도 대체로 정합**(차이는 joint별 작은 offset). gripper 만 큰 발산.

## 3. 이번 세션 산출물 (전부 미커밋)

| 파일 | 내용 | 상태 |
|---|---|---|
| `scripts/sim/inspect_dataset_distribution.py` | LeRobot v3 6축 분포+degree판정+affine 권장 env (Isaac 무의존) | 신규·스모크 OK |
| `docker/policy-client-shim.py` | **per-joint 양방향 affine(Option B)**: `GRIPPER_AFFINE=1` 시 `SOFollower` get_observation(real→model)·send_action(model→real) monkey-patch. arm=`AFFINE_<J>_SIGN/OFFSET`(기본 identity), gripper=anchor or `GRIPPER_A/B` | 신규 블록·로직 검증 |
| `scripts/sim/read_sim_pose.py` | sim 포즈 구동+achieved deg 출력(recorder 동일 `to_lerobot_units`). GUI 크래시 회피 위해 non-headless 시 enable_cameras 강제 | 신규·**Isaac 실행 미검증**(GUI 크래시 수정 후 재시도 단계) |
| `scripts/test/measure_joint_affine.py` | paired-pose(sim·real 같은 포즈 ≥2) → joint별 SIGN/OFFSET·gripper A/B 피팅 → shim env 출력 | 신규·**사용자가 에러 수정함(임시) → 정본 반영 필요** |
| `docs/SIM_REAL_INFERENCE_PARITY.md` | §5.2(gripper scale 확정·정량) ·§5.3(arm per-joint frame·측정법) 신설, §5·6·7·8 갱신 | 갱신 |
| `AGENTS.md`·`CONTEXT.md`·memory | 동기화 | 갱신 |

**진행 상태**: 사용자가 `measure_joint_affine.py` 로 6축 보정값 측정 완료(affine env 획득). 임시로 동작 확인 중.
크래시 이슈(`rtx.scenedb.plugin.dll` access violation)는 `--enable_cameras`(rendering experience)로 해결 — `TROUBLESHOOTING.md §RTX scene DB access violation`.

## 4. 🎯 다음 세션 논의 주제 — "제대로" 동기화

현 affine 은 **band-aid**: 손측정·robot마다·재캘리브마다 재측정·운영 경로 1곳만·kinematic만(동역학/카메라 못 고침)·변환 로직 분산(`lerobot_units`/shim/measure).

**원칙(사용자 제시, 맞음)**: SmolVLA는 프레임 내부 매퍼로 두고(정규화는 OK), **sim·real 각자 좌표계 ↔ 단일 canonical 프레임 변환을 경계에서**. 핵심 결정 = **누가 reference 프레임?** (real이 물리 ground truth+배포 타깃 → real-native 유력.)

### 후보 (전부 재학습 동반)

| | 방법 | 효과 | 비용 |
|---|---|---|---|
| **A 즉시 정석** | 측정한 per-joint 변환을 **추론 매틱이 아니라 sim recorder에 1회 baked** → 데이터셋을 real-native 로 생성. **real 배포 무변환**(shim hack 삭제), sim만 변환. (gripper Option A 발상을 6축 확장) | 운영 경로 hack-free·robust | 데이터 재생성+재학습 |
| **B 최고 parity** | **real 데이터 학습**(`pick_cube_v2`·`autonomous_collect` 보유). frame+동역학+카메라 native. sim=pretrain/aug | 진짜 동등 | real 수집 비용 |
| **C 구조적** | **delta/relative action** 학습 → 영점 offset **불변**(최대 항 구조 제거). scale/sign만 잔존 | 캘리브 드리프트 강건 | 재학습 |

→ 가안: **A 또는 C 로 데이터 계약 고정 + B 보강.**

### 단일 진실원천 코드 (지금 흩어진 변환 통합)

- 구현 정본은 `src/so101_parity/`다. `CalibrationBundle`과 `ModelCodec`이
  `real_native ↔ canonical`, legacy model frame ↔ canonical 변환의 단일 진실원천이며
  sim/real client와 dataset converter가 공유한다.
- 끝판: **`IsaacSO101Robot` 을 LeRobot `Robot` 인터페이스로 구현** → sim·real 둘 다 공식 `RobotClient`+동일 코덱 (CONTEXT P0 권장 구조).

### kinematic 너머 (동작 보장에 필수)

- **gain/동역학**: real P16 droop ↔ sim PD → 데이터 동역학=배포 동역학 정합 or 느린 모션.
- **카메라**: intrinsic/FOV/색 갭 → 실측 정합 / 시각 DR / real fine-tune.
- **제어율·latency·chunk blending** 동일.

### 측정 가능하게 = parity 검증 하네스 (추측 금지)

기존 `scripts/sim/compare_train_vs_rtc.py`·`replay_infer_overlay.py`(recorded ep → 경로별 per-joint MAE) 를
**sim-path vs real-path** 비교로 확장: 같은 obs → 두 경로 action MAE < ε 게이트. + round-trip(state→action→next-state 발산) + FK parity(같은 canonical 포즈 sim TCP vs real TCP). affine 은 이 MAE 를 줄이는 한 수단일 뿐.

## 5. 다음 세션 결정 리스트

1. **canonical 프레임 = real-native? (Y/N)** — 정하면 A/C 의 변환 방향 확정.
2. **A(데이터 baked 변환) vs C(delta action) vs 둘 다** — 재학습 1회에 어느 계약으로.
3. **B(real 데이터 학습) 착수 여부·규모** — 동역학·카메라 갭 해결할지.
4. `so101_frame` 코덱 모듈 신설 위치·인터페이스 (shim/recorder/bridge 배선).
5. parity 검증 하네스부터 만들지 (게이트 먼저 세우고 작업).
6. gain/카메라 정합 범위 (어디까지 맞출지).

## 6. 미해결 / 리스크

- 측정 affine 값(사용자가 이번에 얻은 SIGN/OFFSET·gripper A/B)이 **이 문서에 미기록** → 다음 세션 시작 시 사용자 env 출력 확보 필요.
- `measure_joint_affine.py` 사용자 수정분 미반영(정본 동기화 필요).
- `read_sim_pose.py` 는 GUI 크래시 수정 후 **실제 Isaac 실행 미검증**.
- affine 은 kinematic만 — gain droop·카메라 갭 잔존(같은 동작 완전 보장 아님).
- A/B/C 전부 **재학습 비용** 수반 → 우선순위 결정 필요.

## 7. 참고

- 상세 감사·수식·앵커: `SIM_REAL_INFERENCE_PARITY.md` §5.2(gripper)·§5.3(arm)
- 측정 도구: `scripts/sim/read_sim_pose.py`(sim) + `scripts/test/{read_position,measure_joint_affine}.py`(real)
- 진단: `scripts/sim/inspect_dataset_distribution.py`, `compare_train_vs_rtc.py`, `replay_infer_overlay.py`
- 크래시: `TROUBLESHOOTING.md §RTX scene DB access violation`(GUI=`--enable_cameras`)
- 실기기 경로: `PATH_A_NATIVE.md`(shim·GRIPPER_AFFINE), real 수집: `REALDEVICE_AUTONOMOUS_COLLECT.md`
- 임시 affine 실행: `GRIPPER_AFFINE=1 [AFFINE_<J>_SIGN/OFFSET …] [GRIPPER_A/B …] uv run python ./docker/policy-client-shim.py ...`
