# SO-ARM101 Sim-to-Real

SO-ARM101 6축 로봇 팔용 **Sim-to-Real 파이프라인**. Isaac Sim 5.1 시뮬레이션에서 VLA 정책(ACT · SmolVLA · GR00T-N1.7)을 학습·검증하고, 실기기 SO-101 에 배포한다.

작업은 **2대의 머신**으로 나뉜다.

- **Windows 워크스테이션** — 실기기 SO-101 직결. **native uv**(WSL·Docker 없음)로 teleop·record·calibrate·setup-motors·policy-client.
- **Linux 서버** — 시뮬·학습·추론 서버. **전부 Docker**로 Isaac Sim 폐루프, VLA 학습, policy-server.

스택: **Isaac Sim 5.1 · Isaac Lab 2.3.2 · LeRobot 0.6.0(policy-server/실기기 전용 uv project) · ROS 2 Jazzy**.

## 목차 <!-- omit in toc -->

- [아키텍처 — 2-머신](#아키텍처--2-머신)
- [실행 경로](#실행-경로)
- [EEF-relative action 파이프라인](#eef-relative-action-파이프라인)
- [LeRobot v0.6.0 소스 분석과 구현 기준](#lerobot-v060-소스-분석과-구현-기준)
- [현재 PickCube 환경·에셋·cuRobo 평가](#현재-pickcube-환경에셋curobo-평가)
- [환경 요구사항](#환경-요구사항)
- [사전 설치 확인](#사전-설치-확인)
- [공통 준비](#공통-준비)
- [경로별 가이드](#경로별-가이드)
- [저장소 레이아웃](#저장소-레이아웃)
- [관련 문서](#관련-문서)
- [Reference](#reference)

---

## 아키텍처 — 2-머신

| | Windows 워크스테이션 | Linux 서버 |
|---|---|---|
| **역할** | 실기기 SO-101 제어 | 시뮬·학습·추론 서버 |
| **실행** | native uv + `scripts/real/pyproject.toml` (WSL·Docker 없음) | Docker (전부) |
| **작업** | teleop · record · replay · calibrate · setup-motors · find-port · policy-client | Isaac Sim 폐루프 · VLA 학습 · policy-server · sim policy-client(vla-ros) |
| **LeRobot** | 0.6.0 (Python 3.12 전용 uv project) | 0.6.0 / commit `30da8e6` 기반 patch |
| **로봇 I/O** | COM 포트 직결 (usbipd/WSL 불필요) | 로봇 직결 없음 (sim/추론만) |
| **GPU** | RTX A4000 16GB (실기기 CLI 는 GPU 불요) | RTX PRO 5000 Blackwell 48GB |

```mermaid
flowchart LR
    subgraph WIN["Windows 워크스테이션 (native uv)"]
        ROBOT["SO-101 실기기<br/>leader + follower"]
        CLI["LeRobot CLI<br/>record · calibrate · policy-client"]
        ROBOT --- CLI
    end
    subgraph LNX["Linux 서버 (Docker)"]
        PS["policy-server<br/>async gRPC :8080"]
        SIM["isaac-sim<br/>SimToReal-PickCube"]
        VLA["vla-ros<br/>vla_policy_node"]
        SIM <-->|ROS2| VLA
        VLA <-->|gRPC| PS
    end
    CLI -->|"gRPC (실기기 추론)"| PS

    classDef win fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
    classDef lnx fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    class WIN,ROBOT,CLI win
    class LNX,PS,SIM,VLA lnx
```

---

## 실행 경로

| 경로 | 머신 | 진입점 | 용도 |
|---|---|---|---|
| **실기기 LeRobot** | Windows (native uv) | `scripts/real/lerobot.sh <mode>` | teleop · record · calibrate · setup-motors · find-port |
| **실기기 VLA 추론** | Windows (native uv) | `scripts/real/lerobot.sh policy-client` | joint fallback 또는 EEF FK/IK client → Linux gRPC |
| **sim VLA 폐루프** | Linux (Docker) | `docker compose up policy-server isaac-sim vla-ros` | `SimToReal-SO101-PickCube-Eval-v0` closed-loop 평가 (디바운스 성공; 데이터생성은 `-DR-v0`) |
| **sim SM 데이터 생성** | Linux (Docker) | isaac-sim `datagen` 모드 (`record_state_machine.py`) | State Machine 데모 → LeRobot v3 (GPU 런타임 검증 진행 중) |
| **VLA 학습** | Linux (Docker) | policy-server `train` | ACT · SmolVLA · GR00T-N1.7 + 공통 EEF-relative processor |
| **sim 수동 teleop** (보조) | Linux (host uv) | `uv run scripts/.../teleop_se3_agent.py` | Isaac Lab 로컬 teleop · USD 씬 author |

> **추론 백엔드는 1개**: `policy-server`(gRPC). 실기기 policy-client(Windows)와 sim vla-ros(Linux)가 같은 서버에 접속한다.

---

## EEF-relative action 파이프라인

LeRobot v3에는 state/action을 모두 canonical absolute EEF 10D로 보존한다.

```text
[tcp_grasp xyz(3), Rot6D first two rows(6), absolute gripper feature(1)]
```

학습 preprocessor만 각 action horizon을 현재 observation 기준
`T_rel = inv(T_state) @ T_action`으로 바꾸고, 추론 postprocessor는 full chunk를 한 번에
`T_action = T_state @ T_rel`로 복원한다. 이후 sim/real client가 같은 URDF·robot YAML로
sequential bounded IK를 수행한다. Rot6D/EEF 벡터의 elementwise 평균은 금지하며 overlap은
IK 이후 `latest_only`로 처리한다.

```mermaid
flowchart LR
    J["Joint-space LeRobot v3"] --> C["joint_dataset_to_eef.py"]
    C --> D["Absolute EEF 10D dataset"]
    D --> S["Horizon별 relative stats"]
    D --> P["SE(3) train preprocessor"]
    S --> P
    P --> M["ACT / SmolVLA / GR00T-N1.7"]
    M --> Q["Full-chunk postprocessor"]
    Q --> I["sim/real sequential IK"]
    I --> R["Absolute joint command"]
```

<details>
<summary><strong>학습·추론 좌표계 설정 (action representation 4 mode × EEF pose format 3종)</strong></summary>

#### 1. 좌표계(mode) 4종

| mode | dataset 저장 | 학습 processor | client 경계 |
|---|---|---|---|
| `joint_absolute` | joint-space **absolute** state/action | 변환 없음(canonical 정규화만) | IK **없음**. canonical joint command 직행 |
| `joint_relative` | joint-space **absolute** state/action | chunk 전체를 현재 state 기준 topology-aware Δq로 변환 | IK **없음**. postprocessor가 absolute joint 복원 후 직행 |
| `eef_absolute` | EEF-space **absolute** state/action | 변환 없음(canonical 정규화만) | full chunk → **sequential bounded IK** |
| `eef_relative` | EEF-space **absolute** state/action | chunk 전체를 `T_rel = inv(T_state) @ T_action`로 변환 | full chunk를 `T_state @ T_rel`로 복원 후 **sequential bounded IK** |

- **dataset은 어떤 mode에서도 그 space의 absolute 값만 저장한다.** relative는 학습
  preprocessor가 만들고 추론 postprocessor가 되돌리는 **runtime 변환**이다.
- relative 변환의 기준 state는 chunk 전체가 **동일한 current observation** 하나다
  (프레임 간 temporal delta가 아니다).

#### 2. EEF pose format 3종 (mode가 `eef_*`일 때만)

| 값 | 구성 | 차원 |
|---|---|---|
| `xyz_rot6d_rows` (**project profile 기본값**) | `xyz(3)` + 회전행렬 **첫 두 row**(6) + gripper(1) | 10D |
| `xyz_quaternion_wxyz` | `xyz(3)` + quaternion **wxyz 순서**(4) + gripper(1) | 8D |
| `xyz_rpy` | `xyz(3)` + roll/pitch/yaw(3) + gripper(1) | 7D |

`xyz_rot6d_rows`는 **`env/<profile>.env` 세 profile이 그렇게 설정돼 있다는 뜻**이며 코드의
암묵 기본값이 아니다. `eef_*` mode에서는 pose format을 **항상 명시**해야 한다.

`joint_absolute`/`joint_relative`에서는 pose format을 **반드시 비운다**(빈 문자열).
값을 넣으면 config 단계에서 거부된다.

#### 3. 현재 profile 기본값

`env/act.env` · `env/smolvla.env` · `env/groot_n17.env` 세 profile 모두 동일:

```text
ACTION_REPRESENTATION_MODE=eef_relative
ACTION_REPRESENTATION_POSE_FORMAT=xyz_rot6d_rows
ACTION_REPRESENTATION_STATS_FILE=meta/action_representation_stats.json
```

overlap aggregation은 `latest_only`(EEF/Rot6D 벡터 평균 금지).

#### 4. 학습 시 설정

`env/<profile>.env`의 세 변수를 바꾸면 그 profile의 기본 좌표계가 바뀐다. 학습만
따로 덮어쓰려면 `TRAIN_ACTION_REPRESENTATION_MODE` / `TRAIN_ACTION_REPRESENTATION_POSE_FORMAT`
을 쓴다 — **`TRAIN_*`가 `ACTION_*`보다 우선**한다.

```bash
# profile 기본값(eef_relative + xyz_rot6d_rows)으로 학습
POLICY_PROFILE=act docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  policy-server train

# 일회성 override — quaternion EEF-relative로 학습 (-e 로 container env 주입)
POLICY_PROFILE=act docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e TRAIN_ACTION_REPRESENTATION_MODE=eef_relative \
  -e TRAIN_ACTION_REPRESENTATION_POSE_FORMAT=xyz_quaternion_wxyz \
  -e ACTION_REPRESENTATION_STATS_FILE=meta/action_representation_stats.json \
  policy-server train

# 일회성 override — joint fallback으로 학습 (pose format은 빈 문자열)
# ACTION_REPRESENTATION_POSE_FORMAT 도 함께 비워야 한다. policy-entrypoint.sh 의
# `${TRAIN_...:-${ACTION_...}}` colon-dash fallback 때문에, 이걸 안 비우면 profile 의
# xyz_rot6d_rows 가 joint mode 로 새어 들어와 config 단계에서 거부된다.
POLICY_PROFILE=act docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e TRAIN_ACTION_REPRESENTATION_MODE=joint_absolute \
  -e TRAIN_ACTION_REPRESENTATION_POSE_FORMAT= \
  -e ACTION_REPRESENTATION_POSE_FORMAT= \
  -e ACTION_REPRESENTATION_STATS_FILE=meta/action_representation_stats.json \
  policy-server train
```

`ACTION_REPRESENTATION_STATS_FILE`이 가리키는 artifact에 해당 mode/format/horizon의
stats profile이 **미리 생성돼 있어야** 한다(아래 §데이터 준비와 학습 2단계).

#### 5. 추론 시 설정

추론 좌표계의 **유일한 source of truth는 checkpoint 안의 `action_representation.json`
manifest**다. 환경변수는 override가 아니라 **optional assertion**이다.

| `ACTION_REPRESENTATION_MODE` / `POSE_FORMAT` | 동작 |
|---|---|
| 비움 | manifest 값을 그대로 수용 |
| manifest와 동일 | 검증 통과 |
| manifest와 불일치 | **motor/sim command를 내기 전에 기동 실패**(fail-fast) |

checkpoint 선택은 `POLICY_REPO_ID`(local dir 또는 HF repo id)로 한다.

```bash
# ① policy-server — checkpoint manifest가 좌표계를 결정. 환경변수는 assertion용
POLICY_PROFILE=act docker compose --env-file .env -f docker/docker-compose.yaml up policy-server

# 좌표계를 명시적으로 확인하고 싶을 때만 assertion을 건다
POLICY_PROFILE=act docker compose --env-file .env -f docker/docker-compose.yaml run --rm \
  -e ACTION_REPRESENTATION_MODE=eef_relative \
  -e ACTION_REPRESENTATION_POSE_FORMAT=xyz_rot6d_rows \
  policy-server policy-server

# ② sim 폐루프 — vla-ros client. 좌표계는 서버 checkpoint manifest에서 유도된다
docker compose --env-file .env -f docker/docker-compose.yaml up policy-server isaac-sim vla-ros

# ③ Windows 실기기 — manifest로 client 종류(EEF FK/IK vs joint 경계)를 자동 분기
scripts/real/lerobot.sh policy-client
```

- 실기기 client는 preflight로 `assert_checkpoint_representation.py --emit client_kind`를
  실행해 manifest에서 client를 고르고, chunk overlap은 `--aggregate_fn_name=latest_only`로
  고정한다. **EEF/Rot6D에 `weighted_average`를 쓰지 않는다** — 회전 표현을 elementwise
  평균하면 유효한 SE(3)가 아니게 된다. overlap 병합은 IK 이후 joint queue에서만 한다.
- EEF mode에서 실기기 motor command는 `EEF_IK_REAL_VALIDATED=true`일 때만 나간다
  (자세한 gate 범위는 §검증과 rollout).

> **주의 — absolute dataset을 relative로 덮어쓰지 말 것.**
> LeRobot v3 dataset에는 언제나 해당 space의 **absolute** state/action만 저장한다.
> relative 값을 미리 계산해 dataset에 덮어써 저장하면 stats fingerprint·manifest·
> postprocessor 복원 기준이 전부 어긋나 이중 변환이 발생한다. 변환은 학습
> preprocessor와 추론 postprocessor에서만 일어난다.

</details>

### 데이터 준비와 학습

```bash
# 1) 원본은 보존하고 absolute joint → absolute EEF Rot6D 파생셋 생성
python scripts/convert/joint_dataset_to_eef.py \
  --input-dir datasets/joint_v3 --output-dir datasets/eef_v3 \
  --source-domain sim --rotation-representation rot6d

# 2) policy horizon별 stats profile을 같은 artifact(meta/action_representation_stats.json)에 추가
#    --all 은 그 dataset이 지원하는 representation(absolute/relative)을 모두 만든다.
python scripts/data/generate_action_representation_stats.py --dataset-root datasets/eef_v3 --horizon 100 --all
python scripts/data/generate_action_representation_stats.py --dataset-root datasets/eef_v3 --horizon 50 --all
python scripts/data/generate_action_representation_stats.py --dataset-root datasets/eef_v3 --horizon 40 --all

# 3) .env의 DATASET_ROOT/HF_DATASET_REPO_ID를 EEF dataset으로 지정하고 profile 선택
#    action representation은 profile 변수로 고른다(schema v2, 4 mode × 3 EEF pose format):
#      ACTION_REPRESENTATION_MODE=joint_absolute|joint_relative|eef_absolute|eef_relative
#      ACTION_REPRESENTATION_POSE_FORMAT=xyz_rot6d_rows|xyz_quaternion_wxyz|xyz_rpy  (EEF 전용)
POLICY_PROFILE=act docker compose -f docker/docker-compose.yaml run --rm policy-server train
POLICY_PROFILE=smolvla docker compose -f docker/docker-compose.yaml run --rm policy-server train
POLICY_PROFILE=groot_n17 docker compose -f docker/docker-compose.yaml run --rm policy-server train
```

`POLICY_PUSH_TO_HUB=false`가 프로젝트 기본값이다. LeRobot v0.6 policy config의 upstream
기본값은 `true`이므로, checkpoint를 실제 Hub에 올릴 때만 `.env`에서 명시적으로
`POLICY_PUSH_TO_HUB=true`로 바꾼다.

학습 checkpoint에는 model/config/processor stats와 함께 `action_representation.json`
(**schema v2**)이 **mode와 무관하게 항상** 저장된다(periodic·final·Hub root). manifest는
mode/pose_format/dim, dataset fingerprint·revision, resolved group indices, stats profile hash,
policy horizon/family, LeRobot/project commit, URDF/YAML hash, selective-reuse report를 기록한다.
추론 server와 platform client는 이를 다시 검증하며 누락·변조·kinematics 불일치 시 시작을 거부한다.
v1 manifest checkpoint는 자동 승격되지 않는다(Phase 16 migration 필요).

추론 시 representation 인자(`ACTION_REPRESENTATION_MODE`/`ACTION_REPRESENTATION_POSE_FORMAT`)는
**assertion**이다. checkpoint 의미를 바꾸지 않으며, 값이 다르면 policy-server·sim client·
real client 모두 로봇/sim 명령 이전에 기동을 중단한다. 생략하면 manifest 값을 그대로 쓴다.

legacy checkpoint는 자동 승격되지 않는다. 별도 migration으로 v2 checkpoint를 만든다
(원본은 그대로 두고 새 디렉터리에 생성).

```bash
# manifest 없는 legacy checkpoint → joint_absolute (정확한 flag 필수)
python scripts/convert/migrate_action_representation_checkpoint.py \
  --source outputs/train/old/checkpoints/last/pretrained_model \
  --output outputs/train/old/checkpoints/last/pretrained_model_v2 \
  --dataset-root datasets/joint_v3 --horizon 50 \
  --allow-legacy-joint-absolute-checkpoint

# v1 EEF-relative(xyz_rot6d_rows) checkpoint
python scripts/convert/migrate_action_representation_checkpoint.py \
  --source outputs/train/eef/checkpoints/last/pretrained_model \
  --output outputs/train/eef/checkpoints/last/pretrained_model_v2 \
  --dataset-root datasets/eef_v3 --horizon 50

# checkpoint 계약 확인/assertion (local dir 또는 HF repo id)
python scripts/inference/assert_checkpoint_representation.py \
  --checkpoint outputs/train/eef/checkpoints/last/pretrained_model_v2 --json
```

```bash
# 24 조합(3 policy × 8 representation) 통합 검증
python scripts/contract/validate_action_representation_policies.py --fixture-root scratch/fx --policies act,smolvla,groot
# 실제 CLI checkpoint의 manifest/reload 검증
python scripts/contract/validate_action_representation_checkpoint_cli.py --fixture scratch/fx/joint --mode joint_relative
# Phase 16: migration·routing·CLI assertion
python scripts/contract/validate_action_migration.py --fixture-root scratch/fx
python scripts/contract/validate_action_routing.py
python scripts/contract/validate_representation_cli_assertions.py
```

```bash
# Phase 17: 24 조합 × §25.2 13개 필수 검증 통합 matrix (실제 policy·sync engine·PolicyServer)
#   Docker policy-server:0.6.0 안에서 실행한다(현재 image의 patch가 authoritative).
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  -v lerobot_hf_cache:/workspace/.cache/huggingface \
  -e HF_HOME=/workspace/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e SO101_PROJECT_GIT_COMMIT=$(git rev-parse HEAD) \
  -e SO101_PROJECT_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
  -e SO101_PROJECT_GIT_DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false) \
  -e SO101_DOCKER_IMAGE_ID=$(docker inspect --format '{{.Id}}' policy-server:0.6.0) \
  --entrypoint python policy-server:0.6.0 \
  scripts/contract/validate_action_representation_matrix.py \
    --fixture-root scratch/p17-baseline \
    --output scratch/p17-matrix/phase17_24combo.json

# 결과 집계(24 조합 × 13 check 전부 pass 인지)
jq '.totals' scratch/p17-matrix/phase17_24combo.json
jq '[.combinations[].checks[].status] | group_by(.) | map({(.[0]): length}) | add' \
  scratch/p17-matrix/phase17_24combo.json
```

- 개발 중 부분 실행은 `--policies act` / `--representations joint_absolute` 필터를 쓴다.
  completion 실행은 24 조합 전부여야 하며, 부족하면 runner가 `INCOMPLETE`로 실패한다.
- 결과 artifact: `scratch/p17-matrix/phase17_24combo.json`
  (schema version·생성시각·git SHA/branch/dirty·Docker image ID·LeRobot version/commit·
  device/GPU·seed·24 totals·조합별 13 check·failures/skips, atomic write, mode 0644).
- **provenance**: `SO101_PROJECT_GIT_{COMMIT,BRANCH,DIRTY}`를 host에서 주입해야 worktree
  실제 상태가 기록된다. 주입이 없으면 container 안에서 검출을 시도하고, 검출도 못 하면
  `dirty: null`(unknown)로 남긴다 — clean으로 단정하지 않는다. 각 필드의 출처는
  `.git.provenance_source`에 있다.

```bash
# Phase 18(부분): 24 조합 contract-level rollout dry-run — GPU·정책 weight 불요(CPU)
docker run --rm --ipc=host -v "$PWD":/workspace -w /workspace -e HF_HUB_OFFLINE=1 \
  -e SO101_PROJECT_GIT_COMMIT=$(git rev-parse HEAD) \
  -e SO101_PROJECT_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
  -e SO101_PROJECT_GIT_DIRTY=$([ -n "$(git status --porcelain)" ] && echo true || echo false) \
  -e SO101_DOCKER_IMAGE_ID=$(docker inspect --format '{{.Id}}' policy-server:0.6.0) \
  --entrypoint python policy-server:0.6.0 \
  scripts/contract/validate_action_representation_rollout_dry_run.py \
    --phase17-artifact scratch/p17-matrix/phase17_24combo.json \
    --output scratch/p18-dry-run/phase18_24combo_dry_run.json

jq -c '.status, .phase18_complete, .totals, .acceptance' \
  scratch/p18-dry-run/phase18_24combo_dry_run.json
```

**결과 해석 — 이 artifact는 Phase 18을 완료시키지 않는다.**

| 필드 | 의미 |
|---|---|
| `status = DRY_RUN_PASS` | 24 조합 × 6 stage가 실제 router/platform adapter/action queue로 통과. **계약 수준만** |
| `phase18_complete = false` | **항상 false. 이 runner는 non-promoting이다** — 외부 sim/real report가 둘 다 `REPORT_VERIFIED`여도 승격하지 않는다. Phase 18 승격은 별도 closure 절차(spec 상태표 + §26.2 checkbox 갱신)로만 한다 |
| `acceptance.sim_closed_loop = NOT_RUN` | 학습된 EEF checkpoint·sim 평가 없음(Phase 9 미실행) |
| `acceptance.real_guarded_rollout = BLOCKED_EXTERNAL` | 실기기 승인·작업자·e-stop gate 없음(Phase 10 미실행) |
| `operational_metrics` | 조합별 IK failure·invalid chunk·abort·starvation/empty/stale·residual·routed/published. **`aborts`/`invalid_chunks`/`queue_starvation_ticks`/`empty_chunks`/`stale_chunks`의 0은 장시간 rollout 측정치가 아니라 정상 경로에서 구조적으로 0인 값이다**(짧은 결정적 chunk 1개, 실패 주입 없음). 이 runner는 주입 guard와 evaluator self-test로 **fail-closed 동작만** 검증한다. 실제 rate/threshold acceptance는 외부 evaluator report 몫 |
| `injected_guard_events` | **의도적으로 주입한** guard 수 = 조합당 6건(stale merge 1 + empty pop 1 + invalid chunk 4), 전체 144. operational failure 지표와 분리 |
| `combination_coverage` | 실행된 조합 ID의 set/uniqueness. `exact_24_set=false`면 completion 아님 |
| `phase17_artifact` | **historical 입력 artifact**의 SHA256 + 형식 검증 결과. 현재 Phase 18 실행 환경과의 동일성 비교가 **아니다**(이번 실행 provenance는 top-level `git`/`docker_image_id`/`lerobot`, `provenance_scope="current dry-run execution environment"`에 따로 기록된다). 검증 항목: ① schema version ② §25.1 **24개 expected combination ID set 완전 일치**(중복·누락·초과 0) ③ entry status=pass, `checks` key set이 `CHECK_NAMES`와 정확히 일치 ④ `totals`의 expected/ran/passed/failed 값 정확 일치(24 조합·312 check) ⑤ provenance 형식(commit 40-hex · branch nonempty · dirty bool · `provenance_source` object의 3필드 nonempty · `docker_image_id` `sha256:`+64hex · lerobot version `0.6.0` · lerobot commit 40-hex). 하나라도 어긋나면 전체 FAIL + exit 1 |

- 조합별 stage: `checkpoint_contract_resolve` · `sim_boundary_route`(EEF IK 1회 / joint 0회) ·
  `real_dry_run_boundary`(motor command publish 0을 sink counter로 증명) ·
  `action_queue_operations`(실제 `ActionChunkQueue` latest_only/overlap/stale/empty/refill) ·
  `invalid_chunk_rejection`(NaN·차원·rank·도달불가 chunk가 queue/publish 이전 거부) ·
  `acceptance_gate_real_dry_run`(실제 `evaluate_eef_rollout_metrics.py --mode real-dry-run` 호출).
- runner는 `evaluate_eef_rollout_metrics.py`의 sim/real-dry-run/real pass와 fail-closed 4종을
  synthetic JSONL로 실제 호출하는 acceptance-gate self-test도 1회 수행한다.
- 외부 평가 결과가 생기면 `--sim-eval-report` / `--real-rollout-report`로 넘겨 acceptance를
  조립한다. 인자는 `evaluate_eef_rollout_metrics.py --output`이 만든 **JSON object**여야 하며
  runner가 실제로 load해 ① 파일 존재 ② `mode`가 기대값(sim / real·real-dry-run) ③ `status=PASS`
  ④ `failures`가 빈 list ⑤ `final.event=="final"`을 **전부** 요구한다. 통과하면 상태가
  `REPORT_VERIFIED`가 되고 report의 path/SHA256이 acceptance evidence로 기록된다. 하나라도
  어긋나면 `REPORT_INVALID` + 전체 `status=FAIL` + exit 1이다(존재만 확인하고 통과시키지 않는다).
  report를 주지 않으면 위 표대로 NOT_RUN/BLOCKED_EXTERNAL로 남는다. **둘 다 REPORT_VERIFIED가
  되어도 `phase18_complete`는 false다** — 이 runner는 evidence를 모을 뿐 Phase를 승격하지
  않는다(non-promoting). 승격은 사람이 spec 상태표/§26.2 checkbox를 갱신하는 별도 closure
  절차다.
- **안전 gate 범위**: `EEF_IK_REAL_VALIDATED`는 **EEF IK로 산출한 real joint command**에만
  걸린다. joint-space fallback(`joint_absolute`/`joint_relative`)은 IK를 거치지 않으므로 이
  EEF-specific gate의 대상이 **아니며**, 그래서 fallback으로 즉시 선택 가능하다. 대신 joint
  경로의 실기기 구동은 이 gate가 아니라 **일반 하드웨어 안전 절차**(작업자 입회·e-stop·감속·
  workspace 확인)로 별도 통제해야 한다.
- completion은 24개 **expected combination ID의 정확한 set**(중복·누락·초과 0)까지 확인한다.
  `--policies`에 중복 policy를 주면 argparse 단계에서 거부한다(`--policies act,act,smolvla`처럼
  GR00T 없이 24개를 채워 completion을 위조할 수 없다). subset 필터 실행은 `DRY_RUN_PARTIAL` +
  exit 1이다.

### 검증과 rollout

```bash
# 순수 계약·processor·FK/IK sweep
python scripts/contract/validate_eef_relative_contract.py
python scripts/contract/validate_eef_platform_adapter.py
python scripts/contract/validate_eef_ik_workspace.py

# 학습된 checkpoint의 recorded target 대비 full-chunk open-loop 비교
docker compose -f docker/docker-compose.yaml run --rm policy-server python \
  /workspace/scripts/inference/evaluate_eef_open_loop.py \
  --checkpoint /workspace/outputs/train/<run>/checkpoints/last/pretrained_model \
  --dataset-root /workspace/datasets/eef_v3 \
  --output /workspace/logs/eef_open_loop.json

# sim closed-loop 1-episode. metrics와 eval JSON을 함께 생성
scripts/inference/demo_vla.sh start act --eval 1 --headless
python scripts/contract/evaluate_eef_rollout_metrics.py \
  --mode sim --metrics logs/eef_sim_rollout.jsonl \
  --eval outputs/vla_eval_act.json

# DR batch는 task/episode 수와 허용 성공률을 명시
scripts/inference/demo_vla.sh start smolvla --eval 20 --headless \
  --task SimToReal-SO101-PickCube-DR-Eval-v0
python scripts/contract/evaluate_eef_rollout_metrics.py \
  --mode sim --metrics logs/eef_sim_rollout.jsonl \
  --eval outputs/vla_eval_smolvla.json --min-success-rate 0.8

# Windows: false=motor-off IK target log, true=실기기 command 승인
EEF_IK_REAL_VALIDATED=false scripts/real/lerobot.sh policy-client
python scripts/contract/evaluate_eef_rollout_metrics.py \
  --mode real-dry-run --metrics logs/eef_real_rollout.jsonl

EEF_IK_REAL_VALIDATED=true  scripts/real/lerobot.sh policy-client
python scripts/contract/evaluate_eef_rollout_metrics.py \
  --mode real --metrics logs/eef_real_rollout.jsonl
```

상세 계약·단계별 acceptance criteria는
[`docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`](docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md)에 있다.

---

## LeRobot v0.6.0 소스 분석과 구현 기준

분석 기준은 `ref_repos/lerobot`의 **v0.6.0**, commit
[`30da8e6`](https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce)이다.
reference clone은 read-only 분석 기준이다. 실행 image는 PyPI `lerobot==0.6.0`에
`docker/lerobot_v060_eef_relative_patch.py`를 version-tripwire 방식으로 적용하며 같은 commit을
image label과 checkpoint manifest에 기록한다.

<details>
<summary><strong>lerobot-train 처리 파이프라인과 VLA별 분기</strong></summary>

`lerobot-train`은 모든 policy가 공유하는 **학습 orchestration**이고, 실제 입력 변환·모델·loss·optimizer는
policy별로 분기된다. 따라서 “모든 policy가 같은 train processor를 쓴다”가 아니라,
**공통 runner가 policy별 `PolicyProcessorPipeline`을 호출한다**가 정확하다.

```mermaid
flowchart TD
    CLI["Train config"] --> DS["Dataset + chunk sampling"]
    DS --> P["Policy load"]
    P --> PP["Processor factory"]
    PP --> F{"policy config type"}
    F --> P1["ACT / SmolVLA / π"]
    F --> P2["GR00T N1.7"]
    F --> P3["Other VLA"]
    P1 --> DL["DataLoader"]
    P2 --> DL
    P3 --> DL
    DL --> PRE["Preprocess batch"]
    PRE --> FW["Forward + loss"]
    FW --> BW["Backward"]
    BW --> CLIP["Gradient clip"]
    CLIP --> OPT["Optimizer step"]
    OPT --> SCH["Scheduler step"]
    SCH --> OUT["Log / eval / checkpoint"]
    OUT --> SAVE["Save model + processors"]
```

공통 흐름의 실제 분기점은 다음과 같다.

| 단계 | 공통 처리 | policy별로 달라지는 부분 |
|---|---|---|
| dataset | LeRobot dataset 로드, episode-aware sampling | `action_delta_indices`·`observation_delta_indices`가 chunk/history 길이를 결정 |
| policy 생성 | dataset metadata에서 feature schema 추론 | `get_policy_class()`가 모델 class를 선택하고 pretrained/fresh-init 경로 분기 |
| batch 전처리 | 매 step `preprocessor(batch)` 호출 | rename·normalization·tokenization·padding·frame 변환·relative action 순서가 서로 다름 |
| 학습 update | `forward → backward → clip → optimizer.step → scheduler.step` | 각 policy의 `forward()`가 architecture와 loss를, config가 optimizer/scheduler preset을 정의 |
| 후처리 | offline train loss 계산에는 사용하지 않음 | `postprocessor`는 env eval/추론에서 action decode·unnormalize·absolute 복원에 사용 |
| 저장/재개 | checkpoint와 train state 저장 | processor 두 개도 JSON으로 함께 저장하며, pretrained 재개 시 저장된 pipeline을 복원 |

VLA별 processor 차이는 아래와 같다. 모든 행 앞에는 feature rename과 필요 시 batch dimension 추가,
끝에는 device 이동이 공통으로 붙는다.

| `--policy.type` | 학습 preprocessor의 핵심 순서 | 추론 postprocessor | 내장 relative-action flag |
|---|---|---|---|
| `pi0` | task newline/PaliGemma tokenize → **absolute→relative(선택)** → normalize | unnormalize → **relative→absolute(선택)** | `use_relative_actions` |
| `pi0_fast` | **absolute→relative(선택)** → normalize → state/language 준비 → text tokenizer + action tokenizer | unnormalize → **relative→absolute(선택)** | `use_relative_actions` |
| `pi05` | **absolute→relative(선택)** → normalize → state token 준비 → PaliGemma tokenize | unnormalize → **relative→absolute(선택)** | `use_relative_actions` |
| `groot` | LeRobot 입력을 video/state/action/language/embodiment로 pack → N1.7 VLM encode. checkpoint modality config와 horizon별 stats를 사용 | N1.7 action decode·unnormalize. native `xyz+rot6d` EEF-relative는 SE(3)로 absolute pose 복원 | `use_relative_actions`; native N1.7 경로 우선, generic fallback 존재 |
| `smolvla` | task newline → VLM tokenizer → normalize | unnormalize | 없음 |
| `xvla` | text tokenize → image float/ImageNet normalize → domain ID 추가 → dataset normalize | unnormalize | 없음 |
| `eo1` | normalize → conversation template → Qwen processor | unnormalize | 없음 |
| `molmoact2` | joint sign/offset frame 변환 → gripper-mask normalize/clamp → image/state/language/setup/control token pack | action clamp → masked unnormalize → joint frame 역변환 | 없음 |
| `wall_x` | Qwen 계열 task formatting → normalize | unnormalize | 없음 |
| `evo1` | state/action 차원 padding → normalize | unnormalize → action 차원 복원·선택적 gripper 이진화 | 없음 |

> **Relative action 주의**: v0.6.0의 공용 `RelativeActionsProcessorStep`은 같은 index의
> `observation.state`를 action chunk 전체에서 단순히 빼고, 후처리에서 다시 더한다
> (`relative = action - state`). 즉 joint/EEF라는 좌표 의미를 해석하거나 SE(3) pose composition을
> 수행하지 않는다. EEF의 `rpy`, quaternion, Rot6D에 이 step을 그대로 쓰면 회전의 진짜 상대 pose가
> 아니라 **표현 벡터의 성분별 차이**가 된다. 단, GR00T N1.7 전용 decoder에는 checkpoint가
> `type=eef`, `format=xyz+rot6d`로 선언한 native relative action을
> `T_abs = T_state @ T_rel`로 복원하는 별도 `relative_eef_to_absolute()` 경로가 있다.
> 반대 방향의 범용 absolute EEF→relative 변환을 ACT·SmolVLA까지 제공하는 것은 아니므로,
> SO101의 공통 EEF-relative 입력은 `T_rel = inv(T_state) @ T_action`을 계산하는 별도 processor가
> 필요하다.

이 프로젝트의 세 학습 대상만 보면 ACT와 SmolVLA에는 v0.6.0 내장 relative flag가 없고,
GR00T N1.7에만 전용 지원이 있다. 세 모델에 동일한 EEF-relative 계약을 적용하려면 dataset을
미리 relative로 덮어쓰기보다 공통 custom pre/post processor를 policy pipeline에 삽입하고,
그 processor 설정과 relative-action 통계를 checkpoint에 함께 저장하는 설계가 적합하다.

소스 근거:
[`lerobot_train.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/scripts/lerobot_train.py) ·
[`datasets/factory.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/datasets/factory.py) ·
[`policies/factory.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/factory.py) ·
[`relative_action_processor.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/processor/relative_action_processor.py)

</details>

<details>
<summary><strong>LeRobot v0.6.0이 지원하는 VLA 목록</strong></summary>

upstream README가 **VLA Models**로 분류하고 `lerobot-train --policy.type=...` 및 policy factory에서
선택 가능한 모델은 다음 10개다.

| 모델 | `--policy.type` | policy class | 전용 processor factory |
|---|---|---|---|
| π0 (Pi0) | `pi0` | `PI0Policy` | `make_pi0_pre_post_processors` |
| π0-FAST (Pi0Fast) | `pi0_fast` | `PI0FastPolicy` | `make_pi0_fast_pre_post_processors` |
| π0.5 (Pi05) | `pi05` | `PI05Policy` | `make_pi05_pre_post_processors` |
| GR00T N1.7 | `groot` | `GrootPolicy` | `make_groot_pre_post_processors` |
| SmolVLA | `smolvla` | `SmolVLAPolicy` | `make_smolvla_pre_post_processors` |
| XVLA | `xvla` | `XVLAPolicy` | `make_xvla_pre_post_processors` |
| EO-1 | `eo1` | `EO1Policy` | `make_eo1_pre_post_processors` |
| MolmoAct2 | `molmoact2` | `MolmoAct2Policy` | `make_molmoact2_pre_post_processors` |
| WALL-OSS | `wall_x` | `WallXPolicy` | `make_wall_x_pre_post_processors` |
| EVO1 | `evo1` | `Evo1Policy` | `make_evo1_pre_post_processors` |

- **ACT는 지원되지만 VLA가 아니다.** upstream에서는 `act`를 Imitation Learning으로 분류한다.
- **GR00T는 N1.7만 지원한다.** v0.6.0은 N1.5 config/checkpoint를 명시적으로 거부하며, N1.5가
  필요하면 LeRobot 0.5.1을 사용하라는 오류를 낸다.
- VLA-JEPA·LingBot-VA·FastWAM은 이름에 VLA가 포함되거나 VLA backbone을 사용하지만 upstream
  README 분류상 **World Models**라서 위 VLA 10개 목록에서는 제외했다.
- registry에는 Diffusion, VQ-BeT, MultiTask DiT, TDMPC, Gaussian Actor 같은 비-VLA policy와
  third-party `lerobot_policy_*` plugin 확장 경로도 별도로 존재한다.

소스 근거:
[`README — SoTA Models`](https://github.com/huggingface/lerobot/tree/v0.6.0#sota-models) ·
[`policies/__init__.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/__init__.py) ·
[`policies/factory.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/policies/factory.py)

</details>

<details>
<summary><strong>async policy-server ↔ robot-client 추론 파이프라인</strong></summary>

v0.6.0의 async inference는 Python `asyncio`나 비동기 gRPC stub이 아니다.
**동기 gRPC RPC, client의 control/receiver 두 thread, server의 observation queue**를 조합해
action 실행과 다음 chunk 추론을 겹친 구조다. server는 기동 직후 모델이 없는 빈 container이고,
client handshake가 policy와 checkpoint를 선택한다.

```mermaid
sequenceDiagram
    participant R as Robot
    participant C as RobotClient
    participant S as PolicyServer

    C->>S: Ready
    C->>S: PolicySetup
    S->>S: Load model
    S->>S: Load processors

    par Control thread
        C->>R: Execute queued action
        R-->>C: Capture observation
        C->>S: SendObservations
    and Receiver thread
        C->>S: GetActions
        S->>S: Preprocess
        S->>S: Predict chunk
        S->>S: Postprocess
        S-->>C: TimedAction chunk
        C->>C: Merge action queue
    end
```

### gRPC 계약

| RPC | 방향·형태 | payload와 역할 |
|---|---|---|
| `Ready` | unary → unary | 새 client가 server의 observation queue와 predicted timestep set을 초기화 |
| `SendPolicyInstructions` | unary → unary | pickle `RemotePolicyConfig`: policy type, checkpoint 경로, robot feature schema, device, `actions_per_chunk` 전달. server가 model과 checkpoint의 pre/post processor를 로드 |
| `SendObservations` | client-streaming → unary | pickle `TimedObservation`을 2 MiB 조각으로 전송. 연속 관측 stream 하나가 아니라 **관측 한 건마다 호출하는 blocking RPC** |
| `GetActions` | unary → unary polling | server가 observation을 기다려 추론한 뒤 pickle `list[TimedAction]` 반환. timeout이면 빈 response |

### 실행 순서

1. `RobotClient.__init__()`이 로봇을 연결하고 hardware observation schema를 LeRobot feature schema로 만든다.
2. `Ready → SendPolicyInstructions` handshake 후 server는 `get_policy_class(...).from_pretrained(...)`로
   model을 로드하고, 같은 checkpoint에서 `make_pre_post_processors(...)`를 복원한다.
3. client main thread는 `fps` 주기로 action queue에서 한 action을 꺼내 `robot.send_action()`을 호출한다.
   queue 비율이 `queue_size / action_chunk_size <= chunk_size_threshold`가 되면 새 관측을 보낸다.
4. server의 observation queue는 `maxsize=1`이다. 새 관측이 오는데 queue가 차 있으면 이전 것을
   버리므로, 밀릴 때 backlog를 처리하지 않고 **가장 최신 관측**으로 교체한다.
5. receiver thread의 `GetActions`가 관측 하나를 가져와
   `raw robot obs → LeRobot obs → preprocessor → policy.predict_action_chunk()`를 실행한다.
   이 프로젝트 patch는 **full chunk를 한 번 postprocess한 뒤** `actions_per_chunk`로 자르고
   CPU `TimedAction`으로 만든다.
6. client는 이미 실행한 timestep 이하의 stale action을 버리고, 기존 queue와 새 chunk가 겹치는
   timestep은 `aggregate_fn`으로 결합한다. 기본 `weighted_average`는
   `0.3 × old + 0.7 × new`이다.
7. 현재 chunk가 완전히 소진되기 전에 다음 추론이 진행되므로 정상 튜닝 상태에서는 robot이
   inference를 기다리는 idle frame을 줄일 수 있다. queue가 실제로 비면 fallback action은 없으며,
   새 chunk가 올 때까지 추가 command를 보내지 않는다.

### 핵심 파라미터

| 파라미터 | v0.6.0 source 기본값 | 의미 |
|---|---:|---|
| client/server `fps` | 30 / 30 | client control 주기와 server가 `TimedAction`에 부여하는 timestep 간격. 양쪽을 동일하게 유지 |
| `actions_per_chunk` | 필수 | policy 출력 중 네트워크로 돌려줄 길이. policy의 `chunk_size` 이하여야 함 |
| `chunk_size_threshold` | 0.5 | queue가 최대 수신 chunk의 이 비율 이하일 때 새 관측 송신. 높을수록 빠른 재계획·많은 overlap/RPC |
| `aggregate_fn_name` | `weighted_average` | overlap action 결합. `latest_only`, `average`, `conservative`도 지원 |
| `obs_queue_timeout` | 2 s | `GetActions`가 server observation을 기다리는 최대 시간 |
| `inference_latency` | 1/30 s | server `GetActions` 호출의 최소 목표 간격. 실제 추론이 더 느리면 추가 sleep 없음 |

> upstream async 문서 표에는 `chunk_size_threshold` 기본값이 0.7로 적힌 곳이 있지만,
> v0.6.0의 `RobotClientConfig` 실제 기본값과 예제 명령은 **0.5**다.

### 지원 범위와 주의점

- async server의 source allowlist는
  `act`, `smolvla`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`, `groot`의 **8개**다.
  위의 전체 VLA 10개와 같지 않으며 `pi0_fast`, XVLA, EO-1, MolmoAct2, WALL-OSS, EVO1은 빠져 있다.
  이 프로젝트 대상 ACT·SmolVLA·GR00T N1.7은 모두 allowlist에 포함된다.
- camera key는 checkpoint의 `policy.config.image_features`와 맞아야 한다. `RemotePolicyConfig`에는
  `rename_map`이 있지만 stock `RobotClientConfig` CLI에는 이를 노출하는 field가 없어, key가 다르면
  client 쪽 schema를 맞추거나 별도 client wrapper가 필요하다.
- stock 관측 중복 필터는 image를 비교하지 않고 `observation.state`의 L2 distance만 본다
  (`atol=1`). 이 threshold는 EEF 단위에 맞지 않으므로 EEF-relative processor가 감지되면
  해당 필터를 bypass하고 timestamp/stale-action 규칙만 사용한다.
- stock server는 action chunk를 postprocessor에 한 번에 넣지 않고 `(B, action_dim)`으로 한 step씩
  호출한다. 반면 GR00T N1.7의 native relative decoder는 horizon별 stats와 기준 pose 때문에
  `(B, T, action_dim)` 전체를 요구하고 single-step decode를 명시적으로 거부한다. 따라서
  **v0.6.0 stock async 경로 그대로는 EEF-relative chunk와 호환되지 않는다.**
  이 프로젝트는 server와 sync/eval 경로를 full-chunk postprocess + external absolute FIFO로
  patch하고 processor/stats/manifest를 checkpoint에 함께 저장한다.
- transport는 `grpc.insecure_channel`이고 policy config·observation·action에 Python pickle을 사용한다.
  인증·TLS·payload 검증이 없으므로 인터넷에 직접 노출하면 안 된다. client/server를 같은 신뢰
  boundary에 두고 방화벽, VPN 또는 SSH tunnel을 사용한다.
- server instance에는 session ID가 없고 `Ready`가 전역 queue를 초기화한다. 여러 client가 동시에
  접속하면 서로 model/session 상태를 덮어쓸 수 있어 사실상 **server 하나당 active client 하나** 구조다.

이 프로젝트에서는 실기기가 `RobotClient`를 확장한 `eef_robot_client.py`를 사용하고, sim은
`vla_policy_node`가 같은 gRPC/pickle 계약을 구현하되 ROS observation과 자체 `deque`·inference
thread를 사용한다. EEF mode에서는 `policy-server-affine`을 사용하지 않는다. 그 server는
joint-space absolute fallback의 cross-domain affine 전용이다.

소스 근거:
[`robot_client.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/async_inference/robot_client.py) ·
[`policy_server.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/async_inference/policy_server.py) ·
[`configs.py`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/async_inference/configs.py) ·
[`services.proto`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/transport/services.proto) ·
[`async.mdx`](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/async.mdx)

</details>

---

## 현재 PickCube 환경·에셋·cuRobo 평가

현재 정량평가 기준 환경은 `SimToReal-SO101-PickCube-DR-v0`이다. 한 환경에 SO-101 follower,
40 mm `Cube1` 한 개, 그릇 한 개, 책상과 top/wrist/front 카메라가 있으며, 큐브를 집어 그릇에
놓는 과정을 Isaac Sim 물리로 판정한다.

<table>
  <tr>
    <td width="50%"><img src="docs/pics/cube_desk/current_pickcube_top.png" alt="현재 Isaac Sim PickCube 환경 top camera"></td>
    <td width="50%"><img src="docs/pics/cube_desk/큐브와%20그릇.jpg" alt="실물 큐브와 그릇"></td>
  </tr>
  <tr>
    <td align="center"><sub>현재 Isaac Sim 장면: Cube1 한 개·그릇·SO-101, 매트 없음</sub></td>
    <td align="center"><sub>실물 에셋 원형: 40/50 mm 펠트 큐브와 플라스틱 그릇</sub></td>
  </tr>
</table>

### 등록 환경

| Gym ID | 큐브/그릇 배치 | 성공 종료 | 주 용도 |
|---|---|---|---|
| `SimToReal-SO101-Teleop-v0` | 태스크 오브젝트 없음 | 없음 | 로봇·책상·조명 base substrate |
| `SimToReal-SO101-PickCube-v0` | 고정 실측 배치 | 순간 판정 | 결정적 teleop·datagen |
| `SimToReal-SO101-PickCube-DR-v0` | **full DR** 종형 큐브 영역 + 그릇 arc | 순간 판정 | 데이터 다양화·cuRobo sweep |
| `SimToReal-SO101-PickCube-DRBase-v0` | nominal 근처 좁은 사각형 | 순간 판정 | 제한 영역 DR |
| `SimToReal-SO101-PickCube-Eval-v0` | 고정 실측 배치 | 15-step 디바운스 | 재현성 closed-loop 평가 |
| `SimToReal-SO101-PickCube-DR-Eval-v0` | full DR | 15-step 디바운스 | DR closed-loop 평가 |

### 에셋 형상과 치수

| 에셋 | 현재 형상·치수 | 물리/충돌 표현 |
|---|---|---|
| **SO-101 follower** | `shoulder_pan/lift`·`elbow_flex`·`wrist_flex/roll` 5축 + gripper 1축. URDF 주요 관절 원점 간 거리 약 **116 / 135 / 64 mm**, gripper-frame offset 약 **98 mm** | Isaac용 mesh collider와 cuRobo용 **54-sphere / 9-link** 근사 |
| **Cube1/2** | 한 변 **40 mm**, 35 g, corner radius 8.8 mm인 펠트 rounded box. 현재 task는 **Cube1 한 개**만 활성 | visual과 같은 rounded mesh의 `convexHull` |
| **Cube3/4** | 한 변 **50 mm**, 55 g, corner radius 11 mm. 에셋/단일 사양에는 유지되지만 현재 scene에는 미배치 | `convexHull` |
| **그릇** | 회전체 곡면 bowl, 상단 **Ø150 mm**, 바닥 **Ø65 mm**, 높이 **70 mm**, 벽 4 mm, 외부 base 5 mm + cavity floor 3 mm, 250 g | 오목한 내부를 보존한 watertight mesh + `convexDecomposition` |
| **책상** | **1,600 × 800 × 25 mm**, 상판 높이 705 mm. 현재 scene은 desk mat 없음 | 상판 static box collider |
| **카메라** | top · wrist · front RGB 3-view | static camera cfg, 렌더 시 `--enable_cameras` 필요 |

cuRobo는 삼각 mesh를 직접 충돌검사하지 않고 아래 54개 sphere로 근사한다. 링크별 개수는
base 9 · shoulder 6 · upper arm 8 · lower arm 10 · wrist 5 · gripper 6 · moving jaw 7 · camera mount 3이다.

<table>
  <tr>
    <td width="33%"><img src="docs/pics/cuRobo/so101_base.png" alt="SO-101 visual mesh"></td>
    <td width="33%"><img src="docs/pics/cuRobo/so101_collision_model.png" alt="SO-101 54 sphere collision model"></td>
    <td width="33%"><img src="docs/pics/cuRobo/so101_overlay.png" alt="SO-101 mesh and collision sphere overlay"></td>
  </tr>
  <tr>
    <td align="center"><sub>visual mesh</sub></td>
    <td align="center"><sub>54-sphere collision model</sub></td>
    <td align="center"><sub>mesh/sphere overlay</sub></td>
  </tr>
</table>

### DR 큐브 스폰 영역

full DR은 env-local `x ∈ [-0.24, 0.24] m`, `y ∈ [0.06, 0.26] m`의 좌우대칭 종형 영역이다.
종의 x 반너비는 `(y, half-width) = (0.06,0.24), (0.14,0.24), (0.18,0.20),
(0.22,0.16), (0.26,0.08)` m를 선형 보간한다. 이 외곽에서 다음 영역을 제외한다.

| 제외/제약 | 값 |
|---|---|
| 로봇암 제외 박스 | `x=[-0.09, 0.04]`, `y=[-0.045, 0.155]` m |
| 그릇 이격 | 중심 `(-0.22, 0.265)` m에서 **140 mm** 이상 |
| base 최소 도달거리 | shoulder-pan 축 `(-0.021, 0.023)` m에서 **123 mm** 이상 |
| 큐브 간 최소거리 | **60 mm** |
| DRBase 사각형 | `x=[-0.14, 0.06]`, `y=[0.205, 0.305]` m; 나머지 제약은 동일 |

큐브는 full orientation으로 랜덤화하고, 그릇은 반경 0.44 m 원호에서 -4°~+8°로 움직인다.
DR 환경은 여기에 조명·카메라 focal·로봇 색과 큐브 마찰/질량 randomization을 더한다.

![DR 스폰 영역과 yaw-zero 183-cell 결과](docs/pics/cuRobo/model54_yaw_zero_spawn_map.png)

### cuRobo state machine 정량평가 — 54-sphere 최종

`assets/robots/so101.yml`의 **현재 54-sphere 모델**만 사용해 처음부터 재실행한 결과다.
모든 실행은 `num_envs=64`, 실패 셀 재시도 없음, planning 성공과 Isaac 물리 place 성공을
각각 집계했다. 이전 collision-sphere 모델의 중간 결과와 targeted failure replay는 아래 최종 집계에서 제외했다.

| yaw 조건 | seed | 셀 × trial | planning | place | 성공률 | 경과시간 |
|---|---:|---:|---:|---:|---:|---:|
| zero | 0 | 183 × 1 | 183/183 | **183/183** | **100.00%** | 17m 56s |
| random | 0 | 145 × 3 | 435/435 | **435/435** | **100.00%** | 49m 30s |
| random | 1 | 145 × 3 | 435/435 | **435/435** | **100.00%** | 49m 57s |
| random | 2 | 145 × 3 | 435/435 | **435/435** | **100.00%** | 51m 16s |
| **random 합계** | 0–2 | 145 × 9 | 1305/1305 | **1305/1305** | **100.00%** | 2h 30m 43s |

![54-sphere cuRobo 최종 성공률](docs/pics/cuRobo/model54_final_success_rates.png)

<table>
  <tr>
    <td width="50%"><img src="docs/pics/cuRobo/model54_yaw_zero_spawn_map.png" alt="yaw-zero spawn sweep map"></td>
    <td width="50%"><img src="docs/pics/cuRobo/model54_yaw_random_seed0_spawn_map.png" alt="yaw-random seed0 spawn sweep map"></td>
  </tr>
  <tr>
    <td align="center"><sub>yaw-zero: 183/183, 경계 108/108</sub></td>
    <td align="center"><sub>yaw-random seed 0: 435/435 (145셀 × 3회)</sub></td>
  </tr>
</table>

64-env 실행의 실측 peak VRAM은 **34,110 MiB / 48,935 MiB**였고 OOM이나 48/32-env fallback은 없었다.
grasp manifold, chord-center 보정, 5-frame contact hold와 재현 명령은
[`scripts/cuRobo/README.md`](scripts/cuRobo/README.md)에 정리돼 있다.

---

## 환경 요구사항

### 소프트웨어

| 항목 | Windows (실기기) | Linux (시뮬·학습) |
|---|---|---|
| OS | Windows 11 Pro | Ubuntu 24.04 LTS |
| uv | 최신 (Astral) | 최신 (host uv 보조 경로용) |
| Docker | **불필요** | Docker + NVIDIA Container Toolkit |
| NVIDIA Driver | (Isaac Sim 로컬 실행 시) 580+ | 580+ (CUDA 12.8 컨테이너) |
| WSL2 / usbipd | **불필요 (제거됨)** | 해당 없음 |
| Python | 3.12 (실기기 전용 uv project) | policy=3.12 / Isaac=3.11 |

### 하드웨어

| 장치 | 수량 | 비고 |
|---|---|---|
| SO-101 Leader / Follower Arm | 각 1 | Feetech STS3215 서보 × 6 |
| USB-Serial 어댑터 | 2 | CH343 칩 (Windows COM 포트) |
| 카메라 | 1~3 | top · wrist · front. `ENABLED_CAMERAS` 로 부분집합 선택 |
| NVIDIA GPU (RT 코어 + 16GB+) | 1 (Linux 서버) | 시뮬·학습·추론. **H100/A100 은 RT 코어 부재로 Isaac Sim 미지원**. RTX A4000/A5000/A6000·L40(S)·RTX 6000 Ada·RTX PRO 5000/6000 Blackwell·GeForce RTX 40/50 등 |

### 핵심 의존성

버전은 `pyproject.toml` 에 고정. **ABI 호환성 핀이라 임의 `uv lock --upgrade` 금지.**

| 패키지 | 버전 | 위치 |
|---|---|---|
| Python | 3.12 | Windows 실기기·policy-server |
| lerobot[async,core_scripts,feetech] | 0.6.0 | Windows `scripts/real/pyproject.toml` |
| lerobot[smolvla,async,groot] | 0.6.0 | `policy-server` image + pinned patch |
| Python / torch | 3.11 / 2.7.0+cu128 | Isaac Sim host uv 환경 |
| isaacsim | 5.1.0 `[all,extscache]` | `isaac` 그룹 |
| isaaclab | 2.3.2 `[all,isaacsim]` | `isaac` (직접 의존, 외부 래퍼 제거) |

ABI 핀: `numpy==1.26.0` / `pyarrow<19` / `datasets<4.7` / `h5py<3.16` / `torch==2.7.0+cu128` / `torchcodec<0.6` / `packaging<26` / `setuptools<82`. 이유는 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) 와 `AGENTS.md` 참고.

---

## 사전 설치 확인

```bash
# Windows (Git Bash) — 실기기
uv --version

# Linux 서버 — 시뮬·학습
docker --version
nvidia-smi          # Driver 580+ / CUDA 12.8+
```

미설치 항목은 공식 가이드 참고: [uv](https://docs.astral.sh/uv/getting-started/installation/) · [Docker + NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 공통 준비

### Hub / W&B 인증

```bash
uv run hf auth login        # 또는 export HF_TOKEN=hf_xxx
uv run wandb login          # 선택
```

### `.env` 작성

두 머신이 각자 `.env` 를 둔다. `.env.example` 를 복사해 채운다.

```bash
cp .env.example .env
```

| 블록 | 변수 (발췌) |
|---|---|
| §0 시크릿 | `HF_TOKEN` `HF_USER` `WANDB_API_KEY` |
| §1 모델 프로필 | `POLICY_PROFILE`(smolvla/groot_n17/act) — 활성 모델 1줄 선택 |
| §2 하드웨어 | `TELEOP_PORT` `ROBOT_PORT` `ROBOT_ID` `TELEOP_ID` (Windows=COM, Docker=`/dev/ttyACM*`) |
| §3 카메라 | `ENABLED_CAMERAS` `*_CAM_PORT` `CAM_WIDTH/HEIGHT/FPS` |
| §4 데이터 | `SINGLE_TASK` `HF_DATASET_REPO_ID` `NUM_EPISODES` `RECORD_FPS` |
| §5 학습 | `BATCH_SIZE` `TRAIN_STEPS` `OUTPUT_DIR` (Linux 서버) |
| §6 추론 서버 | `POLICY_SERVER_HOST/PORT` `INFERENCE_LATENCY` `OBS_QUEUE_TIMEOUT` (Linux 서버) |
| §7 추론 클라이언트 | `POLICY_SERVER_ADDRESS` `TASK` `ACTIONS_PER_CHUNK` (실기기) |

- **Linux (Docker)**: compose 가 `--env-file .env` + `env/${POLICY_PROFILE}.env` 로 컨테이너에 주입.
- **Windows (native uv)**: 자동 로드 안 됨 → 셸에서 직접 로드: `set -a; source .env; set +a`.

---

## 경로별 가이드

### Windows native uv — 실기기

WSL·Docker·usbipd 없이 Git Bash 에서 직접 실행한다.

```bash
# 1) Python 3.12 + LeRobot 0.6.0 전용 환경 동기화
uv sync --project scripts/real

# 2) .env/profile 자동 로드 wrapper
scripts/real/lerobot.sh find-port
scripts/real/lerobot.sh setup-motors
scripts/real/lerobot.sh calibrate
scripts/real/lerobot.sh record

# 3) EEF profile이면 FK/IK client, absolute profile이면 stock joint client로 자동 분기
scripts/real/lerobot.sh policy-client
```

> 추가 인자는 wrapper 뒤에 그대로 전달된다. 예:
> `scripts/real/lerobot.sh record --dataset.num_episodes=3`.

### Linux Docker — sim VLA 폐루프

```bash
# 3-서비스 폐루프 (ACT · SmolVLA · GR00T-N1.7)
docker compose --env-file .env -f docker/docker-compose.yaml up policy-server isaac-sim vla-ros
```

`scripts/inference/demo_vla.sh start <act|smolvla|groot>` 가 정책 서버·bridge·vla-ros 를 자동 배선한다(livestream :49100). `--eval` 모드로 closed-loop 평가. 세부는 `AGENTS.md` §시뮬레이션 환경.

### Linux Docker — VLA 학습

```bash
# ACT / SmolVLA / GR00T-N1.7 — 공통 EEF-relative processor로 학습
# (모델 선택 = .env 의 POLICY_PROFILE: act | smolvla | groot_n17)
docker compose -f docker/docker-compose.yaml run --rm policy-server train
```

데이터셋·출력은 `.env` §5(`HF_DATASET_REPO_ID`/`OUTPUT_DIR`)에서 라우팅. RL(강화학습)은 제거됨 — VLA 지도학습만.

### Linux Docker — policy-server

```bash
docker compose -f docker/docker-compose.yaml up -d policy-server      # full-chunk async gRPC
```

실기기(Windows)·sim(vla-ros) 양쪽 클라이언트의 공용 추론 백엔드.

### Linux host uv — sim 수동 teleop (보조)

Isaac Lab 로컬 작업(수동 teleop, USD 씬 author)용. Docker 가 아닌 host uv `isaac` 그룹.

```bash
uv sync --group isaac
# v0 = DR-off 고정배치(결정적). teleop 데이터 다양성 필요하면 --task SimToReal-SO101-PickCube-DR-v0
uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickCube-v0
```

---

## 저장소 레이아웃

| 경로 | 내용 |
|---|---|
| `docs/` | 문서 허브 (`pics/` 이미지, `videos/` 동영상) |
| `datasets/` | LeRobot v3 데이터셋 |
| `outputs/` | 모델 체크포인트·학습 산출물 |
| `logs/` | 런타임 로그 (`.gitignore`) |
| `scratch/` | **임시물 전용** (smoke test·debug dump — `.gitignore`, 커밋 안 함) |
| `scripts/` | 진입 스크립트 (`<범주>/` 단위) |
| `src/` | `sim_to_real` · `so101_contract` 패키지 |
| `docker/` · `env/` | Docker 빌드·entrypoint · 모델 프로필 |
| `ros2_ws/` | sim VLA 노드(`so101_vla_policy`) — Docker vla-ros 가 빌드 |

> **Linux 서버**: `datasets`·`outputs` 는 용량 큰 HDD 로 symlink (예: `/DISK1/so101-sim2real/{datasets,lerobot_outputs}`).

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| [`AGENTS.md`](AGENTS.md) | 내부 구조·규칙·자주 쓰는 명령 (개발자용) |
| [`docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md`](docs/EEF_RELATIVE_ACTION_PIPELINE_SPEC.md) | EEF-relative 데이터·processor·checkpoint·FK/IK·rollout 계약 |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | ABI 불일치 · GPU/드라이버 호환 · 의존성 핀 충돌 · USD/씬 물리 |

---

## Reference

- [Isaac Sim 5.1 + Isaac Lab 2.3 + LeIsaac on Windows](https://hackmd.io/@asierarranz/rkg1tvT93gx)
- [Teleoperation | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/teleoperation)
- [Policy Training & Inference | LeIsaac Document](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support)
- [LeRobot action representations](https://huggingface.co/docs/lerobot/action_representations)
- [Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- [isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
- [LeRobot Installation](https://huggingface.co/docs/lerobot/main/installation)
- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
