# SO-101 Sim-to-Real 장기 자율 개발 마스터플랜 (Codex→Claude 오케스트레이션)

> 이 문서는 Codex `/goal`의 입력 스펙이다. 변하지 않는 목표·아키텍처·계약을 담는다.
> 운영 현황은 [`../TASKS.md`](../TASKS.md), 서사형 인계는 [`../CONTEXT.md`](../CONTEXT.md) 참조.
>
> **2026-06-22 실행 계층 갱신:** 학습 roadmap은 아래 역사적 phase를 유지하지만 sim↔real
> 배포 runtime은 [`PATH_F_CANONICAL_PARITY.md`](PATH_F_CANONICAL_PARITY.md)의
> Isaac Sim 6 / Isaac Lab 3 / ROS Jazzy canonical 경로가 현재 기준이다. Isaac 5.1 경로는 rollback이다.

## Context

목표는 두 가지를 동시에 만족하는 것이다.

1. **개발 목표(WHAT)**: 50개 인간 시연으로 시작 → Isaac Lab 병렬 시뮬에서 RL 전문가 학습 → DR 롤아웃으로 대량 데이터 생성 → GR00T N1.5 증류(IL) → 실 SO-101 배포.
2. **실행 방식(HOW)**: AI Agent가 다른 AI Agent를 오케스트레이션하며 **장기간 무인 자율**로 진행. Context Compaction 전후로 방향이 유지되고, 모든 산출물이 **자기검증(self-verification)** 가능하며, **체크리스트로 현황 관리**가 된다.

확정된 결정(5개):

| 항목 | 결정 |
|---|---|
| 오케스트레이션 토폴로지 | **Codex(상위 플래너) → Claude Code CLI(구현 워커)** 디스패치 |
| Isaac 설치·머신 역할 | Windows와 konan147에 Isaac Sim 6 canonical runtime. 서버는 GPU 중량 작업 전담, Windows는 실기기 client와 parity 검증. Isaac 5.1은 rollback |
| 자율 범위 | **시뮬 전구간 A~E 무인 자율**. 실기기(F~G)는 사용자 개입 게이트 |
| 상태 관리 | **CONTEXT.md(서사 핸드오프) + TASKS.md(구조화 체크리스트)**, git tracked |
| 시뮬 스택 | 기본 실행=`Isaac Sim 6.0.0.1 + Isaac Lab 3 beta2 + PhysX`; 기존 `5.1.0 + Lab 2.3.2`는 학습/rollback 경로 |

Compaction이 일어나도 Codex/Claude는 이 문서 + CONTEXT.md + TASKS.md만 다시 읽으면 방향과 현황을 복구한다.

---

## 0. 자율 계약 + Codex `/goal` 진입

사용자가 Codex에 `/goal`로 넘기면 Codex는 본 문서 + `TASKS.md` + `CONTEXT.md`를 읽고 §2.3 루프를 무인 구동한다.

### 자율 계약 (시작 후 발동, A~E 종료까지 유지)
1. **사용자에게 묻지 않는다.** A~E(시뮬 전구간)가 끝날 때까지 Codex는 자율 판단으로 진행한다. 승인 대기 금지.
2. **사전 승인된 행동**(런타임 게이트 아님): 서버 Isaac Sim 설치, `uv sync`, RL/GR00T 학습 run 시작·중단, 롤아웃 대량 생성, `git commit`·`push origin`, 컨테이너 빌드/실행, `/DISK1/so101-sim2real` 쓰기.
3. **멈추는 경우는 둘뿐**:
   - **실기기 경계(F~G 진입)** — USB·카메라·물리·안전상 사람 필요 → A~E 완료 보고 + F 준비 체크리스트 제시 후 자율 트랙 종료.
   - **복구 불가 블로커** — 동일 task 재시도(스펙 수정 포함) **3회** 실패 시 TASKS.md에 `blocked` + 사유 기록하고, 의존 없는 다른 task로 우회. 우회 불가하면 그때만 보고.
4. **자율 ≠ 무분별**: 비용 큰 작업도 멈추지 않되, GPU 직렬화(§4.3)·검증 게이트(§7)·불변 계약(§1)은 **반드시** 기계적으로 지킨다. 게이트 미통과 task는 done 금지.
5. **진행 상태는 항상 git에 남긴다.** 매 사이클 TASKS.md·CONTEXT.md 갱신 후 push — 그래야 compaction·세션 교체·머신 전환에도 방향이 유지된다.

---

## 1. 불변 계약 (Invariant / North Star — 매 사이클·매 compaction 후 재확인)

시뮬에서 만드는 **모든 롤아웃 데이터·정책 I/O**는 현재 실기기 데이터셋 스키마(`datasets/pick_pen/meta/info.json`, `meta/tasks.parquet`, `data/**/file-*.parquet` 으로 확인)와 **정확히 동일**해야 한다. PickCube 전환 후에도 action/state/camera feature 계약은 그대로 유지하고, task 문자열만 cube task로 바꾼다. 한 글자라도 어긋나면 GR00T fine-tune/배포에서 깨진다.

| 필드 | 고정값 |
|---|---|
| codebase_version | `v3.0` |
| robot_type | `so_follower` |
| action / observation.state | 각 **6-dim joint position** (순서: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) |
| 카메라 | `observation.images.{top, wrist, front}` · 480×640×3 · h264 · **fps 30** |
| task 문자열 | `"pick up the cube and place it in the bowl"` |

이 계약을 **기계가 강제**하도록 `scripts/validate_lerobot_schema.py`(§7)를 가장 먼저 만든다. 모든 데이터 생성 단계는 이 validator를 통과해야 done 처리된다.

---

## 2. 오케스트레이션 아키텍처

### 2.1 토폴로지

```
                ┌──────────────────────────────────────────────┐
                │  ORCHESTRATOR = Codex (상위 플래너/게이트키퍼)   │
                │  - TASKS.md / CONTEXT.md / 본 계획 읽기           │
                │  - 다음 actionable task 선택 + 정밀 스펙 작성     │
                │  - 검증 게이트 판정 → 상태 파일·git 갱신          │
                └───────────────┬──────────────────────────────┘
                                │ dispatch (headless, 1 task = 1 spec)
                ┌───────────────▼──────────────┐
                │  WORKER = Claude Code CLI      │
                │  claude -p "<spec>" \          │
                │    --output-format json \      │
                │    --permission-mode acceptEdits│
                │  - 구현 + 자기검증 명령 실행      │
                │  - 결과 JSON(검증 로그 포함) 반환 │
                └───────┬───────────────┬────────┘
                        │ 로컬(Windows)  │ ssh konan147 (서버)
                  경량 검증·실기기 코드   GPU 작업(Isaac/RL/롤아웃/GR00T)
```

- 같은 워크플로가 **머신 무관**: Codex가 GPU 작업은 `ssh konan147 '...'`로, 실기기/경량은 로컬로 라우팅.
- **1 task = 1 dispatch = 1 검증**. 워커는 작게 유지(컨텍스트 폭발·compaction 방지). 긴 작업은 task를 잘게 쪼갠다.

### 2.2 상태 파일 (compaction-resilient)

| 파일 | 역할 | 갱신 주체 |
|---|---|---|
| `docs/SIM2REAL_MASTERPLAN.md` (이 문서) | 변하지 않는 목표·아키텍처·계약 | 사람(중대 변경만) |
| `TASKS.md` | 구조화 체크리스트(§8). 단일 진실 공급원 | Codex |
| `CONTEXT.md` | 서사형 작업 인계(기존 관례 유지) + 상단에 **North Star 요약** 고정 | Codex/Claude |

**Compaction 복구 프로토콜**: 어떤 에이전트든 세션 시작/compaction 직후 반드시 순서대로 재로드 — ① 마스터플랜 §0·§1·§7 → ② TASKS.md 현재 phase·in_progress·blocked → ③ CONTEXT.md 최근 1~2개 인계. 이 3개만으로 "지금 무엇을, 왜, 다음 명령은" 복구. 추측 금지 — 상태 파일에 없으면 새 task로 만든다.

### 2.3 루프 프로토콜 (Codex 1 사이클)

```
1. RELOAD   : 마스터플랜 §0·§1·§7 + TASKS.md + CONTEXT.md 최근 인계 1~2개 읽기
2. SELECT   : depends_on 충족 + status=todo 중 우선순위 1개 선택 (todo 없으면 §0 자율계약 따라 F 경계 보고 후 종료)
3. SPEC     : 파일경로·변경범위·재사용 심볼·검증명령을 담은 정밀 스펙 작성
4. DISPATCH : claude -p 워커 실행(로컬 or ssh). GPU 중복 점유 금지(§4.3 직렬화)
5. VERIFY   : 워커가 반환한 검증 로그 + Codex가 직접 게이트 명령 재실행
6. RECORD   : TASKS.md 상태 전이(done/blocked), CONTEXT.md 인계 1줄, git commit & push origin
7. LOOP     : 2로. §0 자율계약상 멈추는 경우(F~G 진입 / 복구불가 블로커)에만 종료·보고. 그 외 무인 지속
```

### 2.4 디스패치 메커니즘 (구체)

- 워커 호출(검증된 probe 재사용 — CONTEXT.md 2026-06-03 참조):
  `claude -p "<spec>" --output-format json --permission-mode <검증된 값> --model "sonnet[1m]" --effort high --tools "Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow" --allowedTools "Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow"`
- 워커 결과 JSON 인터페이스는 고정: `{task_id,status,changed_files,verification,notes}`. Codex는 이 로그를 참고하되 §7 게이트 명령을 직접 재실행한 뒤 `done` 처리한다.
- **서버 SSH 주의**: 비대화형 ssh는 PATH에 `claude`/`codex`/`uv`/`docker` 미노출 가능(로그인 셸 아님). 디스패치 스크립트는 절대경로 사용 또는 `bash -lc`로 profile source. 서버 무거운 산출물은 `/`(120GB) 말고 **`/DISK1`(3.4TB)** 로.
- **git sync 허브**: 양 머신 공통 `origin = github PubCyBerry/SO101-Sim2Real`. 상태 파일·코드는 origin 경유. ⚠️ 로컬 working dir 이름(`SO101-LeRobot-VLA`)≠서버 클론(`~/Workspaces/SO101-Sim2Real`)≠로컬 내부 gitea `konan` 리모트 — **sync는 origin 하나로 표준화**, T0.0에서 명시.
- 오케스트레이션 코드: `scripts/orchestrator/` (loop.py = Codex 사이클, dispatch.sh = claude -p 래퍼 로컬/ssh, gate.py = 게이트 판정). T0.4에서 신설.

---

## 3. 머신 역할 분배

| | Windows 워크스테이션 (A4000 16GB) | 서버 konan147 (Blackwell 48GB) |
|---|---|---|
| 주 역할 | 실기기 클라이언트·경량 검증·코드 편집·**오케스트레이터 호스트(Codex)** | **모든 GPU 중량 작업** |
| Isaac Sim | **6.0 canonical client 설치**(A4000) | **6.0 canonical + 5.1 rollback 설치**(Blackwell) |
| GR00T fine-tune | 불가(16GB<25GB) | **전담** |
| 추론 | RobotClient(실기기) | PolicyServer(GR00T, ~6GB) |
| 데이터 저장 | 작업 캐시만 | `/DISK1/so101-sim2real` (롤아웃·체크포인트·HDF5) |

자율 루프(A~E)는 **사실상 서버 중심**. Windows는 오케스트레이터를 띄우고 ssh로 서버를 운전. (Windows에서 Codex 상시 구동이 부담이면 서버에서 Codex 구동도 가능 — T0.0에서 호스트 확정.)

---

## 4. 리소스 예산 (추정)

### 4.1 스토리지 (서버 `/DISK1` 3.4TB 여유 — 제약 아님)
앵커: 현 데이터셋 50ep=333MB(비디오 330MB) → **ep당 ~6.7MB**.

| 산출물 | 추정 | 위치 |
|---|---|---|
| 롤아웃 데이터셋 5,000ep | ~35GB (10kep≈70GB) | `/DISK1` |
| 롤아웃 중간 HDF5(있으면) | 데이터셋과 동급, 변환 후 삭제 | `/DISK1` (휘발) |
| GR00T 체크포인트(3B BF16 ~6GB×N) + optimizer state | run당 ~50–100GB | `/DISK1` |
| Isaac Sim extscache | ~20GB+ (1회) | `/DISK1`(심볼릭) — `/` 120GB 보호 |
| 기존 server outputs | 현재 9.1GB가 `/`에 있음 | **`/DISK1`로 이전** |

### 4.2 VRAM (서버 48GB, 단계는 대부분 순차)

| 단계 | env 수 | 카메라 렌더 | VRAM 추정 |
|---|---|---|---|
| B: state-기반 RL 학습 | 2048–4096 | OFF | ~6–16GB |
| C: 롤아웃(2캠 480×640 렌더) | 16–64로 축소 | ON | ~10–24GB |
| D: GR00T N1.5 full fine-tune | — | — | ~25GB (부족 시 LoRA/`--no-tune_diffusion_model`) |
| F: GR00T 추론 | — | — | ~6GB |

### 4.3 GPU 직렬화 규칙 (오케스트레이터 강제)
48GB 1장. **RL 학습 ∥ 롤아웃 ∥ GR00T 학습 동시 금지.** Codex는 디스패치 전 `nvidia-smi`로 free VRAM 확인하고 `/DISK1/so101-sim2real/run/gpu.lock`으로 중량 작업을 직렬화한다. 임계 미달이면 큐잉. RAM(서버 125GB)·디스크는 병목 아님.

---

## 5. 로드맵 — 현 레포 기준 재구성

> 범례: ✅재사용 가능 / 🔧개조 / 🆕신규(DIY) / ⛔사용자 게이트

### Phase 0 — 부트스트랩 + de-leisaac (sim-critical) 🆕🔧
- 0a. 오케스트레이션 인프라: `scripts/orchestrator/`, `TASKS.md`, CONTEXT.md North Star, git sync 표준화. 🆕
- 0b. **불변 계약 validator** `scripts/validate_lerobot_schema.py` (§7) — 최우선. 🆕
- 0c. 서버 Isaac 설치: user-local `uv` 설치 → `pyproject.toml`/`uv.lock`의 leisaac 제거와 Isaac direct dependency 전환 → `uv sync --group isaac` → headless smoke. extscache→`/DISK1/so101-sim2real`. 🆕 (§0에서 사전 승인됨)
- 0d. **leisaac 제거(sim-critical만)**: §6. 순수 Isaac Lab `ManagerBasedRLEnvCfg`로 pick_pen 재작성(scene+robot+obs+**reward**+termination+events 명시). 🔧
- **게이트**: `uv run`로 `SimToReal-SO101-PickPen-v0` gym.make→reset→random step 500회 무크래시 + obs/action shape가 6-dim 계약 일치.

### Phase A — 씬·드라이브·카메라 정합 🔧
- USD 씬 ✅(`assets/scenes/pen_desk/`, leisaac 비의존). `ASSETS_ROOT`만 자체 경로로 교체.
- SO-101 articulation: **position PD 드라이브**로 Feetech STS3215 근사(stiffness/damping/속도·토크 한계). 🔧
- 카메라 2대(top/wrist) extrinsic/intrinsic을 실기와 정합, 480×640@30 고정.
- **게이트**: 정적 reset에서 펜 4개·펜컵이 의도 영역(그린 타원/주황 호)에 spawn, 관통·바운스 없음.

### Phase B — RL 전문가(state-based) 🆕
- leisaac엔 reward 없음 → §6에서 만든 새 env에 **단계형 reward**(reach→grasp→lift→transport→insert→release) + success 보너스 + action-rate 페널티.
- **rsl_rl PPO**(Isaac Lab 2.3.2 번들). 진입점 `scripts/reinforcement_learning/train.py`(Isaac Lab 표준 래퍼). 🆕
- 커리큘럼: 펜·펜컵 spawn 영역 점진 확대(기존 `domain_randomization.py` 타원/호 ✅).
- **게이트**: eval 롤아웃 success rate ≥ 임계(초기 70%→목표 90%).

### Phase C — 데이터 엔진(롤아웃→LeRobot v3) 🆕
- 학습된 전문가를 DR+2캠 렌더로 롤아웃, **성공 ep만** 필터.
- **롤아웃→LeRobot v3 recorder** 신규(leisaac LeRobotRecorderManager 대체). 🆕
- (선택) squint식 segmentation 배경 오버레이로 시각 갭 축소 — 카메라별 정합.
- **게이트**: 생성 데이터셋이 `validate_lerobot_schema.py` 통과. 소규모(200ep)로 파이프라인 관통 검증 후 확장.

### Phase D — GR00T N1.5 증류(IL) 🔧✅
- 학습 인프라 ✅(`docker/Dockerfile.policy`·policy-entrypoint train·`env/groot.env`). 데이터 소스만 sim 롤아웃으로.
- (i)순차(sim 대량→50 real 미세조정) vs (ii)co-training(혼합) 둘 다 만들어 §E 비교.
- modality config 재사용(2캠+state6+action6).
- **게이트**: 학습 완료 + held-out action MSE 산출 + checkpoint config.type=groot.

### Phase E — 평가 🆕
- open-loop(action MSE) + closed-loop(sim success rate). 3원 비교: ①인간50 only ②sim증강+GR00T ③순수RL.
- **게이트**: 비교표 자동 생성 → 사용자 보고(여기까지 무인).

### Phase F~G — 실기기 배포·Sim2Real 루프 ⛔
- LeRobot Async(서버 PolicyServer=GR00T, Windows RobotClient=so101_follower+2캠) ✅인프라.
- **사용자 개입 필수**(USB·카메라·물리·안전). 자율 루프는 E 완료 시 멈추고 F 준비물 체크리스트를 사용자에게 제시.

---

## 6. de-leisaac 분리 (스코프 핵심)

결합 8파일. **자율 sim 트랙(A~E)에 teleop 디바이스 레이어는 불필요**(인간 teleop 대신 RL전문가+oracle 롤아웃).

### 6a. sim-critical (T0.3에서 즉시 제거·재구현)
| 현 leisaac 의존 | 대체 |
|---|---|
| `SingleArmTask{Env,Scene,Direct}Cfg`, `SingleArmObservationsCfg`, `SingleArmTerminationsCfg` (`pick_pen_env_cfg.py`, `direct/pick_pen_env.py`) | 순수 `ManagerBasedRLEnvCfg`/`DirectRLEnvCfg` + SO-101 `ArticulationCfg` 자체 정의 |
| `parse_usd_and_create_subassets` | `usd-core`(공용 dep) 래퍼 신규 |
| `from leisaac.enhance.envs.mdp import *` (`mdp/__init__.py`) | 표준 `isaaclab.envs.mdp` + 자체 term |
| `is_so101_at_rest_pose` (`terminations.py`) | threshold 판정 자체 구현(간단) |
| `ASSETS_ROOT` (`pen_desk.py`) | 자체 경로 상수 |
| `domain_randomization`/`randomize_camera_uniform` (leisaac 래퍼) | 자체 event term(우리 `domain_randomization.py` 타원/호는 이미 ✅ 독립) |
| pyproject `isaac` 그룹의 `leisaac[isaaclab,gr00t]` + `[tool.uv.sources]` leisaac + uv.lock | 제거, `isaaclab` 직접 의존으로 교체 |

### 6b. deferred (실기기/인간 트랙 — A~E 불필요, F 직전 처리)
`teleop_se3_agent.py`(SO101Keyboard/Gamepad/Leader/Remote/Bi/LeKiwi), `replay.py`, `so101_joint_state_server.py`(FeetechMotorsBus), HDF5 teleop recorder(`StreamingRecorderManager` 등). → 별도 경량 device 패키지로 분리하거나 F 단계에서 재구현. **Phase 0에서는 import만 격리(파일 보존, 자율 트랙에서 미참조)**.

기존에서 차용할 기술: 타원/호 DR(이미 자체), pick_pen geometry helper(`observations.py` 이미 자체), USD 씬 자산, 좌표/물리 상수, TROUBLESHOOTING의 USD 물리 튜닝 노하우.

---

## 7. 자기검증 게이트 (기계 판정)

각 phase done = 아래 명령이 통과해야 함. Codex가 §2.3 VERIFY에서 재실행.

| Phase | 검증 명령(개념) | 통과 기준 |
|---|---|---|
| 0b | `python scripts/validate_lerobot_schema.py datasets/pick_pen` + `--self-test` | 현 데이터셋이 §1 계약 전부 만족(`info.json`, `tasks.parquet`, data parquet schema까지 확인) |
| 0d/A | `uv run scripts/.../env_smoke.py` (gym.make→reset→500 step) | 무크래시 + obs/action 6-dim |
| B | `uv run scripts/.../eval_success.py --ckpt ...` | success_rate ≥ 임계 |
| C | `python scripts/validate_lerobot_schema.py <new_dataset>` | 계약 통과(가장 중요) |
| D | checkpoint config + held-out MSE 스크립트 | type=groot, MSE 산출 |
| E | 비교표 생성 스크립트 | 3원 비교 산출 |

`validate_lerobot_schema.py`는 단일 핵심 도구 — 0b에서 현 데이터셋으로 자기검증(reference oracle) 후 C에서 재사용.

---

## 8. TASKS.md 구조

각 항목 필드: `id | 설명 | machine | depends_on | verify(명령/기준) | status | artifact경로 | [GATED]`.
- Codex는 verify 통과 전 done 금지. blocked는 사유 1줄.
- 상태: `todo | in_progress | blocked | done | gated`.
- 매 사이클 SELECT 전 재로드.

CONTEXT.md: 상단에 **North Star 요약 블록**(불변 계약) 고정 + 기존 `## 작업 인계 (날짜 — 제목)` 관례로 사이클마다 1블록 추가.

---

## 9. 리스크·캐비엇

- **정밀 삽입 난이도**: 가는 펜+좁은 홀더. 성공기준 단계화(입구 진입→완전 삽입)로 출발.
- **액추에이터 갭**: 저가 position 서보 → USD PD 튜닝 + 실데이터 co-training 전제(완전 zero-shot 금지).
- **시각 갭(2캠)**: 카메라별 오버레이·정합.
- **GPU 직렬화 위반**: 동시 학습으로 OOM. §4.3 가드 필수.
- **불변 계약 위반**: validator 게이트로 기계 차단.
- **자율 루프 폭주/비용**: §0 자율계약상 비용 큰 작업도 멈추지 않음 → 대신 GPU 직렬화(§4.3)·검증 게이트(§7)·계약(§1) 미통과 시 done 금지로 폭주를 기계 차단. blocked N회면 우회.
- **레포 3중 리모트 혼선**: origin 단일화 안 하면 상태 분기. T0.0에서 해결.
- **de-leisaac 회귀**: base cfg 재작성이 obs/action 차원을 바꾸면 전 파이프라인 깨짐 → env_smoke가 6-dim 검사.

---

## 10. 부트스트랩 (Codex `/goal` 시작 직후, 무인)

1. **T0.0 Codex preflight 먼저**: `origin` 단일화, 서버 clean 확인, `/DISK1/so101-sim2real` writable 확인, `claude`/`docker`/`nvidia-smi`/`gh`/`jq`/`yq`/`uv` 가용성 기록. `uv` 부재는 blocker가 아니라 T0.2 설치 항목으로 넘긴다.
2. **첫 worker 사이클은 T0.1**: Claude worker가 validator를 작성하고, Codex가 `datasets/pick_pen` + `--self-test`를 직접 재실행해 done 처리한다.
3. **오케스트레이터 스켈레톤**(`scripts/orchestrator/`) + claude -p 디스패치 1-task 드라이런(예: T0.1 재검증)으로 루프 e2e 확인.
4. T0.2 서버 Isaac 설치/의존성 전환 → T0.3 de-leisaac → Phase A~E 무인 진행. E 완료 시 §0대로 멈추고 F 체크리스트 보고.

## 11. 오케스트레이션 시스템 자체 검증 (e2e)

자율 개발을 켜기 전, **루프가 동작하는지** 먼저 증명:
- 드라이런: Codex가 TASKS.md에서 T0.1 SELECT → claude -p DISPATCH → `{task_id,status,changed_files,verification,notes}` JSON 파싱 → validator 게이트 재실행 → TASKS.md done 전이 → CONTEXT.md 인계 → git commit/push. 1바퀴 무인 완주.
- Compaction 복구 모의: 새 세션에서 마스터플랜+TASKS.md+CONTEXT.md만 주고 "다음 명령"을 정확히 도출하는지 확인.
- 서버 라우팅: `ssh konan147 'nvidia-smi'`를 디스패치 경로(비대화형 PATH 포함)로 성공.

---

## 12. 에이전트 보조 도구 (있으면 활용)

상세·설치는 [`AGENT_TOOLING.md`](AGENT_TOOLING.md). 요지만:

- **ovrtx USD 스킬**(`.claude/skills`·`.agents/skills`): Claude는 description으로 자동 호출. **USD/SemanticsAPI 부분만 차용**하고 ovrtx 렌더러 호출은 Isaac Sim/Replicator API로 치환할 것. Phase C 배경 오버레이는 `semantic-labels` 활용.
- **USD Code MCP / Isaac Sim MCP / Kit MCP**(등록되면 자동 노출): USD·Isaac extension/settings/API 막힐 때 우선 조회. 단 **검색기일 뿐** — 답은 실제 import·게이트로 교차검증. (`NVIDIA_API_KEY` 필요 → 발급 전엔 미가용, 없어도 진행.)
- **ovphysx**(검증 라이브러리, 자동 사용 안 됨 → 명시 활용): **Phase A 물리 게이트(TA.2)를 Isaac Sim 부팅 없이 고속 검증**. 독립 스크립트(`scripts/validate_scene_physics.py`)에서만, Isaac과 다른 프로세스로.

도구 가용 여부와 무관하게 §7 검증 게이트·§1 불변 계약은 그대로 강제.

---

## 핵심 파일 (신규 N / 수정 M)

- N `docs/SIM2REAL_MASTERPLAN.md` (이 문서), `TASKS.md`
- N `scripts/validate_lerobot_schema.py` (불변 계약 oracle)
- N `scripts/orchestrator/{loop.py, dispatch.sh, gate.py}` (`gpu.lock` 직렬화 포함)
- N `scripts/reinforcement_learning/train.py` (rsl_rl PPO 래퍼), `scripts/.../eval_success.py`, `scripts/.../env_smoke.py`
- N 롤아웃→LeRobot v3 recorder (`scripts/sim/rollout_to_lerobot.py`)
- M `src/sim_to_real/tasks/pick_pen/{pick_pen_env_cfg.py, direct/pick_pen_env.py, mdp/__init__.py, mdp/terminations.py}` (de-leisaac + reward)
- M `src/sim_to_real/assets/scenes/pen_desk.py` (ASSETS_ROOT 제거)
- M `pyproject.toml` (leisaac 제거, isaaclab 직접 의존, validation 그룹 보존), `uv.lock`
- M `CONTEXT.md` (North Star 블록), `AGENTS.md`·`docs/PATH_C_ISAAC_SIM.md` (leisaac→순수 Isaac Lab 반영)
- 보존(미참조 격리): `scripts/environments/teleoperation/*`, `so101_joint_state_server.py` (deferred device 트랙)
