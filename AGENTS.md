# AGENTS.md

SO-ARM101 6축 로봇 팔 **Sim-to-Real 파이프라인**. Isaac Sim 5.1 시뮬에서 VLA 정책을 학습·검증하고 실기기 SO-101 에 배포한다.

**2-머신 구조**: 실기기는 **Windows 워크스테이션(native uv)**, 시뮬·학습·추론은 **Linux 서버(Docker)**.

본 문서는 **이 저장소에서 작업하는 규칙**만 다룬다. 시스템이 *무엇인지*(계약·스키마·수치)는 명세서에 있다.

| 문서 | 내용 |
|---|---|
| `docs/SPEC.md` | **시스템 명세서 정본** (as-built). env·계약·데이터·런타임·인터페이스·파이프라인·암묵지 |
| `README.md` | 허브 (설치·경로별 quickstart·2-머신 개요) |
| `docs/TROUBLESHOOTING.md` | 트러블슈팅 (ABI·GPU·의존성 핀·USD/씬 물리) |

> **역할 분리**: 어기면 사람 잘못인 것(규칙)은 여기, 코드가 그런 것(사실)은 `docs/spec/`.
> 수치·스키마를 여기에 다시 적지 말 것 — 갈라진다.

## 실행 경로 — 2-머신

| | Windows 워크스테이션 | Linux 서버 |
|---|---|---|
| **실행 방식** | native uv + `pyproject.toml` (WSL·Docker 없음) | Docker (`docker compose ...`) |
| **진입점** | LeRobot CLI (`uv run lerobot-<mode>` · `robot_client`) | policy-server · isaac-sim · vla-ros · pink-ik · curobo-datagen |
| **작업** | teleop · record · replay · calibrate · setup-motors · find-port · **실기기 policy-client** | sim VLA 폐루프 · VLA 학습 · 추론 서버 · datagen |
| **정책** | ACT · SmolVLA · GR00T-N1.5 (서버에서 추론, gRPC) | 동일 (env=추론/데이터 기판, RL 제거) |
| **스택** | LeRobot 0.4.4 (pyproject `teleop`+`async`) | Isaac Sim 5.1 / IsaacLab 2.3.2 / LeRobot 0.5.1 |

> 상세 = `docs/spec/01_OVERVIEW.md` §2 · `docs/spec/06_RUNTIME_SPEC.md`

- **단위·codec 은 단일 소스만 쓴다**: `src/so101_contract/feature_codec.py`(policy-feature ↔ sim) ·
  `leader_calibration.py`(실 leader ↔ sim) · `follower_calibration.py`(실 follower ↔ sim).
  **셋은 다른 변환이다. 수식을 다른 파일에 복제하지 말 것.** 상세 = `docs/spec/04_IO_CONTRACT.md`
- 등록 env 6종·DR 모드·데이터 생성 4경로 = `docs/spec/03_ENV_SPEC.md` §2 · `docs/spec/08_PIPELINES.md`

## 환경 사양

하드웨어·GPU 요구는 `docs/spec/06_RUNTIME_SPEC.md` §8.

**테스트 스위트·lint config 없음** (`tests/`, `ruff.toml`, `pre-commit-config.yaml` 등 미정의).
변경 검증 = 컨테이너 빌드(`docker compose config`/`build`) + 실기기 실행 + `uv run` 시뮬 1회 실행
+ 스크립트 내장 self-check(`docs/spec/08_PIPELINES.md` §10).

## 실기기 native uv (Windows)

실기기 SO-101 제어는 WSL·Docker·usbipd 없이 Windows 호스트의 native uv 로 한다.

- **`.env` 로드**: 자동 안 됨 → Git Bash `set -a; source .env; set +a` 후 `uv run lerobot-<mode>`.
- **주의**: `evdev` 는 linux 전용(Windows 미설치, `sys_platform` 게이트).
  `--robot.type` 거부 시(huggingface/lerobot#3078) robot config 선import 또는 lerobot 0.4.5+.
- CLI 모드·변수→인자 매핑 = `docs/spec/08_PIPELINES.md` §2 (`scripts/real/lerobot.sh` 래퍼)

## Docker (Linux 서버)

서비스 5종·볼륨·entrypoint 모드·env var 69개·모델 프로필 = **`docs/spec/06_RUNTIME_SPEC.md`**.

- 빌드: `docker compose -f docker/docker-compose.yaml build <서비스>`
- `docker/groot_compat_patch.py` 는 **GR00T-N1.5 필수 — 삭제 금지**. 대상 형태가 다르면 빌드를
  중단하는 버전 트립와이어다. lerobot·transformers 업그레이드 시 이 패치부터 점검
  (`docs/spec/09_TACIT_KNOWLEDGE.md` §7.3).

## 진입 스크립트 (`scripts/`)

**배치 규약 (charter)** — 폴더 = 동사/단계 1개. "이 스크립트 어디?"는 *하는 일* 로 1:1 결정.

| 폴더 | 배치 규칙 (한 줄) |
|---|---|
| `assets/` | 기존 로봇 USD 물리 편집 (collision·joint limit) |
| `environments/` | sim 환경 구성·상호작용: 씬 author, env 조회, teleop, 머티리얼 |
| `datagen/` | **새 데이터 생성**: SM·실기기·pink 궤적 record + SM replay |
| `cuRobo/` | **cuRobo 2-proc pick-place SM**: planner · executor · sweep 시각화 |
| `convert/` | 기존 데이터셋 **포맷·프레임 변환** |
| `data/` | 데이터셋 **사후 ops**(변환 아님): 병합·업로드 |
| `inference/` | **VLA 폐루프**(정책 경유) |
| `contract/` | I/O 계약·스키마 검증 |
| `real/` | Windows 실기기 native uv CLI |
| `ece_4560/` | 보관용 과정 프로젝트(격리, 파이프라인 무관) |

> 3중 모호 해소: **새 에피소드 만든다→datagen·cuRobo · 포맷/좌표 바꾼다→convert · 병합/업로드→data.**

스크립트별 인자·동작·검증은 `docs/spec/08_PIPELINES.md`.

## 씬 재생성

USD 6개(`scene.usd` + 객체 5개)는 `scripts/environments/author_pick_cube_scene.py` 로 일괄 재생성한다.
좌표 변경 시 `src/sim_to_real/assets/scenes/cube_desk.py` 의 `SCENE_OFFSET` 을 갱신하고 재실행하며,
`BOWL_CENTER_XY` 같은 world-frame 상수는 `spawn_area.py`·`pick_cube_env_cfg.py` 와 동기화한다.
치수·물리 상수 = `docs/spec/03_ENV_SPEC.md` §9.

## Python 패키지

- **패키지 이름** `sim_to_real` + `so101_contract` (`pyproject.toml`, `where=["src"]` editable).
- **leisaac 은 런타임 의존성이 아니다** — 유용한 코드(`devices`·`datagen`·`assets/robots`·`utils`)만
  vendor 했고 leisaac 내부 import 는 0건이다.
- 의존성 그룹·ABI 핀 = `docs/spec/06_RUNTIME_SPEC.md` §7.

> ⚠ **임의 업그레이드 / `uv lock --upgrade` 금지.** 핀은 Isaac Sim 5.1 번들과 ABI 로 묶여 있다.

## 사용자 환경 컨벤션

- 사용자에게 CLI 안내 시 Windows=Git Bash, Linux=Bash 기준.
- HF/W&B 토큰은 `.env` 에서 읽음 (`.env.example` 템플릿).
- 표준 실행 패턴:
  - 실기기(Windows): `set -a; source .env; set +a` 후 `uv run lerobot-<mode> ...`
  - 시뮬 폐루프(Linux): `docker compose --env-file .env -f docker/docker-compose.yaml up policy-server isaac-sim vla-ros`
  - 시뮬 teleop(Linux host uv): `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task SimToReal-SO101-PickCube-v0 ...`

## 운영 규칙

### 명세서 갱신 — 코드를 바꾸면 문서도 바꾼다

`docs/spec/` 은 코드의 사본이다. 계약·상수·인터페이스를 바꾸면 해당 절도 갱신한다.

```bash
python3 scripts/contract/validate_spec_constants.py   # 상수 대장 ↔ 코드 대조
```

새 수치를 **AGENTS.md·README.md 에 적지 말 것** — 명세서가 정본이고 나머지는 링크한다.

### 에러 수정 후 docs/TROUBLESHOOTING.md 에 기록

새로운 종류의 에러를 진단하고 **수정에 성공**했을 때 `docs/TROUBLESHOOTING.md` 에 항목 추가 (다음 세션·다른 작업자용).

- 양식: **현상 → 오류 메시지(코드 블록) → 원인 → 해결 방법 → 확인 방법** 5블록
- 같은 종류 에러(ABI 불일치, GPU/드라이버 호환, 의존성 핀 충돌, USD/씬 물리 등)는 인접 섹션에 배치
- 수정 실패한 경우도 README 에는 올리지 않음
- **"왜 이 값인가"** 류 설계 근거는 트러블슈팅이 아니라 `docs/spec/09_TACIT_KNOWLEDGE.md` 로 간다

### scratch/ — 임시물 관리

smoke test·일회성 검증·debug dump 가 레포 여기저기 흩어지지 않도록 **임시물은 전부 루트 `scratch/`** 에 둔다.

- `scratch/` 는 `.gitignore` 추적 제외(`scratch/README.md` 만 예외). 커밋 금지.
- 산출물 예: smoke test 스크립트, 디버그 로그, faulthandler 크래시 txt, 한 번 쓰고 버릴 plot/csv.
- 작업 종료 시 정리: 영구 가치 있으면 `scripts/<범주>/` 로 promote(+charter 표 등재), 아니면 삭제.
- 하위 구조 자유 (예: `scratch/2026-06-26-<주제>/`).

### anti-fragmentation — 코드 파편화 방지

ad-hoc 작업으로 코드가 산발하지 않도록:

- **영구 스크립트는 반드시 `scripts/<범주>/`** 아래. 루트·임의 위치 금지. 범주는 위 charter 표
  (+신규 범주 추가 가능, 추가 시 charter 표에 1행 등재).
- **새 파일 만들기 전 기존 모듈·엔트리포인트 확장 우선.** 단일 소스 목록 = `docs/SPEC.md` §4.
- 한 작업 = 한 곳. 헬퍼 산발 금지. 탐색 코드는 `scratch/`, 끝나면 promote-or-delete.

### 단위 및 그리퍼 codec 규약

- **그리퍼 codec = affine only**, offset 전면 제거(`use_default_offset=False`, 절대 joint target).
- **세 계약을 섞지 말 것**: `feature_codec`(policy ↔ sim) · `leader_calibration`(실 leader ↔ sim) ·
  `follower_calibration`(실 follower ↔ sim).
- 수식·상수는 `docs/spec/04_IO_CONTRACT.md` 에 있다. **다른 파일에 복제 금지** — 단일 소스에서 import.

### 5-DOF IK 공통 원칙 (sim)

SO-101 은 팔 5축(+그리퍼)이라 임의 6-DOF pose 를 만족 못 한다. **position 우선·orientation best-effort**:

- 새 IK 경로 추가 시 orientation 을 hard constraint 로 넣지 말 것.
- MoveIt·cuMotion·Lula·RMPFlow·follow-target IK 테스트 스크립트 제거됨.
  (cuRobo 는 `scripts/cuRobo/` pick-place SM 플래너로 복귀 — reachable-manifold 후보 IK.)

### sim 진입 스크립트 AppLauncher 인자 필터

GUI 부팅 진입 스크립트는 `view_eye`/`view_lookat` 같은 **커스텀 인자**를 통째(`AppLauncher(vars(args))`)로
넘기면 Windows 에서 `_prepare_ui` access violation 이 난다. AppLauncher 가 실제 쓰는 키만
화이트리스트(`_LAUNCHER_KEYS`)로 필터해 전달하고, C-레벨 크래시 추적용 `faulthandler.enable(file=...)` 을
부팅 전에 켠다. 적용: `run_cube_desk_ros_bridge.py`, `scripts/cuRobo/pickplace_sm.py`.

⚠ Linux 에선 access violation 대신 **livestream viewport docking 이 조용히 실패**하는 형태로도
나타난다(3-cam 레이아웃 미적용). 상세 = `docs/spec/09_TACIT_KNOWLEDGE.md` §6.1.
