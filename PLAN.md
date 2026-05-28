# SO101-LeRobot-VLA + SO101-Sim2Real 통합 계획

원본: `C:\Users\taehunkim\Workspace\SO101-LeRobot-VLA` (이하 **VLA**), `C:\Users\taehunkim\Workspace\SO101-Sim2Real` (이하 **Sim2Real**)
대상: **VLA** 안에 흡수 (Sim2Real 의 시뮬 레이어를 VLA 위에 얹는다)

---

## 1. 사실 정리 (두 레포 비교)

| 항목 | VLA | Sim2Real |
|---|---|---|
| 역할 | 실기기 (Docker 기반 LeRobot 파이프라인) | 시뮬 (Isaac Lab Sim-to-Real, 펜 Pick&Place) |
| Docker 스택 (`Dockerfile.lerobot`, `Dockerfile.smolvla`, `docker-compose.yaml`, `lerobot-entrypoint.sh`, `server-entrypoint.sh`, `policy-client-shim.py`, `99-feetech.rules`) | ✅ | ✅ **byte-for-byte 동일** |
| Dockerfile.leisaac | ✅ (보존, compose 미연결) | — |
| `pyproject.toml` 패키지 이름 | `robotics-manipulation` | `sim_to_real` |
| `pyproject.toml` 추가 차이 | — | `usd-core>=26.5` 공용 + `[build-system]` + `[tool.setuptools.packages.find] where=["src"]` |
| `src/sim_to_real/` (Python 패키지) | — | 264K — `assets/scenes/pen_desk.py`, `datagen/state_machine/{base,pick_pen}.py`, `tasks/pick_pen/...`, `utils/` |
| `assets/scenes/` USD | — | **159M** — `pen_desk/`, `kitchen_with_orange/`, `robots/` (SO-101 follower USD + URDF) |
| `scripts/` 시뮬 진입점 | — | `author_pick_pen_scene.py`, `oracle_policy_traj.py`, `record_pick_pen.py`, `convert_pick_pen_to_lerobot.py`, `usd_viewer.py` |
| `scripts/environments/list_envs.py` | ✅ | ✅ (Sim2Real 가 약간 최신 — copyright 줄만 차이) |
| `notebooks/training_*.ipynb` | ✅ | — |
| `docs/TROUBLESHOOTING.md` | 766줄, WSL/Docker/CUDA 중심 | 780줄, USD/씬/펜 물리 중심. 헤더 구조 다름 |
| `docs/OpenUSD_Guide.md` | — | ✅ |
| `docs/windows-*-guide.md` 3종 | ✅ | — |
| `docs/gr00t-n16-pickorange-pipeline.md` | ✅ | — |
| `leisaac/` (vendored 포크) | — | **이미 사용자 삭제** |
| Root meta | `AGENTS.md`, `CLAUDE.md`, `README.md`, `.env(.example)`, `.dockerignore`, `.gitignore`, `.gitattributes`, `uv.lock` | + `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `.flake8`, `.pre-commit-config.yaml`, `CONTEXT.md` |

**핵심:** Docker 스택 충돌 0. 시뮬 레이어는 VLA 위에 깨끗하게 얹힘.

---

## 2. 결정 사항

| 결정 | 값 |
|---|---|
| `leisaac/` vendored 디렉터리 | 가져오지 않음 (사용자 선삭제). pyproject `[tool.uv.sources]` 의 `git tag v0.4.0` 만 사용 |
| Sim 실행 경로 | **호스트 uv + Docker (leisaac 서비스) 둘 다 지원** |
| 통합 Python 패키지 이름 | `so101_vla` (기존 두 이름 모두 폐기) |
| 통합 방식 | 단순 파일 복사 (Sim2Real git 히스토리 미보존) |
| dotfiles/메타 (`LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `.flake8`, `.pre-commit-config.yaml`, `CONTEXT.md`) | 가져오지 않음 |
| Docker compose leisaac 진입점 | 신규 `docker/leisaac-entrypoint.sh` (책임 분리) |

---

## 3. 작업 순서 (체크리스트)

### 3.1. 파일 이식 (Sim2Real → VLA)

- [ ] `src/sim_to_real/` → `src/so101_vla/sim/` (디렉터리 rename)
  - 내부 import `sim_to_real.*` → `so101_vla.sim.*` 일괄 치환
  - `tasks/__init__.py` 의 `import_packages` 호출도 새 경로 반영
  - `task gym 등록 id` (`SimToReal-SO101-PickPen-v0`) 는 사용자 가시 식별자라 유지
- [ ] `assets/scenes/pen_desk/` 통째 복사 → `assets/scenes/pen_desk/`
- [ ] `assets/scenes/kitchen_with_orange/` 복사 → `assets/scenes/kitchen_with_orange/` (참조 패턴 원본)
- [ ] `assets/robots/` 복사 → `assets/robots/` (SO-101 follower USD + URDF)
- [ ] sim `scripts/*.py` 5개 복사:
  - `author_pick_pen_scene.py`
  - `oracle_policy_traj.py`
  - `record_pick_pen.py`
  - `convert_pick_pen_to_lerobot.py`
  - `usd_viewer.py`
- [ ] `scripts/environments/list_envs.py` 는 Sim2Real 버전으로 덮어쓰기 (copyright 갱신본)
- [ ] `docs/OpenUSD_Guide.md` 복사

### 3.2. `pyproject.toml` 머지

- [ ] `[project].name` = `so101_vla`
- [ ] `[build-system]` 블록 추가:
  ```toml
  [build-system]
  requires = ["setuptools<82"]
  build-backend = "setuptools.build_meta"
  ```
- [ ] `[tool.setuptools.packages.find]` 추가:
  ```toml
  [tool.setuptools.packages.find]
  where = ["src"]
  ```
- [ ] `[project].dependencies` 에 `usd-core>=26.5` 추가
- [ ] 나머지 `dependency-groups` / `tool.uv` 블록은 VLA 본 그대로 유지 (override 핀 변화 없음)

### 3.3. `uv.lock` 재생성

- [ ] `uv lock` 실행 (override 핀 유지 — `numpy==1.26.0`, `pyarrow<19`, `datasets<4.7`, `packaging<26`)
- [ ] resolve 실패 시 진단 → 핀 조정은 사용자 승인 후

### 3.4. Docker — leisaac 서비스 재활성

- [ ] `docker/Dockerfile.leisaac` 점검 (VLA 에 보존되어 있음)
  - 필요 시 stage 1–4 (`base → uv → python → torch`) 가 lerobot/smolvla 와 일치하는지 확인 → BuildKit 캐시 공유
- [ ] 신규 `docker/leisaac-entrypoint.sh` 작성
  - 모드: `teleop` (keyboard/gamepad/so101leader), `oracle` (`oracle_policy_traj.py`), `record` (`record_pick_pen.py`), `convert-to-lerobot`, `author-scene`, `list-envs`, `bash`, `python`
  - `.env` 의 `TELEOP_DEVICE`, `TELEOP_PORT`, `RECORD_PEN`, `NUM_DEMOS`, `LEROBOT_DATASET_REPO_ID`, `STEP_HZ` 등을 CLI 인자로 매핑
- [ ] `docker/docker-compose.yaml` 에 `leisaac` 서비스 블록 추가
  - GPU 1장 예약 (RT 코어 필수 — H100/A100 미지원 명시)
  - `assets/`, `outputs/` 볼륨 마운트
  - `privileged: true` + `/dev/dri` GPU 디바이스
  - `network_mode: host` (rerun 뷰어 + ZMQ 원격 leader)
  - 명명 볼륨 `lerobot_hf_cache` 공유 (`/root/.cache/huggingface`)

### 3.5. 문서 머지

- [ ] `README.md`:
  - 기존 §아키텍처 / §환경 요구사항 / §실기기 워크플로 유지
  - §시뮬 워크플로 신규 추가 (Sim2Real README 의 펜 씬 / 텔레op / 오라클 / 녹화 / LeRobot v3 변환 절을 PowerShell + Docker 양쪽 명령 병기로 다듬어 옮김)
  - 디렉터리 구조 절도 합치기
- [ ] `AGENTS.md`:
  - §환경 사양 표는 VLA 본 유지 (Windows + Linux 학습 서버 양쪽 다 있음)
  - §시뮬레이션 환경 제약 절은 양쪽 동일하므로 1회만 유지
  - §Docker 컨테이너 구조 절에 leisaac 서비스 추가 설명
  - §사용자 환경 컨벤션·운영 규칙은 VLA 본 유지
- [ ] `docs/TROUBLESHOOTING.md`:
  - VLA 본을 base 로, Sim2Real 본의 §"Isaac Lab / USD / 펜 물리" 카테고리 12개 항목을 카테고리 단위로 append
  - 목차 재생성 (수동)
  - 중복 항목 (e.g. ABI 핀 충돌, RT 코어 제약) 은 VLA 본 우선

### 3.6. 검증

- [ ] `uv sync --group teleop` 성공
- [ ] `uv sync --group isaac` 성공 (호스트가 Windows + RT 코어 GPU 인 경우만)
- [ ] `docker compose -f docker/docker-compose.yaml build lerobot` 성공
- [ ] `docker compose -f docker/docker-compose.yaml build lerobot-policy-server` 성공
- [ ] `docker compose -f docker/docker-compose.yaml build leisaac` 성공 (Linux 서버 또는 RT 코어 워크스테이션)
- [ ] `uv run scripts/oracle_policy_traj.py --num_episodes 1 --pen PenWhite` 1회 (호스트 uv 경로 sanity)
- [ ] `docker compose --env-file .env run --rm leisaac oracle --pen PenWhite --num_episodes 1` (docker 경로 sanity, GPU 가용 시)
- [ ] `docker compose --env-file .env run --rm lerobot info` 회귀 확인 (teleop 경로 무손상)

---

## 4. 리스크 / 미정 사항

| 리스크 | 영향 | 완화 |
|---|---|---|
| `assets/` 159M 가 git 에 그대로 들어감 | repo clone 느려짐, GitHub push limit 우려 | git LFS 도입 또는 `.gitignore` + 별도 배포. 사용자 결정 필요 |
| Sim2Real `import sim_to_real` 의 side-effect (monkey patch + gym 등록) 가 `so101_vla.sim` 으로 옮길 때 누락될 가능성 | gym ID 미등록 → 스크립트 ImportError | 이식 직후 `python -c "import so101_vla.sim; import gymnasium; print(gymnasium.spec('SimToReal-SO101-PickPen-v0'))"` 로 즉시 검증 |
| Docker leisaac 이미지가 Sim2Real 에서 한 번도 빌드 검증된 적 없을 가능성 (VLA 에 dormant 상태로 보존) | docker build 실패 | 빌드 실패 시 base/uv stage 부터 lerobot 이미지와 동기화 |
| Isaac Sim 5.1 + leisaac v0.4.0 가 새 시점에서 nvidia pypi 캐시 일관성 깨질 가능성 | `uv sync --group isaac` resolve 실패 | override 핀은 그대로 두고 nvidia index TLS / 캐시 우선 점검 |
| Sim2Real 의 `usd-core>=26.5` 가 공용 deps 에 들어가면 teleop-only 도커 이미지에도 끌려옴 | teleop 이미지 빌드 시간 +α | 순수 Python 패키지라 영향 작음. 필요 시 `isaac` 그룹으로 분리 |

---

## 5. 다음 단계

사용자 확인 후 위 §3 의 순서대로 진행. 작업 중 결정이 더 필요한 분기 (e.g. `assets/` LFS 도입 여부, TROUBLESHOOTING 목차 순서) 는 그 시점에 추가로 묻는다.
