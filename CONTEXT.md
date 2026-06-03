<!-- ╔═══════════════════════════════════════════════════════════════════════╗
     ║  NORTH STAR — 매 세션/compaction 직후 먼저 읽는다. 변경 금지(상수).      ║
     ╚═══════════════════════════════════════════════════════════════════════╝ -->

## 🧭 North Star (불변 — 매 사이클·compaction 후 재확인)

- **마스터플랜**: [`docs/SIM2REAL_MASTERPLAN.md`](docs/SIM2REAL_MASTERPLAN.md) · **현황**: [`TASKS.md`](TASKS.md)
- **불변 계약**(모든 sim 데이터·정책 I/O가 일치해야 함): `v3.0` · robot_type `so_follower` · action/state 각 **6-dim joint position** (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) · `observation.images.{top,wrist,front}` 480×640×3 h264 **fps 30** · task `"pick up the pen and place it in the holder"`.
- **자율 계약**: Codex `/goal` 시작 후 **A~E 무인 자율**(묻지 않음). 멈추는 경우는 둘뿐 — F~G 실기기 경계 / 복구불가 블로커(동일 task 3회 재시도 후 우회·기록). 게이트 미통과 task는 done 금지.
- **복구 프로토콜**: 세션/compaction 직후 ① 마스터플랜 §0·§1·§7 → ② TASKS.md(현재 phase·in_progress·blocked) → ③ 아래 최근 인계 1~2개 순서로 재로드. 추측 금지 — 상태 파일에 없으면 새 task로.
- **머신**: GPU 중량(Isaac·RL·롤아웃·GR00T) = 서버 konan147(48GB), 산출물 `/DISK1/so101-sim2real`. 경량·실기기·오케스트레이터 = Windows. sync 허브 = `origin`(github PubCyBerry/SO101-Sim2Real).

---

## 작업 인계 (2026-06-03 — T0.0/T0.1 착수 보완 계획 구현)

- **목표**: 보완 계획을 마스터플랜/TASKS에 반영하고, 실제 부트스트랩 일부(T0.0 preflight, T0.1 validator)를 수행.
- **상태**: T0.1·T0.4 완료. T0.0은 origin 표준화와 tool/GPU 확인은 완료했지만 `/DISK1/so101-sim2real` 권한 미준비로 blocked.
- **완료한 일**:
  - 로컬 remote `konan` 제거, 로컬/서버 `origin`을 `https://github.com/PubCyBerry/SO101-Sim2Real.git`로 표준화.
  - 서버 repo clean 확인. 서버 tool 확인: `claude`, `docker`, `nvidia-smi`, `gh`, `jq`, `yq` 있음. `uv`는 없음(T0.2 설치 항목).
  - Claude worker로 `scripts/validate_lerobot_schema.py` 작성 후 Codex가 직접 재검증.
  - 마스터플랜에 RELOAD 범위(§0·§1·§7), 복구불가 3회, worker JSON 인터페이스, `/DISK1/so101-sim2real/run/gpu.lock`, T0.5→T0.2 흡수 반영.
  - `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}` 추가. 로컬 dry run은 WSL 없이 Python subprocess가 `claude.exe --effort high`를 직접 호출하고, `dispatch.sh`는 SSH/Unix 래퍼로 유지.
- **검증 결과**:
  - `python scripts/validate_lerobot_schema.py datasets/pick_pen` 통과.
  - `python scripts/validate_lerobot_schema.py --self-test` 통과.
  - `python -m py_compile scripts/validate_lerobot_schema.py` 통과.
  - `python scripts/orchestrator/gate.py validate-lerobot-schema` 통과.
  - `python scripts/orchestrator/loop.py dry-run-t0.1` 통과(Claude DISPATCH `--effort high` → worker JSON → Codex VERIFY).
- **블로커**: 서버 `/DISK1`는 root 소유이고 `sudo -n true`가 password 요구. 다음 자율 사이클 전 서버에서 1회 실행 필요:
  `sudo mkdir -p /DISK1/so101-sim2real/{cache,outputs,datasets,checkpoints,logs,tmp,run} && sudo chown -R konan147:konan147 /DISK1/so101-sim2real`
- **변경한 파일**: `docs/SIM2REAL_MASTERPLAN.md`, `TASKS.md`, `CONTEXT.md`, `scripts/validate_lerobot_schema.py`, `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}`. 기존 dirty `pyproject.toml`의 `validation = ["ovphysx"]` 변경은 보존(T0.2 소유).
- **다음**: `/DISK1/so101-sim2real` 권한 해소 후 T0.0 verify 재실행 → T0.2(uv 설치 + leisaac 제거/Isaac direct dependency 전환).

---

## 작업 인계 (2026-06-03 — Sim2Real 자율 개발 마스터플랜 수립)

- **목표**: 장기 무인 자율 개발 계획(Codex→Claude 오케스트레이션) 수립 + Codex `/goal` 인계 파일 작성.
- **상태**: 완료(계획·인계 파일). 자율 개발 자체는 미시작 — Codex `/goal`이 부트스트랩(T0.0~)부터 구동.
- **확정 결정 5개**: ①Codex(플래너)→Claude Code CLI(워커) 디스패치 ②서버에 Isaac Sim 5.1 headless 설치 ③시뮬 A~E 무인 자율, F~G 사용자 게이트 ④CONTEXT.md+TASKS.md(git) 상태관리 ⑤**leisaac 전면 제거→순수 Isaac Sim 5.1.0 + Isaac Lab 2.3.2 재구현**.
- **신규 파일**: `docs/SIM2REAL_MASTERPLAN.md`(불변 계획), `TASKS.md`(Phase 0~G 체크리스트), 본 North Star 블록. (미커밋)
- **조사 근거(이번 세션)**: 서버 konan147 = RTX PRO 5000 Blackwell 48GB·RAM 125GB·`/DISK1` 3.4TB 여유, Isaac/uv 미설치, Docker 이미지 3개 빌드됨, 레포 클론 `~/Workspaces/SO101-Sim2Real`. leisaac 결합 8파일(base cfg·device·subasset·recorder) — 단 teleop device는 A~E 자율 트랙엔 불필요(연기). 현 데이터셋 50ep=333MB(ep당 ~6.7MB) → 롤아웃 5k ep≈35GB.
- **다음**: 사용자가 신규 파일 검토 후 Codex에 `/goal docs/SIM2REAL_MASTERPLAN.md` 로 인계. 부트스트랩 첫 task = T0.0(git sync 단일화)·T0.1(validator).

---

## 작업 인계 (2026-06-03 — Claude Code 실행 probe)

- **목표**: Codex가 계획을 세우고 Claude Code CLI에 간단한 구현을 지시할 수 있는지 확인.
- **상태**: 완료.
- **Claude Code 확인**:
  - 실행 파일: `C:\Users\taehunkim\.local\bin\claude.exe`
  - 버전: `2.1.161 (Claude Code)`
  - 프로젝트 설정: `.claude/settings.json`, `.claude/settings.local.json` 없음.
  - 사용자 전역 설정 요약: `model=sonnet[1m]`, `effort=null`, `permissionMode=null`.
  - probe 실행 플래그: `--model sonnet --effort low --permission-mode bypassPermissions --output-format json --no-session-persistence --tools Read,Write,Edit,MultiEdit --allowedTools Read,Write,Edit,MultiEdit`.
  - 실행 결과 JSON의 실제 modelUsage: `claude-sonnet-4-6`.
  - debug log: `outputs/claude_code_probe/claude_debug.log` 에서 `dispatching to firstParty model=claude-sonnet-4-6`, `tool=Write` 확인.
- **Claude에게 지시한 구현**: `outputs/claude_code_probe/joint_summary.py` 생성. JSON joint sample list를 읽어 `timestamp` 제외, numeric joint별 min/max/mean/count/joint_order 출력. `--self-test` 포함. 표준 라이브러리만 사용.
- **검증 결과**:
  - `python outputs/claude_code_probe/joint_summary.py --self-test` 성공.
  - 정상 JSON stdin 요약 성공.
  - invalid JSON, non-list input 모두 stderr 출력 + exit code `2`.
  - `outputs/` 는 `.gitignore` 대상이라 probe 산출물은 git tracked diff 없음.
- **변경한 파일**: `CONTEXT.md` 갱신. 산출물은 ignored 경로 `outputs/claude_code_probe/`.
- **남은 일**: 없음.

## 작업 인계 (2026-06-02 — SmolVLA 카메라 setup 수정 + GR00T modality 점검)

- **GR00T 점검 결과(확실)**: lerobot `GrootPolicy` 경로는 `modality.json` 불필요(소스 참조 0건), 카메라 ≥1개·이름 자유·개수 무제한(`configuration_groot.py:132` input_features VISUAL 동적 수집). 블로그의 modality.json/`so100_dualcam`(2-cam)/`wrist`·`front` 강제는 NVIDIA `Isaac-GR00T` 네이티브(`gr00t_finetune.py`) 전용. 사용자의 3-cam(wrist/front/top) 배포가 정상인 이유.
- **SmolVLA 실버그 발견·수정**: `lerobot/smolvla_base` config(HF에서 확인)가 input_features 로 `observation.images.camera1/2/3` 명시(chunk_size=50). `make_policy`(`factory.py:512`)는 `--policy.path` 시 pretrained input_features 를 데이터셋 키로 덮어쓰지 않음 → wrist/front/top 데이터셋과 mismatch. 기존 `RENAME_MAP` 기본 빈값이라 **SmolVLA train 이 실패하는 상태였음**(GR00T만 검증됐던 탓). 
  - 수정: `env/smolvla.env` 에 `RENAME_MAP` 추가(논문 표준 슬롯 top→camera1, wrist→camera2, front→camera3). `env/groot.env` 는 빈값+이유 주석. `.env`/`.env.example` §5 의 "자동 생성" 오기 제거(프로필로 이관).
  - 추론 물리 매핑도 학습과 동일하게 통일: camera1=top, camera2=wrist, camera3=front. PATH_A §6(train에 --rename_map 추가)·§7, PATH_B §12, TROUBLESHOOTING 1022 의 옛 순서(wrist→camera1)·"자동 생성" 표현 전부 정정.
  - 검증: `docker compose config` 로 smolvla 프로필 RENAME_MAP JSON 주입 + groot 빈값 확인.
- **미반영(미검증)**: article 의 SmolVLA base normalization-key 버그(from_pretrained 시 obs 미정규화로 팔 떨림)는 0.5.x 재현 미확인 → 문서에 안 넣음. 사용자에게 watch-out 으로만 전달.
- 변경: `env/{smolvla,groot}.env`·`.env`·`.env.example`·`docker/policy-entrypoint.sh`·`docs/PATH_A_NATIVE.md`·`docs/PATH_B_DOCKER.md`·`docs/TROUBLESHOOTING.md`. (미커밋)

---

## 작업 인계 (2026-06-02 — 모델 프로필 + 중복인자 정리 + 직접추론 문서)

- **모델 프로필 방식 도입**: 모델별 변수 10개를 `env/<name>.env`(groot.env / smolvla.env)로 분리. `.env` 의 `POLICY_PROFILE` 한 줄로 활성 모델 선택. compose 서비스 `env_file: [../.env, ../env/${POLICY_PROFILE:-groot}.env]` (나중 파일 override). 두 서비스 모두 적용.
  - 검증: `docker compose config` 로 groot 주입 + `${HF_USER}` 보간(taehunkim/...) + `POLICY_PROFILE=smolvla` 셸 오버라이드 전부 정상.
  - `env/` 는 사용자가 .gitignore 에서 제외 → 추적됨. (docker/profiles/ 에 먼저 만들었다가 env/ 로 이동함.)
  - OUTPUT_DIR 은 .env 에서 제거, entrypoint 가 `outputs/train/${JOB_NAME:-run}` 로 파생(프로필 JOB_NAME 따라감).
- **중복 인자 정리**: POLICY_CLIENT_FPS 제거 → 서버·클라가 POLICY_FPS 공유(lerobot-entrypoint 가 POLICY_FPS 읽음). POLICY_TYPE/TRAIN_POLICY_TYPE 은 의미가 달라(클라 항상 필요 vs train 의 path/type 스위치) 병합 안 함 — 설명만.
- **policy.type vs policy.path** (HF 문서 확인): SmolVLA = `--policy.path=lerobot/smolvla_base`(LeRobot 체크포인트 포맷). GR00T = `--policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.5-3B`(NVIDIA native 포맷이라 path 불가). 차이는 scratch-vs-pretrained 아니라 **체크포인트 포맷**. 내가 학습한 체크포인트는 둘 다 LeRobot 포맷 → 재학습 시 --policy.path.
- **직접 추론(서버 없이) 문서화**: `lerobot record --policy.path=<model>` (HF 권장). lerobot-entrypoint `record` 모드가 `shift`+`"$@"` 로 추가 CLI(예 --policy.path) forward 하도록 수정. PATH_B §10 을 "직접 vs async" 2방식 표로 재작성, PATH_A 에 직접추론 절 추가.
- 반영 파일: `.env`·`.env.example`·`env/*.env`·`docker/docker-compose.yaml`·`docker/policy-entrypoint.sh`·`docker/lerobot-entrypoint.sh`·`docs/PATH_A_NATIVE.md`·`docs/PATH_B_DOCKER.md`·`AGENTS.md`. bash -n + compose config 검증 완료.

---

## 작업 인계 (2026-06-02 — 출발모델 변수 통일)

- `BASE_MODEL` 제거, fine-tune 출발 모델을 `POLICY_BASE_MODEL_PATH` 단일 변수로 통일.
- `policy-entrypoint.sh` train 라우팅: `TRAIN_POLICY_TYPE` 비움→`--policy.path=$POLICY_BASE_MODEL_PATH`(LeRobot 체크포인트, SmolVLA 포함), 설정→`--policy.type`+`--policy.base_model_path`(GR00T 등 native 베이스). 0.5.x 의 path/type 동시금지 구조적 회피.
- 반영: `.env`·`.env.example` §1(블록 11→10줄), `docs/PATH_A_NATIVE.md`·`PATH_B_DOCKER.md`·`TROUBLESHOOTING.md`. bash -n + 키 파리티 OK. BASE_MODEL 잔재 0.
- 미적용(설계 제안만): 다모델 확장 시 `env/<model>.env` 프로필 + 다중 `--env-file` 방식 권장(사용자 결정 대기).
- 참고: README 운영시나리오 섹션은 사용자가 정리/제거함 — 재추가 금지.

---

## 작업 인계 (2026-06-02 — .env 재구성 + 모델 토글)

- 목표: 너무 많은 env 변수/혼란 정리, `.env` ↔ `.env.example` reconcile, GR00T↔SmolVLA 전환 단순화.
- 결정(사용자): 토글 블록 2개 방식 / `.env.example` 기본 활성 = GR00T / 두 파일 다 정리 / COMPILE_MODEL 기본 false.
- 적용:
  - `.env.example`·`.env` 동일 구조로 전면 재작성. 섹션: §0 비밀값 / §1 모델토글⭐ / §2 하드웨어 / §3 카메라 / §4 수집 / §5 학습 / §6 서버(+RTC) / §7 클라.
  - **§1 모델 토글**: 두 모델 간 값이 다른 11개 변수만(POLICY_TYPE/BASE_MODEL/TRAIN_POLICY_TYPE/POLICY_BASE_MODEL_PATH/POLICY_TOKENIZER_ASSETS_REPO/POLICY_EMBODIMENT_TAG/POLICY_CHUNK_SIZE/POLICY_N_ACTION_STEPS/ACTIONS_PER_CHUNK/POLICY_REPO_ID/JOB_NAME). [A]GR00T 활성 / [B]SmolVLA 주석. 전환 = 한 블록 토글.
  - `.env` 실제 값(토큰·COM5/COM8·카메라 index·HF_USER 등) 보존. **`POLICY_TYPE=smolvla` leftover 버그 → groot 로 정정** (나머지가 전부 GR00T였음). 누락 키 RENAME_MAP·RTC_* 3개 추가.
  - 키 집합 `.env` == `.env.example` 확인 완료 (diff 동일).
  - `README.md`: "GR00T 빠른 흐름" → "운영 시나리오(학습·배포·추론) + GR00T→SmolVLA 전환". prepare-model 셸 변수 취약/논리오류 수정(학습은 자동 다운로드, 서버는 인자 없는 prepare-model=POLICY_REPO_ID).
  - `docs/PATH_B_DOCKER.md` §4 모드표에 `policy-server-rtc`(compose 기본 CMD) 추가, §9·§11 을 `.env §1 토글` 참조로 정리.
  - `docs/PATH_A_NATIVE.md` §6 native train: `--policy.type`+`--policy.path` 동시 전달(0.5.x 위반) 제거, TRAIN_POLICY_TYPE 비움.
- 기본 시나리오 확정: 이 PC(Windows native uv) = policy-client / konan147(docker) = train + policy-server-rtc.
- 미해결: native(호스트 uv) lerobot 은 Python 3.11 핀이라 0.4.x — 서버 0.5.1 과 async gRPC proto 호환은 별도 검증 필요.

---

## 작업 인계 (2026-06-02 — 0.5.1 디커플링 통합 점검)

- 목표: policy-server 0.4.4→0.5.1 / pyproject.toml 디커플링 후 .env·스크립트·Dockerfile·문서 일관성 점검 및 수정.
- 검토로 찾아 수정한 불일치 (미커밋):
  1. `README.md:5` — "하나의 pyproject.toml 로 묶어" 문구를 policy-server 디커플링 반영으로 수정.
  2. HF 캐시 경로 `/root/.cache/huggingface` → 실제 `/workspace/.cache/huggingface` (HF_HOME, non-root UID): `policy-entrypoint.sh`(주석+사용자 로그 211행), `docs/PATH_B_DOCKER.md`(76·263), `AGENTS.md`(33).
  3. **기능 버그**: `COMPILE_MODEL=true` 기본 + entrypoint 가 정책 무관하게 `--policy.compile_model` 추가 → GrootConfig 에 해당 필드 없어 GR00T train 시 draccus 거부. `policy-entrypoint.sh` train 분기에 `TRAIN_POLICY_TYPE=groot` 면 compile skip+warn 가드 추가. `.env.example`·PATH_B §9 에 주석 보강.
  4. `docs/PATH_B_DOCKER.md:84` 빌드 표 — lerobot 서비스 설명에서 `train` 제거(policy-server 로 이동), policy-server 행에 디커플링 명시.
  5. `pyproject.toml` 헤더 주석 — 존재하지 않는 `Dockerfile.teleop` → `Dockerfile.lerobot`(실제 `uv sync --group teleop --group async`), policy/async 그룹 + 디커플링 설명 추가.
  6. `policy-entrypoint.sh` 주석 `huggingface-cli download` → `hf download` (코드 일치).
- 검증: `bash -n` 으로 두 entrypoint 문법 OK.
- 미해결(별도 판단 필요): `docs/PATH_A_NATIVE.md:334-335` native(0.4.4) train 이 `--policy.type` 과 `--policy.path` 를 동시 전달 — 0.5.1 계약과 다름. 단 native 는 호스트 lerobot 0.4.x 라 동작 가능성. 확인 후 정리 필요.
- 참조용 0.5.2 소스 트리: 작업 디렉터리 `lerobot/` (untracked). robot_client `__main__` 는 `register_third_party_plugins()` + `async_client()`.

---

## 작업 인계 (이전 — GR00T N1.5 학습/추론)

- 목표: `ssh konan147` 원격 서버에서 `policy-server:0.5.1` 컨테이너로 GR00T N1.5를 SO-101 `taehunkim/so101_pick_pen` 데이터셋에 fine-tune.
- 현재 상태: GR00T N1.5 10,000-step full 학습 완료 및 Hugging Face Hub push 완료. 사용자가 원격 inference 컨테이너를 삭제했고, 재실행을 위한 서버/클라이언트/학습 가이드를 Markdown 문서에 반영 중. 이전 코드 변경은 origin/main에 커밋/푸시 완료, 이번 문서 변경은 아직 미커밋.
- 완료한 일:
  - 원격 repo 확인: `/home/konan147/Workspaces/SO101-Sim2Real`, branch `main`, git status clean.
  - 실행 중 컨테이너 없음 확인.
  - `policy-server:0.5.1` 이미지 존재 확인.
  - HF cache 준비 완료:
    - `nvidia/GR00T-N1.5-3B`
    - `lerobot/eagle2hg-processor-groot-n1p5`
  - 1차 smoke 실패 원인 확인: 이미지 내 baked entrypoint가 `POLICY_PATH=lerobot/smolvla_base`를 `--policy.path`로 주입.
  - 2차 smoke 실패 원인 확인: LeRobot 0.5.1 GR00T action head가 Transformers meta tensor 초기화 구간에서 `torch.distributions.Beta` 기본 validation을 실행해 `Tensor.item() cannot be called on meta tensors` 발생.
  - 3차 smoke 진행: Beta patch로 meta tensor 오류는 통과. 다음 오류는 Transformers 5.3이 `GR00TN15.all_tied_weights_keys`를 기대하지만 클래스에 없어 발생.
  - 로컬 파일 수정:
    - `.env`, `.env.example`: 0.5.1 기준 `TRAIN_POLICY_TYPE`, `BASE_MODEL`, `POLICY_BASE_MODEL_PATH`, `DATASET_VIDEO_BACKEND` 계약으로 정리.
    - `docker/policy-entrypoint.sh`: `--policy.path`와 `--policy.type` 동시 전달 방지, GR00T env 매핑 추가.
    - `docker/Dockerfile.policy`: GR00T/Transformers 5.3 호환 패치 추가.
    - 관련 문서/주석의 `Dockerfile.smolvla`, `POLICY_CLIENT_TYPE`, policy-server 0.4.4 표기 정리.
  - 원격에 수정 파일 반영 완료. 원격 `.env`는 토큰/포트 유지, GR00T/0.5.1 학습 키만 변경.
  - `policy-server:0.5.1` 재빌드 완료.
  - 100-step smoke 성공:
    - Job: `smoke_groot_n15_pick_pen_100_20260601_235702`
    - Output: `/home/konan147/Workspaces/SO101-Sim2Real/outputs/train/smoke_groot_n15_pick_pen_100_20260601_235702`
    - Checkpoint: `checkpoints/000100/pretrained_model/config.json`
    - `jq -r .type ...` 결과: `groot`
  - 10,000-step full 학습 시작:
    - Run ID: `groot_n15_full_20260602_000255`
    - PID: `1384429`
    - Container: `so101-groot-n15-full-20260602-000255`
    - Log: `/home/konan147/Workspaces/SO101-Sim2Real/logs/train/groot_n15_full_20260602_000255.log`
    - Output: `/home/konan147/Workspaces/SO101-Sim2Real/outputs/train/so101_groot_n15_pick_pen`
    - W&B: `https://wandb.ai/pubcyberry/lerobot/runs/raxsfmc2`
    - 완료 확인: 10,000/10,000 step, `End of training`, fatal error pattern 없음.
    - 최종 checkpoint: `outputs/train/so101_groot_n15_pick_pen/checkpoints/010000/pretrained_model`
    - 최종 loss 로그: `loss:0.024`, `grdn:0.654`, `lr:1.0e-05`
    - Hub push: `https://huggingface.co/taehunkim/so101_groot_n15_pick_pen`
    - Hub sha: `94940f296903133ef1b02e5145232aa83be6c6df`
  - 원격 inference server 시작:
    - 처음에는 `policy-server-rtc`로 띄웠으나, `GrootPolicy`는 LeRobot 0.5.1에서 `init_rtc_processor` 미지원이라 RTC 없이 fallback함.
    - 불필요한 per-chunk RTC 분기/로그를 피하려고 표준 `policy-server`로 교체.
    - Container: `so101-groot-n15-policy-server`
    - PID: `2509209`
    - Bind: `0.0.0.0:8080`
    - Log: `/home/konan147/Workspaces/SO101-Sim2Real/logs/server/so101-groot-n15-policy-server_20260602_081426.log`
    - Model cache prepared: `taehunkim/so101_groot_n15_pick_pen`
    - `Ready` + `SendPolicyInstructions` RPC 성공.
    - GR00T model load 확인: `Time taken to put policy on cuda: 5.6529 seconds`
    - GPU memory after load: 약 `6077 MiB / 48935 MiB`
  - Git 정리:
    - 로컬 `konan` remote 제거.
    - 로컬 `origin` URL을 GitHub moved target인 `https://github.com/PubCyBerry/SO101-Sim2Real.git`로 갱신.
    - Commit: `0397e0d feat: GR00T N1.5 policy-server 학습/추론 지원`
    - 로컬/원격 서버 모두 `main`이 `0397e0d`로 동기화.
    - 원격 서버의 duplicate tracked changes는 stash 후 fast-forward pull, duplicate stash drop 완료.
  - 문서 업데이트 진행:
    - `README.md`: GR00T N1.5 빠른 흐름 추가. `.env` 핵심값, 100-step smoke, full 학습, 추론 서버, policy client 접속 요약.
    - `docs/PATH_B_DOCKER.md`: GR00T 학습 설정, 100-step smoke/full run, 서버 기동, 클라이언트 연결, `policy-server-rtc` 미사용 사유, SmolVLA/GR00T 카메라 key 차이를 반영.
    - `docs/PATH_A_NATIVE.md`: native policy client 가이드에 GR00T 원격 서버 접속 예시와 0.5.1 `policy` dependency group 표기 반영.
    - `docs/TROUBLESHOOTING.md`: GR00T에서 `policy-server-rtc`가 표준 추론으로 fallback 되는 사례 추가.
- 남은 일:
  - 문서 diff 최종 확인 후 사용자에게 서버/클라이언트/학습 실행 요약 전달.
  - 이번 Markdown 변경 커밋 여부 결정.
  - Windows/실기기 쪽 `policy-client` 또는 `record --policy.path`로 실제 기기 추론 smoke 테스트.
- 결정한 사항:
  - GR00T 학습 CLI는 `--policy.type=groot`, `--policy.base_model_path=nvidia/GR00T-N1.5-3B`.
  - GR00T action horizon 제한에 맞춰 `--policy.chunk_size=16`, `--policy.n_action_steps=16`.
  - SmolVLA용 `.env` `RENAME_MAP`은 `--rename_map="{}"`로 덮어쓴다.
  - smoke는 Hub push 비활성화, full run은 `taehunkim/so101_groot_n15_pick_pen`로 push.
- 검증 결과:
  - 1차 smoke 실패: 원격 `.env`의 `POLICY_PATH=lerobot/smolvla_base`가 entrypoint에서 `--policy.path`로 전달되어 `--policy.type=groot`와 충돌.
  - 2차 smoke 실패: `RuntimeError: Tensor.item() cannot be called on meta tensors`.
  - 3차 smoke 실패: `AttributeError: 'GR00TN15' object has no attribute 'all_tied_weights_keys'`.
  - 4차 smoke 실패: `ValueError: Unsupported video backend: decord`.
  - 5차 smoke 실패: `AttributeError: 'list' object has no attribute 'shape'` (`pixel_values`가 list).
  - 6차 smoke 실패: `NotImplementedError: aten::_sample_dirichlet ... Meta tensors`.
  - 최종 smoke 성공: 100 step 완료 및 checkpoint 저장.
  - full 학습 성공: 10,000 step 완료, checkpoint 001000~010000 생성, Hub push 완료.
- 다음 명령:
  - 문서 변경 확인:
    `git diff --stat -- README.md docs/PATH_A_NATIVE.md docs/PATH_B_DOCKER.md docs/TROUBLESHOOTING.md`
  - 최종 결과 확인:
    `ssh konan147 'cd /home/konan147/Workspaces/SO101-Sim2Real && jq -r .type outputs/train/so101_groot_n15_pick_pen/checkpoints/010000/pretrained_model/config.json'`
  - 서버 상태 확인:
    `ssh konan147 'cd /home/konan147/Workspaces/SO101-Sim2Real && docker compose --env-file .env -f docker/docker-compose.yaml up -d policy-server && docker compose --env-file .env -f docker/docker-compose.yaml logs -f policy-server'`
