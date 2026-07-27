---
name: curobo-datagen
description: Docker 로 cuRobo pick-place state machine 을 실행하고 VLA 학습용 데이터셋을 녹화(IsaacLab HDF5 / LeRobot v3)·변환·검증하는 절차. cuRobo SM 실행, pickplace_sm, curobo_batch_planner, 데이터 녹화/생성(datagen), --record_hdf5, --record_lerobot, HDF5→LeRobot 변환, sweep/fail 재현, planner 컨테이너 기동 등을 요청받으면 반드시 이 스킬을 사용한다. "pick-place 데이터 만들어줘", "SM 돌려줘", "녹화 스모크" 같은 요청도 해당된다.
---

# cuRobo SM 실행·데이터셋 녹화 runbook

2-proc 구조: **planner**(curobo-datagen 컨테이너, ZMQ REP :5599) ↔ **SM**(isaac-sim 컨테이너,
IsaacLab env 실행 + 궤적 replay). 둘 다 `network_mode: host`. 설계 상세는
`scripts/cuRobo/README.md` — 이 스킬은 실행 절차와 함정만 담는다.

## 0. 실행 전 점검 (필수)

GPU 1장 공유 서버다. 기존 세션과 경합하면 사용자 작업을 망친다:

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep -iE "isaac|curobo"   # 기존 SM/planner 세션?
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader     # 여유 메모리?
```

- 이미 `pickplace_sm`/`curobo_batch_planner` 프로세스가 떠 있으면(컨테이너 내부 `ps aux` 로 확인)
  **새로 띄우지 말고 사용자에게 확인**. planner REP 소켓은 1개라 SM 2개가 붙으면 요청이 섞인다.
- 녹화(카메라 on)는 env 당 GPU 메모리를 크게 먹는다(아래 §4 가이드).

## 1. planner 기동 (항상 먼저)

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name curobo-planner \
    curobo-datagen python /workspace/scripts/cuRobo/curobo_batch_planner.py
```

`[planner] ZMQ REP :5599` 로그가 뜬 뒤에 SM 을 실행한다(기동 ~1분, FK bank 로드).
`--max_batch_size` 를 SM `--num_envs` 와 맞추면 batch 재초기화를 회피한다(기본 64 로도 동작).

## 2. SM 실행 — 서브커맨드 3종

`pickplace_sm.py {random|fail|sweep}` (isaac-sim 컨테이너, 기동 2~4분):

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name sm-run isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py random \
    --task SimToReal-SO101-PickCube-DR-v0 --num_envs 1 --livestream 2   # 인터랙티브 관전
```

| 서브커맨드 | 용도 | 핵심 인자 |
|---|---|---|
| `random` | 랜덤 DR 배치. 인터랙티브(livestream 키 N/R/B) 또는 `--auto_trials N` 자동 | `--record_hdf5`/`--record_lerobot` 는 auto 전용 |
| `fail` | sweep 결과의 fail 셀 좌표만 재현 | `--results <sweep.json>` (+`--auto`=headless 집계) |
| `sweep` | DR 스폰영역 정량 평가 → JSON | `--num_envs 12 --headless --out ...` |

원격 관전 = WebRTC :49100 (`.env` 의 `LIVESTREAM=1`+`PUBLIC_IP` 필요, 없으면 검은 화면).

## 3. 데이터셋 녹화 (`random --auto_trials N` 전용)

에피소드 규격(양 백엔드 공통, termination 자동 종료):
`[정지 2s pre-roll] → pick-place → init 복귀 → [정지 1s] → auto-reset(export)`.
플래닝/cold-start 대기·settle 프레임은 기록에 포함되지 않는다.

```bash
# A) IsaacLab HDF5 — multi-env, 실패도 저장(success attr 구분), 사후 변환 필요
... pickplace_sm.py random --task SimToReal-SO101-PickCube-DR-v0 \
    --num_envs 4 --auto_trials 25 --headless --enable_cameras \
    --record_hdf5 /workspace/datasets/pick_cube_sm.hdf5

# B) LeRobot v3 직기록 — single-env 전용, 성공만 저장, 변환 불필요
... pickplace_sm.py random --task SimToReal-SO101-PickCube-DR-v0 \
    --num_envs 1 --auto_trials 25 --headless --enable_cameras \
    --record_lerobot /workspace/datasets/pick_cube_sm_v3
```

| | A) `--record_hdf5` | B) `--record_lerobot` |
|---|---|---|
| multi-env | ✅ env 당 1 demo (`data/demo_N`) | ❌ `--num_envs 1` 강제 |
| 저장 범위 | 실패 포함(`success` attr) | 성공 에피소드만 |
| 메모리 | 이미지 GPU 누적 ~1.2 GB/env/15 s | step 마다 CPU 스트리밍 |
| 보존 | 전체 씬 state(replay·재라벨 가능) | frame(action/state/3-cam)만 |

함정:
- `--enable_cameras` 필수(없으면 즉시 에러). `--auto_trials 0`(인터랙티브)에선 녹화 불가.
- `--record_lerobot` 은 기존 출력 디렉터리를 **덮어쓴다**.
- 두 플래그 동시 사용 불가.
- 트라이얼 단위 seed 재현 없음(run 전체 `--seed` 1회).
- 산출물 경로는 컨테이너 기준 `/workspace/datasets` = 호스트 `./datasets` (compose 볼륨).
- 진행 요약은 `--summary_dir`(기본 scratch) 의 `summary.json`.

## 4. num_envs / 용량 가이드

3-cam 640×480 uint8 @30 Hz ≈ 2.8 MB/frame/env. HDF5 경로는 auto-reset 까지 GPU 에 누적:
15 s 에피소드 ≈ 1.2 GB/env → **`--num_envs 4~8` 권장**(48 GB GPU 기준, sim 자체 사용량 감안).
HDF5 파일은 gzip 후 대략 200 MB/에피소드.

## 5. 변환 (HDF5 → LeRobot v3) — Isaac·lerobot 패키지 불요

```bash
.venv/bin/python scripts/convert/isaaclab2lerobotv3.py \
    --hdf5_files datasets/pick_cube_sm.hdf5 --output_dir datasets/pick_cube_sm_v3 [--overwrite]
```

success demo 만 변환(`--include_failed` 로 해제). 직기록과 **같은 writer** 를 쓰므로 스키마 동일.
HF 업로드는 `scripts/data/upload_to_huggingface.py`.

## 6. 검증 (녹화·변환 후 반드시)

```bash
# 스키마 (h264·so_follower·canonical task 계약)
.venv/bin/python scripts/contract/validate_lerobot_schema.py datasets/pick_cube_sm_v3

# HDF5 구조·에피소드 규격 빠른 검사 (head 60f 정지 / mid 이동 / tail 30f 정지)
.venv/bin/python - <<'EOF'
import h5py, numpy as np
f = h5py.File("datasets/pick_cube_sm.hdf5", "r")
INIT = np.array([0.0, -1.74533, 1.5708, 0.87266, -1.5708, -0.17453], np.float32)
for name in sorted(f["data"], key=lambda n: int(n.split("_")[-1])):
    g = f["data"][name]
    d = np.abs(np.asarray(g["obs_x/joint_pos"]) - INIT).max(axis=1)
    print(f"{name}: T={len(d)} success={g.attrs['success']} "
          f"head60={d[:60].max():.3f} mid_moved={100*np.mean(d[60:-30]>0.15):.0f}% tail30={d[-30:].max():.3f}")
EOF
```

기대값: head60 ≤ 0.01 rad, mid_moved ≥ 90 %, tail30 ≤ 0.07 rad(경계 ±1 프레임), success=True 다수.

## 7. 정리

작업 끝나면 컨테이너를 내린다(공유 GPU): `docker stop curobo-planner sm-run` (docker stop 에 의한
planner exit 137 은 정상). 스모크 산출물은 `datasets/smoke_*` 네이밍으로 만들고 확인 후 삭제.

## 트러블슈팅 요약

| 증상 | 원인/조치 |
|---|---|
| SM 이 plan 응답을 못 받음 | planner 미기동 또는 다른 SM 이 REP 점유 — §0 점검 |
| `--record_* requires --enable_cameras` | AppLauncher 렌더 플래그 누락 |
| livestream 검은 화면 | `.env` `LIVESTREAM=1`+`PUBLIC_IP` 미설정 (LAN IP 만 광고됨) |
| 에피소드가 안 끝남(30 s 후 종료) | init 복귀 실패 → time_out 안전망. success=False 로 저장됨 — planner diag 확인 |
| 변환 결과 스키마 FAIL | lerobot 패키지로 만든 데이터셋인지 확인 — 이 파이프라인은 자체 writer(h264/so_follower) 계약 |
