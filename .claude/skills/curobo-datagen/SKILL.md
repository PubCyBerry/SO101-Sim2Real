---
name: curobo-datagen
description: Docker 로 cuRobo pick-place state machine 을 실행하고 VLA 학습용 데이터셋을 녹화(IsaacLab HDF5 / LeRobot v3)·변환·검증하는 절차. cuRobo SM 실행, pickplace_sm, curobo_batch_planner, 데이터 녹화/생성(datagen), --record_hdf5, --record_lerobot, HDF5→LeRobot 변환, sweep/fail 재현, planner 컨테이너 기동 등을 요청받으면 반드시 이 스킬을 사용한다. "pick-place 데이터 만들어줘", "SM 돌려줘", "녹화 스모크" 같은 요청도 해당된다.
---

# cuRobo SM 실행·데이터셋 녹화 runbook

2-proc 구조: planner(curobo-datagen 컨테이너, ZMQ REP :5599) ↔ SM(isaac-sim 컨테이너,
IsaacLab env 실행 + 궤적 replay). 둘 다 `network_mode: host`. 설계 상세는
`scripts/cuRobo/README.md` 에 있고, 이 스킬은 실행 절차와 함정만 담는다.

## 0. 실행 전 점검 (필수)

GPU 1장 공유 서버다. 기존 세션과 경합하면 사용자 작업을 망친다:

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep -iE "isaac|curobo|planner|datagen"  # 기존 세션?
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader     # 여유 메모리?
```

- 이미 `pickplace_sm`/`curobo_batch_planner` 프로세스가 떠 있으면(컨테이너 내부 `ps aux` 로 확인)
  **새로 띄우지 말고 사용자에게 확인**. planner REP 소켓은 1개라 SM 2개가 붙으면 요청이 섞인다.
- 녹화(카메라 on)는 env 당 GPU 메모리를 크게 먹는다(아래 §4 가이드).

```bash
df -h datasets/    # HDF5 는 에피소드당 ~375 MB. 64 ep 배치 하나가 24 GB 다
```

- **`datasets/` 는 /DISK1 심볼릭이고 `scratch/` 는 루트 파티션이다.** 대용량 산출물을
  `scratch/` 에 쓰면 / 가 찬다(2026-07-28 에 124 GB 로 100% 도달).

## 1. planner 기동 (항상 먼저)

```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name curobo-planner \
    curobo-datagen python /workspace/scripts/cuRobo/curobo_batch_planner.py
```

`[planner] ZMQ REP :5599` 로그가 뜬 뒤에 SM 을 실행한다(기동 ~1분, FK bank 로드).
`--max_batch_size` 를 SM `--num_envs` 와 맞추면 batch 재초기화를 회피한다(기본 64 로도 동작).

## 2. SM 실행 (서브커맨드 3종)

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

공용 `--cube_sizes` = 큐브 **크기 DR**(25/30/35/40 mm 이산) 사다리를 좁힌다. env 당 크기는
런 내내 고정(prestartup USD 편집)이라, 크기별 성공률을 깨끗이 재려면 `--cube_sizes 0.025`
처럼 하나로 고정해 크기마다 따로 돌린다. 생략하면 사다리 전체에서 env 마다 뽑는다.

원격 관전 = WebRTC :49100 (`.env` 의 `LIVESTREAM=1`+`PUBLIC_IP` 필요, 없으면 검은 화면).

## 3. 데이터셋 녹화 (`random --auto_trials N` 전용)

에피소드 규격(양 백엔드 공통, termination 자동 종료):
`[정지 2s pre-roll] → pick-place → init 복귀 → [정지 1s] → auto-reset(export)`.
플래닝/cold-start 대기·settle 프레임은 기록에 포함되지 않는다.

```bash
# A) IsaacLab HDF5: multi-env, 실패도 저장(success attr 구분), 사후 변환 필요
... pickplace_sm.py random --task SimToReal-SO101-PickCube-DR-v0 \
    --num_envs 4 --auto_trials 25 --headless --enable_cameras \
    --record_hdf5 /workspace/datasets/pick_cube_sm.hdf5

# B) LeRobot v3 직기록: single-env 전용, 성공만 저장, 변환 불필요
... pickplace_sm.py random --task SimToReal-SO101-PickCube-DR-v0 \
    --num_envs 1 --auto_trials 25 --headless --enable_cameras \
    --record_lerobot /workspace/datasets/pick_cube_sm_v3
```

| | A) `--record_hdf5` | B) `--record_lerobot` |
|---|---|---|
| multi-env | 가능. env 당 1 demo (`data/demo_N`) | 불가. `--num_envs 1` 강제 |
| 저장 범위 | 실패 포함(`success` attr) | 성공 에피소드만 |
| 메모리 | 이미지 GPU 누적 ~1.2 GB/env/15 s | step 마다 CPU 스트리밍 |
| 보존 | 전체 씬 state(replay·재라벨 가능) | frame(action/state/3-cam)만 |

> **수백~수천 ep 는 위 명령을 직접 쓰지 말 것.** 단일 HDF5 로 몰면 1000 ep = 375 GB 다.
> `scripts/cuRobo/generate_dataset.sh` 가 배치(기본 64 ep)로 쪼개 생성(GPU)과 변환(CPU)을
> 겹치고, 변환 확인된 HDF5 를 지운다 — §3.1.

함정:
- `--enable_cameras` 필수(없으면 즉시 에러). `--auto_trials 0`(인터랙티브)에선 녹화 불가.
- `--record_lerobot` 은 기존 출력 디렉터리를 덮어쓴다.
- 두 플래그 동시 사용 불가.
- 트라이얼 단위 seed 재현 없음(run 전체 `--seed` 1회).
- 산출물 경로는 컨테이너 기준 `/workspace/datasets` = 호스트 `./datasets` (compose 볼륨).
- 진행 요약은 `--summary_dir`(기본 scratch) 의 `summary.json`.

### 3.1 대량 생성 (권장 경로)

```bash
./scripts/cuRobo/generate_dataset.sh 1000 16 64      # TOTAL_EP NUM_ENVS BATCH_EP [OUT_ROOT]
```

배치 N 생성 중 배치 N-1 을 백그라운드 변환한다. planner 는 전 배치 공용으로 1회만 뜬다.
변환 성공(`meta/info.json` 의 `total_episodes > 0`)을 확인한 배치만 HDF5 를 지우고,
실패하면 **보존 + 경고**한다.

| | 직렬 | 이 드라이버 |
|---|---|---|
| 1000 ep | 3.83 h | **2.98 h** |
| HDF5 피크 | 375 GB | **~48 GB** (in-flight 2배치) |

산출은 `OUT_ROOT/v3/batch_NNN/` 로 **배치마다 나뉜다**(writer 가 append 불가).
하나로 합치려면 §5 참조. 설계·실측 = `docs/spec/08_PIPELINES.md` §5.8.

## 4. num_envs / 용량 가이드

**`--num_envs 16` 이 최적**이다(2026-07-28 실측, 구성당 64 ep 생성 + v3 변환, 전 구성 64/64):

| num_envs | 1 | 2 | 4 | 8 | **16** |
|---|---|---|---|---|---|
| s/에피소드 | 31.6 | 27.3 | 24.2 | 16.8 | **13.8** |
| VRAM 피크 | 9.7 | 11.4 | 14.7 | 22.1 | **34.9 GB** |

48 GB 카드에서 16-env 는 OOM 이 아니다. 3-cam 640×480 uint8 @30 Hz ≈ 2.8 MB/frame/env 가
auto-reset 까지 GPU 에 누적된다(≈1 GiB/env/에피소드).

디스크는 `lzf` + frame-chunk 압축 기준 **~375 MB/에피소드**(64 ep = 24 GB). v3 변환 후엔
8 MB/에피소드로 46배 줄어드니, HDF5 는 버리는 중간물로 취급한다.

⚠ **공유 GPU 에서 잰 VRAM 절대값은 믿지 말 것.** 다른 워크로드가 26 GB 를 잡고 있으면
8-env 가 45 GB 로 보인다(실제 22.1). 상대 비교만 유효하다.

## 5. 변환 (HDF5 → LeRobot v3, Isaac·lerobot 패키지 불요)

```bash
.venv/bin/python scripts/convert/isaaclab2lerobotv3.py \
    --hdf5_files datasets/pick_cube_sm.hdf5 --output_dir datasets/pick_cube_sm_v3 [--overwrite]
```

success demo 만 변환(`--include_failed` 로 해제). 직기록과 같은 writer 를 쓰므로 스키마 동일.
HF 업로드는 `scripts/data/upload_to_huggingface.py`.

**배치를 하나로 합치려면** 사후 병합 도구가 없다(카메라마다 mp4 1개에 전 에피소드가 연결돼
있어 파일 복사로는 인덱스·stats 가 깨진다). 대신 변환기에 **쉼표 구분으로 전부 먹인다**:

```bash
--hdf5_files "$(ls -1 OUT_ROOT/hdf5/*.hdf5 | paste -sd,)" --output_dir OUT_ROOT/v3_merged --overwrite
```

단 HDF5 를 전부 남겨야 해서 디스크 이점과 상충한다. 절차 상세 = `08_PIPELINES.md` §5.8.

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

작업 끝나면 컨테이너를 내린다(공유 GPU): `docker stop curobo-planner sm-run datagen_planner` (docker stop 에 의한
planner exit 137 은 정상). 스모크 산출물은 `datasets/smoke_*` 네이밍으로 만들고 확인 후 삭제.

## 트러블슈팅 요약

| 증상 | 원인/조치 |
|---|---|
| SM 이 plan 응답을 못 받음 | planner 미기동 또는 다른 SM 이 REP 점유. §0 점검 |
| `--record_* requires --enable_cameras` | AppLauncher 렌더 플래그 누락 |
| livestream 검은 화면 | `.env` `LIVESTREAM=1`+`PUBLIC_IP` 미설정 (LAN IP 만 광고됨) |
| 에피소드가 안 끝남(30 s 후 종료) | init 복귀 실패 → time_out 안전망. success=False 로 저장됨. planner diag 확인 |
| 변환 결과 스키마 FAIL | lerobot 패키지로 만든 데이터셋인지 확인. 이 파이프라인은 자체 writer(h264/so_follower) 계약 |
