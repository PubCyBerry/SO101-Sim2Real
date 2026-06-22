# SO-101 Canonical Parity 검증 보고서

> 기준일: 2026-06-22
> 결론: software parity path는 통과, physical parity는 실측 gate 대기

## 1. Canonical contract

```text
schema: so101-canonical-v1
arm: absolute URDF radian
gripper: physical jaw aperture mm
action: absolute position target
order: pan, lift, elbow, wrist flex, wrist roll, gripper
rate: 30 Hz
camera: top/wrist/front, RGB uint8 480×640
```

현재 hash:

| artifact | SHA256/content hash |
|---|---|
| contract | `8391f48d2ad25f24c5e2b14a28d5e28fd5551dc85361468988a294b077cd2414` |
| calibration | `6112d10a0a7f2aa3c68788aaf286d30a59218bb53d24593c1b3efe12aaa1c21c` |
| motor profile | `d73d0b9e1117d042fccfb6e230ba83ad6121c4f298f9452de8d493c2b6b161aa` |
| runtime manifest | `9cb11de2a4c92aa1ad3bc9a0730742caa5e82771aabe84186fdfbea35648ffd9` |
| replay checkpoint | `5efd76d1d1eada26d27e122ae6078d0d89c19a563494c5f73788bc05b5be9024` |

## 2. Executor

공통 `CanonicalRuntime`과 `SingleFlightChunkExecutor`가 sim/real adapter를 동일하게 구동한다.

- inference 한 건만 in-flight
- chunk overlap/weighted aggregation 없음
- chunk 종료 step을 다음 chunk 시작점으로 고정
- underrun은 마지막 target hold, logical policy step 정지
- timeout은 adapter safe stop
- prefetch `max(8, ceil(p99×30)+2)`
- velocity/acceleration/jerk limiter 공통 적용
- raw output, limited target, native command, measured state를 JSONL에 기록

동일 초기 상태 mock sim/real test에서 limited canonical target은 모든 step byte-identical이다.
MotionLimiter convergence와 timeout/underrun/chunk boundary test를 포함해 core 19 tests가 통과했다.

## 3. Calibration

### Sim gripper

Isaac 6 geometry에서 jaw surface separation을 자동 측정했다.

| joint rad | aperture mm |
|---:|---:|
| -0.174533 | 2.264 |
| 0.065251 | 20.531 |
| 0.305236 | 37.957 |
| 0.545231 | 52.996 |
| 0.785239 | 67.379 |
| 1.025252 | 80.299 |
| 1.265276 | 91.766 |
| 1.505304 | 101.288 |
| 1.745330 | 109.356 |

곡선은 monotonic이며 PCHIP 수치 round-trip 오차는 `4.27e-14 mm`다.

### Real calibration

아직 측정하지 않았다.

- arm paired pose: 0/10~20
- gripper caliper: 0/7~9
- EEPROM readback: 미실행

따라서 현재 bundle은 의도적으로 다음 상태다.

```text
validated=false
motor_profile.readback_validated=false
real gripper curve=[]
```

`run_real_client.py`와 `launch.py real-motion`은 이 상태에서 motion을 거부한다.

## 4. Dataset

원본 dataset은 변경하지 않았다.

| dataset | frame/episode | video↔Parquet | 처리 |
|---|---:|---|---|
| `pick_cube_v2` | 48,873 / 100 | 3 camera 전부 일치 | provenance/calibration 대기 |
| `pick_cube` | 18,526 / 50 | 일치 | quarantine |
| `pick_pen` | 53,985 / 50 | 일치 | quarantine |

`pick_cube_v2` 변환은 source provenance가 `verified=false`이므로 fail-closed되었다.
`pick_cube`, `pick_pen`은 2026-06-17 이전 source calibration을 복구하지 못해 quarantine report를
생성했다.

Converter는 별도 tiny LeRobot v3 dataset smoke에서 다음을 통과했다.

- source 불변
- action/state canonical 변환
- Parquet 재작성
- global/episode vector stats 재생성
- video hardlink/copy
- 모든 video frame count 재검사
- canonical contract/calibration/provenance metadata 기록

## 5. Dynamics

두 조건의 canonical plan을 생성했다.

| 조건 | step | 시간 |
|---|---:|---:|
| no-load | 5,190 | 173 s |
| cube payload | 5,190 | 173 s |

각 plan은 hold, 각 관절 ±5°/±10° step, ramp, ±15° triangle, 0.2~1.5 Hz multisine,
compound 6-axis, gripper sweep을 포함한다. Isaac adapter 10-step runner smoke는 통과했다.

Fitter는 다음 지표를 계산한다.

- delay
- 2차 ARX DC gain/natural frequency/damping
- observed velocity/acceleration limit
- deadband/backlash
- steady-state gravity droop
- PhysX stiffness/damping/delay/limit fitting hint
- sim-real trajectory/steady-state/gripper/lag gate

실기기 trace가 없으므로 최종 dynamics fit report는 현재 `blocked`가 정상 상태다.

## 6. 현재 gate

| 완료 조건 | 상태 |
|---|---|
| Windows/server compatibility | 통과 |
| lock/commit/manifest 일치 | 통과 |
| quaternion detector warning 0 | 통과 |
| 새 경로 removed write API 0 | 통과 |
| 새 경로 legacy Core API 0 | 통과 |
| 3-camera shape/dtype | 통과 |
| legacy 5.1 rollback | 통과 |
| raw image p99 ≤30 ms | gradient 통과, random worst-case 실패 |
| mock sim/real target bitwise | 통과 |
| timestep 누락/중복 | 통과 |
| queue underrun | 통과 |
| mismatch fail-closed | 통과 |
| arm codec `<1e-5 rad` | synthetic/codec 통과, 실측 대기 |
| gripper `<0.1 mm` | sim 통과, real 대기 |
| EEPROM readback | 대기 |
| arm trajectory p95 `≤5°` | 대기 |
| steady-state `≤2°` | 대기 |
| gripper p95 `≤2 mm` | 대기 |
| lag difference `≤33 ms` | 대기 |

양쪽 migration 재검증은 Windows `bootstrap_windows.ps1 -CheckOnly -Smoke`와
서버 `bootstrap_server.sh --check-only --smoke --probe`로 통과했다.
2026-06-22 기준 Windows→server gradient RGB 3장 transport는 p99 `18.89 ms`,
서버 local은 p99 `10.55 ms`였으며
second lease와 contract mismatch가 모두 거부되고 반환 chunk가 bitwise 동일했다.

## 7. 안전한 실행

```bash
# 전체 software validation
python scripts/parity/launch.py validate

# Windows 고정 stack/Isaac 재검증
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File scripts/parity/bootstrap_windows.ps1 -CheckOnly -Smoke

# konan147 고정 stack/Isaac/Zenoh 재검증
bash scripts/parity/bootstrap_server.sh --check-only --smoke --probe

# replay server 연결 sim client
python scripts/parity/launch.py sim --steps 64

# hardware 접근 없음
python scripts/parity/launch.py real-dry-run

# torque-off EEPROM readback만 수행
python scripts/parity/launch.py real-readback --port COM8
```

Motion은 다음 두 플래그와 validated calibration이 모두 있어야 한다.

```bash
python scripts/parity/launch.py real-motion \
  --port COM8 \
  --enable-motion \
  --confirm-emergency-cutoff-ready
```

사용자가 비상 전원 차단 준비를 확인하기 전에는 위 명령을 실행하지 않는다.

## 8. 실측 완료 절차

1. torque-off readback report 생성
2. paired arm pose 10~20개 입력
3. real gripper caliper 7~9점 입력
4. `fit_calibration_bundle.py`로 validated bundle 생성
5. runtime manifest 재생성·server restart
6. `pick_cube_v2` provenance 확인 후 canonical dataset 변환
7. no-load dynamics real/sim 실행과 fitting
8. cube payload 장착 확인 후 반복
9. safe pose/workspace trajectory parity report 생성

학습 명령은 이 절차에 포함하지 않는다.
