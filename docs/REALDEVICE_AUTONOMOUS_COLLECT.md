# 실기기 SO-101 자율 Pick-Place 데이터 수집 (LeRobot v3) — 인수인계 문서

> 작성: 2026-06-18 세션. Windows 네이티브 LeRobot 0.4.4 + 고전 비전(HSV) + 해석적 IK.
> 코드: `scripts/real/` (미커밋). 디버그 산출물: `outputs/real_debug/`. 데이터셋: `datasets/pick_cube_real/`.
> 상태 한 줄: **파이프라인·캘리브(터치 3.6mm)·근접 큐브 도달 OK. 단 grasp 가 +y 로 ~2-3cm 치우쳐 미스 → `calib_offset_y≈-0.025` 튜닝이 다음 단계. 그릇은 reach 밖(place 블로커).**
>
> **2026-06-18 전환 — ECE4560 assignment8 큐브 스택(bowl→stack).** 목표 변경: 그릇 담기 대신 **좌→우 큐브를 최우측 바닥에 적재(part2=2/part3=3)** + **cubic spline 궤적(assignment9)** → VLA 데이터. 신규 `trajectory.py`(cubic), `move_cubic`(open-loop 재생·매틱 record), 스택 SM·`stage stack/stackcollect`. §14 참조. grasp +y offset 은 적재 정밀도에 그대로 직결 → **먼저 확정 필요**.

---

## 1. 목표 (Goal)

사람 개입 없이 실기기 SO-101 5축 팔이 스스로 pick-place 데이터를 **LeRobot Dataset v3** 로 수집.
- 작업: 책상 위 회색 큐브를 파란 그릇에 담기 (task = `"pick up the cube and place it in the bowl"`).
- 루프 자율 반복: 큐브 집어 그릇에 넣고(record) → 그릇에서 꺼내 책상에 무작위 산포(reset) → 반복.
- 카메라 ≤3 (top/wrist/front). 데이터셋 = LeRobot v3.
- **학습 정책(SmolVLA/GR00T)은 큐브 1개도 못 집음(사용자 확인) → 컨트롤러 = 고전 비전 + 해석적 IK scripted-expert 뿐.** 이 데이터가 재학습용.

## 2. 제약 (Constraints, 사용자 확정)

| 항목 | 결정 |
|---|---|
| 제어 | **Windows 네이티브 uv** — LeRobot 0.4.4 `SOFollower`, COM 직결(pyserial). Docker/WSL2/ROS2/usbipd 금지 (usbipd USB/IP 모션 중 serial corruption 회피, `docs/REALDEVICE_GRASP_PIPELINE.md` §5) |
| 컨트롤러 | HSV 검출 + `SO101Kinematics` 해석적 IK. 학습 정책 없음 |
| reset | 실제 파란 그릇(그릇에 넣고 그릇에서 꺼내 산포). 5축 그릇 추출이 최난도임을 수용 |
| 자율성 | hand-eye 캘리브까지 자율. grasp 반복 실패 시 중단+보고 |

## 3. 실행 환경 (Execution Environment)

- **OS**: Windows 11 Pro. 셸: Git Bash (`uv run`).
- **로봇**: SO-101 follower, **COM8** (`.env` `ROBOT_PORT`), `id=so101_robot`. 모터 calib 파일 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_robot.json` (존재 확인됨).
- **카메라**: OpenCV index **top=2, wrist=1, front=0** (`.env`), MJPG 640×480. LeRobot `OpenCVCameraConfig` 기본 `color_mode=RGB` → `get_observation()` 가 **RGB** 반환.
- **Python**: `uv run --group teleop python ...` (lerobot[feetech] 0.4.4 + opencv 4.11 + cv2.aruco).
- **단위**: SOFollower 네이티브 = arm **degrees**, gripper **[0,100]** (use_degrees=True). 기록도 동일 단위 → **변환 불필요**(`to_lerobot_units` 왕복 안 함).

## 4. 아키텍처 / 신규 코드 (`scripts/real/`)

```
autonomous_collect.py   진입점. 스테이지 다수. SOFollower IO + 단위 glue + 안전,
                        30Hz waypoint executor, pick/place/reset SM, 캘리브, 기록.
so101_kinematics.py     SO101Kinematics 추출본(Isaac 무의존, math 만). FK/IK/ik_reach.
vision.py               HSV 큐브/그릇/그리퍼-tip 검출 + ArUco + 픽셀→baseXY 호모그래피 + 오버레이.
```
**재사용**: `scripts/sim/lerobot_recorder.py`(`LeRobotV3DatasetWriter`), `scripts/sim/lerobot_units.py`, `scripts/sim/upload_to_huggingface.py`. (recorder 는 pyarrow/imageio 지연 import → Isaac 무의존, Windows OK.)

### 스테이지 (`--stage`)
| 스테이지 | 모션 | 용도 |
|---|---|---|
| `check` | 무 | 연결+joint 읽기+3캠 프레임 저장(카메라 매핑·색·밝기 검증) |
| `jointcheck` | 유 | 밝기 판정(어두우면 토크 풀고 대기) + 각 관절 ±30° 추종 점검 |
| `fold` | 유 | 팔을 park pose(0.20,0,0.26)로 접어 워크스페이스 가림 제거 + 3캠 캡처 |
| `arucotest`/`arucosweep` | 유 | 그리퍼 ArUco 가시성 진단(자세/pitch 스윕) |
| `touchcalib` `--detect`/`(touch)`/`--finish` | 유 | **사용자 터치 캘리브**(현행 캘리브 방식, §6) |
| `calibrate` `[--aruco]` | 유 | 자동 그리드 hand-eye(그리퍼-tip 또는 ArUco) — **저각 카메라서 실패**, 터치로 대체 |
| `reachbowl`/`jointreach` | 유 | 그릇 도달성 진단(IK / joint-space 최대 reach) |
| `pick` | 유 | 근접 큐브 1개 pick 검증(그릇 없음) + grasp/hold 프레임 저장 |
| `test` | 유 | pick→bowl 1회(비기록) |
| `collect` `--target N [--upload]` | 유 | test→verify(5ep 게이트)→collect→finalize→업로드 |

## 5. 핵심 결정사항 (Key Decisions)

1. **Windows 네이티브 제어** — USB-IP serial 불안정 회피. `SOFollower(port=COM8, id, max_relative_target=5, cameras, use_degrees=True)`, `connect(calibrate=False)`(input() 프롬프트 회피).
2. **고전 비전 + 해석적 IK** — VLA 0% 라 배제.
3. **`move_to` = waypoint 당 IK 1회만 풀어 고정 joint target** 으로 수렴. (초기 버그: 매 tick `ik_reach` 재해 → pitch 분기 진동 → false "stuck". 수정 후 안정.)
4. **기록 단위 변환 불필요** — 로봇이 네이티브 lerobot 단위. action=`send_action` 반환(clamped), state=`get_observation`, 이미지=RGB 그대로 `add_frame`. (recorder 는 RGB·imageio 호환.)
5. **`max_relative_target=5`** = per-step joint Δ 클램프(잘못된 IK 점프 차단, 1차 안전). + workspace bound(r 0.12-0.42), stuck/timeout watchdog, connect 3회 재시도.
6. **캘리브 = 사용자 터치 방식** — 자동 그리드 hand-eye 가 저각 카메라에서 실패(§7). ArUco 마커도 실패(§7). 결국 **ArUco 를 책상에 정적으로 두고 사용자가 그리퍼 grasp-point 를 마커 중심에 대 (픽셀↔FK XY) 쌍 수집 → homography**. 7점 median **3.6mm**.

## 6. 캘리브레이션 (현행 방식 — 사용자 터치)

`touchcalib` 2-step (마커는 책상 정적, 점당):
1. `--stage touchcalib --detect` : 그리퍼 비킨 채 책상 ArUco 픽셀 검출 → pending.
2. (그리퍼 grasp-point 를 마커 중심에 댐) `--stage touchcalib` : FK 읽어 pending 픽셀과 1쌍 기록.
3. 마커 위치 바꿔 ~6-8회 반복(workspace 골고루).
4. `--stage touchcalib --finish` : `cv2.findHomography(RANSAC)` → `datasets/pick_cube_real/calibration.json`.

**결과(7점)**: residuals(m) `[0.007,0.004,0.002,0.051,0.001,0.001,0.018]`, **median 3.6mm**, inliers(<1.2cm)=5. 1개 outlier(느슨한 터치)는 RANSAC 자동 제외. (이전 overhead 카메라 자동 그리드는 4.6mm 였으나 카메라 이동으로 무효.)

**주의**: 4점은 항상 homography 정확히 맞춤(median 0, 무의미) → **≥6-8점** 필요. 점은 픽셀·base XY **모두 spread**. ArUco 가 프레임 가장자리서 잘리면 검출 실패(전체 보이게).

## 7. 트러블슈팅 (Troubleshooting — 종합)

| 현상 | 원인 | 해결 |
|---|---|---|
| connect `Failed to write 'Lock'/'Torque_Enable' ... no status packet` | feetech serial 간헐 ACK 누락(특히 gripper id=6) | `ArmIO.connect(retries=3)` — 실패 시 disconnect+2s+재시도 |
| 콘솔 로그 `UnicodeEncodeError cp949` (한글/°/≈) | Windows 콘솔 cp949 | `setup_logging` 에서 `sys.stdout/err.reconfigure(utf-8)` |
| clamp WARNING 스텝마다 폭주 | lerobot `ensure_safe_goal_position` root logger | `_ClampFilter` 로 "clamped to be safe" 메시지 필터 |
| 모든 waypoint "stuck" / 팔이 목표 못 감 | `move_to` 가 매 tick `ik_reach` 재해 → pitch 분기 진동 | **waypoint 당 IK 1회**만 풀어 고정 target(§5.3) |
| observe pose `target out of workspace` | observe_tcp r<0.12 | reachable park `(0.20,0,0.26)` (fold 가 실제 도달한 candidate) |
| 캘리브 "residual too high" 오판 | 게이트가 all-pair RMS 사용 | **median + inliers(<1cm)** 로 게이트(RANSAC outlier 강건) |
| 큐브 검출 0/오검출 | sim 회색(2.5cm)≠실기기 크림 soft 큐브, 흰 매트 글자·그릇 하이라이트 병합 | HSV `S<22·aspect0.7-1.4·circ>0.55` (실측 큐브 S 8-11). MASK off |
| 파란 그릇 검출 불안정(폼폼·큐브 오인) | 옅은 파랑+teal 폼폼 인접 | hue≥95 + **최대 circularity** blob(best-effort). **그릇 정적이라 1회 고정 권장** |
| **그릇 place 불가** | 그릇이 base r≈0.46 > 기구학 max ~0.44 | **그릇을 reach 안(r≤0.33)으로 이동 필요**(물리). 또는 평면 zone fallback |
| 먼 +y 큐브(r>0.30) stall | 확장자세서 servo 추종 실패(P_Coefficient=16 soft + 중력 droop) | 근접 zone(r≤0.28)만 사용. 객체 압축 배치 |
| 저각 카메라 자동 hand-eye 7cm 오차 | depth축 foreshortening + 그리퍼-tip 자동검출이 all-purple 팔의 어깨를 tip 으로 오인 | **사용자 터치 캘리브**(§6)로 우회. 검출 맞은 쌍은 2-3mm |
| ArUco 그리퍼 마커 실패 | ① 종이 미끄러짐(rigid 아님) ② 그리퍼 앞면→top-down 자세서 카메라 외면 ③ 4.6cm=거리상 ~60px 너무 작음(14.8cm=196px 검출됨) | 책상 정적 마커(터치 방식)로 전환 |
| 조명 드리프트 | 야간 암흑(Vmean 120→25, 중앙값 4) → 회색 큐브 어둠에 묻힘 | 워크스페이스 일관 조명 필요. `jointcheck` 가 Vmean<60 이면 토크 풀고 대기 |
| grasp ~2-3cm +y 미스 | tip≠TCP / 터치 자세-pick 자세 차이 / 검출 centroid bias 의 잔차 | **`--calib-offset-y ≈ -0.025`** 튜닝(다음 단계). side-approach 무관(직하강도 동일 offset) |
| "table 1→0 LIFTED" 오탐 | 성공판정이 검출 count 기반(노이즈·가림) | **hold_wrist 프레임으로 실제 파지 확인**. 성공판정 개선 필요(wrist-cam 기반) |

## 8. 주요 설정 (Key Settings — `Cfg` in autonomous_collect.py)

```
robot_port=COM8 robot_id=so101_robot   cam top=2/wrist=1/front=0  640x480 MJPG
max_rel_target=5  fps=30  r_min=0.12 r_max=0.42 z_floor=0.005
safe_z=0.15  observe_tcp=(0.20,0,0.26)  grasp_z=0.02  bowl_clear_z=0.18 release_z=0.08
calib_z=0.06(자동)/터치는 z≈0  side_offset=0.035(--side-offset 0=직하강)
calib_offset_x/y=0 (→ y≈-0.025 튜닝 필요)  grip_open=70 grip_close=6 ([0,100])
joint_tol=0.05rad move_timeout=12 stuck_secs=2.5 close_dwell=0.6
calib 그리드(자동) xs(0.16,0.20,0.24,0.27) ys(-0.08,-0.03,0.03,0.08)
scatter xs(0.19,0.30) ys(-0.11,0.11) sep=0.06   verify_eps=5 verify_min_rate=0.6 grasp_attempts=3
```
오버라이드 CLI: `--grasp-z --side-offset --grip-open --grip-close --calib-offset-x/y --max-rel-target --target --upload --aruco --aruco-id --detect --finish`.

### 기하 근거
- base_link ≈ 책상 레벨(z=0). 큐브 평면 z≈0.0125. 기구학 max reach ≈ L1+L2+L3+LIFT_R = 0.116+0.135+0.159+0.030 ≈ 0.44m (PAN_X=0.0388 기준 r).
- 신뢰 grasp zone: **r ≈ 0.14~0.28** (근접). r>0.30 부터 stall. 그릇 r≈0.46 = 도달 불가.

## 9. 타임라인 / 구간별 결과

**2026-06-17 (overhead 카메라)**
- 파이프라인 3모듈 작성·검증. COM8 연결, 카메라 매핑 확정.
- 자동 그리드 hand-eye(그리퍼-tip) **median 4.6mm** (overhead 카메라서 tip=최상단 purple 로 검출됨).
- 진단으로 발견: **그릇 reach 밖(r≈0.46)**, 먼 큐브 stall, **근접 큐브 도달 OK**.
- 야간 조명 암흑(Vmean 25) → 비전 불가, 대기.

**2026-06-18 (저각 정면 카메라로 이동)**
- 조명 복귀(Vmean 121). `jointcheck` 5관절 ±30° 추종 OK(err 1-3°).
- 사용자가 top 카메라를 저각/정면으로 이동(SO-101 body 가 큐브 가려서). → 큐브 4개 선명, 그릇 중앙쪽. 단 **자동 hand-eye 7cm 오차**(depth foreshortening + tip 검출이 어깨 오인).
- ArUco 시도: 종이 미끄러짐 → rigid → 그리퍼 앞면 마커가 카메라 외면 → 4.6cm 너무 작음. 전부 실패.
- **사용자 터치 캘리브 7점 → median 3.6mm** 성공.
- `pick`(근접 큐브 r=0.18): 시퀀스 stall 없이 완주하나 **grasp 가 +y 로 ~2-3cm 치우쳐 미스**(직하강·side-approach 동일). hold_wrist 빈 그리퍼 확인.

## 10. 검증 방법 (Verification)

- **스테이지별 디버그 프레임**: `outputs/real_debug/{check,fold,touch,aruco,arucosweep,pick,episodes}/`.
- **캘리브 품질**: `calibration.json` 의 `residual_rms_m`/per-pair `residuals_m` + median + inliers(<1.2cm).
- **grasp 실제 파지 확인**: `outputs/real_debug/pick/hold_wrist.png`(손목캠, 손가락 사이 큐브 유무) — **count(1→0) 신뢰 금지**.
- **homography sanity**: 검출 큐브 픽셀 → `pixel_to_base_xy` → r 이 reach 안인지.
- **로그**: `outputs/real_debug/autonomous_loop.log` (UTF-8).

## 11. 현재 상태 / 남은 일 (Remaining)

**현재**: 캘리브 양호(터치 3.6mm). 근접 큐브 도달·시퀀스 완주. **단 grasp +y ~2-3cm 미스.**

**다음 단계(우선순위)**:
1. **grasp offset 튜닝** — `--stage pick --calib-offset-y -0.025` (필요시 x 도) 로 grasp_top/hold_wrist 보며 큐브 중앙 파지될 때까지. (image-right ≈ base +y, image-left ≈ base −y/원거리 — 캘리브 쌍에서 도출.)
2. **성공판정 개선** — 검출 count 대신 wrist-cam 의 손가락 사이 큐브 유무 / 특정 큐브 위치 소멸로.
3. **그릇 reach 해결** — 그릇을 r≤0.33 으로 물리 이동, 또는 평면 drop-zone fallback (place-in-bowl→place-on-zone).
4. **reset(그릇 추출)** — 그릇 도달 가능해진 뒤 `extract_one` + `random_scatter_targets` 검증.
5. `--stage test` → `collect --target N --upload`.

**미해결/리스크**: 큐브 검출이 간헐 2-3/4개만(조명·가림). 그릇 검출 best-effort 불안정(정적 고정 권장). soft 큐브 grasp 신뢰성 미검증. 데이터 수집 **0건**.

## 12. 참고 자료 (References)

- 본 repo: `docs/REALDEVICE_GRASP_PIPELINE.md`(이전 WSL2/ROS2 시도·USB-IP 교훈), `docs/PATH_A_NATIVE.md`(Windows 네이티브 lerobot CLI), `docs/SIM_REAL_INFERENCE_PARITY.md`(단위·gripper scale), `AGENTS.md`.
- 코드: `scripts/environments/pick_cube_state_machine.py`(원본 `SO101Kinematics`), `scripts/sim/lerobot_recorder.py`·`lerobot_units.py`·`upload_to_huggingface.py`.
- 외부(설계 참고): AutoEval(autonomous reset+VLM success+safety, arXiv 2503.24278), CIIRC Robot-Vision-PickPlace(homography workspace calib, IEEE CASE 2023), "Working Backwards: Learning to Place by Picking"(2312.02352), LeRobot v3 dataset 문서.
- 계획 파일: `~/.claude/plans/i-want-to-record-kind-galaxy.md`. 세션 인계: `CONTEXT.md` `## 작업 인계 (2026-06-17 야간 …)`.

## 13. 실행 빠른참조 (Runbook)

```bash
# 진단
uv run --group teleop python scripts/real/autonomous_collect.py --stage check
uv run --group teleop python scripts/real/autonomous_collect.py --stage jointcheck

# 캘리브(사용자 터치) — 점당 detect→touch, ~6-8점, 마커 책상 정적
uv run --group teleop python scripts/real/autonomous_collect.py --stage touchcalib --detect
uv run --group teleop python scripts/real/autonomous_collect.py --stage touchcalib            # touch
uv run --group teleop python scripts/real/autonomous_collect.py --stage touchcalib --finish

# grasp 튜닝 → 수집
uv run --group teleop python scripts/real/autonomous_collect.py --stage pick --calib-offset-y -0.025 --side-offset 0
uv run --group teleop python scripts/real/autonomous_collect.py --stage test
uv run --group teleop python scripts/real/autonomous_collect.py --stage collect --target 50 --upload
```
> 디버그 프레임을 매 스테이지 후 확인. `calibration.json`·`touch_pairs.json` 은 `datasets/pick_cube_real/`. 팔은 disconnect 시 토크 해제(limp, 안전).

---

## 14. ECE4560 Assignment8 큐브 스택 (2026-06-18 전환)

출처: `maegantucker.com/ECE4560/` assignment8(IK2 Block Stacking)+assignment9(Cubic Spline). 영상 2개(2블록·3블록 좌→우 적재). 본문 grounded 스펙 + 우리 파이프라인 적용.

### 14.1 과제 매핑

| 과제 항목 | grounded 값 | 우리 구현 |
|---|---|---|
| Part2 | 2블록 적재 | `--num-blocks 2` (기본) |
| Part3(옵션) | 3블록 적재 | `--num-blocks 3` |
| IK | 해석 기하 5-DOF | `SO101Kinematics` (동일 계열) |
| 궤적 | 본문=선형 / **9=cubic spline** | `trajectory.py` cubic(zero-vel 양끝) + `move_cubic` |
| grasp | **수직** raise(+0.03)→descend→토글→retract | `_pick_at`/`_place_at` (side-approach 아님) |
| 적재 z(중심) | 1층0.014/2층0.043/3층0.071 (큐브변 0.0285 증분) | `place_z = running_top + pick_size/2`, `running_top` 초기=base_size. **검증: 동질 2.85cm → 0.043/0.071 정확 일치** |
| gripper | open=50 close=5(과제 스케일) | 우리 HW 자체 70/6 유지(`--grip-open/--grip-close`) |

### 14.2 cubic spline (assignment9, `trajectory.py`)

```
p(t)=a0+a1 t+a2 t²+a3 t³ , a0=θ0, a1=0, a2=3Δ/T², a3=-2Δ/T³ , tlim=clip(t,0,T)
```
per-DOF 독립. `move_cubic`: 세그먼트 시작마다 **측정 6-vec=a0**(랙 흡수), IK 1회로 goal, 6축(arm5°+grip[0,100]) 동시 spline, 30Hz 송신+record. arm Δ0·grip만 변하면 부드러운 그립 토글. open-loop 재생(수렴 watchdog 없음) → 기존 jerky 데이터 결함 회피.

### 14.3 스택 전략 (사용자 확정)

- **전역 최우측 큐브 = 바닥 고정**(재파지 없음). 좌측 (num_blocks-1)개를 좌→우로 그 위에 적재.
- 큐브 이종(4cm 2개+4.8cm 2개): `--base-size`/`--pick-size` 분리. `--cube-size` 는 양쪽 set. 기본 0.04.
- 좌→우 = base y 오름차순(image-right=base +y, calib 규약).
- reset: `--manual-reset`(사용자 재배치 Enter) 또는 자동 `unstack_sequence`(스택 위→산포). **unstack 이 최난도·미검증 리스크.**

### 14.4 데이터 경로

- calib 읽기 = `datasets/pick_cube_real/calibration.json` (`--dataset-root`). 기존 터치 3.6mm 재사용.
- 스택 LeRobot v3 쓰기 = **`datasets/stack_cubes_real/`** (`--stack-dataset-root`). **분리 이유**: writer `overwrite=True` 가 `rmtree(dir)` → calib dir 와 같으면 calibration.json 삭제됨. 별도 dir 로 방지.
- task 문자열 = `"stack the cubes"`. upload repo = `{HF_USER}/so101_real_stack_cubes`.

### 14.5 실행 (Runbook)

```bash
# 0) 캘리브 재사용(터치 3.6mm 있으면 skip). grasp +y offset 먼저 확정:
uv run --group teleop python scripts/real/autonomous_collect.py --stage pick --calib-offset-y -0.025 --side-offset 0

# 1) 스택 1회 검증(비기록). 큐브 2개 책상에 두고(최우측=바닥). 사이즈 실측 주입:
uv run --group teleop python scripts/real/autonomous_collect.py --stage stack \
  --num-blocks 2 --base-size 0.048 --pick-size 0.04 --calib-offset-y -0.025 --grip-open 100
#   → outputs/real_debug/episodes/stacktest_{before,after}.png + grasp_{top,wrist,front}.png 확인

# 2) 데이터 수집(test→verify→record N→upload). 리셋은 수동 권장(초기):
uv run --group teleop python scripts/real/autonomous_collect.py --stage stackcollect \
  --num-blocks 2 --base-size 0.048 --pick-size 0.04 --calib-offset-y -0.025 --grip-open 100 \
  --target 30 --manual-reset --upload
```

### 14.6 리스크 / 미해결

- **grasp +y ~2.5cm offset** — 적재 정밀도에 직결(7 §grasp). `--calib-offset-y` 우선 확정.
- **4.8cm 큐브 jaw open 폭** — `--grip-open 70` 으로 안 벌어지면 `100`. (soft 큐브 close=6 압착.)
- **성공 판정** = table blob count 감소 proxy(약함). hold/after 프레임·wrist-cam 병행 확인 필요.
- **auto unstack reset** 미검증(소프트 스택 위 큐브 5-DOF 추출). 초기엔 `--manual-reset`.
- 이종 큐브에서 `pick_size` 단일값 — 픽 큐브가 전부 같은 변일 때만 정확. 섞으면 층별 사이즈 미반영.
