<!-- ╔═══════════════════════════════════════════════════════════════════════╗
     ║  NORTH STAR — 매 세션/compaction 직후 먼저 읽는다. 변경 금지(상수).      ║
     ╚═══════════════════════════════════════════════════════════════════════╝ -->

## 🧭 North Star (불변 — 매 사이클·compaction 후 재확인)

- **마스터플랜**: [`docs/SIM2REAL_MASTERPLAN.md`](docs/SIM2REAL_MASTERPLAN.md) · **현황**: [`TASKS.md`](TASKS.md)
- **불변 계약**(모든 sim 데이터·정책 I/O가 일치해야 함): `v3.0` · robot_type `so_follower` · action/state 각 **6-dim joint position** (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper) · `observation.images.{top,wrist,front}` 480×640×3 h264 **fps 30** · task `"pick up the cube and place it in the bowl"`.
  (2026-06-08 North Star 변경: 2cam → 3cam. front 카메라 추가. 기존 2cam 데이터셋은 front 없이 수집된 것으로 front 채널 없음 — 신규 수집 시 3cam 기준.)
- **자율 계약**: Codex `/goal` 시작 후 **A~E 무인 자율**(묻지 않음). 멈추는 경우는 둘뿐 — F~G 실기기 경계 / 복구불가 블로커(동일 task 3회 재시도 후 우회·기록). 게이트 미통과 task는 done 금지.
- **복구 프로토콜**: 세션/compaction 직후 ① 마스터플랜 §0·§1·§7 → ② TASKS.md(현재 phase·in_progress·blocked) → ③ 아래 최근 인계 1~2개 순서로 재로드. 추측 금지 — 상태 파일에 없으면 새 task로.
- **머신**: GPU 중량(Isaac·RL·롤아웃·GR00T) = 서버 konan147(48GB), 산출물 `/DISK1/so101-sim2real`. 경량·실기기·오케스트레이터 = Windows. sync 허브 = `origin`(github PubCyBerry/SO101-Sim2Real).

---

## 작업 인계 (2026-06-11 후속 — 셀프피드백 루프: v8 65% 회귀 진단→v10 게이트 수정 85%(영상), 4큐브 정렬게이트 v12 검증중)

- **목표(사용자)**: 알고리즘 셀프피드백 반복 개선 — 만족 수준까지. **매 이터레이션 영상 필수**(사용자 피드백용 + ffmpeg 키프레임 셀프리뷰). 목표 1큐브 ≥90% 유지, 4큐브 1차 mean ≥3.5/4 + all-4 ≥50%, 2차 all-4 ≥75%.
- **버전사(이번 세션, 전부 20ep 1큐브 기준)**: v8(방향분해 ⊥×2.0) **65% 회귀 확정** — ① 저tilt(15°) 후보가 err_perp 작아 승리→scoop 부족 헛닫기(ep2/8/10) ② descend_fix 자유 재계산이 다른 tilt 로 스윙→큐브 0.57m 비산(ep4/12/13). → v9(채점 v6 복원+fix tilt 잠금+hold 폴백) **35% 폭락** — 진범은 별개: **lift 게이트 마진 0**(`_cube_lifted` min_lift 0.08 = lift 명령 0.08) 으로 쥔 큐브(hold 1~2cm)가 +6~7cm 에 머물면 false 판정→멀쩡한 grasp 버리고 재시도하다 떨굼(실패 13ep 중 10ep). v9-novid 40% 로 영상 가설 기각. → **v10 = min_lift 0.05** → **85%(영상 ON, 역대 영상 run 최고 — v6 영상 80%)**. 실패 3ep: ep2(저tilt 헛닫기), ep8(아래), ep16(잡고 운반 중 그릇 밖 낙하).
- **4큐브(v10, 8ep 영상)**: all-4 2/8(25%), mean 2.625/4(구버전 1.625 대비 +1.0). per-cube Cube1 87.5/Cube2 62.5/Cube3 62.5/Cube4 50. 주 실패 = **어긋난 채 닫기**(실패 attempt 닫기 시점 grasp point 오차 평균 2.1cm) → 헛닫기+닫는 손가락이 큐브를 2~7cm 밀어 다음 시도 악화→fail-fast 이탈(영상 ep3: 큐브 2개가 베이스 옆·책상 밖으로 표류). "쥐고 게이트 미달" 패턴은 0건(게이트 수정 유효).
- **v11(--grasp_arm_step_delta 0.05 감속) 기각**: 1큐브 75%(-10%p), 4큐브 ep0~4 all-4 0/5. 느린 접근이 접촉 시간만 늘림, ep4 0.71m 비산 재발.
- **v12(검증중, PID 2019120/2019121)**: **닫기 전 정렬 게이트** `--grasp_misalign_gate 0.02` — 닫기 직전 grasp point↔큐브 오차 > 2cm 면 닫지 않고 attempt 포기(마지막 attempt 제외). 큐브를 안 건드려 재시도가 깨끗. + fail_diag 에 descend_fix/skip_close/abort phase·pre_close_err_m 추가. 결과: `outputs/det_v12_{4cube_8ep,1cube_20ep}.json`.
- **결정적 실패 ep8**(spawn [1.797,-0.450], v8/v10/v11 공통 실패): jacobian refine 폭주(35~78step, ferr 0.02~0.25)·v10 attempt2 는 쥐고도(hold 1.0cm) lift 가 +0.7cm 만 상승. 미해결 — 단일 ep 라 우선순위 낮음.
- **영상**: `outputs/review_videos_v9/`(35% run), `review_videos_v10/`(85%, 20ep), `review_videos_4cube_v10/`(8ep), `review_videos_4cube_v11/`(기각 run ep0~6), `review_videos_4cube_v12/`(생성중). 키프레임 셀프리뷰 = `ffmpeg -vf "select='not(mod(n,30))'" + Read`.
- **함정(이번 세션 신규)**: ① grep 출력이 dotfiles wrapper 로 변형됨 — awk/raw tail 사용 ② v7b sweep 이 ep10 에서 28분 무한정체(CPU 108%, fail-fast 부재 코드) → PID kill, py-spy 는 ptrace 권한 불가 ③ PhysX 잡음 ±2/20ep — ≥3 차이만 판정 근거로.
- **다음**: v12 결과 판정 → 미달 시 다음 레버(저tilt ladder 정리/descend 실행오차 원인/40mm 개방) → 목표 달성 시 결과 보고 + 사용자 영상 리뷰 대기 → 데이터셋 게이트(3cam 계약: SM CAMERA_KEYS=top/wrist 2cam 불일치 해소 필요). 미커밋: `pick_cube_state_machine.py`(v9~v12 수정).

---

## 작업 인계 (2026-06-11 — PickCube RL: v11 PBRS(place potential-based shaping))

> 상세 기록은 **`docs/RL_LSTM_PICKCUBE.md`**(T1~T25 전체 시행착오). 여기는 인계 요약만.

- **위치**: 브랜치 `worktree-lstm-ppo-pickcube` → main 병합 완료. 재개 시 새 worktree/브랜치에서. 실행 = 메인 `.venv` + `PYTHONPATH=$(pwd)/src`. 서버 konan147 GPU(48GB) 공유.
- **목표**: cube_desk 단일 큐브 pick→bowl, LSTM+PPO. scratch(부트스트랩 없는) **성공률 ≥0.80** → 1→2→3→4 커리큘럼.
- **✅ grasp 해결(v4 점화 + v6 신뢰성)**: scratch.grasp/lift/over_bowl **0.85~0.89** 안정. 효과 개입 = grasp_contact(ContactSensor)+close-bridge(3.0)+slew 2.5+RND grasp_focus(v4) + **cube_predisturb 패널티(-3)·cube_lost 추락 종료**(v6, 큐브 변위 -0.40→-0.08·추락 6.9%→3.4%).
- **남은 핵심 문제 = place 정밀도**: grasp 후 `over_bowl 0.86 → placed 0.10`(전이 12%). hover local-optimum. **PBRS 도입(T29)**: Φ=그릇안 1.0+밖 0.3·xy+0.2·z, transport/place_height/insert weight→0. v11(iter 66)은 종료·폐기.
- **obs 87dim**: joint·grasp point·큐브·그릇 pos+rel + 속도 + 큐브 yaw/크기·ee quat·그릇 quat. LSTM(256,1)+PPO, num_envs 16384, gamma 0.997, RND grasp_focus(30dim).
- **outputs 경로**: 새 학습 시 `--log_root_path /home/konan147/Workspaces/SO101-Sim2Real/outputs/rl/rsl_rl` 로 main 레포 outputs 사용. cron_monitor_v4.sh 의 RUN_GLOB 도 main outputs 절대경로로 갱신됨.
- **다음 레버**: place 미해결 시 Φ 가중 튜닝(xy 0.3·z 0.2)·grasp 단계 PBRS화·over_bowl_drop PBRS화. success→0.80 시 커리큘럼 1→2.

---

## 작업 인계 (2026-06-11 — in-process 결정적 grasp SM: 1큐브 90% 달성·4큐브 첫 all-4·리뷰영상 시스템 / 진행중)

- **목표(사용자 확정)**: in-process Isaac Lab SM(`pick_cube_state_machine.py`)으로 cube_desk pick-and-place — expert 데이터셋 생성용, **순수 물리 grasp만**(grasp-assist 금지), 1큐브 고신뢰 먼저 → 4큐브. 계획서 `~/.claude/plans/isaac-sim-cube-desk-memoized-starfish.md`.
- **🔑 핵심 구현 — `--grasp_config_mode deterministic`**: random-FK 5000샘플(단일 grasp ~67%)을 **FD-DLS 결정적 솔버**로 교체. in-sim 가상 FK 위 Gauss-Newton, 난수 0. grasp 단계 = tilt ladder(양수만·중간 tilt 우선 `45,55,35,25,65,15,75`) + scoop 채점 + **개방축↔큐브 면 정렬**(roll task, 잔차 0~2° 달성) + 수평 오차 채점. descend 가 고른 **tilt 를 닫기 단계에 잠금**(자유 재계산이 반대 tilt 로 갈아타며 스윙→큐브 쳐냄 5.5cm 실측). 착좌 게이트(descend 오차 크면 큐브 현위치 기준 보정 descend 1회), 미세 Jacobian refine(팔 도착 후 잔차 폐루프), 폴백 체인(잠긴 tilt 실패→자세유지+닫기만; 그 외→random_fk). 상세 함정 4건 = `docs/TROUBLESHOOTING.md` §SO-101 5DOF grasp 후속 3.
- **결과(1큐브 DR-on 20ep, 동일 seed)**: random_fk(개선환경) **85%** vs deterministic v6 **90%**(18/20, 실패=reach 경계 spawn). 버전사: v1 90%→v2 85%(속도·tilt잠금)→v3 90%(음수tilt 제거)→v4 80%(깊이 -8mm 회귀)→v5b 80%(-4mm+침투필터 -3mm 함정)→**v6 90%(깊이 +5mm 복원)**→**v7 65%(수평 hard gate 7mm — 가장자리 후보 전멸, 회귀 확정)**→v7b(hard 제거·스칼라 가중만) 검증 중→**v8(미검증)**: 수평 오차 **방향 분해**(면을 따라 미끄러진 ⊥개방축 성분 ×2.0 — 모서리/꼭지점 그립의 직접 원인, 무는 방향 ∥ ×0.3) + `_cube_out_of_range` **fail-fast**(책상 낙하/도달영역(scatter+6cm) 밖이면 잔여 재시도 즉시 포기) + attempt별 `cube_moved_m` 변위 진단(잘 집으면 큐브가 거의 안 움직임 — 사용자 영상 관찰의 정량화). ⚠ **20ep ±2 는 PhysX 비결정성 잡음** — 같은 plan·spawn 도 run 간 성패 뒤집힘. 메커니즘 근거 없는 미세튜닝 금지.
- **속도(사용자 요청 18초/4큐브)**: 그리퍼 2속(이동 `--gripper_travel_step_delta` 0.06≈1.8rad/s / close-on-cube 0.02≈0.6rad/s — 과거 전구간 0.005 가 "단계마다 정지" 원인), dwell 단축, lift 0.08/transport 0.12/place 0.12(=transport, place_descend 사실상 생략·그릇 테두리 위 떨굼), approach 0.10. 1큐브 에피소드 최단 7.2초 → 4큐브 ~28초가 현실 하한(관절 5rad/s 캡 유지 시).
- **4큐브**: 구버전 8ep = 평균 1.625/4, **all-4 0/8**(주범: ① transport joint-space 보간이 호를 그리며 그릇 엎음(영상 확인) ② 얕은 그립·모서리(마름모) 그립). 수정(경유점 `transport_via`, 높이, 면 정렬) 후 **ep0 4/4 — 프로젝트 첫 all-4**. 전체 8ep 재검증은 v7b 확정 후.
- **리뷰 영상 시스템(신규)**: `--review_video_dir`(전용 뷰어 카메라 주입, 매 step 캡처 30fps, 에피소드별 `epNN_{ok,fail}.mp4`), `--review_pose_check`(구도 PNG), `--review_cam_eye/target/focal`(확정 구도: eye 2.60,-0.42,1.22 / target 1.76,-0.36,0.70 / focal 17). 영상: `outputs/review_videos/`(1큐브 v3, 10), `outputs/review_videos_v6/`(1큐브 v6, 10), `outputs/review_videos_4cube/`(구버전 8), `outputs/review_videos_4cube_v2/ep00_ok.mp4`(첫 all-4).
- **per-cube 수정**: stale `CUBE_HALF_Z=0.0125`(구 2.5cm) → `CUBE_HALF_HEIGHTS`(30/40mm, 자산은 이미 30/40mm — `author_pick_cube_scene.py::CUBE_SCALES`), 40mm pick 개방 1.4/descend 0.8. sweep 에 `duration_s`·`bowl_tipped`·`fail_diag`(실패 ep 압축 진단) 추가.
- **사용자 피드백 누적(영상 기반, 반영됨)**: 단계 간 정지 제거 / 그릇 누름→테두리 위 떨굼 / 번쩍 들기→높이 하향 / 손목 수평(palm-down) 유지 후 떨굼(`level_axis` task) / 꼭지점 그립→수평 gate(v7 hard 는 회귀, v7b soft 로) / 그릇 쿼터니온 추적(`bowl_quat_w`·`_bowl_tipped`).
- **다음 할 일**: ① v7b 완료(ablation 기록용) → **v8 20ep 검증**(방향분해+fail-fast 포함) ② 통과 시 v8 영상 체인(1큐브 10ep `review_videos_v8/` + 4큐브 8ep `review_videos_4cube_v8/`) → 사용자 피드백 ③ 4큐브 신뢰성 측정 → 데이터셋 생성 게이트(LeRobot v3 — ⚠ SM 의 `CAMERA_KEYS`=top/wrist 2cam, North Star 3cam 계약과 불일치: 데이터셋 단계에서 front 추가 필요) ④ 남는 실패(reach 경계 spawn)는 scatter 안쪽/바깥 한계 1~2cm 축소(env 변경, 사용자 결정) 제안.
- **함정/운영**: 백그라운드 SM 종료는 **python PID 직접 kill**(셸 wrapper kill 은 orphan 잔존 — 이번 세션 v5 잔재 1시간 발견), `pgrep -f` 자기 셸 매칭 주의(브래킷 패턴 `[.]venv/...` 사용). GPU 공유(학습 잡 ~30GB 상주) — headless SM 2개+녹화 1개까지 병렬 가능. multi-env 벡터화는 FSM 구조상 비용>이득, 데이터셋 대량 생성 시 프로세스 샤딩(시드 분할)으로.

---

## 작업 인계 (2026-06-11 — Sim VLA 추론: ROS 2 경로 / 코드 완료·실행 검증 대기)

- **목표**: Isaac Sim SO-101 팔을 학습 VLA(SmolVLA/ACT, Docker `policy-server` gRPC)로 구동. transport **ROS 2** 확정(ZMQ 초안 폐기). 계획서: `~/.claude/plans/scripts-environments-teleoperation-tele-jolly-firefly.md`. 상세 문서: `docs/PATH_E_CUMOTION_ROS.md` §7.
- **아키텍처(3 프로세스)**: ① 호스트 isaac venv `run_cube_desk_ros_bridge.py` 상주(joint_states/clock/tf **+ 신규 카메라 3대** publish, `/isaac_joint_commands` sub→ArticulationController 직접 적용) ② `vla-ros` 컨테이너 `so101_vla_policy` 노드(obs sub→policy-server gRPC→`/isaac_joint_commands` pub) ③ Docker policy-server. cuMotion/MoveIt 미경유.
- **🔑 환경 사실**: 서버 konan147 엔 **ROS 미설치**(`/opt/ros` 없음) — Isaac Sim 은 **번들 jazzy lib** 로 publish(호스트 ROS 불필요). **런타임 분리** rclpy(py3.12) ↔ lerobot(py3.11 venv) → VLA 노드는 별도 py3.12 컨테이너(`Dockerfile.vla_ros`)에서 lerobot pip 설치. 전처리(rename/resize/normalize)는 서버측, 클라는 raw obs 만.
- **단위 계약**: state/action LeRobot 단위(arm deg / gripper [0,100]×31.75) ↔ sim rad. `so101_vla_policy/units.py`(vendored, `scripts/sim/lerobot_units.py` 미러). SmolVLA rename `top→camera1` 등은 env/smolvla.env `RENAME_MAP`.
- **변경/신규 파일**: ① `scripts/sim/lerobot_units.py`(유지·공용 변환) ② `scripts/sim/rollout_to_lerobot.py`(import 교체, 유지) ③ `scripts/sim/run_cube_desk_ros_bridge.py`(**카메라 publish 추가** — USD Camera prim world→opengl 변환 + render product + ROS2CameraHelper, `--no_cameras` 토글) ④ `ros2_ws/src/so101_vla_policy/`(신규 pkg: vla_policy_node + joint_command_to_trajectory shim + units + launch/config) ⑤ `docker/Dockerfile.vla_ros` + `vla-ros-entrypoint.sh` + compose `vla-ros` 서비스 ⑥ docs PATH_C §6(포인터)·PATH_E §7. **teleop_se3_agent.py 는 완전 원복**(ZMQ 초안 삭제).
- **검증 완료(코드 레벨)**: 전체 ast parse·compose yaml·entrypoint sh OK. units 라운드트립 정확(gripper 100°↔1.745rad). bridge convert helper/카메라 상수 import 확인(post-boot). gRPC 프리미티브 import OK.
- **남은 일(사용자/GPU+Docker)**: ① `Dockerfile.vla_ros` **빌드 미검증**(py3.12 lerobot+torch-cpu+numpy<2 resolve, cv_bridge ABI) ② bridge 카메라 런타임(robot USD link prim `gripper`/`shoulder` 존재·convention view) — Isaac 부팅 필요 ③ 풀 파이프라인: PATH_E §7.4 ①②③ 순. ④ action sink shim(`joint_command_to_trajectory`)은 실기기 controller 이름 param 정합 필요(scaffold).
- **리스크**: numpy ABI(cv_bridge↔torch, `numpy<2` 핀), `${HF_USER}` 미보간 시 param 지정, GetActions 빈 chunk timeout 재시도, pickle 0.4.4↔0.5.1(실기기 검증됨).

---

## 작업 인계 (2026-06-10 — 실기기 grasp 파이프라인: feetech deactivate 근본해결 + 제어 파이프라인 동작 / 진행중)

- **목표**: scripted-expert(top 호모그래피 + IK + wrist visual servo)로 SO-101 큐브 grasp 데모 자율 생성 → LeRobot v3.0 녹화. 범위=캘리브→grasp→녹화(유저 확정). 계획서: `~/.claude/plans/context-md-dynamic-meadow.md`.
- **Step 0 완료**: 스택 재기동(follower+move_group+cv2 3캠+rosbridge, FastDDS, env.sh). WSL ws src 는 Windows repo **심링크**(sync 불필요). 250ms 헤더는 빌드에 이미 반영돼 있었음.
- **🔑 근본 블로커 해결 — feetech 단일 read timeout → 전체 deactivate**: `FeetechHardwareInterface::read()` 가 sync_read 1회 실패에 `return_type::ERROR` 반환 → ros2_control 이 하드웨어+3컨트롤러 전부 deactivate. 250ms timeout 으로도 USB-IP(mirrored) 지연 스파이크가 가끔 초과 → 수십 분~시간 내 1회 발생만으로 팔이 죽음(이전 세션 "deactivate" 의 진짜 원인). **수정(적용·재빌드·검증)**: `feetech_ros2_driver.{hpp,cpp}` 에 `consecutive_read_failures_` + `kMaxConsecutiveReadFailures=10` 추가 — 연속 실패 임계 미만이면 마지막 상태 유지 후 OK(cycle skip), 임계 이상(진짜 단선)일 때만 ERROR. 재빌드 후 재기동 → 3컨트롤러 active 유지.
- **🔑 데이터 경로 경험적 판정**:
  - native rclpy **카메라** 수신 OK(`cap.py` 3캠), **rosbag2** OK(joint_states 50Hz + 3캠 25fps 캡처 검증) → 녹화는 rosbag2 가능.
  - native rclpy **joint_states/TF 수신은 불안정** — 단 이는 discovery 문제가 아니라 **하드웨어 deactivate 시 broadcaster 가 죽어 새 메시지 0**(rosbridge 는 마지막값 캐시라 동작)인 정황. → **제어(joint_states/IK/FK/trajectory)는 ros-mcp(rosbridge) 경유**, **이미지 캡처·검출은 native cap.py/detect_*.py** 로 역할 분담.
- **제어 파이프라인 동작 확인**: ros-mcp `/compute_ik`(group manipulator, gripper_frame_link) → `send_action_goal` `/follower/arm_trajectory_controller/follow_joint_trajectory`(5축, 단일점+velocities) → 실기기 이동 성공("Goal successfully reached!") → joint_states+top 이미지로 검증. **pick_ik 는 global 모드라 flip 위험 → 모션 전 joint delta 안전검사 필수.**
- **⚠️ 주의/관측**: ① joint_states `name` 순서=**알파벳순**[elbow_flex,gripper,shoulder_lift,shoulder_pan,wrist_flex,wrist_roll] — 이름기준 재정렬 필수. ② 5-DOF IK 위치오차(z 명령0.051→실제0.043, ~9mm) → 캘리브는 **이동 후 FK XY** 로 페어링. ③ Overrun WARN(read~22ms>20ms@50Hz) 빈발하나 deactivate 아님 — 필요시 update_rate 30~40Hz 하향 여지. ④ 캘리브/grasp 전 **방 조명 필수**(야간 소등 시 3캠 암흑→검출 0). ⑤ WSL 호출은 PowerShell 툴 사용, inline 에 `>`/`2>`/`$()`/`&&` 금지(Git Bash MSYS 경로변환·PowerShell 리다이렉트 가로챔) → 스크립트 파일 패턴.
- **씬**: top 오버헤드(팔=화면 하단 진입), 큐브 4개 상단-중앙(`detect_cubes.py` 4/4: px (253,219)(300,257)(350,166)(400,223)), 그릇 좌측. 현재 EE base (0.186,-0.023,0.043).
- **임시 스크립트(Windows %TEMP%)**: `so101_cal.py`(native IK/FK/move 헬퍼 — native js 불안정으로 보조용), `so101_run.sh`(env.sh source 래퍼); WSL `/tmp` 에 `cap.py`/`detect_cubes.py`/`detect_gripper.py`. Step 4 에서 repo `ros2_ws/src/so101_bringup/scripts/` 로 승격 예정.
- **🔑 5-DOF reach 해결(IK 설정)**: `/compute_ik`(pick_ik)가 전방 좌표를 전부 -31 로 거부했는데, 원인은 5-DOF 가 full 6-DOF pose(위치+방향)를 못 맞춰서. **`kinematics.yaml` 에 `orientation_threshold: 3.15`(≈π, 방향 무시) + `rotation_scale: 0.02`(position 우선) + `position_threshold: 0.01`** 로 position-only IK 화 → 전방 타겟 IK 성공(probe 11/11 ok). **단 `approximate: false` 필수** — `approximate: true` 는 도달 불가 타겟에도 위치 수십 cm 오차 근사해를 success 로 반환(검증서 achieved pose 가 타겟과 30cm 차이로 발각). 현재 `approximate: false` 고정(미검증 — 다음 세션 재probe).
- **🔑✅ serial corruption 해결(3겹)**: (이전엔 미해결 벽으로 봤으나 해결됨)
  1. **Hyper-V 방화벽 해제(유저)** — usbipd-win 의 USB/IP 는 TCP 로 WSL 에 전달되는데 Hyper-V 방화벽이 그 스트림을 간섭해 serial read 가 깨지던 것이 **근본 원인**. 해제 후 timeout/checksum 실패 급감. (`Set-NetFirewallHyperVVMSetting -DefaultInboundAction Allow`)
  2. **feetech read ride-through**(`feetech_ros2_driver.{hpp,cpp}`) — read 실패 시 `communication_protocol_->flush_input()`(=`SerialPort::flashInputBuffer()`) + **항상 OK 반환(절대 deactivate 안 함)**. 단일/연속 실패로 안 죽음.
  3. **sync_read servo ID 검증**(`communication_protocol.hpp`) — 응답의 `response_buffer[0]`(servo ID)가 기대 `ids[i]` 와 다르면 거부. USB-IP 바이트 드롭으로 응답이 한 servo 밀려 **per-servo checksum 은 통과하나 joint↔value 어긋난** frame-shift corruption(명령 안 한 gripper 가 -1.098 등)을 잡아낸다. 이게 캘리브를 막던 핵심. → joint_states 가 일관되게 clean(3회 연속 동일·타겟 일치 검증). read 실패는 ride-through 가 흡수(deactivate 0).
- **🔑 DDS = FastDDS 유지**: CycloneDDS 는 921KB raw Image 를 native·rosbridge 양쪽 모두 0 전달(작은 msg 만 됨) → 카메라 불가. FastDDS(SHM)만 이미지 cross-process 전달. serial 은 DDS 무관(위 방화벽 문제였음)이라 FastDDS 로 충분. `env.sh` FastDDS 고정.
- **플랫폼 안정화 완료**: serial clean + joint_states 신뢰 + 카메라 native OK + IK(position-priority, approximate:false) 전방 도달 + 제어(IK→trajectory) 동작 + rosbag2 녹화 검증. **이제 캘리브 sweep 진입 가능**(진행 중).
- **임시 스크립트(%TEMP%, /tmp 는 wsl --shutdown 시 소실되므로 %TEMP% 사용)**: `so101_cal.py`(IK/FK/move 헬퍼 state/iktest/move/movej), `so101_calib.py`(IK 격자 캘리브 probe/run), `so101_jcalib.py`(joint-space FK 캘리브 jplan/jrun), `cap.py`/`detect_cubes.py`/`detect_gripper.py`(비전), `so101_run.sh`(env source 래퍼), `so101_kill.sh`(self-kill 회피 종료), `so101_build.sh`(feetech 재빌드), `so101_check.sh`(검증).

---

## 작업 인계 (2026-06-09 — 실기기 조작 + VLA teleop 데이터 생성 착수 / 진행중)

- **상위 목표(유저 확인)**: 현재 학습된 VLA 성공률 ~10%(엉망). **재학습용 양질 teleop 데이터를 Claude가 생성**하는 게 목적. 전략 = 시뮬 대량(오라클) + 실기기 소량(sim2real 보정) **둘 다**, **소규모 검증 먼저**. (기존 소량 데이터는 유저가 수동 제작.)
- **VLA 추론 현 구조**: ROS2 미경유, policy-server gRPC 직결 → 추후 수정 필요(유저 언급).

### 실기기(WSL2 ROS2) 진행 — 이번 세션
- **통합 graph**: follower + cv2 카메라 3캠 + rosbridge 전부 **FastDDS** 단일 graph. ros-mcp(127.0.0.1:9090)로 제어/관측. [[project-wsl2-camera-fastdds]]
- **하드웨어 deactivate 2건 해결**:
  - `config/ros2_control/follower_controllers.yaml` `update_rate 100→50Hz` (USB-IP overrun 누적 완화)
  - `feetech_ros2_driver/.../serial_port.hpp` serial timeout **50→250ms** (mirrored USB-IP 지연 스파이크가 50ms 초과 → 수 분마다 deactivate 되던 것 해결). feetech 재빌드 필요. **이게 핵심** — manual·MoveIt 공통.
- **arm 직접 joint 제어**: `FollowJointTrajectory` 액션(send_action_goal)으로 동작 확인. 단일 점 목표 + velocities 포함 형식. 컨트롤러 active 일 때만 수락.
- **수동 grasp 시도 → 실패(예상)**: 캘리브·인식 없이 5축 팔을 2.5cm 큐브에 눈대중 정렬 불가 → MoveIt 전환.
- **MoveIt 데모 실기기 기동**: `follower_moveit_demo.launch.py hardware_type:=real usb_port:=/dev/so101_follower`. move_group(group `manipulator`, base_link→gripper_frame_link) + RViz. **RViz Plan&Execute 동작 = IK 입증**. (RViz marker 안 보임 → Displays>MotionPlanning>Planning Request: "Query Goal State" 체크 + "Interactive Marker Size" 0→0.1.)
- **scripted expert 파이프라인(task #1~4)**:
  - ✅ #1 큐브 검출: `/tmp/detect_cubes.py` HSV(S<55,105<V<215)+면적/aspect 필터로 top 4/4 정확. 그릇/깃털/그리퍼 배제.
  - 🔄 #2 hand-eye 캘리브(진행중): `/compute_fk`(ros-mcp call_service)로 gripper_frame_link base XYZ OK(예: 현재 x0.189 y-0.024 z0.023). 여러 자세 (FK base XY ↔ top 픽셀) → homography 적합 예정. **계획: 거친 homography로 큐브 base XY → IK hover → wrist 카메라로 XY 미세보정(visual servo) → 하강·닫기·들기.**
  - ⏳ #3 단일 큐브 IK grasp 검증 / ⏳ #4 녹화+LeRobot v3.0 변환(so_follower,6-dim,images.{top,wrist,front} 480×640 fps30).

### 알려진 이슈(이번 세션 발견)
- **fresh rclpy 프로세스가 데모 노드 데이터 수신 0**: publisher 디스커버리는 됨(`get_publisher_count`=1)인데 메시지 0개(QoS best_effort로도). ros-mcp(rosbridge)는 정상. FastDDS+mirrored late-joiner 데이터 전송 문제로 추정. **현재 모든 그래프 상호작용 ros-mcp 경유 우회.** recorder(#4)도 fresh 노드라 같은 문제 — 해결 필요(데모 launch 내부 recorder 추가 / rosbag2 / ros-mcp 구독 저장 중 택1).

### 시뮬 오라클 구상(별도, 아직 미진행)
- `pick_cube_state_machine.py`(cube_desk, `SimToReal-SO101-PickCube-v0`) 5축 grasp 실패 = **"타겟엔 IK로 가는데 grasp 자체 실패(손가락이 큐브 못 감쌈/관통/미끄러짐)"**(유저 확인). Franka 7축은 됨. `So101LulaIK`·`grasp_tilt_deg`·`rot_weight_grasp` 등 정교한 시도 누적(`docs/TROUBLESHOOTING` "SO-101 5DOF grasp 불안정").
- **아이디어(유저)**: Claude가 시뮬에서 **직접 interactive 제어**(시뮬은 큐브 pose ground-truth → 인식·캘리브 불필요 = 실기기 블로커 없음)로 grasp 성립 후 오라클 코드화. **매크로 액션 단위**(이동→렌더 캡처→조정 반복).

### 변경 파일(uncommitted, 커밋 안 함)
- 카메라/RMW: `ros2_ws/setup/{env.sh,cyclonedds_localhost.xml,wsl_ros2_sysctl.conf*,wslconfig.example*,99-so101.rules*,06_setup_host_devices.sh*,02_install_ros2_packages.sh}`, `src/so101_bringup/{scripts/cv2_camera_publisher.py*,launch/cameras_cv2.launch.py*,CMakeLists.txt,config/cameras/*.yaml}`
- 실기기 제어: `config/ros2_control/follower_controllers.yaml`(50Hz), `feetech_ros2_driver/.../serial_port.hpp`(250ms)
- 문서: `docs/{PATH_D_ROS2_WSL_MOVEIT.md,TROUBLESHOOTING.md}`, CONTEXT.md (*=신규)

---

## 작업 인계 (2026-06-10 — joint_fk 복원 완료: 1큐브 grasp 해결, 4큐브 full-DR 1.5/4, scatter reach 제한+far_base_first)

- **성과**: 위 DiffIK grip 한계(다음 인계) 결론대로 **joint_fk 복원** 실행. 커밋 `62303d9`(env `SimToReal-SO101-PickCube-v0`, joint-space `SlewLimitedJointPositionAction`)의 `pick_cube_state_machine.py` 를 `git checkout` 으로 복원, 현재 env_cfg(3cam·bowl 수정 반영본)와 런타임 호환 OK. **1큐브 DR-off grasp 1/1 성공**(DiffIK 18회 0/1 대비) — `--controller_mode joint_fk` 가 IK 가 못 만드는 강 tilt 자세를 random-FK 로 직접 탐색해 grip 성립. 원래 목표("grasp 불안정 WIP 해결") 달성.
- **추가 작업(사용자 지시 3건)**: ① env `_CUBE_SCATTER_X_RANGE`=[1.66,2.04]·`_CUBE_SCATTER_Y_RANGE`=[-0.46,-0.345] 로 **scatter 를 SO-101 reach 안쪽으로 제한**(reach 매핑 근거) ② SM `--object_order far_base_first` 추가(robot base 에서 먼 큐브 먼저, 그릇 빈 상태에서 까다로운 큐브 선처리) ③ sweep 에 spawn 위치 로깅 추가(reach 검증).
- **reach 매핑(1큐브 full-scatter 12 ep)**: 실패가 **base 거리와 무관**(먼 0.30m 성공, 가까운 0.11m 실패) — "닿을 수 없는 스폰"이 주 원인 아님. x 극단·너무 가까운 위치만 약간 실패 편향. 즉 진짜 원인은 **joint_fk random-FK 의 marginal grasp(단일 ~67%)**.
- **4큐브 full-DR sweep(far_base_first, 8 ep)**: **평균 1.5/4, all-4 0/8**. scatter reach 제한·far_base_first 로도 개선 안 됨 → CONTEXT 의 알려진 blocker(평균 1.5/4) 재현. 주 원인 = grasp marginal + 4큐브 상호작용(나중 큐브 approach 가 기존 큐브/그릇 침). **즉 1큐브 grasp 는 해결, 4큐브 신뢰 expert 는 미해결 blocker.**
- **문서/커밋**: `docs/TROUBLESHOOTING.md` "SO-101 5DOF grasp" 항목에 후속1(DiffIK 18회)·후속2(joint_fk) 추가. DiffIK 진단 코드는 `diffik_grasp_diag.patch`(commit `12265e1` 대비) 보존. 이 작업 `feat/so101-diffik-grasp` → main 병합.
- **남은 일(다음 세션)**: 4큐브 신뢰성을 올리려면 ① random-FK 스코어러에 tilt/grasp 품질 점수화(단일 67%↑ — CONTEXT 가 >3회 미해결 경고) 또는 ② 4큐브 transport/place 경유점 고도화(상호작용 완화)로 2→3/4 시도. 아니면 **1큐브(또는 fixed 4큐브) joint_fk expert 로 3cam LeRobot v3 dataset 수집** 후 IL/RL 진행(grasp 해결됐으므로).

---

## 작업 인계 (2026-06-09 후속 — DiffIK grasp 18회 진단: grip 근본 한계 확정 → joint_fk 복귀 결정)

- **결론**: 이전 인계의 "grasp 미완"을 18회 headless 실행으로 진단. **갭 roll 정렬은 원인 아님**(misalign 1~6°, 기각). 진짜 원인을 순차 해결 — ① ee 도달: position-only + 단계별 arm stiffness(descend 120)로 3.2cm→**0.4cm** ② 수평 밀림: 밀림이 gripper-local **X축**(jaw 회전 호 방향) 성분임을 분해로 확인 → `_lateral_offset`(local X)로 4.4cm→**1.3cm** ③ z 튐: `grasp_z_gain` 1.2~1.5로 손가락을 큐브 측면 깊이로 → **해결** ④ close 비산: soft close PD(grasp_stiffness 25)+고정 hold (closed-loop close 는 ee가 큐브 쫓아가 18cm 비산 역효과 → 제거).
- **확정된 근본 한계(블로커)**: in-sim DiffIK 5DOF 는 **강 tilt(jaw를 큐브 측면으로) + ee 도달을 동시에 못 함**. position-only=수직(fingerdir z≈-0.9, jaw가 큐브 위 8cm 떠 손가락이 큐브에 안 닿음→안 들림), pose tilt=강tilt시 ee 멀어짐(tilt20→ee1.2cm·jaw위, tilt30→ee4.5cm, tilt35→ee6.6cm+자세붕괴, `--ik_lambda`↓는 DLS 불안정·자세붕괴). 즉 밀림·z·정렬은 다 잡았으나 **손가락이 큐브 측면에 닿질 못해 grip 불가**. 과거 `joint_fk`(in-sim FK 샘플링)는 IK 없이 강 tilt 자세를 직접 탐색해 4/4(20260605) 성공.
- **사용자 결정**: **joint_fk 복귀**. 복원 대상 = **`62303d9`**(2026-06-07, controller_mode 멀티 2687줄, env `SimToReal-SO101-PickCube-v0` North Star joint-space + SlewLimitedJointPositionAction). joint_fk+DR off 4/4 검증, full-DR ~1.5/4(분산 큼). `94780bd`(DiffIK 재작성)에서 joint_fk 제거됨.
- **현 DiffIK 스크립트 추가물(미커밋, grip 한계로 폐기 예정이나 진단·재현 가치)**: `pick_cube_state_machine.py`에 진단 로그(close trajectory·gapdir·cube_yaw·gap_misalign·gbody), `_set_arm_pd`(단계별 stiffness), `_lateral_offset`(local X), `_obj_yaw`, `_success` 추적, CLI `--descend_stiffness/--grasp_stiffness/--grasp_lateral/--grasp_z_gain`. **DiffIK best(grip 직전)**: `--ik_position_only --grasp_lateral 0.01 --grasp_z_gain 1.2 --gripper_close -0.15 --reach_tol 0.02` → 밀림 0.3cm·z 안정이나 손가락이 큐브 안 닿아 grip 실패.
- **남은 일**: ① `git show 62303d9:scripts/environments/pick_cube_state_machine.py` 복원 ② 현재 env_cfg(3cam·bowl충돌 수정 반영본)와 호환 확인(py_compile + 1큐브 smoke — 06-07 이후 env 변경 다수라 인터페이스 깨짐 가능) ③ joint_fk 1큐브→4큐브(DR off)→full-DR 검증 ④ `docs/TROUBLESHOOTING.md` "SO-101 5DOF grasp" 항목 갱신.
- **실행 환경(이번 세션 확립)**: 메인 `.venv`(isaacsim 설치)를 worktree 에서 재사용 — `PYTHONPATH=<worktree>/src` + 메인 `.venv/Scripts/python.exe` 직접 실행, headless, 매 실행 전 `taskkill //F //IM kit.exe`. 진행 로그는 `outputs/so101_sm_progress.txt`(headless 가 stdout 가로챔). PowerShell 툴은 이 환경에서 자동 background 화되니 Bash 우선.

---

## 작업 인계 (2026-06-09 — SO-101 SM: 외부 Lula → in-sim DifferentialIK 재작성 [grasp 튜닝 진행중])

- **목표(사용자)**: `pick_cube_state_machine.py`(SO-101 5DOF, grasp 불안정 WIP)를 Franka 버전처럼 동작.
  방침 = **외부 Lula 버리고 in-sim DifferentialIK 로 재작성**(사용자 승인).
- **진단(확정)**: Lula 는 err 0 수렴(솔버 문제 아님). 두 원인 — ① SO-101 env 가 순수 joint-space 라
  외부 Lula 를 별도 frame 에서 돌리고 런타임 `shift` 로 끼워맞춤 → 자세 의존이라 descend 중 무효 →
  손가락 0.05~0.1m 빗나감. ② 실행 경로가 position-only 라 자세 방치. Franka 는 in-sim DiffIK 라 ① 없음.
- **측정(`inspect_so101_gripper_frame.py`)**: body=`[base,shoulder,upper_arm,lower_arm,wrist,gripper,jaw]`
  (gripper_frame_link 는 USD collapse) → DiffIK `body_name="gripper"`, `body_offset=(-0.0079,-0.000218,-0.0981)`
  (URDF gripper_frame, USD gripper-local ≡ URDF frame 검증).
- **구현 완료**: 신규 `src/sim_to_real/tasks/pick_cube/pick_cube_so101_ik_env_cfg.py`(PickCubeSceneCfg 재사용
  + DifferentialIKAction arm 5joint command_type=pose dls λ0.1 + BinaryJointPositionAction gripper),
  task `SimToReal-SO101-PickCube-IK-v0`(pick_cube/__init__.py), `pick_cube_state_machine.py` 전면 재작성
  (Lula 삭제, Franka 패턴 + radial-yaw+tilt target). **기존 PickCubeEnvCfg(North Star 6dim)는 미변경**.
- **버그 4종 수정**: ① 로그 cp949 인코딩(em dash) silent crash → `open(...,encoding="utf-8")` ②
  `_down_quat_world` yaw 90° 오류(손가락이 큐브 반대로) → `-π/2` 보정 ③ headless viewer skip ④
  **AppLauncher 에 tuple 인자(view_eye) 전달 → Windows GUI `_prepare_ui` access violation** →
  `_LAUNCHER_KEYS` 필터링(docs/TROUBLESHOOTING "_prepare_ui access violation"). 전부 코드 반영됨.
- **검증 현황(단일 큐브, headless, DR off)**: 제어·위치·자세 **완성** — position-only + grasp_z_offset -0.02
  로 `ee 가 큐브에 2cm`(Lula 0.05~0.1m 대비 대폭↑), `fingerdir z=-0.97`(손가락 수직), `descend reached=True`,
  그리퍼 닫힘 `joint=-0.15` 확인. **grasp 미완** — 닫으면 큐브가 옆으로 밀림(close/open/깊이 sweep 무효).
  = 손가락 갭 평면이 큐브 중심을 안 지나는 정렬/roll 문제로 추정(close 시 큐브 -x 밀림).
- **GUI 진단(진행중)**: 이 Windows 박스는 GUI 모드 RTX/Hydra 크래시 2종 — ④(tuple 인자, 수정함) +
  기본 GUI experience(isaaclab.python.kit) rtx.scenedb(→ `--enable_cameras` 로 rendering experience 전환).
  둘 다 적용 후 GUI 로 **손가락 갭 방향** 확인 중. 갭이 좌우 아닌 정면-후방이면 grasp 시 roll 90° 보정 필요.
- **best 파라미터(grasp 직전까지)**: `--ik_position_only --gripper_velocity 1.0 --grasp_z_offset -0.02
  --reach_tol 0.025 --gripper_open 1.5 --gripper_close -0.15`. (pose 모드는 `--grasp_tilt_deg 35` 가 ee 2.8cm.)
- **남은 일**: GUI 로 갭 방향 확인 → roll 보정(또는 ee xy closed-loop) → grasp 성립 → 4큐브 → full-DR 4/4 →
  TROUBLESHOOTING "SO-101 5DOF grasp 불안정" 항목을 "in-sim DiffIK 로 제어 해결" 로 갱신.
- **함정**: Windows 는 **kit 프로세스 1개만**(동시 2개 → KVDB lock → 2번째 부팅 silent 종료, 실행 전
  `Stop-Process`). headless 는 carb 가 stdout 가로채니 진행 로그는 `outputs/so101_sm_progress.txt`(파일)에서 확인.
- **진단 스크립트(보존)**: `inspect_so101_gripper_frame.py`(body 측정), `ik_env_smoke.py`(IK env gym.make 격리).

---

## 작업 인계 (2026-06-09 — PATH E: cube_desk MoveIt2/cuMotion Pick&Place 브릿지 스캐폴딩)

- **목표(사용자)**: cube_desk 장면에서 MoveIt2 path planning(조사 결과 **cuMotion 우선 + OMPL/Pilz 폴백**)으로 SO-101 pick&place state machine 구축. 플랫폼 = Windows+WSL2 먼저, 안 되면 Linux 서버. 충돌 = 전체(그릇+타큐브 obstacle + 잡은 큐브 attach). 통합 = Isaac Sim ROS2 bridge(`topic_based_ros2_control`).
- **아키텍처**: Isaac Sim(Win, cube_desk scene.usd+SO-101) ↔ DDS(7400/7410/9387) ↔ WSL2 ROS2(ros2_control topic_based + move_group cuMotion + cumotion_action_server + moveit_py FSM). 문서 `docs/PATH_E_CUMOTION_PICKPLACE.md`(런북+M0~M4).
- **작성 완료(코드/설정, colcon 빌드+xacro+py_compile 검증됨)**:
  - WSL2: `so101_ros2_control.xacro`(hardware_type:=isaac, topic_based), `isaac_controllers.yaml`, `move_group.launch.py`(use_cumotion arg), `isaac_pick_place.launch.py`, `so101_arm.xrdf`(cuMotion sphere — **M2 정밀화 필요**), `moveit_py_config.yaml`(cumotion 세트), `so101_pick_place_orchestrator.py`+`.launch.py`(moveit_py FSM), `05_install_cumotion.sh`, `cyclonedds_bridge.xml`, CMakeLists.
  - Windows: `scripts/ros2/cube_desk_ros2_sim.py`(AppLauncher + OmniGraph ROS2 브릿지).
- **M0 결과(2026-06-09)**: WSL2 에 `ros-jazzy-joint-state-topic-hardware-interface`(Jazzy 의 topic_based, plugin `joint_state_topic_hardware_interface/JointStateTopicSystem`) 설치 + Isaac ROS apt repo(release-4.4) 등록. **cuMotion 은 WSL2 불가** — `ros-jazzy-isaac-ros-cumotion` 이 `cuda-toolkit-13-0`+`libnvvpi4`+`gxf-isaac-*` 풀스택 요구(CUDA 13 vs 프로젝트 12.8 핀 충돌). **사용자 결정: cuMotion 은 Linux 서버에서** → WSL2 는 OMPL/Pilz(`use_cumotion:=false`, 기본). 코드 cumotion opt-in 화 완료(moveit_py_config pipeline_names 에서 cumotion 제외, orchestrator `use_cumotion` 파라미터).
- **M1 시도 결과(2026-06-09, 블로커)**: Windows Isaac Sim 으로 `cube_desk_ros2_sim.py` 3회 실행 모두 크래시. (1) GUI `_prepare_ui` access violation → headless 전환, (2) headless kit 이 OmniGraph 미로드 → `import omni.graph` 실패 → enable_extension 추가, (3) `enable_extension(isaacsim.ros2.bridge/core.nodes)` 가 viewport→RTX Hydra 를 **다른 그래픽 인터페이스(D3D12→Vulkan)로 재init** → `rtx.scenedb` 크래시(`DriverShaderCacheManager ... different graphics interface`, Aftermath 0xbad00009). **scene/OmniGraph 코드는 실행 전** — 내 코드 무관, Windows Isaac Sim 그래픽 init 근본 문제. headless 전환·확장 최소화는 코드에 반영됨(`--gui` opt-in, 최소 ext). TROUBLESHOOTING 기록함.
- **부팅-시점 로드도 검증·실패(2026-06-09)**: `--kit_args "--enable isaacsim.ros2.bridge ..."` 로 부팅 시점 로드 시 Vulkan 단일이라 재init 은 사라지나, `ROS2 Bridge startup failed` + 부팅 중 `createHydraEngine`→`rtx.scenedb` access violation 여전. GUI/headless·런타임/부팅·최소ext 4가지 모두 RTX/Hydra 에서 크래시 → **이 Windows 박스에서 Isaac Sim ROS2 bridge 불가로 확정**. TROUBLESHOOTING 갱신.
- **Windows 검증 대안(동작함)**: RViz mock(OMPL) pick&place — `ros2 launch so101_bringup follower_moveit_demo.launch.py hardware_type:=mock use_rviz:=true` + `pick_place_orchestrator.launch.py mock_poses:=true` (또는 `ros2_ws/setup/run_mock_pickplace_demo.sh`). physics 없는 kinematic, FSM **4/4 planned** 검증. 큐브/그릇은 SO-101 도달영역(y=0, x 0.22~0.38)에 맞춘 mock 좌표(`MOCK_POSES_BASE`). orchestrator 에 `_relaxed_ik`(자세 sweep, 5-DOF 도달), mock 모드(world 충돌 생략, attach/detach 후 world 제거) 추가됨.
- **Isaac Sim 영상·물리 결과·정식 충돌회피·cuMotion**: 전부 **Linux 서버**(네이티브 Ubuntu) 필요. repo git 동기 후 ROS2 Jazzy+MoveIt+ws(+cuMotion) 재구축 → PATH E 실행. `docs/PATH_E_CUMOTION_PICKPLACE.md`.
- **나머지 검증(플랫폼 확정 후)**: M1 브릿지 smoke → M2 RViz OMPL → M3 FSM 풀 사이클 → M4 튜닝. cuMotion 은 Linux 서버에서만(`docs/PATH_E §cuMotion`).
- **blind 작성 → 런타임 확인 지점**: OmniGraph 노드 타입명(Isaac Sim 5.1 `isaacsim.*`), base_link prim 경로, topic_based plugin 명, cumotion launch arg 이름(`cumotion_planner.*` vs `cumotion_action_server.*`), moveit_py collision/attach 메서드명, cumotion planner_id. `docs/PATH_E_…` §알려진 검증 포인트 참조.
- **5-DOF 주의**: SO-101 은 임의 6-DOF pose 도달 불가 → grasp 자세는 top-down tilt(orchestrator `GRASP_RPY`/`GRASP_TILT_RAD`) + pick_ik approximate. 기존 `pick_cube_state_machine.py`(in-process weighted-DLS)는 그대로 유지(폴백/비교용).

---

## 작업 인계 (2026-06-10 — PATH E grasp 디버깅: 진단 반전 + 위치정합 해결 ✅, 남은 건 접촉 물리(사용자 방향 결정 보류))

직전 인계의 "gripper close 간헐 실패" 블로커를 라이브 디버깅하다 **진단이 뒤집혔다**. close 는 실패가
아니라 **성공**(grip 매번 -0.16 완전닫힘)이고, 진짜 문제는 **grasp 접촉**(패드가 2.5cm 큐브를 못 뭄)이었다.
근본원인 = `_grasp_point` 가 link **원점(pivot)** 을 써서 실제 패드와 7~8cm 어긋난 것. 이를 **실측 패드 기준**
으로 재정의해 **위치 정합을 해결**(패드가 큐브 양옆을 정확히 감쌈)했으나, 그래도 grip 이 큐브를 못 물고
끝까지 닫힌다(접촉 물리 = PATH C 가 강tilt-scoop 으로만 풀었던 11사이클 난제). **다음 방향은 사용자 결정 보류 중.**

> ⚠ **인수인계 필수 3가지** (먼저 읽기):
> 1. **`run_sm.sh` 는 워크트리에 없다.** 내가 만든 임시 SM 단발 실행 헬퍼 — 위치 = 호스트
>    `/DISK1/so101-sim2real/work/ros2_build/run_sm.sh` = 컨테이너 `/build/run_sm.sh`. launch 띄우고
>    `RESULT:` 대기 후 스택 자동 정리(orphan 방지). 로직은 전부 `pick_place_sm.py` 에 있음. (원하면 `scripts/sim/` 로 옮겨 git 포함.)
> 2. **코드 변경 2파일 미커밋**(working tree): `ros2_ws/src/so101_cumotion_pick_place/so101_cumotion_pick_place/pick_place_sm.py`
>    + `.../config/pick_place_params.yaml`. 아래 "변경 내역" 참조. (symlink-install 이라 colcon 재빌드 없이 launch 반영.)
> 3. **headless bridge 가 살아있다**(호스트 PID 278427, `run_cube_desk_ros_bridge.sh --num_cubes 1`). 컨테이너 `so101_ros` up.
>    **GUI bridge 는 디버깅에 쓰지 말 것**(아래 ⚠ 컨트롤러 race). 디버깅은 headless bridge 로.

### 🎯 목표
PATH E(cuMotion+ROS) cube_desk pick-and-place **State Machine 을 동작**시키기. (close 디버깅에서 출발했으나
진단이 grasp 접촉으로 이동.) 사용자 방침: **"강tilt든 top-down이든 무관, pick-place 만 잘 되면 됨."**

### ✅ 완료 / 확정한 것
1. **진단 반전(4회 baseline b1~b4)**: close 명령(-0.16)이 매번 bridge 도달 + grip 매번 -0.16 **완전닫힘**.
   → close 는 실패가 아님. 2.5cm 큐브가 패드 사이에 있었다면 -0.16 까지 못 닫힘(중간 stall). 즉 **패드가
   큐브를 빗나가 헛닫힘**. CONTEXT 의 "grip 1.5 미닫힘" 프레이밍은 틀림.
2. **근본원인(패드 기하)**: 기존 `_grasp_point` = `JAW_LINK`·`gripper_link` **link 원점(pivot/모터 위치)** 중점.
   mesh AABB 측정(`.venv/bin/python3` STL 파싱)으로 확인 — moving_jaw 는 JAW_LINK 프레임에서 **-y 로 ~8.2cm
   뻗은 손가락**, gripper_frame_link(TCP)은 gripper_link z≈-0.098 = **고정 finger tip** 근처. → 원점 중점은
   실제 패드와 **7~8cm 어긋나** 큐브 위 2~4cm 헛집음(g5 등 gpt z=0.094 vs cube 0.049).
3. **위치 정합 해결**: `_grasp_point` = **(고정 finger tip=TCP) + (moving jaw tip=`JAW_TIP_LOCAL`) 중점** 으로
   재정의. tilt 제약 해제(사용자 방침) 후 **g15~17 = FK 3/3 성공, 실행된 grasp 중심이 큐브에서 0.7~1.2cm,
   패드가 큐브 양옆을 감쌈**(g17: jaw_tip y=-0.110·fix_tip y=-0.130 사이에 cube y=-0.125, z≈0.045).
4. **FK/IK/실행 정상 확인(selftest)**: 사용자 요청으로 "좌표 지정→EE 도달 측정" 루틴 추가. 4좌표 모두
   **실행오차(plan↔exec) 7~9mm**, FK샘플오차 14~18mm(gate 0.04 탓). → **FK/IK·실행은 멀쩡.** 앞서 본
   13cm·8cm 괴리는 FK/IK 가 아니라 ⬇ 컨트롤러 race 탓이었음.
5. **⚠ 컨트롤러 spawn race 원인 규명**: GUI(렌더링) bridge 로 시스템 부하↑ 시, launch 의 spawner 들이
   `load_controller`/`configure` 서비스 타임아웃으로 연쇄 실패(`Failed loading`/`already loaded↔no controller`).
   → 팔이 계획대로 안 움직여 grasp 중심 13cm 괴리. **headless bridge(가벼움)에선 0 실패.** (b1~b4 정상,
   g6~g8 GUI bridge 라 실패.)

### 🔴 남은 블로커 — grasp 접촉 물리
위치 정합(패드가 큐브 양옆)했는데도 **grip 이 -0.16 까지 완전히 닫히고 큐브 안 들림**(g15~17). 큐브는 안 움직임
(=안 튕김). 즉 **2.5cm 큐브가 이 그리퍼의 닫힘 aperture 한계 근처**라 패드 표면이 큐브를 물지 못하고 닫힘.
PATH C(in-process)는 같은 그리퍼·큐브로 4/4 성공했는데, 비결이 **"강tilt 로 moving jaw 를 큐브 옆/아래로
퍼올리듯(scoop)"**(memory `path-e-cumotion-ros-status`/in-process SM 6차). 단순 top-down/center 정렬로는 안 뭄.

### ⏭ 진행 예정 — 사용자 결정 보류 중 (AskUserQuestion 했으나 "추가 질문 후 결정" 으로 보류)
사용자가 "큐브에 일정 거리 이상 가까이 접근 못 함" 을 관찰 → 구조 도식 설명 요청 → 이 문서 작성. 결정 대기.
선택지(제시함):
- **A. Sim grasp-assist(부착)** — close+큐브근접 시 bridge 에서 fixed joint 부착, open 시 해제. 위치 정합이
  정확해 물리적 정당. pick-place 안정 완주. (사용자 1차엔 "물리" 택했으나 접촉 벽 확인 후 "어떻게 집든 무관" 으로 완화.)
- **B. 강tilt-scoop 물리 재현(PATH C 방식)** — tilt 다시 강하게 + FK 점수를 moving jaw 가 큐브 아래로 가도록.
  5-DOF ROS 에선 FK manifold 얇아 신뢰성 낮음(~15%).
- **C. 큐브 크기 상향**(2.5→3.0cm 등, aperture 에 맞춤) — North Star 계약(2.5cm) 영향 확인 필요.
- (추가 물리 레버: 느린 close / 마찰↑ / GRIPPER_CLOSED 덜 닫기 — 단 grip 이 끝까지 닫힘=접촉 자체가 없어 효과 제한적 추정.)

### 📊 결과 데이터 (대표, headless bridge, `/build/sm_run_<tag>.log`)
| tag | 의미 | FK grasp | 실행 grasp중심 vs cube | grip | 결과 |
|---|---|---|---|---|---|
| b1~b4 | baseline(원점 중점) | — | 중심 2~4cm 위(헛집음) | -0.16 완전닫힘 | 0/1 |
| g15~17 | 패드중점+tilt해제 | 3/3 d 0.7~1.1cm | **0.7~1.2cm(정합 OK)** | -0.16 완전닫힘 | 0/1(접촉 실패) |
| st1 | selftest(FK/IK검증) | — | 실행오차 7~9mm | — | FK/IK 정상 |
- g13 진단: FK 실패 시 위치상 최근접 config 가 **tilt 1°(top-down)·near_d 1cm** → 강tilt 필터가 reachable
  config 를 버리고 있었음(그래서 tilt 제약 해제함).

### 🔧 변경 내역 (미커밋, 2파일)
- **`pick_place_sm.py`**:
  - `JAW_TIP_LOCAL=(0.0,-0.065,0.019)` 상수(moving jaw 패드 접점, JAW_LINK frame. mesh AABB 기반 추정 — 미세조정 여지).
  - `_jaw_tip`/`_fix_tip` 헬퍼 + `_grasp_point`(두 tip 중점) / `_grasp_axis_vert`(두 tip 벡터) 재정의.
  - `_diag_grasp` 로그 확장: `jaw_tip`·`fix_tip`·`gpt`(grasp중심)·cube.
  - `_fk_sample_goal`: gate 를 stage 별 분리(grasp=fk_pos_gate, approach/transport=wide), **실패 시 near_d/tilt/axis
    진단 로그** 추가, near-sample 추적.
  - `_move_to`: 계획된 TCP/grasp중심을 `self._last_goal_tcp` 등에 저장(selftest 비교용), grasp/approach 는 goal_rs 도 CLOSED 기준.
  - `selftest()` 메서드 + main 에 `selftest` 파라미터 분기.
- **`pick_place_params.yaml`**: `grasp_z_offset` 0.012→**0.0**(중심을 큐브 중심에), `fk_pos_gate` 0.015→**0.025**,
  `grasp_tilt_min` 45→**0.0**(top-down 허용), `grasp_axis_vert_max` 0.4→**0.6**, `selftest: false`(검증 시 true).

### 🔑 진단 도구 (코드에 남김)
- `DIAG grasp` 로그(jaw_tip/fix_tip/gpt/cube/grip/grip_cmd/axis_vert).
- `FK[grasp] 실패` 시 near_d/tilt/axis 로그(reach 문제 vs 필터 문제 판별).
- **selftest 모드**: yaml `selftest: true` → pick 대신 4좌표로 TCP 도달 측정(FK샘플오차/실행오차/전체오차 분리). 검증 후 false 로.

### 🖥 환경 상태(세션 종료 시점)
- headless bridge **살아있음**(호스트 PID 278427). 컨테이너 `so101_ros` **up**(24h). GPU idle.
- 미커밋 2파일(위). 마지막 커밋 `487504b`.
- 호스트 `isaacsim-mcp` 별개 서비스 — 건드리지 말 것.

### ▶ 콜드 스타트(재현, headless 권장)
1. **bridge**(호스트, 워크트리): `scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1` → `[bridge] ready` +
   `gripper grip force: dof[5] kps=80.0` 대기. (이미 278427 떠 있으면 생략.)
2. **SM 1회**(헬퍼): `docker exec so101_ros bash /build/run_sm.sh <tag> 1` → 끝에 `DIAG (tag)` 출력 + 스택 자동 정리.
   - 코드 수정 시: symlink-install 이라 colcon 불요(파이썬). 깨끗하게 하려면 `colcon build --symlink-install
     --base-paths /workspace/ros2_ws/src --packages-select so101_cumotion_pick_place` 1회.
   - 로그: 컨테이너 `/build/sm_run_<tag>.log` = 호스트 `/DISK1/so101-sim2real/work/ros2_build/sm_run_<tag>.log`.
3. **GUI 로 보려면**(컨트롤러 race 주의): bridge 를 `DISPLAY=:1 scripts/sim/run_cube_desk_ros_bridge.sh ...`(로컬
   `:1` 데스크톱) 또는 `--livestream 2`(WebRTC). **단 GUI bridge 부하로 컨트롤러 spawn 이 깨질 수 있음** —
   GUI 로 보려면 spawner 견고화(아래 주의) 선행 필요.

### ⚠ 주의 (함정)
- **GUI bridge 컨트롤러 race(미해결)**: 사용자가 GUI 로 watch 하려면 `follower_split.launch.py` 의 spawner 들에
  `--controller-manager-timeout` 추가 / 순차화 필요. 지금은 headless 로만 안정.
- **셸 `rtk` 래퍼 함정**: 호스트 `ps aux | grep ...` 가 dotfiles 의 `rtk` 함수에 먹혀 깨짐 → **`pgrep -af` 사용**.
  `pkill -f`/`kill $(pgrep)` 는 자기 셸 매칭=자살 → **PID 직접 kill**. orphan 정리도 컨테이너서 PID 직접.
- bridge↔컨테이너 RMW/transport(fastrtps/UDPv4) 일치 필수. GPU 서버 공유(학습 경합 주의).
- `JAW_TIP_LOCAL` 은 mesh AABB 기반 **추정값** — 접촉이 끝내 안 되면 실제 패드 접점으로 미세조정 여지(닫힘 시 두 tip 간격 ~2cm 측정됨 vs 큐브 2.5cm).

---

## 작업 인계 (2026-06-09 — PATH E grasp 물리: planning self-collision 해결 ✅ + grasp 정합 7메커니즘 확립, 남은 건 gripper close 간헐 실패)

직전 인계의 "grasp 물리(jaw 가 큐브 못 감쌈)" 블로커를 11사이클 라이브 디버깅으로 깊이 규명했다.
in-process SM 의 강tilt 전략을 ROS SM 에 포팅하며 **planning self-collision 블로커를 완전 해결**하고
grasp 정합을 단계적으로 끌어올렸으나(큐브를 손가락 사이에 정렬·xy 정합·axis 수평 달성), 마지막
**gripper close 가 grasp 자세 의존적으로 간헐 실패**해 `RESULT 1/1` 미달. **커밋 `853beb8`**(코드 3파일).

- **변경 파일(커밋 `853beb8`)**: `pick_place_sm.py`(대폭) · `pick_place_params.yaml` · `scripts/sim/run_cube_desk_ros_bridge.py`(gripper grip force).
- **✅ 해결한 것**:
  1. **planning self-collision(근본 블로커)**: `set_to_random_positions` 가 joint bounds 만 지키고 self-collision
     무시 → FK 샘플이 shoulder↔lower_arm colliding config 선택 → OMPL "goal tree 샘플 실패"/cuMotion
     `INVALID_INITIAL_CSPACE_POSITION`. **`PlanningScene.is_state_colliding` 로 FK 샘플 배제**(`_colliding`,
     best 후보·정밀화 모두 체크) → approach/grasp/lift **전부 OMPL OK**, self-collision 에러 0건.
  2. **grasp 자세 정합 7메커니즘**(`_fk_sample_goal`): ① grasp 단계는 TCP 대신 **두 손가락 중점**(JAW_LINK·
     FIX_LINK FK 중점, `_grasp_point`)을 큐브에 맞춤 ② tilt **하한 필터**(grasp_tilt_min 45°, 수직 top-down
     배제) ③ 통과 config 중 **거리 최소**(xy 정합, gate 0.015 시 d 0.004 달성) ④ **CLOSED 기준 중점**(gripper
     닫으면 jaw 회전해 중점 z↑3cm → OPEN 정합은 close 후 큐브 위로 뜸) ⑤ **grasp axis 수평**(grasp_axis_vert_max
     0.4, `_grasp_axis_vert` — 두 손가락 벌어짐이 수직이면 큐브 위아래 눌러 실패) ⑥ transport 는 **gate 분리**
     (0.04, 좁은 gate 면 lift FK 도달 실패) ⑦ bridge **gripper grip force**(dof[5] kps 17.8→80).
  - 결과: 큐브를 손가락 사이(y)에 정렬 + xy 0.8cm + axis_vert 0.11(수평) 자세 도달 — **grasp 직전까지 완성**.
- **🔴 남은 블로커(gripper close 간헐 실패)**: close 명령(-0.16)은 항상 bridge 도달(DIAG `grip_cmd=-0.16` 확인)
  하나, grasp 자세에 따라 articulation grip 이 **-0.16(닫힘) ↔ 1.5+(미닫힘) 간헐**. jaw 가 책상/큐브 윗면에
  박히거나 self-collision 반력으로 닫힘 토크(URDF effort 10 한계)를 못 이기는 것으로 추정. grip force 80 으로도
  못 이김. **다음 후보**: ① bridge `enabled_self_collisions` 가 jaw close 경로 막는지(끄거나 gripper 링크쌍만
  disable) ② grasp z 를 큐브 윗면(jaw 0.069 vs 윗면 ~0.064) 아래로 더 내리되 도달성 확보(scatter 축소로 큐브를
  도달 쉬운 위치=base_link x[1.60,1.76]·y[-0.465,-0.365] 권장 — CONTEXT 기존 "사용자 결정 미적용") ③ close 동역학
  (느린 close + dwell↑) ④ closed-loop grasp servoing.
- **🔑 진단 도구(코드에 남김)**: `DIAG grasp` 로그(grip·grip_cmd·axis_vert·ee·jaw·fix·gpt·cube) + store 노드의
  `/isaac_joint_commands` 구독(`last_grip_cmd`). grasp 디버깅 재현 시 그대로 활용.
- **핵심 파라미터(yaml)**: grasp_tilt_deg 60 / grasp_tilt_min 45 / grasp_axis_vert_max 0.4 / grasp_z_offset 0.012
  / fk_samples 40000 / fk_pos_gate 0.015 / fk_pos_gate_transport 0.04 / gripper grip force(bridge) 80.
- **실행 환경(세션 종료 시점)**: 컨테이너 `so101_ros` 살아있음(sleep infinity, GPU 미사용). bridge·launch 전부
  정리됨(GPU idle, gnome 340MiB 뿐). 호스트 `isaacsim-mcp` 별개 서비스 — 건드리지 말 것. 이미지 재빌드 불요
  (두 overlay baked, 소스 미변경분 symlink 반영). **코드 변경은 symlink-install 이라 colcon 증분만 하면 launch 반영.**
- **콜드 스타트(재현, 3터미널)**: ① bridge(host, 워크트리): `scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1`
  → `[bridge] ready` + `gripper grip force: dof[5] kps=80.0` 대기. ② (코드 수정 시) `docker exec so101_ros bash -c
  'source /opt/ros/jazzy/setup.bash && cd /build && colcon build --symlink-install --base-paths /workspace/ros2_ws/src
  --packages-select so101_cumotion_pick_place'`. ③ launch: `docker exec so101_ros bash -c 'source /opt/ros/jazzy/setup.bash
  && source /opt/tbc_overlay/install/setup.bash && source /opt/cumotion_overlay/install/setup.bash && source /build/install/setup.bash
  && export SO101_REPO=/workspace && ros2 launch so101_cumotion_pick_place pick_place.launch.py num_cubes:=1 2>&1 | tee /build/sm_run.log'`.
  로그 grep: `FK\[grasp\]|DIAG grasp|RESULT|grasp 실패|도달 config 없음`.
- **주의**: launch 재실행은 통째로 — 이전 move_group/cumotion/ros2_control orphan 을 정리해야(누적되면 controller_manager
  경합으로 controller 스폰 실패). 정리는 PID 직접 kill 또는 **브래킷 트릭**(`grep -E '[r]os2_control_node|[m]ove_group|...'`)
  으로 — `pkill -f`/`kill $(pgrep -f ...)` 는 docker exec 자기 셸 cmdline 을 매칭해 자살(exit 137). bridge 종료도 PID 직접.

---

## 작업 인계 (2026-06-09 — PATH E 5-DOF planning 블로커 해결: joint-goal FK 샘플링 ✅, 남은 건 grasp 물리)

직전 인계의 "5-DOF grasp IK" 블로커를 **근본 진단 후 해결**했다. 다중 세션 블로커의 진짜 원인은
cuMotion/IK 가 아니라 **MoveIt/cuMotion 의 goal 샘플러가 5-DOF 를 못 다루는 것**이었고, SM 을
**joint-space goal(FK 샘플링)** 로 바꿔 approach→grasp→lift 가 전부 planning OK 가 됐다. 남은 건
grasp 물리(grip 이 큐브를 못 쥠) — in-process SM 의 known-hard 영역.

- **🔑 근본 진단(서버 konan147, GPU idle, 라이브)**:
  - OMPL `Unable to sample valid states for goal tree` + cuMotion `INVERSE_KINEMATICS_FAILURE` 이
    **orientation 제약/완화/position-only 전부에서** 발생. `/compute_ik` 는 5-DOF 에서 exact 6-DOF
    pose 라 거의 -31(NO_IK_SOLUTION). **`/compute_fk` 랜덤 FK 샘플링**(`scripts/sim/probe_ik.py` 신규)
    으로 워크스페이스 매핑: 큐브 위치는 도달 가능하나 achievable orientation manifold 가 thin(예 grasp
    z≈0.05 에서 tilt 4~52°) → planner 의 "랜덤 orientation+IK" 샘플러가 thin manifold 를 못 찾음.
  - 즉 **pose/position goal 자체가 5-DOF 에 비가능**(과거 in-process SM 이 joint_fk 쓴 이유와 동일).
  - 워크스페이스 기하: **world↔base_link = Z 180° 회전**(`base_link=(1.84-wx, -0.565-wy, wz-0.6749)`).
    Cube2/4 authored 위치는 base_link 음의 x(팔 reach 뒤, shoulder_pan ±110° 밖)=도달 불가.
- **✅ 해결(커밋 `4734e5b`)**: `pick_place_sm.py` `_move_to` 를 **joint-goal** 로 전환 —
  `RobotState.set_to_random_positions()` in-process FK 샘플링으로 target(x,y,z) 에 down-ish(tool z
  tilt≤max) tip 을 두는 manipulator config 를 찾고 `set_from_ik` 정밀화 → `set_goal_state(robot_state=)`.
  planner(cuMotion/OMPL)는 joint→joint collision-free 만 푼다. `_fk_sample_goal`/`_tool_tilt` 신규,
  pose-goal 기계(`_pose`/`_yaw_to`/`_grasp_constraints`) 제거. param: `fk_samples`(15000)/`fk_pos_gate`(0.04)
  추가, pose-goal tolerance 제거, height LOW band(approach_height 0.12→0.06, lift 0.12→0.07, transport
  0.15→0.12, grasp_tilt 60→30). yaml top-key `pick_place_object_store:`→`/**:`(store 노드명≠launch
  타겟 노드명 문제로 param 미로딩이던 것 해결). kinematics.yaml pick_ik timeout 0.2→1.0/attempts 10→50.
- **검증(서버, /build/sm_run5.log)**: `OMPL OK → (0.140,-0.125,0.109/0.044/0.119)` approach/grasp/lift 전부 plan+exec.
  단 `grasp 실패(안 들림)`.
- **🔴 남은 블로커(grasp 물리, planning 무관)**: ① grasp config 의 moving_jaw 가 큐브 위(z 0.089 vs
  cube 0.049) — TCP(gripper_frame_link)만 큐브 근처, jaw 가 안 감쌈 ② 그리퍼 완전히 안 닫힘(0.086 vs
  -0.16 목표). **다음 후보**: FK 샘플링 목적함수에 jaw-z 점수(`_finger_min_z` 식, 강tilt 로 moving_jaw 를
  큐브 옆/아래로) / 그리퍼 close dwell·force / 위치 gate(fk_pos_gate) 강화 / set_from_ik 정밀도. in-process
  SM 이 ~1.4/4 로 marginal 했던 동일 난제 — grasp 물리는 별도 phase.
- **사용자 결정(미적용)**: "scatter 범위 축소" — `pick_cube_env_cfg.py::_CUBE_SCATTER_X/Y_RANGE` 를
  도달 가능 world 영역(x[1.60,1.76], y[-0.465,-0.365] 권장)으로. Cube2/4 authored 위치도 reachable 로
  재배치(author_pick_cube_scene.py) 시 4큐브 가능. planning 해결과 별개라 grasp 물리 후 적용 권장.
- **실행 환경 현재 상태(세션 종료 시점)**: 이미지 `so101-cumotion:jazzy` 재빌드 완료(두 overlay baked
  — `/tmp/*_ws` 수동 빌드 불요). 컨테이너 **`so101_ros` 살아있음**(`sleep infinity`, GPU 미사용). bridge·
  move_group·cumotion **전부 정리됨**(GPU idle 375MiB). 호스트의 `isaacsim-mcp`(PID 2315xxx)는 별개
  사전 서비스 — 건드리지 말 것. ROS 패키지 소스 미변경분은 `/build/install` symlink 반영(재빌드 불요).
- **콜드 스타트(재현, 3터미널)**:
  1) bridge(host, 워크트리서): `scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1` → `[bridge] ready` 대기.
  2) (코드 수정 시만) 컨테이너서 colcon 증분: `docker exec so101_ros bash -c 'source /opt/ros/jazzy/setup.bash &&
     cd /build && colcon build --symlink-install --base-paths /workspace/ros2_ws/src --packages-select <pkg>'`.
     install=`/build/install`(worktree `/workspace/ros2_ws/install` 아님).
  3) 전체 launch: `docker exec so101_ros bash -c 'source /opt/ros/jazzy/setup.bash && source /opt/tbc_overlay/install/setup.bash &&
     source /opt/cumotion_overlay/install/setup.bash && source /build/install/setup.bash && export SO101_REPO=/workspace &&
     ros2 launch so101_cumotion_pick_place pick_place.launch.py 2>&1 | tee /build/sm_run.log'`. 결과 grep:
     `OK →|FK-sample 도달 config 없음|grasp 실패|RESULT:`.
  - 컨테이너 docker run 명령(없을 때): `docker run -d --name so101_ros --network host --ipc host --gpus all
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 -e SO101_REPO=/workspace
    -v <worktree>:/workspace -v /DISK1/so101-sim2real/work/ros2_build:/build so101-cumotion:jazzy sleep infinity`.
- **주의**: SM 노드 재실행은 launch 통째로(이전 launch 의 move_group/cumotion orphan 은 `kill -9 <pid>` 로 정리
  후 재launch — 안 그러면 노드 중복). probe_ik.py(reachability 진단)는 move_group 떠 있을 때만(`/compute_fk`
  서비스). bridge 종료는 PID 직접 kill(`pkill -f` 자기매칭 금지). bridge↔컨테이너 RMW/transport(fastrtps/UDPv4) 일치 필수.

---

## 작업 인계 (2026-06-09 — PATH E cuMotion pose-goal 전환 + c-space 6vs5 블로커 플러그인 패치로 해결)

§5 통합(아래 인계) 후 사용자 지시로 **cuMotion pose-goal 방식** 진행. set_from_ik(pick_ik) IK 실패를
우회하려 SM `_move_to` 를 cuMotion task-space goal 직접으로 바꿨고, 그 과정에서 **단일 핵심 블로커
(c-space 6 vs 5)를 진단·패치로 해결**했다. 단 최종 pick-and-place 는 grasp IK 미도달로 미완(이월).

- **커밋**: `a17d2aa`(pose-goal 전환+진단) → `71bcbdc`(플러그인 패치+relaxed orientation).
- **🔑 핵심 블로커 진단·해결 (웹+MCP 조사로 확증)**:
  - 증상: cuMotion 이 `INVALID_INITIAL_CSPACE_POSITION` (`cspace_position[6]` vs `robot[5]`)으로 모든 계획 실패.
  - 원인: MoveIt `request.start_state` 는 전체 로봇 6관절(arm5+gripper)을 담는데 cuMotion cspace 는
    tool_frame(gripper_frame_link) chain 위 **5축뿐**(gripper 분기 관절은 cspace 에 넣어도 무시 — 3회 확인).
    cuMotion MoveIt 플러그인(`CumotionMoveGroupClient::updateGoal`)이 request 를 **무필터 전달** → 6 전달.
  - 조사: **upstream [issue #10](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/issues/10) 와 동일 버그**(open, NVIDIA fix 없음). XRDF 에 비-cspace joint 선언 메커니즘 없음(문서 확인). isaac-sim-mcp: cuMotion core 가 cspace 길이를 controlled joint 수로 엄격 검증.
  - **해결(검증)**: `docker/patches/cumotion_moveit_filter_start_state.patch` — updateGoal 이 start_state.joint_state 를
    planning group(manipulator,5) active 관절로 필터링. Dockerfile 이 clone+apply+build → `/opt/cumotion_overlay`.
    → clean run 에서 INVALID_INITIAL_CSPACE **0건**, cspace_position 이 5개로 필터링, cuMotion 이 task-space IK 수행까지 진입.
- **변경 파일**: `pick_place_sm.py`(_move_to → cuMotion pose-goal + `_grasp_constraints` relaxed orientation:
  position tight + tool z 회전 자유로 5-DOF redundant DOF 해방. `orient_mode` 진단 토글) · `pick_place_params.yaml`
  (pos_tol/ori_tol/yaw_free_tol) · `moveit_py_cumotion.yaml`(wait_for_initial_state_timeout 10→30, startup race) ·
  `Dockerfile.cumotion_ros`(cumotion patch overlay) · `so101.xrdf`(cspace 제약 경위 주석).
- **🔴 남은 블로커(grasp IK, 5-DOF)**: cspace count 는 해결됐으나 cuMotion 이 grasp 접근 자세 미도달 —
  orientation 제약 시 `INVERSE_KINEMATICS_FAILURE`, position-only 시 start config(5개) "invalid"(관절한계/충돌 추정).
  즉 5-DOF 가 down+tilt 접근을 그 위치에서 못 풀거나 cuMotion 이 MoveIt orientation tolerance 를 무시(exact 취급).
  **다음 후보**: ① position-only 가 왜 start invalid 인지(관절한계/충돌/start-state 읽기) 규명 ② cuMotion native
  task-space tolerance(plugin 의 task-space target orientation tolerance) ③ grasp 자세/approach_height 재설계
  ④ 도달 가능 워크스페이스 확인(FK 스윕). grasp 작동(open/close)은 5-DOF 무관(gripper_controller action 별개).
- **그리퍼 5-DOF 확인(사용자 질문)**: cuMotion 은 5축(팔)만 계획, 그리퍼는 `gripper_controller`(ParallelGripperCommand
  action)로 별개 제어 → 패치도 start_state 를 manipulator group(gripper 제외)으로 필터링하므로 정합. tool z 회전
  자유는 2.5cm 큐브엔 무해(대칭). 길쭉한 물체면 jaw 정렬 문제.
- **실행 환경**: 영속 컨테이너 `so101_ros`(현재 살아있음, 정리 필요) + bridge(host). 컨테이너 내 overlay:
  `/tmp/tbc_ws`(topic_based), `/tmp/cu_ws`(패치된 cumotion_moveit). **이미지 재빌드 시 Dockerfile 이 두 overlay 영구화** →
  다음 세션엔 `/tmp/*_ws` 수동 빌드 불요. source 순서: /opt/ros/jazzy → /build/install → tbc → cumotion overlay.

---

## 작업 인계 (2026-06-09 — PATH E §5 4~6: ROS 스택 end-to-end 통합 ✅, 남은 건 5-DOF grasp IK)

bridge 블로커 해소(아래 인계) 후 사용자 지시로 §5 4~6 진행. **`pick_place.launch.py` 전체 스택(bridge +
controllers + move_group + cuMotion + SM)을 서버에서 처음으로 end-to-end 기동**시켰다. SM 이 큐브 포즈를
받아 pick 시도까지 진행하며, 마지막으로 **grasp 접근 pose 의 5-DOF IK 실패**에서 멈춘다(§6 known-hard).

- **커밋**: `15450a8`(bridge B안) → `ddbe664`(ROS 스택 통합 4대 수정).
- **해소한 bringup 4대 함정**(전부 소스/Dockerfile 반영, TROUBLESHOOTING 기록):
  1. **controller_manager SIGSEGV** — Isaac ROS repo 의 `topic_based_ros2_control 99.99.1` 이 ROS 메인 repo
     `hardware_interface 4.44.0` 과 ABI 불일치(`get_lifecycle_id` vtable). 메인 repo 엔 0.3.0 source 만 → PickNik
     소스에서 재빌드해 overlay 설치(`Dockerfile.cumotion_ros` `/opt/tbc_overlay`, bashrc 에서 source).
  2. **`pick_ik/PickIkPlugin` 미설치** → SM set_from_ik SIGSEGV. `Dockerfile` apt 에 `ros-jazzy-pick-ik` 추가.
  3. **launch tuple 에러** — `isaac_ros_cumotion_planning.yaml` 의 `request_adapters: []`(빈 리스트)가 launch_ros 에서
     빈 튜플로 변환돼 Node 파라미터 타입검증 실패. 해당 줄 제거.
  4. **SM `PoseStamped` NameError** — `pick_place_sm.py` import 누락 보강.
  + **use_sim_time**: `follower_split.launch.py` 에 인자 추가(CM/rsp 주입), `pick_place.launch.py` 가 `true` 전달
    (bridge `/clock` 정합).
- **검증된 사실(서버 konan147, GPU idle)**:
  - cuMotion `CumotionPlanner` 플러그인 로드 + URDF/XRDF 로 로봇 로드 성공(tool=gripper_frame_link, base=base_link)
    → **XRDF 유효 = §5 #1 사실상 통과**(Python curobo 미설치라 `gen_so101_xrdf.py` 는 미실행, C++ planner 가 대체 검증).
  - 컨트롤러 3종(joint_state_broadcaster/arm_trajectory_controller/gripper_controller) active, `/follower/joint_states` 흐름.
  - SM: 포즈 수신→`pick order [0]`→`pick-and-place cube[0]`→**`IK 실패 (0.140,-0.125,0.169) tilts=[60,45,75,0]`**.
  - 종료 시 exit -11 은 MoveItPy teardown(무해).
- **🔴 남은 블로커(§5 #6 성공 = grasp)**: 5-DOF SO-101 이 grasp 접근 pose(down+tilt)에 도달 못 함. set_from_ik(pick_ik)
  가 모든 tilt 후보에서 실패. 이전 in-process Lula/ikpy SM 도 못 풀던 동일 난제(§6). **다음 방향 후보**:
  ① SM `_move_to` 를 set_from_ik(joint goal) 대신 **cuMotion pose-goal 직접 사용**(MoveItPy `set_goal_state(pose_stamped_msg, pose_link)`)
     — cuMotion 이 5-DOF 여유/limit 내에서 모션을 직접 풀게. ② grasp 자세/tilt/approach_height 재설계(pick_place_params.yaml).
     단 5-DOF orientation 도달성은 planner 선택과 무관한 본질적 제약이라 grasp 전략 자체 재검토 필요.
- **실행 환경(재현)**: 영속 컨테이너로 단계 디버깅함 —
  `docker run -d --name so101_ros --network host --ipc host --gpus all -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 -e SO101_REPO=/workspace -v <worktree>:/workspace
  -v /DISK1/so101-sim2real/work/ros2_build:/build so101-cumotion:jazzy sleep infinity` → 안에서
  `source /opt/ros/jazzy/setup.bash; source /build/install/setup.bash; source /tmp/tbc_ws/install/setup.bash`(현 컨테이너는
  topic_based 를 /tmp/tbc_ws 에 빌드해둠 — **이미지 재빌드 시 Dockerfile 의 /opt/tbc_overlay 로 영구화**) 후 launch.
  colcon 빌드(symlink-install)는 `/DISK1/so101-sim2real/work/ros2_build`(/build), 소스는 worktree(/workspace). ROS 패키지
  소스 미변경이라 재빌드 불요(symlink 반영). **다음 세션엔 이미지 재빌드(Dockerfile 반영) 후 `/tmp/tbc_ws` overlay 불요.**
- **주의**: 컨테이너 정리 `docker rm -f so101_ros`. bridge 종료는 PID kill. 동시에 bridge(host)+컨테이너 RMW/transport(fastrtps/UDPv4) 일치 필수.

---

## 작업 인계 (2026-06-09 — PATH E B안 완료: bridge device -1 블로커 해소, joint_states publish ✅)

직전 인계의 마지막 블로커(Isaac Lab InteractiveScene bridge 의 `device 0 vs -1`)를 **B안으로 해소**. bridge 를 순수 `isaacsim.core` 로 재작성해 `/isaac_joint_states`·`/clock`·`/tf` 가 모두 정상 publish 됨(서버 konan147, GPU idle, `--num_cubes 1` 실측). PATH_E §5 검증 **1~3 통과**.

- **변경 파일**:
  - `scripts/sim/run_cube_desk_ros_bridge.py` — **전면 재작성(B안)**. `SimulationContext+InteractiveScene` → `World(physics_dt=1/120, rendering_dt=1/30)` CPU 백엔드 + `SingleArticulation`. `add_reference_to_stage` 로 `cube_desk/scene.usd`(+`so101_follower.usd`) 직접 로드. base 고정 = `isaaclab.sim.schemas.modify_articulation_root_properties(fix_root_link=True)` 재사용(→ articulation root 가 `/World/Robot` 로 올라옴). 로봇 pose = `PickCubeEnvCfg._ROBOT_POS/_ROBOT_ROT`. drive gain = `set_gains(kps=17.8, kds=0.6)`(leisaac). OmniGraph 는 `OnPlaybackTick`(수동 evaluate_sync 제거), 루프 = `world.step(render=True)`. TF parent = `/World/Robot/base/base_link` Xform 신설(USD base 링크명이 `base` 라 동명 Xform 으로 frame "base_link" 생성).
  - `scripts/sim/run_cube_desk_ros_bridge.sh` — **(신규)** 런처. `LD_LIBRARY_PATH`(번들 ROS 2 lib)·`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`·`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`·`OMNI_KIT_ACCEPT_EULA` export 후 uv run.
  - docs: TROUBLESHOOTING.md(device -1 → 해결 + 신규 2건), PATH_E_CUMOTION_ROS.md(§검증 상태·실행·파일맵·§5 표), 이 인계.
- **🔑 해결한 환경 함정 3건(전부 TROUBLESHOOTING 기록)**:
  1. **device -1**: Isaac Lab GPU fabric view ↔ OmniGraph 노드 view 충돌. 순수 `World`(CPU 백엔드)는 OmniGraph 가 sim view 단독 소유 → 양쪽 device 일치, 충돌 소멸. 단일 로봇+소수 큐브라 CPU 물리로 충분.
  2. **`librmw_implementation.so` 로드 실패**(`libament_index_cpp.so cannot open`): 호스트 ROS 2 없음 → isaacsim 번들 jazzy/lib 의 `$ORIGIN` RPATH 부재. 동적 링커가 프로세스 시작 시 LD_LIBRARY_PATH 읽으므로 **launch 전 export 필수**(python os.environ 무효). 래퍼가 처리.
  3. **host↔container DDS discovery 실패**: bridge=일반유저·컨테이너=root 의 cross-UID `/dev/shm` fastrtps SHM lock 충돌. **양쪽 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`**(UDP 강제)로 우회. isaacsim 번들은 fastrtps 만 → 컨테이너도 fastrtps(env.sh 의 cyclonedds 는 WSL2 PATH D 전용, PATH E 에서 source 금지).
- **검증 명령(재현)**: 터미널1 `scripts/sim/run_cube_desk_ros_bridge.sh --num_cubes 1` → `[bridge] ready`, 로그에 `expected device` 0건. 터미널2 `docker run --rm --network host --ipc host -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 so101-cumotion:jazzy bash -c 'source /opt/ros/jazzy/setup.bash && ros2 topic echo /isaac_joint_states --once'` → 6관절 값. `/tf` 도 `base_link→Cube1/Bowl`(좌표 검산 일치: world offset 을 `_ROBOT_ROT` z-180° 회전한 값).
- **➡ 다음(PATH_E §5 4~6, 컨테이너 ROS 스택 필요)**: ① `gen_so101_xrdf.py` IK>90% + sphere 튜닝(§5 1) ② `pick_place.launch.py use_rviz:=true`(컨테이너, fastrtps/UDPv4 env) → RViz 수동 plan/execute, 팔 동기 ③ 그리퍼 action ④ 단일→4큐브 SM 성공률. cuMotion plugin 클래스명/노드 파라미터는 설치 버전 문서로 재확인. ROS 스택 launch 시 bridge 와 RMW/transport 일치 필수.
- **주의**: bridge 종료는 PID 직접 kill(`pkill -f` 금지=자기 매칭). 래퍼는 `/World/Robot` 단일 env(env-NS 없음) prim 레이아웃. drive gain 미적용 시 cuMotion 위치명령 미추종.

---

## 작업 인계 (2026-06-09 — PATH E 라이브 검증: 빌드·인프라 OK, bridge joint_states 가 마지막 블로커)

서버 konan147 에서 빌드~실행 검증 진행. **인프라·빌드는 전부 성공**, Isaac Sim bridge 의 joint_states publish 한 지점이 미해결로 남아 **순수 isaacsim 재작성(B안)으로 이월**.

- **성공(재현됨)**:
  - Docker 이미지 `so101-cumotion:jazzy`(18.8GB) 빌드 — `docker/Dockerfile.cumotion_ros`. Isaac ROS 4.4 apt repo(release-4.4 noble main external-main) + CUDA/Jetson-x86/ROS2 repo. 설치: ros-jazzy-moveit(+moveit-py)·ros2-controllers·joint-trajectory-controller·parallel-gripper-controller·**topic-based-ros2-control**·tf-transformations·**isaac-ros-cumotion(+moveit)**·curobo는 C++ `libcumotion_planner_lib.so`(libcudart.so.13 링크, 의존성 전부 해소, Python curobo 불요).
  - **cuMotion MoveIt plugin 클래스 = `isaac_ros_cumotion_moveit/CumotionPlanner`**(추정 CumotionPlannerManager 아님 — yaml 수정함). cuMotion action server 파라미터 = `cumotion_action_server.{urdf_file_path,xrdf_file_path,read_esdf_world,...}`(상류 isaac_ros_cumotion.launch.py include, move_group_cumotion.launch.py 반영).
  - colcon 빌드 5패키지 OK(`/DISK1/so101-sim2real/work/ros2_build`). xacro `hardware_type:=isaac`→TopicBasedSystem OK. MoveItPy/ParallelGripperCommand/tf_transformations import OK. 양쪽 launch `--show-args` OK.
  - cuMotion pipeline yaml 은 **so101_moveit_config/config/isaac_ros_cumotion_planning.yaml** 에 둬야 MoveItConfigsBuilder 가 찾음(so101_cumotion_moveit_config 에서 옮김).
  - Isaac Sim uv 동기(`uv sync --group isaac`, 캐시 `/DISK1/so101-sim2real/cache/uv`). bridge **부팅·OmniGraph 생성·토픽 4개 광고·host↔container DDS discovery 전부 OK**(FastRTPS, --network host).
- **해결한 환경 함정**(TROUBLESHOOTING 기록):
  - Isaac Sim 헤드리스에서 OmniGraph 생성 실패("Unable to create prim")→**rendering experience 필요**(`args.enable_cameras=True`→isaaclab.python.headless.rendering.kit). 기본 headless.kit 는 OmniGraph USD authoring strip.
  - inotify watch 고갈(errno=28)→`sudo sysctl -w fs.inotify.max_user_instances=1024 fs.inotify.max_user_watches=1048576`(세션 적용함, 영구는 /etc/sysctl.d).
  - `sleep`/loop 가 셸 dotfiles(`rtk`)에 깨짐→백그라운드 명령에 sleep 금지(Monitor/직접 Read 사용). `pkill -f run_cube_desk_ros_bridge` 는 자기 명령줄 매칭→자살하므로 금지.
  - bridge launch GPU OOM 은 동시 학습(policy-server 34GB) 탓 — 학습 중단하면 48GB 확보.
- **🔴 미해결 블로커**: bridge 루프에서 OmniGraph JointState/ArticulationController 노드가 `omni.physx.tensors: expected device 0, received device -1` → joint_states 값 미publish(토픽은 광고됨). **Isaac Lab InteractiveScene + OmniGraph 물리노드의 device 바인딩 불일치**. 시도해 실패: graph를 reset 전/후 생성, OnPlaybackTick→OnTick+`evaluate_sync` 강제평가, `PickCubeEnvCfg().sim`(GPU 파이프라인) 사용 — 전부 device -1 지속.
- **➡ 다음(B안, 권장)**: bridge 를 **순수 `isaacsim.core`(World+Articulation)** 로 재작성 — Isaac Lab InteractiveScene 대신 cube_desk `scene.usd`+SO-101 `so101_follower.usd` 직접 stage 로드. NVIDIA 공식 ROS2 예제(create_ros_action_graph)와 동일 경로라 device 0 정합됨. 나머지(컨테이너 ros 스택·SM·cuMotion)는 그대로 재사용. 그 후 PATH_E §5 검증 4~6.
- **현재 bridge 스크립트 상태**: Isaac Lab 버전(rendering exp + OnTick+evaluate_sync + env_cfg.sim + graph-before-reset). joint_states 직전까지 동작. B안에서 scene 로드부만 교체.

---

## 작업 인계 (2026-06-09 — PATH E: cuMotion + ROS 2 cube_desk pick-and-place scaffold)

- **목표(사용자)**: Isaac Sim MCP·NVIDIA Isaac ROS pick-and-place 튜토리얼을 조사해 **cuMotion + ROS** 로 SO-101 cube_desk SM 작성 + 재현 문서.
- **배경/근거**: 기존 in-process Lula IK SM(`pick_cube_state_machine.py`)은 Lula↔USD 정합 잔차(~0.1m)로 grasp 미완. cuMotion 은 articulation frame 에서 직접 collision-free 계획 → 좌표 정합 문제 구조적 제거. `isaac_ros_cumotion` 이 ROS 2 Jazzy/x86_64 공식 지원이라 기존 `ros2_ws`(so101_moveit_config) 재사용 가능.
- **확정 결정(사용자)**: ① 플랫폼 = Linux 서버 네이티브(Jazzy+cuMotion+Isaac Sim 한 머신) ② 인지 = 시뮬 ground-truth 포즈 ③ SM = 커스텀 ROS 2 Python 노드(MoveItPy+cuMotion).
- **구현 완료(코드/설정/문서, 빌드·실행 검증 미실시 — GPU+ROS 서버 필요)**:
  - `scripts/sim/run_cube_desk_ros_bridge.py` — Isaac Sim standalone(InteractiveScene + PickCubeSceneCfg 재사용) + `isaacsim.ros2.bridge` OmniGraph(JointState pub/sub, Clock, ArticulationController) + 물체 포즈 publish(`/cube_poses`,`/bowl_pose`, **base_link frame**).
  - `assets/robots/so101.xrdf`(collision sphere 근사·튜닝 필요) + `scripts/sim/gen_so101_xrdf.py`(curobo 검증 하니스).
  - `ros2_ws/src/so101_cumotion_moveit_config/`(cuMotion planner plugin yaml·moveit_py_cumotion.yaml·move_group_cumotion.launch.py) — so101_moveit_config SRDF/kinematics 재사용.
  - `ros2_ws/src/so101_cumotion_pick_place/`(pick_place_sm.py = MoveItPy manipulator+cuMotion 8단계, params, pick_place.launch.py).
  - `so101_ros2_control.xacro` `hardware_type:=isaac`(TopicBasedSystem, state `/isaac_joint_states`/cmd `/isaac_joint_commands`) + `follower_isaac_controllers.yaml`(100Hz).
  - 문서 `docs/PATH_E_CUMOTION_ROS.md`(셋업/실행/검증 6단계/튜닝), `AGENTS.md` 경로표·스크립트표 갱신.
- **핵심 설계 결정**:
  - 토픽 분리: `/isaac_joint_states`(bridge↔TopicBasedSystem) ≠ `/follower/joint_states`(broadcaster→MoveIt) — 피드백 루프 방지.
  - 프레임: SRDF virtual_joint(world→base_link)=identity → 모든 포즈 base_link frame 통일(bridge 가 robot base 빼서 publish). robot-base-offset static TF **불필요**.
  - 5DOF grasp: `set_from_ik` 에 tilt 후보(60°→±15°→0°) 순차 시도 → 첫 성공 자세로 cuMotion joint-space 계획. cuMotion 실패 시 OMPL fallback.
  - MoveItPy 파라미터 매칭: SM Node `name="pick_place_moveit"` 로 moveit config+named set 전달, store 노드는 params yaml(node key) 로.
- **검증(로컬)**: 작성 Python 6종 `py_compile` OK, yaml/xml/xacro 파싱 OK. (ROS/Isaac/curobo import·빌드·실행은 서버에서 미실시.)
- **남은 일(서버 GPU+ROS 필요)**: ① `gen_so101_xrdf.py` IK>90% 확인 + sphere 튜닝(cuMotion Robot Description Editor) ② `colcon build` 5패키지 ③ bridge↔topic 토픽 흐름 확인(PATH_E §5) ④ RViz dry-run plan/execute ⑤ 단일→4큐브 SM 성공률 ⑥ `isaac_ros_cumotion` plugin 클래스명/노드 파라미터를 설치 버전 문서로 재확인(`move_group_cumotion.launch.py` 주석). 성공 시 TROUBLESHOOTING 기록.
- **주의**: XRDF sphere·cuMotion plugin 파라미터명은 버전 의존이라 서버에서 실측 보정 필요. `SO101_REPO` env 로 XRDF/URDF 절대경로 해결.

---

## 작업 인계 (2026-06-08 — 3cam 전환: top/wrist → top/wrist/front)

- **목표(사용자)**: 레포지토리 전체를 3개 카메라(top/wrist/front) 기준으로 통일.
- **변경 완료**: 환경변수(`.env`, `.env.example`), SmolVLA rename_map(`env/smolvla.env` → camera3=front), GR00T 설명(`env/groot.env`), sim env cfg 2종(`pick_pen_env_cfg.py`, `pick_cube_env_cfg.py` — `_FRONT_CAMERA_*` 상수 + make/add 함수 확장), 스크립트 5종(`rollout_to_lerobot.py`, `pick_cube_state_machine.py`, `camera_shape_smoke.py`, `segmentation_overlay_preview.py`, `validate_lerobot_schema.py`), teleop 스크립트(`teleop_se3_agent.py` — CLI 인자·viewport·tuner·주입 함수), 문서 4종(`PATH_A_NATIVE.md`, `TROUBLESHOOTING.md`, `AGENTS.md`, `CONTEXT.md`/`TASKS.md`).
- **front 카메라 시뮬 기본 좌표(미튜닝)**: PickPen `pos=(2.14, 0.65, 1.10)`, `target=(2.14, -0.15, 0.80)`, `focal=18.0`; PickCube `pos=(1.87, 0.65, 1.10)` (동일 target/focal). `--tune_cameras`로 GUI에서 조정 후 `_FRONT_CAMERA_*` 상수에 붙여넣기.
- **검증 필요(GPU/GUI)**:  
  1. `docker compose ... run --rm lerobot find-cameras` → 3개 카메라 탐지 확인  
  2. `uv run ... teleop_se3_agent.py --task SimToReal-SO101-PickCube-v0 --enable_cameras --tune_cameras` → Front viewport 확인 + 좌표 튜닝  
  3. front 카메라로 데이터 수집 후 `validate_lerobot_schema.py` 통과 확인
- **주의**: 기존 2cam 데이터셋은 `observation.images.front` 채널이 없어 `validate_lerobot_schema.py`가 WARNING만 출력(에러 아님). SmolVLA fine-tune 시 3cam 데이터셋 기준으로 학습해야 `camera3` 키가 모델에 등록된다.

---

## 작업 인계 (2026-06-08 — 그릇 충돌 형상 수정: convexDecomposition → 명시적 패널)

- **목표(사용자)**: 큐브를 그릇에 place 하면 바닥까지 안 가라앉고 높게 쌓여 넘칠 듯 담기는 현상 진단·수정. 실제 그릇처럼 곡면 타고 바닥으로 정착해야.
- **진단(확정)**: `Bowl.usda` Wall mesh 의 `physics:approximation = "convexDecomposition"` 이 오목한 그릇 안쪽 캐비티를 convex hull 로 **메워** 충돌 바닥을 실제(z≈0.012)보다 높임 → 큐브가 가짜 바닥에 얹힘. Wall 이 두께 0 열린 회전면(504 pts)이라 분해가 더 부정확. 유입 = commit `e0eeae3`(기존 480 명시 패널 → 단일 mesh+convexDecomp 경량화). 문서 근거: PhysX/Omni Physics — convexDecomposition 은 hollow 형상을 채워 동적 컨테이너엔 SDF/명시적 충돌 권장.
- **수정(완료, 사용자 승인 = 명시적 패널 안)**: `scripts/environments/author_pick_cube_scene.py`
  - Wall mesh → **시각 전용**(`_bowl_wall_mesh(..., collision=False)`: collision API·approximation 제거).
  - 신규 `_bowl_collision_walls()` — 자오선 경사각(alpha)만큼 기울인 invisible box 패널 링(6 band×24 = **144개**)으로 안쪽 충돌 구성. 연직 패널이면 band 경계 ledge 에 큐브가 얹히므로 tilt 필수. Bottom cylinder(r=0.0325) 유지.
  - 배치 = `_oriented_box()` 가 `pxr.Gf` 로 `Scale·Ry(alpha)·Rz(phi)·Translate` 합성한 baked `matrix4d xformOp:transform`(Euler op-order 모호성 회피). 프로파일 상수(`BOWL_R_BOTTOM/R_TOP/Z_BASE/DEPTH/LATS/LONS`, `BOWL_COLLISION_LATS/LONS/THICKNESS`)로 시각·충돌 공유.
- **검증(완료, 자산 수준)**: `uv run scripts/environments/author_pick_cube_scene.py` 재생성 성공. Wall=collision 없음·approximation 없음, CollisionWalls invisible·패널 144개·collision 있음, 패널 z `0.0168~0.0652`·r `0.0478~0.0757`(그릇 안쪽 연속), scene 합성 OK(196 prim). 첫 패널 매트릭스 손계산 일치.
- **남은 일(물리 거동 = GPU/GUI 필요, 사용자 실행 권장)**: `uv run scripts/environments/pick_cube_state_machine.py --num_envs 1 --active_objects 4` (또는 teleop)로 큐브가 바닥까지 미끄러져 정착하는지 육안 확인. 통과 시 → ① `docs/TROUBLESHOOTING.md` 항목 추가(현상→원인→해결) ② `docs/GRASP_PHYSICS.md` 그릇 물성 절 갱신 ③ place_height/stack 증분 하향 재튜닝 여지.
- **주의**: `pick_cube_state_machine.py` 의 diff 23줄은 **이번 작업과 무관한 기존 WIP**(건드리지 않음).

---

## 작업 인계 (2026-06-06 — teleop-grade pickcube FSM 고도화: 신뢰성 grasp 돌파)

- **브랜치**: `feat/sm-teleop-grade-pickcube` (main `02bdc71` 기준). 편집은 Windows, GPU 실행은 konan147 에 scp 동기화 후 `/DISK1/so101-sim2real/venvs/isaac` 로 실행.
- **목표(사용자)**: 20260605 4-cube 2cam 성공본(`pick_cube_state_machine_explicit_fsm_stackheight_4cube_2cam_20260605`, 9000f/300s)을 teleop 수준으로 고도화. 신뢰성 8~9/10(4큐브 전부), 시간 ≤120s, full-DR(teleop 동일 조건), LeRobot v3 기록.
- **확인된 사실**: 20260605 이 최신·최고. 단 `object_radius_scale=0.0`·`container_angle_scale=0.0` → **DR 완전 비활성 fixed-spawn(curriculum hacking)**. 이제 SM 기본을 full-DR(1.0)로.
- **구현 완료**(`pick_cube_state_machine.py`, 커밋 `c93bef9`·`934b4c9`): full-DR 기본화 / `--object_order raster`(top카메라 좌상단→우하단) 기본 / `DESK_TOP_Z=0.76`(펜값) 제거→live bowl z·`CUBE_DESK_TOP_Z=0.705` / env `SO101_JOINT_TARGET_MAX_VELOCITY` 1.0→5.0 / SM 팔 slew 5 rad/s / 대기 최소화+`_phase` early-exit / 직선 transport(높이 0.18→0.12, idle_home 성공경로 제거) / 수직 retry / Isaac-frame DLS 에 top-down 접근축 task(`_ik_action(topdown,ori_weight)`) + ikpy orientation_mode.
- **🔑 핵심 돌파(grasp)**:
  1. **그리퍼를 5 rad/s 로 빠르게 닫으면 큐브를 못 쥠**. 천천히(`--max_gripper_step_delta 0.005`≈0.15 rad/s) 쥐어야 안착·마찰. 팔은 5 rad/s 유지(env 한계 5 내). 이게 grasp 반복 실패의 진짜 원인.
  2. **strict 수직 top-down pick 불가**(5-DOF reach: jaw 가 큐브 ~3cm 위 정지). → pick=자연 tilt, drop=palm-down(level). `--topdown_pick`(기본 off).
  3. **그리퍼=고정 finger+모터 jaw**(URDF: `gripper_link` 고정 + `moving_jaw_so101_v1_link` revolute).
  4. **🔑 grasp 성패 = 모터 jaw 가 큐브 높이까지 내려오는가 = tilt 강도**. finger AABB 측정(`_diagnostic_pose`):
     - 실패(centered, down≈0.81): 모터 jaw 바닥 z≈0.743 (큐브 윗면 0.741 위) → 닿지 않음. 고정 finger 가 책상(0.705)에 닿아 더 못 내려감.
     - 성공(20260605, down≈0.34 강한 tilt): 모터 jaw 바닥 z≈0.689 (큐브 바닥 0.717 아래) → 큐브를 감쌈. `/DISK1/so101-sim2real/outputs/pickcube_geom_success_20260606.json`.
     - 즉 **강한 tilt 가 모터 jaw 를 큐브 옆/아래로 내려보내 grip 성립**. ikpy 최소동작 IK 는 먼 큐브에서 tilt 가 약해 실패. joint_fk(random-FK)는 자연히 강한 tilt 를 찾음(20260605 4/4).
- **검증된 동작 config**: ikpy + tilt + slow gripper 0.005 + grasp_pick_offset 0.005 + cycles 2 → **1-cube full-DR PASS**(sm14). joint_fk + DR-off + slow slew → 4/4(20260605 재현).
- **🔑 2차 진단(2026-06-06 후속)**: joint_fk full-DR 도 **2/4**(`pickcube_jfk_fulldr_4cube_20260606.json`). ikpy와 동률 → controller 문제 아님.
  - 근본 원인: **random-FK 스코어러가 `dist+continuity_weight*continuity`만 보고 tilt/grasp 품질을 점수화 안 함** → 목표 닿는 pose 중 운에 맡겨 tilt 들쭉날쭉(descend down_dot 0.35~0.84). 좋으면 성공, 약하면 실패.
  - 성공/실패 판별식: **모터 jaw 바닥 z 가 고정 finger 아래로 + 큐브 바닥까지** 내려가면 성공(cube1 성공 jaw 0.694<fix 0.713). 약tilt면 모터 jaw 가 fix 위·큐브 위에 남음(cube3 jaw 0.79>fix 0.75).
  - 2가지 실패: ① 약tilt 로 모터 jaw 가 큐브 중심(0.72~0.74)에 멈춤(cube2, **stochastic**·retry/cycle2 로 복구됨) ② 워크스페이스 가장자리 먼 큐브(cube3 x=2.06, scatter x∈[1.60,2.08] 끝)는 자세가 안 나와 jaw 0.79 갇힘(**deterministic**·retry 무효).
  - **깊은 descend(grasp_pick_offset -0.022) 는 비추**: unreachable 목표라 descend early-exit 안 걸려 full-steps 소진 → 시간↑(≤120s 역효과). tilt 를 z 과주입 대신 **자세로** 해결해야.
- **🔧 채택 수정(미검증, 코드 반영·scp 완료)**:
  - `_fk_solve_joint_target` 에 **`grasp_tilt_weight`** 점수항 추가(descend/grasp 단계만): `score += w*(max(0,jaw_z-floor)+max(0,jaw_z-fix_z))` → 모터 jaw 가 desk(0.705)까지+고정 finger 아래로 내려가는 강tilt pose 를 결정적으로 선택. `_finger_min_z()` 헬퍼(캐시된 link-frame 코너).
  - **`--num_episodes N`** sweep: 한 Isaac 세션에서 reset+SM N회(매 reset DR 재추첨, `torch.rand` default RNG advancing 확인) → all4/per-cube 성공률 집계. dataset 기록 비활성.
- **🔑 3차 — 정직한 baseline sweep(5 ep, full-DR, 검증 params, cycles2)**: `pickcube_sweep_baseline_20260606.json` → **all-4 = 0.0, 평균 1.4/4**, per-cube Cube1 0.4·Cube2 0.4·**Cube3 0.0**·Cube4 0.6. 8~9/10(per-cube ~95% 필요)과 격차 큼. 실패=① grip 안 잡힘(큐브 책상 제자리) ② 일부 충돌로 날아감(z=0.025 바닥 추락 등).
- **❌ 실패한 실험들(모두 baseline 이하)**:
  - tilt_weight 0.4 + offset -0.005: 1/4, 큐브 날아감(자세 thrashing).
  - 깊은 descend(offset -0.022): unreachable 목표 → early-exit 안 됨 → 시간↑·비추.
  - combo(tilt_weight 0.15 + grasp_arm_step 0.05 + continuity 0.04 + descend-only settle): seed7 0/4. **positioning 은 개선**(모터 jaw mvz 0.69~0.71 로 내려옴, grip 2개 유지) 됐으나 **bowl 이 z=0.794 로 변위**(tilted 자세의 transport/place 가 그릇 들이받음) → place 회귀. baseline sweep 은 bowl 0.71 정상.
- **결론**: 정직한 full-DR 4-cube 신뢰성은 grip 한계로 낮음(평균 1.4/4). tilt 점수화로 positioning 은 고쳤으나 후속(transport/place/bowl-knock) 회귀. **추가 튜닝(>3회) 으로 미해결 — 블로커 성격.**
- **추가 코드(기본 OFF, 무해)**: `--grasp_tilt_weight`(FK tilt 점수), `--grasp_arm_step_delta`(grasp 단계 팔 속도), `--num_episodes`(sweep), `_finger_min_z`, descend `require_arm_settled`. py_compile OK, scp 완료.
- **4차 — continuity(부분 효과)**: 0.7 reach sweep 에서 ep 가 **큐브를 x=3.58·z=0.02 로 테이블 밖 사출** → 5 rad/s 팔이 random-FK 의 global 자세 점프를 통과하며 큐브를 쳐냄. `--continuity_weight` 0.015→**0.05**(기본값化) 로 **flinging 정성적으로 크게 감소**(매끄러운 궤적). 이건 견고한 개선.
- **🔴🔴 5차 — 측정 함정 발견(중요)**: continuity 0.05 가 seed7 sweep 에서 평균 3.0/4(all-4 40%) 로 보였으나, **재현 안 됨**:
  - 신선한 단일 seed 11~16 = **1,1,1,1,1,0 (평균 0.83, all-4 0/6)**.
  - 같은 seed7 단일 ep0 를 여러 번 → 2/4 지만 **성공 큐브·step 수가 매번 다름**(Cube1,4 ↔ Cube1,2; 7684↔9853 steps). 같은 초기 레이아웃인데 결과가 다름.
  - **결론: GPU PhysX 가 non-deterministic + grasp 가 marginal → 같은 시나리오도 물리 노이즈만으로 1/4↔3/4 뒤집힘.** 5-ep sweep 은 분산이 커 신뢰 불가. carryover 는 아님(매 reset `start_inside=0` 확인).
  - **정직한 종합 평균 ≈ 1.5/4, all-4 ≈ 10%, 분산 큼.** continuity 의 실제 이득은 baseline(1.4) 대비 작음(노이즈에 묻힘). 과거 "3.0 돌파" 는 lucky sweep artifact.
- **(이전 가설 폐기)** "open-loop SM + sim non-determinism = 근본 블로커" 라 적었으나 **틀림** — 진짜 원인은 위 6차의 env 그리퍼 cap 이었다. determinism 도 정상(determ A==B). 아래 7차로 재측정 진행.
- **🎯 7차 — env 수정 후 재측정(진행 중)**: 그리퍼 cap 1.0 + 팔 2 rad/s + continuity 0.05 로 현재 SM 을 full-DR(seeds 7,11,12) 재측정 → `fixenv_fulldr_{7,11,12}.json`. (이전 ~1.5/4 는 망가진 env 결과라 무효.)
- **권고/다음(env 수정 후)**: ① full-DR 신뢰성 재측정 결과로 deliverable 확정 ② teleop(31s,2rad/s) 기준 효율화(phase floor 축소·재시도↓) 로 ≤60~120s ③ 4/4 seed v3 dataset 재기록(이전 fixedpos 기록은 망가진 env 라 1/4, 폐기) ④ docs/TROUBLESHOOTING 에 "env 그리퍼 cap 5.0 → grasp 실패" 항목 추가.
- **임시 진단 파일(konan147)**: `orig_sm_20260605.py`(원본 추출본), `orig_dropoff_s7.json`(cap5→0/4), `orig_dropoff_gripper1_s7.json`(grip1→4/4). 정리 대상.

---

## 작업 인계 (2026-06-09 — WSL2 실기기 카메라 0fps 해결 + RMW FastDDS 전환)

- **목표**: ros-mcp 로 SO-101 follower + 3캠(top/wrist/front) 실기기 동작. arm 은 즉시 성공(trajectory action), 카메라가 0 fps 였음.
- **근본 원인(3겹)**: ① gscam/v4l2_camera 가 usbipd-win 가상 V4L2 에서 동작 안 함(MMAP/MJPG 디코드) ② net.core.rmem_max 208KB ③ **★ `.wslconfig` networkingMode=mirrored 에서 CycloneDDS 가 sensor_msgs/Image cross-process 전달 실패**(String 은 됨, 크기 무관). 상세는 메모리 `project-wsl2-camera-fastdds`.
- **해결**: ① OpenCV(MJPG) 단일스레드 라운드로빈 publisher `ros2_ws/src/so101_bringup/scripts/cv2_camera_publisher.py` (USB-IP 동시 open/멀티스레드 read 불가) ② `ros2_ws/setup/wsl_ros2_sysctl.conf`→`/etc/sysctl.d/99-ros2-wsl.conf`(rmem 16MB) ③ **`ros2_ws/setup/env.sh` RMW CycloneDDS→FastDDS**(SHM 전송으로 우회). FastDDS 로 3캠 각 ~23fps cross-process 수신 검증.
- **토픽 네이밍 통일**: `/camera/{top,wrist,front}/image_raw` + `/camera/{top,wrist,front}/camera_info` (North Star observation.images 키와 일치). frame_id=`<name>_camera_optical_frame`.
- **검증 완료**: ros-mcp(rosbridge, FastDDS) 8노드 통합 graph, joint_states + 3캠 image 수신, rqt_graph/rqt_image_view 시각 확인.
- **변경 파일(uncommitted, main 작업트리)**: `ros2_ws/setup/env.sh`(RMW), `cyclonedds_localhost.xml`(대용량 튜닝 — FastDDS 전환으로 비활성이나 보존), `wsl_ros2_sysctl.conf`(신규), `src/so101_bringup/scripts/cv2_camera_publisher.py`(신규), `launch/cameras_cv2.launch.py`(신규), `CMakeLists.txt`(scripts install), `config/cameras/so101_{cameras,gs_cam,v4l2_cam}.yaml`(gscam/v4l2 잔재 — 비활성). **커밋 미진행**.
- **실행 방법**: 각 노드 `source ros2_ws/setup/env.sh` 후 — follower `ros2 launch so101_bringup follower.launch.py use_rviz:=false arm_controller:=trajectory_controller`, rosbridge `ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090`, 카메라 `ros2 launch so101_bringup cameras_cv2.launch.py`(또는 `ros2 run so101_bringup cv2_camera_publisher.py`). RMW 안 섞이게 모두 env.sh 필수.
- **주의/남은 일**: ① USB-IP 디바이스는 publisher 종료 후 FD 해제에 수 초 걸림(재기동 시 open 실패면 대기 후 재시도) ② gscam/v4l2 yaml 잔재 정리 또는 cv2 로 일원화 ③ MoveIt/cuMotion(PATH D/E)을 FastDDS 에서 회귀 확인 필요 ④ 카메라 보정(camera_info 현재 미보정 width/height 만).

---

## 작업 인계 (2026-06-06 — controller_mode 에 lula_ik(경로2)/ikpy(경로3) 백엔드 포팅)

- **브랜치**: `feat/cube-ik-backends` (main `d33f61d` 기준). 별도 worktree `cube-ik-port`.
- **배경**: 참고 가이드의 경로 2(Lula `LulaKinematicsSolver` 직접 IK)·경로 3(ikpy)을 추가. main 은 경로 1(RMPFlow, 4-cube blocked) + diff_ik + joint_fk 만 보유했었다.
- **구현**(`scripts/environments/pick_cube_state_machine.py`):
  - `--controller_mode` 에 `lula_ik`, `ikpy` 추가.
  - `SO101LulaIkJointTarget`(경로2) / `SO101IkpyJointTarget`(경로3) — 기존 `SO101RmpFlowJointTarget` 과 **동일한 `compute(target_w)->(q,plan)` 인터페이스**. `_make_cartesian_driver()` 팩토리로 dispatch → `_phase` 의 검증된 2nd-half Jacobian refine·slew 그대로 활용.
  - base pose·grasp offset 은 RmpFlow 가 쓰는 검증 상수(`RMPFLOW_BASE_POS/QUAT_USD`, `RMPFLOW_GRIPPER_FRAME_TARGET_OFFSET`) 재사용. Lula descriptor 도 `so101_robot_description.yaml` 재사용(신규 파일 없음).
  - lula_ik: target stepping(0.04m) + position-only(`target_orientation=None`) + 반환 배열 직접 사용. ikpy: base 상수로 world→base 변환 + seed clamp.
  - `pyproject.toml` isaac 그룹에 `ikpy>=3.4,<3.5`. 함정 4건 `docs/TROUBLESHOOTING.md`.
- **검증**: py_compile OK, ikpy URDF 파싱+IK 오프라인 err 0.0. in-sim lula_ik/ikpy 1-cube 스모크 = (sync 후 진행).
- **한계**: 5-DOF position-only IK 라 4-cube 신뢰성은 검증된 `joint_fk` direct FSM 이 우위. lula_ik/ikpy 는 단일 큐브·대안 백엔드 용도.
- **출처 메모**: 이 작업은 worktree `goofy-honking-pie`(옛 base c172d1f, 미커밋)에서 만든 ikpy/Lula 백엔드를, main 의 explicit-FSM/controller_mode 구조에 맞게 재구현한 것. 옛 worktree 의 `pick_cube_state_machine.py` diff·중복 Lula descriptor 는 폐기.

---

## 작업 인계 (2026-06-05 — FSM 전이 확인 + RMPFlow 4-cube blocked)

- **사용자 질문 답변**: 현재 `scripts/environments/pick_cube_state_machine.py`는 사용자가 제시한 전이 구조를 따른다. enum/result JSON 순서는 `IDLE → OPEN_GRIPPER → MOVE_TO_PRE_PICK → ORIENT_WRIST → DESCEND → GRASP → LIFT → MOVE_TO_PRE_PLACE → PLACE_DESCEND → RELEASE → MARK_DONE → ALL_DONE`이다. cube는 elongated object가 아니므로 `ORIENT_WRIST`는 `cube_has_no_elongated_axis_skip` trace만 남기고 `DESCEND`로 넘어간다.
- **추가 보정**: `GRASP` 상태가 실제로 "그리퍼 닫기 + 대기"가 되도록, slew-limited gripper command가 closed target에 도달하는 최소 step과 `--grasp_settle_steps`를 보장하게 했다. 기본값 기준 open 1.0 → closed 0.0은 200 step + 60 settle.
- **검증**:
  - `uv run python -m py_compile scripts/environments/pick_cube_state_machine.py` 통과.
  - 서버 direct `joint_fk` 4-cube 회귀 proof 통과: `/DISK1/so101-sim2real/outputs/pick_cube_jointfk_graspwait_regression_4cube_nocam_20260605.json` (`status=passed`, `final_inside.Cube1~4=true`).
  - RMPFlow 1-cube smoke는 이전에 pass했지만, 4-cube fixed-spawn은 3개 조합 모두 failed:
    - `/DISK1/so101-sim2real/outputs/pick_cube_rmpflow_refine_4cube_nocam_20260605.json` → `Cube1/4=true`, `Cube2/3=false`
    - `/DISK1/so101-sim2real/outputs/pick_cube_rmpflow_graspwait_4cube_nocam_20260605.json` → `Cube2=true`만 최종 inside
    - `/DISK1/so101-sim2real/outputs/pick_cube_rmpflow_hardfirst_cycles2_shortgrasp_4cube_nocam_20260605.json` → `Cube2/4=true`, `Cube1/3=false`
- **결정**: 자율 루프의 동일 task 3회 실패 기준으로 `TA.CUBE.RMPFLOW_CONTROLLER`는 `blocked` 처리. 4-cube direct FSM 2cam dataset은 이미 PASS이므로 RL/IL은 direct FSM expert 산출물을 기준으로 우회한다.
- **다음 actionable**: RMPFlow를 더 붙잡지 말고, direct FSM 2cam expert dataset/trajectory에서 BC warm-start 또는 phase-aware imitation/DAgger 쪽으로 RL 재진입.

---

## 작업 인계 (2026-06-05 — PickCube 명시 FSM V2 완료, 다음 RMPFlow)

- **사용자 질문 답변**: 현재 `scripts/environments/pick_cube_state_machine.py`는 사용자가 제시한 전이 구조를 명시적으로 따른다. `PickCubeFSMState`와 결과 JSON의 `fsm_state_sequence`는 `IDLE → OPEN_GRIPPER → MOVE_TO_PRE_PICK → ORIENT_WRIST → DESCEND → GRASP → LIFT → MOVE_TO_PRE_PLACE → PLACE_DESCEND → RELEASE → MARK_DONE → ALL_DONE` 순서다.
- **핵심 변경**:
  - `joint_fk` 직접 SM을 명시 FSM trace로 재구성하고, 각 object attempt마다 `OPEN_GRIPPER/MOVE_TO_PRE_PICK/.../MARK_DONE` 이벤트를 남긴다.
  - object 사이에는 `IDLE` home posture로 복귀해 bowl/기존 큐브를 치지 않게 했다.
  - 4개 큐브를 같은 bowl에 쌓을 때 마지막 큐브가 튕겨 나가던 문제를 줄이기 위해 `--stack_place_height_increment` 기반 place 높이 보정을 추가했다.
- **서버 검증 완료**:
  - no-video proof: `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_explicit_fsm_stackheight_4cube_nocam_20260605.json`
  - 2cam LeRobot v3 dataset: `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_explicit_fsm_stackheight_4cube_2cam_20260605`
  - 결과 JSON: `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_explicit_fsm_stackheight_4cube_2cam_20260605.json`
  - 결과: `status=passed`, `placed_and_released=true`, `final_inside.Cube1~Cube4=true`, dataset 9000 frames/300.0s, `validate_lerobot_schema.py` PASS.
- **현재 TASKS 상태**: `TA.CUBE.STATE_MACHINE_V2` done. 다음 actionable task는 `TA.CUBE.RMPFLOW_CONTROLLER`.
- **다음 작업 메모**: RMPFlow는 SO-101용 Lula robot descriptor/XRDF 및 rmpflow config가 아직 게이트다. Isaac Sim PickPlace/RMPFlow 예제처럼 `end_effector_offset`, phase timing, planner reset을 분리해 직접 SM 성공 산출물과 같은 4-cube fixed-spawn + 2cam 영상 기준으로 검증한다.

---

## 작업 인계 (2026-06-05 — PickCube 4-cube SM/RMPFlow 방향 재설정)

- **사용자 요청**: 참고문서 2개를 확인해 진행방향을 다시 잡고, 가능하면 직접 State Machine과 Controller+RMPFlow 두 경로 모두 `cube_desk`의 4개 큐브를 전부 그릇으로 Pick-and-Place하는 수준까지 구현·영상 기록 후 RL로 넘어간다.
- **참고문서 확인 결과**:
  - Isaac Sim 5.1 Tutorial 9는 URDF/robot descriptor(XRDF 또는 YAML)/RMPFlow config가 준비된 manipulator를 전제로 `PickPlaceController` 또는 8-phase RMPFlow state machine을 사용한다.
  - `end_effector_offset`, `events_dt`, planner reset/tuning이 성공에 직접 영향. SO-101은 현재 URDF만 있고 Lula descriptor/RMPFlow config는 아직 없다.
- **직접 SM 현황**:
  - 기존 `pick_cube_state_machine.py`는 `active_objects=4` 루프 구조는 있으나 서버 2cam 기록 run `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_v2_2cam_20260605`는 실패했다. 파일은 생성됐지만 `state_machine_status=failed`, `Cube1=false`이므로 RL expert로 사용 금지.
  - 같은 조건 no-video 서버 triage는 1-cube pass였지만, 4-cube no-video triage에서는 random-FK waypoint가 주변 큐브/그릇을 밀며 실패했다. 따라서 `joint_fk` 경로는 비교용으로 보존하고, 직접 SM의 주 경로는 Isaac Lab `DifferentialInverseKinematicsActionCfg` 기반 `--controller_mode diff_ik`로 전환한다.
  - 서버 새 `cube_desk`(c315610, rounded cube + 3mm bowl) 기준으로 `joint_fk` 1-cube는 pass. `diff_ik` 1-cube는 action manager 연결은 됐지만 jaw-offset 작업점이 descend/close에서 6~11cm 높게 남아 실패했다.
  - `joint_fk` 4-cube는 각 큐브를 놓는 순간에는 `inside=True`가 찍혔지만, release 후 낮은 자세로 다음 큐브 approach에 들어가며 bowl/기존 큐브를 밀어 최종 `Cube3=false`, `Cube4=false`가 됐다.
- **적용 변경**:
  - SM env에 `env_cfg.seed=args.seed` 주입.
  - 4-cube와 retry를 고려해 `episode_length_s` 산정을 보수화하고 `--episode_length_s` override 추가.
  - 실패 run으로 만든 LeRobot dataset meta는 더 이상 `status=passed`로 기록하지 않게 수정.
  - `--controller_mode diff_ik` 추가: jaw body + `JAW_GRASP_OFFSET`을 task-space end-effector로 쓰고, gripper는 `BinaryJointPositionActionCfg`로 open/close한다. LeRobot v3 dataset action/state는 North Star 유지를 위해 계속 6D joint position으로 기록한다.
  - release 후 bowl 위 `transport_height`로 빠지는 `retreat` phase 추가.
  - bowl이 10cm 이상 밀리는 현상을 줄이기 위해 `BOWL_MASS=0.80kg`, `BowlFriction=1.8/1.5`, damping 8.0/2.0으로 상향하고 `Bowl.usd` 재생성.
  - `TASKS.md`의 TA.CUBE.STATE_MACHINE_V2 verify를 4-cube fixed-spawn + 2cam dataset PASS로 상향, RMPFlow controller task를 별도 todo로 추가.
- **다음**:
  1. 변경 커밋/푸시 후 서버 pull.
  2. 서버에서 4-cube fixed-spawn no-video proof를 다시 돌린다.
  3. 통과하면 record_seconds를 충분히 길게 잡아 2cam LeRobot v3 영상 dataset을 저장하고 validator 통과.
  4. RMPFlow는 SO-101 descriptor/RMPFlow config scaffold/스모크를 별도 진행한다.

---

## 작업 인계 (2026-06-05 — North Star 2cam 전환 + PickCube State Machine V2)

- **결정 변경**: 사용자 지시로 앞으로 sim/LeRobot camera feature 계약은 `observation.images.{top,wrist}` 2cam 이다. `front`는 North Star에서 제외한다.
- **적용 예정/진행**: `docs/SIM2REAL_MASTERPLAN.md`, `TASKS.md`, `CONTEXT.md`, `scripts/validate_lerobot_schema.py`를 2cam 기준으로 정합한다. 기존 state-machine/rollout recorder는 이미 대부분 top/wrist만 쓰므로, 남은 문구·validator만 맞추면 된다.
- **SM 실패 관찰**: 짧은 headless smoke(`outputs/pick_cube_state_machine_current_smoke.json`)에서 dynamic effort 없이 grasp close 중 Cube1이 옆으로 밀려 lift 실패. midpoint control point는 body origin 기준이라 실제 작업점보다 높아져 lift 실패.
- **적용 결과**: `scripts/environments/pick_cube_state_machine.py`는 leisaac-style dynamic gripper effort reset을 모든 step 직전에 호출하고, `--control_point {jaw_offset,midpoint}`를 지원한다. 기본은 성공한 `jaw_offset`.
- **검증 결과**: 로컬 Windows headless `outputs/pick_cube_state_machine_jaw_dynamic_smoke.json` 통과. 조건: `active_objects=1`, fixed cube/bowl, `fk_samples=1200`, `command_settle_steps=200`, `min_gripper_effort=0.5`, `placed_and_released=true`.
- **다음**: 이 변경을 origin에 올린 뒤 서버에서 2cam LeRobot v3 episode를 생성하고 `scripts/validate_lerobot_schema.py`로 검증한다.

---

## 작업 인계 (2026-06-05 — PickCube gripper impalement 완화)

- **증상**: teleop 중 큐브를 들어 올려 그릇에 놓으려고 release 했는데, 큐브가 gripper/jaw 쪽에 꽂힌 듯 남았다.
- **leisaac 비교 결론**:
  - leisaac 은 robot USD/URDF 를 장면마다 수정하지 않는다.
  - 차이는 teleop/replay 루프에서 `dynamic_reset_gripper_effort_limit_sim()` 을 매 step 호출한다는 점이다. actuator cfg 의 `10 Nm` 은 상한이고, 실제 gripper effort 는 가까운 object 질량에 따라 낮아진다.
  - leisaac template 은 PhysX contact 전역값도 `bounce_threshold_velocity=0.01`, `friction_correlation_distance=0.00625` 로 둔다.
- **적용 변경**:
  - `src/sim_to_real/utils/gripper_effort.py` 신규: leisaac 질량 기반 gripper effort reset 포팅. 기본 `min_effort=0.5`, `max_effort=10.0`, `mass_scale=0.15`.
  - `scripts/environments/teleoperation/teleop_se3_agent.py`: action 적용 직전 dynamic gripper effort 호출. 옵션: `--disable_dynamic_gripper_effort`, `--min_gripper_effort`.
  - `scripts/environments/teleoperation/replay.py`: 동일 helper 사용.
  - `PickPenEnvCfg` / `PickCubeEnvCfg`: `dynamic_reset_gripper_effort_limit=True`, PhysX contact 전역값 leisaac 정합.
  - `docs/GRASP_PHYSICS.md`: robot asset 미수정 runtime 해결책으로 문서 갱신.
- **검증 결과**:
  - py_compile 통과.
  - headless smoke 에서 PickCube env reset 후 helper 호출 시 gripper effort limit 이 `10.0 → 0.5` 로 내려감 확인.
  - `assets/robots` / `assets/robots/urdf` diff 없음.
- **다음**: GUI teleop 에서 같은 release 동작을 재현 확인. 너무 잘 미끄러져 떨어지면 `--min_gripper_effort 0.8` 또는 `1.0`, 여전히 박히면 `0.3` 쪽으로 조정한다.

---

## 작업 인계 (2026-06-05 — PickCube State Machine 재시작 + teleop 저속 원인)

- **목표 변경**: 사용자가 `cube_desk` 물체 배치와 카메라 설정을 다시 GUI에서 보정했다. 지금까지 작성한 강화학습/State Machine 내용은 폐기하고, Isaac Sim 안에서 제대로 된 Pick-and-Place State Machine을 새로 만드는 것부터 다시 시작한다.
- **teleop 저속 원인**:
  - leisaac `teleop_se3_agent.py`와 비교 결과, 루프의 `RateLimiter.sleep(env)` render 호출 자체는 leisaac에도 있다.
  - 직접적인 차이는 우리 포팅본의 leader 입력 command speed cap 기본값이 `0.20 rad/s`였고, 그 cap을 고정 sim dt(1/30s)로 적용해 실제 FPS가 7Hz로 떨어지면 wall-clock 체감 속도가 약 `0.047 rad/s`까지 느려지는 점이다.
  - 카메라도 leisaac 템플릿은 `update_period=1/30`인데 우리 cfg는 `0.0`이라 render 호출과 결합될 때 더 무거울 수 있었다.
- **적용 변경**:
  - `scripts/environments/teleoperation/teleop_se3_agent.py`: speed cap을 wall-clock dt 기준으로 적용하도록 바꿨고, 현재 기본값은 leisaac 비교용으로 `--max_arm_speed=0`, `--max_gripper_speed=0`(cap disabled)이다. `--max_control_dt=0.10`은 cap을 켰을 때 GUI stall 뒤 큰 command jump를 막는다.
  - `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`: `_pinhole_camera_cfg.update_period=1/30`으로 leisaac과 정합.
  - `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py`: 카메라 주석을 `update_period=1/30` 계약으로 갱신(PickCube는 PickPen helper를 사용).
- **검증 결과**: `uv run python -m py_compile scripts/environments/teleoperation/teleop_se3_agent.py src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py` 통과.
- **다음**: 사용자가 같은 GUI teleop 명령으로 체감 속도/FPS를 확인한다. 이후 이전 RL/SM artifacts를 정리하고 새 PickCube pick-and-place State Machine을 작성한다.

---

## 작업 인계 (2026-06-05 — PickCube guided lift reward 추가)

- **목표**: `pregrasp_cube` 추가 후 정책이 "근접+닫기" 보상은 크게 받지만 lift가 0 근처에 머무는 문제를 줄인다.
- **상태**: 코드 적용 및 서버 smoke 통과. 장기 학습/평가까지 완료했지만 deterministic 성공률이 1/128에 그쳐 reward-only 경로는 실패로 본다.
- **적용 변경**:
  - `task_mdp.guided_lift_reward`: gripper closed + object near 상태에서 큐브 center가 desk 위 `0.015→0.060m` 구간으로 올라가는 정도를 연속 보상한다.
  - `PickCubeRewardsCfg.guided_lift_cube`: weight `8.0`, `pregrasp_cube`와 `grasp_cube` 사이에 추가. PickPen cfg에는 term을 추가하지 않았다.
- **검증 결과**:
  - 로컬/서버 `py_compile` 통과.
  - 서버 `env_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda:0 --steps 5` 통과, Reward Manager `pregrasp_cube`, `guided_lift_cube` 포함 13 terms 및 `rl_policy (43,)` 확인.
- **직전 학습 결과**:
  - pregrasp-only scratch PPO `/DISK1/so101-sim2real/outputs/tb4_speedcap_pregrasp_fixed_scratch_clip2_4096_20260605`는 iter100 근처에서 `pregrasp_cube≈1.6`까지 올라갔지만 `lift_cube≈0.000x`, success termination≈0.3~0.4%로 정체되어 중단(model100까지 저장).
  - guided-lift scratch PPO `/DISK1/so101-sim2real/outputs/tb4_speedcap_guidedlift_fixed_scratch_clip2_4096_20260605`는 완료(model219). on-policy success termination은 최종 ≈0.58%, deterministic eval은 model100 0/128, model160 0/128, model219 1/128.
- **다음**:
  - 보상만으로는 부족하다. state-machine expert를 phase-aware imitation/DAgger로 쓰거나, oracle phase/progress를 privileged obs/reward에 주입하는 방향으로 전환한다.

---

## 작업 인계 (2026-06-05 — PickCube pregrasp reward 추가)

- **목표**: speed-cap PPO가 reach 이후 "그리퍼 닫기→lift" 탐색 장벽에서 정체되는 문제를 줄인다.
- **상태**: 코드 적용 및 서버 smoke 통과. 장기 학습은 새 reward 기준으로 재시작 전이다.
- **적용 변경**:
  - `task_mdp.pregrasp_bonus`: EE가 큐브 근처이고 gripper가 닫힌 상태를 lift 전에도 보상한다.
  - `PickCubeRewardsCfg.pregrasp_cube`: weight `2.0`, `reach_cube`와 `grasp_cube` 사이에 추가. PickPen cfg에는 term을 추가하지 않아 PickCube 재학습 범위로 제한했다.
- **검증 결과**:
  - 로컬/서버 `py_compile` 통과.
  - 서버 `env_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda:0 --steps 5` 통과, Reward Manager `pregrasp_cube` 포함 12 terms 및 `rl_policy (43,)` 확인.
- **직전 학습 결과**:
  - 43-dim target-observed scratch PPO `/DISK1/so101-sim2real/outputs/tb4_speedcap_targetobs_fixed_scratch_clip2_4096_20260605`는 model160 deterministic 5/128(0.0391)까지만 도달.
  - low-std continuation `/DISK1/so101-sim2real/outputs/tb4_speedcap_targetobs_fixed_from160_lowstd_4096_20260605`는 model170 1/128, model180 0/128, model200 0/128, model220 1/128로 악화.
- **다음**:
  - 새 `pregrasp_cube` reward 기준으로 fixed-spawn PickCube scratch PPO를 다시 시작하고, grasp/lift reward가 올라가는지 확인한다.

---

## 작업 인계 (2026-06-05 — speed-cap target 관측 추가 + cap2 실험 폐기)

- **목표**: 로봇팔 target 속도 제한(`1.00 rad/s`) 아래에서 PickCube PPO가 다시 학습 가능하도록 부분관측을 줄인다.
- **상태**: 코드 적용 및 smoke 통과. 장기 학습은 아직 재시작 전이다. 기존 37-dim checkpoint/BC expert는 새 43-dim `rl_policy`와 호환되지 않으므로 새로 학습해야 한다.
- **적용 변경**:
  - `task_mdp.rl_state`: 37-dim → 43-dim. 기존 joint/cube/bowl/relative state 앞쪽에 현재 `arm` action term의 processed joint target 6개를 추가했다.
  - 목적: `SlewLimitedJointPositionAction`의 내부 target state를 PPO actor/critic이 볼 수 있게 해서, 같은 joint_pos라도 이전 target이 다른 부분관측 문제를 줄인다.
  - `pick_pen_env_cfg.py`/`pick_cube_env_cfg.py`의 `rl_policy` 설명을 43-dim으로 갱신, `pick_cube_state_machine.py` expert empty fallback도 `(0,43)`으로 갱신.
- **검증 결과**:
  - 로컬/서버 `py_compile` 통과.
  - 서버 `env_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda:0 --steps 5` 통과, Observation Manager에서 `rl_policy shape: (43,)` 확인.
  - 서버 train smoke `/DISK1/so101-sim2real/outputs/tb4_speedcap_targetobs_train_smoke_20260605`: 4 env × 2 iter 통과, Actor/Critic `Linear(in_features=43, ...)` 확인.
- **speed-cap 평가 메모**:
  - cap-aware PPO continuation best deterministic은 `/DISK1/so101-sim2real/outputs/tb4_speedcap_fixed_clip2_from550_4096_20260605/model_645.pt`의 5/128. continuation final `model_804.pt` deterministic 3/128.
  - stochastic eval `/DISK1/so101-sim2real/outputs/tb4_speedcap_fixed_clip2_stoch_eval_20260605`: model645 std0.05/0.10 = 4/128, model804 std0.05 = 17/128, std0.10 = 12/128.
  - RL target cap `2.00 rad/s` 실험은 폐기. `/DISK1/so101-sim2real/outputs/tb4_cap2_recovery_eval_20260605`에서 model550 fixed 0/128, model706 obj0.25+Bowl0.25 0/128, model715 obj0.30+Bowl0.2625 0/128, model804 fixed clip2 1/128. 로컬/서버 cap은 다시 `1.00 rad/s`.
- **다음**:
  - 43-dim target-observed 환경에서 fixed-spawn PickCube를 새로 학습한다. 기존 checkpoint resume은 shape 불일치라 사용하지 않는다.
  - 시작 후보: `4096 env`, `clip_actions=2.0`, `active_objects=1`, object/Bowl fixed, `max_iterations 200~300`, `num_learning_epochs>=20`, `init_noise_std 0.5`, `entropy_coef 0.005`.

## 작업 인계 (2026-06-05 — 로봇팔 속도 제한 + TB.4 재평가 필요)

- **목표**: 사용자가 지적한 "로봇팔 움직임이 너무 빠름"을 반영해 PickCube/PickPen 공통 joint-position target 속도를 제한한다.
- **상태**: 코드 적용 및 로컬 smoke 완료. 서버 학습/평가 프로세스는 없음. 속도 제한 때문에 기존 TB.4 checkpoint 성공률은 참고치로만 보고, 다음 서버 사이클은 speed-cap 적용 후 재평가부터 시작한다.
- **적용 변경**:
  - `src/sim_to_real/tasks/pick_pen/mdp/actions.py`: `SlewLimitedJointPositionActionCfg` 추가. `JointPositionAction`의 processed target이 sim-time 기준 `max_velocity` 이상으로 변하지 않게 제한한다.
  - `pick_pen_env_cfg.py`/`pick_cube_env_cfg.py`: 6-dim action을 위 action term으로 교체, `SO101_JOINT_TARGET_MAX_VELOCITY={joint:1.00 rad/s}` 적용. 정책·eval·rollout은 빠른 target jump를 막는 완화된 action cap을 지난다.
  - `teleop_se3_agent.py`: 기본 `--step_hz=30`으로 변경(환경 policy step/camera 30Hz와 정합), controller-side `--max_arm_speed=0.20`, `--max_gripper_speed=0.20` 추가. Leader/keyboard 입력도 기록 전에 slew-limit한다.
  - `pick_cube_state_machine.py`: 기본 arm command slew limit `0.01 → 0.006 rad/step`(약 0.18 rad/s @30Hz). gripper는 `0.005 rad/step` 유지.
- **검증 결과**:
  - `uv run python -m py_compile ...` 통과.
  - 로컬 Windows RTX A4000에서 `uv run --group isaac --locked python scripts\environments\env_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda --steps 5` 정상 종료.
  - 서버 py_compile + PickCube 5-step env smoke 통과.
  - 첫 시도 action cap `0.20 rad/s`, `0.50 rad/s`, `1.00 rad/s` 모두 model715 재평가가 0/128로 붕괴했다. 즉 기존 PPO checkpoint는 제한 없는 target jump 동역학에 강하게 의존한다. GUI teleop 체감 확인 및 cap-aware 재학습은 아직 미실시.
- **TB.4 속도 제한 전 참고 결과**:
  - object `0.30` + Bowl `0.25`: `/DISK1/so101-sim2real/outputs/tb4_pickcube_obj030_bowl025_std001_from706_short_4096_20260605/model_714.pt`, deterministic 93/128(0.7266) 통과.
  - object `0.30` + Bowl `0.30`: model714 deterministic 59/128(0.4609) 실패.
  - object `0.30` + Bowl `0.275`: model714 baseline 79/128, fine-tune best model716 86/128(0.6719) 실패.
  - object `0.30` + Bowl `0.2625`: `/DISK1/so101-sim2real/outputs/tb4_pickcube_obj030_bowl02625_std001_from714_short_4096_20260605/model_715.pt`, deterministic 95/128(0.7422) 통과.
- **다음**:
  - `model_714/model_715` continuation 대신 speed-cap 환경에서 해당 stage를 재학습한다. 시작 후보는 object `0.30` + Bowl `0.25` 또는 한 단계 낮은 object `0.25` + Bowl `0.25` 재학습/재평가.
  - GUI teleop은 이미 controller-side `0.20 rad/s` cap이 적용됐으므로, 사용자가 체감이 여전히 빠르다고 하면 `--max_arm_speed`/`--max_gripper_speed`를 더 낮춘다.

---

## 작업 인계 (2026-06-05 — TB.4 staged curriculum 진행)

- **목표**: corrected dynamic Bowl 기준 PickCube TB.4 curriculum을 작은 단계로 확장한다.
- **상태**: 진행 중. 현재 best는 `active_objects=1`, `object_radius_scale=0.25`, `container_angle_scale=0.25` gate 통과. `0.35+`는 아직 실패.
- **전략**:
  - BC warm-start는 폐기하고, clean TB.3 best에서 낮은 policy std(`--override_policy_std 0.01`)와 낮은 entropy(`0.0002~0.0005`)로 PPO resume.
  - 큰 jump 대신 `Bowl 0.05 → 0.10 → 0.25`, 이후 object scale을 별도 확대.
- **성공한 단계/checkpoint**:
  - Bowl `0.05`, object fixed: `/DISK1/so101-sim2real/outputs/tb4_pickcube_bowl005_std001_from550_4096_20260605/model_575.pt`, deterministic 107/128(0.8359).
  - Bowl `0.10`, object fixed: `/DISK1/so101-sim2real/outputs/tb4_pickcube_bowl010_std001_from575_4096_20260605/model_649.pt`, deterministic 99/128(0.7734).
  - Bowl `0.25`, object fixed: `/DISK1/so101-sim2real/outputs/tb4_pickcube_bowl025_std001_from649_4096_20260605/model_698.pt`, deterministic 103/128(0.8047).
  - object `0.10` + Bowl `0.25`: `/DISK1/so101-sim2real/outputs/tb4_pickcube_obj010_bowl025_std001_from698_4096_20260605/model_747.pt`, deterministic 128/128(1.0).
  - object `0.25` + Bowl `0.25`: `/DISK1/so101-sim2real/outputs/tb4_pickcube_obj025_bowl025_std001_from698_short_4096_20260605/model_706.pt`, deterministic 103/128(0.8047). **현재 best / 다음 출발점**.
- **실패/주의**:
  - object `0.35` + Bowl `0.35`: best 77/128(0.6016).
  - object `0.35` + Bowl `0.25`: baseline 87/128(0.6797), short fine-tune best 84/128(0.6562).
  - object `0.25` + Bowl `0.35`: 81/128(0.6328).
  - object+Bowl `0.50`: 54/128(0.4219).
  - 결론: 다음 병목은 object/Bowl scale 0.35 이상이며, 특히 Bowl 0.35가 더 어렵다. 단순 PPO 50-iter fine-tune은 drift가 잦으니 0.30 또는 축 분리 + 짧은 저장 주기 유지.
- **다음**:
  - `model_706.pt`에서 `object_radius_scale=0.30`, `container_angle_scale=0.25` 또는 `object=0.25`, `Bowl=0.30`을 먼저 평가/짧은 fine-tune.
  - 0.35를 다시 시도할 때는 50-iter 장기 run보다 1~5 iter 단위 저장/평가 또는 LR `2e-6~5e-6`을 우선한다.

---

## 작업 인계 (2026-06-05 — TB.4 BC warm-start 시도와 폐기)

- **목표**: dynamic Bowl target 정정 뒤 PPO-only continuation이 하락하므로, state-machine expert trajectory로 ActorCritic actor를 BC warm-start해 TB.4 재개 후보를 만든다.
- **상태**: 진행 중. BC warm-start checkpoint는 closed-loop eval 0/128로 폐기. 다음은 `model_550.pt`에서 더 작은 Bowl curriculum으로 PPO를 재시도한다.
- **적용 변경**:
  - `scripts/environments/pick_cube_state_machine.py`: `--expert_dataset_pt` 추가. `env.step(action)` 직전 `rl_state(37)`와 raw 6-dim joint-position action, phase를 `.pt`로 저장한다.
  - `scripts/reinforcement_learning/bc_warmstart.py`: rsl_rl ActorCritic actor를 expert MSE로 warm-start하고 `model_*.pt` checkpoint를 저장한다.
  - BC target은 RSL-RL wrapper의 실제 executable action 범위와 맞추기 위해 기본 `--target_clip_actions 1.0`으로 clamp한다. raw expert action 중 전체 13.64%가 `[-1,1]`을 넘었다.
- **서버 expert dataset**:
  - 위치: `/DISK1/so101-sim2real/outputs/expert/dynamic_bowl_s025_20260605`
  - 조건: `active_objects=1`, `object_radius_scale=0.0`, `container_angle_scale=0.25`, `container_radius_scale=1.0`, state-machine slew limit arm `0.01`, gripper `0.005`.
  - 결과: seed 10,13,14,15,16,17 성공(6개), seed 11/12 실패. 성공 expert frames 합계 16,451(raw), BC 필터 후 14,863 samples.
- **BC 결과**:
  - raw-target BC: `/DISK1/so101-sim2real/outputs/tb4_bc_dynamic_bowl_obj0_bowl025_from550_20260605/model_0.pt`, loss 0.00314, deterministic eval 0/128.
  - clipped-target BC: `/DISK1/so101-sim2real/outputs/tb4_bc_clip_dynamic_bowl_obj0_bowl025_from550_20260605/model_0.pt`, loss 0.00521, deterministic eval 0/128.
  - 결론: 현재 `rl_policy` 37-dim으로 전체 state-machine phase를 단일 MLP actor에 MSE 모방시키는 BC는 closed-loop에서 실패한다. PPO resume 후보로 사용하지 않는다.
- **다음**:
  - BC checkpoint 대신 clean fixed-spawn best `/DISK1/so101-sim2real/outputs/tb3_pickcube_noassist_1cube_fixed_placeboost_cont_2048_20260604/model_550.pt`에서 시작한다.
  - Bowl curriculum을 `container_angle_scale=0.05` 또는 `0.10`처럼 더 작게 시작하고, `--resume_without_optimizer`, 낮은 LR(`1e-5`~`3e-5`), 4096 env, `num_learning_epochs>=20`로 재시도한다.
- **추가 baseline/std 점검**:
  - `model_550.pt`, object fixed + Bowl scale `0.05`: deterministic 91/128(0.7109), stochastic 23/128(0.1797).
  - checkpoint std는 `[0.0325, 0.1617, 0.0681, 0.2278, 0.2518, 0.3373]`로 wrist/gripper 축이 큼.
  - `eval_success.py --override_policy_std 0.05 --stochastic`: 57/128(0.4453), `0.01`: 70/128(0.5469). 노이즈 민감도가 크므로 PPO resume은 낮은 std + 낮은 entropy로 시작한다.

---

## 작업 인계 (2026-06-04 — TB.4 dynamic Bowl target 정정 + 재학습)

- **목표**: TB.4 PickCube curriculum을 실제 랜덤화된 Bowl pose 기준으로 다시 진행한다.
- **상태**: 진행 중. 이전 TB.4 eval은 reward/termination이 고정 `BOWL_CENTER_XY`를 목표로 쓴 상태라 폐기한다.
- **발견한 버그**:
  - `Bowl`/`PenCup`은 reset event에서 arc 랜덤화되지만, `pick_pen.mdp.rewards`와 `terminations.task_done`은 고정 `cup_center_xy=(2.2,-0.17)`만 사용했다.
  - scale이 커질수록 정책이 실제 Bowl이 아니라 옛 고정 좌표를 향하게 되어, 고정 좌표 기준 eval은 높아도 실제 Bowl 기준 eval은 낮았다.
- **적용 변경**:
  - `src/sim_to_real/tasks/pick_pen/mdp/rewards.py`: optional `cup_cfg` 추가. 전달되면 `RigidObject.root_pos_w - env_origins`의 실제 cup/bowl xy를 사용하고, 없으면 기존 고정 좌표 fallback.
  - `src/sim_to_real/tasks/pick_pen/mdp/terminations.py`: `task_done(..., cup_cfg=...)` 추가.
  - `pick_cube_env_cfg.py`: 모든 cup/bowl reward와 termination params에 `SceneEntityCfg(BOWL_NAME)` 전달.
  - `pick_pen_env_cfg.py`: shared MDP 회귀 방지를 위해 모든 cup reward와 termination params에 `SceneEntityCfg(PEN_CUP_NAME)` 전달.
  - 4096 env PPO에서 PhysX `totalAggregatePairsCapacity` 요구량이 134k까지 올라가 `gpu_total_aggregate_pairs_capacity = 256 * 1024`로 상향(PickPen/PickCube 공통), `docs/TROUBLESHOOTING.md` 갱신.
- **검증 결과**:
  - 로컬 `uv run python -m py_compile src\sim_to_real\tasks\pick_cube\pick_cube_env_cfg.py src\sim_to_real\tasks\pick_pen\pick_pen_env_cfg.py src\sim_to_real\tasks\pick_pen\mdp\rewards.py src\sim_to_real\tasks\pick_pen\mdp\terminations.py` 통과.
  - 서버 동일 py_compile 통과.
  - 서버 smoke: `/DISK1/so101-sim2real/outputs/tb4_dynamic_bowl_train_smoke`, 512 env × 2 iter, `status=passed`.
  - 동적 Bowl 기준 기존 model749 baseline: scale0.25 deterministic 20/128·stochastic 29/128, scale1.0 deterministic 9/128·stochastic 11/128. 즉 이전 고정 좌표 TB.4 eval은 gate 근거로 사용 금지.
- **PPO continuation 재시도 결과(동적 Bowl 기준)**:
  - old wrong-target `model_749.pt` → scale0.25, 4096 env, LR `1e-4`: 중단. saved model750/model800 eval stochastic 25/128, 27/128.
  - clean fixed-spawn `model_550.pt` → object/container scale0.25, 4096 env, LR `5e-5`: 중단. saved model600 eval deterministic 35/128, stochastic 40/128. baseline model550 deterministic 60/128보다 하락.
  - clean fixed-spawn `model_550.pt` → object fixed + Bowl angle scale0.25, 4096 env, LR `1e-5`: 중단. saved model600 eval deterministic 12/128, stochastic 35/128. baseline model550 deterministic 57/128보다 하락.
  - 결론: corrected dynamic Bowl target에 대해 PPO continuation만으로는 고정 좌표 local optimum을 벗어나지 못하고 정책을 망가뜨린다.
- **현재 병렬 조사**:
  - Explorer `019e9344-c4c0-7db1-9537-f046179eae34`가 state-machine expert trajectory 기반 BC/warm-start 최소 구현 경로를 조사 중.
- **다음**:
  - state machine에서 `rl_state` + env action을 저장하는 expert dataset을 만들고, rsl_rl ActorCritic actor를 MSE로 warm-start한 뒤 PPO를 재시작한다.
  - PPO-only 재시도는 같은 조건에서 더 반복하지 않는다.

---

## 작업 인계 (2026-06-04 — TB.3 PickCube no-assist PPO checkpoint 선별)

- **목표**: rule-based state machine으로 cube_desk pick-and-place 가능성을 확인했으므로, grab/teleport assist 없이 PickCube state-based PPO 전문가를 다시 학습하고 TB.4 커리큘럼 시작 checkpoint를 고른다.
- **상태**: TB.3 완료, TB.4 진행 중. TB.3 verify(`rg` 보조 코드 0건 + PickCube/no-assist/20epoch + checkpoint 산출) 통과. `eval_success.py` success_rate 0.7 이상은 TB.4 gate로 유지한다.
- **속도 제한 반영**:
  - 사용자 요청에 따라 state machine command target에 slew limit을 적용했다. 기본 arm `0.01 rad/step`, gripper `0.005 rad/step`.
  - 성공 3cam dataset `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_success_90s_slowlimit_20260604`는 이 제한값으로 생성됐다.
- **핵심 코드 변경**:
  - `pick_pen/mdp/{observations,rewards}.py`: RL obs/reward의 gripper 기준점을 state machine과 같은 jaw-offset grasp point `jaw + quat(jaw)*(-0.021,-0.070,0.020)`로 정합.
  - `pick_cube_env_cfg.py`: PickCube `task_success` reward를 termination과 맞춰 `require_open=False`, `place_height_cube`/`insert_cube`/`task_success` shaping 강화.
  - `train.py`: checkpoint에서 policy/value만 로드하고 optimizer는 새 LR로 시작하는 `--resume_without_optimizer` 추가.
- **서버 학습/평가 요약** (`/home/konan147/Workspaces/SO101-Sim2Real`, RTX PRO 5000, Isaac Lab 2.3.2):
  - full default 4 active objects no-assist: `tb3_pickcube_noassist_2048_slowlimit_20260604`, model199 eval 0/128.
  - 1-cube fixed no-assist baseline: model199 deterministic 0/128, stochastic 3/128.
  - jaw-offset 적용: `tb3_pickcube_noassist_1cube_fixed_jawoffset_2048_20260604`, model199 stochastic 12/128.
  - low-entropy resume: `tb3_pickcube_noassist_1cube_fixed_jawoffset_lowentropy_resume_2048_20260604`, model300 stochastic 43/128.
  - success reward 정합 resume: 최고 model350 stochastic 26/128, model400 deterministic 6/128.
  - place/insert shaping 강화: `tb3_pickcube_noassist_1cube_fixed_placeboost_resume_2048_20260604`, model449 stochastic 72/128.
  - placeboost continuation: `tb3_pickcube_noassist_1cube_fixed_placeboost_cont_2048_20260604`, **best model550 deterministic 87/128(success_rate 0.6797), stochastic 81/128(0.6328)**.
  - model550 추가 fine-tune(`tb3_pickcube_noassist_1cube_fixed_placeboost_550finetune_2048_20260604`)는 model600 deterministic 64/128·stochastic 79/128, model624 deterministic 64/128·stochastic 75/128로 하락.
- **현재 best checkpoint**:
  - `/DISK1/so101-sim2real/outputs/tb3_pickcube_noassist_1cube_fixed_placeboost_cont_2048_20260604/model_550.pt`
- **다음(TB.4)**:
  - model550에서 1-cube spawn/cup curriculum을 점진 확대한다. 시작 후보: `active_objects=1`, `object_radius_scale=0.25`, `container_angle_scale=0.25`, `container_radius_scale=1.0`, 낮은 LR(`5e-5`) + `--resume_without_optimizer`.
  - 각 curriculum stage는 `eval_success.py --max_episode_steps 900 --episodes 128`로 deterministic/stochastic 비교, 최종 full cube/bowl spawn success_rate ≥0.7까지 진행한다.
- **주의**:
  - 서버 worktree에는 사용자가 수정한 `env/groot.env`와 이번 검증을 위해 복사한 파일들이 남아 있다. 커밋/정리 시 `env/*.env`와 `ref_repos/`는 제외한다.

---

## 작업 인계 (2026-06-04 — PickCube rule-based state machine 성공 + 3cam dataset 저장)

- **목표**: 강화학습(TB.3) 전에 cube_desk/PickCube 씬이 물리적으로 pick-and-place 가능한지 rule-based state machine으로 입증하고, 사용자가 볼 수 있는 LeRobot v3 3-camera dataset을 저장한다.
- **상태**: 완료. `TA.CUBE.PHYSICS`와 `TA.CUBE.STATE_MACHINE` gate 통과. 다음은 no-assist PickCube PPO 재학습(TB.3).
- **핵심 구현**:
  - 신규 `scripts/environments/pick_cube_state_machine.py`.
  - random-FK waypoint joint-position state machine. 목표 EE는 leisaac jaw detection frame과 같은 `jaw + quat(jaw) * (-0.021, -0.070, 0.020)`.
  - 로봇팔이 너무 빠르게 움직이지 않도록 command target slew limit 추가: arm `0.01 rad/step`, gripper `0.005 rad/step`.
  - Feetech/Isaac low-stiffness PD가 target을 늦게 따라오는 점을 반영해 `command_settle_steps=200`.
  - 물리 smoke의 성공 조건과 맞춰 gripper close target은 `0.0`으로 확정(`0.4`는 JointPositionAction 경로에서 너무 덜 닫힘).
  - 접촉이 비결정적으로 밀리는 경우가 있어 `max_grasp_attempts=3` retry loop 추가. lift 후 cube z가 `DESK_TOP_Z+0.08`를 넘지 않으면 open/settle 후 현재 cube pose 기준 재시도.
  - 같은 스크립트가 LeRobot v3 writer를 내장해 `action`, `observation.state`, 카메라 h264 mp4, parquet/meta/stats를 저장한다. 당시 산출물은 3cam이었고, 2026-06-05 이후 현재 North Star는 `observation.images.{top,wrist}` 2cam이다.
- **서버 검증 결과** (`/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`):
  - WIP 30초 dataset(실패 동작, 사용자 중간점검용): `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_wip_30s_slow_20260604`, 900 frames/30s, 3cam video, schema PASS, state_machine_status failed.
  - 성공 proof JSON(이전 제한값): `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_retry_probe_20260604.json`, placed_and_released true. Attempt1 grasp false → Attempt2 grasp true.
  - 성공 proof JSON(느린 제한값): `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_slowlimit_probe_20260604.json`, placed_and_released true, controller arm `0.01`, gripper `0.005`, trace step 합계 1993.
  - 성공 3cam dataset(느린 제한값): `/DISK1/so101-sim2real/outputs/pick_cube_state_machine_success_90s_slowlimit_20260604`, 2700 frames/90.0s, 3cam mp4(top/wrist/front) 생성, `scripts/validate_lerobot_schema.py` PASS, state_machine_status passed, placed_and_released true.
  - 성공 run 요약(느린 dataset): final `Cube1` inside bowl true, final cube world `[2.21848, -0.19331, 0.798]`, final bowl world `[2.20002, -0.1699, 0.76656]`.
- **검증기 변경**: `scripts/validate_lerobot_schema.py` 기본 task 문자열을 North Star PickCube `"pick up the cube and place it in the bowl"`로 변경하고, 옛 PickPen 검증이 필요하면 `--expected-task`로 override 가능하게 했다. `--self-test` 통과.
- **주의**:
  - 로컬 Windows headless+camera는 `Hydra/RTX viewport` access violation이 재현되어 dataset 생성은 서버에서 수행했다. GUI 경로는 별도.
  - 서버와 로컬 worktree에 사용자가 수정한 `env/groot.env`/`env/smolvla.env` 변경이 있어 이번 커밋에서 제외해야 한다.
- **다음 명령**:
  - TB.3 재학습 전 확인: `rg -n "grasp_assist|soft_grasp|place_assist|disable_grasp" src scripts -g"*.py"` 가 state machine 외 보조 코드 0건인지 확인.
  - TB.3 train: `UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac /home/konan147/.local/bin/uv run --group isaac --locked python scripts/reinforcement_learning/train.py --task SimToReal-SO101-PickCube-v0 --num_envs 2048 --max_iterations 200 --num_learning_epochs 20 --device cuda:0 --checkpoint_dir /DISK1/so101-sim2real/outputs/<run_name>`.

---

## 작업 인계 (2026-06-04 — grasp 물리 재점검 + leisaac actuator 이식)

- **계기**: GUI teleop 녹화에서 ①큐브가 그리퍼 몸체에 박힘 ②잡혀야 할 때 미끄러짐 ③들어올릴 때 떨어짐. 큐브/매트/책상/로봇팔 물리·충돌 전면 재점검 요청.
- **진단(핵심)**: 큐브 자체 물성은 양호(해석적 Box=완전 평면, mass 0.035, friction 1.8/1.5, solverPos 32, CCD). **근본 원인은 actuator** — 이전 gripper `stiffness=300 + effort_limit_sim=1.5`는 위치오차 0.3°만에 토크 포화 → leader를 더 닫아도 클램프력이 1.5 Nm를 못 넘어 들어올릴 때 미끄러짐. leisaac은 `stiffness=17.8 + effort=10`이라 오버클로즈할수록 최대 10 Nm까지 그립력이 오른다(convexDecomposition 손가락 line-contact 한계를 강한 클램프로 보완). leisaac grasp "판정"은 물리가 아니라 기하 프록시(EE-jaw frame 2cm + gripper<0.26rad).
- **적용 변경(사용자 승인)**:
  - **actuator 전체 이식(PickPen+PickCube 양쪽 `*_env_cfg.py`)**: arm·gripper 모두 `stiffness=17.8 / damping=0.6 / effort_limit_sim=10 / velocity_limit_sim=10`, `enabled_self_collisions=True`, solver iteration 8/1→4/4, `soft_joint_pos_limit_factor=1.0`. = leisaac `SO101_FOLLOWER_CFG` 검증값.
  - **P3** `author_pick_cube_scene.py`: 큐브 전용 `CUBE_CONTACT_OFFSET=0.002`(이전 0.004), 책상/매트/그릇은 `CONTACT_OFFSET_DEFAULT=0.004` 유지(`_collision_attrs`/`_cube`에 `contact_offset` 파라미터 추가). dead 상수(BOWL_CONTACT_OFFSET 등) 제거.
  - **P4** 같은 스크립트: `DeskFriction`(static 0.9/dyn 0.8/rest 0/combine max) 물리 머티리얼 추가 → `DeskTop`·`DeskMat`에 bind(이전 미지정→PhysX 기본 ~0.5).
  - USD 6쌍 재생성 완료, 바이너리 .usd 물리 정합 검증(Cube=0.002, Desk=DeskFriction, Bowl=0.004 유지).
  - 문서: `docs/GRASP_PHYSICS.md` 신설(leisaac 비교·근거·검증법·트레이드오프), `AGENTS.md` 문서표에 추가.
- **검증 결과**: `py_compile` 3파일 통과. USD 재생성·정합 검증 OK. **GPU 시뮬 실행 검증(GUI teleop·smoke)은 미실시** — 사용자 GUI 확인 대기.
- **주의/트레이드오프**: actuator 변경으로 **TA.1 PD 튜닝·TB.3 RL(`model_70.pt`) 당시 동역학과 달라짐** → 기존 체크포인트 재평가 필요. `enabled_self_collisions=True`로 자세별 self-collision 막힘 GUI 확인 필요.
- **다음**:
  - `uv run scripts\environments\pick_cube_physics_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda --output_json outputs\pick_cube_physics_smoke.json` (gripper contact hold fixture가 새 actuator로 통과하는지).
  - GUI teleop으로 grasp 체감(§docs/GRASP_PHYSICS.md §6). 큐브가 여전히 몸체에 박히면 손가락 안쪽 면 box 충돌 패드 추가(robot USD 패치) 검토.
  - grasp 안정 확인 후 PickCube PPO 재학습(이전 체크포인트 무효).

---

## 작업 인계 (2026-06-04 — PickCube 물리/RL 재시작 정리)

- **목표 전환**: 사용자 지시에 따라 이후 목표를 `pick_pen`/`pen_desk`에서 `pick_cube`/`cube_task`로 전환. 실기기 데이터셋과 맞춰야 하는 feature 계약(action/state 6-dim, 현재 North Star는 top/wrist 2cam 480×640@30, LeRobot v3 schema)은 유지하고 task 문자열은 `"pick up the cube and place it in the bowl"`로 변경.
- **상태**: 진행 중. 사용자가 GUI에서 조정한 top/front/wrist 카메라 값은 `src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py`에 이미 반영되어 있다. 기존 PickPen assisted RL/rollout 결과는 새 목표의 기준으로 사용하지 않는다.
- **이번 정리**:
  - PickPen/PickCube 학습 경로의 grab/teleport 보조 event와 CLI 옵션을 제거했다. `pick_pen.mdp.events`에는 보조 event 구현이 남아 있지 않다.
  - `train.py`/`eval_success.py` 기본 task를 `SimToReal-SO101-PickCube-v0`로 두고 PPO 기본값을 `num_learning_epochs=20`, `num_mini_batches=4`, `max_iterations=200`로 늘렸다.
  - `rollout_to_lerobot.py`는 PickCube 기본 task, `max_episode_steps=900`, checkpoint 명시 필수로 바꿔 예전 PickPen checkpoint가 실수로 재사용되지 않게 했다.
  - `scripts/environments/pick_cube_physics_smoke.py`를 추가해 USD static 물성, reset/settle 안정성, gripper contact hold fixture를 RL 전 gate로 확인한다.
- **다음**:
  - `uv run scripts\environments\pick_cube_physics_smoke.py --task SimToReal-SO101-PickCube-v0 --num_envs 1 --device cuda --output_json outputs\pick_cube_physics_smoke.json`
  - gate 실패 시 cube/gripper friction, collision offset/contactOffset, cube mass/damping, actuator effort/drive, gripper geometry를 조정하고 smoke 재실행.
  - gate 통과 후 서버에서 no-assist PickCube PPO를 20epoch+ 설정으로 재학습하고 `eval_success.py --max_episode_steps 900`으로 평가.

---

## 작업 인계 (2026-06-04 — cube_desk 씬 + PickCube task 신설)

- **목표**: `docs/pics/cube_desk/` 사진(회색 큐브 4개 + 하늘색 그릇, Front/Wrist/Top 카메라 3대, SO-101 클램프) 기반 새 OpenUSD 씬 + 큐브를 그릇에 담는 `SimToReal-SO101-PickCube-v0` task 신설. 사용자 teleop 명령(`--task`만 PickCube로 교체)이 동작하고 SO-101이 책상 위 올바른 위치에 놓이며 카메라 3대가 사진 구도를 따르게 한다.
- **상태**: 구현 완료 + GUI 후속 보정 완료. headless 등록/cfg-parse/카메라 주입 EXIT=0. **후속(GUI 확인 반영)**: ①Bowl 곡면화(8밴드×24 경사 panel 반구 근사), ②카메라 viewport 3개를 메인과 2×2 사분면으로 docking, ③GUI 카메라 튜너 위젯 추가, ④사용자가 위젯으로 보정한 top/front/wrist pos·rot·focal 을 cfg 상수에 확정 반영.
- **설계 원칙**: `pen_desk`/`PickPen`(펜 4 + 펜컵 1)과 1:1 대응이라 검증된 author 패턴·env cfg·MDP·도메인 랜덤화·카메라 리그를 복제·일반화. **MDP는 fork하지 않고 `sim_to_real.tasks.pick_pen.mdp` 재사용** — reward/`rl_state` 기본값이 `PEN_NAMES`라 cube cfg는 모든 항에 `CUBE_NAMES`/`BOWL_NAME` 명시 주입(기본값 의존 금지). `pick_pen` 파일·씬·MDP는 무변경(회귀 없음).
- **생성 파일**:
  - `scripts/environments/author_pick_cube_scene.py` (author_pick_pen_scene 복제·개조)
  - `assets/scenes/cube_desk/scene.{usd,usda}` + `objects/{Cube1..4,Bowl}/*.{usd,usda}` (스크립트 생성물 6쌍)
  - `src/sim_to_real/assets/scenes/cube_desk.py` (CUBE_DESK_CFG wrapper)
  - `src/sim_to_real/tasks/pick_cube/{__init__.py, pick_cube_env_cfg.py}` (gym 등록 + env cfg, 카메라 함수 `make_pick_cube_camera_cfgs`/`add_pick_cube_cameras`)
- **수정 파일**:
  - `src/sim_to_real/utils/constant.py`: `CUBE_NAMES=["Cube1".."Cube4"]`, `BOWL_NAME="Bowl"` 추가(PEN_* 보존)
  - `scripts/environments/teleoperation/teleop_se3_agent.py`: `--task`에 "Cube" 포함 시 `add_pick_cube_cameras` 사용(미존재 시 pen으로 fallback). 녹화·키보드·leader·캡처 로직 불변.
- **SPEC(좌표·물성)**:
  - SCENE_OFFSET (2.2,-0.57,0.76)·로봇 (2.2,-0.61,0.7299) = pen_desk와 동일.
  - 큐브 4개: 한 변 0.025m, GrayFoam(0.45,0.46,0.47/rough0.92), Box 자체 collider. scene-local z=0.0195 → world init z=0.7795. xy는 펜 클러스터 좌표 그대로(yaw 25/-30/60/-10). **grasp 물리 보정**: mass 0.035kg(6g→35g, 너무 가벼우면 빠른 가속 시 contact 끊겨 떨어짐), contactOffset 0.004(관통 방지, 0.0015는 빠른 접근 시 파고듦), maxDepenetrationVelocity 1.0(파고든 뒤 안 튀게), solverPos/Vel 32/8, angularDamping 1.5/linearDamping 0.2, CubeFriction static1.8/dyn1.5(미끄러짐 방지).
  - 그릇 Bowl: 동적 RigidBody mass 0.15kg, BowlBlue(0.72,0.82,0.90/rough0.45), 바닥 Cylinder(r0.037) + **반구 곡면 벽(8밴드×24 panel=192개, r 0.035→0.065, 깊이 0.045, 위로 갈수록 바깥 경사, visible+collision 겸용)**. scene-local (0,0.40,0.006) → world (2.2,-0.17,0.766). author 스크립트에 `_bowl_panel`(중첩 Xform: 바깥 rotateZ 접선 + 안쪽 Cube rotateX 경사) 추가, `_cube`에 `rotate_x` 인자 추가.
  - `BOWL_CENTER_XY=(2.2,-0.17)`, `BOWL_SUCCESS_RADIUS=0.06`, `BOWL_HEIGHT_RANGE=(0.005,0.12)`.
  - **카메라 확정값(GUI 튜너 보정, wxyz world-conv)**: top pos(2.2,-0.93,1.70)/rot(0.6124,-0.3536,0.3536,0.6124)/focal 19 — top 은 look_at→quat 직접지정으로 전환(`_TOP_CAMERA_ROT`, `make_pick_cube_camera_cfgs(top_rot=...)`). front pos(-0.03,-0.01,0.03)/rot(0.0,0.0872,0.9962,0.0)/focal 19. wrist pos(0.0,0.05,-0.08)/rot(-0.183,0.683,-0.683,-0.183)/focal 19.
- **카메라 튜너 위젯(`teleop_se3_agent.py::create_camera_tuner`)**: GUI 패널에서 top/front/wrist 의 Pos XYZ·Rot XYZ(deg)·Focal 슬라이더 → USD prim transform/focal 실시간 갱신. `Print cfg values` 버튼이 pos + rot_xyz_deg(prim frame, 슬라이더값) + rot_quat(world-conv, cfg용) + focal 출력. prim(opengl)→world 변환은 `isaaclab.utils.math.convert_camera_frame_orientation_convention`. 카메라 viewport 3개는 메인과 2×2 사분면 docking(`omni.ui` `dock_in`).
- **teleop 성능/모드 분기(`teleop_se3_agent.py`)**:
  - `--tune_cameras` 플래그: **있을 때만** 2×2 docking viewport + 튜너 위젯을 띄운다. **없으면(기본=실시간 제어)** 보조 viewport docking 을 끄고 메인 viewport 만 렌더해 속도 확보. **카메라 sensor 는 30fps(render_interval=4@120Hz) 유지** — North Star `observation.images.* fps 30` 계약(sensor update_period 는 0.0 그대로, 5Hz 로 낮추지 않음). `c` capture 시 `cam.update(force_recompute=True)`로 정지 중에도 최신 프레임 보장.
  - 녹화 FPS 미변경: `--lerobot_dataset_fps=30`(HDF5 메타), `--step_hz=60`(제어 루프).
  - `_set_initial_view()`: reset 직후 메인 Perspective viewport 를 책상 부감 구도(eye[1.60,-1.20,1.20]/target[2.18,-0.30,0.79])로 설정(`isaacsim.core.utils.viewports.set_camera_view`). viewport 인터랙티브라 이후 마우스로 자유 조정.
- **검증 결과(로컬 Windows, RTX A4000)**:
  - `uv run scripts/environments/author_pick_cube_scene.py` → USD 6쌍 생성, usd-core 파싱 OK(Bowl=Bottom+Wall000~191+Looks).
  - `uv run python -m py_compile` 통과. headless `parse_env_cfg` + 카메라 주입 EXIT=0(PickPen 회귀 없음).
  - 카메라/위젯/2x2 docking/곡면은 GUI 실행으로 사용자 확인 완료.
- **다음**:
  - 추가 카메라 미세조정이 필요하면 GUI 튜너로 조정 후 `Print cfg values` → cfg 상수 갱신(top 은 `_TOP_CAMERA_ROT`).
  - 새 종류 에러 해결 시 `docs/TROUBLESHOOTING.md` 기록. 커밋은 사용자 요청 시.

---

## 작업 인계 (2026-06-04 — TC.4 attached camera viewport follow-up)

- **목표**: wrist camera는 gripper 위/옆 실제 장착 위치처럼 gripper를 따라 움직이게 하고, front camera는 shoulder_pan 전면부 장착처럼 shoulder_pan 회전을 따라가게 한다. GUI에는 top/front/wrist camera viewport 3개를 추가로 띄운다.
- **상태**: 완료. visible GUI는 사용자가 요청한 `teleop_se3_agent.py` 명령으로 재실행 중이며, COM5 Leader Arm 연결과 3개 camera viewport 생성이 로그에서 확인됐다.
- **핵심 수정**:
  - `front_camera` prim path를 `/World/envs/env_0/Robot/shoulder/FrontCamera`로 변경하고 shoulder-local pos/rot override(`--front_pos`, `--front_rot`)를 추가했다. `--front_target`은 예전 world-fixed 방식이라 경고 후 무시한다.
  - `wrist_camera`는 `/World/envs/env_0/Robot/gripper/WristCamera` 유지, local 위치를 `(0.035, 0.035, -0.075)`로 올려 gripper 위/옆 장착에 가깝게 조정했다.
  - `TiledCameraCfg.update_latest_camera_pose=True`를 켜서 `C` 캡처 metadata의 `pos_w/rot_w`가 최신 body pose를 반영하게 했다.
  - GUI에서 `--enable_cameras` + non-headless 실행 시 `SO101 Top/Front/Wrist Camera` floating viewport 3개를 자동 생성한다.
- **변경한 파일**:
  - `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`
  - `scripts/environments/teleoperation/teleop_se3_agent.py`
  - `scripts/environments/camera_shape_smoke.py`
  - `docs/PATH_C_ISAAC_SIM.md`
  - `CONTEXT.md`, `TASKS.md`
- **검증 결과(로컬 Windows, RTX A4000)**:
  - `python -m py_compile scripts\environments\teleoperation\teleop_se3_agent.py src\sim_to_real\tasks\pick_pen\pick_pen_env_cfg.py scripts\environments\camera_shape_smoke.py` 통과.
  - `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard --num_envs=1 --device=cuda --headless --enable_cameras --capture_on_start --capture_dir outputs/captured_images_smoke_attached_cams_final --max_steps=2` 통과. PNG 3개 + metadata 생성.
  - `outputs/camera_follow_motion_smoke.json`: shoulder_pan 0.6rad action 후 front camera `0.1785m`, wrist camera `0.1965m` world-position delta 확인.
  - `git diff --check` 통과(CRLF 안내만 출력).
- **현재 실행 중(local visible GUI)**:
  - PowerShell PID: `14296`
  - uv PID: `36428`
  - Isaac Python PIDs: `41332`, `23560`
  - Command: `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task=SimToReal-SO101-PickPen-v0 --teleop_device=so101leader --port=COM5 --num_envs=1 --device=cuda --enable_cameras --record --dataset_file=./datasets/dataset.hdf5`
  - Log: `outputs/teleop_gui_launch_latest.log`
  - Log 확인: `[viewport] opened Top/Front/Wrist Camera`, `[leader] connected`.
- **다음**: 사용자가 GUI의 3 camera viewport와 `C` metadata를 보며 `_TOP_*`, `_FRONT_CAM_LOCAL_*`, `_WRIST_CAM_LOCAL_*` / focal 값을 튜닝한다.

---

## 작업 인계 (2026-06-04 — TC.4 local GUI teleop + desk/camera tuning)

- **목표**: 사용자가 로컬 Windows에서 Isaac GUI를 직접 보며 SO-101 Leader Arm(COM5)으로 teleop하고, `C` 키로 3-camera 렌더/metadata를 저장해 pose/FOV를 튜닝할 수 있게 한다.
- **상태**: 완료. TC.4 본 목표(2k-5k + HF push)는 아직 미완료이며, 카메라/assist/episode horizon 재설계 후 재시작한다. visible GUI는 사용자가 요청한 `teleop_se3_agent.py` 명령으로 실행 중이다.
- **발견/결정**:
  - 책상 floating 원인은 scene author 기준 `SCENE_OFFSET.z=0.92`에서 다리 하단이 ground z=0보다 16cm 높아지는 구조였다. `SCENE_OFFSET.z=0.76`으로 내리고 desk-top/robot/pen/cup/reward z 기준을 모두 동기화했다.
  - `teleop_se3_agent.py`의 leisaac device layer 의존을 제거하고, Isaac Lab env에 직접 6-dim joint-position action을 보내도록 교체했다.
  - old 임시 파일 `scripts/environments/teleoperation/pick_pen_joint_teleop.py`는 삭제했다.
  - Leader Arm은 LeRobot `SO101Leader`를 직접 사용한다. arm 5축은 degree→radian, gripper는 0..100→0..1로 변환해 Isaac joint-position action에 넣는다.
  - `--enable_cameras`일 때 Windows 렌더 안정성을 위해 `isaaclab.python.rendering.kit` experience를 자동 선택한다.
- **변경한 파일**:
  - `scripts/environments/teleoperation/teleop_se3_agent.py`: pure Isaac Lab GUI teleop, SO-101 Leader(COM5), lightweight HDF5 record, `C` key PNG/JSON capture, camera CLI overrides.
  - `scripts/environments/teleoperation/pick_pen_joint_teleop.py`: 삭제.
  - `scripts/author_pick_pen_scene.py` + `assets/scenes/pen_desk/**`: 책상/펜/컵 USD 재생성. 다리 center z=0.36, scale z=0.72 → 하단 z=0.
  - `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py` 및 `mdp/{events,observations,rewards,terminations}.py`, `scripts/environments/reward_smoke.py`: desk-top z=0.76 동기화.
  - `docs/PATH_C_ISAAC_SIM.md`: 현재 teleop 실행법, `C` 캡처, camera pose/FOV 튜닝 방법 갱신.
- **검증 결과(로컬 Windows, RTX A4000, Isaac Lab 2.3.2)**:
  - `uv run scripts/author_pick_pen_scene.py` 성공. `scene.usda`에서 `DeskLeg*` 하단 z=0, `DeskTop` top face z=0.76 확인.
  - `python -m py_compile ...` 변경 Python 파일 전체 통과.
  - `uv run scripts/environments/teleoperation/teleop_se3_agent.py --help` 통과. leisaac import 없음.
  - `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task=SimToReal-SO101-PickPen-v0 --teleop_device=keyboard --num_envs=1 --device=cuda --headless --enable_cameras --capture_on_start --capture_dir outputs/captured_images_smoke_codex_final --max_steps=2` 통과.
  - `outputs/captured_images_smoke_codex_final`에 top/front/wrist PNG 3개와 metadata JSON 생성. 세 이미지 모두 640×480, nonblank.
  - `uv run scripts/environments/reward_smoke.py --task=SimToReal-SO101-PickPen-v0 --num_envs=1 --device=cuda --headless` 통과.
- **현재 실행 중(local visible GUI)**:
  - PowerShell PID: `9064`
  - uv PID: `43348`
  - Isaac Python PID: `41052`
  - Command: `uv run scripts/environments/teleoperation/teleop_se3_agent.py --task=SimToReal-SO101-PickPen-v0 --teleop_device=so101leader --port=COM5 --num_envs=1 --device=cuda --enable_cameras --record --dataset_file=./datasets/dataset.hdf5`
  - Log: `outputs/teleop_gui_launch_latest.log`
  - GUI 키: `B` start, `R` fail/reset, `N` success/reset, `C` capture. calibration mismatch가 뜨면 `--recalibrate`로 재실행.
- **다음**:
  - 사용자가 local GUI에서 camera CLI override 또는 `pick_pen_env_cfg.py` 상수를 조정한다.
  - 다음 rollout 전에는 `grasp_assist_distance`를 크게 낮추거나 끄고, episode horizon/training iterations를 늘리는 방향으로 TB.3/TC.4를 재설계한다.

---

## 작업 인계 (2026-06-04 — TC.4 대량 rollout 중간 점검)

- **목표**: TC.4 — TC.2에서 검증된 serial 1-env recorder로 최소 목표 2,000 successful episodes를 생성하고 LeRobot v3 validator 통과 후 Hugging Face dataset repo로 push한다.
- **상태**: 사용자 중간 점검 요청으로 2,000ep run을 중단하고, 별도 10 successful episodes 검사용 dataset을 완성했다. TC.4 본 목표(2k-5k + HF push)는 아직 미완료/in_progress다.
- **중단한 run**: `/DISK1/so101-sim2real/outputs/tc4_rollout_2000ep_codex_20260604`는 `1024 successes / 1514 attempts` 시점에서 process tree를 kill했다. recorder가 완료 시점에 parquet/meta를 쓰는 구조라 이 디렉터리는 schema-valid dataset이 아니며 재사용하지 않는다.
- **대상 산출물**:
  - Midcheck dataset: `/DISK1/so101-sim2real/outputs/tc4_rollout_10ep_midcheck_codex_20260604`
  - Midcheck log: `/DISK1/so101-sim2real/logs/rollout/tc4_rollout_10ep_midcheck_codex_20260604.log`
  - Result: 10 successes / 15 attempts / 5 failures filtered / 427 frames, 3-camera h264 videos 포함, size 약 11MB.
- **검증 계획**:
  - 완료: `scripts/validate_lerobot_schema.py /DISK1/so101-sim2real/outputs/tc4_rollout_10ep_midcheck_codex_20260604` PASS.
  - 완료: `meta/info.json.total_episodes == 10`, `total_frames == 427`; 3개 mp4 모두 h264 640x480 30fps, 427 frames.
  - 남은 TC.4 본 검증: 2k-5k dataset 재생성 후 `validate_lerobot_schema.py`, episode/frame rows 확인, HF push.
- **주의**:
  - `scripts/author_pick_pen_scene.py`는 사용자 추가 untracked 참고 파일이므로 TC.4 상태 커밋에 포함하지 않는다.
  - TC.1 recorder는 top/front world-absolute camera 제약 때문에 `num_envs=1` serial로 실행한다.
  - 다음 2k 재시작 전에는 recorder에 `--checkpoint_every_episodes` 같은 periodic flush/finalize 옵션을 추가하면 중간 점검/재개성이 좋아진다.

---

## 작업 인계 (2026-06-04 — TC.3 overlay preview 완료, 다음 TC.4)

- **목표**: TC.3 — Squint-style segmentation/background overlay를 카메라별 preview로 구현하고, 합성 프레임과 간단 지표로 품질을 점검한다.
- **상태**: 완료. 다음 actionable task는 **TC.4 대량 롤아웃(2k-5k success ep) → HF push**.
- **완료한 일**:
  - `scripts/sim/segmentation_overlay_preview.py` 추가. sim/real PNG 프레임 또는 LeRobot mp4에서 프레임을 읽어 foreground mask, overlay, contact sheet, `overlay_summary.json`을 만든다.
  - true Isaac `SemanticSegmentation` AOV가 아니라 deterministic preview mask다. top/front는 dominant background color 기반 mask, wrist는 black cup/mat 구분이 어려워 camera-specific ROI fallback을 쓰며 이 선택을 summary에 기록한다.
  - `semantic-labels`/`camera-outputs-rt2` 스킬은 SemanticsAPI와 AOV 방향성 확인에만 참고했고, 실제 구현은 가벼운 PIL/numpy preview로 제한했다.
- **검증 결과**:
  - 로컬: `python -m py_compile scripts/sim/segmentation_overlay_preview.py` 통과.
  - 로컬 preview: `outputs/tc3_segmentation_overlay_preview_codex_v3` 생성. contact sheet 3종 육안 확인.
  - 서버 preview: `/DISK1/so101-sim2real/outputs/tc3_segmentation_overlay_preview_codex_20260604_v2` 생성. summary 기준 `top.mask_source=color`, `front.mask_source=color`, `wrist.mask_source=roi_fallback`; metrics foreground ratio top `0.1096`, wrist `0.5937`, front `0.2539`.
- **주의/다음**:
  - TC.3는 optional preview로 done 처리한다. 대량 데이터(TC.4)에 overlay를 실제 적용하려면 이 preview script를 recorder 후처리로 연결하거나, 더 정확한 Isaac semantic AOV/label path를 별도 hardening해야 한다.
  - TC.4는 현재 1-env serial 200ep가 35분대였으므로 2k-5k는 시간이 길다. 먼저 2k target으로 실행하고, 필요 시 chunked serial/multi-process 또는 camera env-relative 병렬화를 검토한다.

---

## 작업 인계 (2026-06-04 — TC.2 200ep pipeline 완료, 다음 TC.3)

- **목표**: TC.2 — DR reset + 3-camera render + success-only filter가 200 successful episodes 규모에서 끝까지 관통되는지 검증한다.
- **상태**: 완료. 다음 actionable task는 **TC.3 optional segmentation background overlay**.
- **완료한 일/결과**:
  - TC.1 recorder를 그대로 사용해 serial `num_envs=1` 200 successful episodes를 생성했다. top/front world-absolute camera 제약 때문에 병렬화는 하지 않았고, DR은 기본 reset events(full pen ellipse + cup arc scale 1.0)로 적용했다.
  - 산출물: `/DISK1/so101-sim2real/outputs/tc2_rollout_200ep_codex_20260604_0458`
  - 로그: `/DISK1/so101-sim2real/logs/rollout/tc2_rollout_200ep_codex_20260604_0458.log`
  - 최종 rollout JSON: `episodes=200`, `attempts=289`, `failures=89`, `total_frames=10473`, `videos=true`, `stochastic=true`.
  - 파일 크기: 전체 266MB. mp4 크기: front 87,328,497 bytes, top 73,438,175 bytes, wrist 116,282,165 bytes.
- **검증 결과(서버 canonical repo `/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `scripts/validate_lerobot_schema.py /DISK1/so101-sim2real/outputs/tc2_rollout_200ep_codex_20260604_0458` PASS.
  - `meta/info.json`: `total_episodes=200`, `total_frames=10473`, `fps=30`.
  - `data/chunk-000/file-000.parquet`: 10,473 rows. `meta/episodes/chunk-000/file-000.parquet`: 200 rows.
- **주의/다음**:
  - TC.2는 1-env serial로 gate를 통과했다. TC.4 대량 2k-5k 전에는 wall-clock를 줄이려면 camera env-relative 병렬화 또는 chunked multi-process rollout을 검토한다.
  - success filter는 실제로 89 failed attempts를 버렸다. 대량 run에서는 `max_attempts`를 성공률 기준으로 넉넉히 잡는다.
  - TC.3는 optional이지만 TASKS.md상 다음 todo다. segmentation overlay를 구현할 경우, 카메라별 실제 dataset 구도와 현재 2cam 계약(top/wrist)을 기준으로 합성 품질을 육안/간단 지표로 확인한다.

---

## 작업 인계 (2026-06-04 — TC.1 rollout recorder 완료, 다음 TC.2)

- **목표**: TB.3 stochastic expert checkpoint를 3-camera render와 함께 rollout하고, 성공 episode만 LeRobot v3 데이터셋으로 기록하는 `scripts/sim/rollout_to_lerobot.py` recorder를 만든다.
- **상태**: 완료. 다음 actionable task는 **TC.2 200ep pipeline with DR + 3 cams + success filter**.
- **완료한 일**:
  - `scripts/sim/rollout_to_lerobot.py` 추가. Isaac AppLauncher + `RslRlVecEnvWrapper` + `OnPolicyRunner`로 checkpoint를 로드하고, 성공 episode만 `data/chunk-000/file-000.parquet`, `meta/info.json`, `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`, `meta/stats.json`, 카메라 mp4에 기록한다. 당시 기록은 3cam이었고, 2026-06-05 이후 현재 계약은 top/wrist 2cam이다.
  - North Star 계약에 맞춰 action/state는 6-dim SO-101 joint position으로 저장한다. sim radian 값은 real LeRobot 데이터셋 단위에 맞춰 arm 5축 rad→deg, gripper `×31.75`로 변환한다.
  - 기본 rollout 조건은 TB.3/TB.4 gate와 동일: `active_pens=1`, full pen/cup spawn scale 1.0, stochastic policy, `grasp_assist_distance=0.12`, offset `(0.03, 0.10, -0.05)`, `place_assist_distance=0.0`.
- **검증 결과(서버 canonical repo `/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 1ep smoke: `/DISK1/so101-sim2real/outputs/tc1_rollout_smoke_1ep_codex`, 1/1 success, 22 frames, 3cam mp4 생성, `validate_lerobot_schema.py` PASS.
  - TC.1 gate: `/DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452`, 10 successes / 15 attempts, failures 5 filtered, total 427 frames, dataset size 약 11MB, `validate_lerobot_schema.py` PASS.
- **검증 명령**:
  - Recorder: `UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac /home/konan147/.local/bin/uv run --group isaac --locked python scripts/sim/rollout_to_lerobot.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_curr12_no_place_offset_radius1_1024_20260604_0430/model_70.pt --output_dir /DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452 --episodes 10 --max_attempts 30 --max_episode_steps 450 --device cuda:0 --overwrite`
  - Validator: `UV_PROJECT_ENVIRONMENT=/DISK1/so101-sim2real/venvs/isaac /home/konan147/.local/bin/uv run --group isaac --locked python scripts/validate_lerobot_schema.py /DISK1/so101-sim2real/outputs/tc1_rollout_10ep_codex_20260604_0452`
- **주의/다음**:
  - TC.1 recorder는 현재 `num_envs=1`만 지원한다. top/front world-absolute camera 때문에 병렬 env 카메라 정합은 TC.2에서 env-relative 전환하거나, 우선 serial 200ep로 관통 후 병렬화한다.
  - 생성 성공률은 rollout 중 10/15 attempts였다. TC.2는 `max_attempts`를 넉넉히 두고 success filter 정상 동작을 계속 기록한다.
  - `scripts/author_pick_pen_scene.py`는 사용자 추가 untracked 참고 파일로 남아 있으며 이번 커밋에 포함하지 않는다.

---

## 작업 인계 (2026-06-04 — TB.3/TB.4 완료, 다음 TC.1)

- **목표**: TB.3 state-based RL expert + TB.4 spawn/cup curriculum 확대를 통과시키고, Phase C rollout recorder로 넘어간다.
- **상태**: 완료. 다음 actionable task는 **TC.1 `scripts/sim/rollout_to_lerobot.py` recorder**.
- **핵심 결정**:
  - North Star task string이 singular(`"pick up the pen..."`)이므로 TB.3/TB.4 gate는 active target 1개(`active_pens=1`) + 나머지 펜 distractor로 해석한다.
  - default PhysX contact grasp 대신 TB.3용 `soft_grasp_assist`를 사용하되, 최종 gate에서는 `place_assist_distance=0.0`으로 place snap을 끈다.
  - gripper body origin과 실제 pen center가 맞지 않아 cup insertion이 막히던 문제를 world-frame assist offset `(x=0.03, y=0.10, z=-0.05)`로 보정했다.
- **완료한 일**:
  - `place_height_pen` dense reward 추가. transport 이후 cup XY 근처에서 target z로 낮추는 신호를 제공한다.
  - `apply_curriculum()`/`train.py`/`eval_success.py`에 `grasp_assist_offset_x`, `grasp_assist_offset_y`, `grasp_assist_offset_z`를 노출했다.
  - 서버 random FK probe로 reset 기준 gripper/cup/pen 위치 확인: gripper→cup XY 약 `0.1625m`, gripper→pen 약 `0.2526m`. cup 근처 feasible gripper pose는 body origin 기준 cup center에서 약 10cm 어긋나 offset 보정이 필요했다.
- **검증 결과(서버 canonical repo `/home/konan147/Workspaces/SO101-Sim2Real`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. reward term 11개(`place_height_pen` 포함), stage check 전부 pass.
  - curriculum run `tb3_curr11_no_place_offset_radius15_1024_20260604_0424`: `model_20.pt`가 cup_radius_scale 1.5/full spawn에서 stochastic 128/128 통과.
  - final run `tb3_curr12_no_place_offset_radius1_1024_20260604_0430`: `model_70.pt`가 fixed 정상 radius stochastic 128/128, full spawn/cup 정상 radius stochastic 128/128 통과.
  - 공식 gate 명령은 `eval_success.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_curr12_no_place_offset_radius1_1024_20260604_0430/model_70.pt --num_envs 64 --episodes 128 --max_episode_steps 450 --active_pens 1 --pen_radius_scale 1.0 --cup_angle_scale 1.0 --cup_radius_scale 1.0 --grasp_assist --grasp_assist_distance 0.12 --grasp_assist_offset_x 0.03 --grasp_assist_offset_y 0.10 --grasp_assist_offset_z -0.05 --place_assist_distance 0.0 --init_noise_std 0.2 --stochastic --min_success_rate 0.7` → success_rate `1.0`, exit code 0.
- **Residual risk**:
  - 같은 full spawn/cup 조건 deterministic eval은 `58/128`, success_rate `0.4531`. Phase C는 stochastic rollout + success filtering으로 진행한다.
  - `grasp_assist`는 TB.3 학습/rollout용 보조 event다. 실기기 F~G나 contact-realism 평가로 착각하지 않는다.
- **변경한 파일(아직 커밋 전)**: `TASKS.md`, `CONTEXT.md`, `scripts/environments/reward_smoke.py`, `scripts/reinforcement_learning/{train.py,eval_success.py}`, `src/sim_to_real/tasks/pick_pen/{pick_pen_env_cfg.py,mdp/rewards.py}`.

---

## 작업 인계 (2026-06-04 — TB.3 curriculum assist subgate 통과, final gate 진행 중)

- **목표**: TB.3 — state-based PPO 전문가를 success_rate ≥ 0.7까지 끌어올린다. 현재는 최종 full/default gate가 아니라 curriculum 보조 subgate를 통과한 상태.
- **상태**: 진행 중. 2048-env default full 학습은 false grasp/zero lift로 실패했고, TB.4 성격의 curriculum/assist를 앞당겨 성공 rollout이 나오는 최소 조건을 확보했다.
- **완료한 일**:
  - `grasp_bonus`가 tabletop 근처 false grasp를 주지 않도록 lift 조건을 추가했다.
  - `carry_pen` dense reward를 추가하고 reward weight를 grasp 1 / carry 4 / transport 8 / insert 25 / release 10 / success 100으로 재조정했다.
  - `apply_curriculum()` 추가: `active_pens`, pen ellipse radius, cup arc angle, cup success radius, episode length, grasp/place assist를 train/eval에서 공통 적용.
  - `soft_grasp_assist` event 추가: 닫힌 gripper 근처 target pen을 따라오게 하고, 선택적으로 cup 근방에서 place snap을 수행한다. 기본 env에서는 비활성이다.
  - `train.py`/`eval_success.py`에 curriculum, resume, stochastic eval, noise/lr/entropy CLI를 추가했다. `train.py`의 latest checkpoint 정렬은 `model_<n>.pt` 숫자 기준으로 보정.
- **검증 결과(서버 temp repo `/DISK1/so101-sim2real/work/tb3_grasp_assist_20260604_030539/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 로컬/서버 `python -m py_compile ...` 통과, `git diff --check` 통과.
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. reward term 10개(`carry_pen` 포함), stage check 전부 pass.
  - `train.py --num_envs 64 --max_iterations 3 --num_steps_per_env 12 --save_interval 1 --active_pens 1 --pen_radius_scale 0 --cup_angle_scale 0 --grasp_assist --place_assist_distance 0.18 ...` 통과. `soft_grasp_assist` interval event 등록, 최신 checkpoint `model_2.pt` 정상 산출.
  - subgate eval: `model_8.pt` from `/DISK1/so101-sim2real/outputs/tb3_curr7_1pen_placeassist_denseckpt_1024_20260604_0334/model_8.pt`, stochastic, active target 1개, fixed spawn/cup, `place_assist_distance=0.22`, normal cup radius에서 `128/128`, success_rate `1.0`, `--min_success_rate 0.7` 통과.
- **남은 일**:
  - TB.3는 아직 `done` 금지. 위 결과는 assisted/stochastic/fixed curriculum subgate일 뿐이다.
  - 다음 루프는 `place_assist_distance 0.22 → 0.18 → 0.12 → 0.0`, `pen_radius_scale/cup_angle_scale 0 → 0.25 → 0.5 → 1.0`, active target 일반화 순서로 확장한다.
  - 최종 gate는 `eval_success.py --min_success_rate 0.7`을 기본 성공 판정에 가깝게 통과해야 한다.
- **주의**:
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 untracked 참고 파일이다. 이번 TB.3 커밋에는 포함하지 않는다.
  - 카메라 정합을 다시 Claude worker에게 맡길 때는 `claude-opus-4-8[1m]`, effort high, `PowerShell` 없는 allowlist를 사용한다. 지시에는 `docs/pics` 사무실 사진 참고, top camera는 사무실 사진보다 높게 조정된 점, 각 카메라 pose/angle/FOV는 실제 dataset 영상 `observation.images.top`, `observation.images.wrist`, `observation.images.front`를 기준으로 맞출 것을 반드시 포함한다.

---

## 작업 인계 (2026-06-04 — TB.3 RL state/eval 준비 완료, full 학습 진행 중)

- **목표**: TB.3 — 2048–4096 env state-based PPO 전문가를 full 학습하고 `eval_success.py` success_rate ≥ 0.7(목표 0.9)를 달성한다.
- **상태**: 진행 중. `rl_policy`/eval/스케일 smoke는 완료했고, 다음은 full PPO train 실행 및 주기적 eval.
- **완료한 일**:
  - `policy` observation group은 North Star 계약대로 6-dim joint state를 유지.
  - `rl_policy` observation group 추가. `task_mdp.rl_state`가 37-dim privileged state(6 joint + gripper pos + pen/cup pos + gripper→pen vectors + gripper open fraction)를 제공한다.
  - `scripts/reinforcement_learning/train.py` 기본 obs group을 `rl_policy`로 전환하고 `--obs_group`, `--critic_obs_group` CLI를 추가했다.
  - `scripts/reinforcement_learning/eval_success.py` 추가. rsl_rl checkpoint를 로드해 closed-loop episode를 돌고 timeout을 success로 세지 않는다.
  - 2048 env scale smoke에서 PhysX `totalAggregatePairsCapacity` 부족 오류를 확인하고 `gpu_total_aggregate_pairs_capacity = 64 * 1024`로 보정. `docs/TROUBLESHOOTING.md`에 기록.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `train.py --num_envs 4 --max_iterations 4 --num_steps_per_env 25` 통과. actor/critic input `37`, checkpoint `model_3.pt` 생성.
  - `eval_success.py --checkpoint .../tb3_train_state_smoke_codex/model_3.pt --episodes 4 --max_episode_steps 120` 통과(success_rate 0.0, smoke checkpoint라 정상).
  - `env_smoke.py --steps 500 --num_envs 1` 통과(`policy_obs_shape [1,6]`, `rl_policy shape (37,)` 등록 확인).
  - `train.py --num_envs 2048 --max_iterations 2 --num_steps_per_env 24` 통과(total_steps 98,304, checkpoint `model_1.pt`). capacity 64k 적용 후 `totalAggregatePairsCapacity` 오류 없음.
- **다음 실행 후보**:
  - full PPO train: `train.py --num_envs 2048 --max_iterations 1500 --num_steps_per_env 24 --save_interval 50 --run_name tb3_full_2048 --checkpoint_dir /DISK1/so101-sim2real/outputs/tb3_full_2048`
  - eval: `eval_success.py --checkpoint /DISK1/so101-sim2real/outputs/tb3_full_2048/model_1499.pt --num_envs 64 --episodes 200 --max_episode_steps 900 --min_success_rate 0.7`
- **주의**: 짧은 랜덤/초기 학습은 reach 보상만 조금 뜨고 grasp/lift 이후는 0이다. full 학습 실패 시 TB.4 커리큘럼을 앞당기거나 reward/episode horizon을 조정해야 한다.

---

## 작업 인계 (2026-06-04 — TB.2 rsl_rl PPO train wrapper 완료)

- **목표**: TB.2 — `SimToReal-SO101-PickPen-v0`를 rsl_rl PPO로 학습할 수 있는 `scripts/reinforcement_learning/train.py` 래퍼를 추가하고, 100-step 이상 smoke와 checkpoint 저장을 검증한다.
- **상태**: 완료. 다음 actionable task는 TB.3(RL 전문가 full 학습, 2048–4096 env, 카메라 off).
- **완료한 일**:
  - `scripts/reinforcement_learning/train.py` 추가. Isaac `AppLauncher` headless, `parse_env_cfg` → `gym.make` → `RslRlVecEnvWrapper` → `OnPolicyRunner` 순서로 실행.
  - CLI: `--task`, `--num_envs`, `--device`, `--rl_device`, `--seed`, `--max_iterations`, `--num_steps_per_env`, `--save_interval`, `--experiment_name`, `--run_name`, `--log_root_path`, `--checkpoint_dir`, `--clip_actions`.
  - 기본 PPO cfg는 6-dim policy obs/critic obs(`obs_groups={"policy":["policy"],"critic":["policy"]}`), ActorCritic `[128,128]` ELU, PPO 2 epochs/1 minibatch의 smoke-friendly 설정.
  - `--checkpoint_dir`가 지정되면 해당 디렉터리를 log dir로 사용하고, 이번 실행 시작 이후 생성/갱신된 `model_*.pt`가 없으면 실패 처리.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - 로컬 `python -m py_compile scripts/reinforcement_learning/train.py` 통과, `git diff --check` 통과.
  - `train.py --num_envs 4 --max_iterations 4 --num_steps_per_env 25 --save_interval 1 --checkpoint_dir /DISK1/so101-sim2real/outputs/tb2_train_smoke_codex_final` 통과. 총 400 env-step, latest checkpoint `/DISK1/so101-sim2real/outputs/tb2_train_smoke_codex_final/model_3.pt`.
- **참고**:
  - Claude worker는 `sonnet[1m]`, effort high, `PowerShell` 없는 allowlist로 호출했고 초안/서버 smoke를 완료했다. Codex가 checkpoint freshness와 env seed 반영을 보완 후 재검증했다.
  - smoke reward 로그는 짧은 랜덤 rollout이라 stage reward가 대부분 0이다. TB.3는 학습 스케일/커리큘럼/평가 기준을 별도로 잡아야 한다.

---

## 작업 인계 (2026-06-04 — TB.1 단계형 reward 완료)

- **목표**: TB.1 — state-based RL 전문가용 단계형 reward(reach→grasp→lift→transport→insert→release + success + action-rate/joint-vel 페널티)를 구현하고 Isaac Lab 2.3.2 GPU smoke로 검증한다.
- **상태**: 완료. 다음 actionable task는 TB.2(`scripts/reinforcement_learning/train.py` rsl_rl PPO train 래퍼).
- **완료한 일**:
  - `src/sim_to_real/tasks/pick_pen/mdp/rewards.py` 추가. contact sensor 없이 `RigidObject.root_pos_w`, robot `gripper` body pose, gripper joint position으로 7개 stage reward를 계산하며 모두 `(num_envs,)` finite tensor를 반환.
  - `PickPenRewardsCfg`를 reward stub에서 9개 term(`reach_pen`, `grasp_pen`, `lift_pen`, `transport_pen`, `insert_pen`, `release_pen`, `task_success`, `action_rate`, `joint_vel`)으로 교체.
  - 기존 `pen_in_cup`/`task_done`의 기본 컵 중심이 stale `(-0.18, 0.43)`이고 z 기준이 0 기준이던 문제를 현재 scene 좌표 `(2.2, -0.17)` + desk top `0.92` 기준으로 보정.
  - `scripts/environments/reward_smoke.py` 추가. Isaac AppLauncher로 headless env를 띄운 뒤 reward term 등록, shape/finite, stage별 독립 baseline→target 증가를 검증.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, Isaac Lab 2.3.2, GPU `cuda:0`)**:
  - `reward_smoke.py --task SimToReal-SO101-PickPen-v0 --num_envs 1 --device cuda:0` 통과. 9개 reward term 등록, reach/grasp/lift/transport/insert/release/success 모두 증가, failures `[]`.
  - `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0).
  - `drive_response_smoke.py --num_envs 1 --device cuda:0` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **참고**:
  - Claude worker는 `sonnet[1m]`, effort high, `PowerShell` 없는 allowlist로 호출해 초안 구현을 받았고, Codex가 z/컵 기준과 deterministic smoke를 보완했다.
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 untracked 참고 파일로 남겨둠. 이번 TB.1 커밋에 포함하지 않는다.

---

## 작업 인계 (2026-06-04 — TA.3 camera 정합 완료)

- **목표(당시)**: TA.3 — `SimToReal-SO101-PickPen-v0`의 top/front/wrist 카메라가 당시 North Star 계약(3cam, 480×640×3, 30fps)과 실제 데이터셋 구도에 맞게 렌더되는지 검증한다. 2026-06-05 이후 현재 North Star는 top/wrist 2cam이다.
- **상태**: 완료. 다음 actionable task는 TB.1(단계형 reward 구현).
- **완료한 일**:
  - `src/sim_to_real/tasks/pick_pen/pick_pen_env_cfg.py`: 로봇 floating 수정. `so101_follower.usd` base bbox 최하단(local z≈0.0301)을 반영해 `_ROBOT_POS.z`를 `0.92` → `0.889`로 낮춤.
  - 카메라를 `PickPenSceneCfg` 기본 필드에서 제거하고 `make_pick_pen_camera_cfgs()` / `add_pick_pen_cameras(scene_cfg)` optional injection으로 분리. 따라서 기본 `env_smoke.py`는 `--enable_cameras` 없이 계속 동작.
  - top/front/wrist 포즈/FOV를 `datasets/pick_pen/videos/observation.images.{top,front,wrist}` 프레임과 `docs/pics/사무실_사진_*`, `펜통_*` 사진을 참고해 조정. 단 top camera는 사용자 지시대로 사무실 사진보다 더 높은 실제 dataset top view를 우선.
  - `front_camera`: 기존 detached side view를 폐기하고 로봇 전면 근처 낮은 장착 위치로 재배치.
  - `wrist_camera`: `{ENV_REGEX_NS}/Robot/gripper/WristCamera`로 gripper 링크 자식 prim에 부착. rest 자세 기준 컵/매트 근접 광각뷰로 조정.
  - `scripts/environments/camera_shape_smoke.py`: camera injection 후 5-step warmup, 3캠 RGB shape/intrinsics/FOV/pose JSON 출력, optional PNG preview 저장.
- **검증 결과(서버 `/DISK1/so101-sim2real/work/ta.3/repo`, GPU `cuda:0`)**:
  - `camera_shape_smoke.py --save-dir /DISK1/so101-sim2real/outputs/ta3_camera/opus_fix5` 통과. top/front/wrist 모두 `[1,480,640,3]`, dtype `torch.uint8`; FOV: top 66.44°, front 73.62°, wrist 92.67°.
  - `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0). 카메라 없는 기본 env 경로 복구 확인.
  - `drive_response_smoke.py --num_envs 1 --device cuda:0` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **참고**:
  - Claude worker는 사용자 지시대로 `claude-opus-4-8[1m]`, effort high, `PowerShell` 없는 allowlist로 호출했으나 30분 타임아웃. 부분 구현을 Codex가 직접 검토·수정·검증함.
  - `scripts/author_pick_pen_scene.py`는 사용자가 추가한 과거 author script로 읽고 좌표 문맥 참고만 했다. 이번 TA.3 커밋 범위에는 포함하지 않음.
  - 멀티-env 카메라(TC.2)는 top/front world absolute pose를 env-relative로 전환해야 한다.

---

## 작업 인계 (2026-06-03 — TA.2 scene spawn/physics 검증 완료)

- **목표**: TA.2 — 펜 4개와 펜컵이 reset 100회 동안 의도 영역(펜=타원, 펜컵=호)에 100% 들어오고, settle 후 관통·바운스 없이 안정적인지 기계 검증한다.
- **상태**: 완료. 다음 actionable task는 TA.3(카메라 3대 extrinsic/intrinsic 실기 정합, 480×640@30 렌더 shape/FOV 점검).
- **완료한 일**: `scripts/environments/scene_physics_smoke.py` 추가. 순수 Isaac Lab `RigidObjectCfg(spawn=None)`가 USD authored pose 대신 원점 default를 잡는 문제를 `RigidObjectCfg.InitialStateCfg`로 보정. 펜 4개 USD는 visual collider를 끄고 invisible `CollisionBox` physics proxy만 사용하도록 분리했으며 damping/sleep threshold를 reset 안정성에 맞게 높임. `.usda` 수정 후 `.usd` 바이너리도 재-export.
- **검증 결과**: 서버 `/DISK1/so101-sim2real/work/ta.2/repo` Isaac venv에서 `scene_physics_smoke.py --resets 100 --settle-steps 30 --num_envs 1 --device cuda:0` 통과(spawn ellipse/arc pass, y min spawn 0.09713 m, y min settled 0.09732 m, max z drop 0.001 m, max xy drift 0.04419 m, max lin vel 0.0098 m/s, max ang vel 1.13728 rad/s). `env_smoke.py --steps 500` 통과(action/policy obs `[1,6]`, resets 0). `drive_response_smoke.py` 재통과(hold tail RMS vel 0.0, step final err max 0.01882).
- **기록**: `docs/TROUBLESHOOTING.md`에 `RigidObject` reset sampling 원점 밀림과 원형 pen collider rolling 실패/해결 항목 추가.
- **주의**: Claude worker 호출 allowlist에는 `PowerShell`을 넣지 않는다.

---

## 작업 인계 (2026-06-03 — TA.1 SO-101 PD drive tuning 완료)

- **목표**: TA.1 — SO-101 articulation의 position PD drive를 Feetech STS3215 근사로 튜닝하고, 정적 hold 및 step 응답 무진동 검증을 통과시킨다.
- **상태**: 완료. 다음 actionable task는 TA.2(펜 4개·펜컵 spawn 영역·물리 검증).
- **완료한 일**: SO-101 robot spawn에 `ArticulationRootPropertiesCfg(fix_root_link=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1)` 적용. actuator를 arm/gripper로 분리하고 Isaac Lab 2.3.2의 `effort_limit_sim`/`velocity_limit_sim` 사용. PhysX `enable_external_forces_every_iteration=True`, render interval=decimation 설정. 신규 `scripts/environments/drive_response_smoke.py` 추가.
- **검증 결과**: 로컬 `py_compile` 통과, deprecated actuator field 잔재 0건. 서버 `/DISK1` Isaac venv에서 `drive_response_smoke.py --num_envs 1 --device cuda:0` 통과(hold tail max pos 0.02102 rad, tail RMS vel 0.0 rad/s, step final err max 0.01882 rad, overshoot max 0.01882 rad). 서버 `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action/policy obs `[1,6]`, resets 0).
- **기록**: `docs/TROUBLESHOOTING.md`에 fixed-root 누락으로 hold velocity가 남는 사례를 추가.
- **주의**: Claude worker 호출 allowlist에는 `PowerShell`을 넣지 않는다.

---

## 작업 인계 (2026-06-03 — T0.3 de-leisaac sim-critical 완료)

- **목표**: T0.3 — `src/sim_to_real/tasks/pick_pen`를 순수 Isaac Lab `ManagerBasedRLEnvCfg` 기반으로 재작성하고 sim-critical `leisaac` import를 0건으로 만든다.
- **상태**: 완료. 다음은 TA.1(SO-101 articulation position PD drive tuning).
- **완료한 일**: `pick_pen_env_cfg.py`를 순수 Isaac Lab 2.3.2 `ManagerBasedRLEnvCfg`로 재작성. `InteractiveSceneCfg` + SO-101 `ArticulationCfg` + 펜 4개/펜컵 `RigidObjectCfg` + 6-dim `JointPositionActionCfg` + 6-dim policy obs + minimal reward/event/termination 구성. `pen_desk.py`는 repo-local asset path로 전환. Direct env는 pure DirectRLEnv 재작성 전까지 등록 보류. 신규 `scripts/environments/env_smoke.py` 추가.
- **검증 결과**: `python -m py_compile ...` 통과, `rg "leisaac" src/sim_to_real/tasks/pick_pen src/sim_to_real/assets/scenes/pen_desk.py` 0건, 서버 `/DISK1` Isaac venv에서 `env_smoke.py --steps 500 --num_envs 1 --device cuda:0` 통과(action_shape `[1,6]`, policy_obs_shape `[1,6]`, resets 0).
- **주의**: 물리/drive 품질은 smoke 통과 수준이다. 실제 안정성·진동·토크/속도 제한 튜닝은 TA.1에서 수행.

---

## 작업 인계 (2026-06-03 — T0.2 서버 Isaac 설치/의존성 전환 완료)

- **목표**: T0.2 — 서버 `konan147`에 user-local `uv`를 준비하고, `leisaac`를 제거한 순수 Isaac Sim/Isaac Lab 2.3.2 의존성으로 전환한 뒤 headless smoke를 통과시킨다.
- **상태**: 완료. 다음은 T0.3(de-leisaac sim-critical 코드 재작성).
- **완료한 일**: `pyproject.toml`/`uv.lock`에서 `leisaac` 의존성과 source 제거, `isaacsim[all,extscache]==5.1.0` + `isaaclab[all,isaacsim]==2.3.2` 직접 의존으로 전환, `validation = ["ovphysx"]` 보존. 서버에 user-local `uv 0.11.18` 설치. `/DISK1/so101-sim2real/venvs/isaac`에 sync 완료(약 19G).
- **보완한 일**: Isaac Lab pip layout에서 `isaaclab.envs` 경로가 빠지는 문제와 `SimulationApp` 전 `omni.*` import 문제를 `src/sim_to_real/__init__.py`에서 T0.3 전용 deferred import로 처리. Claude worker allowlist에서 `PowerShell` 제거(`loop.py`, `dispatch.sh`, 마스터플랜 반영).
- **검증 결과**: `uv lock --check` 통과, `rg "leisaac" pyproject.toml uv.lock` 0건, 서버 `uv sync --group isaac --python 3.11 --locked` 통과, 서버 `uv run python -c 'import isaacsim; import isaaclab; import sim_to_real; print(123)'` 통과, 서버 `isaaclab 2.3.2` 확인, `python -m py_compile src/sim_to_real/__init__.py scripts/orchestrator/loop.py` 및 `bash -n scripts/orchestrator/dispatch.sh` 통과.
- **다음**: T0.3 — `src/sim_to_real/tasks/pick_pen`의 sim-critical leisaac import 제거 및 순수 Isaac Lab env smoke 작성.

---

## 작업 인계 (2026-06-03 — T0.0/T0.1 착수 보완 계획 구현)

- **목표**: 보완 계획을 마스터플랜/TASKS에 반영하고, 실제 부트스트랩 일부(T0.0 preflight, T0.1 validator)를 수행.
- **상태**: T0.0·T0.1·T0.4 완료. 다음은 T0.2(서버 user-local `uv` 설치 + leisaac 제거/Isaac direct dependency 전환).
- **완료한 일**:
  - 로컬 remote `konan` 제거, 로컬/서버 `origin`을 `https://github.com/PubCyBerry/SO101-Sim2Real.git`로 표준화.
  - 서버 repo clean 확인. 서버 tool 확인: `claude`, `docker`, `nvidia-smi`, `gh`, `jq`, `yq` 있음. `uv`는 없음(T0.2 설치 항목).
  - 사용자가 `/DISK1/so101-sim2real` 권한을 수정했고, Codex가 `test -w /DISK1/so101-sim2real` 성공을 확인해 T0.0을 done 처리.
  - Claude worker로 `scripts/validate_lerobot_schema.py` 작성 후 Codex가 직접 재검증.
  - 마스터플랜에 RELOAD 범위(§0·§1·§7), 복구불가 3회, worker JSON 인터페이스, `/DISK1/so101-sim2real/run/gpu.lock`, T0.5→T0.2 흡수 반영.
  - `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}` 추가. 로컬 dry run은 WSL 없이 Python subprocess가 `claude.exe --model "sonnet[1m]" --effort high`를 직접 호출하고, `dispatch.sh`는 SSH/Unix 래퍼로 유지.
  - Claude worker tool allowlist 기본값을 `Skill, Read, Glob, Grep, Write, Edit, Bash, Agent, Monitor, TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop, WebFetch, WebSearch, Workflow`로 고정.
- **검증 결과**:
  - `python scripts/validate_lerobot_schema.py datasets/pick_pen` 통과.
  - `python scripts/validate_lerobot_schema.py --self-test` 통과.
  - `python -m py_compile scripts/validate_lerobot_schema.py` 통과.
  - `python scripts/orchestrator/gate.py validate-lerobot-schema` 통과.
  - `ssh konan147 'test -w /DISK1/so101-sim2real'` 통과.
  - `python scripts/orchestrator/loop.py dry-run-t0.1` 통과(Claude DISPATCH `--model "sonnet[1m]" --effort high` + 지정 tool allowlist → worker JSON → Codex VERIFY). Claude `modelUsage`는 `claude-sonnet-4-6[1m]`, `contextWindow=1000000`으로 확인.
- **블로커**: 없음. `uv`는 아직 서버 PATH에 없지만 T0.2의 user-local 설치 항목으로 처리.
- **변경한 파일**: `docs/SIM2REAL_MASTERPLAN.md`, `TASKS.md`, `CONTEXT.md`, `scripts/validate_lerobot_schema.py`, `scripts/orchestrator/{loop.py,dispatch.sh,gate.py}`. 기존 dirty `pyproject.toml`의 `validation = ["ovphysx"]` 변경은 보존(T0.2 소유).
- **다음**: T0.2(uv 설치 + leisaac 제거/Isaac direct dependency 전환).

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
