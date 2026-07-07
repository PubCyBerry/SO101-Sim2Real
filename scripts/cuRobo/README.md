# cuRobo pick-place — 인터랙티브 SM (2-proc, 순수 ZMQ)

SO-101 pick-place를 **cuRobo collision-free planner**(planning)와 **IsaacLab env**(execution)로
분리해 돌리는 최소 구성. 두 프로세스는 ZMQ 하나로만 통신한다 (ROS·executor 없음).

```
┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
│ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py                  │
│  cuRobo v0.8 full pick-place │◀──REQ───│  pick_cube env variant           │
│  REP  tcp://*:5599           │───REP──▶│  reset(DR)→plan→env.step→check   │
└──────────────────────────────┘  :5599  └──────────────────────────────────┘
```

- 두 서비스 모두 `network_mode: host` → planner는 `tcp://*:5599`, SM은 `tcp://127.0.0.1:5599`.
- SM은 env 안에서 큐브/그릇 pose를 직접 읽고(`env.scene`) joint를 직접 구동(`env.step`)한다.
  ROS·TF·별도 executor가 필요 없다 (그래서 "isaac sim bridge"가 이 SM 하나로 접힘).

## 파일

| 파일 | 컨테이너 | 역할 |
|---|---|---|
| `curobo_batch_planner.py` | curobo-datagen | ZMQ REP planner. cube/bowl(base_link) → full pick-place 궤적(arm deg + gripper feat). 검증본 재사용. |
| `pickplace_sm.py` | isaac-sim | ZMQ REQ + IsaacLab pick_cube env. 키보드 상태머신(N/R 리셋 · B plan+manipulate), EE pose 매 step 출력. |
| `build_robot_model.py` | curobo-datagen | (기존) SO-101 cuRobo config 빌더. |

## 실행 (터미널 2개)

```bash
# 1) planner
docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
    python /workspace/scripts/cuRobo/curobo_batch_planner.py

# 2) SM (planner "ready" 로그 뜬 뒤)
docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
    python /workspace/scripts/cuRobo/pickplace_sm.py \
    --task SimToReal-SO101-PickCube-DR-v0 --livestream 2
```

- **키보드 인터랙티브**(livestream 입력, `--livestream` 필수):
  - **N** = 새 DR layout 리셋(로봇 → init) · **R** = 이전과 같은 layout 리셋 · **B** = cuRobo plan + manipulation 실행
  - R/N 은 manipulation 중에도 = 진행 동작 취소 + 로봇 pose·scene 리셋. Ctrl-C·창 닫기로 종료.
- 관전: WebRTC `:49100` (원격은 `.env`에 `LIVESTREAM=1` + `PUBLIC_IP`).
- `--task`: env variant 선택 — `PickCube-v0`(고정) · `-DR-v0`(full DR) · `-DRBase-v0` · `-Eval-v0` · `-DR-Eval-v0`.
- `--cam_eye`/`--cam_target`: viewport 카메라(env 원점 상대).

## planner로 보내는 데이터 (`plan_pickplace` 요청)

- `cubes`: `[[x, y, grasp_z, qw,qx,qy,qz]]` — **6-DOF pose**(위치+quat). planner가 quat으로 큐브 yaw face-align.
- `bowl`: `[x, y]` — planner는 place 목표로 **xy만** 소비(그릇은 컨테이너+정적 obstacle이라 6D 불요).
- `start`: `[[6 joint rad, SO101 순서]]` — **reset·settle 후 실제 robot joint**. planner가 이 자세부터 계획(arm 5개 사용).

## robot 시작 자세

`env_cfg.scene.robot.init_state.joint_pos`를 아래로 override하고 `reset_robot_joints` jitter를 0으로 →
**frame 0부터** 이 자세로 스폰(중립→init 이동 transient 제거). settle도 이 자세를 hold(zeros면 팔이 흘러내림).

```
shoulder_pan 0 · shoulder_lift -100 · elbow_flex 90 · wrist_flex 50 · wrist_roll -90 · gripper -10  (deg)
```

## 런타임 검증 포인트 (GPU 필요, 호스트에선 문법만 확인됨)

1. **action 단위** — planner row(arm deg + gripper feat)를 `policy_feature_to_sim_joint_radians`로
   sim radian 변환 후 `env.step`. `curobo_executor.py`와 동일 변환이라 정합할 것으로 보나 첫 실행에서 확인.
2. **base_link 프레임** — `subtract_frame_transforms(robot_base, cube)`로 planner 입력(base_link USD 규약)을 만든다.
   planner가 내부에서 `Rz(90)+BASE_T` 보정. 조준이 빗나가면 이 프레임부터 점검.
3. **성공 판정** — `--bowl_tol`(기본 6cm xy)은 튜닝 knob. 엄밀히는 env `object_in_container`와 맞출 것.
4. **auto-reset** — `env.step`이 termination에서 auto-reset. SM은 terminal state를 다음 step 전에 읽고 break.
5. **livestream 상호작용** — plan 대기·종료 중에도 `simulation_app.update()`로 Kit을 계속 pump해야
   zoom/drag/click·키(N/R/B)가 먹는다(안 그러면 프리즈). plan 대기·키 대기 중에도 `simulation_app.update()` pump 유지. **키 입력은 livestream 필수**(headless 는 입력 없음).
