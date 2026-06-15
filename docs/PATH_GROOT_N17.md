# PATH GR00T-N1.7 — NVIDIA Isaac-GR00T 네이티브 정책 (finetune + serve + sim 폐루프)

SO-101 pick-place 용 **GR00T-N1.7** 정책을 NVIDIA Isaac-GR00T 네이티브 스택으로 학습/서빙하고,
기존 Isaac Sim bridge + ROS VLA + gRPC policy-server 폐루프에 **무수정으로** 끼워 넣는 경로.

> 한눈에: `gr00t` 이미지가 NVIDIA 네이티브(학습·ZMQ 추론)를 담당하고, `policy-server` 이미지의
> `policy-server-groot` bridge 모드가 gRPC↔ZMQ 를 잇는다. `vla_policy_node` 와 NVIDIA repo 는 손대지 않는다.

## 검증 상태 (2026-06-15)

🟢 **전 파이프라인 통합·폐루프 라이브 검증 완료** (1k-step 스모크 체크포인트):

| 단계 | 상태 | 근거 |
|---|---|---|
| convert (v3→v2.1+modality.json) | ✅ | `datasets/groot/.../meta/` 6파일, 240ep·143072frame·6dim·3cam |
| finetune 1k | ✅ | `checkpoint-1000/`, 403s, train_loss 0.82, trainable 1.62B(projector+diffusion) |
| gr00t zmq-server | ✅ | `Server ready — listening 0.0.0.0:5555`, GPU ~6.6GB |
| policy-server-groot bridge | ✅ | e2e smoke gRPC→bridge→ZMQ→GR00T action(16,6) roundtrip 438ms |
| 폐루프 (isaac+vla-ros) | ✅ | vla-ros roundtrip ~100ms, bridge chunk 170+, robot 구동, livestream :8011 |

⚠ **grasp 성공률 0%** — 1k-step 은 plumbing 스모크. 거동은 보이나 미숙(undertrained). **본 학습 필요**(`TRAIN_STEPS`↑, 예: 20k~). 통합/단위·물리 정합은 완전 검증됨.

## 왜 별도 이미지인가

| | policy-server:0.5.1 (SmolVLA) | gr00t-n17:ea (GR00T-N1.7) |
|---|---|---|
| Python / transformers | 3.12 / **5.3.0** | 3.10 / **4.57.3** |
| 추론 프로토콜 | LeRobot gRPC (`AsyncInferenceStub` :8080) | NVIDIA 네이티브 ZMQ (msgpack :5555) |
| 정책 로드 | `lerobot` GrootPolicy(=N1.5 전용) | `gr00t.policy.Gr00tPolicy`(N1.7) |

transformers 4.57 vs 5.3 충돌로 한 venv 공존 불가 → N1.7 은 **NVIDIA repo(`ref_repos/Isaac-GR00T`)를 무수정으로 빌드한 별도 이미지**에서 돌린다. 우리 자산은 bind-mount(`/host`) + entrypoint override 로만 주입한다(업그레이드 = repo 재-pull).

## 아키텍처

```
[학습]
  HF v3 dataset ──(gr00t: convert)──▶ LeRobot v2.1 + meta/modality.json
                 ──(gr00t: finetune)─▶ examples/finetune.sh → checkpoint-<step>
                     base = nvidia/GR00T-N1.7-3B, embodiment = NEW_EMBODIMENT,
                     modality = configs/groot/so101_config.py (3-cam)

[추론 폐루프]
  Isaac Sim bridge ──/isaac_joint_states · /camera/{top,wrist,front}/image_raw──▶ vla_policy_node (불변)
       ▲                                                                              │ gRPC :8080
       │ /isaac_joint_commands                                                        ▼
       └──────────────────────────────── policy-server  [policy-server-groot bridge]
                                          GrootBridgeServer(PolicyServer 서브클래스)
                                          raw lerobot obs ↔ GR00T modality dict 변환
                                                                            │ ZMQ msgpack :5555
                                                                            ▼
                                          gr00t  [zmq-server]  run_gr00t_server.py → Gr00tPolicy(N1.7)
```

추론 시 컨테이너 2개(policy-server bridge + gr00t zmq), localhost ZMQ 1홉. `network_mode: host`.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `docker/docker-compose.yaml` `gr00t` 서비스 | NVIDIA `ref_repos/Isaac-GR00T/docker/Dockerfile` 무수정 빌드. `/host` 에 자산 마운트 + entrypoint override |
| `docker/gr00t-entrypoint.sh` | 모드 분기: `convert` / `finetune` / `zmq-server` / `bash` / `python` |
| `configs/groot/so101_config.py` | NEW_EMBODIMENT modality config (SO100 베이스 + 3-cam top/wrist/front). finetune 시 체크포인트에 baked |
| `configs/groot/so101_modality.json` | 변환 데이터셋 `meta/modality.json` 으로 주입(state/action 0:5+5:6, video 3-cam, annotation) |
| `scripts/policy_server_groot_bridge.py` | `policy-server-groot` 모드 본체. `PolicyServer` 서브클래스 — gRPC 컨트랙트 유지, 추론을 ZMQ 위임 |
| `docker/policy-entrypoint.sh` `policy-server-groot` | bridge 기동(env→CLI 매핑) |
| `env/groot_n17.env` | 프로필. `.env` 의 `POLICY_PROFILE=groot_n17` 로 활성화 |

## obs / action 변환 (bridge)

| | LeRobot (vla_policy_node ↔ gRPC) | GR00T (ZMQ get_action) |
|---|---|---|
| state | `observation.state` (6,): `[shoulder_pan..wrist_roll]` deg + `gripper` [0,100] | `state.single_arm` (1,1,5) + `state.gripper` (1,1,1) f32 |
| video | bare 키 `top`/`wrist`/`front` (480×640×3 uint8 HWC) | `video.{front,wrist,top}` (1,1,H,W,3) uint8 |
| language | `task` (str) | `language["annotation.human.task_description"] = [[task]]` |
| action | `(16,6)` f32: `[arm deg(절대)] + [gripper [0,100]]` | `single_arm` (1,16,5) + `gripper` (1,16,1) f32 (절대=데이터셋 네이티브) |

- GR00T action 의 `single_arm` rep=RELATIVE 는 **내부 표현**일 뿐, `get_action()` 반환은 `unapply_action` 이 현재 state 로 절대값을 복원한다 → bridge 는 그대로 통과(delta 누적 없음).
- RTC 미적용(GR00T 는 `init_rtc_processor` 없음). 노드 client-side `weighted_average` 청크 블렌딩은 그대로 동작.

## 실행 — 전체 파이프라인 기동 (검증된 절차)

전제: `.env` 의 `POLICY_PROFILE=groot_n17`, `HF_USER`/`HF_TOKEN` 설정. 서버 GPU 1장 공유.

### 0) 이미지 빌드 (1회)

**gr00t** (NVIDIA repo 무수정, flash-attn 은 wheel — 소스빌드 없음):
```bash
docker compose -f docker/docker-compose.yaml build gr00t
```

**policy-server**: from-scratch 빌드는 flash-attn 을 sm_120(Blackwell)으로 **소스 컴파일**하느라 ~30-45분
(공식 flash-attn wheel 은 sm_120 미포함). 이미 sm_120 flash-attn 이 든 `policy-server:0.5.1` 이 있으면
bridge 의존(pyzmq/msgpack) + 갱신 entrypoint 만 얹는 **overlay**(수초) 권장:
```bash
docker build -t policy-server:0.5.1 -f - docker/ <<'E'
FROM policy-server:0.5.1
COPY policy-entrypoint.sh /usr/local/bin/entrypoint.sh
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && uv pip install --python /opt/venv/bin/python "pyzmq>=27.1.0" "msgpack>=1.0.0"
E
```
> 기존 sm_120 이미지가 없으면 `docker compose build policy-server`(from-scratch, 느림). bridge 자체는
> flash-attn 미사용(로컬 정책 로드 안 함) — flash-attn 은 SmolVLA 로컬 추론용.

### 1) 데이터 변환 (v3 → v2.1 + modality.json, 1회)
```bash
docker compose --env-file .env -f docker/docker-compose.yaml run --rm gr00t convert
```
확인: `datasets/groot/${HF_USER}/so101_sim_pick_cube/meta/` 에 `info.json`(v2.1)·`episodes.jsonl`·`tasks.jsonl`·`stats.json`·**`modality.json`**.
> ⚠ 1회성(변환 후 v2.1 이라 재실행 시 검증 실패). 재변환 = `datasets/groot/<repo>` 삭제 후 재실행.

### 2) finetune
```bash
# 1k 스모크 (48GB 단일 GPU): batch 8 권장(~39GB). env 기본 BATCH_SIZE=32 는 단일 GPU OOM 위험.
docker compose --env-file .env -f docker/docker-compose.yaml run --rm -e BATCH_SIZE=8 gr00t finetune
```
산출: `outputs/train/so101_groot_n17_pick_cube/checkpoint-<step>/`. (1k → `checkpoint-1000`)
- 본 학습 = `TRAIN_STEPS` 상향(`env/groot_n17.env` 또는 `-e TRAIN_STEPS=20000`) + `GROOT_CHECKPOINT` 의 step 갱신.
- 다중 GPU = `-e NUM_GPUS=N`(finetune.sh 가 torchrun 자동 분기).
- 기본 tune = projector + diffusion head (LLM·vision freeze, trainable ~1.62B). 1k @ batch8 ≈ 7분(Blackwell).

### 3) 추론 파이프라인 기동 (순서 중요)
```bash
# ① GR00T ZMQ 서버 (checkpoint 로드, :5555)
docker compose --env-file .env -f docker/docker-compose.yaml up -d gr00t
docker logs -f so101_gr00t      # "Server ready — listening on 0.0.0.0:5555" 대기

# ② gRPC↔ZMQ bridge (gRPC :8080)
docker compose --env-file .env -f docker/docker-compose.yaml run -d --name policy_groot \
  policy-server policy-server-groot
docker logs policy_groot        # "GrootBridgeServer ... 기동 | gRPC 0.0.0.0:8080"

# ③ VLA ROS 노드 = policy client (gRPC :8080 접속 + ROS 토픽 대기)
docker compose --env-file .env -f docker/docker-compose.yaml up -d vla-ros
docker logs so101_vla_ros       # "sent instructions (type=groot_n17 ...)" + "VLA node up"

# ④ Isaac Sim bridge (host uv) — ⚠ 반드시 .sh 래퍼로 기동
#    (LD_LIBRARY_PATH=isaacsim.ros2.bridge/jazzy/lib 필요. .py 직접 실행 시 ros2 bridge ext
#     미로드 → "Could not create node ... isaacsim.ros2.bridge.ROS2Context")
#  (a) headless 정량 eval:
scripts/sim/run_cube_desk_ros_bridge.sh --eval 10 --eval_warmup 30 --headless --enable_cameras --seed 0
#  (b) livestream 관전 (WebRTC):
PUBLIC_IP=10.10.16.147 scripts/sim/run_cube_desk_ros_bridge.sh \
  --livestream 1 --enable_cameras --seed 0
#     브라우저: http://10.10.16.147:8011/streaming/webrtc-client?server=10.10.16.147
#     또는 Omniverse Streaming Client → 10.10.16.147  (signaling :49100, webrtc client :8011)
```

**확인 지표** (폐루프 동작 중):
- `so101_vla_ros` 로그: `gRPC roundtrip ms (n=10): total ...~100ms` (obs 송신+action 수신 반복)
- `policy_groot` 로그: `[GR00T] chunk #N | infer ~90ms | shape (16, 6)` (GR00T 위임)
- eval = `outputs/vla_eval.json`. SmolVLA 와 동일 데이터라 직접 A/B.

### 4) 정지
```bash
# isaac: 래퍼/python child 직접 종료(TaskStop 만으론 좀비 → 49100 다중 LISTEN → WebRTC 검은화면)
docker rm -f policy_groot so101_vla_ros
docker compose -f docker/docker-compose.yaml stop gr00t
```

## 함정 / 검증 메모

- **modality.json 주입**: `convert_v3_to_v2.py` 는 info/episodes/tasks/stats 만 만들고 `modality.json` 은 안 만든다 → entrypoint 가 `configs/groot/so101_modality.json` 을 복사. 누락 시 finetune `KeyError`.
- **convert 환경 (non-root 함정)**: gr00t main venv 에 lerobot 없음 → repo 별도 uv 프로젝트(`scripts/lerobot_conversion`, lerobot 커밋 핀 ~0.4.0)의 **deps 만** 직접 설치한다(로컬 프로젝트 빌드는 `egg-info` 를 root 소유 `/workspace` 에 쓰려다 `Permission denied` → 금지). uv env/cache 는 host-writable `/host/outputs/.uv-cache`(named volume 아님).
- **non-root HOME/캐시**: gr00t 컨테이너 UID 1000 은 `/` 에 못 쓴다 → triton/wandb/matplotlib 가 `/.triton` 등에 `PermissionError`. entrypoint 가 `HOME=/host/outputs/.gr00t-home` + `XDG_CACHE_HOME`/`TRITON_CACHE_DIR`/`MPLCONFIGDIR`/`WANDB_DIR` 를 host-writable 로 강제.
- **entrypoint shift**: `gr00t-entrypoint.sh` 는 모드 인자 캡처 후 `shift` 한다(안 하면 `finetune.sh` 가 `finetune` 을 `Unknown argument` 로 거부).
- **finetune.sh 인자 규약**: 튜닝값은 **env var**(NUM_GPUS/MAX_STEPS/GLOBAL_BATCH_SIZE/DATALOADER_NUM_WORKERS/USE_WANDB)로, model/dataset/embodiment/modality/output 만 flag. 미인식 flag 거부 → entrypoint 가 `BATCH_SIZE`→GLOBAL_BATCH_SIZE 등으로 매핑.
- **flash-attn sm_120 + overlay**: 공식 wheel 은 Blackwell sm_120 미포함 → from-scratch 는 소스빌드(느림). 기존 sm_120 이미지에 overlay 로 bridge 의존만 추가(§실행 0). bridge 는 flash-attn 미사용.
- **Isaac bridge 는 `.sh` 래퍼 전용**: `run_cube_desk_ros_bridge.sh` 가 `LD_LIBRARY_PATH=…/isaacsim.ros2.bridge/jazzy/lib` 를 export 해야 ros2 bridge ext 가 로드된다(.py 직접 실행 시 `ROS2Context unrecognized`).
- **bridge 가로채기**: `PolicyServer._predict_action_chunk(observation_t)` 오버라이드(raw `TimedObservation` 수신) → lerobot 전/후처리 우회. `SendPolicyInstructions` 오버라이드로 lerobot 정책 로드 생략(SUPPORTED_POLICIES 에 groot_n17 없음 → super 호출 금지).
- **gripper offset**: `GRIPPER_CMD_OFFSET=0.2`(sim cuRobo 데이터 전용, SmolVLA 와 동일). 실기기 모델은 0.
- **GPU 경합**: 서버 GPU 1장 공유 — finetune·zmq-server·Isaac 동시 실행 시 VRAM 주의. 빌드 2개 동시도 부하폭주(flash-attn ×2)로 프로세스 죽음 → 직렬화.
- 새 에러 진단·수정 성공 시 `docs/TROUBLESHOOTING.md` 에 기록(운영 규칙).
