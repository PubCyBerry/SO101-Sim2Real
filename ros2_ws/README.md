# ros2_ws — SO-101 sim VLA 추론 ROS 2 워크스페이스

Isaac Sim 폐루프 VLA 추론을 위한 **단일 패키지** 워크스페이스. Linux 서버의 Docker `vla-ros` 서비스가 이 워크스페이스를 빌드·실행한다. 실기기 제어는 ROS 가 아니라 Windows native uv(LeRobot CLI)가 담당하므로, 옛 follower MoveIt2/실기기 ROS 패키지(`so101_description`·`so101_bringup`·`feetech_ros2_driver`)와 WSL 셋업 스크립트는 **제거됐다**.

## 패키지

| 패키지 | 역할 |
|---|---|
| `so101_vla_policy` | VLA 추론 ROS 2 노드(`vla_policy_node`). `/isaac_joint_states`(isaac-sim PUB) 구독 → policy-server gRPC 추론 호출 → `/isaac_joint_commands` publish. |

구성:
- `so101_vla_policy/vla_policy_node.py` — 추론 노드 본체.
- `so101_vla_policy/units.py` — feature codec 단위 변환(어댑터; 정본은 `src/so101_contract/feature_codec.py`).
- `config/vla_policy.yaml` — 노드 파라미터(토픽명, policy-server 주소 등; 기본값은 `.env`/`env/<profile>.env` 에서 주입).
- `vendor/` — gRPC action 텐서 unpickle 용 mini-lerobot shim(컨테이너 torch CPU wheel).

## 빌드·실행 (Docker `vla-ros`)

호스트에서 직접 빌드하지 않는다. `docker/vla-ros-entrypoint.sh` 가 컨테이너 안에서:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select so101_vla_policy
source install/setup.bash
# vla_policy_node 실행 (config/vla_policy.yaml + .env 주입)
```

전체 sim 폐루프(policy-server + isaac-sim + vla-ros) 기동은 `scripts/inference/demo_vla.sh` 또는 `docker compose up` 참조. 자세한 내용은 루트 `README.md`·`AGENTS.md`.

## 산출물

`build/ install/ log/` 은 `.gitignore` (`ros2_ws/*.jsonl` 포함).
