#!/usr/bin/env python3
"""LeRobot v3 dataset 한 에피소드를 Isaac Sim bridge 로 replay (lerobot-replay 의 sim 판).

dataset 의 ``action`` (또는 ``observation.state``) 프레임을 fps 페이싱으로
``/isaac_joint_commands`` (sensor_msgs/JointState, rad) 에 publish 한다. 실행 중인
bridge(``run_cube_desk_ros_bridge.py``)의 ArticulationController 가 그대로 적용 →
**기본 큐브/그릇 배치**(bridge 를 ``--dr`` 없이 띄운 고정 위치)에서 녹화된
pick-and-place 가 재현된다. 변환은 ``vla_policy_node`` 명령 경로와 동일 단일 소스
(``so101_contract.feature_codec``): arm degrees·gripper[0,100] → sim joint rad.

vla-ros 가 pyarrow<19 + huggingface_hub 를 갖춰(Dockerfile.vla_ros) ``--dataset`` 으로
HF parquet 을 직접 읽으므로 한 명령으로 끝난다(rclpy 도 vla-ros 에 있음). lerobot·pandas
불요(pyarrow.compute 로 필터/정렬). ``--export``/``--npz`` 는 사전 로드용으로 남겨둔다.

사용 (bridge 가 떠 있는 상태에서 vla-ros 한 줄):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --no-deps vla-ros \\
      python3 /workspace/scripts/inference/replay_dataset_to_bridge.py \\
      --dataset taehunkim/so101_pick_cube_test --wait_for_subscriber

선택(사전 export): host uv 에서 npz 로 뽑아 ``--npz`` 로 publish (네트워크/HF 불요 환경용):
  uv run python scripts/inference/replay_dataset_to_bridge.py \\
      --dataset taehunkim/so101_pick_cube_test --export scratch/replay.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# so101_contract 는 host(editable src) · vla-ros(PYTHONPATH=/workspace/src) 모두에서 import 가능.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from so101_contract.feature_codec import (  # noqa: E402
    CAMERA_KEYS,
    FPS as DEFAULT_FPS,
    JOINT_FEATURE_NAMES,
    SO101_JOINT_ORDER,
    clamp_sim_joint_radians,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from so101_contract.leader_calibration import (  # noqa: E402
    SO101_FOLLOWER_USD_JOINT_LIMITS,
    real_leader_to_sim_radians,
)

JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"
_RAD_TO_DEG = 180.0 / np.pi


def _check_usd_limits(actions: np.ndarray) -> None:
    """LeRobot 액션을 sim joint degree 로 변환해 USD joint limit 초과 프레임을 경고한다.

    부호/영점이 맞아도(좌표계 정합) 실기기 가동범위가 sim USD 모델보다 넓으면 그 프레임은
    물리 articulation 이 clamp 한다(arm 1:1 degree·gripper affine, leader_calibration 한계표).
    """
    sim_deg = policy_feature_to_sim_joint_radians(actions) * _RAD_TO_DEG  # (T, 6)
    n = actions.shape[0]
    for i, j in enumerate(SO101_JOINT_ORDER):
        lo, hi = SO101_FOLLOWER_USD_JOINT_LIMITS[j]
        col = sim_deg[:, i]
        over = int(((col < lo) | (col > hi)).sum())
        if over:
            print(f"[replay] ⚠ {j}: {over}/{n} 프레임이 USD limit [{lo:.0f},{hi:.0f}]° 초과 "
                  f"(데이터 {col.min():.1f}..{col.max():.1f}°) → 물리 clamp", flush=True)


def _load_actions(repo_id: str, episode: int, source: str):
    """HF LeRobot v3 dataset 에서 한 에피소드의 (T, 6) 액션을 LeRobot 단위로 로드.

    LeRobotDataset 버전 의존을 피하려 parquet 을 pyarrow 로 직접 읽는다.
    반환: (actions[T,6] float32, fps:int). action 컬럼은 info.json 의 names 순서를
    SO101_JOINT_ORDER 로 재정렬한다(컬럼 순서가 달라도 안전).
    """
    import json

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    col = "action" if source == "action" else "observation.state"

    info = json.load(open(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")))
    fps = int(info.get("fps", DEFAULT_FPS))
    names = list(info["features"][col]["names"])  # 예: ['shoulder_pan.pos', ...]

    api = HfApi()
    data_files = sorted(
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if not data_files:
        raise RuntimeError(f"{repo_id}: data/*.parquet 없음")

    import pyarrow as pa
    import pyarrow.compute as pc

    tables = []
    for rel in data_files:
        local = hf_hub_download(repo_id, rel, repo_type="dataset")
        t = pq.read_table(local, columns=[col, "episode_index", "frame_index"])
        t = t.filter(pc.equal(t["episode_index"], episode))
        if t.num_rows:
            tables.append(t)
    if not tables:
        raise RuntimeError(f"{repo_id}: episode {episode} 프레임 없음")
    tbl = pa.concat_tables(tables)
    tbl = tbl.take(pc.sort_indices(tbl, sort_keys=[("frame_index", "ascending")]))
    raw = np.asarray(tbl[col].to_pylist(), dtype=np.float32)  # (T, 6) info.json names 순서
    # names → SO101_JOINT_ORDER 재정렬.
    want = [f"{j}.pos" for j in SO101_JOINT_ORDER]
    idx = [names.index(w) for w in want]
    actions = raw[:, idx]
    print(f"[replay] loaded {repo_id} ep{episode} '{col}': {actions.shape} @ {fps}fps", flush=True)
    print(f"[replay] LeRobot 단위 per-joint min={actions.min(0).round(1)} "
          f"max={actions.max(0).round(1)}", flush=True)
    _check_usd_limits(actions)
    return actions, fps


def _export(args) -> None:
    actions, fps = _load_actions(args.dataset, args.episode, args.source)
    out = os.path.abspath(args.export)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, actions=actions, fps=fps, source=args.source,
             dataset=args.dataset, episode=args.episode)
    print(f"[replay] EXPORT → {out} ({actions.shape[0]} frames)", flush=True)


def _load_npz(path: str):
    d = np.load(path, allow_pickle=False)
    actions = d["actions"].astype(np.float32)
    fps = int(d["fps"])
    print(f"[replay] loaded npz {path}: {actions.shape} @ {fps}fps", flush=True)
    _check_usd_limits(actions)
    return actions, fps


def _publish(args) -> None:
    if args.npz:
        actions, fps = _load_npz(args.npz)
    elif args.dataset:
        actions, fps = _load_actions(args.dataset, args.episode, args.source)
    else:
        raise SystemExit("publish: --npz 또는 --dataset 필요")
    if args.fps:
        fps = args.fps

    # action → sim joint rad 변환. 두 계약:
    #  codec       = feature_codec(arm deg×π/180 1:1, gripper affine) — 정책 출력용(sim 학습공간).
    #  calibration = leader_calibration(실기기 정규화 [-100,100]→USD-degree per-joint scale+offset
    #                + gripper affine) — 실기기 녹화 데이터 replay 용. arm 범위차(pan 1.1·wrist_roll
    #                1.6 등) 보정 → "real 보다 덜 움직임" 해소.
    _convert = (real_leader_to_sim_radians if args.arm_mapping == "calibration"
                else policy_feature_to_sim_joint_radians)
    targets_rad = np.stack([
        clamp_sim_joint_radians(_convert(row)) for row in actions
    ]).astype(np.float32)
    print(f"[replay] arm_mapping={args.arm_mapping} "
          f"({'실기기 정규화→sim remap' if args.arm_mapping == 'calibration' else 'feature_codec 1:1 deg'})",
          flush=True)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("dataset_replay")
    pub = node.create_publisher(JointState, args.topic, 10)
    names = list(SO101_JOINT_ORDER)
    record = bool(args.record_dir)

    # probe·record 둘 다 achieved state(/isaac_joint_states) 구독.
    achieved = {"rad": None}
    if args.probe_tracking or record:
        def _state_cb(m: JointState) -> None:
            idx = {n: i for i, n in enumerate(m.name)}
            try:
                achieved["rad"] = np.array([m.position[idx[j]] for j in SO101_JOINT_ORDER],
                                           dtype=np.float32)
            except (KeyError, IndexError):
                pass
        node.create_subscription(JointState, args.joint_states_topic, _state_cb, 10)
    if args.probe_tracking:
        node.get_logger().info(f"추종 probe ON: {args.joint_states_topic}(achieved) vs target 기록")

    # record: 3 카메라 구독 + sim observation(state·images)·action 을 LeRobot 단위로 프레임당 기록.
    images: dict = {}
    if record:
        import cv2  # noqa: PLC0415
        from cv_bridge import CvBridge  # noqa: PLC0415 (vla-ros 전용)
        from sensor_msgs.msg import Image  # noqa: PLC0415
        _bridge = CvBridge()

        def _mk_img_cb(cam: str):
            def _cb(m) -> None:
                try:
                    images[cam] = _bridge.imgmsg_to_cv2(m, desired_encoding="rgb8")
                except Exception:  # noqa: BLE001
                    pass
            return _cb
        _cam_topics = {"top": "/camera/top/image_raw", "wrist": "/camera/wrist/image_raw",
                       "front": "/camera/front/image_raw"}
        for _cam, _t in _cam_topics.items():
            node.create_subscription(Image, _t, _mk_img_cb(_cam), 10)
        rec_dir = os.path.abspath(args.record_dir)
        for _cam in CAMERA_KEYS:
            os.makedirs(os.path.join(rec_dir, _cam), exist_ok=True)
        rec_actions: list = []
        rec_states: list = []
        rec_k = 0
        node.get_logger().info(f"record ON → {rec_dir} (state+action+3cam, LeRobot 단위)")

    if args.wait_for_subscriber:
        node.get_logger().info(f"{args.topic} 구독자(bridge) 대기...")
        while rclpy.ok() and pub.get_subscription_count() == 0:
            time.sleep(0.1)
        node.get_logger().info("구독자 연결됨")
    if record:  # 카메라+state 첫 프레임 도착 대기(최대 20s)
        node.get_logger().info("record: 카메라+state 첫 프레임 대기...")
        _t0 = time.monotonic()
        while rclpy.ok() and (achieved["rad"] is None or any(c not in images for c in CAMERA_KEYS)):
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - _t0 > 20:
                node.get_logger().warn("record: 카메라/state 대기 timeout — 진행")
                break
    if args.start_delay > 0:
        time.sleep(args.start_delay)

    period = 1.0 / max(fps, 1)
    node.get_logger().info(
        f"replay 시작: {targets_rad.shape[0]} frames @ {fps}fps → {args.topic} "
        f"(≈{targets_rad.shape[0] / fps:.1f}s){' [loop]' if args.loop else ''}"
    )
    probe_log: list[tuple[np.ndarray, np.ndarray]] = []  # (target_rad, achieved_rad)
    try:
        while rclpy.ok():
            next_t = time.monotonic()
            for i, q in enumerate(targets_rad):
                if not rclpy.ok():
                    break
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = names
                msg.position = [float(v) for v in q]
                pub.publish(msg)
                if args.probe_tracking or record:
                    rclpy.spin_once(node, timeout_sec=0.0)
                if args.probe_tracking and achieved["rad"] is not None:
                    probe_log.append((q.copy(), achieved["rad"].copy()))
                if record and achieved["rad"] is not None and all(c in images for c in CAMERA_KEYS):
                    rec_actions.append(np.asarray(actions[i], dtype=np.float32))           # 원본 action(LeRobot)
                    rec_states.append(sim_joint_radians_to_policy_feature(achieved["rad"]))  # sim state→LeRobot
                    for _cam in CAMERA_KEYS:
                        cv2.imwrite(os.path.join(rec_dir, _cam, f"{rec_k:06d}.png"),
                                    cv2.cvtColor(images[_cam], cv2.COLOR_RGB2BGR))
                    rec_k += 1
                next_t += period
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
            node.get_logger().info("replay 1회 완료")
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.probe_tracking and probe_log:
            _report_tracking(node, probe_log, fps)
        if record and rec_k > 0:
            np.savez(os.path.join(rec_dir, "frames.npz"),
                     action=np.stack(rec_actions), state=np.stack(rec_states),
                     fps=int(fps), task=args.record_task,
                     joint_names=list(JOINT_FEATURE_NAMES), cameras=list(CAMERA_KEYS))
            node.get_logger().info(f"record DONE: {rec_k} frames → {rec_dir} (frames.npz + {CAMERA_KEYS} png)")
        node.destroy_node()
        rclpy.shutdown()


def _report_tracking(node, probe_log, fps: int) -> None:
    """명령 vs 실제 추종 오차 리포트(deg). 정지구간(마지막 1s)=steady-state, 전체=dynamic."""
    tgt = np.stack([t for t, _ in probe_log]) * _RAD_TO_DEG   # (N, 6) deg
    ach = np.stack([a for _, a in probe_log]) * _RAD_TO_DEG
    n = tgt.shape[0]
    tail = max(1, min(fps, n))                                # 마지막 ~1s = 자세 정지구간
    steady = (tgt[-tail:] - ach[-tail:]).mean(axis=0)         # signed: +면 achieved 가 부족(under-shoot)
    dyn = np.abs(tgt - ach).max(axis=0)                       # motion 중 최대 추종 오차
    node.get_logger().info(f"=== 추종 probe (N={n}) — target vs achieved [deg] ===")
    node.get_logger().info("  joint          steady(target-achieved)   max|err|(dynamic)")
    for i, j in enumerate(SO101_JOINT_ORDER):
        node.get_logger().info(f"  {j:14s} {steady[i]:+8.2f}                  {dyn[i]:7.2f}")
    node.get_logger().info(
        "  해석: steady≈0(±1°)이면 sim 이 target 도달 → 변환/스케일 문제(데이터 단위). "
        "steady 가 크게 +(under-shoot)면 actuator PD 추종 지연."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="HF dataset repo_id (예: taehunkim/so101_pick_cube_test)")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--source", choices=["action", "state"], default="action",
                    help="replay 할 컬럼: action(명령, 기본) 또는 state(observation.state)")
    ap.add_argument("--export", metavar="PATH", help="dataset → npz 저장(로드 환경에서 실행, rclpy 불요)")
    ap.add_argument("--npz", metavar="PATH", help="publish 할 npz(--export 산출물)")
    ap.add_argument("--topic", default=JOINT_COMMANDS_TOPIC)
    ap.add_argument("--fps", type=int, default=0, help="페이싱 fps override(0=dataset fps)")
    ap.add_argument("--loop", action="store_true", help="에피소드 반복 재생")
    ap.add_argument("--wait_for_subscriber", action="store_true",
                    help="bridge 가 토픽 구독할 때까지 대기 후 시작")
    ap.add_argument("--start_delay", type=float, default=0.0, help="시작 전 지연(초)")
    ap.add_argument("--probe_tracking", action="store_true",
                    help="명령 vs 실제(/isaac_joint_states) 추종 오차 측정 — 변환/스케일 vs PD 추종 판별")
    ap.add_argument("--joint_states_topic", default="/isaac_joint_states",
                    help="probe 가 구독할 achieved joint state 토픽")
    ap.add_argument("--arm_mapping", choices=["codec", "calibration"], default="codec",
                    help="action→sim 변환: codec(feature_codec 1:1 deg, 기본) 또는 "
                         "calibration(leader_calibration 정규화→sim remap, 실기기 녹화 replay 용)")
    ap.add_argument("--record_dir", metavar="DIR",
                    help="replay 중 sim observation(state·3cam)+action 을 LeRobot 단위로 DIR 에 기록 "
                         "(frames.npz + {top,wrist,front}/*.png). append_sim_episode.py 가 dataset 에 추가.")
    ap.add_argument("--record_task", default="pick up the cube and place it in the bowl",
                    help="기록 episode 의 task 문자열")
    args = ap.parse_args()

    if args.export:
        if not args.dataset:
            raise SystemExit("--export 에는 --dataset 필요")
        _export(args)
    else:
        _publish(args)


if __name__ == "__main__":
    main()
