#!/usr/bin/env python3
"""SO-101 canonical codec, action queue, replay snapshot의 P0 검증."""

from __future__ import annotations

import importlib
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "so101_vla_policy"))

from so101_contract.action_queue import ActionChunkQueue  # noqa: E402
from so101_contract.feature_codec import (  # noqa: E402
    CODEC_VERSION,
    policy_feature_to_sim_joint_radians,
    sim_joint_radians_to_policy_feature,
)
from so101_contract.policy_snapshot import (  # noqa: E402
    load_policy_io_snapshot,
    save_policy_io_snapshot,
)


def _assert_close(actual, expected, *, message: str, atol: float = 1e-5) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def validate_codec() -> None:
    sim_deg = np.asarray(
        [
            [-90.0, -45.0, 0.0, 45.0, 90.0, -10.0],
            [0.0, 10.0, -20.0, 30.0, -40.0, 45.0],
            [90.0, 45.0, 0.0, -45.0, -90.0, 100.0],
        ],
        dtype=np.float32,
    )
    sim_rad = np.deg2rad(sim_deg).astype(np.float32)
    feature = sim_joint_radians_to_policy_feature(sim_rad)
    _assert_close(feature[:, 5], [0.0, 50.0, 100.0], message="gripper endpoint/midpoint")
    _assert_close(feature[:, :5], sim_deg[:, :5], message="arm degree contract")
    _assert_close(
        policy_feature_to_sim_joint_radians(feature),
        sim_rad,
        message="feature -> sim -> feature round-trip",
    )


def validate_ros_adapter() -> None:
    units = importlib.import_module("so101_vla_policy.units")
    sim_rad = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, np.deg2rad(45.0)], dtype=np.float32)
    expected = sim_joint_radians_to_policy_feature(sim_rad)
    _assert_close(units.to_lerobot_units(sim_rad), expected, message="ROS adapter encoder")
    _assert_close(units.from_lerobot_units(expected), sim_rad, message="ROS adapter decoder")
    if units.CODEC_VERSION != CODEC_VERSION:
        raise AssertionError(f"ROS codec version mismatch: {units.CODEC_VERSION} != {CODEC_VERSION}")


def validate_action_queue() -> None:
    queue = ActionChunkQueue("weighted_average")
    queue.merge(
        [
            (0, np.zeros(6, dtype=np.float32)),
            (1, np.zeros(6, dtype=np.float32)),
            (2, np.zeros(6, dtype=np.float32)),
        ]
    )
    timestep, _ = queue.pop_next()
    if timestep != 0:
        raise AssertionError(f"first timestep mismatch: {timestep}")

    queue.merge(
        [
            (1, np.ones(6, dtype=np.float32)),
            (2, np.ones(6, dtype=np.float32)),
            (3, np.ones(6, dtype=np.float32)),
        ]
    )
    if queue.timesteps() != [1, 2, 3]:
        raise AssertionError(f"merged timesteps mismatch: {queue.timesteps()}")
    timestep, action = queue.pop_next()
    if timestep != 1:
        raise AssertionError(f"aggregated timestep mismatch: {timestep}")
    _assert_close(action, np.full(6, 0.7, dtype=np.float32), message="weighted_average semantics")
    if queue.ready_to_send_observation(0.5):
        raise AssertionError("queue refill threshold became ready too early")
    queue.pop_next()
    if not queue.ready_to_send_observation(0.5):
        raise AssertionError("queue refill threshold did not become ready")
    queue.pop_next()
    if not queue.observation_must_go():
        raise AssertionError("empty queue must_go transition mismatch")
    queue.mark_observation_sent(True)
    if queue.observation_must_go():
        raise AssertionError("must_go was not cleared after observation send")

    try:
        import torch
        from lerobot.async_inference.configs import get_aggregate_function
        from lerobot.async_inference.helpers import TimedAction
        from lerobot.async_inference.robot_client import RobotClient
    except ImportError:
        return
    old = torch.zeros(6)
    new = torch.ones(6)
    for name in ("weighted_average", "latest_only", "average", "conservative"):
        expected = get_aggregate_function(name)(old, new).numpy()
        actual = ActionChunkQueue(name)
        actual.merge([(0, old.numpy())])
        actual.merge([(0, new.numpy())])
        _, merged = actual.pop_next()
        _assert_close(merged, expected, message=f"upstream aggregate parity: {name}")

        upstream = RobotClient.__new__(RobotClient)
        upstream.action_queue = Queue()
        upstream.action_queue_lock = threading.Lock()
        upstream.latest_action = 0
        upstream.latest_action_lock = threading.Lock()
        for step in (1, 2):
            upstream.action_queue.put(TimedAction(timestamp=0.0, timestep=step, action=old.clone()))
        incoming = [
            TimedAction(timestamp=1.0, timestep=step, action=new.clone())
            for step in (1, 2, 3)
        ]
        RobotClient._aggregate_action_queues(
            upstream,
            incoming,
            get_aggregate_function(name),
        )
        expected_queue = [
            (item.get_timestep(), item.get_action().numpy())
            for item in upstream.action_queue.queue
        ]

        actual = ActionChunkQueue(name)
        actual.merge([(0, old.numpy()), (1, old.numpy()), (2, old.numpy())])
        actual.pop_next()
        actual.merge([(1, new.numpy()), (2, new.numpy()), (3, new.numpy())])
        actual_queue = []
        while actual.has_actions():
            actual_queue.append(actual.pop_next())
        if [step for step, _ in actual_queue] != [step for step, _ in expected_queue]:
            raise AssertionError(f"upstream queue timestep parity mismatch: {name}")
        for (_, actual_action), (_, expected_action) in zip(actual_queue, expected_queue):
            _assert_close(
                actual_action,
                expected_action,
                message=f"upstream queue action parity: {name}",
            )


def validate_snapshot() -> None:
    observation = {
        "shoulder_pan.pos": 1.0,
        "shoulder_lift.pos": 2.0,
        "elbow_flex.pos": 3.0,
        "wrist_flex.pos": 4.0,
        "wrist_roll.pos": 5.0,
        "gripper.pos": 50.0,
        "top": np.zeros((4, 5, 3), dtype=np.uint8),
        "wrist": np.ones((4, 5, 3), dtype=np.uint8),
        "front": np.full((4, 5, 3), 2, dtype=np.uint8),
        "task": "pick up the cube and place it in the bowl",
    }
    action_feature = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 50.0]], dtype=np.float32)
    action_sim = policy_feature_to_sim_joint_radians(action_feature)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.npz"
        save_policy_io_snapshot(
            path,
            observation=observation,
            request_timestep=7,
            must_go=True,
            action_timesteps=[7],
            actions_feature=action_feature,
            actions_sim_rad=action_sim,
            metadata={"aggregate_fn_name": "weighted_average"},
        )
        loaded = load_policy_io_snapshot(path)

    if loaded["manifest"]["codec_version"] != CODEC_VERSION:
        raise AssertionError("snapshot codec version mismatch")
    if loaded["manifest"]["request_timestep"] != 7:
        raise AssertionError("snapshot timestep mismatch")
    _assert_close(loaded["actions_feature"], action_feature, message="snapshot action feature")
    _assert_close(loaded["actions_sim_rad"], action_sim, message="snapshot sim target")
    for camera in ("top", "wrist", "front"):
        if not np.array_equal(loaded["observation"][camera], observation[camera]):
            raise AssertionError(f"snapshot image mismatch: {camera}")


def main() -> None:
    validate_codec()
    validate_ros_adapter()
    validate_action_queue()
    validate_snapshot()
    print(f"PASS: SO-101 P0 I/O contract ({CODEC_VERSION})")


if __name__ == "__main__":
    main()
