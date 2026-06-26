"""Script to replay recorded state-machine demonstrations from LeRobot v3 dataset."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay state-machine recorded demonstrations.")
parser.add_argument("--task", type=str, default="SimToReal-SO101-PickCube-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./datasets/so101_pick_cube_demos",
    help="Directory containing LeRobot v3 dataset to replay.",
)
parser.add_argument(
    "--replay_mode",
    type=str,
    default="action",
    choices=["action", "state"],
    help="Replay mode: action replays actions, state replays joint states.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="List of episode indices to replay. Empty = replay all.",
)
parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "State machine device type used during recording, e.g. 'so101_state_machine'. "
        "If not set, inferred from the task name."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import contextlib
import os
import time
from pathlib import Path

import gymnasium as gym
import json
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401  register tasks
from sim_to_real.utils.env_utils import get_task_type
from sim_to_real.data.lerobot_units import from_lerobot_units


class RateLimiter:
    """Rate limiter for simulation stepping."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Sleep to maintain target Hz rate."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def apply_damping(env, task_type: str):
    """Apply joint damping each step to match state-machine recording behavior."""
    if task_type == "so101_state_machine":
        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)


class LeRobotV3ReplayLoader:
    """Minimal LeRobot v3 dataset loader for replay."""

    def __init__(self, dataset_dir: str):
        self.root = Path(dataset_dir)
        self.episodes = []
        self._load_metadata()

    def _load_metadata(self):
        """Load episode metadata and action/state data."""
        import pyarrow.parquet as pq

        episodes_file = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        data_file = self.root / "data" / "chunk-000" / "file-000.parquet"

        if not episodes_file.exists() or not data_file.exists():
            raise FileNotFoundError(f"Dataset files not found in {self.root}")

        episodes_table = pq.read_table(episodes_file)
        data_table = pq.read_table(data_file)

        # Convert to numpy arrays.
        episode_indices = episodes_table["episode_index"].to_numpy()
        dataset_from_indices = episodes_table["dataset_from_index"].to_numpy()
        dataset_to_indices = episodes_table["dataset_to_index"].to_numpy()

        actions = data_table["action"].to_numpy()
        states = data_table["observation.state"].to_numpy()

        for ep_idx, from_idx, to_idx in zip(episode_indices, dataset_from_indices, dataset_to_indices):
            self.episodes.append({
                "index": int(ep_idx),
                "actions": [np.array(a, dtype=np.float32) for a in actions[from_idx:to_idx]],
                "states": [np.array(s, dtype=np.float32) for s in states[from_idx:to_idx]],
            })

    def get_num_episodes(self) -> int:
        return len(self.episodes)

    def get_episode(self, index: int) -> dict:
        if index >= len(self.episodes):
            return None
        return self.episodes[index]


def main():
    """Replay recorded demonstrations."""
    if not os.path.exists(args_cli.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args_cli.dataset_dir}")

    loader = LeRobotV3ReplayLoader(args_cli.dataset_dir)
    episode_count = loader.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    episode_indices_to_replay = args_cli.select_episodes or list(range(episode_count))
    num_envs = args_cli.num_envs

    task_type = get_task_type(args_cli.task, args_cli.task_type)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.use_teleop_device(task_type)
    env_cfg.recorders = None
    env_cfg.terminations = None

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Disable gravity for robot links to match state-machine recording behavior.
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    idle_action = torch.zeros(env.action_space.shape)

    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    rate_limiter = RateLimiter(args_cli.step_hz)
    replayed_episode_count = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Per-env episode state: index into episode's action/state list.
            env_episode_step = {i: 0 for i in range(num_envs)}
            env_episode_data = {i: None for i in range(num_envs)}

            has_next_action = True
            while has_next_action:
                actions = idle_action.clone()
                has_next_action = False

                for env_id in range(num_envs):
                    ep_data = env_episode_data[env_id]
                    ep_step = env_episode_step[env_id]

                    # Check if current episode has more steps.
                    if ep_data is not None and ep_step < len(ep_data["actions"]):
                        if args_cli.replay_mode == "state":
                            action = ep_data["states"][ep_step]
                        else:
                            action = ep_data["actions"][ep_step]
                        actions[env_id] = torch.from_numpy(action).to(idle_action.dtype)
                        env_episode_step[env_id] += 1
                        has_next_action = True
                    else:
                        # Current episode exhausted, load next.
                        next_ep_index = None
                        while episode_indices_to_replay:
                            next_ep_index = episode_indices_to_replay.pop(0)
                            if next_ep_index < episode_count:
                                break
                            next_ep_index = None

                        if next_ep_index is not None:
                            replayed_episode_count += 1
                            ep_data = loader.get_episode(next_ep_index)
                            print(f"{replayed_episode_count:4}: Loading episode #{next_ep_index} to env_{env_id}")
                            env_episode_data[env_id] = ep_data
                            env_episode_step[env_id] = 0

                            # Load first action/state of new episode.
                            if ep_data is not None and len(ep_data["actions"]) > 0:
                                if args_cli.replay_mode == "state":
                                    action = ep_data["states"][0]
                                else:
                                    action = ep_data["actions"][0]
                                actions[env_id] = torch.from_numpy(action).to(idle_action.dtype)
                                env_episode_step[env_id] += 1
                                has_next_action = True
                        else:
                            continue

                # Apply damping and step.
                apply_damping(env, task_type)
                env.step(actions)
                rate_limiter.sleep(env)

            break

    print(f"Finished replaying {replayed_episode_count} episode{'s' if replayed_episode_count != 1 else ''}.")
    env.close()


if __name__ == "__main__":
    main()
