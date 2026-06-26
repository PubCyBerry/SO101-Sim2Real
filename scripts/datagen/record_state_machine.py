"""Unified data generation script using state machines for SO-101 PickCube.

Selects the appropriate state machine, runs the recording loop with LeRobot v3 writer.

Usage:
    isaaclab.sh -p scripts/datagen/record_state_machine.py \
        --task SimToReal-SO101-PickCube-v0 \
        --num_envs 1 --num_demos 50 --dataset_dir ./datasets/so101_pick_cube_demos
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import faulthandler
import os
import signal
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State machine data generation for SO-101 PickCube task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="SimToReal-SO101-PickCube-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./datasets/so101_pick_cube_demos",
    help="Directory to export recorded LeRobot v3 dataset.",
)
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of successful demonstrations to record. Set to 0 for infinite."
)
parser.add_argument("--headless", action="store_true", help="Run without GUI.")

# AppLauncher args (viewport, quality render, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Setup fault handler for crash diagnostics on Windows AppLauncher.
faulthandler_file = None
if sys.platform == "win32":
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    faulthandler_file = output_dir / "app_crash.txt"
    with faulthandler_file.open("w"):
        pass  # clear
    faulthandler.enable(file=faulthandler_file.open("a"))

# Filter AppLauncher args to known keys (avoid access violation on Windows).
_LAUNCHER_KEYS = {
    "headless", "device", "num_envs", "experience", "enable_cameras", "physics_dt",
    "rendering_dt", "enable_viewport", "viewport_camera_state"
}
launcher_args = {k: v for k, v in vars(args_cli).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real  # noqa: F401  register tasks
from sim_to_real.datagen.state_machine import PickCubeStateMachine
from sim_to_real.datagen.sm_actions import StateMachineActionsCfg
from sim_to_real.data.lerobot_recorder import LeRobotV3DatasetWriter
from sim_to_real.data.lerobot_units import read_camera_rgb_u8, to_lerobot_units, CAMERA_SCENE_NAMES
from sim_to_real.utils.gripper_effort import dynamic_reset_gripper_effort_limit_sim

# Maps gym task id → (StateMachineClass, device_type)
TASK_REGISTRY = {
    "SimToReal-SO101-PickCube-v0": (PickCubeStateMachine, "so101_state_machine"),
}


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Run state machine data generation with LeRobot v3 writer."""
    task_name = args_cli.task
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{task_name}' is not registered in TASK_REGISTRY.\nAvailable tasks: {list(TASK_REGISTRY.keys())}"
        )
    SMClass, device_type = TASK_REGISTRY[task_name]

    # Setup output directory and writer.
    output_dir = Path(args_cli.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(device_type)
    # SM 은 8D EE-pose(IK) action 을 낸다 → 기본 slew joint action(6D, VLA 추론용)을 datagen 동안만
    # IK action term 으로 교체. IK 가 joint 로 풀어 적용하고, 기록은 그 joint target 을 쓴다(아래).
    env_cfg.actions = StateMachineActionsCfg()
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # Disable gravity for robot links to match leisaac state-machine behavior.
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    rate_limiter = RateLimiter(args_cli.step_hz)

    if hasattr(env, "initialize"):
        env.initialize()

    # Initialize state machine and writer.
    sm = SMClass()
    sm.setup(env)
    env.reset()
    sm.reset()

    writer = LeRobotV3DatasetWriter(output_dir, overwrite=True, enable_videos=True, robot_type="so_follower")

    recorded_demo_count = 0
    start_record_state = False
    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                # Adapt gripper effort each step based on nearest object mass.
                dynamic_reset_gripper_effort_limit_sim(env, device_type)

                if sm.is_episode_done:
                    # Episode finished — check success and commit.
                    try:
                        success = sm.check_success(env)
                    except Exception as e:
                        print("Success check failed:", e)
                        success = False

                    print(f"Episode {'success' if success else 'failed'}!")

                    if start_record_state:
                        print("Stop recording.")
                        start_record_state = False

                    # Commit episode to writer.
                    was_committed = writer.commit_episode(
                        success=success, task_name="pick up the cube and place it in the bowl"
                    )
                    if was_committed:
                        recorded_demo_count += 1
                        print(f"Recorded {recorded_demo_count}/{args_cli.num_demos} successful demonstrations.")

                    # Check if we've reached target demo count.
                    if args_cli.num_demos > 0 and recorded_demo_count >= args_cli.num_demos:
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                        break

                    env.reset()
                    sm.reset()
                else:
                    # Episode in progress — record frame and step.
                    if not start_record_state:
                        print("Start recording.")
                        start_record_state = True

                    sm.pre_step(env)
                    actions = sm.get_action(env)
                    env.step(actions)
                    sm.advance()

                    # Capture frame (action·state 둘 다 joint-space degrees, 우리 codec).
                    #  - observation.state = 현재 관측 joint_pos.
                    #  - action = IK 가 푼 commanded joint target(joint_pos_target). SM 의 8D EE-pose 가
                    #    아니라 그 결과 6D joint target 을 기록해야 VLA(joint-space)·실기기와 호환.
                    #    (env.step 이 IK action term 을 적용한 뒤라 joint_pos_target 이 이번 step 명령값.)
                    robot = env.scene["robot"]
                    joint_pos_rad = robot.data.joint_pos[0].detach().cpu().numpy()
                    joint_pos_feature = to_lerobot_units(joint_pos_rad)

                    joint_target_rad = robot.data.joint_pos_target[0].detach().cpu().numpy()
                    # TODO(datagen): IK joint target 은 slew-limit 이 없어 큰 step 가능. 추론은 slew action
                    # 이라 train/deploy 정합 위해 기록 target 도 slew-limit 하는 게 이상적
                    # (메모리 vla-data-jerky-slew-record-fix 교훈). 현재는 raw IK target 기록.
                    action_record = to_lerobot_units(joint_target_rad)

                    images_dict = {}
                    for cam_key, scene_name in CAMERA_SCENE_NAMES.items():
                        try:
                            rgb_u8 = read_camera_rgb_u8(env, scene_name)
                            images_dict[cam_key] = rgb_u8
                        except Exception as e:
                            print(f"Warning: Failed to capture {cam_key} camera: {e}")

                    writer.add_frame(action_record, joint_pos_feature, images_dict)

                if rate_limiter:
                    rate_limiter.sleep(env)

            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        summary = writer.finalize()
        print(f"\n[INFO] Dataset saved: {summary['output_dir']}")
        print(f"  Total episodes: {summary['total_episodes']}")
        print(f"  Total frames: {summary['total_frames']}")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
