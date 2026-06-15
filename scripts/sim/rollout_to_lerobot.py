"""TC.1 RSL-RL expert rollout -> LeRobot v3 dataset recorder.

This script records successful SimToReal-SO101 PickCube/PickPen episodes only.
It writes the LeRobot v3 data contract used by the real SO-101 datasets:
6-dim joint action/state, 30 FPS timestamps, and three h264 camera videos.

The v3 writer logic lives in `lerobot_recorder.LeRobotV3DatasetWriter` (shared with
`pick_cube_curobo_demo.py`). This script only supplies the RL rollout + (action, state,
images) per frame.

Example:
    uv run --group isaac --locked python scripts/sim/rollout_to_lerobot.py \
        --checkpoint /DISK1/so101-sim2real/outputs/pick_cube_rl/model_200.pt \
        --output_dir /DISK1/so101-sim2real/outputs/tc1_rollout_10ep \
        --episodes 10 --overwrite
"""

from __future__ import annotations

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


# 단위/카메라 변환 상수·헬퍼는 lerobot_units 공용 모듈 (같은 디렉터리, AppLauncher 무의존).
from lerobot_units import (  # noqa: E402
    CAMERA_KEYS,
    CAMERA_SCENE_NAMES,
    read_camera_rgb_u8,
    to_lerobot_units as _to_lerobot_units,
)
# v3 writer 는 lerobot_recorder 공유 모듈 (pyarrow/imageio 지연 import → ABI 안전).
from lerobot_recorder import LeRobotV3DatasetWriter  # noqa: E402

TASK_ID = "TC.1"
PEN_TASK_NAME = "pick up the pen and place it in the holder"
CUBE_TASK_NAME = "pick up the cube and place it in the bowl"


parser = argparse.ArgumentParser(description="TC.1 rollout-to-LeRobot-v3 recorder")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--checkpoint", required=True, help="RSL-RL OnPolicyRunner checkpoint")
parser.add_argument("--output_dir", required=True, help="LeRobot dataset output directory")
parser.add_argument("--episodes", type=int, default=10, help="Number of successful episodes to keep")
parser.add_argument("--max_attempts", type=int, default=None, help="Stop after this many attempted episodes")
parser.add_argument("--max_episode_steps", type=int, default=900)
parser.add_argument("--num_envs", type=int, default=1, help="TC.1 recorder currently supports only 1 env")
parser.add_argument("--rl_device", default=None, help="RL device; defaults to --device")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--critic_obs_group", default=None)
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--init_noise_std", type=float, default=0.2)
parser.add_argument("--num_learning_epochs", type=int, default=20)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists")
parser.add_argument("--no_videos", action="store_true", help="Skip mp4 writing, but keep video metadata contract")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic act_inference instead of stochastic act")
parser.add_argument("--warmup_steps", type=int, default=5, help="Camera/render warm-up steps before recording")

# Curriculum defaults.
parser.add_argument("--active_objects", "--active_pens", dest="active_objects",
                    type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", "--pen_radius_scale", dest="object_radius_scale",
                    type=float, default=1.0)
parser.add_argument("--container_angle_scale", "--cup_angle_scale", dest="container_angle_scale",
                    type=float, default=1.0)
parser.add_argument("--container_radius_scale", "--cup_radius_scale", dest="container_radius_scale",
                    type=float, default=1.0)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_videos

launcher = AppLauncher(args)
simulation_app = launcher.app

# Isaac Sim must be running before these imports.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402, F401
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # registers SimToReal-SO101-PickCube/PickPen-v0

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    add_pick_cube_cameras,
    apply_curriculum as apply_cube_curriculum,
)
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import (  # noqa: E402
    add_pick_pen_cameras,
    apply_curriculum as apply_pen_curriculum,
)


def _build_train_cfg(cli_args: argparse.Namespace) -> dict[str, Any]:
    """Recreate the actor/critic network shape used by train.py/eval_success.py."""
    rl_device = cli_args.rl_device if cli_args.rl_device is not None else cli_args.device
    obs_group = cli_args.obs_group
    critic_group = cli_args.critic_obs_group if cli_args.critic_obs_group is not None else obs_group
    return {
        "seed": cli_args.seed,
        "device": rl_device,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "rollout_tmp",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": {"policy": [obs_group], "critic": [critic_group]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": cli_args.init_noise_std,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": cli_args.num_learning_epochs,
            "num_mini_batches": cli_args.num_mini_batches,
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
        },
    }


def _task_name(task: str) -> str:
    return CUBE_TASK_NAME if "PickCube" in task else PEN_TASK_NAME


def _apply_task_curriculum(env_cfg, cli_args: argparse.Namespace) -> None:
    """task 이름에 맞는 curriculum을 적용한다."""

    params = {
        "active_objects": cli_args.active_objects,
        "object_radius_scale": cli_args.object_radius_scale,
        "container_angle_scale": cli_args.container_angle_scale,
        "container_radius_scale": cli_args.container_radius_scale,
    }
    if cli_args.task and "PickCube" in cli_args.task:
        apply_cube_curriculum(env_cfg, **params)
    else:
        apply_pen_curriculum(
            env_cfg,
            active_pens=params["active_objects"],
            pen_radius_scale=params["object_radius_scale"],
            cup_angle_scale=params["container_angle_scale"],
            cup_radius_scale=params["container_radius_scale"],
        )


def _add_task_cameras(env_cfg, task: str) -> None:
    if "PickCube" in task:
        add_pick_cube_cameras(env_cfg.scene)
    else:
        add_pick_pen_cameras(env_cfg.scene)


def _read_joint_state(raw_env) -> np.ndarray:
    robot = raw_env.unwrapped.scene["robot"]
    return _to_lerobot_units(robot.data.joint_pos[0].detach().cpu().numpy())


def _action_to_record(action_tensor: torch.Tensor) -> np.ndarray:
    action = action_tensor[0].detach().cpu().numpy()
    return _to_lerobot_units(action)


def _capture_images(raw_env) -> dict[str, np.ndarray]:
    return {key: read_camera_rgb_u8(raw_env, CAMERA_SCENE_NAMES[key]) for key in CAMERA_KEYS}


def _reset_wrapper_env(env) -> Any:
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    if reset_out is None:
        return env.get_observations()
    return reset_out


def main() -> None:
    env = None
    writer: LeRobotV3DatasetWriter | None = None
    try:
        if args.num_envs != 1:
            raise ValueError("TC.1 recorder supports --num_envs 1 only; TC.2 will extend camera/env parallelism")
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")

        writer = LeRobotV3DatasetWriter(
            args.output_dir, overwrite=args.overwrite, enable_videos=not args.no_videos
        )

        device: str = args.device
        rl_device: str = args.rl_device if args.rl_device is not None else device
        task_name = _task_name(args.task)
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        _apply_task_curriculum(env_cfg, args)
        policy_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.episode_length_s = args.max_episode_steps * policy_dt
        if not args.no_videos:
            _add_task_cameras(env_cfg, args.task)

        raw_env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=args.clip_actions)

        torch.manual_seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)

        train_cfg = _build_train_cfg(args)
        runner = OnPolicyRunner(env, train_cfg, log_dir="", device=rl_device)
        try:
            runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        except TypeError:
            runner.load(args.checkpoint)

        if args.deterministic:
            policy = runner.get_inference_policy(device=rl_device)
        else:
            runner.eval_mode()
            runner.alg.policy.to(rl_device)
            policy = runner.alg.policy.act

        obs_dict = _reset_wrapper_env(env)
        zero_action = torch.zeros(args.num_envs, 6, device=device)
        for _ in range(max(0, args.warmup_steps)):
            obs_dict, _, _, _ = env.step(zero_action)
        obs_dict = _reset_wrapper_env(env)

        attempts = 1
        successes = 0
        failures = 0
        ep_frames = 0
        max_attempts = args.max_attempts if args.max_attempts is not None else max(args.episodes * 20, args.episodes)

        while successes < args.episodes and attempts <= max_attempts:
            with torch.no_grad():
                actions = policy(obs_dict)

            writer.add_frame(
                _action_to_record(actions),
                _read_joint_state(raw_env),
                {} if args.no_videos else _capture_images(raw_env),
            )
            ep_frames += 1

            obs_dict, _rewards, dones, infos = env.step(actions)
            if bool(dones[0].item()):
                time_outs = infos.get("time_outs", torch.zeros_like(dones))
                success = bool((dones.bool() & ~time_outs.bool())[0].item())
                committed = writer.commit_episode(success, task_name)
                if committed:
                    successes += 1
                    print(
                        json.dumps(
                            {
                                "task_id": TASK_ID,
                                "event": "episode_recorded",
                                "successes": successes,
                                "attempts": attempts,
                                "frames": ep_frames,
                            }
                        ),
                        flush=True,
                    )
                else:
                    failures += 1
                ep_frames = 0
                attempts += 1

        if successes < args.episodes:
            raise RuntimeError(
                f"Only recorded {successes}/{args.episodes} successful episodes "
                f"after {attempts - 1} attempts"
            )

        summary = writer.finalize(task_name)

        result = {
            "task_id": TASK_ID,
            "status": "passed",
            "task": args.task,
            "checkpoint": args.checkpoint,
            "output_dir": summary["output_dir"],
            "episodes": successes,
            "attempts": attempts - 1,
            "failures": failures,
            "total_frames": summary["total_frames"],
            "fps": summary["fps"],
            "videos": summary["videos"],
            "stochastic": not args.deterministic,
            "curriculum": {
                "active_objects": args.active_objects,
                "object_radius_scale": args.object_radius_scale,
                "container_angle_scale": args.container_angle_scale,
                "container_radius_scale": args.container_radius_scale,
            },
        }
        print(json.dumps(result), flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        if writer is not None:
            writer._close_video_writers()
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": tb,
                }
            ),
            flush=True,
        )
        sys.exit(1)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
