"""T0.3 smoke test: gym.make → reset → N random steps, assert 6-dim action/obs.

Usage:
    uv run python scripts/environments/env_smoke.py \
        --task SimToReal-SO101-PickPen-v0 \
        --steps 500 --num_envs 1 --device cuda:0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import time
import traceback

from isaaclab.app import AppLauncher

_RESULT_FILE = "/tmp/smoke_t03_result.json"

parser = argparse.ArgumentParser(description="T0.3 env smoke test")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=1)
# --device is registered by AppLauncher below (avoids conflict)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- imports that require SimApp to be running ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # registers SimToReal-SO101-PickPen-v0

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _write_result(result: dict) -> None:
    result_json = json.dumps(result, indent=2)
    print(result_json)
    sys.stdout.flush()
    with open(_RESULT_FILE, "w") as f:
        f.write(result_json + "\n")


def main() -> None:
    device: str = args.device  # provided by AppLauncher (e.g. "cuda:0")
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        # Ensure episode is long enough to survive all steps without timeout resets
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            args.steps * env_cfg.sim.dt * env_cfg.decimation + 2.0,
        )

        env = gym.make(args.task, cfg=env_cfg)

        # --- action shape check ---
        action_shape = env.action_space.shape
        assert action_shape[-1] == 6, f"Expected 6-dim action, got {action_shape}"

        # --- reset ---
        obs_dict, _ = env.reset()

        # --- locate policy obs tensor and assert 6-dim ---
        policy_obs = obs_dict.get("policy", obs_dict)
        if isinstance(policy_obs, dict):
            obs_tensor = None
            for v in policy_obs.values():
                if isinstance(v, torch.Tensor) and v.shape[-1] == 6:
                    obs_tensor = v
                    break
            if obs_tensor is None:
                obs_tensor = next(v for v in policy_obs.values() if isinstance(v, torch.Tensor))
        else:
            obs_tensor = policy_obs

        assert obs_tensor.shape[-1] == 6, (
            f"Expected policy obs last-dim=6, got shape={obs_tensor.shape}"
        )

        # --- random step loop ---
        t0 = time.time()
        resets = 0
        for _ in range(args.steps):
            action = torch.tensor(
                env.action_space.sample(), dtype=torch.float32, device=device
            )
            obs_dict, _reward, terminated, truncated, _info = env.step(action)
            if (terminated | truncated).any():
                obs_dict, _ = env.reset()
                resets += 1

        elapsed = time.time() - t0
        env.close()

        _write_result({
            "task": args.task,
            "status": "passed",
            "steps": args.steps,
            "resets": resets,
            "num_envs": args.num_envs,
            "action_shape": list(action_shape),
            "policy_obs_shape": list(obs_tensor.shape),
            "elapsed_s": round(elapsed, 2),
        })

    except Exception as exc:
        tb = traceback.format_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        _write_result({
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
        })
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
