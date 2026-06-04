"""State-machine expert BC warm-start for PickCube PPO.

Input is one or more `.pt` files from
`scripts/environments/pick_cube_state_machine.py --expert_dataset_pt`.
The script trains only the ActorCritic actor with MSE(state -> raw action) and
saves a normal rsl_rl `model_*.pt` checkpoint that `train.py` can resume from.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import random
import sys
import traceback

from isaaclab.app import AppLauncher


if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


parser = argparse.ArgumentParser(description="BC warm-start rsl_rl ActorCritic from PickCube state-machine experts")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--expert_dataset_pt", nargs="+", required=True, type=Path)
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--base_checkpoint", default=None, help="Optional rsl_rl checkpoint to initialize from")
parser.add_argument("--output_iteration", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1, help="Dummy env count used to instantiate rsl_rl")
parser.add_argument("--rl_device", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--critic_obs_group", default=None)
parser.add_argument("--init_noise_std", type=float, default=0.5)
parser.add_argument("--num_learning_epochs", type=int, default=20)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--max_samples_per_phase", type=int, default=2000)
parser.add_argument(
    "--target_clip_actions",
    type=float,
    default=1.0,
    help="Clip expert action targets to the executable RSL-RL action range; <=0 disables clipping.",
)
parser.add_argument("--exclude_phase_contains", nargs="*", default=["settle"])
parser.add_argument("--require_success", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--active_objects", "--active_pens", dest="active_objects", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", "--pen_radius_scale", dest="object_radius_scale", type=float, default=0.25)
parser.add_argument("--container_angle_scale", "--cup_angle_scale", dest="container_angle_scale", type=float, default=0.25)
parser.add_argument("--container_radius_scale", "--cup_radius_scale", dest="container_radius_scale", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_curriculum as apply_cube_curriculum  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import apply_curriculum as apply_pen_curriculum  # noqa: E402


TASK_ID = "TB.4.BC"


def _build_train_cfg(args: argparse.Namespace) -> dict:
    rl_device = args.rl_device if args.rl_device is not None else args.device
    obs_group = args.obs_group
    critic_group = args.critic_obs_group if args.critic_obs_group is not None else obs_group
    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "bc_warmstart",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": {"policy": [obs_group], "critic": [critic_group]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": args.init_noise_std,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": args.num_learning_epochs,
            "num_mini_batches": args.num_mini_batches,
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


def _apply_task_curriculum(env_cfg, args: argparse.Namespace) -> None:
    params = {
        "active_objects": args.active_objects,
        "object_radius_scale": args.object_radius_scale,
        "container_angle_scale": args.container_angle_scale,
        "container_radius_scale": args.container_radius_scale,
    }
    if args.task and "PickCube" in args.task:
        apply_cube_curriculum(env_cfg, **params)
    else:
        apply_pen_curriculum(
            env_cfg,
            active_pens=params["active_objects"],
            pen_radius_scale=params["object_radius_scale"],
            cup_angle_scale=params["container_angle_scale"],
            cup_radius_scale=params["container_radius_scale"],
        )


def _phase_allowed(phase: str) -> bool:
    return not any(token and token in phase for token in args.exclude_phase_contains)


def _load_expert_dataset(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict]]:
    obs_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    phase_parts: list[str] = []
    metas: list[dict] = []
    phase_counts: dict[str, int] = defaultdict(int)

    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        meta = dict(data.get("meta", {}))
        meta["path"] = str(path)
        metas.append(meta)
        if args.require_success and not meta.get("placed_and_released", False):
            continue

        obs = data["obs"].to(torch.float32)
        actions = data["actions"].to(torch.float32)
        phases = list(data.get("phases", ["unknown"] * int(obs.shape[0])))
        if obs.shape[0] != actions.shape[0] or obs.shape[0] != len(phases):
            raise ValueError(f"Expert dataset length mismatch: {path}")

        keep: list[int] = []
        for idx, phase in enumerate(phases):
            if not _phase_allowed(phase):
                continue
            if args.max_samples_per_phase > 0 and phase_counts[phase] >= args.max_samples_per_phase:
                continue
            phase_counts[phase] += 1
            keep.append(idx)

        if keep:
            keep_t = torch.tensor(keep, dtype=torch.long)
            obs_parts.append(obs.index_select(0, keep_t))
            action_parts.append(actions.index_select(0, keep_t))
            phase_parts.extend([phases[i] for i in keep])

    if not obs_parts:
        raise RuntimeError("No expert samples after filtering")

    return torch.cat(obs_parts, dim=0), torch.cat(action_parts, dim=0), phase_parts, metas


def main() -> None:
    env = None
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        rl_device = args.rl_device if args.rl_device is not None else args.device
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        obs_cpu, actions_cpu, phases, metas = _load_expert_dataset(args.expert_dataset_pt)
        if args.target_clip_actions > 0:
            clip = abs(args.target_clip_actions)
            actions_cpu = actions_cpu.clamp(-clip, clip)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        _apply_task_curriculum(env_cfg, args)
        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=1.0)

        runner = OnPolicyRunner(env, _build_train_cfg(args), log_dir=str(output_dir), device=rl_device)
        if args.base_checkpoint is not None:
            try:
                runner.load(args.base_checkpoint, load_optimizer=False, map_location=rl_device)
            except TypeError:
                runner.load(args.base_checkpoint)

        policy = runner.alg.policy
        policy.train()
        optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.MSELoss()
        obs = obs_cpu.to(device=rl_device)
        actions = actions_cpu.to(device=rl_device)
        sample_count = int(obs.shape[0])
        steps_per_epoch = max(1, math.ceil(sample_count / max(1, args.batch_size)))
        final_loss = float("nan")

        for _epoch in range(args.epochs):
            perm = torch.randperm(sample_count, device=rl_device)
            for step in range(steps_per_epoch):
                idx = perm[step * args.batch_size : (step + 1) * args.batch_size]
                batch_obs = {args.obs_group: obs.index_select(0, idx)}
                batch_actions = actions.index_select(0, idx)
                pred = policy.act_inference(batch_obs)
                loss = loss_fn(pred, batch_actions)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
                optimizer.step()
                final_loss = float(loss.detach().cpu().item())

        checkpoint_path = output_dir / f"model_{args.output_iteration}.pt"
        phase_hist = dict(sorted((phase, phases.count(phase)) for phase in set(phases)))
        infos = {
            "task_id": TASK_ID,
            "bc": {
                "expert_datasets": [str(p) for p in args.expert_dataset_pt],
                "samples": sample_count,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "final_loss": final_loss,
                "target_clip_actions": args.target_clip_actions,
                "phase_hist": phase_hist,
                "metas": metas,
            },
        }
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": args.output_iteration,
                "infos": infos,
            },
            checkpoint_path,
        )
        result = {
            "task_id": TASK_ID,
            "status": "passed",
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "samples": sample_count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_loss": final_loss,
            "target_clip_actions": args.target_clip_actions,
            "phase_hist": phase_hist,
            "base_checkpoint": args.base_checkpoint,
            "curriculum": {
                "active_objects": args.active_objects,
                "object_radius_scale": args.object_radius_scale,
                "container_angle_scale": args.container_angle_scale,
                "container_radius_scale": args.container_radius_scale,
            },
        }
        (output_dir / "bc_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        env.close()
        env = None
    except Exception as exc:
        payload = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
