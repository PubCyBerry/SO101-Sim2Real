"""TB.2/TB.3 PPO 학습 스크립트 — SimToReal-SO101-PickPen-v0.

사용법:
    # TB.3 상태 기반 (기본값 — rl_policy 그룹 사용)
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickPen-v0 \
        --num_envs 64 --device cuda:0 --max_iterations 100

    # North Star 6-dim 정책만 사용
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickPen-v0 --obs_group policy \
        --num_envs 64 --device cuda:0 --max_iterations 100

    # 비대칭 AC: actor=rl_policy, critic=rl_policy
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickPen-v0 \
        --obs_group rl_policy --critic_obs_group rl_policy \
        --num_envs 64 --device cuda:0 --max_iterations 100
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TB.2 PPO training")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rl_device", default=None, help="RL 연산 디바이스 (기본값: --device와 동일)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--experiment_name", default="so101_pick_pen_ppo")
parser.add_argument("--run_name", default="")
parser.add_argument("--log_root_path", default="outputs/rl/rsl_rl")
parser.add_argument("--checkpoint_dir", default=None, help="체크포인트 저장 경로 (스모크 테스트용)")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument(
    "--obs_group",
    default="rl_policy",
    help="actor 에 사용할 obs 그룹 이름 (기본값: rl_policy)",
)
parser.add_argument(
    "--critic_obs_group",
    default=None,
    help="critic 에 사용할 obs 그룹 이름 (기본값: --obs_group 과 동일)",
)
# --device / --headless 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# headless 기본값 강제 (명시적으로 --no-headless 를 전달하지 않은 경우)
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- Isaac Sim 기동 후 임포트 ----
import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # SimToReal-SO101-PickPen-v0 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _build_train_cfg(args: argparse.Namespace) -> dict:
    """OnPolicyRunner 에 전달할 학습 설정 딕셔너리 반환."""
    rl_device = args.rl_device if args.rl_device is not None else args.device
    obs_group = args.obs_group
    critic_group = args.critic_obs_group if args.critic_obs_group is not None else obs_group
    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": args.num_steps_per_env,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": {"policy": [obs_group], "critic": [critic_group]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.5,
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 1,
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


def _resolve_log_dir(args: argparse.Namespace) -> str:
    if args.checkpoint_dir is not None:
        log_dir = args.checkpoint_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{args.run_name}" if args.run_name else ""
        log_dir = os.path.join(
            args.log_root_path,
            args.experiment_name,
            f"{timestamp}{suffix}",
        )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def main() -> None:
    device: str = args.device
    rl_device: str = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        # 환경 설정 파싱 및 생성
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)

        # rsl_rl VecEnv 래퍼
        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

        # 재현성 시드
        torch.manual_seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)

        # 학습 설정 및 로그 디렉터리
        train_cfg = _build_train_cfg(args)
        log_dir = _resolve_log_dir(args)

        # OnPolicyRunner 생성 및 학습
        run_start_time = time.time()
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=rl_device)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )

        # 체크포인트 목록 수집
        checkpoints = sorted(
            c for c in glob.glob(os.path.join(log_dir, "model_*.pt"))
            if os.path.getmtime(c) >= run_start_time - 1.0
        )
        if not checkpoints:
            print(
                json.dumps({
                    "task_id": "TB.2",
                    "status": "failed",
                    "error": f"체크포인트 없음: {log_dir}",
                }),
                flush=True,
            )
            sys.exit(1)

        total_steps = args.num_envs * args.num_steps_per_env * args.max_iterations
        print(
            json.dumps({
                "task_id": "TB.2",
                "status": "passed",
                "task": args.task,
                "num_envs": args.num_envs,
                "num_steps_per_env": args.num_steps_per_env,
                "max_iterations": args.max_iterations,
                "total_steps": total_steps,
                "log_dir": log_dir,
                "checkpoints": [os.path.basename(c) for c in checkpoints],
                "latest_checkpoint": checkpoints[-1],
            }),
            flush=True,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        print(
            json.dumps({
                "task_id": "TB.2",
                "status": "failed",
                "error": str(exc),
                "traceback": tb,
            }),
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
