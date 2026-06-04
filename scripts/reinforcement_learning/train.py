"""TB.2/TB.3 PPO 학습 스크립트 — SO-101 PickCube/PickPen.

사용법:
    # PickCube 상태 기반 (기본값 — rl_policy 그룹 사용)
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 \
        --num_envs 64 --device cuda:0 --max_iterations 200

    # North Star 6-dim 정책만 사용
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 --obs_group policy \
        --num_envs 64 --device cuda:0 --max_iterations 200

    # 비대칭 AC: actor=rl_policy, critic=rl_policy
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 \
        --obs_group rl_policy --critic_obs_group rl_policy \
        --num_envs 64 --device cuda:0 --max_iterations 200
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TB.2 PPO training")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rl_device", default=None, help="RL 연산 디바이스 (기본값: --device와 동일)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--num_learning_epochs", type=int, default=20,
                    help="PPO update당 learning epoch 수. contact-rich grasp 학습을 위해 기본 20.")
parser.add_argument("--num_mini_batches", type=int, default=4,
                    help="PPO minibatch 수")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--experiment_name", default="so101_pick_cube_ppo")
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
# 커리큘럼 파라미터 — gym.make() 이전에 env_cfg 에 적용
parser.add_argument("--active_objects", "--active_pens", dest="active_objects",
                    type=int, default=4, choices=[1, 2, 3, 4],
                    help="학습에 사용할 대상 수 (1~4, 기본값: 4). --active_pens는 호환 alias.")
parser.add_argument("--object_radius_scale", "--pen_radius_scale", dest="object_radius_scale",
                    type=float, default=1.0,
                    help="대상 reset ellipse 반경 배율. --pen_radius_scale은 호환 alias.")
parser.add_argument("--container_angle_scale", "--cup_angle_scale", dest="container_angle_scale",
                    type=float, default=1.0,
                    help="그릇/컵 reset 각도 범위 배율. --cup_angle_scale은 호환 alias.")
parser.add_argument("--container_radius_scale", "--cup_radius_scale", dest="container_radius_scale",
                    type=float, default=1.0,
                    help="그릇/컵 안 판정 반경 배율. --cup_radius_scale은 호환 alias.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="에피소드 길이(초) override (기본값: env 설정값 30.0)")
parser.add_argument("--resume_checkpoint", default=None,
                    help="이어학습 체크포인트 경로 (.pt). 설정 시 learn() 전 로드.")
parser.add_argument("--init_noise_std", type=float, default=0.5,
                    help="ActorCritic 초기 action noise std")
parser.add_argument("--entropy_coef", type=float, default=0.005,
                    help="PPO entropy coefficient")
parser.add_argument("--learning_rate", type=float, default=3e-4,
                    help="PPO learning rate")
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

import sim_to_real  # noqa: E402  # SimToReal-SO101-PickCube/PickPen-v0 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_curriculum as apply_cube_curriculum  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import apply_curriculum as apply_pen_curriculum  # noqa: E402


TASK_ID = "TB.3"


def _checkpoint_index(path: str) -> int:
    """model_<step>.pt 체크포인트를 숫자 기준으로 정렬하기 위한 키."""
    match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
    if match is None:
        return -1
    return int(match.group(1))


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
            "learning_rate": args.learning_rate,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": args.entropy_coef,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
        },
    }


def _apply_task_curriculum(env_cfg, args: argparse.Namespace) -> None:
    """task 이름에 맞는 curriculum을 적용한다."""

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
        # 커리큘럼 파라미터 적용
        _apply_task_curriculum(env_cfg, args)
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
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
        # 이어학습: resume_checkpoint 지정 시 optimizer 포함 로드
        if args.resume_checkpoint is not None:
            try:
                runner.load(args.resume_checkpoint, load_optimizer=True, map_location=rl_device)
            except TypeError:
                runner.load(args.resume_checkpoint)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )

        # 체크포인트 목록 수집
        checkpoints = sorted(
            (
                c for c in glob.glob(os.path.join(log_dir, "model_*.pt"))
                if os.path.getmtime(c) >= run_start_time - 1.0
            ),
            key=_checkpoint_index,
        )
        if not checkpoints:
            print(
                json.dumps({
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error": f"체크포인트 없음: {log_dir}",
                }),
                flush=True,
            )
            sys.exit(1)

        total_steps = args.num_envs * args.num_steps_per_env * args.max_iterations
        print(
            json.dumps({
                "task_id": TASK_ID,
                "status": "passed",
                "task": args.task,
                "num_envs": args.num_envs,
                "num_steps_per_env": args.num_steps_per_env,
                "max_iterations": args.max_iterations,
                "num_learning_epochs": args.num_learning_epochs,
                "num_mini_batches": args.num_mini_batches,
                "total_steps": total_steps,
                "log_dir": log_dir,
                "checkpoints": [os.path.basename(c) for c in checkpoints],
                "latest_checkpoint": checkpoints[-1],
                "curriculum": {
                    "active_objects": args.active_objects,
                    "object_radius_scale": args.object_radius_scale,
                    "container_angle_scale": args.container_angle_scale,
                    "container_radius_scale": args.container_radius_scale,
                    "episode_length_s": args.episode_length_s,
                    "resume_checkpoint": args.resume_checkpoint,
                    "init_noise_std": args.init_noise_std,
                    "entropy_coef": args.entropy_coef,
                    "learning_rate": args.learning_rate,
                },
            }),
            flush=True,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        print(
            json.dumps({
                "task_id": TASK_ID,
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
