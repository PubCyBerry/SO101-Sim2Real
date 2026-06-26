"""SO-101 PickCube PPO 학습 — 레퍼런스(ref_repos/pick_and_place) 정합 기본 설정.

기본값 = 레퍼런스(IsaacLab Lift-Cube-Place) 레시피:
  - obs   : ref_policy (54-dim 저차원)
  - 정책  : feedforward ActorCritic MLP [128,64,32] + obs_normalization
  - PPO   : γ0.98, lr 8e-5(adaptive), entropy 0.006, num_steps 24, epochs 5, minibatch 4,
            max_grad_norm 0.4, init_noise_std 1.0
  - 보상  : ref dense 4항(env_cfg 기본) — reaching/lifting/tracking/lowering + smoothness
  - DR    : --dr_level 0(완전고정) 기본, 단계적으로 올림
  - 큐브  : --active_objects 1 (Cube1=40mm)

사용 예:
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 --num_envs 4096 --max_iterations 4000 \
        --device cuda:0 --dr_level 0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 PickCube PPO (레퍼런스 정합)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rl_device", default=None, help="RL 연산 디바이스 (기본값: --device와 동일)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=4000)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--num_learning_epochs", type=int, default=5)
parser.add_argument("--num_mini_batches", type=int, default=4)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--experiment_name", default="ref_lift_place")
parser.add_argument("--run_name", default="")
parser.add_argument("--log_root_path", default="outputs/rl/rsl_rl")
parser.add_argument("--checkpoint_dir", default=None, help="체크포인트 저장 경로 (스모크 테스트용)")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--obs_group", default="ref_policy",
                    help="actor 에 사용할 obs 그룹 이름 (기본값: ref_policy 54-dim)")
parser.add_argument("--critic_obs_group", default=None,
                    help="critic 에 사용할 obs 그룹 이름 (기본값: --obs_group 과 동일)")
# 커리큘럼 파라미터 — gym.make() 이전에 env_cfg 에 적용
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4],
                    help="학습에 사용할 큐브 수 (1~4, 기본값: 1 = Cube1 40mm).")
parser.add_argument("--object_radius_scale", type=float, default=1.0,
                    help="큐브 reset scatter 범위 배율(0=고정 spawn). dr_level 이 우선 제어.")
parser.add_argument("--container_angle_scale", type=float, default=1.0,
                    help="그릇 reset 각도 범위 배율.")
parser.add_argument("--container_radius_scale", type=float, default=1.0,
                    help="그릇 안(success) 판정 반경 배율.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="에피소드 길이(초) override (기본값: env 설정값)")
parser.add_argument("--dr_level", type=int, default=0, choices=[0, 1, 2, 3],
                    help="DR 사다리(PickCube). 0=완전고정 → 1=+spawn 랜덤 → 2=+sensor(jitter+"
                         "obs noise) → 3=+물리/시각. 기본 0. -1 전달 시 authored DR 유지.")
# 진행 모니터링용 주기적 에피소드 비디오 녹화
parser.add_argument("--video", action="store_true", default=False,
                    help="학습 중 주기적으로 에피소드 비디오 녹화(headless offscreen).")
parser.add_argument("--video_length", type=int, default=450, help="녹화 길이(policy step 수)")
parser.add_argument("--video_interval", type=int, default=1500, help="녹화 간격(policy step 수)")
parser.add_argument("--resume_checkpoint", default=None,
                    help="이어학습 체크포인트 경로 (.pt). DR 사다리 단계 상승 시 사용.")
parser.add_argument("--resume_without_optimizer", action="store_true",
                    help="체크포인트에서 policy/value만 로드하고 optimizer state는 새 설정으로 시작")
parser.add_argument("--init_noise_std", type=float, default=1.0,
                    help="ActorCritic 초기 action noise std (레퍼런스 1.0)")
parser.add_argument("--override_policy_std", type=float, default=None,
                    help="resume checkpoint 로드 후 policy action std를 이 값으로 덮어씀")
parser.add_argument("--entropy_coef", type=float, default=0.006, help="PPO entropy coefficient")
parser.add_argument("--learning_rate", type=float, default=8e-5, help="PPO learning rate")
parser.add_argument("--gamma", type=float, default=0.98, help="PPO 할인율")
parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--policy_hidden_dims", type=int, nargs="+", default=[128, 64, 32],
                    help="feedforward ActorCritic MLP hidden dims (레퍼런스 [128,64,32])")
parser.add_argument("--obs_normalization", action=argparse.BooleanOptionalAction, default=True,
                    help="actor/critic 관측 정규화(empirical running stats). 기본 on(레퍼런스).")
parser.add_argument("--max_grad_norm", type=float, default=0.4, help="PPO gradient clipping max norm")
parser.add_argument("--schedule", default="adaptive", choices=["fixed", "adaptive"],
                    help="PPO learning rate schedule")
# --device / --headless 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# headless 기본값 강제 (명시적으로 --no-headless 를 전달하지 않은 경우)
args.headless = True
# 비디오 녹화 시 offscreen 렌더를 위해 카메라 활성화 강제
if args.video:
    args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- Isaac Sim 기동 후 임포트 ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # SimToReal-SO101-PickCube/PickPen-v0 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    apply_curriculum as apply_cube_curriculum,
    apply_dr_level as apply_cube_dr_level,
)
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
    obs_groups = {"policy": [obs_group], "critic": [critic_group]}

    policy_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": args.init_noise_std,
        "actor_hidden_dims": list(args.policy_hidden_dims),
        "critic_hidden_dims": list(args.policy_hidden_dims),
        "activation": "elu",
        "actor_obs_normalization": args.obs_normalization,
        "critic_obs_normalization": args.obs_normalization,
    }

    algorithm_cfg = {
        "class_name": "PPO",
        "num_learning_epochs": args.num_learning_epochs,
        "num_mini_batches": args.num_mini_batches,
        "learning_rate": args.learning_rate,
        "schedule": args.schedule,
        "gamma": args.gamma,
        "lam": args.lam,
        "entropy_coef": args.entropy_coef,
        "desired_kl": 0.01,
        "max_grad_norm": args.max_grad_norm,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
    }

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
        "obs_groups": obs_groups,
        "policy": policy_cfg,
        "algorithm": algorithm_cfg,
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
        log_dir = os.path.join(args.log_root_path, args.experiment_name, f"{timestamp}{suffix}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _override_policy_std(policy, value: float | None) -> None:
    if value is None:
        return
    if not hasattr(policy, "std"):
        raise AttributeError("ActorCritic policy does not expose a 'std' parameter")
    with torch.no_grad():
        policy.std.fill_(float(value))


def main() -> None:
    device: str = args.device
    rl_device: str = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # 커리큘럼 파라미터 적용
        _apply_task_curriculum(env_cfg, args)
        # DR 사다리 (PickCube; apply_curriculum 이후 = events/obs 재구성). -1 이면 authored DR 유지.
        if args.task and "PickCube" in args.task and args.dr_level is not None and args.dr_level >= 0:
            apply_cube_dr_level(env_cfg, args.dr_level, active_objects=args.active_objects)
        # --episode_length_s override
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s

        log_dir = _resolve_log_dir(args)

        env = gym.make(args.task, cfg=env_cfg,
                       render_mode="rgb_array" if args.video else None)

        # 주기적 에피소드 비디오 녹화 (RL 래퍼보다 먼저 감싼다)
        if args.video:
            video_dir = os.path.join(log_dir, "videos", "train")
            os.makedirs(video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
            print(json.dumps({"video": True, "video_dir": video_dir,
                              "interval": args.video_interval, "length": args.video_length}),
                  flush=True)

        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

        torch.manual_seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)

        train_cfg = _build_train_cfg(args)

        run_start_time = time.time()
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=rl_device)
        # 이어학습: resume_checkpoint 지정 시 로드 (DR 사다리 단계 상승)
        if args.resume_checkpoint is not None:
            load_optimizer = not args.resume_without_optimizer
            if args.resume_without_optimizer:
                _loaded = torch.load(args.resume_checkpoint, map_location=rl_device, weights_only=False)
                runner.alg.policy.load_state_dict(_loaded["model_state_dict"])
                print(f"[resume] policy-only load from {args.resume_checkpoint} (optimizer fresh)", flush=True)
            else:
                try:
                    runner.load(args.resume_checkpoint, load_optimizer=load_optimizer, map_location=rl_device)
                except TypeError:
                    runner.load(args.resume_checkpoint)
            _override_policy_std(runner.alg.policy, args.override_policy_std)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )

        checkpoints = sorted(
            (
                c for c in glob.glob(os.path.join(log_dir, "model_*.pt"))
                if os.path.getmtime(c) >= run_start_time - 1.0
            ),
            key=_checkpoint_index,
        )
        if not checkpoints:
            print(json.dumps({"task_id": TASK_ID, "status": "failed",
                              "error": f"체크포인트 없음: {log_dir}"}), flush=True)
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
                "total_steps": total_steps,
                "log_dir": log_dir,
                "checkpoints": [os.path.basename(c) for c in checkpoints],
                "latest_checkpoint": checkpoints[-1],
                "curriculum": {
                    "active_objects": args.active_objects,
                    "dr_level": args.dr_level,
                    "obs_group": args.obs_group,
                    "policy_hidden_dims": args.policy_hidden_dims,
                    "gamma": args.gamma,
                    "learning_rate": args.learning_rate,
                    "entropy_coef": args.entropy_coef,
                    "resume_checkpoint": args.resume_checkpoint,
                },
            }),
            flush=True,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        print(json.dumps({"task_id": TASK_ID, "status": "failed",
                          "error": str(exc), "traceback": tb}), flush=True)
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
