"""학습된 정책 평가 — SO-101 PickCube 성공률 측정 (DR 활성).

최종 목표: DR 켠 상태에서 4큐브 전부 배치 성공률 >= 0.90.

사용법:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/reinforcement_learning/eval.py \
        --task SimToReal-SO101-PickCube-v0 --recurrent \
        --checkpoint outputs/rl/rsl_rl/lstm_ppo_pickcube/<run>/model_1500.pt \
        --num_envs 256 --num_episodes 512 --device cuda:0

성공 판정: 에피소드가 timeout(truncation) 이 아니라 success termination 으로
끝났는지로 센다. success termination = task_done(모든 큐브 그릇 안). reward hacking
금지선(성공 반경/그립 물리)을 건드리지 않고 환경 그대로 평가한다.
"""

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PickCube 정책 성공률 평가")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--checkpoint", required=True, help="평가할 .pt 체크포인트 경로")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_episodes", type=int, default=512, help="집계할 최소 에피소드 수")
parser.add_argument("--seed", type=int, default=123)
# --device 는 AppLauncher 가 등록
parser.add_argument("--rl_device", default=None)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--critic_obs_group", default=None)
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="평균 액션(deterministic) 사용. 미설정 시 stochastic.")
# 진행 모니터링용 에피소드 비디오 녹화(평가 시 소수 env 뷰포트 → 학습 속도와 무관)
parser.add_argument("--video", action="store_true", default=False,
                    help="평가 에피소드를 비디오로 녹화(enable_cameras 자동 on).")
parser.add_argument("--video_length", type=int, default=600,
                    help="녹화 길이(policy step). 600≈20s.")
parser.add_argument("--video_dir", default=None,
                    help="비디오 저장 폴더(기본: 체크포인트 dir/videos/eval).")
# 정책 아키텍처 (체크포인트와 일치해야 함)
parser.add_argument("--recurrent", action="store_true", default=False)
parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
parser.add_argument("--rnn_hidden_dim", type=int, default=256)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--obs_normalization", action="store_true", default=False)
parser.add_argument("--init_noise_std", type=float, default=0.5)
# 커리큘럼 (평가는 기본 full 난이도 + DR on)
parser.add_argument("--active_objects", type=int, default=4, choices=[1, 2, 3, 4])
parser.add_argument("--object_radius_scale", type=float, default=1.0)
parser.add_argument("--container_angle_scale", type=float, default=1.0)
# 주의: container_radius_scale 은 1.0 고정 (성공 반경 스케일링 금지)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # 환경 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_curriculum as apply_cube_curriculum  # noqa: E402


def _build_eval_cfg() -> dict:
    rl_device = args.rl_device if args.rl_device is not None else args.device
    obs_group = args.obs_group
    critic_group = args.critic_obs_group if args.critic_obs_group is not None else obs_group
    policy_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": args.init_noise_std,
        "actor_hidden_dims": [256, 128] if args.recurrent else [128, 128],
        "critic_hidden_dims": [256, 128] if args.recurrent else [128, 128],
        "activation": "elu",
        "actor_obs_normalization": args.obs_normalization,
        "critic_obs_normalization": args.obs_normalization,
    }
    if args.recurrent:
        policy_cfg.update({
            "class_name": "ActorCriticRecurrent",
            "rnn_type": args.rnn_type,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "rnn_num_layers": args.rnn_num_layers,
        })
    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "eval",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": {"policy": [obs_group], "critic": [critic_group]},
        "policy": policy_cfg,
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1,
            "learning_rate": 3e-4, "schedule": "fixed", "gamma": 0.99, "lam": 0.95,
            "entropy_coef": 0.0, "desired_kl": 0.01, "max_grad_norm": 1.0,
            "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
        },
    }


def main() -> None:
    device = args.device
    rl_device = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # full 난이도 + DR on. 성공 반경은 1.0 고정(스케일 금지).
        apply_cube_curriculum(
            env_cfg,
            active_objects=args.active_objects,
            object_radius_scale=args.object_radius_scale,
            container_angle_scale=args.container_angle_scale,
            container_radius_scale=1.0,
        )
        env = gym.make(args.task, cfg=env_cfg,
                       render_mode="rgb_array" if args.video else None)

        if args.video:
            vdir = args.video_dir or os.path.join(
                os.path.dirname(os.path.abspath(args.checkpoint)), "videos", "eval")
            os.makedirs(vdir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env, video_folder=vdir,
                step_trigger=lambda step: step == 0,  # 시작 시 1회 녹화
                video_length=args.video_length, disable_logger=True,
            )
            print(json.dumps({"video_dir": vdir, "length": args.video_length}), flush=True)

        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

        torch.manual_seed(args.seed)

        runner = OnPolicyRunner(env, _build_eval_cfg(), log_dir=None, device=rl_device)
        runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        policy = runner.get_inference_policy(device=rl_device)

        num_envs = env.num_envs
        ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        n_success = 0
        n_timeout = 0
        n_episodes = 0
        success_steps_sum = 0

        obs = env.get_observations()  # RslRlVecEnvWrapper.get_observations 는 TensorDict 단일 반환
        max_steps = args.num_episodes * 2000  # 안전 상한
        step = 0
        with torch.inference_mode():
            while n_episodes < args.num_episodes and step < max_steps:
                actions = policy(obs)  # inference policy 는 deterministic mean 사용
                obs, _rew, dones, extras = env.step(actions)
                ep_steps += 1
                step += 1

                done_mask = dones.bool() if dones.dtype != torch.bool else dones
                if done_mask.any():
                    # truncation(timeout) 여부 — Isaac Lab 은 extras["time_outs"] 제공
                    time_outs = extras.get("time_outs", None)
                    if time_outs is None:
                        time_outs = torch.zeros_like(done_mask)
                    time_outs = time_outs.bool()

                    finished = done_mask
                    success = finished & (~time_outs)  # success termination = task_done
                    timeout = finished & time_outs

                    n_episodes += int(finished.sum().item())
                    n_success += int(success.sum().item())
                    n_timeout += int(timeout.sum().item())
                    if success.any():
                        success_steps_sum += int(ep_steps[success].sum().item())
                    # 종료 env step 카운터 리셋
                    ep_steps[finished] = 0

        success_rate = n_success / max(n_episodes, 1)
        mean_success_steps = (success_steps_sum / n_success) if n_success > 0 else None
        result = {
            "status": "passed" if success_rate >= 0.90 else "below_target",
            "task": args.task,
            "checkpoint": args.checkpoint,
            "num_envs": num_envs,
            "episodes": n_episodes,
            "success": n_success,
            "timeout": n_timeout,
            "success_rate": round(success_rate, 4),
            "mean_success_steps": (round(mean_success_steps, 1) if mean_success_steps else None),
            "target": 0.90,
            "deterministic": True,  # get_inference_policy 는 평균 액션
            "dr_active": True,
        }
        print(json.dumps(result), flush=True)
        if success_rate < 0.90:
            sys.exit(2)

    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc()}), flush=True)
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
