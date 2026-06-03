"""TB.3 성공률 평가 스크립트 — SimToReal-SO101-PickPen-v0.

사용법:
    uv run python scripts/reinforcement_learning/eval_success.py \
        --task SimToReal-SO101-PickPen-v0 \
        --checkpoint /path/to/model_100.pt \
        --num_envs 16 --device cuda:0 --episodes 100

JSON 결과 stdout 출력 예:
    {
      "task_id": "TB.3",
      "status": "passed",
      "checkpoint": "...",
      "episodes": 100,
      "successes": 12,
      "success_rate": 0.12,
      "num_envs": 16,
      "max_episode_steps": 900
    }
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TB.3 성공률 평가")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--checkpoint", required=True, help="OnPolicyRunner 체크포인트 경로 (.pt)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rl_device", default=None, help="RL 연산 디바이스 (기본값: --device 와 동일)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episodes", type=int, default=100, help="수집할 에피소드 수")
parser.add_argument("--max_episode_steps", type=int, default=900,
                    help="에피소드 최대 스텝 수 (policy hz 기준 — env.episode_length_s 를 이 값으로 override)")
parser.add_argument("--obs_group", default="rl_policy",
                    help="actor 에 사용할 obs 그룹 이름 (학습 시와 동일해야 함)")
parser.add_argument("--critic_obs_group", default=None,
                    help="critic 에 사용할 obs 그룹 이름 (기본값: --obs_group 과 동일)")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--min_success_rate", type=float, default=None,
                    help="설정하면 success_rate 가 이 값 미만일 때 exit code 1 로 종료")
parser.add_argument("--stochastic", action="store_true",
                    help="deterministic act_inference 대신 stochastic policy.act()로 평가")
parser.add_argument("--init_noise_std", type=float, default=0.5,
                    help="학습 시 ActorCritic init_noise_std (checkpoint load shape 재현용)")
# 커리큘럼 파라미터 — 학습 시와 동일한 설정을 사용해야 분포가 일치
parser.add_argument("--active_pens", type=int, default=4, choices=[1, 2, 3, 4],
                    help="평가에 사용할 펜 수 (학습 시와 동일하게 설정)")
parser.add_argument("--pen_radius_scale", type=float, default=1.0,
                    help="펜 랜덤화 ellipse 반경 배율")
parser.add_argument("--cup_angle_scale", type=float, default=1.0,
                    help="컵 랜덤화 각도 범위 배율")
parser.add_argument("--cup_radius_scale", type=float, default=1.0,
                    help="컵 안 판정 반경 배율 (기본값: 1.0 = 0.05m)")
parser.add_argument("--grasp_assist", action="store_true",
                    help="학습과 동일하게 TB.3 soft grasp assist 활성화")
parser.add_argument("--grasp_assist_distance", type=float, default=0.075,
                    help="grasp assist attach 거리(m)")
parser.add_argument("--grasp_assist_offset_z", type=float, default=0.0,
                    help="gripper body 기준 pen z offset(m)")
parser.add_argument("--place_assist_distance", type=float, default=0.0,
                    help="컵 근방 도달 시 펜을 컵 중심으로 스냅하는 거리(m). 0이면 비활성")
# --device / --headless 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- Isaac Sim 기동 후 임포트 ----
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim_to_real  # noqa: E402  # SimToReal-SO101-PickPen-v0 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import apply_curriculum  # noqa: E402


def _build_train_cfg(args: argparse.Namespace) -> dict:
    """OnPolicyRunner 모델 구조 재현용 설정 딕셔너리 (학습 시와 동일하게 유지)."""
    rl_device = args.rl_device if args.rl_device is not None else args.device
    obs_group = args.obs_group
    critic_group = args.critic_obs_group if args.critic_obs_group is not None else obs_group
    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 1,
        "experiment_name": "eval_tmp",
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


def main() -> None:
    device: str = args.device
    rl_device: str = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        # 환경 설정
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # 커리큘럼 파라미터 적용 (학습 시와 동일하게 설정할 것)
        apply_curriculum(
            env_cfg,
            active_pens=args.active_pens,
            pen_radius_scale=args.pen_radius_scale,
            cup_angle_scale=args.cup_angle_scale,
            cup_radius_scale=args.cup_radius_scale,
            grasp_assist=args.grasp_assist,
            grasp_assist_distance=args.grasp_assist_distance,
            grasp_assist_offset_z=args.grasp_assist_offset_z,
            place_assist_distance=args.place_assist_distance,
        )
        # episode_length_s 를 max_episode_steps 기준으로 override
        policy_dt = env_cfg.sim.dt * env_cfg.decimation  # 초 단위 policy step
        env_cfg.episode_length_s = args.max_episode_steps * policy_dt

        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

        torch.manual_seed(args.seed)

        # OnPolicyRunner 생성 및 체크포인트 로드
        train_cfg = _build_train_cfg(args)
        runner = OnPolicyRunner(env, train_cfg, log_dir="", device=rl_device)
        try:
            runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        except TypeError:
            # rsl_rl 버전에 따라 load() 인자 지원 범위가 다를 수 있음
            runner.load(args.checkpoint)

        if args.stochastic:
            runner.eval_mode()
            runner.alg.policy.to(rl_device)
            policy = runner.alg.policy.act
        else:
            policy = runner.get_inference_policy(device=rl_device)

        # 평가 루프
        # get_observations() → TensorDict (obs group 이름을 키로 가짐)
        # policy(obs) 는 TensorDict 전체를 받아 obs_groups["policy"] 키로 추출·연결
        obs_dict = env.get_observations()
        total_episodes = 0
        total_successes = 0

        while total_episodes < args.episodes:
            with torch.no_grad():
                actions = policy(obs_dict)

            # step() 는 (TensorDict, rew, dones, extras) 4개 반환
            obs_dict, rewards, dones, infos = env.step(actions)

            if dones.any():
                # time_outs: is_finite_horizon=False 인 경우 extras 에 포함
                time_outs = infos.get("time_outs", torch.zeros_like(dones))
                success_mask = dones.bool() & ~time_outs.bool()

                for i in range(env.num_envs):
                    if dones[i].item():
                        total_episodes += 1
                        if success_mask[i].item():
                            total_successes += 1
                        if total_episodes >= args.episodes:
                            break

        success_rate = total_successes / max(1, total_episodes)
        result = {
            "task_id": "TB.3",
            "status": "passed",
            "checkpoint": args.checkpoint,
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": round(success_rate, 4),
            "num_envs": args.num_envs,
            "max_episode_steps": args.max_episode_steps,
            "stochastic": args.stochastic,
            "curriculum": {
                "active_pens": args.active_pens,
                "pen_radius_scale": args.pen_radius_scale,
                "cup_angle_scale": args.cup_angle_scale,
                "cup_radius_scale": args.cup_radius_scale,
                "grasp_assist": args.grasp_assist,
                "grasp_assist_distance": args.grasp_assist_distance,
                "grasp_assist_offset_z": args.grasp_assist_offset_z,
                "place_assist_distance": args.place_assist_distance,
            },
        }
        print(json.dumps(result), flush=True)

        if args.min_success_rate is not None and success_rate < args.min_success_rate:
            sys.exit(1)

    except Exception as exc:
        tb = traceback.format_exc()
        print(
            json.dumps({
                "task_id": "TB.3",
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
