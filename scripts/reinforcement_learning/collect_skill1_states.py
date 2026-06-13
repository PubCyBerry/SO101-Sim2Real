"""Skill chaining P2 — skill1(acquire+transport) 정책을 굴려 '그릇 위 grasp' 종료 상태 수집.

skill1 은 over_bowl_grasped 도달 즉시 종료한다. 그 직전(s_t) scene 상태를 env-local 로
스냅샷해, skill2(place) 학습의 demo-state reset 분포(_bootstrap_demo / _load_demos)로 쓸
demo_*.pt 로 저장한다. 행동 클론이 아니라 **상태 분포 seed**(handoff 지점).

포맷은 SM recorder(pick_cube_state_machine._save_demos)·PickCubeEnv._load_demos 와 동일:
  joint_pos/vel (1,ndof), cube_pos/quat/vel (1,ncube,{3,4,6}) env-local, bowl_* (1,{3,4,6}),
  phases=["LOWER"](패딩 아님→_load_demos 가 유지), meta.success=True.

사용법:
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src .venv/bin/python \
      scripts/reinforcement_learning/collect_skill1_states.py \
      --checkpoint outputs/.../lstm256_skill1_acquire_v1/model_1500.pt \
      --output_dir outputs/demos/skill1_overbowl --num_envs 512 --target_states 2000 \
      --recurrent --rnn_hidden_dim 256 --obs_normalization --active_objects 1 --device cuda:0
"""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="skill1 over-bowl-grasped 상태 수집")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--checkpoint", required=True, help="skill1(acquire) RSL-RL 체크포인트")
parser.add_argument("--output_dir", required=True, help="demo_*.pt 저장 디렉터리")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--target_states", type=int, default=2000, help="수집 목표 상태 수")
parser.add_argument("--max_steps", type=int, default=6000, help="안전 상한(스텝)")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--rl_device", default=None)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--demo_tag", default="s1", help="파일명 태그")
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
# over-bowl-grasped 필터(skill=acquire 종료 조건과 동일 기본값) — cube_lost 종료를 걸러낸다.
parser.add_argument("--over_bowl_xy", type=float, default=0.12)
parser.add_argument("--lift_min", type=float, default=0.02)
parser.add_argument("--grasp_dist", type=float, default=0.08)
parser.add_argument("--close_threshold", type=float, default=0.50)
# 정책 아키텍처(체크포인트와 일치)
parser.add_argument("--recurrent", action="store_true", default=False)
parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
parser.add_argument("--rnn_hidden_dim", type=int, default=256)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--obs_normalization", action="store_true", default=False)
parser.add_argument("--init_noise_std", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim_to_real  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    apply_curriculum as apply_cube_curriculum,
    apply_skill_acquire,
    BOWL_CENTER_XY,
)
from sim_to_real.tasks.common.mdp.rewards import _get_gripper_pos, _container_xy  # noqa: E402
from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z  # noqa: E402
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def _build_eval_cfg() -> dict:
    rl_device = args.rl_device if args.rl_device is not None else args.device
    policy_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": args.init_noise_std,
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
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
        "seed": args.seed, "device": rl_device, "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1, "experiment_name": "collect",
        "run_name": "", "resume": False, "load_run": ".*",
        "load_checkpoint": "model_.*.pt", "logger": "tensorboard",
        "obs_groups": {"policy": [args.obs_group], "critic": [args.obs_group]},
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
        apply_cube_curriculum(
            env_cfg, active_objects=args.active_objects,
            object_radius_scale=1.0, container_angle_scale=1.0, container_radius_scale=1.0,
        )
        # skill1 종료(over_bowl_grasped)로 자연 recycle. 부트스트랩 0 = 실제 정책이 scratch
        # 에서 만드는 handoff 상태(rollout 시 skill1 도 scratch 시작이므로 분포 정합).
        apply_skill_acquire(env_cfg)
        env_cfg.grasp_bootstrap_prob = 0.0
        env_cfg.demo_reset_prob = 0.0

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)
        torch.manual_seed(args.seed)

        runner = OnPolicyRunner(env, _build_eval_cfg(), log_dir=None, device=rl_device)
        runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        policy = runner.get_inference_policy(device=rl_device)

        base = env.unwrapped
        n = base.num_envs
        robot_cfg = SceneEntityCfg("robot", body_names=["gripper"])
        os.makedirs(args.output_dir, exist_ok=True)

        def snapshot():
            """현재(step 직전) env-local scene 상태를 dict(텐서 GPU)로 반환."""
            origins = base.scene.env_origins
            robot = base.scene["robot"]
            cpos, cquat, cvel = [], [], []
            for c in CUBE_NAMES:
                a = base.scene[c]
                cpos.append(a.data.root_pos_w - origins)
                cquat.append(a.data.root_quat_w)
                cvel.append(torch.cat([a.data.root_lin_vel_w, a.data.root_ang_vel_w], dim=-1))
            bowl = base.scene[BOWL_NAME]
            return {
                "jpos": robot.data.joint_pos.clone(),
                "jvel": robot.data.joint_vel.clone(),
                "cpos": torch.stack(cpos, dim=1),                       # (N,ncube,3)
                "cquat": torch.stack(cquat, dim=1),
                "cvel": torch.stack(cvel, dim=1),
                "bpos": (bowl.data.root_pos_w - origins).clone(),       # (N,3)
                "bquat": bowl.data.root_quat_w.clone(),
                "bvel": torch.cat([bowl.data.root_lin_vel_w, bowl.data.root_ang_vel_w], dim=-1),
            }

        def near_over_bowl_grasped(snap) -> torch.Tensor:
            """스냅샷이 over-bowl-grasped 근처인지(=success 종료, cube_lost 아님) 필터."""
            ee = _get_gripper_pos(base, robot_cfg)
            cx, cy = _container_xy(base, BOWL_CENTER_XY, SceneEntityCfg(BOWL_NAME))
            closed = snap["jpos"][:, -1] < args.close_threshold
            ok = torch.zeros(n, dtype=torch.bool, device=device)
            for i in range(len(CUBE_NAMES)):
                local = snap["cpos"][:, i, :]
                world = local + base.scene.env_origins
                lifted = local[:, 2] > (DESK_TOP_Z + args.lift_min)
                xy = torch.hypot(local[:, 0] - cx, local[:, 1] - cy)
                dist = torch.linalg.vector_norm(world - ee, dim=1)
                ok = ok | (lifted & (xy < args.over_bowl_xy) & closed & (dist < args.grasp_dist))
            return ok

        def save_env(snap, i: int, idx: int) -> None:
            traj = {
                "joint_pos": snap["jpos"][i:i + 1].cpu().contiguous(),
                "joint_vel": snap["jvel"][i:i + 1].cpu().contiguous(),
                "cube_pos": snap["cpos"][i:i + 1].cpu().contiguous(),    # (1,ncube,3) env-local
                "cube_quat": snap["cquat"][i:i + 1].cpu().contiguous(),
                "cube_vel": snap["cvel"][i:i + 1].cpu().contiguous(),
                "bowl_pos": snap["bpos"][i:i + 1].cpu().contiguous(),    # (1,3) env-local
                "bowl_quat": snap["bquat"][i:i + 1].cpu().contiguous(),
                "bowl_vel": snap["bvel"][i:i + 1].cpu().contiguous(),
                "phases": ["LOWER"],  # 패딩(_PAD_PHASES) 아님 → _load_demos 가 유지
                "meta": {"success": True, "frames": 1, "active_objects": args.active_objects,
                         "source": "skill1_acquire", "cube_names": list(CUBE_NAMES)},
            }
            torch.save(traj, os.path.join(args.output_dir, f"demo_{args.demo_tag}_{idx:05d}.pt"))

        obs = env.get_observations()
        prev = snapshot()  # s_{t-1} 롤링 버퍼
        saved = 0
        step = 0
        with torch.inference_mode():
            while saved < args.target_states and step < args.max_steps:
                actions = policy(obs)
                obs, _rew, dones, extras = env.step(actions)
                step += 1
                done_mask = dones.bool() if dones.dtype != torch.bool else dones
                if done_mask.any():
                    time_outs = extras.get("time_outs", None)
                    if time_outs is None:
                        time_outs = torch.zeros_like(done_mask)
                    # success(over_bowl_grasped) 종료 = done & ~time_out, 그리고 직전 스냅샷이
                    # over-bowl-grasped 근처여야(cube_lost 종료 제외).
                    succ = done_mask & (~time_outs.bool()) & near_over_bowl_grasped(prev)
                    for i in torch.nonzero(succ, as_tuple=False).flatten().tolist():
                        if saved >= args.target_states:
                            break
                        save_env(prev, i, saved)
                        saved += 1
                    if step % 50 == 0 or succ.any():
                        print(json.dumps({"step": step, "saved": saved}), flush=True)
                prev = snapshot()

        print(json.dumps({"status": "ok", "saved": saved, "steps": step,
                          "output_dir": args.output_dir}), flush=True)

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
