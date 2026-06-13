"""Skill chaining P4 — skill1(acquire)→skill2(place) 2-policy end-to-end eval.

전체 task env(종료=task_done=그릇 안)에서 env 마다 skill1 으로 시작, over-bowl-grasped
도달 시 그 env 를 skill2 로 전환(skill2 LSTM hidden reset → 학습 시 over-bowl 시작과 정합)
하고 placed 까지 굴린다. end-to-end 성공률·전환율·단계 도달률을 집계한다.

두 정책은 같은 arch(rl_policy 87dim·LSTM256) 전제. obs normalization 은 act_inference 내부 적용.

사용법:
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src .venv/bin/python \
      scripts/reinforcement_learning/eval_chain.py \
      --skill1_checkpoint .../skill1/model_1500.pt --skill2_checkpoint .../skill2/model_800.pt \
      --recurrent --rnn_hidden_dim 256 --obs_normalization --num_envs 64 --num_episodes 128 \
      --active_objects 1 --device cuda:0
"""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="skill1→skill2 chained eval")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--skill1_checkpoint", required=True)
parser.add_argument("--skill2_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=128)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=321)
parser.add_argument("--rl_device", default=None)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
# over-bowl-grasped 전환 조건(skill=acquire 종료와 동일 기본값)
parser.add_argument("--over_bowl_xy", type=float, default=0.10)
parser.add_argument("--lift_min", type=float, default=0.02)
parser.add_argument("--grasp_dist", type=float, default=0.07)
parser.add_argument("--close_threshold", type=float, default=0.50)
# 단계 판정 임계값
parser.add_argument("--reach_thr", type=float, default=0.06)
parser.add_argument("--grasp_thr", type=float, default=0.06)
parser.add_argument("--open_thr", type=float, default=0.60)
# 비디오
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--video_dir", default=None)
parser.add_argument("--cam_eye", type=float, nargs=3, default=[-2.739, 5.835, 1.811])
parser.add_argument("--cam_target", type=float, nargs=3, default=[-2.206, 5.025, 1.565])
# 정책 아키텍처(두 체크포인트 공통)
parser.add_argument("--recurrent", action="store_true", default=False)
parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
parser.add_argument("--rnn_hidden_dim", type=int, default=256)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--obs_normalization", action="store_true", default=False)
parser.add_argument("--init_noise_std", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True

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
    BOWL_CENTER_XY, BOWL_SUCCESS_RADIUS, BOWL_HEIGHT_RANGE,
)
from sim_to_real.tasks.pick_cube.mdp.rewards import _over_bowl_grasped_mask  # noqa: E402
from sim_to_real.tasks.common.mdp.rewards import (  # noqa: E402
    _get_gripper_pos, _object_inside_container_mask,
)
from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z  # noqa: E402
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def _build_eval_cfg() -> dict:
    rl_device = args.rl_device if args.rl_device is not None else args.device
    policy_cfg = {
        "class_name": "ActorCritic", "init_noise_std": args.init_noise_std,
        "actor_hidden_dims": [256, 128], "critic_hidden_dims": [256, 128],
        "activation": "elu",
        "actor_obs_normalization": args.obs_normalization,
        "critic_obs_normalization": args.obs_normalization,
    }
    if args.recurrent:
        policy_cfg.update({"class_name": "ActorCriticRecurrent", "rnn_type": args.rnn_type,
                           "rnn_hidden_dim": args.rnn_hidden_dim, "rnn_num_layers": args.rnn_num_layers})
    return {
        "seed": args.seed, "device": rl_device, "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1, "experiment_name": "chain",
        "run_name": "", "resume": False, "load_run": ".*",
        "load_checkpoint": "model_.*.pt", "logger": "tensorboard",
        "obs_groups": {"policy": [args.obs_group], "critic": [args.obs_group]},
        "policy": policy_cfg,
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1,
                      "learning_rate": 3e-4, "schedule": "fixed", "gamma": 0.99, "lam": 0.95,
                      "entropy_coef": 0.0, "desired_kl": 0.01, "max_grad_norm": 1.0,
                      "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2},
    }


def main() -> None:
    device = args.device
    rl_device = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # 전체 task(종료=task_done=그릇 안). skill 프리셋·부트스트랩·demo 없음(진짜 end-to-end).
        apply_cube_curriculum(
            env_cfg, active_objects=args.active_objects,
            object_radius_scale=1.0, container_angle_scale=1.0, container_radius_scale=1.0,
        )
        if hasattr(env_cfg, "grasp_bootstrap_prob"):
            env_cfg.grasp_bootstrap_prob = 0.0
        if hasattr(env_cfg, "demo_reset_prob"):
            env_cfg.demo_reset_prob = 0.0

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            vdir = args.video_dir or os.path.join(
                os.path.dirname(os.path.abspath(args.skill2_checkpoint)), "videos", "chain")
            os.makedirs(vdir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env, video_folder=vdir, step_trigger=lambda s: s == 0,
                video_length=args.video_length, disable_logger=True)
            print(json.dumps({"video_dir": vdir}), flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)
        torch.manual_seed(args.seed)

        # 두 정책 (같은 env·arch, 다른 체크포인트)
        r1 = OnPolicyRunner(env, _build_eval_cfg(), log_dir=None, device=rl_device)
        r1.load(args.skill1_checkpoint, load_optimizer=False, map_location=rl_device)
        r2 = OnPolicyRunner(env, _build_eval_cfg(), log_dir=None, device=rl_device)
        r2.load(args.skill2_checkpoint, load_optimizer=False, map_location=rl_device)
        r1.eval_mode(); r2.eval_mode()
        p1, p2 = r1.alg.policy, r2.alg.policy

        base = env.unwrapped
        n = base.num_envs
        robot_cfg = SceneEntityCfg("robot", body_names=["gripper"])
        cube_cfgs = [SceneEntityCfg(c) for c in CUBE_NAMES[:args.active_objects]]
        bowl_cfg = SceneEntityCfg(BOWL_NAME)

        if args.video:
            try:
                base.sim.set_camera_view(eye=tuple(args.cam_eye), target=tuple(args.cam_target))
            except Exception:
                pass

        # phase: 0=skill1, 1=skill2
        phase = torch.zeros(n, dtype=torch.long, device=device)
        stages = ["reach", "grasp", "over_bowl", "switched", "placed", "placed_open", "success"]
        ever = {s: torch.zeros(n, dtype=torch.bool, device=device) for s in stages[:-1]}
        counts = {s: 0 for s in stages}
        ep_count = 0

        def placed_mask():
            m = torch.ones(n, dtype=torch.bool, device=device)
            for cfg in cube_cfgs:
                cp = base.scene[cfg.name].data.root_pos_w
                m = m & _object_inside_container_mask(
                    base, cp, BOWL_CENTER_XY, BOWL_SUCCESS_RADIUS, BOWL_HEIGHT_RANGE, bowl_cfg)
            return m

        obs = env.get_observations()
        p1.reset(); p2.reset()
        step = 0
        last_placed = torch.zeros(n, dtype=torch.bool, device=device)
        last_placed_open = torch.zeros(n, dtype=torch.bool, device=device)
        with torch.inference_mode():
            while ep_count < args.num_episodes and step < args.max_steps:
                a1 = p1.act_inference(obs)
                a2 = p2.act_inference(obs)
                use2 = (phase == 1).unsqueeze(-1)
                actions = torch.where(use2, a2, a1)

                # over-bowl-grasped 도달한 skill1 env → skill2 전환(+ skill2 hidden reset)
                obg = _over_bowl_grasped_mask(
                    base, cube_cfgs, robot_cfg, BOWL_CENTER_XY, bowl_cfg,
                    args.over_bowl_xy, args.lift_min, args.grasp_dist, args.close_threshold)
                switch = (phase == 0) & obg
                if switch.any():
                    p2.reset(switch)
                    phase = torch.where(switch, torch.ones_like(phase), phase)
                    ever["switched"] |= switch

                # 단계 도달(pre-step live scene)
                ee = _get_gripper_pos(base, robot_cfg)
                gripper = base.scene["robot"].data.joint_pos[:, -1]
                closed = gripper < args.close_threshold
                gopen = gripper > args.open_thr
                cx, cy = BOWL_CENTER_XY
                for cfg in cube_cfgs:
                    cp = base.scene[cfg.name].data.root_pos_w
                    local = cp - base.scene.env_origins
                    dist = torch.linalg.vector_norm(cp - ee, dim=1)
                    lifted = local[:, 2] > (DESK_TOP_Z + args.lift_min)
                    ob = (torch.hypot(local[:, 0] - cx, local[:, 1] - cy) < args.over_bowl_xy) & lifted
                    ever["reach"] |= dist < args.reach_thr
                    ever["grasp"] |= (dist < args.grasp_thr) & closed & lifted
                    ever["over_bowl"] |= ob
                pm = placed_mask()
                ever["placed"] |= pm
                ever["placed_open"] |= pm & gopen
                last_placed = pm
                last_placed_open = pm & gopen

                obs, _rew, dones, extras = env.step(actions)
                step += 1
                done_mask = dones.bool() if dones.dtype != torch.bool else dones
                if done_mask.any():
                    time_outs = extras.get("time_outs", None)
                    if time_outs is None:
                        time_outs = torch.zeros_like(done_mask)
                    for i in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                        ep_count += 1
                        for s in stages[:-1]:
                            if bool(ever[s][i]):
                                counts[s] += 1
                        # end-to-end 성공 = 종료 직전 그릇 안(+open)
                        if bool(last_placed[i]):
                            counts["success"] += 1
                    # 종료 env 리셋: phase·정책 hidden·플래그
                    phase = torch.where(done_mask, torch.zeros_like(phase), phase)
                    p1.reset(done_mask); p2.reset(done_mask)
                    for s in stages[:-1]:
                        ever[s][done_mask] = False

        def rate(s):
            return round(counts[s] / ep_count, 3) if ep_count else None

        print(json.dumps({
            "status": "ok",
            "skill1_checkpoint": args.skill1_checkpoint,
            "skill2_checkpoint": args.skill2_checkpoint,
            "num_envs": n, "episodes_total": ep_count,
            "rates": {s: rate(s) for s in stages},
            "note": "switched=skill2 전환율, success=종료직전 그릇안. end-to-end(부트스트랩·demo 없음).",
        }), flush=True)

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
