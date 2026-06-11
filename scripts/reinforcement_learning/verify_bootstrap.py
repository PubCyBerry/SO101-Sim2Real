"""grasp 부트스트랩 초기상태 검증 — '큐브 잡힌 채 시작'이 물리적으로 성립하는지.

PickCubeEnv 의 grasp_bootstrap(_reset_idx) 가 일부 env 를 큐브-인-그리퍼로 초기화한다.
이 스크립트는 prob=1 로 전체 env 를 부트스트랩하고, 그리퍼를 닫힘 유지하는 액션으로
스텝하며 (1) 큐브가 grasp point 근처 유지(held), (2) 책상 위로 들림(lifted) 을 측정.

사용법:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src .venv/bin/python \
    scripts/reinforcement_learning/verify_bootstrap.py --num_envs 32 \
    --close -0.05 --lift 0.0 --device cuda:0
"""
import argparse, json, sys, traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--close", type=float, default=-0.05, help="부트스트랩 gripper 닫힘 각")
p.add_argument("--lift", type=float, default=0.0, help="grasp point z 들어올림(m)")
p.add_argument("--hold_tol", type=float, default=0.04)
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch, gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import SO101_JOINT_ORDER  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_curriculum  # noqa: E402
from sim_to_real.utils.constant import CUBE_NAMES  # noqa: E402

DESK_TOP = 0.709


def main():
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        apply_curriculum(cfg, active_objects=1)
        cfg.grasp_bootstrap_prob = 1.0
        cfg.grasp_bootstrap_close = args.close
        cfg.grasp_bootstrap_lift = args.lift
        env = gym.make(args.task, cfg=cfg)
        base = env.unwrapped
        dev = base.device; n = base.num_envs
        robot = base.scene["robot"]; cube = base.scene[CUBE_NAMES[0]]
        gid = SO101_JOINT_ORDER.index("gripper")

        env.reset()
        # 첫 step 으로 grasp offset 캐시 → 이후 reset 에서 부트스트랩 적용
        zero = torch.zeros(n, len(SO101_JOINT_ORDER), device=dev)
        env.step(zero)
        env.reset()  # 이제 전체 env 부트스트랩(큐브-인-그리퍼)

        bn = list(robot.data.body_names)
        def grasp_pt():
            if "jaw" in bn and "gripper" in bn:
                j, g = bn.index("jaw"), bn.index("gripper")
                return 0.5*(robot.data.body_pos_w[:, j, :]+robot.data.body_pos_w[:, g, :])
            return robot.data.body_pos_w[:, bn.index("gripper"), :]

        # 그리퍼 닫힘 유지 액션: target=raw+offset(0.8) → raw=-0.9 면 target≈-0.1(닫힘)
        act = torch.zeros(n, len(SO101_JOINT_ORDER), device=dev); act[:, gid] = -0.9
        max_d = torch.zeros(n, device=dev)
        for _ in range(args.steps):
            env.step(act)
            d = torch.linalg.vector_norm(cube.data.root_pos_w - grasp_pt(), dim=1)
            max_d = torch.maximum(max_d, d)
        cube_z = cube.data.root_pos_w[:, 2]
        gp_z = grasp_pt()[:, 2]
        held = max_d < args.hold_tol
        lifted = (cube_z - DESK_TOP) > 0.03
        res = {
            "status": "bootstrap_ok" if held.float().mean() > 0.5 else "bootstrap_fail",
            "num_envs": n, "close": args.close, "lift": args.lift,
            "held_frac": round(float(held.float().mean()), 3),
            "lifted_frac": round(float(lifted.float().mean()), 3),
            "held_and_lifted": round(float((held & lifted).float().mean()), 3),
            "max_dist_mean_m": round(float(max_d.mean()), 4),
            "cube_z_mean": round(float(cube_z.mean()), 4),
            "graspPt_z_mean": round(float(gp_z.mean()), 4),
            "desk_top": DESK_TOP,
        }
        print(json.dumps(res), flush=True)
    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e), "traceback": traceback.format_exc()}), flush=True)
        sys.exit(1)
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        app.close()


if __name__ == "__main__":
    main()
