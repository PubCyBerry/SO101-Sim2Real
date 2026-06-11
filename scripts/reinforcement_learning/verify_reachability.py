"""DR 활성 시 큐브 spawn 이 SO-101 도달 범위 안인지 실증.

요구사항: "domain randomization 적용 시 큐브는 SO-101 도달 가능 범위에서만 spawn".
이 스크립트는 DR(randomize_cubes_scattered)을 켠 채 환경을 여러 번 reset 하여,
각 큐브의 robot base 대비 평면(xy) 거리 분포를 모아 REACH_RADIUS 초과 샘플이
없는지 확인한다(샘플러 in-range 보증). 기하 계산상 최악 코너는 ~0.335 m 이고
SO-101 평면 도달은 ~0.44 m 이므로 통상 전부 통과해야 한다.

사용법:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/reinforcement_learning/verify_reachability.py \
        --num_envs 64 --num_resets 50 --reach_radius 0.44
"""

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PickCube 큐브 DR 도달성 검증")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_resets", type=int, default=50, help="reset 반복 횟수(샘플 수 = num_envs*num_resets*4)")
parser.add_argument("--reach_radius", type=float, default=0.44, help="SO-101 평면 도달 반경(m) 합격선")
parser.add_argument("--seed", type=int, default=7)
# --device 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_curriculum as apply_cube_curriculum  # noqa: E402
from sim_to_real.utils.constant import CUBE_NAMES  # noqa: E402


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # full DR 난이도 (성공 반경 스케일 금지 → 1.0)
        apply_cube_curriculum(env_cfg, active_objects=4, object_radius_scale=1.0,
                              container_angle_scale=1.0, container_radius_scale=1.0)
        env = gym.make(args.task, cfg=env_cfg)
        base_env = env.unwrapped

        torch.manual_seed(args.seed)

        robot = base_env.scene["robot"]
        cubes = [base_env.scene[name] for name in CUBE_NAMES]

        dists = []  # 각 reset 마다 (num_envs*4,) 평면 거리 누적
        for _ in range(args.num_resets):
            base_env.reset()
            base_xy = robot.data.root_pos_w[:, :2]  # (N, 2) world
            for cube in cubes:
                cube_xy = cube.data.root_pos_w[:, :2]
                d = torch.linalg.vector_norm(cube_xy - base_xy, dim=1)  # (N,)
                dists.append(d)

        all_d = torch.cat(dists)
        max_d = float(all_d.max().item())
        mean_d = float(all_d.mean().item())
        p99 = float(torch.quantile(all_d, 0.99).item())
        n_over = int((all_d > args.reach_radius).sum().item())
        n_total = int(all_d.numel())

        result = {
            "status": "passed" if n_over == 0 else "failed",
            "samples": n_total,
            "reach_radius": args.reach_radius,
            "max_dist": round(max_d, 4),
            "p99_dist": round(p99, 4),
            "mean_dist": round(mean_d, 4),
            "over_radius": n_over,
            "over_frac": round(n_over / max(n_total, 1), 4),
            "hint": ("OK — 전 샘플 도달 범위 내" if n_over == 0
                     else "scatter y 상한을 -0.35 로 좁히거나 reach_radius 재검토"),
        }
        print(json.dumps(result), flush=True)
        if n_over > 0:
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
