"""IsaacLab TiledCamera 렌더 throughput 벤치 — P1 비교 기준.

목표:
  IsaacLab PickCube 환경에서 3 카메라(top/wrist/front) 활성화 후
  환경 렌더 step 의 throughput(FPS) 를 측정하여 ovrtx 와 비교한다.

실행:  uv run --group isaac python scripts/perf/tiled_camera_throughput_bench.py
출력:  성능 로그(FPS)

AppLauncher 로 Isaac Sim 부팅 후 gym.make 로 환경 로드.
warmup 후 100 step rollout 의 wall-clock 을 측정.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab TiledCamera throughput bench")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
args.headless = True
args.enable_cameras = True  # TiledCamera 렌더 활성화

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- imports that require SimApp to be running ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # registers env

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (
    add_pick_cube_cameras
)


def main() -> None:
    print("=" * 80)
    print("[bench-isaac] IsaacLab TiledCamera throughput 벤치")
    print("[bench-isaac] gym.make('SimToReal-SO101-PickCube-v0') with 3 cameras", flush=True)

    device: str = args.device  # e.g., "cuda:0"
    task = "SimToReal-SO101-PickCube-v0"

    try:
        print("[bench-isaac] 환경 cfg 파싱...", flush=True)
        env_cfg = parse_env_cfg(task, device=device, num_envs=1)

        # 카메라 3개 주입
        print("[bench-isaac] 카메라 3개 주입 (top/wrist/front)...", flush=True)
        env_cfg.scene = add_pick_cube_cameras(env_cfg.scene)

        # episode 길이 충분히 설정 (100 step + 여유)
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            100 * env_cfg.sim.dt * env_cfg.decimation + 5.0,
        )

        print("[bench-isaac] gym.make 중...", flush=True)
        env = gym.make(task, cfg=env_cfg).unwrapped
        print(f"[bench-isaac] ✓ 환경 생성 완료 (num_envs={env.num_envs})", flush=True)

        # warmup
        print("[bench-isaac] warmup 40 step...", flush=True)
        obs, info = env.reset()
        for warmup_step in range(40):
            if warmup_step % 10 == 0:
                print(f"  warmup {warmup_step}/40", flush=True)
            action = torch.tensor(
                env.action_space.sample(), dtype=torch.float32, device=device
            )
            obs, rewards, terminateds, truncateds, info = env.step(action)

        # 성능 측정
        print("[bench-isaac] 성능 측정: 100 step 렌더 + 물리...", flush=True)
        n_steps = 100

        t0 = time.perf_counter()
        for step_idx in range(n_steps):
            if step_idx % 20 == 0:
                print(f"  step {step_idx}/{n_steps}", flush=True)
            action = torch.tensor(
                env.action_space.sample(), dtype=torch.float32, device=device
            )
            obs, rewards, terminateds, truncateds, info = env.step(action)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        fps = n_steps / elapsed

        print("\n" + "=" * 80)
        print("[bench-isaac] ✓✓ TiledCamera 측정 완료")
        print(f"  벽시계: {elapsed:.3f} sec for {n_steps} step")
        print(f"  쓰루풋: {fps:.2f} FPS (3 카메라 + 물리 step)")
        print(f"  프레임당: {elapsed/n_steps*1000:.2f} ms")
        print("=" * 80)

        env.close()

    except Exception as e:
        print(f"\n[bench-isaac] ❌ 측정 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
