"""IsaacLab 기본 환경 step 성능 벤치 (카메라 제외, 물리만).

목표:
  IsaacLab PickCube 환경(카메라 OFF)의 step 성능을 측정.
  ovrtx 의 순수 렌더 성능과의 비교 기준이 아니라,
  물리 step 만의 오버헤드를 파악하기 위함(ovrtx 는 물리 없음).

실행:  uv run --group isaac python scripts/perf/isaac_env_step_throughput.py
출력:  성능 로그(step FPS)
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab env step throughput (no cameras)")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
args.headless = True
# disable cameras to avoid rendering overhead

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- imports that require SimApp to be running ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    print("=" * 80)
    print("[bench-isaac-step] IsaacLab PickCube env step throughput (no cameras)")
    print("[bench-isaac-step] 물리 step 성능만 측정 (렌더 제외)", flush=True)

    device: str = args.device  # e.g., "cuda:0"
    task = "SimToReal-SO101-PickCube-v0"

    try:
        print("[bench-isaac-step] 환경 cfg 파싱...", flush=True)
        env_cfg = parse_env_cfg(task, device=device, num_envs=1)

        # episode 길이 충분히 설정
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            100 * env_cfg.sim.dt * env_cfg.decimation + 5.0,
        )

        print("[bench-isaac-step] gym.make 중...", flush=True)
        env = gym.make(task, cfg=env_cfg).unwrapped
        print(f"[bench-isaac-step] ✓ 환경 생성 완료 (num_envs={env.num_envs})", flush=True)

        # warmup
        print("[bench-isaac-step] warmup 40 step...", flush=True)
        obs, info = env.reset()
        for warmup_step in range(40):
            if warmup_step % 10 == 0:
                print(f"  warmup {warmup_step}/40", flush=True)
            action = torch.tensor(
                env.action_space.sample(), dtype=torch.float32, device=device
            )
            obs, rewards, terminateds, truncateds, info = env.step(action)

        # 성능 측정
        print("[bench-isaac-step] 성능 측정: 100 step (물리만, 렌더 제외)...", flush=True)
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
        print("[bench-isaac-step] ✓✓ 측정 완료 (카메라/렌더 OFF)")
        print(f"  벽시계: {elapsed:.3f} sec for {n_steps} step")
        print(f"  쓰루풋: {fps:.2f} FPS (물리 step 만)")
        print(f"  프레임당: {elapsed/n_steps*1000:.2f} ms")
        print("=" * 80)

        env.close()

    except Exception as e:
        print(f"\n[bench-isaac-step] ❌ 측정 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
