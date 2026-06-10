"""IK env 최소 스모크 — PickCubeSo101IkEnvCfg 가 gym.make + reset + step 되는지만 격리 확인.

SM 스크립트의 부가 로직(_apply_dr/viewer/state machine) 없이 IK env 자체만 테스트한다.

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/ik_env_smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_LOG = os.path.abspath("outputs/ik_smoke.txt")
os.makedirs(os.path.dirname(_LOG), exist_ok=True)
open(_LOG, "w").close()


def log(m: str) -> None:
    with open(_LOG, "a") as f:
        f.write(m + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(m, file=sys.__stderr__, flush=True)  # SM 과 동일(print 범인 여부 확인)


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(vars(args)).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402, F401
from sim_to_real.tasks.pick_cube.pick_cube_so101_ik_env_cfg import (  # noqa: E402
    PickCubeSo101IkEnvCfg,
)


def main() -> None:
    log("[smoke] main entered")
    cfg = PickCubeSo101IkEnvCfg()
    log("[smoke] cfg constructed")
    cfg.scene.num_envs = 1
    cfg.seed = 0
    log("[smoke] seed set")
    cfg.actions.arm.scale = 1.0
    log("[smoke] arm.scale set")
    # _apply_dr 복제 (object_radius_scale=0, container_angle_scale=0)
    cfg.events.randomize_cubes = None
    log("[smoke] randomize_cubes=None")
    if getattr(cfg.events, "randomize_bowl", None) is not None:
        lo, hi = cfg.events.randomize_bowl.params["angle_range_deg"]
        cfg.events.randomize_bowl.params["angle_range_deg"] = (lo * 0.0, hi * 0.0)
    log("[smoke] bowl angle scaled (DR applied)")
    log("[smoke] calling gym.make")
    env = gym.make("SimToReal-SO101-PickCube-IK-v0", cfg=cfg).unwrapped
    log("[smoke] gym.make done")
    env.reset()
    log("[smoke] reset done")
    robot = env.scene["robot"]
    log(f"[smoke] action space={env.action_space.shape} body0={robot.data.body_names[:3]}")
    a = torch.zeros((1, 8), device=env.device)
    env.step(a)
    log("[smoke] step done")
    env.close()
    log("[smoke] DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        import traceback

        log(f"[smoke] EXIT ({type(e).__name__}): " + traceback.format_exc())
        raise
    finally:
        app.close()
