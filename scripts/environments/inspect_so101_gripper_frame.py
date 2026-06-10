"""SO-101 gripper/jaw body 기하 측정 — DiffIK body_name/body_offset 결정용 (1회 진단).

기존 PickCubeEnvCfg(SO-101)를 띄워 rest 자세에서 다음을 출력한다:
  · robot.data.body_names / joint_names (DiffIK body_name 후보 확인, gripper_frame_link 존재 여부)
  · "gripper" body world pose
  · 관심 body들의 "gripper" local frame 기준 위치(= body_offset 후보)
  · 두 손가락(jaw 모터 손가락 + gripper 고정 손가락) midpoint 의 gripper-local offset

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac python \
        scripts/environments/inspect_so101_gripper_frame.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

_LOG = os.path.join(tempfile.gettempdir(), "so101_inspect.txt")
open(_LOG, "w").close()


def log(msg: str) -> None:
    with open(_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, file=sys.__stderr__, flush=True)


parser = argparse.ArgumentParser(description="SO-101 gripper frame inspector")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_inv  # noqa: E402

import sim_to_real  # noqa: E402, F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import PickCubeEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = PickCubeEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    # zero action 으로 rest 자세 정착
    for _ in range(30):
        if not simulation_app.is_running():
            return
        env.step(torch.zeros((1, 6), device=env.device))

    robot = env.scene["robot"]
    names = list(robot.data.body_names)
    log("=" * 70)
    log(f"BODY NAMES ({len(names)}): {names}")
    log(f"JOINT NAMES: {list(robot.data.joint_names)}")
    log("=" * 70)

    gp = robot.data.body_pos_w[0, names.index("gripper")]
    gq = robot.data.body_quat_w[0, names.index("gripper")]
    log(f"'gripper' body  world pos={_fmt(gp)}  quat(wxyz)={_fmt(gq)}")

    def local_of(world_pos: torch.Tensor) -> torch.Tensor:
        """world point → 'gripper' body local frame."""
        return quat_apply(quat_inv(gq).unsqueeze(0), (world_pos - gp).unsqueeze(0)).squeeze(0)

    # 관심 body 들의 gripper-local 위치 (body_offset 후보)
    finger_bodies = []
    for cand in ("gripper_frame_link", "jaw", "moving_jaw_so101_v1_link",
                 "fixed_jaw", "wrist", "gripper"):
        if cand in names:
            wp = robot.data.body_pos_w[0, names.index(cand)]
            log(f"  body '{cand}': world={_fmt(wp)}  gripper-local={_fmt(local_of(wp))}")
            if "jaw" in cand or cand == "gripper":
                finger_bodies.append((cand, wp))

    # 두 손가락 midpoint (현 SM 이 grasp point 로 쓰던 정의)
    if len(finger_bodies) >= 2:
        mid = 0.5 * (finger_bodies[0][1] + finger_bodies[1][1])
        log(f"  midpoint({finger_bodies[0][0]},{finger_bodies[1][0]}): "
            f"world={_fmt(mid)}  gripper-local={_fmt(local_of(mid))}")

    # URDF gripper_frame_joint origin 을 gripper-local 로 직접 적용했을 때 world 위치 검증
    urdf_off = torch.tensor([-0.0079, -0.000218121, -0.0981274], device=env.device)
    world_from_urdf = gp + quat_apply(gq.unsqueeze(0), urdf_off.unsqueeze(0)).squeeze(0)
    log(f"  URDF gripper_frame offset applied → world={_fmt(world_from_urdf)}")

    log("=" * 70)
    log("DONE — 위 'gripper-local' 값 중 두 손가락 사이 grasp 중심을 body_offset 으로 채택.")
    env.close()


def _fmt(t: torch.Tensor) -> str:
    return "(" + ", ".join(f"{x:.4f}" for x in t.detach().cpu().tolist()) + ")"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("EXCEPTION:\n" + traceback.format_exc())
        raise
    finally:
        simulation_app.close()
