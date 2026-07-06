"""Interactive SO-101 pick-place state machine — Isaac side (ZMQ).

Minimal 2-process, pure-ZMQ pick-place. This is the **isaac-sim** half; the planner
half is ``curobo_batch_planner.py`` (curobo-datagen container).

    ┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
    │ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py  (this file)     │
    │  cuRobo v0.8 collision-free  │◀──REQ───│  IsaacLab pick_cube env variant  │
    │  full pick-place planner     │───REP──▶│  reset(DR)→read cube/bowl→plan→  │
    │  REP  tcp://*:5599           │  :5599  │  env.step replay→success check   │
    └──────────────────────────────┘         └──────────────────────────────────┘

No ROS, no separate executor: inside the env we read cube/bowl poses (``env.scene``)
and drive joints (``env.step``) directly — the cuRobo planner is the only remote piece.
The SM loop over episodes = reset → plan → replay → check → next.

Run (two terminals; both services use network_mode: host so ZMQ localhost works):

    # 1) planner  (curobo-datagen)
    docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
        python /workspace/scripts/cuRobo/curobo_batch_planner.py

    # 2) this SM  (isaac-sim)
    docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
        python /workspace/scripts/cuRobo/pickplace_sm.py \
        --task SimToReal-SO101-PickCube-DR-v0 --episodes 5 --livestream 2

Watch: WebRTC livestream at :49100 (LIVESTREAM=1 + PUBLIC_IP for remote relay).
Env variants (``--task``): see src/sim_to_real/tasks/pick_cube/__init__.py
  PickCube-v0 (fixed) · -DR-v0 (full DR) · -DRBase-v0 · -Eval-v0 · -DR-Eval-v0.
"""
import argparse

from isaaclab.app import AppLauncher

# ── CLI — AppLauncher needs a SimulationApp before ANY isaac/env import ──────────
parser = argparse.ArgumentParser(description="SO-101 pick-place SM (Isaac side, ZMQ→cuRobo)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-DR-v0",
                    help="registered pick_cube env variant (tasks/pick_cube/__init__.py)")
parser.add_argument("--episodes", type=int, default=3, help="pick-place episodes to run")
parser.add_argument("--planner", default="tcp://127.0.0.1:5599", help="cuRobo planner ZMQ REQ endpoint")
parser.add_argument("--grasp_z", type=float, default=0.06, help="grasp height in robot-base frame (m)")
parser.add_argument("--settle", type=int, default=30, help="physics steps to settle after each reset")
parser.add_argument("--bowl_tol", type=float, default=0.06,
                    help="success = cube-center within this xy radius of bowl center (m). Tuning knob.")
parser.add_argument("--pause", action="store_true", help="wait for Enter between episodes (step-through)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ⚠ AppLauncher gets a WHITELIST only (AGENTS.md): passing vars(args) whole feeds custom
#   args (--task/--episodes/…) into _prepare_ui and breaks livestream viewport docking.
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

# ── isaac / project imports (only valid after SimulationApp exists) ──────────────
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
import gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: F401,E402  (gym.register side effect)
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from so101_contract.feature_codec import SO101_JOINT_ORDER, policy_feature_to_sim_joint_radians  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras  # noqa: E402

CUBE, BOWL = "Cube1", "Bowl"  # pick_cube leaf scene entity names (single cube + bowl)

# Robot start pose (degrees) — arm holds this while the cube settles, and it is the plan's
# start joint state. wrist_roll -90 = top-down-tilt-ready; gripper -10° = feature 0 (open).
INIT_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
    "wrist_flex": 50.0, "wrist_roll": -90.0, "gripper": -10.0,
}
INIT_RAD = {j: math.radians(d) for j, d in INIT_POSE_DEG.items()}
INIT_ACTION = [INIT_RAD[j] for j in SO101_JOINT_ORDER]  # env action / planner start order (rad)


def _cube_bowl_in_base(env):
    """Cube (6D) + bowl (xy) in the robot base_link frame — the planner's expected input
    frame (it applies Rz(90)+BASE_T internally, see curobo_batch_planner.usd_to_urdf)."""
    robot = env.scene["robot"]
    base_p, base_q = robot.data.root_pos_w[:1], robot.data.root_quat_w[:1]
    cube = env.scene[CUBE]
    cp, cq = subtract_frame_transforms(base_p, base_q, cube.data.root_pos_w[:1], cube.data.root_quat_w[:1])
    bowl = env.scene[BOWL]
    bp, _ = subtract_frame_transforms(base_p, base_q, bowl.data.root_pos_w[:1], bowl.data.root_quat_w[:1])
    cp, cq, bp = cp[0].tolist(), cq[0].tolist(), bp[0].tolist()
    # 6D cube [x, y, grasp_z, qw,qx,qy,qz] → planner face-aligns to cube yaw. z = fixed grasp
    # height in base frame (matches curobo_executor --grasp_z; measured z is noisy at rest).
    return [cp[0], cp[1], args.grasp_z, *cq], [bp[0], bp[1]]


def _cube_in_bowl(env):
    """Success proxy: cube center within --bowl_tol (xy) of the bowl center. World frame."""
    c = env.scene[CUBE].data.root_pos_w[0]
    b = env.scene[BOWL].data.root_pos_w[0]
    return bool(torch.linalg.norm(c[:2] - b[:2]) < args.bowl_tol)


def _recv_plan(sock, poller):
    """Block for the planner reply while KEEPING the Kit app pumping — otherwise the app
    freezes during the (multi-second) plan and the WebRTC livestream stops accepting
    zoom/drag/click input. Pump every ~50 ms until the reply lands."""
    while not poller.poll(50):
        simulation_app.update()
    return json.loads(sock.recv())


def main():
    # num_envs=1 for the interactive SM (one scene to watch).
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    remove_pick_cube_cameras(env_cfg)  # SM plans on state only → skip camera spawn/render
    # Robot spawns AT the start pose from frame 0 (no neutral→init transient that corrupted
    # early inference), with reset jitter zeroed so the start is deterministic.
    env_cfg.scene.robot.init_state.joint_pos = dict(INIT_RAD)
    if hasattr(env_cfg.events, "reset_robot_joints"):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print(f"[sm] env={args.task} action_dim={env.action_space.shape} device={env.device}", flush=True)

    robot = env.scene["robot"]
    so101_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]  # articulation → SO101 order
    hold = torch.tensor(INIT_ACTION, device=env.device, dtype=torch.float32).unsqueeze(0)

    sock = zmq.Context().socket(zmq.REQ)
    sock.connect(args.planner)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[sm] planner {args.planner}", flush=True)

    n_ok = 0
    for ep in range(args.episodes):
        env.reset()  # robot → INIT (deterministic); DR variant scatters the cube
        for _ in range(args.settle):  # HOLD init pose while the cube settles (zeros would drift the arm)
            env.step(hold)

        cube_req, bowl_req = _cube_bowl_in_base(env)
        start = robot.data.joint_pos[0][so101_idx].tolist()  # actual start pose (rad, SO101 order)
        sock.send_string(json.dumps({"cmd": "plan_pickplace", "cubes": [cube_req],
                                     "bowl": bowl_req, "start": [start]}))
        rep = _recv_plan(sock, poller)
        traj = rep.get("trajectories", [None])[0] if rep.get("ok") else None
        if traj is None:
            print(f"[sm] ep{ep} plan FAIL cube={cube_req[:2]} rep_ok={rep.get('ok')}", flush=True)
            continue

        # planner row = (arm deg ×5, gripper feature) → sim joint radians (SO101 order) → action.
        tgt = np.stack([policy_feature_to_sim_joint_radians(np.asarray(r, np.float32)) for r in traj])
        success = False
        for row in tgt:
            action = torch.as_tensor(row, device=env.device, dtype=torch.float32).unsqueeze(0)
            _obs, _rew, term, trunc, _info = env.step(action)
            if bool(term[0]) or bool(trunc[0]):
                success = _cube_in_bowl(env)  # read terminal state before env auto-resets next step
                break
        else:
            success = _cube_in_bowl(env)
        n_ok += success
        print(f"[sm] ep{ep} {'OK ' if success else 'FAIL'} cube={cube_req[:2]} waypoints={len(tgt)}",
              flush=True)
        if args.pause:
            input("[sm] Enter for next episode… ")

    print(f"[sm] done {n_ok}/{args.episodes} placed — livestream stays interactive; "
          f"close the stream window or Ctrl-C to exit", flush=True)
    while simulation_app.is_running():  # keep rendering so the stream stays interactive after the run
        simulation_app.update()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
