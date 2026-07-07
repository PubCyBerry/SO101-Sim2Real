"""Interactive SO-101 pick-place state machine — Isaac side (ZMQ).

Minimal 2-process, pure-ZMQ pick-place. This is the **isaac-sim** half; the planner
half is ``curobo_batch_planner.py`` (curobo-datagen container).

    ┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
    │ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py  (this file)     │
    │  cuRobo v0.8 collision-free  │◀──REQ───│  IsaacLab pick_cube env variant  │
    │  full pick-place planner     │───REP──▶│  reset(DR)→read cube/bowl→plan→  │
    │  REP  tcp://*:5599           │  :5599  │  env.step replay→success check   │
    └──────────────────────────────┘         └──────────────────────────────────┘

No ROS, no separate executor: we read cube/bowl poses (``env.scene``) and drive joints
(``env.step``) directly — the cuRobo planner is the only remote piece.

Livestream keyboard (needs --livestream; keys arrive via the WebRTC client):
  N = reset to a NEW random DR layout (robot → init)
  R = reset to the SAME layout as before (robot → init)
  B = request cuRobo planning (ZMQ) and run the pick-place
R/N (incl. mid-manipulation) cancel remaining robot actions and reset the robot pose +
scene. Ctrl-C / close the stream to stop.

Run (two terminals; both services use network_mode: host so ZMQ localhost works):

    # 1) planner  (curobo-datagen)
    docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
        python /workspace/scripts/cuRobo/curobo_batch_planner.py

    # 2) this SM  (isaac-sim)
    docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
        python /workspace/scripts/cuRobo/pickplace_sm.py \
        --task SimToReal-SO101-PickCube-DR-v0 --livestream 2

Watch: WebRTC livestream at :49100 (LIVESTREAM=1 + PUBLIC_IP for remote relay).
Env variants (--task): PickCube-v0 (fixed) · -DR-v0 · -DRBase-v0 · -Eval-v0 · -DR-Eval-v0
(see src/sim_to_real/tasks/pick_cube/__init__.py).
"""
import argparse

from isaaclab.app import AppLauncher

# ── CLI — AppLauncher needs a SimulationApp before ANY isaac/env import ──────────
parser = argparse.ArgumentParser(description="SO-101 pick-place SM (Isaac side, ZMQ→cuRobo)")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-DR-v0",
                    help="registered pick_cube env variant (tasks/pick_cube/__init__.py)")
parser.add_argument("--planner", default="tcp://127.0.0.1:5599", help="cuRobo planner ZMQ REQ endpoint")
parser.add_argument("--grasp_z", type=float, default=0.06, help="grasp height in robot-base frame (m)")
parser.add_argument("--settle", type=int, default=5, help="physics steps to settle after each reset")
parser.add_argument("--bowl_tol", type=float, default=0.06,
                    help="success = cube-center within this xy radius of bowl center (m). Tuning knob.")
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.2, 0.8, 1.2],
                    help="viewport/livestream camera eye (env-relative)")
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.1, 0.7],
                    help="viewport/livestream camera lookat (env-relative)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ⚠ AppLauncher gets a WHITELIST only (AGENTS.md): passing vars(args) whole feeds custom
#   args (--task/--cam_eye/…) into _prepare_ui and breaks livestream viewport docking.
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

# ── isaac / project imports (only valid after SimulationApp exists) ──────────────
import json  # noqa: E402
import math  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
import gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: F401,E402  (gym.register side effect)
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms  # noqa: E402
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


# ── task-space reads (robot base frame) ─────────────────────────────────────────
def _cube_bowl_in_base(env):
    """Cube (6D) + bowl (xy) in the robot base_link frame — the planner's expected input
    frame (it applies Rz(90)+BASE_T internally, see curobo_batch_planner.usd_to_urdf).
    Returns cube [x, y, grasp_z, qw,qx,qy,qz] (z = fixed grasp height, since the measured
    z is noisy at rest) and bowl [x, y]. Planner face-aligns to the cube yaw."""
    robot = env.scene["robot"]
    base_p, base_q = robot.data.root_pos_w[:1], robot.data.root_quat_w[:1]
    cube = env.scene[CUBE]
    cp, cq = subtract_frame_transforms(base_p, base_q, cube.data.root_pos_w[:1], cube.data.root_quat_w[:1])
    bowl = env.scene[BOWL]
    bp, _ = subtract_frame_transforms(base_p, base_q, bowl.data.root_pos_w[:1], bowl.data.root_quat_w[:1])
    cp, cq, bp = cp[0].tolist(), cq[0].tolist(), bp[0].tolist()
    return [cp[0], cp[1], args.grasp_z, *cq], [bp[0], bp[1]]


def _cube_in_bowl(env):
    """Success proxy: cube center within --bowl_tol (xy) of the bowl center. World frame."""
    c = env.scene[CUBE].data.root_pos_w[0]
    b = env.scene[BOWL].data.root_pos_w[0]
    return bool(torch.linalg.norm(c[:2] - b[:2]) < args.bowl_tol)


# ── per-step task-space logging ─────────────────────────────────────────────────
def _wrap180(rad):
    return (math.degrees(rad) + 180.0) % 360.0 - 180.0


def _log_state(env):
    """Print EE (gripper) 6D pose + cube 6D pose + bowl xy each step, all in the robot base
    frame (task-space). pose = position (m), Euler-XYZ (deg), quaternion (wxyz)."""
    robot = env.scene["robot"].data
    bp, bq = robot.root_pos_w[:1], robot.root_quat_w[:1]
    ee, cube, bowl = env.scene["ee_frame"].data, env.scene[CUBE].data, env.scene[BOWL].data
    ep, eq = subtract_frame_transforms(bp, bq, ee.target_pos_w[:1, 0], ee.target_quat_w[:1, 0])
    cp, cq = subtract_frame_transforms(bp, bq, cube.root_pos_w[:1], cube.root_quat_w[:1])
    wp, _ = subtract_frame_transforms(bp, bq, bowl.root_pos_w[:1], bowl.root_quat_w[:1])
    ep, eq, cp, cq, wp = ep[0], eq[0], cp[0], cq[0], wp[0]
    ee_e = tuple(_wrap180(a[0].item()) for a in euler_xyz_from_quat(eq.unsqueeze(0)))
    cu_e = tuple(_wrap180(a[0].item()) for a in euler_xyz_from_quat(cq.unsqueeze(0)))
    print(f"[ee]   pos=({ep[0]:+.3f},{ep[1]:+.3f},{ep[2]:+.3f}) "
          f"eul=({ee_e[0]:+6.1f},{ee_e[1]:+6.1f},{ee_e[2]:+6.1f}) "
          f"quat=({eq[0]:+.3f},{eq[1]:+.3f},{eq[2]:+.3f},{eq[3]:+.3f})", flush=True)
    print(f"[cube] pos=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f}) "
          f"eul=({cu_e[0]:+6.1f},{cu_e[1]:+6.1f},{cu_e[2]:+6.1f}) "
          f"quat=({cq[0]:+.3f},{cq[1]:+.3f},{cq[2]:+.3f},{cq[3]:+.3f})  "
          f"bowl_xy=({wp[0]:+.3f},{wp[1]:+.3f})", flush=True)


def _step(env, action):
    """env.step + per-step EE/cube/bowl task-space state print."""
    out = env.step(action)
    _log_state(env)
    return out


# ── livestream keyboard (R/N/B one-shot flag) ────────────────────────────────────
def _key_listener():
    """Subscribe to carb keyboard (comes through the WebRTC livestream client). R/N/B set a
    one-shot flag consumed by the loop. Returns a dict holding the flag + the subscription."""
    state = {"key": None}
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    inp = carb.input.acquire_input_interface()

    def _on_kbd(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                state["key"] = "R"
            elif event.input == carb.input.KeyboardInput.N:
                state["key"] = "N"
            elif event.input == carb.input.KeyboardInput.B:
                state["key"] = "B"
        return True

    state["_sub"] = inp.subscribe_to_keyboard_events(keyboard, _on_kbd)  # keep ref alive
    return state


def _poll_key(state):
    """Consume and return a pending R/N/B key press (or None)."""
    k, state["key"] = state["key"], None
    return k


def _wait_key(state):
    """Pump the app (livestream stays live) until a key is pressed. Returns the key, or None
    if the window closed. Honors an already-pending press (doesn't drop it)."""
    while simulation_app.is_running():
        k = _poll_key(state)
        if k is not None:
            return k
        simulation_app.update()
    return None


# ── DR layout snapshot / restore (so R replays the exact same scene) ─────────────
def _capture_layout(env):
    """Snapshot the DR cube + bowl world poses so R can restore this exact layout."""
    c, b = env.scene[CUBE].data, env.scene[BOWL].data
    return {"cube": (c.root_pos_w[:1].clone(), c.root_quat_w[:1].clone()),
            "bowl": (b.root_pos_w[:1].clone(), b.root_quat_w[:1].clone())}


def _restore_layout(env, layout):
    """Overwrite the freshly-DR'd cube + bowl with the saved poses (zero velocity)."""
    for name, k in ((CUBE, "cube"), (BOWL, "bowl")):
        pos, quat = layout[k]
        obj = env.scene[name]
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


# ── planner request/reply ────────────────────────────────────────────────────────
def _recv_plan(sock, poller):
    """Block for the planner reply while KEEPING the Kit app pumping — else the app freezes
    during the (multi-second) plan and the WebRTC livestream stops accepting input. The
    short 2 ms poll timeout keeps the wait render-bound (~30 FPS), not poll-bound. Returns
    None if the window is closed."""
    while not poller.poll(2):
        if not simulation_app.is_running():
            return None
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
    # Replay the WHOLE planned trajectory: drop the early-cut terminations. `success` fires
    # while the held cube merely passes over the bowl mid-transit (z≈0.13, inside task_done's
    # [+0.005,+0.18] height band) → env would auto-reset before place/release ever runs. Keep
    # only time_out (30 s = 900 steps ≫ ~442-row traj); judge success at the end via _cube_in_bowl.
    for _term in ("success", "cube_lost"):
        if hasattr(env_cfg.terminations, _term):
            setattr(env_cfg.terminations, _term, None)
    # viewport / livestream camera — env-relative origin (env0 = world origin here)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args.cam_eye)
    env_cfg.viewer.lookat = tuple(args.cam_target)
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

    key = _key_listener()
    layout = None

    def _reset(which):
        """Scene reset: robot → init (env.reset), cube → NEW DR (N) or SAVED layout (R)."""
        nonlocal layout
        env.reset()
        if which == "R" and layout is not None:
            _restore_layout(env, layout)     # overwrite the fresh DR with the saved layout
        for _ in range(args.settle):         # settle (hold init; zeros would drift the arm)
            _step(env, hold)
        if which == "N":
            layout = _capture_layout(env)    # remember this new layout for a later R

    def _manipulate():
        """B: request the cuRobo plan for the current cube and replay it. → (status, val):
        ("closed",_) window closed · ("abort","N"/"R") R/N mid-run · ("done", success_bool)."""
        cube_req, bowl_req = _cube_bowl_in_base(env)
        start = robot.data.joint_pos[0][so101_idx].tolist()  # actual start pose (rad, SO101 order)
        sock.send_string(json.dumps({"cmd": "plan_pickplace", "cubes": [cube_req],
                                     "bowl": bowl_req, "start": [start]}))
        rep = _recv_plan(sock, poller)
        if rep is None:
            return "closed", None
        traj = rep.get("trajectories", [None])[0] if rep.get("ok") else None
        if traj is None:
            print(f"[sm] plan FAIL cube={cube_req[:2]}", flush=True)
            return "done", False
        # planner row = (arm deg ×5, gripper feature) → sim joint radians (SO101 order) → action.
        tgt = np.stack([policy_feature_to_sim_joint_radians(np.asarray(r, np.float32)) for r in traj])
        success = None
        for row in tgt:
            k = _poll_key(key)
            if k in ("N", "R"):              # cancel remaining actions → next reset command
                return "abort", k
            action = torch.as_tensor(row, device=env.device, dtype=torch.float32).unsqueeze(0)
            _obs, _rew, term, trunc, _info = _step(env, action)
            if bool(term[0]) or bool(trunc[0]):
                success = _cube_in_bowl(env)  # read terminal state before env auto-resets
                break
        if success is None:
            success = _cube_in_bowl(env)
        return "done", bool(success)

    _reset("N")  # initial scene = new DR layout
    print("[sm] ready — B=plan+manipulate · N=new layout · R=same layout (livestream keys)",
          flush=True)
    n_ok = n_run = 0
    while simulation_app.is_running():
        k = _wait_key(key)
        if k is None:
            break                            # window closed
        if k in ("N", "R"):
            _reset(k)
            print(f"[sm] reset ({k})", flush=True)
            continue
        status, val = _manipulate()          # k == "B"
        if status == "closed":
            break
        if status == "abort":                # R/N mid-run → cancel + reset robot pose + scene
            _reset(val)
            print(f"[sm] ABORT → reset ({val})", flush=True)
            continue
        n_run += 1
        n_ok += val
        print(f"[sm] manipulate {'OK ' if val else 'FAIL'} ({n_ok}/{n_run} placed)", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
