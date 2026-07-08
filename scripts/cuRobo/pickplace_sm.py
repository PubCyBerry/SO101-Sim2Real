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

Multi-env (``--num_envs N``): all N envs reset/plan/replay in **lockstep**. One batch ZMQ
request carries per-env cube/bowl/start; the planner returns per-env trajectories
(null = plan fail → that env holds init). Shorter trajectories pad with their last row;
success is judged per env at the end. Viewer camera follows env 0.

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
parser.add_argument("--num_envs", type=int, default=1, help="parallel envs (lockstep plan+replay)")
parser.add_argument("--grasp_z", type=float, default=0.06, help="grasp height in robot-base frame (m)")
parser.add_argument("--settle", type=int, default=5, help="physics steps to settle after each reset")
parser.add_argument("--bowl_tol", type=float, default=0.06,
                    help="success = cube-center within this xy radius of bowl center (m). Tuning knob.")
parser.add_argument("--seed", type=int, default=0, help="base seed for auto trials and env reset")
parser.add_argument("--auto_trials", type=int, default=0,
                    help="run this many random trials without keyboard input, then exit")
parser.add_argument("--record_viewport_dir", default=None,
                    help="directory for auto-trial viewport MP4 files. Use 'none' to disable video.")
parser.add_argument("--summary_dir", default=None,
                    help="directory for auto-trial summary.json. Defaults to record_viewport_dir or scratch.")
parser.add_argument("--record_fps", type=int, default=30, help="viewport MP4 FPS")
parser.add_argument("--record_every", type=int, default=1, help="record every N simulation steps")
parser.add_argument("--planner_knobs_json", default=None,
                    help="JSON object forwarded to cuRobo planner request.knobs")
parser.add_argument("--log_every", type=int, default=1,
                    help="print EE/cube state every N env steps. Use 0 to disable per-step logs.")
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
from pathlib import Path  # noqa: E402

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


class ViewportVideoRecorder:
    """Active viewport RGB를 step마다 MP4로 기록한다(auto trial 전용)."""

    def __init__(self, out_dir, fps=30, every=1):
        self.out_dir = Path(out_dir) if out_dir else None
        self.fps = int(fps)
        self.every = max(1, int(every))
        self.writer = None
        self.path = None
        self.step_i = 0
        self._imageio = None
        self._annotator = None
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio  # noqa: PLC0415
        import omni.kit.viewport.utility as viewport_utils  # noqa: PLC0415
        import omni.replicator.core as rep  # noqa: PLC0415

        viewport = viewport_utils.get_active_viewport()
        if viewport is None or not viewport.render_product_path:
            raise RuntimeError(
                "active viewport render product 없음. --record_viewport_dir 사용 시 GUI 또는 --livestream 2로 실행하세요."
            )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(viewport.render_product_path)
        self._imageio = imageio
        self._annotator = annotator
        self.render_product_path = str(viewport.render_product_path)

    @property
    def enabled(self):
        return self.out_dir is not None

    def start(self, trial_idx):
        if not self.enabled:
            return None
        self.close()
        self.path = self.out_dir / f"trial_{trial_idx:03d}.mp4"
        self.step_i = 0
        self.writer = self._imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        )
        return str(self.path)

    def capture(self):
        if self.writer is None:
            return
        self.step_i += 1
        if self.step_i % self.every:
            return
        arr = np.asarray(self._annotator.get_data())
        if arr.size == 0:
            return
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            self.writer.append_data(arr.astype(np.uint8))

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ── task-space reads (robot base frame, per-env) ─────────────────────────────────
def _cubes_bowls_in_base(env):
    """Per-env cube (6D) + bowl (xy) in each robot's base_link frame — the planner input
    (it applies Rz(90)+BASE_T internally, see curobo_batch_planner.usd_to_urdf).
    Returns (cubes [N][x,y,grasp_z,qw,qx,qy,qz], bowls [N][x,y]). z = fixed grasp height
    (measured z is noisy at rest); planner face-aligns to the cube yaw quat."""
    robot = env.scene["robot"].data
    cp, cq = subtract_frame_transforms(robot.root_pos_w, robot.root_quat_w,
                                       env.scene[CUBE].data.root_pos_w,
                                       env.scene[CUBE].data.root_quat_w)
    wp, _ = subtract_frame_transforms(robot.root_pos_w, robot.root_quat_w,
                                      env.scene[BOWL].data.root_pos_w,
                                      env.scene[BOWL].data.root_quat_w)
    cubes = [[cp[i, 0].item(), cp[i, 1].item(), args.grasp_z, *cq[i].tolist()]
             for i in range(env.num_envs)]
    bowls = [[wp[i, 0].item(), wp[i, 1].item()] for i in range(env.num_envs)]
    return cubes, bowls


def _cubes_in_bowl(env):
    """Per-env success proxy: cube center within --bowl_tol (xy) of its bowl. → [bool ×N]."""
    c = env.scene[CUBE].data.root_pos_w
    b = env.scene[BOWL].data.root_pos_w
    return (torch.linalg.norm(c[:, :2] - b[:, :2], dim=1) < args.bowl_tol).tolist()


# ── per-step task-space logging ─────────────────────────────────────────────────
def _wrap180(rad):
    return (math.degrees(rad) + 180.0) % 360.0 - 180.0


def _log_state(env):
    """Print env-0 EE/cube 6D pose + bowl xy each step, robot-base frame (multi-env 은 env 0 만
    — 전 env 출력은 로그 홍수). pose = position (m), Euler-XYZ (deg), quaternion (wxyz)."""
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


def _step(env, action, recorder=None):
    """env.step + per-step EE/cube/bowl task-space state print."""
    out = env.step(action)
    _step.i = getattr(_step, "i", 0) + 1
    if args.log_every > 0 and _step.i % args.log_every == 0:
        _log_state(env)
    if recorder is not None:
        recorder.capture()
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
    if state is None:
        return None
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


# ── DR layout snapshot / restore (so R replays the exact same scene, all envs) ───
def _capture_layout(env):
    """Snapshot per-env DR cube + bowl world poses so R can restore this exact layout."""
    c, b = env.scene[CUBE].data, env.scene[BOWL].data
    return {"cube": (c.root_pos_w.clone(), c.root_quat_w.clone()),
            "bowl": (b.root_pos_w.clone(), b.root_quat_w.clone())}


def _restore_layout(env, layout):
    """Overwrite the freshly-DR'd cube + bowl with the saved poses (zero velocity), all envs."""
    for name, k in ((CUBE, "cube"), (BOWL, "bowl")):
        pos, quat = layout[k]
        obj = env.scene[name]
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))


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
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    remove_pick_cube_cameras(env_cfg)  # SM plans on state only → skip camera spawn/render
    # Robot spawns AT the start pose from frame 0 (no neutral→init transient that corrupted
    # early inference), with reset jitter zeroed so the start is deterministic.
    env_cfg.scene.robot.init_state.joint_pos = dict(INIT_RAD)
    if hasattr(env_cfg.events, "reset_robot_joints"):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    # Replay the WHOLE planned trajectory: drop the early-cut terminations. `success` fires
    # while the held cube merely passes over the bowl mid-transit (z≈0.13, inside task_done's
    # [+0.005,+0.18] height band) → env would auto-reset before place/release ever runs. Keep
    # only time_out (30 s = 900 steps ≫ ~442-row traj); judge success at the end via _cubes_in_bowl.
    for _term in ("success", "cube_lost"):
        if hasattr(env_cfg.terminations, _term):
            setattr(env_cfg.terminations, _term, None)
    # viewport / livestream camera — env-relative origin (env0 = world origin here)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args.cam_eye)
    env_cfg.viewer.lookat = tuple(args.cam_target)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print(f"[sm] env={args.task} action_dim={env.action_space.shape} device={env.device}", flush=True)

    robot = env.scene["robot"]
    so101_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]  # articulation → SO101 order
    hold = torch.tensor(INIT_ACTION, device=env.device,
                        dtype=torch.float32).repeat(env.num_envs, 1)  # (N,6) init hold

    sock = zmq.Context().socket(zmq.REQ)
    sock.connect(args.planner)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[sm] planner {args.planner}", flush=True)

    key = None if args.auto_trials > 0 else _key_listener()
    layout = None
    last_manip = {}
    planner_knobs = json.loads(args.planner_knobs_json) if args.planner_knobs_json else None
    if planner_knobs is not None and not isinstance(planner_knobs, dict):
        raise ValueError("--planner_knobs_json must decode to a JSON object")

    def _reset(which, seed=None, recorder=None):
        """Scene reset: robot → init (env.reset), cube → NEW DR (N) or SAVED layout (R)."""
        nonlocal layout
        if seed is None:
            env.reset()
        else:
            env.reset(seed=int(seed))
        if which == "R" and layout is not None:
            _restore_layout(env, layout)     # overwrite the fresh DR with the saved layout
        for _ in range(args.settle):         # settle (hold init; zeros would drift the arm)
            _step(env, hold, recorder)
        if which == "N":
            layout = _capture_layout(env)    # remember this new layout for a later R

    def _manipulate(recorder=None, plan_seed=None):
        """B: batch-plan all envs' cubes and replay the trajectories in lockstep.

        → (status, val): ("closed",_) window closed · ("abort","N"/"R") R/N mid-run ·
        ("done", (n_planned, n_placed)). plan-fail env 는 init hold(카운트는 실패 처리),
        짧은 궤적은 마지막 row 로 패딩 — 모든 env 가 같은 step 수를 소화한다."""
        nonlocal last_manip
        cubes, bowls = _cubes_bowls_in_base(env)
        starts = [robot.data.joint_pos[i][so101_idx].tolist() for i in range(env.num_envs)]
        req = {"cmd": "plan_pickplace", "cubes": cubes, "bowl": bowls, "start": starts}
        req_knobs = dict(planner_knobs or {})
        if plan_seed is not None:
            req_knobs.setdefault("seed", int(plan_seed))
        if req_knobs:
            req["knobs"] = req_knobs
        sock.send_string(json.dumps(req))
        rep = _recv_plan(sock, poller)
        if rep is None:
            return "closed", None
        trajs = rep.get("trajectories") if rep.get("ok") else None
        last_manip = {
            "request": {"cubes": cubes, "bowls": bowls, "starts": starts},
            "planner_ok": bool(rep.get("ok")),
            "diagnostics": rep.get("diagnostics", []),
            "planned": [],
            "placed": [],
            "n_steps": 0,
        }
        if not trajs or all(t is None for t in trajs):
            print(f"[sm] plan FAIL (all {env.num_envs} envs)", flush=True)
            last_manip["planned"] = [False] * env.num_envs
            last_manip["placed"] = [False] * env.num_envs
            return "done", (env.num_envs, 0)
        planned = [t is not None for t in trajs]
        n_steps = max(len(t) for t in trajs if t is not None)
        last_manip["planned"] = [bool(v) for v in planned]
        last_manip["n_steps"] = int(n_steps)
        # planner row = (arm deg ×5, gripper feature) → sim joint radians (SO101 order).
        # plan-fail env 는 init hold, 짧은 궤적은 last-row 패딩 → (T, N, 6) lockstep 텐서.
        per_env = []
        for t in trajs:
            if t is None:
                per_env.append(np.tile(np.asarray(INIT_ACTION, np.float32), (n_steps, 1)))
                continue
            rows = np.stack([policy_feature_to_sim_joint_radians(np.asarray(r, np.float32))
                             for r in t])
            if len(rows) < n_steps:
                rows = np.concatenate([rows, np.tile(rows[-1:], (n_steps - len(rows), 1))])
            per_env.append(rows)
        tgt = np.stack(per_env, axis=1)  # (T, N, 6)
        success = None
        for step_rows in tgt:
            k = _poll_key(key)
            if k in ("N", "R"):              # cancel remaining actions → next reset command
                return "abort", k
            action = torch.as_tensor(step_rows, device=env.device, dtype=torch.float32)
            _obs, _rew, term, trunc, _info = _step(env, action, recorder)
            if bool(term.any()) or bool(trunc.any()):
                success = _cubes_in_bowl(env)  # read terminal state before env auto-resets
                break
        if success is None:
            success = _cubes_in_bowl(env)
        last_manip["placed"] = [bool(v) for v in success]
        n_placed = sum(1 for i, ok in enumerate(success) if planned[i] and ok)
        for i, (p, ok) in enumerate(zip(planned, success)):
            print(f"[sm]   env{i}: plan={'ok' if p else 'FAIL'} placed={bool(ok)}", flush=True)
        return "done", (env.num_envs, n_placed)

    if args.auto_trials > 0:
        record_dir = args.record_viewport_dir
        if record_dir is not None and record_dir.lower() in {"", "none", "off", "false", "0"}:
            record_dir = None
        summary_dir = args.summary_dir or record_dir or "/workspace/scratch/curobo-auto-trials"
        recorder = ViewportVideoRecorder(record_dir, fps=args.record_fps, every=args.record_every)
        trials = []
        print(f"[sm] auto_trials={args.auto_trials} seed={args.seed} "
              f"record_dir={record_dir} summary_dir={summary_dir}", flush=True)
        try:
            for trial_i in range(1, args.auto_trials + 1):
                if not simulation_app.is_running():
                    break
                trial_seed = int(args.seed + trial_i - 1)
                video_path = recorder.start(trial_i) if recorder.enabled else None
                print(f"[sm] auto trial {trial_i}/{args.auto_trials} seed={trial_seed}", flush=True)
                status = "unknown"
                val = None
                try:
                    _reset("N", seed=trial_seed, recorder=recorder)
                    status, val = _manipulate(recorder=recorder, plan_seed=trial_seed)
                finally:
                    recorder.close()
                n_attempt, n_placed = val if status == "done" else (env.num_envs, 0)
                trial_summary = {
                    "trial": trial_i,
                    "seed": trial_seed,
                    "status": status,
                    "video": video_path,
                    "n_attempt": int(n_attempt),
                    "n_placed": int(n_placed),
                    **last_manip,
                }
                trials.append(trial_summary)
                print(f"[sm] auto trial {trial_i}: status={status} placed={n_placed}/{n_attempt} "
                      f"video={video_path}", flush=True)
                if status == "closed":
                    break
        finally:
            summary = {
                "task": args.task,
                "num_envs": args.num_envs,
                "base_seed": args.seed,
                "record_fps": args.record_fps,
                "record_every": args.record_every,
                "viewport_render_product": getattr(recorder, "render_product_path", None),
                "trials": trials,
            }
            summary_path = Path(summary_dir) / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[sm] auto summary → {summary_path}", flush=True)
            env.close()
            simulation_app.close()
        return

    _reset("N")  # initial scene = new DR layout
    print(f"[sm] ready — B=plan+manipulate · N=new layout · R=same layout "
          f"(livestream keys, {env.num_envs} envs)", flush=True)
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
        n_attempt, n_placed = val
        n_run += n_attempt
        n_ok += n_placed
        print(f"[sm] manipulate {n_placed}/{n_attempt} ({n_ok}/{n_run} placed total)", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
