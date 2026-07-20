"""SO-101 pick-place state machine — Isaac side (ZMQ → cuRobo).

Minimal 2-process, pure-ZMQ pick-place. This is the **isaac-sim** half; the planner
half is ``curobo_batch_planner.py`` (curobo-datagen container).

    ┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
    │ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py  (this file)     │
    │  cuRobo v0.8 collision-free  │◀──REQ───│  IsaacLab pick_cube env variant  │
    │  full pick-place planner     │───REP──▶│  reset→read cube/bowl→plan→       │
    │  REP  tcp://*:5599           │  :5599  │  env.step replay→success check   │
    └──────────────────────────────┘         └──────────────────────────────────┘

No ROS, no separate executor: we read cube/bowl poses (``env.scene``) and drive joints
(``env.step``) directly — the cuRobo planner is the only remote piece.

Multi-env (``--num_envs N``, 기본 1): N envs reset/plan/replay in **lockstep**. One batch
ZMQ request carries per-env cube/bowl/start; the planner returns per-env trajectories
(null = plan fail → that env holds init). Shorter trajectories pad with their last row;
success is judged per env. Viewer camera follows env 0.

## 서브커맨드 (mode)

    random  — 통상 랜덤 DR 배치. 인터랙티브(livestream 키) 또는 --auto_trials N 자동.
    fail    — sweep 결과(--results)의 place/plan-fail 셀 좌표만 재현(인터랙티브).
    sweep   — DR 스폰영역 전체 grid+boundary 정량 평가 → JSON(--out). 자동/headless.

인터랙티브 키(livestream 필수, WebRTC 클라이언트로 전달):
  N = random: 새 DR 레이아웃 · fail: 다음 fail batch     (로봇 → init)
  R = random: 같은 레이아웃 · fail: 같은 batch           (로봇 → init)
  B = cuRobo plan 요청(ZMQ) + pick-place 실행
R/N (조작 중 포함) = 남은 로봇 동작 취소 + 재배치. Ctrl-C / 스트림 닫기로 종료.

실행(터미널 2개; 둘 다 network_mode: host 라 ZMQ localhost):

    # 1) planner  (curobo-datagen)
    docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
        python /workspace/scripts/cuRobo/curobo_batch_planner.py

    # 2) SM  (isaac-sim) — 예: 통상 랜덤, livestream 관전
    docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
        python /workspace/scripts/cuRobo/pickplace_sm.py random \
        --task SimToReal-SO101-PickCube-DR-v0 --livestream 2

관전: WebRTC :49100 (원격 relay = .env 의 LIVESTREAM=1 + PUBLIC_IP).
Env variants(--task): PickCube-v0(고정) · -DR-v0 · -DRBase-v0 · -Eval-v0 · -DR-Eval-v0
(src/sim_to_real/tasks/pick_cube/__init__.py).
"""
import argparse

from isaaclab.app import AppLauncher

# ── CLI (subcommands) — AppLauncher needs a SimulationApp before ANY isaac import ──
# 공용 인자 + AppLauncher 인자는 부모 파서 하나에 정의하고, 각 서브커맨드가 상속한다.
_common = argparse.ArgumentParser(add_help=False)
_common.add_argument("--task", default="SimToReal-SO101-PickCube-DR-v0",
                     help="registered pick_cube env variant (tasks/pick_cube/__init__.py)")
_common.add_argument("--planner", default="tcp://127.0.0.1:5599", help="cuRobo planner ZMQ REQ endpoint")
_common.add_argument("--num_envs", type=int, default=1, help="parallel envs (lockstep plan+replay)")
_common.add_argument("--grasp_z", type=float, default=0.06, help="grasp height in robot-base frame (m)")
_common.add_argument("--settle", type=int, default=5, help="physics steps to settle after each reset")
_common.add_argument("--bowl_tol", type=float, default=0.06,
                     help="success = cube-center within this xy radius of bowl center (m)")
_common.add_argument("--seed", type=int, default=0, help="base seed for reset/plan")
_common.add_argument("--planner_knobs_json", default=None,
                     help="JSON object forwarded to cuRobo planner request.knobs")
_common.add_argument("--log_every", type=int, default=1,
                     help="print EE/cube state every N env steps. 0 = disable per-step logs")
_common.add_argument("--cam_eye", type=float, nargs=3, default=[0.2, 0.8, 1.2],
                     help="viewport/livestream camera eye (env-relative)")
_common.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.1, 0.7],
                     help="viewport/livestream camera lookat (env-relative)")
AppLauncher.add_app_launcher_args(_common)

parser = argparse.ArgumentParser(description="SO-101 pick-place SM (Isaac side, ZMQ→cuRobo)")
_sub = parser.add_subparsers(dest="mode", required=True)

_p_random = _sub.add_parser("random", parents=[_common], help="통상 랜덤 DR 배치(인터랙티브/자동)")
_p_random.add_argument("--auto_trials", type=int, default=0,
                       help="키 입력 없이 이만큼 랜덤 trial 실행 후 종료(0 = 인터랙티브)")
_p_random.add_argument("--record_viewport_dir", default=None,
                       help="auto-trial viewport MP4 디렉터리. 'none' = 비활성")
_p_random.add_argument("--summary_dir", default=None,
                       help="auto-trial summary.json 디렉터리(기본 record_viewport_dir 또는 scratch)")
_p_random.add_argument("--record_fps", type=int, default=30, help="viewport MP4 FPS")
_p_random.add_argument("--record_every", type=int, default=1, help="N step 마다 1 프레임 기록")

_p_fail = _sub.add_parser("fail", parents=[_common], help="sweep 결과의 fail 좌표만 재현")
_p_fail.add_argument("--results", required=True,
                     help="sweep JSON 경로. place/plan-fail 셀 좌표만 batch 로 로드")

_p_sweep = _sub.add_parser("sweep", parents=[_common], help="스폰영역 grid+boundary 정량 평가")
_p_sweep.add_argument("--nx", type=int, default=15, help="interior grid x 해상도")
_p_sweep.add_argument("--ny", type=int, default=8, help="interior grid y 해상도")
_p_sweep.add_argument("--boundary_n", type=int, default=20,
                      help="경계(최외곽 bell/y·최내곽 base/bowl/exclude) 곡선당 샘플 수")
_p_sweep.add_argument("--trials", type=int, default=1,
                      help="셀당 반복 횟수(판정 노이즈 평균). >1 이면 --yaw random 권장")
_p_sweep.add_argument("--yaw", default="0", help="큐브 yaw(도) 고정값 또는 'random'")
_p_sweep.add_argument("--out", default="/workspace/outputs/curobo_sweep/sweep_results.json",
                      help="sweep 결과 JSON 경로(호스트 ./outputs 마운트)")
args = parser.parse_args()

# ⚠ AppLauncher gets a WHITELIST only (AGENTS.md): passing vars(args) whole feeds custom
#   args (--task/--cam_eye/…) into _prepare_ui and breaks livestream viewport docking.
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

# ── isaac / project imports (only valid after SimulationApp exists) ──────────────
import faulthandler  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

faulthandler.enable()  # C-레벨 크래시 시 파이썬 스택 덤프

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
import gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: F401,E402  (gym.register side effect)
from isaaclab.utils.math import (  # noqa: E402
    euler_xyz_from_quat, quat_from_euler_xyz, quat_mul, subtract_frame_transforms,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from so101_contract.feature_codec import SO101_JOINT_ORDER, policy_feature_to_sim_joint_radians  # noqa: E402
from sim_to_real.tasks.pick_cube import spawn_area as SA  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras  # noqa: E402

CUBE, BOWL = "Cube1", "Bowl"  # pick_cube leaf scene entity names (single cube + bowl)
FIXED_INNER_CENTER = np.array([0.0215, 0.0147, 0.0463], dtype=np.float64)  # fixed-jaw pad center (tcp-frame)
PAD_LOW_OFF = 0.075  # tcp → fixed-jaw tip approach 거리 (진단 로그용)

# Robot start pose (degrees) — arm holds this while the cube settles, and it is the plan's
# start joint state. wrist_roll -90 = top-down-tilt-ready; gripper -10° = feature 0 (open).
INIT_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
    "wrist_flex": 50.0, "wrist_roll": -90.0, "gripper": -10.0,
}
INIT_RAD = {j: math.radians(d) for j, d in INIT_POSE_DEG.items()}
INIT_ACTION = [INIT_RAD[j] for j in SO101_JOINT_ORDER]  # env action / planner start order (rad)


# ══ 포맷/수학 헬퍼 ════════════════════════════════════════════════════════════════
def _wrap180(rad):
    return (math.degrees(rad) + 180.0) % 360.0 - 180.0


def _quat_rotate_np(q, v):
    """wxyz quaternion 으로 vector 를 회전한다."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    qv = np.array([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(qv, w * v + np.cross(qv, v))


def _fmt_vec(v):
    return "(" + ",".join(f"{float(x):+.3f}" for x in v) + ")"


def _fmt_quat(q):
    return "(" + ",".join(f"{float(x):+.3f}" for x in q) + ")"


def _euler_deg_from_wxyz_np(q):
    qt = torch.tensor(np.asarray(q, dtype=np.float64), dtype=torch.float32).unsqueeze(0)
    return tuple(_wrap180(a[0].item()) for a in euler_xyz_from_quat(qt))


class ViewportVideoRecorder:
    """Active viewport RGB 를 step 마다 MP4 로 기록한다(random --auto_trials 전용)."""

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


# ══ task-space reads (robot base frame, per-env) ══════════════════════════════════
def _cubes_bowls_in_base(env):
    """Per-env cube (6D) + bowl (xy) in each robot's base_link frame — the planner input
    (it applies Rz(90)+BASE_T internally, see curobo_batch_planner.usd_to_urdf).
    Returns (cubes [N][x,y,grasp_z,qw,qx,qy,qz], bowls [N][x,y]). z = fixed grasp height
    (measured z is noisy at rest); planner extracts cube face normals directly from the quat."""
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


def _log_state(env):
    """Print env-0 EE/cube 6D pose + bowl xy each step, world frame (multi-env 은 env 0 만).
    pose = position (m), Euler-XYZ (deg), quaternion (wxyz)."""
    ee, cube, bowl = env.scene["ee_frame"].data, env.scene[CUBE].data, env.scene[BOWL].data
    ep = ee.target_pos_w[0, 0].detach().cpu().numpy()
    eq = ee.target_quat_w[0, 0].detach().cpu().numpy()
    cp = cube.root_pos_w[0].detach().cpu().numpy()
    cq = cube.root_quat_w[0].detach().cpu().numpy()
    wp = bowl.root_pos_w[0].detach().cpu().numpy()
    ee_e = _euler_deg_from_wxyz_np(eq)
    cu_e = _euler_deg_from_wxyz_np(cq)
    xax = _quat_rotate_np(eq, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    yax = _quat_rotate_np(eq, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    zax = _quat_rotate_np(eq, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    dx, dy, dz = FIXED_INNER_CENTER
    fixed_inner = ep + dx * xax + dy * yax + dz * zax
    fixed_tip = ep + PAD_LOW_OFF * zax
    print(f"[world ee]   pos={_fmt_vec(ep)} "
          f"eul=({ee_e[0]:+6.1f},{ee_e[1]:+6.1f},{ee_e[2]:+6.1f}) "
          f"quat={_fmt_quat(eq)}", flush=True)
    print(f"[world cube] pos={_fmt_vec(cp)} "
          f"eul=({cu_e[0]:+6.1f},{cu_e[1]:+6.1f},{cu_e[2]:+6.1f}) "
          f"quat={_fmt_quat(cq)}  bowl={_fmt_vec(wp)}", flush=True)
    print(f"[world jaw]  fixed_tip={_fmt_vec(fixed_tip)} fixed_inner={_fmt_vec(fixed_inner)}",
          flush=True)


def _step(env, action, recorder=None):
    """env.step + per-step EE/cube/bowl task-space state print (--log_every)."""
    out = env.step(action)
    _step.i = getattr(_step, "i", 0) + 1
    if args.log_every > 0 and _step.i % args.log_every == 0:
        _log_state(env)
    if recorder is not None:
        recorder.capture()
    return out


# ══ livestream keyboard (R/N/B one-shot flag) ═════════════════════════════════════
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


# ══ DR layout snapshot / restore (so R replays the exact same scene, all envs) ════
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


# ══ planner request/reply ═════════════════════════════════════════════════════════
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


def _sweep_summary(cells, n_targets):
    """sweep 셀 집계 + 스폰영역 기하 메타 → plot_sweep 가 읽는 JSON dict."""
    return {
        "task": args.task, "num_envs": args.num_envs, "bowl_tol": args.bowl_tol,
        "yaw": args.yaw, "trials": args.trials, "seed": args.seed,
        "grid": {"nx": args.nx, "ny": args.ny, "boundary_n": args.boundary_n},
        "spawn": {
            "bell": [list(p) for p in SA.CUBE_SCATTER_BELL],
            "x_range": list(SA.CUBE_SCATTER_X_RANGE), "y_range": list(SA.CUBE_SCATTER_Y_RANGE),
            "exclude_box": list(SA.CUBE_ARM_EXCLUDE),
            "bowl_center_xy": list(SA.BOWL_CENTER_XY), "bowl_sep": SA.MIN_BOWL_SEP,
            "base_xy": list(SA.BASE_XY), "base_sep": SA.MIN_BASE_SEP,
        },
        "n_targets": int(n_targets),
        "cells": list(cells.values()),
    }


def _build_env():
    """pick_cube env 를 SM 규약대로 생성한다(카메라 제거·init 자세 고정·조기종료 해제·뷰포트)."""
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    remove_pick_cube_cameras(env_cfg)  # SM plans on state only → skip camera spawn/render
    # SM 은 렌더 없는 state-only → 시각 DR 제거 + 표준 physics 복원. -DR 은 robot-color(Replicator)
    # 용으로 replicate_physics=False + robot-color/lights/focal 이벤트를 켜는데, 이는 렌더가 있어야
    # 의미 있고 headless 에선 physx view(get_dof_velocities)를 깨뜨린다. datagen(카메라 경로)은 유지.
    env_cfg.scene.replicate_physics = True
    for _ev in ("randomize_robot_color", "randomize_lights", "randomize_camera_focal"):
        if hasattr(env_cfg.events, _ev):
            setattr(env_cfg.events, _ev, None)
    # Robot spawns AT the start pose from frame 0 (no neutral→init transient), reset jitter zeroed.
    env_cfg.scene.robot.init_state.joint_pos = dict(INIT_RAD)
    if hasattr(env_cfg.events, "reset_robot_joints"):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    # Replay the WHOLE planned trajectory: drop early-cut terminations. `success` fires while the
    # held cube merely passes over the bowl mid-transit → env would auto-reset before release runs.
    # Keep only time_out (30 s = 900 steps ≫ ~442-row traj); judge success at end via _cubes_in_bowl.
    for _term in ("success", "cube_lost"):
        if hasattr(env_cfg.terminations, _term):
            setattr(env_cfg.terminations, _term, None)
    env_cfg.viewer.origin_type = "env"  # env-relative camera (env0 = world origin here)
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args.cam_eye)
    env_cfg.viewer.lookat = tuple(args.cam_target)
    env_cfg.seed = args.seed
    return gym.make(args.task, cfg=env_cfg).unwrapped


def main():
    env = _build_env()
    print(f"[sm] mode={args.mode} env={args.task} action_dim={env.action_space.shape} "
          f"device={env.device}", flush=True)
    robot = env.scene["robot"]
    so101_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]  # articulation → SO101 order
    hold = torch.tensor(INIT_ACTION, device=env.device,
                        dtype=torch.float32).repeat(env.num_envs, 1)  # (N,6) init hold

    sock = zmq.Context().socket(zmq.REQ)
    sock.connect(args.planner)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[sm] planner {args.planner}", flush=True)

    interactive = args.mode == "fail" or (args.mode == "random" and args.auto_trials == 0)
    key = _key_listener() if interactive else None
    layout = None
    last_manip = {}
    planner_knobs = json.loads(args.planner_knobs_json) if args.planner_knobs_json else None
    if planner_knobs is not None and not isinstance(planner_knobs, dict):
        raise ValueError("--planner_knobs_json must decode to a JSON object")

    # ── 씬 조작 primitives ────────────────────────────────────────────────────────
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

    def _reset_to_targets(batch_xy, yaws_rad, seed):
        """env.reset(robot→init) 후 Cube1 을 batch_xy(env-local) 로, Bowl 을 nominal 고정으로
        덮어써 settle. DR 배치를 통제된 타깃으로 치환(bowl 고정 = 성공맵 교란 제거). fail/sweep 공용.
        batch_xy/yaws_rad 길이 = num_envs (부분 batch 는 호출 측에서 첫 타깃으로 패딩)."""
        env.reset(seed=int(seed))
        cube, bowl = env.scene[CUBE], env.scene[BOWL]
        origins = env.scene.env_origins
        cdef = cube.data.default_root_state.clone()   # (N,13) env-local (origin 미포함)
        bdef = bowl.data.default_root_state.clone()
        tx = torch.tensor([p[0] for p in batch_xy], device=env.device, dtype=torch.float32)
        ty = torch.tensor([p[1] for p in batch_xy], device=env.device, dtype=torch.float32)
        th = torch.tensor(list(yaws_rad), device=env.device, dtype=torch.float32)
        z0 = torch.zeros_like(th)
        cq = quat_mul(cdef[:, 3:7], quat_from_euler_xyz(z0, z0, th))  # 안착 자세 + yaw
        cpos = torch.stack([tx, ty, cdef[:, 2]], dim=-1) + origins
        cube.write_root_pose_to_sim(torch.cat([cpos, cq], dim=-1))
        cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        bpos = bdef[:, :3] + origins                                 # bowl → nominal 고정
        bowl.write_root_pose_to_sim(torch.cat([bpos, bdef[:, 3:7]], dim=-1))
        bowl.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        for _ in range(args.settle):
            _step(env, hold)

    def _manipulate(recorder=None, plan_seed=None):
        """B: batch-plan all envs' cubes and replay the trajectories in lockstep.

        → (status, val): ("closed",_) window closed · ("abort","N"/"R") R/N mid-run ·
        ("done", (n_attempt, n_placed)). plan-fail env 는 init hold(실패 처리),
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
        _manipulate.request_i = getattr(_manipulate, "request_i", 0) + 1
        request_i = _manipulate.request_i
        print(f"[sm] send plan_request #{request_i}: envs={env.num_envs} "
              f"cube0={_fmt_vec(cubes[0][:3])} bowl0={_fmt_vec(bowls[0])}", flush=True)
        sock.send_string(json.dumps(req))
        rep = _recv_plan(sock, poller)
        if rep is None:
            return "closed", None
        print(f"[sm] recv plan_reply #{request_i}: ok={bool(rep.get('ok'))}", flush=True)
        trajs = rep.get("trajectories") if rep.get("ok") else None
        last_manip = {
            "request": {"cubes": cubes, "bowls": bowls, "starts": starts},
            "planner_ok": bool(rep.get("ok")),
            "diagnostics": rep.get("diagnostics", []),
            "planned": [],
            "placed": [],
            "n_steps": 0,
        }
        for i, diag in enumerate(last_manip["diagnostics"]):
            print(f"[sm]   diag env{i}: ok={diag.get('ok')} fail={diag.get('fail')} "
                  f"phases={diag.get('phases')} candidates={diag.get('num_candidates')}",
                  flush=True)
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

    # ══ mode: random ═════════════════════════════════════════════════════════════
    def run_random():
        if args.auto_trials > 0:
            _run_random_auto()
        else:
            _run_random_interactive()

    def _run_random_auto():
        """키 입력 없이 랜덤 DR trial 을 --auto_trials 회 실행 + (선택) viewport MP4 기록."""
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
                status, val = "unknown", None
                try:
                    _reset("N", seed=trial_seed, recorder=recorder)
                    status, val = _manipulate(recorder=recorder, plan_seed=trial_seed)
                finally:
                    recorder.close()
                n_attempt, n_placed = val if status == "done" else (env.num_envs, 0)
                trials.append({
                    "trial": trial_i, "seed": trial_seed, "status": status, "video": video_path,
                    "n_attempt": int(n_attempt), "n_placed": int(n_placed), **last_manip,
                })
                print(f"[sm] auto trial {trial_i}: status={status} placed={n_placed}/{n_attempt} "
                      f"video={video_path}", flush=True)
                if status == "closed":
                    break
        finally:
            summary = {
                "task": args.task, "num_envs": args.num_envs, "base_seed": args.seed,
                "record_fps": args.record_fps, "record_every": args.record_every,
                "viewport_render_product": getattr(recorder, "render_product_path", None),
                "trials": trials,
            }
            summary_path = Path(summary_dir) / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[sm] auto summary → {summary_path}", flush=True)

    def _run_random_interactive():
        """livestream 키: N=새 DR · R=같은 레이아웃 · B=plan+manipulate."""
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

    # ══ mode: fail — sweep 결과의 fail 좌표만 재현(인터랙티브) ══════════════════════
    def run_fail():
        fdata = json.loads(Path(args.results).read_text(encoding="utf-8"))
        fails = [c for c in fdata["cells"] if c["n_placed"] < c["n"]]  # place+plan fail 전부
        fails.sort(key=lambda c: (c["y"], c["x"]))
        Nf = env.num_envs
        fbatches = [fails[i:i + Nf] for i in range(0, len(fails), Nf)] or [[]]
        fb = {"i": 0}

        def _load_fail(advance):
            if advance:
                fb["i"] = (fb["i"] + 1) % len(fbatches)
            real = list(fbatches[fb["i"]])
            if not real:
                print("[fail] fail 셀 없음(전부 성공)", flush=True)
                return
            padded = real + [real[0]] * (Nf - len(real))  # 남는 env 는 첫 fail 좌표 복제
            _reset_to_targets([(c["x"], c["y"]) for c in padded], [0.0] * Nf, args.seed + fb["i"])
            print(f"[fail] batch {fb['i'] + 1}/{len(fbatches)} — {len(real)} fail 좌표:", flush=True)
            for c in real:
                typ = "plan-fail " if c["n_planned"] == 0 else "place-fail"
                print(f"[fail]   ({c['x']:+.3f}, {c['y']:+.3f})  {c['kind']:12s} {typ} "
                      f"fails={c.get('fails')}", flush=True)

        _load_fail(advance=False)
        print(f"[sm] FAIL-REPLAY ready — N=다음 batch · R=같은 batch · B=plan+run "
              f"({len(fails)} fail cells, {len(fbatches)} batches, {Nf} envs)", flush=True)
        while simulation_app.is_running():
            k = _wait_key(key)
            if k is None:
                break
            if k == "N":
                _load_fail(advance=True)
                continue
            if k == "R":
                _load_fail(advance=False)
                continue
            status, val = _manipulate()          # B
            if status == "closed":
                break
            if status == "abort":
                _load_fail(advance=(val == "N"))
                continue
            n_attempt, n_placed = val
            print(f"[sm] manipulate {n_placed}/{n_attempt}", flush=True)

    # ══ mode: sweep — 스폰영역 grid+boundary 정량 평가 → JSON ═══════════════════════
    def run_sweep():
        targets_all = SA.sweep_targets(args.nx, args.ny, args.boundary_n)
        N = env.num_envs
        yaw_mode = str(args.yaw).strip().lower()
        rng = np.random.default_rng(args.seed)
        cells = {}
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = [targets_all[i:i + N] for i in range(0, len(targets_all), N)]
        n_bnd = sum(1 for _, _, k in targets_all if k != "interior")
        print(f"[sweep] targets={len(targets_all)} (boundary={n_bnd}) chunks={len(chunks)} "
              f"num_envs={N} trials={args.trials} yaw={args.yaw} out={out_path}", flush=True)

        def _record_cell(x, y, kind, planned, placed, fail):
            ckey = f"{round(x, 4)}_{round(y, 4)}_{kind}"
            c = cells.setdefault(ckey, {"x": float(x), "y": float(y), "kind": kind,
                                        "n": 0, "n_planned": 0, "n_placed": 0, "fails": []})
            c["n"] += 1
            c["n_planned"] += int(planned)
            c["n_placed"] += int(placed)
            if fail:
                c["fails"].append(fail)

        counter = 0
        try:
            for trial in range(args.trials):
                for ci, chunk in enumerate(chunks):
                    if not simulation_app.is_running():
                        break
                    real = list(chunk)
                    batch = real + [real[0]] * (N - len(real))       # 부분 chunk 패딩
                    if yaw_mode == "random":
                        yaws = [float(rng.uniform(0.0, 2.0 * math.pi)) for _ in batch]
                    else:
                        yaws = [math.radians(float(args.yaw))] * N
                    seed = int(args.seed + counter)
                    counter += 1
                    try:
                        _reset_to_targets(batch, yaws, seed)
                        status, _val = _manipulate(plan_seed=seed)
                    except Exception:  # 한 chunk 예외로 전체 sweep 죽지 않게(+traceback 노출)
                        print(f"[sweep] chunk {ci + 1} EXCEPTION — cells 를 error 처리:", flush=True)
                        traceback.print_exc()
                        for x, y, kind in real:
                            _record_cell(x, y, kind, planned=False, placed=False, fail="exception")
                        continue
                    if status == "closed":
                        break
                    planned = last_manip.get("planned", [False] * N)
                    placed = last_manip.get("placed", [False] * N)
                    diags = last_manip.get("diagnostics", [])
                    for i, (x, y, kind) in enumerate(real):
                        pl = bool(planned[i]) if i < len(planned) else False
                        ok = pl and (bool(placed[i]) if i < len(placed) else False)
                        fail = (diags[i].get("fail")
                                if not ok and i < len(diags) and isinstance(diags[i], dict) else None)
                        _record_cell(x, y, kind, pl, ok, fail)
                    done = sum(v["n"] for v in cells.values())
                    placed_tot = sum(v["n_placed"] for v in cells.values())
                    print(f"[sweep] trial {trial} chunk {ci + 1}/{len(chunks)} "
                          f"→ cumulative placed {placed_tot}/{done}", flush=True)
                    with out_path.open("w", encoding="utf-8") as fp:  # 증분 저장(중단 안전)
                        json.dump(_sweep_summary(cells, len(targets_all)), fp, indent=2)
        finally:
            with out_path.open("w", encoding="utf-8") as fp:
                json.dump(_sweep_summary(cells, len(targets_all)), fp, indent=2)
            print(f"[sweep] done → {out_path}", flush=True)

    try:
        {"random": run_random, "fail": run_fail, "sweep": run_sweep}[args.mode]()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
