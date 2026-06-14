"""1-큐브 cuRobo pick-place 데모 — env 클라이언트 (PICKCUBE_CUROBO P2 첫 통합).

⚠ cuRobo 는 isaac 와 한 프로세스 공존 불가(warp 1.14 ↔ omni.warp.core 1.8.2 배타, ABI 게이트 확증).
→ cuRobo 플래닝은 **별도 사이드카 프로세스**(`scripts/planning/curobo_planner_server.py`, ZMQ)에 두고
이 스크립트(Isaac env)는 ZMQ REQ 로 (cube_base+yaw)→config, (start_q,goal_q)→trajectory 만 받는다.

흐름(1 큐브, 고정 spawn): reset → ready 자세 정착 → cube/bowl world pose → base 변환 →
planner ik(D9: 해석적 seed/orientation → cuRobo IK refine = 정확 config) → planner plan_cspace
(ready→pre→grasp→lift→bowl) → 각 궤적 실행(q_bias 중력보상 C6 + gripper open/close) → 성공 판정.

해석적 IK·q_bias·world_to_base·gripper 매핑은 `pick_cube_state_machine.py` 검증분과 동일(복제).

선행: 다른 터미널에서 planner 서버 기동
    uv run --no-sync --group isaac python scripts/planning/curobo_planner_server.py --port 5599
실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
        scripts/sim/pick_cube_curobo_demo.py --video --planner_port 5599
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── argparse + AppLauncher (isaac import 전, _LAUNCHER_KEYS 화이트리스트) ──────
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--planner_host", default="127.0.0.1")
parser.add_argument("--planner_port", type=int, default=5599)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_name", default="so101_curobo_pickplace")
parser.add_argument("--video_length", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gripper_speed", type=float, default=5.0)
parser.add_argument("--grip_open", type=float, default=0.85, help="그리퍼 열림 target(rad)")
parser.add_argument("--grip_close", type=float, default=-0.05)
parser.add_argument("--loop", type=int, default=0, help="시퀀스 반복 횟수(0=auto: livestream 무한·그외 1회)")
parser.add_argument("--ready_steps", type=int, default=70)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--close_steps", type=int, default=10)
parser.add_argument("--pre_z", type=float, default=0.10, help="큐브 위 pre-grasp 높이(m)")
parser.add_argument("--lift_z", type=float, default=0.16)
parser.add_argument("--bowl_z", type=float, default=0.12, help="그릇 위 release 높이(m)")
parser.add_argument("--grasp_dz", type=float, default=0.0, help="grasp TCP z = 큐브중심 + 이 값")
parser.add_argument("--side_offset", type=float, default=0.035, help="side-approach 횡 비킴 하한(m)")
parser.add_argument("--active_objects", type=int, default=4, help="집을 큐브 수(1~4)")
parser.add_argument("--cube_clear", type=float, default=0.022, help="roll 선택 이웃 큐브 여유(m)")
parser.add_argument("--bowl_clear", type=float, default=0.055, help="roll 선택 그릇 여유(m)")
parser.add_argument("--no_dr", action="store_true", help="DR 끄고 고정 spawn (기본=DR ON, 매 round 재배치)")
parser.add_argument("--view_eye", type=float, nargs=3, default=[0.9, -0.9, 1.15])
parser.add_argument("--view_lookat", type=float, nargs=3, default=[0.20, 0.10, 0.70])
parser.add_argument("--public_ip", default="", help="원격 WebRTC livestream 공개 IP(tailscale). 지정 시 livestream mode 1 강제")
parser.add_argument("--cameras", action="store_true", help="top/wrist/front 카메라 리그 주입 + 3-패널 docking viewport")
parser.add_argument("--layout", default="assets/layouts/pick_cube_3cam.json",
                    help="viewport docking layout JSON(ui.Workspace dump). ROOT 상대경로. 없으면 수동 dock fallback")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 원격 WebRTC livestream — PUBLIC_IP env + mode 1 (mode 2 는 PUBLIC_IP 무시 → LAN IP 검은화면).
# 근거: pick_cube_state_machine.py 동일 패턴 + memory isaac-livestream-tailscale-publicip.
if args.public_ip:
    os.environ["PUBLIC_IP"] = args.public_ip
    args.livestream = 1
    print(f"[demo] PUBLIC_IP={args.public_ip} → livestream mode 1", flush=True)

os.makedirs("outputs", exist_ok=True)
faulthandler.enable(open(os.path.join(ROOT, "outputs/curobo_demo_faulthandler.txt"), "w"))

_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
if args.video or args.cameras:
    _launcher_args["enable_cameras"] = True
app_launcher = AppLauncher(_launcher_args)
simulation_app = app_launcher.app

# ── isaac 부팅 후 import ──────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

import sim_to_real  # noqa: E402, F401  (Gym 환경 등록)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
    PickCubeEnvCfg,
    add_pick_cube_cameras,
)

# cube_desk 카메라 prim 경로 (단일 env env_0) — teleop CAMERA_PRIM_PATHS 와 동일.
CAM_PATHS = [
    ("Top Camera", "/World/envs/env_0/TopCamera"),
    ("Wrist Camera", "/World/envs/env_0/Robot/gripper/WristCamera"),
    ("Front Camera", "/World/envs/env_0/Robot/shoulder/FrontCamera"),
]
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kinematics import SO101Kinematics  # noqa: E402  (PAN_X·_fold_45 — closing-axis)

# ── SM 검증분 복제 상수/헬퍼 (frame 정합) ──────────────────────────────────
DESK_TOP_Z = 0.705
GRIPPER_ACTION_OFFSET = 0.20
GRASP_OFF = np.array([-0.0079, -0.000218121, -0.0981274])
BASE_XY_OFFSET = (0.0204, 0.0157)
BASE_Z_OFFSET = 0.0325
BASE_YAW_OFFSET = math.pi / 2
BIAS_KI = 0.06
BIAS_MAX = 0.35


def _quat_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _world_to_base_np(p_w, root_p, root_yaw):
    d = p_w - root_p
    c, s = math.cos(-root_yaw), math.sin(-root_yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    return [-(by - BASE_XY_OFFSET[1]), bx - BASE_XY_OFFSET[0], float(d[2]) - BASE_Z_OFFSET]


def log(m):
    print(m, flush=True)


class Planner:
    """ZMQ REQ 클라이언트 → cuRobo 사이드카."""

    def __init__(self, host, port):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, 60000)
        self.s.connect(f"tcp://{host}:{port}")

    def call(self, d):
        import json
        self.s.send(json.dumps(d).encode())
        import json as _j
        return _j.loads(self.s.recv())

    def ik(self, tcp_base, yaw, grip=0.785, pitch_max_deg=-20.0, roll=0.0):
        r = self.call({"cmd": "ik", "tcp_base": list(tcp_base), "yaw": yaw,
                       "grip": grip, "pitch_max_deg": pitch_max_deg, "roll": roll})
        return r

    def set_world(self, cuboids):
        return self.call({"cmd": "set_world", "cuboids": cuboids})

    def plan(self, start_q, goal_q):
        return self.call({"cmd": "plan", "start_q": list(start_q), "goal_q": list(goal_q)})


def dock_camera_viewports():
    """top/wrist/front 카메라 viewport 3개 생성 + Perspective 와 수직 분할 docking (teleop 패턴).
    livestream WebRTC 가 kit 창 전체(4-패널)를 캡처 → 사용자가 카메라 피드 봄."""
    try:
        import omni.kit.app
        import omni.ui as ui
        from pxr import Sdf
        em = omni.kit.app.get_app().get_extension_manager()
        for e in ("omni.kit.viewport.window", "omni.kit.viewport.utility"):
            try:
                if not em.is_extension_enabled(e):
                    em.set_extension_enabled_immediate(e, True)
            except Exception:
                pass
        from omni.kit.viewport.utility import create_viewport_window
    except Exception as exc:
        log(f"[demo] viewport 모듈 불가: {exc}")
        return
    created = {}
    for title, path in CAM_PATHS:
        try:
            created[title] = create_viewport_window(name=f"SO101 {title}", camera_path=Sdf.Path(path))
            log(f"[demo] viewport {title}: {path}")
        except Exception as exc:
            log(f"[demo] viewport {title} 실패: {exc}")
    app = omni.kit.app.get_app()
    for _ in range(3):
        app.update()
    # 저장된 layout JSON(ui.Workspace dump) 복원. window title 이 "SO101 Top/Wrist/Front Camera"
    # 로 일치하므로 위치·크기가 그대로 복원된다. 실패 시 수동 dock_in fallback.
    layout_path = os.path.join(ROOT, args.layout) if not os.path.isabs(args.layout) else args.layout
    if os.path.isfile(layout_path):
        try:
            import json
            with open(layout_path) as fh:
                dump = json.load(fh)
            ui.Workspace.restore_workspace(dump)
            for _ in range(3):
                app.update()
            log(f"[demo] layout 복원: {layout_path}")
            return
        except Exception as exc:
            log(f"[demo] layout 복원 실패({exc}) → 수동 dock fallback")
    else:
        log(f"[demo] layout 파일 없음({layout_path}) → 수동 dock fallback")
    try:
        main_vp = ui.Workspace.get_window("Viewport")
        t, w, f = created.get("Top Camera"), created.get("Wrist Camera"), created.get("Front Camera")
        if main_vp is not None and t is not None:
            t.dock_in(main_vp, ui.DockPosition.RIGHT, 0.5)
        if t is not None and w is not None:
            w.dock_in(t, ui.DockPosition.BOTTOM, 0.5)
        if w is not None and f is not None:
            f.dock_in(w, ui.DockPosition.BOTTOM, 0.5)
        for _ in range(3):
            app.update()
        log("[demo] 카메라 viewport docked (L=Perspective, R=top/wrist/front)")
    except Exception as exc:
        log(f"[demo] docking 실패: {exc}")


def main() -> int:
    # ── env 구성 (1 env, 고정 spawn, 컨트롤러가 종료 관리) ──
    env_cfg = PickCubeEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.success = None
    env_cfg.terminations.cube_lost = None
    if args.no_dr:
        env_cfg.events.randomize_cubes = None   # 고정 spawn
        env_cfg.events.randomize_bowl = None
    # DR ON(기본): 매 env.reset() 마다 큐브·그릇 재샘플 → 매 round 다른 레이아웃
    env_cfg.viewer.eye = tuple(args.view_eye)
    env_cfg.viewer.lookat = tuple(args.view_lookat)
    mv = dict(env_cfg.actions.arm.max_velocity)
    mv["gripper"] = args.gripper_speed
    env_cfg.actions.arm.max_velocity = mv

    if args.cameras:
        add_pick_cube_cameras(env_cfg.scene)   # top/wrist/front 카메라 리그 주입

    if args.video:
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        os.makedirs("docs", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder="docs", name_prefix=args.video_name,
            step_trigger=lambda step: step == 0, video_length=args.video_length,
            disable_logger=True)
        base = env.unwrapped
    else:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        base = env

    env.reset()
    log("[demo] env reset 완료")
    if args.cameras and not args.headless:
        dock_camera_viewports()

    scene = base.scene
    device = base.device
    robot = scene["robot"]
    g_idx = list(robot.data.body_names).index("gripper")
    cubes = list(CUBE_NAMES[: args.active_objects])
    CUBE_SIZES = {"Cube1": 0.030, "Cube2": 0.030, "Cube3": 0.040, "Cube4": 0.040}
    q_bias = torch.zeros((1, 5), device=device)

    pl = Planner(args.planner_host, args.planner_port)
    ping = pl.call({"cmd": "ping"})
    log(f"[demo] planner ping: {ping.get('ok')}  joints={ping.get('joints')}")

    # ── 상태 헬퍼 ──
    def snap():
        d = robot.data
        s = {
            "jp": d.joint_pos[0, :6].detach().cpu().numpy(),
            "rp": d.root_pos_w[0, :3].detach().cpu().numpy(),
            "ryaw": _quat_to_yaw(d.root_quat_w[0].detach().cpu().numpy()),
            "bowl": scene[BOWL_NAME].data.root_pos_w[0, :3].detach().cpu().numpy(),
            "obj": {}, "objyaw": {},
        }
        for c in cubes:
            s["obj"][c] = scene[c].data.root_pos_w[0, :3].detach().cpu().numpy()
            s["objyaw"][c] = _quat_to_yaw(scene[c].data.root_quat_w[0].detach().cpu().numpy())
        return s

    CONTROL_DT = 1.0 / 30.0  # decimation 4 × sim.dt(1/120)
    nstep = [0]

    def act(q_arm, grip_target):
        """[q_arm(5)+q_bias, grip] 1 step. q_bias 중력보상(C6)."""
        nonlocal q_bias
        nstep[0] += 1
        q_cmd = torch.tensor([q_arm], device=device, dtype=torch.float32)
        q_now = robot.data.joint_pos[:, :5]
        q_bias = torch.clamp(q_bias + BIAS_KI * (q_cmd - q_now), -BIAS_MAX, BIAS_MAX)
        grip = torch.tensor([[grip_target - GRIPPER_ACTION_OFFSET]], device=device)
        env.step(torch.cat([q_cmd + q_bias, grip], dim=-1))

    def to_base(p_w, sn):
        return _world_to_base_np(np.asarray(p_w, float), sn["rp"], sn["ryaw"])

    def yaw_base(sn, yaw_w=0.0):
        return yaw_w - sn["ryaw"] + BASE_YAW_OFFSET

    def cur_arm():
        return robot.data.joint_pos[0, :5].detach().cpu().tolist()

    def run_traj(traj, grip_target, hold=0, stride=1):
        """plan 궤적 실행. stride>1 = 매 stride 번째 waypoint 만(PD 가 보간) → crisp·빠름."""
        n = len(traj)
        for i in range(0, n, stride):
            if not simulation_app.is_running():
                return False
            act(traj[i][:5], grip_target)
        act(traj[-1][:5], grip_target)   # 항상 마지막 도달
        for _ in range(hold):
            act(traj[-1][:5], grip_target)
        return True

    def settle(q_arm, grip_target, n):
        for _ in range(n):
            if not simulation_app.is_running():
                return
            act(q_arm, grip_target)

    # ready/rest 자세 — SM HOME_Q 풍(shoulder 올림·elbow 접힘·wrist cam 위). 단 극단 fold(-1.5,1.4)는
    # cuRobo self-collision 경계 바로 안쪽 → q_bias sag 로 측정 config 가 경계 넘어 plan 시작 실패.
    # 경계서 충분히 떨어진 안전 fold(-1.3,1.2)로 backoff(sag 흡수). 수평 중립 아님.
    READY = [0.0, -1.3, 1.2, math.radians(-20.0), math.radians(-90.0)]

    def plan_and_run(start_q, goal_q6, grip, hold=0, tag="", stride=3):
        p = pl.plan(start_q + [0.785], goal_q6) if len(start_q) == 5 else pl.plan(start_q, goal_q6)
        if not p["ok"]:
            log(f"[demo] {tag} plan 실패 — 중단")
            return False
        log(f"[demo] {tag} plan OK n={p['n']} → 실행(stride{stride})")
        return run_traj(p["traj"], grip, hold=hold, stride=stride)

    def seq_exec(goal_q6, grip, steps, tag, hold=0):
        """grasp 미세동작 — plan_cspace 없이 joint 직접 보간 실행(짧고 target 근처라 충돌계획 불요·
        plan_cspace 의 이웃 과다기각 회피). clearance roll-선택이 이미 face 확보."""
        start = cur_arm()
        g = goal_q6[:5]
        for i in range(1, steps + 1):
            if not simulation_app.is_running():
                return False
            a = i / steps
            act([start[j] * (1 - a) + g[j] * a for j in range(5)], grip)
        for _ in range(hold):
            act(g, grip)
        log(f"[demo] {tag} (direct {steps}step)")
        return True

    def ik_q(p_w, yaw_b, pmax, sn, roll=0.0, name=""):
        r = pl.ik(to_base(p_w, sn), yaw_b, grip=args.grip_open, pitch_max_deg=pmax, roll=roll)
        if name:
            log(f"[demo] ik {name}: ok={r['ok']} pos_err={r.get('pos_err_mm'):.2f}mm pitch={r.get('pitch_deg'):.0f}")
        return r["q"] if r["ok"] else None

    def closing_axis(cube_w, sn, cube_yaw, roll=0.0):
        """fixed-finger(닫힘축) world 방위 (SM `_closing_axis` 복제). side-approach 방향.
        cube_yaw = 큐브 실제 world yaw (DR ±30°) → 잠금이 큐브 face 에 정렬."""
        tb = to_base(cube_w, sn)
        q1 = -math.atan2(tb[1], tb[0] - SO101Kinematics.PAN_X)
        yb = yaw_base(sn, cube_yaw)
        q5b = SO101Kinematics._fold_45(yb + q1)
        return (q5b - q1) - BASE_YAW_OFFSET + sn["ryaw"] + roll

    placed = set()  # 그릇에 담은 큐브

    def set_obstacles(target, sn):
        """target·placed 제외 큐브 + 그릇을 cuboid 장애물로 planner 주입(base frame).
        → plan_cspace·IK 가 이웃·그릇 회피 (cuRobo per-env world)."""
        cubs = []
        for c in cubes:
            if c == target or c in placed:
                continue
            cb = to_base(sn["obj"][c], sn)
            s = CUBE_SIZES[c]
            cubs.append({"name": c, "pose": [cb[0], cb[1], cb[2], 1, 0, 0, 0],
                         "dims": [s + 0.002, s + 0.002, s + 0.002]})
        bb = to_base(sn["bowl"], sn)                      # 그릇 = 낮은 박스(위는 비워 placement 가능)
        cubs.append({"name": "bowl", "pose": [bb[0], bb[1], bb[2], 1, 0, 0, 0],
                     "dims": [0.12, 0.12, 0.03]})          # 낮게(0.03) → 위 접근 path 안 막음
        r = pl.set_world(cubs)
        log(f"[demo]   set_world obstacles={r.get('n_obstacles')} (target={target} 제외)")

    def select_grasp(target, sn):
        """clearance·wrist-flip 기준 roll(±90/0/π) 선정. (roll, q_desc, q_slide) 또는 None.
        SM `_evaluate_all_grasps` 단순화 — 이웃/그릇 안 치는 face + wrist 회전 최소."""
        cube_w = sn["obj"][target]
        cyaw = sn["objyaw"][target]                       # 큐브 실제 yaw (DR ±30°)
        px, py, pz = float(cube_w[0]), float(cube_w[1]), float(cube_w[2])
        size = CUBE_SIZES[target]
        so = max(args.side_offset, size * 0.5 + 0.018)
        gz = pz + args.grasp_dz
        yb = yaw_base(sn, cyaw)                            # grasp_yaw 를 큐브 face 에 정렬
        bx, by = float(sn["bowl"][0]), float(sn["bowl"][1])
        others = [sn["obj"][c] for c in cubes if c != target and c not in placed]
        cur_roll = cur_arm()[4]
        fh = 0.045
        best = None
        for roll in (math.pi / 2, -math.pi / 2, 0.0, math.pi):
            ax = closing_axis(cube_w, sn, cyaw, roll)
            dx, dy = math.cos(ax), math.sin(ax)
            fx, fy = -dy, dx
            bx0, by0 = px - so * dx, py - so * dy
            pts = [(px + fh * fx, py + fh * fy), (px - fh * fx, py - fh * fy),
                   (bx0, by0), (0.5 * (bx0 + px), 0.5 * (by0 + py))]
            clear = 1e9
            for cxx, cyy in pts:
                clear = min(clear, math.hypot(cxx - bx, cyy - by) - args.bowl_clear)
                for o in others:
                    clear = min(clear, math.hypot(cxx - float(o[0]), cyy - float(o[1])) - args.cube_clear)
            q_d = ik_q([bx0, by0, gz], yb, -70, sn, roll=roll)
            q_s = ik_q([px, py, gz], yb, -70, sn, roll=roll)
            if q_d is None or q_s is None:
                continue
            clear_norm = max(0.0, min(1.0, clear / 0.04))
            bias = 0.1 if abs(abs(roll) - math.pi / 2) < 1e-3 else 0.0
            flip = abs(q_s[4] - cur_roll)
            score = clear_norm + bias - 0.30 * flip
            if best is None or score > best[0]:
                best = (score, roll, q_d, q_s, clear)
        if best is None:
            return None
        log(f"[demo]   roll 선정={math.degrees(best[1]):.0f}° clear={best[4]*1000:.0f}mm score={best[0]:.2f}")
        return best[1], best[2], best[3]

    def pick_cube(target, start_q=None):
        """1 큐브 side-approach pick-place (장애물 회피 + roll 선택). placed 추가 여부 반환.
        start_q=plan 시작 config(없으면 cur_arm). cube1 은 exact READY(측정 sag 회피)."""
        log(f"[demo] --- {target} pick 시작 ---")
        sn = snap()   # READY 복귀 없음 — 직전 release(그릇 위)에서 바로 다음 큐브로
        set_obstacles(target, sn)
        sel = select_grasp(target, sn)
        if sel is None:
            log(f"[demo] 🔴 {target} grasp roll 전부 도달불가 — skip")
            return False
        roll, q_desc, q_slid = sel
        cube_w = sn["obj"][target]
        cyaw = sn["objyaw"][target]
        px, py, pz = float(cube_w[0]), float(cube_w[1]), float(cube_w[2])
        size = CUBE_SIZES[target]
        so = max(args.side_offset, size * 0.5 + 0.018)
        yb = yaw_base(sn, cyaw)
        ax = closing_axis(cube_w, sn, cyaw, roll)
        dx, dy = math.cos(ax), math.sin(ax)
        pre_w  = [px - so * dx, py - so * dy, pz + args.pre_z]
        lift_w = [px, py, pz + args.lift_z]
        bowl_a = [float(sn["bowl"][0]), float(sn["bowl"][1]), DESK_TOP_Z + args.bowl_z]
        q_pre  = ik_q(pre_w, yb, -45, sn, roll=roll, name="pre(off)")
        q_lift = ik_q(lift_w, yb, -30, sn, roll=roll, name="lift")
        q_bowl = ik_q(bowl_a, yb, -10, sn, roll=roll, name="bowl")  # 먼 그릇 → 강한 tilt 허용
        if None in (q_pre, q_lift, q_bowl):
            log(f"[demo] 🔴 {target} pre/lift/bowl IK 도달불가 — skip")
            return False

        ok = True
        ok = ok and plan_and_run(start_q if start_q is not None else cur_arm(),
                                 q_pre, args.grip_open, hold=2, tag="→pre")  # cube1=exact READY, 이후=현재(post-release)
        ok = ok and seq_exec(q_desc, args.grip_open, 16, "pre→descend", hold=1)               # 미세: 직접(빠른 하강)
        ok = ok and seq_exec(q_slid, args.grip_open, 14, "slide→center", hold=2)               # 미세: 직접(빠른 slide=push↓)
        if ok:
            log("[demo]   grasp close")
            settle(q_slid[:5], args.grip_close, args.close_steps)
        ok = ok and seq_exec(q_lift, args.grip_close, 16, "grasp→lift", hold=2)                 # 미세: 직접(수직↑)
        if ok:  # lift→bowl 긴 transit: 충돌계획, 실패 시 직접 fallback(팔 높이라 sweep 가 desk 큐브 위)
            if not plan_and_run(cur_arm(), q_bowl, args.grip_close, hold=3, tag="lift→bowl"):
                log("[demo]   lift→bowl plan 실패 → 직접 fallback")
                ok = seq_exec(q_bowl, args.grip_close, 30, "lift→bowl(direct)", hold=3)
        if ok:
            settle(q_bowl[:5], args.grip_open, 12)   # release
        settle(q_bowl[:5] if ok else READY, args.grip_open, 6)

        cp = snap()["obj"][target]
        bp = sn["bowl"]
        in_xy = float(np.linalg.norm(cp[:2] - bp[:2])) < BOWL_SUCCESS_RADIUS
        z_rel = float(cp[2]) - DESK_TOP_Z
        ok_p = in_xy and (BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10)
        log(f"[demo]   {target} placed={ok_p} (xy={float(np.linalg.norm(cp[:2]-bp[:2]))*1000:.0f}mm)")
        return ok_p

    def run_round():
        """4 큐브 전부 시도. placed 수 반환."""
        nonlocal placed
        placed = set()
        settle(READY, args.grip_open, 35)   # 라운드 시작: READY 수렴(cube1 plan start)
        s0 = nstep[0]
        for idx, target in enumerate(cubes):
            if not simulation_app.is_running():
                break
            if pick_cube(target, start_q=(READY if idx == 0 else None)):
                placed.add(target)
        settle(READY, args.grip_open, 30)   # 라운드 끝: 대기 자세 복귀(흘러내림 방지)
        sim_t = (nstep[0] - s0) * CONTROL_DT
        log("=" * 60)
        log(f"[demo] ===== ROUND 결과: {len(placed)}/{len(cubes)} placed | "
            f"{nstep[0]-s0} steps = {sim_t:.1f}s (sim, {len(cubes)}큐브) =====")
        log("=" * 60)
        return len(placed)

    watch = bool(getattr(args, "livestream", 0)) and not args.headless and not args.video
    if watch:
        log("[demo] 🔴 LIVESTREAM watch 모드 — 접속 대기 10초 후 4-큐브 pick-place 반복")
        for _ in range(300):
            if not simulation_app.is_running():
                break
            act(READY, args.grip_open)

    loop_n = args.loop if args.loop > 0 else (10**9 if watch else 1)
    i = 0
    while i < loop_n and simulation_app.is_running():
        if i > 0:
            env.reset()
            q_bias = torch.zeros((1, 5), device=device)
        log(f"[demo] ===== ROUND {i+1} 시작 =====")
        run_round()
        for _ in range(40):   # 라운드 간 대기 — READY 유지(cur_arm 홀딩=droop 방지)
            if not simulation_app.is_running():
                break
            act(READY, args.grip_open)
        i += 1

    if watch:
        log("[demo] 종료 — 창 닫을 때까지 idle (READY 유지)")
        while simulation_app.is_running():
            act(READY, args.grip_open)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
