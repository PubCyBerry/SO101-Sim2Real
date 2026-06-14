"""multi-env cuRobo pick-place 배치 — env 클라이언트 (PICKCUBE_CUROBO P3).

단일-env `pick_cube_curobo_demo.py` 의 **lock-step 벡터화판**(D12). 전 env 가 동일 phase tape
(고정 순서 Cube1→4)를 따르며, IK/plan/step 을 배치로 처리한다. cuRobo `BatchMotionPlanner`
(multi_env=True)가 "N문제 N world 1 GPU call" 을 담당(사이드카 `curobo_planner_server.py` 배치
엔드포인트). 단일-env 데모(livestream/카메라/layout)는 그대로 보존하고 본 파일은 headless 정량용.

스칼라 기하 헬퍼(closing_axis·to_base·yaw_base·grasp 선정)는 데모서 **복제 후 per-env 루프**로
요청을 빌드하고, GPU 비용(ik_batch·plan_batch)과 env.step 만 배치한다. q_bias 중력보상(C6)·side-
approach·DR cube_yaw·per-env world 회피·READY backoff 는 단일-env 검증분과 동일.

선행: 다른 터미널에서 배치 planner 서버 기동
    uv run --no-sync --group isaac python scripts/planning/curobo_planner_server.py --port 5601
실행(headless 정량, N=256):
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync --group isaac python \
        scripts/sim/pick_cube_curobo_batch.py --headless --num_envs 256 --planner_port 5601 --loop 1
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── argparse + AppLauncher ────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--planner_host", default="127.0.0.1")
parser.add_argument("--planner_port", type=int, default=5601)
parser.add_argument("--num_envs", type=int, default=4, help="Isaac 물리 배치 env 수")
parser.add_argument("--chunk", type=int, default=64, help="planner 배치 청크(VRAM 상한). 0/≥num_envs=단일배치")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gripper_speed", type=float, default=5.0)
parser.add_argument("--grip_open", type=float, default=0.85)
parser.add_argument("--grip_close", type=float, default=-0.05)
parser.add_argument("--loop", type=int, default=1, help="라운드 반복 횟수")
parser.add_argument("--ik_seeds", type=int, default=48, help="배치 IK seed 수(낮추면 빠름·성공↓)")
parser.add_argument("--plan_attempts", type=int, default=4)
parser.add_argument("--close_steps", type=int, default=10)
parser.add_argument("--max_retry", type=int, default=1, help="큐브당 grasp 실패 시 재시도 횟수(0=재시도 없음)")
parser.add_argument("--order_mode", default="fixed", choices=["fixed", "isolated"],
                    help="픽 순서: fixed=인덱스순(92.8% 동치), isolated=가장 고립 먼저")
parser.add_argument("--face_yaw", action="store_true", help="grasp yaw=수직면 azimuth(tumbled 정합)")
parser.add_argument("--pre_z", type=float, default=0.10)
parser.add_argument("--lift_z", type=float, default=0.16)
parser.add_argument("--bowl_z", type=float, default=0.12, help="그릇 위 release 높이(m). 낮으면 rim 충돌·placement 실패")
parser.add_argument("--grasp_dz", type=float, default=0.0)
parser.add_argument("--side_offset", type=float, default=0.035)
parser.add_argument("--active_objects", type=int, default=4)
parser.add_argument("--cube_clear", type=float, default=0.022)
parser.add_argument("--bowl_clear", type=float, default=0.055)
parser.add_argument("--no_dr", action="store_true", help="DR 끄고 고정 spawn")
parser.add_argument("--stride", type=int, default=3, help="plan traj subsample stride")
parser.add_argument("--taxonomy", default="", help="결과 JSON 저장 경로(빈값=미저장)")
parser.add_argument("--dump_fail", default="", help="최악 실패 layout 초기 pose 덤프 경로(headless 진단용)")
parser.add_argument("--dump_n", type=int, default=4, help="덤프할 최악 실패 env 수")
parser.add_argument("--load_fail", default="", help="덤프된 실패 layout 재현(N=layout 수, DR off). livestream 관전용")
parser.add_argument("--view_eye", type=float, nargs=3, default=[2.2, -2.2, 2.4])
parser.add_argument("--view_lookat", type=float, nargs=3, default=[0.4, 0.4, 0.75])
parser.add_argument("--public_ip", default="", help="원격 WebRTC livestream(선택, 보통 headless)")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.public_ip:
    os.environ["PUBLIC_IP"] = args.public_ip
    args.livestream = 1
    print(f"[batch] PUBLIC_IP={args.public_ip} → livestream mode 1", flush=True)

os.makedirs("outputs", exist_ok=True)
faulthandler.enable(open(os.path.join(ROOT, "outputs/curobo_batch_faulthandler.txt"), "w"))

_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
_launcher_args = {k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS}
app_launcher = AppLauncher(_launcher_args)
simulation_app = app_launcher.app

# ── isaac 부팅 후 import ──────────────────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

import gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: E402, F401
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    BOWL_HEIGHT_RANGE,
    BOWL_SUCCESS_RADIUS,
    PickCubeEnvCfg,
)
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kinematics import SO101Kinematics  # noqa: E402

# ── SM 검증분 복제 상수/스칼라 헬퍼 (단일-env 데모와 동일) ─────────────────
DESK_TOP_Z = 0.705
GRIPPER_ACTION_OFFSET = 0.20
BASE_XY_OFFSET = (0.0204, 0.0157)
BASE_Z_OFFSET = 0.0325
BASE_YAW_OFFSET = math.pi / 2
BIAS_KI = 0.06
BIAS_MAX = 0.35
CUBE_SIZES = {"Cube1": 0.030, "Cube2": 0.030, "Cube3": 0.040, "Cube4": 0.040}
READY = [0.0, -1.3, 1.2, math.radians(-20.0), math.radians(-90.0)]


def log(m):
    print(m, flush=True)


def _quat_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _cube_face_yaw(q) -> float:
    """큐브 grasp 면 azimuth — 회전행렬의 **수평 body 축**(수직면 법선) 방향.

    full_orient(6면 착지)에서 naive yaw 는 옆면 착지 큐브의 수직면과 안 맞아 jaw 가
    비스듬히 잡는다. 세 body 축 중 world-z 성분 최대=윗/밑면 → 나머지 둘=수평(수직면
    법선). xy 크기 큰 쪽의 azimuth 반환(큐브 4-fold 대칭이라 closing_axis 의 _fold_45 가 정렬)."""
    w, x, y, z = (float(v) for v in q)
    ex = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    ey = (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    ez = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    axes = (ex, ey, ez)
    vert = max(range(3), key=lambda i: abs(axes[i][2]))           # 수직(윗/밑) 축
    horiz = [axes[i] for i in range(3) if i != vert]
    h = max(horiz, key=lambda a: math.hypot(a[0], a[1]))          # 수평면 법선
    return math.atan2(h[1], h[0])


def _world_to_base_np(p_w, root_p, root_yaw):
    d = np.asarray(p_w, float) - np.asarray(root_p, float)
    c, s = math.cos(-root_yaw), math.sin(-root_yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    return [float(-(by - BASE_XY_OFFSET[1])), float(bx - BASE_XY_OFFSET[0]), float(d[2]) - BASE_Z_OFFSET]


class BatchPlanner:
    """ZMQ REQ → cuRobo 배치 사이드카."""

    def __init__(self, host, port):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, 180000)
        self.s.connect(f"tcp://{host}:{port}")

    def call(self, d):
        self.s.send(json.dumps(d).encode())
        return json.loads(self.s.recv())

    def init_batch(self, n_envs, cubes, bowl_dims, ik_seeds, max_attempts, chunk):
        return self.call({"cmd": "init_batch", "n_envs": n_envs, "cubes": cubes,
                          "bowl_dims": list(bowl_dims), "ik_seeds": ik_seeds,
                          "max_attempts": max_attempts, "chunk": chunk})

    def world_batch(self, worlds):
        return self.call({"cmd": "world_batch", "worlds": worlds})

    def ik_batch(self, probs):
        return self.call({"cmd": "ik_batch", "probs": probs})

    def plan_batch(self, start_q, goal_q):
        return self.call({"cmd": "plan_batch", "start_q": start_q, "goal_q": goal_q})


def main() -> int:
    N = args.num_envs
    fail_layouts = None
    if args.load_fail:
        with open(args.load_fail) as fh:
            fail_layouts = json.load(fh)
        N = len(fail_layouts)
        args.no_dr = True   # 덤프 layout 을 reset 후 직접 override
        log(f"[batch] load_fail: {N} 실패 layout 재현 (DR off)")
    # ── env 구성 (N env, DR ON 기본) ──
    env_cfg = PickCubeEnvCfg()
    env_cfg.scene.num_envs = N
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.success = None
    env_cfg.terminations.cube_lost = None
    if args.no_dr:
        env_cfg.events.randomize_cubes = None
        env_cfg.events.randomize_bowl = None
    env_cfg.viewer.eye = tuple(args.view_eye)
    env_cfg.viewer.lookat = tuple(args.view_lookat)
    mv = dict(env_cfg.actions.arm.max_velocity)
    mv["gripper"] = args.gripper_speed
    env_cfg.actions.arm.max_velocity = mv

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    log(f"[batch] env reset 완료 (N={N})")

    scene = env.scene
    device = env.device
    robot = scene["robot"]
    cubes = list(CUBE_NAMES[: args.active_objects])
    q_bias = torch.zeros((N, 5), device=device)

    pl = BatchPlanner(args.planner_host, args.planner_port)
    ping = pl.call({"cmd": "ping"})
    log(f"[batch] planner ping: {ping.get('ok')} joints={ping.get('joints')}")
    cube_specs = [{"name": c, "dims": [CUBE_SIZES[c]] * 3} for c in cubes]
    r = pl.init_batch(N, cube_specs, [0.12, 0.12, 0.03], args.ik_seeds, args.plan_attempts, args.chunk)
    if not r.get("ok"):
        log(f"[batch] 🔴 init_batch 실패: {r}")
        return 1
    log(f"[batch] init_batch OK n_envs={r.get('n_envs')} chunk={r.get('chunk')}")

    CONTROL_DT = 1.0 / 30.0
    nstep = [0]

    # ── 배치 stepping ──
    def act(q_arm, grip):
        """q_arm (N,5) np/tensor, grip (N,) np → 1 step. q_bias 중력보상(C6)."""
        nonlocal q_bias
        nstep[0] += 1
        q_cmd = torch.as_tensor(np.asarray(q_arm, np.float32), device=device)
        q_now = robot.data.joint_pos[:, :5]
        q_bias = torch.clamp(q_bias + BIAS_KI * (q_cmd - q_now), -BIAS_MAX, BIAS_MAX)
        g = torch.as_tensor(np.asarray(grip, np.float32) - GRIPPER_ACTION_OFFSET,
                            device=device).reshape(N, 1)
        env.step(torch.cat([q_cmd + q_bias, g], dim=-1))

    def cur_arm():
        return robot.data.joint_pos[:, :5].detach().cpu().numpy()   # (N,5)

    def grip_vec(active, val_active):
        """active=True env 는 val_active, 나머지는 grip_open(idle)."""
        g = np.full(N, args.grip_open, np.float32)
        g[active] = val_active
        return g

    def settle(q_arm, grip, n):
        for _ in range(n):
            if not simulation_app.is_running():
                return False
            act(q_arm, grip)
        return True

    def seq_exec(goal, active, grip, steps):
        """미세동작 직접 보간(N,5). inactive env 는 hold(goal=start)."""
        start = cur_arm()
        g = np.array(goal, np.float32).copy()
        g[~active] = start[~active]
        for i in range(1, steps + 1):
            if not simulation_app.is_running():
                return False
            a = i / steps
            act(start * (1 - a) + g * a, grip)
        return True

    def run_traj(trajs, grip, stride):
        """plan 배치 궤적 실행. trajs (N,H,6). inactive env 는 서버서 flat(goal=start)."""
        H = trajs.shape[1]
        for i in range(0, H, stride):
            if not simulation_app.is_running():
                return False
            act(trajs[:, i, :5], grip)
        act(trajs[:, -1, :5], grip)
        return True

    # ── 상태 스냅 (배치 → per-env 스칼라 dict 리스트, 데모 헬퍼 재사용) ──
    def snap_all():
        d = robot.data
        jp = d.joint_pos[:, :6].detach().cpu().numpy()
        rp = d.root_pos_w[:, :3].detach().cpu().numpy()
        rq = d.root_quat_w.detach().cpu().numpy()
        bowl = scene[BOWL_NAME].data.root_pos_w[:, :3].detach().cpu().numpy()
        objp = {c: scene[c].data.root_pos_w[:, :3].detach().cpu().numpy() for c in cubes}
        objq = {c: scene[c].data.root_quat_w.detach().cpu().numpy() for c in cubes}
        out = []
        for e in range(N):
            sn = {"jp": jp[e], "rp": rp[e], "ryaw": _quat_to_yaw(rq[e]),
                  "bowl": bowl[e], "obj": {}, "objyaw": {}}
            for c in cubes:
                sn["obj"][c] = objp[c][e]
                sn["objyaw"][c] = (_cube_face_yaw(objq[c][e]) if args.face_yaw
                                   else _quat_to_yaw(objq[c][e]))
            out.append(sn)
        return out

    # ── 실패 layout 덤프/재현 (livestream 진단용) ──
    def capture_layout0():
        """각 env 의 초기 큐브/그릇 pose 를 env_origin 상대로 캡처(타일링 무관 재현용)."""
        orig = scene.env_origins.detach().cpu().numpy()
        cp = {c: scene[c].data.root_pos_w[:, :3].detach().cpu().numpy() for c in cubes}
        cq = {c: scene[c].data.root_quat_w.detach().cpu().numpy() for c in cubes}
        bp = scene[BOWL_NAME].data.root_pos_w[:, :3].detach().cpu().numpy()
        bq = scene[BOWL_NAME].data.root_quat_w.detach().cpu().numpy()
        out = []
        for e in range(N):
            rel = {c: [float(cp[c][e, 0] - orig[e, 0]), float(cp[c][e, 1] - orig[e, 1]),
                       float(cp[c][e, 2] - orig[e, 2])] + [float(x) for x in cq[c][e]] for c in cubes}
            bowl = [float(bp[e, 0] - orig[e, 0]), float(bp[e, 1] - orig[e, 1]),
                    float(bp[e, 2] - orig[e, 2])] + [float(x) for x in bq[e]]
            out.append({"cubes": rel, "bowl": bowl})
        return out

    def apply_fail_layouts(layouts):
        """덤프 layout 을 reset 후 env 에 주입(world = env_origin + 상대pose). 속도 0."""
        orig = scene.env_origins.detach().cpu().numpy()
        zero_v = torch.zeros((N, 6), device=device)
        for c in cubes:
            poses = []
            for e in range(N):
                rel = layouts[e]["cubes"][c]
                poses.append([rel[0] + float(orig[e, 0]), rel[1] + float(orig[e, 1]),
                              rel[2] + float(orig[e, 2]), rel[3], rel[4], rel[5], rel[6]])
            pt = torch.tensor(poses, device=device, dtype=torch.float32)
            scene[c].write_root_pose_to_sim(pt)
            scene[c].write_root_velocity_to_sim(zero_v)
        bposes = []
        for e in range(N):
            rb = layouts[e]["bowl"]
            bposes.append([rb[0] + float(orig[e, 0]), rb[1] + float(orig[e, 1]),
                           rb[2] + float(orig[e, 2]), rb[3], rb[4], rb[5], rb[6]])
        scene[BOWL_NAME].write_root_pose_to_sim(torch.tensor(bposes, device=device, dtype=torch.float32))
        scene[BOWL_NAME].write_root_velocity_to_sim(zero_v)

    def to_base(p_w, sn):
        return _world_to_base_np(p_w, sn["rp"], sn["ryaw"])

    def yaw_base(sn, yaw_w=0.0):
        return yaw_w - sn["ryaw"] + BASE_YAW_OFFSET

    def closing_axis(cube_w, sn, cube_yaw, roll=0.0):
        tb = to_base(cube_w, sn)
        q1 = -math.atan2(tb[1], tb[0] - SO101Kinematics.PAN_X)
        yb = yaw_base(sn, cube_yaw)
        q5b = SO101Kinematics._fold_45(yb + q1)
        return (q5b - q1) - BASE_YAW_OFFSET + sn["ryaw"] + roll

    # ── per-env world 주입 (env별 target·placed 제외, base frame) ──
    def push_world(sn_list, tgt, placed):
        worlds = []
        for e in range(N):
            sn = sn_list[e]
            obs = []
            for ci, c in enumerate(cubes):
                cb = to_base(sn["obj"][c], sn)
                enabled = (c != tgt[e]) and (not placed[e][ci])
                obs.append({"name": c, "pose": [cb[0], cb[1], cb[2], 1, 0, 0, 0],
                            "enabled": bool(enabled)})
            bb = to_base(sn["bowl"], sn)
            obs.append({"name": "bowl", "pose": [bb[0], bb[1], bb[2], 1, 0, 0, 0], "enabled": True})
            worlds.append(obs)
        pl.world_batch(worlds)

    # 픽 순서(A): env별 미placed·retry미소진 큐브 선정.
    #   order_mode="fixed": 고정 인덱스 순(옛 92.8% 동치). "isolated": 가장 고립(회귀 격리용).
    def select_targets(sn_list, placed, failed):
        tgt = [None] * N
        for e in range(N):
            sn = sn_list[e]
            cand = [c for ci, c in enumerate(cubes)
                    if not placed[e][ci] and failed[e][ci] <= args.max_retry]
            if not cand:
                continue
            if args.order_mode == "fixed" or len(cand) == 1:
                tgt[e] = cand[0]
                continue
            best = None
            for c in cand:
                pc = sn["obj"][c][:2]
                mind = min(float(np.linalg.norm(sn["obj"][o][:2] - pc)) for o in cand if o != c)
                if best is None or mind > best[0]:
                    best = (mind, c)
            tgt[e] = best[1]
        return tgt

    # ── grasp roll 선정 (배치 IK, per-env 스코어) ──
    ROLLS = (math.pi / 2, -math.pi / 2, 0.0, math.pi)

    _DUMMY_PROB = {"tcp_base": [0.2, 0.0, 0.05], "yaw": 0.0, "grip": 0.785,
                   "pitch_max_deg": -70.0, "roll": 0.0}

    def select_grasp(sn_list, tgt, placed, active):
        """env별 target(tgt[e]) grasp (roll, q_desc, q_slide) 선정. inactive/None=건너뜀.

        닫힘축(ax)=jaw 분리 방향. clearance 를 **닫힘축 방향 finger 점**으로 측정해 이웃을
        향해 닫는 roll 을 penalize → 이웃과 수직으로 닫는 roll 선택(예: X-나란 큐브 → Y-close).
        (이전 버그: finger 점을 닫힘축 **수직**(fx,fy)으로 잡아 X-close 가 오히려 우대됨.)"""
        cur = cur_arm()
        per_roll = []
        for roll in ROLLS:
            pd, ps, meta = [], [], []
            for e in range(N):
                if not active[e] or tgt[e] is None:
                    pd.append(dict(_DUMMY_PROB)); ps.append(dict(_DUMMY_PROB))
                    meta.append((-1e9, 0.0)); continue
                target = tgt[e]
                sn = sn_list[e]
                cube_w = sn["obj"][target]
                cyaw = sn["objyaw"][target]
                px, py, pz = float(cube_w[0]), float(cube_w[1]), float(cube_w[2])
                size = CUBE_SIZES[target]
                so = max(args.side_offset, size * 0.5 + 0.018)
                gz = pz + args.grasp_dz
                yb = yaw_base(sn, cyaw)
                ax = closing_axis(cube_w, sn, cyaw, roll)
                dx, dy = math.cos(ax), math.sin(ax)      # 닫힘축(jaw 분리) 방향
                bx0, by0 = px - so * dx, py - so * dy     # 근측 finger(side-approach 접근점)
                # clearance: 이웃 큐브·그릇 대비. finger 점 = **닫힘축 방향** ±fh(이웃 향한 닫음 검출)
                bx, by = float(sn["bowl"][0]), float(sn["bowl"][1])
                others = [sn["obj"][c] for ci, c in enumerate(cubes)
                          if c != target and not placed[e][ci]]
                fh = 0.045
                pts = [(px + fh * dx, py + fh * dy), (bx0, by0),
                       (0.5 * (bx0 + px), 0.5 * (by0 + py))]
                clear = 1e9
                for cxx, cyy in pts:
                    clear = min(clear, math.hypot(cxx - bx, cyy - by) - args.bowl_clear)
                    for o in others:
                        clear = min(clear, math.hypot(cxx - float(o[0]), cyy - float(o[1])) - args.cube_clear)
                pd.append({"tcp_base": to_base([bx0, by0, gz], sn), "yaw": yb,
                           "grip": args.grip_open, "pitch_max_deg": -70.0, "roll": roll})
                ps.append({"tcp_base": to_base([px, py, gz], sn), "yaw": yb,
                           "grip": args.grip_open, "pitch_max_deg": -70.0, "roll": roll})
                meta.append((clear, float(cur[e][4])))
            rd = pl.ik_batch(pd)
            rs = pl.ik_batch(ps)
            per_roll.append((roll, rd, rs, meta))
        sel = [None] * N
        for e in range(N):
            if not active[e]:
                continue
            best = None
            for roll, rd, rs, meta in per_roll:
                if not (rd["oks"][e] and rs["oks"][e]):
                    continue
                clear, cur_roll = meta[e]
                q_s = rs["qs"][e]
                clear_norm = max(0.0, min(1.0, clear / 0.04))
                bias = 0.1 if abs(abs(roll) - math.pi / 2) < 1e-3 else 0.0
                flip = abs(q_s[4] - cur_roll)
                # clearance 지배(1.5×) → 이웃 향한 closing 회피. flip 경감(0.15). 침투 강패널티.
                score = 1.5 * clear_norm + bias - 0.15 * flip + (-1.0 if clear < 0 else 0.0)
                if best is None or score > best[0]:
                    best = (score, roll, rd["qs"][e], rs["qs"][e])
            sel[e] = None if best is None else (best[1], best[2], best[3])
        return sel

    # ── IK 배치 헬퍼: 각 env 의 (world pt, yaw, roll) → q (6) ──
    def ik_points(sn_list, points_w, yaws, rolls, pitch_max, active):
        """points_w (N,3 world list), yaws/rolls/active (N) → (qs(list/None), oks(np))."""
        probs = []
        for e in range(N):
            sn = sn_list[e]
            probs.append({"tcp_base": to_base(points_w[e], sn), "yaw": float(yaws[e]),
                          "grip": args.grip_open, "pitch_max_deg": pitch_max, "roll": float(rolls[e])})
        r = pl.ik_batch(probs)
        oks = np.array(r["oks"], bool) & active
        return r["qs"], oks

    # ── lock-step pick-place: env별 target(tgt[e]) 동시 처리 (phase 만 균일) ──
    def pick_step(sn_list, tgt, placed, active, start_q6):
        push_world(sn_list, tgt, placed)
        sel = select_grasp(sn_list, tgt, placed, active)
        active = active & np.array([s is not None for s in sel], bool)
        n_reach = int(active.sum())
        log(f"[batch] pick: grasp 도달 {n_reach}/{int(np.array([t is not None for t in tgt]).sum())}")
        if n_reach == 0:
            return placed

        # per-env grasp 파라미터
        q_desc = np.tile(np.array(READY + [args.grip_open]), (N, 1)).astype(np.float32)
        q_slid = q_desc.copy()
        rolls = np.zeros(N, np.float32)
        pre_w = [[0.3, 0.0, 0.9]] * N
        lift_w = [[0.3, 0.0, 0.9]] * N
        bowl_w = [[0.3, 0.0, 0.9]] * N
        yaws = np.zeros(N, np.float32)
        for e in range(N):
            if sel[e] is None or not active[e]:
                continue
            roll, qd, qs = sel[e]
            rolls[e] = roll
            q_desc[e] = qd
            q_slid[e] = qs
            sn = sn_list[e]
            target = tgt[e]
            cube_w = sn["obj"][target]
            cyaw = sn["objyaw"][target]
            px, py, pz = float(cube_w[0]), float(cube_w[1]), float(cube_w[2])
            size = CUBE_SIZES[target]
            so = max(args.side_offset, size * 0.5 + 0.018)
            ax = closing_axis(cube_w, sn, cyaw, roll)
            dx, dy = math.cos(ax), math.sin(ax)
            yaws[e] = yaw_base(sn, cyaw)
            pre_w[e] = [px - so * dx, py - so * dy, pz + args.pre_z]
            lift_w[e] = [px, py, pz + args.lift_z]
            bowl_w[e] = [float(sn["bowl"][0]), float(sn["bowl"][1]), DESK_TOP_Z + args.bowl_z]

        q_pre, ok_pre = ik_points(sn_list, pre_w, yaws, rolls, -45.0, active)
        q_lift, ok_lift = ik_points(sn_list, lift_w, yaws, rolls, -30.0, active)
        q_bowl, ok_bowl = ik_points(sn_list, bowl_w, yaws, rolls, -10.0, active)
        active = active & ok_pre & ok_lift & ok_bowl
        log(f"[batch] {target}: pre/lift/bowl IK 후 active {int(active.sum())}/{N}")

        # ① start → pre (plan_batch, transit)
        def goal6(qlist, fallback_arm):
            g = []
            for e in range(N):
                if active[e] and qlist[e] is not None:
                    g.append([float(v) for v in qlist[e]])
                else:
                    g.append(fallback_arm[e].tolist() + [args.grip_open])  # hold
            return g
        start6 = [list(start_q6[e]) for e in range(N)]
        arm_now = cur_arm()
        gp = goal6(q_pre, arm_now)
        rp = pl.plan_batch(start6, gp)
        if not rp.get("ok"):
            log("[batch] pick: →pre plan_batch 실패 — skip")
            return placed
        run_traj(np.array(rp["trajs"], np.float32), grip_vec(active, args.grip_open), args.stride)
        active = active & np.array(rp["success"], bool)

        # ② pre→descend, ③ descend→slide (직접 보간)
        seq_exec(q_desc[:, :5], active, grip_vec(active, args.grip_open), 16)
        seq_exec(q_slid[:, :5], active, grip_vec(active, args.grip_open), 14)
        # ④ grasp close
        settle(q_slid[:, :5], grip_vec(active, args.grip_close), args.close_steps)
        # ⑤ grasp→lift (직접)
        seq_exec(q_lift_arm := np.array([(q_lift[e] if (active[e] and q_lift[e]) else q_slid[e].tolist())
                                         for e in range(N)], np.float32)[:, :5],
                 active, grip_vec(active, args.grip_close), 16)
        # ⑥ lift→bowl (plan_batch transit, 실패 env 는 직접 fallback)
        arm_now = cur_arm()
        gb = goal6(q_bowl, arm_now)
        start_lift = [arm_now[e].tolist() + [args.grip_close] for e in range(N)]
        rb = pl.plan_batch(start_lift, gb)
        bowl_arm = np.array([(q_bowl[e][:5] if (active[e] and q_bowl[e]) else arm_now[e].tolist())
                             for e in range(N)], np.float32)
        if rb.get("ok"):
            run_traj(np.array(rb["trajs"], np.float32), grip_vec(active, args.grip_close), args.stride)
            plan_ok = np.array(rb["success"], bool)
            fb = active & ~plan_ok
            if fb.any():
                seq_exec(bowl_arm, fb, grip_vec(fb, args.grip_close), 30)
        else:
            seq_exec(bowl_arm, active, grip_vec(active, args.grip_close), 30)
        # ⑦ release
        settle(bowl_arm, grip_vec(active, args.grip_open), 12)

        # placed 판정 (per env, env별 target)
        sn2 = snap_all()
        for e in range(N):
            if not active[e]:
                continue
            cp = sn2[e]["obj"][tgt[e]]
            bp = sn_list[e]["bowl"]
            in_xy = float(np.linalg.norm(cp[:2] - bp[:2])) < BOWL_SUCCESS_RADIUS
            z_rel = float(cp[2]) - DESK_TOP_Z
            if in_xy and (BOWL_HEIGHT_RANGE[0] <= z_rel <= BOWL_HEIGHT_RANGE[1] + 0.10):
                placed[e][cubes.index(tgt[e])] = True
        return placed

    def run_round():
        placed = [[False] * len(cubes) for _ in range(N)]
        if fail_layouts is not None:                       # 덤프 layout 재현 → 주입 후 정착
            apply_fail_layouts(fail_layouts)
        settle(np.tile(READY, (N, 1)), np.full(N, args.grip_open, np.float32), 35)
        layout0 = capture_layout0()                        # 정착된 초기 layout(덤프용)
        s0 = nstep[0]
        ncubes = len(cubes)
        failed = [[0] * ncubes for _ in range(N)]          # env별 큐브별 실패 횟수(retry 소진)
        # 동적 픽: 매 step env별 가장 고립된 미placed 큐브 선정(A) → 실패 env 만 bounded 재시도.
        first = True
        for _ in range(ncubes * (args.max_retry + 1) + 2):  # budget cap
            if not simulation_app.is_running():
                break
            sn_list = snap_all()
            tgt = select_targets(sn_list, placed, failed)
            active = np.array([t is not None for t in tgt], bool)
            if not active.any():
                break
            if first:
                start_q6 = [list(READY) + [args.grip_open] for _ in range(N)]
                first = False
            else:
                start_q6 = [cur_arm()[e].tolist() + [args.grip_open] for e in range(N)]
            placed = pick_step(sn_list, tgt, placed, active, start_q6)
            for e in range(N):                              # 시도했는데 미placed → 실패+1
                if active[e]:
                    ci = cubes.index(tgt[e])
                    if not placed[e][ci]:
                        failed[e][ci] += 1
        settle(np.tile(READY, (N, 1)), np.full(N, args.grip_open, np.float32), 30)
        per_env = [sum(p) for p in placed]
        total = sum(per_env)
        sim_t = (nstep[0] - s0) * CONTROL_DT
        log("=" * 64)
        log(f"[batch] ROUND 결과: placed 총 {total}/{N * len(cubes)} "
            f"(env평균 {total / N:.2f}/{len(cubes)}) | {nstep[0]-s0} steps = {sim_t:.1f}s")
        log("=" * 64)
        return per_env, layout0

    results = []
    last_layout0 = None
    # livestream(public_ip) 면 사용자가 관전하도록 무한 반복
    watch = bool(getattr(args, "livestream", 0)) and not args.headless
    loop_n = (10 ** 9 if watch else max(1, args.loop))
    t_wall0 = time.time()
    i = 0
    while i < loop_n and simulation_app.is_running():
        if i > 0:
            env.reset()
            q_bias = torch.zeros((N, 5), device=device)
        log(f"[batch] ===== ROUND {i+1} =====")
        per_env, last_layout0 = run_round()
        results.append(per_env)
        i += 1
    wall = time.time() - t_wall0

    flat = [v for r in results for v in r]
    n_cubes = len(cubes)
    if flat:
        success_rate = sum(flat) / (len(flat) * n_cubes)
        full = sum(1 for v in flat if v == n_cubes)
        log(f"[batch] ===== 종합: {len(flat)} env-round | cube 성공률 {success_rate*100:.1f}% | "
            f"all-{n_cubes} env {full}/{len(flat)} ({full/len(flat)*100:.1f}%) | wall {wall:.1f}s =====")
        if args.taxonomy:
            with open(args.taxonomy, "w") as fh:
                json.dump({"num_envs": N, "loops": args.loop, "n_cubes": n_cubes,
                           "per_round_placed": results, "success_rate": success_rate,
                           "full_success_envs": full, "wall_s": wall}, fh, indent=2)
            log(f"[batch] taxonomy 저장: {args.taxonomy}")

    # 최악 실패 layout 덤프 (마지막 round 의 layout0 기준, placed 적은 순)
    if args.dump_fail and last_layout0 is not None and results:
        last = results[-1]
        order = sorted(range(N), key=lambda e: last[e])          # placed 적은 순
        worst = order[: args.dump_n]
        dump = [{"placed": last[e], **last_layout0[e]} for e in worst]
        with open(args.dump_fail, "w") as fh:
            json.dump(dump, fh, indent=2)
        log(f"[batch] dump_fail 저장: {args.dump_fail} (worst {len(dump)} env, placed={[last[e] for e in worst]})")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
