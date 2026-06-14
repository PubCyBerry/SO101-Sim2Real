"""cuRobo planner 사이드카 — ZMQ REP 서버 (PICKCUBE_CUROBO P2 핵심).

⚠ **isaac 절대 import 금지**(isaaclab.app 부팅 시 omni.warp.core 1.8.2 가 cuRobo 의 warp 1.14 를
밀어내 깨짐 — in-process 불가, ABI 게이트로 확증). 이 프로세스는 cuRobo + warp 1.14 만.
Isaac env 프로세스(omni.warp.core)와 ZMQ 로 분리 통신. plan 은 phase 당 1회라 IPC 오버헤드 무시.

프로토콜 (JSON, REQ/REP):
  요청 {"cmd": "ping"}                                          → {"ok": true}
  요청 {"cmd": "ik", "tcp_base":[x,y,z], "yaw":r, "grip":g,     → {"ok": bool, "q":[6], "pos_err_mm":f, "pitch_deg":f}
        "pitch_max_deg":-20}                                      (D9: 해석적 seed/orientation → cuRobo IK refine = 정확 config)
  요청 {"cmd": "plan", "start_q":[6], "goal_q":[6]}             → {"ok": bool, "traj":[[6]...], "dt":f}
  요청 {"cmd": "shutdown"}                                       → {"ok": true} 후 종료

실행:
    uv run --no-sync --group isaac python scripts/planning/curobo_planner_server.py --port 5599
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YML = os.path.join(ROOT, "assets/robots/so101_curobo.yml")
sys.path.insert(0, os.path.join(ROOT, "scripts/sim"))
from so101_kinematics import SO101Kinematics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--scene", type=str, default="", help="cuRobo scene yml(장애물). 빈값=free space")
    args = ap.parse_args()

    import torch
    import zmq

    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import GoalToolPose, JointState, Pose

    # 동적 장애물(set_world) 지원 — per-env world(다른 큐브·그릇).
    # collision_cache 만으론 scene_collision_checker 가 안 생겨 더미 scene_model 필요(checker 트리거).
    # 실제 장애물은 update_world 로 교체. 더미 _placeholder 는 10m 밖이라 무해.
    CACHE = {"cuboid": 16}
    EMPTY_SCENE = os.path.join(ROOT, "assets/robots/curobo_empty_scene.yml")
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=YML, scene_model=EMPTY_SCENE, collision_cache=CACHE))
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    jn = planner.joint_names
    dof = len(jn)
    tool = planner.tool_frames[0]
    # position 우선·orientation flex(0.5rad) → 도달가능 큐브 위치 정확히, 자세 best-effort.
    # 동일 scene+cache → set_world 시 IK 도 장애물 회피.
    ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=YML, num_seeds=64, self_collision_check=True,
        position_tolerance=0.004, orientation_tolerance=0.5,
        scene_model=EMPTY_SCENE, collision_cache=CACHE,
    ))
    ak = SO101Kinematics()

    def set_world(cuboids):
        # plan_cspace(궤적)만 충돌-aware. IK 는 충돌-free(reachability·D9 refine 전용) — IK 에도
        # 장애물 주면 이웃 근처 grasp config 를 그리퍼 sphere 스침으로 과다 기각(수동 가능한데 실패).
        # 이웃 회피는 env 쪽 clearance roll-선택 + plan_cspace 가 담당(SM 패턴).
        cubs = [Cuboid(name=c["name"], dims=list(c["dims"]), pose=list(c["pose"])) for c in cuboids]
        planner.update_world(Scene(cuboid=cubs))
        return len(cubs)

    print(f"[planner] warmup OK  joints={jn}  collision_cache={CACHE}", flush=True)

    def cu_fk(q6):
        q = torch.tensor([q6], device="cuda", dtype=torch.float32)
        p = planner.kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=jn)).tool_poses.get_link_pose(tool)
        return p.position.reshape(3), p.quaternion.reshape(4)

    def solve_ik(tcp, yaw, grip, pitch_max_deg, roll=0.0):
        """D9: 해석적 ik_reach → feasible orientation+seed → cuRobo IK refine = 정확 goal config."""
        sol = ak.ik_reach(tuple(tcp), yaw, pitch_max=math.radians(pitch_max_deg), roll_offset=roll)
        if sol is None:
            return None, float("nan"), float("nan")
        q5, pitch = sol
        seed = q5 + [grip]
        _, Q = cu_fk(seed)  # 해석적 config 의 실제 cuRobo orientation = feasible
        pos = torch.tensor([tcp], device="cuda", dtype=torch.float32)
        goal = GoalToolPose.from_poses(
            {tool: Pose(position=pos, quaternion=Q.reshape(1, 4))},
            ordered_tool_frames=ik.tool_frames, num_goalset=1,
        )
        seed_js = JointState.from_position(
            torch.tensor([seed], device="cuda", dtype=torch.float32), joint_names=jn)
        res = ik.solve_pose(goal, current_state=seed_js)
        if bool(res.success.any()):
            q = res.js_solution.position.reshape(-1, dof)[0].tolist()
            perr = res.position_error.reshape(-1)[0].item() * 1000.0
            return q, perr, math.degrees(pitch)
        # 폴백: 해석적 config 그대로(부정확하지만 None 보단 나음)
        return seed, float("nan"), math.degrees(pitch)

    def plan(start_q, goal_q):
        s = JointState.from_position(torch.tensor([start_q], device="cuda", dtype=torch.float32), joint_names=jn)
        g = JointState.from_position(torch.tensor([goal_q], device="cuda", dtype=torch.float32), joint_names=jn)
        res = planner.plan_cspace(g, s, max_attempts=8)
        if res is None or not bool(res.success.any()):
            return None, 0.0
        interp = res.get_interpolated_plan()
        p = interp.position.squeeze(0)
        if p.ndim == 3:  # (1, horizon, dof) → (horizon, dof)
            p = p[0]
        traj = p.cpu().tolist()
        dt = float(getattr(planner.trajopt_solver.config, "interpolation_dt", 0.02))
        return traj, dt

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    addr = f"tcp://127.0.0.1:{args.port}"
    sock.bind(addr)
    print(f"[planner] ZMQ REP bind {addr} — 대기", flush=True)

    while True:
        try:
            req = json.loads(sock.recv())
        except Exception as e:  # noqa: BLE001
            sock.send(json.dumps({"ok": False, "err": f"bad request: {e}"}).encode())
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                rep = {"ok": True, "joints": jn}
            elif cmd == "set_world":
                n = set_world(req.get("cuboids", []))
                rep = {"ok": True, "n_obstacles": n}
            elif cmd == "ik":
                q, perr, pit = solve_ik(req["tcp_base"], float(req.get("yaw", 0.0)),
                                        float(req.get("grip", 0.785)),
                                        float(req.get("pitch_max_deg", -20.0)),
                                        float(req.get("roll", 0.0)))
                rep = {"ok": q is not None, "q": q, "pos_err_mm": perr, "pitch_deg": pit}
            elif cmd == "plan":
                traj, dt = plan(req["start_q"], req["goal_q"])
                rep = {"ok": traj is not None, "traj": traj, "dt": dt,
                       "n": (len(traj) if traj else 0)}
            elif cmd == "shutdown":
                sock.send(json.dumps({"ok": True}).encode())
                break
            else:
                rep = {"ok": False, "err": f"unknown cmd {cmd}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rep = {"ok": False, "err": str(e)}
        sock.send(json.dumps(rep).encode())

    print("[planner] 종료", flush=True)
    sock.close()
    ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
