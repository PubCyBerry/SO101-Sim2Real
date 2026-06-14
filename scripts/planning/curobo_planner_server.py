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

P3 배치(multi-env, lock-step) — additive. BatchMotionPlanner(multi_env) + 배치 IKSolver(collision-free).
  요청 {"cmd":"init_batch", "n_envs":N, "cubes":[{"name","dims":[3]}...], "bowl_dims":[3],   → {"ok", "joints", "n_envs"}
        "ik_seeds":64, "max_attempts":4}
  요청 {"cmd":"world_batch", "worlds":[ per-env [{"name","pose":[7],"enabled":bool}...] ]}    → {"ok"}
  요청 {"cmd":"ik_batch", "probs":[ {"tcp_base":[3],"yaw","grip","pitch_max_deg","roll"}×N ]}  → {"ok","qs":[[6]/null×N],"oks":[bool×N],"pos_err_mm":[N],"pitch_deg":[N]}
  요청 {"cmd":"plan_batch", "start_q":[N×6], "goal_q":[N×6]}                                    → {"ok","trajs":[[H×6]×N],"success":[bool×N],"n":H,"dt"}

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

    from curobo.batch_motion_planner import BatchMotionPlanner
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

    # ── P3 배치(multi-env, lock-step) — additive. 위 단일-env 경로는 그대로 둠 ──────────
    # BatchMotionPlanner(multi_env=True) = N문제 N world 1 GPU call. IK 는 collision-free
    # 배치(D11), plan_cspace 만 충돌-aware(per-env world). 첫 init_batch 때 lazy 빌드(VRAM).
    BATCH: dict = {}

    def init_batch(n_envs, cubes, bowl_dims, ik_seeds=64, max_attempts=4, chunk=0):
        import yaml
        # 물리배치(Isaac n_envs) ↔ planner 배치(chunk) 분리. planner 는 chunk 슬롯만 빌드(VRAM),
        # n_envs 문제는 chunk 단위로 나눠 처리(부분청크 패딩). chunk=0/≥n_envs 면 단일배치(n_envs).
        chunk = n_envs if (chunk <= 0 or chunk >= n_envs) else int(chunk)
        cube_names = [c["name"] for c in cubes]
        names = cube_names + ["bowl"]
        n_cuboid = max(8, len(names))
        # per-env world buffer = scene_model 이 **길이 chunk 의 scene 리스트**일 때만 chunk env 할당
        # (SceneCollisionCfg.__post_init__: num_envs=len(scene_model)). 동일 스키마 chunk개 임시 YAML.
        far = [10.0, 10.0, -10.0, 1.0, 0.0, 0.0, 0.0]
        one = {"cuboid": {}}
        for c in cubes:
            one["cuboid"][c["name"]] = {"dims": list(c["dims"]), "pose": list(far)}
        one["cuboid"]["bowl"] = {"dims": list(bowl_dims), "pose": list(far)}
        scene_list = [json.loads(json.dumps(one)) for _ in range(chunk)]
        scene_path = os.path.join(ROOT, f"outputs/_batch_scene_{chunk}.yml")
        with open(scene_path, "w") as fh:
            yaml.safe_dump(scene_list, fh)
        bp = BatchMotionPlanner(MotionPlannerCfg.create(
            robot=YML, scene_model=scene_path, collision_cache={"cuboid": n_cuboid},
            multi_env=True, max_batch_size=chunk))
        bp.warmup(enable_graph=False, num_warmup_iterations=5)
        bik = InverseKinematics(InverseKinematicsCfg.create(
            robot=YML, num_seeds=ik_seeds, self_collision_check=True,
            position_tolerance=0.004, orientation_tolerance=0.5,
            max_batch_size=chunk))
        BATCH.update(n_envs=n_envs, chunk=chunk, planner=bp, ik=bik, names=names,
                     cube_names=cube_names, worlds=None, max_attempts=int(max_attempts),
                     dt=float(getattr(bp.trajopt_solver.config, "interpolation_dt", 0.02)))
        print(f"[planner] init_batch OK  n_envs={n_envs}  chunk={chunk}  "
              f"obstacles={names}  ik_seeds={ik_seeds}", flush=True)
        return jn

    def _chunks(n, c):
        return [(a, min(a + c, n)) for a in range(0, n, c)]

    def _pose(pose7):
        p = torch.tensor([pose7[:3]], device="cuda", dtype=torch.float32)
        q = torch.tensor([pose7[3:7]], device="cuda", dtype=torch.float32)
        return Pose(position=p, quaternion=q)

    def world_batch(worlds):
        # n_envs 개 world 스펙을 저장만(planner 슬롯엔 plan_batch 가 chunk 단위로 로드).
        BATCH["worlds"] = worlds
        return True

    def _load_chunk_world(env_lo, env_hi):
        """worlds[env_lo:env_hi] 를 planner 슬롯 0..(hi-lo) 에 로드(per-chunk)."""
        worlds = BATCH.get("worlds")
        if not worlds:
            return
        chk = BATCH["planner"].scene_collision_checker
        for slot, env_idx in enumerate(range(env_lo, env_hi)):
            for ob in worlds[env_idx]:
                chk.update_obstacle_pose(ob["name"], _pose(ob["pose"]), slot)
                chk.enable_obstacle(ob["name"], bool(ob.get("enabled", True)), slot)

    def _ik_one(probs):
        """≤chunk 문제 collision-free IK(정확히 chunk 로 패딩). D9: 해석적 ik_reach → feasible
        orientation+seed → cuRobo IK refine. cu_fk·solve_pose 가 num_envs=chunk 와 정합하도록 패딩."""
        bik = BATCH["ik"]
        chunk = BATCH["chunk"]
        m = len(probs)
        seeds, tcps, ana_ok, pitches = [], [], [], []
        for pr in probs:
            sol = ak.ik_reach(tuple(pr["tcp_base"]), float(pr.get("yaw", 0.0)),
                              pitch_max=math.radians(float(pr.get("pitch_max_deg", -20.0))),
                              roll_offset=float(pr.get("roll", 0.0)))
            if sol is None:
                seeds.append([0.0] * dof); tcps.append([0.0, 0.0, 0.0])
                ana_ok.append(False); pitches.append(float("nan"))
            else:
                q5, pitch = sol
                seeds.append(q5 + [float(pr.get("grip", 0.785))])
                tcps.append(list(pr["tcp_base"])); ana_ok.append(True)
                pitches.append(math.degrees(pitch))
        while len(seeds) < chunk:   # 부분청크 → chunk 로 패딩(결과는 [:m] 슬라이스)
            seeds.append([0.0] * dof); tcps.append([0.0, 0.0, 0.0])
        seed_t = torch.tensor(seeds, device="cuda", dtype=torch.float32)          # (chunk, dof)
        ks = BATCH["planner"].kinematics.compute_kinematics(
            JointState.from_position(seed_t, joint_names=jn))
        Q = ks.tool_poses.get_link_pose(tool).quaternion.reshape(chunk, 4)
        pos = torch.tensor(tcps, device="cuda", dtype=torch.float32)              # (chunk, 3)
        goal = GoalToolPose.from_poses(
            {tool: Pose(position=pos, quaternion=Q)},
            ordered_tool_frames=bik.tool_frames, num_goalset=1)
        seed_js = JointState.from_position(seed_t, joint_names=jn)
        res = bik.solve_pose(goal, current_state=seed_js)
        succ = res.success.reshape(-1)
        qsol = res.js_solution.position.reshape(chunk, dof)
        perr = res.position_error.reshape(-1) * 1000.0
        qs, oks, errs = [], [], []
        for i in range(m):
            ok = bool(ana_ok[i]) and bool(succ[i].item())
            oks.append(ok)
            qs.append(qsol[i].tolist() if ok else (seeds[i] if ana_ok[i] else None))
            errs.append(float(perr[i].item()) if ana_ok[i] else float("nan"))
        return qs, oks, errs, pitches[:m]

    def ik_batch(probs):
        """n_envs 문제 IK — chunk 단위로 나눠 _ik_one 처리 후 concat. collision-free(world 불요)."""
        qs, oks, errs, pits = [], [], [], []
        for a, b in _chunks(len(probs), BATCH["chunk"]):
            q, o, e, p = _ik_one(probs[a:b])
            qs += q; oks += o; errs += e; pits += p
        return qs, oks, errs, pits

    def _plan_one(start_q, goal_q, env_lo, env_hi):
        """≤chunk 문제 plan_cspace. per-chunk world 로드 + chunk 로 패딩. (per-env trim 리스트, success) 반환."""
        bp = BATCH["planner"]
        chunk = BATCH["chunk"]
        m = len(start_q)
        _load_chunk_world(env_lo, env_hi)
        sq = list(start_q) + [start_q[-1]] * (chunk - m)   # chunk 로 패딩
        gq = list(goal_q) + [goal_q[-1]] * (chunk - m)
        s = JointState.from_position(torch.tensor(sq, device="cuda", dtype=torch.float32), joint_names=jn)
        g = JointState.from_position(torch.tensor(gq, device="cuda", dtype=torch.float32), joint_names=jn)
        res = bp.plan_cspace(g, s, max_attempts=BATCH["max_attempts"])
        if res is None:
            return [None] * m, [False] * m
        succ = res.success
        succ = succ.any(dim=-1) if succ.ndim == 2 else succ
        succ = succ.reshape(-1).tolist()
        p = res.interpolated_trajectory.position
        if p.ndim == 4:
            p = p[:, 0]
        p = p.detach().cpu()
        lt = getattr(res, "interpolated_last_tstep", None)
        if lt is not None:
            lt = [max(1, int(x)) for x in lt.reshape(-1).tolist()]
        else:
            lt = [p.shape[1]] * p.shape[0]
        rows = [p[i, :lt[i]].tolist() for i in range(m)]   # per-env 가변길이(trim)
        return rows, succ[:m]

    def plan_batch(start_q, goal_q):
        """n_envs (start,goal) — chunk 단위 plan(per-chunk world). per-env trim 후 전역 Hmax 로 goal-pad
        해 균일화(lock-step batched step). inactive env 는 클라가 start=goal 로 전송(hold)."""
        n = len(start_q)
        all_rows, all_succ = [], []
        for a, b in _chunks(n, BATCH["chunk"]):
            rows, succ = _plan_one(start_q[a:b], goal_q[a:b], a, b)
            all_rows += rows
            all_succ += succ
        Hmax = max((len(r) for r in all_rows if r), default=2)
        trajs = []
        for r in all_rows:
            if not r:
                r = [list(start_q[len(trajs)])] if len(trajs) < n else [[0.0] * dof]
            row = list(r)
            if len(row) < Hmax:
                row += [row[-1]] * (Hmax - len(row))
            trajs.append(row)
        return trajs, all_succ, int(Hmax)

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
            elif cmd == "init_batch":
                joints = init_batch(int(req["n_envs"]), req["cubes"], req["bowl_dims"],
                                    ik_seeds=int(req.get("ik_seeds", 64)),
                                    max_attempts=int(req.get("max_attempts", 4)),
                                    chunk=int(req.get("chunk", 0)))
                rep = {"ok": True, "joints": joints, "n_envs": BATCH["n_envs"], "chunk": BATCH["chunk"]}
            elif cmd == "world_batch":
                world_batch(req["worlds"])
                rep = {"ok": True}
            elif cmd == "ik_batch":
                qs, oks, errs, pits = ik_batch(req["probs"])
                rep = {"ok": True, "qs": qs, "oks": oks, "pos_err_mm": errs, "pitch_deg": pits}
            elif cmd == "plan_batch":
                trajs, success, H = plan_batch(req["start_q"], req["goal_q"])
                rep = {"ok": trajs is not None, "trajs": trajs, "success": success,
                       "n": H, "dt": BATCH.get("dt", 0.02)}
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
