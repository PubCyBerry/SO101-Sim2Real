"""SO-101 native pose-goal 인터랙티브 Viser — 공식 motion_planning.py --visualize 구조 정합.

**정정(소스 검증)**: "5축 SO-101 은 cuRobo pose-goal 비가능"(옛 D2/C4/D9)은 틀렸다. 옛 결론은
`orientation_tolerance`(수렴 게이트)만 키워본 데서 나왔는데 그건 optimizer cost 를 못 끈다. 진짜
position-only API = **`ToolPoseCriteria.track_position()`**(회전 weight=0 → angular_distance·gradient
0 → orientation 게이트 무조건 통과 + optimizer 자세 무시). 커널 `wp_tool_pose` 까지 확인.

공식 예제(`ref_repos/curobo/.../motion_planning.py::interactive_motion_planning`)와 **동일 UI**:
  - 궤적 그래프(pos/vel/acc/jerk) 이미지 패널
  - **Move** 버튼  : position-only `plan_pose`(gizmo position 으로 이동, orientation 무시)
  - **Grasp** 버튼 : holder-up 일관 3단계. ① 후보(tilt×roll×az) 배치 IK → reachable 중 카메라 홀더
    최상(Δz↑) config q* 채택 ② 같은 자세 q* 로 hover IK → plan_cspace(joint-goal, 충돌회피)로 이동
    ③ hover↔grasp 직접 joint 보간 하강/상승(D11) + gripper open→close. hover·grasp 모두 q*=holder-up
    이라 모션 내내 홀더 위(중간 wrist flip 없음). 단일 plan_pose "찌르기" 아님

5-DOF 차이:
  - 공식 Move 는 full 6-DOF pose-goal → 5축 거의 실패. 여기선 **position-only** 로 작동.
  - 공식 Grasp(`plan_grasp`)는 내부적으로 full-pose criteria 로 approach/grasp/lift 를 풀어 5축에서
    실패 → 여기선 **자세 goal-set** 으로 대체(cuRobo 가 reachable 자세 자동 선택). 진짜 3단 grasp 는
    Stage 2 SM 에서.
  - 장애물 회피: graph planner(warmup enable_graph) + trajopt. feasible 타깃 ~90%. 실패 시 사유 표시
    (대개 5축이 그 위치를 pillar 안 겹치고 도달 못 하는 내재 infeasible).

실행:
    uv run --no-sync --group isaac python scripts/sim/motion_plan_so101_viser.py --port 8088
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YML = os.path.join(ROOT, "assets/robots/so101_curobo.yml")
SCENE = os.path.join(ROOT, "assets/robots/so101_scene.yml")


def _create_trajectory_image(trajectory, joint_names, title=""):
    """궤적(pos/vel/acc/jerk)을 PNG 배열로 렌더 — Viser GUI 패널용 (공식 예제 동일)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    traj = trajectory.squeeze(0)
    pos = np.atleast_2d(traj.position[0].cpu().numpy())  # (horizon, dof)
    dt_val = traj.dt.item() if traj.dt is not None else 0.02
    t = np.arange(pos.shape[0]) * dt_val

    vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None
    acc = np.atleast_2d(traj.acceleration[0].cpu().numpy()) if traj.acceleration is not None else None
    jrk = np.atleast_2d(traj.jerk[0].cpu().numpy()) if traj.jerk is not None else None

    n_plots = 1 + (vel is not None) + (acc is not None) + (jrk is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(5, 2 * n_plots), dpi=100, sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_data = [(pos, "Position (rad)")]
    if vel is not None:
        plot_data.append((vel, "Velocity (rad/s)"))
    if acc is not None:
        plot_data.append((acc, "Accel (rad/s²)"))
    if jrk is not None:
        plot_data.append((jrk, "Jerk (rad/s³)"))

    for ax, (data, ylabel) in zip(axes, plot_data):
        for j in range(data.shape[1]):
            label = joint_names[j] if j < len(joint_names) else f"J{j}"
            if len(label) > 8:
                label = label[:6] + ".."
            ax.plot(t, data[:, j], linewidth=1.0, label=label)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_array = np.array(Image.open(buf))
    plt.close(fig)
    buf.close()
    return img_array


def _qmul_wxyz(a, b):
    """쿼터니언 Hamilton 곱 (wxyz, world-frame 회전 a 를 b 에 적용 = a⊗b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _roty(deg):
    import math
    t = math.radians(deg) / 2.0
    return [math.cos(t), 0.0, math.sin(t), 0.0]


def _rotz(deg):
    import math
    t = math.radians(deg) / 2.0
    return [math.cos(t), 0.0, 0.0, math.sin(t)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--no_scene", action="store_true", help="장애물 씬 없이(robot 만)")
    ap.add_argument("--play_dt", type=float, default=0.02, help="궤적 재생 step 간 sleep(s)")
    ap.add_argument("--max_attempts", type=int, default=8, help="plan_pose 재시도(graph_attempt=1 권장)")
    ap.add_argument("--grasp_yaws", type=int, default=12, help="Grasp 후보 azimuth(world-z, uniform) 분할 수")
    args = ap.parse_args()

    # Grasp goal-set: top-down + tilt cone × finger-yaw (5-DOF reachable 자세 manifold 커버).
    # tilt = approach pitch(0=top-down, 90=수평), roll = approach축 회전(카메라 홀더 up/down).
    # 먼 타깃은 5축이 top-down 불가 → 수평(tilt↑)+roll(홀더 위)으로 잡아야(사용자 지적). collision-aware
    # IK 가 후보 중 reachable+홀더 clear 선택.
    GRASP_TILTS_DEG = [0.0, 25.0, 50.0, 75.0, 90.0]
    GRASP_ROLLS_DEG = [-90.0, -45.0, 0.0, 45.0, 90.0, 180.0]  # approach축 회전(홀더 up/down 탐색)
    HOLDER_UP_MIN = 0.005  # grasp 채택 최소 홀더-그리퍼 높이차(m). 미만이면 holder-down → 거부

    import math

    import torch

    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose, ToolPoseCriteria
    from curobo.viewer import ViserVisualizer
    from curobo._src.types.robot import RobotCfg

    URDF = os.path.join(ROOT, "assets/robots/urdf/so_arm101.urdf")

    # ── Viser + 로봇 ───────────────────────────────────────────────────────
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=YML),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    # ── MotionPlanner (native pose-goal) ───────────────────────────────────
    # seeds=8·act_dist=0.025: feasible 타깃 회피 성공률 ↑(probe 90%). graph_attempt=1(기본) 유지
    # — graph 즉시(=0)는 5-DOF cramped 에서 오히려 악화(probe 62.5%).
    n_grasp = max(args.grasp_yaws, 1) * len(GRASP_TILTS_DEG) * len(GRASP_ROLLS_DEG)
    mp_kwargs = dict(robot=YML, num_trajopt_seeds=8, optimizer_collision_activation_distance=0.025,
                     orientation_tolerance=0.12)
    if not args.no_scene:
        mp_kwargs["scene_model"] = SCENE
    config = MotionPlannerCfg.create(**mp_kwargs)
    planner = MotionPlanner(config)
    joint_names = planner.joint_names
    tool = planner.tool_frames[0]

    # ── grasp 자세 선정용 배치 IK + holder FK (collision-free, D11) ─────────
    # 후보 N개를 각자 독립 goal 로 배치 IK → reachable 중 카메라 홀더가 가장 위(Δz)인 config 채택.
    kin_cfg = KinematicsCfg.from_robot_yaml_file(YML, urdf_path=URDF)
    fk_kin = Kinematics(kin_cfg)
    # use_cuda_graph=False: 후보 배치(N개)와 hover(1개) 두 batch 크기로 호출 → graph 재캡처 회피
    # (graph on 이면 "CUDA graph reset is not available" → CUDA context 깨짐).
    grasp_ik = InverseKinematics(InverseKinematicsCfg.create(
        robot=RobotCfg.create({"kinematics": kin_cfg}), num_seeds=30, self_collision_check=True,
        position_tolerance=0.008, orientation_tolerance=0.12, max_batch_size=n_grasp,
        use_cuda_graph=False))
    _hidx = fk_kin.config.kinematics_config.get_sphere_index_from_link_name("wrist_cam_mount_link").tolist()
    _gidx = fk_kin.config.kinematics_config.get_sphere_index_from_link_name("gripper_link").tolist()

    # 장애물 frame (드래그 가능)
    obstacle_frames = {}
    old_obstacle_poses = {}
    if not args.no_scene and config.scene_collision_cfg is not None:
        scene_cfg = config.scene_collision_cfg.scene_model
        obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
        old_obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames
        }

    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    # criteria 두 모드: Move=position-only(자세 무시), Grasp=full-pose(goal-set 자세 매칭)
    POSITION_ONLY = {tool: ToolPoseCriteria.track_position()}
    FULL_POSE = {tool: ToolPoseCriteria()}

    current_state = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=joint_names
    )

    # ── GUI: 궤적 plot 패널 + 버튼 ─────────────────────────────────────────
    server = viser_viz._server
    traj_plot = server.gui.add_image(
        _create_trajectory_image(
            JointState.from_position(
                planner.default_joint_state.position.unsqueeze(0).unsqueeze(0),
                joint_names=joint_names,
            ),
            joint_names, title="No trajectory yet",
        ),
        label="Joint Trajectory", format="png",
    )
    status_md = server.gui.add_markdown("**준비됨** — gizmo position 드래그 후 Move/Grasp")

    def update_obstacles():
        for k in obstacle_frames:
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def gizmo_pose():
        gp = {k.replace("target_", ""): v for k, v in viser_viz.get_control_frame_pose().items()}[tool]
        xyz = [float(v) for v in gp.position.reshape(3).tolist()]
        wxyz = [float(v) for v in gp.quaternion.reshape(4).tolist()]
        return xyz, wxyz

    grip_idx = joint_names.index("gripper") if "gripper" in joint_names else None

    def execute(traj, grip=None):
        nonlocal current_state
        t = traj.squeeze(0)
        n = t.position.shape[-2]
        for i in range(n):
            q = t.position[0, i, :].clone()
            if grip is not None and grip_idx is not None:
                q[grip_idx] = grip  # gripper open/close 시각화(plan 무관)
            wp = JointState.from_position(q.unsqueeze(0), joint_names=t.joint_names)
            viser_viz.set_joint_state(wp.squeeze(0))
            time.sleep(args.play_dt)
        last = t.position[0, -1, :].clone()
        if grip is not None and grip_idx is not None:
            last[grip_idx] = grip
        current_state = JointState.from_position(last.unsqueeze(0), joint_names=t.joint_names)

    # ── stuck 복구: chaining 으로 current_state 가 self-collision 경계 config 에 안착하면
    #    이후 모든 plan 이 "Start in collision" 으로 실패(영구 stuck). 실패 시 home 복귀 재시도. ──
    home_q = planner.default_joint_state.position.reshape(-1)[:len(joint_names)].clone()

    def go_home():
        nonlocal current_state
        wp = JointState.from_position(home_q.unsqueeze(0), joint_names=joint_names)
        viser_viz.set_joint_state(wp.squeeze(0))
        current_state = JointState.from_position(home_q.unsqueeze(0), joint_names=joint_names)

    def _at_home():
        return bool(torch.allclose(current_state.position.reshape(-1)[:len(joint_names)], home_q, atol=1e-3))

    def _plan_recover(goal):
        """current_state 에서 plan_pose. 실패 & not-home → home 복귀 후 1회 재시도. (res, homed)."""
        active = planner.kinematics.get_active_js(current_state.clone())
        res = planner.plan_pose(goal, active, max_attempts=args.max_attempts, enable_graph_attempt=1)
        if res is not None and bool(res.success.any()):
            return res, False
        if not _at_home():
            print("[recover] start-state 막힘 → home 복귀 후 재시도")
            go_home()
            active = planner.kinematics.get_active_js(current_state.clone())
            res = planner.plan_pose(goal, active, max_attempts=args.max_attempts, enable_graph_attempt=1)
            if res is not None and bool(res.success.any()):
                return res, True
        return None, False

    def _cspace_recover(goal_q6):
        """joint-goal plan_cspace (충돌회피 transit). 실패 & not-home → home 복귀 재시도."""
        gs = JointState.from_position(goal_q6.reshape(1, -1), joint_names=joint_names)
        active = planner.kinematics.get_active_js(current_state.clone())
        res = planner.plan_cspace(gs, active, max_attempts=args.max_attempts)
        if res is not None and bool(res.success.any()):
            return res, False
        if not _at_home():
            print("[recover] cspace start 막힘 → home 복귀 재시도")
            go_home()
            active = planner.kinematics.get_active_js(current_state.clone())
            res = planner.plan_cspace(gs, active, max_attempts=args.max_attempts)
            if res is not None and bool(res.success.any()):
                return res, True
        return None, False

    is_moving = False

    def _run(label, build_goal, criteria):
        """공통: criteria 주입 → goal 빌드 → plan_pose → plot + 실행. 실패 시 사유 표시."""
        nonlocal is_moving
        if is_moving:
            print(f"[skip] 이미 실행 중")
            return

        def work():
            nonlocal is_moving
            is_moving = True
            try:
                update_obstacles()
                planner.update_tool_pose_criteria(criteria)
                xyz, wxyz = gizmo_pose()
                goal = build_goal(xyz, wxyz)
                res, homed = _plan_recover(goal)
                if res is None:
                    why = "도달불가/충돌 (5-DOF 내재 infeasible — home 에서도 실패)"
                    print(f"[{label}] FAIL — {why}")
                    status_md.content = f"❌ **{label} 실패** pos={[round(v,3) for v in xyz]} — {why}"
                    return
                interp = res.get_interpolated_plan()
                nwp = interp.position.shape[-2]
                tt = getattr(res, "total_time", 0.0)
                tag = " (home 복귀 후)" if homed else ""
                print(f"[{label}] OK{tag} pos={[round(v,3) for v in xyz]} waypoints={nwp} time={tt:.3f}s")
                traj_plot.image = _create_trajectory_image(
                    interp, joint_names, title=f"{label}  |  {tt:.3f}s  |  {nwp} wp")
                status_md.content = f"✅ **{label} OK**{tag} pos={[round(v,3) for v in xyz]}  ({nwp} wp, {tt:.2f}s)"
                execute(interp)
            except Exception:
                traceback.print_exc()
                status_md.content = "⚠️ 예외 — 콘솔 로그 확인"
            finally:
                is_moving = False

        threading.Thread(target=work, daemon=True).start()

    # Move = position-only single goal (gizmo position 도달, 자세 무시)
    def _move_goal(xyz, wxyz):
        pos = torch.tensor([xyz], device="cuda", dtype=torch.float32)
        quat = torch.tensor([wxyz], device="cuda", dtype=torch.float32)  # 무시(track_position)
        return GoalToolPose.from_poses({tool: Pose(position=pos, quaternion=quat)},
                                       ordered_tool_frames=planner.tool_frames)

    # Grasp 후보 자세: top-down([0,1,0,0]) 에 roll(approach축 회전=홀더 up/down) → tilt(pitch) →
    #   world-z azimuth. tilt 0=top-down·90=수평, roll 로 카메라 홀더 방향. collision-aware IK 가
    #   reachable+홀더 clear 한 것 선택. (먼 타깃=수평+roll, 가까운=top-down)
    def _grasp_candidates(xyz):
        az_n = max(args.grasp_yaws, 1)
        q_td = [0.0, 1.0, 0.0, 0.0]
        cands = []
        for tl in GRASP_TILTS_DEG:
            for rl in GRASP_ROLLS_DEG:
                base = _qmul_wxyz(_roty(tl), _qmul_wxyz(_rotz(rl), q_td))  # roll(홀더)→tilt(pitch)
                for k in range(az_n):
                    az = -180.0 + 360.0 * k / az_n
                    cands.append(_qmul_wxyz(_rotz(az), base))  # world-z azimuth
        return cands

    def _interp_exec(q_to, steps, grip):
        """현재 config → q_to 직접 joint 보간 실행(grasp 미세동작, D11: trajopt 과다기각 회피)."""
        nonlocal current_state
        q_from = current_state.position.reshape(-1)[:len(joint_names)].clone()
        q_to = q_to.reshape(-1)[:len(joint_names)].clone()
        for i in range(1, steps + 1):
            a = i / steps
            q = q_from * (1.0 - a) + q_to * a
            if grip is not None and grip_idx is not None:
                q[grip_idx] = grip
            viser_viz.set_joint_state(JointState.from_position(q.unsqueeze(0), joint_names=joint_names).squeeze(0))
            time.sleep(args.play_dt)
        last = q_to.clone()
        if grip is not None and grip_idx is not None:
            last[grip_idx] = grip
        current_state = JointState.from_position(last.unsqueeze(0), joint_names=joint_names)

    # Grasp = holder-up 일관 swoop 3단계 (공식 plan_grasp 모션을 5축 용으로 직접 구현):
    #   ① grasp config 선정 : 후보 N개 배치 IK → reachable 중 카메라 홀더 최상(Δz↑) config q* 채택
    #   ② approach : 같은 자세 q* 로 hover(큐브 위) IK → plan_cspace(joint-goal, 충돌회피)로 이동
    #                (hover·grasp 모두 q* = holder-up → 모션 내내 홀더 위, 중간 wrist flip 없음)
    #   ③ grasp/lift : hover↔grasp 직접 joint 보간(D11) + gripper open→close
    DESCEND_STEPS, LIFT_STEPS = 22, 22
    GRASP_OFF = 0.05           # hover = 큐브 +z OFF (작게 = q* 자세 유지하며 reachable)
    GRIP_OPEN, GRIP_CLOSE = 1.2, 0.2

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            print("[skip] 이미 실행 중")
            return

        def work():
            nonlocal is_moving
            is_moving = True
            try:
                update_obstacles()
                xyz, _ = gizmo_pose()
                hover_xyz = [xyz[0], xyz[1], xyz[2] + GRASP_OFF]

                # ① grasp config = 후보 배치 IK → reachable 중 홀더 최상(Δz) 채택 (holder-up)
                cands = _grasp_candidates(xyz)
                nc = len(cands)
                qb = torch.tensor(cands, device="cuda", dtype=torch.float32)
                gset = GoalToolPose.from_poses(
                    {tool: Pose(position=torch.tensor([xyz] * nc, device="cuda", dtype=torch.float32),
                                quaternion=qb)}, num_goalset=1)
                ik_res = grasp_ik.solve_pose(gset)
                succ = ik_res.success.reshape(-1)
                if not bool(succ.any()):
                    status_md.content = f"❌ **Grasp 실패** pos={[round(v,3) for v in xyz]} — 5축 grasp 자세 도달불가(reach-edge/높은 z)"
                    print("[Grasp] grasp-config IK FAIL (0 reachable)"); return
                sols = ik_res.js_solution.position.reshape(nc, -1, len(joint_names))[:, 0, :]
                S = fk_kin.compute_kinematics(JointState.from_position(sols, joint_names=joint_names)) \
                    .get_link_spheres().reshape(nc, -1, 4)
                holder_dz = S[:, _hidx, 2].mean(1) - S[:, _gidx, 2].mean(1)  # 홀더-그리퍼 높이차
                # holder-up 강제: 카메라 홀더가 그리퍼보다 위인 reachable 자세만. 없으면 거부(바닥 박기 금지).
                up_ok = succ & (holder_dz > HOLDER_UP_MIN)
                if not bool(up_ok.any()):
                    best_down = float(holder_dz.masked_fill(~succ, -1e9).max().item())
                    status_md.content = (f"❌ **Grasp 거부** pos={[round(v,3) for v in xyz]} — holder-up grasp 자세 "
                                         f"도달불가(최선 Δz={best_down:+.3f}↓, 바닥 충돌 회피). 다른 위치/거리")
                    print(f"[Grasp] no holder-up reachable (best Δz={best_down:+.3f}) → refuse"); return
                bi = int(holder_dz.masked_fill(~up_ok, -1e9).argmax().item())
                q_grasp = sols[bi]
                q_star = qb[bi:bi + 1]      # 선택된 holder-up 자세
                up = float(holder_dz[bi].item())

                # ② hover = 같은 자세 q* IK(q_grasp seed) → plan_cspace 로 이동(holder-up 일관)
                hov_goal = GoalToolPose.from_poses(
                    {tool: Pose(position=torch.tensor([hover_xyz], device="cuda", dtype=torch.float32),
                                quaternion=q_star)}, num_goalset=1)
                hov_res = grasp_ik.solve_pose(
                    hov_goal, current_state=JointState.from_position(q_grasp.reshape(1, -1), joint_names=joint_names))
                use_hover = bool(hov_res.success.any())
                approach_goal = (hov_res.js_solution.position.reshape(-1, len(joint_names))[0]
                                 if use_hover else q_grasp)
                resA, homed = _cspace_recover(approach_goal)
                if resA is None:
                    status_md.content = f"❌ **Grasp approach 실패** — transit 도달불가(home 에서도)"
                    print("[Grasp] approach(cspace) FAIL"); return
                interpA = resA.get_interpolated_plan()
                n_appr = interpA.position.shape[-2]
                traj_plot.image = _create_trajectory_image(interpA, joint_names, title=f"Grasp 1/3 approach  |  {n_appr} wp")
                tag = " (home 경유)" if homed else ""
                status_md.content = (f"✅ **Grasp 1/3 approach**{tag} reachable {int(succ.sum())}/{nc} · "
                                     f"홀더Δz={up:+.3f}{'↑' if up > 0 else '↓'} · {n_appr} wp")
                print(f"[Grasp] approach{tag}: cand={bi} reachable {int(succ.sum())}/{nc} holderΔz={up:+.3f} hover={use_hover} wp={n_appr}")
                execute(interpA, grip=GRIP_OPEN)

                # ③ descend → close → lift (직접 보간, q_grasp ↔ hover 모두 holder-up)
                hover_cfg = current_state.position.reshape(-1)[:len(joint_names)].clone()
                status_md.content = f"✅ **Grasp 2/3 descend** ({DESCEND_STEPS} step)"
                print("[Grasp] descend → cube")
                _interp_exec(q_grasp, steps=DESCEND_STEPS, grip=GRIP_OPEN)
                _interp_exec(q_grasp, steps=8, grip=GRIP_CLOSE)    # 제자리 close
                status_md.content = f"✅ **Grasp 3/3 lift** ({LIFT_STEPS} step)"
                print("[Grasp] lift → hover")
                _interp_exec(hover_cfg, steps=LIFT_STEPS, grip=GRIP_CLOSE)
                status_md.content = (f"✅ **Grasp 완료** approach {n_appr}wp → descend {DESCEND_STEPS} → "
                                     f"lift {LIFT_STEPS} · 홀더 {up:+.3f}{'↑' if up > 0 else '↓'}")
                print(f"[Grasp] done — approach {n_appr}wp / descend {DESCEND_STEPS} / lift {LIFT_STEPS}")
            except Exception:
                traceback.print_exc()
                status_md.content = "⚠️ 예외 — 콘솔 로그 확인"
            finally:
                is_moving = False

        threading.Thread(target=work, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(lambda _: _run("Move", _move_goal, POSITION_ONLY))
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    def on_home(_):
        nonlocal is_moving
        if is_moving:
            return
        go_home()
        status_md.content = "🏠 **Home 복귀** — current_state 초기화"
        print("[Home] reset to default")
    home_btn = server.gui.add_button("Home", color="gray")
    home_btn.on_click(on_home)

    print(f"\n[so101-viser] http://localhost:{args.port}  (tailscale: 100.79.237.116:{args.port})")
    print(f"[so101-viser] tool={tool}  joints={joint_names}")
    print("[so101-viser] Move=position-only · Grasp=holder-up 3단계(approach cspace→descend/lift) · Home=리셋")
    print("[so101-viser] 실패 시 home 복귀 자동 재시도(stuck 복구). 막히면 Home 클릭. 장애물 드래그 가능. Ctrl+C 종료.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
