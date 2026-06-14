"""SO-101 joint-goal 모션플래닝 인터랙티브 Viser (PICKCUBE_CUROBO P1+/P2 선검증).

공식 motion_planning.py 의 pose-goal(plan_pose)은 6-DOF orientation 까지 hard 매칭이라 5축 SO-101
에서 거의 항상 실패한다(C4/D2 벽, probe 로 확증). cuRobo IK 도 orientation cost 를 못 꺼 5-DOF
position-only 가 안 된다(reach 안 점도 실패 확인). 이 스크립트는 **P2 메커니즘 그대로** 우회한다:

    gizmo position  →  해석적 IK(SO101Kinematics, position-우선·orientation best-effort)
                    →  goal joint config  →  plan_cspace  →  실행

- IK = `so101_kinematics.SO101Kinematics.ik_reach` (C3 검증, top-down 우선 pitch sweep). 5축이라
  높고 먼 horizontal pose 는 거절(grasp 도메인상 정상) — 책상면 큐브존이 정상 reach.
- plan_cspace 가 충돌-free joint→joint 궤적 (장애물 pillar·self-collision 회피).
- "Plan & Move" 버튼 1클릭 = one-shot global plan (P2 미리보기). gizmo·장애물 드래그 가능.

⚠ gizmo orientation 은 무시(position-only). grasp 자세는 ik_reach 의 pitch sweep 이 결정.

실행:
    uv run --no-sync --group isaac python scripts/sim/motion_plan_so101_viser.py --port 8088
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YML = os.path.join(ROOT, "assets/robots/so101_curobo.yml")
SCENE = os.path.join(ROOT, "assets/robots/so101_scene.yml")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kinematics import SO101Kinematics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--no_scene", action="store_true", help="장애물 씬 없이(robot 만)")
    ap.add_argument("--play_dt", type=float, default=0.02, help="궤적 재생 step 간 sleep(s)")
    ap.add_argument("--grasp_yaw", type=float, default=0.0, help="손가락 닫힘축 yaw(rad)")
    ap.add_argument("--pitch_min_deg", type=float, default=-90.0, help="ik_reach pitch 하한(top-down)")
    ap.add_argument("--pitch_max_deg", type=float, default=20.0,
                    help="ik_reach pitch 상한(수평쪽). 데모는 home 등 flat pose 까지 reach 하려 넓힘")
    args = ap.parse_args()

    import torch

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import ContentPath, JointState, Pose
    from curobo.viewer import ViserVisualizer

    # ── Viser + 로봇 ───────────────────────────────────────────────────────
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=YML),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    # ── MotionPlanner (joint-goal plan_cspace, 장애물 회피) ─────────────────
    mp_kwargs = dict(robot=YML)
    if not args.no_scene:
        mp_kwargs["scene_model"] = SCENE
    config = MotionPlannerCfg.create(**mp_kwargs)
    planner = MotionPlanner(config)
    joint_names = planner.joint_names
    dof = len(joint_names)
    tool = planner.tool_frames[0]
    gripper_default = float(planner.default_joint_state.position.reshape(-1)[5].item())

    # 해석적 IK (position-우선, Isaac 불필요)
    kin = SO101Kinematics()

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

    current_state = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=joint_names
    )

    def update_obstacles():
        for k in obstacle_frames:
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def execute(traj):
        nonlocal current_state
        t = traj.squeeze(0)
        n = t.position.shape[-2]
        for i in range(n):
            wp = JointState.from_position(
                t.position[0, i, :].unsqueeze(0), joint_names=t.joint_names
            )
            viser_viz.set_joint_state(wp.squeeze(0))
            time.sleep(args.play_dt)
        current_state = JointState.from_position(
            t.position[0, -1, :].unsqueeze(0), joint_names=t.joint_names
        )

    is_moving = False

    def on_plan_move(_):
        nonlocal is_moving
        if is_moving:
            print("[skip] 이미 실행 중")
            return

        def work():
            nonlocal is_moving
            is_moving = True
            try:
                update_obstacles()
                # 1) gizmo target → 해석적 IK. position 은 그대로, orientation 은 yaw(world-z)만
                #    뽑아 grasp_yaw(손가락 닫힘축)로 매핑 → gizmo roll 돌리면 wrist_roll 반응.
                gp = {k.replace("target_", ""): v for k, v in viser_viz.get_control_frame_pose().items()}[tool]
                xyz = [float(v) for v in gp.position.reshape(3).tolist()]
                w, qx, qy, qz = (float(v) for v in gp.quaternion.reshape(4).tolist())  # wxyz
                gizmo_yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                grasp_yaw = args.grasp_yaw + gizmo_yaw
                sol = kin.ik_reach(
                    (xyz[0], xyz[1], xyz[2]), grasp_yaw=grasp_yaw,
                    pitch_min=math.radians(args.pitch_min_deg),
                    pitch_max=math.radians(args.pitch_max_deg),
                )
                if sol is None:
                    print(f"[IK] 도달불가 — pos={[round(v,3) for v in xyz]} (top-down reach 밖)")
                    return
                q5, pitch = sol
                ferr = math.dist(kin.fk_tcp(q5), xyz) * 1000.0
                print(f"[IK] OK pitch={math.degrees(pitch):.0f}°  grasp_yaw={math.degrees(grasp_yaw):.0f}°  "
                      f"fk_err={ferr:.2f}mm  q(deg)={[round(math.degrees(v)) for v in q5]}")

                # 2) plan_cspace (충돌-free joint 궤적)
                q_goal = torch.tensor([q5 + [gripper_default]], device="cuda", dtype=torch.float32)
                goal_state = JointState.from_position(q_goal, joint_names=joint_names)
                active = planner.kinematics.get_active_js(current_state.clone())
                res = planner.plan_cspace(goal_state, active, max_attempts=5)
                if res is None or not bool(res.success.any()):
                    print("[plan_cspace] 실패 — start/goal 충돌 또는 경로 없음")
                    return
                interp = res.get_interpolated_plan()
                nwp = interp.position.shape[-2]
                print(f"[plan_cspace] OK waypoints={nwp}  time={getattr(res,'total_time',0.0):.3f}s → 실행")
                execute(interp)
                print("[done]")
            except Exception:
                traceback.print_exc()
            finally:
                is_moving = False

        threading.Thread(target=work, daemon=True).start()

    server = viser_viz._server
    btn = server.gui.add_button("Plan & Move", color="green")
    btn.on_click(on_plan_move)

    print(f"\n[joint-goal] SO-101 Viser: http://localhost:{args.port}")
    print(f"[joint-goal] tool={tool}  joints={joint_names}  gripper_default={gripper_default:.3f}")
    print("[joint-goal] gizmo position 드래그 → 'Plan & Move' 클릭. orientation 무시(해석적 IK).")
    print("[joint-goal] 책상면 큐브존(낮은 z)이 정상 reach. 장애물 pillar 드래그 가능. Ctrl+C 종료.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
