"""SO-101 reactive MPC 검증 Viser (PICKCUBE_CUROBO P1 검증).

cuRobo 공식 `reactive_control.py` 의 interactive MPC 예제를 SO-101 (우리 so101.xrdf + urdf)로 적용.
Viser(:port) 에서 EE 기즈모를 끌면 SO-101 이 MPC 로 실시간 추종 + 장애물 회피 → P1 robot/충돌 모델을
동적 검증한다. (Isaac 불필요 — 순수 cuRobo + Viser.)

실행:
    uv run --no-sync --group isaac python scripts/sim/reactive_so101_viser.py --port 8086
    uv run --no-sync --group isaac python scripts/sim/reactive_so101_viser.py --port 8086 --no_scene
"""

from __future__ import annotations

import argparse
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
YML = os.path.join(ROOT, "assets/robots/so101_curobo.yml")  # cuRobo native(mesh_link_names 포함)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--no_scene", action="store_true", help="장애물 씬 없이(robot 만)")
    args = ap.parse_args()

    import torch

    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import ContentPath, GoalToolPose, JointState, Pose
    from curobo.viewer import ViserVisualizer

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=YML),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=True,   # 충돌 sphere 도 같이 표시(P1 검증)
        add_robot_to_scene=True,
    )

    cfg_kwargs = dict(
        robot=YML,
        use_cuda_graph=True,
        optimization_dt=0.03,
        interpolation_steps=4,
        optimizer_collision_activation_distance=0.03,
    )
    if not args.no_scene:
        cfg_kwargs["scene_model"] = "collision_test.yml"
    config = ModelPredictiveControlCfg.create(**cfg_kwargs)

    obstacle_frames = {}
    old_obstacle_poses = {}
    if not args.no_scene and config.scene_collision_cfg is not None:
        scene_cfg = config.scene_collision_cfg.scene_model
        obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
        old_obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames
        }

    mpc = ModelPredictiveControl(config)

    current_state = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names
    )
    current_state.velocity = torch.zeros_like(current_state.position)
    current_state.acceleration = torch.zeros_like(current_state.position)
    mpc.setup(current_state)

    kin_result = mpc.compute_kinematics(current_state)
    mpc.update_goal_tool_poses(
        GoalToolPose.from_poses(
            kin_result.tool_poses.to_dict(), ordered_tool_frames=mpc.tool_frames, num_goalset=1
        ),
        run_ik=False,
    )

    print(f"\n[reactive] SO-101 MPC Viser: http://localhost:{args.port}")
    print(f"[reactive] tool_frames={mpc.tool_frames}  joints={mpc.joint_names}")
    print("[reactive] EE 기즈모 드래그 → 추종. Ctrl+C 종료.\n")

    previous_target_poses = None
    pose_changed = False
    while True:
        if obstacle_frames:
            obstacle_poses = {
                k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
                for k in obstacle_frames
            }
            for k in obstacle_poses:
                if obstacle_poses[k] != old_obstacle_poses[k]:
                    mpc.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                    pose_changed = True
            old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = target_poses
        else:
            for fn in target_poses:
                if target_poses[fn] != previous_target_poses[fn]:
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            tlp = {k.replace("target_", ""): v for k, v in target_poses.items()}
            mpc.update_goal_tool_poses(
                GoalToolPose.from_poses(tlp, ordered_tool_frames=mpc.tool_frames, num_goalset=1),
                run_ik=False,
            )
            pose_changed = False

        res = mpc.optimize_action_sequence(current_state)
        if res.action_sequence is not None and res.action_sequence.position.shape[1] > 0:
            current_state = JointState.from_position(
                res.action_sequence.position[:, -1, :].clone(), joint_names=mpc.joint_names
            )
            current_state.velocity = res.action_sequence.velocity[:, -1, :]
            current_state.acceleration = res.action_sequence.acceleration[:, -1, :]
            viser_viz.set_joint_state(current_state.squeeze(0))
        time.sleep(0.001)


if __name__ == "__main__":
    raise SystemExit(main())
