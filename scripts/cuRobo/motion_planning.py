# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan collision-free trajectories and grasp motions with GPU-accelerated optimization.

.. raw:: html

    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/get_started_motion_plan.webm" type="video/webm"></video>

Motion planning finds a smooth, collision-free path from a start joint
configuration to a goal end-effector pose. cuRobo formulates this as a
trajectory optimization problem: it parameterizes the path as a sequence of
knot points and jointly optimizes them for smoothness (velocity, acceleration,
jerk costs) and feasibility (collision avoidance, joint limits). Multiple
trajectory seeds are optimized in parallel on the GPU, and the best
collision-free result is returned.

cuRobo also supports **grasp planning**, which chains three trajectory segments
-- approach, grasp, and lift -- into a single call. The planner first solves
for a collision-free path to a pre-grasp pose (offset along the approach axis),
then plans the final approach to the grasp pose with finger-link collisions
disabled, and finally plans a lift motion away from the surface.

By the end of this tutorial you will have:

- Initialized a GPU-accelerated motion planner for the Franka Panda robot
- Defined a box obstacle in the planning scene
- Planned a collision-free trajectory from a start configuration to a goal pose
- Planned a three-phase grasp motion (approach, grasp, lift)
- Saved trajectory plots as PDF files

Step 1: Run the tutorial
--------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning

This runs both the pose-to-pose planner and the grasp planner in sequence.
To run only one mode:

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning --mode pose
   python -m curobo.examples.getting_started.motion_planning --mode grasp

Step 2: Check the output
--------------------------

When the tutorial finishes successfully you will see::

    === Pose-to-Pose Motion Planning ===
    ✓ Planning succeeded!
    Trajectory has 250 waypoints
    Duration: 5.00s
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/motion_plan.pdf

    === Grasp Planning ===
    ✓ Grasp planning succeeded!
      Approach: 120 waypoints
      Grasp:    45 waypoints
      Lift:     60 waypoints
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/grasp_plan.pdf

Output files are written to ``~/.cache/curobo/`` by default. Override with
``--output-dir`` or by setting ``curobo._src.runtime.cache_dir`` in Python.

Step 3: Understand the pipeline
---------------------------------

**Pose-to-pose planning** walks through five stages:

1. **Initialize MotionPlanner**: Load the robot config (``franka.yml``) and a
   scene (``collision_test.yml``). The ``warmup()`` call pre-compiles CUDA
   graphs for faster subsequent planning calls.

2. **Set start and goal**: The start is a ``JointState`` (joint angles in
   radians), and the goal is a ``Pose`` (position + quaternion for the
   end-effector). Obstacle poses use the format
   ``[x, y, z, qw, qx, qy, qz]`` (meters, wxyz quaternion).

3. **Plan**: ``plan_pose`` runs IK to find a goal configuration, then
   optimizes a trajectory across multiple seeds. It returns a result with
   ``success``, the optimized trajectory, and an interpolated version.

4. **Save**: The interpolated joint trajectory is plotted with
   ``matplotlib`` and saved as a PDF.

**Grasp planning** extends this with ``plan_grasp``, which chains three
trajectory segments:

1. **Approach**: Plan from the current configuration to a pre-grasp pose,
   offset along ``grasp_approach_axis`` by ``grasp_approach_offset`` (default
   -15 cm along the tool Z axis). Finger-link collisions are disabled so the
   gripper can reach into tight spaces.

2. **Grasp**: Plan the final approach from the pre-grasp pose to the grasp
   pose itself.

3. **Lift**: Plan a retraction from the grasp pose, offset along
   ``grasp_lift_axis`` by ``grasp_lift_offset``.

The planner accepts a *goal set* of candidate grasp poses via ``ToolPose``,
selects the most reachable one, and returns all three trajectory segments in
a ``GraspPlanResult``.

Step 4: Interactive motion planning with Viser
------------------------------------------------

For an interactive version with a web-based 3D viewer, run:

.. code-block:: bash

   python -m curobo.examples.getting_started.motion_planning --visualize

Open http://localhost:8080 in your browser. Drag the target frame to set
the goal pose, move obstacles, and click "Move" to plan and execute.
Click "Grasp" to plan and execute a three-phase grasp motion.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

import curobo.runtime as runtime
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
import time

# ── SO-101(5축) 대응 ──────────────────────────────────────────────────────────
# SO-101 = shoulder_pan(수직축) + 평행 3축(planar 3R) + wrist_roll = 5-DOF.
# 임의 6-DOF pose 못 만족 → 예제 기본 full-pose 는 IK 수렴 실패("Motion planning
# failed"). tool pose criteria 로 orientation 자유도를 완화하면 풀린다.
#   full      = 6-DOF pose (franka 기본).  5축이면 대부분 실패.
#   position  = 위치만 추적 [1,1,1,0,0,0].  항상 풀림(자세 무시).
#   yaw-free  = 위치+roll+pitch, yaw(수직축=base-Z) 자유 [1,1,1,1,1,0]. 5축 상한.
# 오차축은 base 프레임(커널 q_delta=q_cur⊗q_goal⁻¹) → index5=base-Z=world yaw.
SO101_ROBOT = "/workspace/assets/robots/so101.yml"  # compose ../assets 마운트 경로
POSE_MODES = ("full", "position", "yaw-free")
# headless 예제 기본 goal (base frame). rest EE(moving_jaw)≈(0.317,0.019,0.255) 오프라인
# FK 근거 상수 — SO-101 reach 안. viser 는 이 상수 안 쓰고 드래그 타깃 사용.
DEFAULT_GOAL = (0.30, 0.0, 0.20)


def _apply_pose_mode(planner, mode):
    """plan_pose 의 orientation 제약 설정. warmup 후 호출(in-place, CUDA-graph 안전).
    franka 예제와 코드 경로는 동일 — 5축은 이 criteria 로 orientation 자유도만 완화."""
    if mode == "full":
        return
    if mode == "position":
        crit = ToolPoseCriteria.track_position()
    elif mode == "yaw-free":
        crit = ToolPoseCriteria.track_position_and_orientation(rpy=[1.0, 1.0, 0.0])
    else:
        raise ValueError(f"unknown pose mode: {mode!r} (choose from {POSE_MODES})")
    planner.update_tool_pose_criteria({tf: crit for tf in planner.tool_frames})


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    save_path: str,
    title: str = "Joint Trajectory",
    phase_boundaries: Optional[List[int]] = None,
    phase_labels: Optional[List[str]] = None,
):
    """Plot joint positions over time and save to *save_path*.

    Args:
        positions: Joint positions tensor of shape ``(timesteps, n_joints)``.
        joint_names: Label for each joint.
        dt: Time step between waypoints (seconds).
        save_path: Output file path (e.g. ``"trajectory.pdf"``).
        title: Plot title.
        phase_boundaries: Timestep indices where a new phase starts.
        phase_labels: Label for each phase (length must match *phase_boundaries*).
    """
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    n_steps = pos_np.shape[0]
    t = [i * dt for i in range(n_steps)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)

    if phase_boundaries and phase_labels:
        for idx, label in zip(phase_boundaries, phase_labels):
            ax.axvline(x=idx * dt, color="grey", linestyle="--", linewidth=0.8)
            ax.text(
                idx * dt, ax.get_ylim()[1], f" {label}", fontsize=8, va="top",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def pose_planning_example(robot=SO101_ROBOT, scene="collision_test.yml", mode="position",
                          goal_xyz=DEFAULT_GOAL, output_dir=None):
    """Plan a collision-free trajectory to a goal pose.

    Args:
        robot: Robot config file.
        scene: Scene collision model.
        mode: Pose-tracking mode (see POSE_MODES). SO-101 needs position/yaw-free.
        goal_xyz: Cartesian goal position (base frame). Orientation = identity.
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.

    Returns:
        True if planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _apply_pose_mode(planner, mode)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    gx, gy, gz = goal_xyz
    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[gx, gy, gz]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )
    print(f"[mode={mode}] goal=({gx:.3f},{gy:.3f},{gz:.3f}) quat=identity  robot={Path(robot).name}")

    result = planner.plan_pose(goal_pose, q_start)

    interp_dt = planner.trajopt_solver.config.interpolation_dt
    if result is not None and result.success.any():
        print("✓ Planning succeeded!")
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        print(f"Trajectory has {n_waypoints} waypoints")
        print(f"Duration: {n_waypoints * interp_dt:.2f}s")

        _plot_trajectory(
            interpolated.position.squeeze(0),
            planner.joint_names,
            dt=interp_dt,
            save_path=str(output_dir / "motion_plan.pdf"),
            title="Pose-to-Pose Trajectory",
        )
        return True
    else:
        print("✗ Planning failed - try adjusting the goal or obstacles")
        return False


def grasp_planning_example(robot=SO101_ROBOT, scene="collision_test.yml", mode="position",
                           goal_xyz=DEFAULT_GOAL, output_dir=None):
    """Plan a three-phase grasp motion (approach, grasp, lift).

    주의: plan_grasp 는 grasp phase 를 내부적으로 full-pose criteria 로 세팅하므로
    (motion_planner.py) 5축 SO-101 에선 mode 와 무관하게 grasp phase 가 실패할 수 있다.
    개념검증 핵심은 plan_pose(Move) 쪽 — 여기선 로딩·approach 까지만 최소 확인.

    Returns:
        True if grasp planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        max_goalset=10,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _apply_pose_mode(planner, mode)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    # goal 위치 주변에 후보 3개
    cx, cy, cz = goal_xyz
    n_grasps = 3
    positions = torch.zeros(1, 1, 1, n_grasps, 3, device="cuda", dtype=torch.float32)
    positions[..., 0] = cx
    positions[0, 0, 0, :, 1] = cy + torch.linspace(-0.04, 0.04, n_grasps)
    positions[..., 2] = cz

    quaternions = torch.zeros(1, 1, 1, n_grasps, 4, device="cuda", dtype=torch.float32)
    quaternions[..., 0] = 1.0

    grasp_poses = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=positions,
        quaternion=quaternions,
    )

    result = planner.plan_grasp(
        current_state=q_start,
        grasp_poses=grasp_poses,
        grasp_approach_offset=0.1,
        grasp_lift_offset=0.1,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
        grasp_lift_in_tool_frame=True,
    )

    if result.success is not None and result.success.any():
        print("✓ Grasp planning succeeded!")

        approach = result.approach_interpolated_trajectory
        grasp = result.grasp_interpolated_trajectory
        lift = result.lift_interpolated_trajectory

        if approach is not None:
            print(f"  Approach: {approach.position.shape[-2]} waypoints")
        if grasp is not None:
            print(f"  Grasp:    {grasp.position.shape[-2]} waypoints")
        if lift is not None:
            print(f"  Lift:     {lift.position.shape[-2]} waypoints")

        all_positions = []
        phase_boundaries = []
        phase_labels = []
        offset = 0
        n_joints = approach.position.shape[-1] if approach is not None else None
        for traj, label in [
            (approach, "Approach"),
            (grasp, "Grasp"),
            (lift, "Lift"),
        ]:
            if traj is not None:
                phase_boundaries.append(offset)
                phase_labels.append(label)
                pos = traj.position.reshape(-1, traj.position.shape[-1])
                all_positions.append(pos)
                offset += pos.shape[0]

        if all_positions:
            combined = torch.cat(all_positions, dim=0)
            _plot_trajectory(
                combined,
                planner.joint_names,
                dt=0.02,
                save_path=str(output_dir / "grasp_plan.pdf"),
                title="Grasp Trajectory (Approach → Grasp → Lift)",
                phase_boundaries=phase_boundaries,
                phase_labels=phase_labels,
            )
        return True
    else:
        status = getattr(result, "status", "unknown")
        print(f"✗ Grasp planning failed: {status}")
        return False


def interactive_motion_planning(robot_file=SO101_ROBOT, scene_file="collision_test.yml", port=8080, mode="position"):
    """Launch an interactive Viser viewer for motion planning.

    Provides a web-based 3D viewer where you can:
    - Drag the target frame to set the goal pose
    - Drag obstacles to reposition them
    - Click "Move" to plan and execute a collision-free trajectory
    - Click "Grasp" to plan a three-phase grasp motion (approach, grasp, lift)
    """
    import threading

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    config = MotionPlannerCfg.create(robot=robot_file, scene_model=scene_file)
    planner = MotionPlanner(config)

    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    current_state = planner.default_joint_state.clone().unsqueeze(0)

    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _apply_pose_mode(planner, mode)
    print(f"[pose-mode={mode}] Move/Grasp 는 이 criteria 로 plan (5축이면 position/yaw-free)")

    is_moving = False

    def _create_trajectory_image(trajectory, joint_names, title=""):
        """Render a joint trajectory as a PNG image array for the Viser GUI."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

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
        from PIL import Image
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        buf.close()
        return img_array

    server = viser_viz._server
    traj_plot = server.gui.add_image(
        _create_trajectory_image(
            JointState.from_position(
                planner.default_joint_state.position.unsqueeze(0).unsqueeze(0),
                joint_names=planner.joint_names,
            ),
            planner.joint_names,
            title="No trajectory yet",
        ),
        label="Joint Trajectory",
        format="png",
    )

    def update_obstacles():
        for k in obstacle_frames.keys():
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                traj_plot.image = _create_trajectory_image(
                    interp, planner.joint_names,
                    title=f"Pose Plan  |  {result.total_time:.3f}s",
                )
                execute_trajectory(interp)
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_grasp_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            grasp_poses = GoalToolPose.from_poses(target_poses, num_goalset=1)
            result = planner.plan_grasp(
                grasp_poses,
                active_js,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                grasp_lift_in_tool_frame=True,
            )
            if result is not None and result.success is not None and result.success.any():
                traj_plot.image = _create_trajectory_image(
                    result.approach_interpolated_trajectory, planner.joint_names,
                    title="Approach",
                )
                execute_trajectory(result.approach_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.grasp_interpolated_trajectory, planner.joint_names,
                    title="Grasp",
                )
                execute_trajectory(result.grasp_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.lift_interpolated_trajectory, planner.joint_names,
                    title="Lift",
                )
                execute_trajectory(result.lift_interpolated_trajectory)
            else:
                print("Grasp planning failed")
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    print(f"\nInteractive Motion Planner running at http://localhost:{port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Drag obstacles to reposition them")
    print("  - Click 'Move' for pose-to-pose planning")
    print("  - Click 'Grasp' for approach-grasp-lift planning")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def test():
    """개념검증 self-check (컨테이너 GPU 필요): SO-101 은 position-only 로 plan_pose 성립."""
    robot = SO101_ROBOT
    assert pose_planning_example(robot=robot, mode="position"), (
        "SO-101 position-only plan_pose 실패 — 도달가능 goal 인데 안 되면 config/goal 점검"
    )
    print("✓ SO-101 position-only pose planning OK (5-DOF full-pose 는 --pose-mode full 로 재현)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Motion Planning with cuRobo (SO-101 5-DOF 기본)",
    )
    parser.add_argument(
        "--mode",
        choices=["pose", "grasp", "all"],
        default="pose",
        help="Which headless example to run (default: pose)",
    )
    parser.add_argument(
        "--pose-mode",
        choices=POSE_MODES,
        default="position",
        help="Orientation 제약 (default: position). SO-101 은 position/yaw-free, franka 는 full",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        default=list(DEFAULT_GOAL),
        metavar=("X", "Y", "Z"),
        help=f"headless Cartesian goal, base frame (default: {DEFAULT_GOAL})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run as self-test with assertions",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch interactive Viser viewer with Move and Grasp buttons",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=SO101_ROBOT,
        help=f"Robot config file (default: SO-101). franka 는 'franka.yml' + --pose-mode full",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="collision_test.yml",
        help="Scene config file (default: collision_test.yml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: ~/.cache/curobo/examples/motion_planning/)",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.visualize:
        interactive_motion_planning(
            robot_file=args.robot, scene_file=args.scene, port=args.port, mode=args.pose_mode,
        )
        return

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    goal = tuple(args.goal)
    if args.mode in ("pose", "all"):
        print("=== Pose-to-Pose Motion Planning ===")
        pose_planning_example(
            robot=args.robot, scene=args.scene, mode=args.pose_mode, goal_xyz=goal,
            output_dir=output_dir,
        )
        print()

    if args.mode in ("grasp", "all"):
        print("=== Grasp Planning ===")
        grasp_planning_example(
            robot=args.robot, scene=args.scene, mode=args.pose_mode, goal_xyz=goal,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
