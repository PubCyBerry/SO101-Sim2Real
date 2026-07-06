# cuRobo v2 — Examples

The 10 example scripts under `curobo/examples/`. Run each with `python -m curobo.examples.<dotted.module>`.

## getting_started/

### curobo/examples/getting_started/build_robot_model.py
Build a cuRobo robot config from a URDF (adds the collision spheres + collision matrix cuRobo needs beyond a raw URDF).
- Run: `python -m curobo.examples.getting_started.build_robot_model`
- API: `from curobo.robot_builder import RobotBuilder`; `builder.fit_collision_spheres(...)`, `builder.compute_collision_matrix(...)`, `builder.build()`.

### curobo/examples/getting_started/forward_kinematics.py
GPU forward kinematics with autodiff (joint angles → 6-DOF pose), single + batched.
- Run: `python -m curobo.examples.getting_started.forward_kinematics`
- API: `from curobo.kinematics import Kinematics, KinematicsCfg`; `from curobo.types import JointState`; `KinematicsCfg.from_robot_yaml_file("franka.yml")`, `kin.compute_kinematics(JointState.from_position(q, joint_names=...))`.

### curobo/examples/getting_started/inverse_kinematics.py
GPU IK with multi-seed optimization; single / batched (100 poses) / collision-free, runtime world updates, + 3 interactive Viser modes.
- Run: `python -m curobo.examples.getting_started.inverse_kinematics [--mode single|batch|collision_free|all] [--visualize] [--differential] [--reachability] [--robot franka.yml] [--port 8080] [--test]`
- API: `from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg`; `from curobo.scene import Cuboid, Scene`; `from curobo.types import ContentPath, GoalToolPose, Pose`; `from curobo.viewer import ViserVisualizer`. Calls: `InverseKinematicsCfg.create(robot=..., num_seeds=32, scene_model="collision_table.yml", self_collision_check=True, collision_cache={"cuboid":10}, max_batch_size=...)`, `ik.solve_pose(GoalToolPose.from_poses({ik.tool_frames[0]: Pose(...)}, num_goalset=1))`, `ik.update_world(Scene(cuboid=[...]))`, `ik.scene_collision_checker.update_obstacle_pose(...)`. Result: `.success`, `.js_solution.position`, `.position_error`.

### curobo/examples/getting_started/motion_planning.py
Collision-free trajectory + grasp motion planning.
- Run: `python -m curobo.examples.getting_started.motion_planning`
- API: `import curobo.runtime as runtime`; `from curobo.motion_planner import MotionPlanner, MotionPlannerCfg`; `from curobo.types import ContentPath, GoalToolPose, JointState, Pose`. Calls: `MotionPlannerCfg.create(robot=..., scene_model=...)`, `planner.plan_pose(goal_pose, q_start)`.

### curobo/examples/getting_started/reactive_control.py
Real-time moving-goal tracking via MPC.
- Run: `python -m curobo.examples.getting_started.reactive_control`
- API: `from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg`. Calls: `ModelPredictiveControlCfg.create(...)`, `mpc.setup(current_state)`, `mpc.update_goal_tool_poses(...)`, `mpc.optimize_action_sequence(current_state)`.

### curobo/examples/getting_started/humanoid_retargeting.py
SOMA adapter retargeting BVH motion-capture onto a Unitree G1 using cuRobo IK + MPC.
- Run: `python -m curobo.examples.getting_started.humanoid_retargeting`
- API: `from curobo.motion_retargeter import (MotionRetargeter, MotionRetargeterCfg, RetargetResult, SequenceGoalToolPose, ToolPoseCriteria)`; `from curobo.scene import Sphere`. Calls: `MotionRetargeterCfg.create(...)`, `retargeter.solve_frame(tool_pose)` / `solve_sequence(...)`.

### curobo/examples/getting_started/volumetric_mapping.py
Fuse RGB-D frames into a TSDF world model, compute ESDF for collision-aware planning.
- Run: `python -m curobo.examples.getting_started.volumetric_mapping`
- API: `from curobo.perception import FilterDepth, Mapper, MapperCfg`; `from curobo.profiling import CudaEventTimer`; `from curobo.scene import Cuboid, Mesh, SceneData`; `from curobo.types import CameraObservation, DeviceCfg, Pose`. Calls: `Mapper(MapperCfg(...))`, `FilterDepth(...)`, `mapper.integrate(observation)`, `mapper.compute_esdf()`.

## guides/

### curobo/examples/guides/custom_optimization.py
Use `RosenbrockRollout` with cuRobo optimizers on the canonical non-convex test problem.
- Run: `python -m curobo.examples.guides.custom_optimization`
- API: `from curobo.optim import (MPPI, EvolutionStrategies, EvolutionStrategiesCfg, LBFGSOpt, LBFGSOptCfg, MPPICfg, MultiStageOptimizer, ScipyOpt, ScipyOptCfg, TorchOpt, TorchOptCfg)`; `from curobo.rollout import RosenbrockCfg, RosenbrockRollout`. Calls: `optimizer.optimize(init_action)`.

## reference/

### curobo/examples/reference/robot_pose_calibration.py
Interactive Viser demo of `PoseDetector` (ICP) and `SDFPoseDetector` (mesh SDF) for robot pose calibration.
- Run: `python -m curobo.examples.reference.robot_pose_calibration`
- API: `from curobo.kinematics import Kinematics, KinematicsCfg`; `from curobo.perception import (DetectorCfg, PoseDetector, RobotMesh, SDFDetectorCfg, SDFPoseDetector)`; `from curobo.viewer import ViserVisualizer`.

### curobo/examples/reference/sphere_fit_comparison.py
Visualise different sphere-fitting methods in Viser (load URDF, fit spheres per collision link, compare).
- Run: `python -m curobo.examples.reference.sphere_fit_comparison`
- API: `from curobo.config_io import join_path, load_yaml`; `from curobo.robot_parser import UrdfRobotParser`; `from curobo.scene import Sphere`; `from curobo.sphere_fit import (SphereFitType, estimate_sphere_count, fit_spheres_to_mesh)`.
