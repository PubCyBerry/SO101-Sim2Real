# cuRobo v2 (0.8.0) — Public API Map

The 23 public shim modules (`curobo/*.py`) and the `_src` symbols they re-export.

Every `curobo/*.py` module is a thin shim that re-exports the real implementation from `curobo._src.**`. The `from ... import X as Y` aliasing is captured verbatim (Public Name = `Y`, `-> _src symbol` = full dotted path of the original).

---

## Package root

### `curobo/__init__.py`
cuRobo top-level package — provides accelerated robotics modules (numerical optimization, kinematics, geometry/collision, graph search) and high-level APIs (collision-free IK, MPC, motion planning). Only re-exports `__version__`.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `__version__` | const | `curobo._version.__version__` (via `curobo._src.util.version.get_version`) |

### `curobo/_version.py`
Version information for cuRobo package — computes `__version__` at import.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `__version__` | const | `curobo._src.util.version.get_version()` |

---

## Solvers (high-level entry points)

### `curobo/inverse_kinematics.py`
Collision-aware inverse kinematics solving with optimization.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `InverseKinematics` | class | `curobo._src.solver.solver_ik.IKSolver` |
| `InverseKinematicsCfg` | cfg | `curobo._src.solver.solver_ik_cfg.IKSolverCfg` |
| `InverseKinematicsResult` | result | `curobo._src.solver.solver_ik_result.IKSolverResult` |

### `curobo/trajectory_optimizer.py`
Collision-aware trajectory generation with optimization.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `TrajectoryOptimizer` | class | `curobo._src.solver.solver_trajopt.TrajOptSolver` |
| `TrajectoryOptimizerCfg` | cfg | `curobo._src.solver.solver_trajopt_cfg.TrajOptSolverCfg` |
| `TrajectoryOptimizerResult` | result | `curobo._src.solver.solver_trajopt_result.TrajOptSolverResult` |

### `curobo/motion_planner.py`
High-level motion planning combining trajectory optimization and graph planning.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `MotionPlanner` | class | `curobo._src.motion.motion_planner.MotionPlanner` |
| `MotionPlannerCfg` | cfg | `curobo._src.motion.motion_planner_cfg.MotionPlannerCfg` |
| `GraspPlanResult` | result | `curobo._src.motion.motion_planner_result.GraspPlanResult` |

### `curobo/batch_motion_planner.py`
Batch motion planning — solves multiple independent planning problems in parallel with a single IK + trajectory optimization pass.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `BatchMotionPlanner` | class | `curobo._src.motion.motion_planner_batch.BatchMotionPlanner` |
| `MotionPlannerCfg` | cfg | `curobo._src.motion.motion_planner_cfg.MotionPlannerCfg` |

### `curobo/model_predictive_control.py`
Model predictive control for real-time trajectory tracking with warm-start optimization and obstacle avoidance.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `ModelPredictiveControl` | class | `curobo._src.solver.solver_mpc.MPCSolver` |
| `ModelPredictiveControlCfg` | cfg | `curobo._src.solver.solver_mpc_cfg.MPCSolverCfg` |
| `ModelPredictiveControlResult` | result | `curobo._src.solver.solver_mpc_result.MPCSolverResult` |

### `curobo/motion_retargeter.py`
IK- and MPC-based motion retargeting — produces joint trajectories from per-frame tool pose targets (e.g. humanoid retargeting).

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `MotionRetargeter` | class | `curobo._src.motion.motion_retargeter.MotionRetargeter` |
| `MotionRetargeterCfg` | cfg | `curobo._src.motion.motion_retargeter_cfg.MotionRetargeterCfg` |
| `RetargetResult` | result | `curobo._src.motion.motion_retargeter_result.RetargetResult` |
| `ToolPoseCriteria` | class | `curobo._src.cost.tool_pose_criteria.ToolPoseCriteria` |
| `GoalToolPose` | class | `curobo._src.types.tool_pose.GoalToolPose` |
| `SequenceGoalToolPose` | class | `curobo._src.types.sequence_tool_pose.SequenceGoalToolPose` |

---

## Kinematics & robot model

### `curobo/kinematics.py`
Differentiable forward kinematics with Jacobian, center-of-mass, and collision-sphere generation.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `Kinematics` | class | `curobo._src.robot.kinematics.kinematics.Kinematics` |
| `KinematicsCfg` | cfg | `curobo._src.robot.kinematics.kinematics_cfg.KinematicsCfg` |
| `KinematicsState` | result | `curobo._src.robot.kinematics.kinematics_state.KinematicsState` |

### `curobo/robot_builder.py`
Build cuRobo robot configurations from URDF (fit collision spheres, compute collision matrix, save YAML/XRDF); plus a collision debugger.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `RobotBuilder` | class | `curobo._src.robot.builder.builder_robot.RobotBuilder` |
| `RobotDebugger` | class | `curobo._src.robot.builder.debugger_robot.RobotDebugger` |

### `curobo/robot_parser.py`
Robot kinematic parsers — turn robot description files into link/joint data structures used by `Kinematics`.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `UrdfRobotParser` | class | `curobo._src.robot.parser.parser_urdf.UrdfRobotParser` |

---

## Scene, geometry & collision

### `curobo/scene.py`
Scene representation — build/manage scenes with cuboid, sphere, capsule, cylinder, mesh, and voxel-grid obstacles. Note `Scene` aliases the config type `SceneCfg`.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `Scene` | cfg | `curobo._src.geom.types.SceneCfg` |
| `SceneData` | class | `curobo._src.geom.data.data_scene.SceneData` |
| `Obstacle` | class | `curobo._src.geom.types.Obstacle` |
| `Cuboid` | class | `curobo._src.geom.types.Cuboid` |
| `Sphere` | class | `curobo._src.geom.types.Sphere` |
| `Capsule` | class | `curobo._src.geom.types.Capsule` |
| `Cylinder` | class | `curobo._src.geom.types.Cylinder` |
| `Mesh` | class | `curobo._src.geom.types.Mesh` |
| `VoxelGrid` | class | `curobo._src.geom.types.VoxelGrid` |

### `curobo/collision_checking.py`
Robot-scene collision checking for custom (differentiable) pipelines outside the main solvers.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `RobotCollisionChecker` | class | `curobo._src.collision.collision_robot_scene.RobotSceneCollision` |
| `RobotCollisionCheckerCfg` | cfg | `curobo._src.collision.collision_robot_scene_cfg.RobotSceneCollisionCfg` |

### `curobo/sphere_fit.py`
Fit sphere approximations to triangle meshes for fast GPU collision checking, and inspect fit quality.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `fit_spheres_to_mesh` | func | `curobo._src.geom.sphere_fit.fit_spheres.fit_spheres_to_mesh` |
| `estimate_sphere_count` | func | `curobo._src.geom.sphere_fit.sphere_count.estimate_sphere_count` |
| `SphereFitMetrics` | class | `curobo._src.geom.sphere_fit.types.SphereFitMetrics` |
| `SphereFitResult` | result | `curobo._src.geom.sphere_fit.types.SphereFitResult` |
| `SphereFitType` | class | `curobo._src.geom.sphere_fit.types.SphereFitType` |

---

## Optimizers & rollouts (low-level)

### `curobo/optim.py`
Public surface for cuRobo's underlying optimizers (MPPI, evolution strategies, L-BFGS, PyTorch, SciPy) and multi-stage chaining; for custom optimization pipelines.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `MPPI` | class | `curobo._src.optim.particle.mppi.MPPI` |
| `MPPICfg` | cfg | `curobo._src.optim.particle.mppi.MPPICfg` |
| `EvolutionStrategies` | class | `curobo._src.optim.particle.evolution_strategies.EvolutionStrategies` |
| `EvolutionStrategiesCfg` | cfg | `curobo._src.optim.particle.evolution_strategies.EvolutionStrategiesCfg` |
| `LBFGSOpt` | class | `curobo._src.optim.gradient.lbfgs.LBFGSOpt` |
| `LBFGSOptCfg` | cfg | `curobo._src.optim.gradient.lbfgs.LBFGSOptCfg` |
| `MultiStageOptimizer` | class | `curobo._src.optim.multi_stage_optimizer.MultiStageOptimizer` |
| `ScipyOpt` | class | `curobo._src.optim.external.scipy_opt.ScipyOpt` |
| `ScipyOptCfg` | cfg | `curobo._src.optim.external.scipy_opt.ScipyOptCfg` |
| `TorchOpt` | class | `curobo._src.optim.external.torch_opt.TorchOpt` |
| `TorchOptCfg` | cfg | `curobo._src.optim.external.torch_opt.TorchOptCfg` |

### `curobo/rollout.py`
Rollouts define the cost/dynamics an optimizer minimizes. Exposes the Rosenbrock test rollout for validating custom optimizer configs.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `RosenbrockRollout` | class | `curobo._src.rollout.rollout_rosenbrock.RosenbrockRollout` |
| `RosenbrockCfg` | cfg | `curobo._src.rollout.rollout_rosenbrock.RosenbrockCfg` |

---

## Perception

### `curobo/perception.py`
Sensor processing for robot perception — robot segmentation, volumetric mapping from depth, and robot pose estimation.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `RobotSegmenter` | class | `curobo._src.perception.robot_segmenter.RobotSegmenter` |
| `FilterDepth` | class | `curobo._src.perception.filter_depth.FilterDepth` |
| `Mapper` | class | `curobo._src.perception.mapper.Mapper` |
| `MapperCfg` | cfg | `curobo._src.perception.mapper.MapperCfg` |
| `RobotMesh` | class | `curobo._src.perception.pose_estimation.mesh_robot.RobotMesh` |
| `PoseDetector` | class | `curobo._src.perception.pose_estimation.pose_detector.PoseDetector` |
| `DetectorCfg` | cfg | `curobo._src.perception.pose_estimation.pose_detector_cfg.DetectorCfg` |
| `SDFPoseDetector` | class | `curobo._src.perception.pose_estimation.sdf_pose_detector.SDFPoseDetector` |
| `SDFDetectorCfg` | cfg | `curobo._src.perception.pose_estimation.sdf_pose_detector_cfg.SDFDetectorCfg` |

---

## Common data types

### `curobo/types.py`
Common data types used throughout cuRobo — robot/joint states, poses, tool poses, camera observations, content paths, tensor device config.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `JointState` | class | `curobo._src.state.state_joint.JointState` |
| `RobotState` | class | `curobo._src.state.state_robot.RobotState` |
| `Pose` | class | `curobo._src.types.pose.Pose` |
| `ToolPose` | class | `curobo._src.types.tool_pose.ToolPose` |
| `GoalToolPose` | class | `curobo._src.types.tool_pose.GoalToolPose` |
| `CameraObservation` | class | `curobo._src.types.camera.CameraObservation` |
| `ContentPath` | class | `curobo._src.types.content_path.ContentPath` |
| `DeviceCfg` | cfg | `curobo._src.types.device_cfg.DeviceCfg` |

---

## Visualization

### `curobo/viewer.py`
Visualization backends (Viser web viewer, USD writer). Both are lazy factory functions raising `ImportError` if the optional dep (`viser` / `usd-core`) is missing — NOT direct class re-exports.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `ViserVisualizer` | func | lazily wraps `curobo._src.util.viser_visualizer.ViserVisualizer` |
| `UsdWriter` | func | lazily wraps `curobo._src.util.usd_writer.UsdWriter` |

---

## Flat-name re-export modules (runtime / IO / logging)

### `curobo/runtime.py`
Compile-time and runtime flags (CUDA graph usage, debug switches, cache dir, torch-compile options). All re-exported flat from `curobo._src.runtime`:

`cache_dir`, `cuda_core_backend`, `cuda_event_timers`, `cuda_graph_reset`, `cuda_graphs`, `cuda_streams`, `debug`, `debug_cuda_compile`, `debug_cuda_graphs`, `debug_nan`, `debug_timers`, `debug_trajopt`, `kernel_backend`, `profiler`, `torch_compile`, `torch_compile_slow`, `torch_jit` — all `-> curobo._src.runtime.<name>` (const/flag).

### `curobo/config_io.py`
File utilities. All re-exported flat from `curobo._src.util.config_io`:

`copy_file_to_path`, `file_exists`, `get_filename`, `get_files_from_dir`, `get_path_of_dir`, `is_file_xrdf`, `join_path`, `load_yaml`, `merge_dict_a_into_b`, `resolve_config`, `write_yaml` — all `-> curobo._src.util.config_io.<name>` (func).

### `curobo/logging.py`
Public logging helpers. All re-exported flat from `curobo._src.util.logging`:

`log_and_raise`, `log_debug`, `log_info`, `log_warn`, `setup_logger` — all `-> curobo._src.util.logging.<name>` (func).

### `curobo/profiling.py`
Lightweight GPU timing helpers; timers are no-ops (return `0.0`) when `curobo.runtime.cuda_event_timers` is disabled.

| Public Name | Kind | -> _src symbol |
|---|---|---|
| `CudaEventTimer` | class | `curobo._src.util.cuda_event_timer.CudaEventTimer` |
