# cuRobo v2 (0.8.0) — Source API Index

Exhaustive reference index for the **cuRobo v2 (0.8.0)** source vendored at
`ref_repos/curobo/`. Lists every public module, class, function, dataclass field
(the tunable "variables"), enum, example, and bundled config.

- **Version:** 0.8.0 (`curobo._version.__version__`)
- **Source:** `ref_repos/curobo/curobo/` (package `curobo`, impl under `curobo/_src/`)
- **License:** Apache-2.0 · NVIDIA
- **Research:** cuRoboV2 — https://arxiv.org/abs/2603.05493
- **This project's use:** batch motion-planning / collision-free IK backend for SO-101 datagen
  (see `docs/PINK_IK_PICKPLACE.md`, memory `curobo-so101-config-v08-validated`).

> ⚠️ **v2 is a full rewrite.** The public API is *completely different* from classic
> cuRobo / MotionGen (`MotionGen`, `MotionGenConfig.load_from_robot_config` no longer exist).
> Any prior cuRobo knowledge or config does **not** transfer — use this index, not memory.

---

## 1. Architecture (what changed in 0.8.0)

cuRobo v2 is a **flat, inheritance-free architecture**. Behaviour is **composed** from
standalone classes, not built by subclassing.

- **Two `typing.Protocol`s** define the seams:
  - `Optimizer` — `curobo._src.optim.optimizer_protocol.Optimizer`
  - `Rollout` — `curobo._src.rollout.rollout_protocol.Rollout`
- **Optimizer stack** (all standalone, satisfy `Optimizer`): MPPI, Evolution Strategies,
  Gradient Descent, L-BFGS, L-SR1, Conjugate Gradient, + external Scipy/Torch wrappers,
  chained by `MultiStageOptimizer`.
- **Solvers** (`IKSolver`, `TrajOptSolver`, `MPCSolver`) compose a `SolverCore` +
  optimizer stack + `RobotRollout` + `RobotCostManager` + `SceneCollision`.
- **CUDA graphs** are a **constructor parameter** (`use_cuda_graph=True`), not a mixin/subclass.
- **Impl reorganised** under `curobo/_src/` with clear module boundaries: `types`, `state`,
  `transition`, `robot`, `geom`, `collision`, `solver`, `motion`, `optim`, `rollout`, `cost`,
  `perception`, `graph_planner`, `curobolib`, `util`.
- **New tool-frame abstractions:** `Pose`, `GoalToolPose`, `SequenceGoalToolPose` — multi-tool
  targets go through `GoalToolPose.from_poses({tool_frame: Pose(...)}, ordered_tool_frames=...)`.
- **New `Mapper` subsystem** (`curobo._src.perception.mapper`): block-sparse TSDF with primitive
  stamping (`update_static_obstacles`) + ESDF extraction (`compute_esdf`); the resulting
  `VoxelGrid` feeds motion planning directly for voxel-based collision.
- **Batch-first:** `MotionPlanner` / `IKSolver` take `max_batch_size`, `multi_env`, `max_goalset`;
  batched grasp planning (`plan_grasp`) is first-class.

Scale: ~560 files, ~297 public classes, ~478 module-level functions under `_src/`.

---

## 2. Public API quick-start

```python
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Scene, Cuboid, Sphere, Mesh
from curobo.types import JointState, Pose, GoalToolPose

# Forward kinematics
kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
state = kin.compute_kinematics(JointState.from_position(q, joint_names=kin.joint_names))

# Inverse kinematics (multi-seed, GPU, batched, collision-aware)
ik = InverseKinematics(InverseKinematicsCfg.create(robot="franka.yml", num_seeds=32))
result = ik.solve_pose(GoalToolPose.from_poses({ik.tool_frames[0]: Pose(position=p, quaternion=q)}, num_goalset=1))
#   -> result.success, result.js_solution.position, result.position_error

# Motion planning
planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_table.yml"))
plan = planner.plan_pose(goal_pose, q_start)

# MPC (reactive, moving goal)
mpc = ModelPredictiveControl(ModelPredictiveControlCfg.create(robot="franka.yml"))
mpc.setup(current_state); mpc.update_goal_tool_poses(...); mpc.optimize_action_sequence(current_state)
```

**High-level entry points** (the 5 solvers + retargeter):

| Task | Public class | Config | Key call |
|---|---|---|---|
| Forward kinematics | `Kinematics` | `KinematicsCfg` | `compute_kinematics(js)` |
| Inverse kinematics | `InverseKinematics` | `InverseKinematicsCfg` | `solve_pose(GoalToolPose)` |
| Trajectory optimization | `TrajectoryOptimizer` | `TrajectoryOptimizerCfg` | `solve_pose` / `solve_cspace` |
| Motion planning | `MotionPlanner` / `BatchMotionPlanner` | `MotionPlannerCfg` | `plan_pose` / `plan_cspace` / `plan_grasp` |
| Model predictive control | `ModelPredictiveControl` | `ModelPredictiveControlCfg` | `optimize_action_sequence` |
| Motion retargeting | `MotionRetargeter` | `MotionRetargeterCfg` | `solve_frame` / `solve_sequence` |

Each `*Cfg` exposes a `.create(...)` classmethod whose kwargs are the primary tunable
"variables" (IK ~40, MotionPlanner ~57, TrajOpt ~64, MPC ~79) — fully enumerated in
[05 Solvers](cuRobo_v2_0.8.0_INDEX/05_solvers.md) and [06 Motion Planning](cuRobo_v2_0.8.0_INDEX/06_motion_planning.md).

---

## 3. Section index

| # | Section | Covers |
|---|---|---|
| 01 | [Public API map](cuRobo_v2_0.8.0_INDEX/01_public_api.md) | The 23 `curobo/*.py` shim modules → `_src` symbols |
| 02 | [Types, State & Transition](cuRobo_v2_0.8.0_INDEX/02_types_state.md) | `Pose`, `ToolPose`, `GoalToolPose`, `JointState`, `RobotState`, `RobotStateTransition`, `ContentPath`, `DeviceCfg` |
| 03 | [Robot Model](cuRobo_v2_0.8.0_INDEX/03_robot.md) | `Kinematics`, `RobotBuilder`, `UrdfRobotParser`, `Dynamics`, `KinematicsParams`, `KinematicsLoaderCfg` |
| 04 | [Geometry, Scene & Collision](cuRobo_v2_0.8.0_INDEX/04_geom_collision.md) | `Scene`/`SceneCfg`, obstacle types, sphere-fitting, `RobotSceneCollision`, `AttachmentManager` |
| 05 | [Solvers](cuRobo_v2_0.8.0_INDEX/05_solvers.md) | `IKSolver`/`TrajOptSolver`/`MPCSolver`/`SeedIKSolver` + **full `.create()` variable surface** |
| 06 | [Motion Planning & Retargeting](cuRobo_v2_0.8.0_INDEX/06_motion_planning.md) | `MotionPlanner`, `BatchMotionPlanner`, `MotionRetargeter`, `GraspPlanResult` |
| 07 | [Optimizers](cuRobo_v2_0.8.0_INDEX/07_optim.md) | `Optimizer` protocol, MPPI, ES, L-BFGS, L-SR1, CG, `MultiStageOptimizer`, samplers |
| 08 | [Rollout & Cost](cuRobo_v2_0.8.0_INDEX/08_rollout_cost.md) | `Rollout` protocol, `RobotRollout`, `RobotCostManager`, every cost term, `ToolPoseCriteria`, `PoseCostMetric` |
| 09 | [Perception](cuRobo_v2_0.8.0_INDEX/09_perception.md) | `Mapper` (TSDF/ESDF), `FilterDepth`, `RobotSegmenter`, `PoseDetector`/`SDFPoseDetector` |
| 10 | [Graph Planner](cuRobo_v2_0.8.0_INDEX/10_graph_planner.md) | `PRMGraphPlanner` + config, graph construction, node sampling, path search |
| 11 | [curobolib (CUDA ops)](cuRobo_v2_0.8.0_INDEX/11_curobolib.md) | `torch.autograd.Function` wrappers + cuda-core launch layer (Python-facing) |
| 12 | [Utilities & Runtime](cuRobo_v2_0.8.0_INDEX/12_util_runtime.md) | config I/O, logging, trajectory interp, samplers, CUDA-graph/stream, USD/Viser viz, `runtime` flags |
| 13 | [Examples](cuRobo_v2_0.8.0_INDEX/13_examples.md) | 10 example scripts: what each shows + how to run |
| 14 | [Content Configs & Assets](cuRobo_v2_0.8.0_INDEX/14_configs.md) | Bundled robot / scene / task YAML+XRDF configs |
| 15 | [Benchmarks & Docs Map](cuRobo_v2_0.8.0_INDEX/15_benchmarks_docs.md) | `benchmark/*.py` + `docs/*.rst` topic map |

---

## 4. Public modules (`curobo/*.py`)

Thin shims re-exporting from `curobo._src`. See [01 Public API](cuRobo_v2_0.8.0_INDEX/01_public_api.md) for the full symbol map.

| Module | Purpose |
|---|---|
| `kinematics` | Forward kinematics (`Kinematics`, `KinematicsCfg`, `KinematicsState`) |
| `inverse_kinematics` | IK solver (`InverseKinematics`, `InverseKinematicsCfg`, `InverseKinematicsResult`) |
| `trajectory_optimizer` | Trajectory optimization (`TrajectoryOptimizer`, `…Cfg`, `…Result`) |
| `motion_planner` | Motion planning (`MotionPlanner`, `MotionPlannerCfg`, `GraspPlanResult`) |
| `batch_motion_planner` | Batched motion planning (`BatchMotionPlanner`) |
| `model_predictive_control` | MPC (`ModelPredictiveControl`, `…Cfg`, `…Result`) |
| `motion_retargeter` | IK/MPC motion retargeting (`MotionRetargeter`, `SequenceGoalToolPose`, `ToolPoseCriteria`) |
| `scene` | Scene + obstacle types (`Scene`, `Cuboid`, `Sphere`, `Capsule`, `Cylinder`, `Mesh`, `VoxelGrid`) |
| `collision_checking` | Robot-scene collision for custom pipelines (`RobotCollisionChecker`) |
| `perception` | `Mapper`, `FilterDepth`, `RobotSegmenter`, pose detectors |
| `sphere_fit` | Mesh→sphere fitting (`fit_spheres_to_mesh`, `SphereFitType`) |
| `optim` | Optimizer surface (`MPPI`, `LBFGSOpt`, `EvolutionStrategies`, …) |
| `rollout` | Test rollout (`RosenbrockRollout`) |
| `robot_builder` | `RobotBuilder`, `RobotDebugger` |
| `robot_parser` | `UrdfRobotParser` |
| `types` | `JointState`, `RobotState`, `Pose`, `ToolPose`, `GoalToolPose`, `CameraObservation`, `ContentPath`, `DeviceCfg` |
| `viewer` | `ViserVisualizer`, `UsdWriter` (lazy) |
| `runtime` | Runtime flags (CUDA graphs, debug, cache, torch compile) |
| `config_io` | YAML/path utilities |
| `profiling` | `CudaEventTimer` |
| `logging` | `setup_logger`, `log_warn`/`log_info`/`log_debug`, `log_and_raise` |

---

## 5. Bundled configs (`curobo/content/configs/`)

See [14 Configs](cuRobo_v2_0.8.0_INDEX/14_configs.md) for details.

- **Robots** (`robot/`): `franka.yml`, `ur10e.yml` / `ur10e.xrdf`, `dual_ur10e.yml`,
  `simple_mimic_robot.yml`, `unitree_g1.yml`, `unitree_g1_29dof_retarget.yml`
- **Scenes** (`scene/`): `collision_base_stand.yml`, `collision_primitives_3d.yml`,
  `collision_table.yml`, `collision_test.yml`
- **Tasks** (`task/`): `metrics_base.yml`; `ik/{particle,lbfgs,lbfgs_retarget,transition}_ik.yml`;
  `trajopt/{particle,lbfgs_bspline,transition_bspline}_trajopt.yml`;
  `mpc/{lbfgs,lbfgs_retarget,transition_bspline}_mpc.yml`;
  `graph_planner/{exact,transition}_graph_planner.yml`

---

## 6. Enum quick-index

| Enum | Members | Location |
|---|---|---|
| `SolveMode` | SINGLE, BATCH, MULTI_ENV | `_src/solver/solve_mode.py` |
| `ControlSpace` | POSITION(0), VELOCITY(1), ACCELERATION(2), BSPLINE_3/4/5 | `_src/types/control_space.py` |
| `JointType` | FIXED(-1), X/Y/Z_PRISM(0-2), X/Y/Z_ROT(3-5), + _NEG variants (6-11) | `_src/robot/types/joint_types.py` |
| `TrajInterpolationType` | LINEAR, CUBIC, QUARTIC, QUINTIC, LINEAR_CUDA, BSPLINE_KNOTS_CUDA | `_src/util/trajectory.py` |
| `LineSearchType` | GREEDY, ARMIJO, WOLFE, STRONG_WOLFE, APPROX_WOLFE, APPROX_STRONG_WOLFE | `_src/optim/gradient/line_search_strategy.py` |
| `SquashType` | CLAMP(0), CLAMP_RESCALE(1), TANH(2), IDENTITY(3) | `_src/optim/particle/particle_opt_utils.py` |
| `SampleMode` | MEAN, BEST, SAMPLE | `_src/optim/components/particle_opt_core.py` |
| `CovType` | SIGMA_I, DIAG_A | `_src/optim/components/gaussian_distribution.py` |
| `BaseActionType` | REPEAT, NULL, RANDOM | `_src/optim/particle/mppi.py` |
| `PoseErrorType` | SINGLE_GOAL(0), BATCH_GOAL(1), GOALSET(2), BATCH_GOALSET(3) | `_src/cost/cost_pose_type.py` |
| `CSpaceCostType` | POSITION(0), STATE(1) | `_src/cost/cost_cspace_type.py` |
| `SphereFitType` | SURFACE, VOXEL, MORPHIT | `_src/geom/sphere_fit/types.py` |

---

## 7. SO-101 relevance notes

- SO-101 is a **5-DOF arm** → arbitrary 6-DOF pose targets are not always solvable. Prefer
  **position-only / best-effort orientation** goals (see project 5-DOF IK principle).
- The project's validated SO-101 cuRobo config lives at `assets/robots/so101.yml`
  (memory `curobo-so101-config-v08-validated`); `RobotBuilder` (section 03) is how it was authored.
- Install pins for the datagen image (torch 2.7 / curobo 0.8 / warp coexistence) are in
  memory `curobo-v08-docker-install-recipe` — not repeated here.
