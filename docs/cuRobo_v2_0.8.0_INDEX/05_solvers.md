# cuRobo v2 — Solvers (IK, TrajOpt, MPC, SeedIK)

The IK / trajectory-optimization / MPC solvers and their full config surfaces — the heart of the tunable variables.

---

## Shared "Standard Solver" surface (documented once)

`IKSolver`, `TrajOptSolver`, and `MPCSolver` are **composition wrappers** over a single `SolverCore` instance (`self.core = SolverCore(config.core_cfg, ...)`). Each re-exposes the same delegated read-only properties and pass-through methods. These are listed here once; per-solver sections list only the *extra* methods.

**Delegated properties** (all three solvers): `optimizer`, `metrics_rollout`, `auxiliary_rollout`, `kinematics`, `transition_model`, `action_dim`, `action_horizon`, `joint_names`, `tool_frames`, `default_joint_position`, `default_joint_state`, `device_cfg`, `scene_collision_checker`, `goal_registry_manager`, `solve_state`, `seed_manager`. (TrajOpt adds `additional_metrics_rollouts`, `optimizer_rollouts`, `horizon`, `opt_dim`, `interpolation_steps`; MPC adds `additional_metrics_rollouts`.)

**Delegated methods** (all three): `get_all_rollout_instances(**kwargs)`, `compute_kinematics(state)`, `get_active_js(full_js)`, `get_full_js(active_js)`, `sample_configs(num_samples, rejection_ratio=10)`, `update_pose_cost_metric(pose_cost_metric)`, `update_link_inertial(link_name, mass=None, com=None, inertia=None)`, `update_links_inertial(link_properties)`, `debug_dump(file_path)`, `prepare_action_seeds(...)`, `prepare_trajectory_seeds(...)`, `enable_tool_pose_tracking(tool_frames=None)`, `disable_tool_pose_tracking(tool_frames=None)`, `enable_joint_position_tracking()`, `disable_joint_position_tracking()`, `reset_shape()`, `reset_seed()`, `reset_cuda_graph()`, `destroy()`, `problem_batch_size` (property).

---

# IK Solver

### curobo/_src/solver/solver_ik.py
Inverse Kinematics solver for reaching target tool-frame poses. Wraps SolverCore with LM seed generation, multi-link goal handling, batch padding, and solution ranking by pose error. Supports goalset IK, velocity-aware IK via optimization_dt, optional collision checking.

- **IKSolver** — Inverse Kinematics solver (composition over `SolverCore` + optional `SeedIKSolver`).
  - `__init__(self, config: IKSolverCfg, scene_collision_checker: Optional[SceneCollision] = None)` — builds core; instantiates an LM `SeedIKSolver` when `config.use_lm_seed`. Multi-link tuning: >1 tool frame → seed_solver_num_seeds=128, max_iterations=20; >2 → 64, 30, tile_threads=256; `num_seeds > seed_solver_num_seeds` → doubles it. `override_iters_for_multi_link_ik` bumps lbfgs iterations.
  - `solve_pose(goal_tool_poses: GoalToolPose, current_state=None, seed_config=None, return_seeds: int = 1, run_optimizer: bool = True) -> IKSolverResult` — main entry. Pads batch to `max_batch_size`, runs LM seed solve then multi-stage optimizer, ranks solutions by cost, returns top `return_seeds`.
  - `get_unique_solution(roundoff_decimals: int = 2) -> torch.Tensor` — dedupe successful solutions to `[num_unique, dof]` by rounding.
  - `update_world(scene_cfg: SceneCfg) -> None` — reload collision model from a new scene.
  - `update_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria])` — updates both seed solver and core.
  - `reset_seed()`, `destroy()` — also reset/destroy the seed solver.
  - Extra properties: `problem_batch_size` (= `solve_state.get_ik_batch_size()`).
  - Plus shared Standard Solver surface (above).
- `_pad_batch_inputs(goal_tool_poses, current_state, seed_config, batch_size, max_batch)` — module helper: pad inputs to `max_batch` by repeating first element (reused by TrajOpt).
- `_slice_batch_result(result, batch_size)` — module helper: slice padded result tensors back to original batch (reused by TrajOpt).

### curobo/_src/solver/solver_ik_cfg.py
Configuration dataclass for the Inverse Kinematics solver. Bundles SolverCoreCfg with IK-specific tolerances, seed counts, velocity-aware IK settings, and the `create` factory.

- **IKSolverCfg** (@dataclass) — Configuration specific to the IK solver.
  - `core_cfg: SolverCoreCfg` (no default — required)
  - `robot_config: RobotCfg` (no default — required)
  - `max_batch_size: int = 1`
  - `multi_env: bool = False`
  - `max_goalset: int = 1`
  - `num_seeds: int = 32`
  - `position_tolerance: float = 0.005`
  - `orientation_tolerance: float = 0.05`
  - `optimizer_collision_activation_distance: float = 0.01`
  - `non_terminal_tool_pose_weight_factor: float = 0.0`
  - `success_requires_convergence: bool = True`
  - `override_iters_for_multi_link_ik: Optional[int] = None`
  - `use_lm_seed: bool = True`
  - `exit_early: bool = True`
  - `exit_early_batch_success_threshold: float = 1.0`
  - `optimization_dt: Optional[float] = None`
  - `seed_position_weight: float = 1.0`
  - `seed_orientation_weight: float = 1.0`
  - `seed_velocity_weight: float = 0.0`
  - `seed_acceleration_weight: float = 0.0`
  - `seed_solver_num_seeds: int = 32`
  - `self_collision_check: bool = True`
  - Convenience read-only properties into `core_cfg`: `device_cfg`, `use_cuda_graph`, `random_seed`, `store_debug`, `scene_collision_cfg`, `optimizer_configs`, `optimizer_rollout_configs`, `metrics_rollout_config`.
  - `@staticmethod create(...)` — full signature verbatim:
    - `robot: Union[str, Dict[str, Any], RobotCfg]`
    - `optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"]`
    - `metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml"`
    - `transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml"`
    - `scene_model: Optional[Union[str, Dict[str, Any]]] = None`
    - `collision_cache: Optional[Dict[str, int]] = None`
    - `self_collision_check: bool = True`
    - `device_cfg: DeviceCfg = DeviceCfg()`
    - `num_seeds: int = 32`
    - `position_tolerance: float = 0.005`
    - `orientation_tolerance: float = 0.05`
    - `use_cuda_graph: bool = True`
    - `random_seed: int = 123`
    - `optimizer_collision_activation_distance: float = 0.01`
    - `store_debug: bool = False`
    - `override_optimizer_num_iters: Dict[str, Optional[int]] = {"particle": None, "lbfgs": None}`
    - `transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg`
    - `cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg`
    - `override_iters_for_multi_link_ik: Optional[int] = None`
    - `optimization_dt: Optional[float] = None`
    - `load_collision_spheres: bool = True`
    - `velocity_regularization_weight: Optional[float] = None`
    - `acceleration_regularization_weight: Optional[float] = None`
    - `success_requires_convergence: bool = True`
    - `seed_position_weight: float = 1.0`
    - `seed_orientation_weight: float = 1.0`
    - `seed_velocity_weight: float = 0.0`
    - `seed_acceleration_weight: float = 0.0`
    - `seed_solver_num_seeds: int = 32`
    - `max_batch_size: int = 1`
    - `multi_env: bool = False`
    - `max_goalset: int = 1`
    - Returns `IKSolverCfg`.

### curobo/_src/solver/solver_ik_result.py
- **IKSolverResult**(BaseSolverResult) — Result specific to IK. Adds no new fields (inherits all `BaseSolverResult` fields; `pass`-body placeholder for future IK-specific fields).

---

# TrajOpt Solver

### curobo/_src/solver/solver_trajopt.py
Trajectory optimization solver for collision-free, time-optimal paths. Wraps SolverCore with multi-seed trajectory generation, iterative dt-finetuning, interpolation to dense waypoints, solution ranking. Supports Cartesian (`solve_pose`) and cspace (`solve_cspace`) goals.

- **TrajOptSolver** — Trajectory Optimization solver.
  - `__init__(self, config: TrajOptSolverCfg, scene_collision_checker=None)` — builds core; adds an `"interpolated_rollout"` to `core.additional_metrics_rollouts`; creates a `TrajectorySeedGenerator` and interpolation buffers.
  - `solve_pose(goal_tool_poses: GoalToolPose, current_state: JointState, seed_config=None, seed_traj=None, return_seeds: int = 1, num_seeds: Optional[int] = None, dt=None, use_implicit_goal: bool = False, finetune_attempts: int = 1, goal_state: Optional[JointState] = None, initial_iters: Optional[int] = None, time_optimal_iters: Optional[int] = None, finetune_iters: Optional[int] = None, finetune_dt_scale: float = 0.55) -> TrajOptSolverResult` — Cartesian-space time-optimal trajectory; multi-seed + iterative dt-finetuning.
  - `solve_cspace(goal_state: JointState, current_state: JointState, seed_traj=None, return_seeds: int = 1, num_seeds: Optional[int] = None, dt=None, finetune_attempts: int = 1, initial_iters=None, time_optimal_iters=None, finetune_iters=None) -> TrajOptSolverResult` — joint-to-joint trajectory; runs FK on goal to derive tool pose, calls `_solve_impl(use_implicit_goal=True)`.
  - `compute_trajectory_dt(trajectory: JointState, epsilon: float = 1e-3, scale_dt: bool = True) -> torch.Tensor` — compute time-optimal dt from vel/acc/jerk limits, clamped to `[minimum_trajectory_dt, maximum_trajectory_dt]`.
  - `get_interpolated_trajectory(js_optimized: JointState) -> (JointState, tstep, bool)` — dense interpolation into buffer (BSpline knots or linear).
  - Extra properties: `horizon`, `opt_dim` (= action_dim*action_horizon), `interpolation_steps`, `problem_batch_size` (= `solve_state.get_trajopt_batch_size()`).
  - Plus shared Standard Solver surface (above).

### curobo/_src/solver/solver_trajopt_cfg.py
Configuration dataclass for the trajectory optimization solver. Adds dt bounds, interpolation settings, and the `create` factory.

- **TrajOptSolverCfg** (@dataclass) — Configuration specific to the TrajOpt solver.
  - `core_cfg: SolverCoreCfg` (required)
  - `robot_config: RobotCfg` (required)
  - `max_batch_size: int = 1`
  - `multi_env: bool = False`
  - `max_goalset: int = 1`
  - `num_seeds: int = 4`
  - `position_tolerance: float = 0.005`
  - `orientation_tolerance: float = 0.05`
  - `optimizer_collision_activation_distance: float = 0.01`
  - `non_terminal_tool_pose_weight_factor: float = 0.0`
  - `self_collision_check: bool = True`
  - `minimum_trajectory_dt: float = 0.002`
  - `maximum_trajectory_dt: float = 0.2`
  - `interpolation_dt: float = 0.025`
  - `interpolation_type: TrajInterpolationType = TrajInterpolationType.BSPLINE_KNOTS_CUDA`
  - `interpolation_buffer_size: int = 5000`
  - Convenience read-only properties into `core_cfg` (same 8 as IKSolverCfg).
  - `@staticmethod create(...)` — full signature verbatim:
    - `robot: Union[str, Dict[str, Any], RobotCfg]`
    - `optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"]`
    - `metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml"`
    - `transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml"`
    - `scene_model: Optional[Union[str, Dict[str, Any]]] = None`
    - `collision_cache: Optional[Dict[str, int]] = None`
    - `self_collision_check: bool = True`
    - `device_cfg: DeviceCfg = DeviceCfg()`
    - `num_seeds: int = 4`
    - `position_tolerance: float = 0.005`
    - `orientation_tolerance: float = 0.05`
    - `use_cuda_graph: bool = True`
    - `random_seed: int = 123`
    - `optimizer_collision_activation_distance: float = 0.01`
    - `store_debug: bool = False`
    - `minimum_trajectory_dt: float = 0.002`
    - `maximum_trajectory_dt: float = 0.2`
    - `load_collision_spheres: bool = True`
    - `override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None}`
    - `transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg`
    - `cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg`
    - `max_batch_size: int = 1`
    - `multi_env: bool = False`
    - `max_goalset: int = 1`
    - Returns `TrajOptSolverCfg`. (Interpolation type auto-selected: `LINEAR_CUDA` if transition control_space is POSITION else `BSPLINE_KNOTS_CUDA`.)

### curobo/_src/solver/solver_trajopt_result.py
- **TrajOptSolverResult**(BaseSolverResult) — Result specific to TrajOpt. Extra fields on top of common `BaseSolverResult`:
  - `solution: Optional[torch.Tensor] = None` (shape `(batch, return_seeds, horizon, dof)`)
  - `js_solution: Optional[JointState] = None`
  - `interpolated_trajectory: Optional[JointState] = None`
  - `interpolated_last_tstep: Optional[torch.Tensor] = None`
  - `interpolated_metrics: Optional[RolloutMetrics] = None`
  - `maximum_trajectory_dt: Optional[torch.Tensor] = None`
  - `minimum_trajectory_dt: Optional[torch.Tensor] = None`
  - Key methods: `motion_time() -> torch.Tensor` (=(horizon-1)*dt); `get_interpolated_plan() -> JointState` (trims to last valid tstep); `process_metrics_and_rank_seeds()` (feasibility+convergence at last step, ranks by dt+jerk+acc smooth cost); `get_topk_seeds(topk) -> TrajOptSolverResult`; `copy_successful_solutions(other)`; `copy_at_batch_indices(other, mask)`; `clone()`.

---

# MPC Solver

### curobo/_src/solver/solver_mpc.py
MPCSolver — Model Predictive Control for reaching targets in Cartesian and Joint space. Owns an internal `IKSolver` (for goal retargeting) and a `TrajectoryExecutionManager`.

- **MPCSolver** — model predictive control solver.
  - `__init__(self, config: MPCSolverCfg, scene_collision_checker=None)` — builds core, `TrajectoryExecutionManager(interpolation_steps)`, and an internal `IKSolver` (`ik/lbfgs_retarget_ik.yml`, 1 seed); enables tool-pose tracking, disables joint-position tracking.
  - `setup(current_state: JointState, tool_frames: Optional[List[str]] = None, dt: Optional[torch.Tensor] = None) -> None` — required before solving; builds goal buffer, seeds trajectory, cold-start solve, resets robot.
  - `optimize_next_action(current_state: JointState) -> MPCSolverResult` — cold/warm-start as needed; returns single-step `next_action` + full `action_buffer`.
  - `optimize_action_sequence(current_state: JointState) -> MPCSolverResult` — always re-optimizes; returns full `action_sequence` over horizon.
  - `cold_start_solve(current_state)` / `warm_start_solve(current_state)` — full vs warm optimizer passes (`cold_start_optimization_num_iters` vs `warm_start_optimization_num_iters`).
  - `update_goal_tool_poses(goal_tool_poses, robot_ids=None, run_ik=True, use_ik_goal=True, use_best_effort_ik=False) -> bool` — set Cartesian goal, optionally IK-retarget to a feasible joint goal.
  - `update_goal_state(goal_state, robot_ids=None)`, `update_current_state(current_state)`, `update_seed_trajectory(seed_trajectory)`, `update_seed_trajectory_from_goal_state(goal_joint_state)`.
  - `set_default_goal_from_current_state(current_state, robot_ids=None)`, `reset_robot(current_state)`, `reset_robot_id(current_state, robot_ids)`.
  - `prepare_safe_deceleration_trajectory(current_state, failed_mask, deceleration_time=None, deceleration_profile=None) -> torch.Tensor` — safe stop fallback for infeasible robots.
  - `update_tool_pose_criteria(tool_pose_criteria)` — updates both internal `ik_solver` and core.
  - Extra property: `problem_batch_size` (= `solve_state.get_trajopt_batch_size()`).
  - Plus shared Standard Solver surface (above).

### curobo/_src/solver/solver_mpc_cfg.py
Configuration dataclass for the MPC solver. Adds cold/warm-start iters, interpolation steps, optimization_dt, deceleration, and the `create` factory.

- **MPCSolverCfg** (@dataclass) — Configuration specific to the MPC solver.
  - `core_cfg: SolverCoreCfg` (required)
  - `robot_config: RobotCfg` (required)
  - `max_batch_size: int = 1`
  - `multi_env: bool = False`
  - `max_goalset: int = 1`
  - `num_seeds: int = 1`
  - `position_tolerance: float = 0.005`
  - `orientation_tolerance: float = 0.05`
  - `optimizer_collision_activation_distance: float = 0.01`
  - `non_terminal_tool_pose_weight_factor: float = 0.001`  (note: nonzero, unlike IK/TrajOpt)
  - `self_collision_check: bool = True`
  - `interpolation_steps: int = 4`
  - `optimization_dt: float = 0.02`
  - `warm_start_optimization_num_iters: int = 200`
  - `cold_start_optimization_num_iters: int = 300`
  - `use_deceleration_on_failure: bool = True`
  - `deceleration_time: Optional[float] = None`
  - `deceleration_profile: str = "exponential"`
  - `max_deceleration_time: float = 2.0`
  - Convenience read-only properties into `core_cfg` (same 8 as IKSolverCfg).
  - `@staticmethod create(...)` — full signature verbatim:
    - `robot: Union[str, Dict[str, Any], RobotCfg]`
    - `optimizer_configs: List[Union[str, Dict[str, Any]]] = ["mpc/lbfgs_mpc.yml"]`
    - `metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml"`
    - `transition_model: Union[str, Dict[str, Any]] = "mpc/transition_bspline_mpc.yml"`
    - `scene_model: Optional[Union[str, Dict[str, Any]]] = None`
    - `collision_cache: Optional[Dict[str, int]] = None`
    - `self_collision_check: bool = True`
    - `device_cfg: DeviceCfg = DeviceCfg()`
    - `position_tolerance: float = 0.005`
    - `orientation_tolerance: float = 0.05`
    - `use_cuda_graph: bool = True`
    - `random_seed: int = 123`
    - `optimizer_collision_activation_distance: float = 0.01`
    - `store_debug: bool = False`
    - `override_optimizer_num_iters: Dict[str, Optional[int]] = {"lbfgs": None}`
    - `transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg`
    - `cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg`
    - `optimization_dt: float = 0.02`
    - `interpolation_steps: int = 4`
    - `use_deceleration_on_failure: bool = True`
    - `deceleration_time: Optional[float] = None`
    - `deceleration_profile: str = "exponential"`
    - `max_deceleration_time: float = 2.0`
    - `load_collision_spheres: bool = True`
    - `num_control_points: Optional[int] = None`
    - `squared_l2_regularization_weight: Optional[List[float]] = None`
    - `warm_start_optimization_num_iters: int = 200`
    - `cold_start_optimization_num_iters: int = 300`
    - `max_batch_size: int = 1`
    - `multi_env: bool = False`
    - `max_goalset: int = 1`
    - `**kwargs`
    - Returns `MPCSolverCfg`. (Writes `optimization_dt`/`interpolation_steps`/`num_control_points`/regularization into the transition-model dict before building core.)

### curobo/_src/solver/solver_mpc_result.py
- **MPCSolverResult**(BaseSolverResult) — Result specific to MPC. Extra fields:
  - `next_action: Optional[JointState] = None`
  - `action_sequence: Optional[JointState] = None`
  - `full_action_sequence: Optional[JointState] = None`
  - `robot_state_sequence: Optional[RobotState] = None`
  - `action_buffer: Optional[torch.Tensor] = None`
  - `action_dt: Optional[float] = None`
  - Method: `clone() -> MPCSolverResult`.

---

# Shared solver infrastructure

### curobo/_src/solver/solver_core_cfg.py
SolverCoreCfg and factory functions for building solver infrastructure. Holds config needed by SolverCore to construct rollouts, optimizers, collision checker, seed manager. Each solver nests a SolverCoreCfg.

- **SolverCoreCfg** (@dataclass) — Config consumed by SolverCore.
  - `metrics_rollout_config: RobotRolloutCfg` (required)
  - `optimizer_rollout_configs: List[RobotRolloutCfg]` (required)
  - `optimizer_configs: List[Any]` (required)
  - `scene_collision_cfg: Optional[SceneCollisionCfg] = None`
  - `device_cfg: DeviceCfg = field(default_factory=DeviceCfg)`
  - `use_cuda_graph: bool = True`
  - `random_seed: int = 123`
  - `store_debug: bool = False`
  - `__post_init__`: raises if `optimizer_configs` empty; disables `use_cuda_graph` when `store_debug` True.
- `resolve_yaml_configs(robot, optimizer_configs, metrics_rollout, transition_model, scene_model, device_cfg, load_collision_spheres=True, num_envs=1) -> Tuple[RobotCfg, List[Dict], Dict, Dict, Optional[Dict]]` — resolve YAML paths to config dicts/objects.
- `create_solver_core_cfg(robot_config, optimizer_dicts, metrics_rollout_dict, transition_model_dict, scene_model_dict, device_cfg, collision_cache=None, self_collision_check=True, optimizer_collision_activation_distance=0.01, use_cuda_graph=True, random_seed=123, store_debug=False, override_optimizer_num_iters=None, transition_model_config_instance_type=RobotStateTransitionCfg, cost_manager_config_instance_type=RobotCostManagerCfg) -> SolverCoreCfg` — build SolverCoreCfg from resolved dicts.
- `create_rollout_configs(optimization_dicts, transition_model_dict, robot_config, device_cfg, optimizer_collision_activation_distance, transition_model_config_instance_type=RobotStateTransitionCfg, cost_manager_config_instance_type=RobotCostManagerCfg, self_collision_check=True) -> List[RobotRolloutCfg]` — one rollout cfg per optimizer stage.
- `create_metrics_rollout_config(metrics_rollout_dict, transition_model_dict, robot_config, device_cfg, transition_model_config_instance_type=RobotStateTransitionCfg, cost_manager_config_instance_type=RobotCostManagerCfg) -> RobotRolloutCfg` — build metrics rollout cfg.
- `create_scene_collision_cfg(scene_model_dict, collision_cache, device_cfg) -> Optional[SceneCollisionCfg]` — scene collision cfg from scene dict/list.

### curobo/_src/solver/solver_core.py
SolverCore — shared infrastructure component (NOT a base class). Manages rollouts, optimizers, collision checker, kinematics, seeds, goal buffers. Owned by each solver via composition.

- **SolverCore** — manages shared solver infrastructure.
  - `__init__(self, config: SolverCoreCfg, scene_collision_checker: Optional[SceneCollision] = None)` — builds collision checker, `metrics_rollout` + `auxiliary_rollout`, optimizer stages (`MultiStageOptimizer`), `AttachmentManager`, `SeedManager`, `GoalManager`, default init state.
  - Properties: `action_dim`, `kinematics`, `transition_model`, `action_horizon`, `default_joint_position`, `default_joint_state`, `joint_names`, `tool_frames`, `solve_state`.
  - `get_all_rollout_instances(include_optimizer_rollouts=True, include_auxiliary_rollout=True) -> List[RobotRollout]`.
  - `update_rollout_params(goal_buffer, include_auxiliary_rollout=True)`, `reset_shape()`, `reset_seed()`, `reset_cuda_graph()`, `destroy()`.
  - `prepare_goal_buffer(solve_state, goal_tool_poses, current_state=None, use_implicit_goal=False, seed_goal_state=None, goal_state=None) -> (GoalRegistry, bool)`.
  - `prepare_action_seeds(batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None) -> torch.Tensor`, `prepare_trajectory_seeds(...)`.
  - `enable_tool_pose_tracking(tool_frames=None, non_terminal_weight_factor=0.0)`, `disable_tool_pose_tracking(tool_frames=None)`, `enable_joint_position_tracking()`, `disable_joint_position_tracking()`.
  - `update_pose_cost_metric(pose_cost_metric: Dict[str, PoseCostMetric])`, `update_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria])`.
  - `sample_configs(num_samples, rejection_ratio=10, optimizer_collision_activation_distance=0.01) -> torch.Tensor` — rejection-sample feasible joint configs.
  - `compute_kinematics(state) -> KinematicsState`, `get_active_js(full_js)`, `get_full_js(active_js)`, `update_link_inertial(...)`, `update_links_inertial(...)`, `debug_dump(file_path)`.

### curobo/_src/solver/solver_base_result.py
Base result dataclass for solvers. Shared by IK/TrajOpt/MPC results.

- **BaseSolverResult** (@dataclass) — base result. Fields:
  - `success: torch.Tensor` (required)
  - `solution: Optional[torch.Tensor] = None`
  - `js_solution: Optional[JointState] = None`
  - `position_error: Optional[torch.Tensor] = None`
  - `rotation_error: Optional[torch.Tensor] = None`
  - `cspace_error: Optional[torch.Tensor] = None`
  - `goalset_index: Optional[torch.Tensor] = None`
  - `solve_time: float = 0.0`
  - `total_time: float = 0.0`
  - `debug_info: Dict = field(default_factory=dict)`
  - `optimized_seeds: Optional[torch.Tensor] = None`
  - `metrics: Optional[RolloutMetrics] = None`
  - `position_tolerance: float = 0.0`
  - `orientation_tolerance: float = 0.0`
  - `seed_rank: Optional[torch.Tensor] = None`
  - `seed_cost: Optional[torch.Tensor] = None`
  - `batch_size: int = 0`
  - `num_seeds: int = 0`
  - `total_cost_reshaped: Optional[torch.Tensor] = None`
  - `solution_state: Optional[RobotState] = None`
  - `feasible: Optional[torch.Tensor] = None`  (constraint feasibility per (batch, seed), independent of pose convergence)
  - Methods: `clone() -> BaseSolverResult`; `copy_successful_solutions(other) -> None` (per-(batch,seed) copy of successes); `copy_at_batch_indices(other, mask) -> None` (overwrite whole batch slices where mask True).

### curobo/_src/solver/solve_mode.py
Solve mode enum for optimization-based solvers.

- **SolveMode** (Enum) `{SINGLE="single", BATCH="batch", MULTI_ENV="multi_env"}`
- `SolveModeInput = Union[SolveMode, Literal["single", "batch", "multi_env"]]` (type alias)
- `parse_solve_mode(mode: SolveModeInput) -> SolveMode` — coerce enum/string to `SolveMode`.

### curobo/_src/solver/solve_state.py
Solve state types for optimization-based solvers.

- **SolveState** (@dataclass) — stores current problem type of a solver.
  - `solve_type: SolveMode` (required)
  - `batch_size: int` (required)
  - `num_envs: int` (required)
  - `num_goalset: int = 1`
  - `multi_env: bool = False`
  - `batch_mode: bool = False`
  - `num_seeds: Optional[int] = None`
  - `num_ik_seeds: Optional[int] = None`
  - `num_graph_seeds: Optional[int] = None`
  - `num_trajopt_seeds: Optional[int] = None`
  - `tool_frames: Optional[List[str]] = None`
  - `__post_init__`: derives `multi_env`/`batch_mode`; back-fills `num_seeds` from ik/trajopt/graph seeds.
  - Methods: `clone()`, `get_batch_size()` (=num_seeds*batch_size), `get_ik_batch_size()`, `get_trajopt_batch_size()`.
- **MotionPlanSolveState** (@dataclass) — motion planner state.
  - `solve_type: SolveMode`
  - `ik_solve_state: SolveState`
  - `trajopt_solve_state: SolveState`

### curobo/_src/solver/__init__.py
Solver module aggregator. Exports `BaseSolverResult`, `SolverCore`, `SolverCoreCfg`, `IKSolver`/`IKSolverCfg`/`IKSolverResult`, `TrajOptSolver`/`TrajOptSolverCfg`/`TrajOptSolverResult`, `MPCSolver`/`MPCSolverCfg`/`MPCSolverResult`, `SolveMode`/`SolveModeInput`/`parse_solve_mode`, `SolveState`/`MotionPlanSolveState`, `SeedManager`/`GoalManager`, `SeedIKSolver`/`SeedIKSolverCfg`.

---

# Managers

### curobo/_src/solver/manager_goal.py
Goal registry manager for optimization-based solvers. Encapsulates goal-buffer creation, updates, and access (target poses, states, config).

- **GoalManager** — manages goal buffer for solvers.
  - `__init__(self, device_cfg: DeviceCfg)`.
  - `create_goal_buffer(solve_state, goal_tool_poses=None, goal_js=None, current_js=None, seed_goal_js=None, current_state_dt=None) -> GoalRegistry`.
  - `update_goal_buffer(solve_state, goal_tool_poses=None, current_js=None, seed_goal_js=None, goal_js=None, use_implicit_goal=False, current_state_dt=None) -> (GoalRegistry, bool)` — in-place update or reallocate; returns `(buffer, update_reference)`.
  - `update_from_goal_registry(solve_state, goal: GoalRegistry) -> (GoalRegistry, bool)`.
  - `update_goal_tool_poses(goal_tool_poses) -> GoalRegistry`, `update_current_state(current_state) -> GoalRegistry`, `update_goal_state(goal_state) -> GoalRegistry`.
  - `update_batch_helper(batch_size) -> torch.Tensor`.
  - Properties: `goal_buffer`, `solve_state`, `batch_helper` (column index tensor). `get_batch_size()`, `get_ik_batch_size()`, `get_trajopt_batch_size()`.

### curobo/_src/solver/manager_seed.py
Action seed manager for optimization-based solvers. Generates single-step (IK) and multi-step (trajectory) seeds via Halton sampling.

- **SeedManager** — generates/prepares action seeds.
  - `__init__(self, device_cfg: DeviceCfg, action_dim: int, action_bound_lows: torch.Tensor, action_bound_highs: torch.Tensor, random_seed: int = 123, action_horizon: int = 1)`.
  - `prepare_action_seeds(batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None) -> torch.Tensor` — single-step seeds `(batch*num_seeds, 1, dof)`.
  - `prepare_trajectory_seeds(batch_size, num_seeds, current_state, seed_config=None, seed_traj=None) -> torch.Tensor` — trajectory seeds `(batch*num_seeds, horizon, dof)`.
  - `generate_random_actions(batch_size, num_seeds) -> torch.Tensor` — Halton random seeds.
  - `prepare_deceleration_trajectory_seeds(batch_size, num_seeds, current_state, deceleration_time=None, deceleration_profile="exponential") -> torch.Tensor`.
  - `reset_seed() -> None`.

---

# Seed IK Solver (`seed_ik/`)

### curobo/_src/solver/seed_ik/seed_ik_solver.py
Seed IK Solver using a Levenberg-Marquardt algorithm — fast approximate IK to seed the main optimizer. Supports Cholesky/QR/SVD LM steps, CUDA-graph capture, single/batch modes, multi-seed returns.

- **SeedIKSolver** — LM-based seed IK solver.
  - `__init__(self, config: SeedIKSolverCfg)` — builds jacobian-enabled `Kinematics`, `SeedIKErrorCalculator`, `SeedIterationStateManager`, Halton sampler; velocity/accel state buffers for velocity-aware IK.
  - `solve_single(goal_tool_poses: GoalToolPose, current_state=None, seed_config=None, return_seeds: int = 1) -> IKSolverResult` — single problem, multiple goal poses.
  - `solve_batch(goal_tool_poses: GoalToolPose, current_state=None, seed_config=None, return_seeds: int = 1) -> IKSolverResult` — batched problems.
  - `update_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria])`, `reset_seed()`, `destroy()`.
  - `compute_kinematics(joint_position: JointState)`, property `kinematics`, `joint_limits`, `get_default_joint_position()`.
  - `n_residuals` (property) = 2*3*num_links (+dof for joint-limit, +dof each for velocity/accel weights).

### curobo/_src/solver/seed_ik/seed_ik_solver_cfg.py
- **SeedIKSolverCfg** (@dataclass) — Configuration for Seed IK Solver (Levenberg-Marquardt based).
  - `robot_config: RobotCfg` (required)
  - `device_cfg: DeviceCfg = DeviceCfg()`
  - `max_iterations: int = 16`
  - `inner_iterations: int = 4`
  - `position_tolerance: float = 0.005`
  - `orientation_tolerance: float = 0.05`
  - `convergence_position_tolerance: float = 0.00001`
  - `convergence_orientation_tolerance: float = 0.00001`
  - `convergence_joint_limit_weight: float = 1.0`
  - `lambda_initial: float = 0.2`
  - `lambda_factor: float = 2.0`
  - `lambda_max: float = 1.0e10`
  - `lambda_min: float = 1e-5`
  - `joint_limit_margin: float = 0.001`
  - `batch_success_threshold: float = 1.0`
  - `max_step_size: float = 0.0`
  - `num_seeds: int = 1`
  - `joint_limit_weight: float = 1.0`
  - `use_cuda_graph: bool = True`
  - `use_backward: bool = True`
  - `rho_min: float = 1e-3`
  - `tile_threads: int = 32`
  - `sampler_seed: int = 451`
  - `max_problems_mini_batch: int = (200 * 512)`
  - `start_cspace_dist_weight: float = 0.01`
  - `position_weight: float = 1.0`
  - `orientation_weight: float = 1.0`
  - `velocity_weight: float = 0.0`
  - `acceleration_weight: float = 0.0`
  - `__post_init__`: requires `max_iterations >= inner_iterations` and divisible.
  - `@staticmethod create(robot: Union[str, Dict, RobotCfg], device_cfg: DeviceCfg = DeviceCfg(), **kwargs) -> SeedIKSolverCfg` — build from robot path/dict/RobotCfg; extra params passed via `**kwargs`. (In `IKSolver`, called with `num_seeds`, `max_iterations`, `inner_iterations`, `joint_limit_weight=1.0`, `lambda_initial=1.0`, `lambda_factor=2.0`, `lambda_max=1.0e10`, `lambda_min=1e-5`, `rho_min=1e-5`, `use_backward=True`, `tile_threads`, and seed pos/ori/vel/acc weights.)

### curobo/_src/solver/seed_ik/seed_ik_state.py
- **SeedIKState** (@dataclass) — State container for Seed IK Solver iterations.
  - `success: Optional[torch.Tensor] = None`
  - `improvement: Optional[torch.Tensor] = None`
  - `joint_position: Optional[torch.Tensor] = None`
  - `error_norm: Optional[torch.Tensor] = None`
  - `jTerror: Optional[torch.Tensor] = None`
  - `jacobian: Optional[torch.Tensor] = None`
  - `lambda_damping: Optional[torch.Tensor] = None`
  - `position_errors: Optional[torch.Tensor] = None`
  - `orientation_errors: Optional[torch.Tensor] = None`
  - Methods: `clone()`, `copy_(other)`.

### curobo/_src/solver/seed_ik/seed_ik_error_calculator.py
Error and Jacobian Calculator for Seed IK Solver. Unified pose + joint-limit (+ optional velocity/acceleration regularization) residual & jacobian computation over CUDA streams.

- **ErrorJacobianResult** (@dataclass) — container for error/jacobian results.
  - `position_errors: torch.Tensor`
  - `orientation_errors: torch.Tensor`
  - `jTerror: torch.Tensor`
  - `jacobian: torch.Tensor`
  - `error_norm: torch.Tensor`
  - `joint_position: torch.Tensor`
- **SeedIKErrorCalculator** — unified IK error/jacobian calculator.
  - `__init__(self, robot_model, config, action_min, action_max, device_cfg)` — sets up `ToolPoseCost` and per-residual CUDA stream pairs.
  - `setup_batch_tensors(batch_size, num_seeds=1)`.
  - `compute_error_and_jacobian(joint_position, goal_poses, idxs_goal, current_position=None, current_velocity=None, dt=None, velocity_clamping_active=False) -> ErrorJacobianResult` — combines pose + joint-limit (+vel/accel) residuals.
  - `update_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria])`, `stream_context(stream_name)`.

### curobo/_src/solver/seed_ik/seed_iteration_state_manager.py
Iteration State Manager for Seed IK Solver. Handles LM step acceptance (trust-region ratio), damping updates, state-value selection, convergence checking.

- **SeedIterationStateManager** — manages LM iteration state updates.
  - Class constant `EPSILON_DIVISION_SAFETY = 1e-8`.
  - `__init__(self, action_min, action_max, rho_min, lambda_factor, lambda_min, lambda_max, convergence_position_tolerance, convergence_orientation_tolerance, convergence_joint_limit_weight)`.
  - `update_iteration_state(current_state: SeedIKState, candidate_state: SeedIKState, predicted_reduction: torch.Tensor, batch_size: int) -> SeedIKState` — trust-region step accept, damping update (÷/× lambda_factor, clamp), value selection, convergence flag.
  - Nested **SelectedStateValues** (@dataclass): `joint_position`, `jTerror`, `jacobian`, `position_errors`, `orientation_errors`.

### curobo/_src/solver/seed_ik/__init__.py
Seed IK Solver module aggregator. Exports `SeedIKSolverCfg`, `SeedIKSolver`, `SeedIKErrorCalculator`, `SeedIterationStateManager`, `SeedIKState`.
