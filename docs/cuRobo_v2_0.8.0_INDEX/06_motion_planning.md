# cuRobo v2 — Motion Planning & Retargeting
MotionPlanner, BatchMotionPlanner, MotionRetargeter and their configs/results.

### curobo/_src/motion/motion_planner.py
Single-problem motion planner with retry logic and graph-planner seeding.

- **MotionPlanner** — solves one planning problem at a time (`batch_size=1`); wraps IKSolver + TrajOptSolver + optional PRMGraphPlanner, with retry/seed-repair/graph-seed logic.
  - `__init__(self, config: MotionPlannerCfg)` — build IK/trajopt/graph solvers over a shared (optional) scene collision checker.
  - `destroy()` — release all CUDA graph resources (idempotent, guards `_destroyed`); also called by `__del__` and `__exit__`.
  - `warmup(enable_graph: bool = True, warmup_joint_index: int = 0, warmup_joint_delta: float = 0.2, num_warmup_iterations: int = 10) -> bool` — JIT-warmup IK/TrajOpt (and graph planner); handles goalset vs single-goal paths; temporarily disables `exit_early`.
  - `plan_pose(goal_tool_poses: GoalToolPose, current_state: JointState, use_implicit_goal: bool = True, max_attempts: int = 5, enable_graph_attempt: int = 1) -> Optional[TrajOptSolverResult]` — plan trajectory to target tool pose(s); auto-detects goalset from `goal_tool_poses.num_goalset` (goalset>1 → simpler IK+TrajOpt loop, else single-goal retry loop with graph seeding).
  - `plan_cspace(goal_state: JointState, current_state: JointState, max_attempts: int = 5, enable_graph_attempt: int = 1) -> Optional[TrajOptSolverResult]` — plan collision-free trajectory to a joint configuration, with retry and graph-seeding from `enable_graph_attempt`.
  - `plan_grasp(grasp_poses: GoalToolPose, current_state: JointState, grasp_approach_axis: str = "z", grasp_approach_offset: float = -0.15, grasp_approach_in_tool_frame: bool = True, grasp_lift_axis: str = "z", grasp_lift_offset: float = -0.15, grasp_lift_in_tool_frame: bool = True, plan_approach_to_grasp: bool = True, plan_grasp_to_lift: bool = True, disable_collision_links: List[str] = None) -> GraspPlanResult` — 4-stage grasp motion: goalset select → approach pose → linear approach→grasp (with `ToolPoseCriteria.linear_motion`) → lift; disables/re-enables contact-link spheres per stage; `disable_collision_links` defaults to `kinematics_config.grasp_contact_link_names`.
  - `compute_kinematics(state: JointState) -> KinematicsState` — FK via the trajopt solver.
  - `enable_link_collision(enable_collision_links: List[str])` / `disable_link_collision(disable_collision_links: List[str])` — toggle collision spheres per link.
  - `update_world(scene_cfg: SceneCfg)` — load collision model into scene checker and reset graph buffer.
  - `clear_scene_cache()` — clear scene checker cache and reset graph buffer.
  - `reset_seed()` — reset IK/TrajOpt/graph seeds and graph buffer.
  - `update_link_inertial(link_name, mass=None, com=None, inertia=None)` / `update_links_inertial(link_properties: dict)` — push inertial overrides to both solvers.
  - `update_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria])` — set per-link pose-tracking criteria on IK and TrajOpt.
  - properties: `attachment_manager`, `joint_names -> List[str]`, `action_dim -> int`, `tool_frames -> List[str]`, `default_joint_state -> JointState`, `kinematics -> Kinematics`.
- `_axis_string_to_vector(axis: str) -> List[float]` — map `"x"/"y"/"z"` to a unit vector (module-level, used by both planners; raises on invalid axis).

### curobo/_src/motion/motion_planner_batch.py
Batch motion planner for solving multiple planning problems in parallel.

- **BatchMotionPlanner** — solves `batch_size` independent problems in a single IK→(optional graph seed)→TrajOpt pass; graph seeding only when `multi_env=False`; no per-problem retry-with-graph like `MotionPlanner` but supports first-success-wins across attempts.
  - `__init__(self, config: MotionPlannerCfg)` — reads `max_batch_size`/`multi_env` from `config.ik_solver_config`; builds graph planner only if `not multi_env` and a graph config exists.
  - `destroy()` — release CUDA graph resources (also via `__exit__`; no `__del__` here).
  - `warmup(enable_graph: bool = True, num_warmup_iterations: int = 5) -> bool` — batched cspace warmup (perturbs joint 0 by 0.2) plus optional graph warmup.
  - `plan_pose(goal_tool_poses: GoalToolPose, current_state: JointState, use_implicit_goal: bool = True, max_attempts: int = 1, success_ratio: float = 1.0, enable_graph_attempt: int = 0) -> Optional[TrajOptSolverResult]` — batched pose planning; each attempt re-solves the full batch, per-problem results locked in on first success via `copy_at_batch_indices`; exits early when solved fraction ≥ `success_ratio`.
  - `plan_cspace(goal_states: JointState, current_state: JointState, max_attempts: int = 1, success_ratio: float = 1.0, enable_graph_attempt: int = 0) -> Optional[TrajOptSolverResult]` — batched joint-space planning, same first-success-wins / `success_ratio` logic.
  - `plan_grasp(grasp_poses: GoalToolPose, current_state: JointState, grasp_approach_axis: str = "z", grasp_approach_offset: float = -0.15, grasp_approach_in_tool_frame: bool = True, grasp_lift_axis: str = "z", grasp_lift_offset: float = -0.15, grasp_lift_in_tool_frame: bool = True, plan_approach_to_grasp: bool = True, plan_grasp_to_lift: bool = True, disable_collision_links: List[str] = None) -> GraspPlanResult` — batched 4-stage grasp; ALL B problems planned at every stage (CUDA-graph stability), failed problems get FK-of-current-state substituted as goal (`_substitute_fallback_goal`) so the optimizer doesn't diverge; per-problem grasp pose picked by each problem's `goalset_index` (`_extract_per_problem_grasp`).
  - `compute_kinematics(state) -> KinematicsState`, `update_world(scene_cfg)`, `clear_scene_cache()`, `reset_seed()`, `enable_link_collision(...)`, `disable_link_collision(...)`, `update_tool_pose_criteria(...)`, `update_link_inertial(...)`, `update_links_inertial(...)` — mirror `MotionPlanner`.
  - properties: `attachment_manager`, `batch_size -> int` (= `config.ik_solver_config.max_batch_size`), `joint_names`, `action_dim`, `tool_frames`, `default_joint_state`, `kinematics`.

Batch differences vs `MotionPlanner`: single-pass by default (`max_attempts=1`), `success_ratio` early-exit, `enable_graph_attempt=0` (graph from first attempt), graph seeding disabled entirely under `multi_env=True`, no seed-repair/finetune-escalation, failed-goal fallback substitution, and per-problem success is a `[B]` mask (`.any(dim=-1)`) rather than a scalar.

### curobo/_src/motion/motion_planner_cfg.py
Configuration for the motion planner. (inferred first line)

- **MotionPlannerCfg** (@dataclass) — bundles IK, TrajOpt, graph-planner and scene-collision configs for `MotionPlanner`/`BatchMotionPlanner`.
  - `ik_solver_config: IKSolverCfg`
  - `trajopt_solver_config: TrajOptSolverCfg`
  - `graph_planner_config: PRMGraphPlannerCfg = None`
  - `scene_collision_cfg: Optional[SceneCollisionCfg] = None`
  - `device_cfg: DeviceCfg = DeviceCfg()`
  - `create(robot: Union[str, Dict[str, Any], RobotCfg], ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["ik/particle_ik.yml", "ik/lbfgs_ik.yml"][1:], ik_transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml", metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml", trajopt_optimizer_configs: List[Union[str, Dict[str, Any]]] = ["trajopt/lbfgs_bspline_trajopt.yml"], trajopt_transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml", graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml", graph_planner_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml", graph_planner_transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml", scene_model: Optional[Union[str, Dict[str, Any]]] = None, collision_cache: Optional[Dict[str, int]] = None, self_collision_check: bool = True, device_cfg: DeviceCfg = DeviceCfg(), num_ik_seeds: int = 32, num_trajopt_seeds: int = 4, position_tolerance: float = 0.005, orientation_tolerance: float = 0.05, use_cuda_graph: bool = True, random_seed: int = 123, optimizer_collision_activation_distance: float = 0.01, store_debug: bool = False, transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg, cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg, max_batch_size: int = 1, multi_env: bool = False, max_goalset: int = 1) -> MotionPlannerCfg` — @staticmethod; resolves robot/scene configs then builds `IKSolverCfg.create`, `TrajOptSolverCfg.create`, `PRMGraphPlannerCfg.create`. `num_envs = max_batch_size if multi_env else 1`. `ik_optimizer_configs` default is the SLICE `[...][1:]` → effectively `["ik/lbfgs_ik.yml"]` (particle_ik dropped). `collision_cache` alone (no `scene_model`) still pre-allocates a `SceneCollisionCfg` for later voxel/mesh updates.

### curobo/_src/motion/motion_planner_result.py
(inferred) Result dataclasses for motion / grasp planning.

- **MotionPlannerResult** (@dataclass) — result of a motion planning operation.
  - `success: Optional[torch.Tensor] = None`
- **GraspPlanResult** (@dataclass) — result of a grasp planning operation (returned by `plan_grasp`).
  - `success: Optional[torch.Tensor] = None`
  - `approach_success: Optional[torch.Tensor] = None`
  - `grasp_success: Optional[torch.Tensor] = None`
  - `lift_success: Optional[torch.Tensor] = None`
  - `approach_trajectory: Optional[JointState] = None`
  - `approach_trajectory_dt: Optional[torch.Tensor] = None`
  - `approach_interpolated_trajectory: Optional[JointState] = None`
  - `grasp_trajectory: Optional[JointState] = None`
  - `grasp_trajectory_dt: Optional[torch.Tensor] = None`
  - `grasp_interpolated_trajectory: Optional[JointState] = None`
  - `lift_trajectory: Optional[JointState] = None`
  - `lift_trajectory_dt: Optional[torch.Tensor] = None`
  - `lift_interpolated_trajectory: Optional[JointState] = None`
  - `approach_interpolated_last_tstep: Optional[torch.Tensor] = None`
  - `grasp_interpolated_last_tstep: Optional[torch.Tensor] = None`
  - `lift_interpolated_last_tstep: Optional[torch.Tensor] = None`
  - `status: Optional[str] = None`
  - `planning_time: float = 0.0`
  - `goalset_index: Optional[torch.Tensor] = None`
  - Note: `plan_grasp` also sets non-dataclass attrs at runtime — `goalset_result`, `approach_result` (assigned dynamically on the instance).

### curobo/_src/motion/motion_retargeter.py
Motion retargeter: IK and MPC-based retargeting for humanoid robots.

- **MotionRetargeter** — high-level per-frame retargeting; frame 0 uses global IK (many seeds), later frames use warm-started local IK or MPC per `config.use_mpc`. Tracks internal state (`_prev_solution`, `_prev_velocity`, `_mpc_state`) across `solve_frame` calls.
  - `__init__(self, config: MotionRetargeterCfg)` — builds global IK solver plus either an MPC solver (`use_mpc=True`) or local IK solver.
  - `solve_frame(goal_tool_poses: GoalToolPose) -> RetargetResult` — solve one frame; routes to global IK (first frame) / MPC / local IK; raises if `goal_tool_poses.batch_size != num_envs`.
  - `solve_sequence(tool_poses: SequenceGoalToolPose) -> RetargetResult` — reset state then loop `solve_frame` over all frames (tqdm `trange`), stacking `joint_state` on dim=1 and concatenating MPC trajectories; raises if `tool_poses.num_envs != num_envs`.
  - `reset() -> None` — clear warm-start state so the next `solve_frame` uses global IK.
  - properties: `joint_names -> List[str]`, `action_dim -> int`, `tool_frames -> List[str]`, `kinematics -> Kinematics`, `default_joint_state -> JointState`, `num_dof -> int` (deprecated alias for `action_dim`), `config -> MotionRetargeterCfg`.
  - private builders: `_build_global_ik_solver` (num_seeds=`num_seeds_global`, no velocity limit, override lbfgs iters=`global_ik_num_iters`), `_build_local_ik_solver` (num_seeds=`num_seeds_local`, `optimization_dt`, velocity/accel reg weights, sets `use_lm_seed=False`/`exit_early=False`), `_build_mpc_solver` (`MPCSolverCfg.create` with `num_control_points`, warm/cold-start iters).

### curobo/_src/motion/motion_retargeter_cfg.py
Configuration for MotionRetargeter.

- **MotionRetargeterCfg** (@dataclass) — config for `MotionRetargeter`; use `create` to construct.
  - `robot: Union[str, Dict[str, Any]]` — robot config YAML filename or dict.
  - `tool_pose_criteria: Dict[str, ToolPoseCriteria]` — per-link position(xyz)/rotation(rpy) tracking weights.
  - `num_envs: int = 1` — clips retargeted in parallel; short clips padded by repeating last frame.
  - `use_mpc: bool = False` — MPC for frames 1+ (smoother, 2–4x slower) vs warm-started IK.
  - `self_collision_check: bool = True`
  - `scene_model: Optional[Union[str, Dict[str, Any]]] = None`
  - `optimization_dt: float = 0.05` — timestep for velocity-limited IK/MPC.
  - `num_seeds_global: int = 64` — seeds for frame-0 global IK.
  - `position_tolerance: float = 0.005`
  - `orientation_tolerance: float = 0.05`
  - `device_cfg: DeviceCfg = field(default_factory=DeviceCfg)`
  - `load_collision_spheres: bool = True` — auto-disabled when both `self_collision_check=False` and `scene_model=None`.
  - `ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(default_factory=lambda: ["ik/lbfgs_retarget_ik.yml"])`
  - `mpc_optimizer_configs: List[Union[str, Dict[str, Any]]] = field(default_factory=lambda: ["mpc/lbfgs_retarget_mpc.yml"])`
  - `num_seeds_local: int = 1` — seeds for warm-started local IK (ignored when `use_mpc=True`).
  - `num_control_points: Optional[int] = None` — B-spline control points for MPC (ignored when `use_mpc=False`). NOTE: dataclass default is `None`, but `create` defaults it to `12`.
  - `steps_per_target: int = 8` — MPC optimization steps per input frame. NOTE: `create` defaults it to `4`.
  - `velocity_regularization_weight: Optional[float] = None` — penalizes `(q - q_prev)/dt` (YAML default 0.001 for IK).
  - `acceleration_regularization_weight: Optional[float] = None` — penalizes `(v - v_prev)/dt` (YAML default 0.01 for IK).
  - `collision_activation_distance: float = 0.01`
  - `global_ik_num_iters: Optional[int] = None` — L-BFGS iters for frame-0 global IK (YAML default 200).
  - `local_ik_num_iters: Optional[int] = None` — L-BFGS iters for warm-started local IK (ignored when `use_mpc=True`).
  - `mpc_warm_start_num_iters: int = 100` — iters for warm-started MPC steps.
  - `mpc_cold_start_num_iters: int = 300` — iters for the first MPC step (should be ≥ warm-start).
  - property `tool_frames -> List[str]` — ordered `tool_pose_criteria` keys.
  - `create(robot: Union[str, Dict[str, Any]], tool_pose_criteria: Dict[str, ToolPoseCriteria], num_envs: int = 1, use_mpc: bool = False, self_collision_check: bool = True, scene_model: Optional[Union[str, Dict[str, Any]]] = None, optimization_dt: float = 0.05, num_seeds_global: int = 64, load_collision_spheres: bool = True, ik_optimizer_configs: Optional[List[Union[str, Dict[str, Any]]]] = None, mpc_optimizer_configs: Optional[List[Union[str, Dict[str, Any]]]] = None, num_seeds_local: int = 1, num_control_points: Optional[int] = 12, steps_per_target: int = 4, position_tolerance: float = 0.005, orientation_tolerance: float = 0.05, device_cfg: DeviceCfg = DeviceCfg(), velocity_regularization_weight: Optional[float] = None, acceleration_regularization_weight: Optional[float] = None, collision_activation_distance: float = 0.01, global_ik_num_iters: Optional[int] = None, local_ik_num_iters: Optional[int] = None, mpc_warm_start_num_iters: int = 100, mpc_cold_start_num_iters: int = 300) -> MotionRetargeterCfg` — @staticmethod; `None` optimizer configs fall back to the retarget YAML defaults.
  - `__post_init__` — auto-disables `load_collision_spheres` when no collision checking; raises (`log_and_raise`) if `load_collision_spheres` is False while `self_collision_check` is True or `scene_model` is set.

### curobo/_src/motion/motion_retargeter_result.py
Result type for MotionRetargeter.

- **RetargetResult** (@dataclass) — returned by `MotionRetargeter.solve_frame` / `solve_sequence`.
  - `joint_state: JointState` — final joint state; `solve_frame` position shape `(num_envs, num_dof)`, `solve_sequence` position shape `(num_envs, num_output_frames, num_dof)`.
  - `trajectory: Optional[JointState] = None` — full intermediate MPC trajectory (MPC mode only; None for IK mode).

### curobo/_src/motion/__init__.py
Motion module for motion planning and retargeting.

- Re-exports (`__all__`): `MotionPlanner`, `MotionPlannerCfg`, `GraspPlanResult`. (Note: `BatchMotionPlanner`, `MotionRetargeter`, `MotionRetargeterCfg`, `RetargetResult` are NOT re-exported here — import from their submodules.)
