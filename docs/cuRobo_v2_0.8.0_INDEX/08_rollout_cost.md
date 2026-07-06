# cuRobo v2 — Rollout & Cost Functions

The `Rollout` protocol, `RobotRollout` + `RobotCostManager`, and every cost term with its full config-field surface. Rollouts compute the costs/constraints an optimizer minimizes.

## Rollout — `_src/rollout/`

### curobo/_src/rollout/rollout_protocol.py
Structural-typing interface optimizers/solvers use to drive rollouts.
- **Rollout**(Protocol) — props `action_dim`, `action_horizon`, `action_bound_lows`, `action_bound_highs`, `dt`, `sum_horizon`; methods `evaluate_action(act_seq, **kwargs) -> RolloutResult`, `compute_metrics_from_state`, `compute_metrics_from_action`, `update_params`, `update_batch_size`, `update_dt`, `reset`, `reset_shape`, `reset_seed`.

### curobo/_src/rollout/rollout_robot.py
Robot rollout — forward-simulates joint trajectories and evaluates costs.
- **RobotRollout** — `evaluate_action(act_seq)`, `compute_metrics_from_state(state) -> RolloutMetrics`, `compute_metrics_from_action(act_seq) -> RolloutMetrics`, `compute_state_from_action`, `update_params(goal, num_particles=None)`, `update_goal_dt`, `update_batch_size`, `update_dt`, `reset`, `filter_robot_state`, `get_robot_command`, `sample_random_actions(n=0, bounded=True)`, `enable_cost_component(name)`, `disable_cost_component(name)`, `get_all_cost_components() -> Dict[str, BaseCost]`, `get_cost_component_by_name(name)`. Props: `action_dim`, `action_horizon`, `horizon`, `action_bound_lows/highs`, `dt`, `batch_size` (get/set), `default_joint_state/position`.

### curobo/_src/rollout/rollout_robot_cfg.py
Config for RobotRollout (device, costs, transitions, collision).
- **RobotRolloutCfg**(@dataclass) — `device_cfg: DeviceCfg`, `sum_horizon: bool = False`, `sampler_seed: int = 1312`, `cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg`, `transition_model_config_instance_type: Type[RobotStateTransitionCfg]`, `transition_model_cfg`, `cost_cfg`, `constraint_cfg`, `hybrid_cost_constraint_cfg`, `convergence_cfg`, `scene_collision_cfg` (all `Optional`). Classmethods `create_with_component_types(...)`, `get_cost_manager_configs(...)`.

### curobo/_src/rollout/rollout_rosenbrock.py
Rollout evaluating the Rosenbrock cost — canonical optimizer test problem.
- **RosenbrockCfg**(@dataclass) — `device_cfg`, `a: float = 1.0`, `b: float = 100.0`, `dimensions: int = 2`, `time_horizon: int = 1`, `time_action_horizon: int = 1`, `sum_horizon: bool = False`, `sampler_seed: int = 1312`. Classmethod `create(config_dict, device_cfg=DeviceCfg())`.
- **RosenbrockRollout** — `__init__(config=None, use_cuda_graph=False)`; implements Rollout (`evaluate_action`, `compute_metrics_from_state/action`, `update_params(a=None, b=None)`, ...).

### curobo/_src/rollout/metrics.py
Cost/constraint collections and rollout results.
- **CostCollection**(@dataclass) — `values, names, weights, sq_weights`; `add`, `get_sum(sum_horizon=True)`, `merge`, `clone`.
- **CostsAndConstraints**(@dataclass) — `costs, constraints, hybrid_costs_constraints`; `get_sum_cost`, `get_sum_constraint`, `get_feasible`.
- **RolloutResult**(@dataclass, Sequence) — `actions, costs_and_constraints, state, debug`.
- **RolloutMetrics**(RolloutResult) — extra `feasible`, `convergence: CostCollection`.
- **CostCollectionSum**(torch.autograd.Function) — `forward`/`backward`.

### curobo/_src/rollout/goal_registry.py
Stores goal specs; indexes across seeds and batch.
- **GoalRegistry**(@dataclass) — `name="goal"`, `batch_size=-1`, `num_goalset=1`, `num_seeds=1`, `goal_js`, `seed_goal_js`, `link_goal_poses: Optional[GoalToolPose]`, `current_js`, `current_state_dt`, `idxs_link_pose`, `idxs_goal_js`, `idxs_current_js`, `idxs_seed_goal_js`, `idxs_enable`, `idxs_env`, `seed_enable_implicit_goal_js`, `update_idxs_buffers=True`. Methods `repeat_seeds`, `clone`, `apply_kernel`, `copy_`, `create_index_buffers`.

### curobo/_src/rollout/cost_manager/
- **cost_manager_robot.py** — **RobotCostManager** (`register_cost(name, component)`, `get_cost`, `enable/disable_cost_component`, `setup_batch_tensors(batch, horizon)`, `initialize_from_config(...)`, `compute_costs(...)`, `compute_convergence(...)`, `update_params(**kwargs)`).
- **cost_manager_robot_cfg.py** — **RobotCostManagerCfg**(@dataclass): `class_type: type = None`, `self_collision_cfg: Optional[SelfCollisionCostCfg]`, `scene_collision_cfg: Optional[SceneCollisionCostCfg]`, `cspace_cfg: Optional[CSpaceCostCfg]`, `start_cspace_dist_cfg`, `target_cspace_dist_cfg: Optional[CSpaceDistCostCfg]`, `tool_pose_cfg: Optional[ToolPoseCostCfg]`. `create(data_dict, scene_collision_checker=None, device_cfg=DeviceCfg()) -> RobotCostManagerCfg`; methods `update_collision_activation_distance`, `disable_self_collision`, `update_regularization_weight`.

## Cost functions — `_src/cost/`

### curobo/_src/cost/cost_base.py + cost_base_cfg.py
- **BaseCostCfg**(@dataclass) — `weight: Union[torch.Tensor,float,List[float]]`, `class_type: Type[BaseCost] = BaseCost`, `device_cfg: DeviceCfg = DeviceCfg()`, `convert_to_binary: bool = False`, `use_grad_input: bool = False`.
- **BaseCost** — `setup_batch_tensors(batch, horizon)`, `forward(**kwargs)`, `enable_cost`, `disable_cost`, `update_dt(dt)`, `reset(...)`; prop `enabled`.

### C-space costs
- **cost_cspace_cfg.py** — **CSpaceCostCfg**(BaseCostCfg): `dof: int = 0`, `cost_type: Optional[CSpaceCostType] = None`, `joint_limits: Optional[JointLimits]`, `squared_l2_regularization_weight: Optional[List[float]]`, `retime_weights: bool = False`, `retime_regularization_weights: bool = False`, `activation_distance: Union[torch.Tensor,float] = 0.0`, `cspace_target_weight`, `cspace_non_terminal_weight_factor`, `cspace_target_dof_weight`. Methods `set_bounds`, `initialize_from_transition_model`, `update_dof`.
- **cost_cspace_type.py** — `CSpaceCostType` {POSITION=0, STATE=1}
- **cost_cspace_base.py / cost_cspace_position.py / cost_cspace_state.py** — **BaseCSpaceCost**(BaseCost) → **PositionCSpaceCost**, **StateCSpaceCost** (each `setup_batch_tensors`, `forward`).
- **cost_cspace_dist_cfg.py** — **CSpaceDistCostCfg**(BaseCostCfg): `class_type=CSpaceDistCost`, `use_null_space: bool = False`, `only_terminal_cost: bool = True`, `terminal_dof_weight`, `non_terminal_dof_weight`. **cost_cspace_dist.py** — **CSpaceDistCost** (`forward`, `forward_out_distance`).

### Collision costs
- **cost_self_collision_cfg.py** — **SelfCollisionCostCfg**(BaseCostCfg): `class_type=SelfCollisionCost`, `self_collision_kin_config: Optional[SelfCollisionKinematicsCfg]`, `store_pair_distance: bool = False`. **cost_self_collision.py** — **SelfCollisionCost** (`forward`, `reset`).
- **cost_scene_collision_cfg.py** — **SceneCollisionCostCfg**(BaseCostCfg): `class_type=SceneCollisionCost`, `use_sweep: bool = False`, `use_sweep_kernel: bool = False`, `use_speed_metric: bool = False`, `activation_distance: Union[torch.Tensor,float] = 0.0`, `sum_distance: bool = True`, `num_spheres: int = 0`, `_num_scene_collision_checkers: int = 0`, `_scene_collision_checker: Optional[SceneCollision]`. **cost_scene_collision.py** — **SceneCollisionCost** (`update_num_spheres`, `forward`, `get_gradient_buffer`) — CUDA-only.

### Tool-pose costs
- **cost_tool_pose_cfg.py** — **ToolPoseCostCfg**(BaseCostCfg): `class_type=ToolPoseCost`, `tool_frames: Optional[List[str]]`, `tool_pose_criteria: Dict[str, ToolPoseCriteria] = {}`, `use_lie_group: bool = False` (+ private terminal/non-terminal tolerance & axes-weight fields, `_project_distance_to_goal`). Methods `set_tool_frames`; props `num_links`, `rotation_method`.
- **cost_tool_pose.py** — **ToolPoseCost** (`setup_batch_tensors`, `forward`, `update_tool_pose_criteria`).
- **cost_pose_type.py** — `PoseErrorType` {SINGLE_GOAL=0, BATCH_GOAL=1, GOALSET=2, BATCH_GOALSET=3}
- **cost_pose_metric.py** — **PoseCostMetric**(@dataclass): `hold_partial_pose: bool = False`, `release_partial_pose: bool = False`, `hold_vec_weight`, `reach_partial_pose: bool = False`, `reach_full_pose: bool = False`, `reach_vec_weight`, `offset_position`, `offset_rotation`, `offset_tstep_fraction: float = -1.0`, `remove_offset_waypoint: bool = False`, `include_link_pose: bool = False`, `project_to_goal_frame: Optional[bool]`. Classmethods `create_grasp_approach_metric(...)`, `reset_metric()`, `reach_position_metric(...)`.

### tool_pose_criteria.py
Per-link pose tracking axes/weights/tolerances.
- **ToolPoseCriteria**(@dataclass) — `terminal_pose_axes_weight_factor`, `non_terminal_pose_axes_weight_factor`, `terminal_pose_convergence_tolerance`, `non_terminal_pose_convergence_tolerance`, `project_distance_to_goal: Union[torch.Tensor,bool] = False`, `device_cfg: DeviceCfg = DeviceCfg()`. Static factories: **`track_position(xyz=[1.,1.,1.])`**, **`track_orientation(...)`**, **`track_position_and_orientation(...)`**, **`linear_motion(...)`**, **`disabled()`**.
- **StackedToolPoseCriteria**(@dataclass) — stacked tensors over `tool_frames`; `from_tool_pose_criteria(...)`, `update_tool_pose_criteria(...)`.

### Support-polygon (humanoid balance)
- **cost_support_polygon_cfg.py** — **CostSupportPolygonCfg**(BaseCostCfg): `class_type=CostSupportPolygon`, `foot_sphere_indices`, `foot_link_names`, `inside_cost_weight: float = 0.001`. **cost_support_polygon.py** — **CostSupportPolygon** (`build_convex_hull`, `forward(robot_com, robot_spheres)`).

### Warp kernel files (compact — device code)
- **warp_bound_util.py** — `@wp.func` bound helpers: `shrink_bounds_with_activation_distance`, `aggregate_bound_cost`, `aggregate_bound_cost_l1`, `aggregate_squared_l2_regularization`, `aggregate_energy_regularization`.
- **wp_cspace_position.py** — `PositionCSpaceFunction`(autograd) + `forward_cspace_position_warp`.
- **wp_cspace_state.py** — `StateCSpaceFunction`(autograd) + `forward_cspace_state_warp`.
- **wp_torch_cspace_dist.py** — `L2DistFunction`(autograd) + `forward_l2_warp`.
- **wp_torch_pose_dist.py** ≈ **wp_tool_pose.py** (near-duplicate) — pose-distance kernels: `scale_quaternion_difference_by_axis`, `compute_position_error`, `compute_rotation_error{,_axis_angle,_lie_group}`, `create_goalset_pose_distance_kernel_with_constants(num_goalset, rotation_method=0)`; `ToolPoseDistance`(autograd).
