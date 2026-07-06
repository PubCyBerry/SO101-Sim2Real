# cuRobo v2 — Graph Planner (PRM)

PRM roadmap planner, graph construction, node sampling, and path search.

### curobo/_src/graph_planner/graph_planner_prm.py
(inferred) PRM roadmap motion planner — batched start/goal path finding via feasible sampling, linear steering, roadmap graph, and NetworkX shortest-path search.

- **PRMGraphPlanner** — orchestrates the full PRM pipeline: feasibility rollout, sampling strategy, graph construction, path search, pruning, and interpolation.
  - `__init__(self, config: PRMGraphPlannerCfg, scene_collision_checker: Optional[SceneCollision] = None)` — build all components (feasibility/auxiliary rollouts, node manager, sampler, constructor, pruner, path finder).
  - `find_path(self, x_start: torch.Tensor, x_goal: torch.Tensor, interpolate_waypoints: bool = True, interpolation_steps: int = 100, interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR, validate_interpolated_trajectory: bool = True) -> GraphPlannerResult` — main entry: plan batched paths, optionally interpolate + revalidate; returns `GraphPlannerResult` with `solve_time`/`joint_names` set.
  - `check_samples_feasibility(self, action_samples)` — collision/constraint feasibility mask for a 2D (batch, action_dim) tensor; chunks by `feasibility_buffer_size`.
  - `extend_roadmap_with_random_samples(self, num_samples: int, neighbors_per_node: int = 10)` — sample uniform feasible nodes, add to buffer, connect to graph.
  - `extend_roadmap_with_ellipsoidal_samples(self, x_start, x_goal, max_sampling_radius, num_samples, neighbors_per_node: int = 5)` — informed-ellipsoid sampling between start/goal, add + connect.
  - `get_interpolated_trajectory(self, paths, success, interpolation_steps, interpolation_type)` — per-joint `linear_smooth` interpolation over successful waypoint paths (supports LINEAR/CUBIC/QUINTIC).
  - `reset_buffer(self)` — clear node manager + graph constructor state.
  - `reset_seed(self)` — reset sampler and path-finder seeds.
  - `reset_cuda_graph(self)` — reset auxiliary rollout cuda graph if present.
  - `get_all_rollout_instances(self) -> List[RobotRollout]` — [feasibility_rollout, auxiliary_rollout].
  - `warmup(self, num_warmup_iterations: int = 10, max_batch_size: int = 4)` — JIT/cuda-graph warmup by running dummy find_path calls.
  - `compute_kinematics(self, state: JointState) -> KinematicsState` — forward kinematics via robot model.
  - Properties: `action_dim`, `kinematics`, `transition_model`, `default_joint_state`, `joint_names`, `action_bound_lows`, `action_bound_highs`, `n_nodes`, `cspace_distance_weight`.

### curobo/_src/graph_planner/graph_planner_prm_cfg.py
(inferred) Configuration dataclass + factory for the PRM graph planner.

- **PRMGraphPlannerCfg** — all tunable PRM parameters (roadmap size, sampling, growth factors, rollout/collision configs).
  - `max_nodes: int` — Maximum number of nodes in the graph.
  - `feasibility_buffer_size: int` — Max points to check for feasibility per call; larger queries are split.
  - `steer_buffer_size: int` — Max points allowed to steer between two nodes.
  - `exploration_radius: float` — Max radius to sample around linear start→goal path (like c_max in BIT*).
  - `new_nodes_per_iteration: int` — Number of nodes to sample per iteration.
  - `max_path_finding_iterations: int` — Max iterations to find a path.
  - `min_finetune_iterations: int` — Min iterations to finetune path.
  - `use_default_position_heuristic: bool` — Connect start/goal through default joint position as heuristic.
  - `cspace_similarity_threshold: float` — Threshold for cspace distance similarity.
  - `sample_rejection_ratio: int` — Node sample rejection ratio (should be > 1).
  - `neighbors_per_node: int` — Nearest neighbors to steer to when connecting a sampled node.
  - `rollout_config: RobotRolloutCfg` — Configuration for the rollout function.
  - `sampler_seed: int` — Seed for the node sampler.
  - `sampler_buffer_size: int` — Number of samples stored in the sampler buffer.
  - `use_cuda_graph_for_rollout: bool` — Whether to use cuda graph.
  - `connect_terminal_nodes_with_nearest: bool` — Connect start/goal to nearest graph nodes when adding; set True if performance is poor.
  - `exploration_radius_growth_factor: float` — Exploration radius growth factor (>1).
  - `neighbors_per_node_growth_factor: float` — Nearest neighbors growth factor (>1).
  - `new_nodes_per_iteration_growth_factor: float` — New-nodes-per-iteration growth factor (>1).
  - `ellipsoid_projection_method: str = "householder"` — Ellipsoid projection: "svd", "householder", or "approximate".
  - `scene_collision_cfg: Optional[SceneCollisionCfg] = None` — World collision checker config.
  - `device_cfg: DeviceCfg = DeviceCfg()` — Tensor device configuration.
  - `graph_path_finder_seed: int = 42`
  - `@staticmethod create(robot: Union[str, Dict[str, Any], RobotCfg], graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml", rollout: Union[str, Dict[str, Any]] = "metrics_base.yml", transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml", scene_model: Optional[Union[str, Dict[str, Any]]] = None, collision_cache: Optional[Dict[str, int]] = None, self_collision_check: bool = True, device_cfg: DeviceCfg = DeviceCfg(), use_cuda_graph_for_rollout: bool = True, transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg, cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg, graph_path_finder_seed: int = 42) -> "PRMGraphPlannerCfg"` — build cfg from file paths / dicts / objects; resolves robot, scene collision, rollout+transition, and `graph_planner` YAML params.

### curobo/_src/graph_planner/result.py
(inferred) Result dataclass for graph planner queries.

- **GraphPlannerResult** — Data class stores information about the graph planner result.
  - `success: torch.Tensor` — Success flag for each query, shape (B).
  - `plan_waypoints: Optional[List[Union[torch.Tensor, None]]] = None` — Plan waypoints, shape (B, N, action_dim).
  - `interpolated_waypoints: Optional[torch.Tensor] = None` — Interpolated waypoints, shape (B, interpolation_steps, action_dim).
  - `joint_names: Optional[List[str]] = None` — Joint names for each index in action_dim.
  - `path_length: Optional[torch.Tensor] = None` — Path length, shape (B).
  - `solve_time: float = 0.0` — Solve time.
  - `valid_query: bool = True` — Valid query flag.
  - `debug_info: Optional[Any] = None` — Debug info.

### curobo/_src/graph_planner/graph/constructor.py
(inferred) Builds and maintains the PRM roadmap graph structure: node addition, edge creation, terminal (start/goal) and default-position node handling.

- **GraphConstructor** — Handles construction and maintenance of the PRM graph structure.
  - `__init__(self, config: PRMGraphPlannerCfg, linear_connector: LinearConnector, distance_calculator: DistanceNeighborCalculator, node_manager: GraphNodeManager, action_dim: int, check_feasibility_fn, device_cfg)`
  - `steer_and_register_edges(self, start_nodes, goal_nodes, add_exact_node=False)` — steer start→goal linearly until infeasible, then register resulting nodes + edges (both shape (B, action_dim+1)).
  - `connect_nodes(self, new_nodes, add_exact_node=False, neighbors_per_node=10)` — find nearest neighbors for new nodes and steer/register edges to them.
  - `initialize_default_node(self, default_joint_state: JointState) -> Tuple[Optional[torch.Tensor], bool]` — add default joint position as roadmap node if feasible + heuristic enabled (cached once).
  - `initialize_terminal_graph_connections(self, x_init_batch, x_goal_batch, default_joint_state) -> Tuple[torch.Tensor, torch.Tensor]` — add start/goal (and default) nodes, wire bidirectional connections, return (start_nodes_in_roadmap, goal_nodes_in_roadmap).
  - `reset(self)` — clear cached default-node state.

### curobo/_src/graph_planner/graph/node_manager.py
(inferred) Roadmap node storage manager (preallocated buffer, dedup, index bookkeeping) plus the ConnectedGraph debug container and JIT node-buffer helpers.

- **ConnectedGraph** — Data class storing the created graph (nodes/edges/connectivity) for debugging.
  - `nodes: torch.Tensor` — Shape (B, N, DOF).
  - `edges: torch.Tensor` — Shape (num_edges, 2, DOF); each edge is a dof config.
  - `connectivity: torch.Tensor` — Shape (num_edges, 3); columns [start_node_idx, end_node_idx, edge_distance].
  - `robot_state_nodes: Optional[RobotState] = None`
  - `shortest_path_lengths: Optional[torch.Tensor] = None`
  - `set_shortest_path_lengths(self, shortest_path_lengths)` — store per-node shortest-path lengths.
  - `get_node_distance(self)` — concat nodes with shortest_path_lengths column (or None).
- **GraphNodeManager** — Manages graph node storage and operations for the PRM planner.
  - `__init__(self, config, distance_calculator=None, graph_path_finder=None, auxiliary_rollout=None, device_cfg=None)` — allocate `(max_nodes, action_dim+1)` node buffer + steer index buffer.
  - `add_nodes_to_buffer(self, new_nodes)` — append nodes to preallocated buffer (raises if buffer too small).
  - `get_connected_graph(self)` — materialize a `ConnectedGraph` from current edges + robot states.
  - `register_nodes_and_connections(self, node_set, start_nodes, add_exact_node=False)` — add nodes to roadmap and register weighted edges to the path finder.
  - `add_nodes_to_roadmap(self, nodes, add_exact_node=False) -> torch.Tensor` — dedup vs threshold, add new nodes, return (B, action_dim+1) node set with roadmap indices.
  - `add_initial_exact_nodes_to_roadmap(self, nodes) -> torch.Tensor` — add first exact (zero-distance dedup) nodes when buffer empty.
  - `get_nodes_in_path(self, path_list)` — map index paths → node configs (None passthrough).
  - `reset_buffer(self)` — reset path finder + zero node buffer + count.
  - `reset_graph_path_finder(self)` — reset path finder graph only.
  - Properties: `action_dim`, `n_nodes`, `node_idx_padding_buffer`, `preallocated_node_buffer`, `valid_node_buffer` (buffer sliced to n_nodes).
- `jit_add_nodes_to_buffer(preallocated_node_buffer, new_nodes, used_node_count, action_dim, device, dtype) -> torch.Tensor` — JIT: write nodes + arange indices into buffer.
- `jit_add_all_nodes_to_buffer(all_nodes, preallocated_node_buffer, used_node_count, action_dim) -> Tuple[torch.Tensor, torch.Tensor, int]` — JIT: append all nodes, return (buffer, new_nodes_in_roadmap, updated_count).

### curobo/_src/graph_planner/graph/node_distance.py
(inferred) Weighted C-space distance, nearest-neighbor search, and unique-node dedup (all JIT-compiled).

- **DistanceNeighborCalculator** — distance calculations, nearest neighbors, and unique node operations in configuration space.
  - `__init__(self, action_dim: int, cspace_distance_weight: torch.Tensor, device_cfg: DeviceCfg)`
  - `calculate_weighted_distance(self, pt, batch_pts) -> torch.Tensor` — weighted L2 distance between point(s) and batch.
  - `find_nearest_neighbors(self, new_nodes, existing_nodes, neighbors_per_node) -> torch.Tensor` — top-k nearest existing nodes per new node → (B, K, DOF+1).
  - `get_unique_nodes(self, nodes, similarity_threshold) -> Tuple[torch.Tensor, torch.Tensor]` — dedup by threshold → (unique_nodes, inverse_indices).
  - `get_unique_nodes_zero_distance(self, nodes) -> Tuple[torch.Tensor, torch.Tensor]` — dedup at exact/zero-distance.
  - Static JIT: `jit_calculate_weighted_distance`, `jit_find_nearest_neighbors` (cdist + stable_topk), `jit_get_unique_nodes`, `jit_get_unique_nodes_zero_distance`.

### curobo/_src/graph_planner/graph/connector_linear.py
(inferred) Straight-line C-space connector — interpolates between configs at similarity-threshold resolution and returns the last collision-free node.

- **LinearConnector** — generates linear connections between configurations, checks collisions along paths, finds furthest feasible point.
  - `__init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None)`
  - `set_dependencies(self, action_dim: int, cspace_distance_weight: torch.Tensor, check_feasibility_fn)` — inject DOF, distance weights, feasibility fn.
  - `steer_until_infeasible(self, start_nodes, desired_nodes) -> torch.Tensor` — interpolate start→goal at `cspace_similarity_threshold` resolution, return last feasible (B, DOF+1) node per path.

### curobo/_src/graph_planner/graph/node_sampling_strategy.py
(inferred) Feasible configuration sampling — uniform (Halton), unit-ball, and informed-ellipsoid (householder/svd/approximate) with collision rejection.

- **NodeSamplingStrategy** — samples robot configurations for graph planning (uniform, in-ellipsoid, with feasibility filtering).
  - `__init__(self, config: PRMGraphPlannerCfg, action_lower_bounds, action_upper_bounds, cspace_distance_weight, action_dim, check_feasibility_fn, device_cfg=None)` — builds a Halton `SampleBuffer` + identity rotation frame.
  - `generate_action_samples(self, n_samples, bounded=True, unit_ball=False)` — Halton samples: bounded, unbounded, or projected to unit ball/sphere.
  - `check_samples_feasibility(self, action_samples)` — feasibility mask via injected fn.
  - `get_feasible_sample_set(self, x_samples)` — filter to collision-free samples.
  - `generate_feasible_action_samples(self, num_samples)` — oversample by `sample_rejection_ratio`, keep first `num_samples` feasible.
  - `generate_feasible_samples(self, num_samples) -> torch.Tensor` — same, for roadmap vertices.
  - `generate_feasible_samples_in_ellipsoid(self, x_start, x_goal, num_samples, max_sampling_radius) -> torch.Tensor` — informed-ellipsoid feasible samples (method chosen by `ellipsoid_projection_method`).
  - `compute_distance_from_line(self, vertices, x_start, x_goal)` — perpendicular distance of vertices to the start→goal segment.
  - `reset_seed(self)` — reset the Halton generator.
  - Static JIT: `jit_transform_unit_ball_to_ellipsoid_householder`, `jit_transform_unit_ball_to_ellipsoid_svd`, `jit_transform_unit_ball_to_ellipsoid_approximate`, `jit_compute_distance_from_line`.

### curobo/_src/graph_planner/search/path_finder_networkx.py
(inferred) NetworkX-backed roadmap graph with buffered edge/node insertion and shortest-path queries.

- **NetworkXPathFinder** — maintains an `nx.Graph`, buffers edges/nodes, and answers path-existence / shortest-path queries.
  - `__init__(self, seed: int = 42)`
  - `reset_graph(self)` — clear graph + edge/node buffers.
  - `reset_seed(self)` — reseed numpy + random (networkx has no per-graph seed reset).
  - `add_node(self, i)` / `add_nodes(self, node_list)` — buffer node(s).
  - `add_edge(self, start_i, end_i, weight)` / `add_edges(self, edge_list)` — buffer edge(s).
  - `update_graph(self)` — flush buffered edges/nodes into the graph.
  - `get_edges(self, attribue="weight")` — list of (u, v, weight) edges.
  - `path_exists(self, start_node_idx, goal_node_idx)` — `nx.has_path` after ensuring nodes exist.
  - `get_shortest_path(self, start_node_idx, goal_node_idx, return_length=False)` — `nx.bidirectional_dijkstra` (weight="weight"); returns path or (path, length).
  - `get_path_lengths(self, goal_node_idx)` — per-node shortest-path lengths from goal (`nx.shortest_path_length`), padded list.

### curobo/_src/graph_planner/search/path_pruner.py
(inferred) Path shortcut pruner — attempts direct connections between all node pairs on a path to bypass intermediate waypoints.

- **PathPruner** — prunes PRM paths by shortcutting from each node to other nodes in the path.
  - `__init__(self, config: PRMGraphPlannerCfg, device_cfg: Optional[DeviceCfg] = None)`
  - `set_dependencies(self, action_dim, cspace_distance_weight, preallocated_node_buffer, steer_and_register_edges_fn, find_path_for_index_pairs_fn)` — inject roadmap buffer + steer/find-path callbacks (raises if any None).
  - `prune_path_with_shortcuts(self, paths: List[List[int]], start_idx: List[int], goal_idx: List[int]) -> Tuple[List[List[int]], List[float]]` — register all-pairs shortcut edges, re-run shortest path, return (pruned_paths, lengths).
