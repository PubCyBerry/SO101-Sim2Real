# cuRobo v2 — Types, State & Transition

Core data types (Pose, ToolPose, JointState, ...) and state-transition models.

---

## types/

### curobo/_src/types/pose.py
Pose representation used in CuRobo (position + quaternion/rotation), the primary SE(3) type.

- **Pose**(Sequence) — batched rigid-body pose (position + wxyz quaternion, optional rotation matrix). `@dataclass`.
  - `position: Optional[T_BPosition] = None` — x, y, z in meters
  - `quaternion: Optional[T_BQuaternion] = None` — wxyz order
  - `rotation: Optional[T_BRotation] = None` — 3x3 rotation matrix
  - `batch_size: int = 1` — initialized from input
  - `name: str = "ee_link"` — link name this pose represents
  - `normalize_rotation: bool = False` — normalize quaternion on init (recommended for external sources)
  - Constructors (classmethods/staticmethods):
    - `from_matrix(matrix: Union[np.ndarray, torch.Tensor])` — build from 4x4 (or Nx4x4) homogeneous matrix
    - `from_euler_xyz(cls, euler_xyz: torch.Tensor, position: Optional[torch.Tensor] = None)` — extrinsic XYZ euler → Pose
    - `from_euler_xyz_intrinsic(cls, euler_xyz: torch.Tensor, position: Optional[torch.Tensor] = None)` — intrinsic XYZ euler (URDF joint-chain convention) → Pose
    - `from_numpy(cls, position: np.ndarray, quaternion: np.ndarray, device_cfg: DeviceCfg = DeviceCfg())`
    - `from_list(cls, pose: List[float], device_cfg: DeviceCfg = DeviceCfg(), q_xyzw=False)` — [x,y,z, q...] (q_xyzw reorders)
    - `from_batch_list(cls, pose: List[List[float]], device_cfg: DeviceCfg = DeviceCfg(), q_xyzw=False)`
  - Key methods:
    - `get_rotation() / get_rotation_matrix()` — 3x3 matrix (from rotation or quaternion)
    - `get_matrix(out_matrix=None) / get_affine_matrix(out_matrix=None)` — 4x4 tensor; `get_numpy_matrix() / get_numpy_affine_matrix()` — numpy variants
    - `get_pose_vector()` — cat(position, quaternion) [..., 7]
    - `inverse()` — inverse pose
    - `multiply(other_pose, out_position=None, out_quaternion=None)` — pose composition
    - `compute_offset_pose(offset) / compute_local_pose(world_pose)` — offset / relative pose
    - `distance(other_pose, use_phi3=False)` — (linear, angular) distance; `angular_distance(...)`, `linear_distance(...)`
    - `transform_points(points, out_buffer=None, gp_out=None, gq_out=None, gpt_out=None)`; `batch_transform_points(...)`; `batch_transform_points_inverse(...)`; deprecated `transform_point(...)`
    - `stack(other_pose) / repeat(n) / repeat_seeds(num_seeds) / cat(pose_list)` — batching ops
    - `unsqueeze(dim=-1) / squeeze(dim=-1) / get_index(b, n=None) / __getitem__ / __setitem__`
    - `clone() / detach() / contiguous() / copy_(pose) / to(device_cfg=None, device=None) / requires_grad_(bool)`
    - `apply_kernel(kernel_mat)` — matmul kernel onto position/quaternion
    - `to_list(q_xyzw=False) / tolist(q_xyzw=False)`
    - properties: `device`, `ndim`, `shape`, deprecated `batch`

### curobo/_src/types/tool_pose.py
Tool pose types for FK output and goal specification. ToolPose: 4D `[batch, horizon, num_links, 3/4]` FK output. GoalToolPose: 5D `[batch, horizon, num_links, num_goalset, 3/4]` goal-side only.

- **ToolPose**(Sequence) — 4D FK output (no goalset dim; current robot state). `@dataclass`.
  - `tool_frames: List[str]` — link/frame names, len == position.shape[2]
  - `position: torch.Tensor` — `[batch, horizon, num_links, 3]`
  - `quaternion: torch.Tensor` — `[batch, horizon, num_links, 4]` (wxyz)
  - Key methods:
    - `get_link_pose(link_name, make_contiguous=False)` — single link as 2D `Pose [B*H, 3/4]`
    - `to_dict(make_contiguous=True)` — Dict[str, Pose]
    - `reorder_links(ordered_tool_frames)` — new ToolPose with reordered/subset links
    - `as_goal(ordered_tool_frames=None)` — → GoalToolPose (adds num_goalset=1 dim)
    - `copy_(other) / clone() / detach() / contiguous() / requires_grad_(bool) / __getitem__ / __len__`
    - properties: `batch_size`, `horizon`, `num_links`, `shape`, `ndim`, `device`
- **GoalToolPose**(Sequence) — 5D goal spec `[batch, horizon, num_links, num_goalset, 3/4]` (goal/target side). `@dataclass`.
  - `tool_frames: List[str]` — len == position.shape[2]
  - `position: torch.Tensor` — `[batch, horizon, num_links, num_goalset, 3]`
  - `quaternion: torch.Tensor` — `[batch, horizon, num_links, num_goalset, 4]` (wxyz)
  - `from_poses(cls, pose_dict: Dict[str, Pose], ordered_tool_frames: Optional[List[str]] = None, num_goalset: int = 1)` — build from per-link Poses → `[batch, 1, num_links, num_goalset, 3/4]`
  - Key methods: `get_link_pose(link_name, make_contiguous=False)`, `to_dict(make_contiguous=True)`, `reorder_links(ordered_tool_frames)`, `copy_(other)`, `clone()`, `detach()`, `__getitem__`, `__len__`
  - properties: `batch_size`, `horizon`, `num_links`, `num_goalset`, `shape`, `ndim`, `device`

### curobo/_src/types/sequence_tool_pose.py
Time-first sequence of goal tool poses for retargeting. Layout `[num_frames, num_envs, num_links, num_goalset, 3/4]` (time-first for per-frame contiguity).

- **SequenceGoalToolPose** — time-series of goal tool poses for batch offline retargeting. `@dataclass`.
  - `tool_frames: List[str]` — link names, matches dim 2
  - `position: torch.Tensor` — `(num_frames, num_envs, num_links, num_goalset, 3)`
  - `quaternion: torch.Tensor` — `(num_frames, num_envs, num_links, num_goalset, 4)` wxyz
  - `get_frame(t: int)` — frame t as `GoalToolPose [num_envs, 1, num_links, num_goalset, 3/4]` (view, horizon=1)
  - `clone()`
  - properties: `num_frames`, `num_envs`, `num_links`, `num_goalset`, `device`

### curobo/_src/types/robot.py
Robot configuration to load in CuRobo.

- **RobotCfg** — top-level robot config (kinematics + optional dynamics + device). `@dataclass`.
  - `kinematics: KinematicsCfg` — kinematics config for cuda robot model (required)
  - `dynamics: Optional[DynamicsCfg] = None` — optional inverse-dynamics config
  - `device_cfg: DeviceCfg = DeviceCfg()`
  - `create(data: Union[Dict[str, Any], "RobotCfg"], device_cfg: DeviceCfg = DeviceCfg(), load_collision_spheres: bool = True, num_envs: int = 1)` — build from dict (or passthrough RobotCfg); dict may include `load_dynamics: bool`; unwraps `"robot_cfg"` key; skips collision spheres when `load_collision_spheres=False`
  - `from_basic(urdf_path: str, base_link: str, tool_frames: List[str], device_cfg: DeviceCfg = DeviceCfg(), load_dynamics: bool = False)` — build from basic URDF params
  - `write_config(file_path)` — serialize to YAML (stores `load_dynamics` flag, not full dynamics)
  - property `cspace -> CSpaceParams` (= kinematics.cspace)

### curobo/_src/types/camera.py
Camera observation container (rgb/depth/segmentation + intrinsics + pose) with pointcloud helpers. (inferred)

- **CameraObservation** — batched camera image bundle with projection/pointcloud utilities. `@dataclass`.
  - `name: str = "camera_image"`
  - `rgb_image: Optional[torch.Tensor] = None` — BxHxWxchannels
  - `depth_image: Optional[torch.Tensor] = None`
  - `image_segmentation: Optional[torch.Tensor] = None`
  - `projection_matrix: Optional[torch.Tensor] = None`
  - `projection_rays: Optional[torch.Tensor] = None`
  - `resolution: Optional[List[int]] = None`
  - `pose: Optional[Pose] = None`
  - `intrinsics: Optional[torch.Tensor] = None` — (b,3,3): [[fx,0,cx],[0,fy,cy],[0,0,1]]
  - `timestamp: Optional[torch.Tensor] = None`
  - `depth_to_meter: float = 0.001`
  - Key methods:
    - `filter_depth(distance=0.01)` — zero out depth below threshold (in-place)
    - `get_pointcloud(project_to_pose=False)` — depth+rays → pointcloud, optionally transformed to world
    - `update_projection_rays()` — recompute rays from intrinsics/depth
    - `extract_depth_from_structured_pointcloud(pointcloud, output_image=None)` — Z of structured (camera-frame) pointcloud → depth image
    - `stack(new_observation, dim=0) / copy_(new_data) / clone() / to(device) / save_to_file(file_path)`
    - property `shape` (rgb_image.shape)

### curobo/_src/types/content_path.py
Contains a class for storing file paths.

- **ContentPath** — root/absolute/relative paths for robot config + assets + scene. `@dataclass(frozen=True)`.
  - `robot_config_root_path: str = get_robot_configs_path()`
  - `robot_xrdf_root_path: str = get_robot_configs_path()`
  - `robot_urdf_root_path: str = get_assets_path()`
  - `robot_asset_root_path: str = get_assets_path()`
  - `scene_config_root_path: str = get_scene_configs_path()`
  - `world_asset_root_path: str = get_assets_path()`
  - `robot_config_absolute_path: Optional[str] = None`
  - `robot_xrdf_absolute_path: Optional[str] = None`
  - `robot_urdf_absolute_path: Optional[str] = None`
  - `robot_asset_absolute_path: Optional[str] = None`
  - `scene_config_absolute_path: Optional[str] = None`
  - `robot_config_file: Optional[str] = None` — relative; joined onto root → absolute in `__post_init__`
  - `robot_xrdf_file: Optional[str] = None`
  - `robot_urdf_file: Optional[str] = None`
  - `robot_asset_subroot_path: Optional[str] = None`
  - `scene_config_file: Optional[str] = None`
  - `get_robot_configuration_path()` — return robot config absolute path, falling back to XRDF
  - (`__post_init__` raises if a `*_file` and its `*_absolute_path` are both provided)

### curobo/_src/types/device_cfg.py
Device configuration for tensor operations.

- **DeviceCfg** — device + dtype settings for all tensor ops. `@dataclass(frozen=True)`.
  - `device: torch.device = torch.device("cuda", 0)`
  - `dtype: torch.dtype = torch.float32`
  - `collision_geometry_dtype: torch.dtype = torch.float32`
  - `collision_gradient_dtype: torch.dtype = torch.float32`
  - `collision_distance_dtype: torch.dtype = torch.float32`
  - `from_basic(device: str, dev_id: int)` — build DeviceCfg from device string + id
  - Key methods: `to_device(data_tensor)`, `to_int8_device(data_tensor)`, `cpu()`, `as_torch_dict()`, `is_same_torch_device(other)` (treats "cuda" == "cuda:0")

### curobo/_src/types/control_space.py
Enum of control spaces + spline helpers. (inferred)

- **ControlSpace**(Enum) `{POSITION=0, VELOCITY=1, ACCELERATION=2, BSPLINE_3=3, BSPLINE_4=4, BSPLINE_5=5}`
  - `bspline_types()` — [BSPLINE_3, BSPLINE_4, BSPLINE_5]
  - `position_types()` — [POSITION] + bspline_types()
  - `spline_degree(control_space)` — 3/4/5 for bsplines else 0
  - `spline_total_knots(control_space, action_knots)` — action_knots (+ degree + 1 for bsplines)
  - `spline_total_interpolation_steps(control_space, action_knots, interpolation_steps)` — total_knots * interpolation_steps + 1

### curobo/_src/types/tensor.py
Aliases for structured Tensors, improving readability. All are `torch.Tensor` aliases documenting shape:
`T_DOF` [dof], `T_BDOF` [batch, dof], `T_BHDOF_float` [batch, horizon, dof], `T_HDOF_float` [horizon, dof], `T_BHValue_bool`, `T_BValue_float`, `T_BHValue_float`, `T_BValue_bool`, `T_BValue_int`, `T_BPosition` [batch, 3], `T_BQuaternion` [batch, 4], `T_BRotation` [batch, 3, 3].

### curobo/_src/types/base.py
Deprecated shim — emits a deprecation warning; use `curobo.types.tensor` instead. No public API.

### curobo/_src/types/math.py
Deprecated shim — re-exports `curobo._src.types.pose` and warns; use `curobo.types.pose` instead.

### curobo/_src/types/file_path.py
Empty module (license header only). No public API.

### curobo/_src/types/__init__.py
Empty package init (license header only).

---

## state/

### curobo/_src/state/state_joint.py
Joint-space robot state (position and optional derivatives). Convention: `joint_state`/`js` for objects, `q` for raw position tensors.

- **JointState**(State) — batched joint state with position + velocity/acceleration/jerk + names + dt. `@dataclass`.
  - `position: Union[List[float], T_DOF]`
  - `velocity: Union[List[float], T_DOF, None] = None`
  - `acceleration: Union[List[float], T_DOF, None] = None`
  - `joint_names: Optional[List[str]] = None`
  - `jerk: Union[List[float], T_DOF, None] = None`
  - `device_cfg: DeviceCfg = DeviceCfg()`
  - `dt: Optional[torch.Tensor] = None`
  - `aux_data: dict = field(default_factory=lambda: {})`
  - `knot: Optional[torch.Tensor] = None`
  - `knot_dt: Optional[torch.Tensor] = None`
  - `control_space: Optional[ControlSpace] = None`
  - Constructors (staticmethods):
    - `from_numpy(joint_names, position, velocity=None, acceleration=None, jerk=None, device_cfg=DeviceCfg())` — missing derivatives filled with zeros
    - `from_position(position, joint_names=None)` — zero derivatives
    - `from_state_tensor(state_tensor, joint_names=None, dof=7)` — split packed [pos|vel|acc|jerk]
    - `from_list(position, velocity, acceleration, device_cfg: DeviceCfg())`
    - `zeros(size: Tuple[int], device_cfg: DeviceCfg, joint_names=None)`
  - Core methods: `to(device_cfg)`, `clone()`, `detach()`, `copy_(in_joint_state, allow_clone=True)`, `copy_reference(in_joint_state)`, `copy_data(...)` (deprecated)
  - Shape/index ops: `unsqueeze(idx)`, `squeeze(dim=0)`, `view(*shape)`, `__getitem__`, `__setitem__`, `__len__`, `data_ptr()`
  - Reorder: `reorder(joint_names)` (new), `reindex(joint_names)` (in-place)
  - properties (trivial getters): `device`, `dtype`, `shape`, `ndim`
  - Deprecated method wrappers (delegate to standalone ops): `blend(coeff, new_state)`, `get_state_tensor()`, `stack(new_state)`, `cat(other_js, dim)`, `repeat(repeat_input)`, `repeat_seeds(num_seeds)`, `apply_kernel(kernel_mat)`, `scale(dt)`, `scale_by_dt(dt, new_dt)`, `scale_time(new_dt)`, `calculate_fd_from_position(dt=None)`, `get_augmented_joint_state(joint_names, lock_joints=None)`, `append_joints(joint_state)`, `gather_by_seed_index(idx)`, `copy_only_index(...)`, `copy_at_index(...)`, `copy_at_batch_seed_indices(...)`, `get_trajectory_at_horizon_index(horizon_index)`, `trim_trajectory(start_idx, end_idx=None)`, `index_dof(idx)`

### curobo/_src/state/state_robot.py
RobotState — joint state + torque + FK/kinematics state. (inferred)

- **RobotState**(State) — bundles JointState with optional torque and cuda-robot-model (FK) state. `@dataclass`.
  - `joint_state: JointState`
  - `joint_torque: Optional[torch.Tensor] = None`
  - `cuda_robot_model_state: Optional[KinematicsState] = None`
  - Key methods:
    - `get_link_pose(link_name)` — Pose from tool_poses
    - `copy_at_batch_seed_indices(other, batch_idx, seed_idx)` — copy js + spheres + tool_poses + torque at [batch, seed] (handles merged batch*seed FK layout)
    - `copy_only_index(other, index)` — copy at indices
    - `clone() / copy_(other) / detach() / __getitem__ / __len__ / data_ptr()`
  - properties: `robot_spheres`, `link_poses`, `tool_poses` (all → cuda_robot_model_state), `tool_frames`
- `_kinematics_uses_merged_batch_seed_dim(joint_position, robot_spheres, tool_position=None)` — detect FK layout `[batch*num_seeds, horizon, ...]` vs joint `[B, S, ...]` (module-level, prefixed but load-bearing)

### curobo/_src/state/filter_coeff.py
Filter coefficients for blending joint states. (inferred)

- **FilterCoeff** — per-derivative blend weights. `@dataclass`.
  - `position: float = 0.0`
  - `velocity: float = 0.0`
  - `acceleration: float = 0.0`
  - `jerk: float = 0.0`

### curobo/_src/state/state_base.py
Base State dataclass. (inferred)

- **State**(Sequence) — empty base class (`pass`) for JointState/RobotState. `@dataclass`.

### curobo/_src/state/state_joint_ops.py
Operations on JointState objects (module-level functions; JointState methods delegate here).

- `blend_joint_states(target, new_state, coeff: FilterCoeff)` — blend two states via FilterCoeff (in-place on target)
- `joint_state_to_tensor(joint_state)` — pack [pos, vel, acc, jerk] → [..., 4*dof]
- `stack_joint_states(js1, js2)` — stack along second-to-last dim
- `cat_joint_states(js1, js2, dim)` — concatenate along dim (dim=-1 merges joint_names)
- `repeat_joint_state(joint_state, repeat_input: List[int])` — repeat along dims
- `repeat_joint_state_seeds(joint_state, num_seeds)` — repeat for multiple seeds
- `apply_kernel_to_joint_state(joint_state, kernel_mat)` — matmul kernel onto all fields
- `scale_joint_state(joint_state, dt)` — scale velocity·dt, accel·dt², jerk·dt³
- `scale_joint_state_by_dt(joint_state, dt, new_dt)` — rescale from dt to new_dt
- `scale_joint_state_time(joint_state, new_dt)` — rescale using stored dt
- `calculate_fd_from_position(joint_state, dt=None)` — finite-difference vel/acc/jerk (in-place)
- `reorder_joint_state(joint_state, ordered_joint_names)` — new state with reordered joints
- `reindex_joint_state_inplace(joint_state, joint_names)` — reorder in-place (None return)
- `augment_joint_state(joint_state, joint_names, lock_joints=None)` — append locked joints then reorder
- `append_joints_to_state(joint_state, other_js)` — append other's joints (handles 1D/2D/≥3D shapes)

### curobo/_src/state/state_joint_trajectory_ops.py
Trajectory operations on JointState objects (module-level functions).

- `gather_joint_state_by_seed(joint_state, idx)` — gather (batch, num_seeds, horizon, dof) by seed idx (batch, topk)
- `copy_joint_state_only_index(target, source, idx)` — copy fields at idx (in-place)
- `copy_joint_state_at_index(target, source, idx)` — write source into target at idx (None return)
- `copy_joint_state_at_batch_seed_indices(target, source, batch_idx, seed_idx)` — copy at [batch, seed] incl. knot/knot_dt
- `get_joint_state_at_horizon_index(joint_state, horizon_index)` — slice horizon dim
- `trim_joint_state_trajectory(joint_state, start_idx, end_idx=None)` — trim horizon range
- `index_joint_state_dof(joint_state, idx)` — index along DOF dim (subsets joint_names)

### curobo/_src/state/state_joint_jit_helpers.py
JIT helper functions for JointState operations (torch-jit accelerated internals).

- `jit_js_scale(vel, acc, jerk, dt, new_dt)` — scale derivatives by dt/new_dt powers (jit)
- `jit_get_index(position, velocity, acc, jerk, dt, idx: torch.Tensor)` — tensor-index all fields (jit)
- `fn_get_index(position, velocity, acc, jerk, dt, idx)` — non-jit index variant (slice-aware)
- `jit_get_index_int(position, velocity, acc, jerk, dt, idx: int)` — int-index all fields (jit)
- `jit_inplace_reindex(position, velocity, acceleration, jerk, knot, new_index)` — index_select on last dim (jit)
- `jit_joint_state_repeat_seeds(position, velocity, acceleration, jerk, dt, num_seeds)` — repeat across seeds (jit)
- `jit_joint_state_copy(position, velocity, acceleration, jerk, dt, in_position, ...)` — in-place copy_ of matched fields (jit)
- `clone_state_jit(position, velocity, acceleration, jerk, dt)` — clone-if-not-none tuple (jit)
- `trim_trajectory_jit(position, velocity, acceleration, jerk, start_idx, end_idx)` — slice horizon (jit-decorator commented out)

### curobo/_src/state/__init__.py
State types for robot joint and robot states. Package init, no public API.

---

## transition/

### curobo/_src/transition/robot_state_transition.py
RobotStateTransition — integrates actions into state sequences via control-space-specific step functions + FK/dynamics. (inferred)

- **RobotStateTransition** — action→state integrator; dispatches to StateFrom* step fns by control space, runs FK (and optional inverse dynamics), returns RobotState.
  - `__init__(config: RobotStateTransitionCfg)` — builds Kinematics, allocates state_seq buffer, picks `_rollout_step_fn`/`_cmd_step_fn` per control_space (ACCELERATION→StateFromAcceleration, POSITION→StateFromPositionTeleport if teleport_mode else StateFromPositionClique, BSPLINE_*→StateFromBSplineKnot; VELOCITY raises), sets up JointStateFilter
  - `forward(start_state, act_seq, start_state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, idxs_env=None) -> RobotState` — main entry: tensor_step then compute_augmented_state
  - `tensor_step(state, act, state_seq, state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None)` — rollout step fn forward
  - `robot_cmd_tensor_step(state, act, state_seq, state_idx=None, implicit_goal_state=None, implicit_goal_state_idx=None, use_implicit_goal_state=None)` — command step fn forward
  - `compute_augmented_state(state_seq, idxs_env=None) -> RobotState` — FK (+ inverse dynamics on separate CUDA streams) → RobotState
  - `get_robot_command(current_state, act_seq, shift_steps=1, state_idx=None, implicit_goal_state=None, implicit_goal_state_idx=None, use_implicit_goal_state=None) -> JointState` — extract executable command (full buffer or integrated)
  - `get_state_from_action(start_state, act_seq, state_idx=None) -> JointState` — full state seq from action trajectory
  - `get_action_from_state(state) -> torch.Tensor` — position/velocity/acceleration by control space
  - `integrate_action(act_seq)`, `integrate_action_step(act, dt)`, `filter_robot_state(current_state)`
  - `update_batch_size(batch_size, force_update=False)`, `update_cmd_batch_size(batch_size)`, `update_traj_dt(dt, base_dt=None, max_dt=None, base_ratio=None)`
  - `get_state_bounds()`, `get_full_dof_from_solution(q_js)` — re-add locked joints
  - Dynamics editing: `update_link_mass(link_name, mass)`, `update_link_inertial(link_name, mass=None, com=None, inertia=None)`, `update_links_inertial(link_properties)`
  - properties: `action_bound_lows`, `action_bound_highs`, `init_action_mean`, `get_init_action_mean()`, `default_joint_position`, `cspace_distance_weight`, `null_space_weight`, `null_space_maximum_distance`, `max_acceleration`, `max_jerk`, `max_velocity`, `action_horizon`, `horizon`, `n_knots`, `control_space`, `device_cfg`, `teleport_mode`, `return_full_act_buffer`, `state_finite_difference_mode`, `filter_robot_command`, `compute_inverse_dynamics`

### curobo/_src/transition/robot_state_transition_cfg.py
Config for RobotStateTransition + trajectory time parameters. (inferred)

- **TimeTrajCfg** — per-trajectory dt schedule (constant base_dt then blend to max_dt). `@dataclass`.
  - `base_dt: float`
  - `base_ratio: float`
  - `max_dt: float`
  - `get_dt_array(num_points)` — dt array (base_ratio fraction constant, remainder linspaced to max_dt)
  - `update_dt(all_dt=None, base_dt=None, max_dt=None, base_ratio=None)` — mutate dt params (all_dt sets base=max)
- **RobotStateTransitionCfg** — full transition config. `@dataclass(frozen=False)`.
  - `robot_config: RobotCfg`
  - `dt_traj_params: TimeTrajCfg`
  - `device_cfg: DeviceCfg`
  - `vel_scale: float = 1.0`
  - `state_estimation_variance: float = 0.0`
  - `batch_size: int = 1`
  - `horizon: int = 5`
  - `n_knots: int = 0`
  - `control_space: ControlSpace = ControlSpace.ACCELERATION`
  - `state_filter_cfg: Optional[FilterCfg] = None`
  - `teleport_mode: bool = False`
  - `return_full_act_buffer: bool = False`
  - `state_finite_difference_mode: str = "BACKWARD"`
  - `filter_robot_command: bool = False`
  - `interpolation_steps: int = 1`
  - `class_type: Type[RobotStateTransition] = RobotStateTransition`
  - `create(data_dict_in, robot_cfg: Union[Dict, RobotCfg], device_cfg=DeviceCfg())` — build from dict (parses TimeTrajCfg, ControlSpace, FilterCfg)
  - (`__post_init__` validates bspline params: interpolation_steps 1..32, n_knots > 5, no teleport; computes horizon from spline steps)

### curobo/_src/transition/fns_state_transition.py
State-integration step functions — action→state via CUDA trajectory kernels (module docstring: none). (inferred)

- **StateFromBase** — abstract base for step functions (holds batch_size/horizon/device, dt buffers).
  - `__init__(device_cfg: DeviceCfg, batch_size=1, horizon=1)`
  - `update_dt(dt)`, `update_batch_size(batch_size=None, horizon=None, force_update=False)`, abstract `forward(start_state, u_act, out_state_seq, start_state_idx=None, **kwargs) -> JointState`
- **StateFromPositionTeleport**(StateFromBase) — copies action directly into position (teleport, no integration).
  - `__init__(device_cfg, batch_size=1, horizon=1)`; `forward(...)` copies u_act → out_state_seq.position
- **StateFromAcceleration**(StateFromBase) — integrate acceleration action via AccelerationTensorStepIdxKernel.
  - `__init__(device_cfg, dt_h: torch.Tensor, dof: int, batch_size=1, horizon=1)`; requires `start_state_idx`
- **StateFromPositionClique**(StateFromBase) — position-clique integration via CliqueTensorStepIdxKernel; optional SMA filtering of vel/acc/jerk.
  - `__init__(device_cfg, dt_h, dof, filter_velocity=False, filter_acceleration=False, filter_jerk=False, batch_size=1, horizon=1)` — action_horizon = horizon-4; CUDA only
  - `forward(start_state, u_act, out_state_seq, start_state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs)`; `filter_signal(signal)`
- **StateFromBSplineKnot**(StateFromBase) — BSpline knot integration via BSplineIdxKernel (float32/CUDA only).
  - `__init__(device_cfg, dof, batch_size=1, horizon=1, n_knots=4, interpolation_steps=1, use_implicit_goal_state=False, control_space=ControlSpace.BSPLINE_4)` — n_knots>5; padded_horizon from spline steps
  - `forward(start_state, u_act, out_state_seq, start_state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs)`
- `filter_signal_jit(signal, kernel)` — conv1d SMA filter over [b, h, dof] with float dtype check (module-level; wraps jit core)

### curobo/_src/transition/__init__.py
State transition models for robot dynamics and integration. Package init, no public API.
