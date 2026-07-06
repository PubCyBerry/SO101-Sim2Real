# cuRobo v2 — Robot Model (kinematics, dynamics, builder, parser)

The GPU kinematic/dynamic robot model, URDF/XRDF parsing, and the `RobotBuilder` authoring pipeline. `curobo/_src/robot/`.

## Kinematics — `_src/robot/kinematics/`

### curobo/_src/robot/kinematics/kinematics.py
Builds the GPU kinematic representation ("CUDA Accelerated Robot Model").
- **Kinematics** — differentiable FK with Jacobian, COM, collision spheres.
  - `compute_kinematics(joint_state, idxs_env=None) -> KinematicsState` — batched FK.
  - `get_robot_as_mesh`, `get_robot_as_spheres`, `get_link_poses`, `get_link_transform`, `get_all_link_transforms`, `get_link_mesh`, `get_self_collision_config`.
  - `get_full_js`, `get_mimic_js`, `get_active_js`, `update_kinematics_config`.
  - props: `tool_frames`, `joint_names`, `all_articulated_joint_names`, `dof` / `get_dof()`, `total_spheres`, `robot_spheres`, `base_link`, `default_joint_position`, `default_joint_state`, `lock_jointstate`, `kinematics_config`, `get_joint_limits()`.

### curobo/_src/robot/kinematics/kinematics_cfg.py
- **KinematicsCfg**(@dataclass) — `device_cfg: DeviceCfg`, `tool_frames: List[str]`, `kinematics_config: KinematicsParams`, `self_collision_config: Optional[SelfCollisionKinematicsCfg]`, `kinematics_parser: Optional[RobotParser]`, `generator_config: Optional[KinematicsLoaderCfg]`.
  - **`from_robot_yaml_file(file_path, tool_frames=None, device_cfg=DeviceCfg(), urdf_path=None, **kwargs) -> KinematicsCfg`** — load a cuRobo-format yml/xrdf (xrdf needs `urdf_path`). Primary entry point.
  - `from_content_path(content_path, tool_frames=None, device_cfg=DeviceCfg(), **kwargs)`, `from_basic_urdf(...)`, `from_config_file(...)`, `from_data_dict(...)`, `from_config(config: KinematicsLoaderCfg)`.
  - props `cspace -> CSpaceParams`, `dof -> int`, `get_joint_limits() -> JointLimits`.

### curobo/_src/robot/kinematics/kinematics_state.py
- **KinematicsState**(@dataclass) — `tool_poses: Optional[ToolPose]`, `tool_jacobians`, `robot_spheres`, `robot_com`, `robot_collision_geometry: Optional[RobotCollisionGeometry]`. Methods `tool_frames`, `get_link_spheres`, `clone`, `detach`, `copy_`.

### curobo/_src/robot/kinematics/kinematics_reducer.py
- **KinematicsReducer** — reduce DOF of a `KinematicsParams`; `reduce_dof`, `reconstruct_joint_state`.

## Builder & parser — `_src/robot/builder/`, `_src/robot/parser/`

### curobo/_src/robot/builder/builder_robot.py
High-level API for building/editing robot configs from URDF (the `curobo.robot_builder.RobotBuilder` public entry).
- **RobotBuilder**
  - `from_config(config_path, device_cfg)` — construct from a URDF+config.
  - **`fit_collision_spheres(sphere_density=1.0, surface_radius=0.002, fit_type=SphereFitType.MORPHIT, use_collision_mesh=False, iterations=200, coverage_weight=None, protrusion_weight=None, compute_metrics=False, clip_links=None) -> Dict[str, List[Dict]]`** — fit collision spheres per link.
  - `refit_link_spheres(...)`.
  - **`compute_collision_matrix(prune_collisions=True, num_samples=1000, batch_size=10000, seed=345, custom_ignore=None) -> Dict[str, List[str]]`** — build the self-collision ignore matrix.
  - `add_collision_ignore(...)`, `remove_collision_ignore(...)`.
  - **`build() -> KinematicsLoaderCfg`**, `save(...)`, `save_xrdf(...)`, `visualize(...)`.
  - props `tool_frames`, `collision_link_names`, `collision_spheres`, `collision_matrix`, `num_spheres`, `link_metrics`.

### curobo/_src/robot/builder/debugger_robot.py
- **RobotDebugger** — debug collision configs: `from_xrdf`, `check_default_joint_configuration_collision`, `check_collision_at_config`, `sample_collision_checks`, `find_never_colliding_pairs`, `visualize_collision_at_config`, `print_collision_matrix_stats`, props `robot_config`, `robot_model`.

### curobo/_src/robot/parser/parser_base.py + parser_urdf.py
- **RobotParser**(base) — `build_link_parent`, `get_link_parameters`, `add_absolute_path_to_link_meshes`, `get_link_mesh`, `get_link_geometry`, `get_chain`, `get_controlled_joint_names`, `get_link_names`.
- **UrdfRobotParser**(RobotParser) — parse kinematics from URDF; `get_urdf_string`, `root_link`, `get_link_names_from_urdf`, `get_joint_names_from_urdf`.

## Loader — `_src/robot/loader/`

### curobo/_src/robot/loader/kinematics_loader_cfg.py
- **KinematicsLoaderCfg**(@dataclass) — robot representation generator config (loaded from dict). Fields: `base_link: str`, `device_cfg: DeviceCfg`, `tool_frames: Optional[List[str]]`, `collision_link_names`, `collision_spheres: Union[None,str,Dict]`, `collision_sphere_buffer: Union[float,Dict[str,float]]`, `self_collision_buffer: Optional[Dict[str,float]]`, `self_collision_ignore: Optional[Dict[str,List[str]]]`, `debug`, `asset_root_path: str`, `mesh_link_names`, `grasp_contact_link_names`, `load_tool_frames_with_mesh: bool`, `urdf_path: Optional[str]`, `lock_joints: Optional[Dict[str,float]]`, `extra_links: Optional[Dict[str,LinkParams]]`, `add_object_link: bool`, `use_external_assets: bool`, `external_asset_path`, `external_robot_configs_path`, `extra_collision_spheres: Optional[Dict[str,int]]`, `load_collision_spheres: bool`, `num_envs: int`, `cspace: Union[None,CSpaceParams,Dict]`, `load_meshes: bool`, `use_global_cumul: bool`, `format_version: float`.

### curobo/_src/robot/loader/kinematics_loader.py + util.py
- **KinematicsLoader**(KinematicsLoaderCfg) — `kinematics_config`, `self_collision_config`, `kinematics_parser`, `initialize_tensors`, `add_link(link_params)`, `add_fixed_link`, `get_joint_limits`.
- `load_robot_yaml(content_path: ContentPath = ContentPath()) -> dict` — load robot repr from yaml or xrdf.

## Dynamics — `_src/robot/dynamics/`

### curobo/_src/robot/dynamics/dynamics.py + dynamics_cfg.py
- **Dynamics** — differentiable dynamics via native CUDA RNEA: `setup_batch_size`, `compute_inverse_dynamics(joint_state, f_ext)`, `update_link_mass`, `update_link_com`, `update_link_inertia`, `update_link_inertial`, `update_links_inertial`.
- **DynamicsCfg**(@dataclass) — `kinematics_config: KinematicsParams`, `device_cfg: DeviceCfg`, `gravity: List[float]`; `get_gravity_spatial()`.

## Data model — `_src/robot/types/`

### joint_types.py
- `JointType` {FIXED=-1, X_PRISM=0, Y_PRISM=1, Z_PRISM=2, X_ROT=3, Y_ROT=4, Z_ROT=5, X_PRISM_NEG=6, Y_PRISM_NEG=7, Z_PRISM_NEG=8, X_ROT_NEG=9, Y_ROT_NEG=10, Z_ROT_NEG=11}

### link_params.py
- **LinkParams**(@dataclass) — `link_name, joint_name, joint_type: JointType, fixed_transform, parent_link_name, child_link_name, joint_limits, joint_axis, joint_id, joint_velocity_limits, joint_offset, mimic_joint_name, joint_effort_limit, link_mass, link_com, link_inertia`. Methods `create(dict_data)`, `get_link_com_and_mass()`.

### joint_limits.py
- **JointLimits**(@dataclass) — `joint_names: List[str]`, `position/velocity/acceleration/jerk: torch.Tensor`, `effort: Optional[torch.Tensor]`, `device_cfg`. Methods `from_data_dict`, `clone`, `copy_`, `validate_shape`, props `position_lower_limits`, `position_upper_limits`.

### cspace_params.py
- **CSpaceParams**(@dataclass) — `joint_names, default_joint_position, cspace_distance_weight, null_space_weight, null_space_maximum_distance, device_cfg, max_acceleration, max_jerk, velocity_scale, acceleration_scale, jerk_scale, position_limit_clip`. Methods `inplace_reindex`, `copy_`, `clone`, `scale_joint_limits`, `load_from_joint_limits`.

### kinematics_params.py
- **KinematicsParams**(@dataclass) — the core GPU kinematics description. Tensor fields: `fixed_transforms, link_map, joint_map, joint_map_type, joint_offset_map, tool_frame_map, link_chain_data, link_chain_offsets, joint_links_data, joint_links_offsets, joint_affects_endeffector, link_spheres, link_sphere_idx_map, reference_link_spheres, link_masses_com, link_inertias, link_level_data, link_level_offsets`; plus `tool_frames: List[str]`, `joint_limits: JointLimits`, `non_fixed_joint_names`, `num_dof: int`, `mesh_link_names`, `joint_names`, `lock_jointstate`, `mimic_joints`, `link_name_to_idx_map`, `total_spheres: int`, `cspace: Optional[CSpaceParams]`, `base_link: str`, `grasp_contact_link_names`, `max_level_width: int`, `device_cfg`. Selected methods: `clone`, `make_contiguous`, `copy_`, `get_sphere_index_from_link_name`, `update_link_spheres`, `get_link_spheres`, `disable_link_spheres`, `enable_link_spheres`, `update_link_mass/com/inertia`, `get_robot_collision_geometry`, `num_links`, `num_spheres`, `num_envs`, `n_tree_levels`, `export_to_urdf`.

### collision_geometry.py / self_collision_params.py
- **RobotCollisionGeometry**(@dataclass) — `link_sphere_idx_map: torch.Tensor`, `num_links: int`; `clone`, `copy_`, `detach`.
- **SelfCollisionKinematicsCfg**(@dataclass) — self-collision CUDA-kernel attrs: `num_spheres: int`, `sphere_padding`, `collision_pairs`, `_num_checks_per_thread_large/small_collision_pairs`, `_max_threads_per_block_large/small_collision_pairs`. Methods `create_from_sphere_pair_distances`, `create_from_link_pairs`, `compute_sphere_pair_distance_with_link_pair_ignores`.
