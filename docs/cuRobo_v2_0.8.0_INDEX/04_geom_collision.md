# cuRobo v2 — Geometry, Scene & Collision

Obstacle/scene types, sphere-fitting, differentiable transforms, and sphere-obstacle / robot-scene collision. `curobo/_src/geom/` + `curobo/_src/collision/`.

## Scene & obstacle types — `_src/geom/types.py`

The public `curobo.scene` module re-exports these (with `SceneCfg` exposed as `Scene`).

- **Obstacle**(@dataclass) — base for all obstacles. `name: str`, `pose: Optional[List[float]]`, `scale`, `color`, `texture_id`, `texture`, `material: Material`, `device_cfg`. Methods `get_trimesh_mesh`, `save_as_mesh`, `get_cuboid`, `get_mesh`, `get_transform_matrix`, `get_sphere`, `get_bounding_spheres`.
- **Cuboid**(Obstacle) — `dims: List[float]`.
- **Capsule**(Obstacle) — `radius: float`, `base: List[float]`, `tip: List[float]`.
- **Cylinder**(Obstacle) — `radius: float`, `height: float`.
- **Sphere**(Obstacle) — `radius: float`, `position: Optional[List[float]]`.
- **Mesh**(Obstacle) — `file_path`, `file_string`, `urdf_path`, `vertices`, `faces`, `vertex_colors`, `vertex_normals`, `face_colors`. Methods `to_cpu`, `to_gpu`, `get_mesh_data`, `from_pointcloud`.
- **PointCloud**(Obstacle) — `points`, `points_features`. `from_camera_observation(...)`.
- **VoxelGrid**(Obstacle) — `dims: List[float]`, `voxel_size: float`, `feature_tensor`, `xyzr_tensor`, `feature_dtype: torch.dtype`. Methods `get_grid_shape`, `create_xyzr_tensor`, `get_occupied_voxels`, `clone`.
- **Material**(@dataclass) — `metallic: float`, `roughness: float`.
- **SceneCfg**(@dataclass, Sequence) — "Representation of World for use in CuRobo" (public name **`Scene`**). Fields: `sphere: Optional[List[Sphere]]`, `cuboid`, `capsule`, `cylinder`, `mesh`, `voxel`, `objects: Optional[List[Obstacle]]`. Methods `create`, `create_obb_world`, `create_mesh_scene`, `create_merged_mesh_world`, `get_obb_world`, `get_mesh_world`, `get_collision_check_world`, `add_obstacle`, `get_obstacle`, `remove_obstacle`, `save_scene_as_mesh`, `get_cache_dict`, `randomize_color`, `add_material`, `remove_absolute_paths`.
- module funcs: `tensor_sphere`, `tensor_capsule`, `tensor_cube`, `batch_tensor_cube`.

## Transforms & math — `_src/geom/`
- **transform.py** — differentiable point/pose transforms (Warp-backed). Funcs: `transform_points`, `batch_transform_points`, `batch_transform_points_inverse`, `get_inv_transform`, `matrix_to_quaternion`, `quaternion_to_matrix`, `pose_to_matrix`, `pose_multiply`, `pose_inverse`, `quaternion_rate_to_axis_angle_rate`. autograd Functions: `TransformPoint`, `BatchTransformPoint`, `BatchTransformPointInverse`, `BatchTransformPose`, `TransformPose`, `PoseInverse`, `QuatToMatrix`, `MatrixToQuaternion`.
- **quaternion.py** — `normalize_quaternion`, `quat_multiply`, `angular_distance_phi3`, `angular_distance_axis_angle`.
- **cv.py** — depth↔pointcloud: `project_depth_to_pointcloud`, `get_projection_rays`, `extract_depth_from_structured_pointcloud`, `project_depth_using_rays`.
- **sdf_grid.py** — deprecated SDF grid: `lookup_distance`, `compute_sdf_gradient`, `SDFGrid`(autograd).
- **convex_polygon_helper.py** — **ConvexPolygon2DHelper** (`build_convex_hull`, `compute_point_hull_distance`).

## GPU obstacle data — `_src/geom/data/`
Warp-side storage for each obstacle type (used by the collision checker).
- **data_scene.py** — **SceneData**(@dataclass): `cuboids: Optional[CuboidData]`, `meshes: Optional[MeshData]`, `voxels: Optional[VoxelData]`, `num_envs`, `device_cfg`, `scene_model`. Methods `from_scene_cfg`, `from_batch_scene_cfg`, `add_obstacle`, `update_obstacle_pose`, `enable_obstacle`, `get_obstacle_names`, `has_cuboids/meshes/voxels`, `to_warp`. Also `SceneDataWarp` struct.
- **data_cuboid.py** — **CuboidData** (`from_scene_cfg`, `add`, `update_pose`, `update_dims`, `set_enabled`, `to_warp`) + `CuboidDataWarp` struct + SDF `@wp.func`s (`compute_local_sdf(_with_grad)`).
- **data_mesh.py** — **MeshData** + **WarpMeshCache** + `MeshDataWarp` struct + SDF wp funcs.
- **data_voxel.py** — **VoxelData** (`create_from_voxel_grids`, `update_features`, `get_voxel_grid`, `to_warp`) + `VoxelDataWarp` struct + voxel-SDF wp funcs.
- **helper_pose.py** — wp funcs `get_obs_idx`, `load_inv_position/quat`, `load_transform_from_inv_pose`.

## Sphere-obstacle collision — `_src/geom/collision/`
- **collision_scene.py** — **SceneCollisionCfg**(@dataclass): `device_cfg`, `scene_model: Optional[Union[SceneCfg,List[SceneCfg]]]`, `num_envs: int`, `max_distance: float`, `cache: Optional[Dict[str,int]]`. **SceneCollision**(@dataclass): `data: SceneData`, `checker: CollisionChecker`, `scene_model`. Methods `from_config`, `get_sphere_distance`, `get_swept_sphere_distance`, `get_sphere_collision`, `load_collision_model`, `update_obstacle_pose`, `enable_obstacle`, `get_obstacle_names`, `clear_cache`, `update_voxel_data`, `get_voxel_grid`. `create_scene_collision(config) -> SceneCollision`.
- **checker_collision.py** — **CollisionChecker**(@dataclass): `device_cfg`, `max_distance`. Methods `get_sphere_distance`, `get_swept_sphere_distance`, `get_sphere_collision`, `get_swept_sphere_collision`.
- **buffer_collision.py** — **CollisionBuffer**(@dataclass): `distance`, `gradient`, `shape`, `device_cfg`; `from_shape`, `zero_`, `resize`, `clone`.
- **wp_autograd.py** — `SphereObstacleCollision`, `SweptSphereObstacleCollision` (autograd).
- warp kernels (compact): **wp_collision_kernel.py** `sphere_obstacle_collision_kernel`; **wp_sweep_collision_kernel.py** `swept_sphere_obstacle_collision_kernel` (`SWEEP_STEPS=3`); **wp_collision_common.py** `apply_collision_activation`, `load_sphere_query`, `accumulate_collision` (struct `SphereQueryData`); **wp_speed_metric.py** `apply_speed_metric`.

## Sphere fitting — `_src/geom/sphere_fit/`
- **types.py** — `SphereFitType` {SURFACE='surface', VOXEL='voxel', MORPHIT='morphit'}; **SphereFitMetrics**(@dataclass: `num_spheres, coverage, protrusion, protrusion_dist_mean/p95, surface_gap_mean/p95, max_uncovered_gap, volume_ratio`); **SphereFitResult**(@dataclass: `centers, radii, num_spheres, metrics, fit_time_s, used_mesh, history, debug_info`).
- **fit_spheres.py** — **`fit_spheres_to_mesh(mesh, num_spheres, sphere_density, surface_radius, fit_type, iterations, compute_metrics, coverage_weight, protrusion_weight, clip_plane, device_cfg) -> SphereFitResult`** (public `curobo.sphere_fit`).
- **fit_morphit.py** — **MorphItConfig**(@dataclass: `num_spheres, device, num_inside_samples, iterations, center_lr, radius_lr, grad_clip_norm, loss_weights: MorphItLossWeights, density_control_interval, radius_threshold_ratio, coverage_threshold_ratio, max_spheres, clip_plane, clip_plane_buffer, verbose_frequency`) + **MorphItLossWeights** (`coverage, protrusion, tangency, overlap, halfplane, protrusion_samples`); `morphit_sphere_fit(...)`.
- **fit_voxel.py** — `sample_even_fit_mesh`, `voxel_fit_mesh`.
- **sphere_count.py** — `estimate_sphere_count(...)` (consts `_BASE_DENSITY=1/15`, `_MIN_SPHERES=3`, `_MAX_SPHERES=100`).
- **metrics.py** — `compute_sphere_fit_metrics`, `populate_metrics`.
- **wp_mesh_query.py** — **WarpMeshQuery** (`query_outside_mask`, `query_sdf`, `query_closest_point`); **WarpSphereSDFFunction**(autograd).

## Robot-scene collision — `_src/collision/`
Differentiable, robot-level collision (public `curobo.collision_checking.RobotCollisionChecker`).
- **collision_robot_scene_cfg.py** — **RobotSceneCollisionCfg**(@dataclass): `kinematics: Kinematics`, `sampler: SampleBuffer`, `bound_scale: torch.Tensor`, `cspace_cost: PositionCSpaceCost`, `self_collision_cost: Optional[SelfCollisionCost]`, `collision_cost: Optional[SceneCollisionCost]`, `collision_constraint: Optional[SceneCollisionCost]`, `scene_model: Optional[SceneCollision]`, `rejection_ratio: int`, `device_cfg`, `contact_distance: float`. Method `load_from_config(...)`.
- **collision_robot_scene.py** — **RobotSceneCollision**(RobotSceneCollisionCfg) — `setup_batch_tensors`, `get_kinematics`, `update_world`, `clear_scene_cache`, `get_collision_distance`, `get_collision_constraint`, `get_self_collision_distance`, `get_self_collision`, `get_collision_vector`, `get_scene_self_collision_distance_from_joints`, `get_bound`, `sample`, `validate`, `sample_trajectory`, `validate_trajectory`, `pose_distance`, `get_point_robot_distance`, `get_active_js`, prop `tool_frames`.
- **attachment_manager.py** — **AttachmentManager** — attach/detach obstacles to robot links: `kinematics_params`, `fit_spheres`, `update`, `attach`, `attach_from_scene`, `detach`.
