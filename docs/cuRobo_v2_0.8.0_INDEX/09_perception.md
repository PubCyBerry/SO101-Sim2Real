# cuRobo v2 — Perception (TSDF/ESDF Mapper, segmentation, pose estimation)

Block-sparse volumetric mapping (`Mapper`), depth filtering, robot segmentation, and ICP/SDF pose detectors. `curobo/_src/perception/`.

Warp GPU-kernel modules (`wp_*.py`, everything under `mapper/kernel/`, `mapper/esdf/kernel/`, `mapper/marching_cubes/kernel/`) are listed compactly (module + kernel names only) — they are device code, not the Python API.

## Top-level

### curobo/_src/perception/filter_depth.py
- **FilterDepthConfig**(@dataclass) — `depth_minimum_distance`, `depth_maximum_distance: float`, `flying_pixel_threshold: Optional[float]`, `bilateral_kernel_size: Optional[int]`, `bilateral_sigma_spatial`, `bilateral_sigma_depth: float`.
- **FilterDepth** — `update_config`, `from_config`. (public `curobo.perception.FilterDepth`)

### curobo/_src/perception/robot_segmenter.py
- **RobotSegmenter** — segment the robot from depth via collision spheres. `from_robot_file`, `get_pointcloud_from_depth`, `update_camera_projection`, `get_robot_mask`, `get_robot_mask_from_active_js`, props `kinematics`, `base_link`.

### curobo/_src/perception/optim_pose_lm.py
Levenberg-Marquardt SE(3) utilities: `compute_predicted_reduction`, `trust_region_update`, `solve_lm_step`.

## Mapper (block-sparse TSDF/ESDF) — `_src/perception/mapper/`

### mapper.py
- **Mapper** — unified block-sparse volumetric mapper (public `curobo.perception.Mapper`).
  - `integrate(observation)` — fuse an RGB-D `CameraObservation` (batched cameras supported).
  - `compute_esdf()` — extract Euclidean SDF.
  - `update_static_obstacles(...)` — stamp primitives into the static channel.
  - `extract_mesh()`, `render{,_color,_depth,_color_only,_shaded,_depth_colormap,_normal_colormap}(...)`, `refine_pose(...)`, `reset()`, `get_stats()`, `memory_usage_mb()`. Props `integrator`, `tsdf`.

### mapper_cfg.py
- **MapperCfg**(@dataclass) — `extent_meters_xyz: Tuple`, `voxel_size: float`, `esdf_voxel_size: float`, `extent_esdf_meters_xyz`, `grid_center: Optional[torch.Tensor]`, `truncation_distance`, `minimum_tsdf_weight`, `depth_minimum_distance`, `depth_maximum_distance`, `decay_factor`, `frustum_decay_factor`, `rgb_scale: int`, `block_fill_ratio`, `hash_load_factor`, `roughness: float`, `integration_method: str`, `seeding_method: str`, `edt_solver: str`, `enable_static: bool`, `static_obstacle_color: Tuple`, `num_cameras: int`, `device: str`. Methods `grid_shape`, `block_size`, `max_blocks`, `hash_capacity`, `voxel_to_world`, `world_to_voxel`, `get_grid_bounds`.

### integrator_tsdf.py / integrator_esdf.py
- **BlockSparseTSDFIntegratorCfg** / **BlockSparseTSDFIntegrator** — `integrate`, `extract_mesh`, `recycle_empty_blocks`, `extract_surface/occupied_voxels`, `update_static_obstacles`, `get_stats`. (Cfg fields: `voxel_size, origin, truncation_distance, max_blocks, hash_capacity, depth_min/max_distance, frustum_decay, time_decay, minimum_tsdf_weight, grid_shape, roughness, image_height/width, enable_static, static_obstacle_color, integration_method, num_cameras, device`.)
- **BlockSparseESDFIntegratorCfg** / **BlockSparseESDFIntegrator** — `compute_esdf`, `get_voxel_grid`, `integrate`, `extract_occupied_voxels`. (Cfg adds `esdf_voxel_size, esdf_grid_shape, blend_esdf, use_cuda_graph, adjacent_skip_steps, seeding_method, edt_solver`.)

### storage.py / renderer.py / pose_refiner.py / mesh_extractor.py / block_allocation.py
- **storage.py** — **BlockSparseTSDFCfg** + **BlockSparseTSDFData** (23-field GPU hash/block store; `to_warp`) + **BlockSparseTSDF** (`data`, `reset`, `get_stats`, `compact_hash_table`, `memory_usage_mb`, `prepare_frame`).
- **renderer.py** — **BlockSparseTSDFRendererCfg** + **BlockSparseTSDFRenderer** (`render`, `render_color/depth/normals/shaded/...`); `depth_to_colormap`, `normals_to_colormap`.
- **pose_refiner.py** — **BlockSparseRefinementState**, **BlockSparseRaycastRefinerCfg**, **BlockSparseRaycastPoseRefiner** (`refine_pose`).
- **mesh_extractor.py** — host `extract_mesh_block_sparse(tsdf, level, surface_only, refine_iterations, minimum_tsdf_weight) -> (verts, tris, normals, colors)` + block-sparse Marching-Cubes wp kernels.
- **block_allocation.py** — `calculate_tsdf_max_blocks(grid_shape, voxel_size, block_dim, truncation_dist, roughness) -> int`.

### mapper/esdf/ (Euclidean Distance Transform)
- **edt_jump_flooding.py** — **JumpFloodingEDT** (`propagate`).
- **edt_parallel_banding.py** — **ParallelBandingEDT** (exact EDT; `propagate`).
- kernels (compact): `esdf/kernel/wp_jfa.py` (`jfa_propagate_kernel_18/26`, `smooth_esdf`), `esdf/kernel/wp_resample.py` (`seed_esdf_sites_from_tsdf_kernel`, `compute_decoupled_esdf_*`, `compute_esdf_from_min_tsdf_kernel`).

### mapper/marching_cubes/kernel/ (compact)
- `wp_mc_common.py` — `MCLookupTables`, `interpolate_edge_vertex`, `get_edge_vertex` (consts `TRIANGLE_TABLE`, `NUM_TRIANGLES_TABLE`, `EDGE_OWNER_OFFSETS`).
- `wp_mc_filter.py` — `count/compact_valid_triangles_kernel`, `filter_triangles`.
- `wp_mc_sampling.py` — `sample_sdf_with_weight`, `trilinear_sample_sdf_weighted`, `estimate_sdf_gradient_weighted`, `refine_vertex_weighted` (const `UNOBSERVED_SDF=1000.0`).

### mapper/kernel/ (block-sparse TSDF GPU core — compact)
- `warp_types.py` — `BlockSparseTSDFWarp` struct + hash consts (`HASH_EMPTY`, `HASH_TOMBSTONE`, `BLOCK_KEY_*`).
- `wp_hash.py` — hash-table ops (`hash_lookup`, `find_or_insert_block`, `read/write_tsdf_voxel`, `pack/unpack_*`).
- `wp_coord.py` — voxel↔world↔block coordinate conversions.
- `wp_integrate_sort_filter.py` — **SortFilterIntegrator** + `integrate_depth_sort_filter` + kernels.
- `wp_integrate_voxel_project.py` — **VoxelProjectIntegrator** + kernels.
- `wp_raycast.py` / `wp_raycast_common.py` / `wp_raycast_pose_refine.py` — raycast render/sample kernels.
- `wp_tsdf_sample.py`, `wp_decay.py`, `wp_esdf_seed.py`, `wp_filter_depth.py`, `wp_stamp_obstacles.py`, `wp_voxel_extraction.py`, `wp_integrate_common.py` — sampling / frustum decay / obstacle stamping / voxel extraction kernels.

### mapper/util/
- `utils_coords.py` — `voxel_to_world`, `world_to_voxel(_continuous)`, `get_grid_bounds`, `get_grid_extent`.
- `utils_quantization.py` — `pack/unpack_site_coords`, `get_sdf_from_float16_grids` (consts `SITE_COORD_BITS=10`).

## Pose estimation — `_src/perception/pose_estimation/`

- **pose_detector.py** — **PoseDetector** — point-to-plane ICP with Huber loss; `detect`, `detect_from_points`.
- **pose_detector_cfg.py** — **DetectorCfg**(@dataclass): `n_mesh_points_coarse`, `n_observed_points_coarse`, `n_rotation_samples`, `n_iterations_coarse`, `distance_threshold_coarse`, `n_mesh_points_fine`, `n_observed_points_fine`, `n_iterations_fine`, `distance_threshold_fine`, `use_svd: bool`, `use_huber_loss: bool`, `huber_delta: float`, `save_iterations: bool`, `device_cfg`.
- **sdf_pose_detector.py** — **SDFPoseDetector** (mesh-SDF alignment; `detect`, `detect_from_points`) + **SDFRefinementState**.
- **sdf_pose_detector_cfg.py** — **SDFDetectorCfg**(@dataclass): `max_iterations`, `inner_iterations`, `convergence_threshold`, `rotation_convergence_threshold`, `use_cuda_graph`, `distance_threshold`, `min_valid_ratio`, `use_huber`, `huber_delta`, `lambda_initial`, `lambda_factor`, `lambda_min`, `lambda_max`, `rho_min`, `n_points`, `device_cfg`.
- **mesh_robot.py** — **RobotMesh** (`from_trimesh`, `from_kinematics`, `update`, `sample_surface_points`, `get_trimesh`) + **SurfaceSampleCache**.
- **geometry.py** — **RigidObjectGeometry** (0-DOF), **ArticulatedRobotGeometry** (n-DOF) — `update`, `sample_surface_points`, `get_dof`.
- **detection_result.py** — **DetectionResult**(@dataclass): `pose: Pose`, `config`, `confidence: float`, `alignment_error: float`, `n_iterations: int`, `compute_time: float`.
- **util.py** — `extract_observed_points`, `omega_to_quaternion`, `huber_loss`, `find_nearest_neighbors`, `resample_points`, `compute_pose_point_to_plane_svd/cholesky`.
- **wp_mesh_sdf_alignment.py** (compact) — `mesh_surface_distance_query_kernel`, `jacobian_reduce_kernel`.
