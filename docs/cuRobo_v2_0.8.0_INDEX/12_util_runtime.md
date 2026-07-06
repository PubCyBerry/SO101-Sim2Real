# cuRobo v2 — Utilities, Runtime & Config I/O

File/config I/O, logging, trajectory interpolation, samplers, CUDA-graph/stream helpers, USD & Viser visualization, and the global runtime flags. `curobo/_src/util/`, `_src/runtime.py`, `_src/context.py`.

## Runtime flags — `_src/runtime.py`

Module-level flags (mutate to change behaviour; public via `curobo.runtime`):
- `torch_compile = False`, `torch_compile_slow = False`, `torch_jit = False`
- `cuda_graphs = True`, `cuda_graph_reset = False`, `cuda_streams = True`
- `cuda_event_timers = True`, `cuda_core_backend = True`, `kernel_backend = "auto"`
- `cache_dir = ~/.cache/curobo`
- `debug = False`, `debug_cuda_graphs = False`, `debug_cuda_compile = False`, `debug_nan = False`, `debug_timers = False`, `debug_trajopt = False`
- `profiler = False`

### curobo/_src/context.py
- **CuroboRuntime** — global runtime for kernel management (`get_cuda_core_cache`, `get_warp_cache`); `init()`, `get_runtime() -> CuroboRuntime`.

### curobo/_src/util_file.py
Re-export shim: `from ...util.config_io import *; from curobo.content import *`.

## Config I/O & logging — `_src/util/`
- **config_io.py** — `join_path`, `resolve_config`, `load_yaml`, `write_yaml`, `copy_file_to_path`, `get_filename`, `get_path_of_dir`, `get_files_from_dir`, `file_exists`, `merge_dict_a_into_b`, `is_platform_windows/linux`, `is_file_xrdf`, `create_dir_if_not_exists`. (public `curobo.config_io`)
- **logging.py** — `setup_logger`, `setup_curobo_logger`, `log_warn`, `log_debug`, `log_info`, `log_and_raise`, `deprecated`. (public `curobo.logging`)
- **version.py** — `get_version`.

## Trajectory — `_src/util/`
- **trajectory.py** — `TrajInterpolationType` {LINEAR, CUBIC, QUARTIC, QUINTIC, LINEAR_CUDA, BSPLINE_KNOTS_CUDA}; funcs `get_batch_interpolated_trajectory`, `get_cpu_linear_interpolation`, `linear_smooth`, `calculate_dt_no_clamp`, `calculate_traj_steps`, `get_interpolated_trajectory`.
- **trajectory_seed_generator.py** — **TrajectorySeedGenerator** (`generate_constant_seeds`, `generate_interpolated_seeds`, `generate_deceleration_seeds`); `interpolate_kernel`.
- **trajectory_execution_manager.py** — **TrajectoryExecutionManager** (`get_current_metrics`, `update_robot_state_trajectory`, `get_next_command`, `get_command_sequence`, `has_valid_next_command`, `get_action_buffer`).
- **state_filter.py** — **FilterCfg**(@dataclass, frozen: `filter_coeff: FilterCoeff`, `dt: float`, `control_space: ControlSpace`, `device_cfg`, `enable: bool`, `teleport_mode: bool`; `create`); **JointStateFilter** (`filter_joint_state`, `integrate_jerk/acc/vel/pos`, `reset`).

## Sampling — `_src/util/sampling/`
- **sequencer_base.py** — **BaseSequencer**(ABC): `random`, `reset`, `fast_forward`.
- **sequencer_halton.py / _random.py / _roberts.py** — **HaltonSequencer**, **RandomSequencer**, **RobertsSequencer**.
- **sample_buffer.py** — **SampleBuffer** (`reset`, `fast_forward`, `get_samples`, `get_gaussian_samples`, `bound_samples`, `gaussian_transform`, factories `create_halton/random/roberts_sample_buffer`).

## CUDA / Warp helpers — `_src/util/`
- **cuda_graph_util.py** — **GraphExecutor** (`warmup`, `reset`, `debug_dump`, `is_initialized`); `create_graph_executor`.
- **cuda_stream_util.py** — `cuda_stream_context`, `synchronize_cuda_streams`, `create_cuda_stream_pair`.
- **cuda_event_timer.py** — **CudaEventTimer** (`start`, `stop`). (public `curobo.profiling`)
- **torch_util.py** — `is_cuda_graph_available`, `is_torch_compile_available`, `get_torch_compile_options`, `get_torch_jit_decorator`, `get_profiler_decorator`, `profile_class_methods`, `empty_decorator`.
- **warp.py** — `init_warp`, `warp_support_sdf_struct`, `warp_support_kernel_key`, `warp_support_bvh_constructor_type`, `get_warp_device_stream`.
- **warp_interpolation.py** — `linear_interpolate_batch_dt_trajectory_kernel`, `get_cuda_linear_interpolation`.
- **warp_tile_mlp.py** — **WarpTileMLP**(torch.nn.Module) + **WarpTileMLPFunction**(autograd) + **WarpTileMLPTensors**.

## Tensor / misc — `_src/util/`
- **tensor_util.py** — `check_tensor_shapes`, `copy_tensor`, `copy_or_clone`, `clone_if_not_none`, `cat_sum`, `cat_max`, `tensor_repeat_seeds`, `fd_tensor`, `check_nan_last_dimension`, `shift_buffer`, `find_first/last_idx`, `round_away_from_zero`, `stable_topk` (Protocol `TensorLike`).
- **error_metrics.py** — `rotation_error_quaternion`, `rotation_error_matrix`.
- **benchmark_metrics.py** — **Statistic**(@dataclass: `mean, std, median, percent_25/75/98, min, max`), **CuroboMetrics** (per-trajectory: `skip, success, collision, joint_limit_violation, self_collision, position_error, orientation_error, ..., motion_time, solve_time, jerk, energy, torque, power`), **CuroboGroupMetrics** (aggregate; `from_list`, `print_summary`); `percent_true`.
- **helpers.py** — `default_to_regular`, `list_idx_if_not_none`, `robust_floor`. **python_util.py** — `ceildiv`.

## USD & Viser — `_src/util/`
- **usd_util.py** — `join_usd_path`, `set_prim_translate`, `set_prim_transform`, `get_prim_world_pose`, `get_transform`, `get_position_quat`, `create_stage`.
- **usd_scene_parser.py** — **UsdSceneParser** (`load_stage(_from_file)`, `get_pose`, `get_obstacles_from_stage`) + primitive attr getters.
- **usd_writer.py** — **UsdWriter** (public `curobo.viewer.UsdWriter`) — `create_stage`, `add_{cuboid,cylinder,sphere,mesh}_to_stage`, `add_world_to_stage`, `write_stage_to_file`, `create_animation`, `write_trajectory_animation(_with_robot_usd)`, `load_robot(_usd)`, `update_robot_joint_state`, `create_grid_usd`.
- **viser_visualizer.py** — **ViserVisualizer** (public `curobo.viewer.ViserVisualizer`) — `update_robot_spheres`, `add_frame`, `add_control_frame`, `get_control_frame_pose`, `set_joint_state`, `set_joint_positions`, `add_batched_spheres`, `add_sphere`, `add_line_segments`, `add_mesh`, `add_scene`, `add_point_cloud`, `add_image`, `reset_robot`, prop `joint_names`.
- **xrdf_util.py** — `return_value_if_exists`, `convert_xrdf_to_curobo`, `convert_curobo_to_xrdf`.

## Re-export shims
`util/__init__.py` (empty), `util/logger.py` (→ logging), `util/torch_utils.py` (→ torch_util), `util/xrdf_utils.py` (→ xrdf_util).
