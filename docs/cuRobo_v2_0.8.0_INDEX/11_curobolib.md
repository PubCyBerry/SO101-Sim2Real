# cuRobo v2 — curobolib (CUDA op Python wrappers)

The torch.autograd.Function op wrappers and the cuda-core launch layer (Python-facing only).

---

## cuda_ops/ — torch.autograd.Function wrappers (most-used)

These wrap the backend `launch_*` kernels in autograd Functions and validate tensors before each launch.

### curobo/_src/curobolib/cuda_ops/kinematics.py
(inferred) Autograd wrapper for the fused forward/inverse kinematics CUDA kernel (link poses, robot spheres, jacobian, CoM).
- **KinematicsFusedFunction**(Function) — fused batched kinematics forward + backward; forward fills preallocated buffers, backward returns joint-space gradient.
  - `create_buffers(batch: int, horizon: int, kinematics_config: KinematicsParams, device_cfg: DeviceCfg = DeviceCfg())` — allocate all forward+backward output/grad buffers; returns dict keyed `batch_link_position`, `batch_link_quaternion`, `batch_robot_spheres`, `batch_com`, `batch_jacobian`, `batch_cumul_mat`, `grad_out_q`, `grad_out_q_jacobian`, `grad_in_link_pos`, `grad_in_link_quat`, `grad_in_robot_spheres`, `grad_in_com`.
  - `forward(ctx, joint_seq, batch_link_position, batch_link_quaternion, batch_robot_spheres, batch_com, batch_jacobian, batch_cumul_mat, kinematics_config, grad_out, grad_out_q_jacobian, grad_in_link_pos, grad_in_link_quat, grad_in_robot_spheres, grad_in_com, compute_jacobian, compute_spheres, compute_com, env_query_idx, horizon)` — runs `check_*_tensors` on all KinematicsParams tensors, `kinematics_config.validate_shapes()`, then `kinematics_cu.launch_kinematics_forward(...)`; returns `(batch_link_position, batch_link_quaternion, batch_robot_spheres, batch_com, batch_jacobian)`.
  - `backward(ctx, grad_in_link_pos, grad_in_link_quat, grad_in_spheres, grad_in_com, grad_in_link_jacobian)` — `@once_differentiable`; falls back to ctx-stored grad buffers when upstream grads are None, requires `grad_in_link_quat` 16-byte aligned, calls `launch_kinematics_backward` and (if `compute_jacobian`) `launch_kinematics_jacobian_backward`, summing into `grad_joint`.

### curobo/_src/curobolib/cuda_ops/geometry.py
(inferred) Autograd wrapper for the self-collision distance CUDA kernel (world collision uses Warp, not this).
- **SelfCollisionDistance**(torch.autograd.Function) — max self-collision penetration distance over sphere pairs; gradient is the precomputed collision vector.
  - `forward(ctx, robot_spheres, out_distance, out_vec, pair_distance, sparse_idx, weight, sphere_padding, pair_locations, block_batch_max_value, block_batch_max_index, num_blocks_per_batch, max_threads_per_block, store_pair_distance, return_loss)` — validates tensors (float32 / uint8 `sparse_idx` / int16 `pair_locations`,`block_batch_max_index`), calls `geometry_cu.self_collision_distance(...)`, saves `(out_vec, out_distance)`; returns `out_distance`.
  - `backward(ctx, grad_out_distance)` — `@once_differentiable`; returns saved `out_vec` (× upstream grad if `return_loss`) as sphere gradient, rest None.

### curobo/_src/curobolib/cuda_ops/optimization.py
(inferred) Wolfe line-search driver + LBFGS step autograd Function.
- `wolfe_line_search(iteration_state: OptimizationIterationState, line_search_context: LineSearchContext, exploration_idx, selected_idx, search_cost, search_action, search_gradient, step_direction, strong_wolfe: bool, approx_wolfe: bool)` — validate all iteration-state/context tensors + exhaustive shape checks, then `optimization_cu.launch_line_search(...)` to update best/current/converged state; returns `(iteration_state, exploration_idx, selected_idx)`.
- **LBFGScu**(Function) — one L-BFGS two-loop-recursion step producing a search direction.
  - `forward(ctx, step_vec, rho_buffer, y_buffer, s_buffer, q, grad_q, x_0, grad_0, epsilon=0.1, stable_mode=False, use_shared_buffers=True)` — checks float32 tensors, derives `m, b, v_dim` from `y_buffer.shape`, calls `optimization_cu.launch_lbfgs_step(...)`, returns step vector reshaped to `step_vec.shape`.
  - `backward(ctx, grad_output)` — returns 6× None (non-differentiable).

### curobo/_src/curobolib/cuda_ops/trajectory.py
(inferred) B-spline interpolation helper + three trajectory-parameterization autograd Functions (clique, acceleration integration, b-spline).
- `get_bspline_interpolation(input_trajectory: JointState, output_trajectory: JointState, interpolation_dt, current_state: JointState, goal_state: JointState, start_idx, goal_idx, use_implicit_goal_state, interpolated_horizon, bspline_degree=4)` — validates all JointState tensors, calls `trajectory_cu.launch_bspline_interpolation_single_dt_kernel(...)`; returns `output_trajectory`.
- **CliqueTensorStepIdxKernel**(torch.autograd.Function) — finite-difference (clique/stencil) mapping of action deltas → position/velocity/acceleration/jerk with start+goal indices.
  - `forward(ctx, u_act, start_position, start_velocity, start_acceleration, goal_position, goal_velocity, goal_acceleration, start_idx, goal_idx, out_position, out_velocity, out_acceleration, out_jerk, out_dt, traj_dt, use_implicit_goal_state, out_grad_position)` — asserts `u_act.shape[-2] == horizon-4`, calls `launch_differentiation_position_forward_kernel(...)`; returns `(out_position, out_velocity, out_acceleration, out_jerk)`.
  - `backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j)` — calls `launch_differentiation_position_backward_kernel(...)`, returns `out_grad_position` for `u_act` grad.
- **AccelerationTensorStepIdxKernel**(torch.autograd.Function) — integrate acceleration action → pos/vel/acc/jerk (forward-only; backward raises).
  - `forward(ctx, u_act, start_position, start_velocity, start_acceleration, start_idx, out_position, out_velocity, out_acceleration, out_jerk, traj_dt, out_grad_position)` — calls `launch_integration_acceleration_kernel(..., True)`; returns 4 tensors.
  - `backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j)` — `raise NotImplementedError()` if grad needed for `u_act`.
- **BSplineIdxKernel**(torch.autograd.Function) — differentiable B-spline knot→trajectory expansion with boundary states.
  - `forward(ctx, u_act, start_position, start_velocity, start_acceleration, start_jerk, goal_position, goal_velocity, goal_acceleration, goal_jerk, start_idx, goal_idx, out_position, out_velocity, out_acceleration, out_jerk, out_dt, traj_dt, use_implicit_goal_state, out_grad_position, bspline_degree, use_flat_gradient=False)` — calls `launch_bspline_interpolation_forward_kernel(...)`; saves `n_knots`, `bspline_degree`, `use_flat_gradient`; returns 4 tensors.
  - `backward(ctx, grad_out_p, grad_out_v, grad_out_a, grad_out_j)` — validates padded-horizon consistency across grads, calls `launch_bspline_interpolation_backward_kernel(...)`; returns `out_grad_position` for `u_act` grad.

### curobo/_src/curobolib/cuda_ops/dynamics.py
PyTorch autograd function for RNEA inverse dynamics (wraps the CUDA RNEA forward kernel). Const `_CACHE_FLOATS_PER_LINK = 20`.
- **RNEAForwardFunction**(torch.autograd.Function) — differentiable inverse dynamics τ = RNEA(q, q̇, q̈, f_ext).
  - `create_buffers(batch_size, num_links, ...)` — allocate the forward-cache (`(batch_size, num_links * 20)`) + output buffers.
  - `forward(ctx, q, qd, qdd, tau, ..., gravity, level_starts, level_links, ...)` — call `launch_rnea_forward(...)`; saves intermediates to `forward_cache`.
  - `backward(ctx, grad_tau)` — call `launch_rnea_backward(...)`; returns VJP dL/dq, dL/dq̇, dL/dq̈ (and dL/df_ext when external forces present).

### curobo/_src/curobolib/cuda_ops/tensor_checks.py
Tensor validation utilities for kernel launches. (Every kernel-bound tensor must be on the expected device, contiguous, and correct dtype; `.contiguous()` fallback is unsafe under CUDA graphs, so these assert instead.)
- `check_float32_tensors(device, **tensors) — validate device + contiguous + float32.`
- `check_float16_tensors(device, **tensors) — validate device + contiguous + float16.`
- `check_int8_tensors(device, **tensors) — validate device + contiguous + int8.`
- `check_uint8_tensors(device, **tensors) — validate device + contiguous + uint8.`
- `check_int16_tensors(device, **tensors) — validate device + contiguous + int16.`
- `check_int32_tensors(device, **tensors) — validate device + contiguous + int32.`
- `check_bool_tensors(device, **tensors) — validate device + contiguous + bool.`
- (`_check_tensors(device, expected_dtype, **tensors)` — private core; None/device/contiguity/dtype checks via `log_and_raise`.)

### curobo/_src/curobolib/cuda_ops/__init__.py
Python bindings for curobolib kernels. (Docstring-only package marker.)

---

## backends/ — backend selection

### curobo/_src/curobolib/backends/__init__.py
Backend selector for curobolib kernels. Supports two backends — `'cuda_core'` (runtime cuda.core compilation, default) and `'pybind'` (pre-compiled PyBind11). Selection priority: env `CUROBO_KERNEL_BACKEND` → config `cuda_core_backend` → auto-detect with fallback. Module-level `__getattr__` returns a lazy `_BackendProxy` so `from ...backends import kinematics as kinematics_cu` succeeds even with no backend installed at import time (selection deferred to first attribute access).
- `get_backend() -> dict` — return active backend module dict (kinematics/optimization/trajectory/geometry/dynamics/pba), initializing + logging backend name on first call.
- `get_backend_name() -> str` — return active backend name (`'cuda_core'` / `'pybind'`), initializing if needed.
- **_BackendProxy** — lazy per-module proxy; `_MODULES = frozenset({"kinematics", "optimization", "trajectory", "geometry", "dynamics", "pba"})`; raises via `log_and_raise` if the active backend lacks the module (dynamics/pba are cuda_core-only).
- (private: `_try_load_cuda_core_backend`, `_try_load_pybind_backend`, `_load_cuda_core_backend`, `_load_pybind_backend`, `_auto_select_backend`.)

### curobo/_src/curobolib/backends/pybind/__init__.py
PyBind backend for CuRobo kernels. (Docstring-only; pre-compiled extensions loaded lazily; not present in this tree.)

---

## backends/cuda_core_backend/ — cuda.core launch layer

### curobo/_src/curobolib/backends/cuda_core_backend/__init__.py
cuda.core backend for CuRobo kernels with runtime compilation. Re-exports kernel modules (`dynamics`, `geometry`, `kinematics`, `optimization`, `pba`, `trajectory`) plus utilities `CudaCoreKernelCache`, `get_cuda_home`, `CudaCoreKernelCfg`.

### curobo/_src/curobolib/backends/cuda_core_backend/kernel_cache.py
Kernel cache for runtime CUDA kernel compilation using cuda.core.
- `get_cuda_home() -> Optional[str]` — CUDA header dir via `cuda.pathfinder.find_nvidia_header_directory("nvrtc")`.
- **CudaCoreKernelCache** — compiles + caches CUDA kernels via cuda.core, manages device/arch, wraps PyTorch streams.
  - Instance attrs (set in `__init__`): `compiled_kernels: Dict[str, any] = {}`, `device: Optional[any] = None`, `arch: Optional[str] = None`.
  - `initialize()` — create `cuda.core.Device()`, `set_current()`, set `arch = f"sm_{device.arch}"`.
  - `get_stream_wrapper(torch_stream)` — wrap a PyTorch stream in a cuda.core Stream (via inner `PyTorchStreamWrapper` exposing `__cuda_stream__`); no new stream created.
  - `get_kernel_hash(source_files: List[Path], kernel_name: str, compile_flags: List[str]) -> str` — SHA256 over file contents + kernel name + flags + arch (cache key).
  - `get_or_compile_kernel(source_files: List[Path], kernel_name: str, include_dirs: List[Path], compile_flags: List[str])` — cache lookup by hash, else compile; `torch.cuda.synchronize()` after compile so cubin is fully loaded.
  - (private `_compile_kernel`: NVRTC via cuda.core `Program`/`ProgramOptions` — `std="c++17"`, `arch`, `ftz=True`, `fma=True`, `prec_div=False`, `prec_sqrt=False`, `lineinfo=True`, `device_code_optimize=True`; `debug=True` if `debug_cuda_compile`; compiles to `"cubin"`. `_read_sources` prepends stdint typedefs + `#define CUDA_CORE_COMPILE true`.)

### curobo/_src/curobolib/backends/cuda_core_backend/kernel_config.py
Base configuration for CUDA kernel compilation using cuda.core.
- **CudaCoreKernelCfg** — base config; `__init__(self, kernel_subdir: str)` sets `_kernel_dir = <curobolib>/kernels/<kernel_subdir>`.
  - `kernel_dir` (property) — the kernels subdirectory Path.
  - `get_compile_flags(debug: bool = False) -> List[str]` — debug flags `["-G", "-g", "--generate-line-info", "--device-debug"]` else release `["-O3", "--ftz=true", "--fmad=true", "--prec-div=false", "--prec-sqrt=false", "--generate-line-info"]`.
  - `get_base_include_dirs() -> List[Path]` — `[kernels/, kernels/common, kernels/third_party]`.

### curobo/_src/curobolib/backends/cuda_core_backend/launch_helper.py
(inferred) Thin cuda.core kernel-launch wrapper with optional debug sync + error check.
- `launch_kernel(kernel_name, stream, config, kernel, *kernel_args)` — `cuda.core.launch(...)`; if `runtime.debug` sync stream; raise via `log_and_raise` on nonzero `cudaGetLastError()`.

### curobo/_src/curobolib/backends/cuda_core_backend/util.py
(inferred) Small integer helper.
- `ceil_div(a: int, b: int) -> int` — ceiling division `(a + b - 1) // b`.

---

### curobo/_src/curobolib/backends/cuda_core_backend/kinematics.py
cuda.core backend for kinematics kernels (same signatures as PyBind11 for backend swapping).
- `launch_kinematics_forward(link_pos, link_quat, batch_robot_spheres, batch_center_of_mass, batch_jacobian, global_cumul_mat, joint_vec, fixed_transform, robot_spheres, link_masses_com, link_map, joint_map, joint_map_type, tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets, joint_links_data, joint_links_offsets, joint_affects_endeffector, joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints, num_spheres, compute_jacobian, compute_com)` — single fused kernel when `num_spheres < 100` (`kinematics_fused[_jacobian]_kernel`), else two-kernel path (`kinematics_cumul_kernel` + `kinematics_spheres_links[_jacobian]_kernel`); in-place outputs.
- `launch_kinematics_backward(grad_out, grad_nlinks_pos, grad_nlinks_quat, grad_spheres, grad_center_of_mass, batch_center_of_mass, global_cumul_mat, joint_vec, fixed_transform, robot_spheres, link_masses_com, link_map, joint_map, joint_map_type, tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets, joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints, num_spheres, compute_com)` — launches `kinematics_fused_backward_unified_kernel<float, float, MAX_JOINTS, use_warp_reduce, compute_com>`; joint grad written into `grad_out` in-place.
- `launch_kinematics_jacobian_backward(grad_joint, grad_jacobian, global_cumul_mat, joint_map_type, joint_map, link_map, link_chain_data, link_chain_offsets, joint_links_data, joint_links_offsets, joint_affects_endeffector, tool_frame_map, joint_offset_map, batch_size, n_joints, n_tool_frames)` — launches `kinematics_jacobian_gradient_backward_kernel<float, MAX_JOINTS, use_warp_reduce>`; accumulates into `grad_joint`.

### curobo/_src/curobolib/backends/cuda_core_backend/kinematics_config.py
(inferred) Kinematics kernel compile config + launch-config calculators.
- **KinematicsKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"kinematics"`.
  - `get_kernel_files(kernel_type) -> List[str]` — maps `"forward"→["kinematics_forward_kernel.cuh"]`, `"backward"→["kinematics_backward_kernel.cuh"]`, `"jacobian_backward"→["kinematics_jacobian_backward_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base dirs + `kernels/kinematics/`.
- **KinematicsLaunchCfg** — static launch-config calculators; class constants (tunables):
  - `MAX_FW_BATCH_PER_BLOCK = 8`
  - `MAX_BW_BATCH_PER_BLOCK = 32`
  - `DEFAULT_MAX_THREADS = 128`
  - `DEFAULT_MAX_SHARED_MEM = 48 * 1024`
  - `calculate_forward_config(batch_size, num_links, num_spheres, n_tool_frames, compute_jacobian, max_threads=None, max_shared_mem=None) -> Tuple[LaunchConfig, LaunchConfig]` — single fused (`num_spheres<100`, 4 threads/batch) vs separate cumul+spheres configs.
  - `calculate_backward_config(batch_size, num_links, num_spheres, n_tool_frames, n_joints, max_threads=None, max_shared_mem=None) -> Tuple[LaunchConfig, int, bool, int]` — returns `(config, threads_per_batch, use_warp_reduce, max_joints_template)`; `use_warp_reduce = num_spheres < 5000`; `max_joints_template ∈ {16,64,128}`.
  - `calculate_jacobian_backward_config(batch_size, num_links, n_tool_frames, n_joints, max_threads=None, max_shared_mem=None) -> Tuple[LaunchConfig, int, bool, int]` — as above with `use_warp_reduce = n_tool_frames < 5000`.

### curobo/_src/curobolib/backends/cuda_core_backend/geometry.py
cuda.core backend for geometry/collision kernels (only self-collision uses CUDA; world collision uses Warp).
- Module constants: `COLLISION_PAIR_SIZE = 8`, `STATIC_SMEM_OVERHEAD = COLLISION_PAIR_SIZE * 33`.
- `self_collision_distance(out_distance, out_vec, pair_distance, sparse_index, robot_spheres, sphere_padding, weight, pair_locations, block_batch_max_value, block_batch_max_index, num_blocks_per_batch, max_threads_per_block, batch_size, horizon, nspheres, num_collision_pairs, store_pair_distance, compute_grad)` — single kernel `self_collision_max_distance_kernel` when `num_blocks_per_batch==1`, else two-pass `self_collision_max_block_kernel` + `self_collision_max_reduce_kernel`; dynamic smem = `16*nspheres`; in-place outputs.
- (private `_validate_and_configure_shared_memory(kernel, dynamic_smemsize, nspheres, kernel_name)` — checks `cudaDevAttrMaxSharedMemoryPerBlockOptin`, sets `cudaFuncAttributeMaxDynamicSharedMemorySize` when `>48000`.)

### curobo/_src/curobolib/backends/cuda_core_backend/geometry_config.py
Configuration for geometry/collision kernel compilation.
- **GeometryKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"geometry"`.
  - `get_kernel_files(kernel_type) -> List[str]` — `"self_collision"→["self_collision/self_collision_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base + `kernels/geometry/`, `.../common`, `.../self_collision`.

### curobo/_src/curobolib/backends/cuda_core_backend/optimization.py
cuda.core backend for optimization kernels (line search + LBFGS).
- `launch_line_search(best_cost, best_action, best_iteration, current_iteration, converged_global, convergence_iteration, cost_delta_threshold, cost_relative_threshold, exploration_cost, exploration_action, exploration_gradient, exploration_idx, selected_cost, selected_action, selected_gradient, selected_idx, search_cost, search_action, search_gradient, step_direction, search_magnitudes, armijo_threshold_c_1, curvature_threshold_c_2, strong_wolfe, approx_wolfe, n_linesearch, opt_dim, batchsize)` — selects `kernel_line_search<float, 4>` if `n_linesearch==4` else `<float, -1>`; in-place state update.
- `launch_lbfgs_step(step_vec, rho_buffer, y_buffer, s_buffer, q, grad_q, x_0, grad_0, epsilon, batch_size, history_m, v_dim, stable_mode, use_shared_buffers) -> List[torch.Tensor]` — requires `0 <= history_m <= 31`; selects `kernel_lbfgs_step_shared_memory<float, false, history_m>` vs `kernel_lbfgs_step<...>`; configures extended dynamic smem (Volta+) when needed; returns `[step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]`.

### curobo/_src/curobolib/backends/cuda_core_backend/optimization_config.py
Configuration for optimization kernel compilation.
- **OptimizationKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"optimization"`.
  - `get_kernel_files(kernel_type) -> List[str]` — `"line_search"→["line_search/line_search_kernel.cuh"]`, `"lbfgs"→["lbfgs/lbfgs_step_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base + `kernels/optimization/`, `.../line_search`, `.../lbfgs`.
- **LineSearchLaunchCfg** — `calculate_config(opt_dim: int, batchsize: int) -> LaunchConfig` — `block=opt_dim`, `grid=batchsize`, `shmem_size=0`.
- **LBFGSLaunchCfg** — `calculate_config(batch_size, v_dim, history_m, use_shared_buffers) -> Tuple[LaunchConfig, bool, int]` — `block=v_dim`, `grid=batch_size`; picks basic vs shared-buffer smem (`max_shared_base=48000`, `max_shared_allowed=65536`); returns `(config, use_shared_buffers_actual, max_shared_memory_needed)`.

### curobo/_src/curobolib/backends/cuda_core_backend/trajectory.py
cuda.core backend for trajectory kernels (B-spline + legacy differentiation/integration).
- `launch_bspline_interpolation_forward_kernel(out_position, out_velocity, out_acceleration, out_jerk, out_dt, u_position, start_position, start_velocity, start_acceleration, start_jerk, goal_position, goal_velocity, goal_acceleration, goal_jerk, start_idx, goal_idx, traj_dt, use_implicit_goal_state, batch_size, horizon, dof, n_knots, bspline_degree)` — `interpolate_bspline_kernel<float, DEGREE, BasisBackend=2>`.
- `launch_bspline_interpolation_backward_kernel(out_grad_position, grad_position, grad_velocity, grad_acceleration, grad_jerk, traj_dt, dt_idx, use_implicit_goal_state, batch_size, padded_horizon, dof, n_knots, bspline_degree, use_direct_polynomial)` — `horizon=padded_horizon-1` (must be ≥5); `bspline_backward_kernel<DEGREE, float, BasisBackend=2>`.
- `launch_bspline_interpolation_single_dt_kernel(out_position, out_velocity, out_acceleration, out_jerk, out_dt, knots, knot_dt, start_position, start_velocity, start_acceleration, start_jerk, goal_position, goal_velocity, goal_acceleration, goal_jerk, start_idx, goal_idx, interpolation_dt, use_implicit_goal_state, interpolation_horizon, batch_size, max_out_tsteps, dof, n_knots, bspline_degree)` — `interpolate_bspline_single_dt_kernel<float, DEGREE, BasisBackend=2>`.
- `launch_differentiation_position_forward_kernel(out_position, out_velocity, out_acceleration, out_jerk, out_dt, u_position, start_position, start_velocity, start_acceleration, goal_position, goal_velocity, goal_acceleration, start_idx, goal_idx, traj_dt, use_implicit_goal_state, batch_size, horizon, dof)` — `position_clique_loop_idx_fwd_kernel<float,true>`.
- `launch_differentiation_position_backward_kernel(out_grad_position, grad_position, grad_velocity, grad_acceleration, grad_jerk, traj_dt, dt_idx, use_implicit_goal_state, batch_size, horizon, dof)` — `position_clique_loop_idx_bwd_kernel<float,true>`.
- `launch_integration_acceleration_kernel(out_position, out_velocity, out_acceleration, out_jerk, u_acc, start_position, start_velocity, start_acceleration, start_idx, traj_dt, batch_size, horizon, dof, use_rk2=True)` — `acceleration_loop_idx_rk2_kernel<float,max_horizon>` (or non-rk2 variant).

### curobo/_src/curobolib/backends/cuda_core_backend/trajectory_config.py
Configuration for trajectory kernel compilation + B-spline layout/launch calculators.
- **TrajectoryKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"trajectory"`.
  - `get_kernel_files(kernel_type) -> List[str]` — `bspline_forward/backward/single_dt→["bspline/bspline_kernel.cuh"]`, `differentiation_forward/backward→["legacy/differentiation_position_kernel.cuh"]`, `integration→["legacy/integration_acceleration_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base + `kernels/trajectory/`, `.../bspline`, `.../bspline/basis`, `.../legacy`.
- **BSplineBackwardLayout** — thread-organization struct; instance fields all default `0`: `interpolation_steps: int = 0`, `knots_per_warp: int = 0`, `warps_for_n_knots: int = 0`, `threads_for_n_knots: int = 0`, `padded_horizon: int = 0`, `n_knots: int = 0`, `padded_n_knots: int = 0`, `horizon: int = 0`, `dof: int = 0`.
- `get_spline_support_size(degree: int) -> int` — `degree + 1`.
- `get_total_knots(n_knots: int, degree: int) -> int` — `n_knots + support_size`.
- `compute_bspline_backward_layout(horizon, dof, n_knots, bspline_degree) -> BSplineBackwardLayout` — port of C++ warp-parallel thread layout for b-spline gradient.
- **BSplineLaunchCfg** — `calculate_forward_config(batch_size, dof, horizon)` (block≤128), `calculate_backward_config(batch_size, dof, n_knots, horizon, bspline_degree)` (uses `threads_for_n_knots`; asserts `interpolation_steps≤32`), `calculate_single_dt_config(batch_size, dof, max_out_tsteps)` (block≤256).
- **LegacyTrajectoryLaunchCfg** — `calculate_differentiation_forward_config(batch_size, dof, horizon)` (block≤128), `calculate_differentiation_backward_config(batch_size, dof, horizon)` (k_size uses `horizon-4`, block≤128), `calculate_integration_config(batch_size, dof)` (block≤512).

### curobo/_src/curobolib/backends/cuda_core_backend/dynamics.py
cuda.core backend for RNEA dynamics kernels (runtime compile + launch of forward/backward).
- `launch_rnea_forward(tau, q, qd, qdd, fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map, joint_offset_map, gravity, level_starts, level_links, forward_cache, batch_size, num_links, num_dof, n_levels, threads_per_batch=1, f_ext=None)` — inverse dynamics τ=RNEA(q,q̇,q̈,f_ext); `rnea_forward_kernel<N_LINKS, N_DOF, TPB, HAS_EXTERNAL_FORCES>`; f_ext ptr 0 when absent; saves intermediates to `forward_cache`.
- `launch_rnea_backward(grad_q, grad_qd, grad_qdd, grad_tau, q, qd, fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map, joint_offset_map, gravity, level_starts, level_links, forward_cache, batch_size, num_links, num_dof, n_levels, threads_per_batch=1, grad_f_ext=None)` — VJP dL/dq,dL/dqd,dL/dqdd (and dL/df_ext when `grad_f_ext` given); `rnea_backward_kernel<N_LINKS, N_DOF, TPB, HAS_EXTERNAL_FORCES>`.

### curobo/_src/curobolib/backends/cuda_core_backend/dynamics_config.py
Configuration for dynamics (RNEA) CUDA kernel compilation and launching.
- **DynamicsKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"dynamics"`.
  - `get_kernel_files(kernel_type) -> List[str]` — `"forward"→["rnea_forward_kernel.cuh"]`, `"backward"→["rnea_backward_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base + `kernels/dynamics/`, `kernels/kinematics/`.
- **DynamicsLaunchCfg** — warp-aligned occupancy-maximizing launch calc; class constants (tunables):
  - `DEFAULT_MAX_BATCHES_PER_BLOCK = 256`
  - `DEFAULT_MAX_BW_BATCHES_PER_BLOCK = 256`
  - `DEFAULT_MAX_SHARED_MEM = 48 * 1024`
  - `DEFAULT_SM_SHARED_MEM_CAPACITY = 100 * 1024`
  - `WARP_SIZE = 32`
  - `calculate_forward_config(batch_size, num_links, threads_per_batch=1, max_batches_per_block=None, max_shared_mem=None) -> LaunchConfig` — smem/batch = `num_links*12*4` (v+a).
  - `calculate_backward_config(batch_size, num_links, threads_per_batch=1, max_batches_per_block=None, max_shared_mem=None) -> LaunchConfig` — block-shared `num_links*12*4` + per-batch `num_links*30*4`.
  - (private `_warp_align_batches(batches_per_block, threads_per_batch, smem_per_block_fn, sm_shared_mem_capacity) -> int` — warp-round-down picking option maximizing resident-block threads.)

### curobo/_src/curobolib/backends/cuda_core_backend/pba.py
cuda.core backend for PBA+ 3D Euclidean Distance Transform (Parallel Banding Algorithm; 5 launches / 3 unique kernels: FloodZ → MaurerAxis → ColorAxis → MaurerAxis → ColorAxis).
- Kernel-name constants: `_FLOOD_Z="curobo::parallel_banding::kernel_flood_z"`, `_MAURER=...kernel_maurer_axis`, `_COLOR=...kernel_color_axis`.
- `launch_pba3d(site_index, buffer, nx, ny, nz, m3=2) -> None` — runs exact 3D Voronoi/EDT on `site_index` in-place (PBA axis map sx=nz, sy=ny, sz=nx); result copied back from `buffer`.
- (private `_get_compiled_kernels(cache)` — compile/retrieve the three PBA+ kernels.)

### curobo/_src/curobolib/backends/cuda_core_backend/pba_config.py
Configuration for PBA+ 3D EDT kernel compilation.
- **PBAKernelCfg**(CudaCoreKernelCfg) — `__init__` → subdir `"parallel_banding"`.
  - `get_kernel_files() -> List[str]` — `["pba3d_kernel.cuh"]`.
  - `get_include_dirs() -> List[Path]` — base + `kernels/parallel_banding/`.
- **PBALaunchCfg** — static launch calcs (PBA convention sx,sy,sz):
  - `flood_z(sx, sy) -> LaunchConfig` — grid `(cdiv(sx,32), cdiv(sy,4))`, block `(32,4)`.
  - `maurer_axis(sx, sz) -> LaunchConfig` — grid `(cdiv(sx,32), cdiv(sz,4))`, block `(32,4)`.
  - `color_axis(sx, sz, m3=2) -> LaunchConfig` — grid `(cdiv(sx,32), sz)`, block `(32, m3)`.
- `_cdiv(a: int, b: int) -> int` — ceiling division (module-private helper).
