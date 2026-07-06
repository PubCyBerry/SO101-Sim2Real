# cuRobo v2 — Optimizers (MPPI, ES, L-BFGS, L-SR1, CG, external)

The standalone optimizer stack behind every solver, the `Optimizer` protocol, and the particle samplers. All optimizers are composed (not inherited) and satisfy the protocol.

## Standard Optimizer Interface

Every optimizer satisfies `Optimizer` (`curobo/_src/optim/optimizer_protocol.py`, a runtime-checkable `typing.Protocol`):

- **properties:** `config`, `device_cfg`, `opt_dt` (get/set), `use_cuda_graph`, `enabled`, `action_horizon`, `action_dim`, `opt_dim`, `outer_iters`, `horizon`, `action_bound_lows`, `action_bound_highs`, `action_step_max`, `solve_time`, `solver_names`, `rollout_fn`
- **methods:** `optimize(...)`, `reinitialize(...)`, `shift(...)`, `update_num_problems(...)`, `update_rollout_params(...)`, `update_goal_dt(...)`, `get_all_rollout_instances()`, `compute_metrics(...)`, `enable()`, `disable()`, `reset_shape()`, `reset_seed()`, `reset_cuda_graph()`, `get_recorded_trace()`, `update_solver_params(...)`, `update_niters(...)`, `debug_dump(...)`

Per-optimizer sections below list only the *distinctive* methods/properties on top of this interface.

## Shared `*OptCfg` field block

All gradient & particle `*OptCfg` dataclasses share this base block (listed once):
`num_iters: int = 100`, `solver_type: str`, `solver_name: str`, `device_cfg: DeviceCfg`, `store_debug: bool = False`, `debug_info: Any = None`, `num_problems: int = 1`, `num_particles: Optional[int] = None`, `sync_cuda_time: bool = True`, `use_coo_sparse: bool = True`, `step_scale: float = 1.0`, `inner_iters: int`, `_num_rollout_instances: int`.

**Gradient** configs additionally share: `cost_convergence: float = 1e-11`, `cost_delta_threshold: float = 0.0`, `cost_relative_threshold: float = 0.0`, `converged_ratio: float = 0.8`, `fixed_iters: bool = True`, `convergence_iteration: int = 0`, `minimum_iters: Optional[int] = None`, `return_best_action: bool = True`.

Only per-optimizer *extras* are enumerated below.

---

### curobo/_src/optim/optimizer_protocol.py
Runtime-checkable protocol defining the optimizer interface.
- **Optimizer**(Protocol) — see Standard Optimizer Interface above.

### curobo/_src/optim/optim_factory.py
Optimizer creation entry point mapping `solver_type` strings to optimizer classes.
- `create_optimization_config(config_dict: Dict, device_cfg: DeviceCfg)` — build a typed `*OptCfg` from a task-yml dict.
- `create_optimizer(rollout: List[Rollout], use_cuda_graph: bool = False)` — instantiate the optimizer named by the config.

### curobo/_src/optim/multi_stage_optimizer.py
Chains several optimizers in sequence (e.g. particle→gradient).
- **MultiStageOptimizer** — `__init__(optimizers: List, rollout_list: Optional[List[Rollout]] = None)`. Standard interface; extra props `outer_iters`, `solver_names`, `solve_time`.

### curobo/_src/optim/optimization_iteration_state.py
Per-iteration optimization variables.
- **OptimizationIterationState**(@dataclass) — `action: torch.Tensor`, `cost/gradient/exploration_action/exploration_gradient/exploration_cost/step_direction/best_action/best_cost/best_iteration/current_iteration/state/converged/jacobian` (all `Optional`). Methods `data_ptr`, `clone`, `copy_(other)`.

## Gradient optimizers — `_src/optim/gradient/`

### gradient/lbfgs.py
L-BFGS via two-loop recursion for quasi-Newton step directions.
- **LBFGSOptCfg**(@dataclass) — common gradient block + `solver_type/name="lbfgs"`, `inner_iters: int = 25`, `_num_rollout_instances: int = 2`, `line_search_scale: List[float] = [0.1,0.3,0.7,1.0]`, `line_search_type: LineSearchType = APPROX_WOLFE`, `use_cuda_kernel_line_search: bool = True`, `fix_terminal_action: bool = False`, `line_search_wolfe_c_1: float = 1e-5`, `line_search_wolfe_c_2: float = 0.9`, `history: int = 7`, `epsilon: float = 0.01`, `use_cuda_kernel_step_direction: bool = True`, `stable_mode: bool = True`, `use_cuda_kernel_shared_buffers: bool = True`, `initial_step_scale: float = 0.1`. Methods `create_data_dict`, `update_niters`.
- **LBFGSOpt** — standard interface; extra props `action_horizon_step_max`, `action_horizon_bounds_lows/highs`.

### gradient/lsr1.py
L-SR1 via symmetric rank-1 updates.
- **LSR1Opt** — standard interface (uses an `LBFGSOptCfg`-style config); extra prop `action_horizon_step_max`.
- `jit_lsr1_compute_step_direction(y_buffer, s_buffer, grad, m, epsilon, stable_mode, hessian_0)`.

### gradient/conjugate_gradient.py
Nonlinear Conjugate Gradient (Fletcher-Reeves / Polak-Ribiere / Dai-Yuan).
- **ConjugateGradientOptCfg**(@dataclass) — common gradient block + `solver_type/name="conjugate_gradient"`, `inner_iters: int = 25`, `line_search_scale = [0.1,0.3,0.7,1.0]`, `line_search_type: LineSearchType = APPROX_WOLFE`, `use_cuda_kernel_line_search: bool = True`, `fix_terminal_action: bool = False`, `line_search_wolfe_c_1: float = 1e-5`, `line_search_wolfe_c_2: float = 0.9`, `initial_step_scale: float = 0.1`, `cg_method: str = "FR"`, `max_beta: float = 10.0`.
- **ConjugateGradientOpt** — standard interface.
- `jit_cg_compute_step_direction(grad, prev_grad, prev_step, max_beta, method)`, `jit_cg_shift_buffers(shift_steps, action_dim)`.

### gradient/gradient_descent.py
Scaled gradient descent.
- **GradientDescentOptCfg**(@dataclass) — common gradient block + `solver_type/name="gradient_descent"`, `inner_iters: int = 25`, `_num_rollout_instances: int = 1`, `gradient_descent_step_scale: float = 0.001`.
- **GradientDescentOpt** — standard interface. **LineSearchGradientDescentOpt**(GradientDescentOpt) — line-search variant.

### gradient/line_search_strategy.py
Line search strategies for step-direction methods.
- `LineSearchType` {GREEDY="greedy", ARMIJO="armijo", WOLFE="wolfe", STRONG_WOLFE="strong_wolfe", APPROX_WOLFE="approx_wolfe", APPROX_STRONG_WOLFE="approx_strong_wolfe"}
- **LineSearchStrategy**(ABC) — `search(...)`, `update_num_problems`; concrete: **GreedyLineSearchStrategy**, **ArmijoLineSearchStrategy**, **BaseWolfeLineSearchStrategy** → **WolfeLineSearchStrategy**, **StrongWolfeLineSearchStrategy**, **ApproxWolfeLineSearchStrategy**, **ApproxStrongWolfeLineSearchStrategy**.
- **LineSearchStrategyFactory** — `get_strategy(strategy_type)`, `register_strategy(strategy_type, strategy)`.

### gradient/ — support modules
- **line_search_context.py** — **LineSearchContext**(@dataclass): `device_cfg, line_search_scale, line_search_c_1, line_search_c_2, num_problems, opt_dim, action_horizon, action_dim, step_scale, fix_terminal_action, action_horizon_step_max, use_cuda_kernel_line_search, compute_costs_and_gradients: Callable, convergence_iteration, cost_delta_threshold, cost_relative_threshold`; prop `n_linesearch`, `update_num_problems`.
- **line_search_state.py** — **LineSearchState**(@dataclass): `action, cost, gradient, idxs`.
- **line_search_result.py** — **LineSearchResult**(@dataclass): `selected_state, exploration_state`.
- **lbfgs_jit_helpers.py** — `jit_lbfgs_compute_step_direction(...)`, `jit_lbfgs_update_buffers(...)`, `lbfgs_shift_buffers_jit`, `lbfgs_reset_jit`, `lbfgs_reset_problem_ids_jit`.
- **update_best_solution.py** — `update_best_solution(iteration_state, action_horizon, action_dim, cost_delta_threshold, cost_relative_threshold, convergence_iteration, ...)`.
- **util.py** — empty (license header only).

## Particle optimizers — `_src/optim/particle/`

### particle/mppi.py
Model Predictive Path Integral optimizer.
- `BaseActionType` {REPEAT="REPEAT", NULL="NULL", RANDOM="RANDOM"}
- **MPPICfg**(@dataclass) — common block + `solver_type/name="mppi"`, `inner_iters: int = 1`, `_num_rollout_instances: int = 1`, `gamma: float = 1.0`, `sample_mode: SampleMode = SampleMode.MEAN`, `seed: int = 0`, `store_rollouts: bool = False`, `null_act_frac: float = 0.0`, `init_mean: Optional[torch.Tensor] = None`, `init_cov: float = 0.5`, `base_action: BaseActionType = REPEAT`, `step_size_mean: float = 0.9`, `step_size_cov: float = 0.1`, `squash_fn: SquashType = CLAMP`, `cov_type: CovType = DIAG_A`, `sample_params: Optional[ParticleSamplerCfg] = None`, `update_cov: bool = True`, `random_mean: bool = False`, `beta: float = 0.1`, `alpha: float = 1.0`, `kappa: float = 0.01`, `sample_per_problem: bool = True`.
- **MPPI** — standard interface + `sample_actions`, `update_seed`, `update_init_mean`, `get_rollouts`, `reset_distribution/mean/covariance`, `initialize_samples`, `update_samples`, `generate_noise`; props `mean_action` (get/set), `best_traj`, `cov_action`, `scale_tril`, `inv_cov_action`, `entropy`, `top_trajs`, etc.
- jit helpers: `jit_calculate_exp_util`, `jit_compute_total_cost`, `jit_diag_a_cov_update`, `jit_blend_cov`, `jit_blend_mean`, `jit_mean_cov_diag_a`.

### particle/evolution_strategies.py
Evolution Strategies optimizer.
- **EvolutionStrategiesCfg**(MPPICfg) — extra field `learning_rate: float = 0.1`.
- **EvolutionStrategies** — standard interface + same extra methods/props as MPPI. `calc_exp(total_costs)`, `compute_es_mean(...)`.

### particle/particle_opt_utils.py
- `SquashType` {CLAMP=0, CLAMP_RESCALE=1, TANH=2, IDENTITY=3}
- `scale_ctrl(ctrl, action_lows, action_highs, squash_fn=CLAMP)`, `gaussian_entropy(cov=None, L=None)`, `cost_to_go(cost_seq, gamma_seq, only_first=False)`, `matrix_cholesky(A)`, `batch_cholesky(A)`.

### particle/sample_strategies/
- **particle_sampler_cfg.py** — **ParticleSamplerCfg**(@dataclass): `device_cfg, fixed_samples: bool = True, sample_ratio: Dict[str,float], seed: int = 0, filter_coeffs: Optional[List[float]] = [0.3,0.3,0.4], n_knots: int = 3, scale_tril: Optional[float] = None, covariance_matrix: Optional[torch.tensor] = None, sample_method: str = "halton", degree: int = 3, stencil_type: str = "3point"`.
- **particle_sampler.py** — **ParticleSampler** (`get_samples`, factories `create_halton/random/knot/stomp_particle_sampler`); **MixedParticleSampler**; `create_particle_sampler(sample_type, sample_config, horizon, action_dim, **kwargs)`.
- **processor_standard.py / processor_knot.py / processor_stomp.py** — `StandardParticleProcessor`, `KnotParticleProcessor` (`bspline(...)`), `StompParticleProcessor` — each `process_samples(samples, filter_smooth=False)`.
- **stomp_covariance.py** — `get_stomp_cov(horizon, zero_out_boundary=True, stencil_type="3point", ...)`.

## Components — `_src/optim/components/`
- **gaussian_distribution.py** — `CovType` {SIGMA_I, DIAG_A}; **GaussianDistribution** (`reset_mean/covariance`, `update_mean`, `get_samples`, `generate_noise`, `shift`, props `full_scale_tril`, `full_inv_cov`).
- **particle_opt_core.py** — `SampleMode` {MEAN, BEST, SAMPLE}; **ParticleOptCore** (shared particle infra: `sample_actions`, `update_seed`, `reset_distribution`, props `particles_per_problem`, `total_num_particles`, ...).
- **gradient_opt_core.py** — **GradientOptCore** (shared gradient/line-search infra; standard interface + `finish_init`).
- **action_bounds.py** — **ActionBounds** (`refresh(...)`) — horizon-expanded action limits.
- **best_tracker.py** — **BestTracker** (`resize`, `clear`, `update`, `check_convergence`).
- **quasi_newton_buffers.py** — **QuasiNewtonBuffers** (`resize`, `set_reference`, `update`, `shift`) — limited-memory ring buffers.
- **debug_recorder.py** — **DebugRecorder** (`record`, `get_trace`).

## External optimizers — `_src/optim/external/`
- **torch_opt.py** — **TorchOptCfg**(@dataclass): common block + `solver_type/name="torch"`, `inner_iters: int = 1`, `torch_optim_name: str = "Adam"`, `torch_optim_kwargs: dict = {}`, `torch_optim_class: Optional[Any] = None`. **TorchOpt** — wraps any `torch.optim` optimizer.
- **scipy_opt.py** — **ScipyOptCfg**(@dataclass): common block + `solver_type/name="scipy"`, `inner_iters: int = 1`, `scipy_minimize_method: str = "SLSQP"`, `scipy_minimize_kwargs: dict = {}`, `use_float64_on_cpu: bool = False`. **ScipyOpt** — evaluate cost on GPU, minimize on CPU.

## Util — `_src/optim/util/`
- **levenberg_marquardt_step.py** — **LevenbergMarquardtState**(@dataclass): `jacobian, jTerror, lambda_damping, joint_position_in, joint_position_out, pred_reduction` (+ shape helpers); **LevenbergMarquardtStep** (`__call__(state) -> (Tensor, Tensor)`, static `create_lm_warp_kernel(dof, n_res)`).
