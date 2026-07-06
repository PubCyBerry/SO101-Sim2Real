# cuRobo v2 — Benchmarks & Documentation Map

The benchmark scripts (repo-root `benchmark/`) and the reStructuredText documentation topic map (`docs/`).

## Benchmarks — `benchmark/`

| Script | Measures |
|---|---|
| `ik_benchmark.py` | IK solver timing/success across robots (`IKSolver` / `IKSolverCfg` + `CudaEventTimer`) |
| `motion_plan_benchmark.py` | Motion-planning benchmark; also loads a Pinocchio robot model for dynamics comparison |
| `cost_gradient_benchmark.py` | Forward kinematics, pose cost, and self/scene collision gradient time + memory. Unit-tagged result keys `time_ms`, `time_per_sample_ms`, `memory_mb` |

## Documentation (`docs/*.rst`)

Sphinx source — the conceptual/tutorial docs (rendered HTML is the official cuRobo docs site).

### Top-level
- `index.rst` — "cuRobo: CUDA Accelerated Robot Library" (landing page)
- `news.rst` — "Updates"
- `technical_reports.rst` — "Technical Reports" (research page)

### getting-started/
- `index.rst` — Getting Started
- `installation.rst` — Installation
- `build_robot_model.rst` — Build Robot Model
- `forward_kinematics.rst` — Forward Kinematics
- `inverse_kinematics.rst` — Inverse Kinematics
- `motion_planning.rst` — Motion Planning
- `reactive_control.rst` — Reactive Control
- `humanoid_retargeting.rst` — Humanoid Motion Retargeting
- `volumetric_mapping.rst` — Volumetric Mapping

### guides/
- `index.rst` — Guides
- `custom_cost.rst` — Extending RobotCostManager with a Custom Cost
- `custom_optimization.rst` — Writing Custom Optimization Problems
- `optimization_motion.rst` — Writing Motion Optimization Problems
- `optimization_problem.rst` — Writing Optimization Problems

### concepts/
- `index.rst` — Concepts
- `graph_planner.rst` — Graph Planner
- `optimization_solver.rst` — Optimization Solvers
- `rollout_class.rst` — Rollout Classes

### reference/
- `index.rst` — Reference
- `api_overview.rst` — Python API
- `benchmarks.rst` — Benchmarks & Profiling
- `runtime_configuration.rst` — Runtime Configuration
- `self_collision.rst` — Robot Self-Collision
- `sphere_fitting.rst` — Fitting Spheres to Geometry
- `styleguide.rst` — Style Guide

### snippets/
- `citation.rst`, `citation_v2.rst` — citation include snippets

## Changelog highlights (v0.8.0)

From `CHANGELOG.md` — see `../cuRobo_v2_0.8.0_INDEX.md` §1 for the architecture summary.
- Flat, inheritance-free rewrite (curobov2); `Optimizer` + `Rollout` Protocols; composed optimizer stack.
- Impl reorganised under `curobo/_src/` (`types`, `geom`, `motion`, `solver`, `rollout`, `optim`, `cost`, `perception`, `transition`, `graph_planner`).
- New `MotionPlanner`/`MotionPlannerCfg` with first-class batch + grasp planning.
- New tool-frame abstractions `Pose`, `GoalToolPose`, `SequenceToolPose`.
- New `Mapper` subsystem (block-sparse TSDF + ESDF, multi-camera fusion).
- CPU backend for `Pose`; self-collision debugger script; benchmark overhaul.
- **Breaking:** the major refactor breaks most of the classic cuRobo API.
