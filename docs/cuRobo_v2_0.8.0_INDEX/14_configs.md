# cuRobo v2 — Content Configs & Assets

Bundled robot / scene / task YAML+XRDF configs under `curobo/content/configs/`, plus the robot asset roots. These are the string names passed to `robot=`, `scene_model=`, and `optimizer_configs=`.

## Robots — `content/configs/robot/`

Top-level YAML shape: `robot_cfg: { kinematics: {...}, load_dynamics: bool }`. Key kinematics keys: `format_version`, `base_link`, `tool_frames`, `collision_link_names`, `collision_spheres`, `collision_sphere_buffer`, `self_collision_buffer`, `self_collision_ignore`, `lock_joints`, `mesh_link_names`, `extra_links`, `grasp_contact_link_names`, `asset_root_path`, `urdf_path`, `cspace`, `use_global_cumul`.

| File | Robot | URDF |
|---|---|---|
| `franka.yml` | Franka Panda (7-DOF) | `robot/franka_description/franka_panda.urdf` (base_link `panda_link0`) |
| `ur10e.yml` | Universal Robots UR10e (6-DOF) | `robot/ur_description/ur10e.urdf` |
| `ur10e.xrdf` | UR10e (XRDF collision/self-collision spec — needs a separate `urdf_path`) | — |
| `dual_ur10e.yml` | Dual UR10e | `robot/ur_description/dual_ur10e.urdf` |
| `simple_mimic_robot.yml` | Simple mimic-joint test robot | `robot/simple/simple_mimic_robot.urdf` |
| `unitree_g1.yml` | Unitree G1, 29-DOF with hands | `robot/g1/g1_29dof_with_hand_rev_1_0.urdf` |
| `unitree_g1_29dof_retarget.yml` | Unitree G1 29-DOF, retargeting variant | `robot/g1/g1_29dof_rev_1_0.urdf` |

## Scenes — `content/configs/scene/`

Obstacle-world YAMLs (keys are obstacle types: `cuboid:`, `mesh:`, etc., each entry with `dims` + `pose` = `[x,y,z,qw,qx,qy,qz]`).

| File | Defines |
|---|---|
| `collision_table.yml` | A single `table` cuboid (`dims: [4,4,0.2]`, pose below origin) — the standard flat-table world |
| `collision_base_stand.yml` | Base/stand obstacle world |
| `collision_primitives_3d.yml` | Assorted 3D primitive obstacles |
| `collision_test.yml` | Small test obstacle world (used by interactive IK examples) |

## Tasks — `content/configs/task/`

Solver/optimizer-stage YAMLs. Shape: `rollout: { cost_cfg, constraint_cfg }` + `optimizer: { solver_type, solver_name, num_iters, history, inner_iters, line_search_type, ... }`. Passed via `optimizer_configs=[...]` / `transition_model=` / `metrics_rollout=` to the `*Cfg.create()` methods.

| File | Solver | Stage / notes |
|---|---|---|
| `metrics_base.yml` | (shared) | Base rollout/metrics config used by all solvers as `metrics_rollout` |
| `ik/particle_ik.yml` | IK | Particle (MPPI/ES) seed stage |
| `ik/lbfgs_ik.yml` | IK | L-BFGS gradient stage (`solver_type: lbfgs`, `num_iters: 100`, `history: 7`, `inner_iters: 20`, `line_search_type: approx_wolfe`) — the default IK optimizer |
| `ik/lbfgs_retarget_ik.yml` | IK | L-BFGS retargeting stage |
| `ik/transition_ik.yml` | IK | Transition (integration) model for IK |
| `trajopt/particle_trajopt.yml` | TrajOpt | Particle stage |
| `trajopt/lbfgs_bspline_trajopt.yml` | TrajOpt | L-BFGS + B-spline stage (default trajopt optimizer) |
| `trajopt/transition_bspline_trajopt.yml` | TrajOpt | Transition B-spline model |
| `mpc/lbfgs_mpc.yml` | MPC | L-BFGS stage |
| `mpc/lbfgs_retarget_mpc.yml` | MPC | L-BFGS retargeting stage |
| `mpc/transition_bspline_mpc.yml` | MPC | Transition B-spline model |
| `graph_planner/exact_graph_planner.yml` | Graph (PRM) | Exact roadmap planner |
| `graph_planner/transition_graph_planner.yml` | Graph (PRM) | Transition/retarget roadmap planner |
| `README.md` | — | Docs for the task-config directory |

## Asset roots — `content/assets/robot/`

Mesh/URDF asset roots referenced by `asset_root_path` in the robot configs:
`franka_description/` (+ `meshes/{visual,collision}`), `g1/` (+ `meshes`), `ur_description/` (+ `meshes/{ur5e,ur10e}/{visual,collision}`), `simple/`.
