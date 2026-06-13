"""grasp_close=0 진단 — bootstrap full-grasp env의 reset 직후 ee vs cube 기하 실측.

가설: bootstrap이 큐브를 고정 _grasp_offset(env0 default 자세 1회 캐시)에 놓는데
reset마다 팔 ±0.05rad jitter → live ee 어긋남 → grasp point 밖 → grasp_close=0.

reset→step(캐시)→재reset(bootstrap 적용) 순서로 측정.
"""
import argparse, torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real  # noqa: F401  (Gym 등록)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import apply_skill_full_bc
from sim_to_real.tasks.common.mdp.rewards import _get_gripper_pos, grasp_close_reward, grasp_align_reward
from sim_to_real.utils.constant import CUBE_NAMES, BOWL_NAME
from isaaclab.managers import SceneEntityCfg

dev = "cuda:0"
env_cfg = parse_env_cfg(args.task, device=dev, num_envs=args.num_envs)
apply_skill_full_bc(env_cfg)
# bootstrap 파라미터 = train.py와 동일
env_cfg.grasp_bootstrap_prob = 0.7
env_cfg.grasp_bootstrap_prob_final = 0.0
env_cfg.grasp_bootstrap_anneal_steps = 1e9   # anneal 거의 안 됨 → prob≈0.7 유지
env_cfg.grasp_bootstrap_pregrasp_frac = 0.3
env_cfg.grasp_bootstrap_close = -0.05

env = gym.make(args.task, cfg=env_cfg, render_mode=None)
ue = env.unwrapped

# 1) 첫 reset (bootstrap skip — grasp_offset None)
ue.reset()
# 2) step 1회 → _cache_grasp_geom 가 _grasp_offset 캐시
zero = torch.zeros((args.num_envs, ue.action_space.shape[1] if hasattr(ue.action_space,'shape') else 6), device=dev)
try:
    ue.step(zero)
except Exception:
    ue.step(torch.zeros((args.num_envs, 6), device=dev))
print(f"[diag] _grasp_offset cached = {ue._grasp_offset}")

# 3) 전체 강제 재reset → 이제 bootstrap 적용됨
all_ids = torch.arange(args.num_envs, device=dev)
ue._reset_idx(all_ids)
# write_*_to_sim 반영 위해 physics 1 step + .data 버퍼 refresh
ue.scene.write_data_to_sim()
ue.sim.step(render=False)
ue.scene.update(ue.sim.get_physics_dt())

# 4) rollout trace — 2-phase 배치(ue.step~2 에 발생)부터 grip 유지까지 추적
robot_cfg = SceneEntityCfg("robot", body_names=["gripper"]); robot_cfg.resolve(ue.scene)
cube_cfg = SceneEntityCfg(CUBE_NAMES[0]); cube_cfg.resolve(ue.scene)
robot = ue.scene["robot"]
full0 = (ue.bootstrap_kind == 1).clone()   # 이번 reset 의 full-grasp env (마킹 시점)
pre0 = (ue.bootstrap_kind == 2).clone()
n_full = full0.sum().item(); n_pre = pre0.sum().item()
desk_z = 0.726
print(f"\n[diag] bootstrap_kind: full={n_full} pre={n_pre} scratch={(ue.bootstrap_kind==0).sum().item()} / {args.num_envs}")
print("\n=== rollout (zero action) — 2-phase 배치는 step~2 에 발생, 이후 full-grasp grip 유지 추적 ===")
print(f"{'step':>4} {'cube_z':>7} {'ee_z':>7} {'dz':>6} {'dxy':>6} {'gclose':>7} {'gc>1.5%':>7} {'lost%':>6} {'gripJ':>6}")
for s in range(36):
    ee = _get_gripper_pos(ue, robot_cfg)
    cube_p = ue.scene[CUBE_NAMES[0]].data.root_pos_w
    local_z = (cube_p - ue.scene.env_origins)[:, 2]
    d = cube_p - ee
    dz = torch.abs(d[:, 2]); dxy = torch.linalg.vector_norm(d[:, :2], dim=1)
    gc = grasp_close_reward(ue, robot_cfg=robot_cfg, object_cfgs=[cube_cfg],
                            container_center_xy=(2.2, -0.17), container_cfg=SceneEntityCfg(BOWL_NAME))
    grip_j = robot.data.joint_pos[full0, -1].mean() if n_full > 0 else float('nan')
    lost = (local_z[full0] < (desk_z - 0.02)).float().mean() * 100  # 책상 밑으로 떨어짐
    if s <= 6 or s % 3 == 0 or s == 35:
        print(f"{s:4d} {cube_p[full0,2].mean():7.3f} {ee[full0,2].mean():7.3f} "
              f"{dz[full0].mean():6.3f} {dxy[full0].mean():6.3f} {gc[full0].mean():7.3f} "
              f"{(gc[full0]>1.5).float().mean()*100:7.1f} {lost:6.1f} {grip_j:6.3f}")
    ue.step(torch.zeros((args.num_envs, 6), device=dev))
print("\n[diag] 판정: 배치 후(step≳3) cube_z≈ee_z + dz<0.06 + gclose>1.5 비율↑ = grip OK(2-phase 성공).")
print("        cube_z 급락 / gclose→0 = grip 실패.")

env.close()
simulation_app.close()
