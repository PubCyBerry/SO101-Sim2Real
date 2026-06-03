"""TB.1 staged reward smoke: build rewards + deterministic predicate checks.

Usage:
    uv run python scripts/environments/reward_smoke.py \
        --task SimToReal-SO101-PickPen-v0 \
        --num_envs 1 --device cuda:0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TB.1 staged reward smoke")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # registers SimToReal-SO101-PickPen-v0
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from sim_to_real.tasks.pick_pen import mdp as task_mdp  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import PEN_CUP_CENTER_XY  # noqa: E402
from sim_to_real.utils.constant import PEN_NAMES  # noqa: E402

_DESK_TOP_Z = 0.92
_REQUIRED_TERMS = {
    "reach_pen",
    "grasp_pen",
    "lift_pen",
    "transport_pen",
    "insert_pen",
    "release_pen",
    "task_success",
    "action_rate",
    "joint_vel",
}


def _tensor_list(value: torch.Tensor) -> list[float]:
    return [round(float(v), 5) for v in value.detach().cpu().flatten().tolist()]


def _check_term(name: str, value: torch.Tensor, num_envs: int, failures: list[str]) -> dict:
    shape_ok = tuple(value.shape) == (num_envs,)
    finite_ok = bool(torch.isfinite(value).all().item())
    if not shape_ok:
        failures.append(f"{name}: shape {tuple(value.shape)} != ({num_envs},)")
    if not finite_ok:
        failures.append(f"{name}: non-finite value {_tensor_list(value)}")
    return {
        "shape": list(value.shape),
        "finite": finite_ok,
        "value": _tensor_list(value),
    }


def _term_values(env) -> dict[str, torch.Tensor]:
    robot_gripper = SceneEntityCfg("robot", body_names=["gripper"])
    return {
        "reach_pen": task_mdp.reach_reward(env, robot_cfg=robot_gripper, cup_center_xy=PEN_CUP_CENTER_XY),
        "grasp_pen": task_mdp.grasp_bonus(env, robot_cfg=robot_gripper, cup_center_xy=PEN_CUP_CENTER_XY),
        "lift_pen": task_mdp.lift_reward(env),
        "transport_pen": task_mdp.transport_reward(env, cup_center_xy=PEN_CUP_CENTER_XY),
        "insert_pen": task_mdp.insert_reward(env, cup_center_xy=PEN_CUP_CENTER_XY),
        "release_pen": task_mdp.release_bonus(env, cup_center_xy=PEN_CUP_CENTER_XY),
        "task_success": task_mdp.task_success_bonus(env, cup_center_xy=PEN_CUP_CENTER_XY),
    }


def _local_xyz(env, xyz: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor([xyz], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)


def _gripper_pos(env) -> torch.Tensor:
    cfg = SceneEntityCfg("robot", body_names=["gripper"])
    cfg.resolve(env.scene)
    robot = env.scene["robot"]
    body_ids = cfg.body_ids
    if isinstance(body_ids, int):
        return robot.data.body_pos_w[:, body_ids, :]
    if isinstance(body_ids, slice):
        return robot.data.body_pos_w[:, body_ids, :][:, 0, :]
    return robot.data.body_pos_w[:, body_ids[0], :]


def _sync_scene(env) -> None:
    env.scene.update(0.0)


def _write_pen_world_pose(env, name: str, pos_w: torch.Tensor) -> None:
    obj = env.scene[name]
    env_ids = torch.arange(env.num_envs, device=env.device)
    pose = obj.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = pos_w
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids)


def _write_pen_local_pose(env, name: str, pos_local: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    _write_pen_world_pose(env, name, pos_local + env.scene.env_origins[env_ids])


def _write_all_pens_outside(env) -> None:
    z = _DESK_TOP_Z + 0.015
    positions = [
        (2.00, -0.39, z),
        (2.07, -0.39, z),
        (2.14, -0.39, z),
        (2.21, -0.39, z),
    ]
    for name, xyz in zip(PEN_NAMES, positions, strict=True):
        _write_pen_local_pose(env, name, _local_xyz(env, xyz))
    _sync_scene(env)


def _set_gripper_open(env, open_: bool) -> None:
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, -1] = 0.8 if open_ else 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    _sync_scene(env)


def _assert_increase(
    checks: list[dict],
    name: str,
    before: torch.Tensor,
    after: torch.Tensor,
    failures: list[str],
    min_delta: float = 1e-4,
) -> None:
    delta = after - before
    ok = bool((delta > min_delta).all().item())
    checks.append({
        "stage": name,
        "before": _tensor_list(before),
        "after": _tensor_list(after),
        "delta": _tensor_list(delta),
        "ok": ok,
    })
    if not ok:
        failures.append(f"{name}: expected increase, before={_tensor_list(before)}, after={_tensor_list(after)}")


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.episode_length_s = 20.0
        env = gym.make(args.task, cfg=env_cfg)
        env.reset()

        reward_term_names = set(env.unwrapped.reward_manager.active_terms)
        failures: list[str] = []
        missing = sorted(_REQUIRED_TERMS - reward_term_names)
        if missing:
            failures.append(f"missing reward terms: {missing}")

        # Initial finite check from the normal reset state.
        initial_terms = _term_values(env.unwrapped)
        finite_terms = {
            name: _check_term(name, value, args.num_envs, failures)
            for name, value in initial_terms.items()
        }

        stage_checks: list[dict] = []

        # reach/grasp: independent baseline on desk, then place PenWhite at the gripper.
        _write_all_pens_outside(env.unwrapped)
        _set_gripper_open(env.unwrapped, open_=False)
        baseline = _term_values(env.unwrapped)
        near_gripper = _gripper_pos(env.unwrapped).clone()
        _write_pen_world_pose(env.unwrapped, "PenWhite", near_gripper)
        _sync_scene(env.unwrapped)
        target = _term_values(env.unwrapped)
        _assert_increase(stage_checks, "reach", baseline["reach_pen"], target["reach_pen"], failures)
        _assert_increase(stage_checks, "grasp", baseline["grasp_pen"], target["grasp_pen"], failures)

        # lift: independent desk baseline, then lift one pen above the desk.
        _write_all_pens_outside(env.unwrapped)
        baseline = _term_values(env.unwrapped)
        lifted = _local_xyz(env.unwrapped, (2.05, -0.35, _DESK_TOP_Z + 0.13))
        _write_pen_local_pose(env.unwrapped, "PenWhite", lifted)
        _sync_scene(env.unwrapped)
        target = _term_values(env.unwrapped)
        _assert_increase(stage_checks, "lift", baseline["lift_pen"], target["lift_pen"], failures)

        # transport: lifted far-from-cup baseline, then same pen above the cup XY.
        _write_all_pens_outside(env.unwrapped)
        far_lifted = _local_xyz(env.unwrapped, (2.00, -0.45, _DESK_TOP_Z + 0.13))
        _write_pen_local_pose(env.unwrapped, "PenWhite", far_lifted)
        _sync_scene(env.unwrapped)
        baseline = _term_values(env.unwrapped)
        over_cup = _local_xyz(env.unwrapped, (PEN_CUP_CENTER_XY[0], PEN_CUP_CENTER_XY[1], _DESK_TOP_Z + 0.13))
        _write_pen_local_pose(env.unwrapped, "PenWhite", over_cup)
        _sync_scene(env.unwrapped)
        target = _term_values(env.unwrapped)
        _assert_increase(stage_checks, "transport", baseline["transport_pen"], target["transport_pen"], failures)

        # insert: all pens outside baseline, then all pens inside cup volume with gripper closed.
        _write_all_pens_outside(env.unwrapped)
        _set_gripper_open(env.unwrapped, open_=False)
        baseline = _term_values(env.unwrapped)
        for index, name in enumerate(PEN_NAMES):
            offset = (index - 1.5) * 0.008
            inside = _local_xyz(env.unwrapped, (PEN_CUP_CENTER_XY[0] + offset, PEN_CUP_CENTER_XY[1], _DESK_TOP_Z + 0.07))
            _write_pen_local_pose(env.unwrapped, name, inside)
        _sync_scene(env.unwrapped)
        target = _term_values(env.unwrapped)
        _assert_increase(stage_checks, "insert", baseline["insert_pen"], target["insert_pen"], failures)

        # release/success: same inserted pose, then open the gripper.
        baseline = _term_values(env.unwrapped)
        _set_gripper_open(env.unwrapped, open_=True)
        target = _term_values(env.unwrapped)
        _assert_increase(stage_checks, "release", baseline["release_pen"], target["release_pen"], failures)
        _assert_increase(stage_checks, "success", baseline["task_success"], target["task_success"], failures)

        # Final finite check after staged manipulation.
        final_terms = _term_values(env.unwrapped)
        final_finite_terms = {
            name: _check_term(name, value, args.num_envs, failures)
            for name, value in final_terms.items()
        }

        env.close()
        env = None

        result = {
            "task_id": "TB.1",
            "task": args.task,
            "status": "passed" if not failures else "failed",
            "num_envs": args.num_envs,
            "reward_terms": sorted(reward_term_names),
            "finite_terms_initial": finite_terms,
            "finite_terms_final": final_finite_terms,
            "stage_checks": stage_checks,
            "failures": failures,
        }
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        if failures:
            sys.exit(1)

    except Exception as exc:
        tb = traceback.format_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print(json.dumps({
            "task_id": "TB.1",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
        }, indent=2))
        sys.stdout.flush()
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
