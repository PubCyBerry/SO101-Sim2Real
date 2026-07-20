"""공식 Isaac Lab Mimic SkillGen flow로 SO-101 demonstration을 증강한다.

공식 ``generate_dataset.py --use_skillgen``과 동일한 async generation 모듈을 사용하되,
Franka-only planner factory 대신 프로젝트의 SO-101 adapter를 주입한다.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import random
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToReal-SO101-PickCube-SkillGen-DRBase-v0")
parser.add_argument("--input-file", required=True, help="SM annotated source HDF5")
parser.add_argument("--output-file", required=True, help="성공 generated HDF5")
parser.add_argument("--summary-file", required=True, help="attempt/planner 통계 JSON")
parser.add_argument("--generation-num-trials", type=int, default=10, help="목표 성공 episode 수")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=41000)
parser.add_argument("--record-images", action="store_true", help="HDF5에 top/wrist/front uint8 기록")
parser.add_argument(
    "--keep-failed",
    action="store_true",
    help="실패 증빙 전용: 별도 *_failed.hdf5에 failed attempt를 보존",
)
parser.add_argument("--debug-planner", action="store_true", help="cuRobo 실패 status 상세 출력")
parser.add_argument("--visualize-spheres", action="store_true", help="검수 run에서 collision sphere 표시")
parser.add_argument("--visualize-plan", action="store_true", help="검수 run에서 planner trajectory 표시")
parser.add_argument("--max-failures", type=int, default=2500, help="무한 재시도를 막는 실패 상한")
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab_mimic.datagen import generation
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import setup_output_paths

import sim_to_real  # noqa: F401
from sim_to_real.datagen.skillgen_curobo import SO101CuroboPlanner, make_so101_planner_cfg
from sim_to_real.datagen.skillgen_recorders import (
    SO101SkillGenJointRecorderManagerCfg,
    SO101SkillGenRecorderManagerCfg,
)
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras


_failure_categories: Counter[str] = Counter()
_attempt_diagnostics: list[dict] = []


def _classify_failed_attempt(results: dict, *, planner_failed: bool) -> tuple[str, dict]:
    """공식 DataGenerator result를 planning/IK/grasp/place/timeout으로 분류한다."""
    if planner_failed or "actions" not in results:
        return "planning", {"steps": 0}

    actions = results["actions"]
    states = results.get("states", [])
    observations = results.get("observations", [])
    if not isinstance(actions, torch.Tensor) or actions.numel() == 0 or not states:
        return "timeout", {"steps": 0}

    actions_cpu = actions.detach().float().cpu()
    cube_z = torch.stack(
        [state["rigid_object"]["Cube1"]["root_pose"][0, 2].detach().float().cpu() for state in states]
    )
    initial_cube_z = float(cube_z[0].item())
    max_lift = float(cube_z.max().item() - initial_cube_z)
    lifted = max_lift >= 0.04

    gripper = actions_cpu[:, -1]
    close_indices = torch.where(gripper < 0.0)[0]
    released = False
    if close_indices.numel():
        released = bool(torch.any(gripper[int(close_indices[0].item()) + 1 :] > 0.0).item())

    tracking_errors: list[torch.Tensor] = []
    for action, obs in zip(actions_cpu, observations, strict=False):
        try:
            ee_pos = obs["subtask_terms"]["ee_pose"][0, :3].detach().float().cpu()
        except (KeyError, TypeError, IndexError):
            continue
        tracking_errors.append(torch.linalg.vector_norm(action[:3] - ee_pos))
    if tracking_errors:
        errors = torch.stack(tracking_errors)
        tracking_p95 = float(torch.quantile(errors, 0.95).item())
        tracking_max = float(errors.max().item())
    else:
        tracking_p95 = 0.0
        tracking_max = 0.0

    details = {
        "steps": int(actions_cpu.shape[0]),
        "max_lift_m": max_lift,
        "released_after_close": released,
        "eef_position_tracking_p95_m": tracking_p95,
        "eef_position_tracking_max_m": tracking_max,
    }
    # 8 cm P95는 slew 지연을 넘어 trajectory 전반에서 target을 놓친 경우다.
    if tracking_p95 > 0.08:
        return "ik_tracking", details
    if not lifted:
        return "grasp", details
    if not released:
        return "timeout", details
    return "place", details


async def _run_data_generator_with_diagnostics(
    env,
    env_id,
    env_reset_queue,
    env_action_queue,
    data_generator,
    success_term,
    pause_subtask=False,
    motion_planner=None,
):
    """공식 coroutine과 동일하게 실행하되 attempt별 failure evidence를 보존한다."""
    while True:
        planning_failures_before = (
            motion_planner.skillgen_stats["planning_failures"] if motion_planner is not None else 0
        )
        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            pause_subtask=pause_subtask,
            motion_planner=motion_planner,
        )
        success = bool(results["success"])
        if success:
            generation.num_success += 1
            _attempt_diagnostics.append({"attempt": generation.num_attempts, "result": "success"})
        else:
            generation.num_failures += 1
            planning_failures_after = (
                motion_planner.skillgen_stats["planning_failures"] if motion_planner is not None else 0
            )
            category, details = _classify_failed_attempt(
                results,
                planner_failed=planning_failures_after > planning_failures_before,
            )
            _failure_categories[category] += 1
            _attempt_diagnostics.append(
                {
                    "attempt": generation.num_attempts,
                    "result": "failure",
                    "category": category,
                    **details,
                }
            )
        generation.num_attempts += 1


def _prepare_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {path}")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _strip_visual_events(env_cfg) -> None:
    for event_name in (
        "randomize_lights",
        "randomize_camera_focal",
        "randomize_robot_color",
        "randomize_cube1_material",
    ):
        if hasattr(env_cfg.events, event_name):
            delattr(env_cfg.events, event_name)


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("Gate E/F baseline requires --num-envs 1")
    if args_cli.generation_num_trials < 1:
        raise ValueError("generation-num-trials must be positive")
    if args_cli.record_images and not args_cli.enable_cameras:
        raise ValueError("--record-images requires --enable_cameras")

    input_path = Path(args_cli.input_file).resolve()
    output_path = Path(args_cli.output_file).resolve()
    summary_path = Path(args_cli.summary_file).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    _prepare_path(output_path, args_cli.overwrite)
    _prepare_path(summary_path, args_cli.overwrite)

    output_dir, output_name = setup_output_paths(str(output_path))
    recorder_cfg = (
        SO101SkillGenRecorderManagerCfg()
        if args_cli.record_images
        else SO101SkillGenJointRecorderManagerCfg()
    )
    env_cfg, success_term = setup_env_config(
        env_name=args_cli.task,
        output_dir=output_dir,
        output_file_name=output_name,
        num_envs=1,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        recorder_cfg=recorder_cfg,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.datagen_config.seed = args_cli.seed
    env_cfg.datagen_config.use_skillgen = True
    env_cfg.datagen_config.generation_guarantee = True
    env_cfg.datagen_config.generation_keep_failed = args_cli.keep_failed
    env_cfg.datagen_config.max_num_failures = args_cli.max_failures
    if not args_cli.record_images:
        remove_pick_cube_cameras(env_cfg)
        _strip_visual_events(env_cfg)

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    generation.num_success = 0
    generation.num_failures = 0
    generation.num_attempts = 0
    _failure_categories.clear()
    _attempt_diagnostics.clear()

    started_at = time.time()
    planner = None
    status = "failed"
    error = None
    env = None
    env_closed = False
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        if not isinstance(env, ManagerBasedRLMimicEnv):
            raise TypeError(f"expected ManagerBasedRLMimicEnv, got {type(env)}")
        env.reset(seed=args_cli.seed)

        planner = SO101CuroboPlanner(
            env=env,
            robot=env.scene["robot"],
            config=make_so101_planner_cfg(
                debug=args_cli.debug_planner,
                visualize_spheres=args_cli.visualize_spheres,
                visualize_plan=args_cli.visualize_plan,
            ),
            env_id=0,
        )
        original_run_data_generator = generation.run_data_generator
        generation.run_data_generator = _run_data_generator_with_diagnostics
        try:
            async_components = setup_async_generation(
                env=env,
                num_envs=1,
                input_file=str(input_path),
                success_term=success_term,
                motion_planners={0: planner},
            )
        finally:
            generation.run_data_generator = original_run_data_generator
        tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        try:
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
            )
            env_closed = True
        finally:
            tasks.cancel()
            try:
                async_components["event_loop"].run_until_complete(tasks)
            except asyncio.CancelledError:
                pass
        status = (
            "complete"
            if generation.num_success >= args_cli.generation_num_trials
            else "failed"
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None and not env_closed:
            env.close()
        planner_stats = planner.skillgen_stats if planner is not None else {}
        summary = {
            "status": status,
            "error": error,
            "official_flow": "isaaclab_mimic.datagen.generation + CuroboPlanner",
            "use_skillgen": True,
            "num_envs": 1,
            "task_id": args_cli.task,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "seed": args_cli.seed,
            "record_images": args_cli.record_images,
            "keep_failed": args_cli.keep_failed,
            "visualize_spheres": args_cli.visualize_spheres,
            "visualize_plan": args_cli.visualize_plan,
            "attempts": generation.num_attempts,
            "successes": generation.num_success,
            "failures": generation.num_failures,
            "success_rate": (
                generation.num_success / generation.num_attempts
                if generation.num_attempts
                else 0.0
            ),
            "failure_categories": {
                category: int(_failure_categories.get(category, 0))
                for category in ("planning", "ik_tracking", "grasp", "place", "timeout")
            },
            "attempt_diagnostics": _attempt_diagnostics,
            "planner": planner_stats,
            "elapsed_seconds": time.time() - started_at,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
