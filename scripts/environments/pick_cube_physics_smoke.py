"""PickCube 강화학습 전 물리 검증 smoke.

검사 범위:
  - USD 정적 물성: cube/bowl/desk/mat collision, mass, friction, contact offset
  - reset/settle 안정성: 큐브·그릇이 꺼지거나 과속/NaN으로 튀지 않는지
  - fixture grasp-hold: gripper-jaw 사이 큐브가 닫힌 그리퍼에 유지되는지
"""

from __future__ import annotations

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PickCube physics smoke")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--resets", type=int, default=20)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--grasp_hold_steps", type=int, default=120)
parser.add_argument("--skip_grasp_hold", action="store_true")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_json", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
from pxr import Usd  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_USD_PATH  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import BOWL_NAME, BOWL_SUCCESS_RADIUS, CUBE_NAMES  # noqa: E402

# mat translate.z = SCENE_OFFSET.z(0.705) + 0.002 = 0.707.
# mat top = 0.707 + half(0.004) = 0.709.  settle 체크: min_cube_bottom_z >= 0.709 - 0.006.
_DESK_TOP_Z = 0.709
_CUBE_HALF = 0.0125
_CUBE_BOTTOM_TOL = 0.006
_LIN_VEL_THRESH = 0.45
_ANG_VEL_THRESH = 5.0
_GRASP_DROP_TOL = 0.035
_GRASP_DIST_TOL = 0.055


def _attr(prim, name: str, default: Any = None) -> Any:
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    value = attr.Get()
    return default if value is None else value


def _rel_targets(prim, name: str) -> list[str]:
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(t) for t in rel.GetTargets()]


def _static_usd_checks() -> dict[str, Any]:
    scene_path = Path(CUBE_DESK_USD_PATH)
    scene_stage = Usd.Stage.Open(str(scene_path))
    if scene_stage is None:
        raise RuntimeError(f"Could not open scene USD: {scene_path}")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, **data: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), **data})

    desk_top = scene_stage.GetPrimAtPath("/Scene/DeskTop")
    desk_mat = scene_stage.GetPrimAtPath("/Scene/DeskMat")
    for label, prim in (("DeskTop", desk_top), ("DeskMat", desk_mat)):
        add_check(
            f"{label}.collision",
            bool(_attr(prim, "physics:collisionEnabled", False)),
            contact_offset=float(_attr(prim, "physxCollision:contactOffset", -1.0)),
            rest_offset=float(_attr(prim, "physxCollision:restOffset", 999.0)),
        )

    mat_z = float(_attr(desk_mat, "xformOp:translate")[2])
    mat_scale_z = float(_attr(desk_mat, "xformOp:scale")[2])
    mat_top_z = mat_z + 0.5 * mat_scale_z
    add_check("DeskMat.top_z", abs(mat_top_z - (_DESK_TOP_Z + mat_scale_z)) < 0.02, mat_top_z=round(mat_top_z, 5))

    for cube_name in CUBE_NAMES:
        cube_path = scene_path.parent / "objects" / cube_name / f"{cube_name}.usd"
        stage = Usd.Stage.Open(str(cube_path))
        if stage is None:
            add_check(f"{cube_name}.usd_open", False, path=str(cube_path))
            continue
        root = stage.GetPrimAtPath(f"/{cube_name}")
        box = stage.GetPrimAtPath(f"/{cube_name}/Box")
        mass = float(_attr(root, "physics:mass", -1.0))
        contact_offset = float(_attr(box, "physxCollision:contactOffset", -1.0))
        scale = tuple(float(v) for v in _attr(box, "xformOp:scale", (0.0, 0.0, 0.0)))
        add_check(
            f"{cube_name}.physics",
            0.02 <= mass <= 0.08
            and bool(_attr(root, "physics:rigidBodyEnabled", False))
            and bool(_attr(root, "physxRigidBody:enableCCD", False))
            and int(_attr(root, "physxRigidBody:solverPositionIterationCount", 0)) >= 16
            and bool(_attr(box, "physics:collisionEnabled", False))
            and 0.002 <= contact_offset <= 0.01
            and all(0.03 <= v <= 0.06 for v in scale)  # Cube1/2=40mm, Cube3/4=50mm
            and _rel_targets(box, "material:binding:physics"),
            mass=mass,
            contact_offset=contact_offset,
            scale=scale,
            physics_material=_rel_targets(box, "material:binding:physics"),
        )

    bowl_path = scene_path.parent / "objects" / BOWL_NAME / f"{BOWL_NAME}.usd"
    bowl_stage = Usd.Stage.Open(str(bowl_path))
    if bowl_stage is None:
        add_check("Bowl.usd_open", False, path=str(bowl_path))
    else:
        root = bowl_stage.GetPrimAtPath("/Bowl")
        bottom = bowl_stage.GetPrimAtPath("/Bowl/Bottom")
        wall = bowl_stage.GetPrimAtPath("/Bowl/Wall")
        # 단일 Mesh 벽: convexDecomposition 충돌 + collisionEnabled 확인
        wall_ok = (
            wall.IsValid()
            and _attr(wall, "physics:approximation") == "convexDecomposition"
            and bool(_attr(wall, "physics:collisionEnabled", False))
        )
        add_check(
            "Bowl.physics",
            float(_attr(root, "physics:mass", -1.0)) >= 0.05
            and bool(_attr(root, "physics:rigidBodyEnabled", False))
            and bool(_attr(bottom, "physics:collisionEnabled", False))
            and wall_ok,
            mass=float(_attr(root, "physics:mass", -1.0)),
            wall_mesh=wall_ok,
            wall_approximation=str(_attr(wall, "physics:approximation")) if wall.IsValid() else "missing",
            bottom_contact_offset=float(_attr(bottom, "physxCollision:contactOffset", -1.0)),
        )

    return {
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }


def _settle_checks(env, device: str) -> dict[str, Any]:
    scene = env.unwrapped.scene
    origins = scene.env_origins
    objects = {name: scene[name] for name in [*CUBE_NAMES, BOWL_NAME]}
    zero_action = torch.zeros(args.num_envs, 6, device=device)
    max_z_drop = 0.0
    max_lin_vel = 0.0
    max_ang_vel = 0.0
    min_cube_bottom_z = float("inf")
    nan_triggered = False

    for _ in range(args.resets):
        env.reset()
        for _step in range(args.settle_steps):
            env.step(zero_action)
        for name, obj in objects.items():
            pos_local = obj.data.root_pos_w - origins
            lin_vel = obj.data.root_lin_vel_w
            ang_vel = obj.data.root_ang_vel_w
            if not torch.isfinite(pos_local).all() or not torch.isfinite(lin_vel).all() or not torch.isfinite(ang_vel).all():
                nan_triggered = True
            default_local = obj.data.default_root_state[:, :3]
            max_z_drop = max(max_z_drop, float((default_local[:, 2] - pos_local[:, 2]).max().item()))
            max_lin_vel = max(max_lin_vel, float(torch.linalg.norm(lin_vel, dim=-1).max().item()))
            max_ang_vel = max(max_ang_vel, float(torch.linalg.norm(ang_vel, dim=-1).max().item()))
            if name in CUBE_NAMES:
                min_cube_bottom_z = min(min_cube_bottom_z, float((pos_local[:, 2] - _CUBE_HALF).min().item()))

    passed = (
        not nan_triggered
        and min_cube_bottom_z >= _DESK_TOP_Z - _CUBE_BOTTOM_TOL
        and max_lin_vel <= _LIN_VEL_THRESH
        and max_ang_vel <= _ANG_VEL_THRESH
    )
    return {
        "pass": passed,
        "resets": args.resets,
        "settle_steps": args.settle_steps,
        "min_cube_bottom_z": round(min_cube_bottom_z, 5),
        "desk_top_z": _DESK_TOP_Z,
        "max_z_drop_m": round(max_z_drop, 5),
        "max_lin_vel_ms": round(max_lin_vel, 5),
        "max_ang_vel_rads": round(max_ang_vel, 5),
        "nan_triggered": nan_triggered,
    }


def _body_pos(robot, body_name: str) -> torch.Tensor:
    body_id = robot.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_id, :]


def _write_cube_pose(cube, pos_w: torch.Tensor) -> None:
    pose = cube.data.root_state_w[:, :7].clone()
    pose[:, :3] = pos_w
    pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=pos_w.device, dtype=pos_w.dtype)
    cube.write_root_pose_to_sim(pose)
    cube.write_root_velocity_to_sim(torch.zeros((pos_w.shape[0], 6), device=pos_w.device, dtype=pos_w.dtype))


def _grasp_hold_check(env, device: str) -> dict[str, Any]:
    if args.skip_grasp_hold:
        return {"pass": True, "skipped": True}

    env.reset()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    cube = scene["Cube1"]

    action = torch.zeros(args.num_envs, 6, device=device)
    action[:, 5] = 1.0
    for _ in range(40):
        env.step(action)

    grip_pos = _body_pos(robot, "gripper")
    jaw_pos = _body_pos(robot, "jaw")
    midpoint = 0.5 * (grip_pos + jaw_pos)
    # 살짝 아래로 둬서 gripper/jaw contact가 큐브 옆면을 누르게 한다.
    fixture_pos = midpoint.clone()
    fixture_pos[:, 2] -= 0.005
    _write_cube_pose(cube, fixture_pos)
    for _ in range(5):
        env.step(action)

    start_z = cube.data.root_pos_w[:, 2].clone()
    max_dist = 0.0
    min_z = float(start_z.min().item())
    action[:, 5] = 0.0
    for _ in range(args.grasp_hold_steps):
        env.step(action)
        grip_pos = _body_pos(robot, "gripper")
        jaw_pos = _body_pos(robot, "jaw")
        midpoint = 0.5 * (grip_pos + jaw_pos)
        cube_pos = cube.data.root_pos_w
        max_dist = max(max_dist, float(torch.linalg.norm(cube_pos - midpoint, dim=-1).max().item()))
        min_z = min(min_z, float(cube_pos[:, 2].min().item()))

    z_drop = max(0.0, float(start_z.max().item()) - min_z)
    passed = z_drop <= _GRASP_DROP_TOL and max_dist <= _GRASP_DIST_TOL
    return {
        "pass": passed,
        "hold_steps": args.grasp_hold_steps,
        "start_z": round(float(start_z[0].item()), 5),
        "min_z": round(min_z, 5),
        "z_drop_m": round(z_drop, 5),
        "max_dist_to_gripper_midpoint_m": round(max_dist, 5),
        "z_drop_tol_m": _GRASP_DROP_TOL,
        "dist_tol_m": _GRASP_DIST_TOL,
        "note": "fixture grasp-hold with PhysX contact only",
    }


def main() -> None:
    env = None
    try:
        static_usd = _static_usd_checks()
        device = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        total_steps = args.resets * (args.settle_steps + 1) + args.grasp_hold_steps + 100
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, total_steps * env_cfg.sim.dt * env_cfg.decimation + 5.0)
        env = gym.make(args.task, cfg=env_cfg)
        settle = _settle_checks(env, device)
        grasp_hold = _grasp_hold_check(env, device)
        env.close()
        env = None

        passed = static_usd["pass"] and settle["pass"] and grasp_hold["pass"]
        result = {
            "task_id": "TA.CUBE.PHYSICS",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "static_usd": static_usd,
            "settle": settle,
            "grasp_hold": grasp_hold,
            "next_if_failed": "RL 시작 전 cube/gripper friction, collision offsets, mass, actuator force/drive, 또는 gripper geometry를 조정",
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not passed:
            sys.exit(1)
    except Exception as exc:
        result = {
            "task_id": "TA.CUBE.PHYSICS",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(1)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
