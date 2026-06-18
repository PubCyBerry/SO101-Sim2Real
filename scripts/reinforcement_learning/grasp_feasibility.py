"""SO-101 grasp 물리 가능성 진단 — IK/접근을 배제하고 hold 물리만 격리 검증.

질문: "큐브를 집게 사이 grasp point 에 정확히 놓았을 때, SO-101 이 물리적으로
잡고(닫기) 들 수 있는가?" RL/IK 가 큐브에 도달하는 문제와 분리한다.

절차(각 env 병렬):
  1) reset 후 그리퍼 open(0.0) 으로 몇 step 안정.
  2) 큐브를 현재 jaw grasp point(=jaw_pos + JAW_GRASP_OFFSET 회전)로 텔레포트.
  3) 닫는 동안(pin_steps) 매 step 큐브를 grasp point 에 재고정(속도 0)하며 그리퍼를
     닫는다(target 1.0). → 집게가 큐브에 접촉/클램프하도록.
  4) pin 해제 후 정지 hold(gravity 부하) + 완만한 lift(shoulder_lift) 를 가하며
     큐브가 grasp point 를 따라오는지(거리 유지) 측정.
  5) 판정: post-release 구간 max distance(cube, grasp_point) < hold_tol 이면 "잡힘".

reward hacking 아님 — 진단 전용(학습/보상과 무관). grasp-assist(weld) 미사용,
순수 물리 접촉만으로 잡히는지 본다.

사용법:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src \
    .venv/bin/python scripts/reinforcement_learning/grasp_feasibility.py \
    --num_envs 32 --cube Cube1 --device cuda:0
"""

import argparse
import json
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 grasp 물리 가능성 진단")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--cube", default="Cube1", help="테스트할 큐브 prim 이름 (Cube1=40mm, Cube3=50mm)")
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--pin_steps", type=int, default=8, help="닫는 동안 큐브를 고정할 step 수(초기 접촉 형성)")
parser.add_argument("--no_dynamic_effort", action="store_true", default=False,
                    help="동적 gripper effort 조정 비활성(현 RL 학습과 동일한 10Nm 고정). 비교 실험용.")
parser.add_argument("--hold_steps", type=int, default=90, help="pin 해제 후 hold+lift step 수")
# 규약(reward 코드 기준): open = joint_pos 높음(>0.6), close = 낮음(→ 하한 -0.174).
parser.add_argument("--open_target", type=float, default=1.4, help="그리퍼 열기 목표각(rad, 높을수록 열림)")
parser.add_argument("--close_target", type=float, default=-0.15, help="그리퍼 닫기 목표각(rad, 낮을수록 닫힘)")
parser.add_argument("--lift_target", type=float, default=0.6, help="shoulder_lift 들기 목표 offset(rad)")
parser.add_argument("--hold_tol", type=float, default=0.04, help="잡힘 판정: cube↔grasp_point 거리 상한(m)")
parser.add_argument("--seed", type=int, default=0)
# --device 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim_to_real  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import SO101_JOINT_ORDER  # noqa: E402

_JAW_GRASP_OFFSET = (-0.021, -0.070, 0.020)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    qw = quat[:, 0:1]
    qv = quat[:, 1:4]
    uv = torch.cross(qv, vec, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return vec + 2.0 * (qw * uv + uuv)


def _grasp_point(robot, num_envs, device) -> torch.Tensor:
    """두 손가락(jaw=가동, gripper=고정) 중점 = 실제 집게 사이 grasp 중심 (num_envs, 3).

    중점이 양 손가락 사이라 gripper open 시 큐브를 놓을 위치로 가장 정확하다.
    한쪽만 있으면 jaw+offset 으로 폴백.
    """
    body_names = robot.data.body_names
    if "jaw" in body_names and "gripper" in body_names:
        j = body_names.index("jaw")
        g = body_names.index("gripper")
        return 0.5 * (robot.data.body_pos_w[:, j, :] + robot.data.body_pos_w[:, g, :])
    if "jaw" in body_names:
        idx = body_names.index("jaw")
        off = torch.tensor(_JAW_GRASP_OFFSET, device=device, dtype=robot.data.body_pos_w.dtype)
        off = off.unsqueeze(0).expand(num_envs, -1)
        return robot.data.body_pos_w[:, idx, :] + _quat_apply_wxyz(robot.data.body_quat_w[:, idx, :], off)
    idx = body_names.index("gripper")
    return robot.data.body_pos_w[:, idx, :]


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # 동적 effort 조정 토글 (PickCubeEnv.step 이 이 플래그를 본다)
        env_cfg.dynamic_reset_gripper_effort_limit = (not args.no_dynamic_effort)
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        device = base.device
        n = base.num_envs

        robot = base.scene["robot"]
        cube = base.scene[args.cube]
        gripper_id = SO101_JOINT_ORDER.index("gripper")  # 5
        lift_id = SO101_JOINT_ORDER.index("shoulder_lift")  # 1

        env.reset()

        def step(action: torch.Tensor):
            env.step(action)

        # action: default(0) 로부터 offset. arm 0 → default 자세 유지, gripper 마지막.
        def make_action(gripper_target: float, lift_offset: float = 0.0) -> torch.Tensor:
            a = torch.zeros(n, len(SO101_JOINT_ORDER), device=device)
            a[:, gripper_id] = gripper_target
            a[:, lift_id] = lift_offset
            return a

        print(json.dumps({"body_names": list(robot.data.body_names)}), flush=True)

        # 1) open + settle (open = 높은 joint 값)
        for _ in range(args.settle_steps):
            step(make_action(args.open_target))

        # 2) 큐브를 grasp point 로 텔레포트
        gp = _grasp_point(robot, n, device)
        quat = torch.zeros(n, 4, device=device); quat[:, 0] = 1.0  # identity wxyz
        pose = torch.cat([gp, quat], dim=-1)
        cube.write_root_pose_to_sim(pose)
        cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

        # 3) 닫는 동안 큐브 pin (재고정) — 집게가 클램프하도록
        for _ in range(args.pin_steps):
            step(make_action(args.close_target))
            gp = _grasp_point(robot, n, device)
            pose = torch.cat([gp, quat], dim=-1)
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))

        # 4) pin 해제 후 hold + lift, 큐브가 grasp point 를 따라오는지 거리 추적
        max_dist = torch.zeros(n, device=device)
        desk_top = 0.709
        held_above = torch.ones(n, dtype=torch.bool, device=device)
        for t in range(args.hold_steps):
            lift = args.lift_target * min(1.0, (t + 1) / 30.0)  # 점진 들기
            step(make_action(args.close_target, lift_offset=lift))
            gp = _grasp_point(robot, n, device)
            cpos = cube.data.root_pos_w
            d = torch.linalg.vector_norm(cpos - gp, dim=1)
            max_dist = torch.maximum(max_dist, d)

        final_cube = cube.data.root_pos_w
        final_gp = _grasp_point(robot, n, device)
        final_dist = torch.linalg.vector_norm(final_cube - final_gp, dim=1)
        gripper_joint = robot.data.joint_pos[:, gripper_id]  # 닫혔는지(<0.5)
        # 잡힘: post-release 내내 grasp point 근처 유지
        held = (max_dist < args.hold_tol)
        # 큐브가 책상 위로 들렸는지(grasp point 가 들렸고 따라옴)
        lifted = (final_cube[:, 2] - desk_top) > 0.03

        held_frac = float(held.float().mean().item())
        held_and_lifted = float((held & lifted).float().mean().item())

        result = {
            "status": "grasp_feasible" if held_frac >= 0.5 else "grasp_infeasible",
            "cube": args.cube,
            "num_envs": n,
            "held_frac": round(held_frac, 3),
            "held_and_lifted_frac": round(held_and_lifted, 3),
            "max_dist_mean_m": round(float(max_dist.mean().item()), 4),
            "max_dist_median_m": round(float(max_dist.median().item()), 4),
            "final_dist_mean_m": round(float(final_dist.mean().item()), 4),
            "gripper_joint_mean_rad": round(float(gripper_joint.mean().item()), 4),
            "hold_tol_m": args.hold_tol,
            "note": ("닫힌 그리퍼가 큐브를 grasp point 근처로 유지 → 물리적 grasp 가능"
                     if held_frac >= 0.5 else
                     "닫아도 큐브가 빠져나감 → 현 물리로는 grasp 어려움(보상으로 못 고침)"),
        }
        print(json.dumps(result), flush=True)

    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc()}), flush=True)
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
