"""SO-101 XRDF 검증·스모크 (PATH E, cuMotion+ROS).

`assets/robots/so101.xrdf` 가 URDF 와 정합하는지, cuMotion(curobo) 가 로드해 FK/IK 가
도는지 대상 머신(Linux 서버, curobo 설치됨)에서 확인한다. sphere 자동 생성기가 아니라
**검증 하니스**다 — sphere 반경/중심 튜닝은 Isaac Sim cuMotion Robot Description Editor 에서.

실행(서버, isaac_ros_cumotion/curobo 가 있는 ROS 환경):
    python scripts/sim/gen_so101_xrdf.py \
        --xrdf assets/robots/so101.xrdf \
        --urdf assets/robots/urdf/so_arm101.urdf

확인 항목:
  1) XRDF + URDF 로 CudaRobotModel 생성 (스키마/링크명/관절명 정합)
  2) 무작위 c-space 샘플에서 FK → tool frame pose 출력
  3) 도달 가능한 무작위 목표에 대한 IK 성공률(>90% 기대)
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="SO-101 XRDF 검증/스모크")
    ap.add_argument("--xrdf", default="assets/robots/so101.xrdf")
    ap.add_argument("--urdf", default="assets/robots/urdf/so_arm101.urdf")
    ap.add_argument("--samples", type=int, default=50)
    args = ap.parse_args()

    try:
        import torch
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    except Exception as exc:  # noqa: BLE001
        print(f"[xrdf] curobo import 실패 — cuMotion/curobo 가 설치된 ROS 환경에서 실행하세요: {exc}")
        return 2

    tensor_args = TensorDeviceType()

    # curobo 는 XRDF + URDF 를 함께 받아 RobotConfig 를 만든다.
    xrdf = load_yaml(args.xrdf)
    robot_cfg = RobotConfig.from_dict(
        {"robot_cfg": {"kinematics": {"xrdf": xrdf, "urdf_path": args.urdf}}},
        tensor_args,
    )

    model = CudaRobotModel(robot_cfg.kinematics)
    print(f"[xrdf] OK — controlled joints: {model.joint_names}")
    print(f"[xrdf] tool/link frames: {model.link_names}")

    # FK 스모크
    q = torch.zeros((1, model.get_dof()), device=tensor_args.device)
    state = model.get_state(q)
    print(f"[xrdf] FK(zero) tool pos: {state.ee_position.cpu().numpy().tolist()}")

    # IK 성공률
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, None, num_seeds=20, tensor_args=tensor_args
    )
    ik = IKSolver(ik_cfg)
    q_rand = ik.sample_configs(args.samples)
    goal = model.get_state(q_rand)
    from curobo.types.math import Pose

    result = ik.solve_batch(Pose(goal.ee_position, goal.ee_quaternion))
    succ = float(result.success.float().mean().item())
    print(f"[xrdf] IK 성공률({args.samples} samples): {succ * 100:.1f}%")
    return 0 if succ > 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
