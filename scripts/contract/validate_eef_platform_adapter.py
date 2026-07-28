#!/usr/bin/env python3
"""SO-101 real/sim EEF FK/IK platform adapter 계약 검증."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from so101_contract.eef_ik import IKConfig  # noqa: E402
from so101_contract.eef_policy_io import (  # noqa: E402
    EEFPlatformAdapterConfig,
    SO101EEFPolicyIO,
)
from so101_contract.follower_calibration import sim_radians_to_real_follower  # noqa: E402


def main() -> None:
    adapter = SO101EEFPolicyIO.from_files(
        ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf",
        ROOT / "assets" / "robots" / "so101.yml",
        ik_config=IKConfig(
            position_tolerance_m=5e-4,
            orientation_tolerance_rad=1e-2,
        ),
        config=EEFPlatformAdapterConfig(
            max_arm_step_rad=0.20,
            max_gripper_step_rad=0.20,
            real_hardware_ik_validated=True,
        ),
    )

    current = np.asarray([0.10, -0.65, 0.80, 0.25, -0.10, 0.20], dtype=np.float32)
    real_current = sim_radians_to_real_follower(current)
    np.testing.assert_allclose(
        adapter.observation_from_sim(current),
        adapter.observation_from_real(real_current),
        atol=2e-6,
        err_msg="matching sim/real joints must produce the same canonical EEF observation",
    )

    # 실제 FK로 만든 도달 가능한 target chunk를 IK가 원 joint trajectory로 복원해야 한다.
    joints = np.stack(
        [
            current,
            current + np.asarray([0.015, -0.010, 0.012, 0.008, -0.006, 0.04]),
            current + np.asarray([0.030, -0.020, 0.024, 0.016, -0.012, 0.08]),
            current + np.asarray([0.045, -0.030, 0.036, 0.024, -0.018, 0.12]),
        ]
    ).astype(np.float32)
    actions = np.stack([adapter.observation_from_sim(joint) for joint in joints])
    sim_result = adapter.action_chunk_to_sim(actions, current)
    if not sim_result.success or sim_result.platform_actions is None:
        raise AssertionError(f"reachable sim chunk failed: {sim_result.reason}")
    np.testing.assert_allclose(
        sim_result.platform_actions,
        joints,
        atol=5e-3,
        err_msg="FK→IK joint round-trip exceeds tolerance",
    )
    real_result = adapter.action_chunk_to_real(actions, real_current)
    if not real_result.success or real_result.platform_actions is None:
        raise AssertionError(f"reachable real chunk failed: {real_result.reason}")
    np.testing.assert_allclose(
        real_result.platform_actions,
        sim_radians_to_real_follower(joints),
        atol=0.35,
        err_msg="canonical IK→real follower conversion mismatch",
    )

    # 도달 불가능한 pose가 한 개라도 있으면 command chunk 전체가 폐기되어야 한다.
    invalid = actions.copy()
    invalid[2, :3] = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    failed = adapter.action_chunk_to_sim(invalid, current)
    if failed.success or failed.platform_actions is not None or failed.canonical_joint_radians is not None:
        raise AssertionError("failed IK chunk leaked commands into the execution result")
    if not failed.replan_required or failed.failed_index != 2:
        raise AssertionError(f"failed chunk policy mismatch: {failed}")
    np.testing.assert_array_equal(failed.hold_canonical_joint_radians, current)

    # Offline sweep 승인 전 real command가 절대 생성되지 않는 deployment gate.
    unvalidated = SO101EEFPolicyIO(
        adapter.ik,
        config=EEFPlatformAdapterConfig(
            max_arm_step_rad=0.20,
            max_gripper_step_rad=0.20,
            real_hardware_ik_validated=False,
        ),
    )
    gated = unvalidated.action_chunk_to_real(actions, real_current)
    if gated.success or gated.platform_actions is not None:
        raise AssertionError("unvalidated real IK unexpectedly produced hardware commands")
    dry_run = unvalidated.action_chunk_to_real_dry_run(actions, real_current)
    if not dry_run.success or dry_run.platform_actions is None:
        raise AssertionError(f"motor-off real dry-run could not produce review targets: {dry_run.reason}")

    print("[eef-platform-adapter] PASS")


if __name__ == "__main__":
    main()
