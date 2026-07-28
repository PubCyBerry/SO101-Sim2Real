#!/usr/bin/env python3
"""Phase 18 — 24 조합 contract-level rollout dry-run (§18, §25.2 마지막 문단).

**이 runner는 Phase 18을 완료시키지 않는다.** 학습된 EEF checkpoint·sim closed-loop
평가·실기기 승인이 없는 상태에서 *계약 수준*으로 실행 가능한 것만 실제로 돌린다.

24 조합(3 policy × 8 representation) **각각**에서 새로 실행하는 것:

1. checkpoint 계약 resolve → 실제 ``ActionRepresentationRouter`` 생성
2. mode별 실제 absolute full chunk 생성(EEF는 실제 FK, joint은 arm radian)
3. **sim boundary**: route → canonical joint command → ``sim_publish_command``
   (EEF는 bounded sequential IK 정확히 1회, joint은 IK 0회)
4. **real boundary**: ``real_dry_run`` route에서 motor command publish가 **0회**임을
   sink counter로 증명. real hardware gate가 닫힌 router는 ``real`` 자체를 거부
5. **실제 ``ActionChunkQueue``** 로 latest_only/overlap/stale/empty/refill 동작 실행
6. malformed/NaN/IK 실패 chunk가 **queue/publish 이전에** 거부되는지 계측
7. 조합별 operational dry-run 지표를 JSONL로 쓰고 실제
   ``evaluate_eef_rollout_metrics.py --mode real-dry-run`` 으로 gate

추가로 한 번만 실행하는 것:

- Phase 17 artifact(``scratch/p17-matrix/phase17_24combo.json``) schema/provenance/
  24×13 PASS 검증과 SHA256 기록
- ``evaluate_eef_rollout_metrics.py`` acceptance-gate self-test(sim/real-dry-run/real
  pass + fail-closed 케이스를 synthetic JSONL로 실제 호출)

의도적으로 주입한 stale/invalid/NaN/IK-실패 guard는 ``expected_guard``로 분리 집계하며
operational failure 지표(``ik_failures``·``invalid_chunks``·``aborts``)에 넣지 않는다.
0-failure acceptance를 속이지 않기 위해서다.

.. code-block:: bash

    python scripts/contract/validate_action_representation_rollout_dry_run.py \\
        --phase17-artifact scratch/p17-matrix/phase17_24combo.json \\
        --output scratch/p18-dry-run/phase18_24combo_dry_run.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_action_routing as p16r  # noqa: E402

# provenance/atomic-write/필수 check 목록은 Phase 17 runner의 것을 그대로 재사용한다.
import validate_action_representation_matrix as p17  # noqa: E402

from so101_contract.action_checkpoint_contract import (  # noqa: E402
    resolve_checkpoint_contract,
)
from so101_contract.action_manifest import (  # noqa: E402
    write_action_representation_manifest,
)
from so101_contract.action_queue import ActionChunkQueue  # noqa: E402
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
    PoseFormat,
    combination_id,
)
from so101_contract.action_routing import make_router  # noqa: E402
from so101_contract.eef_ik import IKConfig  # noqa: E402
from so101_contract.eef_policy_io import (  # noqa: E402
    EEFPlatformAdapterConfig,
    SO101EEFPolicyIO,
)
from so101_contract.eef_relative_action import rot6d_rows_to_matrix  # noqa: E402
from so101_contract.follower_calibration import (  # noqa: E402
    sim_radians_to_real_follower,
)
from so101_contract.joint_feature_codec import sim_publish_command  # noqa: E402
from so101_contract.pose_codec import (  # noqa: E402
    convert_pose_format,
    rotation_geodesic_angle,
)

SCHEMA_VERSION = "so101_action_representation_rollout_dry_run_v1"
POLICY_FAMILIES = ("act", "smolvla", "groot")
HORIZON = p16r.HORIZON
EVALUATOR = Path(__file__).resolve().parent / "evaluate_eef_rollout_metrics.py"

#: Phase 8 sweep 기반 dry-run 허용치(실기기 gate가 아니라 계약 residual 상한).
TOL_POSITION_M = 5.0e-3
TOL_ROTATION_RAD = 5.0e-2

#: 조합마다 반드시 존재해야 하는 dry-run stage. 누락은 조합 실패다.
STAGE_NAMES = (
    "checkpoint_contract_resolve",
    "sim_boundary_route",
    "real_dry_run_boundary",
    "action_queue_operations",
    "invalid_chunk_rejection",
    "acceptance_gate_real_dry_run",
)


def _specs() -> list[ActionRepresentationSpec]:
    """8 representation. Phase 17/§25.1과 같은 순서."""
    entries = [
        ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_ABSOLUTE),
        ActionRepresentationSpec(mode=ActionRepresentationMode.JOINT_RELATIVE),
    ]
    for pose_format in (
        PoseFormat.XYZ_ROT6D_ROWS,
        PoseFormat.XYZ_QUATERNION_WXYZ,
        PoseFormat.XYZ_RPY,
    ):
        for mode in (
            ActionRepresentationMode.EEF_ABSOLUTE,
            ActionRepresentationMode.EEF_RELATIVE,
        ):
            entries.append(ActionRepresentationSpec(mode=mode, pose_format=pose_format))
    return entries


class _MotorCommandSink:
    """실기기 motor command publish를 계측하는 sink.

    ``real_dry_run``은 여기에 절대 도달하면 안 된다. 하드코딩 boolean 대신 실제
    호출 카운터로 0을 증명한다.
    """

    def __init__(self) -> None:
        self.published: list[dict] = []
        self.refused: list[str] = []

    def publish(self, platform: str, actions: np.ndarray) -> None:
        if platform == "real_dry_run":
            self.refused.append("real_dry_run")
            raise AssertionError(
                "motor command publish was attempted on a real_dry_run rollout"
            )
        self.published.append({"platform": platform, "count": int(len(actions))})

    def count(self, platform: str | None = None) -> int:
        if platform is None:
            return sum(entry["count"] for entry in self.published)
        return sum(entry["count"] for entry in self.published if entry["platform"] == platform)


class StageRecorder:
    def __init__(self, combination: str) -> None:
        self.combination = combination
        self.stages: dict[str, dict] = {}

    def run(self, name: str):
        if name not in STAGE_NAMES:
            raise KeyError(f"unknown stage: {name}")
        recorder = self

        class _Ctx:
            def __enter__(self):
                self.evidence: dict = {}
                self.started = time.perf_counter()
                return self.evidence

            def __exit__(self, exc_type, exc, tb):
                duration = round(time.perf_counter() - self.started, 4)
                if exc is None:
                    recorder.stages[name] = {
                        "status": "pass",
                        "evidence": self.evidence,
                        "duration_s": duration,
                    }
                    return False
                recorder.stages[name] = {
                    "status": "fail",
                    "evidence": self.evidence,
                    "duration_s": duration,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": "".join(
                        traceback.format_exception(exc_type, exc, tb, limit=12)
                    ),
                }
                print(f"    FAIL {recorder.combination}/{name}: {type(exc).__name__}: {exc}")
                return True

        return _Ctx()

    def to_dict(self) -> dict:
        missing = [name for name in STAGE_NAMES if name not in self.stages]
        for name in missing:
            self.stages[name] = {
                "status": "fail",
                "evidence": {},
                "duration_s": 0.0,
                "error": "stage did not run (missing stage is a combination failure)",
            }
        ordered = {name: self.stages[name] for name in STAGE_NAMES}
        failed = [name for name, value in ordered.items() if value["status"] == "fail"]
        return {
            "combination": self.combination,
            "status": "fail" if failed else "dry_run_pass",
            "missing_stages": missing,
            "failed_stages": failed,
            "stages": ordered,
        }


def _expect_guard(callable_, exceptions, description: str) -> str:
    try:
        callable_()
    except exceptions as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"expected guard did not trigger: {description}")


def _policy_io(*, real_validated: bool) -> SO101EEFPolicyIO:
    return SO101EEFPolicyIO.from_files(
        p16r.URDF,
        p16r.ROBOT_YAML,
        ik_config=IKConfig(orientation_weight=0.15, max_iterations=80),
        config=EEFPlatformAdapterConfig(
            max_arm_step_rad=1.0,
            max_gripper_step_rad=1.0,
            real_hardware_ik_validated=real_validated,
        ),
    )


def _pose_residuals(spec, chunk: np.ndarray, canonical: np.ndarray, policy_io) -> tuple[float, float]:
    poses = np.asarray(chunk, dtype=np.float64)[:, : spec.pose_dim]
    if spec.pose_format is not PoseFormat.XYZ_ROT6D_ROWS:
        poses = convert_pose_format(poses, spec.pose_format, PoseFormat.XYZ_ROT6D_ROWS)
    achieved = np.stack(
        [
            np.asarray(
                policy_io.kinematics.forward_xyz_rot6d(joints[:5].astype(np.float64)),
                dtype=np.float64,
            )
            for joints in canonical
        ]
    )
    position = float(np.max(np.linalg.norm(achieved[:, :3] - poses[:, :3], axis=-1)))
    rotation = float(
        np.max(
            np.abs(
                rotation_geodesic_angle(
                    rot6d_rows_to_matrix(achieved[:, 3:9]),
                    rot6d_rows_to_matrix(poses[:, 3:9]),
                )
            )
        )
    )
    return position, rotation


def _check_step_limits(canonical: np.ndarray, current: np.ndarray, adapter_config) -> dict:
    """slew/step-limit 준수 증거. 첫 step은 현재 자세 기준으로 잰다."""
    sequence = np.concatenate([np.asarray(current, dtype=np.float64)[None, :], canonical])
    deltas = np.abs(np.diff(sequence, axis=0))
    arm_step = float(np.max(deltas[:, :5]))
    gripper_step = float(np.max(deltas[:, 5]))
    max_arm = float(adapter_config.max_arm_step_rad)
    max_gripper = float(adapter_config.max_gripper_step_rad)
    if arm_step > max_arm + 1e-9:
        raise AssertionError(f"arm step limit violated: {arm_step} > {max_arm}")
    if gripper_step > max_gripper + 1e-9:
        raise AssertionError(f"gripper step limit violated: {gripper_step} > {max_gripper}")
    return {
        "max_arm_step_rad": arm_step,
        "max_gripper_step_rad": gripper_step,
        "limit_arm_step_rad": max_arm,
        "limit_gripper_step_rad": max_gripper,
    }


# --- stage 구현 ---------------------------------------------------------------


def _stage_contract(evidence, directory: Path, spec, family):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = p16r._manifest(spec, policy_type=family)
    write_action_representation_manifest(directory, manifest)
    contract = resolve_checkpoint_contract(directory, expected_policy_type=family)
    if contract.spec != spec:
        raise AssertionError(f"resolved spec mismatch: {contract.spec} != {spec}")
    if contract.policy_type != family:
        raise AssertionError(f"resolved policy mismatch: {contract.policy_type} != {family}")
    evidence.update(
        {
            "checkpoint_dir": str(directory),
            "manifest_sha256": manifest["manifest_sha256"],
            "mode": spec.mode.value,
            "pose_format": spec.pose_format.value,
            "action_dim": contract.action_dim,
            "policy_type": contract.policy_type,
        }
    )
    return contract


def _stage_sim(evidence, contract, spec, chunk, policy_io, sink, metrics):
    counting = p16r._CountingPolicyIO(policy_io)
    router = make_router(contract, policy_io=counting)
    current = p16r._current_state()
    routed = router.route(np.asarray(chunk, dtype=np.float32), current, platform="sim")
    metrics["chunks_received"] += 1
    if not routed.success:
        metrics["ik_failures"] += 1
        raise AssertionError(f"sim route failed: {routed.reason} (index={routed.failed_index})")

    expected_ik = 1 if spec.is_eef else 0
    if routed.ik_calls != expected_ik or counting.calls != expected_ik:
        raise AssertionError(
            f"mode={spec.mode.value} expects {expected_ik} platform IK call(s), got "
            f"routed={routed.ik_calls}, adapter={counting.calls}"
        )
    canonical = routed.canonical_joint_radians
    if canonical.shape != (len(chunk), 6):
        raise AssertionError(f"sim route changed horizon/dof: {canonical.shape}")
    if canonical.dtype != np.float32 or routed.platform_dtype != "float32":
        raise AssertionError("sim command dtype must be float32")
    if not np.all(np.isfinite(canonical)):
        raise AssertionError("sim command contains non-finite joint targets")

    published = np.stack([sim_publish_command(row) for row in routed.platform_actions])
    if not np.array_equal(published, routed.platform_actions):
        raise AssertionError("sim publish applied a second unit conversion")
    sink.publish("sim", published)
    metrics["commands_published"] += int(len(published))
    metrics["ik_steps"] += int(len(canonical)) if spec.is_eef else 0

    limits = _check_step_limits(canonical, current, router.adapter_config)
    residuals = {"position_residual_max_m": 0.0, "orientation_residual_max_rad": 0.0}
    if spec.is_eef:
        position, rotation = _pose_residuals(spec, chunk, canonical, policy_io)
        if position > TOL_POSITION_M or rotation > TOL_ROTATION_RAD:
            raise AssertionError(
                f"sim pose residual over tolerance: pos={position} rot={rotation}"
            )
        residuals = {
            "position_residual_max_m": position,
            "orientation_residual_max_rad": rotation,
        }
    metrics["position_residual_max_m"] = max(
        metrics["position_residual_max_m"], residuals["position_residual_max_m"]
    )
    metrics["orientation_residual_max_rad"] = max(
        metrics["orientation_residual_max_rad"], residuals["orientation_residual_max_rad"]
    )

    evidence.update(
        {
            "route": list(routed.route),
            "ik_calls": routed.ik_calls,
            "adapter_calls": counting.calls,
            "horizon": int(len(canonical)),
            "dtype": {"input": routed.input_dtype, "platform": routed.platform_dtype},
            "commands_published": int(len(published)),
            "sim_publish": "canonical_passthrough",
            "limits": limits,
            **residuals,
            "joint_absolute_fallback_selectable": not spec.is_eef,
        }
    )
    return router, routed


def _stage_real_dry_run(evidence, contract, spec, chunk, policy_io, sink, metrics):
    router = make_router(contract, policy_io=policy_io)
    real_state = np.asarray(sim_radians_to_real_follower(p16r._current_state()), dtype=np.float32)
    routed = router.route(
        np.asarray(chunk, dtype=np.float32), real_state, platform="real_dry_run"
    )
    metrics["dry_run_chunks"] += 1
    if not routed.success:
        metrics["ik_failures"] += 1
        raise AssertionError(f"real dry-run route failed: {routed.reason}")
    if routed.ik_calls != (1 if spec.is_eef else 0):
        raise AssertionError(f"real dry-run IK call count wrong: {routed.ik_calls}")

    before = sink.count()
    guard = _expect_guard(
        lambda: sink.publish("real_dry_run", routed.platform_actions),
        AssertionError,
        "real_dry_run must never reach the motor command sink",
    )
    after = sink.count()
    if after != before or sink.count("real_dry_run") != 0:
        raise AssertionError(
            f"real_dry_run published motor commands: {sink.count('real_dry_run')}"
        )
    metrics["commands_sent"] += sink.count("real_dry_run")

    # `EEF_IK_REAL_VALIDATED` gate는 **IK로 만든** joint command에만 걸린다.
    # EEF mode는 gate가 닫히면 `real` 플랫폼 자체가 거부돼야 하고, joint mode는 IK를
    # 쓰지 않으므로 gate 대상이 아니며 그래서 joint_absolute fallback이 즉시 선택 가능하다.
    gated_router = make_router(contract, policy_io=_policy_io(real_validated=False))
    if spec.is_eef:
        gate_guard = _expect_guard(
            lambda: _require_route_success(
                gated_router.route(
                    np.asarray(chunk, dtype=np.float32), real_state, platform="real"
                )
            ),
            (AssertionError, ValueError, PermissionError, RuntimeError),
            "real platform must be refused while EEF_IK_REAL_VALIDATED is false",
        )
    else:
        gated = gated_router.route(
            np.asarray(chunk, dtype=np.float32), real_state, platform="real"
        )
        if not gated.success:
            raise AssertionError(
                "joint mode must stay selectable as the real fallback without IK validation: "
                f"{gated.reason}"
            )
        if gated.ik_calls != 0:
            raise AssertionError(f"joint real route called IK {gated.ik_calls} times")
        gate_guard = (
            "not_applicable: joint mode routes without IK, so the EEF hardware gate does "
            "not apply and joint_absolute stays selectable as the real fallback"
        )
    evidence.update(
        {
            "dry_run_chunks": 1,
            "motor_commands_published": sink.count("real_dry_run"),
            "sink_refusals": len(sink.refused),
            "expected_guard": {
                "publish_attempt_refused": guard,
                "real_hardware_gate_closed": gate_guard,
            },
            "real_boundary": "real_follower -> canonical radian (exactly once)",
        }
    )


def _require_route_success(routed):
    if not routed.success:
        raise PermissionError(f"route refused: {routed.reason}")
    return routed


def _stage_queue(evidence, spec, routed, metrics):
    """실제 ActionChunkQueue로 latest_only/overlap/stale/empty/refill 실행."""
    queue = ActionChunkQueue(aggregate_fn_name="latest_only")
    actions = [np.asarray(row, dtype=np.float32) for row in routed.platform_actions]

    # 1) 최초 chunk merge + refill 판단
    queue.merge([(index, action) for index, action in enumerate(actions)])
    if len(queue) != len(actions):
        raise AssertionError(f"queue did not accept the full chunk: {len(queue)}")
    ready_full = queue.ready_to_send_observation(0.5)

    # 2) 2 step 실행
    dequeued = [queue.pop_next() for _ in range(2)]
    if [timestep for timestep, _ in dequeued] != [0, 1]:
        raise AssertionError(f"queue popped out of order: {dequeued}")
    metrics["commands_routed"] += len(dequeued)

    # 3) overlap merge: latest_only는 새 chunk 값을 그대로 취한다
    overlap_value = actions[2] + np.float32(0.01)
    queue.merge([(2, overlap_value), (3, actions[3]), (4, actions[0])])
    overlap_timestep, overlap_action = queue.pop_next()
    if overlap_timestep != 2:
        raise AssertionError(f"overlap merge lost ordering: {overlap_timestep}")
    if not np.allclose(overlap_action, overlap_value, atol=0.0, rtol=0.0):
        raise AssertionError("latest_only did not take the newest overlapping action")

    # 4) stale chunk: 이미 실행한 timestep 이하는 버려진다(주입 guard)
    stale_before = len(queue)
    queue.merge([(0, actions[0]), (1, actions[1])])
    stale_dropped = stale_before - len(queue) if len(queue) < stale_before else 0
    if any(timestep <= queue.latest_action for timestep in queue.timesteps()):
        raise AssertionError(f"stale timesteps survived the merge: {queue.timesteps()}")
    # 주입한 stale merge 자체가 guard 1건이다(버려진 timestep 수와 별개로 센다).
    metrics["injected_guard_events"] += 1
    stale_guard_events = 1

    # 5) empty merge는 상태를 바꾸지 않는다
    snapshot = queue.timesteps()
    queue.merge([])
    if queue.timesteps() != snapshot:
        raise AssertionError("empty merge mutated the queue")

    # 6) drain → starvation(빈 queue pop은 IndexError)
    drained = 0
    while queue.has_actions():
        queue.pop_next()
        drained += 1
    metrics["commands_routed"] += drained
    starvation_guard = _expect_guard(
        queue.pop_next, IndexError, "empty queue pop must raise instead of publishing"
    )
    # 의도적으로 비운 queue다. operational starvation 지표(`queue_starvation_ticks`)에
    # 넣으면 0-failure acceptance를 속이게 되므로 guard 카운터로 따로 센다.
    metrics["injected_guard_events"] += 1
    empty_pop_guard_events = 1

    # 7) refill: 비었으면 must_go 관측을 요구한다
    if not queue.ready_to_send_observation(0.0):
        raise AssertionError("drained queue must request a refill observation")
    if not queue.observation_must_go():
        raise AssertionError("drained queue must mark the next observation must_go")
    queue.mark_observation_sent(True)
    queue.merge([(index + 10, action) for index, action in enumerate(actions)])
    if len(queue) != len(actions):
        raise AssertionError("refill merge did not restore the queue")

    evidence.update(
        {
            "aggregate_fn": "latest_only",
            "chunk_size": queue.action_chunk_size,
            "ready_to_send_at_full_queue": ready_full,
            "dequeued_timesteps": [timestep for timestep, _ in dequeued],
            "overlap": "latest_only took the newest action",
            "refilled_len": len(queue),
            "expected_guard": {
                "stale_merge_events": stale_guard_events,
                "stale_timesteps_dropped": stale_dropped,
                "empty_queue_pop_events": empty_pop_guard_events,
                "empty_queue_pop_refused": starvation_guard,
                "injected_guard_events": stale_guard_events + empty_pop_guard_events,
            },
        }
    )


def _stage_invalid_chunks(evidence, contract, spec, chunk, policy_io, sink, metrics):
    """malformed/NaN/IK 실패 chunk가 queue/publish 이전에 거부되는지 계측."""
    counting = p16r._CountingPolicyIO(policy_io)
    router = make_router(contract, policy_io=counting)
    current = p16r._current_state()
    queue = ActionChunkQueue(aggregate_fn_name="latest_only")
    published_before = sink.count()
    guards: dict[str, str] = {}

    def _route_and_publish(bad_chunk: np.ndarray):
        routed = router.route(np.asarray(bad_chunk, dtype=np.float32), current, platform="sim")
        if not routed.success:
            raise PermissionError(f"routing refused: {routed.reason}")
        queue.merge(list(enumerate(routed.platform_actions)))
        sink.publish("sim", routed.platform_actions)

    nan_chunk = np.asarray(chunk, dtype=np.float32).copy()
    nan_chunk[1, 0] = np.nan
    guards["nan_chunk"] = _expect_guard(
        lambda: _route_and_publish(nan_chunk),
        (PermissionError, ValueError, AssertionError),
        "NaN chunk must be refused before the command queue",
    )

    malformed = np.asarray(chunk, dtype=np.float32)[:, :-1]
    guards["malformed_dim_chunk"] = _expect_guard(
        lambda: _route_and_publish(malformed),
        (PermissionError, ValueError, AssertionError),
        "wrong action dimension must be refused",
    )

    rank1 = np.asarray(chunk, dtype=np.float32)[0]
    guards["rank1_chunk"] = _expect_guard(
        lambda: _route_and_publish(rank1),
        (PermissionError, ValueError, AssertionError),
        "2D chunk contract must reject a rank-1 action",
    )

    unreachable = np.asarray(chunk, dtype=np.float32).copy()
    if spec.is_eef:
        unreachable[:, 0] += np.float32(5.0)  # 도달 불가 위치 → IK 실패
        guard_name = "unreachable_eef_chunk"
    else:
        unreachable[:, 0] += np.float32(50.0)  # step/limit 위반 joint target
        guard_name = "out_of_limit_joint_chunk"
    guards[guard_name] = _expect_guard(
        lambda: _route_and_publish(unreachable),
        (PermissionError, ValueError, AssertionError),
        f"{guard_name} must be refused before publish",
    )

    if len(queue) != 0:
        raise AssertionError(f"rejected chunks reached the command queue: {queue.timesteps()}")
    if sink.count() != published_before:
        raise AssertionError("rejected chunks reached the motor command sink")
    # 주입한 invalid chunk 4종은 guard 카운터로만 센다(operational `invalid_chunks`는 0 유지).
    metrics["injected_guard_events"] += len(guards)
    evidence.update(
        {
            "queue_len_after_rejections": len(queue),
            "commands_published_delta": sink.count() - published_before,
            "adapter_calls_during_rejection": counting.calls,
            "injected_guard_events": len(guards),
            "expected_guard": guards,
        }
    )


def _stage_acceptance_gate(evidence, directory: Path, metrics):
    """조합별 operational 지표를 실제 evaluator(real-dry-run mode)로 gate."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "final",
        "chunks_received": metrics["chunks_received"],
        "dry_run_chunks": metrics["dry_run_chunks"],
        "commands_sent": metrics["commands_sent"],
        "ik_failures": metrics["ik_failures"],
        "aborts": metrics["aborts"],
        "invalid_chunks": metrics["invalid_chunks"],
        "commands_published": metrics["commands_published"],
        "commands_routed": metrics["commands_routed"],
        "queue_starvation_ticks": metrics["queue_starvation_ticks"],
        "empty_chunks": metrics["empty_chunks"],
        "stale_chunks": metrics["stale_chunks"],
        "injected_guard_events": metrics["injected_guard_events"],
        "ik_steps": metrics["ik_steps"],
        "position_residual_max_m": metrics["position_residual_max_m"],
        "orientation_residual_max_rad": metrics["orientation_residual_max_rad"],
    }
    metrics_path = directory / "real_dry_run_metrics.jsonl"
    metrics_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--mode",
            "real-dry-run",
            "--metrics",
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "real-dry-run acceptance gate failed:\n"
            + (result.stdout + result.stderr).strip()[-1200:]
        )
    evidence.update(
        {
            "mode": "real-dry-run",
            "evaluator": str(EVALUATOR.relative_to(ROOT)),
            "metrics": record,
            "stdout_tail": result.stdout.strip().splitlines()[-3:],
        }
    )
    return record


# --- 조합 실행 ----------------------------------------------------------------


def run_combination(family: str, spec, *, workspace: Path, policy_io) -> dict:
    combination = combination_id(family, spec)
    recorder = StageRecorder(combination)
    combo_dir = workspace / combination
    sink = _MotorCommandSink()
    metrics = {
        "chunks_received": 0,
        "dry_run_chunks": 0,
        "commands_sent": 0,
        "commands_published": 0,
        "commands_routed": 0,
        "ik_failures": 0,
        "ik_steps": 0,
        "invalid_chunks": 0,
        "aborts": 0,
        "queue_starvation_ticks": 0,
        "empty_chunks": 0,
        "stale_chunks": 0,
        "injected_guard_events": 0,
        "position_residual_max_m": 0.0,
        "orientation_residual_max_rad": 0.0,
    }

    contract = None
    with recorder.run("checkpoint_contract_resolve") as evidence:
        contract = _stage_contract(evidence, combo_dir, spec, family)

    chunk = p16r._absolute_chunk(spec, policy_io)
    routed = None
    if contract is not None:
        with recorder.run("sim_boundary_route") as evidence:
            _, routed = _stage_sim(evidence, contract, spec, chunk, policy_io, sink, metrics)

        with recorder.run("real_dry_run_boundary") as evidence:
            _stage_real_dry_run(evidence, contract, spec, chunk, policy_io, sink, metrics)

    if routed is not None:
        with recorder.run("action_queue_operations") as evidence:
            _stage_queue(evidence, spec, routed, metrics)

    if contract is not None:
        with recorder.run("invalid_chunk_rejection") as evidence:
            _stage_invalid_chunks(evidence, contract, spec, chunk, policy_io, sink, metrics)

    final_metrics = None
    with recorder.run("acceptance_gate_real_dry_run") as evidence:
        final_metrics = _stage_acceptance_gate(evidence, combo_dir, metrics)

    payload = recorder.to_dict()
    payload["operational_metrics"] = final_metrics or metrics
    payload["motor_commands_published_real"] = sink.count("real_dry_run")
    shutil.rmtree(combo_dir, ignore_errors=True)
    return payload


# --- 한 번만 실행하는 검증 ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


_SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_LEROBOT_VERSION = "0.6.0"


def phase17_expected_combination_ids() -> tuple[str, ...]:
    """Phase 17 runner 자신의 policy family × 8 spec에서 유도한 24개 조합 ID."""
    # Phase 17 runner는 8개 spec을 Phase 15 validator(`p17.p15`)에서 가져온다.
    ids = tuple(
        combination_id(family, spec)
        for family in p17.POLICY_FAMILIES
        for _fixture, spec in p17.p15._specs()
    )
    if len(set(ids)) != 24:
        raise AssertionError(f"Phase 17 expected id set is not 24 unique ids: {len(set(ids))}")
    if set(ids) != set(expected_combination_ids()):
        raise AssertionError(
            "Phase 17 runner and this dry-run runner disagree on the 24 combination ids"
        )
    return ids


def _require_hex(value, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.match(value) is None:
        raise AssertionError(f"Phase 17 {label} is not a valid hex digest: {value!r}")
    return value


def verify_phase17_artifact(path: Path) -> dict:
    """Phase 17 artifact의 schema/provenance/**정확한 24×13 PASS**를 검증한다.

    ID uniqueness만 보면 임의의 24개 ID도 통과한다. 그래서 여기서는 §25.1 expected ID
    set과의 완전 일치, entry별 check key set 일치, totals 값 일치, provenance 필드의
    형식(40-hex commit·nonempty branch·bool dirty·source object·image sha256·LeRobot
    version/commit)까지 모두 확인하고 하나라도 어긋나면 fail-closed한다.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Phase 17 artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Phase 17 artifact is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Phase 17 artifact must be a JSON object: {path}")
    if payload.get("schema_version") != p17.SCHEMA_VERSION:
        raise AssertionError(
            f"unexpected Phase 17 schema: {payload.get('schema_version')!r} "
            f"(expected {p17.SCHEMA_VERSION!r})"
        )

    check_names = tuple(p17.CHECK_NAMES)
    expected_checks = 24 * len(check_names)
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise TypeError(f"Phase 17 artifact has no totals object: {type(totals).__name__}")
    expected_totals = {
        "expected_combinations": 24,
        "ran_combinations": 24,
        "passed_combinations": 24,
        "failed_combinations": 0,
        "expected_checks": expected_checks,
        "ran_checks": expected_checks,
        "passed_checks": expected_checks,
    }
    mismatched = {
        key: totals.get(key)
        for key, value in expected_totals.items()
        if totals.get(key) != value
    }
    if mismatched:
        raise AssertionError(
            f"Phase 17 totals do not match the required 24×{len(check_names)} matrix: "
            f"{mismatched} (expected {expected_totals})"
        )
    if totals.get("complete") is not True:
        raise AssertionError("Phase 17 artifact is not marked complete")

    combinations = payload.get("combinations")
    if not isinstance(combinations, list):
        raise TypeError(
            f"Phase 17 artifact combinations must be a list: {type(combinations).__name__}"
        )
    expected_ids = phase17_expected_combination_ids()
    ran_ids = [entry.get("combination") for entry in combinations]
    if any(not isinstance(name, str) or not name for name in ran_ids):
        raise AssertionError(f"Phase 17 artifact has malformed combination ids: {ran_ids}")
    duplicates = sorted({name for name in ran_ids if ran_ids.count(name) > 1})
    missing = sorted(set(expected_ids) - set(ran_ids))
    unexpected = sorted(set(ran_ids) - set(expected_ids))
    if duplicates or missing or unexpected or len(ran_ids) != 24:
        raise AssertionError(
            "Phase 17 combination id set does not match the 24 expected ids: "
            f"count={len(ran_ids)}, duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )

    for entry in combinations:
        name = entry["combination"]
        if entry.get("status") != "pass":
            raise AssertionError(f"{name} combination status={entry.get('status')!r} is not pass")
        checks = entry.get("checks")
        if not isinstance(checks, dict):
            raise TypeError(f"{name} has no checks object: {type(checks).__name__}")
        if tuple(checks.keys()) != check_names:
            raise AssertionError(
                f"{name} check key set does not match CHECK_NAMES: got {list(checks)}"
            )
        bad = [key for key, value in checks.items() if value.get("status") != "pass"]
        if bad:
            raise AssertionError(f"{name} has non-pass checks: {bad}")

    git = payload.get("git")
    if not isinstance(git, dict):
        raise TypeError(f"Phase 17 artifact has no git object: {type(git).__name__}")
    _require_hex(git.get("commit"), _SHA1_HEX, "git.commit")
    branch = git.get("branch")
    if not isinstance(branch, str) or not branch.strip():
        raise AssertionError(f"Phase 17 git.branch must be a nonempty string: {branch!r}")
    if not isinstance(git.get("dirty"), bool):
        raise AssertionError(
            f"Phase 17 git.dirty must be a bool (unknown provenance is not acceptable "
            f"for an authoritative artifact): {git.get('dirty')!r}"
        )
    source = git.get("provenance_source")
    if not isinstance(source, dict):
        raise TypeError(
            f"Phase 17 git.provenance_source must be an object: {type(source).__name__}"
        )
    for field in ("commit", "branch", "dirty"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(
                f"Phase 17 git.provenance_source.{field} must be a nonempty string: {value!r}"
            )

    _require_hex(
        str(payload.get("docker_image_id", "")).removeprefix("sha256:"),
        _SHA256_HEX,
        "docker_image_id",
    )
    if not str(payload.get("docker_image_id", "")).startswith("sha256:"):
        raise AssertionError(
            f"Phase 17 docker_image_id must be sha256-prefixed: {payload.get('docker_image_id')!r}"
        )
    lerobot = payload.get("lerobot")
    if not isinstance(lerobot, dict):
        raise TypeError(f"Phase 17 artifact has no lerobot object: {type(lerobot).__name__}")
    if lerobot.get("version") != _EXPECTED_LEROBOT_VERSION:
        raise AssertionError(
            f"Phase 17 lerobot.version={lerobot.get('version')!r} "
            f"(expected {_EXPECTED_LEROBOT_VERSION!r})"
        )
    _require_hex(lerobot.get("commit"), _SHA1_HEX, "lerobot.commit")

    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": payload["schema_version"],
        "generated_at": payload.get("generated_at"),
        "totals": totals,
        "git": git,
        "docker_image_id": payload.get("docker_image_id"),
        "lerobot": lerobot,
        "combination_ids_verified": len(ran_ids),
        "checks_per_combination": len(check_names),
        "verified": {
            "exact_expected_id_set": True,
            "check_key_set_matches_CHECK_NAMES": True,
            "totals_exact": True,
            "provenance_format": True,
        },
        "provenance_complete": True,
        "provenance_note": (
            "historical input artifact의 provenance 형식과 SHA256을 검증한 것이다. 현재 "
            "Phase 18 실행 환경과의 동일성 비교가 아니며, 이번 실행의 provenance는 "
            "top-level git/docker_image_id/lerobot에 따로 기록된다."
        ),
    }


def expected_combination_ids() -> tuple[str, ...]:
    """§25.1의 24개 조합 ID. completion은 이 set과 정확히 일치해야 한다."""
    return tuple(
        combination_id(family, spec) for family in POLICY_FAMILIES for spec in _specs()
    )


def load_external_report(path: Path, *, expected_mode: str) -> dict:
    """외부 sim/real evaluator report(JSON object)를 실제로 읽고 검증한다.

    ``evaluate_eef_rollout_metrics.py --output``이 만든 형태를 기대한다. 파일이 없거나
    mode/status/failures/final event 중 하나라도 어긋나면 **예외**를 던져 전체 실행을
    실패시킨다. 존재만 확인하고 통과시키면 acceptance를 위조하게 된다.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{expected_mode} acceptance report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{expected_mode} acceptance report is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{expected_mode} acceptance report must be a JSON object: {path}")

    mode = payload.get("mode")
    allowed_modes = {"sim": {"sim"}, "real": {"real", "real-dry-run"}}[expected_mode]
    if mode not in allowed_modes:
        raise ValueError(
            f"{expected_mode} acceptance report declares mode={mode!r}; "
            f"expected one of {sorted(allowed_modes)}"
        )
    status = payload.get("status")
    if status != "PASS":
        raise ValueError(f"{expected_mode} acceptance report status={status!r} is not PASS")
    failures = payload.get("failures")
    if not isinstance(failures, list):
        raise TypeError(
            f"{expected_mode} acceptance report must carry a 'failures' list, got {type(failures).__name__}"
        )
    if failures:
        raise ValueError(f"{expected_mode} acceptance report has failures: {failures}")
    final = payload.get("final")
    if not isinstance(final, dict) or final.get("event") != "final":
        raise ValueError(
            f"{expected_mode} acceptance report has no final metrics record "
            f"(final.event={None if not isinstance(final, dict) else final.get('event')!r})"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "mode": mode,
        "status": status,
        "failures": failures,
        "final_event": final.get("event"),
        "derived": payload.get("derived"),
    }


def acceptance_gate_self_test(workspace: Path) -> dict:
    """evaluator를 synthetic JSONL로 실제 호출한다(pass 3 + fail-closed 4)."""
    workspace.mkdir(parents=True, exist_ok=True)
    sim_pass = {
        "event": "final",
        "inference_requests": 10,
        "chunks_received": 10,
        "commands_published": 40,
        "ik_steps": 40,
        "ik_failures": 0,
        "invalid_chunks": 0,
        "aborts": 0,
        "position_residual_max_m": 1e-4,
        "orientation_residual_max_rad": 1e-3,
        "queue_starvation_ticks": 0,
        "empty_chunks": 0,
        "stale_chunks": 0,
    }
    sim_eval = {"all_cubes_success_rate": 1.0}
    dry_pass = {
        "event": "final",
        "chunks_received": 4,
        "dry_run_chunks": 4,
        "commands_sent": 0,
        "ik_failures": 0,
        "aborts": 0,
    }
    real_pass = {
        "event": "final",
        "chunks_received": 4,
        "commands_sent": 8,
        "residual_samples": 8,
        "ik_failures": 0,
        "aborts": 0,
        "position_residual_max_m": 1e-3,
        "orientation_residual_max_rad": 1e-2,
    }

    def _write(name: str, record: dict) -> Path:
        path = workspace / f"{name}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path

    eval_path = workspace / "sim_eval.json"
    eval_path.write_text(json.dumps(sim_eval), encoding="utf-8")

    cases: list[tuple[str, list[str], bool]] = [
        ("sim_pass", ["--mode", "sim", "--metrics", str(_write("sim_pass", sim_pass)),
                      "--eval", str(eval_path)], True),
        ("real_dry_run_pass", ["--mode", "real-dry-run",
                               "--metrics", str(_write("dry_pass", dry_pass))], True),
        ("real_pass", ["--mode", "real", "--metrics", str(_write("real_pass", real_pass))], True),
        ("sim_fail_ik_failure", ["--mode", "sim",
                                 "--metrics", str(_write("sim_ik", {**sim_pass, "ik_failures": 1})),
                                 "--eval", str(eval_path)], False),
        ("sim_fail_success_rate", ["--mode", "sim", "--metrics", str(_write("sim_ok2", sim_pass)),
                                   "--eval", str(_eval_file(workspace, 0.5))], False),
        ("real_dry_run_fail_motor_command", ["--mode", "real-dry-run",
                                             "--metrics", str(_write("dry_cmd", {**dry_pass, "commands_sent": 1}))],
         False),
        ("fail_missing_final_event", ["--mode", "real-dry-run",
                                      "--metrics", str(_write("dry_nofinal", {**dry_pass, "event": "tick"}))],
         False),
    ]
    results = []
    for name, argv, expect_pass in cases:
        completed = subprocess.run(
            [sys.executable, str(EVALUATOR), *argv], capture_output=True, text=True, check=False
        )
        passed = completed.returncode == 0
        if passed != expect_pass:
            raise AssertionError(
                f"acceptance gate self-test case {name!r} expected "
                f"{'pass' if expect_pass else 'fail-closed'}, got returncode "
                f"{completed.returncode}:\n{(completed.stdout + completed.stderr)[-800:]}"
            )
        results.append(
            {
                "case": name,
                "expected": "pass" if expect_pass else "fail_closed",
                "returncode": completed.returncode,
                "report": _parse_evaluator_stdout(completed, case=name),
            }
        )
    return {"status": "pass", "cases": results}


def _parse_evaluator_stdout(completed, *, case: str) -> dict:
    """evaluator stdout(JSON report)에서 쓸모 있는 evidence를 뽑는다.

    판정 자체는 returncode가 하지만, artifact에 마지막 줄 ``}``만 남기면 증거로서
    무가치하다. parser는 fail-closed다 — stdout이 JSON object가 아니거나 필수 key가
    없으면 예외를 던져 self-test 전체를 실패시킨다(malformed를 성공으로 보지 않는다).
    """
    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise AssertionError(
            f"acceptance gate self-test case {case!r} produced no evaluator report on stdout; "
            f"stderr={(completed.stderr or '').strip()[-400:]!r}"
        )
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"acceptance gate self-test case {case!r} printed malformed evaluator output: {exc}; "
            f"stdout={stdout[-400:]!r}"
        ) from exc
    if not isinstance(report, dict):
        raise AssertionError(
            f"acceptance gate self-test case {case!r} evaluator report is not an object: "
            f"{type(report).__name__}"
        )
    missing = [key for key in ("status", "mode", "failures", "final") if key not in report]
    if missing:
        raise AssertionError(
            f"acceptance gate self-test case {case!r} evaluator report is missing {missing}"
        )
    failures = report["failures"]
    if not isinstance(failures, list):
        raise AssertionError(
            f"acceptance gate self-test case {case!r} evaluator 'failures' is not a list: "
            f"{type(failures).__name__}"
        )
    final = report["final"]
    if not isinstance(final, dict):
        raise AssertionError(
            f"acceptance gate self-test case {case!r} evaluator 'final' is not an object: "
            f"{type(final).__name__}"
        )
    expected_status = "PASS" if completed.returncode == 0 else "FAIL"
    if report["status"] != expected_status:
        raise AssertionError(
            f"acceptance gate self-test case {case!r} evaluator status={report['status']!r} "
            f"disagrees with returncode {completed.returncode}"
        )
    return {
        "status": report["status"],
        "mode": report["mode"],
        "final_event": final.get("event"),
        "failures": failures,
        "derived": report.get("derived"),
    }


def _eval_file(workspace: Path, success_rate: float) -> Path:
    path = workspace / f"sim_eval_{success_rate}.json"
    path.write_text(json.dumps({"all_cubes_success_rate": success_rate}), encoding="utf-8")
    return path


# --- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase17-artifact",
        type=Path,
        default=ROOT / "scratch" / "p17-matrix" / "phase17_24combo.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scratch" / "p18-dry-run" / "phase18_24combo_dry_run.json",
    )
    parser.add_argument("--policies", default=",".join(POLICY_FAMILIES))
    parser.add_argument(
        "--representations",
        default="",
        help="빠른 개발용 필터(쉼표 구분 stats profile kind). completion 실행에서는 비워 둔다.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sim-eval-report",
        type=Path,
        default=None,
        help="외부 sim closed-loop evaluator report(JSON). 주면 sim acceptance를 조립한다.",
    )
    parser.add_argument(
        "--real-rollout-report",
        type=Path,
        default=None,
        help="외부 real guarded rollout report(JSON). 주면 real acceptance를 조립한다.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    families = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = [name for name in families if name not in POLICY_FAMILIES]
    if unknown:
        parser.error(f"unknown policy families: {unknown}")
    duplicates = sorted({name for name in families if families.count(name) > 1})
    if duplicates:
        # 중복 policy는 조합 수만 부풀리고 실제 coverage는 늘리지 않는다. 여기서 막지 않으면
        # `--policies act,act,smolvla`가 GR00T 없이 24개를 채워 completion을 위조한다.
        parser.error(f"duplicate policy families are refused: {duplicates}")
    representation_filter = {
        name.strip() for name in args.representations.split(",") if name.strip()
    }
    unknown_representations = representation_filter - {
        spec.stats_profile_kind for spec in _specs()
    }
    if unknown_representations:
        parser.error(f"unknown representations: {sorted(unknown_representations)}")

    workspace = Path(
        args.work_dir or tempfile.mkdtemp(prefix="p18-dry-run-", dir=str(ROOT / "scratch"))
    )
    workspace.mkdir(parents=True, exist_ok=True)
    policy_io = _policy_io(real_validated=True)

    started = time.time()
    phase17: dict = {}
    phase17_error: str | None = None
    try:
        phase17 = verify_phase17_artifact(args.phase17_artifact)
    except Exception as exc:  # noqa: BLE001 - artifact 실패도 결과에 남긴다
        phase17_error = f"{type(exc).__name__}: {exc}"
        print(f"FAIL phase17_artifact: {phase17_error}")

    gate_self_test: dict = {}
    gate_error: str | None = None
    try:
        gate_self_test = acceptance_gate_self_test(workspace / "_gate_self_test")
    except Exception as exc:  # noqa: BLE001
        gate_error = f"{type(exc).__name__}: {exc}"
        print(f"FAIL acceptance_gate_self_test: {gate_error}")

    combinations: list[dict] = []
    try:
        for family in families:
            for spec in _specs():
                if representation_filter and spec.stats_profile_kind not in representation_filter:
                    continue
                print(f"[dry-run] {combination_id(family, spec)}")
                combinations.append(
                    run_combination(family, spec, workspace=workspace, policy_io=policy_io)
                )
    finally:
        if not args.keep_work:
            shutil.rmtree(workspace, ignore_errors=True)

    expected_total = len(families) * 8
    passed = [entry for entry in combinations if entry["status"] == "dry_run_pass"]
    failed = [entry for entry in combinations if entry["status"] != "dry_run_pass"]
    ran_ids = [entry["combination"] for entry in combinations]
    expected_ids = expected_combination_ids()
    coverage = {
        "expected_ids": list(expected_ids),
        "ran_ids": sorted(ran_ids),
        "duplicate_ids": sorted({name for name in ran_ids if ran_ids.count(name) > 1}),
        "missing_ids": sorted(set(expected_ids) - set(ran_ids)),
        "unexpected_ids": sorted(set(ran_ids) - set(expected_ids)),
    }
    coverage["exact_24_set"] = (
        len(ran_ids) == 24
        and not coverage["duplicate_ids"]
        and not coverage["missing_ids"]
        and not coverage["unexpected_ids"]
    )
    stages_total = sum(len(entry["stages"]) for entry in combinations)
    stages_passed = sum(
        1
        for entry in combinations
        for value in entry["stages"].values()
        if value["status"] == "pass"
    )
    complete_run = (
        coverage["exact_24_set"]
        and not failed
        and not phase17_error
        and not gate_error
        and not representation_filter
    )

    sim_status = "NOT_RUN"
    sim_detail = (
        "학습된 EEF checkpoint와 sim closed-loop 평가가 이 worktree에 없다. "
        "Phase 9 절차(대표 조합 closed-loop)는 미실행."
    )
    sim_evidence = None
    real_status = "BLOCKED_EXTERNAL"
    real_detail = (
        "Windows 실기기 승인/작업자/e-stop gate가 없다. EEF_IK_REAL_VALIDATED는 닫힌 상태."
    )
    real_evidence = None
    report_error: str | None = None
    if args.sim_eval_report is not None:
        try:
            sim_evidence = load_external_report(args.sim_eval_report, expected_mode="sim")
        except Exception as exc:  # noqa: BLE001 - 잘못된 report는 전체 실패다
            report_error = f"sim_eval_report: {type(exc).__name__}: {exc}"
            sim_status = "REPORT_INVALID"
            sim_detail = report_error
            print(f"FAIL {report_error}")
        else:
            sim_status = "REPORT_VERIFIED"
            sim_detail = (
                f"external sim evaluator report verified (status=PASS, failures=0): "
                f"{sim_evidence['path']}"
            )
    if args.real_rollout_report is not None:
        try:
            real_evidence = load_external_report(args.real_rollout_report, expected_mode="real")
        except Exception as exc:  # noqa: BLE001
            message = f"real_rollout_report: {type(exc).__name__}: {exc}"
            report_error = message if report_error is None else f"{report_error}; {message}"
            real_status = "REPORT_INVALID"
            real_detail = message
            print(f"FAIL {message}")
        else:
            real_status = "REPORT_VERIFIED"
            real_detail = (
                f"external real rollout report verified (status=PASS, failures=0): "
                f"{real_evidence['path']}"
            )

    if failed or phase17_error or gate_error or report_error:
        status = "FAIL"
    elif not complete_run:
        status = "DRY_RUN_PARTIAL"
    else:
        status = "DRY_RUN_PASS"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "phase18_complete": False,
        "phase18_completion_note": (
            "이 artifact는 contract-level dry-run만 증명한다. 대표 조합 sim closed-loop"
            "(Phase 9)과 real guarded rollout(Phase 10)이 실행되어야 Phase 18이 완료된다. "
            "이 runner는 **non-promoting** 설계다 — 외부 sim/real report가 둘 다 "
            "REPORT_VERIFIED여도 phase18_complete는 false로 남으며, Phase 18 승격은 별도 "
            "closure 절차(spec 상태표·§26.2 checkbox 갱신)로만 이뤄진다."
        ),
        "metrics_semantics": (
            "operational_metrics의 aborts·invalid_chunks·queue_starvation_ticks·"
            "empty_chunks·stale_chunks가 0인 것은 장시간 rollout 측정 결과가 아니라 정상 "
            "경로에서 **구조적으로** 0이기 때문이다(짧은 결정적 chunk 1개, 실패 주입 없음). "
            "이 runner는 주입 guard와 evaluator self-test로 **fail-closed 동작만** 검증한다. "
            "실제 rate/threshold acceptance는 외부 sim/real evaluator report가 담당한다."
        ),
        "acceptance": {
            "contract_dry_run": {
                "status": status,
                "combinations": len(combinations),
                "expected_combinations": expected_total,
            },
            "sim_closed_loop": {
                "status": sim_status,
                "detail": sim_detail,
                "report": sim_evidence,
            },
            "real_guarded_rollout": {
                "status": real_status,
                "detail": real_detail,
                "report": real_evidence,
            },
        },
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_s": round(time.time() - started, 2),
        # 이번 dry-run 실행 환경의 provenance. `phase17_artifact.git`과 별개다.
        "provenance_scope": "current dry-run execution environment",
        "git": p17._git_identity(),
        "docker_image_id": p17._docker_image_id(),
        "lerobot": p17._lerobot_identity(),
        "seed": args.seed,
        "required_stages": list(STAGE_NAMES),
        "phase17_artifact": phase17 or {"error": phase17_error},
        "acceptance_gate_self_test": gate_self_test or {"status": "fail", "error": gate_error},
        "totals": {
            "expected_combinations": 24,
            "ran_combinations": len(combinations),
            "passed_combinations": len(passed),
            "failed_combinations": len(failed),
            "expected_stages": 24 * len(STAGE_NAMES),
            "ran_stages": stages_total,
            "passed_stages": stages_passed,
            "complete_dry_run": complete_run,
            "exact_24_combination_set": coverage["exact_24_set"],
        },
        "combination_coverage": coverage,
        "failures": [
            {
                "combination": entry["combination"],
                "failed_stages": entry["failed_stages"],
                "missing_stages": entry["missing_stages"],
                "errors": {
                    name: entry["stages"][name].get("error") for name in entry["failed_stages"]
                },
            }
            for entry in failed
        ],
        "combinations": combinations,
    }
    p17._atomic_write_json(args.output, payload)

    print(
        f"\n{len(passed)}/{expected_total} combinations dry-run passed; "
        f"{stages_passed}/{expected_total * len(STAGE_NAMES)} stages passed"
    )
    print(f"status={status} phase18_complete=False sim={sim_status} real={real_status}")
    for entry in failed:
        print(f"  FAILED {entry['combination']}: {entry['failed_stages'] or entry['missing_stages']}")
    if failed or phase17_error or gate_error or report_error:
        if report_error:
            print(f"  FAILED external acceptance report: {report_error}")
        return 1
    if not complete_run:
        print(
            f"  INCOMPLETE: ran {len(combinations)} of 24 combinations "
            f"(missing={coverage['missing_ids']}, duplicates={coverage['duplicate_ids']}, "
            f"unexpected={coverage['unexpected_ids']})"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
