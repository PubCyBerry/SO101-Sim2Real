from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from so101_parity.calibration import CalibrationBundle, MonotonePchip
from so101_parity.contract import PolicyIOContract, default_contract_dict
from so101_parity.dynamics import build_excitation_plan, identify_joint_response
from so101_parity.executor import (
    Chunk,
    MotionLimiter,
    SingleFlightChunkExecutor,
    prefetch_lead_from_p99,
)
from so101_parity.model_codec import ModelCodec
from so101_parity.lease import LeaseError, MotionLease
from so101_parity.motor_profile import verify_motor_profile
from so101_parity.runtime import (
    CanonicalObservation,
    CanonicalRuntime,
    RuntimeHashes,
)
from so101_parity.trace import JsonlTraceWriter, load_trace


class ContractTest(unittest.TestCase):
    def test_default_contract(self) -> None:
        contract = PolicyIOContract.from_dict(default_contract_dict())
        self.assertEqual(contract.schema, "so101-canonical-v1")
        self.assertEqual(contract.fps, 30)
        self.assertEqual(contract.arm_unit, "urdf_radian")


class CalibrationTest(unittest.TestCase):
    @staticmethod
    def _bundle() -> CalibrationBundle:
        raw = {
            "schema": "so101-canonical-calibration-v1",
            "policy_io_schema": "so101-canonical-v1",
            "calibration_id": "test",
            "validated": True,
            "joint_order": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
            "arm": {
                name: {
                    "sign": -1.0 if index % 2 else 1.0,
                    "offset_rad": index * 0.01,
                    "real_deg_range": [-180, 180],
                }
                for index, name in enumerate(
                    [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                    ]
                )
            },
            "gripper": {
                "real": {"native": [0, 50, 100], "aperture_mm": [4, 24, 46]},
                "sim": {"native": [-0.2, 0.7, 1.6], "aperture_mm": [4, 24, 46]},
            },
            "motor_profile": {
                "expected": {
                    "operating_mode": 3,
                    "p_coefficient": 16,
                    "i_coefficient": 0,
                    "d_coefficient": 32,
                },
                "readback_validated": True,
            },
        }
        return CalibrationBundle(CalibrationBundle.with_hash(raw))

    def test_pchip_round_trip(self) -> None:
        curve = MonotonePchip([0, 20, 50, 100], [5, 12, 25, 45])
        values = np.linspace(0, 100, 101)
        restored = curve.inverse()(curve(values))
        self.assertLess(float(np.max(np.abs(values - restored))), 1e-9)

    def test_arm_and_gripper_round_trip(self) -> None:
        bundle = self._bundle()
        native = np.array([10, -20, 30, -40, 50, 62], dtype=np.float32)
        restored = bundle.canonical_to_real(bundle.real_to_canonical(native))
        self.assertLess(float(np.max(np.abs(native - restored))), 1e-4)

    def test_arm_affine_fit(self) -> None:
        real_deg = np.array(
            [
                [-20, -10, 0, 10, 20],
                [-10, 0, 10, 20, 30],
                [0, 10, 20, 30, 40],
                [10, 20, 30, 40, 50],
            ],
            dtype=np.float64,
        )
        signs = np.array([1, -1, 1, -1, 1], dtype=np.float64)
        offsets = np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=np.float64)
        canonical = np.deg2rad(real_deg) * signs + offsets
        fitted = CalibrationBundle.fit_arm_affine(canonical, real_deg)
        for index, name in enumerate(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        ):
            self.assertEqual(fitted[name]["sign"], signs[index])
            self.assertAlmostEqual(fitted[name]["offset_rad"], offsets[index], places=12)
            self.assertLess(fitted[name]["fit_rmse_rad"], 1e-12)

    def test_model_codec_round_trip(self) -> None:
        bundle = self._bundle()
        canonical = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 25.0], dtype=np.float32)
        for frame in ("canonical", "sim_legacy_rad_scale_v1", "real_lerobot_range_v1"):
            codec = ModelCodec(frame, bundle)
            restored = codec.model_to_canonical(codec.canonical_to_model(canonical))
            self.assertLess(float(np.max(np.abs(canonical - restored))), 1e-5)

    def test_calibration_bundle_fitter_cli(self) -> None:
        base_path = Path("calibration/so101_canonical.json")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        real_deg = np.stack(
            [
                np.linspace(-30 + index * 2, 30 + index * 2, 10)
                for index in range(5)
            ],
            axis=1,
        )
        signs = np.array([1, -1, 1, -1, 1], dtype=np.float64)
        offsets = np.array([0.01, -0.02, 0.03, -0.04, 0.05], dtype=np.float64)
        canonical = np.deg2rad(real_deg) * signs + offsets
        poses = {
            "schema": "so101-paired-arm-poses-v1",
            "poses": [
                {
                    "canonical_rad": canonical[index].tolist(),
                    "real_deg": real_deg[index].tolist(),
                }
                for index in range(10)
            ],
        }
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            pose_path = directory / "poses.json"
            gripper_path = directory / "gripper.csv"
            readback_path = directory / "readback.json"
            output_path = directory / "calibration.json"
            report_path = directory / "report.json"
            pose_path.write_text(json.dumps(poses), encoding="utf-8")
            gripper_path.write_text(
                "native,aperture_mm\n"
                "0,2\n15,15\n30,29\n45,43\n60,57\n80,76\n100,96\n",
                encoding="utf-8",
            )
            readback_path.write_text(
                json.dumps(
                    {
                        "profile": {"ok": True},
                        "motor_profile_hash": base["motor_profile"]["profile_hash"],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/parity/fit_calibration_bundle.py",
                    "--base",
                    str(base_path),
                    "--paired-arm-poses",
                    str(pose_path),
                    "--real-gripper",
                    str(gripper_path),
                    "--motor-readback",
                    str(readback_path),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["validated"])
            self.assertLess(report["arm_roundtrip_max_rad"], 1e-5)
            self.assertLess(report["gripper_roundtrip_max_mm"], 0.1)


class ExecutorTest(unittest.TestCase):
    def test_exact_chunk_boundary(self) -> None:
        executor = SingleFlightChunkExecutor(prefetch_lead=2)
        first = np.arange(24, dtype=np.float32).reshape(4, 6)
        executor.seed(Chunk(0, 0, first))
        self.assertFalse(executor.should_request())
        executor.tick()
        executor.tick()
        self.assertTrue(executor.should_request())
        ticket = executor.begin_request()
        second = np.arange(24, 48, dtype=np.float32).reshape(4, 6)
        executor.accept(Chunk(ticket.request_id, ticket.requested_start_step, second))
        actual = [executor.tick().target, executor.tick().target, executor.tick().target]
        self.assertTrue(np.array_equal(actual[0], first[2]))
        self.assertTrue(np.array_equal(actual[1], first[3]))
        self.assertTrue(np.array_equal(actual[2], second[0]))
        self.assertEqual(executor.underruns, 0)

    def test_underrun_holds_without_advancing_policy_step(self) -> None:
        executor = SingleFlightChunkExecutor(prefetch_lead=1)
        first = np.arange(12, dtype=np.float32).reshape(2, 6)
        executor.seed(Chunk(0, 0, first))
        executor.tick()
        executor.tick()
        held_1 = executor.tick()
        held_2 = executor.tick()
        self.assertTrue(held_1.underrun)
        self.assertTrue(held_2.underrun)
        self.assertEqual(held_1.policy_step, 2)
        self.assertEqual(held_2.policy_step, 2)
        self.assertEqual(executor.step, 2)
        self.assertTrue(np.array_equal(held_1.target, first[1]))

        ticket = executor.begin_request()
        second = np.arange(12, 24, dtype=np.float32).reshape(2, 6)
        executor.accept(Chunk(ticket.request_id, ticket.requested_start_step, second))
        resumed = executor.tick()
        self.assertFalse(resumed.underrun)
        self.assertEqual(resumed.policy_step, 2)
        self.assertTrue(np.array_equal(resumed.target, second[0]))

    def test_prefetch_formula(self) -> None:
        self.assertEqual(prefetch_lead_from_p99(0.0), 8)
        self.assertEqual(prefetch_lead_from_p99(0.25), 10)

    def test_request_timeout(self) -> None:
        executor = SingleFlightChunkExecutor(request_timeout_ms=100)
        ticket = executor.begin_request(now_ns=1_000_000_000)
        self.assertFalse(executor.request_timed_out(now_ns=1_099_999_999))
        self.assertTrue(executor.request_timed_out(now_ns=1_100_000_000))
        executor.fail_request(ticket.request_id, timeout=True)
        self.assertEqual(executor.timeouts, 1)

    def test_motion_limiter_is_deterministic(self) -> None:
        kwargs = dict(
            fps=30,
            max_velocity=np.ones(6),
            max_acceleration=np.ones(6) * 2,
            max_jerk=np.ones(6) * 10,
        )
        first = MotionLimiter(**kwargs)
        second = MotionLimiter(**kwargs)
        first.reset(np.zeros(6))
        second.reset(np.zeros(6))
        for _ in range(20):
            self.assertTrue(np.array_equal(first.apply(np.ones(6)), second.apply(np.ones(6))))

    def test_motion_limiter_converges_without_limit_cycle(self) -> None:
        limiter = MotionLimiter(
            fps=30,
            max_velocity=np.array([2, 2, 2, 2.5, 3, 80], dtype=np.float64),
            max_acceleration=np.array([4, 4, 4, 5, 6, 160], dtype=np.float64),
            max_jerk=np.array([20, 20, 20, 25, 30, 800], dtype=np.float64),
        )
        target = np.array([0, -1.3, 1.2, -0.349, -1.571, 40], dtype=np.float64)
        limiter.reset(np.zeros(6))
        positions = [limiter.apply(target) for _ in range(300)]
        self.assertLess(float(np.max(np.abs(positions[-1] - target))), 0.002)


class LeaseTest(unittest.TestCase):
    def test_single_active_client_and_expiry(self) -> None:
        lease = MotionLease(duration_ms=100)
        first = lease.acquire("sim", now_ns=1_000_000_000)
        renewed = lease.validate_and_renew("sim", first.token, now_ns=1_050_000_000)
        self.assertGreater(renewed.expires_at_ns, first.expires_at_ns)
        with self.assertRaises(LeaseError):
            lease.acquire("real", now_ns=1_060_000_000)
        with self.assertRaises(LeaseError):
            lease.release("sim", "wrong-token", now_ns=1_060_000_000)
        lease.release("sim", renewed.token, now_ns=1_060_000_000)
        self.assertIsNone(lease.snapshot(now_ns=1_060_000_001))
        real = lease.acquire("real", now_ns=1_070_000_000)
        self.assertEqual(real.client_id, "real")
        lease.release("real", real.token, now_ns=1_080_000_000)
        real = lease.acquire("real", now_ns=1_200_000_000)
        self.assertEqual(real.client_id, "real")


class RuntimeTest(unittest.TestCase):
    class _Adapter:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            self.state = np.zeros(6, dtype=np.float32)
            self.commands: list[np.ndarray] = []

        def capture(self) -> CanonicalObservation:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            return CanonicalObservation(
                self.state.copy(),
                {"top": image, "wrist": image, "front": image},
            )

        def canonical_to_native(self, target: np.ndarray) -> np.ndarray:
            return np.asarray(target, dtype=np.float32)

        def advance(self, native_command: np.ndarray) -> None:
            self.state = np.asarray(native_command, dtype=np.float32).copy()
            self.commands.append(self.state.copy())

        def safe_stop(self, reason: str) -> None:
            raise AssertionError(reason)

    def test_sim_and_real_use_bitwise_identical_targets(self) -> None:
        def infer(ticket, _observation):
            row = np.full(6, ticket.request_id * 0.1, dtype=np.float32)
            return Chunk(
                ticket.request_id,
                ticket.requested_start_step,
                np.repeat(row[None, :], 16, axis=0),
                inference_latency_ms=10.0,
                checkpoint_hash="checkpoint",
            )

        hashes = RuntimeHashes("contract", "runtime", "checkpoint", "calibration", "motor")
        limiter_args = dict(
            fps=30,
            max_velocity=np.ones(6) * 10,
            max_acceleration=np.ones(6) * 100,
            max_jerk=np.ones(6) * 1000,
        )
        with tempfile.TemporaryDirectory() as directory:
            traces = []
            adapters = []
            for domain in ("sim", "real"):
                adapter = self._Adapter(domain)
                path = Path(directory) / f"{domain}.jsonl"
                with JsonlTraceWriter(path) as trace:
                    runtime = CanonicalRuntime(
                        adapter=adapter,
                        infer=infer,
                        limiter=MotionLimiter(**limiter_args),
                        hashes=hashes,
                        trace=trace,
                        prefetch_lead=8,
                        request_timeout_ms=1000,
                    )
                    runtime.limiter.reset(np.zeros(6, dtype=np.float32))
                    runtime.run(6)
                traces.append(load_trace(path))
                adapters.append(adapter)
            self.assertEqual(len(traces[0]), 6)
            self.assertEqual(len(traces[1]), 6)
            for left, right in zip(adapters[0].commands, adapters[1].commands, strict=True):
                self.assertEqual(left.tobytes(), right.tobytes())


class MotorProfileTest(unittest.TestCase):
    class _Calibration:
        def __init__(self, identifier: int) -> None:
            self.id = identifier
            self.drive_mode = 0
            self.homing_offset = identifier * 10
            self.range_min = 100
            self.range_max = 4000

    class _Bus:
        def __init__(self) -> None:
            self.calibration = {
                f"joint_{index}": MotorProfileTest._Calibration(index)
                for index in range(1, 3)
            }

        def sync_read(self, register, normalize=False):
            self.normalize = normalize
            values = {
                "Operating_Mode": 3,
                "P_Coefficient": 16,
                "I_Coefficient": 0,
                "D_Coefficient": 32,
                "Return_Delay_Time": 0,
                "Acceleration": 254,
                "Torque_Enable": 0,
            }
            if register in values:
                return {name: values[register] for name in self.calibration}
            key = {
                "Homing_Offset": "homing_offset",
                "Min_Position_Limit": "range_min",
                "Max_Position_Limit": "range_max",
            }[register]
            return {
                name: getattr(item, key)
                for name, item in self.calibration.items()
            }

        def read(self, register, motor, normalize=False):
            del motor, normalize
            return {
                "Max_Torque_Limit": 500,
                "Protection_Current": 250,
                "Overload_Torque": 25,
            }[register]

    def test_readback_matches_without_writes(self) -> None:
        bus = self._Bus()
        expected = {
            "operating_mode": 3,
            "required_preflight_torque_enable": 0,
            "p_coefficient": 16,
            "i_coefficient": 0,
            "d_coefficient": 32,
            "return_delay_time": 0,
            "acceleration": 254,
            "gripper": {
                "max_torque_limit": 500,
                "protection_current": 250,
                "overload_torque": 25,
            },
            "joints": {
                name: {
                    "id": item.id,
                    "drive_mode": item.drive_mode,
                    "homing_offset": item.homing_offset,
                    "range_min": item.range_min,
                    "range_max": item.range_max,
                }
                for name, item in bus.calibration.items()
            },
        }
        self.assertTrue(verify_motor_profile(bus, expected)["ok"])


class DynamicsTest(unittest.TestCase):
    def test_plan_contains_required_excitations(self) -> None:
        plan = build_excitation_plan(
            np.array([0.0, -1.3, 1.2, -0.349, -1.571, 40.0]),
            condition="no_load",
        )
        phases = {row.phase for row in plan}
        self.assertIn("initial_hold", phases)
        self.assertTrue(any(name.startswith("step_shoulder_pan_") for name in phases))
        self.assertTrue(any(name.startswith("ramp_") for name in phases))
        self.assertTrue(any(name.startswith("triangle_") for name in phases))
        self.assertIn("multisine_0p2_to_1p5_hz", phases)
        self.assertIn("compound_6axis", phases)
        self.assertTrue(any(name.startswith("gripper_sweep_") for name in phases))

    def test_joint_response_identification(self) -> None:
        target = np.repeat(np.random.default_rng(3).normal(size=60), 10)
        measured = np.zeros(len(target))
        for index in range(3, len(target)):
            measured[index] = 0.85 * measured[index - 1] + 0.15 * target[index - 2]
        result = identify_joint_response(target, measured)
        self.assertLessEqual(abs(int(result["delay_steps"]) - 2), 1)
        self.assertTrue(result["stable"])
        self.assertLess(float(result["residual_rmse"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
