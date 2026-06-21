from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so101_parity.calibration import CalibrationBundle
from so101_parity.contract import JOINT_ORDER


def _write_video(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (640, 480),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV smoke video writer를 열지 못했다")
    try:
        for index in range(count):
            frame = np.full((480, 640, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class DatasetConverterTest(unittest.TestCase):
    def test_tiny_dataset_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            source = root / "source"
            destination = root / "canonical"
            (source / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (source / "data" / "chunk-000").mkdir(parents=True)

            count = 5
            native = np.asarray(
                [
                    [-10, -20, 30, 40, -50, 0],
                    [-5, -10, 20, 30, -40, 20],
                    [0, 0, 10, 20, -30, 40],
                    [5, 10, 0, 10, -20, 60],
                    [10, 20, -10, 0, -10, 80],
                ],
                dtype=np.float32,
            )
            vector_type = pa.list_(pa.float32(), 6)
            table = pa.table(
                {
                    "action": pa.array(native.tolist(), type=vector_type),
                    "observation.state": pa.array(native.tolist(), type=vector_type),
                    "timestamp": pa.array(np.arange(count) / 30.0, type=pa.float32()),
                    "frame_index": pa.array(np.arange(count), type=pa.int64()),
                    "episode_index": pa.array(np.zeros(count), type=pa.int64()),
                    "index": pa.array(np.arange(count), type=pa.int64()),
                    "task_index": pa.array(np.zeros(count), type=pa.int64()),
                }
            )
            pq.write_table(table, source / "data" / "chunk-000" / "file-000.parquet")
            pq.write_table(
                pa.table(
                    {
                        "episode_index": pa.array([0], type=pa.int64()),
                        "length": pa.array([count], type=pa.int64()),
                    }
                ),
                source / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            )
            pq.write_table(
                pa.table({"task_index": [0], "task": ["smoke"]}),
                source / "meta" / "tasks.parquet",
            )
            features = {
                key: {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [f"{name}.pos" for name in JOINT_ORDER],
                }
                for key in ("action", "observation.state")
            }
            for camera in ("top", "wrist", "front"):
                features[f"observation.images.{camera}"] = {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                }
                _write_video(
                    source
                    / "videos"
                    / f"observation.images.{camera}"
                    / "chunk-000"
                    / "file-000.mp4",
                    count,
                )
            (source / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v3.0",
                        "robot_type": "so_follower",
                        "fps": 30,
                        "total_episodes": 1,
                        "total_frames": count,
                        "features": features,
                    }
                ),
                encoding="utf-8",
            )
            (source / "meta" / "stats.json").write_text(
                json.dumps({"action": {}, "observation.state": {}}),
                encoding="utf-8",
            )

            calibration_raw = {
                "schema": "so101-canonical-calibration-v1",
                "policy_io_schema": "so101-canonical-v1",
                "calibration_id": "dataset-smoke",
                "validated": True,
                "joint_order": list(JOINT_ORDER),
                "arm": {
                    name: {
                        "sign": 1.0,
                        "offset_rad": 0.0,
                        "real_deg_range": [-180.0, 180.0],
                    }
                    for name in JOINT_ORDER[:5]
                },
                "gripper": {
                    "real": {
                        "native": [0.0, 50.0, 100.0],
                        "aperture_mm": [2.0, 50.0, 100.0],
                    },
                    "sim": {
                        "native": [-0.2, 0.7, 1.6],
                        "aperture_mm": [2.0, 50.0, 100.0],
                    },
                },
                "motor_profile": {
                    "expected": {"p_coefficient": 16},
                    "readback_validated": True,
                },
            }
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(CalibrationBundle.with_hash(calibration_raw)),
                encoding="utf-8",
            )
            provenance_path = root / "provenance.json"
            calibration_bundle = CalibrationBundle.load(calibration_path)
            provenance_path.write_text(
                json.dumps(
                    {
                        "schema": "so101-dataset-provenance-v1",
                        "verified": True,
                        "source_frame": "real_lerobot_range_v1",
                        "canonical_calibration_hash": calibration_bundle.calibration_hash,
                    }
                ),
                encoding="utf-8",
            )
            report_path = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/parity/convert_dataset_canonical.py",
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                    "--source-frame",
                    "real_lerobot_range_v1",
                    "--provenance",
                    str(provenance_path),
                    "--calibration",
                    str(calibration_path),
                    "--report",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            converted = pq.read_table(
                destination / "data" / "chunk-000" / "file-000.parquet"
            )
            action = np.asarray(
                converted["action"].combine_chunks().values,
                dtype=np.float32,
            ).reshape(count, 6)
            np.testing.assert_allclose(action[:, :5], np.deg2rad(native[:, :5]), atol=1e-6)
            self.assertTrue(np.all(np.diff(action[:, 5]) > 0))


if __name__ == "__main__":
    unittest.main()
