#!/usr/bin/env python
"""실측 arm/gripper 자료로 canonical calibration bundle을 생성한다."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from so101_parity.calibration import CalibrationBundle, CalibrationError, MonotonePchip
from so101_parity.contract import JOINT_ORDER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("calibration/so101_canonical.json"),
    )
    parser.add_argument(
        "--paired-arm-poses",
        type=Path,
        required=True,
        help="poses[].canonical_rad/real_deg가 각각 길이 5인 JSON",
    )
    parser.add_argument(
        "--real-gripper",
        type=Path,
        required=True,
        help="native,aperture_mm 열을 가진 CSV",
    )
    parser.add_argument(
        "--motor-readback",
        type=Path,
        help="verify_motor_profile 결과 JSON. ok=true일 때만 readback_validated를 설정한다.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/parity/calibration_fit_report.json"),
    )
    parser.add_argument("--arm-rmse-limit-deg", type=float, default=0.5)
    parser.add_argument("--gripper-roundtrip-limit-mm", type=float, default=0.1)
    return parser


def _load_arm_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = raw.get("poses", [])
    if not 10 <= len(poses) <= 20:
        raise CalibrationError(
            f"paired arm pose는 10~20개여야 한다: {len(poses)}개"
        )
    canonical = np.asarray([row["canonical_rad"] for row in poses], dtype=np.float64)
    real = np.asarray([row["real_deg"] for row in poses], dtype=np.float64)
    if canonical.shape != (len(poses), 5) or real.shape != canonical.shape:
        raise CalibrationError("paired pose의 canonical_rad/real_deg는 각각 길이 5여야 한다")
    if not np.all(np.isfinite(canonical)) or not np.all(np.isfinite(real)):
        raise CalibrationError("paired pose에 NaN/Inf가 있다")
    return canonical, real


def _load_gripper(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 7 <= len(rows) <= 9:
        raise CalibrationError(f"real gripper 측정점은 7~9개여야 한다: {len(rows)}개")
    native = np.asarray([float(row["native"]) for row in rows], dtype=np.float64)
    aperture = np.asarray([float(row["aperture_mm"]) for row in rows], dtype=np.float64)
    order = np.argsort(native)
    native = native[order]
    aperture = aperture[order]
    if np.any(np.diff(native) <= 0):
        raise CalibrationError("real gripper native 값은 중복 없이 증가해야 한다")
    MonotonePchip(native, aperture)
    return native, aperture


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    canonical, real_deg = _load_arm_poses(args.paired_arm_poses)
    real_native, aperture_mm = _load_gripper(args.real_gripper)

    fitted = CalibrationBundle.fit_arm_affine(canonical, real_deg)
    arm_rmse_deg = {}
    for name in JOINT_ORDER[:5]:
        row = dict(base["arm"][name])
        row.update(fitted[name])
        row["fit_pose_range_deg"] = [
            float(np.min(real_deg[:, JOINT_ORDER.index(name)])),
            float(np.max(real_deg[:, JOINT_ORDER.index(name)])),
        ]
        base["arm"][name] = row
        arm_rmse_deg[name] = float(np.rad2deg(fitted[name]["fit_rmse_rad"]))

    real_curve = MonotonePchip(real_native, aperture_mm)
    aperture_probe = np.linspace(float(aperture_mm.min()), float(aperture_mm.max()), 1001)
    gripper_roundtrip_mm = float(
        np.max(
            np.abs(
                real_curve(real_curve.inverse()(aperture_probe))
                - aperture_probe
            )
        )
    )
    base["gripper"]["real"] = {
        "native_unit": "lerobot_range_0_100",
        "native": real_native.tolist(),
        "aperture_mm": aperture_mm.tolist(),
        "max_interpolation_error_mm": gripper_roundtrip_mm,
        "source_report": str(args.real_gripper),
    }

    motor_readback_ok = False
    if args.motor_readback:
        readback = json.loads(args.motor_readback.read_text(encoding="utf-8"))
        profile_result = readback.get("profile", readback)
        motor_readback_ok = bool(profile_result.get("ok", False))
        if not motor_readback_ok:
            raise CalibrationError("motor readback report가 ok=true가 아니다")
        expected_hash = CalibrationBundle.with_hash(base)["motor_profile"]["profile_hash"]
        supplied_hash = readback.get("motor_profile_hash")
        if supplied_hash and supplied_hash != expected_hash:
            raise CalibrationError(
                "motor readback의 profile hash가 calibration bundle과 다르다"
            )
        base["motor_profile"]["readback_validated"] = True
        base["motor_profile"]["readback_report"] = str(args.motor_readback)

    arm_ok = max(arm_rmse_deg.values()) <= args.arm_rmse_limit_deg
    gripper_ok = gripper_roundtrip_mm < args.gripper_roundtrip_limit_mm
    sim_ok = bool(base.get("gripper", {}).get("sim", {}).get("native"))
    base["validated"] = bool(arm_ok and gripper_ok and sim_ok and motor_readback_ok)
    base["calibration_id"] = "so101-follower-canonical-measured"
    base["source_measurements"] = {
        "paired_arm_poses": str(args.paired_arm_poses),
        "real_gripper": str(args.real_gripper),
        "motor_readback": str(args.motor_readback) if args.motor_readback else None,
    }
    output = CalibrationBundle.with_hash(base)
    bundle = CalibrationBundle(output)

    arm_roundtrip_max_rad = 0.0
    for canonical_row, real_row in zip(canonical, real_deg, strict=True):
        native = np.concatenate((real_row, [float(real_native[len(real_native) // 2])]))
        converted = bundle.real_to_canonical(native)
        restored = bundle.canonical_to_real(converted)
        arm_roundtrip_max_rad = max(
            arm_roundtrip_max_rad,
            float(np.max(np.abs(np.deg2rad(restored[:5] - real_row)))),
        )

    report = {
        "status": "passed" if output["validated"] else "blocked",
        "validated": output["validated"],
        "pose_count": int(len(canonical)),
        "gripper_anchor_count": int(len(real_native)),
        "arm_rmse_deg": arm_rmse_deg,
        "arm_roundtrip_max_rad": arm_roundtrip_max_rad,
        "gripper_roundtrip_max_mm": gripper_roundtrip_mm,
        "motor_readback_validated": motor_readback_ok,
        "calibration_hash": bundle.calibration_hash,
        "motor_profile_hash": bundle.motor_profile_hash,
        "gates": {
            "arm_fit": arm_ok,
            "arm_roundtrip_lt_1e-5_rad": arm_roundtrip_max_rad < 1e-5,
            "gripper_roundtrip_lt_0_1_mm": gripper_ok,
            "sim_gripper_curve": sim_ok,
            "motor_readback": motor_readback_ok,
        },
    }
    _write_json(args.output, output)
    _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if output["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
