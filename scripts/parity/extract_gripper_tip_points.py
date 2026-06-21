#!/usr/bin/env python
"""SO-101 USD에서 jaw/gripper distal tip point cloud를 별도 프로세스로 추출한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_USD_PATH = _REPO_ROOT / "assets" / "robots" / "so101_follower.usd"


def mesh_points(stage, cache, path: str) -> np.ndarray:
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(
        cache.GetLocalToWorldTransform(mesh.GetPrim()),
        dtype=np.float64,
    ).reshape(4, 4)
    return np.hstack((points, np.ones((len(points), 1)))) @ transform[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-usd", type=Path, default=_ROBOT_USD_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/parity/gripper_tip_points.npz"),
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.robot_usd))
    cache = UsdGeom.XformCache()
    jaw = mesh_points(stage, cache, "/visuals/jaw/moving_jaw_so101_v1/mesh")
    gripper = np.concatenate(
        (
            mesh_points(stage, cache, "/visuals/gripper/sts3215_03a_v1/mesh"),
            mesh_points(stage, cache, "/visuals/gripper/wrist_roll_follower_so101_v1/mesh"),
        ),
        axis=0,
    )
    jaw_tip = jaw[jaw[:, 1] <= -0.060]
    gripper_tip = gripper[gripper[:, 2] <= -0.075]
    # USD mesh는 face-vertex 단위 중복점이 많다. 1 µm grid로 deterministic dedupe한다.
    jaw_tip = np.unique(np.round(jaw_tip, 6), axis=0)
    gripper_tip = np.unique(np.round(gripper_tip, 6), axis=0)
    if len(jaw_tip) < 4 or len(gripper_tip) < 4:
        raise RuntimeError(
            f"tip point 추출 실패: jaw={len(jaw_tip)}, gripper={len(gripper_tip)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        jaw_tip=jaw_tip.astype(np.float32),
        gripper_tip=gripper_tip.astype(np.float32),
    )
    print(
        f"{args.output}: jaw={len(jaw_tip)}, gripper={len(gripper_tip)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
