#!/usr/bin/env python
"""LeRobot v3 dataset을 원본 불변 상태로 canonical frame에 변환한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so101_parity.calibration import CalibrationBundle
from so101_parity.contract import CAMERA_ORDER, PolicyIOContract, canonical_json
from so101_parity.manifest import file_sha256
from so101_parity.model_codec import ModelCodec


STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
VECTOR_KEYS = ("action", "observation.state")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--source-frame",
        choices=("real_lerobot_range_v1", "sim_legacy_rad_scale_v1", "canonical"),
        required=True,
    )
    parser.add_argument("--provenance", type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration/so101_canonical.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/parity/policy_io.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--quarantine-reason")
    parser.add_argument("--video-copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _vector_matrix(table: pa.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    if not pa.types.is_fixed_size_list(column.type) or column.type.list_size != 6:
        raise RuntimeError(f"{key}는 fixed_size_list<float>[6]이어야 한다: {column.type}")
    return np.asarray(column.values, dtype=np.float32).reshape(len(column), 6)


def _fixed_vector_array(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 6)


def _stats(values: np.ndarray) -> dict[str, list]:
    matrix = np.asarray(values)
    count = int(matrix.shape[0])
    return {
        "min": np.min(matrix, axis=0).astype(np.float64).tolist(),
        "max": np.max(matrix, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(matrix, axis=0, dtype=np.float64).tolist(),
        "std": np.std(matrix, axis=0, dtype=np.float64).tolist(),
        "count": [count],
        "q01": np.quantile(matrix, 0.01, axis=0).astype(np.float64).tolist(),
        "q10": np.quantile(matrix, 0.10, axis=0).astype(np.float64).tolist(),
        "q50": np.quantile(matrix, 0.50, axis=0).astype(np.float64).tolist(),
        "q90": np.quantile(matrix, 0.90, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).astype(np.float64).tolist(),
    }


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"video open 실패: {path}")
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if count <= 0:
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
        return count
    finally:
        capture.release()


def _source_fingerprint(source: Path, files: list[Path]) -> str:
    rows = []
    for path in sorted(files):
        stat = path.stat()
        row = {
            "path": path.relative_to(source).as_posix(),
            "size": stat.st_size,
        }
        if path.suffix in {".json", ".parquet"}:
            row["sha256"] = file_sha256(path)
        rows.append(row)
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def _audit(source: Path, contract: PolicyIOContract) -> tuple[dict, dict, list[Path]]:
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if info.get("codebase_version") != "v3.0":
        errors.append(f"codebase_version={info.get('codebase_version')!r}")
    if int(info.get("fps", -1)) != contract.fps:
        errors.append(f"fps={info.get('fps')!r}")

    features = info.get("features", {})
    for key in VECTOR_KEYS:
        feature = features.get(key, {})
        if feature.get("shape") != [6] or feature.get("dtype") != "float32":
            errors.append(f"{key} feature={feature!r}")
    for camera in CAMERA_ORDER:
        key = f"observation.images.{camera}"
        feature = features.get(key, {})
        if feature.get("shape") != list(contract.image_shape) or feature.get("dtype") != "video":
            errors.append(f"{key} feature={feature!r}")

    data_files = sorted((source / "data").rglob("*.parquet"))
    episode_files = sorted((source / "meta" / "episodes").rglob("*.parquet"))
    video_files = sorted((source / "videos").rglob("*.mp4"))
    if not data_files or not episode_files or not video_files:
        errors.append("data/meta episodes/video 파일이 모두 있어야 한다")

    data_rows: dict[str, int] = {}
    total_rows = 0
    for path in data_files:
        relative = path.relative_to(source / "data").with_suffix("").as_posix()
        rows = pq.read_metadata(path).num_rows
        data_rows[relative] = rows
        total_rows += rows

    episode_rows = []
    for path in episode_files:
        table = pq.read_table(path, columns=["episode_index", "length"])
        episode_rows.extend(zip(table["episode_index"].to_pylist(), table["length"].to_pylist()))
    episode_length_sum = int(sum(int(length) for _, length in episode_rows))
    if total_rows != int(info.get("total_frames", -1)):
        errors.append(f"data rows={total_rows}, info total_frames={info.get('total_frames')}")
    if episode_length_sum != total_rows:
        errors.append(f"episode length sum={episode_length_sum}, data rows={total_rows}")
    if len(episode_rows) != int(info.get("total_episodes", -1)):
        errors.append(
            f"episode rows={len(episode_rows)}, info total_episodes={info.get('total_episodes')}"
        )

    video_counts: dict[str, int] = {}
    for path in video_files:
        relative = path.relative_to(source / "videos")
        camera_key = relative.parts[0]
        data_key = Path(*relative.parts[1:]).with_suffix("").as_posix()
        count = _video_frame_count(path)
        video_counts[relative.as_posix()] = count
        expected = data_rows.get(data_key)
        if expected is None:
            errors.append(f"{relative}: 대응 data parquet 없음")
        elif count != expected:
            errors.append(f"{relative}: video frames={count}, data rows={expected}")
        if camera_key not in {f"observation.images.{name}" for name in CAMERA_ORDER}:
            errors.append(f"알 수 없는 camera video key: {camera_key}")

    files = [info_path, *data_files, *episode_files, source / "meta" / "tasks.parquet", *video_files]
    audit = {
        "ok": not errors,
        "errors": errors,
        "total_frames": total_rows,
        "total_episodes": len(episode_rows),
        "data_files": len(data_files),
        "video_files": len(video_files),
        "video_frame_counts": video_counts,
        "source_fingerprint": _source_fingerprint(source, [path for path in files if path.exists()]),
    }
    return audit, info, data_files


def _load_provenance(
    path: Path | None,
    source_frame: str,
    calibration: CalibrationBundle,
) -> dict:
    if path is None:
        raise RuntimeError("변환에는 --provenance JSON이 필요하다")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw.get("verified", False):
        raise RuntimeError("dataset provenance가 verified=true가 아니다")
    if raw.get("source_frame") != source_frame:
        raise RuntimeError(
            f"provenance source_frame={raw.get('source_frame')!r}, CLI={source_frame!r}"
        )
    if raw.get("canonical_calibration_hash") != calibration.calibration_hash:
        raise RuntimeError(
            "dataset provenance의 canonical_calibration_hash가 현재 bundle과 다르다"
        )
    snapshot_text = raw.get("source_calibration_snapshot")
    snapshot_hash = raw.get("source_calibration_sha256")
    if snapshot_text:
        snapshot = Path(snapshot_text)
        if not snapshot.is_file():
            raise RuntimeError(f"source calibration snapshot이 없다: {snapshot}")
        if snapshot_hash and file_sha256(snapshot) != snapshot_hash:
            raise RuntimeError("source calibration snapshot hash가 provenance와 다르다")
    return raw


def _copy_video_tree(source: Path, destination: Path, mode: str) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(path, target)
                continue
            except OSError:
                pass
        shutil.copy2(path, target)


def _replace_stats_columns(
    table: pa.Table,
    per_episode: dict[int, dict[str, dict[str, list]]],
) -> pa.Table:
    episode_indices = [int(value) for value in table["episode_index"].to_pylist()]
    result = table
    for key in VECTOR_KEYS:
        for stat_name in STAT_NAMES:
            column_name = f"stats/{key}/{stat_name}"
            if column_name not in result.column_names:
                continue
            column_index = result.schema.get_field_index(column_name)
            column_type = result.schema.field(column_index).type
            values = [per_episode[index][key][stat_name] for index in episode_indices]
            result = result.set_column(
                column_index,
                column_name,
                pa.array(values, type=column_type),
            )
    return result


def _convert(
    source: Path,
    destination: Path,
    info: dict,
    data_files: list[Path],
    codec: ModelCodec,
    contract: PolicyIOContract,
    calibration: CalibrationBundle,
    provenance: dict,
    video_copy_mode: str,
) -> dict:
    if destination.exists():
        raise RuntimeError(f"destination이 이미 존재한다: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    all_values = {key: [] for key in VECTOR_KEYS}
    episode_values: dict[int, dict[str, list[np.ndarray]]] = {}
    try:
        for source_path in data_files:
            table = pq.read_table(source_path)
            episodes = np.asarray(table["episode_index"], dtype=np.int64)
            transformed: dict[str, np.ndarray] = {}
            for key in VECTOR_KEYS:
                native = _vector_matrix(table, key)
                canonical = codec.model_chunk_to_canonical(native)
                transformed[key] = canonical
                all_values[key].append(canonical)
                for episode in np.unique(episodes):
                    episode_values.setdefault(
                        int(episode),
                        {name: [] for name in VECTOR_KEYS},
                    )[key].append(canonical[episodes == episode])
                index = table.schema.get_field_index(key)
                table = table.set_column(index, key, _fixed_vector_array(canonical))
            target = staging / source_path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target, compression="snappy")

        global_stats = {
            key: _stats(np.concatenate(all_values[key], axis=0))
            for key in VECTOR_KEYS
        }
        per_episode = {
            episode: {
                key: _stats(np.concatenate(values[key], axis=0))
                for key in VECTOR_KEYS
            }
            for episode, values in episode_values.items()
        }

        source_meta = source / "meta"
        target_meta = staging / "meta"
        target_meta.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_meta / "tasks.parquet", target_meta / "tasks.parquet")
        for path in sorted((source_meta / "episodes").rglob("*.parquet")):
            table = _replace_stats_columns(pq.read_table(path), per_episode)
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target, compression="snappy")

        stats = json.loads((source_meta / "stats.json").read_text(encoding="utf-8"))
        stats.update(global_stats)
        _write_json(target_meta / "stats.json", stats)

        converted_info = dict(info)
        converted_info["canonical"] = {
            "schema": contract.schema,
            "contract_hash": contract.contract_hash,
            "calibration_hash": calibration.calibration_hash,
            "source_frame": codec.frame,
            "source_provenance": provenance,
        }
        _write_json(target_meta / "info.json", converted_info)
        _copy_video_tree(source / "videos", staging / "videos", video_copy_mode)

        os.replace(staging, destination)
        return {
            "global_stats": global_stats,
            "destination": str(destination),
            "video_copy_mode": video_copy_mode,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    report: dict = {
        "source": str(args.source),
        "destination": str(args.destination) if args.destination else None,
        "source_frame": args.source_frame,
    }
    try:
        contract = PolicyIOContract.load(args.contract)
        calibration = CalibrationBundle.load(args.calibration)
        audit, info, data_files = _audit(args.source, contract)
        report["audit"] = audit
        report["contract_hash"] = contract.contract_hash
        report["calibration_hash"] = calibration.calibration_hash
        if not audit["ok"]:
            raise RuntimeError("dataset audit 실패: " + "; ".join(audit["errors"]))

        if args.audit_only:
            report["status"] = "quarantined" if args.quarantine_reason else "audited"
            report["quarantine_reason"] = args.quarantine_reason
            _write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        if args.destination is None:
            raise RuntimeError("변환에는 --destination이 필요하다")
        provenance = _load_provenance(
            args.provenance,
            args.source_frame,
            calibration,
        )
        if args.source_frame == "real_lerobot_range_v1":
            calibration.require_validated(require_motor_profile=False)
        elif args.source_frame == "sim_legacy_rad_scale_v1" and not calibration.has_sim_gripper_curve:
            raise RuntimeError("sim gripper curve가 없다")
        codec = ModelCodec(args.source_frame, calibration)
        report["conversion"] = _convert(
            args.source,
            args.destination,
            info,
            data_files,
            codec,
            contract,
            calibration,
            provenance,
            args.video_copy_mode,
        )
        converted_audit, _, _ = _audit(args.destination, contract)
        report["converted_audit"] = converted_audit
        if not converted_audit["ok"]:
            raise RuntimeError("변환 후 dataset audit 실패")
        report["status"] = "passed"
        _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
