#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LeRobot v3 데이터셋 스키마 검증 스크립트.

Usage:
    python scripts/validate_lerobot_schema.py <dataset_root>
    python scripts/validate_lerobot_schema.py --self-test
"""

import argparse
import glob
import io
import json
import os
import sys

# Windows cp949 터미널에서도 한국어 출력 가능하도록 UTF-8 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------- 불변 상수 ----------

EXPECTED_CODEBASE_VERSION = "v3.0"
EXPECTED_ROBOT_TYPE = "so_follower"
EXPECTED_FPS = 30
EXPECTED_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
REQUIRED_CAMERAS = ["top", "wrist"]
OPTIONAL_CAMERAS = ["front"]
EXPECTED_TASK = "pick up the cube and place it in the bowl"

EXPECTED_CAMERA_INFO = {
    "video.codec": "h264",
    "video.fps": 30,
    "video.channels": 3,
    "video.height": 480,
    "video.width": 640,
}

REQUIRED_DATA_COLS = {
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


# ---------- 검증 함수 ----------


def validate_info(dataset_root: str, errors: list[str], warnings: list[str] | None = None) -> None:
    """meta/info.json 검증."""
    path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.exists(path):
        errors.append(f"파일 없음: {path}")
        return

    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"info.json 읽기 실패: {exc}")
        return

    def chk(key, expected):
        val = info.get(key)
        if val != expected:
            errors.append(f"info.json [{key}] = {val!r}, 기대: {expected!r}")

    chk("codebase_version", EXPECTED_CODEBASE_VERSION)
    chk("robot_type", EXPECTED_ROBOT_TYPE)
    chk("fps", EXPECTED_FPS)

    features = info.get("features", {})

    # action / observation.state 검증
    for key in ("action", "observation.state"):
        feat = features.get(key)
        if feat is None:
            errors.append(f"info.json features에 {key!r} 없음")
            continue
        if feat.get("dtype") != "float32":
            errors.append(f"info.json features[{key}].dtype = {feat.get('dtype')!r}, 기대: 'float32'")
        if feat.get("shape") != [6]:
            errors.append(f"info.json features[{key}].shape = {feat.get('shape')!r}, 기대: [6]")
        if feat.get("names") != EXPECTED_JOINT_NAMES:
            errors.append(f"info.json features[{key}].names 불일치: {feat.get('names')!r}")

    # 카메라 검증 — REQUIRED_CAMERAS(에러), OPTIONAL_CAMERAS(경고)
    def _check_camera(cam: str) -> list[str]:
        issues: list[str] = []
        key = f"observation.images.{cam}"
        feat = features.get(key)
        if feat is None:
            issues.append(f"info.json features에 {key!r} 없음")
            return issues
        if feat.get("dtype") != "video":
            issues.append(f"info.json features[{key}].dtype = {feat.get('dtype')!r}, 기대: 'video'")
        if feat.get("shape") != [480, 640, 3]:
            issues.append(f"info.json features[{key}].shape = {feat.get('shape')!r}, 기대: [480,640,3]")
        info_block = feat.get("info", {})
        for k, v in EXPECTED_CAMERA_INFO.items():
            if info_block.get(k) != v:
                issues.append(
                    f"info.json features[{key}].info[{k!r}] = {info_block.get(k)!r}, 기대: {v!r}"
                )
        return issues

    for cam in REQUIRED_CAMERAS:
        errors.extend(_check_camera(cam))
    for cam in OPTIONAL_CAMERAS:
        issues = _check_camera(cam)
        if issues and warnings is not None:
            warnings.extend(f"[선택 카메라 {cam}] {msg}" for msg in issues)


def validate_tasks(dataset_root: str, errors: list[str], expected_task: str = EXPECTED_TASK) -> None:
    """meta/tasks.parquet 검증 (pyarrow 전용)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        errors.append("pyarrow 미설치 — pip install pyarrow")
        return

    path = os.path.join(dataset_root, "meta", "tasks.parquet")
    if not os.path.exists(path):
        errors.append(f"파일 없음: {path}")
        return

    try:
        table = pq.read_table(path)
    except Exception as exc:  # pyarrow가 파일 손상/스키마 오류마다 다른 예외를 낸다.
        errors.append(f"tasks.parquet 읽기 실패: {exc}")
        return
    cols = table.column_names

    missing_columns = False
    if "task_index" not in cols:
        errors.append("tasks.parquet에 task_index 컬럼 없음")
        missing_columns = True
    if "__index_level_0__" not in cols:
        errors.append("tasks.parquet에 __index_level_0__ 컬럼 없음 (태스크 텍스트)")
        missing_columns = True

    if missing_columns:
        return  # 컬럼 없으면 이후 접근 불가

    task_indices = table.column("task_index").to_pylist()
    task_texts = table.column("__index_level_0__").to_pylist()

    if len(task_indices) != 1:
        errors.append(f"tasks.parquet 행 수 = {len(task_indices)}, 기대: 1")
    if task_indices and task_indices[0] != 0:
        errors.append(f"tasks.parquet task_index = {task_indices[0]}, 기대: 0")
    if task_texts and task_texts[0] != expected_task:
        errors.append(f"tasks.parquet 태스크 문자열 = {task_texts[0]!r}, 기대: {expected_task!r}")


def validate_data_parquet(dataset_root: str, errors: list[str]) -> None:
    """data/**/file-*.parquet 스키마 검증 (첫 번째 파일만)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return  # validate_tasks에서 이미 오류 추가됨

    pattern = os.path.join(dataset_root, "data", "**", "file-*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        errors.append(f"data parquet 파일 없음: {pattern}")
        return

    try:
        schema = pq.read_schema(files[0])
    except Exception as exc:  # pyarrow가 파일 손상/스키마 오류마다 다른 예외를 낸다.
        errors.append(f"data parquet 스키마 읽기 실패({files[0]}): {exc}")
        return
    col_names = {field.name for field in schema}

    missing = REQUIRED_DATA_COLS - col_names
    if missing:
        errors.append(f"data parquet 컬럼 누락: {sorted(missing)}")

    import pyarrow as pa

    # action / observation.state 타입 검증
    for col in ("action", "observation.state"):
        if col not in col_names:
            continue
        field = schema.field(col)
        t = field.type
        if not (
            pa.types.is_fixed_size_list(t)
            and t.list_size == 6
            and pa.types.is_float32(t.value_type)
        ):
            errors.append(
                f"data parquet [{col}] 타입 = {t}, 기대: fixed_size_list<float32>[6]"
            )


# ---------- self-test ----------


def run_self_test() -> None:
    """임시 fixture를 만들어 검증기 동작 확인."""
    import tempfile

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("SKIP: pyarrow 미설치", file=sys.stderr)
        sys.exit(1)

    def make_valid_dataset(root: str) -> None:
        """최소 유효 데이터셋 픽스처 생성."""
        os.makedirs(os.path.join(root, "meta"), exist_ok=True)
        os.makedirs(os.path.join(root, "data", "chunk-000"), exist_ok=True)

        info = {
            "codebase_version": "v3.0",
            "robot_type": "so_follower",
            "fps": 30,
            "features": {
                "action": {"dtype": "float32", "shape": [6], "names": EXPECTED_JOINT_NAMES},
                "observation.state": {"dtype": "float32", "shape": [6], "names": EXPECTED_JOINT_NAMES},
                **{
                    f"observation.images.{cam}": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.codec": "h264",
                            "video.fps": 30,
                            "video.channels": 3,
                            "video.height": 480,
                            "video.width": 640,
                        },
                    }
                    for cam in REQUIRED_CAMERAS + OPTIONAL_CAMERAS
                },
            },
        }
        with open(os.path.join(root, "meta", "info.json"), "w") as f:
            json.dump(info, f)

        # tasks.parquet
        tasks_table = pa.table(
            {"task_index": [0], "__index_level_0__": [EXPECTED_TASK]}
        )
        pq.write_table(tasks_table, os.path.join(root, "meta", "tasks.parquet"))

        # data parquet
        fsl_type = pa.list_(pa.float32(), 6)
        data_schema = pa.schema([
            pa.field("action", fsl_type),
            pa.field("observation.state", fsl_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ])
        empty = pa.table(
            {n: pa.array([], type=f.type) for n, f in zip(data_schema.names, data_schema)},
            schema=data_schema,
        )
        pq.write_table(empty, os.path.join(root, "data", "chunk-000", "file-000.parquet"))

    failures_caught = []

    # ---- 케이스 1: 유효 데이터셋은 PASS ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        errs: list[str] = []
        validate_info(tmp, errs)
        validate_tasks(tmp, errs)
        validate_data_parquet(tmp, errs)
        if errs:
            print(f"FAIL self-test(valid): 유효 픽스처에서 오류 발생: {errs}", file=sys.stderr)
            sys.exit(1)

    # ---- 케이스 2: 잘못된 codebase_version ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        p = os.path.join(tmp, "meta", "info.json")
        with open(p) as f:
            data = json.load(f)
        data["codebase_version"] = "v2.0"
        with open(p, "w") as f:
            json.dump(data, f)
        errs = []
        validate_info(tmp, errs)
        if not any("codebase_version" in e for e in errs):
            failures_caught.append("codebase_version 불일치 미검출")

    # ---- 케이스 3: 잘못된 fps ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        p = os.path.join(tmp, "meta", "info.json")
        with open(p) as f:
            data = json.load(f)
        data["fps"] = 60
        with open(p, "w") as f:
            json.dump(data, f)
        errs = []
        validate_info(tmp, errs)
        if not any("fps" in e for e in errs):
            failures_caught.append("fps 불일치 미검출")

    # ---- 케이스 4: 태스크 문자열 오류 ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        tasks_table = pa.table({"task_index": [0], "__index_level_0__": ["wrong task"]})
        pq.write_table(tasks_table, os.path.join(tmp, "meta", "tasks.parquet"))
        errs = []
        validate_tasks(tmp, errs)
        if not any("태스크 문자열" in e for e in errs):
            failures_caught.append("태스크 문자열 불일치 미검출")

    # ---- 케이스 5: data parquet 컬럼 누락 ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        broken = pa.table({"action": pa.array([], type=pa.list_(pa.float32(), 6))})
        pq.write_table(broken, os.path.join(tmp, "data", "chunk-000", "file-000.parquet"))
        errs = []
        validate_data_parquet(tmp, errs)
        if not any("컬럼 누락" in e for e in errs):
            failures_caught.append("data 컬럼 누락 미검출")

    # ---- 케이스 6: action shape 오류 ----
    with tempfile.TemporaryDirectory() as tmp:
        make_valid_dataset(tmp)
        wrong_schema = pa.schema([
            pa.field("action", pa.list_(pa.float32(), 3)),  # 6이 아닌 3
            pa.field("observation.state", pa.list_(pa.float32(), 6)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ])
        broken = pa.table(
            {n: pa.array([], type=f.type) for n, f in zip(wrong_schema.names, wrong_schema)},
            schema=wrong_schema,
        )
        pq.write_table(broken, os.path.join(tmp, "data", "chunk-000", "file-000.parquet"))
        errs = []
        validate_data_parquet(tmp, errs)
        if not any("action" in e for e in errs):
            failures_caught.append("action shape 불일치 미검출")

    if failures_caught:
        for msg in failures_caught:
            print(f"FAIL self-test: {msg}", file=sys.stderr)
        sys.exit(1)

    print("PASS self-test: 모든 케이스 통과 (유효 1 + 오류 검출 5)")


# ---------- 메인 ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot v3 데이터셋 스키마 검증기")
    parser.add_argument("dataset_root", nargs="?", help="데이터셋 루트 경로")
    parser.add_argument("--self-test", action="store_true", help="내부 픽스처로 검증기 동작 확인")
    parser.add_argument("--expected-task", default=EXPECTED_TASK, help="tasks.parquet에 기대하는 task 문자열")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.dataset_root:
        parser.print_help()
        sys.exit(1)

    root = args.dataset_root
    if not os.path.isdir(root):
        print(f"ERROR: 디렉토리 없음: {root}", file=sys.stderr)
        sys.exit(1)

    errors: list[str] = []
    warnings: list[str] = []
    validate_info(root, errors, warnings)
    validate_tasks(root, errors, expected_task=args.expected_task)
    validate_data_parquet(root, errors)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"PASS: {os.path.abspath(root)} - 스키마 검증 완료 (오류 0건, 경고 {len(warnings)}건)")


if __name__ == "__main__":
    main()
