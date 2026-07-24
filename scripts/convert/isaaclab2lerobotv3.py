"""IsaacLab HDF5(datagen record) → LeRobot Dataset v3 변환 — **env-free**.

``pickplace_sm.py --record_hdf5`` 가 기록한 캐노니컬 키를 그대로 읽는다:

- ``data/demo_N/obs_x/joint_pos``            (T, 6)  절대 radian, SO101 순서 (obs_t)
- ``data/demo_N/obs_x/images/{top,wrist,front}`` (T, H, W, 3) uint8
- ``data/demo_N/applied_target``             (T, 6)  적용 joint target radian (action_t)
- demo attrs: ``success``(bool) · ``num_samples`` · ``seed``

백엔드 = 기존 :class:`LeRobotV3DatasetWriter` (``record_state_machine``·``--record_lerobot``
직기록과 **동일 writer = 동일 스키마 계약**: h264 video·so_follower·FPS 30). lerobot 패키지도
Isaac Sim 도 불요 — h5py lazy slice 라 에피소드 통째 GPU/RAM 로드 없음. 단위 변환은
``so101_contract.feature_codec`` 단일 소스(radian → PolicyFeature: arm degree + gripper [0,100]).

에피소드 머리(2 s pre-roll)·꼬리(1 s post-hold)는 녹화 시점에 이미 규격이므로 기본
``--skip_frames 0``. 실패(success=False)·최소길이 미달 demo 는 스킵. HF 업로드는 별도
``scripts/data/upload_to_huggingface.py``.

실행:

  python scripts/convert/isaaclab2lerobotv3.py \\
      --hdf5_files datasets/pick_cube_sm.hdf5 --output_dir datasets/pick_cube_sm_v3

검증: ``python scripts/contract/validate_lerobot_schema.py <output_dir>``
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))

from so101_contract.feature_codec import (  # noqa: E402
    CAMERA_KEYS,
    sim_joint_radians_to_policy_feature as to_lerobot_units,
)


def _load_writer_cls():
    """``sim_to_real`` 패키지 __init__(gym 등록 → isaaclab import) 우회 파일 로드.

    cube_specs 의 importlib 파일로드 패턴 — 부모 패키지를 stub 으로 선등록한 뒤
    lerobot_units → lerobot_recorder 순서로 파일을 직접 exec 한다(so101_contract 만 의존).
    """
    for pkg in ("sim_to_real", "sim_to_real.data"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    data_dir = _SRC / "sim_to_real" / "data"
    _load("sim_to_real.data.lerobot_units", data_dir / "lerobot_units.py")
    rec = _load("sim_to_real.data.lerobot_recorder", data_dir / "lerobot_recorder.py")
    return rec.LeRobotV3DatasetWriter


def _demo_sort_key(name: str):
    """demo_10 이 demo_2 앞에 오는 사전순 함정 회피 — 숫자 기준 정렬."""
    try:
        return (0, int(name.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, name)


def convert_demo(writer, demo: h5py.Group, skip_frames: int, min_frames: int) -> bool:
    """demo group 1개 → writer 에피소드 버퍼(스트리밍). 유효하지 않으면 False (스킵)."""
    try:
        target = demo["applied_target"]
        joint_pos = demo["obs_x/joint_pos"]
        images = {key: demo[f"obs_x/images/{key}"] for key in CAMERA_KEYS}
    except KeyError as e:
        print(f"  누락 키 {e} — 스킵 (datagen record 형식 아님)")
        return False

    # pre/post-step 훅 차이로 stream 길이가 ±1 어긋날 수 있음 → 공통 길이
    n = min(len(target), len(joint_pos), *(len(v) for v in images.values()))
    if n - skip_frames < min_frames:
        print(f"  프레임 {n} < 최소 {min_frames + skip_frames} — 스킵")
        return False

    for t in range(skip_frames, n):
        writer.add_frame(
            to_lerobot_units(np.asarray(target[t], dtype=np.float64)),
            to_lerobot_units(np.asarray(joint_pos[t], dtype=np.float64)),
            {key: np.asarray(images[key][t]) for key in CAMERA_KEYS},
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IsaacLab HDF5(datagen record) → LeRobot v3 (env-free, LeRobotV3DatasetWriter).")
    parser.add_argument("--hdf5_files", required=True, help="HDF5 파일 경로(쉼표 구분 복수 가능).")
    parser.add_argument("--output_dir", required=True, help="LeRobot v3 출력 디렉터리.")
    parser.add_argument("--task", default="pick up the cube and place it in the bowl",
                        help="LeRobot task 문자열(계약 canonical 기본값).")
    parser.add_argument("--skip_frames", type=int, default=0,
                        help="에피소드 선두 절삭 프레임 수(녹화가 규격이라 기본 0).")
    parser.add_argument("--min_frames", type=int, default=10, help="최소 프레임 미달 demo 스킵.")
    parser.add_argument("--include_failed", action="store_true",
                        help="success=False demo 도 포함(기본은 성공만).")
    parser.add_argument("--overwrite", action="store_true", help="기존 output_dir 삭제 후 재생성.")
    args = parser.parse_args()

    writer_cls = _load_writer_cls()
    writer = writer_cls(args.output_dir, overwrite=args.overwrite,
                        enable_videos=True, robot_type="so_follower")

    n_saved = 0
    files = [p.strip() for p in args.hdf5_files.split(",") if p.strip()]
    for file_i, path in enumerate(files, 1):
        print(f"[{file_i}/{len(files)}] {path}")
        with h5py.File(path, "r") as h5:
            data = h5["data"]
            for name in sorted(data.keys(), key=_demo_sort_key):
                demo = data[name]
                success = bool(demo.attrs.get("success", False))
                if not success and not args.include_failed:
                    print(f"  {name}: success=False — 스킵")
                    continue
                print(f"  {name}: success={success}")
                if convert_demo(writer, demo, args.skip_frames, args.min_frames):
                    writer.commit_episode(success=True, task_name=args.task)
                    n_saved += 1
                    print(f"  → episode {n_saved} 저장")
                else:
                    writer.commit_episode(success=False, task_name=args.task)  # 버퍼 폐기

    summary = writer.finalize(args.task)
    print(f"완료: {n_saved} 에피소드 → {summary['output_dir']}")


if __name__ == "__main__":
    main()
