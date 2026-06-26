"""LeRobot v3.0 데이터셋 writer — AppLauncher 무의존 공유 모듈.

`rollout_to_lerobot.py`(RSL-RL expert rollout)와 `pick_cube_curobo_demo.py`(cuRobo SM)가
**동일한 v3 데이터 계약**으로 기록하도록 writer 로직을 한곳에 모은다. 스키마는 실기기 SO-101
데이터셋(North Star)과 byte 단위로 같다:

- codebase_version `v3.0` · robot_type `so_follower`
- action/observation.state = 6-dim(arm 5축 + gripper), `lerobot_units` 단위(arm deg·gripper [0,100])
- observation.images.{top,wrist,front} 480×640×3 h264 yuv420p · fps 30

caller 가 이미 `to_lerobot_units` 로 변환한 6-dim 배열과 (H,W,3) uint8 이미지를 넘긴다(이 모듈은 단위
변환을 하지 않는다 — `lerobot_units` 상수만 재사용).

ABI: `pyarrow`(<19 핀)·`imageio` 는 Isaac Sim 과의 ABI 충돌을 피하려 **AppLauncher 부팅 후** import
해야 한다 → 이 모듈은 모든 무거운 import 를 메서드 내부에서 지연 수행한다(import 시점 무관).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sim_to_real.data.lerobot_units import (
    CAMERA_KEYS,
    FPS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_FEATURE_NAMES,
)


# ── 프레임/에피소드 버퍼 (rollout_to_lerobot 와 동일 구조) ────────────────────
@dataclass
class FrameRecord:
    action: np.ndarray
    state: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    frames: list[FrameRecord] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass
class ImageStats:
    # 모든 640×480 pixel을 float64로 변환하면 video encoding보다 통계 계산이 더 느려진다.
    # 규칙 격자 1/64 표본이면 channel mean/std 오차는 충분히 작고 처리량은 크게 줄어든다.
    sample_stride: int = 8
    count: int = 0
    channel_min: np.ndarray = field(default_factory=lambda: np.full(3, 1.0, dtype=np.float64))
    channel_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    channel_sumsq: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def update(self, image_u8: np.ndarray) -> None:
        values = image_u8[:: self.sample_stride, :: self.sample_stride].astype(np.float32) / 255.0
        flat = values.reshape(-1, 3)
        self.count += flat.shape[0]
        self.channel_min = np.minimum(self.channel_min, flat.min(axis=0))
        self.channel_max = np.maximum(self.channel_max, flat.max(axis=0))
        self.channel_sum += flat.sum(axis=0, dtype=np.float64)
        self.channel_sumsq += np.square(flat).sum(axis=0, dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        if self.count <= 0:
            mean = np.full(3, 0.5, dtype=np.float64)
            std = np.zeros(3, dtype=np.float64)
            min_v = np.zeros(3, dtype=np.float64)
            max_v = np.ones(3, dtype=np.float64)
        else:
            mean = self.channel_sum / self.count
            var = np.maximum(self.channel_sumsq / self.count - np.square(mean), 0.0)
            std = np.sqrt(var)
            min_v = self.channel_min
            max_v = self.channel_max

        def nested(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(v)]] for v in values.tolist()]

        return {
            "min": nested(min_v),
            "max": nested(max_v),
            "mean": nested(mean),
            "std": nested(std),
            "count": [int(self.count)],
            "q01": nested(min_v),
            "q10": nested(mean),
            "q50": nested(mean),
            "q90": nested(mean),
            "q99": nested(max_v),
        }


def _numeric_stats(array_like: list[Any] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array_like)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        width = arr.shape[1] if arr.ndim > 1 else 1
        zeros = [0.0 for _ in range(width)]
        return {k: zeros for k in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")} | {"count": [0]}
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


class LeRobotV3DatasetWriter:
    """LeRobot v3.0 데이터셋을 점진적으로 기록한다(성공 에피소드만 flush).

    사용:
        writer = LeRobotV3DatasetWriter(out_dir, overwrite=True)
        # 에피소드 루프
        writer.add_frame(action_6, state_6, {"top": img, "wrist": img, "front": img})
        ...
        writer.commit_episode(success=True, task_name="pick up the cube and place it in the bowl")
        # 종료
        summary = writer.finalize()
    """

    def __init__(
        self,
        output_dir: Path | str,
        *,
        overwrite: bool = False,
        enable_videos: bool = True,
        robot_type: str = "so_follower",
        video_quality: int = 8,
        video_preset: str | None = None,
        video_codec: str = "libx264",
        video_ffmpeg_exe: str | None = None,
        video_nvenc_cq: int = 23,
    ) -> None:
        self.root = Path(output_dir)
        self.enable_videos = enable_videos
        self.robot_type = robot_type
        if not 0 <= int(video_quality) <= 10:
            raise ValueError(f"video_quality must be in [0,10], got {video_quality}")
        self.video_quality = int(video_quality)
        self.video_preset = video_preset.strip() if video_preset else None
        if video_codec not in {"libx264", "h264_nvenc"}:
            raise ValueError(f"unsupported video_codec: {video_codec}")
        if not 0 <= int(video_nvenc_cq) <= 51:
            raise ValueError(f"video_nvenc_cq must be in [0,51], got {video_nvenc_cq}")
        self.video_codec = video_codec
        self.video_ffmpeg_exe = video_ffmpeg_exe.strip() if video_ffmpeg_exe else None
        self.video_nvenc_cq = int(video_nvenc_cq)
        self._prepare_output_dir(self.root, overwrite)
        self.root = self.root.resolve()

        self._rows: list[dict[str, Any]] = []
        self._episodes_meta: list[dict[str, Any]] = []
        self._image_stats = {cam: ImageStats() for cam in CAMERA_KEYS}
        self._current = EpisodeRecord()
        self._episode_index = 0
        self._task_name: str | None = None

        self._writers: dict[str, Any] = {}
        if self.enable_videos:
            self._open_video_writers()

    # ── 디렉터리 ──────────────────────────────────────────────────────────
    @staticmethod
    def _prepare_output_dir(path: Path, overwrite: bool) -> None:
        resolved = path.resolve()
        if resolved.exists():
            if not overwrite:
                raise FileExistsError(f"output_dir already exists: {resolved}")
            unsafe_targets = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
            if resolved in unsafe_targets or len(resolved.parts) < 4:
                raise ValueError(f"Refusing to remove unsafe output_dir: {resolved}")
            shutil.rmtree(resolved)

        (resolved / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (resolved / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMERA_KEYS:
            (resolved / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ── 비디오 writer (지연 import) ───────────────────────────────────────
    def _open_video_writers(self) -> None:
        # imageio-ffmpeg 번들 바이너리는 NVENC 없이 빌드된 경우가 많다. 시스템 FFmpeg를
        # 명시하면 import 전에 환경변수를 주입해 실제 encoder 목록을 사용하게 한다.
        if self.video_ffmpeg_exe:
            os.environ["IMAGEIO_FFMPEG_EXE"] = self.video_ffmpeg_exe
        import imageio.v2 as imageio  # noqa: PLC0415  (ABI: AppLauncher 부팅 후 import)

        for cam in CAMERA_KEYS:
            p = self.root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
            ffmpeg_params = ["-pix_fmt", "yuv420p"]
            if self.video_preset:
                ffmpeg_params = ["-preset", self.video_preset, *ffmpeg_params]
            quality: int | None = self.video_quality
            if self.video_codec == "h264_nvenc":
                # NVENC는 imageio의 generic qscale 대신 constant-quality(CQ)를 직접 지정한다.
                quality = None
                ffmpeg_params = ["-cq", str(self.video_nvenc_cq), *ffmpeg_params]
            self._writers[cam] = imageio.get_writer(
                p,
                fps=FPS,
                codec=self.video_codec,
                quality=quality,
                macro_block_size=1,
                ffmpeg_params=ffmpeg_params,
            )

    def _close_video_writers(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers = {}

    # ── 프레임/에피소드 ───────────────────────────────────────────────────
    def add_frame(
        self,
        action: np.ndarray,
        state: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
    ) -> None:
        """현재 에피소드 버퍼에 1 프레임 추가(caller 가 단위 변환·이미지 캡처 완료)."""
        self._current.frames.append(
            FrameRecord(
                action=np.asarray(action, dtype=np.float32),
                state=np.asarray(state, dtype=np.float32),
                images=images or {},
            )
        )

    def commit_episode(self, success: bool, task_name: str) -> bool:
        """성공 에피소드면 rows/meta/video 에 flush, 버퍼는 항상 비운다. flush 여부 반환."""
        committed = False
        if success and self._current.length > 0:
            self._append_success_episode(self._current, self._episode_index, task_name)
            self._episode_index += 1
            self._task_name = task_name
            committed = True
        self._current.clear()
        return committed

    def _append_success_episode(self, episode: EpisodeRecord, episode_index: int, task_name: str) -> None:
        start_index = len(self._rows)
        length = episode.length
        for frame_idx, frame in enumerate(episode.frames):
            self._rows.append(
                {
                    "action": frame.action.tolist(),
                    "observation.state": frame.state.tolist(),
                    "timestamp": frame_idx / FPS,
                    "frame_index": frame_idx,
                    "episode_index": episode_index,
                    "index": start_index + frame_idx,
                    "task_index": 0,
                }
            )

        video_from = start_index / FPS
        video_to = (start_index + length) / FPS
        meta: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [task_name],
            "length": length,
            "dataset_from_index": start_index,
            "dataset_to_index": start_index + length,
        }
        for cam in CAMERA_KEYS:
            meta[f"videos/observation.images.{cam}/chunk_index"] = 0
            meta[f"videos/observation.images.{cam}/file_index"] = 0
            meta[f"videos/observation.images.{cam}/from_timestamp"] = video_from
            meta[f"videos/observation.images.{cam}/to_timestamp"] = video_to

        for frame in episode.frames:
            for cam, image in frame.images.items():
                if cam in self._writers:
                    self._writers[cam].append_data(image)
                if cam in self._image_stats:
                    self._image_stats[cam].update(image)

        self._episodes_meta.append(meta)

    # ── 메타 파일 ─────────────────────────────────────────────────────────
    def _write_data_parquet(self) -> None:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        fsl6 = pa.list_(pa.float32(), 6)
        table = pa.table(
            {
                "action": pa.array([r["action"] for r in self._rows], type=fsl6),
                "observation.state": pa.array([r["observation.state"] for r in self._rows], type=fsl6),
                "timestamp": pa.array([r["timestamp"] for r in self._rows], type=pa.float32()),
                "frame_index": pa.array([r["frame_index"] for r in self._rows], type=pa.int64()),
                "episode_index": pa.array([r["episode_index"] for r in self._rows], type=pa.int64()),
                "index": pa.array([r["index"] for r in self._rows], type=pa.int64()),
                "task_index": pa.array([r["task_index"] for r in self._rows], type=pa.int64()),
            }
        )
        pq.write_table(table, self.root / "data" / "chunk-000" / "file-000.parquet")

    def _write_tasks(self, task_name: str) -> None:
        # LeRobot v3 는 tasks.parquet 을 pandas DataFrame(인덱스=task 문자열, 컬럼=task_index)으로
        # 읽어 task_index→문자열 매핑한다. pyarrow 직접 write 면 pandas index 메타데이터가 없어
        # 룩업이 깨진다("Task cannot be None"). 실기기셋과 동일하게 pandas 로 unnamed-index 기록.
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame({"task_index": [0]}, index=[task_name])
        df.to_parquet(self.root / "meta" / "tasks.parquet")

    def _write_episodes(self) -> None:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        table = pa.table(
            {
                "episode_index": pa.array([e["episode_index"] for e in self._episodes_meta], type=pa.int64()),
                "tasks": pa.array([e["tasks"] for e in self._episodes_meta], type=pa.list_(pa.string())),
                "length": pa.array([e["length"] for e in self._episodes_meta], type=pa.int64()),
                "data/chunk_index": pa.array([0 for _ in self._episodes_meta], type=pa.int64()),
                "data/file_index": pa.array([0 for _ in self._episodes_meta], type=pa.int64()),
                "dataset_from_index": pa.array([e["dataset_from_index"] for e in self._episodes_meta], type=pa.int64()),
                "dataset_to_index": pa.array([e["dataset_to_index"] for e in self._episodes_meta], type=pa.int64()),
                **{
                    f"videos/observation.images.{cam}/{name}": pa.array(
                        [e[f"videos/observation.images.{cam}/{name}"] for e in self._episodes_meta],
                        type=pa.float64() if "timestamp" in name else pa.int64(),
                    )
                    for cam in CAMERA_KEYS
                    for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
                },
                "meta/episodes/chunk_index": pa.array([0 for _ in self._episodes_meta], type=pa.int64()),
                "meta/episodes/file_index": pa.array([0 for _ in self._episodes_meta], type=pa.int64()),
            }
        )
        pq.write_table(table, self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    def _write_info(self) -> None:
        video_info = {
            "video.height": IMAGE_HEIGHT,
            "video.width": IMAGE_WIDTH,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": IMAGE_CHANNELS,
            "has_audio": False,
        }
        features: dict[str, Any] = {
            "action": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
            "observation.state": {"dtype": "float32", "names": JOINT_FEATURE_NAMES, "shape": [6]},
        }
        for cam in CAMERA_KEYS:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                "names": ["height", "width", "channels"],
                "info": video_info,
            }
        features.update(
            {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            }
        )
        total_episodes = len(self._episodes_meta)
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": len(self._rows),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": FPS,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        with (self.root / "meta" / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def _write_stats(self) -> None:
        stats = {
            "action": _numeric_stats([r["action"] for r in self._rows]),
            "observation.state": _numeric_stats([r["observation.state"] for r in self._rows]),
            "timestamp": _numeric_stats([r["timestamp"] for r in self._rows]),
            "frame_index": _numeric_stats([r["frame_index"] for r in self._rows]),
            "episode_index": _numeric_stats([r["episode_index"] for r in self._rows]),
            "index": _numeric_stats([r["index"] for r in self._rows]),
            "task_index": _numeric_stats([r["task_index"] for r in self._rows]),
        }
        for cam in CAMERA_KEYS:
            stats[f"observation.images.{cam}"] = self._image_stats[cam].to_json()
        with (self.root / "meta" / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 종료 ──────────────────────────────────────────────────────────────
    @property
    def num_episodes(self) -> int:
        return len(self._episodes_meta)

    @property
    def num_frames(self) -> int:
        return len(self._rows)

    def finalize(self, task_name: str | None = None) -> dict[str, Any]:
        """모든 메타 파일 기록 + 비디오 close. 요약 반환."""
        self._close_video_writers()
        task = task_name or self._task_name or "pick up the cube and place it in the bowl"
        self._write_data_parquet()
        self._write_tasks(task)
        self._write_episodes()
        self._write_info()
        self._write_stats()
        return {
            "output_dir": str(self.root),
            "total_episodes": self.num_episodes,
            "total_frames": self.num_frames,
            "fps": FPS,
            "videos": self.enable_videos,
        }
