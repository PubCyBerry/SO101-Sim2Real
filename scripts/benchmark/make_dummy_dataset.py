#!/usr/bin/env python3
"""VLA/WAM 학습 비용 벤치마크용 dummy LeRobot v3 데이터셋 생성기.

프로젝트 실데이터(`src/sim_to_real/data/lerobot_recorder.py`)와 **같은 스키마**를 쓴다 —
6D joint state/action(arm 5 radian + gripper policy feature [0,100]) + 640×480@30fps 카메라.
단 카메라는 벤치마크 요구대로 **top/wrist 2대**만 둔다. 상수는 `so101_contract.feature_codec`
단일 소스에서 읽는다.

영상은 `use_videos=True` 로 **실제 mp4 로 인코딩**한다 — dataloader 의 video decode 비용이
VLA 학습 스텝 시간의 큰 축이라 image dtype fixture 로는 벤치마크가 왜곡된다.

`meta/info.json` 에 schema v2 joint 블록과 `meta/action_representation_stats.json` 을 함께 써서
act/smolvla/groot 경로(`--policy.action_representation.mode=joint_absolute`)가 dataset 만으로
resolve 되게 한다.

컨테이너 안에서 실행한다(호스트 uv 환경엔 lerobot 0.6.0 이 없다):

.. code-block:: bash

    docker run --rm -u $(id -u):$(id -g) \\
      -v "$PWD/scripts:/workspace/scripts:ro" -v "$PWD/src:/workspace/src:ro" \\
      -v /DISK1/so101-sim2real/datasets:/workspace/datasets \\
      --entrypoint python policy-server:0.6.0 \\
      /workspace/scripts/benchmark/make_dummy_dataset.py \\
      --output-dir /workspace/datasets/benchmark_dummy_2cam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs.video import RGBEncoderConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from so101_contract.action_dataset_contract import (  # noqa: E402
    ACTION_DATASET_CONTRACT_VERSION,
    DATASET_CONTRACT_BLOCK,
    resolve_action_representation_contract,
)
from so101_contract.action_representation import (  # noqa: E402
    ActionRepresentationMode,
    ActionRepresentationSpec,
)
from so101_contract.action_representation_stats import (  # noqa: E402
    ActionStatsSampling,
    calculate_action_representation_stats,
    empty_stats_artifact,
    load_lerobot_v3_episodes,
    upsert_stats_profile,
    write_action_stats_artifact,
)
from so101_contract.feature_codec import (  # noqa: E402
    FPS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_FEATURE_NAMES,
)
from so101_contract.joint_topology import so101_arm_joint_topology  # noqa: E402

JOINT_TOPOLOGY = so101_arm_joint_topology()
FEATURE_NAMES = list(JOINT_FEATURE_NAMES)
STATE_DIM = len(FEATURE_NAMES)
TASK = "pick the cube and place it in the bowl"


def _joint_row(episode: int, frame: int, frames: int) -> np.ndarray:
    """joint limit 안에서 매끄럽게 도는 궤적. 에피소드마다 위상을 바꿔 다양성을 준다."""
    phase = 2.0 * np.pi * frame / max(frames - 1, 1)
    offset = 0.37 * episode
    arm = np.asarray(
        [
            0.60 * np.sin(phase + offset),
            -0.70 + 0.45 * np.sin(0.5 * phase + offset),
            0.80 + 0.40 * np.cos(0.5 * phase + offset),
            0.25 * np.sin(2.0 * phase),
            -0.10 + 0.30 * np.cos(phase + offset),
        ],
        dtype=np.float64,
    )
    # gripper 는 policy feature 단위 [0,100] — 에피소드 중반에 닫고 끝에 연다.
    gripper = 100.0 if frame < frames // 3 or frame > 2 * frames // 3 else 8.0
    return np.concatenate([arm, [gripper]]).astype(np.float32)


def _frame_image(
    rng: np.random.Generator,
    camera_index: int,
    episode: int,
    frame: int,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """h264 비트레이트가 퇴화하지 않을 정도의 합성 장면.

    배경 그라디언트(정적) + 움직이는 블록(모션 벡터) + 약한 노이즈(고주파). 순수 노이즈는
    비트레이트를, 단색은 decode 비용을 비현실적으로 만든다.
    """
    ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    image = np.empty((height, width, IMAGE_CHANNELS), dtype=np.float32)
    image[..., 0] = 0.45 * xs + 0.25 * ys + 20.0 * camera_index
    image[..., 1] = 0.35 * ys + 30.0 * camera_index
    image[..., 2] = 128.0 - 0.20 * xs + 0.15 * ys

    # 프레임에 따라 대각선으로 움직이는 밝은 블록 (grasp 대상 큐브 대역)
    box = max(height, width) // 10
    t = frame / max(frames - 1, 1)
    cy = int((0.2 + 0.6 * t) * (height - box))
    cx = int((0.7 - 0.5 * t) * (width - box))
    image[cy : cy + box, cx : cx + box] = (240.0, 90.0 + 40.0 * camera_index, 60.0)

    # 정적 고주파 텍스처 + 노이즈 — 단색/저주파만이면 h264 비트레이트가 비현실적으로 낮아져
    # decode 비용을 과소평가한다.
    image += 12.0 * np.sin(xs * 0.7)[..., None] * np.cos(ys * 0.5)[..., None]
    image += rng.normal(0.0, 6.0, size=image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def create_dataset(
    output_dir: Path,
    *,
    repo_id: str,
    cameras: tuple[str, ...],
    episodes: int,
    frames_per_episode: int,
    height: int,
    width: int,
    fps: int,
    horizons: tuple[int, ...],
    seed: int,
    write_contract: bool,
    vcodec: str,
    crf: int,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output_dir}")
    if frames_per_episode < max(horizons):
        raise ValueError(f"frames_per_episode({frames_per_episode}) < max horizon({max(horizons)})")

    feature = {"dtype": "float32", "shape": (STATE_DIM,), "names": FEATURE_NAMES}
    features: dict[str, dict] = {
        "observation.state": dict(feature),
        "action": dict(feature),
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, IMAGE_CHANNELS),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="so101_follower",
        use_videos=True,
        # lerobot 0.6.0 기본은 libsvtav1 이지만 프로젝트 recorder 는 h264 로 쓴다.
        # decode 비용이 codec 에 따라 크게 달라지므로 실데이터와 같은 h264 로 맞춘다.
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec, crf=crf),
    )
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        for frame in range(frames_per_episode):
            row: dict[str, object] = {
                "observation.state": _joint_row(episode, frame, frames_per_episode),
                "action": _joint_row(episode, frame + 1, frames_per_episode),
                "task": TASK,
            }
            for camera_index, camera in enumerate(cameras):
                row[f"observation.images.{camera}"] = _frame_image(
                    rng, camera_index, episode, frame, frames_per_episode, height, width
                )
            dataset.add_frame(row)
        dataset.save_episode()
    dataset.finalize()

    stats_path = None
    if write_contract:
        stats_path = write_contract_metadata(output_dir, horizons)

    return {
        "dataset_root": str(output_dir.resolve()),
        "repo_id": repo_id,
        "cameras": list(cameras),
        "resolution": [height, width],
        "fps": fps,
        "episodes": episodes,
        "frames_per_episode": frames_per_episode,
        "total_frames": episodes * frames_per_episode,
        "state_action_dim": STATE_DIM,
        "feature_names": FEATURE_NAMES,
        "vcodec": vcodec,
        "crf": crf,
        "stats_horizons": list(horizons),
        "stats_artifact": str(stats_path) if stats_path else None,
    }


def _write_schema_v2_block(output_dir: Path) -> None:
    """joint absolute dataset 계약 블록 — 프로젝트 실데이터와 같은 meta 를 갖게 한다."""
    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[DATASET_CONTRACT_BLOCK] = {
        "version": ACTION_DATASET_CONTRACT_VERSION,
        "space": "joint",
        "storage_reference": "absolute",
        "transform_group": "arm_joints",
        "groups": {
            "arm_joints": {"start": 0, "end": STATE_DIM - 1},
            "gripper_position": {"start": STATE_DIM - 1, "end": STATE_DIM},
        },
        # joint metadata 는 **dataset feature 이름**으로 키가 잡힌다("shoulder_pan.pos").
        # topology 쪽 이름("shoulder_pan")과 다르므로 여기서 갈아끼운다.
        "joints": [
            {**joint.to_dict(), "name": feature_name}
            for joint, feature_name in zip(JOINT_TOPOLOGY.joints, FEATURE_NAMES, strict=False)
        ],
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def write_contract_metadata(output_dir: Path, horizons: tuple[int, ...]) -> Path:
    """schema v2 joint 블록 + horizon 별 stats profile.

    stats profile 은 (kind, **horizon**, dataset fingerprint) 로 조회된다 —
    정책마다 chunk_size 가 달라(ACT 100 · SmolVLA 50 · GR00T 40/16) 필요한 horizon 을 전부
    만들어 두지 않으면 학습 시작 시
    ``KeyError: no action stats profile matches the transform/sampler/dataset`` 로 죽는다.
    """
    _write_schema_v2_block(output_dir)
    artifact = empty_stats_artifact()
    stats_file = None
    for mode in (
        ActionRepresentationMode.JOINT_ABSOLUTE,
        ActionRepresentationMode.JOINT_RELATIVE,
    ):
        spec = ActionRepresentationSpec(mode=mode)
        contract = resolve_action_representation_contract(output_dir, spec)
        episodes_arrays = load_lerobot_v3_episodes(
            output_dir,
            state_key=contract.state_key,
            action_key=contract.action_key,
            state_dim=contract.state_dim,
            action_dim=contract.action_dim,
        )
        for horizon in horizons:
            result = calculate_action_representation_stats(
                episodes_arrays,
                ActionStatsSampling(action_delta_indices=tuple(range(horizon))),
                contract.transform,
                dataset_fingerprint=contract.fingerprint,
            )
            artifact, _ = upsert_stats_profile(artifact, result)
        stats_file = spec.stats_file
    return write_action_stats_artifact(output_dir, artifact, output_file=stats_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101_benchmark_dummy_2cam")
    parser.add_argument("--cameras", default="top,wrist", help="쉼표 구분 카메라 키")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=200,
        help="ACT chunk_size=100 등 긴 horizon 을 덮으려면 >=128 권장",
    )
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--horizons",
        default="100,50,40,32,16",
        help="stats profile 을 만들 action horizon 목록 (정책별 chunk_size 를 모두 덮어야 한다)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="기존 데이터셋의 meta 블록·stats 만 다시 쓴다 (영상 재인코딩 안 함)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vcodec", default="h264", help="lerobot 기본값은 libsvtav1")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--no-contract",
        action="store_true",
        help="schema v2 블록·stats 생략 (순수 stock lerobot 데이터셋)",
    )
    args = parser.parse_args()

    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    if args.stats_only:
        path = write_contract_metadata(args.output_dir, horizons)
        print(json.dumps({"stats_artifact": str(path), "stats_horizons": list(horizons)}, indent=2))
        return

    result = create_dataset(
        args.output_dir,
        repo_id=args.repo_id,
        cameras=tuple(c.strip() for c in args.cameras.split(",") if c.strip()),
        episodes=args.episodes,
        frames_per_episode=args.frames_per_episode,
        height=args.height,
        width=args.width,
        fps=args.fps,
        horizons=horizons,
        seed=args.seed,
        write_contract=not args.no_contract,
        vcodec=args.vcodec,
        crf=args.crf,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
