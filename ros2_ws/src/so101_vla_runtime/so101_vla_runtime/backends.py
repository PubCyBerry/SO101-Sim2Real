"""Replay, ACT/SmolVLA direct, GR00T ZMQ canonical backend."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from so101_parity.calibration import CalibrationBundle
from so101_parity.model_codec import ModelCodec


class Backend(Protocol):
    checkpoint_hash: str

    def infer(
        self,
        *,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        start_step: int,
    ) -> np.ndarray: ...


class ReplayBackend:
    def __init__(self, checkpoint: Path, checkpoint_hash: str) -> None:
        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
        if raw.get("schema") != "so101-deterministic-replay-v1":
            raise ValueError("지원하지 않는 replay checkpoint")
        self.chunks = tuple(np.asarray(chunk, dtype=np.float32) for chunk in raw["chunks"])
        if not self.chunks or any(chunk.ndim != 2 or chunk.shape[1] != 6 for chunk in self.chunks):
            raise ValueError("replay chunk shape은 모두 (N, 6)이어야 한다")
        self.task = str(raw["task"])
        self.checkpoint_hash = checkpoint_hash

    def infer(
        self,
        *,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        start_step: int,
    ) -> np.ndarray:
        del canonical_state, images
        if task != self.task:
            raise ValueError(f"replay task 불일치: {task!r} != {self.task!r}")
        index = (start_step // len(self.chunks[0])) % len(self.chunks)
        return self.chunks[index].copy()


class DirectLeRobotBackend:
    """ACT/SmolVLA를 LeRobot pre/postprocessor와 함께 직접 실행한다."""

    def __init__(
        self,
        *,
        policy_type: str,
        checkpoint: str,
        checkpoint_hash: str,
        model_codec: ModelCodec,
        actions_per_chunk: int,
        device: str,
        rename_map: dict[str, str] | None = None,
    ) -> None:
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.model_codec = model_codec
        self.actions_per_chunk = int(actions_per_chunk)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()
        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": rename_map or {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        self.checkpoint_hash = checkpoint_hash

    def infer(
        self,
        *,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        start_step: int,
    ) -> np.ndarray:
        del start_step
        state = self.model_codec.canonical_to_model(canonical_state)
        observation = {
            "observation.state": self.torch.from_numpy(state),
            "task": task,
        }
        for name, image in images.items():
            tensor = self.torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
            observation[f"observation.images.{name}"] = tensor
        with self.torch.inference_mode():
            processed = self.preprocessor(observation)
            chunk = self.policy.predict_action_chunk(processed)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : self.actions_per_chunk, :]
            actions = [
                self.postprocessor(chunk[:, index, :]).squeeze(0).detach().cpu().numpy()
                for index in range(chunk.shape[1])
            ]
        return self.model_codec.model_chunk_to_canonical(np.stack(actions).astype(np.float32))


def _encode_msgpack(value):
    if isinstance(value, np.ndarray):
        import msgpack

        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
    raise TypeError(f"msgpack unsupported type: {type(value)}")


def _decode_msgpack(value):
    if isinstance(value, dict) and "__ndarray_class__" in value:
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


class GrootZmqBackend:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_ms: int,
        checkpoint_hash: str,
        model_codec: ModelCodec,
        actions_per_chunk: int,
    ) -> None:
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.zmq = zmq
        self.model_codec = model_codec
        self.actions_per_chunk = int(actions_per_chunk)
        self.checkpoint_hash = checkpoint_hash
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)

    def infer(
        self,
        *,
        canonical_state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        start_step: int,
    ) -> np.ndarray:
        del start_step
        state = self.model_codec.canonical_to_model(canonical_state)
        observation = {
            "video": {
                name: image.reshape(1, 1, *image.shape)
                for name, image in images.items()
            },
            "state": {
                "single_arm": state[:5].reshape(1, 1, 5),
                "gripper": state[5:].reshape(1, 1, 1),
            },
            "language": {"annotation.human.task_description": [[task]]},
        }
        request = {"endpoint": "get_action", "data": {"observation": observation, "options": None}}
        self.socket.send(self.msgpack.packb(request, default=_encode_msgpack))
        response = self.msgpack.unpackb(self.socket.recv(), object_hook=_decode_msgpack)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        action = response[0]
        arm = np.asarray(action["single_arm"], dtype=np.float32)
        gripper = np.asarray(action["gripper"], dtype=np.float32)
        if arm.ndim == 3:
            arm = arm[0]
        if gripper.ndim == 3:
            gripper = gripper[0]
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        model_chunk = np.concatenate([arm, gripper], axis=-1)[: self.actions_per_chunk]
        return self.model_codec.model_chunk_to_canonical(model_chunk)


def make_backend(
    manifest: dict,
    manifest_dir: Path,
    calibration: CalibrationBundle,
) -> Backend:
    from so101_parity.manifest import file_sha256, tree_sha256

    checkpoint_ref = Path(str(manifest["checkpoint_ref"]))
    checkpoint = checkpoint_ref if checkpoint_ref.is_absolute() else manifest_dir / checkpoint_ref
    actual_hash = file_sha256(checkpoint) if checkpoint.is_file() else tree_sha256(checkpoint)
    if actual_hash != manifest["checkpoint_hash"]:
        raise ValueError(
            f"checkpoint hash 불일치: actual={actual_hash}, expected={manifest['checkpoint_hash']}"
        )
    codec = ModelCodec(str(manifest["model_frame"]), calibration)
    backend = str(manifest["backend"])
    if backend == "replay":
        return ReplayBackend(checkpoint, actual_hash)
    if backend in ("act", "smolvla"):
        return DirectLeRobotBackend(
            policy_type=backend,
            checkpoint=str(checkpoint),
            checkpoint_hash=actual_hash,
            model_codec=codec,
            actions_per_chunk=int(manifest["chunk_size"]),
            device=str(manifest.get("device", "cuda")),
            rename_map=dict(manifest.get("rename_map", {})),
        )
    if backend == "groot_zmq":
        return GrootZmqBackend(
            endpoint=str(manifest.get("groot_zmq_endpoint", "tcp://127.0.0.1:5555")),
            timeout_ms=int(manifest.get("groot_zmq_timeout_ms", 60000)),
            checkpoint_hash=actual_hash,
            model_codec=codec,
            actions_per_chunk=int(manifest["chunk_size"]),
        )
    raise ValueError(f"지원하지 않는 backend: {backend!r}")
