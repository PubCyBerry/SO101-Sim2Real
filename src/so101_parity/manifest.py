"""Runtime/checkpoint manifest hash와 fail-closed 검증."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contract import canonical_json, content_hash


class RuntimeManifestError(ValueError):
    """Runtime artifact가 선언된 manifest와 다를 때 발생한다."""


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: str | Path, *, exclude_names: Iterable[str] = ()) -> str:
    base = Path(root)
    excluded = set(exclude_names)
    if not base.exists():
        raise RuntimeManifestError(f"hash 대상이 없다: {base}")
    rows: list[tuple[str, str]] = []
    if base.is_file():
        rows.append((base.name, file_sha256(base)))
    else:
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if path.name in excluded:
                continue
            rows.append((path.relative_to(base).as_posix(), file_sha256(path)))
    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeManifest:
    raw: Mapping[str, Any]
    manifest_hash: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeManifest":
        if raw.get("schema") != "so101-runtime-manifest-v1":
            raise RuntimeManifestError(f"지원하지 않는 runtime manifest: {raw.get('schema')!r}")
        expected = content_hash(raw, "manifest_hash")
        supplied = raw.get("manifest_hash")
        if supplied != expected:
            raise RuntimeManifestError(
                f"runtime manifest hash 불일치: supplied={supplied}, expected={expected}"
            )
        required = (
            "backend",
            "checkpoint_ref",
            "model_frame",
            "task",
            "chunk_size",
            "contract_hash",
            "calibration_hash",
            "motor_profile_hash",
            "checkpoint_hash",
            "pixi_lock_hash",
            "runtime_config_hash",
            "stack",
        )
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise RuntimeManifestError(f"runtime manifest 필수 필드 누락: {missing}")
        return cls(dict(raw), expected)

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeManifest":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def assert_hashes(
        self,
        *,
        contract_hash: str,
        calibration_hash: str,
        motor_profile_hash: str,
        checkpoint_hash: str,
        pixi_lock_hash: str | None = None,
        runtime_config_hash: str | None = None,
    ) -> None:
        actual = {
            "contract_hash": contract_hash,
            "calibration_hash": calibration_hash,
            "motor_profile_hash": motor_profile_hash,
            "checkpoint_hash": checkpoint_hash,
        }
        if pixi_lock_hash is not None:
            actual["pixi_lock_hash"] = pixi_lock_hash
        if runtime_config_hash is not None:
            actual["runtime_config_hash"] = runtime_config_hash
        mismatches = [
            f"{key}: runtime={value}, manifest={self.raw.get(key)}"
            for key, value in actual.items()
            if value != self.raw.get(key)
        ]
        if mismatches:
            raise RuntimeManifestError("runtime hash 불일치: " + "; ".join(mismatches))

    @staticmethod
    def with_hash(raw: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        result["manifest_hash"] = content_hash(result, "manifest_hash")
        return result
