"""Constants and asset paths shared across the sim_to_real package.

Mirrors the role of ``leisaac.utils.constant`` so other modules can keep the
same import surface (``ASSETS_ROOT`` / ``SINGLE_ARM_JOINT_NAMES``).
"""

from __future__ import annotations

# 큐브 크기/질량 단일 진실 소스(cube_specs) 를 패키지 consumer 가 같은 import
# 표면(sim_to_real.utils.constant)에서 가져가도록 re-export.
from .cube_specs import (  # noqa: F401
    CUBE_HALF_EXTENTS,
    CUBE_MASSES,
    CUBE_SIZES,
    CUBE_SPECS,
    MAX_CUBE_FOOTPRINT_RADIUS,
    MAX_CUBE_SIZE,
    CubeSpec,
)

# 단일 큐브 씬(2026-06-26): 40mm Cube1 1개만. 매트 제거 + 1-cube 구성.
CUBE_NAMES: list[str] = ["Cube1"]
BOWL_NAME: str = "Bowl"
