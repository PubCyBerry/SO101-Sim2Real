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

CUBE_NAMES: list[str] = ["Cube1", "Cube2", "Cube3", "Cube4"]
BOWL_NAME: str = "Bowl"
