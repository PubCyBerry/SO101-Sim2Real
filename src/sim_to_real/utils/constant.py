"""Constants and asset paths shared across the sim_to_real package.

Mirrors the role of ``leisaac.utils.constant`` so other modules can keep the
same import surface (``ASSETS_ROOT`` / ``SINGLE_ARM_JOINT_NAMES``).
"""

from __future__ import annotations

PEN_NAMES: list[str] = ["PenWhite", "PenGray", "PenBlack", "PenBlue"]
PEN_CUP_NAME: str = "PenCup"

CUBE_NAMES: list[str] = ["Cube1", "Cube2", "Cube3", "Cube4"]
BOWL_NAME: str = "Bowl"
