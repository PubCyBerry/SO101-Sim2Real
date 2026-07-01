"""SO101 Follower USD의 재료 색상을 실물 사진 기준으로 패치하는 headless 스크립트.

실행:
    uv run scripts/environments/utils/patch_robot_colors.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO101 follower USD 재료 색상 패치")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import os
import sys
from pxr import Gf, Usd

USD_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "so101_follower.usd")
)

# ── 색상 팔레트 (사진 기준) ───────────────────────────────────────────────────
PLASTIC_PURPLE = Gf.Vec3f(0.40, 0.03, 0.75)   # 보라색 3D프린트 PLA
SERVO_DARK     = Gf.Vec3f(0.08, 0.08, 0.10)   # STS3215 서보 케이싱 (거의 검정)
METAL_SILVER   = Gf.Vec3f(0.55, 0.55, 0.58)   # 나사·베어링 (실버)

ROUGHNESS_PLASTIC = 0.60
ROUGHNESS_SERVO   = 0.45
ROUGHNESS_METAL   = 0.25

METALLIC_PLASTIC  = 0.0
METALLIC_SERVO    = 0.0
METALLIC_METAL    = 0.8

SERVO_KEYWORDS = ("servo", "motor", "sts", "actuator", "driver", "controller")
METAL_KEYWORDS = ("screw", "bolt", "nut", "bearing", "shaft", "axle",
                  "steel", "metal", "aluminum", "aluminium", "bracket_metal")


def classify(path_str: str) -> str:
    lower = path_str.lower()
    if any(k in lower for k in SERVO_KEYWORDS):
        return "servo"
    if any(k in lower for k in METAL_KEYWORDS):
        return "metal"
    return "plastic"


# UsdPreviewSurface 속성명
PREVIEW_COLOR_ATTR     = "inputs:diffuseColor"
PREVIEW_ROUGHNESS_ATTR = "inputs:roughness"
PREVIEW_METALLIC_ATTR  = "inputs:metallic"

# OmniPBR (MDL) 속성명
MDL_COLOR_ATTRS     = ("inputs:diffuse_color_constant", "inputs:albedo_add",
                       "inputs:base_color")
MDL_ROUGHNESS_ATTRS = ("inputs:reflection_roughness_constant", "inputs:roughness_constant")
MDL_METALLIC_ATTRS  = ("inputs:metallic_constant",)


def _first_valid(prim, names):
    """유효한 첫 번째 속성과 그 값을 반환."""
    for name in names:
        attr = prim.GetAttribute(name)
        if attr.IsValid() and attr.Get() is not None:
            return attr
    return None


def patch(stage: Usd.Stage) -> int:
    modified = 0
    sys.stderr.write(f"\n[패치 시작] {USD_PATH}\n")
    sys.stderr.write("── 재료 목록 ────────────────────────────────────────────────\n")

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        path_str = str(prim.GetPath())

        # UsdPreviewSurface 시도
        dc_attr = prim.GetAttribute(PREVIEW_COLOR_ATTR)
        roughness_attr = prim.GetAttribute(PREVIEW_ROUGHNESS_ATTR)
        metallic_attr  = prim.GetAttribute(PREVIEW_METALLIC_ATTR)

        # MDL fallback
        if not (dc_attr.IsValid() and dc_attr.Get() is not None):
            dc_attr       = _first_valid(prim, MDL_COLOR_ATTRS)
            roughness_attr = _first_valid(prim, MDL_ROUGHNESS_ATTRS)
            metallic_attr  = _first_valid(prim, MDL_METALLIC_ATTRS)

        if dc_attr is None:
            # 색상 속성이 없는 Shader면 이름만 기록
            sys.stderr.write(f"  [skip   ] {path_str} (색상 속성 없음)\n")
            continue

        cls = classify(path_str)
        if cls == "servo":
            new_color, new_r, new_m = SERVO_DARK, ROUGHNESS_SERVO, METALLIC_SERVO
        elif cls == "metal":
            new_color, new_r, new_m = METAL_SILVER, ROUGHNESS_METAL, METALLIC_METAL
        else:
            new_color, new_r, new_m = PLASTIC_PURPLE, ROUGHNESS_PLASTIC, METALLIC_PLASTIC

        old = dc_attr.Get()
        old_str = f"({old[0]:.2f},{old[1]:.2f},{old[2]:.2f})" if hasattr(old, '__iter__') else str(old)
        new_str = f"({new_color[0]:.2f},{new_color[1]:.2f},{new_color[2]:.2f})"

        dc_attr.Set(new_color)
        if roughness_attr and roughness_attr.IsValid():
            roughness_attr.Set(new_r)
        if metallic_attr and metallic_attr.IsValid():
            metallic_attr.Set(new_m)

        sys.stderr.write(f"  [{cls:7s}] {path_str}\n")
        sys.stderr.write(f"    {old_str}  ->  {new_str}\n")
        modified += 1

    return modified


stage = Usd.Stage.Open(USD_PATH)
n = patch(stage)

if n == 0:
    sys.stderr.write("\n  [경고] 수정된 Shader 없음 — 모든 Shader 속성을 출력합니다:\n")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Shader":
            attrs = [a.GetName() for a in prim.GetAttributes()]
            sys.stderr.write(f"  {prim.GetPath()}: {attrs}\n")
else:
    stage.GetRootLayer().Save()
    sys.stderr.write(f"\n[완료] {n}개 재료 패치 -> 저장: {USD_PATH}\n")

simulation_app.close()
