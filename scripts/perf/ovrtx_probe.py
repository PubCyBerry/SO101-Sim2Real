"""ovrtx P0 게이트 probe (Track A).

검증 목표 (실패 시 Track A 중단 또는 USD flatten 우회 평가):
  게이트: cube_desk scene.usd (payload 참조로 큐브4·그릇 구성) 가 ovrtx 로 로드·렌더되어
          RGB PNG 1장이 나오는가.

실행:  .venv-ovrtx/bin/python scripts/perf/ovrtx_probe.py
출력:  docs/ovrtx_poc.png

전용 .venv-ovrtx 로 실행 (핀 환경 격리).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import ovrtx
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_USD = os.path.join(_REPO, "assets", "scenes", "cube_desk", "scene.usd")
OUT_PNG = os.path.join(_REPO, "docs", "ovrtx_poc.png")

# cube_desk world bbox center(2.2,-0.21,1.26) size(5,4,2.5) — usd-core 측정.
# 카메라를 bbox 밖(앞·위)에 두고 책상면(z~0.75) 조준.
EYE = np.array([2.2, -4.8, 2.2])
LOOKAT = np.array([2.0, -0.3, 0.78])
UP = np.array([0.0, 0.0, 1.0])
RES = (1280, 720)


def lookat_matrix(eye, target, up):
    """USD 카메라(-Z 주시, row-major point*M) world transform 4x4."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    upc = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3] = right        # x
    m[1, :3] = upc          # y
    m[2, :3] = -fwd         # z (카메라는 -Z 를 봄)
    m[3, :3] = eye          # translation
    return m


def usda_matrix(m):
    return "( " + ", ".join(
        "(" + ", ".join(f"{v:.8f}" for v in row) + ")" for row in m
    ) + " )"


def main() -> None:
    print("=" * 70)
    print(f"[probe] ovrtx {ovrtx.__version__}")
    print(f"[probe] scene: {SCENE_USD}")

    cam_m = lookat_matrix(EYE, LOOKAT, UP)
    root_usda = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1.0
)

def Xform "World"
{{
    def "Scene" (
        prepend references = @{SCENE_USD}@
    )
    {{
    }}

    def Camera "Camera"
    {{
        float focalLength = 18.0
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        float2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = {usda_matrix(cam_m)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DomeLight "DomeLight"
    {{
        float inputs:intensity = 1000
    }}

    def DistantLight "KeyLight"
    {{
        float inputs:intensity = 4000
        float inputs:angle = 1.0
        matrix4d xformOp:transform = ( (1,0,0,0),(0,0.7071,0.7071,0),(0,-0.7071,0.7071,0),(0,0,0,1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
}}

def "Render"
{{
    def RenderProduct "Camera"
    {{
        int2 resolution = {RES}
        rel camera = </World/Camera>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
}}
"""

    print("[probe] Renderer 생성 (첫 실행 shader 컴파일 수십초)...", file=sys.stderr)
    renderer = ovrtx.Renderer()
    print("[probe] open_usd_from_string (scene 참조 + 카메라 + RenderProduct)...")
    renderer.open_usd_from_string(root_usda)

    print("[probe] warmup 40 step...")
    for _ in range(40):
        renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

    print("[probe] 최종 렌더 step...")
    products = renderer.step(render_products={"/Render/Camera"}, delta_time=1.0 / 60)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    saved = False
    for _name, product in products.items():
        for frame in product.frames:
            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            print(f"[probe] frame pixels shape={pixels.shape} dtype={pixels.dtype}")
            Image.fromarray(pixels).save(OUT_PNG)
            saved = True

    if saved:
        print(f"[probe] ✓✓ 게이트 PASS — cube_desk scene.usd 렌더됨 → {OUT_PNG}")
    else:
        print("[probe] ❌ 게이트 실패 — 렌더 출력 없음.")
    print("=" * 70)


if __name__ == "__main__":
    main()
