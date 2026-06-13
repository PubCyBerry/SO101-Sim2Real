"""ovrtx P1 게이트 probe (Track A) — 3카메라 멀티뷰 렌더 throughput 측정.

검증 목표:
  P1 게이트: cube_desk scene.usd 를 ovrtx 로 로드한 후,
             3 카메라(top/wrist/front) 를 동시에 720p 로 렌더하고
             warmup 후 단독 렌더 throughput(FPS) 를 실측한다.
             (로봇은 미렌더 — 고정 world frame 카메라만)

실행:  .venv-ovrtx/bin/python scripts/perf/ovrtx_render_layer_probe.py
출력:  docs/ovrtx_cam_{top,wrist,front}.png + 성능 로그

전용 .venv-ovrtx 로 실행 (핀 환경 격리).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import ovrtx
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_USD = os.path.join(_REPO, "assets", "scenes", "cube_desk", "scene.usd")
OUT_DIR = os.path.join(_REPO, "docs")

# ─────────────────────────────────────────────────────────────────────────────
# 카메라 설정 (pick_cube_env_cfg.py::make_pick_cube_camera_cfgs 기준)
# ─────────────────────────────────────────────────────────────────────────────

# top: world frame 절대 좌표 (로봇 뒤, 높은 곳에서 내려보는 급경사 oblique)
#      pos=(1.87,-0.58,1.72), rot quat wxyz=(0.5716,-0.4238,0.4466,0.5424), focal=23
TOP_POS = np.array([1.87, -0.58, 1.72])
TOP_TARGET = np.array([2.14, -0.15, 0.76])  # lookat 계산용 (fallback rot 대신)
TOP_FOCAL = 23.0

# wrist: gripper 링크 자식 prim — P1 에선 로봇 미렌더이므로 대신 고정 world frame 카메라 배치
#        (로봇 자식 prim 으로 두면 로봇이 없어 렌더 불가)
#        gripper 기준 local pos(0.0,0.045,-0.04), rot quat wxyz=(-0.3642,0.6061,-0.6061,-0.3642)
#        → 추정 world pos(wrist 링크가 gripper_frame_link 근처에 있으므로) (1.82, 0.15, 0.72)
#        (대체로 top 근처이지만 악수 방향)
WRIST_POS = np.array([1.82, 0.15, 0.72])
WRIST_TARGET = np.array([2.0, -0.2, 0.75])
WRIST_FOCAL = 23.0

# front: shoulder 링크 자식 — 로봇 미렌더이므로 고정 world frame
#        shoulder local pos(0.050,0.0,0.0), rot quat wxyz=(0.0,0.0,1.0,0.0)
#        → world pos (1.81,-0.57,0.75) (shoulder_pan=0 기록용)
FRONT_POS = np.array([1.81, -0.57, 0.75])
FRONT_TARGET = np.array([2.0, -0.2, 0.7])
FRONT_FOCAL = 23.0

# 해상도 (720p 기준)
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
    print("=" * 80)
    print(f"[ovrtx-p1] ovrtx {ovrtx.__version__}")
    print(f"[ovrtx-p1] scene: {SCENE_USD}")
    print(f"[ovrtx-p1] target: 3-camera (top/wrist/front) 동시 720p 렌더 throughput")

    up = np.array([0.0, 0.0, 1.0])

    # 각 카메라 lookat matrix 계산
    top_m = lookat_matrix(TOP_POS, TOP_TARGET, up)
    wrist_m = lookat_matrix(WRIST_POS, WRIST_TARGET, up)
    front_m = lookat_matrix(FRONT_POS, FRONT_TARGET, up)

    # render layer USDA 구성 (scene 참조 + 3 카메라 + RenderProduct 3개)
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

    def Camera "TopCamera"
    {{
        float focalLength = {TOP_FOCAL}
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        float2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = {usda_matrix(top_m)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def Camera "WristCamera"
    {{
        float focalLength = {WRIST_FOCAL}
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        float2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = {usda_matrix(wrist_m)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def Camera "FrontCamera"
    {{
        float focalLength = {FRONT_FOCAL}
        float horizontalAperture = 20.955
        float verticalAperture = 11.787
        float2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = {usda_matrix(front_m)}
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
    def RenderProduct "Top"
    {{
        int2 resolution = {RES}
        rel camera = </World/TopCamera>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}

    def RenderProduct "Wrist"
    {{
        int2 resolution = {RES}
        rel camera = </World/WristCamera>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}

    def RenderProduct "Front"
    {{
        int2 resolution = {RES}
        rel camera = </World/FrontCamera>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
}}
"""

    print("[ovrtx-p1] Renderer 생성 (첫 실행 shader 컴파일 수십초)...", file=sys.stderr)
    renderer = ovrtx.Renderer()

    print("[ovrtx-p1] open_usd_from_string (scene 참조 + 3 카메라 + 3 RenderProduct)...")
    renderer.open_usd_from_string(root_usda)

    print("[ovrtx-p1] warmup 40 step (카메라별)...", file=sys.stderr)
    for step_idx in range(40):
        if step_idx % 10 == 0:
            print(f"  warmup step {step_idx}/40", file=sys.stderr)
        renderer.step(
            render_products={"/Render/Top", "/Render/Wrist", "/Render/Front"},
            delta_time=1.0 / 60
        )

    print("[ovrtx-p1] 최종 렌더 (3 카메라 PNG 저장)...", file=sys.stderr)
    products = renderer.step(
        render_products={"/Render/Top", "/Render/Wrist", "/Render/Front"},
        delta_time=1.0 / 60
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    cam_names = {"Top": "top", "Wrist": "wrist", "Front": "front"}

    for prod_key, prod in products.items():
        # prod_key = "/Render/Top" → "Top"
        key = prod_key.split("/")[-1]
        cam_name = cam_names.get(key, key.lower())

        for frame in prod.frames:
            var = frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            out_path = os.path.join(OUT_DIR, f"ovrtx_cam_{cam_name}.png")
            Image.fromarray(pixels).save(out_path)
            print(f"[ovrtx-p1] {cam_name} camera: {out_path} (shape={pixels.shape}, dtype={pixels.dtype})")

    # ─────────────────────────────────────────────────────────────────────────
    # 성능 측정: warmup 이후 100 step 동안 wall-clock 측정
    # ─────────────────────────────────────────────────────────────────────────

    print("[ovrtx-p1] 성능 측정: 100 step 동시 렌더 wall-clock...", file=sys.stderr)
    n_steps = 100

    t0 = time.perf_counter()
    for step_idx in range(n_steps):
        if step_idx % 20 == 0:
            print(f"  perf step {step_idx}/{n_steps}", file=sys.stderr)
        renderer.step(
            render_products={"/Render/Top", "/Render/Wrist", "/Render/Front"},
            delta_time=1.0 / 60
        )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = n_steps / elapsed

    print("\n" + "=" * 80)
    print("[ovrtx-p1] ✓✓ P1 측정 완료")
    print(f"  벽시계: {elapsed:.3f} sec for {n_steps} step (3 camera simultaneous)")
    print(f"  쓰루풋: {fps:.2f} FPS (3 카메라 동시, 각 1280×720 RGBA)")
    print(f"  프레임당: {elapsed/n_steps*1000:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
