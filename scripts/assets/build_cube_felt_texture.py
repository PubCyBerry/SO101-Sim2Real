"""큐브용 회색 펠트(보풀) 텍스처 생성 — albedo + tangent-space normal map.

실물 큐브(docs/pics/cube_desk/큐브와 그릇.jpg)는 회색 테리/펠트 천으로 감싼 폼 큐브다.
사진에서 마스킹한 펠트 평균색(약간 쿨한 회색, sRGB ~0.58)을 base 로 잡고, 저주파 얼룩 +
고주파 보풀 grain 을 얹어 펠트 표면을 절차적으로 합성한다. 노이즈는 FFT 기반(주기적)이라
타일링 시 이음매가 없다 — UsdUVTexture wrap=repeat 로 박스 큐브 면에 반복 매핑한다.

실행 (프로젝트 .venv, cv2+numpy 필요):
    .venv/bin/python scripts/assets/build_cube_felt_texture.py

생성물:
    assets/scenes/cube_desk/textures/cube_felt_albedo.png   (sRGB 8bit)
    assets/scenes/cube_desk/textures/cube_felt_normal.png   (raw 8bit, [0,1] tangent-space)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "cube_desk" / "textures"
N = 1024
SEED = 7

# 사진에서 마스킹한 펠트 평균 sRGB(약간 쿨한 회색). 실물은 균일한 밝은 회색 펠트라
# 저주파 얼룩은 약하게(돌처럼 보이지 않게) 두고 고주파 보풀(nap)을 살린다.
BASE_RGB = np.array([162.0, 164.0, 168.0])  # R,G,B — 살짝 쿨(blue 우세), 밝은 회색


def _periodic_noise(n: int, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """FFT 가우시안 저역통과 백색잡음 → 주기적(타일링) 노이즈. 단위분산 정규화."""
    white = rng.standard_normal((n, n))
    fx = np.fft.fftfreq(n)
    fxx, fyy = np.meshgrid(fx, fx)
    gauss = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fxx ** 2 + fyy ** 2))
    out = np.fft.ifft2(np.fft.fft2(white) * gauss).real
    return (out - out.mean()) / (out.std() + 1e-9)


def main() -> None:
    rng = np.random.default_rng(SEED)
    mottle = _periodic_noise(N, sigma_px=42.0, rng=rng)   # 저주파 얼룩(천 톤 불균일)
    fiber = _periodic_noise(N, sigma_px=0.9, rng=rng)      # 고주파 보풀 grain(nap)

    # ── albedo (sRGB) ───────────────────────────────────────────────────────
    alb = BASE_RGB[None, None, :].astype(np.float32) + np.zeros((N, N, 3), np.float32)
    alb = alb + mottle[..., None] * 4.0      # ±~4 톤 얼룩(약 — 균일 펠트)
    alb = alb + fiber[..., None] * 16.0       # ±~16 보풀 명암(주 — nap)

    # 흰 시접 무늬 — 전개도 밴드 컬럼(u<0.5)에 길쭉한 둥근사각 윤곽선. UV(u,v) 매핑은
    #   author_pick_cube_scene._net_uv 와 일치(앞 v[0,.25]·윗[.25,.5]·뒤[.5,.75]).
    #   v∈[0.06,0.69] → 앞·윗·뒤 3면 걸침. rounded-rect SDF 의 |·|<stroke 가 윤곽선.
    yy, xx = np.mgrid[0:N, 0:N]
    u = (xx + 0.5) / N
    v = (yy + 0.5) / N
    cu, cv = 0.25, 0.375            # 밴드 중심(u=0.25 면 Y중앙, v=0.375 윗면 중심)
    hu, hv = 0.205, 0.345           # 반폭 — 긴 변을 면 Y-모서리(u=0/0.5) 가까이, 약간 더 길게
    corner, stroke = 0.06, 0.013
    qu = np.abs(u - cu) - (hu - corner)
    qv = np.abs(v - cv) - (hv - corner)
    sdf = (np.sqrt(np.maximum(qu, 0) ** 2 + np.maximum(qv, 0) ** 2)
           + np.minimum(np.maximum(qu, qv), 0.0) - corner)
    edge = np.clip(1.0 - (np.abs(sdf) - stroke * 0.5) / (1.5 / N), 0.0, 1.0)  # 안티앨리어스
    edge = np.where(u < 0.5, edge, 0.0)[..., None]
    seam_rgb = np.array([238.0, 238.0, 232.0])  # 약간 크림빛 흰색
    alb = alb * (1.0 - edge) + seam_rgb[None, None, :] * edge

    alb = np.clip(alb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "cube_felt_albedo.png"), alb[..., ::-1])  # RGB→BGR

    # ── normal map (tangent-space, [0,1]) ────────────────────────────────────
    # 높이 = 보풀(주). 저주파 얼룩은 거의 빼 미세 nap 요철만 남긴다.
    height = fiber * 1.0 + mottle * 0.04
    gx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    gy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    strength = 1.3
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    nrm = np.stack([nx, ny, nz], axis=-1)
    nrm_rgb = np.clip((nrm * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)  # R=x,G=y,B=z
    cv2.imwrite(str(OUT_DIR / "cube_felt_normal.png"), nrm_rgb[..., ::-1])  # RGB→BGR

    print(f"[INFO]: wrote {OUT_DIR/'cube_felt_albedo.png'}  ({N}x{N})")
    print(f"[INFO]: wrote {OUT_DIR/'cube_felt_normal.png'}  ({N}x{N})")
    print(f"[INFO]: albedo mean sRGB rgb = {alb.reshape(-1,3).mean(0).round(1)}")


if __name__ == "__main__":
    main()
