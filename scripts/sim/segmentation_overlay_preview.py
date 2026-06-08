"""TC.3 segmentation-style background overlay preview.

The production idea is Squint-style compositing: keep simulated foreground
(robot, pen, cup) and replace the broad simulated background with real camera
backgrounds. This preview tool is deliberately lightweight: it can operate on
PNG reference frames already extracted during TA.3, and it also accepts LeRobot
dataset videos when imageio is available.

Outputs per camera:
  - foreground mask PNG
  - composite PNG
  - contact sheet: real | sim | mask | overlay
  - overlay_summary.json with simple mask/appearance metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CAMERAS = ("top", "wrist", "front")
DEFAULT_SIM_DIR = Path("outputs/ta3_camera_renders/opus_fix5")
DEFAULT_REAL_DIR = Path("outputs/ta3_camera_refs")
DEFAULT_OUTPUT_DIR = Path("outputs/tc3_segmentation_overlay_preview")
MIN_FOREGROUND_BY_CAMERA = {
    "top": 0.06,
    # The wrist camera often sees a black cup on a black mat. Color-only
    # masking collapses there, so require a larger ratio before trusting it.
    "wrist": 0.12,
    "front": 0.10,
}
ROI_PRESETS = {
    "top": (0.52, 0.55, 0.43, 0.42),
    "wrist": (0.50, 0.54, 0.45, 0.42),
    "front": (0.50, 0.54, 0.45, 0.42),
}


def _read_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _read_video_frame(path: Path, frame_index: int) -> Image.Image:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Video frame extraction requires imageio") from exc

    reader = imageio.get_reader(path)
    try:
        frame = reader.get_data(frame_index)
    finally:
        reader.close()
    return Image.fromarray(frame[..., :3]).convert("RGB")


def _video_path(dataset_root: Path, camera: str) -> Path:
    path = dataset_root / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_frame(
    *,
    camera: str,
    image_dir: Path | None,
    dataset_root: Path | None,
    image_pattern: str,
    frame_index: int,
) -> Image.Image:
    if dataset_root is not None:
        return _read_video_frame(_video_path(dataset_root, camera), frame_index)
    if image_dir is None:
        raise ValueError("image_dir or dataset_root must be provided")
    return _read_rgb(image_dir / image_pattern.format(camera=camera))


def _dominant_palette(rgb: np.ndarray, border: int, bins: int, max_colors: int) -> np.ndarray:
    """Return coarse dominant RGB palette from border + full frame samples."""
    h, w, _ = rgb.shape
    border_pixels = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[h - border :].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, w - border :].reshape(-1, 3),
        ],
        axis=0,
    )
    # Full-frame dominant colors catch big tabletop/mat regions that do not touch every border.
    stride = max(1, min(h, w) // 160)
    sampled = rgb[::stride, ::stride].reshape(-1, 3)
    pixels = np.concatenate([border_pixels, sampled], axis=0)

    quant = (pixels // bins) * bins + bins // 2
    colors, counts = np.unique(quant, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:max_colors]
    return colors[order].astype(np.float32)


def _roi_mask(camera: str, size: tuple[int, int], blur_radius: float) -> Image.Image:
    w, h = size
    cx_s, cy_s, rx_s, ry_s = ROI_PRESETS.get(camera, ROI_PRESETS["top"])
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * cx_s, h * cy_s
    rx, ry = w * rx_s, h * ry_s
    ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1.0
    mask = Image.fromarray(ellipse.astype(np.uint8) * 255)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def _foreground_mask(
    camera: str,
    image: Image.Image,
    *,
    border: int,
    bins: int,
    bg_distance: float,
    min_foreground: float,
    max_foreground: float,
    blur_radius: float,
) -> tuple[Image.Image, str]:
    """Build a segmentation proxy mask from dominant simulated background colors.

    This is not a semantic annotator. It is a deterministic preview mask whose
    behavior can be visually checked before investing in a heavier AOV path.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    palette = _dominant_palette(rgb, border=border, bins=bins, max_colors=10)
    diff = rgb.astype(np.float32)[..., None, :] - palette[None, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)).min(axis=-1)

    background = dist < bg_distance
    mask = (~background).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)

    # Clean isolated pixels and soften edges for compositing.
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
    if blur_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    ratio = float((mask_arr > 0.5).mean())
    min_ratio = max(min_foreground, MIN_FOREGROUND_BY_CAMERA.get(camera, min_foreground))
    # Keep the preview honest: if the heuristic collapses, fall back to a camera
    # ROI matte and record that choice in overlay_summary.json.
    if ratio < min_ratio or ratio > max_foreground:
        return _roi_mask(camera, image.size, blur_radius), "roi_fallback"

    return mask_img, "color"


def _composite(sim: Image.Image, real: Image.Image, mask: Image.Image) -> Image.Image:
    real = real.resize(sim.size, Image.Resampling.BILINEAR)
    return Image.composite(sim, real, mask)


def _label_panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, panel.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255), font=font)
    return panel


def _contact_sheet(real: Image.Image, sim: Image.Image, mask: Image.Image, overlay: Image.Image) -> Image.Image:
    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    panels = [
        _label_panel(real.resize(sim.size, Image.Resampling.BILINEAR), "real background"),
        _label_panel(sim, "sim foreground"),
        _label_panel(mask_rgb, "foreground mask"),
        _label_panel(overlay, "overlay"),
    ]
    sheet = Image.new("RGB", (sim.width * 4, sim.height), color=(0, 0, 0))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (idx * sim.width, 0))
    return sheet


def _metrics(sim: Image.Image, real: Image.Image, overlay: Image.Image, mask: Image.Image) -> dict[str, Any]:
    sim_arr = np.asarray(sim, dtype=np.float32)
    real_arr = np.asarray(real.resize(sim.size, Image.Resampling.BILINEAR), dtype=np.float32)
    overlay_arr = np.asarray(overlay, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    fg = mask_arr > 0.5
    bg = ~fg
    bg_delta = float(np.mean(np.abs(overlay_arr[bg] - real_arr[bg]))) if bg.any() else 0.0
    fg_delta = float(np.mean(np.abs(overlay_arr[fg] - sim_arr[fg]))) if fg.any() else 0.0
    return {
        "foreground_ratio": round(float(fg.mean()), 4),
        "background_l1_to_real": round(bg_delta, 3),
        "foreground_l1_to_sim": round(fg_delta, 3),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="TC.3 segmentation-style overlay preview")
    parser.add_argument("--sim_dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--real_dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--sim_dataset", type=Path, default=None)
    parser.add_argument("--real_dataset", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sim_image_pattern", default="{camera}_camera.png")
    parser.add_argument("--real_image_pattern", default="{camera}_t001.png")
    parser.add_argument("--sim_frame_index", type=int, default=30)
    parser.add_argument("--real_frame_index", type=int, default=30)
    parser.add_argument("--border", type=int, default=24)
    parser.add_argument("--color_bins", type=int, default=32)
    parser.add_argument("--bg_distance", type=float, default=24.0)
    parser.add_argument("--min_foreground", type=float, default=0.03)
    parser.add_argument("--max_foreground", type=float, default=0.75)
    parser.add_argument("--blur_radius", type=float, default=1.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "task_id": "TC.3",
        "status": "passed",
        "output_dir": str(args.output_dir.resolve()),
        "notes": [
            "This is a deterministic foreground-mask preview, not a true Isaac SemanticSegmentation AOV.",
            "Use it to inspect camera-specific background replacement before production overlay.",
        ],
        "cameras": {},
    }

    for camera in CAMERAS:
        sim = _resolve_frame(
            camera=camera,
            image_dir=args.sim_dir,
            dataset_root=args.sim_dataset,
            image_pattern=args.sim_image_pattern,
            frame_index=args.sim_frame_index,
        )
        real = _resolve_frame(
            camera=camera,
            image_dir=args.real_dir,
            dataset_root=args.real_dataset,
            image_pattern=args.real_image_pattern,
            frame_index=args.real_frame_index,
        )
        if sim.size != (640, 480):
            sim = sim.resize((640, 480), Image.Resampling.BILINEAR)
        if real.size != (640, 480):
            real = real.resize((640, 480), Image.Resampling.BILINEAR)

        mask, mask_source = _foreground_mask(
            camera,
            sim,
            border=args.border,
            bins=args.color_bins,
            bg_distance=args.bg_distance,
            min_foreground=args.min_foreground,
            max_foreground=args.max_foreground,
            blur_radius=args.blur_radius,
        )
        overlay = _composite(sim, real, mask)
        sheet = _contact_sheet(real, sim, mask, overlay)

        mask_path = args.output_dir / f"{camera}_mask.png"
        overlay_path = args.output_dir / f"{camera}_overlay.png"
        sheet_path = args.output_dir / f"{camera}_contact.png"
        mask.save(mask_path)
        overlay.save(overlay_path)
        sheet.save(sheet_path)

        summary["cameras"][camera] = {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "contact_sheet": str(sheet_path),
            "mask_source": mask_source,
            "metrics": _metrics(sim, real, overlay, mask),
        }

    _write_json(args.output_dir / "overlay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
