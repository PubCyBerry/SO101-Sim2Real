"""실기기 SO-101 비전 — HSV 검출 + top 카메라 픽셀→base XY 호모그래피.

`ros2_ws/.../realdevice/detect_cubes.py`·`detect_gripper.py` 의 HSV 파이프라인을 포팅하고
파란 그릇·그릇 내부 큐브 검출을 추가한다. **입력은 RGB**(LeRobot `OpenCVCamera` 기본 color_mode=RGB,
`get_observation()` 가 RGB 반환) — 내부에서 BGR 로 변환해 기존에 튜닝된 BGR2HSV 임계값을 그대로 쓴다.

좌표 매핑: 평면 책상 가정 하에 `cv2.findHomography` 로 (top 픽셀 ↔ base XY) 단일 H 를 추정한다.
hand-eye 캘리브(`autonomous_collect.py`)가 그리퍼 tip 픽셀↔FK base XY 쌍을 모아 H 를 만든다.

HSV 임계값은 모듈 상단 상수로 노출 — preflight 프레임 확인 후 실조명에 맞게 조정한다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ── HSV 임계값 (BGR2HSV, OpenCV H∈[0,180]) — 실기기 fold top 프레임서 튜닝(2026-06-17) ──
# 회색 큐브: 매우 낮은 채도(S 8-11 실측) + 중간 밝기 + 정사각·원형 판별.
CUBE_S_MAX = 30
CUBE_V_MIN = 110
CUBE_V_MAX = 240
CUBE_AREA_MIN = 300
CUBE_AREA_MAX = 5000
CUBE_ASPECT_MIN = 0.7
CUBE_ASPECT_MAX = 1.4
CUBE_CIRC_MIN = 0.55       # 4πA/P² — 큐브 0.77-0.82, 클러터/엣지 <0.3
# 정적 마스크 off(실기기 fold 뷰는 sim 과 레이아웃 다름) — 모양 판별로 충분
MASK_TOP_ROWS = 0
MASK_BOTTOM_ROW = 480

# 파란 그릇: teal 폼폼(H≈87) 제외 위해 hue 하한 95↑. 폼폼은 fuzzy(circ≈0.13),
# 그릇은 둥글어(circ≈0.4+) → 최대 (circularity 가중) contour 채택(best-effort).
BOWL_HSV_LO = (95, 25, 110)
BOWL_HSV_HI = (128, 255, 255)
BOWL_AREA_MIN = 800
BOWL_CIRC_MIN = 0.25

# 보라 그리퍼 tip
GRIP_HSV_LO = (115, 60, 40)
GRIP_HSV_HI = (165, 255, 255)
GRIP_AREA_MIN = 300


@dataclass
class Blob:
    px: int
    py: int
    area: float


@dataclass
class Bowl:
    px: int
    py: int
    area: float
    radius: int  # 등가 원 반경(px) — 내부 ROI 판정용


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """RGB(get_observation) → BGR(cv2 HSV 파이프라인 입력)."""
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


def _morph(mask: np.ndarray, open_k: int = 3, close_k: int = 7) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return mask


def detect_gray_cubes(rgb: np.ndarray, roi_mask: np.ndarray | None = None,
                      area_min: int = CUBE_AREA_MIN, area_max: int = CUBE_AREA_MAX,
                      apply_static_mask: bool = True) -> list[Blob]:
    """회색 큐브 검출 → 면적 내림차순 Blob 리스트. roi_mask 지정 시 그 영역만."""
    bgr = _to_bgr(rgb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s < CUBE_S_MAX) & (v > CUBE_V_MIN) & (v < CUBE_V_MAX)).astype(np.uint8) * 255
    if apply_static_mask:
        mask[:MASK_TOP_ROWS, :] = 0
        mask[MASK_BOTTOM_ROW:, :] = 0
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    mask = _morph(mask, 3, 7)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Blob] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min or a > area_max:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h else 0.0
        if ar < CUBE_ASPECT_MIN or ar > CUBE_ASPECT_MAX:
            continue
        p = cv2.arcLength(c, True)
        circ = 4.0 * math.pi * a / (p * p) if p else 0.0
        if circ < CUBE_CIRC_MIN:
            continue
        out.append(Blob(px=x + w // 2, py=y + h // 2, area=float(a)))
    out.sort(key=lambda b: b.area, reverse=True)
    return out


def detect_blue_bowl(rgb: np.ndarray) -> Bowl | None:
    """파란 그릇. teal 폼폼(fuzzy, low circ)을 제치고 **가장 둥근** 파란 contour 채택.

    점수 = area * (0.3 + circularity) → 큰데 둥근 것 우선. 폼폼(circ≈0.13)보다
    그릇(circ≈0.4+)이 이김. hue 하한 95 로 teal(H≈87) 1차 배제.
    """
    bgr = _to_bgr(rgb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BOWL_HSV_LO), np.array(BOWL_HSV_HI))
    mask = _morph(mask, 5, 11)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < BOWL_AREA_MIN:
            continue
        p = cv2.arcLength(c, True)
        circ = 4.0 * math.pi * a / (p * p) if p else 0.0
        if circ < BOWL_CIRC_MIN:
            continue
        score = a * (0.3 + circ)
        if score > best_score:
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            best = Bowl(px=cx, py=cy, area=float(a), radius=int(np.sqrt(a / np.pi)))
            best_score = score
    return best


def detect_cubes_in_bowl(rgb: np.ndarray, bowl: Bowl, shrink: float = 0.85) -> list[Blob]:
    """그릇 내부(원형 ROI) 회색 큐브. 깊이로 작아 보이므로 면적 하한 완화."""
    h, w = rgb.shape[:2]
    roi = np.zeros((h, w), np.uint8)
    cv2.circle(roi, (bowl.px, bowl.py), int(bowl.radius * shrink), 255, -1)
    return detect_gray_cubes(rgb, roi_mask=roi, area_min=150, area_max=CUBE_AREA_MAX,
                             apply_static_mask=False)


# ArUco 마커(그리퍼 부착). 저각+all-purple 팔에서 tip 색검출 불가 → ArUco 코너로 robust 검출.
# DICT_4X4_50. OpenCV 4.7+ ArucoDetector API.
_ARUCO_DETECTOR = None


def _aruco_detector():
    global _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        _ARUCO_DETECTOR = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    return _ARUCO_DETECTOR


def detect_aruco(rgb: np.ndarray, marker_id: int | None = None) -> tuple[int, int] | None:
    """ArUco 마커 중심 픽셀(4 코너 평균). marker_id 지정 시 그 id만, 없으면 첫 마커."""
    gray = cv2.cvtColor(_to_bgr(rgb), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector().detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten()
    for i, mid in enumerate(ids):
        if marker_id is None or int(mid) == marker_id:
            c = corners[i][0]  # (4,2)
            cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
            return (int(round(cx)), int(round(cy)))
    return None


def detect_gripper_tip(rgb: np.ndarray) -> tuple[int, int] | None:
    """보라 그리퍼 최대 contour 의 최상단(min y) 점 = tip 픽셀."""
    bgr = _to_bgr(rgb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(GRIP_HSV_LO), np.array(GRIP_HSV_HI))
    mask = _morph(mask, 3, 9)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < GRIP_AREA_MIN:
        return None
    pts = c.reshape(-1, 2)
    tip = pts[int(np.argmin(pts[:, 1]))]
    return (int(tip[0]), int(tip[1]))


# ── 호모그래피 (top 픽셀 ↔ base XY) ──────────────────────────────────────────
def compute_homography(pixels: list[tuple[float, float]],
                       base_xys: list[tuple[float, float]]) -> tuple[np.ndarray | None, list[float]]:
    """findHomography(pixel→base XY) + per-pair 재투영 잔차(m) 반환."""
    if len(pixels) < 4:
        return None, []
    src = np.array(pixels, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(base_xys, dtype=np.float64).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.01)
    if H is None:
        return None, []
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    res = [float(np.hypot(*(proj[i] - np.array(base_xys[i])))) for i in range(len(base_xys))]
    return H, res


def pixel_to_base_xy(H: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float]:
    pt = np.array([[pixel]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return (float(out[0]), float(out[1]))


def order_cubes_left_to_right(cubes: list[Blob], H: np.ndarray,
                              offset_xy: tuple[float, float] = (0.0, 0.0)
                              ) -> list[tuple[Blob, tuple[float, float]]]:
    """검출 큐브에 base XY 를 붙여 **좌→우** 정렬 → [(Blob, (x,y))].

    캘리브 규약상 image-right = base +y → "왼쪽"(영상 왼쪽) = base -y.
    따라서 base y 오름차순 = 좌→우. 스택 SM 이 픽 순서·바닥(최우측=last) 선택에 사용.
    """
    out: list[tuple[Blob, tuple[float, float]]] = []
    for b in cubes:
        x, y = pixel_to_base_xy(H, (b.px, b.py))
        out.append((b, (x + offset_xy[0], y + offset_xy[1])))
    out.sort(key=lambda t: t[1][1])
    return out


def save_homography(path: Path, H: np.ndarray, pairs: list, residuals: list[float],
                    extra: dict | None = None) -> None:
    data = {
        "homography_matrix": H.tolist() if H is not None else None,
        "num_pairs": len(pairs),
        "residual_rms_m": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
        "residual_max_m": float(np.max(residuals)) if residuals else None,
        "pairs": pairs,
        "residuals_m": residuals,
    }
    if extra:
        data.update(extra)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_homography(path: Path) -> np.ndarray | None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    h = data.get("homography_matrix")
    return np.array(h, dtype=np.float64) if h is not None else None


# ── 디버그 오버레이 (Claude 가 Read 로 감사) ─────────────────────────────────
def save_debug_overlay(path: Path, rgb: np.ndarray, *, cubes: list[Blob] | None = None,
                       bowl: Bowl | None = None, gripper_tip: tuple[int, int] | None = None,
                       label: str | None = None) -> None:
    """RGB 프레임에 검출 결과를 그려 BGR PNG 로 저장(cv2.imwrite=BGR)."""
    img = _to_bgr(rgb).copy()
    if bowl is not None:
        cv2.circle(img, (bowl.px, bowl.py), bowl.radius, (255, 0, 0), 2)
        cv2.circle(img, (bowl.px, bowl.py), 3, (255, 0, 0), -1)
    for b in (cubes or []):
        cv2.rectangle(img, (b.px - 14, b.py - 14), (b.px + 14, b.py + 14), (0, 255, 0), 2)
        cv2.putText(img, f"{int(b.area)}", (b.px - 14, b.py - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if gripper_tip is not None:
        cv2.circle(img, gripper_tip, 6, (0, 0, 255), -1)
    if label:
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def mean_brightness(rgb: np.ndarray) -> tuple[float, float]:
    """HSV V 채널 평균·중앙값 (조명 밝기 판정용). bright≈120 / dark≈25."""
    v = cv2.cvtColor(_to_bgr(rgb), cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v.mean()), float(np.median(v))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    """RGB 프레임을 색 정확히(BGR 변환 후) PNG 저장."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _to_bgr(rgb))
