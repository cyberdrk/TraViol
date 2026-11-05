"""Licence plate localisation (Python 3 / OpenCV 4+ port of original_2018/pyANPD.py).

Plates are high-contrast, horizontally textured rectangles, so the frame is
reduced to vertical edges (Sobel in x), binarised with Otsu's threshold, and
closed with a wide rectangular kernel so each plate's characters merge into one
blob. Blobs are kept if their rotated bounding box has a plate-like aspect
ratio, area and tilt, and if enough of the box is edge pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Plate geometry presets: (kernel size, aspect-ratio range, area range in px²)
PRESETS = {
    "rect": ((17, 4), (2.2, 12.0), (500, 18000)),  # long single-row plates
    "square": ((7, 6), (1.0, 2.0), (300, 8000)),  # two-row plates
}


@dataclass
class Plate:
    box: np.ndarray  # 4×2 corner points of the rotated rectangle
    bbox: tuple[int, int, int, int]  # x, y, w, h (axis-aligned)
    angle: float
    edge_density: float
    crop: np.ndarray  # de-rotated colour crop of the plate


def _edge_map(gray: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:
    sharpen = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]], dtype=np.float32)
    g = cv2.filter2D(gray, -1, sharpen)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    g = cv2.Sobel(g, -1, 1, 0)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)


def _plausible(rect, img_w: int, aspect_range, area_range) -> bool:
    (_, _), (w, h), _ = rect
    if w <= 0 or h <= 0 or w >= img_w / 2 or h >= img_w / 2:
        return False
    aspect = max(w, h) / min(w, h)
    if not (aspect_range[0] <= aspect <= aspect_range[1]):
        return False
    if not (area_range[0] < w * h < area_range[1]):
        return False
    # Reject boxes tilted more than 45° from horizontal.
    box = cv2.boxPoints(rect)
    p0 = box[0]
    others = sorted(box[1:], key=lambda p: (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2)
    long_side = others[1]
    dx, dy = abs(long_side[0] - p0[0]), abs(long_side[1] - p0[1])
    tilt = 90.0 if dx == 0 else math.degrees(math.atan(dy / dx))
    return tilt <= 45


def _derotate(img: np.ndarray, rect) -> np.ndarray:
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (int(round(w)), int(round(h))), (cx, cy))


def find_plates(image: np.ndarray, kind: str = "rect", min_edge_density: float = 0.45) -> list[Plate]:
    """Return plate candidates in a BGR frame, best (densest edges) first."""
    kernel, aspect_range, area_range = PRESETS[kind]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = _edge_map(gray, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    plates: list[Plate] = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        if not _plausible(rect, image.shape[1], aspect_range, area_range):
            continue
        mask_crop = _derotate(edges, rect)
        if mask_crop.size == 0:
            continue
        _, mask_crop = cv2.threshold(mask_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        density = float((mask_crop == 255).mean())
        if density < min_edge_density:
            continue
        box = cv2.boxPoints(rect).astype(int)
        x, y, w, h = cv2.boundingRect(box)
        plates.append(Plate(box=box, bbox=(x, y, w, h), angle=rect[2], edge_density=density,
                            crop=_derotate(image, rect)))
    plates.sort(key=lambda p: p.edge_density, reverse=True)
    return plates


def draw_plates(image: np.ndarray, plates: list[Plate]) -> np.ndarray:
    out = image.copy()
    for p in plates:
        cv2.drawContours(out, [p.box], 0, (127, 0, 255), 2)
    return out
