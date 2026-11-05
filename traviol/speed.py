"""Vehicle speed from two frames (Python 3 port of original_2018/trorb.py).

ORB (Oriented FAST and Rotated BRIEF) finds distinctive corners on the car in
both frames and matches them by their binary descriptors. The typical shift of
the matched points, converted to metres and divided by the time between the
frames, is the speed:

    t = n_frames / fps          speed = distance / t

The 2018 version averaged the pixel shift in the raw image. This version takes
the median (robust to bad matches) and, when a BirdsEye transform is given,
measures the shift in the top-down view, where one pixel is the same distance
everywhere on the road.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import BirdsEye


@dataclass
class SpeedEstimate:
    kmh: float
    metres: float
    seconds: float
    shift_px: float
    matches: int
    points_a: np.ndarray  # matched keypoints in frame A (image coordinates)
    points_b: np.ndarray


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def estimate_speed(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    box_a: tuple[int, int, int, int],
    n_frames: int,
    fps: float,
    metres_per_px: float | None = None,
    birds_eye: BirdsEye | None = None,
    max_matches: int = 60,
) -> SpeedEstimate | None:
    """Estimate a vehicle's speed between two frames n_frames apart.

    box_a: the vehicle's box in frame A; keypoints are taken only inside it.
    metres_per_px: calibration in the raw image, or None to use birds_eye.metres_per_px.
    """
    orb = cv2.ORB_create(nfeatures=5000, scoreType=cv2.ORB_FAST_SCORE)
    x, y, w, h = box_a
    mask = np.zeros(frame_a.shape[:2], np.uint8)
    mask[y : y + h, x : x + w] = 255
    kp_a, des_a = orb.detectAndCompute(_gray(frame_a), mask)
    kp_b, des_b = orb.detectAndCompute(_gray(frame_b), None)
    if des_a is None or des_b is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)[:max_matches]
    if len(matches) < 4:
        return None

    pa = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pb = np.float32([kp_b[m.trainIdx].pt for m in matches])

    if birds_eye is not None:
        qa, qb = birds_eye.points(pa), birds_eye.points(pb)
        scale = metres_per_px if metres_per_px else birds_eye.metres_per_px
    else:
        qa, qb = pa, pb
        scale = metres_per_px
    if not scale:
        raise ValueError("pass metres_per_px, or a BirdsEye with metres_long set")

    shifts = np.linalg.norm(qb - qa, axis=1)
    # Keep the consistent majority: a car moves rigidly, so outlier matches stand out.
    med = np.median(shifts)
    good = np.abs(shifts - med) <= max(2.0, 0.25 * med)
    shift = float(np.median(shifts[good]))

    seconds = n_frames / fps
    metres = shift * scale
    return SpeedEstimate(
        kmh=metres / seconds * 3.6,
        metres=metres,
        seconds=seconds,
        shift_px=shift,
        matches=int(good.sum()),
        points_a=pa[good],
        points_b=pb[good],
    )
