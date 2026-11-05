"""Bird's-eye view of the road (the 2018 `warpred.py` step).

A camera looks at the road at an angle, so equal distances on the road cover
fewer pixels the further away they are. Picking the four corners of a stretch
of road that is rectangular in reality (top-left, top-right, bottom-right,
bottom-left, the order used in 2018) gives a perspective transform to a
top-down view, where pixel distances are uniform and a stop line is a
horizontal row.
"""

from __future__ import annotations

import cv2
import numpy as np


class BirdsEye:
    def __init__(self, corners, out_size: tuple[int, int] = (400, 800), metres_long: float | None = None):
        """corners: four (x, y) image points TL, TR, BR, BL of a real-world rectangle on the road.
        out_size: (width, height) of the bird's-eye image.
        metres_long: real length of that rectangle along the road, for metric calibration.
        """
        w, h = out_size
        self.size = out_size
        src = np.asarray(corners, dtype=np.float32)
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.metres_per_px = (metres_long / h) if metres_long else None

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.H, self.size)

    def points(self, pts) -> np.ndarray:
        """Map image points (N×2) into bird's-eye coordinates."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
