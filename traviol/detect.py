"""Moving-vehicle detection (Python 3 port of original_2018/km.py).

A Mixture-of-Gaussians background model learns the empty road, so anything
that moves shows up in the foreground mask. Erosion removes speckle, dilation
fills each car into one blob, and every blob above a minimum area becomes a
vehicle box. The 2018 script took the centroid of the whole mask (one vehicle
per frame); this version boxes each blob separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Vehicle:
    box: tuple[int, int, int, int]  # x, y, w, h

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    @property
    def front(self) -> tuple[float, float]:
        """Bottom-centre of the box: the point that crosses a stop line first
        for traffic moving towards the camera."""
        x, y, w, h = self.box
        return x + w / 2, y + h


class VehicleDetector:
    def __init__(self, min_area_fraction: float = 0.01, history: int = 300, learning_rate: float = 0.01):
        # MOG2 is the improved, core-OpenCV successor to the MOG model used in 2018.
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)
        self.min_area_fraction = min_area_fraction
        self.learning_rate = learning_rate
        self.mask: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> list[Vehicle]:
        fg = self.subtractor.apply(frame, learningRate=self.learning_rate)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows (value 127)
        fg = cv2.erode(fg, None, iterations=2)
        fg = cv2.dilate(fg, None, iterations=6)
        self.mask = fg
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_fraction * frame.shape[0] * frame.shape[1]
        return [Vehicle(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) >= min_area]


def draw_vehicles(frame: np.ndarray, vehicles: list[Vehicle], color=(0, 0, 255)) -> np.ndarray:
    out = frame.copy()
    for v in vehicles:
        x, y, w, h = v.box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    return out
