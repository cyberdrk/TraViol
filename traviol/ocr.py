"""Read plate text with Tesseract (Python 3 port of original_2018/tesse.py).

Tesseract needs clean, large, dark-on-light text, so the crop is upscaled,
binarised with Otsu's threshold and lightly denoised before recognition.
Requires the `tesseract` binary on PATH (https://github.com/tesseract-ocr/tesseract).
"""

from __future__ import annotations

import re

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # OCR is optional; detection and speed work without it
    pytesseract = None

# Single line of plate characters.
TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


def preprocess(crop: np.ndarray, target_height: int = 80) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = max(1.0, target_height / max(1, gray.shape[0]))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 3)
    if binary.mean() < 127:  # make the text dark on a light background
        binary = 255 - binary
    return cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def read_plate(crop: np.ndarray) -> str:
    """Return the plate text, or "" if Tesseract finds nothing usable."""
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed: pip install pytesseract")
    text = pytesseract.image_to_string(preprocess(crop), config=TESSERACT_CONFIG)
    return re.sub(r"[^A-Z0-9-]", "", text.upper())
