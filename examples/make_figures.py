"""Render the README figures in docs/figures from the sample frame and a synthetic clip.

    python examples/make_figures.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traviol.detect import VehicleDetector, draw_vehicles  # noqa: E402
from traviol.geometry import BirdsEye  # noqa: E402
from traviol.plates import draw_plates, find_plates  # noqa: E402
from traviol.speed import estimate_speed  # noqa: E402

OUT = ROOT / "docs" / "figures"
CORNERS = [(109, 218), (1007, 179), (1278, 525), (28, 610)]
STOP_LINE = 0.5
GAP = 3


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(out, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def fit(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(ROOT / "samples" / "road_frame.jpg"))

    # 1-2. Background subtraction and detection on the synthetic clip
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic_clip.py"), "--px-per-frame", "12",
                        "--out", str(clip)], check=True, capture_output=True)
        cap = cv2.VideoCapture(str(clip))
        det = VehicleDetector()
        frames, boxes = [], []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            boxes.append(det(f))
            if len(frames) == 52:
                mask = det.mask.copy()
        cap.release()

    k = 51
    detected = draw_vehicles(frames[k], boxes[k])
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 3. ORB matches between frames GAP apart
    est = estimate_speed(frames[k - GAP], frames[k], boxes[k - GAP][0].box, GAP, 30, metres_per_px=0.03)
    pair = np.hstack([frames[k - GAP], frames[k]])
    w = frames[k].shape[1]
    for (ax, ay), (bx, by) in zip(est.points_a[:25], est.points_b[:25]):
        color = tuple(int(c) for c in np.random.default_rng(int(ax)).integers(80, 255, 3))
        cv2.line(pair, (int(ax), int(ay)), (int(bx) + w, int(by)), color, 2, cv2.LINE_AA)
        cv2.circle(pair, (int(ax), int(ay)), 5, color, -1)
        cv2.circle(pair, (int(bx) + w, int(by)), 5, color, -1)
    pair = label(pair, f"ORB matches, {GAP} frames apart: {est.shift_px:.0f} px -> {est.kmh:.1f} km/h")

    # 4. Bird's-eye view with the stop line
    be = BirdsEye(CORNERS)
    corners_img = frame.copy()
    cv2.polylines(corners_img, [np.int32(CORNERS)], True, (0, 220, 255), 3, cv2.LINE_AA)
    for (x, y), name in zip(CORNERS, ["TL", "TR", "BR", "BL"]):
        cv2.circle(corners_img, (x, y), 9, (0, 220, 255), -1)
        cv2.putText(corners_img, name, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    top = be.warp(frame)
    ly = int(STOP_LINE * (top.shape[0] - 1))
    cv2.line(top, (0, ly), (top.shape[1], ly), (0, 0, 255), 4)
    cv2.putText(top, "stop line", (10, ly - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 5. Plate localisation on the real frame
    plates = find_plates(frame)
    x, y, pw, ph = plates[0].bbox
    zoom = draw_plates(frame, plates)[max(0, y - 70) : y + ph + 60, max(0, x - 130) : x + pw + 130]
    zoom = cv2.resize(zoom, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    H = 360
    cv2.imwrite(str(OUT / "1_background_mask.jpg"), fit(label(mask_bgr, "Mixture-of-Gaussians foreground mask"), H))
    cv2.imwrite(str(OUT / "2_vehicle_detected.jpg"), fit(label(detected, "Vehicle detected"), H))
    cv2.imwrite(str(OUT / "3_orb_speed.jpg"), fit(pair, H))
    road = np.hstack([fit(label(corners_img, "Road rectangle (TL, TR, BR, BL)"), H), fit(label(top, "Bird's-eye"), H)])
    cv2.imwrite(str(OUT / "4_birds_eye.jpg"), road)
    cv2.imwrite(str(OUT / "5_plate.jpg"), fit(label(zoom, "Plate located"), H))
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
