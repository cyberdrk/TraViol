"""Build a test clip with a known ground-truth speed from samples/road_frame.jpg.

The car is removed from the frame by patching in the road beside it, the empty road is shown for one
second so the background model can learn it, then the car is pasted back moving
towards the camera at a constant number of pixels per frame.

    python examples/make_synthetic_clip.py --px-per-frame 18 --out samples/synthetic.mp4
    # with --metres-per-pixel 0.03 at 30 fps: 18 × 0.03 × 30 × 3.6 = 58.3 km/h
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAR_BOX = (665, 0, 220, 135)  # x, y, w, h of the car in road_frame.jpg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px-per-frame", type=float, default=18)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--out", default=str(ROOT / "samples" / "synthetic.mp4"))
    args = ap.parse_args()

    frame = cv2.imread(str(ROOT / "samples" / "road_frame.jpg"))
    h, w = frame.shape[:2]
    x, y, cw, ch = CAR_BOX
    car = frame[y : y + ch, x : x + cw].copy()

    # Fill the car's area with the road beside it (same rows, so the crossing
    # stripes line up), feathering the seams so the background model sees no edge.
    road = frame.copy()
    pad = 6
    y0, y1, x0, x1 = y, y + ch + pad, x - pad, x + cw + pad
    donor = frame[y0:y1, x0 - (x1 - x0) : x0].astype(np.float32)
    alpha = np.ones((y1 - y0, x1 - x0, 1), np.float32)
    ramp = np.linspace(0, 1, pad + 1)[1:]
    alpha[:, :pad, 0] = ramp
    alpha[:, -pad:, 0] = ramp[::-1]
    alpha[-pad:, :, 0] *= ramp[::-1, None]
    patch = alpha * donor + (1 - alpha) * frame[y0:y1, x0:x1].astype(np.float32)
    road[y0:y1, x0:x1] = patch.astype(np.uint8)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for _ in range(int(args.fps)):  # one second of empty road
        writer.write(road)
    top = 0.0
    while top + ch < h:
        f = road.copy()
        t = int(round(top))
        f[t : t + ch, x : x + cw] = car
        writer.write(f)
        top += args.px_per_frame
    writer.release()
    print(f"wrote {args.out}: car moves {args.px_per_frame} px/frame at {args.fps} fps")


if __name__ == "__main__":
    main()
