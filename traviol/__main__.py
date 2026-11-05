"""TraViol command-line pipeline: python -m traviol VIDEO [options]

For every vehicle in the video it measures speed, checks red-light crossings,
and for each violation saves the frame and the plate crop, reads the plate if
Tesseract is available, and optionally uploads the evidence to Dropbox.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2

from .detect import VehicleDetector, draw_vehicles
from .geometry import BirdsEye
from .plates import find_plates
from .speed import estimate_speed


def parse_corners(text: str) -> list[tuple[float, float]]:
    nums = [float(v) for v in text.replace(";", ",").split(",")]
    if len(nums) != 8:
        raise argparse.ArgumentTypeError("corners need 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (TL, TR, BR, BL)")
    return list(zip(nums[0::2], nums[1::2]))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m traviol", description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default="violations", help="folder for evidence and violations.json")
    ap.add_argument("--speed-limit", type=float, default=50, help="km/h (default 50)")
    ap.add_argument("--gap", type=int, default=3, help="frames between the two speed measurements (default 3)")
    ap.add_argument("--metres-per-pixel", type=float, help="calibration in the raw image")
    ap.add_argument("--corners", type=parse_corners, help="road rectangle TL,TR,BR,BL for the bird's-eye view")
    ap.add_argument("--road-length", type=float, help="real length of that rectangle in metres")
    ap.add_argument("--stop-line", type=float, help="stop line position in the bird's-eye view, 0 (far) to 1 (near)")
    ap.add_argument("--red", type=float, nargs=2, metavar=("START", "END"), action="append", default=[],
                    help="seconds during which the signal is red (repeatable)")
    ap.add_argument("--ocr", action="store_true", help="read plates with Tesseract")
    ap.add_argument("--upload", action="store_true", help="upload evidence to Dropbox (needs DROPBOX_ACCESS_TOKEN)")
    args = ap.parse_args()

    birds_eye = BirdsEye(args.corners, metres_long=args.road_length) if args.corners else None
    if not args.metres_per_pixel and not (birds_eye and birds_eye.metres_per_px):
        ap.error("calibrate with --metres-per-pixel, or --corners with --road-length")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector()
    frames: deque = deque(maxlen=args.gap + 1)
    tracks: dict[int, dict] = {}  # id -> {"history": deque[(frame_no, Vehicle)], "flagged": set()}
    next_id = 0
    log = []
    frame_no = -1

    def is_red(t: float) -> bool:
        return any(a <= t <= b for a, b in args.red)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        frames.append((frame_no, frame))
        vehicles = detector(frame)

        # Nearest-centroid tracking: good enough for one lane of traffic at a time.
        for v in vehicles:
            cx, cy = v.centre
            best, best_d = None, max(frame.shape[:2]) * 0.15
            for tid, tr in tracks.items():
                last_no, last_v = tr["history"][-1]
                if frame_no - last_no > args.gap + 2:
                    continue
                lx, ly = last_v.centre
                d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = next_id
                next_id += 1
                tracks[best] = {"history": deque(maxlen=args.gap + 1), "flagged": set()}
            tracks[best]["history"].append((frame_no, v))

            tr = tracks[best]
            reasons = []
            evidence = {}

            # Overspeeding: compare with the same vehicle `gap` frames ago.
            first_no, first_v = tr["history"][0]
            if "speed" not in tr["flagged"] and frame_no - first_no == args.gap:
                old = dict(frames).get(first_no)
                if old is not None:
                    # Measure in the bird's-eye view when it is metric; otherwise in the raw image.
                    metric_view = birds_eye if (birds_eye and birds_eye.metres_per_px) else None
                    est = estimate_speed(old, frame, first_v.box, args.gap, fps,
                                         metres_per_px=None if metric_view else args.metres_per_pixel,
                                         birds_eye=metric_view)
                    if est is not None:
                        evidence["speed_kmh"] = round(est.kmh, 1)
                        if est.kmh > args.speed_limit:
                            reasons.append("overspeeding")
                            tr["flagged"].add("speed")

            # Red light: the vehicle's front crosses the stop line while the signal is red.
            if birds_eye and args.stop_line is not None and "red" not in tr["flagged"] and len(tr["history"]) > 1:
                line_y = args.stop_line * (birds_eye.size[1] - 1)
                prev_y = birds_eye.points([tr["history"][-2][1].front])[0][1]
                now_y = birds_eye.points([v.front])[0][1]
                if prev_y < line_y <= now_y and is_red(frame_no / fps):
                    reasons.append("red light")
                    tr["flagged"].add("red")

            if reasons:
                x, y, w, h = v.box
                car = frame[max(0, y) : y + h, max(0, x) : x + w]
                plates = find_plates(car)
                stem = f"vehicle{best:03d}_frame{frame_no:05d}"
                cv2.imwrite(str(out / f"{stem}.jpg"), draw_vehicles(frame, [v]))
                record = {"vehicle": best, "frame": frame_no, "time_s": round(frame_no / fps, 2),
                          "violations": reasons, **evidence, "frame_image": f"{stem}.jpg"}
                if plates:
                    cv2.imwrite(str(out / f"{stem}_plate.png"), plates[0].crop)
                    record["plate_image"] = f"{stem}_plate.png"
                    if args.ocr:
                        from .ocr import read_plate

                        record["plate_text"] = read_plate(plates[0].crop)
                if args.upload:
                    from .notify import upload

                    for key in ("frame_image", "plate_image"):
                        if key in record:
                            upload(out / record[key])
                log.append(record)
                print(json.dumps(record))

    (out / "violations.json").write_text(json.dumps(log, indent=2))
    print(f"{len(log)} violation(s) written to {out / 'violations.json'}")


if __name__ == "__main__":
    main()
