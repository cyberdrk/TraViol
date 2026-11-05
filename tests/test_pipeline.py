"""End-to-end checks on a synthetic clip with a known speed, plus the 2018 sample frame."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traviol.plates import find_plates  # noqa: E402

CORNERS = "109,218,1007,179,1278,525,28,610"  # road rectangle used in the 2018 bird's-eye script


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic_clip.py"),
                    "--px-per-frame", "12", "--out", str(path)], check=True, capture_output=True)
    return path


def run(clip, tmp_path, *extra):
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "traviol", str(clip), "--out", str(out),
                    "--metres-per-pixel", "0.03", *extra], check=True, capture_output=True, cwd=ROOT)
    return json.loads((out / "violations.json").read_text())


def test_speed_matches_ground_truth(clip, tmp_path):
    # 12 px/frame × 0.03 m/px × 30 fps × 3.6 = 38.88 km/h
    log = run(clip, tmp_path, "--speed-limit", "30")
    assert log and log[0]["violations"] == ["overspeeding"]
    assert log[0]["speed_kmh"] == pytest.approx(38.88, abs=1.0)


def test_no_violation_under_limit(clip, tmp_path):
    assert run(clip, tmp_path, "--speed-limit", "50") == []


def test_red_light_crossing(clip, tmp_path):
    log = run(clip, tmp_path, "--speed-limit", "80", "--corners", CORNERS, "--stop-line", "0.5", "--red", "0", "10")
    assert [r["violations"] for r in log] == [["red light"]]


def test_green_light_is_not_a_violation(clip, tmp_path):
    assert run(clip, tmp_path, "--speed-limit", "80", "--corners", CORNERS, "--stop-line", "0.5") == []


def test_plate_found_on_sample_frame():
    plates = find_plates(cv2.imread(str(ROOT / "samples" / "road_frame.jpg")))
    assert plates, "no plate found"
    x, y, w, h = plates[0].bbox
    assert 730 < x + w / 2 < 810 and 100 < y + h / 2 < 130  # the plate on the silver car
