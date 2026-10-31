import csv

import cv2
import numpy as np

from gaze_model import GazeSample, ScreenGeometry
from session_recorder import SessionRecorder


def test_recorder_writes_aligned_video_and_csv(tmp_path):
    rec = SessionRecorder(tmp_path, ScreenGeometry())
    feats = {"face_bbox": (0.4, 0.3, 0.6, 0.7), "blink_l": 0.1, "blink_r": 0.1, "head_yaw": 1.0, "head_pitch": -2.0,
             "head_roll": 0.5, "head_dist_cm": 55.0,
             "sample": GazeSample(np.array([0.1, -0.1, 0.3]), np.eye(3), np.array([0.0, 0.0, -55.0]))}
    for i in range(12):
        frame = np.full((180, 320, 3), i * 20, np.uint8)
        rec.write(frame, 100.0 + i / 30, "cal_points" if i < 6 else "live", (384, 216) if i < 6 else None,
                  i / 30, feats if i % 4 else None)
    rec.close()
    rows = list(csv.DictReader(open(tmp_path / "frames.csv")))
    assert [int(r["frame"]) for r in rows] == list(range(12))
    assert rows[0]["has_face"] == "0" and rows[1]["has_face"] == "1" and rows[1]["dx"] == "0.1"
    cap = cv2.VideoCapture(str(tmp_path / "video.mp4"))
    n, means = 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        means.append(f.mean())
        n += 1
    assert n == 12
    assert abs(means[5] - 100) < 4  # frame order preserved (frame i has brightness 20*i)
