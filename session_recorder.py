"""
Session recorder: colour video + per-frame metadata, for offline gaze models (offline_l2cs.py).

Writes into <dir>/:
  video.mp4   H.264 (libx264, CRF 18) of every processed frame, in order. The container frame
              rate is nominal; real timing is in frames.csv.
  frames.csv  one row per video frame: frame index, monotonic timestamp, phase
              (cal_points / cal_sweep / val_still / val_moving / live), target dot, time since
              that dot appeared, blink scores, face box (normalised), Kinect depth face distance (if kinect2pipe --ir) and the MediaPipe
              features + head pose (same columns as the calibration/validation logs).
  meta.json   screen geometry and recording settings.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from gaze_model import ScreenGeometry
from validation import RAW_FIELDS, raw_columns

FIELDS = ["frame", "t", "phase", "target_x", "target_y", "t_since_onset", "has_face", "blink_l", "blink_r",
          "face_x0", "face_y0", "face_x1", "face_y1", "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
          "head_dist_cm", "depth_cm"] + RAW_FIELDS


class SessionRecorder:
    def __init__(self, out_dir: Path, screen: ScreenGeometry, nominal_fps: int = 30, extra_meta: Optional[dict] = None):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found; needed for --record")
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.screen = screen
        self.nominal_fps = nominal_fps
        self.proc: Optional[subprocess.Popen] = None
        self.size: Optional[tuple[int, int]] = None
        self.n = 0
        self.csv_file = open(self.dir / "frames.csv", "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        self.writer.writeheader()
        (self.dir / "meta.json").write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "screen": vars(screen),
            "video": "video.mp4", "frames": "frames.csv",
            "clock": "time.time() seconds",
            **(extra_meta or {}),
        }, indent=2))

    def _start_ffmpeg(self, w: int, h: int):
        self.size = (w, h)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.nominal_fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             str(self.dir / "video.mp4")],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray, t: float, phase: str, target: Optional[tuple], t_since_onset: float,
              features: Optional[dict], depth_cm: Optional[float] = None):
        h, w = frame_bgr.shape[:2]
        if self.proc is None:
            self._start_ffmpeg(w, h)
        if (w, h) != self.size:
            return  # resolution changed mid-session (mode switch); skip rather than corrupt the stream
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError:
            return
        row = {"frame": self.n, "t": t, "phase": phase, "t_since_onset": t_since_onset,
               "target_x": target[0] if target else "", "target_y": target[1] if target else "",
               "has_face": int(features is not None), "depth_cm": "" if depth_cm is None else depth_cm}
        if features is not None:
            x0, y0, x1, y1 = features["face_bbox"]
            row.update({"blink_l": features["blink_l"], "blink_r": features["blink_r"],
                        "face_x0": x0, "face_y0": y0, "face_x1": x1, "face_y1": y1,
                        "head_yaw_deg": features["head_yaw"], "head_pitch_deg": features["head_pitch"],
                        "head_roll_deg": features["head_roll"], "head_dist_cm": features["head_dist_cm"],
                        **raw_columns(features["sample"])})
        self.writer.writerow(row)
        self.n += 1

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
            self.proc = None
        self.csv_file.close()
        print(f"Session recorded: {self.n} frames -> {self.dir}")
