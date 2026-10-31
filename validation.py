"""
Validation run ([v] in the tracker, after calibration).

Pass 1 "still": head still, 5 dots.  Pass 2 "moving": keep looking at each dot while slowly moving
the head around.  Every frame is logged to logs/validation_<time>.csv with the angular error and
head pose / distance, and a summary breaks the error down by pass, head yaw, head pitch and
distance, which is what tells you whether head movement or distance is the problem.

Re-summarise an old run:  uv run python validation.py logs/validation_YYYYmmdd_HHMMSS.csv
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gaze_model import ScreenGeometry, angular_error_deg

LOG_DIR = Path(__file__).resolve().parent / "logs"
POINTS = [(0.5, 0.5), (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
FIELDS = ["t", "pass", "target_x", "target_y", "raw_x", "raw_y", "filt_x", "filt_y", "err_raw_deg",
          "err_filt_deg", "head_yaw_deg", "head_pitch_deg", "head_roll_deg", "head_dist_cm", "depth_cm"]
# Raw model inputs, so fits can be re-run offline on real data (see save_raw_samples)
RAW_FIELDS = ["dx", "dy", "lid"] + [f"R{i}{j}" for i in range(3) for j in range(3)] + ["Px", "Py", "Pz"]


def raw_columns(sample) -> dict:
    vals = list(sample.feat) + list(np.asarray(sample.R).ravel()) + list(sample.P)
    return {k: float(x) for k, x in zip(RAW_FIELDS, vals)}


def save_raw_samples(path: Path, samples, targets, extra: Optional[list] = None):
    """Write GazeSamples + targets as CSV (one row per frame)."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_x", "target_y"] + RAW_FIELDS + (["phase"] if extra else []))
        w.writeheader()
        for i, (s, t) in enumerate(zip(samples, targets)):
            row = {"target_x": t[0], "target_y": t[1], **raw_columns(s)}
            if extra:
                row["phase"] = extra[i]
            w.writerow(row)


class ValidationSession:
    def __init__(self, screen: ScreenGeometry, still_sec: float = 2.5, moving_sec: float = 5.0, settle_sec: float = 0.8):
        self.screen = screen
        self.plan = [("still", p, still_sec) for p in POINTS] + [("moving", p, moving_sec) for p in POINTS]
        self.settle_sec = settle_sec
        self.idx = -1
        self.step_t0 = 0.0
        self.rows: list[dict] = []

    @property
    def is_active(self) -> bool:
        return 0 <= self.idx < len(self.plan)

    def start(self, now: float):
        self.idx, self.step_t0, self.rows = 0, now, []
        print("\n=== Validation: pass 1 keep head STILL, pass 2 keep eyes on dot while MOVING head ===")

    def target_px(self) -> tuple[float, float]:
        _, (fx, fy), _ = self.plan[self.idx]
        return fx * self.screen.w_px, fy * self.screen.h_px

    def tick(self, now: float) -> bool:
        """Advance through the plan; returns True when the run has just ended."""
        if not self.is_active:
            return False
        if now - self.step_t0 >= self.plan[self.idx][2]:
            self.idx += 1
            self.step_t0 = now
        return not self.is_active

    def record(self, now: float, features: dict, raw_px, filt_px, depth_cm: Optional[float]):
        if not self.is_active or now - self.step_t0 < self.settle_sec:
            return
        tgt = self.target_px()
        P = features["sample"].P
        self.rows.append({
            "t": now, "pass": self.plan[self.idx][0], "target_x": tgt[0], "target_y": tgt[1],
            "raw_x": raw_px[0], "raw_y": raw_px[1], "filt_x": filt_px[0], "filt_y": filt_px[1],
            "err_raw_deg": angular_error_deg(self.screen, P, raw_px, tgt),
            "err_filt_deg": angular_error_deg(self.screen, P, filt_px, tgt),
            "head_yaw_deg": features["head_yaw"], "head_pitch_deg": features["head_pitch"],
            "head_roll_deg": features["head_roll"], "head_dist_cm": features["head_dist_cm"],
            "depth_cm": depth_cm if depth_cm is not None else float("nan"),
            **raw_columns(features["sample"]),
        })

    def finish(self) -> str:
        self.idx = len(self.plan)
        if not self.rows:
            return "Validation: no samples recorded (face not tracked?)."
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"validation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS + RAW_FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        return f"Validation saved to {path}\n" + summarize(self.rows)

    def render(self, canvas: np.ndarray):
        canvas.fill(20)
        tx, ty = map(int, self.target_px())
        name = self.plan[self.idx][0]
        cv2.circle(canvas, (tx, ty), 12, (255, 255, 255), -1)
        cv2.circle(canvas, (tx, ty), 5, (0, 140, 255), -1)
        msg = ("VALIDATION 1/2: keep your head STILL and look at the dot" if name == "still"
               else "VALIDATION 2/2: EYES on the dot, move your head gently (about as far as in calibration)")
        cv2.putText(canvas, msg, (60, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, f"step {self.idx + 1}/{len(self.plan)}", (60, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2)


def _bin_table(rows, key, edges, label, absval=False) -> list[str]:
    out = [f"  by {label}:"]
    vals = np.array([abs(r[key]) if absval else r[key] for r in rows], dtype=float)
    errs = np.array([r["err_filt_deg"] for r in rows], dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.sum() >= 5:
            out.append(f"    {lo:>5.0f} .. {hi:<5.0f} n={m.sum():4d}  mean={errs[m].mean():5.2f} deg  median={np.median(errs[m]):5.2f} deg")
    return out


def summarize(rows: list[dict]) -> str:
    rows = [r for r in rows if not math.isnan(float(r["err_filt_deg"]))]
    if not rows:
        return "no rows"
    lines = ["Validation summary (angular error, filtered gaze; raw in brackets):"]
    for name in ("still", "moving"):
        sub = [r for r in rows if r["pass"] == name]
        if sub:
            ef = np.array([r["err_filt_deg"] for r in sub], dtype=float)
            er = np.array([r["err_raw_deg"] for r in sub], dtype=float)
            lines.append(f"  {name:<7} n={len(sub):4d}  mean={ef.mean():5.2f} deg  median={np.median(ef):5.2f}  "
                         f"p90={np.percentile(ef, 90):5.2f}  [raw mean {er.mean():5.2f}]")
    # head pose relative to the median pose of the "still" pass
    still = [r for r in rows if r["pass"] == "still"] or rows
    y0 = float(np.median([r["head_yaw_deg"] for r in still]))
    p0 = float(np.median([r["head_pitch_deg"] for r in still]))
    rel = [dict(r, dyaw=float(r["head_yaw_deg"]) - y0, dpitch=float(r["head_pitch_deg"]) - p0) for r in rows]
    lines += _bin_table(rel, "dyaw", [0, 3, 6, 10, 15, 25, 90], "|head yaw change| (deg)", absval=True)
    lines += _bin_table(rel, "dpitch", [0, 3, 6, 10, 15, 90], "|head pitch change| (deg)", absval=True)
    dkey = "depth_cm" if any(not math.isnan(float(r["depth_cm"])) for r in rows) else "head_dist_cm"
    drows = [r for r in rows if not math.isnan(float(r[dkey]))]
    lines += _bin_table(drows, dkey, [0, 45, 55, 65, 75, 90, 300],
                        "distance (cm, Kinect depth)" if dkey == "depth_cm" else "distance (cm, MediaPipe estimate)")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = [{k: (v if k == "pass" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    print(summarize(data))
