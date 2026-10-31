"""
Fullscreen calibration for the head-invariant gaze model (gaze_model.HeadFrameGazeModel).

Two phases:
  1. 9-point grid: hold the head still and fixate each dot. Every frame of a stable fixation is
     kept (not just one median point) together with the head pose of that frame.
  2. Guided head sweep: eyes stay on the centre dot while the head goes up, down, left, right
     (rotation), then sways left and right (translation). Each step only completes once the head
     has really moved that far, so the model gets eye-counter-rotation data it can use to tell
     head rotation apart from eye rotation. (A free-form sweep let people keep their head still.)
"""

import os

os.environ.setdefault("QT_QPA_FONTDIR", "")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import math
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from gaze_model import GazeSample, HeadFrameGazeModel, ScreenGeometry, head_angles

# (label, axis, goal). axis: "pitch"/"yaw" in degrees of head rotation, "x" in cm of head sway.
# Steps come in pairs: the first of a pair accepts either sign, the second must go the opposite way,
# so the sequence works whether or not the camera image is mirrored.
SWEEP_STEPS = [
    ("tilt head UP", "pitch", 8.0), ("tilt head DOWN", "pitch", 8.0),
    ("turn head LEFT", "yaw", 12.0), ("turn head RIGHT", "yaw", 12.0),
    ("sway whole head LEFT (keep facing screen)", "x", 5.0), ("sway whole head RIGHT (keep facing screen)", "x", 5.0),
]


def head_offsets(s: GazeSample) -> dict:
    yaw, pitch = head_angles(s.R)
    return {"yaw": math.degrees(yaw), "pitch": math.degrees(pitch), "x": float(s.P[0])}


class GazeCalibrationEngine:
    def __init__(self, screen_w: int = 2560, screen_h: int = 1440, dwell_sec: float = 0.9,
                 screen: Optional[ScreenGeometry] = None, use_sweep: bool = True):
        self.screen = screen or ScreenGeometry(w_px=screen_w, h_px=screen_h)
        self.screen_w = self.screen.w_px
        self.screen_h = self.screen.h_px
        self.dwell_sec = dwell_sec
        self.use_sweep = use_sweep

        self.targets_norm = [(fx, fy) for fy in (0.15, 0.5, 0.85) for fx in (0.15, 0.5, 0.85)]

        self.model = HeadFrameGazeModel(self.screen)
        self.active_idx = 0
        self.phase = "idle"  # "points" | "sweep" | "idle"
        self.is_calibrated = False
        self.train_rms_deg: Optional[float] = None

        self.history: deque = deque(maxlen=12)       # recent samples for stability check
        self.dwell_buffer: List[GazeSample] = []      # samples collected during the current dwell
        self.feature_stability_threshold = 0.08       # std of [dx, dy] (~2 deg of eye rotation)
        self.fixation_start_t: Optional[float] = None
        self.dwell_progress = 0.0

        self.samples: List[GazeSample] = []
        self.targets: List[Tuple[float, float]] = []
        self.sweep_count = 0
        self.sweep_start_t = 0.0
        self.sweep_step = 0
        self.sweep_ref: Optional[dict] = None
        self.sweep_ref_buf: List[dict] = []
        self.sweep_signs: dict = {}
        self.sweep_delta = 0.0
        self.phases: List[str] = []

        self.current_live_gaze: Optional[Tuple[float, float]] = None
        self.flash_until = 0.0
        self.onset_t = 0.0  # when the current target / sweep step appeared (for recordings)

    @property
    def is_active(self) -> bool:
        return self.phase != "idle"

    def start(self):
        self.active_idx = 0
        self.phase = "points"
        self.samples.clear()
        self.targets.clear()
        self.phases.clear()
        self.history.clear()
        self.dwell_buffer.clear()
        self.fixation_start_t = None
        self.dwell_progress = 0.0
        self.sweep_count = 0
        self.sweep_step = 0
        self.sweep_ref = None
        self.sweep_ref_buf = []
        self.sweep_signs = {}
        self.onset_t = time.time()
        print("\n=== Calibration started ===")
        print(f"Phase 1: keep your head still and look at each of the {len(self.targets_norm)} dots.")

    def get_current_target_px(self) -> Tuple[int, int]:
        if self.phase == "sweep":
            return self.screen_w // 2, self.screen_h // 2
        nx, ny = self.targets_norm[min(self.active_idx, len(self.targets_norm) - 1)]
        return int(nx * self.screen_w), int(ny * self.screen_h)

    def update(self, sample: Optional[GazeSample], is_blinking: bool, now: Optional[float] = None) -> bool:
        """Feed one frame. Returns True when calibration has just finished."""
        if not self.is_active:
            return False
        now = time.time() if now is None else now
        self.current_live_gaze = self.model.predict_px(sample) if sample is not None else None

        if sample is None or is_blinking:
            self.fixation_start_t = None
            self.dwell_buffer.clear()
            self.dwell_progress = max(0.0, self.dwell_progress - 0.04)
            return False

        if self.phase == "sweep":
            self.samples.append(sample)
            self.targets.append(self.get_current_target_px())
            self.phases.append("sweep")
            self.sweep_count += 1
            return self._sweep_update(sample)

        self.history.append(sample)
        if len(self.history) < 8:
            self.dwell_progress = 0.0
            return False

        feats = np.array([s.feat[:2] for s in self.history])
        dispersion = float(np.hypot(*feats.std(axis=0)))
        if dispersion < self.feature_stability_threshold:
            if self.fixation_start_t is None:
                self.fixation_start_t = now
                self.dwell_buffer.clear()
            self.dwell_buffer.append(sample)
            self.dwell_progress = min(1.0, (now - self.fixation_start_t) / self.dwell_sec)
            if self.dwell_progress >= 1.0:
                return self._lock_point(self.dwell_buffer)
        else:
            self.fixation_start_t = None
            self.dwell_buffer.clear()
            self.dwell_progress = max(0.0, self.dwell_progress - 0.05)
        return False

    def _sweep_update(self, sample: GazeSample) -> bool:
        off = head_offsets(sample)
        if self.sweep_ref is None:  # reference pose = median of the first frames of the sweep
            self.sweep_ref_buf.append(off)
            self.dwell_progress = 0.0
            if len(self.sweep_ref_buf) >= 6:
                self.sweep_ref = {k: float(np.median([o[k] for o in self.sweep_ref_buf])) for k in off}
            return False
        _label, axis, goal = SWEEP_STEPS[self.sweep_step]
        delta = off[axis] - self.sweep_ref[axis]
        first_of_pair = self.sweep_step % 2 == 0
        if first_of_pair:
            signed = abs(delta)
        else:
            signed = -self.sweep_signs.get(axis, 1.0) * delta
        self.sweep_delta = delta
        self.dwell_progress = float(np.clip(signed / goal, 0.0, 1.0))
        if signed >= goal:
            if first_of_pair:
                self.sweep_signs[axis] = math.copysign(1.0, delta)
            return self._advance_sweep()
        return False

    def _advance_sweep(self) -> bool:
        print(f"  sweep step done: {SWEEP_STEPS[self.sweep_step][0]}")
        self.sweep_step += 1
        self.onset_t = time.time()
        self.dwell_progress = 0.0
        self.flash_until = time.time() + 0.25
        if self.sweep_step >= len(SWEEP_STEPS):
            return self._finish()
        return False

    def manual_capture(self, sample: Optional[GazeSample] = None) -> bool:
        """Enter/Space: accept the current point now (or skip the current sweep step)."""
        if self.phase == "sweep":
            if self.sweep_ref is None:
                return False
            print(f"  sweep step skipped: {SWEEP_STEPS[self.sweep_step][0]}")
            return self._advance_sweep()
        if self.phase != "points":
            return False
        batch = self.dwell_buffer or list(self.history)[-8:]
        if not batch and sample is not None:
            batch = [sample]
        if not batch:
            return False
        return self._lock_point(batch)

    def _lock_point(self, batch: List[GazeSample]) -> bool:
        feats = np.array([s.feat[:2] for s in batch])
        med = np.median(feats, axis=0)
        dist = np.linalg.norm(feats - med, axis=1)
        mad = float(np.median(dist)) + 1e-6
        kept = [s for s, d in zip(batch, dist) if d <= 3.0 * mad] or batch
        tgt = self.get_current_target_px()
        self.samples.extend(kept)
        self.targets.extend([tgt] * len(kept))
        self.phases.extend(["points"] * len(kept))
        self.flash_until = time.time() + 0.25
        print(f"Point {self.active_idx + 1}/{len(self.targets_norm)} locked ({len(kept)} frames) -> {tgt}")

        self.active_idx += 1
        self.onset_t = time.time()
        self.history.clear()
        self.dwell_buffer.clear()
        self.fixation_start_t = None
        self.dwell_progress = 0.0

        if self.active_idx >= len(self.targets_norm):
            if self.use_sweep:
                self.phase = "sweep"
                self.sweep_start_t = time.time()
                print("Phase 2: eyes on the centre dot; follow the head-movement prompts.")
                return False
            return self._finish()
        return False

    def _finish(self) -> bool:
        self.phase = "idle"
        try:
            from validation import LOG_DIR, save_raw_samples
            path = LOG_DIR / f"calibration_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            save_raw_samples(path, self.samples, self.targets, self.phases)
            print(f"Calibration frames saved to {path}")
        except OSError as e:
            print(f"Could not save calibration frames: {e}")
        try:
            self.train_rms_deg = self.model.fit(self.samples, self.targets)
            self.is_calibrated = True
            print(f"Calibration fitted on {len(self.samples)} frames, training RMS {self.train_rms_deg:.2f} deg of eye angle.")
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Calibration fit failed: {e}")
        return True

    def predict_px(self, sample: GazeSample) -> Optional[Tuple[float, float]]:
        return self.model.predict_px(sample)

    # ------------------------------------------------------------------ rendering
    def render(self, canvas: np.ndarray):
        if not self.is_active:
            return
        canvas.fill(20)
        now = time.time()
        tgt_x, tgt_y = self.get_current_target_px()

        for i in range(min(self.active_idx, len(self.targets_norm))):
            nx, ny = self.targets_norm[i]
            p = (int(nx * self.screen_w), int(ny * self.screen_h))
            cv2.circle(canvas, p, 10, (50, 180, 50), -1)

        if self.current_live_gaze is not None:
            gx, gy = int(self.current_live_gaze[0]), int(self.current_live_gaze[1])
            if 0 <= gx < self.screen_w and 0 <= gy < self.screen_h:
                cv2.circle(canvas, (gx, gy), 18, (80, 80, 180), 1)

        pulse = (math.sin(now * 6.0) + 1.0) / 2.0
        r = int(28 + pulse * 6)
        if now < self.flash_until:
            cv2.circle(canvas, (tgt_x, tgt_y), 36, (255, 255, 0), -1)
        else:
            cv2.circle(canvas, (tgt_x, tgt_y), r, (100, 100, 100), 2)
            if self.dwell_progress > 0.01:
                cv2.ellipse(canvas, (tgt_x, tgt_y), (r + 4, r + 4), -90, 0, int(self.dwell_progress * 360), (0, 255, 180), 4)
            cv2.circle(canvas, (tgt_x, tgt_y), 9, (255, 255, 255), -1)
            cv2.circle(canvas, (tgt_x, tgt_y), 4, (0, 140, 255), -1)

        if self.phase == "points":
            title = f"CALIBRATION 1/2: point {self.active_idx + 1} of {len(self.targets_norm)}"
            sub = "Keep your head still, look at the dot. Auto-advances when steady, or press [Enter]/[Space]."
        else:
            if self.sweep_ref is None:
                title = "CALIBRATION 2/2: head movement - hold still a moment..."
                sub = "Keep your EYES on the dot the whole time."
            else:
                label, axis, goal = SWEEP_STEPS[self.sweep_step]
                unit = "cm" if axis == "x" else "deg"
                title = f"CALIBRATION 2/2: step {self.sweep_step + 1}/{len(SWEEP_STEPS)}: {label}"
                sub = (f"EYES stay on the dot. Slowly go until the ring fills "
                       f"({abs(self.sweep_delta):.0f}/{goal:.0f} {unit}), then on to the next.  [Enter] skips.")
            for i, (lbl, _, _) in enumerate(SWEEP_STEPS):
                col = (100, 255, 100) if i < self.sweep_step else ((255, 255, 255) if i == self.sweep_step else (110, 110, 110))
                cv2.putText(canvas, ("[x] " if i < self.sweep_step else "[ ] ") + lbl, (60, 220 + 36 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        cv2.putText(canvas, title, (60, 90), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(canvas, sub, (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.putText(canvas, "[Enter/Space] Lock  |  [c] Restart  |  [q/ESC] Quit", (60, self.screen_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 130, 130), 1)
