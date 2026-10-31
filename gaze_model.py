"""
Head-invariant gaze model.

The old pipeline fitted a bilinear map on the *final* screen point (head ray + eye ray already
combined). Calibration is done with the head roughly still, so that map learned the eye gain and
then applied it to head rotation/translation as well: any head movement after calibration moved
the cursor by the wrong amount.

This module separates the two:
  1. Eye features are measured in an eye-aligned 2D frame (corner-to-corner axis), so head roll
     and the image aspect ratio no longer leak into them.
  2. Calibration fits features -> eye-in-head angles (yaw, pitch), using the head pose recorded
     with every sample to compute what the eye angle must have been to hit the target.
  3. Live, the fitted eye angles are rotated by the live head pose and intersected with the
     screen plane, so head movement is handled geometrically rather than by the fitted map.
Head yaw/pitch are also regressors, which lets the fit absorb the projection bias of 2D iris
landmarks under head rotation, provided the calibration contains some head movement (the
head-sweep phase in calibration.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# MediaPipe FaceLandmarker indices. "R" = subject's right eye (image left on an unmirrored feed).
# Corners are listed image-left -> image-right for both eyes so both axes point the same way.
R_EYE = {"a": 33, "b": 133, "iris": 468, "upper": 159, "lower": 145}
L_EYE = {"a": 362, "b": 263, "iris": 473, "upper": 386, "lower": 374}
BLINK_THRESHOLD = 0.45


@dataclass
class ScreenGeometry:
    """Screen plane z = 0 in camera space, camera on the top edge (cm, y up).

    cam_x_cm: horizontal position of the colour lens relative to the screen centre (negative =
    left of centre as seen by the user), measured with screen_ruler.py."""

    w_px: int = 2560
    h_px: int = 1440
    w_cm: float = 59.7
    h_cm: float = 33.6
    cam_x_cm: float = 0.0

    def cm_to_px(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        return self.w_px / 2.0 + (x_cm + self.cam_x_cm) / self.w_cm * self.w_px, -y_cm / self.h_cm * self.h_px

    def px_to_cm(self, px: float, py: float) -> np.ndarray:
        return np.array([(px - self.w_px / 2.0) / self.w_px * self.w_cm - self.cam_x_cm, -py / self.h_px * self.h_cm, 0.0])

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[tuple[float, float]]:
        if abs(direction[2]) < 1e-4:
            return None
        t = -origin[2] / direction[2]
        if t <= 0.0:
            return None
        hit = origin + t * direction
        return self.cm_to_px(hit[0], hit[1])


def angles_to_dir(yaw: float, pitch: float) -> np.ndarray:
    return np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])


def dir_to_angles(v: np.ndarray) -> tuple[float, float]:
    v = v / np.linalg.norm(v)
    return math.atan2(v[0], v[2]), math.asin(float(np.clip(v[1], -1.0, 1.0)))


def _eye(landmarks, idx: dict, img_w: int, img_h: int) -> tuple[float, float, float]:
    def p(i):
        return np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])

    a, b = p(idx["a"]), p(idx["b"])
    width = float(np.linalg.norm(b - a))
    if width < 1e-6:
        return 0.0, 0.0, 0.0
    u = (b - a) / width
    v = np.array([-u[1], u[0]])  # image-down when u is image-right
    half = width / 2.0
    rel = p(idx["iris"]) - (a + b) / 2.0
    dx = float(rel @ u) / half
    dy = float(rel @ v) / half
    lid = float((p(idx["lower"]) - p(idx["upper"])) @ v) / width
    return dx, dy, lid


def eye_features(landmarks, img_w: int, img_h: int, blink_l: float, blink_r: float) -> np.ndarray:
    """[dx, dy, lid] fused across eyes, measured in each eye's own corner-aligned frame."""
    r = np.array(_eye(landmarks, R_EYE, img_w, img_h))
    l = np.array(_eye(landmarks, L_EYE, img_w, img_h))
    if blink_l > BLINK_THRESHOLD and blink_r <= BLINK_THRESHOLD:
        return r
    if blink_r > BLINK_THRESHOLD and blink_l <= BLINK_THRESHOLD:
        return l
    return (r + l) / 2.0


def head_angles(R: np.ndarray) -> tuple[float, float]:
    """Yaw/pitch (rad) of the head forward axis, used as regressors."""
    return dir_to_angles(R @ np.array([0.0, 0.0, 1.0]))


def prior_eye_angles(feat: Sequence[float]) -> tuple[float, float]:
    """Uncalibrated guess (the old hand-tuned gain), used only for feedback before calibration."""
    yaw = math.asin(float(np.clip(feat[0] * 0.45, -0.65, 0.65)))
    pitch = math.asin(float(np.clip(-feat[1] * 0.45, -0.55, 0.55)))
    return yaw, pitch


@dataclass
class GazeSample:
    feat: np.ndarray       # [dx, dy, lid]
    R: np.ndarray          # 3x3 head rotation (camera frame)
    P: np.ndarray          # head position (cm, camera frame)


def _design(feat: np.ndarray, hang: np.ndarray, use_head: bool) -> np.ndarray:
    """Linear terms only: quadratic/interaction terms overfitted badly on real data (2026-10-31 runs)."""
    cols = [feat[:, 0], feat[:, 1], feat[:, 2]]
    if use_head:
        cols += [hang[:, 0], hang[:, 1]]
    return np.column_stack(cols)


# Head terms are only identifiable when calibration actually contains head rotation.
MIN_HEAD_YAW_RANGE_RAD = math.radians(8.0)
MIN_HEAD_PITCH_RANGE_RAD = math.radians(6.0)


class HeadFrameGazeModel:
    """features (+ head angles) -> eye-in-head yaw/pitch, then ray-cast with the live head pose."""

    def __init__(self, screen: Optional[ScreenGeometry] = None, ridge: float = 1e-3):
        self.screen = screen or ScreenGeometry()
        self.ridge = ridge
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None  # (n_terms + 1, 2)
        self.use_head = False

    @property
    def is_fitted(self) -> bool:
        return self.W is not None

    def target_eye_angles(self, s: GazeSample, target_px: tuple[float, float]) -> tuple[float, float]:
        """Eye-in-head angles needed for the ray from the head to hit the target."""
        d = self.screen.px_to_cm(*target_px) - s.P
        return dir_to_angles(s.R.T @ d)

    def fit(self, samples: Sequence[GazeSample], targets_px: Sequence[tuple[float, float]]) -> float:
        """Least-squares fit; returns training RMS error in degrees of eye angle."""
        if len(samples) < 12:
            raise ValueError(f"need at least 12 samples, got {len(samples)}")
        feat = np.array([s.feat for s in samples], dtype=np.float64)
        hang = np.array([head_angles(s.R) for s in samples], dtype=np.float64)
        Y = np.array([self.target_eye_angles(s, t) for s, t in zip(samples, targets_px)], dtype=np.float64)
        span = hang.max(axis=0) - hang.min(axis=0)
        self.use_head = bool(span[0] >= MIN_HEAD_YAW_RANGE_RAD and span[1] >= MIN_HEAD_PITCH_RANGE_RAD)
        if not self.use_head:
            print(f"Gaze model: head range too small (yaw {math.degrees(span[0]):.1f}, pitch "
                  f"{math.degrees(span[1]):.1f} deg); fitting without head terms.")
        X = _design(feat, hang, self.use_head)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        Xs = np.column_stack([np.ones(len(X)), (X - self.mean) / self.std])
        reg = self.ridge * len(X) * np.eye(Xs.shape[1])
        reg[0, 0] = 0.0  # don't shrink the intercept
        self.W = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)
        resid = Xs @ self.W - Y
        return float(np.degrees(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))))

    def eye_angles(self, s: GazeSample) -> tuple[float, float]:
        if self.W is None:
            return prior_eye_angles(s.feat)
        X = _design(np.asarray(s.feat, dtype=np.float64)[None, :], np.array([head_angles(s.R)]), self.use_head)
        Xs = np.concatenate([[1.0], ((X - self.mean) / self.std)[0]])
        yaw, pitch = Xs @ self.W
        return float(yaw), float(pitch)

    def predict_px(self, s: GazeSample) -> Optional[tuple[float, float]]:
        yaw, pitch = self.eye_angles(s)
        return self.screen.intersect(s.P, s.R @ angles_to_dir(yaw, pitch))

    def head_px(self, s: GazeSample) -> Optional[tuple[float, float]]:
        return self.screen.intersect(s.P, s.R @ np.array([0.0, 0.0, 1.0]))


def angular_error_deg(screen: ScreenGeometry, P: np.ndarray, pred_px, target_px) -> float:
    """Visual angle between the predicted and target screen points as seen from the head."""
    a = screen.px_to_cm(*pred_px) - P
    b = screen.px_to_cm(*target_px) - P
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
