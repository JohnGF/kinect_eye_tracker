"""Synthetic head/eye world used by the tests: known geometry -> landmark-like features."""

import math

import numpy as np

from gaze_model import GazeSample, ScreenGeometry, dir_to_angles

SCREEN = ScreenGeometry()


def rot(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def features(eye_yaw, eye_pitch, head_yaw, head_pitch, rng=None, noise=0.0):
    """Plausible iris-offset features incl. the projection leak from head rotation."""
    n = (lambda: rng.normal(0, noise)) if rng is not None and noise else (lambda: 0.0)
    se, sp = math.sin(eye_yaw), math.sin(eye_pitch)
    dx = 2.2 * se - 0.4 * se ** 3 + 0.25 * math.sin(head_yaw) + n()
    dy = -1.6 * sp + 0.15 * math.sin(head_pitch) + n()
    lid = 0.30 + 0.35 * sp + n()
    return np.array([dx, dy, lid])


def sample_looking_at(target_px, P, R, rng=None, noise=0.0) -> GazeSample:
    d = SCREEN.px_to_cm(*target_px) - P
    ey, ep = dir_to_angles(R.T @ d)
    hy, hp = dir_to_angles(R @ np.array([0.0, 0.0, 1.0]))
    return GazeSample(feat=features(ey, ep, hy, hp, rng, noise), R=R, P=np.asarray(P, dtype=float))


def head_facing(P, target_px, yaw_off=0.0, pitch_off=0.0):
    """Rotation pointing the head roughly at a screen point, plus offsets (rad)."""
    yaw, pitch = dir_to_angles(SCREEN.px_to_cm(*target_px) - np.asarray(P, dtype=float))
    return rot(yaw + yaw_off, pitch + pitch_off)


GRID9 = [(fx * SCREEN.w_px, fy * SCREEN.h_px) for fy in (0.15, 0.5, 0.85) for fx in (0.15, 0.5, 0.85)]
CENTER = (SCREEN.w_px / 2, SCREEN.h_px / 2)
P0 = np.array([0.0, -16.0, -60.0])


def calibration_session(rng, noise=0.01, sweep=True):
    samples, targets = [], []
    for t in GRID9:
        for _ in range(15):
            R = head_facing(P0, CENTER, rng.normal(0, 0.01), rng.normal(0, 0.01))
            samples.append(sample_looking_at(t, P0 + rng.normal(0, 0.3, 3), R, rng, noise))
            targets.append(t)
    if sweep:
        for yo in np.linspace(-0.26, 0.26, 9):
            for po in np.linspace(-0.17, 0.17, 7):
                R = head_facing(P0, CENTER, yo, po)
                samples.append(sample_looking_at(CENTER, P0 + rng.normal(0, 0.5, 3), R, rng, noise))
                targets.append(CENTER)
    return samples, targets


def random_test_set(rng, n=400, yaw_deg=15, pitch_deg=10, shift_cm=8, dz_cm=10, noise=0.01):
    out = []
    for _ in range(n):
        P = P0 + np.array([rng.uniform(-shift_cm, shift_cm), rng.uniform(-5, 5), rng.uniform(-dz_cm, dz_cm)])
        R = head_facing(P, CENTER, math.radians(rng.uniform(-yaw_deg, yaw_deg)), math.radians(rng.uniform(-pitch_deg, pitch_deg)))
        t = (rng.uniform(0.1, 0.9) * SCREEN.w_px, rng.uniform(0.1, 0.9) * SCREEN.h_px)
        out.append((sample_looking_at(t, P, R, rng, noise), t))
    return out
