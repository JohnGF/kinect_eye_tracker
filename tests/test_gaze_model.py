import math
from types import SimpleNamespace

import numpy as np
import pytest

from gaze_model import (
    HeadFrameGazeModel, ScreenGeometry, angles_to_dir, angular_error_deg, dir_to_angles,
    eye_features, prior_eye_angles,
)
from sim import CENTER, GRID9, SCREEN, calibration_session, random_test_set


def mean_err(model_predict, test_set):
    errs = []
    for s, t in test_set:
        p = model_predict(s)
        errs.append(angular_error_deg(SCREEN, s.P, p, t) if p is not None else 90.0)
    return float(np.mean(errs))


def old_pipeline(cal_samples, cal_targets):
    """Reproduces the previous tracker: prior eye gain + bilinear map on the compound screen point."""
    def raw(s):
        yaw, pitch = prior_eye_angles(s.feat)
        return SCREEN.intersect(s.P, s.R @ angles_to_dir(yaw, pitch))

    five = [CENTER, GRID9[0], GRID9[2], GRID9[6], GRID9[8]]
    A, B = [], []
    for s, t in zip(cal_samples, cal_targets):
        if t in five:
            rx, ry = raw(s)
            A.append([1, rx, ry, rx * ry])
            B.append(t)
    W, *_ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)

    def predict(s):
        rx, ry = raw(s)
        return tuple(np.array([1, rx, ry, rx * ry]) @ W)
    return predict


def test_angle_roundtrip():
    for yaw, pitch in [(0.0, 0.0), (0.3, -0.2), (-0.5, 0.4)]:
        assert dir_to_angles(angles_to_dir(yaw, pitch)) == pytest.approx((yaw, pitch), abs=1e-9)


def test_screen_px_cm_roundtrip():
    g = ScreenGeometry()
    assert g.cm_to_px(*g.px_to_cm(700.0, 300.0)[:2]) == pytest.approx((700.0, 300.0))


def test_new_model_handles_head_motion_old_does_not():
    rng = np.random.default_rng(0)
    cal_s, cal_t = calibration_session(rng)
    model = HeadFrameGazeModel(SCREEN)
    train_rms = model.fit(cal_s, cal_t)
    assert train_rms < 1.5

    still = random_test_set(rng, yaw_deg=1, pitch_deg=1, shift_cm=0.5, dz_cm=0.5)
    moving = random_test_set(rng, yaw_deg=15, pitch_deg=10, shift_cm=8, dz_cm=10)

    old = old_pipeline(cal_s, cal_t)
    new_still, new_moving = mean_err(model.predict_px, still), mean_err(model.predict_px, moving)
    old_moving = mean_err(old, moving)
    print(f"new still={new_still:.2f} deg  new moving={new_moving:.2f} deg  old moving={old_moving:.2f} deg")
    assert new_still < 1.0
    assert new_moving < 1.5
    assert old_moving > 2 * new_moving


def test_head_sweep_is_what_removes_projection_leak():
    rng = np.random.default_rng(1)
    moving = random_test_set(rng, yaw_deg=15, pitch_deg=10)
    with_sweep = HeadFrameGazeModel(SCREEN)
    with_sweep.fit(*calibration_session(rng, sweep=True))
    no_sweep = HeadFrameGazeModel(SCREEN)
    no_sweep.fit(*calibration_session(rng, sweep=False))
    assert mean_err(with_sweep.predict_px, moving) < mean_err(no_sweep.predict_px, moving)


def _landmarks(roll_rad: float, iris_shift: float, w=1920, h=1080):
    """Fake 478-point landmark list: two eyes with the iris shifted along the eye axis, then rolled."""
    pts = {}
    c = np.array([960.0, 540.0])
    u = np.array([math.cos(roll_rad), math.sin(roll_rad)])
    v = np.array([-u[1], u[0]])
    for (a, b, iris, up, lo), off in [((33, 133, 468, 159, 145), -60.0), ((362, 263, 473, 386, 374), 60.0)]:
        mid = c + off * u
        pts[a], pts[b] = mid - 20 * u, mid + 20 * u
        pts[iris] = mid + iris_shift * 20 * u
        pts[up], pts[lo] = mid - 6 * v, mid + 6 * v
    lms = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    for i, p in pts.items():
        lms[i] = SimpleNamespace(x=p[0] / w, y=p[1] / h, z=0.0)
    return lms


def test_eye_features_are_roll_invariant():
    flat = eye_features(_landmarks(0.0, 0.3), 1920, 1080, 0.0, 0.0)
    rolled = eye_features(_landmarks(math.radians(20), 0.3), 1920, 1080, 0.0, 0.0)
    assert flat == pytest.approx([0.3, 0.0, 0.3], abs=1e-6)
    assert rolled == pytest.approx(flat, abs=1e-6)


def test_camera_offset_roundtrip_and_shift():
    g = ScreenGeometry(cam_x_cm=-9.4)
    assert g.cm_to_px(*g.px_to_cm(700.0, 300.0)[:2]) == pytest.approx((700.0, 300.0))
    # the point straight below the lens is 9.4 cm left of the screen centre
    assert g.cm_to_px(0.0, 0.0)[0] == pytest.approx(1280 - 9.4 / 59.7 * 2560)
