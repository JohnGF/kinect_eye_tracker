"""
Gaze and Head Pose filtering algorithms.
Implements:
1. OneEuroFilter: Adaptive low-pass filter for smooth tracking with dynamic cutoff.
2. AngularDeadbandFilter: Deadband filter decoupling postural sway/breathing from intentional head movement.
3. IVTGazeFilter2D: I-VT (Velocity-Threshold Identification) state machine for gaze tracking.
   Completely locks gaze to the spatial centroid during fixations (zero micro-jitter)
   and releases lock instantaneously during saccades (zero latency).
4. GazeFilter2D: 2D coordinate filter wrapper.
"""

import math
from typing import Tuple, Optional
import numpy as np


class OneEuroFilter:
    """
    Adaptive low-pass filter designed for human-computer interaction.
    High smoothing when speed is low (eliminates micro-saccade jitter),
    zero lag when speed is high (responsive saccades).
    """

    def __init__(self, t0: float, x0: float, dx0: float = 0.0,
                 min_cutoff: float = 1.2, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    @staticmethod
    def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
        return a * x + (1.0 - a) * x_prev

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev

        # Estimate derivative (speed)
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class AngularDeadbandFilter:
    """
    Angular deadband filter for head pose orientation (pitch, yaw, roll).
    Small fluctuations below threshold_deg (e.g. cardiac pulse, breathing sway < 0.8°)
    are held stationary or heavily smoothed. Intentional head turns (> 0.8°) are passed
    through smoothly with exponential ease-in.
    """

    def __init__(self, threshold_deg: float = 0.8, smoothing: float = 0.25):
        self.threshold_deg = float(threshold_deg)
        self.smoothing = float(smoothing)
        self.locked_val: Optional[float] = None
        self.prev_val: Optional[float] = None

    def filter(self, val: float) -> float:
        if self.locked_val is None or self.prev_val is None:
            self.locked_val = val
            self.prev_val = val
            return val

        delta = val - self.locked_val
        if abs(delta) < self.threshold_deg:
            # Within deadband: hold locked value with negligible micro-drift
            filtered = self.locked_val + 0.02 * delta
            self.locked_val = filtered
            self.prev_val = filtered
            return filtered
        else:
            # Exceeded deadband: intentional head turn. Ease toward new value.
            filtered = self.prev_val + self.smoothing * (val - self.prev_val)
            self.locked_val = filtered
            self.prev_val = filtered
            return filtered

    def reset(self):
        self.locked_val = None
        self.prev_val = None


class HeadRotationDeadband:
    """
    Stabilizes head rotation matrix R in SO(3).
    Decouples breathing sway and involuntary head micro-tremor (< threshold_deg)
    from intentional head turns.
    """

    def __init__(self, threshold_deg: float = 0.8, alpha: float = 0.25):
        # In SO(3), Frobenius norm ||R1 - R2||_F approx sqrt(2) * angle_rad for small angles
        self.threshold = math.sqrt(2.0) * (threshold_deg * math.pi / 180.0)
        self.alpha = float(alpha)
        self.locked_R: Optional[np.ndarray] = None

    def filter(self, R: np.ndarray) -> np.ndarray:
        if self.locked_R is None:
            self.locked_R = np.copy(R)
            return R

        diff = float(np.linalg.norm(R - self.locked_R, "fro"))
        if diff < self.threshold:
            # Within deadband: hold locked orientation (eliminates breathing wobble)
            return self.locked_R
        else:
            # Intentional rotation: smoothly ease toward new orientation
            blended = self.locked_R + self.alpha * (R - self.locked_R)
            u, _, vt = np.linalg.svd(blended)
            self.locked_R = u @ vt
            return self.locked_R

    def reset(self):
        self.locked_R = None


class IVTGazeFilter2D:
    """
    I-VT (Velocity-Threshold Identification) state machine filter for 2D screen gaze.
    Commercial eye-trackers classify eye movements into Fixations and Saccades based on
    angular velocity:
    - Fixation (velocity < velocity_threshold_deg_s): Lock to the spatial centroid
      of the fixation cluster. Eye micro-tremor and sensor noise are zeroed out.
    - Saccade (velocity >= velocity_threshold_deg_s): Release lock immediately
      and track with zero lag.
    - Hysteresis: Transitions from Saccade back to Fixation require sustained low velocity
      for min_fixation_sec (e.g. 100ms) to avoid chatter.
    """

    STATE_FIXATION = "FIXATION"
    STATE_SACCADE = "SACCADE"

    def __init__(
        self,
        velocity_threshold_deg_s: float = 35.0,
        min_fixation_sec: float = 0.10,
        pixel_pitch_cm: float = 59.7 / 2560.0,
        default_distance_cm: float = 60.0,
        centroid_alpha: float = 0.04
    ):
        self.vel_thresh = float(velocity_threshold_deg_s)
        self.min_fixation_sec = float(min_fixation_sec)
        self.pixel_pitch_cm = float(pixel_pitch_cm)
        self.default_distance_cm = float(default_distance_cm)
        self.centroid_alpha = float(centroid_alpha)

        self.state = self.STATE_FIXATION
        self.t_prev: Optional[float] = None
        self.x_prev: Optional[float] = None
        self.y_prev: Optional[float] = None

        # Fixation centroid
        self.centroid_x: Optional[float] = None
        self.centroid_y: Optional[float] = None

        # Saccade -> Fixation dwell timer
        self.low_vel_start_t: Optional[float] = None

        # Diagnostics
        self.current_vel_deg_s: float = 0.0

        # Fast 1-Euro filter during saccades for clean arrival without overshoot
        self.saccade_fx: Optional[OneEuroFilter] = None
        self.saccade_fy: Optional[OneEuroFilter] = None

    def filter(self, t: float, x: float, y: float, distance_cm: Optional[float] = None) -> Tuple[float, float]:
        dist_cm = distance_cm if distance_cm and distance_cm > 15.0 else self.default_distance_cm

        if self.t_prev is None or self.x_prev is None or self.y_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.y_prev = y
            self.centroid_x = x
            self.centroid_y = y
            self.state = self.STATE_FIXATION
            self.low_vel_start_t = t
            self.current_vel_deg_s = 0.0
            self.saccade_fx = OneEuroFilter(t, x, min_cutoff=4.0, beta=0.1)
            self.saccade_fy = OneEuroFilter(t, y, min_cutoff=4.0, beta=0.1)
            return x, y

        dt = t - self.t_prev
        if dt <= 0.0:
            return (self.centroid_x if self.state == self.STATE_FIXATION and self.centroid_x is not None else x,
                    self.centroid_y if self.state == self.STATE_FIXATION and self.centroid_y is not None else y)
        if dt > 0.25:
            # Reconnection or post-blink resumption
            self.reset()
            return self.filter(t, x, y, distance_cm)

        # 1. Screen Euclidean displacement in cm
        dx_px = x - self.x_prev
        dy_px = y - self.y_prev
        disp_cm = math.hypot(dx_px, dy_px) * self.pixel_pitch_cm

        # 2. Visual angle displacement in degrees
        angle_deg = math.degrees(2.0 * math.atan2(disp_cm / 2.0, dist_cm))
        vel_deg_s = angle_deg / dt
        self.current_vel_deg_s = vel_deg_s

        self.t_prev = t
        self.x_prev = x
        self.y_prev = y

        # 3. State Machine Transition Logic
        if self.state == self.STATE_FIXATION:
            if vel_deg_s >= self.vel_thresh:
                # Saccade onset! Release fixation lock immediately
                self.state = self.STATE_SACCADE
                self.low_vel_start_t = None
                self.saccade_fx = OneEuroFilter(t, x, min_cutoff=4.0, beta=0.1)
                self.saccade_fy = OneEuroFilter(t, y, min_cutoff=4.0, beta=0.1)
                return self.saccade_fx(t, x), self.saccade_fy(t, y)
            else:
                # Steady fixation: update running centroid gently to accommodate slow pursuit/drift
                if self.centroid_x is None or self.centroid_y is None:
                    self.centroid_x = x
                    self.centroid_y = y
                else:
                    self.centroid_x = (1.0 - self.centroid_alpha) * self.centroid_x + self.centroid_alpha * x
                    self.centroid_y = (1.0 - self.centroid_alpha) * self.centroid_y + self.centroid_alpha * y
                return self.centroid_x, self.centroid_y

        else:  # STATE_SACCADE
            if self.saccade_fx is None or self.saccade_fy is None:
                self.saccade_fx = OneEuroFilter(t, x, min_cutoff=4.0, beta=0.1)
                self.saccade_fy = OneEuroFilter(t, y, min_cutoff=4.0, beta=0.1)
            out_x = self.saccade_fx(t, x)
            out_y = self.saccade_fy(t, y)

            if vel_deg_s < self.vel_thresh:
                if self.low_vel_start_t is None:
                    self.low_vel_start_t = t
                elif (t - self.low_vel_start_t) >= self.min_fixation_sec:
                    # Sustained low velocity confirmed: lock new fixation centroid
                    self.state = self.STATE_FIXATION
                    self.centroid_x = out_x
                    self.centroid_y = out_y
            else:
                self.low_vel_start_t = None

            return out_x, out_y

    def reset(self):
        self.t_prev = None
        self.x_prev = None
        self.y_prev = None
        self.centroid_x = None
        self.centroid_y = None
        self.low_vel_start_t = None
        self.state = self.STATE_FIXATION
        self.current_vel_deg_s = 0.0
        self.saccade_fx = None
        self.saccade_fy = None


class GazeFilter2D:
    """Pair of 1-Euro filters for 2D screen coordinates (X, Y)."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.fx = None
        self.fy = None

    def filter(self, t: float, x: float, y: float, **kwargs) -> tuple[float, float]:
        if self.fx is None or self.fy is None:
            self.fx = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta)
            self.fy = OneEuroFilter(t, y, min_cutoff=self.min_cutoff, beta=self.beta)
            return x, y
        return self.fx(t, x), self.fy(t, y)

    def reset(self):
        self.fx = None
        self.fy = None

