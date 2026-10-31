#!/usr/bin/env python3
"""
Kinect v2 Eye & Head Gaze Tracker
- 6-DoF head pose + eye-in-head angles (gaze_model.py) ray-cast onto the screen plane.
- Selectable tracking modes: HYBRID (Head + Eyes), HEAD_ONLY (Nose pointer), EYE_ONLY.
- Fullscreen 9-point + head-sweep calibration; [v] validation run logs accuracy vs head pose.
- 1-Euro adaptive low-pass filter for rock-steady fixations and instant saccades.
- 10-Channel Lab Streaming Layer (LSL) broadcast for neuroscience/BCI synchronization.
"""

import os

# Universal Qt font config: use OS fontconfig and silence missing font warnings
os.environ.setdefault("QT_QPA_FONTDIR", "")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import sys
import time
import math
import argparse
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("Error: mediapipe is required. Run via: uv run tracker.py")
    sys.exit(1)

from shm_client import KinectFrameSource, IrDepthSource, face_depth_cm
from calibration import GazeCalibrationEngine
from filters import GazeFilter2D, IVTGazeFilter2D, HeadRotationDeadband
from gaze_model import BLINK_THRESHOLD, GazeSample, ScreenGeometry, eye_features, head_angles
from validation import ValidationSession
from session_recorder import SessionRecorder

WINDOW_NAME = "Kinect Gaze Tracking"

# Screen physical dimensions (DP-1: 2560x1440, 59.7cm x 33.6cm)
SCREEN_W = 2560
SCREEN_H = 1440
SCREEN_W_CM = 59.7
SCREEN_H_CM = 33.6
# Kinect colour lens position relative to screen centre (cm, negative = left), via screen_ruler.py
CAMERA_X_CM = -9.4
# MediaPipe's head distance assumes its own virtual camera, not the Kinect v2 colour lens.
# Measured with distance_sweep.py against Kinect depth (2026-10-31, 58-114 cm, RMSE 1.9 cm vs 16.6 raw):
# true distance = 1.248 x MediaPipe. Applied to the whole head position (a pure perspective scale).
MEDIAPIPE_DIST_SCALE = 1.248


MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def ensure_model_exists():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print(f"Downloading MediaPipe Face Landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# KinectGaze channels, in push order. Labels match nautilus_bci's recorder (KNOWN_CHANNELS).
LSL_CHANNELS = [
    ("gaze_x", "px"), ("gaze_y", "px"), ("head_x", "px"), ("head_y", "px"),
    ("head_yaw", "deg"), ("head_pitch", "deg"), ("head_roll", "deg"),
    ("head_dist", "cm"), ("blink_l", "score"), ("blink_r", "score"),
]


def make_lsl_info(screen: ScreenGeometry):
    """StreamInfo for KinectGaze with channel labels/units and the screen geometry needed to
    turn the pixel channels back into angles."""
    from pylsl import StreamInfo
    info = StreamInfo(name="KinectGaze", type="Gaze", channel_count=len(LSL_CHANNELS),
                      nominal_srate=30, channel_format="float32", source_id="kinect_v2_eyetracker")
    desc = info.desc()
    desc.append_child_value("manufacturer", "Microsoft Kinect v2 + MediaPipe FaceLandmarker")
    chans = desc.append_child("channels")
    for label, unit in LSL_CHANNELS:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "Gaze")
    geo = desc.append_child("screen")
    for key, value in (("width_px", screen.w_px), ("height_px", screen.h_px), ("width_cm", screen.w_cm),
                       ("height_cm", screen.h_cm), ("camera_x_cm", screen.cam_x_cm),
                       ("mediapipe_dist_scale", MEDIAPIPE_DIST_SCALE)):
        geo.append_child_value(key, str(value))
    desc.append_child_value("note", "gaze/head px are screen pixels (0,0 = top-left); head_yaw/pitch fixed "
                                    "2026-10-31 (earlier recordings had them swapped)")
    return info


class KinectEyeTrackerApp:
    def __init__(self, device: str = "/dev/video10", enable_lsl: bool = False, fullscreen: bool = True,
                 headless: bool = False, use_sweep: bool = True, record_dir: str = None):
        ensure_model_exists()
        self.device = device
        self.enable_lsl = enable_lsl
        self.fullscreen = fullscreen and not headless
        self.headless = headless

        # Ingestion & Calibration
        self.source = KinectFrameSource(v4l2_device=device)
        self.screen = ScreenGeometry(SCREEN_W, SCREEN_H, SCREEN_W_CM, SCREEN_H_CM, CAMERA_X_CM)
        self.calibrator = GazeCalibrationEngine(screen=self.screen, dwell_sec=0.9, use_sweep=use_sweep)
        self.validation = ValidationSession(self.screen)
        # Optional parallel IR + depth stream (kinect2pipe --ir): gives true face distance
        self.ir_source = IrDepthSource()
        self.depth_cm = None
        self.recorder = (SessionRecorder(record_dir, self.screen,
                                         extra_meta={"mediapipe_dist_scale_applied": MEDIAPIPE_DIST_SCALE})
                         if record_dir else None)
        if self.recorder:
            print(f"Recording session video + frames to {record_dir}")
            if not self.ir_source.available():
                print("  (no IR/depth stream: start kinect2pipe with --ir to log true face distance)")
        self.gaze_filter = IVTGazeFilter2D(velocity_threshold_deg_s=35.0, min_fixation_sec=0.10)
        self.head_filter = GazeFilter2D(min_cutoff=0.8, beta=0.03)
        self.head_deadband = HeadRotationDeadband(threshold_deg=0.8, alpha=0.25)

        # 1-click recenter / drift nulling offsets
        self.drift_offset_x = 0.0
        self.drift_offset_y = 0.0

        # Vision model
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # Tracking Mode: 0=HYBRID (Head+Eye), 1=HEAD_ONLY (Nose Pointer), 2=EYE_ONLY
        self.modes = ["HYBRID (Head + Eyes)", "HEAD-ONLY (Nose Pointer)", "EYE-ONLY (Fixed Head)"]
        self.mode_idx = 0

        # LSL Outlet
        self.lsl_outlet = None
        if self.enable_lsl:
            self._init_lsl()

        # Canvas & trails
        self.canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        self.show_pip = False
        self.recent_gaze_trail = []
        self.recent_head_trail = []

    def _init_lsl(self):
        try:
            from pylsl import StreamOutlet
            self.lsl_outlet = StreamOutlet(make_lsl_info(self.screen))
            print("Lab Streaming Layer (LSL) outlet 'KinectGaze' (10 channels) initialized.")
        except ImportError:
            print("pylsl not available; continuing without LSL outlet.")

    def extract_3d_features(self, frame_bgr: np.ndarray):
        """
        Extracts 6-DoF Head Pose and Eye Gaze vectors in metric 3D space.
        Returns: Compound gaze ray and direct head forward ray intersected with screen plane.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)
        if not result or not result.face_landmarks or not result.facial_transformation_matrixes:
            return None

        landmarks = result.face_landmarks[0]
        blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
        mat = np.array(result.facial_transformation_matrixes[0])

        # 6-DoF Head Pose (MediaPipe metric space, cm)
        R_head_raw = mat[:3, :3]
        P_head = mat[:3, 3].astype(np.float64) * MEDIAPIPE_DIST_SCALE

        # Decouple postural breathing sway from intentional head rotation via angular deadband
        R_head = self.head_deadband.filter(R_head_raw)

        # Head angles from the forward axis (yaw = left/right, pitch = up/down) + roll about it.
        # (The previous Euler extraction had yaw and pitch swapped.)
        hyaw, hpitch = head_angles(R_head)
        head_yaw_deg, head_pitch_deg = math.degrees(hyaw), math.degrees(hpitch)
        head_roll_deg = math.degrees(math.atan2(R_head[1, 0], R_head[0, 0]))

        blink_l = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_r = blendshapes.get("eyeBlinkRight", 0.0)
        is_blinking = (blink_l > BLINK_THRESHOLD and blink_r > BLINK_THRESHOLD)

        # Iris offset in each eye's own corner-aligned frame (roll/aspect invariant), monocular fallback
        feat = eye_features(landmarks, w, h, blink_l, blink_r)
        sample = GazeSample(feat=feat, R=R_head, P=P_head)

        gaze_px = self.calibrator.predict_px(sample)
        head_px = self.calibrator.model.head_px(sample)
        if gaze_px is None or head_px is None:
            return None

        r_iris = (int(landmarks[468].x * w), int(landmarks[468].y * h))
        l_iris = (int(landmarks[473].x * w), int(landmarks[473].y * h))
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        return {
            "sample": sample,
            "gaze_px": gaze_px,
            "head_px": head_px,
            "is_blinking": is_blinking,
            "blink_l": blink_l,
            "blink_r": blink_r,
            "head_yaw": head_yaw_deg,
            "head_pitch": head_pitch_deg,
            "head_roll": head_roll_deg,
            "head_dist_cm": float(-P_head[2]),
            "face_bbox": (min(xs), min(ys), max(xs), max(ys)),
            "r_iris": r_iris,
            "l_iris": l_iris,
        }

    def setup_window(self):
        if self.headless:
            return
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
        if not self.fullscreen:
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    def cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        print(f"Tracking Mode changed to: {self.modes[self.mode_idx]}")

    def run(self):
        self.setup_window()
        self.calibrator.start()

        print("\n=== Kinect Gaze & Head Tracker Running ===")
        print("  Controls:")
        print("    [h]               : Cycle Tracking Mode (Hybrid / Head-Only / Eye-Only)")
        print("    [Enter] / [Space] : Lock current calibration target")
        print("    [c]               : Restart calibration (9 points + head sweep)")
        print("    [v]               : Validation run (accuracy vs head pose / distance, saved to logs/)")
        print("    [r]               : 1-Click Recenter / Drift Nulling (Look at screen center)")
        print("    [f]               : Toggle Fullscreen / Windowed")
        print("    [p]               : Toggle Picture-in-Picture webcam feed")
        print("    [q] / [ESC]       : Quit")
        print("==========================================\n")

        last_features = None

        while True:
            t_now = time.time()
            success, frame, ts = self.source.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            features = self.extract_3d_features(frame)
            if features is not None:
                last_features = features
                if self.ir_source.available():
                    ok_ir, _ir, depth_mm, _ts = self.ir_source.read()
                    if ok_ir:
                        self.depth_cm = face_depth_cm(depth_mm, features["face_bbox"])

            sample = features["sample"] if features else None
            is_blinking = features["is_blinking"] if features else True

            # Calibration stage
            if self.calibrator.is_active:
                cal_finished = self.calibrator.update(sample, is_blinking)
                if cal_finished:
                    print("\nCalibration completed successfully! Transitioning to live screen tracking.")
                    self.gaze_filter.reset()
                    self.head_filter.reset()
                    self.recent_gaze_trail.clear()
                    self.recent_head_trail.clear()
                else:
                    self.calibrator.render(self.canvas)

            else:
                # Live screen tracking stage
                if not self.headless:
                    self.canvas.fill(16)  # Dark slate background

                smooth_gx, smooth_gy = 0.0, 0.0
                smooth_hx, smooth_hy = 0.0, 0.0

                if features and not features["is_blinking"]:
                    # 1. Gaze tracking with I-VT fixation-saccade classification & drift nulling
                    raw_gx, raw_gy = features["gaze_px"]
                    cal_gx = raw_gx + self.drift_offset_x
                    cal_gy = raw_gy + self.drift_offset_y
                    smooth_gx, smooth_gy = self.gaze_filter.filter(
                        t_now, cal_gx, cal_gy, distance_cm=features["head_dist_cm"]
                    )
                    if self.validation.is_active:
                        self.validation.record(t_now, features, (cal_gx, cal_gy), (smooth_gx, smooth_gy), self.depth_cm)

                    # 2. Head tracking (nose pointer): purely geometric, not passed through the eye calibration
                    raw_hx, raw_hy = features["head_px"]
                    smooth_hx, smooth_hy = self.head_filter.filter(t_now, raw_hx, raw_hy)

                    # Update history trails
                    self.recent_gaze_trail.append((int(smooth_gx), int(smooth_gy)))
                    if len(self.recent_gaze_trail) > 16:
                        self.recent_gaze_trail.pop(0)

                    self.recent_head_trail.append((int(smooth_hx), int(smooth_hy)))
                    if len(self.recent_head_trail) > 16:
                        self.recent_head_trail.pop(0)

                    # Broadcast 10 channels to LSL if enabled
                    if self.lsl_outlet is not None:
                        self.lsl_outlet.push_sample([
                            float(smooth_gx), float(smooth_gy),
                            float(smooth_hx), float(smooth_hy),
                            float(features["head_yaw"]), float(features["head_pitch"]),
                            float(features["head_roll"]), float(features["head_dist_cm"]),
                            float(features["blink_l"]), float(features["blink_r"])
                        ])

                # Active reticle position depends on mode:
                # 0: Hybrid (Gaze), 1: Head-Only (Head Aim), 2: Eye-Only
                active_pt = None
                if self.mode_idx == 0 and self.recent_gaze_trail:
                    active_pt = self.recent_gaze_trail[-1]
                elif self.mode_idx == 1 and self.recent_head_trail:
                    active_pt = self.recent_head_trail[-1]
                elif self.mode_idx == 2 and self.recent_gaze_trail:
                    active_pt = self.recent_gaze_trail[-1]

                # Draw subtle gaze trail
                for i in range(1, len(self.recent_gaze_trail)):
                    alpha = i / len(self.recent_gaze_trail)
                    thickness = int(1 + alpha * 3)
                    color = (int(0 * alpha), int(160 * alpha), int(255 * alpha))
                    cv2.line(self.canvas, self.recent_gaze_trail[i - 1], self.recent_gaze_trail[i], color, thickness)

                # If in Hybrid mode, draw dashed line between Head Pointer and Eye Gaze
                if self.mode_idx == 0 and self.recent_gaze_trail and self.recent_head_trail:
                    gx, gy = self.recent_gaze_trail[-1]
                    hx, hy = self.recent_head_trail[-1]
                    cv2.line(self.canvas, (hx, hy), (gx, gy), (80, 80, 160), 1)

                    # Draw Head Aim reticle (Magenta)
                    cv2.circle(self.canvas, (hx, hy), 14, (200, 50, 200), 1)
                    cv2.line(self.canvas, (hx - 10, hy), (hx + 10, hy), (200, 50, 200), 1)
                    cv2.line(self.canvas, (hx, hy - 10), (hx, hy + 10), (200, 50, 200), 1)

                # Draw main active reticle
                if active_pt:
                    ax, ay = active_pt
                    reticle_color = (0, 220, 255) if self.mode_idx != 1 else (255, 100, 255)
                    cv2.circle(self.canvas, (ax, ay), 24, reticle_color, 2)
                    cv2.circle(self.canvas, (ax, ay), 6, (0, 255, 120), -1)
                    cv2.line(self.canvas, (ax - 18, ay), (ax + 18, ay), reticle_color, 1)
                    cv2.line(self.canvas, (ax, ay - 18), (ax, ay + 18), reticle_color, 1)

                    coord_str = f"Target Point: ({ax}, {ay}) px  |  Mode: {self.modes[self.mode_idx]}"
                    cv2.putText(self.canvas, coord_str, (50, 60), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

                    if last_features:
                        depth_str = f" | Depth={self.depth_cm:.1f} cm" if self.depth_cm else ""
                        pose_str = (f"Head: Yaw={last_features['head_yaw']:+.1f} "
                                    f"Pitch={last_features['head_pitch']:+.1f} "
                                    f"Roll={last_features['head_roll']:+.1f} deg | "
                                    f"Dist={last_features['head_dist_cm']:.1f} cm{depth_str}")
                        cv2.putText(self.canvas, pose_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 2)

                    # Display I-VT classification state and angular gaze velocity
                    state_name = getattr(self.gaze_filter, "state", "TRACKING")
                    vel_deg_s = getattr(self.gaze_filter, "current_vel_deg_s", 0.0)
                    state_color = (0, 255, 120) if state_name == "FIXATION" else (0, 165, 255)
                    ivt_str = f"[{state_name}] Velocity: {vel_deg_s:5.1f}°/s | Drift: ({self.drift_offset_x:+.0f}, {self.drift_offset_y:+.0f}) px"
                    cv2.putText(self.canvas, ivt_str, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_color, 2)

                # Bottom HUD
                hud_str = "[h] Mode  |  [c] Recalibrate  |  [v] Validate  |  [r] Recenter Gaze  |  [f] Fullscreen  |  [p] PiP  |  [q] Quit"
                cv2.putText(self.canvas, hud_str, (50, SCREEN_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1)

            if self.validation.is_active and not self.calibrator.is_active:
                if self.validation.tick(t_now):
                    print(self.validation.finish())
                elif not self.headless:
                    self.validation.render(self.canvas)

            if self.recorder is not None:
                if self.calibrator.is_active:
                    phase = "cal_" + self.calibrator.phase
                    target, onset = self.calibrator.get_current_target_px(), self.calibrator.onset_t
                elif self.validation.is_active:
                    phase = "val_" + self.validation.plan[self.validation.idx][0]
                    target, onset = self.validation.target_px(), self.validation.step_t0
                else:
                    phase, target, onset = "live", None, t_now
                self.recorder.write(frame, t_now, phase, target, t_now - onset, features,
                                    self.depth_cm if features is not None else None)

            # Draw Picture-in-Picture webcam feed only during live tracking (never during calibration)
            if self.show_pip and not self.calibrator.is_active:
                pip_w = 320
                pip_h = 180
                pip_frame = cv2.resize(frame, (pip_w, pip_h))

                if last_features:
                    scale_x = pip_w / frame.shape[1]
                    scale_y = pip_h / frame.shape[0]
                    if last_features["r_iris"]:
                        rx = int(last_features["r_iris"][0] * scale_x)
                        ry = int(last_features["r_iris"][1] * scale_y)
                        cv2.circle(pip_frame, (rx, ry), 3, (0, 255, 0), -1)
                    if last_features["l_iris"]:
                        lx = int(last_features["l_iris"][0] * scale_x)
                        ly = int(last_features["l_iris"][1] * scale_y)
                        cv2.circle(pip_frame, (lx, ly), 3, (0, 255, 0), -1)

                cv2.rectangle(pip_frame, (0, 0), (pip_w - 1, pip_h - 1), (80, 80, 80), 2)
                x_offset = SCREEN_W - pip_w - 40
                y_offset = SCREEN_H - pip_h - 40
                self.canvas[y_offset:y_offset + pip_h, x_offset:x_offset + pip_w] = pip_frame

            if not self.headless:
                cv2.imshow(WINDOW_NAME, self.canvas)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key in (10, 13, ord(' ')):  # Enter or Space
                if self.calibrator.is_active:
                    if self.calibrator.manual_capture(sample):
                        print("Calibration completed via manual lock!")
                        self.gaze_filter.reset()
                        self.head_filter.reset()
                        self.recent_gaze_trail.clear()
                        self.recent_head_trail.clear()
            elif key == ord('h'):
                self.cycle_mode()
            elif key == ord('v'):
                if self.calibrator.is_calibrated and not self.calibrator.is_active:
                    self.gaze_filter.reset()
                    self.validation.start(t_now)
                else:
                    print("Calibrate first ([c]) before running validation.")
            elif key == ord('c'):
                self.drift_offset_x = 0.0
                self.drift_offset_y = 0.0
                self.calibrator.start()
            elif key == ord('r'):
                # 1-Click Recenter / Drift Nulling: while looking at screen center, nulls drift
                if self.recent_gaze_trail and not self.calibrator.is_active:
                    curr_x, curr_y = self.recent_gaze_trail[-1]
                    self.drift_offset_x += (SCREEN_W / 2.0) - curr_x
                    self.drift_offset_y += (SCREEN_H / 2.0) - curr_y
                    self.gaze_filter.reset()
                    print(f"Gaze recentered to screen center! Cumulative offset: ({self.drift_offset_x:+.1f}, {self.drift_offset_y:+.1f}) px")
            elif key == ord('f'):
                self.toggle_fullscreen()
            elif key == ord('p'):
                self.show_pip = not self.show_pip

        if self.validation.is_active:
            print(self.validation.finish())
        self.source.release()
        self.ir_source.release()
        if self.recorder is not None:
            self.recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinect v2 Eye & Head Gaze Tracker")
    parser.add_argument("--device", default="/dev/video10", help="V4L2 fallback device")
    parser.add_argument("--lsl", action="store_true", help="Publish gaze stream to Lab Streaming Layer")
    parser.add_argument("--windowed", action="store_true", help="Start in windowed mode instead of fullscreen")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the head-sweep calibration phase")
    parser.add_argument("--record", nargs="?", const="auto", metavar="DIR",
                        help="Record colour video + per-frame metadata for offline models (default dir: logs/session_<time>)")
    args = parser.parse_args()

    app = KinectEyeTrackerApp(
        device=args.device,
        enable_lsl=args.lsl,
        fullscreen=not args.windowed,
        headless=args.headless,
        use_sweep=not args.no_sweep,
        record_dir=(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", time.strftime("session_%Y%m%d_%H%M%S"))
                    if args.record == "auto" else args.record),
    )
    app.run()
