#!/usr/bin/env python3
"""
Distance sweep: calibrate MediaPipe's head distance against Kinect depth.

MediaPipe's facial transformation matrix assumes its own virtual camera, not the Kinect v2
colour lens (84 x 54 deg field of view), so its distance comes out scaled. This records
MediaPipe distance, Kinect depth and iris size while you move from as close as possible to as
far as possible, then fits the mapping.

  uv run python distance_sweep.py                 # record (needs kinect2pipe --ir), then analyse
  uv run python distance_sweep.py analyze DIR     # re-analyse a recording

Output: logs/distance_sweep_<time>/samples.csv and fit.json
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NOSE_TO_EYES_CM = 2.0  # depth measures the nose tip; the eyes are about this much further back
# Kinect v2 colour camera nominal focal length (px, 1920x1080) and mean human iris diameter (cm)
COLOR_FX_PX = 1081.37
IRIS_DIAMETER_CM = 1.17
FIELDS = ["t", "mp_dist_cm", "Px", "Py", "Pz", "depth_cm", "iris_px", "face_w_norm", "head_yaw_deg", "head_pitch_deg"]


def iris_diameter_px(landmarks, w: int, h: int) -> float:
    """Mean horizontal iris diameter of both eyes (landmarks 469/471 and 474/476)."""
    def d(a, b):
        return float(np.hypot((landmarks[a].x - landmarks[b].x) * w, (landmarks[a].y - landmarks[b].y) * h))
    return (d(469, 471) + d(474, 476)) / 2.0


def record() -> Path:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    from gaze_model import head_angles
    from shm_client import IrDepthSource, KinectFrameSource, face_depth_cm
    from tracker import MODEL_PATH, ensure_model_exists

    ensure_model_exists()
    src, ir = KinectFrameSource(), IrDepthSource()
    keepalive = cv2.VideoCapture("/dev/video10")
    if not ir.available():
        print("No IR/depth stream: start kinect2pipe with --ir first.")
        sys.exit(1)
    lm = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH), num_faces=1,
        output_facial_transformation_matrixes=True))

    out = HERE / "logs" / time.strftime("distance_sweep_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    f = open(out / "samples.csv", "w", newline="")
    wr = csv.DictWriter(f, fieldnames=FIELDS)
    wr.writeheader()

    W, H = 2560, 1440
    win = "Distance sweep"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    depths, n = [], 0
    last = None
    while True:
        ok, frame, _ = src.read()
        if ok:
            fh, fw = frame.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            ok_ir, _irf, depth, _ = ir.read()
            if res.face_landmarks and res.facial_transformation_matrixes and ok_ir:
                L = res.face_landmarks[0]
                M = np.array(res.facial_transformation_matrixes[0])
                xs, ys = [p.x for p in L], [p.y for p in L]
                dcm = face_depth_cm(depth, (min(xs), min(ys), max(xs), max(ys)))
                yaw, pitch = np.degrees(head_angles(M[:3, :3]))
                row = {"t": time.time(), "mp_dist_cm": -M[2, 3], "Px": M[0, 3], "Py": M[1, 3], "Pz": M[2, 3],
                       "depth_cm": "" if dcm is None else dcm, "iris_px": iris_diameter_px(L, fw, fh),
                       "face_w_norm": max(xs) - min(xs), "head_yaw_deg": yaw, "head_pitch_deg": pitch}
                wr.writerow(row)
                n += 1
                last = row
                if dcm is not None:
                    depths.append(dcm + NOSE_TO_EYES_CM)

        canvas = np.full((H, W, 3), 20, np.uint8)
        cv2.circle(canvas, (W // 2, H // 2), 10, (255, 255, 255), -1)
        lines = [
            "DISTANCE SWEEP - look at the centre dot",
            "1. Slowly lean IN as close as you can while still seeing the whole screen",
            "2. Then slowly move BACK as far as your chair allows",
            "3. Come back to normal, press [q]. Go slowly; a couple of passes is better than one.",
        ]
        for i, s in enumerate(lines):
            cv2.putText(canvas, s, (60, 90 + 50 * i), cv2.FONT_HERSHEY_DUPLEX if i == 0 else cv2.FONT_HERSHEY_SIMPLEX,
                        1.1 if i == 0 else 0.9, (255, 255, 255) if i == 0 else (200, 200, 200), 2)
        if depths:
            lo, hi = min(depths), max(depths)
            cv2.putText(canvas, f"covered: {lo:.0f} - {hi:.0f} cm   ({n} frames)", (60, H - 160),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 160), 2)
            x0, x1 = 60, W - 60
            to_x = lambda d: int(x0 + (np.clip(d, 30, 120) - 30) / 90 * (x1 - x0))  # noqa: E731
            cv2.rectangle(canvas, (x0, H - 120), (x1, H - 90), (70, 70, 70), -1)
            cv2.rectangle(canvas, (to_x(lo), H - 120), (to_x(hi), H - 90), (0, 200, 140), -1)
            cv2.line(canvas, (to_x(depths[-1]), H - 130), (to_x(depths[-1]), H - 80), (255, 255, 255), 3)
            for d in range(30, 121, 10):
                cv2.putText(canvas, f"{d}", (to_x(d) - 15, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        if last is not None and last["depth_cm"] != "":
            cv2.putText(canvas, f"now: Kinect depth {last['depth_cm'] + NOSE_TO_EYES_CM:5.1f} cm   MediaPipe "
                        f"{last['mp_dist_cm']:5.1f} cm", (60, H - 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 255), 2)
        cv2.imshow(win, canvas)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    f.close()
    keepalive.release()
    cv2.destroyAllWindows()
    print(f"recorded {n} frames -> {out}")
    return out


KINECT_MIN_VALID_CM = 58.0  # depth spec minimum is 50 cm; below ~55 cm it reads the headset/background


def analyze(out: Path) -> dict:
    rows = [r for r in csv.DictReader(open(out / "samples.csv"))]
    mp_all = np.array([float(r["mp_dist_cm"]) for r in rows])
    iris_all = COLOR_FX_PX * IRIS_DIAMETER_CM / np.array([float(r["iris_px"]) for r in rows])
    yaw_all = np.abs([float(r["head_yaw_deg"]) for r in rows])
    depth_all = np.array([float(r["depth_cm"]) + NOSE_TO_EYES_CM if r["depth_cm"] not in ("", "nan") else np.nan
                          for r in rows])

    # Valid depth: inside the sensor's working range and not the wall/headset (robust ratio band).
    valid = (depth_all >= KINECT_MIN_VALID_CM) & (depth_all < 200) & (mp_all > 5) & (yaw_all < 20)
    r0 = np.nanmedian(depth_all[valid] / mp_all[valid])
    valid &= np.abs(depth_all / mp_all / r0 - 1) < 0.2
    true, mpd, iris = depth_all[valid], mp_all[valid], iris_all[valid]
    if len(true) < 50:
        print(f"only {len(true)} frames with valid depth (>= {KINECT_MIN_VALID_CM:.0f} cm); sit further back")
        return {}

    def rmse(pred, ref=true):
        return float(np.sqrt(np.mean((pred - ref) ** 2)))

    k = float(np.median(true / mpd))
    a, b = np.polyfit(mpd, true, 1)
    k_iris = float(np.median(true / iris))

    # Close range has no valid depth: use the iris-size distance (scaled on the valid range) as an
    # independent reference to check whether the MediaPipe mapping extrapolates.
    close = (mp_all > 5) & (yaw_all < 20) & ~valid & (mp_all < mpd.min())
    close_check = None
    if close.sum() >= 20:
        ref = k_iris * iris_all[close]
        close_check = {"n": int(close.sum()), "mp_range_cm": [float(mp_all[close].min()), float(mp_all[close].max())],
                       "iris_ref_range_cm": [float(ref.min()), float(ref.max())],
                       "scale_rmse_vs_iris_cm": rmse(k * mp_all[close], ref),
                       "affine_rmse_vs_iris_cm": rmse(a * mp_all[close] + b, ref),
                       "raw_rmse_vs_iris_cm": rmse(mp_all[close], ref)}

    per_bin = []
    for lo in np.arange(np.floor(true.min() / 10) * 10, true.max(), 10):
        m = (true >= lo) & (true < lo + 10)
        if m.sum() >= 10:
            per_bin.append({"range_cm": [float(lo), float(lo + 10)], "n": int(m.sum()),
                            "ratio_median": float(np.median(true[m] / mpd[m])),
                            "ratio_iqr": float(np.subtract(*np.percentile(true[m] / mpd[m], [75, 25])))})

    fit = {
        "camera": "Kinect v2 colour, 1920x1080 (kinect2pipe)", "mediapipe": "FaceLandmarker facial_transformation_matrix",
        "frames_valid_depth": int(len(true)), "true_range_cm": [float(true.min()), float(true.max())],
        "scale": {"k": k, "rmse_cm": rmse(k * mpd)},
        "affine": {"a": float(a), "b": float(b), "rmse_cm": rmse(a * mpd + b)},
        "raw_mediapipe_rmse_cm": rmse(mpd),
        "iris_size": {"scale": k_iris, "rmse_cm": rmse(k_iris * iris)},
        "close_range_check": close_check,
        "per_10cm": per_bin,
        "notes": (f"true = Kinect depth to nose tip + {NOSE_TO_EYES_CM} cm; depth used only >= {KINECT_MIN_VALID_CM} cm "
                  "and within 20% of the median ratio; |head yaw| < 20 deg"),
    }
    (out / "fit.json").write_text(json.dumps(fit, indent=2))

    print(f"\n{len(true)} frames with valid depth, true distance {true.min():.0f}-{true.max():.0f} cm")
    print(f"  MediaPipe as-is                 : RMSE {fit['raw_mediapipe_rmse_cm']:5.2f} cm")
    print(f"  scale  true = {k:.3f} * mp        : RMSE {fit['scale']['rmse_cm']:5.2f} cm")
    print(f"  affine true = {a:.3f} * mp {b:+.1f}  : RMSE {fit['affine']['rmse_cm']:5.2f} cm")
    print(f"  iris size x{k_iris:.3f}               : RMSE {fit['iris_size']['rmse_cm']:5.2f} cm")
    print("  ratio per 10 cm: " + "  ".join(f"{p['range_cm'][0]:.0f}-{p['range_cm'][1]:.0f}: {p['ratio_median']:.3f}"
                                             for p in per_bin))
    if close_check:
        c = close_check
        print(f"  close range ({c['n']} frames, MediaPipe {c['mp_range_cm'][0]:.0f}-{c['mp_range_cm'][1]:.0f} cm, "
              f"no valid depth) vs iris reference {c['iris_ref_range_cm'][0]:.0f}-{c['iris_ref_range_cm'][1]:.0f} cm:")
        print(f"      raw RMSE {c['raw_rmse_vs_iris_cm']:.1f} cm, scale RMSE {c['scale_rmse_vs_iris_cm']:.1f} cm, "
              f"affine RMSE {c['affine_rmse_vs_iris_cm']:.1f} cm")
    print(f"saved {out / 'fit.json'}")
    return fit


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "analyze":
        analyze(Path(sys.argv[2]))
    else:
        analyze(record())
