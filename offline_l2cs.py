# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "safetensors>=0.4",
#     "opencv-python-headless>=4.9",
#     "numpy>=1.26",
# ]
# ///
"""
Offline L2CS-Net gaze on a recorded session (tracker.py --record), compared with the MediaPipe
iris model on exactly the same frames.

  uv run offline_l2cs.py logs/session_YYYYmmdd_HHMMSS            # infer (cached) + evaluate
  uv run offline_l2cs.py logs/session_... --device cpu --force   # re-run inference on CPU

Steps
  1. Inference: crop the face (MediaPipe box from frames.csv, squared + padded), resize to
     448x448, ImageNet-normalise, ResNet50 L2CS (Gaze360 weights) -> gaze yaw/pitch per frame.
     Cached to <session>/l2cs.csv.
  2. Calibration: the true gaze direction for each calibration frame is the ray from the head
     (MediaPipe position) to the shown dot. A small linear map L2CS angles (+ head angles) ->
     true angles is fitted on calibration frames.
  3. Evaluation on the validation frames: angular error, still vs moving, horizontal vs vertical,
     raw and with a 5-frame median filter. The same table is printed for the MediaPipe model.

Weights: models/l2cs/l2cs_gaze360_resnet50.safetensors (py-feat re-host of the official
L2CS-Net weights). Gaze360-trained weights are for non-commercial research use only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gaze_model import (  # noqa: E402
    GazeSample, HeadFrameGazeModel, ScreenGeometry, angles_to_dir, angular_error_deg, dir_to_angles, head_angles,
)

WEIGHTS = HERE / "models" / "l2cs" / "l2cs_gaze360_resnet50.safetensors"
RAW = ["dx", "dy", "lid"] + [f"R{i}{j}" for i in range(3) for j in range(3)] + ["Px", "Py", "Pz"]
CAL_SETTLE_S, VAL_SETTLE_S = 0.4, 0.8


# ----------------------------------------------------------------------------- model
def build_model(device: str):
    import torch.nn as nn
    from safetensors.torch import load_file
    from torchvision.models.resnet import Bottleneck, ResNet

    class L2CS(ResNet):
        def __init__(self):
            super().__init__(Bottleneck, [3, 4, 6, 3])
            del self.fc
            self.fc_yaw_gaze = nn.Linear(2048, 90)
            self.fc_pitch_gaze = nn.Linear(2048, 90)
            self.fc_finetune = nn.Linear(2048 + 3, 3)  # unused, present in the checkpoint

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1)
            return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)

    model = L2CS()
    missing, unexpected = model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval().to(device)


def face_crop(frame: np.ndarray, box, pad: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    c, d = a + int(round(side)), b + int(round(side))
    crop = np.zeros((d - b, c - a, 3), np.uint8)  # zero-pad if the box leaves the frame
    sa, sb, sc, sd = max(a, 0), max(b, 0), min(c, w), min(d, h)
    crop[sb - b:sd - b, sa - a:sc - a] = frame[sb:sd, sa:sc]
    return crop


def run_inference(session: Path, rows: list[dict], device: str, batch: int = 32) -> dict[int, tuple[float, float]]:
    import torch

    model = build_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    idx_tensor = torch.arange(90, device=device, dtype=torch.float32)
    wanted = {int(r["frame"]): r for r in rows if r["has_face"] == "1"}
    cap = cv2.VideoCapture(str(session / "video.mp4"))
    out: dict[int, tuple[float, float]] = {}
    buf_imgs, buf_idx = [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.from_numpy(np.stack(buf_imgs)).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        with torch.no_grad():
            yaw_l, pitch_l = model(x)
        yaw = (torch.softmax(yaw_l, 1) * idx_tensor).sum(1) * 4 - 180
        pitch = (torch.softmax(pitch_l, 1) * idx_tensor).sum(1) * 4 - 180
        for i, y, p in zip(buf_idx, yaw.cpu().numpy(), pitch.cpu().numpy()):
            out[i] = (float(y), float(p))
        buf_imgs.clear()
        buf_idx.clear()

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = wanted.get(n)
        if r is not None:
            box = [float(r[k]) for k in ("face_x0", "face_y0", "face_x1", "face_y1")]
            crop = cv2.cvtColor(cv2.resize(face_crop(frame, box), (448, 448)), cv2.COLOR_BGR2RGB)
            buf_imgs.append(crop)
            buf_idx.append(n)
            if len(buf_imgs) >= batch:
                flush()
        n += 1
        if n % 500 == 0:
            print(f"  {n} frames", flush=True)
    flush()
    if n != len(rows):
        print(f"warning: video has {n} frames, frames.csv has {len(rows)} rows")
    return out


# ----------------------------------------------------------------------------- evaluation
# Session-level factor applied to MediaPipe's head position (set from Kinect depth, see depth_scale)
P_SCALE = 1.0
PER_FRAME_DEPTH = False  # True: rescale each frame's head position by that frame's depth
# Depth measures the nearest face surface (nose tip); the eyes sit roughly this much further back.
NOSE_TO_EYES_CM = 2.0


def sample_of(r) -> GazeSample:
    v = [float(r[k]) for k in RAW]
    P = np.array(v[12:15])
    k = P_SCALE
    if PER_FRAME_DEPTH and r.get("depth_cm") not in (None, "", "nan") and P[2] < -1:
        d = float(r["depth_cm"]) + NOSE_TO_EYES_CM
        if 25 < d < 200:
            k = d / -P[2]  # keep MediaPipe's direction to the head, take the distance from depth
    return GazeSample(np.array(v[:3]), np.array(v[3:12]).reshape(3, 3), k * P)


def depth_scale(rows: list[dict]) -> float | None:
    """Ratio true eye distance (Kinect depth) / MediaPipe distance, median over the session."""
    pairs = [(float(r["depth_cm"]) + NOSE_TO_EYES_CM, -float(r["Pz"])) for r in rows
             if r.get("depth_cm") not in (None, "", "nan") and r["has_face"] == "1"]
    pairs = [(d, m) for d, m in pairs if 25 < d < 200 and m > 1]
    if len(pairs) < 30:
        return None
    d, m = np.array(pairs).T
    print(f"distance: Kinect depth median {np.median(d):.1f} cm vs MediaPipe {np.median(m):.1f} cm "
          f"-> scale {np.median(d / m):.3f} (frame-to-frame IQR {np.subtract(*np.percentile(d / m, [75, 25])):.3f})")
    return float(np.median(d / m))


def target_angles(screen: ScreenGeometry, s: GazeSample, target) -> tuple[float, float]:
    return dir_to_angles(screen.px_to_cm(*target) - s.P)


def fit_linear(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    reg = ridge * len(X) * np.eye(Xs.shape[1])
    reg[0, 0] = 0
    W = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)
    return lambda Z: np.column_stack([np.ones(len(Z)), (Z - mu) / sd]) @ W


def median_filter(a: np.ndarray, groups: list, k: int = 5) -> np.ndarray:
    """Running median within contiguous runs of the same target (no smoothing across dots)."""
    out = a.copy()
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        for j, i in enumerate(idx):
            lo, hi = max(0, j - k // 2), min(len(idx), j + k // 2 + 1)
            out[i] = np.median(a[idx[lo:hi]], axis=0)
    return out


def report(name: str, screen: ScreenGeometry, test: list[dict], pred_px: list) -> str:
    lines = [f"{name}"]
    for ph in sorted(set(r["phase"] for r in test)):
        e, ex, ey = [], [], []
        for r, p in zip(test, pred_px):
            if r["phase"] != ph or p is None:
                continue
            s, t = sample_of(r), (float(r["target_x"]), float(r["target_y"]))
            e.append(angular_error_deg(screen, s.P, p, t))
            d = float(np.linalg.norm(screen.px_to_cm(*t) - s.P))
            ex.append(math.degrees(abs(p[0] - t[0]) * screen.w_cm / screen.w_px / d))
            ey.append(math.degrees(abs(p[1] - t[1]) * screen.h_cm / screen.h_px / d))
        if e:
            lines.append(f"  {ph:<10} n={len(e):4d}  mean {np.mean(e):5.2f}  median {np.median(e):5.2f}  p90 "
                         f"{np.percentile(e, 90):5.2f} deg   (median x {np.median(ex):4.1f}, y {np.median(ey):4.1f})")
    return "\n".join(lines)


def tgt(r) -> tuple[float, float]:
    return float(r["target_x"]), float(r["target_y"])


def predict_all(screen: ScreenGeometry, l2cs: dict, train: list[dict], test: list[dict]) -> dict[str, list]:
    """Fit every method on `train`, return predicted screen points for `test`, keyed by method name."""
    def l2cs_X(rs, with_head):
        L = np.radians([l2cs[int(r["frame"])] for r in rs])
        if not with_head:
            return L
        return np.column_stack([L, np.array([head_angles(sample_of(r).R) for r in rs])])

    def to_px(rs, ang):
        return [screen.intersect(sample_of(r).P, angles_to_dir(a[0], a[1])) if np.all(np.isfinite(a)) else None
                for r, a in zip(rs, ang)]

    Y = np.array([target_angles(screen, sample_of(r), tgt(r)) for r in train])
    groups = [(r["phase"], r["target_x"], r["target_y"]) for r in test]
    out = {}
    for with_head in (False, True):
        ang = fit_linear(l2cs_X(train, with_head), Y)(l2cs_X(test, with_head))
        label = "L2CS + linear calibration" + (" + head angles" if with_head else "")
        out[label] = to_px(test, ang)
        out[label + ", 5-frame median"] = to_px(test, median_filter(ang, groups))

    mp = HeadFrameGazeModel(screen)
    mp.fit([sample_of(r) for r in train], [tgt(r) for r in train])
    mp_px = [mp.predict_px(sample_of(r)) for r in test]
    out["MediaPipe iris model (current tracker)"] = mp_px
    mp_ang = np.array([dir_to_angles(screen.px_to_cm(*p) - sample_of(r).P) if p else (np.nan, np.nan)
                       for r, p in zip(test, mp_px)])
    out["MediaPipe iris model, 5-frame median"] = to_px(test, median_filter(mp_ang, groups))
    return out


def evaluate(session: Path, rows: list[dict], l2cs: dict, screen: ScreenGeometry):
    ok = lambda r: r["has_face"] == "1" and max(float(r["blink_l"]), float(r["blink_r"])) < 0.45  # noqa: E731
    cal = [r for r in rows if ok(r) and r["phase"].startswith("cal_") and float(r["t_since_onset"]) >= CAL_SETTLE_S
           and int(r["frame"]) in l2cs]
    val = [r for r in rows if ok(r) and r["phase"].startswith("val_") and float(r["t_since_onset"]) >= VAL_SETTLE_S
           and int(r["frame"]) in l2cs]
    if len(cal) < 30:
        print(f"not enough calibration frames ({len(cal)}); run a calibration first.")
        return

    Ycal = np.array([target_angles(screen, sample_of(r), tgt(r)) for r in cal])
    Lcal = np.radians([l2cs[int(r["frame"])] for r in cal])
    corr = [np.corrcoef(Lcal[:, i], Ycal[:, j])[0, 1] for i, j in ((0, 0), (1, 1), (0, 1), (1, 0))]
    print(f"\nL2CS vs true angle correlation on calibration frames: yaw~yaw {corr[0]:+.2f}, pitch~pitch {corr[1]:+.2f} "
          f"(cross: yaw~pitch {corr[2]:+.2f}, pitch~yaw {corr[3]:+.2f})")

    if val:
        header = f"Validation run: fit on {len(cal)} calibration frames, scored on {len(val)} validation frames"
        test, preds = val, predict_all(screen, l2cs, cal, val)
    else:
        # No [v] run: leave-one-dot-out over the 8 outer calibration dots. The centre dot is never
        # held out because the head-sweep frames also look at the centre (that would leak).
        centre = (screen.w_px // 2, screen.h_px // 2)
        dots = sorted({tgt(r) for r in cal if r["phase"] == "cal_points"} - {centre})
        header = (f"No validation run: leave-one-dot-out over {len(dots)} calibration dots "
                  f"(train on the other dots + head sweep, score the held-out dot)")
        test, preds = [], {}
        for d in dots:
            held = [r for r in cal if r["phase"] == "cal_points" and tgt(r) == d]
            train = [r for r in cal if tgt(r) != d]
            for k, v in predict_all(screen, l2cs, train, held).items():
                preds.setdefault(k, []).extend(v)
            test += held

    text = header + "\n\n" + "\n\n".join(report(k, screen, test, v) for k, v in preds.items())
    print("\n" + text)
    (session / "l2cs_report.txt").write_text(text + "\n")
    print(f"\nreport saved to {session / 'l2cs_report.txt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--force", action="store_true", help="re-run inference even if l2cs.csv exists")
    ap.add_argument("--no-depth", action="store_true", help="ignore Kinect depth; use MediaPipe distance as-is")
    ap.add_argument("--depth-mode", choices=["frame", "session"], default="frame",
                    help="frame: each frame's distance from depth (default); session: one median scale factor")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.session / "frames.csv")))
    meta = json.loads((a.session / "meta.json").read_text())
    screen = ScreenGeometry(**meta["screen"])
    cache = a.session / "l2cs.csv"

    if cache.exists() and not a.force:
        l2cs = {int(r["frame"]): (float(r["yaw_deg"]), float(r["pitch_deg"])) for r in csv.DictReader(open(cache))}
        print(f"using cached {cache} ({len(l2cs)} frames)")
    else:
        import torch
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"running L2CS-Net on {device} ...")
        l2cs = run_inference(a.session, rows, device)
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "yaw_deg", "pitch_deg"])
            for k in sorted(l2cs):
                w.writerow([k, *l2cs[k]])
        print(f"inference done: {len(l2cs)} frames -> {cache}")

    global P_SCALE, PER_FRAME_DEPTH
    k = None if a.no_depth else depth_scale(rows)
    if k is not None:
        P_SCALE = k
        PER_FRAME_DEPTH = a.depth_mode == "frame"
        print(f"head distance from Kinect depth ({a.depth_mode} mode; session scale {k:.3f}); compare with --no-depth")
    elif not a.no_depth:
        applied = float(meta.get("mediapipe_dist_scale_applied", 1.0))  # newer sessions are already scaled
        P_SCALE = 1.248 / applied  # distance_sweep.py fit for the Kinect v2 colour camera
        print(f"no depth in this session; MediaPipe distance x {P_SCALE:.3f} (distance_sweep fit; {applied} already applied)")
    evaluate(a.session, rows, l2cs, screen)


if __name__ == "__main__":
    main()
