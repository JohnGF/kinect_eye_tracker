#!/usr/bin/env python3
"""
Check / view the colour + IR + depth streams from kinect2pipe running with --ir.

  uv run python ir_preview.py            # side-by-side colour | IR | depth window, [q] quits
  uv run python ir_preview.py --check 10 # headless: fps of both streams, drops, colour/IR sync

The Kinect only streams while something holds /dev/video10 open (kinect2pipe's watcher), so this
script opens the V4L2 device too, just to keep the camera awake.
"""

import argparse
import struct
import sys
import time

import cv2
import numpy as np

from shm_client import IR_SHM_PATH, IrDepthSource, KinectFrameSource

TICK_MS = 0.125  # Kinect device timestamp unit


def ir_to_gray(ir: np.ndarray) -> np.ndarray:
    return np.clip(np.sqrt(ir.astype(np.float32)) * 3.8, 0, 255).astype(np.uint8)


def depth_to_color(depth: np.ndarray, near=400, far=1500) -> np.ndarray:
    d = np.clip((depth.astype(np.float32) - near) / (far - near), 0, 1)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth == 0] = 0
    return img


def check(seconds: float) -> int:
    color, ir = KinectFrameSource(), IrDepthSource()
    keepalive = cv2.VideoCapture("/dev/video10")
    t_end = time.time() + seconds
    n_c = n_i = 0
    seqs, offsets = [], []
    last_c_seq = -1
    while time.time() < t_end:
        ok, _frame, _ = color.read()
        if ok and color.mode == "shm":

            s = struct.unpack_from("<Q", color.mmap_obj, 24)[0]
            if s != last_c_seq:
                n_c += 1
                last_c_seq = s
        ok_i, _ir, depth, _ = ir.read(only_new=True)
        if ok_i:
            n_i += 1
            seqs.append(struct.unpack_from("<I", ir.mm, 44)[0])
            if color.last_device_ts and ir.last_device_ts:
                offsets.append((int(ir.last_device_ts) - int(color.last_device_ts)) * TICK_MS)
        time.sleep(0.002)
    keepalive.release()
    print(f"colour: {n_c / seconds:5.1f} fps   ({color.mode})")
    if not ir.available():
        print(f"IR/depth: not available - {IR_SHM_PATH} missing. Restart kinect2pipe with --ir.")
        return 1
    gaps = int(np.sum(np.diff(seqs) - 1)) if len(seqs) > 1 else 0
    print(f"IR/depth: {n_i / seconds:5.1f} fps   dropped {gaps} frames (device seq gaps)")
    if offsets:
        o = np.abs(offsets)
        print(f"IR - colour device-time offset: median {np.median(o):.1f} ms, max {o.max():.1f} ms")
    if depth is not None:
        h, w = depth.shape
        c = depth[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
        print(f"centre depth: {np.median(c[c > 0]) / 10 if (c > 0).any() else float('nan'):.1f} cm")
    ok = n_c > 0.5 * 30 * seconds and n_i > 0.5 * 30 * seconds
    print("PASS" if ok else "FAIL: expected ~30 fps on both streams")
    return 0 if ok else 1


def view():
    color, ir = KinectFrameSource(), IrDepthSource()
    keepalive = cv2.VideoCapture("/dev/video10")
    cv2.namedWindow("kinect colour | IR | depth", cv2.WINDOW_NORMAL)
    while True:
        ok, frame, _ = color.read()
        if not ok:
            time.sleep(0.005)
            continue
        h = 424
        tiles = [cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))]
        ok_i, irf, depth, _ = ir.read()
        if ok_i:
            tiles.append(cv2.cvtColor(ir_to_gray(irf), cv2.COLOR_GRAY2BGR))
            tiles.append(depth_to_color(depth))
        else:
            tiles.append(np.zeros((h, 512, 3), np.uint8))
            cv2.putText(tiles[-1], "no IR shm (run kinect2pipe --ir)", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow("kinect colour | IR | depth", np.hstack(tiles))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    keepalive.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, metavar="SECONDS", help="headless stream check")
    a = ap.parse_args()
    sys.exit(check(a.check) if a.check else view())
