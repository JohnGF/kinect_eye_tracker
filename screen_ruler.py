#!/usr/bin/env python3
"""
Fullscreen ruler to measure where the Kinect colour lens sits relative to the screen.

Move the mouse to the top edge of the screen directly below the camera lens and click.
The x position (px and cm from screen centre) is printed and saved to logs/camera_position.txt.
[q]/[Esc] quits.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

import time
from pathlib import Path

import cv2
import numpy as np

from gaze_model import ScreenGeometry

S = ScreenGeometry()
WIN = "Screen ruler"
state = {"mouse": (S.w_px // 2, 200), "picked": None}


def px_to_cm_x(px: float) -> float:
    return (px - S.w_px / 2) / S.w_px * S.w_cm


def on_mouse(event, x, y, _flags, _param):
    state["mouse"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["picked"] = x
        line = f"camera x = {x} px = {px_to_cm_x(x):+.1f} cm from screen centre ({time.strftime('%Y-%m-%d %H:%M:%S')})"
        print(line, flush=True)
        out = Path(__file__).resolve().parent / "logs" / "camera_position.txt"
        out.parent.mkdir(exist_ok=True)
        with open(out, "a") as f:
            f.write(line + "\n")


def draw() -> np.ndarray:
    img = np.full((S.h_px, S.w_px, 3), 24, np.uint8)
    for x in range(0, S.w_px + 1, 20):
        major = x % 100 == 0
        cv2.line(img, (x, 0), (x, 60 if major else 25), (255, 255, 255) if major else (150, 150, 150), 2 if major else 1)
        if x % 200 == 0:
            cv2.putText(img, f"{x}", (x - 25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"{px_to_cm_x(x):+.0f}cm", (x - 35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cx = S.w_px // 2
    cv2.line(img, (cx, 0), (cx, S.h_px), (0, 120, 255), 1)
    cv2.putText(img, "screen centre (0 cm)", (cx + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)

    mx, _ = state["mouse"]
    cv2.line(img, (mx, 0), (mx, S.h_px), (0, 255, 120), 2)
    cv2.putText(img, f"x = {mx} px  =  {px_to_cm_x(mx):+.1f} cm", (min(mx + 15, S.w_px - 480), 260),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 120), 2)
    if state["picked"] is not None:
        p = state["picked"]
        cv2.line(img, (p, 0), (p, S.h_px), (255, 80, 255), 3)
        cv2.putText(img, f"SAVED camera x = {p} px = {px_to_cm_x(p):+.1f} cm", (60, S.h_px // 2 + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 80, 255), 3)

    msgs = ["Move the mouse so the green line is directly BELOW the Kinect's colour camera lens, then CLICK.",
            "(The colour lens is the small round one; the bigger dark window is the IR depth camera.)",
            "Click again to correct.  [q] / [Esc] when done."]
    for i, m in enumerate(msgs):
        cv2.putText(img, m, (60, S.h_px // 2 - 60 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
    return img


if __name__ == "__main__":
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        cv2.imshow(WIN, draw())
        if cv2.waitKey(16) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
