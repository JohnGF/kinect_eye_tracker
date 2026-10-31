"""
Client for reading Kinect frames from lock-free shared memory (/dev/shm/kinect2pipe_shm)
with fallback to V4L2 device.
"""

import math
import os
import mmap
import struct
import time
from typing import Optional, Tuple
import cv2
import numpy as np

SHM_PATH = "/dev/shm/kinect2pipe_shm"
HEADER_FMT = "<IIIIIIQQ24s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 64 bytes
EXPECTED_MAGIC = 0x4B325050  # "K2PP"


class KinectFrameSource:
    """Reads frames from shared memory (/dev/shm/kinect2pipe_shm) or V4L2 fallback."""

    def __init__(self, v4l2_device: str = "/dev/video10", prefer_shm: bool = True):
        self.v4l2_device = v4l2_device
        self.prefer_shm = prefer_shm
        self.shm_file = None
        self.mmap_obj = None
        self.cap = None
        self.last_seq = 0
        self.last_device_ts = 0
        self.mode = "shm" if (prefer_shm and os.path.exists(SHM_PATH)) else "v4l2"
        self._init_source()

    def _init_source(self):
        if self.mode == "shm":
            try:
                self.shm_file = open(SHM_PATH, "r+b")
                self.mmap_obj = mmap.mmap(self.shm_file.fileno(), 0, prot=mmap.PROT_READ)
                # Verify magic
                magic = struct.unpack_from("<I", self.mmap_obj, 0)[0]
                if magic != EXPECTED_MAGIC:
                    print(f"Warning: SHM magic mismatch (got 0x{magic:X}, expected 0x{EXPECTED_MAGIC:X}).")
            except Exception as e:
                print(f"Failed to open SHM ({e}), falling back to V4L2 {self.v4l2_device}")
                self._cleanup_shm()
                self.mode = "v4l2"

        if self.mode == "v4l2":
            self.cap = cv2.VideoCapture(self.v4l2_device)
            if not self.cap.isOpened():
                print(f"Warning: Could not open V4L2 capture device {self.v4l2_device}")

    def _cleanup_shm(self):
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except Exception:
                pass
            self.mmap_obj = None
        if self.shm_file is not None:
            try:
                self.shm_file.close()
            except Exception:
                pass
            self.shm_file = None

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Reads next frame.
        Returns: (success, bgr_image, timestamp_ns)
        """
        if self.mode == "shm":
            if not os.path.exists(SHM_PATH) or self.mmap_obj is None:
                self._cleanup_shm()
                self.mode = "v4l2"
                self._init_source()
            else:
                return self._read_shm()

        if self.mode == "v4l2" and self.cap is not None:
            ret, frame = self.cap.read()
            return ret, frame, int(time.time() * 1e9)

        return False, None, 0

    def _read_shm(self) -> Tuple[bool, Optional[np.ndarray], int]:
        # Seqlock read protocol:
        # 1. Read sequence (must be even, meaning writer is not in the middle of a write)
        # 2. Read frame
        # 3. Read sequence again and ensure seq1 == seq2
        max_retries = 5
        for _ in range(max_retries):
            try:
                seq1 = struct.unpack_from("<Q", self.mmap_obj, 24)[0]
                if seq1 % 2 != 0:  # Writer is writing, yield and retry
                    time.sleep(0.001)
                    continue

                width, height, yuv_bytes, cam_mode = struct.unpack_from("<IIII", self.mmap_obj, 8)
                timestamp_ns, device_ts = struct.unpack_from("<QI", self.mmap_obj, 32)

                # Copy raw YUV bytes
                raw_yuv = np.frombuffer(
                    self.mmap_obj,
                    dtype=np.uint8,
                    count=yuv_bytes,
                    offset=HEADER_SIZE
                ).copy()  # copy before re-checking seq so a torn frame is detected

                seq2 = struct.unpack_from("<Q", self.mmap_obj, 24)[0]
                if seq1 == seq2:
                    self.last_device_ts = device_ts  # Kinect clock (0.125 ms ticks), pairs with IR frames
                    yuv_frame = raw_yuv.reshape((height * 3 // 2, width))
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                    return True, bgr_frame, timestamp_ns

            except Exception:
                time.sleep(0.001)

        return False, None, 0

    def release(self):
        self._cleanup_shm()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------------------------
# Parallel IR + depth stream (kinect2pipe --ir) at /dev/shm/kinect2pipe_ir_shm
# Header (64 B): magic, version, width, height, payload_bytes, flags | seq @24 | ts_ns @32 |
#                device_ts @40 | device_seq @44 | reserved
# Payload: uint16 IR[h*w] (raw amplitude) then uint16 depth_mm[h*w] (0 = invalid)
# ---------------------------------------------------------------------------------------------
IR_SHM_PATH = "/dev/shm/kinect2pipe_ir_shm"
IR_MAGIC = 0x4B324952  # "K2IR"

# Approximate Kinect v2 fields of view (deg), used to map colour coords onto the depth image.
COLOR_HFOV, COLOR_VFOV = 84.1, 53.8
DEPTH_HFOV, DEPTH_VFOV = 70.6, 60.0


class IrDepthSource:
    """Seqlock reader for the IR + depth segment. Silently unavailable if kinect2pipe runs without --ir."""

    def __init__(self, path: str = IR_SHM_PATH):
        self.path = path
        self.file = None
        self.mm = None
        self.last_seq = 0
        self.last_device_ts = 0

    def available(self) -> bool:
        if self.mm is not None:
            return True
        if not os.path.exists(self.path):
            return False
        try:
            self.file = open(self.path, "rb")
            self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
            if struct.unpack_from("<I", self.mm, 0)[0] != IR_MAGIC:
                self.release()
                return False
            return True
        except (OSError, ValueError):
            self.release()
            return False

    def read(self, only_new: bool = False):
        """Returns (ok, ir_uint16[h,w], depth_mm_uint16[h,w], timestamp_ns)."""
        if not self.available():
            return False, None, None, 0
        for _ in range(5):
            seq1 = struct.unpack_from("<Q", self.mm, 24)[0]
            if seq1 % 2 or seq1 == 0:
                time.sleep(0.001)
                continue
            if only_new and seq1 == self.last_seq:
                return False, None, None, 0
            w, h, payload, flags = struct.unpack_from("<IIII", self.mm, 8)
            ts_ns, dev_ts = struct.unpack_from("<QI", self.mm, 32)
            n = w * h
            buf = np.frombuffer(self.mm, dtype=np.uint16, count=2 * n, offset=HEADER_SIZE).copy()
            if struct.unpack_from("<Q", self.mm, 24)[0] == seq1:
                self.last_seq = seq1
                self.last_device_ts = dev_ts
                depth = buf[n:].reshape(h, w) if flags & 1 else np.zeros((h, w), np.uint16)
                return True, buf[:n].reshape(h, w), depth, ts_ns
        return False, None, None, 0

    def release(self):
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:
                pass
        if self.file is not None:
            self.file.close()
        self.mm = None
        self.file = None


def color_norm_to_depth_px(u: float, v: float, depth_w: int = 512, depth_h: int = 424) -> Tuple[float, float]:
    """Map normalised colour-image coords to depth-image pixels by matching viewing angles.
    Ignores the ~5 cm colour/depth baseline, so callers should sample a generous region."""
    tx = (2 * u - 1) * math.tan(math.radians(COLOR_HFOV / 2))
    ty = (2 * v - 1) * math.tan(math.radians(COLOR_VFOV / 2))
    du = 0.5 + tx / (2 * math.tan(math.radians(DEPTH_HFOV / 2)))
    dv = 0.5 + ty / (2 * math.tan(math.radians(DEPTH_VFOV / 2)))
    return du * depth_w, dv * depth_h


# Colour/depth camera baseline and depth focal length (Kinect v2 nominal): a face at Z cm appears
# shifted by about DEPTH_FX * BASELINE / Z depth pixels between the two images (~33 px at 55 cm).
DEPTH_FX_PX = 365.0
COLOR_DEPTH_BASELINE_CM = 5.2


def _depth_box_value(depth_mm, box_px, near_mm, far_mm):
    h, w = depth_mm.shape
    (ax, ay), (bx, by) = box_px
    ax, bx = int(max(0, min(ax, bx))), int(min(w, max(ax, bx)))
    ay, by = int(max(0, min(ay, by))), int(min(h, max(ay, by)))
    patch = depth_mm[ay:by, ax:bx]
    vals = patch[(patch > near_mm) & (patch < far_mm)]
    return None if vals.size < 20 else float(np.percentile(vals, 20))


def face_depth_cm(depth_mm: np.ndarray, face_bbox_norm, pad: float = 0.1,
                  near_mm: int = 250, far_mm: int = 2500) -> Optional[float]:
    """Face (nose-tip) distance from the depth image: 20th percentile of valid depth in the face box.

    Two passes: a padded box gives a first distance, which sets the colour->depth parallax shift;
    the second pass samples the shifted, tighter box."""
    if depth_mm is None:
        return None
    h, w = depth_mm.shape
    x0, y0, x1, y1 = face_bbox_norm
    a = color_norm_to_depth_px(x0, y0, w, h)
    b = color_norm_to_depth_px(x1, y1, w, h)
    bw, bh = (b[0] - a[0]) * 0.4, (b[1] - a[1]) * 0.4
    first = _depth_box_value(depth_mm, ((a[0] - 2 * bw, a[1] - bh), (b[0] + 2 * bw, b[1] + bh)), near_mm, far_mm)
    if first is None:
        return None
    shift = -DEPTH_FX_PX * COLOR_DEPTH_BASELINE_CM / (first / 10.0)
    px, py = (b[0] - a[0]) * pad, (b[1] - a[1]) * pad
    second = _depth_box_value(depth_mm, ((a[0] + shift - px, a[1] - py), (b[0] + shift + px, b[1] + py)), near_mm, far_mm)
    return (second if second is not None else first) / 10.0
