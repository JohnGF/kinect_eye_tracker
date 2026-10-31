import math
import struct

import numpy as np
import pytest

from calibration import SWEEP_STEPS, GazeCalibrationEngine
from gaze_model import ScreenGeometry
from shm_client import HEADER_SIZE, IR_MAGIC, IrDepthSource, color_norm_to_depth_px, face_depth_cm
from sim import CENTER, P0, head_facing, sample_looking_at
from validation import summarize


def write_ir_shm(path, ir, depth, seq=2, dev_ts=1234, flags=1):
    h, w = ir.shape
    hdr = struct.pack("<IIIIIIQQII16s", IR_MAGIC, 1, w, h, ir.nbytes * 2, flags, seq, 99, dev_ts, 7, b"")
    assert len(hdr) == HEADER_SIZE == 64
    path.write_bytes(hdr + ir.astype("<u2").tobytes() + depth.astype("<u2").tobytes())


def test_ir_reader_roundtrip(tmp_path):
    ir = np.arange(424 * 512, dtype=np.uint16).reshape(424, 512)
    depth = np.full((424, 512), 650, np.uint16)
    p = tmp_path / "ir_shm"
    write_ir_shm(p, ir, depth)
    src = IrDepthSource(str(p))
    ok, ir2, d2, ts = src.read()
    assert ok and ts == 99 and src.last_device_ts == 1234
    assert np.array_equal(ir2, ir) and np.array_equal(d2, depth)
    assert src.read(only_new=True)[0] is False  # same seq -> nothing new


def test_ir_reader_rejects_odd_seq_and_bad_magic(tmp_path):
    p = tmp_path / "ir_shm"
    write_ir_shm(p, np.zeros((424, 512), np.uint16), np.zeros((424, 512), np.uint16), seq=3)
    assert IrDepthSource(str(p)).read()[0] is False
    p.write_bytes(b"\0" * 64)
    assert IrDepthSource(str(p)).available() is False
    assert IrDepthSource(str(tmp_path / "missing")).available() is False


def test_face_depth_ignores_background_and_corrects_parallax():
    depth = np.full((424, 512), 2000, np.uint16)  # wall
    cx, cy = color_norm_to_depth_px(0.5, 0.5)
    assert (cx, cy) == pytest.approx((256, 212))
    shift = int(-365.0 * 5.2 / 62.0)  # where a face at 62 cm really appears in the depth image
    depth[160:270, 215 + shift:300 + shift] = 620   # face
    depth[160:270, 300 + shift:340 + shift] = 900   # chair/headset just right of the face
    assert face_depth_cm(depth, (0.45, 0.4, 0.55, 0.6)) == pytest.approx(62.0)
    assert face_depth_cm(np.zeros((424, 512), np.uint16), (0.45, 0.4, 0.55, 0.6)) is None


def test_calibration_engine_end_to_end(tmp_path, monkeypatch):
    import validation
    monkeypatch.setattr(validation, "LOG_DIR", tmp_path)
    rng = np.random.default_rng(3)
    eng = GazeCalibrationEngine(screen=ScreenGeometry())
    eng.start()
    t = 0.0
    while eng.phase == "points":
        tgt = eng.get_current_target_px()
        R = head_facing(P0, CENTER, rng.normal(0, 0.005), rng.normal(0, 0.005))
        eng.update(sample_looking_at(tgt, P0, R, rng, 0.005), False, now=t)
        t += 1 / 30
        assert t < 60, "points phase never completed"
    assert eng.phase == "sweep"

    # Follow the prompts like a person: ease towards the asked-for direction until the step completes.
    off = {"yaw": 0.0, "pitch": 0.0, "x": 0.0}
    done, frames = False, 0
    while not done:
        if eng.sweep_ref is not None:
            _, axis, goal = SWEEP_STEPS[eng.sweep_step]
            want = (1 if eng.sweep_step % 2 == 0 else -1) * goal * 1.3
            off = {k: (v + 0.15 * ((want if k == axis else 0.0) - v)) for k, v in off.items()}
        P = P0 + np.array([off["x"], 0.0, 0.0])
        R = head_facing(P, CENTER, math.radians(off["yaw"]), math.radians(off["pitch"]))
        done = eng.update(sample_looking_at(CENTER, P, R, rng, 0.005), False, now=t)
        t += 1 / 30
        frames += 1
        assert frames < 2000, f"sweep stuck at step {eng.sweep_step}"
    assert eng.sweep_step == len(SWEEP_STEPS) and eng.model.use_head
    assert done and eng.is_calibrated and not eng.is_active
    assert eng.train_rms_deg < 1.0
    assert len(list(tmp_path.glob("calibration_*.csv"))) == 1
    probe = sample_looking_at((900, 600), P0 + np.array([3, 0, 3]), head_facing(P0, CENTER, 0.1, -0.05))
    px = eng.predict_px(probe)
    assert math.hypot(px[0] - 900, px[1] - 600) < 80


def test_summary_bins():
    rows = []
    for i in range(40):
        rows.append(dict(**{"pass": "still" if i < 20 else "moving"}, err_filt_deg=1.0 if i < 20 else 3.0,
                         err_raw_deg=1.5, head_yaw_deg=0.0 if i < 20 else 12.0, head_pitch_deg=0.0,
                         head_dist_cm=60.0, depth_cm=float("nan")))
    s = summarize(rows)
    assert "still   n=  20  mean= 1.00" in s
    assert "moving  n=  20  mean= 3.00" in s
    assert "MediaPipe estimate" in s
