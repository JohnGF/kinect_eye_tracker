# Kinect v2 head & gaze tracker

Webcam-style head pose and screen-gaze estimation with a Kinect v2 colour camera and MediaPipe
FaceLandmarker, streamed to [Lab Streaming Layer](https://labstreaminglayer.org/) for the
[nautilus_bci](https://github.com/JohnGF/nautilus_bci) multimodal BCI platform.

It is a **concept-validation** tracker: head pose is reliable, while screen gaze is coarse
(about 6° median, see [Results](#results)). Use a dedicated eye tracker (e.g. Pupil Labs) where
precise gaze matters.

## What it does

- **Head pose** (yaw, pitch, roll, position) from MediaPipe's facial transformation matrix, with
  the distance corrected for the Kinect lens (`MEDIAPIPE_DIST_SCALE`, see Results).
- **Gaze** (`gaze_model.py`): iris offset measured in each eye's corner-aligned frame, mapped to
  eye-in-head angles by a linear calibration, then ray-cast onto the screen with the live head
  pose, so head movement after calibration is handled geometrically.
- **Calibration:** 9 dots with the head still, then a guided head sweep with the eyes on the centre
  dot: tilt up/down, turn left/right, sway left/right. Each step completes only once the head has
  really moved (8°, 12°, 5 cm).
- **Validation (`v`):** 5 dots with the head still, 5 while moving it; per-frame CSV in `logs/` and
  an error summary by head yaw, pitch and distance.
- **Recording (`--record`):** colour video plus per-frame metadata, for offline models such as
  L2CS-Net (`offline_l2cs.py`).
- **LSL outlet** `KinectGaze` (type `Gaze`, 10 × float32): `gaze_x, gaze_y, head_x, head_y` (screen px),
  `head_yaw, head_pitch, head_roll` (deg), `head_dist_cm`, `blink_l, blink_r`. Samples are pushed per
  processed frame (about 15–16 Hz on the test machine; the declared nominal rate is 30 Hz).
- Display filters: I-VT fixation/saccade filter on gaze, 1-Euro filter and a 0.8° rotation
  deadband on head pose.

## Requirements

- Linux, Kinect v2, [libfreenect2](https://github.com/OpenKinect/libfreenect2) and the
  [JohnGF/kinect2pipe](https://github.com/JohnGF/kinect2pipe) fork feeding a v4l2loopback device
  (default `/dev/video10`): `./kinect_start.sh` there.
- The tracker reads frames from the fork's lock-free shared memory (`/dev/shm/kinect2pipe_shm`,
  see `shm_client.py`) and falls back to the V4L2 device, so upstream
  [swedishborgie/kinect2pipe](https://github.com/swedishborgie/kinect2pipe) also works. The optional
  IR + depth segment (`/dev/shm/kinect2pipe_ir_shm`) needs the fork started with `--ir`.
- [uv](https://docs.astral.sh/uv/); Python 3.11+. Dependencies are in `pyproject.toml` / `uv.lock`.

## Quick start

```bash
./run_kinect_tracker.sh --lsl            # fullscreen: calibrate, then live tracking
./run_kinect_tracker.sh --lsl --record   # same, and record logs/session_<time>/
```

Set your screen size and camera position at the top of `tracker.py` (`SCREEN_*`, `CAMERA_X_CM`).
`screen_ruler.py` shows a fullscreen ruler: click under the camera lens to read `CAMERA_X_CM`.

| Key | Action |
|---|---|
| `Enter` / `Space` | lock the current calibration dot / skip a head-sweep step |
| `c` | recalibrate |
| `v` | validation run |
| `r` | recentre: look at the screen centre and press `r` |
| `h` | mode: hybrid / head only / eye only |
| `p`, `f` | picture-in-picture camera view, fullscreen toggle |
| `q` / `Esc` | quit |

## Tools

| Script | Purpose |
|---|---|
| `validation.py <csv>` | re-summarise a validation log |
| `offline_l2cs.py <session>` | L2CS-Net gaze on a recorded session, calibrated and scored against the MediaPipe model on the same frames. Self-contained uv script (torch); runs on a GPU box or cluster. |
| `distance_sweep.py` | measure the MediaPipe → true distance factor against Kinect depth (needs `--ir`) |
| `ir_preview.py [--check N]` | view colour / IR / depth side by side, or a headless fps and sync check |
| `screen_ruler.py` | measure the camera's horizontal position on the screen |

L2CS-Net weights are not included. Download the safetensors re-host of the official Gaze360
weights from Hugging Face (`py-feat/l2cs`):

```bash
mkdir -p models/l2cs && curl -L -o models/l2cs/l2cs_gaze360_resnet50.safetensors \
  https://huggingface.co/py-feat/l2cs/resolve/main/l2cs_gaze360_resnet50.safetensors
```

The L2CS-Net code is MIT; the Gaze360-trained weights are for **non-commercial research use only**
(Gaze360 dataset licence).

## Results

One participant, one desk setup (Kinect on top of a 59.7 × 33.6 cm, 2560 × 1440 monitor,
9.4 cm left of centre, eyes about 57 cm away), 31 Oct 2026. Treat these as a measurement of this
setup, not general constants.

### MediaPipe distance vs Kinect depth

MediaPipe computes head position for its own virtual camera, not the Kinect v2 colour lens, so its
distance is scaled. A sweep from as close as possible to as far as the chair allowed
(`distance_sweep.py`; Kinect depth to the nose tip + 2 cm as ground truth, valid from 58 to 114 cm):

| Model | RMSE |
|---|---|
| MediaPipe as-is | 16.6 cm |
| **true = 1.248 × MediaPipe** | **1.9 cm** |
| true = 1.383 × MediaPipe − 8.1 cm | 1.3 cm |

The pure scale is used: it agrees with the expected field-of-view mismatch (about 1.21), and at
close range, where Kinect depth is invalid (minimum range 50 cm), it matches an independent
iris-size distance estimate better than the affine fit does (2.8 vs 4.8 cm). With this factor,
a fixed scale gave the same gaze accuracy as per-frame depth, so the IR stream is not needed at
run time.

### Gaze accuracy

Median angular error on separate validation dots after calibration (same session, same frames for
every method):

| Method | Head still | Head moving |
|---|---|---|
| MediaPipe iris model, raw MediaPipe distance | 9.7° | 7.7° |
| MediaPipe iris model, distance × 1.248 | 8.8° | 6.8° |
| L2CS-Net (Gaze360) + linear calibration + head angles, raw distance | 7.5° | 8.0° |
| **L2CS-Net + linear calibration + head angles, distance × 1.248** | **5.8°** | **6.4°** |

Findings along the way:

- The original calibration fitted a map on the final screen point, so its about 5× eye gain also
  applied to head movement. Separating eye-in-head angles from head pose fixed the head-motion
  dependence, but needs calibration data with real head movement (the guided sweep).
- MediaPipe's iris landmarks jitter by about 2–5° horizontally and 3–7° vertically at this
  distance, and do not respond consistently to eye counter-rotation when the head turns; this
  limits the iris model to roughly 6–9°.
- L2CS-Net is steadier frame to frame (1–3°) but has a target-dependent bias, so it needs a
  calibration; with the distance correction it is the most accurate option tested, mainly
  vertically (about 3.5° vs about 7°).

## Development

```bash
uv run --with pytest pytest -q tests
```

`tests/sim.py` is a synthetic head/eye world used to check that gaze error stays low when the head
moves after calibration; other tests cover the shared-memory readers, the calibration flow, the
session recorder and the validation summary.

`gaze_tracking_roadmap.md` is the original design roadmap (written before the results above).

## License

MIT, see [LICENSE](LICENSE). Model weights downloaded separately carry their own terms.
