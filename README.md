# OBS Face Follow

An OBS Studio Python script plugin that detects a face in a video source and
automatically pans a PTZ camera to keep the face horizontally centred.

---

## Features

| # | Feature |
|---|---------|
| 1 | **Separate management window** – a standalone Qt dialog for configuring and controlling the tracker |
| 2 | **Activate / deactivate button** – one-click start and stop with clear visual status |
| 3 | **Horizontal-only tracking** – only pan commands are issued; tilt is always zero |
| 4 | **Proportional tracking** – the further the face drifts from centre, the faster the camera pans |
| 5 | **User-settable deadzone** – configurable fraction of the half-frame width; no movement is triggered inside this zone |
| 6 | **Three PTZ control methods** – URL-based HTTP GET, ONVIF (standard PTZ), or the [obs-ptz](https://github.com/glikely/obs-ptz) OBS plugin |

---

## Files

| File | Purpose |
|------|---------|
| `obs_face_tracker.py` | OBS Python script entry point – load this via **Tools → Scripts** |
| `face_detector.py` | OpenCV Haar-cascade face detection |
| `ptz_controller.py` | PTZ camera control (URL / ONVIF / obs-ptz) |
| `tracker_window.py` | PyQt5 management dialog |
| `requirements.txt` | Python dependencies |
| `tests/` | pytest unit tests |

---

## Quick Start

### 1 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 – Load the script in OBS Studio

1. Open OBS Studio.
2. Go to **Tools → Scripts**.
3. Click the **+** button and select `obs_face_tracker.py`.
4. Click **Open Tracker Window** in the Scripts dialog.

### 3 – Configure the tracker

**Control tab**

* Choose **Camera Device Index** (0 = first webcam) or **Video URL / RTSP Stream** for the
  video source that OpenCV will read for face detection.
* Click **▶ Start Tracking** to begin.

**Tracking tab**

| Setting | Description |
|---------|-------------|
| Deadzone | Fraction of the half-frame width where no pan is issued (default 0.10 = 10 %). Increase to reduce jitter. |
| Speed Multiplier | How fast the camera moves relative to the face offset (default 0.5). |
| Detection Scale | Haar-cascade scale factor – higher is faster but less accurate (default 1.3). |
| Min Neighbours | Minimum detections required to confirm a face – higher = fewer false positives (default 5). |

**PTZ Settings tab**

Choose one of three control methods:

#### URL-based (HTTP GET)

Fill in the pan-left, pan-right, and stop URLs for your camera's HTTP API.
Optionally specify a speed query-string parameter name (e.g. `speed`).

#### ONVIF

Enter the camera's IP, port, username, and password.  The profile token is
auto-detected from the camera if left blank.  Use the **Test ONVIF Connection**
button to verify credentials.

#### obs-ptz Plugin

Enter the exact OBS source name of your PTZ camera (as configured in the
[obs-ptz](https://github.com/glikely/obs-ptz) plugin).  The script will update
the `pan_speed` property on that source.

---

## Tracking Algorithm

```
offset = (face_centre_x − frame_centre_x) / (frame_width / 2)
          ∈ [−1.0, +1.0]

if |offset| ≤ deadzone  →  stop
if  offset  >  deadzone  →  pan_right(speed = min(1.0, (offset − dz) × speed_multiplier × 2))
if  offset  < −deadzone  →  pan_left (speed = min(1.0, (|offset| − dz) × speed_multiplier × 2))
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Requirements

* Python 3.9+
* OBS Studio 28+ (for Python scripting)
* `opencv-python`, `numpy`, `requests`, `PyQt5` (see `requirements.txt`)
* For ONVIF: an ONVIF-compatible PTZ camera
* For obs-ptz: the [obs-ptz plugin](https://github.com/glikely/obs-ptz) installed in OBS