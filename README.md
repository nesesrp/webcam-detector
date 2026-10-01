# Webcam Detector

A beginner project for **real-time object detection** from a computer webcam using [Ultralytics YOLO](https://docs.ultralytics.com/). It uses a model pretrained on the COCO dataset, so it recognizes 80 everyday objects such as people, phones, cups and chairs out of the box.

## Features

- Real-time object detection with a webcam
- Command-line options for camera, model, confidence, image size and class filtering
- Bounding boxes, labels and confidence scores drawn on each frame
- Live FPS counter (averaged over recent frames for a stable reading)
- Live counter showing the number of people on screen
- Adjust the confidence threshold live with the keyboard
- Save screenshots of the annotated frame with a single key press
- Pause and resume the live view
- Record the annotated video to an `.mp4` file
- Session summary printed on exit (max people, most seen objects, average FPS)

## Tech Stack

- Python 3.12
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLO11 nano model)
- PyTorch
- OpenCV

## Setup

```bash
git clone https://github.com/nesesrp/webcam-detector.git
cd webcam-detector

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note for Intel Macs:** PyTorch no longer publishes wheels for Intel (x86_64) macOS after version 2.2.2, which supports Python up to 3.12. Use Python 3.12 rather than a newer version, otherwise `pip` will try to build packages like NumPy from source and fail.

## Usage

```bash
python detect.py
```

- The model weights (`yolo11n.pt`) are downloaded automatically on the first run.
- Optional command-line options (run `python detect.py --help` for details):

| Option | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index to open |
| `--model` | `yolo11n.pt` | YOLO model weights to load |
| `--conf` | `0.5` | Minimum confidence score (0–1) |
| `--imgsz` | `320` | Inference image size, smaller = faster |
| `--classes` | all | Only detect these classes, e.g. `--classes person cup` |

- On macOS, allow camera access for your terminal or editor when prompted (System Settings → Privacy & Security → Camera).
- Keyboard controls:

| Key | Action |
|---|---|
| `p` | Pause / resume the live view |
| `+` / `-` | Raise / lower the confidence threshold by 0.05 |
| `s` | Save a screenshot of the current frame to `captures/` |
| `r` | Start / stop recording an `.mp4` video to `captures/` |
| `q` | Quit |

- Screenshots also work while paused, so you can freeze a moment and then save it.
- Recordings use the FPS measured when recording starts, so playback runs at roughly real speed. Frames are not recorded while paused.
- When you quit, a session summary is printed to the terminal:

```
=== Session Summary ===
Duration: 01:35
Frames processed: 960
Average FPS: 12.0
Max people at once: 3 (at 00:42)
Most seen objects (share of frames they appeared in):
  person: 94%
  cup: 42%
  cell phone: 12%
Screenshots saved: 2
Recordings saved: 1
```

You can tune detection in `detect.py`:

| Parameter | Default | Effect |
|---|---|---|
| `conf` | `0.5` | Minimum confidence score; raise it to reduce false positives |
| `imgsz` | `320` | Input size; smaller is faster, larger is more accurate |
| `FPS_WINDOW` | `20` | Number of recent frames used to average the FPS |

## Project Structure

```
webcam-detector/
├── detect.py          # Real-time webcam detection
├── captures/          # Screenshots and recordings (created on first save, git-ignored)
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Real-time webcam detection with a pretrained model
- [x] FPS counter
- [x] People counter
- [x] Stable (averaged) FPS counter
- [x] Screenshot capture
- [x] Pause and video recording
- [x] Session summary on exit
- [ ] Object detection on a single image
- [ ] Count people passing through a doorway using object tracking
- [ ] Web interface with Streamlit
- [ ] Save detections to a CSV file

## License

This project uses the [Ultralytics](https://github.com/ultralytics/ultralytics) library, which is licensed under AGPL-3.0.
