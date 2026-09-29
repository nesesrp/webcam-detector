# Webcam Detector

A beginner project for **real-time object detection** from a computer webcam using [Ultralytics YOLO](https://docs.ultralytics.com/). It uses a model pretrained on the COCO dataset, so it recognizes 80 everyday objects such as people, phones, cups and chairs out of the box.

## Features

- Real-time object detection with a webcam
- Bounding boxes, labels and confidence scores drawn on each frame
- Live FPS counter
- Live counter showing the number of people on screen

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
- On macOS, allow camera access for your terminal or editor when prompted (System Settings → Privacy & Security → Camera).
- Press `q` to close the window.

You can tune detection in `detect.py`:

| Parameter | Default | Effect |
|---|---|---|
| `conf` | `0.5` | Minimum confidence score; raise it to reduce false positives |
| `imgsz` | `320` | Input size; smaller is faster, larger is more accurate |

## Project Structure

```
webcam-detector/
├── detect.py          # Real-time webcam detection
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Real-time webcam detection with a pretrained model
- [x] FPS counter
- [x] People counter
- [ ] Object detection on a single image
- [ ] Count people passing through a doorway using object tracking
- [ ] Web interface with Streamlit
- [ ] Save detections to a CSV file

## License

This project uses the [Ultralytics](https://github.com/ultralytics/ultralytics) library, which is licensed under AGPL-3.0.
