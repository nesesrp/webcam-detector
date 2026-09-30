import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class ID of "person" in the COCO dataset
PERSON_CLASS_ID = 0

# Number of recent frames used to average the FPS (higher = smoother but slower to react)
FPS_WINDOW = 20

# Folder where screenshots and recordings are saved
CAPTURES_DIR = Path("captures")


def capture_path(prefix, extension):
    """Return a unique, timestamped file path inside the captures folder."""
    CAPTURES_DIR.mkdir(exist_ok=True)
    return CAPTURES_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.{extension}"


# Load the model (downloaded automatically on first run, "n" = nano, the fastest)
model = YOLO("yolo11n.pt")

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check System Settings > Privacy > Camera permissions.")

prev_time = time.time()
frame_times = deque(maxlen=FPS_WINDOW)
paused = False
writer = None  # cv2.VideoWriter while recording, otherwise None

while True:
    # While paused, skip reading and detection so the last frame stays on screen
    if not paused:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        # Run detection (conf: minimum confidence score, imgsz: smaller = faster)
        results = model(frame, conf=0.5, imgsz=320, verbose=False)

        # Draw boxes and labels on the frame
        annotated = results[0].plot()

        # Calculate FPS as an average over the last frames so the value doesn't flicker
        now = time.time()
        frame_times.append(now - prev_time)
        prev_time = now
        fps = len(frame_times) / sum(frame_times)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Count detected people and draw the count on the frame
        people_count = int((results[0].boxes.cls == PERSON_CLASS_ID).sum())
        cv2.putText(annotated, f"People: {people_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if writer is not None:
            writer.write(annotated)

    # Status indicators are drawn on a copy so they don't end up in screenshots or recordings
    display = annotated.copy()
    if writer is not None:
        width = display.shape[1]
        cv2.circle(display, (width - 100, 25), 8, (0, 0, 255), -1)
        cv2.putText(display, "REC", (width - 85, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if paused:
        cv2.putText(display, "PAUSED", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO Webcam", display)

    # Wait longer while paused to avoid busy-looping on a frozen frame
    key = cv2.waitKey(30 if paused else 1) & 0xFF

    # Press 'p' to pause or resume
    if key == ord("p"):
        paused = not paused
        if not paused:
            # Reset timing so the pause doesn't drag the FPS average down
            prev_time = time.time()
            frame_times.clear()

    # Press 's' to save the current annotated frame
    if key == ord("s"):
        filename = capture_path("capture", "jpg")
        cv2.imwrite(str(filename), annotated)
        print(f"Saved {filename}")

    # Press 'r' to start or stop recording
    if key == ord("r"):
        if writer is None:
            filename = capture_path("recording", "mp4")
            height, width = annotated.shape[:2]
            # Use the measured FPS so the video plays back at real speed
            record_fps = max(1, round(fps))
            writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"),
                                     record_fps, (width, height))
            if writer.isOpened():
                print(f"Recording to {filename} at {record_fps} FPS")
            else:
                print("Could not start recording.")
                writer = None
        else:
            writer.release()
            writer = None
            print("Recording stopped.")

    # Press 'q' to quit
    if key == ord("q"):
        break

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()
