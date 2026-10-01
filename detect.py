import argparse
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class ID of "person" in the COCO dataset
PERSON_CLASS_ID = 0

# Number of recent frames used to average the FPS (higher = smoother but slower to react)
FPS_WINDOW = 20

# How much the confidence threshold changes per '+' / '-' key press, and its allowed range
CONF_STEP = 0.05
CONF_MIN, CONF_MAX = 0.05, 0.95

# Folder where screenshots and recordings are saved
CAPTURES_DIR = Path("captures")


def capture_path(prefix, extension):
    """Return a unique, timestamped file path inside the captures folder."""
    CAPTURES_DIR.mkdir(exist_ok=True)
    return CAPTURES_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.{extension}"


def format_duration(seconds):
    """Format seconds as MM:SS."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Real-time object detection from a webcam.")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index to open (default: 0)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model weights to load (default: yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="minimum confidence score, between 0 and 1 (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="inference image size, smaller = faster (default: 320)")
    parser.add_argument("--classes", nargs="+", metavar="NAME",
                        help="only detect these classes, e.g. --classes person cup")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    return args


def resolve_class_ids(model, class_names):
    """Convert class names like "person" to the model's class IDs."""
    if not class_names:
        return None
    ids_by_name = {name: class_id for class_id, name in model.names.items()}
    unknown = [name for name in class_names if name not in ids_by_name]
    if unknown:
        raise SystemExit(f"Unknown class name(s): {', '.join(unknown)}\n"
                         f"Available: {', '.join(sorted(ids_by_name))}")
    return [ids_by_name[name] for name in class_names]


def print_session_summary(stats):
    """Print statistics collected during the session."""
    duration = time.time() - stats["start_time"]
    frames = stats["frames"]

    print("\n=== Session Summary ===")
    print(f"Duration: {format_duration(duration)}")
    if frames == 0:
        print("No frames were processed.")
        return

    print(f"Frames processed: {frames}")
    print(f"Average FPS: {frames / stats['active_time']:.1f}")
    if stats["max_people"] > 0:
        print(f"Max people at once: {stats['max_people']} "
              f"(at {format_duration(stats['max_people_at'])})")
    else:
        print("No people detected.")

    if stats["class_frames"]:
        print("Most seen objects (share of frames they appeared in):")
        for name, count in stats["class_frames"].most_common(5):
            print(f"  {name}: {count / frames:.0%}")
    else:
        print("No objects detected.")

    print(f"Screenshots saved: {stats['screenshots']}")
    print(f"Recordings saved: {stats['recordings']}")


args = parse_args()

# Load the model (downloaded automatically on first run, "n" = nano, the fastest)
model = YOLO(args.model)
class_ids = resolve_class_ids(model, args.classes)

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check System Settings > Privacy > Camera permissions.")

conf = args.conf  # Can be changed at runtime with '+' / '-'
prev_time = time.time()
frame_times = deque(maxlen=FPS_WINDOW)
paused = False
writer = None  # cv2.VideoWriter while recording, otherwise None

# Statistics printed when the program exits
stats = {
    "start_time": time.time(),
    "active_time": 0.0,  # Time spent processing frames, excluding pauses
    "frames": 0,
    "max_people": 0,
    "max_people_at": 0.0,  # Seconds since start when max_people was reached
    "class_frames": Counter(),  # Number of frames each class appeared in
    "screenshots": 0,
    "recordings": 0,
}

while True:
    # While paused, skip reading and detection so the last frame stays on screen
    if not paused:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        # Run detection (conf: minimum confidence score, imgsz: smaller = faster)
        results = model(frame, conf=conf, imgsz=args.imgsz, classes=class_ids, verbose=False)

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
        cv2.putText(annotated, f"Conf: {conf:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update session statistics
        stats["frames"] += 1
        stats["active_time"] += frame_times[-1]
        if people_count > stats["max_people"]:
            stats["max_people"] = people_count
            stats["max_people_at"] = now - stats["start_time"]
        detected_names = {results[0].names[int(c)] for c in results[0].boxes.cls}
        stats["class_frames"].update(detected_names)

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
        cv2.putText(display, "PAUSED", (10, 150),
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

    # Press '+' (or '=' so Shift isn't needed) / '-' to raise or lower the confidence threshold
    if key in (ord("+"), ord("="), ord("-")):
        step = -CONF_STEP if key == ord("-") else CONF_STEP
        conf = round(min(CONF_MAX, max(CONF_MIN, conf + step)), 2)
        print(f"Confidence threshold: {conf:.2f}")

    # Press 's' to save the current annotated frame
    if key == ord("s"):
        filename = capture_path("capture", "jpg")
        cv2.imwrite(str(filename), annotated)
        print(f"Saved {filename}")
        stats["screenshots"] += 1

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
                stats["recordings"] += 1
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

print_session_summary(stats)
