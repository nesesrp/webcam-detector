import time

import cv2
from ultralytics import YOLO

# Class ID of "person" in the COCO dataset
PERSON_CLASS_ID = 0

# Load the model (downloaded automatically on first run, "n" = nano, the fastest)
model = YOLO("yolo11n.pt")

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check System Settings > Privacy > Camera permissions.")

prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from camera.")
        break

    # Run detection (conf: minimum confidence score, imgsz: smaller = faster)
    results = model(frame, conf=0.5, imgsz=320, verbose=False)

    # Draw boxes and labels on the frame
    annotated = results[0].plot()

    # Calculate FPS and draw it on the frame
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Count detected people and draw the count on the frame
    people_count = int((results[0].boxes.cls == PERSON_CLASS_ID).sum())
    cv2.putText(annotated, f"People: {people_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Webcam", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
