import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Lower latency: use substream (subtype=1) if your camera supports it.
RTSP_URL = "rtsp://pyuser:yacine03041973@192.168.1.70:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Camera", 960, 540)

SCALE = 0.6  # change this value (0.5 = 50%, 0.7 = 70%, etc.)
IMGSZ = 640
CONF = 0.4
IOU = 0.5
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"

while True:
    # Drop older frames to keep latency low.
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        continue

    # Resize BEFORE inference (better performance)
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * SCALE), int(h * SCALE)))

    results = model.predict(
        resized,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False,
        half=USE_HALF,
        device=DEVICE,
        stream=False,
    )
    annotated = results[0].plot()

    cv2.imshow("AI Camera", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
