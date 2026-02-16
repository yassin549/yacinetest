import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

RTSP_URL = "rtsp://mluser:yacine03041973@192.168.1.29:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(RTSP_URL)

cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Camera", 960, 540)

SCALE = 0.6  # change this value (0.5 = 50%, 0.7 = 70%, etc.)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize BEFORE inference (better performance)
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * SCALE), int(h * SCALE)))

    results = model(resized, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("AI Camera", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
