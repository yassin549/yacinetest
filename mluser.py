import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

cv2.setUseOptimized(True)
cpu_threads = max(1, (os.cpu_count() or 2) // 2)
cv2.setNumThreads(cpu_threads)
torch.set_num_threads(cpu_threads)
torch.set_num_interop_threads(1)

model = YOLO("yolov8n.pt")

# Lower latency: use substream (subtype=1) if your camera supports it.
RTSP_URL = "rtsp://pyuser:yacine03041973@192.168.1.70:554/cam/realmonitor?channel=1&subtype=1"

# Force low-latency FFmpeg options for RTSP (OpenCV picks these up).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;2000000|buffer_size;102400"
)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Camera", 960, 540)

SCALE = 0.5  # change this value (0.5 = 50%, 0.7 = 70%, etc.)
IMGSZ = 416
CONF = 0.4
IOU = 0.5
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"
MAX_DET = 50
INFER_EVERY = 2  # send one frame to detector every N displayed frames
DRAW_LABELS = True  # show class labels; set False for maximum smoothness
SHOW_CONF = True    # confidence text adds extra draw cost

class LatestFrame:
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame

    def get(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)

latest = LatestFrame(cap).start()

class AsyncDetector:
    def __init__(self, model):
        self.model = model
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending_frame = None
        self.last_output = (np.empty((0, 4), dtype=np.int32), None, None, None)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame):
        # Keep only the newest frame to avoid queue buildup/latency.
        with self.input_lock:
            self.pending_frame = frame

    def _worker(self):
        while self.running:
            with self.input_lock:
                frame = self.pending_frame
                self.pending_frame = None

            if frame is None:
                time.sleep(0.001)
                continue

            results = self.model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False,
                half=USE_HALF,
                device=DEVICE,
                max_det=MAX_DET,
                stream=False,
            )
            r0 = results[0]
            boxes = r0.boxes
            if boxes is None or len(boxes) == 0:
                packed = (np.empty((0, 4), dtype=np.int32), None, None, r0.names)
            else:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
                cls = boxes.cls.cpu().numpy().astype(np.int32) if boxes.cls is not None else None
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                packed = (xyxy, cls, confs, r0.names)

            with self.output_lock:
                self.last_output = packed

    def get(self):
        with self.output_lock:
            return self.last_output

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)

if DEVICE != "cpu":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Model fusing + warmup reduces first-frame latency.
    model.fuse()
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    _ = model.predict(
        dummy,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False,
        half=USE_HALF,
        device=DEVICE,
    )

frame_idx = 0
detector = AsyncDetector(model).start()

while True:
    frame = latest.get()
    if frame is None:
        time.sleep(0.005)
        continue

    # Resize BEFORE inference (better performance)
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * SCALE), int(h * SCALE)))

    if frame_idx % INFER_EVERY == 0:
        detector.submit(resized)

    annotated = resized.copy()
    boxes, cls, confs, names = detector.get()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if DRAW_LABELS and cls is not None and confs is not None and names is not None:
            label = names[int(cls[i])] if not SHOW_CONF else f"{names[int(cls[i])]} {confs[i] * 100:.1f}%"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("AI Camera", annotated)

    if cv2.waitKey(1) == 27:
        break

    frame_idx += 1

detector.stop()
latest.stop()
cap.release()
cv2.destroyAllWindows()
