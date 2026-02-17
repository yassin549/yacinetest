import os
import threading
import time
from urllib.parse import quote

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from insightface.app import FaceAnalysis
from db import DBClient
from matching import FaceRegistry, assign_actions_from_pose, face_quality, iou

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_VLM = True
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    _HAS_VLM = False

cv2.setUseOptimized(True)
cpu_threads = max(1, (os.cpu_count() or 2) // 2)
cv2.setNumThreads(cpu_threads)
torch.set_num_threads(cpu_threads)
torch.set_num_interop_threads(1)

# Data + DB
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "people.db")
PERSONS_DIR = os.path.join(DATA_DIR, "persons")
CREDENTIALS_PATH = "credentials.txt"

# Face identity
INSIGHTFACE_MODEL = "buffalo_s"  # lightweight model pack
FACE_EVERY = 4  # process faces every N frames
COSINE_THRESHOLD = 0.35  # smaller is stricter (cosine distance)
MIN_FACE_SIZE = 40  # ignore tiny faces
FACE_SAVE_COOLDOWN_SEC = 2.0

# Lightweight action recognition via pose (optional)
POSE_MODEL_PATH = "yolov8n-pose.pt"
POSE_ENABLED = os.path.exists(POSE_MODEL_PATH)
ACTION_EVERY = 6  # run action model every N frames
ACTION_MIN_INTERVAL_SEC = 2.0

# Optional: store BLIP captions as actions
USE_CAPTION_AS_ACTION = False
CAPTION_ACTION_MIN_INTERVAL_SEC = 5.0

# Force low-latency FFmpeg options for RTSP (OpenCV picks these up).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;2000000|buffer_size;102400"
)

SCALE = 0.45  # change this value (0.5 = 50%, 0.7 = 70%, etc.)
IMGSZ = 320
CONF = 0.45
IOU = 0.5
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"
MAX_DET = 50
INFER_EVERY = 3  # send one frame to detector every N displayed frames
DRAW_LABELS = True  # show class labels; set False for maximum smoothness
SHOW_CONF = True    # confidence text adds extra draw cost

CAPTION_ENABLED = False
CAPTION_EVERY_SEC = 5.0
CAPTION_IMG_SIZE = 384
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_LEN = 30

DISPLAY_FPS = 30
DISPLAY_SLEEP_MIN = 0.001

db_client = DBClient(DB_PATH, DATA_DIR, PERSONS_DIR)
face_registry = FaceRegistry(embedding_dim=128)

last_face_saved_at = {}
last_action_at = {}
last_caption_action_at = 0.0
last_person_boxes = []
last_person_ids = []
TRACK_IOU = 0.3
face_app = None

def _load_credentials_file(path):
    user = None
    password = None
    if not os.path.exists(path):
        return user, password
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
            elif "=" in line:
                key, val = line.split("=", 1)
            else:
                continue
            key = key.strip().lower()
            val = val.strip()
            if key in ("username", "user"):
                user = val
            elif key in ("password", "pass"):
                password = val
    return user, password

def _build_rtsp_url():
    env_url = os.getenv("RTSP_URL")
    if env_url:
        return env_url

    user = os.getenv("RTSP_USERNAME")
    password = os.getenv("RTSP_PASSWORD")
    if not (user and password):
        file_user, file_pass = _load_credentials_file(CREDENTIALS_PATH)
        user = user or file_user
        password = password or file_pass

    host = os.getenv("RTSP_HOST", "192.168.1.29")
    port = int(os.getenv("RTSP_PORT", "554"))
    path = os.getenv("RTSP_PATH", "/cam/realmonitor?channel=1&subtype=0")
    if not path.startswith("/"):
        path = f"/{path}"

    if not (user and password):
        raise RuntimeError(
            "Missing RTSP credentials. Set RTSP_URL or RTSP_USERNAME/RTSP_PASSWORD "
            "or provide credentials.txt with username/password."
        )
    return f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}{path}"

class LatestFrame:
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.index = 0
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
                self.index += 1

    def get(self):
        with self.lock:
            return self.frame, self.index

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)

class AsyncDetector:
    def __init__(self, model):
        self.model = model
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending = None  # (frame_id, frame)
        self.last_output = (0, np.empty((0, 4), dtype=np.int32), None, None, None)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame_id, frame):
        # Keep only the newest frame to avoid queue buildup/latency.
        with self.input_lock:
            self.pending = (frame_id, frame)

    def _worker(self):
        while self.running:
            with self.input_lock:
                item = self.pending
                self.pending = None

            if item is None:
                time.sleep(0.001)
                continue

            frame_id, frame = item
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
                packed = (frame_id, np.empty((0, 4), dtype=np.int32), None, None, r0.names)
            else:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
                cls = boxes.cls.cpu().numpy().astype(np.int32) if boxes.cls is not None else None
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                packed = (frame_id, xyxy, cls, confs, r0.names)

            with self.output_lock:
                self.last_output = packed

    def get(self):
        with self.output_lock:
            return self.last_output

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)

def _create_detection_model():
    model = YOLO("yolov8n.pt")
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
    return model

class AsyncCaptioner:
    def __init__(self):
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending_frame = None
        self.last_caption = ""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

        if not CAPTION_ENABLED or not _HAS_VLM:
            self.enabled = False
            return

        self.enabled = True
        self.device = torch.device("cpu")
        self.processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
        self.model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def start(self):
        if self.enabled:
            self.thread.start()
        return self

    def submit(self, frame):
        if not self.enabled:
            return
        with self.input_lock:
            self.pending_frame = frame

    def _worker(self):
        while self.running:
            with self.input_lock:
                frame = self.pending_frame
                self.pending_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            # Convert BGR to RGB and resize for VLM.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = CAPTION_IMG_SIZE / max(h, w)
            resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            image = Image.fromarray(resized)

            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=CAPTION_MAX_LEN,
                    num_beams=3,
                )
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()

            with self.output_lock:
                self.last_caption = caption

    def get(self):
        if not self.enabled:
            return ""
        with self.output_lock:
            return self.last_caption

    def stop(self):
        self.running = False
        if self.enabled:
            self.thread.join(timeout=1)

def _face_quality(face_bgr):
    return face_quality(face_bgr)

def _upsert_person_face(person_id, person_bgr, embedding, score):
    db_client.upsert_person_face(person_id, person_bgr, embedding, score)
    face_registry.upsert(person_id, embedding)

def _create_person(person_bgr, embedding, score):
    person_id = db_client.create_person(embedding)
    _upsert_person_face(person_id, person_bgr, embedding, score)
    return person_id

def _match_face(embedding):
    return face_registry.match(embedding, COSINE_THRESHOLD)

def _record_action(person_id, label, confidence, source):
    if person_id is None:
        return
    now = time.monotonic()
    key = (person_id, label, source)
    last = last_action_at.get(key, 0.0)
    if now - last < ACTION_MIN_INTERVAL_SEC:
        return
    last_action_at[key] = now
    db_client.record_action(person_id, label, confidence, source)

def _assign_actions_from_pose(pose_results, person_boxes, person_ids):
    assign_actions_from_pose(pose_results, person_boxes, person_ids, _record_action)

class AsyncFaceWorker:
    def __init__(self):
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending = None  # (frame_id, frame, person_boxes)
        self.last_output = (0, [], [])
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.last_person_boxes = []
        self.last_person_ids = []

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame_id, frame, person_boxes):
        if not person_boxes:
            return
        with self.input_lock:
            self.pending = (frame_id, frame, person_boxes)

    def _match_prev_ids(self, cur_boxes):
        if not self.last_person_boxes or not self.last_person_ids:
            return [None] * len(cur_boxes)
        ids = [None] * len(cur_boxes)
        for i, box in enumerate(cur_boxes):
            best_i = -1
            best_iou = 0.0
            for j, pbox in enumerate(self.last_person_boxes):
                overlap = iou(box, pbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_i = j
            if best_i != -1 and best_iou >= TRACK_IOU:
                ids[i] = self.last_person_ids[best_i]
        return ids

    def _worker(self):
        global last_face_saved_at
        try:
            while self.running:
                with self.input_lock:
                    item = self.pending
                    self.pending = None

                if item is None:
                    time.sleep(0.001)
                    continue

                frame_id, frame, person_boxes = item
                person_ids = []
                now = time.monotonic()

                for (x1, y1, x2, y2) in person_boxes:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        person_ids.append(None)
                        continue
                    faces = face_app.get(roi)
                    if not faces:
                        person_ids.append(None)
                        continue
                    # pick largest face in ROI
                    faces = sorted(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True,
                    )
                    face = faces[0]
                    fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                        person_ids.append(None)
                        continue
                    face_bgr = roi[fy1:fy2, fx1:fx2]
                    person_bgr = roi
                    embedding = face.normed_embedding.astype(np.float32)
                    match_id, dist = _match_face(embedding)

                    quality = _face_quality(face_bgr)
                    area = (fw * fh) / max(1.0, (x2 - x1) * (y2 - y1))
                    if match_id is None:
                        score = 0.5 * float(face.det_score) + 0.3 * min(1.0, area * 2.0) + 0.2 * min(1.0, quality / 300.0)
                        person_id = _create_person(person_bgr, embedding, score)
                    else:
                        score = max(0.0, 1.0 - dist) * 0.6 + min(1.0, area * 2.0) * 0.2 + min(1.0, quality / 300.0) * 0.2
                        person_id = match_id
                        last_saved = last_face_saved_at.get(person_id, 0.0)
                        if now - last_saved >= FACE_SAVE_COOLDOWN_SEC:
                            _upsert_person_face(person_id, person_bgr, embedding, score)
                            last_face_saved_at[person_id] = now
                    person_ids.append(person_id)

                tracked = self._match_prev_ids(person_boxes)
                person_ids = [pid if pid is not None else tracked[i] for i, pid in enumerate(person_ids)]

                self.last_person_boxes = list(person_boxes)
                self.last_person_ids = list(person_ids)

                with self.output_lock:
                    self.last_output = (frame_id, list(person_boxes), list(person_ids))
        finally:
            db_client.close_thread_conn()

    def get(self):
        with self.output_lock:
            return self.last_output

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)

class AsyncPoseWorker:
    def __init__(self, model):
        self.model = model
        self.input_lock = threading.Lock()
        self.pending = None  # (frame_id, frame, person_boxes, person_ids)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        if self.model is None:
            return self
        self.thread.start()
        return self

    def submit(self, frame_id, frame, person_boxes, person_ids):
        if self.model is None or not person_boxes:
            return
        with self.input_lock:
            self.pending = (frame_id, frame, person_boxes, person_ids)

    def _worker(self):
        try:
            while self.running:
                with self.input_lock:
                    item = self.pending
                    self.pending = None

                if item is None:
                    time.sleep(0.001)
                    continue

                frame_id, frame, person_boxes, person_ids = item
                pose_results = self.model.predict(
                    frame,
                    imgsz=IMGSZ,
                    conf=0.25,
                    iou=0.5,
                    verbose=False,
                    device=DEVICE,
                )
                if not person_ids:
                    person_ids = [None] * len(person_boxes)
                _assign_actions_from_pose(pose_results, person_boxes, person_ids)
        finally:
            db_client.close_thread_conn()

    def stop(self):
        self.running = False
        if self.model is not None:
            self.thread.join(timeout=1)

def _close_thread_db():
    db_client.close_thread_conn()

def _bootstrap_runtime():
    global face_app
    os.makedirs(PERSONS_DIR, exist_ok=True)
    conn = db_client.ensure_db()
    known_ids, known_embeddings = db_client.load_known_faces(conn)
    conn.close()
    face_registry.load(known_ids, known_embeddings)
    face_app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(480, 480))

def main():
    global last_caption_action_at
    _bootstrap_runtime()
    rtsp_url = _build_rtsp_url()
    model = _create_detection_model()
    pose_model = YOLO(POSE_MODEL_PATH) if POSE_ENABLED else None
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Camera", 960, 540)

    latest = LatestFrame(cap).start()
    detector = AsyncDetector(model).start()
    captioner = AsyncCaptioner().start()
    face_worker = AsyncFaceWorker().start()
    pose_worker = AsyncPoseWorker(pose_model).start()

    frame_idx = 0
    next_caption_at = time.monotonic()
    next_display_at = time.perf_counter()
    display_interval = 1.0 / max(1, DISPLAY_FPS)
    try:
        while True:
            # Display pacing to reduce bursty frame timing and make motion smoother.
            now_perf = time.perf_counter()
            if now_perf < next_display_at:
                time.sleep(max(DISPLAY_SLEEP_MIN, next_display_at - now_perf))
            next_display_at = max(next_display_at + display_interval, now_perf)

            frame, frame_id = latest.get()
            if frame is None:
                time.sleep(0.005)
                continue

            # Resize BEFORE inference (better performance)
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (int(w * SCALE), int(h * SCALE)))

            if frame_idx % INFER_EVERY == 0:
                detector.submit(frame_id, resized)

            annotated = resized.copy()
            _, boxes, cls, confs, names = detector.get()
            person_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if DRAW_LABELS and cls is not None and confs is not None and names is not None:
                    name_idx = int(cls[i])
                    if isinstance(names, dict):
                        name_str = names.get(name_idx, str(name_idx))
                    else:
                        name_str = names[name_idx] if name_idx < len(names) else str(name_idx)
                    label = name_str if not SHOW_CONF else f"{name_str} {confs[i] * 100:.1f}%"
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
                if names is not None and cls is not None:
                    name_idx = int(cls[i])
                    if isinstance(names, dict):
                        name_str = names.get(name_idx, "")
                    else:
                        name_str = names[name_idx] if name_idx < len(names) else ""
                    if name_str == "person":
                        person_boxes.append((x1, y1, x2, y2))

            now = time.monotonic()
            if CAPTION_ENABLED and now >= next_caption_at:
                captioner.submit(resized)
                next_caption_at = now + CAPTION_EVERY_SEC

            caption = captioner.get()
            if caption:
                cv2.putText(
                    annotated,
                    caption,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Face identity matching (async)
            if frame_idx % FACE_EVERY == 0 and person_boxes:
                face_worker.submit(frame_id, resized, person_boxes)
            _, face_boxes, face_ids = face_worker.get()
            person_ids = []
            if face_boxes:
                # Use the most recent face result even if it's slightly stale.
                person_ids = list(face_ids)
            if len(person_ids) != len(person_boxes):
                person_ids = [None] * len(person_boxes)

            # Action recognition from pose model (async)
            if POSE_ENABLED and frame_idx % ACTION_EVERY == 0 and person_boxes:
                pose_worker.submit(frame_id, resized, person_boxes, person_ids)

            # Optional: store caption as action for all visible persons
            if USE_CAPTION_AS_ACTION and caption and person_boxes:
                if now - last_caption_action_at >= CAPTION_ACTION_MIN_INTERVAL_SEC:
                    for pid in person_ids:
                        if pid is not None:
                            _record_action(pid, caption, None, "caption")
                    last_caption_action_at = now

            if person_boxes and person_ids:
                last_person_boxes[:] = person_boxes
                last_person_ids[:] = person_ids

            cv2.imshow("AI Camera", annotated)

            if cv2.waitKey(1) == 27:
                break

            frame_idx += 1
    finally:
        detector.stop()
        captioner.stop()
        face_worker.stop()
        pose_worker.stop()
        latest.stop()
        cap.release()
        cv2.destroyAllWindows()
        _close_thread_db()

if __name__ == "__main__":
    main()
