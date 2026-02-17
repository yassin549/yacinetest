import logging
import os
import threading
import time

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from ultralytics import YOLO

from config import AppConfig, load_config
from db import DBClient
from matching import FaceRegistry, assign_actions_from_pose, face_quality, iou

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    _HAS_VLM = True
except Exception:
    BlipForConditionalGeneration = None
    BlipProcessor = None
    _HAS_VLM = False

cv2.setUseOptimized(True)
CPU_THREADS = max(1, (os.cpu_count() or 2) // 2)
cv2.setNumThreads(CPU_THREADS)
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)


def _configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class LatestFrame:
    def __init__(self, capture, logger):
        self.capture = capture
        self.logger = logger
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
        self.thread.join(timeout=2)


class AsyncDetector:
    def __init__(self, model, cfg: AppConfig, logger):
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending = None
        self.last_output = (0, np.empty((0, 4), dtype=np.int32), None, None, None)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame_id, frame):
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
            try:
                results = self.model.predict(
                    frame,
                    imgsz=self.cfg.imgsz,
                    conf=self.cfg.conf,
                    iou=self.cfg.iou,
                    verbose=False,
                    half=(self.cfg.device != "cpu"),
                    device=self.cfg.device,
                    max_det=self.cfg.max_det,
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
            except Exception:
                self.logger.exception("detector worker error")

    def get(self):
        with self.output_lock:
            return self.last_output

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


class AsyncCaptioner:
    def __init__(self, cfg: AppConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.pending_frame = None
        self.last_caption = ""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)

        if not cfg.caption_enabled or not _HAS_VLM:
            self.enabled = False
            return

        self.enabled = True
        self.device = torch.device("cpu")
        self.processor = BlipProcessor.from_pretrained(cfg.caption_model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(cfg.caption_model_id)
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

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                scale = self.cfg.caption_img_size / max(h, w)
                resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                image = Image.fromarray(resized)

                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=self.cfg.caption_max_len,
                        num_beams=3,
                    )
                caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
                with self.output_lock:
                    self.last_caption = caption
            except Exception:
                self.logger.exception("caption worker error")

    def get(self):
        if not self.enabled:
            return ""
        with self.output_lock:
            return self.last_caption

    def stop(self):
        self.running = False
        if self.enabled:
            self.thread.join(timeout=2)


class AppRuntime:
    def __init__(self, cfg: AppConfig):
        _configure_logging(cfg.log_level)
        self.log = logging.getLogger("runtime")
        self.cfg = cfg
        self.pose_enabled = os.path.exists(cfg.pose_model_path)
        self.db_client = DBClient(cfg.db_path, cfg.data_dir, cfg.persons_dir)
        self.face_registry = FaceRegistry(embedding_dim=128)
        self.face_app = None

        self.last_face_saved_at = {}
        self.last_action_at = {}
        self.last_caption_action_at = 0.0

        # Force low-latency FFmpeg options for RTSP.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;2000000|buffer_size;102400"
        )

    def _create_detection_model(self):
        model = YOLO(self.cfg.yolo_model_path)
        if self.cfg.device != "cpu":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            model.fuse()
            dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
            _ = model.predict(
                dummy,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                verbose=False,
                half=True,
                device=self.cfg.device,
            )
        return model

    def _bootstrap(self):
        os.makedirs(self.cfg.persons_dir, exist_ok=True)
        conn = self.db_client.ensure_db()
        known_ids, known_embeddings = self.db_client.load_known_faces(conn)
        conn.close()
        self.face_registry.load(known_ids, known_embeddings)

        self.face_app = FaceAnalysis(name=self.cfg.insightface_model, providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=-1, det_size=(self.cfg.face_det_size, self.cfg.face_det_size))

    def _match_face(self, embedding):
        return self.face_registry.match(embedding, self.cfg.cosine_threshold)

    def _upsert_person_face(self, person_id, person_bgr, embedding, score):
        self.db_client.upsert_person_face(person_id, person_bgr, embedding, score)
        self.face_registry.upsert(person_id, embedding)

    def _create_person(self, person_bgr, embedding, score):
        person_id = self.db_client.create_person(embedding)
        self._upsert_person_face(person_id, person_bgr, embedding, score)
        return person_id

    def _record_action(self, person_id, label, confidence, source):
        if person_id is None:
            return
        now = time.monotonic()
        key = (person_id, label, source)
        last = self.last_action_at.get(key, 0.0)
        if now - last < self.cfg.action_min_interval_sec:
            return
        self.last_action_at[key] = now
        self.db_client.record_action(person_id, label, confidence, source)

    def _assign_actions_from_pose(self, pose_results, person_boxes, person_ids):
        assign_actions_from_pose(pose_results, person_boxes, person_ids, self._record_action)

    def _make_face_worker(self):
        runtime = self

        class AsyncFaceWorker:
            def __init__(self):
                self.input_lock = threading.Lock()
                self.output_lock = threading.Lock()
                self.pending = None
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
                    if best_i != -1 and best_iou >= 0.3:
                        ids[i] = self.last_person_ids[best_i]
                return ids

            def _worker(self):
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
                            try:
                                faces = runtime.face_app.get(roi)
                            except Exception:
                                runtime.log.exception("face_app.get failed")
                                person_ids.append(None)
                                continue
                            if not faces:
                                person_ids.append(None)
                                continue

                            faces = sorted(
                                faces,
                                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                                reverse=True,
                            )
                            face = faces[0]
                            fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                            roi_h, roi_w = roi.shape[:2]
                            fx1 = max(0, min(fx1, roi_w - 1))
                            fy1 = max(0, min(fy1, roi_h - 1))
                            fx2 = max(0, min(fx2, roi_w))
                            fy2 = max(0, min(fy2, roi_h))
                            fw = fx2 - fx1
                            fh = fy2 - fy1
                            if fw < runtime.cfg.min_face_size or fh < runtime.cfg.min_face_size:
                                person_ids.append(None)
                                continue

                            face_bgr = roi[fy1:fy2, fx1:fx2]
                            if face_bgr.size == 0:
                                person_ids.append(None)
                                continue

                            person_bgr = roi
                            embedding = face.normed_embedding.astype(np.float32)
                            match_id, dist = runtime._match_face(embedding)

                            quality = face_quality(face_bgr)
                            area = (fw * fh) / max(1.0, (x2 - x1) * (y2 - y1))
                            if match_id is None:
                                score = 0.5 * float(face.det_score) + 0.3 * min(1.0, area * 2.0) + 0.2 * min(1.0, quality / 300.0)
                                person_id = runtime._create_person(person_bgr, embedding, score)
                            else:
                                score = max(0.0, 1.0 - dist) * 0.6 + min(1.0, area * 2.0) * 0.2 + min(1.0, quality / 300.0) * 0.2
                                person_id = match_id
                                last_saved = runtime.last_face_saved_at.get(person_id, 0.0)
                                if now - last_saved >= runtime.cfg.face_save_cooldown_sec:
                                    runtime._upsert_person_face(person_id, person_bgr, embedding, score)
                                    runtime.last_face_saved_at[person_id] = now
                            person_ids.append(person_id)

                        tracked = self._match_prev_ids(person_boxes)
                        person_ids = [pid if pid is not None else tracked[i] for i, pid in enumerate(person_ids)]
                        self.last_person_boxes = list(person_boxes)
                        self.last_person_ids = list(person_ids)
                        with self.output_lock:
                            self.last_output = (frame_id, list(person_boxes), list(person_ids))
                except Exception:
                    runtime.log.exception("face worker fatal error")
                finally:
                    runtime.db_client.close_thread_conn()

            def get(self):
                with self.output_lock:
                    return self.last_output

            def stop(self):
                self.running = False
                self.thread.join(timeout=2)

        return AsyncFaceWorker()

    def _make_pose_worker(self, pose_model):
        runtime = self

        class AsyncPoseWorker:
            def __init__(self, model):
                self.model = model
                self.input_lock = threading.Lock()
                self.pending = None
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
                        try:
                            pose_results = self.model.predict(
                                frame,
                                imgsz=runtime.cfg.imgsz,
                                conf=0.25,
                                iou=0.5,
                                verbose=False,
                                device=runtime.cfg.device,
                            )
                            if not person_ids:
                                person_ids = [None] * len(person_boxes)
                            runtime._assign_actions_from_pose(pose_results, person_boxes, person_ids)
                        except Exception:
                            runtime.log.exception("pose worker inference error")
                finally:
                    runtime.db_client.close_thread_conn()

            def stop(self):
                self.running = False
                if self.model is not None:
                    self.thread.join(timeout=2)

        return AsyncPoseWorker(pose_model)

    def run(self):
        self._bootstrap()

        model = self._create_detection_model()
        pose_model = YOLO(self.cfg.pose_model_path) if self.pose_enabled else None

        cap = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError("Unable to open RTSP stream")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cfg.headless:
            cv2.namedWindow("AI Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AI Camera", 960, 540)

        latest = LatestFrame(cap, self.log).start()
        detector = AsyncDetector(model, self.cfg, self.log).start()
        captioner = AsyncCaptioner(self.cfg, self.log).start()
        face_worker = self._make_face_worker().start()
        pose_worker = self._make_pose_worker(pose_model).start()

        frame_idx = 0
        next_caption_at = time.monotonic()
        next_display_at = time.perf_counter()
        display_interval = 1.0 / max(1, self.cfg.display_fps)

        try:
            while True:
                now_perf = time.perf_counter()
                if now_perf < next_display_at:
                    time.sleep(max(self.cfg.display_sleep_min, next_display_at - now_perf))
                next_display_at = max(next_display_at + display_interval, now_perf)

                frame, frame_id = latest.get()
                if frame is None:
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]
                resized = cv2.resize(frame, (int(w * self.cfg.scale), int(h * self.cfg.scale)))

                if frame_idx % self.cfg.infer_every == 0:
                    detector.submit(frame_id, resized)

                annotated = resized.copy()
                _, boxes, cls, confs, names = detector.get()

                person_boxes = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if not self.cfg.headless:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if self.cfg.draw_labels and cls is not None and confs is not None and names is not None and not self.cfg.headless:
                        name_idx = int(cls[i])
                        if isinstance(names, dict):
                            name_str = names.get(name_idx, str(name_idx))
                        else:
                            name_str = names[name_idx] if name_idx < len(names) else str(name_idx)
                        label = name_str if not self.cfg.show_conf else f"{name_str} {confs[i] * 100:.1f}%"
                        cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    if names is not None and cls is not None:
                        name_idx = int(cls[i])
                        if isinstance(names, dict):
                            name_str = names.get(name_idx, "")
                        else:
                            name_str = names[name_idx] if name_idx < len(names) else ""
                        if name_str == "person":
                            person_boxes.append((x1, y1, x2, y2))

                now = time.monotonic()
                if self.cfg.caption_enabled and now >= next_caption_at:
                    captioner.submit(resized)
                    next_caption_at = now + self.cfg.caption_every_sec

                caption = captioner.get()
                if caption and not self.cfg.headless:
                    cv2.putText(annotated, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

                if frame_idx % self.cfg.face_every == 0 and person_boxes:
                    face_worker.submit(frame_id, resized, person_boxes)
                _, face_boxes, face_ids = face_worker.get()

                person_ids = []
                if face_boxes:
                    person_ids = list(face_ids)
                if len(person_ids) != len(person_boxes):
                    person_ids = [None] * len(person_boxes)

                if self.pose_enabled and frame_idx % self.cfg.action_every == 0 and person_boxes:
                    pose_worker.submit(frame_id, resized, person_boxes, person_ids)

                if self.cfg.use_caption_as_action and caption and person_boxes:
                    if now - self.last_caption_action_at >= self.cfg.caption_action_min_interval_sec:
                        for pid in person_ids:
                            if pid is not None:
                                self._record_action(pid, caption, None, "caption")
                        self.last_caption_action_at = now

                if self.cfg.headless:
                    frame_idx += 1
                    continue

                cv2.imshow("AI Camera", annotated)
                if cv2.waitKey(1) == 27:
                    break
                frame_idx += 1
        except KeyboardInterrupt:
            self.log.info("shutdown requested")
        finally:
            detector.stop()
            captioner.stop()
            face_worker.stop()
            pose_worker.stop()
            latest.stop()
            cap.release()
            if not self.cfg.headless:
                cv2.destroyAllWindows()
            self.db_client.close_thread_conn()


def main():
    cfg = load_config(".env")
    device = 0 if torch.cuda.is_available() else "cpu"
    cfg = AppConfig(**{**cfg.__dict__, "device": device})
    runtime = AppRuntime(cfg)
    runtime.log.info("starting runtime", extra={"device": str(device)})
    runtime.run()


if __name__ == "__main__":
    main()
