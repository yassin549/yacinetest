import asyncio
import http.server
import json
import logging
import os
import socketserver
import threading
import time

import cv2
import numpy as np
import torch
import websockets
from PIL import Image
from ultralytics import YOLO

from config import AppConfig, load_config
from db import DBClient

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
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class FrameStreamer:
    def __init__(self, jpeg_quality=72):
        self.lock = threading.Lock()
        self.jpg = None
        self.index = 0
        self.jpeg_quality = int(jpeg_quality)

    def update(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self.lock:
            self.jpg = buf.tobytes()
            self.index += 1

    def get_jpg(self):
        with self.lock:
            return self.jpg

    def get_latest(self):
        with self.lock:
            return self.jpg, self.index


class MJPEGServer:
    def __init__(self, host, port, streamer: FrameStreamer, logger):
        self.host = host
        self.port = port
        self.streamer = streamer
        self.logger = logger
        self.httpd = None
        self.thread = None

    def start(self):
        streamer = self.streamer
        logger = self.logger

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/snapshot"):
                    jpg = streamer.get_jpg()
                    if jpg is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    try:
                        self.wfile.write(jpg)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        return
                    return

                if self.path != "/stream":
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()

                try:
                    while True:
                        jpg = streamer.get_jpg()
                        if jpg is None:
                            time.sleep(0.03)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                except Exception:
                    logger.exception("mjpeg client error")

            def log_message(self, *_args):
                return

        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.logger.info("mjpeg server started", extra={"host": self.host, "port": self.port})
        return self

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
        if self.thread is not None:
            self.thread.join(timeout=2)


class WSFrameServer:
    def __init__(self, host, port, streamer: FrameStreamer, logger, flags, stream_fps=24):
        self.host = host
        self.port = port
        self.streamer = streamer
        self.logger = logger
        self.flags = flags
        self.stream_fps = max(1, int(stream_fps))
        self.thread = None
        self.loop = None

    async def _handler(self, websocket, _path=None):
        last_sent = -1
        send_interval = 1.0 / float(self.stream_fps)
        next_send_at = 0.0

        async def _recv_loop():
            try:
                async for msg in websocket:
                    if isinstance(msg, bytes):
                        continue
                    try:
                        payload = json.loads(msg)
                    except Exception:
                        continue
                    if payload.get("type") == "settings":
                        self.flags.update_from(payload)
            except websockets.ConnectionClosed:
                return

        recv_task = asyncio.create_task(_recv_loop())
        try:
            while True:
                jpg, idx = self.streamer.get_latest()
                if jpg is None or idx == last_sent:
                    await asyncio.sleep(0.01)
                    continue
                now = time.perf_counter()
                if now < next_send_at:
                    await asyncio.sleep(next_send_at - now)
                next_send_at = time.perf_counter() + send_interval
                await websocket.send(jpg)
                last_sent = idx
        except websockets.ConnectionClosed:
            return
        except Exception:
            self.logger.exception("ws client error")
        finally:
            recv_task.cancel()

    async def _run(self):
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            max_size=None,
            compression=None,
        ):
            self.logger.info("ws stream started", extra={"host": self.host, "port": self.port})
            await asyncio.Future()

    def start(self):
        self.logger.info("ws stream start requested", extra={"host": self.host, "port": self.port})

        def _thread_main():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self._run())
            except Exception:
                self.logger.exception("ws stream failed to start")

        self.thread = threading.Thread(target=_thread_main, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=2)


class LatestFrame:
    def __init__(self, capture, rtsp_url, logger):
        self.capture = capture
        self.rtsp_url = rtsp_url
        self.logger = logger
        self.lock = threading.Lock()
        self.frame = None
        self.index = 0
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.fail_count = 0

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                self.fail_count += 1
                if self.fail_count >= 50:
                    self.logger.warning("reopening rtsp stream after consecutive read failures")
                    try:
                        self.capture.release()
                    except Exception:
                        pass
                    self.capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.fail_count = 0
                time.sleep(0.01)
                continue
            self.fail_count = 0
            with self.lock:
                self.frame = frame
                self.index += 1

    def get(self):
        with self.lock:
            return self.frame, self.index

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
        self.thread = None
        self.processor = None
        self.model = None
        self.device = torch.device("cpu")
        self.model_ready = False
        self.model_failed = False

        self.enabled = bool(cfg.caption_enabled and _HAS_VLM)
        if cfg.caption_enabled and not _HAS_VLM:
            logger.warning("captioning disabled: transformers package is not installed")
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        if self.enabled and self.thread is not None:
            self.thread.start()
        return self

    def submit(self, frame):
        if not self.enabled:
            return
        with self.input_lock:
            self.pending_frame = frame

    def _worker(self):
        if not self._init_model():
            return

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
                    out = self.model.generate(**inputs, max_length=self.cfg.caption_max_len, num_beams=1)

                caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
                with self.output_lock:
                    self.last_caption = caption
            except Exception:
                self.logger.exception("caption worker error")

    def _init_model(self):
        if self.model_ready:
            return True
        if self.model_failed:
            return False

        try:
            self.processor = BlipProcessor.from_pretrained(self.cfg.caption_model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(self.cfg.caption_model_id)
            self.model.to(self.device)
            self.model.eval()
            self.model_ready = True
            self.logger.info("caption model ready", extra={"caption_model_id": self.cfg.caption_model_id})
            return True
        except Exception:
            self.model_failed = True
            self.logger.exception("captioning disabled: failed to initialize model")
            return False

    def get(self):
        if not self.enabled:
            return ""
        with self.output_lock:
            return self.last_caption

    def stop(self):
        self.running = False
        if self.enabled and self.thread is not None:
            self.thread.join(timeout=2)


class RuntimeFlags:
    def __init__(self, draw_boxes: bool, draw_labels: bool, show_conf: bool):
        self.lock = threading.Lock()
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.show_conf = show_conf

    def update_from(self, payload):
        with self.lock:
            if "draw_boxes" in payload:
                self.draw_boxes = bool(payload["draw_boxes"])
            if "draw_labels" in payload:
                self.draw_labels = bool(payload["draw_labels"])
            if "show_conf" in payload:
                self.show_conf = bool(payload["show_conf"])

    def snapshot(self):
        with self.lock:
            return self.draw_boxes, self.draw_labels, self.show_conf


class AppRuntime:
    def __init__(self, cfg: AppConfig):
        _configure_logging(cfg.log_level)
        self.log = logging.getLogger("runtime")
        self.cfg = cfg
        self.db_client = DBClient(cfg.db_path, cfg.data_dir, cfg.persons_dir)
        self.flags = RuntimeFlags(cfg.draw_boxes, cfg.draw_labels, cfg.show_conf)

        self.last_action_at = {}
        self.last_caption_action_at_by_camera = {}
        self.last_caption_text_by_camera = {}
        self.fallback_person_id = None
        self.fallback_person_lock = threading.Lock()
        self.stop_event = threading.Event()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;2000000|buffer_size;102400"
        )

    def _create_detection_model(self):
        model = YOLO(self.cfg.yolo_model_path)
        try:
            model.fuse()
        except Exception:
            pass

        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        _ = model.predict(
            dummy,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            verbose=False,
            half=False,
            device=self.cfg.device,
        )
        return model

    def _bootstrap(self):
        os.makedirs(self.cfg.persons_dir, exist_ok=True)
        conn = self.db_client.ensure_db()
        conn.close()

    def _resolve_action_person_id(self, person_id):
        if person_id is not None:
            return person_id
        with self.fallback_person_lock:
            if self.fallback_person_id is None:
                self.fallback_person_id = self.db_client.create_placeholder_person()
            return self.fallback_person_id

    def _record_action(self, person_id, label, confidence, source):
        person_id = self._resolve_action_person_id(person_id)
        now = time.monotonic()
        key = (person_id, label, source)
        last = self.last_action_at.get(key, 0.0)
        if now - last < self.cfg.action_min_interval_sec:
            return
        self.last_action_at[key] = now
        self.db_client.record_action(person_id, label, confidence, source)

    def run(self):
        self._bootstrap()
        self.stop_event.clear()

        transport = os.getenv("RTSP_TRANSPORT", "udp").lower()
        if transport not in ("udp", "tcp"):
            transport = "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|max_delay;0|stimeout;2000000|buffer_size;102400"
        )

        try:
            self._run_camera_pipeline(
                {"camera_id": "cam1", "rtsp_url": self.cfg.rtsp_url, "mjpeg_port": 8765, "ws_port": 8766}
            )
        except KeyboardInterrupt:
            self.log.info("shutdown requested")
        finally:
            self.stop_event.set()
            self.db_client.close_thread_conn()

    def _run_camera_pipeline(self, camera):
        camera_id = camera["camera_id"]
        rtsp_url = camera["rtsp_url"]
        model = self._create_detection_model() if self.cfg.detection_enabled else None
        captioner = AsyncCaptioner(self.cfg, self.log).start()

        streamer = FrameStreamer(jpeg_quality=self.cfg.stream_jpeg_quality)
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"Waiting for {camera_id}...",
            (18, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (180, 255, 180),
            2,
        )
        streamer.update(placeholder)
        mjpeg = MJPEGServer("127.0.0.1", camera["mjpeg_port"], streamer, self.log).start()
        ws_server = WSFrameServer(
            "127.0.0.1",
            camera["ws_port"],
            streamer,
            self.log,
            self.flags,
            stream_fps=self.cfg.stream_fps,
        ).start()

        def _open_capture():
            cap_local = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap_local.isOpened():
                return cap_local
            cap_local.release()
            return None

        cap = _open_capture()
        while cap is None and not self.stop_event.is_set():
            self.log.warning("rtsp not available, retrying...", extra={"camera_id": camera_id})
            time.sleep(2.0)
            cap = _open_capture()

        if cap is None:
            captioner.stop()
            mjpeg.stop()
            ws_server.stop()
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        latest = LatestFrame(cap, rtsp_url, self.log).start()

        frame_idx = 0
        next_caption_at = time.monotonic()
        next_display_at = time.perf_counter()
        display_interval = 1.0 / max(1, self.cfg.display_fps)
        last_boxes = np.empty((0, 4), dtype=np.int32)
        last_cls = None
        last_confs = None
        last_names = None

        try:
            while not self.stop_event.is_set():
                now_perf = time.perf_counter()
                if now_perf < next_display_at:
                    time.sleep(max(self.cfg.display_sleep_min, next_display_at - now_perf))
                next_display_at = max(next_display_at + display_interval, now_perf)

                frame, _frame_id = latest.get()
                if frame is None:
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]
                resized = cv2.resize(frame, (int(w * self.cfg.scale), int(h * self.cfg.scale)))
                annotated = resized.copy()

                if model is not None and frame_idx % self.cfg.infer_every == 0:
                    try:
                        results = model.predict(
                            resized,
                            imgsz=self.cfg.imgsz,
                            conf=self.cfg.conf,
                            iou=self.cfg.iou,
                            verbose=False,
                            half=False,
                            device=self.cfg.device,
                            max_det=self.cfg.max_det,
                            stream=False,
                        )
                        r0 = results[0]
                        boxes_obj = r0.boxes
                        if boxes_obj is None or len(boxes_obj) == 0:
                            last_boxes = np.empty((0, 4), dtype=np.int32)
                            last_cls = None
                            last_confs = None
                            last_names = r0.names
                        else:
                            last_boxes = boxes_obj.xyxy.cpu().numpy().astype(np.int32)
                            last_cls = boxes_obj.cls.cpu().numpy().astype(np.int32) if boxes_obj.cls is not None else None
                            last_confs = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else None
                            last_names = r0.names
                    except Exception:
                        self.log.exception("detector inference error", extra={"camera_id": camera_id})

                boxes = last_boxes
                cls = last_cls
                confs = last_confs
                names = last_names
                draw_boxes, draw_labels, show_conf = self.flags.snapshot()
                person_seen = False

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if draw_boxes:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if draw_labels and cls is not None and confs is not None and names is not None:
                        name_idx = int(cls[i])
                        if isinstance(names, dict):
                            name_str = names.get(name_idx, str(name_idx))
                        else:
                            name_str = names[name_idx] if name_idx < len(names) else str(name_idx)
                        label = name_str if not show_conf else f"{name_str} {confs[i] * 100:.1f}%"
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
                        if name_str == "person":
                            person_seen = True

                now = time.monotonic()
                if self.cfg.caption_enabled and now >= next_caption_at:
                    captioner.submit(resized)
                    next_caption_at = now + self.cfg.caption_every_sec
                caption = captioner.get()

                if model is not None and person_seen:
                    self._record_action(None, "person_detected", None, f"{camera_id}:detector")

                if self.cfg.use_caption_as_action and caption:
                    last = self.last_caption_action_at_by_camera.get(camera_id, 0.0)
                    last_text = self.last_caption_text_by_camera.get(camera_id, "")
                    if now - last >= self.cfg.caption_action_min_interval_sec and caption != last_text:
                        self._record_action(None, caption, None, f"{camera_id}:caption")
                        self.last_caption_action_at_by_camera[camera_id] = now
                        self.last_caption_text_by_camera[camera_id] = caption

                streamer.update(annotated)
                frame_idx += 1
        finally:
            captioner.stop()
            latest.stop()
            cap.release()
            mjpeg.stop()
            ws_server.stop()


def main():
    cfg = load_config(".env")
    cfg = AppConfig(**{**cfg.__dict__, "device": "cpu", "headless": True})
    runtime = AppRuntime(cfg)
    runtime.log.info("starting runtime", extra={"device": "cpu"})
    runtime.run()


if __name__ == "__main__":
    main()
