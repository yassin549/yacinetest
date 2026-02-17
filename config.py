import os
from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "data"
    persons_dir: str = "data/persons"
    db_path: str = "data/people.db"
    env_path: str = ".env"

    insightface_model: str = "buffalo_s"
    pose_model_path: str = "yolov8n-pose.pt"
    yolo_model_path: str = "yolov8n.pt"

    scale: float = 0.45
    imgsz: int = 320
    conf: float = 0.45
    iou: float = 0.5
    max_det: int = 50
    device: str = "cpu"

    infer_every: int = 3
    face_every: int = 4
    action_every: int = 6

    min_face_size: int = 40
    face_save_cooldown_sec: float = 2.0
    cosine_threshold: float = 0.35
    action_min_interval_sec: float = 2.0
    face_det_size: int = 480

    caption_enabled: bool = False
    caption_every_sec: float = 5.0
    caption_img_size: int = 384
    caption_model_id: str = "Salesforce/blip-image-captioning-base"
    caption_max_len: int = 30
    use_caption_as_action: bool = False
    caption_action_min_interval_sec: float = 5.0

    display_fps: int = 30
    display_sleep_min: float = 0.001
    draw_labels: bool = True
    show_conf: bool = True
    headless: bool = False

    rtsp_url: str = ""
    log_level: str = "INFO"


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                val = val[1:-1]
            os.environ.setdefault(key, val)


def _build_rtsp_url_from_env() -> str:
    env_url = os.getenv("RTSP_URL")
    if env_url:
        return env_url

    user = os.getenv("RTSP_USERNAME")
    password = os.getenv("RTSP_PASSWORD")
    host = os.getenv("RTSP_HOST", "192.168.1.29")
    port = int(os.getenv("RTSP_PORT", "554"))
    path = os.getenv("RTSP_PATH", "/cam/realmonitor?channel=1&subtype=0")
    if not path.startswith("/"):
        path = f"/{path}"

    if not (user and password):
        raise RuntimeError(
            "Missing RTSP credentials. Set RTSP_URL or RTSP_USERNAME/RTSP_PASSWORD in environment or .env file."
        )
    return f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}{path}"


def load_config(env_path: str = ".env") -> AppConfig:
    load_dotenv(env_path)

    cfg = AppConfig(
        env_path=env_path,
        data_dir=os.getenv("DATA_DIR", "data"),
        persons_dir=os.getenv("PERSONS_DIR", os.path.join(os.getenv("DATA_DIR", "data"), "persons")),
        db_path=os.getenv("DB_PATH", os.path.join(os.getenv("DATA_DIR", "data"), "people.db")),
        insightface_model=os.getenv("INSIGHTFACE_MODEL", "buffalo_s"),
        pose_model_path=os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt"),
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        scale=float(os.getenv("SCALE", "0.45")),
        imgsz=int(os.getenv("IMGSZ", "320")),
        conf=float(os.getenv("CONF", "0.45")),
        iou=float(os.getenv("IOU", "0.5")),
        max_det=int(os.getenv("MAX_DET", "50")),
        device=os.getenv("DEVICE", "cpu"),
        infer_every=int(os.getenv("INFER_EVERY", "3")),
        face_every=int(os.getenv("FACE_EVERY", "4")),
        action_every=int(os.getenv("ACTION_EVERY", "6")),
        min_face_size=int(os.getenv("MIN_FACE_SIZE", "40")),
        face_save_cooldown_sec=float(os.getenv("FACE_SAVE_COOLDOWN_SEC", "2.0")),
        cosine_threshold=float(os.getenv("COSINE_THRESHOLD", "0.35")),
        action_min_interval_sec=float(os.getenv("ACTION_MIN_INTERVAL_SEC", "2.0")),
        face_det_size=int(os.getenv("FACE_DET_SIZE", "480")),
        caption_enabled=_parse_bool(os.getenv("CAPTION_ENABLED"), False),
        caption_every_sec=float(os.getenv("CAPTION_EVERY_SEC", "5.0")),
        caption_img_size=int(os.getenv("CAPTION_IMG_SIZE", "384")),
        caption_model_id=os.getenv("CAPTION_MODEL_ID", "Salesforce/blip-image-captioning-base"),
        caption_max_len=int(os.getenv("CAPTION_MAX_LEN", "30")),
        use_caption_as_action=_parse_bool(os.getenv("USE_CAPTION_AS_ACTION"), False),
        caption_action_min_interval_sec=float(os.getenv("CAPTION_ACTION_MIN_INTERVAL_SEC", "5.0")),
        display_fps=int(os.getenv("DISPLAY_FPS", "30")),
        display_sleep_min=float(os.getenv("DISPLAY_SLEEP_MIN", "0.001")),
        draw_labels=_parse_bool(os.getenv("DRAW_LABELS"), True),
        show_conf=_parse_bool(os.getenv("SHOW_CONF"), True),
        headless=_parse_bool(os.getenv("HEADLESS"), False),
        rtsp_url=_build_rtsp_url_from_env(),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )

    if cfg.scale <= 0:
        raise ValueError("SCALE must be > 0")
    if cfg.imgsz <= 0:
        raise ValueError("IMGSZ must be > 0")
    if cfg.infer_every < 1 or cfg.face_every < 1 or cfg.action_every < 1:
        raise ValueError("INFER_EVERY, FACE_EVERY, ACTION_EVERY must be >= 1")
    if cfg.display_fps < 1:
        raise ValueError("DISPLAY_FPS must be >= 1")
    if cfg.min_face_size < 1:
        raise ValueError("MIN_FACE_SIZE must be >= 1")

    return cfg
