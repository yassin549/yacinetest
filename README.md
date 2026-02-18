# MLUser Production Runtime

MLUser is a Python RTSP vision runtime with a Tauri + Svelte desktop control UI.

## What this build does

- Runs YOLO object detection on an RTSP stream.
- Streams annotated frames over local MJPEG and WebSocket endpoints.
- Stores deduplicated action events in SQLite.
- Optionally runs BLIP captioning and logs captions as actions.

## Repository layout

- `mluser.py`: runtime entrypoint.
- `runtime.py`: production pipeline (capture, detect, stream, action logging).
- `config.py`: `.env` parsing and validated runtime config.
- `db.py`: SQLite schema and thread-local connection handling.
- `matching.py`: face/pose helper utilities and embedding registry.
- `src/`: Svelte frontend.
- `src-tauri/`: Tauri desktop wrapper.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Rust toolchain (for Tauri desktop app)

## Python setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set at least:

```env
RTSP_URL=rtsp://user:pass@host:554/path
HEADLESS=true
LOG_LEVEL=INFO
```

Recommended production values:

```env
SCALE=0.35
IMGSZ=224
INFER_EVERY=8
STREAM_FPS=30
STREAM_JPEG_QUALITY=60
CAPTION_ENABLED=false
USE_CAPTION_AS_ACTION=false
```

Run runtime:

```bash
python mluser.py
```

## Desktop app (Tauri + Svelte)

Install frontend dependencies:

```bash
npm install
```

Run desktop dev mode:

```bash
npm run tauri dev
```

Build desktop bundle:

```bash
npm run tauri build
```

## Tests

Run smoke tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Deployment notes

- Keep secrets only in `.env`.
- Do not commit runtime artifacts (`data/`, `.env`, `dist/`, `src-tauri/target/`).
- Keep model files outside source control unless explicitly required.
- Prefer CPU-only deployment unless GPU acceleration is configured intentionally.
