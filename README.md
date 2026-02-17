# yacinetest

how to run : 
Create .env from .env.example and set your RTSP credentials.
  2. Install deps: pip install -r requirements.txt
  3. Run headless for deployment:
      - HEADLESS=true
      - python mluser.py
## Install

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root.

You can copy from `.env.example`.

Set either full RTSP URL:

```bash
RTSP_URL=rtsp://user:pass@192.168.1.29:554/cam/realmonitor?channel=1&subtype=0
```

Or set credential parts:

```bash
RTSP_USERNAME=your_user
RTSP_PASSWORD=your_password
RTSP_HOST=192.168.1.29
RTSP_PORT=554
RTSP_PATH=/cam/realmonitor?channel=1&subtype=0
```

Optional performance tuning in `.env`:

```bash
SCALE=0.4
IMGSZ=320
INFER_EVERY=4
FACE_EVERY=6
ACTION_EVERY=8
FACE_DET_SIZE=320
```

Production defaults:

```bash
HEADLESS=true
LOG_LEVEL=INFO
```

## Run

```bash
python mluser.py
```

## Module Layout

- `mluser.py`: thin entrypoint.
- `runtime.py`: production runtime loop, worker orchestration, graceful shutdown, logging.
- `config.py`: validated environment-driven app configuration.
- `db.py`: SQLite schema management and thread-local DB access.
- `matching.py`: face registry, embedding matching, and pose/action matching logic.

## Production Notes

- Use `.env` for all secrets and runtime settings.
- Set `HEADLESS=true` for server deployment.
- Use `LOG_LEVEL=INFO` (or `DEBUG` for diagnosis).
- Avoid committing local runtime artifacts (`data/`, `.env`, `*.db*`).

## Smoke Tests

```bash
python -m unittest tests/test_smoke.py
```
