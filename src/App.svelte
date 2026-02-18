<script>
  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";

  const WS_PORT = 8766;
  const MJPEG_PORT = 8765;

  let status = "Disconnected";
  let isRunning = false;
  let logs = [];
  let actions = [];
  let actionsTimer = null;
  let togglesPending = false;
  let togglesTimer = null;
  let connectRetryTimer = null;

  let objectIdentification = true;
  let vlmEnabled = false;
  let actionsHint = "";

  let ws = null;
  let streamCanvas = null;
  let streamFrameEl = null;
  let streamResizeObserver = null;
  let syncedTopHeight = 0;
  let streamCtx = null;
  let streamStatus = "connecting";
  let connecting = true;
  let mjpegVisible = false;
  let renderBusy = false;
  let queuedFrame = null;
  let fps = 0;
  let fpsFrames = 0;
  let fpsStart = 0;
  let frameWidth = 0;
  let frameHeight = 0;

  const envBool = (value, fallback) => {
    if (value === undefined || value === null || value === "") return fallback;
    const normalized = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off"].includes(normalized)) return false;
    return fallback;
  };

  const addLog = (line) => {
    logs = [line, ...logs].slice(0, 8);
  };

  const parseSource = (source) => {
    const raw = String(source || "");
    if (raw.includes(":")) return raw.slice(raw.indexOf(":") + 1);
    return raw;
  };

  const refreshActions = async () => {
    try {
      const rows = await invoke("read_actions", { limit: 140 });
      const mapped = rows.map((row) => ({ ...row, source_label: parseSource(row.source) }));
      const deduped = [];
      for (const row of mapped) {
        const prev = deduped[deduped.length - 1];
        if (
          prev &&
          prev.person_id === row.person_id &&
          prev.action_label === row.action_label &&
          prev.source_label === row.source_label
        ) {
          continue;
        }
        deduped.push(row);
      }
      actions = deduped;
    } catch (err) {
      addLog(String(err));
    }
  };

  const loadActionsHint = () => {
    actionsHint = vlmEnabled
      ? "VLM enabled: image descriptions are logged here."
      : "Enable VLM to see image descriptions in the terminal.";
  };

  const loadSettings = async () => {
    try {
      const env = await invoke("read_env");
      objectIdentification = envBool(
        env.DETECTION_ENABLED,
        envBool(env.DRAW_BOXES, true) && envBool(env.DRAW_LABELS, true) && envBool(env.SHOW_CONF, true)
      );
      vlmEnabled = envBool(env.CAPTION_ENABLED, false);
      loadActionsHint();
    } catch (err) {
      addLog(String(err));
    }
  };

  const sendOverlaySettings = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        type: "settings",
        draw_boxes: objectIdentification,
        draw_labels: objectIdentification,
        show_conf: objectIdentification
      })
    );
  };

  const applyFeatureToggles = async () => {
    togglesPending = true;
    try {
      await invoke("apply_settings", {
        updates: {
          DETECTION_ENABLED: objectIdentification ? "true" : "false",
          DRAW_BOXES: objectIdentification ? "true" : "false",
          DRAW_LABELS: objectIdentification ? "true" : "false",
          SHOW_CONF: objectIdentification ? "true" : "false",
          FACE_ENABLED: "false",
          POSE_ENABLED: "false",
          CAPTION_ENABLED: vlmEnabled ? "true" : "false",
          USE_CAPTION_AS_ACTION: vlmEnabled ? "true" : "false",
          CAPTION_ALLOW_SCENE_ACTION: "true",
          CAPTION_EVERY_SEC: "12.0",
          CAPTION_IMG_SIZE: "224",
          SCALE: "0.35",
          IMGSZ: "224",
          INFER_EVERY: "8",
          STREAM_FPS: "30",
          STREAM_JPEG_QUALITY: "60"
        }
      });
      await loadSettings();
      await refreshActions();
      sendOverlaySettings();
    } catch (err) {
      addLog(String(err));
    } finally {
      togglesPending = false;
    }
  };

  const queueApplyFeatureToggles = () => {
    if (togglesTimer) {
      clearTimeout(togglesTimer);
      togglesTimer = null;
    }
    togglesTimer = setTimeout(() => {
      togglesTimer = null;
      applyFeatureToggles();
    }, 280);
  };

  const connect = async () => {
    status = "Starting runtime...";
    try {
      await invoke("start_runtime");
      isRunning = true;
      status = "Connected";
      await refreshActions();
      await loadSettings();
      if (!actionsTimer) actionsTimer = setInterval(refreshActions, 1200);
      startStream();
    } catch (err) {
      const msg = String(err);
      if (msg.toLowerCase().includes("already running")) {
        isRunning = true;
        status = "Connected";
        startStream();
        return;
      }
      status = "Retrying...";
      addLog(msg);
      if (!connectRetryTimer) {
        connectRetryTimer = setTimeout(() => {
          connectRetryTimer = null;
          connect();
        }, 1500);
      }
    }
  };

  const renderFrameData = async (frameData) => {
    if (!streamCanvas) return;

    const blob = new Blob([frameData], { type: "image/jpeg" });
    const bitmap = await createImageBitmap(blob);
    frameWidth = bitmap.width;
    frameHeight = bitmap.height;

    if (!streamCtx) {
      streamCtx = streamCanvas.getContext("2d", { alpha: false, desynchronized: true });
    }
    if (!streamCtx) {
      bitmap.close();
      return;
    }

    mjpegVisible = false;
    connecting = false;

    const dpr = Math.min(1.0, window.devicePixelRatio || 1);
    const targetWidth = streamCanvas.clientWidth || bitmap.width;
    const targetHeight = streamCanvas.clientHeight || bitmap.height;
    const canvasWidth = Math.max(1, Math.floor(targetWidth * dpr));
    const canvasHeight = Math.max(1, Math.floor(targetHeight * dpr));
    if (streamCanvas.width !== canvasWidth || streamCanvas.height !== canvasHeight) {
      streamCanvas.width = canvasWidth;
      streamCanvas.height = canvasHeight;
    }

    streamCtx.setTransform(1, 0, 0, 1, 0, 0);
    streamCtx.clearRect(0, 0, streamCanvas.width, streamCanvas.height);
    const scale = Math.min(streamCanvas.width / bitmap.width, streamCanvas.height / bitmap.height);
    const drawW = Math.floor(bitmap.width * scale);
    const drawH = Math.floor(bitmap.height * scale);
    const dx = Math.floor((streamCanvas.width - drawW) / 2);
    const dy = Math.floor((streamCanvas.height - drawH) / 2);
    streamCtx.drawImage(bitmap, dx, dy, drawW, drawH);
    bitmap.close();

    const now = performance.now();
    if (fpsStart === 0) {
      fpsStart = now;
      fpsFrames = 0;
    }
    fpsFrames += 1;
    const elapsed = now - fpsStart;
    if (elapsed >= 1000) {
      fps = (fpsFrames * 1000) / elapsed;
      fpsFrames = 0;
      fpsStart = now;
    }
  };

  const startStream = () => {
    if (ws) return;

    streamStatus = "connecting";
    connecting = true;
    ws = new WebSocket(`ws://127.0.0.1:${WS_PORT}/stream`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      streamStatus = "live";
      mjpegVisible = false;
      connecting = false;
      sendOverlaySettings();
    };

    ws.onclose = () => {
      ws = null;
      streamStatus = isRunning ? "reconnecting" : "disconnected";
      mjpegVisible = true;
      connecting = isRunning;
      if (isRunning) setTimeout(startStream, 600);
    };

    ws.onerror = () => {
      streamStatus = "error";
      mjpegVisible = true;
      connecting = false;
    };

    ws.onmessage = async (evt) => {
      queuedFrame = evt.data;
      if (renderBusy) return;
      renderBusy = true;
      try {
        while (queuedFrame) {
          const next = queuedFrame;
          queuedFrame = null;
          await renderFrameData(next);
        }
      } finally {
        renderBusy = false;
      }
    };
  };

  const stopStream = () => {
    if (ws) ws.close();
    ws = null;
    streamStatus = "disconnected";
    connecting = false;
    mjpegVisible = false;
    fps = 0;
    fpsFrames = 0;
    fpsStart = 0;
    queuedFrame = null;
    renderBusy = false;
  };

  const toggleFullscreen = async () => {
    try {
      const el = document.getElementById("camera-frame");
      if (!el) return;
      if (!document.fullscreenElement) {
        await el.requestFullscreen();
        return;
      }
      if (document.fullscreenElement === el) {
        await document.exitFullscreen();
        return;
      }
      await document.exitFullscreen();
      await el.requestFullscreen();
    } catch (err) {
      addLog(String(err));
    }
  };

  onMount(() => {
    refreshActions();
    loadSettings();
    actionsTimer = setInterval(refreshActions, 1200);
    connect();

    if (streamFrameEl && typeof ResizeObserver !== "undefined") {
      streamResizeObserver = new ResizeObserver((entries) => {
        const next = entries?.[0]?.contentRect?.height || 0;
        syncedTopHeight = next > 0 ? Math.round(next) : 0;
      });
      streamResizeObserver.observe(streamFrameEl);
    }
  });

  onDestroy(() => {
    if (actionsTimer) {
      clearInterval(actionsTimer);
      actionsTimer = null;
    }
    if (togglesTimer) {
      clearTimeout(togglesTimer);
      togglesTimer = null;
    }
    if (connectRetryTimer) {
      clearTimeout(connectRetryTimer);
      connectRetryTimer = null;
    }
    if (streamResizeObserver) {
      streamResizeObserver.disconnect();
      streamResizeObserver = null;
    }
    stopStream();
  });
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-dot"></div>
      <div>
        <div class="title">AI Camera</div>
        <div class="subtitle">Vision Console</div>
      </div>
    </div>
    <div class="top-metrics">
      <div class="pill {isRunning ? 'ok' : 'idle'}">{status}</div>
      <div class="metric">
        <span class="metric-label">FPS</span>
        <span class="metric-value">{fps > 0 ? fps.toFixed(1) : "--"}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Events</span>
        <span class="metric-value">{actions.length}</span>
      </div>
    </div>
  </header>

  <main class="workspace">
    <div class="panel stream-panel">
      <div
        class="stream-frame"
        id="camera-frame"
        bind:this={streamFrameEl}
        style:aspect-ratio={frameWidth > 0 && frameHeight > 0 ? `${frameWidth} / ${frameHeight}` : "16 / 9"}
      >
        <canvas class="stream-canvas" bind:this={streamCanvas}></canvas>
        {#if mjpegVisible}
          <img class="stream-img" src={`http://127.0.0.1:${MJPEG_PORT}/stream`} alt="Live Stream" />
        {/if}
        {#if connecting}
          <div class="stream-loading">
            <div class="spinner"></div>
            <div>Connecting…</div>
          </div>
        {/if}
        <div class="stream-tag">RTSP Live</div>
        <div class="stream-resolution">{frameWidth > 0 ? `${frameWidth}x${frameHeight}` : ""}</div>
      </div>
    </div>

    <div class="panel terminal-wrap" style:height={syncedTopHeight > 0 ? `${syncedTopHeight}px` : null}>
      <div class="terminal-shell">
        <div class="terminal-bar">
          <div class="dot red"></div>
          <div class="dot yellow"></div>
          <div class="dot green"></div>
          <div class="terminal-title">Quick Actions</div>
          <div class="terminal-count">{actions.length} items</div>
        </div>
        <div class="terminal">
          {#if actions.length === 0}
            <div class="terminal-line muted">{actionsHint || "No actions yet"}</div>
          {:else}
            {#each actions as action}
              <div class="terminal-line">
                [{action.created_at}] P{action.person_id} {action.action_label}
                {#if action.action_confidence !== null && action.action_confidence !== undefined}
                  ({(action.action_confidence * 100).toFixed(0)}%)
                {/if}
                <span class="source">#{action.source_label}</span>
              </div>
            {/each}
          {/if}
        </div>
      </div>
    </div>

    <div class="panel camera-info">
      <div class="camera-header">
        <div class="camera-title">Camera 1</div>
        <div class="camera-status {streamStatus}">{streamStatus}</div>
      </div>
      <div class="info-grid">
        <div class="info-label">FPS</div>
        <div class="info-value">{fps > 0 ? fps.toFixed(1) : "--"}</div>
        <div class="info-label">Resolution</div>
        <div class="info-value">{frameWidth > 0 ? `${frameWidth}x${frameHeight}` : "--"}</div>
        <div class="info-label">Mode</div>
        <div class="info-value">{objectIdentification ? "Detection On" : "Detection Off"}</div>
        <div class="info-label">VLM</div>
        <div class="info-value">{vlmEnabled ? "Enabled" : "Disabled"}</div>
      </div>
      <button class="fs-btn" on:click={toggleFullscreen}>Fullscreen</button>
    </div>

    <div class="panel controls">
      <div class="feature-toggles">
        <label class="toggle-row">
          <span class="toggle-copy">
            <span class="toggle-title">Object Identification</span>
            <span class="toggle-hint">Show detections and confidence overlays.</span>
          </span>
          <span class="switch">
            <input type="checkbox" bind:checked={objectIdentification} on:change={queueApplyFeatureToggles} />
            <span class="slider"></span>
          </span>
        </label>
        <label class="toggle-row">
          <span class="toggle-copy">
            <span class="toggle-title">VLM Captions</span>
            <span class="toggle-hint">Log scene-aware action descriptions.</span>
          </span>
          <span class="switch">
            <input type="checkbox" bind:checked={vlmEnabled} on:change={queueApplyFeatureToggles} />
            <span class="slider"></span>
          </span>
        </label>
        {#if togglesPending}
          <span class="apply-state">Applying...</span>
        {/if}
      </div>
    </div>
  </main>
</div>

<style>
  .shell {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
    gap: 10px;
    padding: 10px 12px 12px;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    border: 1px solid rgba(42, 111, 107, 0.2);
    border-radius: 12px;
    padding: 10px 12px;
    background: rgba(246, 239, 229, 0.72);
    backdrop-filter: blur(6px);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent), var(--accent-2));
    box-shadow: 0 0 0 6px rgba(42, 111, 107, 0.13);
  }

  .title {
    font-weight: 700;
    letter-spacing: 0.02em;
  }

  .subtitle {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .top-metrics {
    display: flex;
    align-items: stretch;
    gap: 8px;
  }

  .pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 12px;
    border-radius: 10px;
    border: 1px solid var(--border);
    font-size: 12px;
    font-weight: 600;
    min-height: 38px;
    white-space: nowrap;
  }

  .pill.ok {
    background: rgba(42, 111, 107, 0.14);
    color: #1f5f5b;
    border-color: rgba(42, 111, 107, 0.3);
  }

  .pill.idle {
    background: rgba(107, 104, 100, 0.08);
    color: var(--muted);
  }

  .metric {
    min-width: 86px;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 7px 10px;
    background: rgba(248, 242, 234, 0.8);
    display: grid;
    gap: 2px;
  }

  .metric-label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  .metric-value {
    font-size: 16px;
    line-height: 1;
    color: #2b2a27;
    font-weight: 700;
  }

  .workspace {
    min-height: 0;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    grid-template-areas:
      "stream terminal"
      "camera controls";
    grid-template-rows: auto auto;
    gap: 10px;
  }

  .panel {
    border: 1px solid rgba(99, 92, 81, 0.18);
    border-radius: 12px;
    background: rgba(246, 239, 229, 0.55);
    overflow: hidden;
  }

  .stream-panel {
    grid-area: stream;
    padding: 0;
    align-self: start;
  }

  .stream-frame {
    position: relative;
    width: 100%;
    height: auto;
    background: #090d0b;
  }

  .stream-tag,
  .stream-resolution {
    position: absolute;
    font-size: 11px;
    color: #cbe8d2;
    background: rgba(10, 20, 14, 0.62);
    border: 1px solid rgba(121, 177, 140, 0.22);
    border-radius: 999px;
    padding: 3px 8px;
  }

  .stream-tag {
    left: 10px;
    top: 10px;
  }

  .stream-resolution {
    right: 10px;
    bottom: 10px;
  }

  .stream-canvas {
    width: 100%;
    height: 100%;
    display: block;
    background: #090d0b;
  }

  .stream-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    position: absolute;
    inset: 0;
  }

  .stream-loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    gap: 10px;
    color: #cde7d0;
    font-size: 12px;
    background: radial-gradient(circle at 50% 45%, rgba(10, 20, 14, 0.45), rgba(5, 8, 6, 0.75));
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 3px solid rgba(143, 200, 160, 0.2);
    border-top-color: #9fe870;
    animation: spin 1s linear infinite;
  }

  .camera-info {
    grid-area: camera;
    padding: 12px;
    display: grid;
    gap: 10px;
  }

  .camera-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .camera-title {
    font-weight: 700;
    font-size: 15px;
  }

  .camera-status {
    border-radius: 999px;
    border: 1px solid var(--border);
    padding: 3px 10px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #5f5b56;
    background: rgba(217, 208, 194, 0.45);
  }

  .camera-status.live {
    color: #1d6844;
    border-color: rgba(39, 127, 83, 0.35);
    background: rgba(39, 127, 83, 0.12);
  }

  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    row-gap: 8px;
    column-gap: 10px;
  }

  .info-label {
    font-size: 12px;
    color: var(--muted);
  }

  .info-value {
    font-size: 12px;
    color: #2d2a26;
    font-weight: 600;
    text-align: right;
  }

  .fs-btn {
    border: 1px solid var(--border);
    background: rgba(251, 246, 238, 0.8);
    border-radius: 8px;
    font-size: 12px;
    padding: 7px 12px;
    cursor: pointer;
    justify-self: start;
  }

  .terminal-wrap {
    grid-area: terminal;
    height: 100%;
    min-height: 0;
    display: flex;
    overflow: hidden;
  }

  .terminal-shell {
    flex: 1;
    height: 100%;
    min-height: 0;
    display: grid;
    grid-template-rows: auto 1fr;
    border-radius: 12px;
    overflow: hidden;
    background: #0a0f0d;
    border: 1px solid rgba(120, 150, 130, 0.24);
    box-shadow: inset 0 0 24px rgba(20, 40, 25, 0.45);
  }

  .terminal-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    background: linear-gradient(180deg, #171d1a, #111613);
    border-bottom: 1px solid rgba(120, 150, 130, 0.18);
  }

  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .dot.red {
    background: #ff5f56;
  }

  .dot.yellow {
    background: #ffbd2e;
  }

  .dot.green {
    background: #27c93f;
  }

  .terminal-title {
    margin-left: 6px;
    color: #c7d4c6;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .terminal-count {
    margin-left: auto;
    color: #7ca086;
    font-size: 11px;
  }

  .terminal {
    color: #9fe870;
    font-family: var(--mono);
    padding: 10px 12px;
    overflow-y: auto;
    overflow-x: hidden;
    min-height: 0;
  }

  .terminal-line {
    font-size: 12px;
    line-height: 1.42;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .terminal-line.muted {
    color: #7aa68b;
  }

  .terminal-line .source {
    color: #6edbb5;
    margin-left: 8px;
  }

  .controls {
    grid-area: controls;
    padding: 12px;
  }

  .feature-toggles {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
    align-items: stretch;
  }

  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    background: rgba(251, 246, 238, 0.74);
  }

  .toggle-copy {
    display: grid;
    gap: 2px;
  }

  .toggle-title {
    color: #2d2a26;
    font-size: 13px;
    font-weight: 600;
  }

  .toggle-hint {
    color: var(--muted);
    font-size: 11px;
  }

  .switch {
    position: relative;
    display: inline-block;
    width: 42px;
    height: 24px;
  }

  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    inset: 0;
    cursor: pointer;
    background-color: #b9b2a8;
    transition: 0.2s;
    border-radius: 999px;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    top: 3px;
    background-color: white;
    transition: 0.2s;
    border-radius: 50%;
  }

  .switch input:checked + .slider {
    background-color: var(--accent);
  }

  .switch input:checked + .slider:before {
    transform: translateX(18px);
  }

  .apply-state {
    font-size: 11px;
    color: var(--accent);
    border: 1px solid rgba(42, 111, 107, 0.25);
    background: rgba(42, 111, 107, 0.1);
    border-radius: 999px;
    padding: 4px 10px;
    width: fit-content;
  }

  #camera-frame:fullscreen {
    background: #000;
    width: 100vw;
    height: 100vh;
  }

  #camera-frame:fullscreen .stream-canvas {
    width: 100%;
    height: 100%;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 980px) {
    .shell {
      padding: 8px;
      gap: 8px;
    }

    .topbar {
      flex-wrap: wrap;
    }

    .workspace {
      grid-template-columns: 1fr;
      grid-template-areas:
        "stream"
        "camera"
        "terminal"
        "controls";
    }

    .stream-frame {
      min-height: 0;
    }

    .top-metrics {
      width: 100%;
      justify-content: flex-start;
    }
  }
</style>
