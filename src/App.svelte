<script>
  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";

  const WS_PORT = 8766;
  const MJPEG_PORT = 8765;
  const tabs = [
    { id: "live", label: "Live Stream" },
    { id: "profiles", label: "Face Identification" }
  ];

  let activeTab = "live";
  let status = "Disconnected";
  let isRunning = false;
  let actions = [];
  let people = [];
  let selectedPersonId = null;
  let personProfile = null;

  let actionsTimer = null;
  let peopleTimer = null;
  let togglesPending = false;
  let togglesTimer = null;
  let connectRetryTimer = null;

  let objectIdentification = true;
  let actionsHint = "";
  let personFaceImages = {};

  let ws = null;
  let streamCanvas = null;
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

  const parseSource = (source) => {
    const raw = String(source || "");
    if (raw.includes(":")) return raw.slice(raw.indexOf(":") + 1);
    return raw;
  };

  const refreshActions = async () => {
    try {
      const rows = await invoke("read_actions", { limit: 160 });
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
      console.error(err);
    }
  };

  const refreshPeople = async () => {
    try {
      const rows = await invoke("read_people", { limit: 220 });
      people = rows;
      const keepIds = new Set(people.map((p) => p.id));
      personFaceImages = Object.fromEntries(
        Object.entries(personFaceImages).filter(([id]) => keepIds.has(Number(id)))
      );
      if (!people.length) {
        selectedPersonId = null;
        personProfile = null;
        return;
      }
      await Promise.all(people.map((person) => loadFaceImage(person.id)));
      if (selectedPersonId === null || !people.some((p) => p.id === selectedPersonId)) {
        selectedPersonId = people[0].id;
      }
      await loadPersonProfile(selectedPersonId);
    } catch (err) {
      console.error(err);
    }
  };

  const loadPersonProfile = async (personId) => {
    if (!personId) {
      personProfile = null;
      return;
    }
    try {
      const profile = await invoke("read_person_profile", {
        person_id: personId,
        sightings_limit: 120,
        actions_limit: 120
      });
      personProfile = profile;
      if (personProfile?.person?.id) {
        await loadFaceImage(personProfile.person.id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadFaceImage = async (personId) => {
    if (!personId || personFaceImages[personId] !== undefined) return;
    try {
      const encoded = await invoke("read_face_image", { person_id: personId });
      const image = encoded ? `data:image/jpeg;base64,${encoded}` : "";
      personFaceImages = { ...personFaceImages, [personId]: image };
    } catch (err) {
      console.error(err);
      personFaceImages = { ...personFaceImages, [personId]: "" };
    }
  };

  const selectPerson = async (personId) => {
    selectedPersonId = personId;
    await loadPersonProfile(personId);
  };

  const loadActionsHint = () => {
    actionsHint = "Face ID and VLM captions are always active. Use Object Identification to control overlays.";
  };

  const loadSettings = async () => {
    try {
      const env = await invoke("read_env");
      objectIdentification = envBool(
        env.DETECTION_ENABLED,
        envBool(env.DRAW_BOXES, true) && envBool(env.DRAW_LABELS, true) && envBool(env.SHOW_CONF, true)
      );
      loadActionsHint();
    } catch (err) {
      console.error(err);
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
          FACE_ENABLED: "true",
          POSE_ENABLED: "false",
          CAPTION_ENABLED: "true",
          USE_CAPTION_AS_ACTION: "true",
          CAPTION_ALLOW_SCENE_ACTION: "true",
          CAPTION_EVERY_SEC: "12.0",
          CAPTION_IMG_SIZE: "224",
          SCALE: "0.35",
          IMGSZ: "224",
          INFER_EVERY: "8",
          FACE_EVERY: "4",
          STREAM_FPS: "30",
          STREAM_JPEG_QUALITY: "60"
        }
      });
      await loadSettings();
      await refreshActions();
      await refreshPeople();
      sendOverlaySettings();
    } catch (err) {
      console.error(err);
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
      await refreshPeople();
      await loadSettings();
      if (!actionsTimer) actionsTimer = setInterval(refreshActions, 1200);
      if (!peopleTimer) peopleTimer = setInterval(refreshPeople, 1800);
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
      console.error(msg);
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

    streamCtx.drawImage(bitmap, 0, 0, streamCanvas.width, streamCanvas.height);
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
      console.error(err);
    }
  };

  onMount(() => {
    refreshActions();
    refreshPeople();
    loadSettings();
    actionsTimer = setInterval(refreshActions, 1200);
    peopleTimer = setInterval(refreshPeople, 1800);
    connect();
  });

  onDestroy(() => {
    if (actionsTimer) {
      clearInterval(actionsTimer);
      actionsTimer = null;
    }
    if (peopleTimer) {
      clearInterval(peopleTimer);
      peopleTimer = null;
    }
    if (togglesTimer) {
      clearTimeout(togglesTimer);
      togglesTimer = null;
    }
    if (connectRetryTimer) {
      clearTimeout(connectRetryTimer);
      connectRetryTimer = null;
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
    <div class="tab-row">
      {#each tabs as tab}
        <button
          class={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
          on:click={() => (activeTab = tab.id)}
        >
          {tab.label}
        </button>
      {/each}
      <div class="tab-indicator" style={`transform: translateX(${activeTab === "profiles" ? "100%" : "0"});`}></div>
    </div>
    <div class="top-metrics">
      <div class={`pill ${isRunning ? "ok" : "idle"}`}>{status}</div>
      <div class="metric">
        <span class="metric-label">FPS</span>
        <span class="metric-value">{fps > 0 ? fps.toFixed(1) : "--"}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Profiles</span>
        <span class="metric-value">{people.length}</span>
      </div>
    </div>
  </header>

  <main class="workspace">
    <section class={`page ${activeTab === "live" ? "active" : ""}`} aria-hidden={activeTab !== "live"}>
      <div class="live-grid">
        <div class="stream-column">
          <div class="stream-frame" id="camera-frame">
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
            <button class="fullscreen-btn" on:click={toggleFullscreen}>Fullscreen</button>
            <div class="stream-badges">
              <span class="badge">{streamStatus}</span>
              <span class="badge">{frameWidth > 0 ? `${frameWidth}x${frameHeight}` : "--"}</span>
            </div>
          </div>
        </div>
        <div class="terminal-column">
          <div class="quick-shell">
            <div class="quick-head">
              <div class="quick-title">Quick Actions</div>
              <div class="quick-count">{actions.length} entries</div>
            </div>
            <div class="mini-log terminal">
              {#if actions.length === 0}
                <div class="terminal-line muted">No events yet</div>
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
          <div class="toggle-shell">
            <div class="toggle-row">
              <div>
                <div class="toggle-title">Object Identification</div>
                <div class="toggle-hint">Show detection boxes and labels.</div>
              </div>
              <label class="switch">
                <input type="checkbox" bind:checked={objectIdentification} on:change={queueApplyFeatureToggles} />
                <span class="slider"></span>
              </label>
            </div>
            {#if togglesPending}
              <div class="state-chip">Applying settings…</div>
            {/if}
            <div class="hint-line">{actionsHint}</div>
          </div>
        </div>
      </div>
    </section>

    <section class={`page profiles ${activeTab === "profiles" ? "active" : ""}`} aria-hidden={activeTab !== "profiles"}>
      <div class="profiles-shell">
        <div class="people-list">
          <div class="list-head">Detected People</div>
          <div class="list-scroll">
            {#if people.length === 0}
              <div class="empty-note">No face profiles yet.</div>
            {:else}
              {#each people as person}
                <button
                  class={`person-item ${selectedPersonId === person.id ? "active" : ""}`}
                  on:click={() => selectPerson(person.id)}
                >
                  <div class="person-head">
                    <div class="person-avatar">
                      {#if personFaceImages[person.id]}
                        <img src={personFaceImages[person.id]} alt={`P${person.id}`} />
                      {:else}
                        <span>?</span>
                      {/if}
                    </div>
                    <span class="pid">P{person.id}</span>
                  </div>
                  <span class="seen">{person.seen_count} sightings</span>
                  <span class="last">{person.last_seen_at}</span>
                </button>
              {/each}
            {/if}
          </div>
        </div>
        <div class="profile-pane">
          <div class="list-head">Profile Details</div>
          {#if !personProfile}
            <div class="empty-note">Select a person to view history.</div>
          {:else}
            <div class="profile-top">
              <div class="profile-avatar profile-avatar-lg">
                {#if personFaceImages[personProfile.person.id]}
                  <img src={personFaceImages[personProfile.person.id]} alt={`P${personProfile.person.id}`} />
                {:else}
                  <span>?</span>
                {/if}
              </div>
              <div class="profile-id">P{personProfile.person.id}</div>
            </div>
            <div class="profile-grid">
              <div class="profile-k">First Seen</div><div class="profile-v">{personProfile.person.created_at}</div>
              <div class="profile-k">Last Seen</div><div class="profile-v">{personProfile.person.last_seen_at}</div>
              <div class="profile-k">Sightings</div><div class="profile-v">{personProfile.person.seen_count}</div>
              <div class="profile-k">Best Face Score</div>
              <div class="profile-v">{personProfile.person.best_face_score.toFixed(1)}</div>
            </div>
            <div class="section-label">Recent Sightings</div>
            <div class="mini-log">
              {#if personProfile.sightings.length === 0}
                <div class="terminal-line muted">No sightings recorded</div>
              {:else}
                {#each personProfile.sightings.slice(0, 16) as sighting}
                  <div class="terminal-line">[{sighting.seen_at}] {sighting.camera_id} score {sighting.face_score?.toFixed(1) ?? "--"}</div>
                {/each}
              {/if}
            </div>
            <div class="section-label">Profile Actions</div>
            <div class="mini-log">
              {#if personProfile.actions.length === 0}
                <div class="terminal-line muted">No actions yet</div>
              {:else}
                {#each personProfile.actions.slice(0, 16) as action}
                  <div class="terminal-line">
                    [{action.created_at}] {action.action_label}
                    {#if action.action_confidence !== null && action.action_confidence !== undefined}
                      ({(action.action_confidence * 100).toFixed(0)}%)
                    {/if}
                    <span class="source">#{parseSource(action.source)}</span>
                  </div>
                {/each}
              {/if}
            </div>
          {/if}
        </div>
      </div>
    </section>
  </main>
</div>
<style>
  :global(:root) {
    --bg: #f4efe8;
    --panel: rgba(255, 255, 255, 0.84);
    --border: rgba(96, 107, 99, 0.25);
    --accent: #2a6f6b;
    --accent-2: #7eb594;
    --muted: #6b6864;
    --mono: "IBM Plex Mono", "Consolas", monospace;
    --sans: "Sora", "Segoe UI", sans-serif;
  }

  :global(body) {
    margin: 0;
    font-family: var(--sans);
    background: linear-gradient(165deg, #f8f4ed 0%, #ede5db 46%, #e4dacc 100%);
    height: 100vh;
    overflow: hidden;
  }

  :global(#app) {
    height: 100vh;
    overflow: hidden;
  }

  .shell {
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    height: 100vh;
    overflow: hidden;
  }

  .topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    border-radius: 10px;
    padding: 8px 16px;
    background: rgba(246, 239, 229, 0.9);
    border: 1px solid rgba(42, 111, 107, 0.2);
    flex-shrink: 0;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent), var(--accent-2));
  }

  .title {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
    line-height: 1.2;
  }

  .subtitle {
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    line-height: 1.2;
  }

  .top-metrics {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
  }

  .pill {
    padding: 0 10px;
    border-radius: 10px;
    border: 1px solid var(--border);
    min-height: 24px;
    display: grid;
    place-items: center;
    font-size: 10px;
    font-weight: 600;
  }

  .pill.ok {
    background: rgba(42, 111, 107, 0.14);
    color: #1f5f5b;
  }

  .pill.idle {
    background: rgba(107, 104, 100, 0.08);
    color: var(--muted);
  }

  .metric {
    border-radius: 8px;
    border: 1px solid var(--border);
    padding: 2px 8px;
    background: rgba(248, 242, 234, 0.7);
    display: grid;
    gap: 0;
    min-width: 56px;
    text-align: center;
  }

  .metric-label {
    font-size: 8px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .metric-value {
    font-size: 13px;
    font-weight: 700;
    color: #2b2a27;
  }

  .tab-row {
    position: relative;
    flex: 1;
    max-width: 360px;
    margin: 0 auto;
    background: rgba(10, 20, 14, 0.06);
    border-radius: 999px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    overflow: hidden;
  }

  .tab-btn {
    border: none;
    background: transparent;
    padding: 6px 0;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    color: var(--muted);
    transition: color 0.3s ease;
  }

  .tab-btn.active {
    color: #0c261c;
  }

  .tab-indicator {
    position: absolute;
    inset: 3px;
    width: 50%;
    border-radius: 999px;
    background: rgba(126, 181, 148, 0.35);
    transition: transform 0.35s ease;
  }
  .workspace {
    flex: 1;
    min-height: 0;
    position: relative;
  }

  .page {
    position: absolute;
    inset: 0;
    opacity: 0;
    pointer-events: none;
    transform: translateY(12px);
    transition: opacity 0.35s ease, transform 0.35s ease;
  }

  .page.active {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
  }

  .live-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(0, 0.9fr);
    gap: 10px;
    height: 100%;
    align-items: stretch;
  }

  .stream-column,
  .terminal-column {
    background: var(--panel);
    border-radius: 12px;
    padding: 8px;
    border: 1px solid var(--border);
    box-shadow: 0 12px 28px rgba(20, 31, 21, 0.1);
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 0;
    overflow: hidden;
  }

  .stream-frame {
    position: relative;
    background: transparent;
    border-radius: 10px;
    overflow: hidden;
    width: 100%;
    aspect-ratio: 16 / 9;
  }

  .stream-canvas,
  .stream-img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
  }

  .stream-loading {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    gap: 10px;
    background: rgba(0, 0, 0, 0.55);
    color: #cde7d0;
    font-size: 12px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  .spinner {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 3px solid rgba(143, 200, 160, 0.3);
    border-top-color: #9fe870;
    animation: spin 1s linear infinite;
  }

  .fullscreen-btn {
    position: absolute;
    bottom: 12px;
    right: 12px;
    padding: 6px 12px;
    font-size: 12px;
    border: none;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.85);
    cursor: pointer;
    font-weight: 600;
  }

  .stream-badges {
    position: absolute;
    top: 12px;
    left: 12px;
    display: flex;
    gap: 6px;
  }

  .badge {
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(19, 28, 24, 0.8);
    color: #cbe8d2;
    font-size: 11px;
  }

  .terminal-column {
    justify-content: space-between;
    min-height: 100%;
  }

  .quick-shell {
    border-radius: 10px;
    background: #0b120f;
    border: 1px solid rgba(120, 150, 130, 0.25);
    padding: 8px;
    color: #d4e5d8;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .quick-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .quick-title {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #c7d4c6;
  }

  .quick-count {
    font-size: 10px;
    color: #8ba993;
  }

  .terminal {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.4;
    color: #9fe870;
  }

  .terminal-line {
    margin-bottom: 6px;
    white-space: normal;
    word-break: break-word;
  }

  .terminal-line .source {
    color: #6edbb5;
    margin-left: 8px;
  }

  .terminal-line.muted {
    color: #7aa68b;
  }
  .toggle-shell {
    border-radius: 10px;
    border: 1px solid rgba(120, 150, 130, 0.2);
    padding: 10px;
    background: rgba(251, 246, 238, 0.84);
    display: grid;
    gap: 8px;
    flex-shrink: 0;
  }

  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .toggle-title {
    font-weight: 600;
    font-size: 13px;
    color: #2d2a26;
  }

  .toggle-hint {
    font-size: 11px;
    color: var(--muted);
  }

  .switch {
    position: relative;
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
    border-radius: 999px;
    background-color: #b9b2a8;
    transition: 0.2s;
  }

  .slider:before {
    content: "";
    position: absolute;
    width: 18px;
    height: 18px;
    left: 3px;
    top: 3px;
    border-radius: 50%;
    background: #fff;
    transition: 0.2s;
  }

  .switch input:checked + .slider {
    background: var(--accent);
  }

  .switch input:checked + .slider:before {
    transform: translateX(18px);
  }

  .state-chip {
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(42, 111, 107, 0.2);
    color: #1f5f5b;
    font-size: 11px;
    width: fit-content;
  }

  .hint-line {
    font-size: 11px;
    color: #7ba187;
  }

  .profiles-shell {
    height: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(10, 20, 14, 0.75);
    display: grid;
    grid-template-columns: minmax(0, 0.45fr) minmax(0, 0.55fr);
    gap: 10px;
    padding: 10px;
    overflow: hidden;
    box-shadow: 0 14px 28px rgba(15, 20, 17, 0.45);
  }

  .people-list,
  .profile-pane {
    background: rgba(10, 20, 14, 0.6);
    border: 1px solid rgba(120, 150, 130, 0.3);
    border-radius: 12px;
    display: grid;
    grid-template-rows: auto 1fr;
    overflow: hidden;
  }
  .people-list {
    min-height: 0;
  }
  .profile-pane {
    min-height: 0;
  }

  .list-head {
    padding: 10px 12px;
    font-size: 11px;
    color: #c7d4c6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid rgba(120, 150, 130, 0.25);
    background: linear-gradient(180deg, #171d1a, #111613);
  }

  .list-scroll {
    overflow-y: auto;
    padding: 10px;
    display: grid;
    gap: 8px;
  }

  .person-item {
    border-radius: 10px;
    border: 1px solid rgba(120, 150, 130, 0.25);
    background: rgba(22, 35, 26, 0.65);
    color: #d4e5d8;
    padding: 8px;
    font-family: var(--mono);
    text-align: left;
    display: grid;
    gap: 4px;
  }

  .person-head {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .person-avatar,
  .profile-avatar {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid rgba(120, 150, 130, 0.4);
    background: rgba(16, 27, 19, 0.85);
    display: grid;
    place-items: center;
    color: #9abca2;
    font-size: 12px;
    font-family: var(--mono);
  }

  .person-avatar img,
  .profile-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .profile-top {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 12px 0;
  }

  .profile-avatar-lg {
    width: 64px;
    height: 64px;
    font-size: 16px;
  }

  .profile-id {
    color: #9fe870;
    font-size: 20px;
    font-weight: 700;
    font-family: var(--mono);
  }

  .person-item.active {
    border-color: rgba(110, 219, 181, 0.6);
    background: rgba(38, 72, 57, 0.8);
  }

  .pid {
    color: #9fe870;
    font-size: 13px;
    font-weight: 700;
  }

  .seen,
  .last {
    font-size: 11px;
    color: #9abca2;
  }

  .profile-grid {
    display: grid;
    grid-template-columns: 0.45fr 0.55fr;
    row-gap: 8px;
    column-gap: 14px;
    padding: 12px;
    color: #d0e0d3;
  }

  .profile-k {
    font-size: 11px;
    color: #8ba993;
  }

  .profile-v {
    font-size: 12px;
    font-family: var(--mono);
    text-align: right;
  }

  .section-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8ba993;
    padding: 8px 12px 4px;
  }

  .mini-log {
    padding: 0 12px 8px;
    overflow-y: auto;
  }

  .empty-note {
    color: #7aa68b;
    padding: 12px;
    font-size: 12px;
  }

  @media (max-width: 1024px) {
    .topbar {
      flex-wrap: wrap;
    }

    .tab-row {
      flex-basis: 100%;
      max-width: 100%;
    }

    .live-grid {
      grid-template-columns: 1fr;
      height: auto;
    }

    .terminal-column {
      order: -1;
    }

    .profiles-shell {
      grid-template-columns: 1fr;
      height: auto;
    }
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
