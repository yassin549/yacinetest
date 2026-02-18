#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
};

use rusqlite::{params, Connection};
use serde::Serialize;
use tauri::State;

#[derive(Default)]
struct RuntimeState {
    child: Mutex<Option<Child>>,
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("runtime already running")]
    AlreadyRunning,
    #[error("runtime not running")]
    NotRunning,
    #[error("failed to start runtime: {0}")]
    StartFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

fn find_project_root() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let mut dir = parent.to_path_buf();
            for _ in 0..6 {
                candidates.push(dir.clone());
                if let Some(next) = dir.parent() {
                    dir = next.to_path_buf();
                } else {
                    break;
                }
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd;
        for _ in 0..6 {
            candidates.push(dir.clone());
            if let Some(next) = dir.parent() {
                dir = next.to_path_buf();
            } else {
                break;
            }
        }
    }

    for dir in candidates {
        if dir.join("mluser.py").exists() {
            return dir;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn base_dir() -> PathBuf {
    find_project_root()
}

fn env_path() -> PathBuf {
    let mut path = base_dir();
    path.push(".env");
    path
}

fn parse_env(contents: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = if let Some(stripped) = line.strip_prefix("export ") {
            stripped.trim()
        } else {
            line
        };
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim().to_string();
            let mut val = v.trim().to_string();
            if val.len() >= 2 {
                let bytes = val.as_bytes();
                if (bytes[0] == b'"' && bytes[val.len() - 1] == b'"')
                    || (bytes[0] == b'\'' && bytes[val.len() - 1] == b'\'')
                {
                    val = val[1..val.len() - 1].to_string();
                }
            }
            map.insert(key, val);
        }
    }
    map
}

fn write_env_file(path: &PathBuf, updates: &HashMap<String, String>) -> std::io::Result<()> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut map = parse_env(&existing);
    for (k, v) in updates {
        map.insert(k.clone(), v.clone());
    }

    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();

    let mut out = String::new();
    for k in keys {
        let v = map.get(&k).cloned().unwrap_or_default();
        out.push_str(&k);
        out.push('=');
        out.push_str(&v);
        out.push('\n');
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, out)
}

fn start_runtime_inner(state: &RuntimeState) -> Result<(), AppError> {
    let mut child_guard = state.child.lock().unwrap();
    if child_guard.is_some() {
        return Err(AppError::AlreadyRunning);
    }

    let cwd = base_dir();

    let mut cmd = Command::new("python");
    cmd.arg("mluser.py")
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let child = cmd.spawn().or_else(|_| {
        let mut fallback = Command::new("py");
        fallback
            .arg("mluser.py")
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        fallback.spawn()
    });

    let child = child.map_err(|err| AppError::StartFailed(err.to_string()))?;
    *child_guard = Some(child);
    Ok(())
}

fn stop_runtime_inner(state: &RuntimeState) -> Result<(), AppError> {
    let mut child_guard = state.child.lock().unwrap();
    let Some(mut child) = child_guard.take() else {
        return Err(AppError::NotRunning);
    };

    let _ = child.kill();
    let _ = child.wait();
    Ok(())
}

fn resolve_db_path(env: &HashMap<String, String>) -> PathBuf {
    if let Some(path) = env.get("DB_PATH") {
        return base_dir().join(path);
    }
    let data_dir = env.get("DATA_DIR").map(String::as_str).unwrap_or("data");
    base_dir().join(Path::new(data_dir)).join("people.db")
}

#[derive(Serialize)]
struct RuntimeInfo {
    pose_model_exists: bool,
    yolo_model_exists: bool,
}

#[tauri::command]
fn get_runtime_info(_app: tauri::AppHandle) -> Result<RuntimeInfo, String> {
    let env = read_env(_app)?;
    let pose_path = env
        .get("POSE_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("yolov8n-pose.pt"));
    let yolo_path = env
        .get("YOLO_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("yolov8n.pt"));
    Ok(RuntimeInfo {
        pose_model_exists: base_dir().join(pose_path).exists(),
        yolo_model_exists: base_dir().join(yolo_path).exists(),
    })
}

#[tauri::command]
fn read_env(_app: tauri::AppHandle) -> Result<HashMap<String, String>, String> {
    let path = env_path();
    let contents = fs::read_to_string(path).unwrap_or_default();
    Ok(parse_env(&contents))
}

#[tauri::command]
fn write_env(_app: tauri::AppHandle, updates: HashMap<String, String>) -> Result<(), String> {
    let path = env_path();
    write_env_file(&path, &updates).map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Serialize)]
struct ActionRow {
    person_id: i64,
    action_label: String,
    action_confidence: Option<f64>,
    source: String,
    created_at: String,
}

#[tauri::command]
fn read_actions(_app: tauri::AppHandle, limit: Option<u32>) -> Result<Vec<ActionRow>, String> {
    let env = read_env(_app)?;
    let db_path = resolve_db_path(&env);
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT person_id, action_label, action_confidence, source, created_at
             FROM actions
             ORDER BY id DESC
             LIMIT ?1",
        )
        .map_err(|e| e.to_string())?;
    let limit_val = limit.unwrap_or(50) as i64;
    let rows = stmt
        .query_map(params![limit_val], |row| {
            Ok(ActionRow {
                person_id: row.get(0)?,
                action_label: row.get(1)?,
                action_confidence: row.get(2)?,
                source: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

#[tauri::command]
fn start_runtime(_app: tauri::AppHandle, state: State<RuntimeState>) -> Result<(), String> {
    start_runtime_inner(&state).map_err(|e| e.to_string())
}

#[tauri::command]
fn stop_runtime(state: State<RuntimeState>) -> Result<(), String> {
    stop_runtime_inner(&state).map_err(|e| e.to_string())
}

#[tauri::command]
fn apply_settings(
    _app: tauri::AppHandle,
    state: State<RuntimeState>,
    updates: HashMap<String, String>,
) -> Result<(), String> {
    let path = env_path();
    write_env_file(&path, &updates).map_err(|e| e.to_string())?;
    if state.child.lock().unwrap().is_some() {
        let _ = stop_runtime_inner(&state);
        let _ = start_runtime_inner(&state);
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![
            read_env,
            write_env,
            read_actions,
            apply_settings,
            get_runtime_info,
            start_runtime,
            stop_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
