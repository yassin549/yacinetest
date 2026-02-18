import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class DBClient:
    def __init__(self, db_path, data_dir, persons_dir):
        self.db_path = db_path
        self.data_dir = data_dir
        self.persons_dir = persons_dir
        self.write_lock = threading.Lock()
        self.thread_local = threading.local()

    def _apply_pragmas(self, conn):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        conn.commit()

    def ensure_db(self):
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        self._apply_pragmas(conn)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                best_face_path TEXT,
                best_face_score REAL DEFAULT 0.0,
                embedding BLOB,
                embedding_dim INTEGER
            )
            """
        )
        # Legacy table (no longer used for storing multiple images/embeddings).
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                action_label TEXT NOT NULL,
                action_confidence REAL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES persons(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_actions_person_id ON actions(person_id)")
        try:
            cur.execute("ALTER TABLE persons ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("ALTER TABLE persons ADD COLUMN embedding_dim INTEGER")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        return conn

    def get_conn(self):
        conn = getattr(self.thread_local, "db_conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            self._apply_pragmas(conn)
            self.thread_local.db_conn = conn
        return conn

    def close_thread_conn(self):
        conn = getattr(self.thread_local, "db_conn", None)
        if conn is not None:
            conn.close()
            self.thread_local.db_conn = None

    def load_known_faces(self, conn):
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, embedding, embedding_dim
            FROM persons
            WHERE embedding IS NOT NULL AND embedding_dim IS NOT NULL
            """
        )
        ids = []
        embeds = []
        for person_id, emb_blob, emb_dim in cur.fetchall():
            arr = np.frombuffer(emb_blob, dtype=np.float32)
            if emb_dim and arr.size == emb_dim:
                ids.append(person_id)
                embeds.append(arr)
        if embeds:
            embeds = np.stack(embeds, axis=0)
        else:
            embeds = np.empty((0, 128), dtype=np.float32)
        return ids, embeds

    def save_person_image(self, person_id, person_bgr):
        os.makedirs(self.persons_dir, exist_ok=True)
        path = os.path.join(self.persons_dir, f"person_{person_id}.jpg")
        cv2.imwrite(path, person_bgr)
        return path

    def create_person(self, embedding):
        with self.write_lock:
            conn = self.get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO persons (created_at, last_seen_at, best_face_path, best_face_score, embedding, embedding_dim) VALUES (?, ?, ?, ?, ?, ?)",
                (now_iso(), now_iso(), None, 0.0, embedding.astype(np.float32).tobytes(), int(embedding.size)),
            )
            person_id = cur.lastrowid
            conn.commit()
        return person_id

    def create_placeholder_person(self):
        with self.write_lock:
            conn = self.get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO persons (created_at, last_seen_at, best_face_path, best_face_score, embedding, embedding_dim) VALUES (?, ?, ?, ?, ?, ?)",
                (now_iso(), now_iso(), None, 0.0, None, None),
            )
            person_id = cur.lastrowid
            conn.commit()
        return person_id

    def upsert_person_face(self, person_id, person_bgr, embedding, score):
        with self.write_lock:
            conn = self.get_conn()
            cur = conn.cursor()
            cur.execute(
                "UPDATE persons SET last_seen_at = ? WHERE id = ?",
                (now_iso(), person_id),
            )
            cur.execute(
                "SELECT best_face_score FROM persons WHERE id = ?",
                (person_id,),
            )
            row = cur.fetchone()
            best_score = row[0] if row and row[0] is not None else 0.0
            if score > best_score:
                img_path = self.save_person_image(person_id, person_bgr)
                cur.execute(
                    "UPDATE persons SET best_face_path = ?, best_face_score = ?, embedding = ?, embedding_dim = ? WHERE id = ?",
                    (img_path, float(score), embedding.astype(np.float32).tobytes(), int(embedding.size), person_id),
                )
            conn.commit()

    def record_action(self, person_id, label, confidence, source):
        with self.write_lock:
            conn = self.get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT person_id, action_label, source
                FROM actions
                ORDER BY id DESC
                LIMIT 1
                """
            )
            last = cur.fetchone()
            if last and last[0] == person_id and last[1] == label and last[2] == source:
                return
            cur.execute(
                """
                INSERT INTO actions (person_id, action_label, action_confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, label, confidence, source, now_iso()),
            )
            cur.execute(
                "UPDATE persons SET last_seen_at = ? WHERE id = ?",
                (now_iso(), person_id),
            )
            conn.commit()
