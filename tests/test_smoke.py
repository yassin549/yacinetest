import os
import threading
import tempfile
import unittest

import numpy as np

from db import DBClient, now_iso
from matching import FaceRegistry


class SmokeTests(unittest.TestCase):
    def test_db_schema_and_known_faces_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "people.db")
            persons_dir = os.path.join(tmpdir, "persons")
            client = DBClient(db_path, tmpdir, persons_dir)
            conn = client.ensure_db()

            emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO persons (created_at, last_seen_at, best_face_score, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_iso(), now_iso(), 0.5, emb.tobytes(), emb.size),
            )
            person_id = cur.lastrowid
            conn.commit()

            ids, embeds = client.load_known_faces(conn)
            conn.close()

            self.assertIn(person_id, ids)
            idx = ids.index(person_id)
            np.testing.assert_allclose(embeds[idx], emb)

    def test_get_db_returns_thread_local_connections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "people.db")
            persons_dir = os.path.join(tmpdir, "persons")
            client = DBClient(db_path, tmpdir, persons_dir)
            conn = client.ensure_db()
            conn.close()

            main_conn = client.get_conn()
            result = {}

            def _worker():
                worker_conn = client.get_conn()
                result["worker_conn_id"] = id(worker_conn)
                cur = worker_conn.cursor()
                cur.execute("SELECT 1")
                result["worker_ok"] = cur.fetchone()[0] == 1
                client.close_thread_conn()

            t = threading.Thread(target=_worker)
            t.start()
            t.join()

            self.assertNotEqual(id(main_conn), result["worker_conn_id"])
            self.assertTrue(result["worker_ok"])
            client.close_thread_conn()

    def test_face_registry_match_smoke(self):
        registry = FaceRegistry(embedding_dim=3)
        registry.load([123], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

        pid, dist = registry.match(np.array([1.0, 0.0, 0.0], dtype=np.float32), threshold=0.35)
        self.assertEqual(pid, 123)
        self.assertIsNotNone(dist)

        pid2, dist2 = registry.match(np.array([0.0, 1.0, 0.0], dtype=np.float32), threshold=0.35)
        self.assertIsNone(pid2)
        self.assertIsNone(dist2)


if __name__ == "__main__":
    unittest.main()
