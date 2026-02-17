import os
import tempfile
import unittest

from config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("RTSP_URL=rtsp://user:pass@127.0.0.1:554/stream\n")
                f.write("SCALE=0.4\n")
                f.write("HEADLESS=true\n")
                f.write("LOG_LEVEL=DEBUG\n")

            prev = dict(os.environ)
            try:
                cfg = load_config(env_path)
            finally:
                os.environ.clear()
                os.environ.update(prev)

            self.assertEqual(cfg.rtsp_url, "rtsp://user:pass@127.0.0.1:554/stream")
            self.assertAlmostEqual(cfg.scale, 0.4)
            self.assertTrue(cfg.headless)
            self.assertEqual(cfg.log_level, "DEBUG")

    def test_invalid_numeric_config_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("RTSP_URL=rtsp://user:pass@127.0.0.1:554/stream\n")
                f.write("INFER_EVERY=0\n")

            prev = dict(os.environ)
            try:
                with self.assertRaises(ValueError):
                    load_config(env_path)
            finally:
                os.environ.clear()
                os.environ.update(prev)


if __name__ == "__main__":
    unittest.main()
