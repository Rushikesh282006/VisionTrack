"""
Integration tests for VisionTrack Day 6 FastAPI Service.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app


class TestVisionTrackAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Using context manager to trigger lifespan startup
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()
        cls.sample_image = Path("data/samples/bus.jpg")

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_root_endpoint(self):
        """Tests that GET / returns service status and docs URL."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("docs_url", data)
        self.assertEqual(data["version"], "1.0.0")

    def test_health_endpoint(self):
        """Tests that GET /health returns healthy system status and uptime."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("device", data)
        self.assertIn("uptime_seconds", data)
        self.assertTrue(data["model_loaded"])

    def test_detect_image_valid(self):
        """Tests POST /detect/image with real vehicle image (bus.jpg)."""
        self.assertTrue(self.sample_image.exists(), "Sample bus.jpg missing")
        with open(self.sample_image, "rb") as f:
            res = self.client.post(
                "/detect/image",
                files={"file": ("bus.jpg", f, "image/jpeg")},
                data={"conf_threshold": 0.25}
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_detections"], 0)
        self.assertIn("bus", data["class_counts"])
        self.assertGreater(data["inference_time_ms"], 0.0)
        self.assertEqual(len(data["detections"]), data["total_detections"])

    def test_detect_image_corrupted(self):
        """Tests POST /detect/image rejects invalid image data with HTTP 400."""
        res = self.client.post(
            "/detect/image",
            files={"file": ("corrupt.txt", b"plain text is not an image", "text/plain")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("not a valid", res.json()["detail"])

    def test_track_video_synthetic(self):
        """Tests POST /track/video with synthetic 5-frame video."""
        temp_dir = Path("outputs/test_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        synthetic_video = temp_dir / "api_test_video.mp4"

        # Generate small 5-frame video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(synthetic_video), fourcc, 10.0, (200, 200))
        for _ in range(5):
            writer.write(np.full((200, 200, 3), 100, dtype=np.uint8))
        writer.release()

        with open(synthetic_video, "rb") as f:
            res = self.client.post(
                "/track/video",
                files={"file": ("api_test_video.mp4", f, "video/mp4")},
                data={"tripwire_ratio": 0.5, "max_frames": 5}
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["frames_processed"], 5)
        self.assertGreater(data["avg_fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
