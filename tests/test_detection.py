"""
Unit tests for VisionTrack Detection & Utils modules.
"""

import unittest
from pathlib import Path
from src.utils import validate_image_path, Timer, get_device
from src.detection import YOLODetector, DetectionResult


class TestVisionTrackDay1(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.sample_image = self.project_root / "data" / "samples" / "bus.jpg"
        self.model_path = self.project_root / "yolo26n.pt"

    def test_validate_image_path_success(self):
        """Tests that a valid existing image path is resolved correctly."""
        resolved = validate_image_path(self.sample_image)
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.suffix.lower(), ".jpg")

    def test_validate_image_path_not_found(self):
        """Tests that a non-existent path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            validate_image_path("non_existent_image.png")

    def test_timer_context_manager(self):
        """Tests that the Timer context manager accurately records non-zero elapsed time."""
        with Timer("TestOperation") as timer:
            _ = sum(i * i for i in range(100_000))
        self.assertGreater(timer.elapsed, 0.0)

    def test_yolo_detector_inference(self):
        """Tests end-to-end detector prediction on CPU."""
        detector = YOLODetector(
            model_path=self.model_path,
            device="cpu",
            conf_threshold=0.25
        )
        result = detector.predict(self.sample_image, imgsz=320, return_annotated=False)
        self.assertIsInstance(result, DetectionResult)
        self.assertGreater(len(result.detections), 0)
        self.assertIn("bus", result.count_by_class())
        self.assertIn("person", result.count_by_class())


if __name__ == "__main__":
    unittest.main()

