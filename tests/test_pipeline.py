"""
Unit tests for VisionTrack Day 5 Video Pipeline & HUD Module.
"""

import unittest
from pathlib import Path
import cv2
import numpy as np

from src.tracking.tracker import Track, SimpleTracker
from src.tracking.counter import VehicleCounter
from src.pipeline import draw_track_trail, render_pipeline_hud, VideoPipeline


class DummyDetector:
    """Mock detector returning predictable detections for pipeline testing."""
    def predict(self, frame, imgsz=640, return_annotated=False):
        class DummyDet:
            def __init__(self, bbox, class_id, conf):
                self.bbox = bbox
                self.class_id = class_id
                self.confidence = conf

        class DummyResult:
            def __init__(self, detections):
                self.detections = detections

        # Vehicle moving downwards across y=100
        return DummyResult([
            DummyDet((40.0, 80.0, 80.0, 120.0), 0, 0.9)
        ])


class TestDay5VideoPipeline(unittest.TestCase):

    def setUp(self):
        self.canvas = np.zeros((300, 400, 3), dtype=np.uint8)
        self.class_names = {0: "car", 1: "truck", 2: "bus"}

    def test_draw_track_trail_empty_or_single(self):
        """Tests that trails with < 2 points do not modify the image."""
        original = self.canvas.copy()
        res = draw_track_trail(self.canvas, history=[(10.0, 10.0)])
        np.testing.assert_array_equal(original, res)

    def test_draw_track_trail_multiple_points(self):
        """Tests drawing trail connecting multiple history points."""
        history = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
        res = draw_track_trail(self.canvas, history, color=(0, 255, 255), thickness=2)
        self.assertEqual(res.shape, (300, 400, 3))
        # Check that pixels along the trail are colored
        self.assertTrue(np.any(res > 0))

    def test_render_pipeline_hud(self):
        """Tests HUD rendering with tracks and telemetry."""
        track = Track(track_id=1, bbox=(50.0, 50.0, 100.0, 100.0), class_id=0, confidence=0.88)
        track.update(bbox=(50.0, 60.0, 100.0, 110.0), confidence=0.9)  # hits=2 -> Tracked

        counts = {"incoming": 1, "outgoing": 0, "total": 1}
        annotated = render_pipeline_hud(
            frame=self.canvas,
            tracks=[track],
            tripwire_start=(0.0, 150.0),
            tripwire_end=(400.0, 150.0),
            counts=counts,
            fps=32.5,
            class_names=self.class_names
        )

        self.assertEqual(annotated.shape, self.canvas.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_video_pipeline_end_to_end_synthetic(self):
        """Generates a synthetic 5-frame MP4 and processes it with VideoPipeline."""
        temp_dir = Path("outputs/test_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        input_vid = temp_dir / "synthetic_input.mp4"
        output_vid = temp_dir / "synthetic_output.mp4"

        # 1. Create a 5-frame dummy video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_vid), fourcc, 10.0, (200, 200))
        for _ in range(5):
            frame = np.full((200, 200, 3), 128, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # 2. Setup pipeline
        detector = DummyDetector()
        tracker = SimpleTracker(max_cost=0.7)
        counter = VehicleCounter(line_start=(0.0, 100.0), line_end=(200.0, 100.0))
        pipeline = VideoPipeline(detector, tracker, counter, self.class_names)

        # 3. Process video
        metrics = pipeline.process_video(
            input_path=str(input_vid),
            output_path=str(output_vid),
            max_frames=5
        )

        self.assertEqual(metrics["frames_processed"], 5)
        self.assertGreater(metrics["avg_fps"], 0.0)
        self.assertTrue(output_vid.exists())
        self.assertGreater(output_vid.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
