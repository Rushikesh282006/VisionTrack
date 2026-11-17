"""
Unit tests for VisionTrack Day 3 Training & Metrics Module.
"""

import unittest
from pathlib import Path
from src.training import compute_iou, load_train_config


class TestDay3TrainingPipeline(unittest.TestCase):

    def test_iou_identical_boxes(self):
        """Tests that identical bounding boxes have an IoU of 1.0."""
        box = (10.0, 20.0, 110.0, 120.0)
        iou = compute_iou(box, box)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_iou_disjoint_boxes(self):
        """Tests that non-overlapping boxes have an IoU of 0.0."""
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (50.0, 50.0, 70.0, 70.0)
        iou = compute_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_iou_partial_overlap(self):
        """
        Tests known partial overlap:
        Box 1: (0, 0, 10, 10) -> Area 100
        Box 2: (5, 0, 15, 10) -> Area 100
        Intersection: (5, 0, 10, 10) -> Area 50
        Union: 100 + 100 - 50 = 150
        IoU: 50 / 150 = 1/3 ~ 0.333333
        """
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (5.0, 0.0, 15.0, 10.0)
        iou = compute_iou(box1, box2)
        self.assertAlmostEqual(iou, 50.0 / 150.0, places=5)

    def test_iou_nested_box(self):
        """Tests nested box where Box 2 is inside Box 1."""
        box1 = (0.0, 0.0, 10.0, 10.0)  # Area 100
        box2 = (0.0, 0.0, 5.0, 5.0)     # Area 25
        # Intersection = 25, Union = 100
        iou = compute_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.25, places=5)

    def test_iou_zero_division_guard(self):
        """Tests that degraded zero-area boxes do not crash with ZeroDivisionError."""
        box1 = (0.0, 0.0, 0.0, 0.0)
        box2 = (0.0, 0.0, 0.0, 0.0)
        iou = compute_iou(box1, box2)
        self.assertEqual(iou, 0.0)

    def test_load_train_config_keys(self):
        """Tests that train_config.yaml loads with all required training parameters."""
        config_path = "configs/train_config.yaml"
        self.assertTrue(Path(config_path).exists(), f"{config_path} does not exist")
        cfg = load_train_config(config_path)
        self.assertIsInstance(cfg, dict)

        required_keys = ["model", "data", "epochs", "batch", "imgsz", "optimizer", "lr0"]
        for key in required_keys:
            self.assertIn(key, cfg, f"Missing required configuration key: {key}")


if __name__ == "__main__":
    unittest.main()
