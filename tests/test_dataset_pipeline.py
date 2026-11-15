"""
Unit tests for VisionTrack Day 2 Dataset Engineering module.
"""

import unittest
from src.dataset import (
    kitti_bbox_to_yolo,
    parse_kitti_line,
    convert_kitti_file,
    split_dataset,
    yolo_to_pixel_bbox,
    validate_yolo_line,
)


class TestDay2DatasetModule(unittest.TestCase):

    def setUp(self):
        self.img_w = 1242
        self.img_h = 375
        self.class_mapping = {
            "Car": 0, "Van": 0, "Truck": 1, "Tram": 2,
            "Pedestrian": 3, "Person_sitting": 3, "Cyclist": 4
        }

    def test_kitti_bbox_to_yolo_math(self):
        """Tests forward normalization coordinate math."""
        bbox = (100.0, 50.0, 300.0, 150.0)
        res = kitti_bbox_to_yolo(bbox, self.img_w, self.img_h)
        self.assertIsNotNone(res)
        xc, yc, w, h = res
        self.assertAlmostEqual(xc, 200.0 / 1242.0, places=6)
        self.assertAlmostEqual(yc, 100.0 / 375.0, places=6)
        self.assertAlmostEqual(w, 200.0 / 1242.0, places=6)
        self.assertAlmostEqual(h, 100.0 / 375.0, places=6)

    def test_roundtrip_coordinate_invariance(self):
        """Tests that (pixel -> YOLO -> pixel) returns the original bounding box."""
        original_box = (100.0, 50.0, 300.0, 150.0)
        yolo_box = kitti_bbox_to_yolo(original_box, self.img_w, self.img_h)
        self.assertIsNotNone(yolo_box)
        recovered_box = yolo_to_pixel_bbox(yolo_box, self.img_w, self.img_h)
        self.assertEqual(recovered_box, (100, 50, 300, 150))

    def test_parse_kitti_line(self):
        """Tests parsing a raw KITTI 15-column line."""
        line = "Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56"
        res = parse_kitti_line(line, self.class_mapping)
        self.assertIsNotNone(res)
        cls_id, bbox = res
        self.assertEqual(cls_id, 1)
        self.assertAlmostEqual(bbox[0], 599.41)

    def test_split_dataset_proportions(self):
        """Tests that splitting partitions deterministically into 80/10/10."""
        samples = [f"sample_{i:03d}" for i in range(100)]
        splits = split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 10)
        # Ensure zero overlap
        all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        self.assertEqual(len(all_ids), 100)

    def test_validate_yolo_line(self):
        """Tests validator checks."""
        # Valid line
        valid, msg = validate_yolo_line("0 0.5 0.5 0.2 0.3")
        self.assertTrue(valid)
        self.assertEqual(msg, "Valid")

        # Invalid class ID
        valid, msg = validate_yolo_line("9 0.5 0.5 0.2 0.3")
        self.assertFalse(valid)

        # Non-positive dimension
        valid, msg = validate_yolo_line("0 0.5 0.5 -0.1 0.3")
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
