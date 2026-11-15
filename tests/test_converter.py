"""
Unit tests for KITTI to YOLO coordinate converter.
"""

import unittest
from pathlib import Path
from src.dataset.converter import KITTItoYOLOConverter, YOLOBoundingBox


class TestAnnotationConverter(unittest.TestCase):

    def setUp(self):
        self.converter = KITTItoYOLOConverter(Path("configs/dataset.yaml"))
        self.img_w = 1242
        self.img_h = 375

    def test_coordinate_conversion_math(self):
        """Tests that pixel box converts to exact normalized YOLO coordinates."""
        # Box from 100, 50 to 300, 150
        # center_x = 200, center_y = 100, width = 200, height = 100
        bbox = (100.0, 50.0, 300.0, 150.0)
        res = self.converter.convert_bbox(bbox, self.img_w, self.img_h)
        self.assertIsNotNone(res)
        xc, yc, w, h = res

        expected_xc = 200.0 / 1242.0
        expected_yc = 100.0 / 375.0
        expected_w = 200.0 / 1242.0
        expected_h = 100.0 / 375.0

        self.assertAlmostEqual(xc, expected_xc, places=6)
        self.assertAlmostEqual(yc, expected_yc, places=6)
        self.assertAlmostEqual(w, expected_w, places=6)
        self.assertAlmostEqual(h, expected_h, places=6)

    def test_inverted_coordinates_rejected(self):
        """Tests that inverted boxes (xmax <= xmin) are rejected."""
        bbox = (300.0, 50.0, 100.0, 150.0)
        res = self.converter.convert_bbox(bbox, self.img_w, self.img_h)
        self.assertIsNone(res)

    def test_clipping_out_of_bounds(self):
        """Tests that coordinates beyond image boundaries are clamped."""
        # Box extending past right edge (1300 > 1242)
        bbox = (1200.0, 50.0, 1300.0, 100.0)
        res = self.converter.convert_bbox(bbox, self.img_w, self.img_h)
        self.assertIsNotNone(res)
        xc, yc, w, h = res
        # Clamped box is (1200, 50, 1242, 100) -> width is 42
        expected_w = 42.0 / 1242.0
        self.assertAlmostEqual(w, expected_w, places=6)
        self.assertLessEqual(xc + w / 2, 1.0)

    def test_subpixel_box_rejected(self):
        """Tests that boxes smaller than min_box_dimension_px are rejected."""
        bbox = (100.0, 50.0, 101.0, 50.5)  # 1px wide, 0.5px tall
        res = self.converter.convert_bbox(bbox, self.img_w, self.img_h)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
