"""
Unit tests for KITTI to YOLO coordinate converter.
"""

import unittest
from src.dataset import kitti_bbox_to_yolo


class TestAnnotationConverter(unittest.TestCase):

    def setUp(self):
        self.img_w = 1242
        self.img_h = 375

    def test_coordinate_conversion_math(self):
        """Tests that pixel box converts to exact normalized YOLO coordinates."""
        bbox = (100.0, 50.0, 300.0, 150.0)
        res = kitti_bbox_to_yolo(bbox, self.img_w, self.img_h)
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
        res = kitti_bbox_to_yolo(bbox, self.img_w, self.img_h)
        self.assertIsNone(res)

    def test_clipping_out_of_bounds(self):
        """Tests that coordinates beyond image boundaries are clamped."""
        bbox = (1200.0, 50.0, 1300.0, 100.0)
        res = kitti_bbox_to_yolo(bbox, self.img_w, self.img_h)
        self.assertIsNotNone(res)
        xc, yc, w, h = res
        expected_w = 42.0 / 1242.0
        self.assertAlmostEqual(w, expected_w, places=6)
        self.assertLessEqual(xc + w / 2, 1.0)


if __name__ == "__main__":
    unittest.main()
