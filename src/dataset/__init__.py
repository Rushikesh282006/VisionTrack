"""
VisionTrack Dataset Engineering Module
======================================
Modular functions for KITTI-to-YOLO conversion, leak-free dataset splitting,
OpenCV annotation rendering, and data integrity validation.
"""

from src.dataset.converter import (
    kitti_bbox_to_yolo,
    parse_kitti_line,
    convert_kitti_file,
)
from src.dataset.split import split_dataset
from src.dataset.visualize import yolo_to_pixel_bbox, render_yolo_boxes
from src.dataset.validator import validate_yolo_line

__all__ = [
    "kitti_bbox_to_yolo",
    "parse_kitti_line",
    "convert_kitti_file",
    "split_dataset",
    "yolo_to_pixel_bbox",
    "render_yolo_boxes",
    "validate_yolo_line",
]
