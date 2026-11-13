"""
VisionTrack Package
===================
Real-Time Intelligent Traffic Monitoring and Vehicle Detection System.
"""

from src.detection import YOLODetector, Detection, DetectionResult
from src.utils import setup_logger, get_device, validate_image_path, Timer

__all__ = [
    "YOLODetector",
    "Detection",
    "DetectionResult",
    "setup_logger",
    "get_device",
    "validate_image_path",
    "Timer",
]

