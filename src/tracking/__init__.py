"""
VisionTrack Multi-Object Tracking & Vehicle Counting Module
===========================================================
Modular tracker, Hungarian matching, Track lifecycle management,
and directional virtual tripwire counting.
"""

from src.tracking.tracker import (
    Track,
    SimpleTracker,
    compute_iou_cost_matrix,
    associate_detections_to_tracks,
)
from src.tracking.counter import (
    Point,
    ccw,
    do_segments_intersect,
    VehicleCounter,
)

__all__ = [
    "Track",
    "SimpleTracker",
    "compute_iou_cost_matrix",
    "associate_detections_to_tracks",
    "Point",
    "ccw",
    "do_segments_intersect",
    "VehicleCounter",
]
