"""
Unit tests for VisionTrack Day 4 Tracking & Counting Module.
"""

import unittest
import numpy as np
from src.tracking import (
    Track,
    SimpleTracker,
    VehicleCounter,
    compute_iou_cost_matrix,
    associate_detections_to_tracks,
    do_segments_intersect,
)


class TestDay4TrackingPipeline(unittest.TestCase):

    def test_compute_iou_cost_matrix(self):
        """Tests that cost matrix computes 1.0 - IoU correctly."""
        tracks = [(10.0, 10.0, 50.0, 50.0), (100.0, 100.0, 150.0, 150.0)]
        dets = [
            (10.0, 10.0, 50.0, 50.0),   # Identical to track 0 -> cost 0.0
            (500.0, 500.0, 550.0, 550.0) # Disjoint -> cost 1.0
        ]
        cost_mat = compute_iou_cost_matrix(tracks, dets)
        self.assertEqual(cost_mat.shape, (2, 2))
        self.assertAlmostEqual(cost_mat[0, 0], 0.0, places=4)
        self.assertAlmostEqual(cost_mat[0, 1], 1.0, places=4)

    def test_cost_matrix_empty_guard(self):
        """Tests empty tracks or detections input handling."""
        mat = compute_iou_cost_matrix([], [(10.0, 10.0, 20.0, 20.0)])
        self.assertEqual(mat.shape, (0, 1))

    def test_associate_detections_to_tracks(self):
        """Tests Hungarian matching and cost threshold rejection."""
        # 2 tracks, 2 detections
        # Pair (0, 0) has cost 0.1 (good match)
        # Pair (1, 1) has cost 0.9 (cost > max_cost 0.7 -> rejected)
        cost_matrix = np.array([
            [0.1, 1.0],
            [1.0, 0.9]
        ], dtype=np.float32)

        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            cost_matrix, max_cost=0.7
        )

        self.assertEqual(matches, [(0, 0)])
        self.assertIn(1, unmatched_tracks)
        self.assertIn(1, unmatched_dets)

    def test_track_lifecycle(self):
        """Tests state transitions: New -> Tracked -> Lost -> Removed."""
        track = Track(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_id=0, confidence=0.9)
        self.assertEqual(track.state, "New")
        self.assertEqual(track.hits, 1)

        # Frame 2: Matched! Hits becomes 2 -> Tracked
        track.update(bbox=(2.0, 2.0, 12.0, 12.0), confidence=0.92)
        self.assertEqual(track.state, "Tracked")
        self.assertEqual(track.hits, 2)
        self.assertEqual(len(track.history), 2)

        # Frame 3: Missed! Becomes Lost
        track.mark_missed(max_lost_frames=5)
        self.assertEqual(track.state, "Lost")
        self.assertEqual(track.lost_frames, 1)

        # Exceed max_lost_frames -> Removed
        for _ in range(5):
            track.mark_missed(max_lost_frames=5)
        self.assertEqual(track.state, "Removed")

    def test_track_noise_rejection(self):
        """Tests that a single-frame unconfirmed 'New' track is removed immediately when missed."""
        track = Track(track_id=99, bbox=(0.0, 0.0, 10.0, 10.0), class_id=0, confidence=0.3)
        track.mark_missed(max_lost_frames=30)
        self.assertEqual(track.state, "Removed")

    def test_do_segments_intersect(self):
        """Tests 2D line segment intersection."""
        # Cross (+) intersection
        p1 = (0.0, 5.0)
        p2 = (10.0, 5.0)
        q1 = (5.0, 0.0)
        q2 = (5.0, 10.0)
        self.assertTrue(do_segments_intersect(p1, p2, q1, q2))

        # Parallel horizontal lines (never intersect)
        p3 = (0.0, 0.0)
        p4 = (10.0, 0.0)
        self.assertFalse(do_segments_intersect(p1, p2, p3, p4))

        # Non-touching segments
        p5 = (0.0, 5.0)
        p6 = (4.0, 5.0)  # stops before x=5
        self.assertFalse(do_segments_intersect(p5, p6, q1, q2))

    def test_vehicle_counter_directional(self):
        """Tests counting across virtual tripwire and directional classification."""
        counter = VehicleCounter(line_start=(0.0, 100.0), line_end=(200.0, 100.0))

        # Track 1: Moving Downwards (y: 80 -> 120) across y=100 (incoming)
        t1 = Track(track_id=1, bbox=(40.0, 70.0, 60.0, 90.0), class_id=0, confidence=0.85)
        t1.update(bbox=(40.0, 110.0, 60.0, 130.0), confidence=0.88)  # hits=2 -> Tracked

        res = counter.update([t1])
        self.assertEqual(res["incoming"], 1)
        self.assertEqual(res["outgoing"], 0)
        self.assertEqual(res["total"], 1)

        # Ensure deduplication: same track on next frame is not counted again
        t1.update(bbox=(40.0, 130.0, 60.0, 150.0), confidence=0.89)
        res2 = counter.update([t1])
        self.assertEqual(res2["total"], 1)

        # Track 2: Moving Upwards (y: 120 -> 80) across y=100 (outgoing)
        t2 = Track(track_id=2, bbox=(40.0, 110.0, 60.0, 130.0), class_id=1, confidence=0.9)
        t2.update(bbox=(40.0, 70.0, 60.0, 90.0), confidence=0.91)  # hits=2 -> Tracked

        res3 = counter.update([t2])
        self.assertEqual(res3["incoming"], 1)
        self.assertEqual(res3["outgoing"], 1)
        self.assertEqual(res3["total"], 2)


if __name__ == "__main__":
    unittest.main()
