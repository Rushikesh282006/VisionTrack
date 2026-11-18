"""
VisionTrack - Multi-Object Tracker (MOT)
========================================
Core tracking engine, Hungarian assignment, and Track lifecycle management developed during Day 4.
"""

from typing import Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy greedy bipartite matching fallback when scipy is not installed.
        Greedily pairs lowest-cost (highest-IoU) candidates.
        """
        if cost_matrix.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        rows, cols = cost_matrix.shape
        matched_r: set[int] = set()
        matched_c: set[int] = set()
        row_ind: list[int] = []
        col_ind: list[int] = []

        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            r = int(idx // cols)
            c = int(idx % cols)
            if r not in matched_r and c not in matched_c:
                matched_r.add(r)
                matched_c.add(c)
                row_ind.append(r)
                col_ind.append(c)
                if len(matched_r) == min(rows, cols):
                    break

        return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)

from src.training.metrics import compute_iou


def compute_iou_cost_matrix(
    tracks_boxes: list[tuple[float, float, float, float]],
    detections_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Computes an (M, N) cost matrix where cost[i, j] = 1.0 - IoU(track_i, det_j).

    Args:
        tracks_boxes: List of M boxes (xmin, ymin, xmax, ymax) from existing tracks.
        detections_boxes: List of N boxes (xmin, ymin, xmax, ymax) from new detections.

    Returns:
        np.ndarray: Matrix of shape (M, N) containing float costs in [0.0, 1.0].
    """
    rows, cols = len(tracks_boxes), len(detections_boxes)

    if not tracks_boxes or not detections_boxes:
        return np.zeros((rows, cols), dtype=np.float32)

    cost_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            cost_matrix[i, j] = 1.0 - compute_iou(tracks_boxes[i], detections_boxes[j])

    return cost_matrix


def associate_detections_to_tracks(
    cost_matrix: np.ndarray,
    max_cost: float = 0.7,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Associates tracks with detections using Hungarian Algorithm with cost gating.

    Args:
        cost_matrix: (M, N) matrix of costs.
        max_cost: Threshold above which a match is rejected (default 0.7 -> IoU < 0.3).

    Returns:
        matches: List of tuples (track_idx, det_idx) for valid matches.
        unmatched_tracks: List of track indices that have no valid match.
        unmatched_detections: List of detection indices that have no valid match.
    """
    num_tracks, num_detections = cost_matrix.shape

    if num_tracks == 0 or num_detections == 0:
        return [], list(range(num_tracks)), list(range(num_detections))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))

    matched_tracks = {r for r, _ in matches}
    matched_detections = {c for _, c in matches}

    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    unmatched_detections = [j for j in range(num_detections) if j not in matched_detections]

    return matches, unmatched_tracks, unmatched_detections


class Track:
    """
    Represents a single tracked vehicle across time.
    """

    def __init__(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        class_id: int,
        confidence: float,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.state = "New"  # "New", "Tracked", "Lost", "Removed"
        self.hits = 1
        self.lost_frames = 0

        # Center point (cx, cy)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history: list[tuple[float, float]] = [(cx, cy)]

    def update(self, bbox: tuple[float, float, float, float], confidence: float):
        """Called when this track is successfully matched to a detection."""
        self.lost_frames = 0
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1

        if self.hits >= 2:
            self.state = "Tracked"

        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        self.history.append((new_cx, new_cy))

        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self, max_lost_frames: int = 30):
        """Called when this track is NOT matched in the current frame."""
        self.lost_frames += 1

        if self.state == "New":
            self.state = "Removed"
        elif self.lost_frames > max_lost_frames:
            self.state = "Removed"
        else:
            self.state = "Lost"


class SimpleTracker:
    """
    Multi-Object Tracker coordinating Track lifecycles across video frames.
    """

    def __init__(self, max_cost: float = 0.7, max_lost_frames: int = 30):
        self.max_cost = max_cost
        self.max_lost_frames = max_lost_frames
        self.tracks: list[Track] = []
        self.next_id: int = 1
        self.frame_count: int = 0

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], int, float]],
    ) -> list[Track]:
        """
        Updates tracks with new detections from the current frame.

        Args:
            detections: List of (bbox, class_id, confidence).

        Returns:
            list[Track]: Confirmed active tracks for visualization and counting.
        """
        self.frame_count += 1
        active_tracks = [t for t in self.tracks if t.state != "Removed"]

        if not active_tracks:
            for bbox, class_id, conf in detections:
                new_track = Track(self.next_id, bbox, class_id, conf)
                self.tracks.append(new_track)
                self.next_id += 1
            return [t for t in self.tracks if t.state == "Tracked"]

        track_boxes = [t.bbox for t in active_tracks]
        det_boxes = [d[0] for d in detections]

        cost_matrix = compute_iou_cost_matrix(track_boxes, det_boxes)
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            cost_matrix, max_cost=self.max_cost
        )

        for track_idx, det_idx in matches:
            bbox, class_id, conf = detections[det_idx]
            active_tracks[track_idx].update(bbox, conf)

        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_missed(self.max_lost_frames)

        for det_idx in unmatched_dets:
            bbox, class_id, conf = detections[det_idx]
            new_track = Track(self.next_id, bbox, class_id, conf)
            self.tracks.append(new_track)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.state != "Removed"]

        return [t for t in self.tracks if t.state == "Tracked"]
