"""
VisionTrack - Virtual Tripwire & Vehicle Counter
================================================
Directional line-crossing detection and traffic counting developed during Day 4.
"""

from typing import Optional
from src.tracking.tracker import Track

Point = tuple[float, float]


def ccw(A: Point, B: Point, C: Point) -> float:
    """Computes orientation of triplet (A, B, C)."""
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])


def do_segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """
    Returns True if line segment AB intersects line segment CD using the CCW orientation test.
    """
    cd_opposite = (ccw(A, B, C) > 0) != (ccw(A, B, D) > 0)
    ab_opposite = (ccw(C, D, A) > 0) != (ccw(C, D, B) > 0)
    return cd_opposite and ab_opposite


class VehicleCounter:
    """
    Directional vehicle counter using a virtual tripwire line.
    """

    def __init__(
        self,
        line_start: tuple[float, float],
        line_end: tuple[float, float],
        class_names: Optional[dict[int, str]] = None,
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.class_names = class_names or {
            0: "car",
            1: "truck",
            2: "bus",
            3: "pedestrian",
            4: "cyclist",
        }

        # Set to prevent counting the same vehicle multiple times
        self.counted_ids: set[int] = set()

        # Dictionary storing counts by direction and category
        self.counts = {
            "incoming": {name: 0 for name in self.class_names.values()},
            "outgoing": {name: 0 for name in self.class_names.values()},
        }

    def update(self, tracks: list[Track]) -> dict:
        """
        Checks active tracks against the tripwire line and updates counts.

        Returns:
            dict: Summary with {"incoming": int, "outgoing": int, "total": int, "details": dict}
        """
        for track in tracks:
            if (
                track.track_id in self.counted_ids
                or track.state != "Tracked"
                or len(track.history) < 2
            ):
                continue

            prev_pt = track.history[-2]
            curr_pt = track.history[-1]

            if do_segments_intersect(prev_pt, curr_pt, self.line_start, self.line_end):
                self.counted_ids.add(track.track_id)
                direction = "incoming" if curr_pt[1] > prev_pt[1] else "outgoing"
                class_name = self.class_names.get(track.class_id, "car")
                if class_name in self.counts[direction]:
                    self.counts[direction][class_name] += 1
                else:
                    self.counts[direction][class_name] = 1

        total_in = sum(self.counts["incoming"].values())
        total_out = sum(self.counts["outgoing"].values())
        total = total_in + total_out

        return {
            "incoming": total_in,
            "outgoing": total_out,
            "total": total,
            "details": self.counts,
        }
