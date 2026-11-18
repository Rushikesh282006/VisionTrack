"""
VisionTrack - Real-Time Video Processing & Analytics Pipeline
==============================================================
Integrates YOLO Detection, Multi-Object Tracking, Virtual Tripwires,
and Live HUD Telemetry Dashboard developed during Day 5.
"""

import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from src.tracking.tracker import Track, SimpleTracker
from src.tracking.counter import VehicleCounter, Point


def draw_track_trail(
    image: np.ndarray,
    history: list[Point],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws a motion trail connecting the recent history points of a tracked vehicle.

    Args:
        image: BGR image array.
        history: List of (cx, cy) center points.
        color: BGR color tuple.
        thickness: Line thickness in pixels.

    Returns:
        np.ndarray: Image with the trajectory polyline rendered.
    """
    if len(history) < 2:
        return image

    for i in range(1, len(history)):
        pt1 = (int(round(history[i - 1][0])), int(round(history[i - 1][1])))
        pt2 = (int(round(history[i][0])), int(round(history[i][1])))
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    return image


def render_pipeline_hud(
    frame: np.ndarray,
    tracks: list[Track],
    tripwire_start: tuple[float, float],
    tripwire_end: tuple[float, float],
    counts: dict,
    fps: float,
    class_names: dict[int, str],
) -> np.ndarray:
    """
    Renders all visual telemetry:
    1. Tripwire line with text label.
    2. Active track bounding boxes, IDs, and motion trails.
    3. Semi-transparent top HUD dashboard with FPS and directional counters.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Tripwire Line & Label
    p1 = (int(tripwire_start[0]), int(tripwire_start[1]))
    p2 = (int(tripwire_end[0]), int(tripwire_end[1]))
    cv2.line(canvas, p1, p2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        "TRIPWIRE",
        (p1[0] + 5, max(p1[1] - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # 2. Draw each active Track
    for t in tracks:
        if t.state == "Tracked":
            # Motion trail
            draw_track_trail(canvas, t.history, color=(0, 255, 255), thickness=2)

            # Bounding box
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # Label pill
            cls_name = class_names.get(t.class_id, "car")
            label_text = f"ID:{t.track_id} {cls_name}"
            cv2.putText(
                canvas,
                label_text,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # 3. Semi-Transparent HUD Banner at top
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # 4. Telemetry Text
    in_count = counts.get("incoming", 0)
    out_count = counts.get("outgoing", 0)
    tot_count = counts.get("total", 0)

    t1 = f"VISIONTRACK ITS  |  FPS: {fps:.1f}  |  ACTIVE: {len(tracks)}"
    t2 = f"INCOMING: {in_count}  |  OUTGOING: {out_count}  |  TOTAL: {tot_count}"

    cv2.putText(canvas, t1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, t2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return canvas


class VideoPipeline:
    """
    End-to-end video processing engine integrating Detection, Tracking,
    Virtual Tripwire Counting, and Live HUD Rendering.
    """

    def __init__(
        self,
        detector,
        tracker: SimpleTracker,
        counter: VehicleCounter,
        class_names: Optional[dict[int, str]] = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.class_names = class_names or {
            0: "car",
            1: "truck",
            2: "bus",
            3: "pedestrian",
            4: "cyclist",
        }

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """
        Runs the full pipeline on a video file and saves annotated MP4.

        Returns:
            dict: Summary metrics (frames_processed, avg_fps, counts).
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

        frame_count = 0
        total_time = 0.0
        last_counts = {"incoming": 0, "outgoing": 0, "total": 0}

        try:
            while cap.isOpened():
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                t_start = time.perf_counter()

                # 1. DETECT
                det_result = self.detector.predict(frame, imgsz=640, return_annotated=False)
                detections = [
                    (tuple(getattr(d, "box", getattr(d, "bbox", None))), d.class_id, d.confidence)
                    for d in det_result.detections
                ]

                # 2. TRACK
                tracks = self.tracker.update(detections)

                # 3. COUNT
                last_counts = self.counter.update(tracks)

                # 4. TIMING & FPS
                t_elapsed = time.perf_counter() - t_start
                total_time += t_elapsed
                fps_current = 1.0 / max(t_elapsed, 1e-5)

                # 5. RENDER HUD
                annotated = render_pipeline_hud(
                    frame=frame,
                    tracks=tracks,
                    tripwire_start=self.counter.line_start,
                    tripwire_end=self.counter.line_end,
                    counts=last_counts,
                    fps=fps_current,
                    class_names=self.class_names,
                )

                if writer:
                    writer.write(annotated)

        finally:
            cap.release()
            if writer:
                writer.release()

        avg_fps = frame_count / max(total_time, 1e-5) if frame_count > 0 else 0.0
        return {
            "frames_processed": frame_count,
            "avg_fps": avg_fps,
            "counts": last_counts,
        }
