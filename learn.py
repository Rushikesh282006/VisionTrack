from networkx.algorithms.bipartite import color
def kitti_bbox_to_yolo(
    bbox: tuple[float, float, float, float], 
    img_width: int, 
    img_height: int
) -> tuple[float, float, float, float]:
    """
    Converts KITTI pixel bounding box to normalized YOLO format.
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.
        img_width: Physical width of the image in pixels.
        img_height: Physical height of the image in pixels.
        
    Returns:
        (x_center, y_center, width, height) normalized to [0.0, 1.0].
    """
    x_min,y_min,x_max,y_max = bbox

    if x_min >= x_max or y_min >= y_max :
        return None

    width = x_max - x_min
    height = y_max - y_min

    x_center =  x_min + width/2
    y_center =  y_min + height/2

    x_c_norm = x_center / img_width
    y_c_norm = y_center / img_height

    width = width / img_width
    height = height / img_height
    
    return (x_c_norm,y_c_norm,width,height) 


def parse_kitti_line(
    line: str, 
    class_mapping: dict[str, int]
) -> tuple[int, tuple[float, float, float, float]] | None:
    
    line_list = line.split()

    if len(line_list) < 15:
        return None    
    elif line_list[0] not in class_mapping:
        return None
    else:    
        list_key = class_mapping[line_list[0]]
        return (list_key,(float(line_list[4]),float(line_list[5]),float(line_list[6]),float(line_list[7]),))
    

def convert_kitti_file(
    file_path: str,
    img_width: int,
    img_height: int,
    class_mapping: dict[str, int]
) -> list[str]:
    
        yolo_lines = []
        
        with open(file_path, "r") as file:
            for line in file:
                temp = parse_kitti_line(line,class_mapping)
                if temp != None:
                    yolo_lines.append(temp)

        file.close()

        yolo_str = []

        for line in yolo_lines:
            if line is not None:
                temp = kitti_bbox_to_yolo(line[1],img_width,img_height)
                yolo_str.append(f"{line[0]} {temp[0]} {temp[1]} {temp[2]} {temp[3]}")

        return yolo_str
    

import random

def split_dataset(
    sample_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict[str, list[str]]:
    """
    Splits sample IDs into train, val, and test sets reproducibly.
    
    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    if train_ratio+test_ratio+val_ratio != 1 :
        raise ValueError("The Sum of ratios should be 1")

    random.seed(seed)

    shuffled_data = sample_ids.copy()
    random.shuffle(shuffled_data)
    n  = len(shuffled_data)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train : n_train + n_val]
    test_data = shuffled_data[n_train + n_val :]
    data_dict = {"train_data":train_data,"test_data":test_data,"val_data":val_data}

    return data_dict

from pathlib import Path

# if __name__ == "__main__":
#     labels_dir = Path("data/raw/kitti/training/label_2")
    
#     # Grab all 7,481 file stems (e.g. '000000', '000001', ...)
#     sample_ids = [p.stem for p in sorted(labels_dir.glob("*.txt"))]
    
#     splits = split_dataset(sample_ids)
    
#     print("\n--- KITTI DATASET PARTITIONING ---")
#     print(f"Total files: {len(sample_ids)}")
#     print(f"Train split: {len(splits['train_data'])} ({len(splits['train_data'])/len(sample_ids)*100:.1f}%)")
#     print(f"Val split:   {len(splits['val_data'])} ({len(splits['val_data'])/len(sample_ids)*100:.1f}%)")
#     print(f"Test split:  {len(splits['test_data'])} ({len(splits['test_data'])/len(sample_ids)*100:.1f}%)")






def yolo_to_pixel_bbox(
    yolo_box: tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> tuple[int, int, int, int]:
    """
    Converts normalized YOLO coordinates back into pixel coordinates (xmin, ymin, xmax, ymax).
    
    Args:
        yolo_box: (x_center, y_center, width, height) normalized in [0.0, 1.0].
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        
    Returns:
        (xmin, ymin, xmax, ymax) as integer pixel coordinates.
    """
    
    x_c_px = yolo_box[0] * img_width

    y_c_px = yolo_box[1] * img_height

    w_px = yolo_box[2] * img_width

    h_px = yolo_box[3] * img_height

    x_min = x_c_px - w_px/2

    y_min = y_c_px - h_px/2

    x_max = x_min + w_px

    y_max = y_min + h_px

    return (int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max)))    


import cv2
import numpy as np

def render_yolo_boxes(
    image: np.ndarray,
    yolo_boxes: list[tuple[int, tuple[float, float, float, float]]],
    class_names: dict[int, str]
) -> np.ndarray:
    """
    Renders bounding boxes and class labels onto an image.
    
    Args:
        image: BGR numpy image array.
        yolo_boxes: List of (class_id, (x_center, y_center, width, height)).
        class_names: Dict mapping class_id -> class_name (e.g. {0: "car", 1: "truck"}).
        
    Returns:
        Annotated BGR image array.
    """
    img_h,img_w = image.shape[:2]

    for class_id,bbox_norm in yolo_boxes:
        xmin,ymin,xmax,ymax = yolo_to_pixel_bbox(bbox_norm,img_w,img_h)
        cv2.rectangle(
            image, 
            pt1=(xmin, ymin), 
            pt2=(xmax, ymax), 
            color=(0, 255, 0), 
            thickness=2
        )

        cls_name = class_names.get(class_id, f"class_{class_id}")
        label_text = f"{cls_name} ({class_id})"

        cv2.putText(
            image,
            text=label_text,
            org=(xmin, max(ymin - 10, 15)), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA             
        )
    
    return image


if __name__ == "__main__":
    from pathlib import Path

    # 1. Image dimensions
    W, H = 1242, 375

    # 2. Class mapping
    names = {0: "car", 1: "truck", 2: "bus", 3: "pedestrian", 4: "cyclist"}
    
    # 3. Create a blank dark gray canvas (simulating a road frame)
    canvas = np.full((H, W, 3), 40, dtype=np.uint8)

    # 4. Our 3 converted YOLO boxes from 000001.txt: (class_id, (xc, yc, w, h))
    test_boxes = [
        (1, (0.494831, 0.460867, 0.024428, 0.087600)),  # Truck
        (0, (0.326667, 0.512880, 0.029130, 0.057547)),  # Car
        (4, (0.549750, 0.477173, 0.009968, 0.079947)),  # Cyclist
    ]

    # 5. Render
    annotated = render_yolo_boxes(canvas, test_boxes, names)

    # 6. Save image to outputs
    output_path = Path("outputs/test_render.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved visual verification image to: {output_path}")


def validate_yolo_line(
    line: str, 
    valid_class_ids: set[int] = {0, 1, 2, 3, 4}
) -> tuple[bool, str]:
    """
    Validates a single YOLO annotation line.
    
    Returns:
        (True, "Valid") if the line passes all integrity checks.
        (False, "Error reason") if any check fails.

    """

    line_list = line.split()
    
    try:
        class_id = int(line_list[0])
        xc, yc, w, h = float(line_list[1]), float(line_list[2]), float(line_list[3]), float(line_list[4])
    except ValueError:
        return False, "Failed to parse numbers"

    # 1. Class ID check
    if class_id not in valid_class_ids:
        return False, f"Class ID {class_id} is invalid"

    # 2. Positive dimensions check (width & height must be > 0)
    if w <= 0.0 or h <= 0.0:
        return False, "Box dimensions must be strictly positive"

    # 3. Normalized range check [0.0, 1.0]
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and w <= 1.0 and h <= 1.0):
        return False, "Coordinates or dimensions exceed [0.0, 1.0]"

    return True, "Valid"




import numpy as np
# You can import your compute_iou from src.training!
from src.training.metrics import compute_iou

def compute_iou_cost_matrix(
    tracks_boxes: list[tuple[float, float, float, float]],
    detections_boxes: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """
    Computes an (M, N) cost matrix where cost[i, j] = 1.0 - IoU(track_i, det_j).
    
    Args:
        tracks_boxes: List of M boxes (xmin, ymin, xmax, ymax) from existing tracks.
        detections_boxes: List of N boxes (xmin, ymin, xmax, ymax) from new detections.
        
    Returns:
        np.ndarray: Matrix of shape (M, N) containing float costs in [0.0, 1.0].
                    If either input list is empty, return an empty array with matching shape.
    """
    # Your implementation here
    rows,cols = len(tracks_boxes),len(detections_boxes)

    if not tracks_boxes or not detections_boxes:
        return np.zeros((rows,cols))

    cost_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            cost_matrix[i][j] = 1.0 - compute_iou(tracks_boxes[i],detections_boxes[j])

        
    return cost_matrix




# from scipy.optimize import linear_sum_assignment

def associate_detections_to_tracks(
    cost_matrix: np.ndarray,
    max_cost: float = 0.7
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
    # Your implementation here
    num_tracks, num_detections = cost_matrix.shape

    if num_tracks == 0 or num_detections == 0:
        return [], list(range(num_tracks)), list(range(num_detections))


    row_ind, col_ind = 1,2 #linear_sum_assignment(cost_matrix)
    matches = []

    for r,c in zip(row_ind,col_ind):
        if cost_matrix[r,c] <= max_cost:
            matches.append((r,c))

    matched_tracks = {r for r, c in matches}
    matched_detections = {c for r, c in matches}

    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    unmatched_detections = [j for j in range(num_detections) if j not in matched_detections]
    
    return (matches,unmatched_tracks,unmatched_detections)


class Track:
    """
    Represents a single tracked vehicle across time.
    """
    def __init__(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        class_id: int,
        confidence: float
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.state = "New"       # "New", "Tracked", "Lost", "Removed"
        self.hits = 1
        self.lost_frames = 0
        
        # Center point (cx, cy)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.history: list[tuple[float, float]] = [(cx, cy)]

    def update(self, bbox: tuple[float, float, float, float], confidence: float):
        """Called when this track is successfully matched to a detection."""
        # 1. Update self.bbox and self.confidence
        # 2. Reset self.lost_frames to 0
        # 3. Increment self.hits by 1
        # 4. If self.hits >= 2, set self.state to "Tracked"
        # 5. Compute new center (cx, cy) and append to self.history
        self.lost_frames = 0

        self.bbox = bbox
        self.confidence = confidence

        self.hits += 1

        if self.hits >= 2:
            self.state = "Tracked"

        new_cx = (bbox[0]+bbox[2]) / 2.0
        new_cy = (bbox[1]+bbox[3]) / 2.0

        self.history.append((new_cx,new_cy))


        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self, max_lost_frames: int = 30):
        """Called when this track is NOT matched in the current frame."""
        # 1. Increment self.lost_frames by 1
        # 2. If self.lost_frames > max_lost_frames, set self.state to "Removed"
        # 3. Otherwise, set self.state to "Lost"
        
        self.lost_frames += 1

        if self.state == "New":
            self.state = "Removed"
        elif self.lost_frames > max_lost_frames:
            self.state = "Removed"
        else:
            self.state = "Lost"       


Point = tuple[float, float]

def ccw(A: Point, B: Point, C: Point) -> float:
    """Computes orientation of triplet (A, B, C)."""
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def do_segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """
    Returns True if line segment AB intersects line segment CD.
    """
    # 1. Check if C and D are on opposite sides of AB
    # 2. Check if A and B are on opposite sides of CD
    # (Hint: If two numbers have opposite signs, their product is negative or (ccw1 > 0 != ccw2 > 0))ab = ccw(A,B,C)
    # True if one is > 0 and the other is NOT > 0 (opposite signs!)
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
        class_names: dict[int, str] = {0: "car", 1: "truck", 2: "bus", 3: "pedestrian", 4: "cyclist"}
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.class_names = class_names
        
        # Set to prevent counting the same vehicle multiple times
        self.counted_ids: set[int] = set()
        
        # Dictionary storing counts by direction and category
        self.counts = {
            "incoming": {name: 0 for name in class_names.values()},
            "outgoing": {name: 0 for name in class_names.values()}
        }

    def update(self, tracks: list[Track]) -> dict:
        """
        Checks active tracks against the tripwire line and updates counts.

        Returns:
            dict: Summary with {"incoming": ..., "outgoing": ..., "total": ...}
        """
        # Step 1: Loop through each track in tracks
        #
        # Step 2: Skip conditions:
        #   - If track.track_id is already in self.counted_ids: continue
        #   - If track.state != "Tracked": continue (ignore unconfirmed noise)
        #   - If len(track.history) < 2: continue (need 2 points to form a line)
        #
        # Step 3: Extract points:
        #   prev_pt = track.history[-2]
        #   curr_pt = track.history[-1]
        #
        # Step 4: Check if motion segment crossed the tripwire:
        #   if do_segments_intersect(prev_pt, curr_pt, self.line_start, self.line_end):
        #       - Add track.track_id to self.counted_ids
        #       - Determine direction:
        #           "incoming" if curr_pt[1] > prev_pt[1] else "outgoing"
        #       - Get class name: self.class_names.get(track.class_id, "car")
        #       - Increment self.counts[direction][cls_name] += 1
        #
        # Step 5: Compute total count and return summary dictionary
        for track in tracks:
            if track.track_id in self.counted_ids or track.state != "Tracked" or len(track.history) < 2:
                continue
            else:
                prev_pt = track.history[-2]
                curr_pt = track.history[-1]

                if do_segments_intersect(prev_pt,curr_pt,self.line_start,self.line_end):

                    self.counted_ids.add(track.track_id)

                    direction = "incoming" if curr_pt[1] > prev_pt[1] else "outgoing"

                    class_name = self.class_names.get(track.class_id,"car")

                    self.counts[direction][class_name] += 1

        total_in = sum(self.counts["incoming"].values())

        total_out = sum(self.counts["outgoing"].values())

        total = total_in + total_out

        return {"incoming":total_in,"outgoing":total_out,"total":total}


#Day 5

import cv2
import numpy as np

Point = tuple[float, float]

def draw_track_trail(
    image: np.ndarray,
    history: list[Point],
    color: tuple[int, int, int] = (0, 255, 255),  # Yellow trail in BGR
    thickness: int = 2
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
    # If there are fewer than 2 points, we can't draw a line connecting them!
    if len(history) < 2:
        return image

    # Loop through consecutive pairs of points (history[i - 1] to history[i])
    # and draw lines using cv2.line(image, pt1, pt2, color, thickness)
    for i in range(1, len(history)):
        pt1 = (int(history[i - 1][0]),int(history[i - 1][1]))
        pt2 = (int(history[i][0]),int(history[i][1]))
        cv2.line(image, pt1, pt2, color, thickness)

    return image
    
def render_pipeline_hud(
    frame: np.ndarray,
    tracks: list,
    tripwire_start: tuple[float, float],
    tripwire_end: tuple[float, float],
    counts: dict,
    fps: float,
    class_names: dict[int, str]
) -> np.ndarray:
    """
    Renders all visual telemetry:
    1. Motion trails for each track.
    2. Bounding boxes & ID labels.
    3. The tripwire line.
    4. Top telemetry dashboard banner with live FPS and traffic stats.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Step 1: Draw Tripwire Line (Red: (0, 0, 255))
    # Step 2: For each track, draw trail, bounding box, and ID pill
    # Step 3: Draw semi-transparent HUD banner at the top (y from 0 to 80)
    # Step 4: Put text for FPS and Counts

    p1 = (int(tripwire_start[0]), int(tripwire_start[1]))
    p2 = (int(tripwire_end[0]), int(tripwire_end[1]))

    cv2.line(canvas, p1, p2, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "TRIPWIRE", (p1[0] + 5, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    for t in tracks:
        if t.state == "Tracked":
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"ID:{t.track_id}", max((x1, y1 - 8),15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            draw_track_trail(canvas, t.history)
    
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    t1 = f"VISIONTRACK ITS  |  FPS: {fps:.1f}  |  ACTIVE: {len(tracks)}"
    t2 = f"INCOMING: {counts.get('incoming', 0)}  |  OUTGOING: {counts.get('outgoing', 0)}  |  TOTAL: {counts.get('total', 0)}"

    cv2.putText(canvas, t1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, t2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return canvas





import time
from pathlib import Path
from typing import Optional

class VideoPipeline:
    """
    End-to-end video processing engine integrating Detection, Tracking,
    Virtual Tripwire Counting, and Live HUD Rendering.
    """
    def __init__(
        self,
        detector,               # YOLODetector instance
        tracker,                # SimpleTracker instance
        counter,                # VehicleCounter instance
        class_names: dict[int, str]
    ):
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.class_names = class_names

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Runs the full pipeline on a video file and saves annotated MP4.
        
        Returns:
            dict: Summary metrics (total_frames, avg_fps, final_counts).
        """
        # Step 1: Open cv2.VideoCapture(input_path)
        # Step 2: Read width, height, fps
        # Step 3: Initialize cv2.VideoWriter if output_path is provided
        # Step 4: Loop frames:
        #         - Measure t_start
        #         - detector.predict -> format detections as [((xmin, ymin, xmax, ymax), cls_id, conf), ...]
        #         - tracker.update(detections) -> active tracks
        #         - counter.update(tracks) -> counts
        #         - render_pipeline_hud -> annotated frame
        #         - writer.write(annotated)
        #         - update FPS
        # Step 5: cap.release(), writer.release()
        # Instead of: with open(input_path) as file:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


        writer = None

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path,fourcc,fps,(width,height))


        
        frame_count = 0
        total_time = 0.0

        while cap.isOpened():
            if max_frames and frame_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break  # Reached the end of the video!

            frame_count += 1

            t_start = time.perf_counter()

            # 1. DETECT: Run detector on the frame
            det_result = self.detector.predict(frame, imgsz=640, return_annotated=False)
            # Format for tracker: list of (bbox, class_id, confidence)
            detections = [(d.bbox, d.class_id, d.confidence) for d in det_result.detections]

            # 2. TRACK: Feed detections into tracker
            tracks = self.tracker.update(detections)

            # 3. COUNT: Check for tripwire line crossings
            counts = self.counter.update(tracks)

            # Calculate instantaneous FPS
            t_elapsed = time.perf_counter() - t_start
            total_time += t_elapsed
            fps_current = 1.0 / max(t_elapsed, 1e-5)

            # 4. RENDER: Draw the HUD, boxes, and trails using render_pipeline_hud
            annotated = render_pipeline_hud(
                frame=frame,
                tracks=tracks,
                tripwire_start=self.counter.line_start,
                tripwire_end=self.counter.line_end,
                counts=counts,
                fps=fps_current,
                class_names=self.class_names
            )

            if writer:
                writer.write(annotated)


        cap.release()

        if writer:
            writer.release()

        avg_fps = frame_count / max(total_time, 1e-5)
        return {
            "frames_processed": frame_count,
            "avg_fps": avg_fps,
            "counts": counts
        }






#Day 6

from pydantic import BaseModel, Field
from typing import Literal

class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox

class DetectionResponse(BaseModel):
    success: bool = True
    total_detections: int
    class_counts: dict[str, int]
    inference_time_ms: float
    detections: list[DetectionItem]

class HealthResponse(BaseModel):
    status: str = "healthy"
    device: Literal["cpu","cuda","cuda:0"] = "cpu"
    model_loaded: bool = True
    uptime_seconds: float



import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import time

APP_START_TIME = time.time() 
app = FastAPI(title="VisionTrack API", version="1.0.0")



@app.get("/health",response_model=HealthResponse)
def health_check():
    uptime = time.time() - APP_START_TIME
    return HealthResponse(uptime_seconds=round(uptime,2))

@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    # Step 1 & 2: Read bytes and decode
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image format.")

    # Step 3: Run YOLO detection on the decoded BGR image
    result = detector.predict(img, imgsz=640, return_annotated=False)

    # Step 4: Convert detections into Pydantic models
    items = []
    for d in result.detections:
        x1, y1, x2, y2 = d.box
        items.append(
            DetectionItem(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                box=BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
            )
        )

    # Step 5: Return validated response
    return DetectionResponse(
        total_detections=len(items),
        class_counts=result.count_by_class(),
        inference_time_ms=round(result.inference_time_ms, 2),
        detections=items
    )


import streamlit as st
from src.detection import YOLODetector
from src.utils import get_device
from pathlib import Path

@st.cache_resource
def get_cached_detector():
    model_path = Path("models/yolo26n.pt") if Path("models/yolo26n.pt").exists() else Path("yolo26n.pt")
    return YOLODetector(model_path=model_path, device=get_device())

# 2. Sidebar controls
def setup_sidebar():
    """
    Renders sidebar controls and returns user parameter settings.
    Returns:
        dict: {"mode": ..., "conf_threshold": ..., "tripwire_ratio": ...}
    """
    st.sidebar.title("VisionTrack ITS")
    mode = st.sidebar.selectbox("Select Mode", ["Image Detection", "Video Tracking", "Live Demo"]) 
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    tripwire_ratio = st.sidebar.slider("Tripwire Position (%)", min_value=20, max_value=80, value=55, step=5) / 100
    

    return {"mode":mode,"conf_threshold":conf_threshold,"tripwire_ratio":tripwire_ratio}