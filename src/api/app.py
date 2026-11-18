"""
VisionTrack - REST API Application
==================================
Production FastAPI web service for vehicle detection, tracking, and traffic analytics.
"""

import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.detection import YOLODetector
from src.tracking.tracker import SimpleTracker
from src.tracking.counter import VehicleCounter
from src.pipeline import VideoPipeline
from src.utils import get_device, setup_logger
from src.api.schemas import (
    DetectionResponse,
    DetectionItem,
    BoundingBox,
    HealthResponse,
    VideoTrackResponse,
)

logger = setup_logger("VisionTrackAPI")

CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "pedestrian",
    4: "cyclist",
}


def find_weights() -> Path:
    candidates = [
        Path("models/yolo26n.pt"),
        Path("yolo26n.pt"),
        Path("weights/best.pt"),
        Path("models/yolov8n.pt"),
        Path("yolov8n.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("yolov8n.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and caches model weights once during startup."""
    app.state.start_time = time.time()
    app.state.device = get_device()
    model_path = find_weights()
    logger.info(f"Loading YOLO weights from {model_path} on {app.state.device}...")

    try:
        app.state.detector = YOLODetector(model_path=model_path, device=app.state.device)
        app.state.model_loaded = True
        logger.info("YOLO model successfully loaded and ready for inference.")
    except Exception as e:
        logger.warning(f"Could not load detector at startup: {e}")
        app.state.detector = None
        app.state.model_loaded = False

    yield
    logger.info("VisionTrack API shutting down.")


app = FastAPI(
    title="VisionTrack Intelligent Traffic Monitoring API",
    description="Real-time multi-class vehicle detection, MOT tracking, and virtual tripwire analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients (Streamlit, React, mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root status and documentation link."""
    return {
        "service": "VisionTrack ITS API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """System diagnostic, device detection, and uptime reporting."""
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    device = getattr(app.state, "device", "cpu")
    loaded = getattr(app.state, "model_loaded", False)

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        device=str(device),
        model_loaded=loaded,
        uptime_seconds=round(uptime, 2),
    )


@app.post("/detect/image", response_model=DetectionResponse, tags=["Inference"])
async def detect_image(
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    conf_threshold: float = Form(0.25, description="Confidence score threshold"),
):
    """
    Performs vehicle detection on an uploaded image and returns structured JSON predictions.
    """
    if not getattr(app.state, "model_loaded", False) or app.state.detector is None:
        raise HTTPException(status_code=503, detail="YOLO detector model is not initialized.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid or readable image.")

    result = app.state.detector.predict(img, imgsz=640, conf_threshold=conf_threshold, return_annotated=False)

    items = []
    for d in result.detections:
        box_coords = getattr(d, "box", getattr(d, "bbox", (0.0, 0.0, 0.0, 0.0)))
        x1, y1, x2, y2 = box_coords
        items.append(
            DetectionItem(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                box=BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2),
            )
        )

    return DetectionResponse(
        success=True,
        total_detections=len(items),
        class_counts=result.count_by_class(),
        inference_time_ms=round(result.inference_time_ms, 2),
        detections=items,
    )


@app.post("/track/video", response_model=VideoTrackResponse, tags=["Inference"])
async def track_video(
    file: UploadFile = File(..., description="MP4, AVI, or MOV video file"),
    tripwire_ratio: float = Form(0.55, description="Tripwire Y-position ratio from top (0.0 to 1.0)"),
    max_frames: Optional[int] = Form(None, description="Max frames to process (useful for previews)"),
):
    """
    Uploads a video, runs detection, multi-object tracking, and virtual tripwire counting.
    """
    if not getattr(app.state, "model_loaded", False) or app.state.detector is None:
        raise HTTPException(status_code=503, detail="YOLO detector model is not initialized.")

    # Save uploaded bytes to a temporary video file
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    tmp_out_path = tempfile.mktemp(suffix=".mp4")

    try:
        cap = cv2.VideoCapture(tmp_in_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot read or decode uploaded video file.")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()

        trip_y = float(h * tripwire_ratio)
        tracker = SimpleTracker(max_cost=0.7, max_lost_frames=30)
        counter = VehicleCounter(line_start=(0.0, trip_y), line_end=(float(w), trip_y), class_names=CLASS_NAMES)
        pipeline = VideoPipeline(app.state.detector, tracker, counter, class_names=CLASS_NAMES)

        metrics = pipeline.process_video(
            input_path=tmp_in_path,
            output_path=tmp_out_path,
            max_frames=max_frames,
        )

        counts = metrics["counts"]
        return VideoTrackResponse(
            success=True,
            frames_processed=metrics["frames_processed"],
            avg_fps=round(metrics["avg_fps"], 2),
            incoming_count=counts.get("incoming", 0),
            outgoing_count=counts.get("outgoing", 0),
            total_count=counts.get("total", 0),
            details=counts,
        )

    finally:
        # Cleanup temporary files
        try:
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(tmp_out_path).unlink(missing_ok=True)
        except Exception:
            pass
