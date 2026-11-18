"""
VisionTrack API - Pydantic Request & Response Schemas
=====================================================
Defines strict data contracts and validation schemas for the REST API.
"""

from typing import Literal
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized or pixel bounding box coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class DetectionItem(BaseModel):
    """A single localized object prediction."""
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class DetectionResponse(BaseModel):
    """Response payload for image detection endpoint."""
    success: bool = True
    total_detections: int
    class_counts: dict[str, int]
    inference_time_ms: float
    detections: list[DetectionItem]


class HealthResponse(BaseModel):
    """System diagnostic and health status."""
    status: str = "healthy"
    device: str = "cpu"
    model_loaded: bool = True
    uptime_seconds: float


class VideoTrackResponse(BaseModel):
    """Response payload for video tracking and analytics endpoint."""
    success: bool = True
    frames_processed: int
    avg_fps: float
    incoming_count: int
    outgoing_count: int
    total_count: int
    details: dict
