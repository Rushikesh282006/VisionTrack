"""
VisionTrack - Object Detection Engine
=====================================
Encapsulates YOLO object detection with type hints, structured outputs,
and clean separation between model logic and presentation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import cv2
from ultralytics import YOLO

from src.utils import setup_logger, validate_image_path, Timer


@dataclass
class Detection:
    """Represents a single detected bounding box and classification."""
    box: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in pixel coords
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> Tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0


@dataclass
class DetectionResult:
    """Encapsulates all detections from a single frame/image alongside metadata."""
    detections: List[Detection]
    inference_time_ms: float
    orig_shape: Tuple[int, int]  # (height, width)
    annotated_image: Optional[np.ndarray] = None

    def count_by_class(self) -> dict[str, int]:
        """Returns a frequency count of detected object classes."""
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


class YOLODetector:
    """
    Modular wrapper around Ultralytics YOLO models for CPU and edge inference.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = "yolo26n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.logger = setup_logger("YOLODetector")
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        self.logger.info(f"Loading YOLO model from: {self.model_path.name} on {self.device}")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names

    def predict(
        self,
        image_source: Union[str, Path, np.ndarray],
        imgsz: int = 640,
        return_annotated: bool = True,
        conf_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Runs inference on an image path or loaded numpy array.

        Args:
            image_source: File path or loaded BGR image array.
            imgsz: Input resolution square dimension (e.g. 640, 416, 320).
            return_annotated: Whether to render bounding boxes onto the frame.
            conf_threshold: Optional per-request confidence threshold override.

        Returns:
            DetectionResult instance containing parsed detections and latency.
        """
        # Validate file path if string or Path provided
        if isinstance(image_source, (str, Path)):
            resolved_path = validate_image_path(image_source)
            source_input = str(resolved_path)
        else:
            source_input = image_source

        active_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        with Timer() as timer:
            results = self.model.predict(
                source=source_input,
                device=self.device,
                imgsz=imgsz,
                conf=active_conf,
                iou=self.iou_threshold,
                save=False,
                verbose=False,
            )

        inference_time_ms = timer.elapsed * 1000.0

        # Parse Ultralytics result
        first_result = results[0]
        orig_shape = first_result.orig_shape  # (height, width)
        detections: List[Detection] = []

        if first_result.boxes is not None:
            for box in first_result.boxes:
                coords = box.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
                cls_id = int(box.cls[0].cpu().item())
                confidence = float(box.conf[0].cpu().item())
                name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        box=(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                        confidence=confidence,
                        class_id=cls_id,
                        class_name=name,
                    )
                )

        annotated_image = None
        if return_annotated:
            annotated_image = first_result.plot()  # Ultralytics rendered BGR image

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            orig_shape=orig_shape,
            annotated_image=annotated_image,
        )

    @staticmethod
    def save_image(image: np.ndarray, output_path: Union[str, Path]) -> Path:
        """Saves an image array to disk, creating parent folders if needed."""
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_path), image)
        return target_path

