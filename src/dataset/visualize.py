"""
VisionTrack - Annotation Visualizer
===================================
Inverse coordinate transforms and OpenCV rendering developed during Day 2.
"""

from pathlib import Path
import cv2
import numpy as np


def yolo_to_pixel_bbox(
    yolo_box: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
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
    x_c, y_c, w, h = yolo_box

    x_c_px = x_c * img_width
    y_c_px = y_c * img_height
    w_px = w * img_width
    h_px = h * img_height

    x_min = x_c_px - w_px / 2.0
    y_min = y_c_px - h_px / 2.0
    x_max = x_min + w_px
    y_max = y_min + h_px

    # Clamp to boundaries
    xmin = max(0, min(img_width - 1, int(round(x_min))))
    ymin = max(0, min(img_height - 1, int(round(y_min))))
    xmax = max(0, min(img_width - 1, int(round(x_max))))
    ymax = max(0, min(img_height - 1, int(round(y_max))))

    return (xmin, ymin, xmax, ymax)


def render_yolo_boxes(
    image: np.ndarray,
    yolo_boxes: list[tuple[int, tuple[float, float, float, float]]],
    class_names: dict[int, str],
) -> np.ndarray:
    """
    Renders bounding boxes and class labels onto an image using OpenCV.

    Args:
        image: BGR numpy image array.
        yolo_boxes: List of (class_id, (x_center, y_center, width, height)).
        class_names: Dict mapping class_id -> class_name (e.g. {0: "car", 1: "truck"}).

    Returns:
        Annotated BGR image array.
    """
    img_h, img_w = image.shape[:2]

    for class_id, bbox_norm in yolo_boxes:
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(bbox_norm, img_w, img_h)

        # Draw bounding rectangle
        cv2.rectangle(
            image,
            pt1=(xmin, ymin),
            pt2=(xmax, ymax),
            color=(0, 255, 0),
            thickness=2,
        )

        cls_name = class_names.get(class_id, f"class_{class_id}")
        label_text = f"{cls_name} ({class_id})"

        # Draw label text with anti-aliasing
        cv2.putText(
            image,
            text=label_text,
            org=(xmin, max(ymin - 10, 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image
