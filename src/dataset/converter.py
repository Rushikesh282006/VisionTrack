"""
VisionTrack - KITTI to YOLO Coordinate Converter
=================================================
Core annotation conversion functions developed during Day 2.
"""

from pathlib import Path
from typing import Optional


def kitti_bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Optional[tuple[float, float, float, float]]:
    """
    Converts KITTI pixel bounding box to normalized YOLO format.

    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.
        img_width: Physical width of the image in pixels.
        img_height: Physical height of the image in pixels.

    Returns:
        (x_center, y_center, width, height) normalized to [0.0, 1.0], or None if invalid.
    """
    x_min, y_min, x_max, y_max = bbox

    # Guard clause: check for inverted or zero-area boxes
    if x_min >= x_max or y_min >= y_max:
        return None

    # Boundary clamping
    x_min = max(0.0, min(float(img_width), x_min))
    x_max = max(0.0, min(float(img_width), x_max))
    y_min = max(0.0, min(float(img_height), y_min))
    y_max = max(0.0, min(float(img_height), y_max))

    width = x_max - x_min
    height = y_max - y_min

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    x_c_norm = x_center / img_width
    y_c_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return (x_c_norm, y_c_norm, width_norm, height_norm)


def parse_kitti_line(
    line: str,
    class_mapping: dict[str, int],
) -> Optional[tuple[int, tuple[float, float, float, float]]]:
    """
    Parses a raw line from a KITTI label file and extracts class and bounding box.

    Args:
        line: Raw 15-column KITTI line string.
        class_mapping: Dict mapping string class name to integer ID.

    Returns:
        (class_id, (xmin, ymin, xmax, ymax)) or None if line is invalid or ignored.
    """
    line_list = line.strip().split()

    if len(line_list) < 15:
        return None
    elif line_list[0] not in class_mapping:
        return None
    else:
        class_id = class_mapping[line_list[0]]
        return (
            class_id,
            (
                float(line_list[4]),
                float(line_list[5]),
                float(line_list[6]),
                float(line_list[7]),
            ),
        )


def convert_kitti_file(
    file_path: str,
    img_width: int,
    img_height: int,
    class_mapping: dict[str, int],
) -> list[str]:
    """
    Reads a KITTI label file and converts all valid objects into YOLO lines.

    Args:
        file_path: Path to raw KITTI label file.
        img_width: Physical image width in pixels.
        img_height: Physical image height in pixels.
        class_mapping: Dict mapping class names to IDs.

    Returns:
        List of formatted YOLO strings: 'class_id xc yc w h'.
    """
    yolo_lines: list[str] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parsed = parse_kitti_line(line, class_mapping)
            if parsed is not None:
                class_id, bbox = parsed
                norm_box = kitti_bbox_to_yolo(bbox, img_width, img_height)
                if norm_box is not None:
                    yolo_lines.append(
                        f"{class_id} {norm_box[0]:.6f} {norm_box[1]:.6f} {norm_box[2]:.6f} {norm_box[3]:.6f}"
                    )

    return yolo_lines
