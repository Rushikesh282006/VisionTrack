"""
VisionTrack - Dataset Quality Validator
=======================================
Integrity auditing functions for converted YOLO annotations developed during Day 2.
"""

from pathlib import Path


def validate_yolo_line(
    line: str,
    valid_class_ids: set[int] = {0, 1, 2, 3, 4},
) -> tuple[bool, str]:
    """
    Validates a single YOLO annotation line against structural and numerical invariants.

    Returns:
        (True, "Valid") if the line passes all integrity checks.
        (False, "Error reason") if any check fails.
    """
    line_list = line.strip().split()

    # 1. Field count check
    if len(line_list) != 5:
        return False, f"Expected 5 fields, got {len(line_list)}"

    # 2. Numerical parse check
    try:
        class_id = int(line_list[0])
        xc, yc, w, h = (
            float(line_list[1]),
            float(line_list[2]),
            float(line_list[3]),
            float(line_list[4]),
        )
    except ValueError:
        return False, "Failed to parse numbers in annotation line"

    # 3. Class ID check
    if class_id not in valid_class_ids:
        return False, f"Class ID {class_id} is invalid (allowed: {valid_class_ids})"

    # 4. Positive dimensions check (width & height must be strictly positive)
    if w <= 0.0 or h <= 0.0:
        return False, f"Box dimensions must be strictly positive (got w={w}, h={h})"

    # 5. Normalized range check [0.0, 1.0] with small float epsilon
    eps = 1e-4
    if not (0.0 - eps <= xc <= 1.0 + eps and 0.0 - eps <= yc <= 1.0 + eps):
        return False, f"Center coordinates out of bounds: ({xc}, {yc})"

    if not (0.0 < w <= 1.0 + eps and 0.0 < h <= 1.0 + eps):
        return False, f"Box dimensions out of bounds: ({w}, {h})"

    # 6. Boundary spill check
    xmin = xc - w / 2.0
    xmax = xc + w / 2.0
    ymin = yc - h / 2.0
    ymax = yc + h / 2.0
    if xmin < -eps or xmax > 1.0 + eps or ymin < -eps or ymax > 1.0 + eps:
        return False, f"Box edges spill outside frame: x=[{xmin:.4f}, {xmax:.4f}], y=[{ymin:.4f}, {ymax:.4f}]"

    return True, "Valid"
