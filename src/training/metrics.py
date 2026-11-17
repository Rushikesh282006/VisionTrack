def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """
    Computes Intersection over Union (IoU) of two boxes in (xmin, ymin, xmax, ymax) format.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2

    x1_int = max(b1x1, b2x1)
    y1_int = max(b1y1, b2y1)
    x2_int = min(b1x2, b2x2)
    y2_int = min(b1y2, b2y2)

    inter_w = max(0.0, x2_int - x1_int)
    inter_h = max(0.0, y2_int - y1_int)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, b1x2 - b1x1) * max(0.0, b1y2 - b1y1)
    box2_area = max(0.0, b2x2 - b2x1) * max(0.0, b2y2 - b2y1)

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area