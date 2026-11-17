# """
# VisionTrack - Main Detection Pipeline
# =====================================
# Demonstrates baseline object detection using our modular YOLODetector.
# """

# from pathlib import Path
# from src.detection import YOLODetector
# from src.utils import setup_logger, get_device

# logger = setup_logger("VisionTrackMain")


# def main() -> None:
#     # 1. Select execution device
#     device = get_device(prefer_cuda=False)

#     # 2. Paths
#     project_root = Path(__file__).resolve().parent
#     model_path = project_root / "yolo26n.pt"
#     image_path = project_root / "data" / "samples" / "bus.jpg"
#     output_path = project_root / "outputs" / "bus_detected.jpg"

#     # 3. Initialize modular detector
#     detector = YOLODetector(
#         model_path=model_path,
#         device=device,
#         conf_threshold=0.25,
#     )

#     # 4. Warm-up run (to eliminate cold-start graph initialization overhead)
#     logger.info("Executing warm-up run...")
#     _ = detector.predict(image_path, imgsz=640, return_annotated=False)

#     # 5. Production inference run
#     logger.info(f"Running inference on: {image_path.name}")
#     result = detector.predict(image_path, imgsz=640, return_annotated=True)

#     # 6. Display detection results
#     print("\n" + "=" * 45)
#     print("           DETECTION RESULTS")
#     print("=" * 45)
#     high_conf_detections = [det for det in result.detections if det.confidence > 0.85]
    
#     for det in high_conf_detections:
#         print(
#             f"Object: {det.class_name:<12} | "
#             f"Confidence: {det.confidence:.2f} | "
#             f"Box: ({det.box[0]:.1f}, {det.box[1]:.1f}, {det.box[2]:.1f}, {det.box[3]:.1f})"
#         )

#     print("-" * 45)
#     print(f"Total objects detected: {len(result.detections)}")
#     print(f"Class breakdown:        {result.count_by_class()}")
#     print(f"Warm inference latency: {result.inference_time_ms:.1f} ms ({result.inference_time_ms / 1000.0:.3f} s)")
#     print("=" * 45)

#     # 7. Save annotated output image
#     if result.annotated_image is not None:
#         saved_file = detector.save_image(result.annotated_image, output_path)
#         logger.info(f"Annotated result saved to: {saved_file}")

#     print("\nDetection completed successfully!")


# if __name__ == "__main__":
#     main()





def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float]
) -> float:
    """
    Computes Intersection over Union (IoU) of two boxes in (xmin, ymin, xmax, ymax) format.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    # Your implementation here
    
    b1x1,b1y1,b1x2,b1y2 = box1

    b2x1,b2y1,b2x2,b2y2 = box2
    


    x1_int = max(b1x1, b2x1)
    y1_int = max(b1y1, b2y1)

    x2_int = min(b1x2, b2x2)
    y2_int = min(b1y2, b2y2) 

    inter_w = max(0.0, x2_int - x1_int)
    inter_h = max(0.0, y2_int - y1_int)
    inter_area = inter_w * inter_h

    box1_area = (b1x2-b1x1) * (b1y2-b1y1)
    box2_area = (b2x2-b2x1) * (b2y2-b2y1)

    union_area = box1_area + box2_area - inter_area


    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area