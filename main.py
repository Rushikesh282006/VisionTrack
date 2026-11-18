"""
VisionTrack - Unified CLI Platform
==================================
Command-line interface for Real-Time Vehicle Detection, Multi-Object Tracking,
and Virtual Tripwire Traffic Analytics.

Usage:
    # 1. Single-Image Detection Mode:
    python main.py --mode image --source data/samples/bus.jpg

    # 2. Video Tracking & Analytics Mode:
    python main.py --mode video --source path/to/traffic.mp4

    # 3. Automated End-to-End Demo Mode:
    python main.py --mode demo
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from src.detection import YOLODetector
from src.tracking.tracker import SimpleTracker
from src.tracking.counter import VehicleCounter
from src.pipeline import VideoPipeline
from src.utils import setup_logger, get_device, validate_image_path

logger = setup_logger("VisionTrackCLI")

CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "pedestrian",
    4: "cyclist",
}


def resolve_model_path(custom_path: str = None) -> Path:
    """Finds the best available model weights."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    candidates = [
        Path("models/yolo26n.pt"),
        Path("yolo26n.pt"),
        Path("weights/best.pt"),
        Path("models/yolov8n.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Default to first candidate
    return candidates[0]


def run_image(args: argparse.Namespace) -> None:
    """Runs single-frame object detection."""
    source_path = Path(args.source or "data/samples/bus.jpg")
    output_path = Path(args.output or "outputs/bus_detected.jpg")
    model_path = resolve_model_path(args.model)
    device = args.device or get_device()

    logger.info(f"Mode: IMAGE DETECTION | Model: {model_path} | Device: {device}")
    logger.info(f"Input: {source_path}")

    validated_src = validate_image_path(source_path)
    detector = YOLODetector(model_path=model_path, device=device, conf_threshold=args.conf)

    # Warm-up run
    _ = detector.predict(validated_src, imgsz=640, return_annotated=False)

    # Inference
    result = detector.predict(validated_src, imgsz=640, return_annotated=True)

    print("\n" + "=" * 55)
    print("                DETECTION RESULTS")
    print("=" * 55)
    for det in result.detections:
        print(
            f"Object: {det.class_name:<12} | "
            f"Confidence: {det.confidence:.2f} | "
            f"Box: ({det.box[0]:.1f}, {det.box[1]:.1f}, {det.box[2]:.1f}, {det.box[3]:.1f})"
        )
    print("-" * 55)
    print(f"Total objects detected: {len(result.detections)}")
    print(f"Class breakdown:        {result.count_by_class()}")
    print(f"Inference latency:      {result.inference_time_ms:.1f} ms ({result.inference_time_ms / 1000.0:.3f} s)")
    print("=" * 55)

    if result.annotated_image is not None:
        saved_file = detector.save_image(result.annotated_image, output_path)
        logger.info(f"Annotated result saved to: {saved_file}")


def run_video(args: argparse.Namespace) -> None:
    """Runs video tracking, tripwire counting, and HUD rendering."""
    if not args.source:
        logger.error("--source is required for video mode!")
        sys.exit(1)

    input_path = Path(args.source)
    if not input_path.exists():
        logger.error(f"Input video file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output or "outputs/traffic_tracked.mp4")
    model_path = resolve_model_path(args.model)
    device = args.device or get_device()

    # Read dimensions to set up tripwire
    cap = cv2.VideoCapture(str(input_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    trip_y = float(args.tripwire_y) if args.tripwire_y is not None else h * 0.55
    line_start = (0.0, trip_y)
    line_end = (float(w), trip_y)

    logger.info(f"Mode: VIDEO TRACKING | Source: {input_path} | Output: {output_path}")
    logger.info(f"Tripwire: {line_start} -> {line_end}")

    detector = YOLODetector(model_path=model_path, device=device, conf_threshold=args.conf)
    tracker = SimpleTracker(max_cost=0.7, max_lost_frames=30)
    counter = VehicleCounter(line_start=line_start, line_end=line_end, class_names=CLASS_NAMES)
    pipeline = VideoPipeline(detector, tracker, counter, class_names=CLASS_NAMES)

    metrics = pipeline.process_video(
        input_path=str(input_path),
        output_path=str(output_path),
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 55)
    print("             VIDEO TRACKING SUMMARY")
    print("=" * 55)
    print(f"Frames processed: {metrics['frames_processed']}")
    print(f"Average FPS:      {metrics['avg_fps']:.1f}")
    print(f"Incoming counts:  {metrics['counts']['incoming']}")
    print(f"Outgoing counts:  {metrics['counts']['outgoing']}")
    print(f"Total counted:    {metrics['counts']['total']}")
    print("=" * 55)
    logger.info(f"Processed video exported to: {output_path}")


def run_demo(args: argparse.Namespace) -> None:
    """Generates a realistic multi-frame traffic simulation and runs full pipeline."""
    demo_dir = Path("outputs/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_input = demo_dir / "demo_traffic.mp4"
    demo_output = Path(args.output or "outputs/traffic_tracked.mp4")

    logger.info("Generating synthetic 60-frame traffic sequence...")
    w, h, fps = 640, 360, 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(demo_input), fourcc, fps, (w, h))

    # Simulate roadway with moving vehicle boxes
    for frame_idx in range(60):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)  # Dark asphalt

        # Road markings
        cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 1)
        dash_x = (frame_idx * 5) % 40
        for x in range(dash_x, w, 40):
            cv2.line(frame, (x, h // 2), (x + 20, h // 2), (255, 255, 255), 2)

        # Vehicle 1: Car moving DOWNWARDS across tripwire (y: 100 -> 280)
        v1_y = int(100 + frame_idx * 3.0)
        cv2.rectangle(frame, (180, v1_y), (260, v1_y + 40), (0, 165, 255), -1)  # Orange car
        cv2.putText(frame, "CAR", (190, v1_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Vehicle 2: Truck moving UPWARDS across tripwire (y: 260 -> 80)
        v2_y = int(260 - frame_idx * 3.0)
        cv2.rectangle(frame, (380, v2_y), (480, v2_y + 60), (200, 200, 0), -1)  # Cyan truck
        cv2.putText(frame, "TRUCK", (395, v2_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic video generated at: {demo_input}")

    # Set up pipeline for demo
    args.source = str(demo_input)
    args.output = str(demo_output)
    args.tripwire_y = h * 0.5
    run_video(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionTrack: Real-Time Traffic Detection, Tracking & Counting Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["image", "video", "demo"],
        default="image",
        help="Pipeline execution mode",
    )
    parser.add_argument("--source", type=str, default=None, help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None, help="Path for saving output")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model weights (.pt)")
    parser.add_argument("--device", type=str, default=None, help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tripwire-y", type=float, default=None, help="Y-coordinate for tripwire line")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process in video mode")

    args = parser.parse_args()

    if args.mode == "image":
        run_image(args)
    elif args.mode == "video":
        run_video(args)
    elif args.mode == "demo":
        run_demo(args)


if __name__ == "__main__":
    main()
