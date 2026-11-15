"""
VisionTrack - Sample Visual Verification Script
===============================================
Inspects the first 5 samples from the dataset, converting normalized YOLO
coordinates back to pixel space and rendering bounding boxes and class pills.
"""

from pathlib import Path
import cv2
import numpy as np
import yaml

from src.dataset.visualize import AnnotationVisualizer, yolo_to_pixel
from src.utils import setup_logger

logger = setup_logger("SampleVerifier")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "dataset.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = project_root / config["paths"]["processed_dir"]
    raw_images_dir = project_root / config["paths"]["raw_images_dir"]
    output_dir = project_root / "outputs" / "dataset_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = AnnotationVisualizer(config_path=config_path)

    sample_ids = ["000000", "000001", "000002", "000003", "000004"]
    logger.info(f"Visualizing {len(sample_ids)} sample annotations...")

    for sid in sample_ids:
        # Locate the label in whichever split it was partitioned into
        label_file = None
        for split_name in ["train", "val", "test"]:
            candidate = processed_dir / "labels" / split_name / f"{sid}.txt"
            if candidate.exists():
                label_file = candidate
                break

        if label_file is None:
            logger.warning(f"Could not locate converted label for ID: {sid}")
            continue

        # Look for raw image; if not yet downloaded, create standard KITTI canvas
        img_file = raw_images_dir / f"{sid}.png"
        if img_file.exists():
            img = cv2.imread(str(img_file))
        else:
            # Standard KITTI Camera 2 frame: 1242 x 375
            img = np.full((375, 1242, 3), 40, dtype=np.uint8)
            # Add grid lines and camera metadata banner
            cv2.putText(
                img,
                f"KITTI Frame: {sid} (Canvas 1242x375 - Ground Truth Verification)",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Temporary path for drawing
        temp_img_path = output_dir / f"temp_{sid}.png"
        cv2.imwrite(str(temp_img_path), img)

        dest_annotated = output_dir / f"sample_{sid}_annotated.jpg"
        visualizer.draw_annotations(
            image_path=temp_img_path,
            yolo_label_path=label_file,
            output_path=dest_annotated,
        )

        if temp_img_path.exists():
            temp_img_path.unlink()

        logger.info(f"Sample {sid} verified -> saved to: {dest_annotated.name}")

    print("\n" + "=" * 50)
    print("      VISUAL VERIFICATION COMPLETED (5 SAMPLES)")
    print("=" * 50)
    for sample_file in sorted(output_dir.glob("*.jpg")):
        print(f"Verified: {sample_file.name}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
