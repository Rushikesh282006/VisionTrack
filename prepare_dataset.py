"""
VisionTrack - Day 2 Dataset Preparation Pipeline
================================================
Orchestrates:
1. KITTI raw label parsing and coordinate conversion to normalized YOLO format.
2. Leak-free deterministic dataset partitioning (Train / Val / Test).
3. Data integrity auditing and edge-case validation.
4. Markdown summary generation and YOLO YAML configuration export.
"""

from pathlib import Path
import cv2
import yaml

from src.dataset.converter import KITTItoYOLOConverter
from src.dataset.split import DatasetSplitter
from src.dataset.validator import DatasetValidator
from src.utils import setup_logger, Timer

logger = setup_logger("PrepareDataset")


def main() -> None:
    logger.info("Starting VisionTrack Day 2 Dataset Preparation Pipeline...")

    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "dataset.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_labels_dir = project_root / config["paths"]["raw_labels_dir"]
    raw_images_dir = project_root / config["paths"]["raw_images_dir"]
    processed_dir = project_root / config["paths"]["processed_dir"]

    if not raw_labels_dir.exists():
        raise FileNotFoundError(f"Raw KITTI labels not found at: {raw_labels_dir}")

    # 1. Collect all label sample IDs
    label_files = sorted(raw_labels_dir.glob("*.txt"))
    sample_ids = [f.stem for f in label_files]
    total_samples = len(sample_ids)
    logger.info(f"Discovered {total_samples} raw KITTI annotation files.")

    # 2. Partition dataset into Train / Val / Test splits
    splitter = DatasetSplitter(config_path=config_path)
    splits = splitter.create_splits(sample_ids)
    splitter.save_split_manifests(splits, output_dir=processed_dir)

    # 3. Create target processed directories
    for split_name in ["train", "val", "test"]:
        (processed_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)

    # 4. Convert all annotations
    converter = KITTItoYOLOConverter(config_path=config_path)
    split_lookup = {}
    for split_name, ids in splits.items():
        for sid in ids:
            split_lookup[sid] = split_name

    logger.info("Converting annotations to YOLO format across splits...")
    with Timer("ConversionPipeline") as timer:
        converted_count = 0
        for sid in sample_ids:
            label_file = raw_labels_dir / f"{sid}.txt"
            img_file = raw_images_dir / f"{sid}.png"

            # Determine exact image dimensions
            if img_file.exists():
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                else:
                    w, h = 1242, 375
            else:
                w, h = 1242, 375

            yolo_boxes = converter.convert_label_file(label_file, img_width=w, img_height=h)

            # Target destination
            target_split = split_lookup[sid]
            target_label_file = processed_dir / "labels" / target_split / f"{sid}.txt"

            with open(target_label_file, "w", encoding="utf-8") as f_out:
                for box in yolo_boxes:
                    f_out.write(box.to_yolo_line() + "\n")

            converted_count += 1
            if converted_count % 1500 == 0 or converted_count == total_samples:
                logger.info(f"Processed {converted_count}/{total_samples} label files...")

    logger.info(f"Conversion completed in {timer.elapsed:.2f} seconds.")

    # 5. Run Quality Audit across all splits
    validator = DatasetValidator(config_path=config_path)
    all_processed_labels = processed_dir / "all_labels"
    all_processed_labels.mkdir(exist_ok=True)

    # Temporary audit of converted files
    logger.info("Auditing converted annotations...")
    total_boxes = 0
    class_dist = {}
    out_of_bounds = 0
    zero_area = 0
    invalid_classes = 0
    empty_files = []

    for split_name in ["train", "val", "test"]:
        split_dir = processed_dir / "labels" / split_name
        split_report = validator.audit_directory(split_dir)
        total_boxes += split_report.total_boxes
        out_of_bounds += split_report.out_of_bounds_errors
        zero_area += split_report.zero_area_errors
        invalid_classes += split_report.invalid_class_errors
        empty_files.extend(split_report.empty_label_files)
        for cid, cnt in split_report.class_distribution.items():
            class_dist[cid] = class_dist.get(cid, 0) + cnt

    from src.dataset.validator import ValidationReport
    combined_report = ValidationReport(
        total_samples=total_samples,
        total_boxes=total_boxes,
        class_distribution=class_dist,
        empty_label_files=empty_files,
        out_of_bounds_errors=out_of_bounds,
        zero_area_errors=zero_area,
        invalid_class_errors=invalid_classes,
    )

    split_counts = {k: len(v) for k, v in splits.items()}
    summary_path = project_root / "outputs" / "dataset_summary.md"
    validator.generate_markdown_summary(
        report=combined_report,
        output_path=summary_path,
        split_counts=split_counts, 
    )

    # 6. Generate Ultralytics fine-tuning configuration file: traffic_kitti.yaml
    yolo_yaml_path = project_root / "configs" / "traffic_kitti.yaml"
    yolo_config_data = {
        "path": "data/processed/kitti_yolo",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config["names"],
    }
    with open(yolo_yaml_path, "w", encoding="utf-8") as f_yaml:
        yaml.dump(yolo_config_data, f_yaml, sort_keys=False)

    logger.info(f"Exported YOLO fine-tuning config: {yolo_yaml_path}")
    print("\n" + "=" * 50)
    print("      DAY 2 DATASET PIPELINE COMPLETED!")
    print("=" * 50)
    print(f"Total files converted:   {total_samples:,}")
    print(f"Total bounding boxes:    {total_boxes:,}")
    print(f"Class distribution:      {class_dist}")
    print(f"Data audit status:       {'100% CLEAN' if combined_report.is_clean else 'ISSUES FOUND'}")
    print(f"Summary report:          {summary_path.relative_to(project_root)}")
    print(f"YOLO training config:    {yolo_yaml_path.relative_to(project_root)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
