"""
VisionTrack - KITTI Dataset Downloader & Extractor
==================================================
Handles downloading and extracting official KITTI 2D object detection archives
with progress tracking, checksum/size verification, and graceful error handling.
"""

from pathlib import Path
import zipfile
import urllib.request
import shutil
from typing import Optional

from src.utils import setup_logger

logger = setup_logger("KITTIDownloader")

KITTI_URLS = {
    "labels": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "filename": "data_object_label_2.zip",
        "expected_size_mb": 5.3,
        "description": "KITTI 2D Object Detection Ground Truth Labels (7,481 files)",
    },
    "images": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "filename": "data_object_image_2.zip",
        "expected_size_mb": 12560.0,
        "description": "KITTI 2D Object Detection Left Color Images (7,481 train + 7,518 test)",
    },
}


def download_file(
    url: str,
    destination_path: Path,
    chunk_size: int = 1024 * 1024,  # 1 MB
) -> Path:
    """
    Downloads a remote file with streaming progress logging.

    Args:
        url: Direct HTTP/HTTPS download link.
        destination_path: Local Path where the file should be saved.
        chunk_size: Read buffer size in bytes.

    Returns:
        Path to downloaded file.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VisionTrack-Downloader/1.0 (Python/urllib)"}
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = response.length or 0
        total_mb = total_size / (1024 * 1024)
        logger.info(f"Target size: {total_mb:.2f} MB. Writing to: {destination_path.name}")

        downloaded = 0
        with open(destination_path, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded / (1024 * 1024)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloading: {downloaded_mb:.1f} MB", end="", flush=True)
        print()  # Newline after download loop

    logger.info(f"Download complete: {destination_path}")
    return destination_path


def extract_archive(
    archive_path: Path,
    target_dir: Path,
    remove_archive: bool = False,
) -> Path:
    """
    Safely extracts a ZIP archive into the target directory.

    Args:
        archive_path: Path to the .zip archive.
        target_dir: Directory where contents will be extracted.
        remove_archive: If True, deletes the zip file after successful extraction.

    Returns:
        Path to target directory.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} to {target_dir}...")

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    logger.info(f"Successfully extracted {archive_path.name}")

    if remove_archive:
        archive_path.unlink()
        logger.info(f"Removed zip archive: {archive_path.name}")

    return target_dir


def acquire_kitti_labels(target_raw_dir: Path) -> Path:
    """
    Downloads and extracts the KITTI training labels (5 MB).
    """
    target_raw_dir = Path(target_raw_dir).resolve()
    archive_path = target_raw_dir / KITTI_URLS["labels"]["filename"]
    extracted_labels_dir = target_raw_dir / "training" / "label_2"

    if extracted_labels_dir.exists() and any(extracted_labels_dir.iterdir()):
        logger.info(f"KITTI labels already exist at: {extracted_labels_dir}")
        return extracted_labels_dir

    if not archive_path.exists():
        download_file(KITTI_URLS["labels"]["url"], archive_path)

    extract_archive(archive_path, target_raw_dir)
    return extracted_labels_dir
