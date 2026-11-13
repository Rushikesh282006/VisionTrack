"""
VisionTrack - Utility Functions and Helpers
===========================================
Provides logging configuration, performance timing, device management,
and filesystem validation utilities.
"""

from pathlib import Path
import logging
import time
from typing import Union, Callable, Any
import functools
import torch


def setup_logger(name: str = "VisionTrack", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized console logger.

    Args:
        name: Name of the logger instance.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_device(prefer_cuda: bool = False) -> str:
    """
    Determines and validates the execution device (CPU or CUDA).

    Args:
        prefer_cuda: Whether to use CUDA if available. Defaults to False
                     to honor local CPU development constraints.

    Returns:
        Device string ('cpu' or 'cuda').
    """
    logger = setup_logger("DeviceManager")
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA accelerator detected: {device_name}")
    else:
        device = "cpu"
        logger.info("Using device: CPU")
    return device


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validates that the given image path exists and has a supported format.

    Args:
        path: String or Path object pointing to an image file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a supported image format.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_path = Path(path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Path is not a regular file: {image_path}")
    if image_path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file extension '{image_path.suffix}'. "
            f"Supported extensions: {supported_extensions}"
        )
    return image_path


class Timer:
    """
    Context manager for measuring elapsed execution time with high precision.
    """
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start

