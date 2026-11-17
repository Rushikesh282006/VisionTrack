"""
VisionTrack Training & Metrics Module
=====================================
Contains fine-tuning pipelines, loss metric computations (IoU, CIoU),
and hyperparameter configuration loaders.
"""

from src.training.metrics import compute_iou
from src.training.train import load_train_config, train_model

__all__ = [
    "compute_iou",
    "load_train_config",
    "train_model",
]
