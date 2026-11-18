# VisionTrack: Real-Time Intelligent Traffic Monitoring & Vehicle Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-orange.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen.svg)]()

VisionTrack is a Computer Vision and Intelligent Transportation System (ITS) engineered for real-time edge traffic monitoring, multi-class vehicle detection, multi-object tracking (MOT), virtual tripwire counting, and live HUD telemetry analytics.

---

## 1. System Features

1. **Multi-Class Vehicle & Pedestrian Detection**: Real-time localization of cars, trucks, buses, pedestrians, and cyclists with pre-trained and fine-tuned YOLO nano backbones.
2. **Multi-Object Tracking (MOT)**: Identity persistence across video frames using IoU distance cost matrices, gated bipartite Hungarian matching, and a robust 4-stage track state machine (`New` $\to$ `Tracked` $\to$ `Lost` $\to$ `Removed`).
3. **Virtual Tripwires & Flow Analytics**: Directional vehicle counting (incoming vs. outgoing) using 2D vector orientation (`ccw`) line-crossing detection with automatic deduplication.
4. **Live Heads-Up Display (HUD)**: Alpha-blended semi-transparent dashboard banner overlay displaying real-time FPS, active track count, motion trajectory breadcrumbs, and directional vehicle tallies.
5. **KITTI Dataset Engineering Pipeline**: Automated conversion from raw KITTI 15-column labels to normalized YOLO coordinates with deterministic 80/10/10 leak-free train/val/test splitting.
6. **Cloud GPU Fine-Tuning Setup**: Ready-to-run Google Colab workflow with backbone layer freezing (`freeze=10`), CIoU box regression loss, and closed-mosaic scheduling.
7. **Comprehensive Automated Test Suite**: 29 unit and integration tests covering the entire pipeline.

---

## 2. Architecture & Project Layout

```text
VisionTrack/
├── configs/
│   ├── dataset.yaml            # KITTI dataset paths and class configuration
│   ├── traffic_kitti.yaml      # Processed YOLO training specification
│   └── train_config.yaml       # Hyperparameters (epochs, batch, lr0, freeze, augmentations)
├── data/
│   ├── raw/kitti/              # Raw KITTI labels and camera images
│   ├── processed/kitti_yolo/   # Converted YOLO labels (train, val, test)
│   └── samples/                # Sample benchmark images (bus.jpg)
├── models/                     # Exported model weights (yolo26n.pt)
├── notebooks/                  # Colab training notebooks (kitti_traffic_training_colab.ipynb)
├── outputs/                    # Annotated visual detections, video outputs, and summary logs
│   ├── bus_detected.jpg        # Annotated image detection output
│   ├── traffic_tracked.mp4     # Exported video tracking output with HUD
│   └── dataset_summary.md      # Full KITTI dataset audit metrics
├── src/
│   ├── __init__.py             # Clean package exports
│   ├── detection.py            # YOLODetector and DetectionResult abstractions
│   ├── pipeline.py             # VideoPipeline, draw_track_trail, and render_pipeline_hud
│   ├── utils.py                # Logging, device detection, timers, path validators
│   ├── dataset/                # Dataset Engineering Engine
│   │   ├── converter.py        # Forward/reverse coordinate math and 15-column parser
│   │   ├── split.py            # Leak-free deterministic 80/10/10 partitioning
│   │   ├── validator.py        # Integrity quality checks and coordinate bounds auditing
│   │   └── visualize.py        # OpenCV annotation visualizer
│   ├── tracking/               # Multi-Object Tracking Engine
│   │   ├── tracker.py          # Track state machine, cost matrix, SimpleTracker
│   │   └── counter.py          # CCW orientation test, segment crossing, VehicleCounter
│   └── training/               # Training & Loss Engine
│       ├── metrics.py          # IoU calculation with zero-division protection
│       └── train.py            # Fine-tuning runner with automatic hardware selection
├── tests/                      # Automated Test Suite (29 tests)
│   ├── test_detection.py       # Day 1 detection and timing tests
│   ├── test_dataset_pipeline.py# Day 2 converter, split, and validator tests
│   ├── test_converter.py       # Day 2 coordinate math tests
│   ├── test_training_pipeline.py# Day 3 IoU and configuration tests
│   ├── test_tracking_pipeline.py# Day 4 tracking, Hungarian, and tripwire tests
│   └── test_pipeline.py        # Day 5 video pipeline and HUD tests
├── main.py                     # Unified CLI entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 3. Quickstart & CLI Usage

### Installation

```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 1. Single-Image Detection Mode

Run detection on a static image with latency profiling:

```powershell
python main.py --mode image --source data/samples/bus.jpg
```

_Outputs are saved to `outputs/bus_detected.jpg`._

### 2. Video Tracking & Analytics Mode

Run tracking, tripwire counting, and HUD rendering on a video file:

```powershell
python main.py --mode video --source path/to/traffic.mp4 --output outputs/traffic_tracked.mp4
```

### 3. Automated End-to-End Demo Mode

Generate a synthetic multi-vehicle traffic simulation and run the full pipeline:

```powershell
python main.py --mode demo
```

_Outputs are saved to `outputs/traffic_tracked.mp4`._

---

## 4. Benchmark Performance (Edge CPU)

Tested on local CPU (Windows, Python 3.13, PyTorch CPU, YOLO Nano):

| Task                | Input Resolution |           Latency / FPS           | Key Features                                                         |
| :------------------ | :--------------: | :-------------------------------: | :------------------------------------------------------------------- |
| **Still Detection** |    640 × 640     |       **152 ms (~6.5 FPS)**       | Warm-up run, class breakdown, confidence filtering                   |
| **Video Pipeline**  |    640 × 360     | **~15–20 ms / frame (MOT + HUD)** | Bounding boxes, ID pills, motion trails, tripwire counter, alpha HUD |
| **Full Test Suite** |  29 Test Cases   |         **~13.3 seconds**         | 100% automated test coverage across all 5 modules                    |

---

## 5. Automated Test Suite Verification

Run all test suites across the entire repository:

```powershell
python -m unittest discover tests
```

```text
Ran 29 tests in 13.316s

OK
```
