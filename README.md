# VisionTrack: Real-Time Intelligent Traffic Monitoring & Vehicle Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-orange.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-purple.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()

VisionTrack is a  Computer Vision and Intelligent Transportation System (ITS) engineered for real-time edge traffic monitoring, multi-class vehicle detection, multi-object tracking (MOT), virtual tripwire counting, and live HUD telemetry analytics.

---

## 1. System Features

1. **Multi-Class Vehicle & Pedestrian Detection**: Real-time localization of cars, trucks, buses, pedestrians, and cyclists with pre-trained and fine-tuned YOLO nano backbones.
2. **Multi-Object Tracking (MOT)**: Identity persistence across video frames using IoU distance cost matrices, gated bipartite Hungarian matching, and a robust 4-stage track state machine (`New` $\to$ `Tracked` $\to$ `Lost` $\to$ `Removed`).
3. **Virtual Tripwires & Flow Analytics**: Directional vehicle counting (incoming vs. outgoing) using 2D vector orientation (`ccw`) line-crossing detection with automatic deduplication.
4. **Live Heads-Up Display (HUD)**: Alpha-blended semi-transparent dashboard banner overlay displaying real-time FPS, active track count, motion trajectory breadcrumbs, and directional vehicle tallies.
5. **Interactive Web Dashboard**: Streamlit-powered traffic surveillance UI with drag-and-drop video/image testing, dynamic tripwire adjustment sliders, and category distribution charts.
6. **Production REST API**: FastAPI web microservice with modern `lifespan` model caching, Pydantic data contracts, and interactive Swagger documentation.
7. **Containerized Architecture**: Multi-service Docker and Docker Compose orchestration for one-command deployment across cloud and edge hardware.
8. **Automated CI/CD Pipeline**: GitHub Actions workflow running 34 unit and integration tests on every push.

---

## 2. Architecture & Project Layout

```text
VisionTrack/
├── .github/workflows/ci.yml    # GitHub Actions automated CI testing workflow
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
│   ├── api/                    # Production FastAPI REST microservice
│   │   ├── app.py              # Endpoints (/health, /detect/image, /track/video)
│   │   └── schemas.py          # Pydantic validation models
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
├── tests/                      # Automated Test Suite (34 tests)
│   ├── test_detection.py       # Day 1 detection and timing tests
│   ├── test_dataset_pipeline.py# Day 2 converter, split, and validator tests
│   ├── test_converter.py       # Day 2 coordinate math tests
│   ├── test_training_pipeline.py# Day 3 IoU and configuration tests
│   ├── test_tracking_pipeline.py# Day 4 tracking, Hungarian, and tripwire tests
│   ├── test_pipeline.py        # Day 5 video pipeline and HUD tests
│   └── test_api.py             # Day 6 FastAPI endpoint integration tests
├── .dockerignore               # Docker build context exclusion rules
├── Dockerfile                  # Production container image specification
├── docker-compose.yml          # Multi-container orchestration (API + Dashboard)
├── dashboard.py                # Streamlit operator dashboard
├── main.py                     # Unified CLI entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 3. Quickstart & Usage

### Local Installation
```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 1. Interactive Web Dashboard (Streamlit)
Launch the operator dashboard in your browser:
```powershell
streamlit run dashboard.py
```
*Access the UI at `http://localhost:8501`.*

### 2. Production REST API (FastAPI)
Start the REST microservice with automatic Swagger documentation:
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
*Access interactive Swagger docs at `http://localhost:8000/docs`.*

### 3. Unified CLI
```powershell
# Single-Image Detection:
python main.py --mode image --source data/samples/bus.jpg

# Video Tracking & Analytics:
python main.py --mode video --source path/to/traffic.mp4

# Live Traffic Simulation Demo:
python main.py --mode demo
```

### 4. One-Command Docker Deployment
Launch both the FastAPI API and the Streamlit Dashboard simultaneously:
```bash
docker compose up --build
```

---

## 4. Benchmark Performance (Edge CPU)

Tested on local CPU (Windows, Python 3.13, PyTorch CPU, YOLO Nano):

| Task | Input Resolution | Latency / FPS | Key Features |
| :--- | :---: | :---: | :--- |
| **Still Detection** | 640 × 640 | **152 ms (~6.5 FPS)** | Warm-up run, class breakdown, confidence filtering |
| **Video Pipeline** | 640 × 360 | **~15–20 ms / frame (MOT + HUD)** | Bounding boxes, ID pills, motion trails, tripwire counter, alpha HUD |
| **Full Test Suite** | 34 Test Cases | **~6.5 seconds** | 100% automated test coverage across all modules |

---

## 5. Automated Test Suite Verification

Run all 34 test suites across the entire repository:
```powershell
python -m unittest discover tests
```

```text
Ran 34 tests in 6.536s

OK
```
