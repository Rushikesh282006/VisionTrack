# VisionTrack: Real-Time Intelligent Traffic Monitoring & Vehicle Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-orange.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-purple.svg)](https://github.com/ultralytics/ultralytics)

VisionTrack is a portfolio-grade Computer Vision and Machine Learning system engineered for real-time edge traffic monitoring, multi-class vehicle detection, pedestrian awareness, and telemetry analytics.

---

## 1. Project Objectives

1. **Multi-Class Vehicle & Vulnerable Road User Detection**: Accurate real-time localization of cars, buses, trucks, motorcycles, and pedestrians.
2. **Virtual Line Counting & Flow Analytics**: Bidirectional vehicle counting across virtual tripwires.
3. **Multi-Object Tracking (MOT)**: Temporal object identity persistence across consecutive video frames.
4. **Edge CPU Inference Optimization**: High throughput and sub-200ms prediction latencies on commodity multi-core CPUs.
5. **Production Microservice Architecture**: FastAPI backend + interactive analytics dashboard + containerized deployment with Docker.

---

## 2. System Architecture

```text
VisionTrack/
├── configs/            # Experiment configurations and class mappings
├── data/
│   ├── raw/            # Raw datasets (KITTI, custom videos)
│   ├── processed/      # Preprocessed, YOLO-formatted splits
│   └── samples/        # Verification and benchmark images
├── models/             # Exported weights (.pt, ONNX)
├── notebooks/          # Exploratory data analysis and metric reviews
├── outputs/            # Annotated visual detections and evaluation logs
├── src/
│   ├── __init__.py     # Package exports
│   ├── detection.py    # Modular YOLODetector engine
│   └── utils.py        # Profiling, logging, and filesystem validation
├── tests/              # Unit and integration test suite
├── bench_mark.py       # Latency and resolution benchmarking script
├── main.py             # Inference pipeline entry point
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 3. Hardware Constraints & Engineering Philosophy

- **Local Development**: Intel/AMD x86-64 CPU, 16 GB RAM, integrated graphics (no local NVIDIA GPU).
- **Execution Strategy**:
  - Model architecture inspection, dataset engineering, and CPU-optimized inference run locally.
  - Compute-heavy transfer learning and model fine-tuning are dispatched to Google Colab GPU runtimes.
- **Warm-Up Profiling**: To eliminate cold-start graph initialization anomalies, all latency benchmarks run a mandatory warm-up pass before collecting performance metrics.

---

## 4. Benchmark Latency Results (Day 1 Preliminary)

Benchmarked on local CPU using 5 consecutive runs per input resolution (model: `YOLO26n`):

| Image Size | Average Inference Latency | Inferred Throughput | Target Use-Case |
|:---:|:---:|:---:|:---:|
| **640 × 640** | **0.346 s** (~346 ms) | ~2.9 FPS | High-accuracy surveillance stills |
| **416 × 416** | **0.206 s** (~206 ms) | ~4.9 FPS | Balanced edge video processing |
| **320 × 320** | **0.192 s** (~192 ms) | ~5.2 FPS | High-speed edge IoT monitoring |

---

## 5. Quickstart

### Installation
```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Running Detection
```powershell
python main.py
```

### Running Test Suite
```powershell
python -m unittest tests/test_detection.py
```

