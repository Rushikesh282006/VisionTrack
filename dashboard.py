"""
VisionTrack - Interactive Traffic Operator Dashboard
====================================================
Streamlit web application for real-time traffic detection,
multi-object tracking, and virtual tripwire analytics.

Usage:
    streamlit run dashboard.py
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detection import YOLODetector
from src.pipeline import VideoPipeline
from src.tracking.counter import VehicleCounter
from src.tracking.tracker import SimpleTracker
from src.utils import get_device

# Page configuration
st.set_page_config(
    page_title="VisionTrack | Traffic Surveillance Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "pedestrian",
    4: "cyclist",
}


@st.cache_resource
def get_cached_detector() -> YOLODetector:
    """Loads and caches YOLO detector in memory once."""
    candidates = [
        Path("models/yolo26n.pt"),
        Path("yolo26n.pt"),
        Path("weights/best.pt"),
        Path("models/yolov8n.pt"),
    ]
    model_path = candidates[0]
    for c in candidates:
        if c.exists():
            model_path = c
            break

    return YOLODetector(model_path=model_path, device=get_device())


def setup_sidebar() -> dict:
    """Renders sidebar controls and parameters."""
    st.sidebar.title("🚦 VisionTrack ITS")
    st.sidebar.markdown(
        f"**Hardware Device**: `{get_device().upper()}`\n\n"
        "**System Status**: 🟢 `Online`"
    )
    st.sidebar.divider()

    mode = st.sidebar.selectbox(
        "Operating Mode",
        ["Image Detection", "Video Tracking", "Live Synthetic Demo"],
    )

    st.sidebar.subheader("Telemetry Tuning")
    conf = st.sidebar.slider(
        "Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05
    )
    tripwire_pct = st.sidebar.slider(
        "Virtual Tripwire Y-Position (%)", min_value=20, max_value=80, value=55, step=5
    )

    st.sidebar.divider()
    st.sidebar.info(
        "**VisionTrack 2.0**\n"
        "Anchor-free YOLO Detection + MOT Tracking + Tripwire Analytics."
    )

    return {
        "mode": mode,
        "conf": conf,
        "tripwire_ratio": tripwire_pct / 100.0,
    }


def render_image_studio(detector: YOLODetector, conf: float) -> None:
    """Studio for still image detection and telemetry breakdown."""
    st.header("📸 Traffic Image Detection Studio")
    st.caption("Upload road surveillance imagery or test with standard KITTI benchmark scenes.")

    uploaded = st.file_uploader(
        "Upload Traffic Image", type=["jpg", "jpeg", "png"], key="img_uploader"
    )

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.success(f"Loaded uploaded image: `{uploaded.name}`")
    else:
        sample_path = Path("data/samples/bus.jpg")
        if sample_path.exists():
            img = cv2.imread(str(sample_path))
            st.info("Displaying default benchmark sample (`bus.jpg`). Upload your own image above anytime!")
        else:
            img = None
            st.warning("No sample image found. Please upload a JPEG or PNG image.")

    if img is not None:
        result = detector.predict(img, imgsz=640, conf_threshold=conf, return_annotated=True)

        # KPI Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Objects Detected", len(result.detections))
        m2.metric("Inference Latency", f"{result.inference_time_ms:.1f} ms")
        fps_equiv = 1000.0 / max(result.inference_time_ms, 1e-4)
        m3.metric("Throughput (Inferred)", f"{fps_equiv:.1f} FPS")
        m4.metric("Active Hardware", get_device().upper())

        # Visual Comparison
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Original Camera Feed")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption="Raw Sensor Data",
                use_container_width=True,
            )
        with col_right:
            st.subheader("YOLO Neural Overlay")
            st.image(
                cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                caption="Localized Vehicles & Vulnerable Road Users",
                use_container_width=True,
            )

        # Tabulated Detections & Class Breakdown
        st.divider()
        st.subheader("Telemetry Breakdown")

        tab_table, tab_chart = st.tabs(["📋 Detection List", "📊 Category Distribution"])
        with tab_table:
            if result.detections:
                rows = []
                for d in result.detections:
                    box = getattr(d, "box", getattr(d, "bbox", (0, 0, 0, 0)))
                    rows.append({
                        "Class": d.class_name.capitalize(),
                        "Confidence": f"{d.confidence:.2%}",
                        "X_min": int(box[0]),
                        "Y_min": int(box[1]),
                        "X_max": int(box[2]),
                        "Y_max": int(box[3]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No objects detected above the selected confidence threshold.")

        with tab_chart:
            counts = result.count_by_class()
            if counts:
                df_counts = pd.DataFrame(
                    list(counts.items()), columns=["Category", "Count"]
                ).set_index("Category")
                st.bar_chart(df_counts)
            else:
                st.info("No class data to display.")


def render_video_studio(detector: YOLODetector, conf: float, tripwire_ratio: float) -> None:
    """Studio for video upload, MOT tracking, and directional counting."""
    st.header("🎥 Real-Time Video Tracking & Virtual Tripwire Studio")
    st.caption("Processes video feeds with multi-object tracking (MOT) and tripwire flow analytics.")

    uploaded_video = st.file_uploader(
        "Upload Traffic Video", type=["mp4", "avi", "mov"], key="vid_uploader"
    )
    max_frames = st.slider("Max Frames to Process (0 = All)", 0, 300, 60, step=10)
    frames_limit = None if max_frames == 0 else max_frames

    if uploaded_video is not None:
        # Save uploaded video to temp file
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(uploaded_video.read())
            tmp_in_path = tmp_in.name

        tmp_out_path = tempfile.mktemp(suffix=".mp4")

        if st.button("🚀 Run Video Pipeline", type="primary"):
            with st.spinner("Processing video stream through Detection -> Tracking -> Counting -> HUD..."):
                cap = cv2.VideoCapture(tmp_in_path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cap.release()

                trip_y = float(h * tripwire_ratio)
                tracker = SimpleTracker(max_cost=0.7, max_lost_frames=30)
                counter = VehicleCounter(
                    line_start=(0.0, trip_y), line_end=(float(w), trip_y), class_names=CLASS_NAMES
                )
                pipeline = VideoPipeline(detector, tracker, counter, class_names=CLASS_NAMES)

                metrics = pipeline.process_video(
                    input_path=tmp_in_path,
                    output_path=tmp_out_path,
                    max_frames=frames_limit,
                )

            st.success("Video processing complete!")

            # Telemetry KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Counted", metrics["counts"]["total"])
            k2.metric("Incoming (Down)", metrics["counts"]["incoming"])
            k3.metric("Outgoing (Up)", metrics["counts"]["outgoing"])
            k4.metric("Average FPS", f"{metrics['avg_fps']:.1f}")

            # Display video
            st.subheader("Annotated Video Stream with Live HUD")
            st.video(tmp_out_path)


def render_demo_studio(detector: YOLODetector, conf: float, tripwire_ratio: float) -> None:
    """Generates synthetic multi-vehicle traffic sequence and streams tracking."""
    st.header("⚡ Live Traffic Simulation Demo")
    st.caption("Generates an interactive synthetic highway sequence with bidirectional vehicles crossing the tripwire.")

    if st.button("▶️ Generate & Run Live Traffic Simulation", type="primary"):
        with st.spinner("Generating simulation & executing pipeline..."):
            temp_dir = Path("outputs/demo")
            temp_dir.mkdir(parents=True, exist_ok=True)
            demo_in = temp_dir / "dashboard_demo_input.mp4"
            demo_out = temp_dir / "dashboard_demo_output.mp4"

            # Generate 60-frame traffic sequence
            w, h, fps = 640, 360, 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(demo_in), fourcc, fps, (w, h))

            for frame_idx in range(60):
                frame = np.full((h, w, 3), 42, dtype=np.uint8)
                # Road lines
                dash_x = (frame_idx * 5) % 40
                for x in range(dash_x, w, 40):
                    cv2.line(frame, (x, h // 2), (x + 20, h // 2), (255, 255, 255), 2)

                # Vehicle 1: Downward Car (crossing tripwire)
                v1_y = int(80 + frame_idx * 3.5)
                cv2.rectangle(frame, (180, v1_y), (260, v1_y + 45), (0, 165, 255), -1)
                cv2.putText(frame, "CAR", (190, v1_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Vehicle 2: Upward Truck (crossing tripwire)
                v2_y = int(270 - frame_idx * 3.5)
                cv2.rectangle(frame, (380, v2_y), (480, v2_y + 60), (220, 200, 0), -1)
                cv2.putText(frame, "TRUCK", (395, v2_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                writer.write(frame)
            writer.release()

            # Process with VideoPipeline
            trip_y = float(h * tripwire_ratio)
            tracker = SimpleTracker(max_cost=0.7, max_lost_frames=30)
            counter = VehicleCounter(
                line_start=(0.0, trip_y), line_end=(float(w), trip_y), class_names=CLASS_NAMES
            )
            pipeline = VideoPipeline(detector, tracker, counter, class_names=CLASS_NAMES)

            metrics = pipeline.process_video(
                input_path=str(demo_in),
                output_path=str(demo_out),
            )

        st.success("Simulation & tracking execution finished!")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Counted", metrics["counts"]["total"])
        d2.metric("Incoming Vehicles", metrics["counts"]["incoming"])
        d3.metric("Outgoing Vehicles", metrics["counts"]["outgoing"])
        d4.metric("Pipeline FPS", f"{metrics['avg_fps']:.1f}")

        st.subheader("Live Video Playback with HUD Overlay")
        st.video(str(demo_out))


def main() -> None:
    settings = setup_sidebar()
    detector = get_cached_detector()

    mode = settings["mode"]
    conf = settings["conf"]
    trip_ratio = settings["tripwire_ratio"]

    if mode == "Image Detection":
        render_image_studio(detector, conf)
    elif mode == "Video Tracking":
        render_video_studio(detector, conf, trip_ratio)
    elif mode == "Live Synthetic Demo":
        render_demo_studio(detector, conf, trip_ratio)


if __name__ == "__main__":
    main()
