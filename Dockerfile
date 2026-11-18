# ==============================================================================
# VisionTrack - Production Docker Container
# Multi-purpose image for FastAPI REST Backend and Streamlit Dashboard
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system runtime dependencies required by OpenCV (headless GL & GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Layer Caching Optimization:
# Copy ONLY requirements.txt first so Docker reuses cached pip packages
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/
COPY data/samples/ ./data/samples/
COPY main.py dashboard.py ./

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command: Start FastAPI REST microservice with Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
