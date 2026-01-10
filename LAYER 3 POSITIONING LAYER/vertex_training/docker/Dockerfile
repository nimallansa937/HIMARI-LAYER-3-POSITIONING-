# HIMARI Layer 3 - Vertex AI Training Container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Vertex AI
RUN pip install --no-cache-dir \
    google-cloud-storage==2.10.0 \
    google-cloud-aiplatform==1.38.0 \
    torch==2.0.1 \
    numpy==1.24.3

# Copy the entire Layer 3 codebase
COPY src/ /app/src/
COPY config/ /app/config/
COPY vertex_ai/trainer/ /app/trainer/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Set the entrypoint to the training script
ENTRYPOINT ["python", "/app/trainer/train.py"]
