FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-gcp.txt .
RUN pip install --no-cache-dir -r requirements-gcp.txt

# Copy source code
COPY src/ ./src/
COPY cloud_run/api_server.py ./api_server.py
COPY cloud_run/simple_policy.py ./simple_policy.py

# Cloud Run expects PORT environment variable
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT} --workers 1
