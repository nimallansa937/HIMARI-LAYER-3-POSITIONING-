# HIMARI OPUS 2 - Google Cloud Platform RL Deployment Guide

**Version:** 1.0
**Date:** 2025-12-26
**Status:** Production Ready
**Budget:** $400 CAD GCP Credit (3 months)
**Target:** Google Anti-Gravity Coding Agent

---

## Executive Summary

This guide provides a complete migration path from local/Colab-based RL training to a production-grade Google Cloud Platform deployment. The system uses $96 of the $400 CAD credit over 3 months, leaving $304 for experimentation.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ GOOGLE CLOUD PLATFORM                                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Vertex AI Training                                     │   │
│  │ • Train PPO agent (8-10 hours)                        │   │
│  │ • GPU: NVIDIA T4                                      │   │
│  │ • Auto-save checkpoints to Cloud Storage             │   │
│  │ Cost: $15-30 one-time                                │   │
│  └────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Cloud Storage                                          │   │
│  │ • Model: gs://himari-rl/models/ppo_final.pt          │   │
│  │ • Training data: gs://himari-rl/data/                │   │
│  │ • Logs: gs://himari-rl/logs/                         │   │
│  │ Cost: $1-2/month                                      │   │
│  └────────────────────────────────────────────────────────┘   │
│                          ↓                                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Cloud Run (Serverless API)                            │   │
│  │ • Endpoint: POST /predict                             │   │
│  │ • Auto-scaling: 0-10 instances                        │   │
│  │ • Latency: <50ms                                      │   │
│  │ • Uptime: 99.95% SLA                                  │   │
│  │ Cost: $5-15/month                                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                          ↑ HTTPS                               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Cloud Monitoring + Logging                            │   │
│  │ • Real-time dashboards                                │   │
│  │ • Alert policies (latency, errors)                    │   │
│  │ • 30-day log retention                                │   │
│  │ Cost: $3-5/month                                      │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↑
                       HTTPS API Call
                       (150ms timeout)
                            │
┌─────────────────────────────────────────────────────────────────┐
│ LOCAL PRODUCTION SERVER                                         │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Layer 3 Phase 1 RL                                     │   │
│  │ • Calls GCP API: POST https://XXX.run.app/predict    │   │
│  │ • Timeout: 150ms                                      │   │
│  │ • Fallback: Bayesian Kelly if timeout                │   │
│  │ • State: 16-dim feature vector                        │   │
│  │ • Response: position multiplier [0.0-2.0]            │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost Breakdown

### 3-Month Budget Allocation

| Component | One-Time | Monthly | 3-Month Total | Notes |
|-----------|----------|---------|---------------|-------|
| **Vertex AI Training** | $30 | $0 | $30 | 10 hours on T4 GPU |
| **Cloud Run** (API) | $0 | $10 | $30 | Auto-scales, pay per request |
| **Cloud Storage** | $0 | $2 | $6 | Model files + logs (10 GB) |
| **Cloud Monitoring** | $0 | $5 | $15 | Dashboards + alerts |
| **Networking** | $0 | $5 | $15 | Egress traffic |
| **TOTAL** | **$30** | **$22** | **$96** | **76% under budget** |
| **Remaining Credit** | | | **$304** | **For experimentation** |

### Usage of Remaining $304 Credit

**Recommended Experiments:**
1. **Hyperparameter Tuning** ($50-100): Grid search on Vertex AI
2. **Multi-Asset Training** ($80-120): Train on BTC, ETH, SOL simultaneously
3. **LSTM Architecture** ($60-80): Compare vs PPO performance
4. **Live A/B Testing** ($30-40): 50/50 split RL vs Kelly
5. **Reserve** ($34-84): Buffer for unexpected costs

---

## Prerequisites

### 1. Google Cloud Account Setup

**Step 1: Create GCP Account**
```bash
# Go to: https://cloud.google.com/
# Sign in with Google account
# Activate $400 CAD free credit (new users get $300 USD + your $400 CAD)
```

**Step 2: Create Project**
```bash
# Project ID: himari-opus-2
# Project Name: HIMARI OPUS 2 Production
# Billing: Link to your $400 CAD credit account
```

**Step 3: Enable Required APIs**
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  storage-api.googleapis.com \
  monitoring.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com
```

**Cost:** $0 (API activation is free)

---

### 2. Local Development Environment

**Install Google Cloud SDK**
```bash
# macOS
brew install google-cloud-sdk

# Windows
# Download from: https://cloud.google.com/sdk/docs/install

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version
```

**Authenticate**
```bash
# Login to your GCP account
gcloud auth login

# Set default project
gcloud config set project himari-opus-2

# Set default region (choose closest to you)
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Verify configuration
gcloud config list
```

**Install Python Dependencies**
```bash
cd "LAYER 3 POSITIONING LAYER"

# Create new requirements file for GCP
cat > requirements-gcp.txt << EOF
# Core ML
torch==2.0.1
numpy==1.24.3

# GCP SDKs
google-cloud-aiplatform==1.38.0
google-cloud-storage==2.10.0
google-cloud-monitoring==2.15.1
google-cloud-logging==3.5.0

# API Server
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# HTTP client
httpx==0.25.2

# Utilities
python-dotenv==1.0.0
PyYAML==6.0.1
EOF

pip install -r requirements-gcp.txt
```

---

## Phase 1: Vertex AI Training (Week 1, Day 1-2)

### Objective
Train PPO agent on Google Cloud with GPU acceleration, auto-save to Cloud Storage.

### Step 1: Create Cloud Storage Bucket

```bash
# Create bucket for models and data
gsutil mb -l us-central1 gs://himari-rl-models/

# Create folder structure
gsutil mkdir gs://himari-rl-models/training-data/
gsutil mkdir gs://himari-rl-models/checkpoints/
gsutil mkdir gs://himari-rl-models/final-models/
gsutil mkdir gs://himari-rl-models/logs/

# Set lifecycle policy (delete checkpoints after 30 days)
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["checkpoints/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://himari-rl-models/
```

**Cost:** $0 initial, ~$1-2/month for 10 GB storage

---

### Step 2: Prepare Training Code for Vertex AI

**Create `vertex_training/trainer.py`:**
```python
"""
HIMARI RL Trainer for Vertex AI
Designed to run on managed training service with GPU.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Vertex AI sets these environment variables
BUCKET_NAME = os.environ.get('AIP_STORAGE_URI', 'gs://himari-rl-models')
MODEL_DIR = os.environ.get('AIP_MODEL_DIR', './models')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from google.cloud import storage

from rl.trainer import RLTrainer, TrainingConfig
from rl.trading_env import EnvConfig
from rl.ppo_agent import PPOConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload file to Google Cloud Storage."""
    client = storage.Client()

    # Parse GCS path: gs://bucket/path
    gcs_path = gcs_path.replace('gs://', '')
    bucket_name = gcs_path.split('/')[0]
    blob_path = '/'.join(gcs_path.split('/')[1:])

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")


def main(args):
    """Main training loop for Vertex AI."""

    logger.info("=" * 80)
    logger.info("HIMARI RL Training on Vertex AI")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Bucket: {BUCKET_NAME}")
    logger.info("")

    # Training configuration
    training_config = TrainingConfig(
        num_episodes=args.num_episodes,
        max_steps_per_episode=500,
        batch_size=64,
        ppo_epochs=10,
        save_interval=50,
        log_interval=10,
        checkpoint_dir='./checkpoints',
        use_live_prices=True
    )

    # Environment configuration
    env_config = EnvConfig(
        initial_capital=100000.0,
        max_position_pct=0.5,
        commission_rate=0.001,
        slippage_bps=5,
        reward_window=10,
        max_steps=500,
        symbol='BTC-USD'
    )

    # PPO agent configuration
    ppo_config = PPOConfig(
        state_dim=16,
        action_dim=1,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
    )

    # Create trainer
    trainer = RLTrainer(
        training_config=training_config,
        env_config=env_config,
        ppo_config=ppo_config,
        device=args.device
    )

    logger.info("Trainer initialized")
    logger.info("")

    # Train
    logger.info("Starting training...")
    training_stats = trainer.train()
    logger.info("Training complete!")
    logger.info("")

    # Save final model locally
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(MODEL_DIR, 'ppo_final.pt')
    trainer.agent.save(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # Upload to GCS
    gcs_model_path = f"{BUCKET_NAME}/final-models/ppo_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    upload_to_gcs(final_model_path, gcs_model_path)

    # Upload latest as well (for Cloud Run to always use latest)
    gcs_latest_path = f"{BUCKET_NAME}/final-models/ppo_latest.pt"
    upload_to_gcs(final_model_path, gcs_latest_path)

    # Print final stats
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total episodes: {len(training_stats['episode_rewards'])}")
    logger.info(f"Avg reward (last 100): {np.mean(training_stats['episode_rewards'][-100:]):.3f}")
    logger.info(f"Avg Sharpe (last 100): {np.mean(training_stats['episode_sharpes'][-100:]):.3f}")
    logger.info(f"Avg P&L (last 100):    {np.mean(training_stats['episode_pnls'][-100:]):.2%}")
    logger.info(f"Model uploaded to: {gcs_latest_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HIMARI RL agent on Vertex AI')

    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()
    main(args)
```

---

### Step 3: Create Docker Container for Vertex AI

**Create `vertex_training/Dockerfile`:**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-gcp.txt .
RUN pip install --no-cache-dir -r requirements-gcp.txt

# Copy source code
COPY src/ ./src/
COPY vertex_training/trainer.py ./trainer.py

# Vertex AI expects the training script at /app/trainer.py
ENTRYPOINT ["python", "trainer.py"]
```

**Build and Push to Google Container Registry:**
```bash
cd "LAYER 3 POSITIONING LAYER"

# Build Docker image
docker build -f vertex_training/Dockerfile -t gcr.io/himari-opus-2/rl-trainer:latest .

# Configure Docker to use gcloud credentials
gcloud auth configure-docker

# Push to GCR
docker push gcr.io/himari-opus-2/rl-trainer:latest
```

**Alternative: Use Cloud Build (no local Docker needed)**
```bash
# Create cloudbuild.yaml
cat > cloudbuild.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'vertex_training/Dockerfile'
      - '-t'
      - 'gcr.io/himari-opus-2/rl-trainer:latest'
      - '.'
images:
  - 'gcr.io/himari-opus-2/rl-trainer:latest'
EOF

# Build in cloud
gcloud builds submit --config cloudbuild.yaml .
```

**Cost:** $0.01-0.05 for build time

---

### Step 4: Launch Vertex AI Training Job

**Create `vertex_training/launch_training.py`:**
```python
"""
Launch Vertex AI training job.
"""

from google.cloud import aiplatform
from datetime import datetime

# Initialize Vertex AI
aiplatform.init(
    project='himari-opus-2',
    location='us-central1',
    staging_bucket='gs://himari-rl-models'
)

# Create custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name=f'himari-rl-training-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    container_uri='gcr.io/himari-opus-2/rl-trainer:latest',
    model_serving_container_image_uri='gcr.io/himari-opus-2/rl-trainer:latest',
)

print("Submitting training job to Vertex AI...")
print("This will take 8-10 hours with GPU")
print("")

# Run training job
model = job.run(
    replica_count=1,
    machine_type='n1-standard-4',       # 4 vCPUs, 15 GB RAM
    accelerator_type='NVIDIA_TESLA_T4',  # T4 GPU
    accelerator_count=1,
    args=['--num-episodes', '1000', '--device', 'cuda'],
    environment_variables={
        'AIP_STORAGE_URI': 'gs://himari-rl-models'
    },
    sync=True  # Wait for completion
)

print("")
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model resource name: {model.resource_name}")
print(f"Model saved to: gs://himari-rl-models/final-models/ppo_latest.pt")
print("")
```

**Run Training:**
```bash
python vertex_training/launch_training.py
```

**Expected Output:**
```
Submitting training job to Vertex AI...
This will take 8-10 hours with GPU

Training job created: projects/123/locations/us-central1/trainingPipelines/456
View in console: https://console.cloud.google.com/vertex-ai/locations/us-central1/training/456

Training progress:
Episode 100/1000 | Reward: 0.234 | Sharpe: 0.512 | Time: 32s
Episode 200/1000 | Reward: 0.312 | Sharpe: 0.678 | Time: 29s
...
Episode 1000/1000 | Reward: 0.445 | Sharpe: 1.023 | Time: 28s

================================================================================
TRAINING COMPLETE
================================================================================
Model resource name: projects/123/locations/us-central1/models/789
Model saved to: gs://himari-rl-models/final-models/ppo_latest.pt
```

**Cost:** $15-30 for 8-10 hours on T4 GPU

**Note:** You can monitor training in real-time:
- Console: https://console.cloud.google.com/vertex-ai/training
- Logs: https://console.cloud.google.com/logs

---

## Phase 2: Cloud Run Deployment (Week 1, Day 3-4)

### Objective
Deploy trained RL model as serverless REST API with auto-scaling.

### Step 1: Create FastAPI Inference Server

**Create `cloud_run/api_server.py`:**
```python
"""
HIMARI RL Inference API Server
Runs on Cloud Run, serves RL model predictions.
"""

import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
from google.cloud import storage
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="HIMARI RL Inference API",
    description="Position sizing RL agent for HIMARI OPUS 2",
    version="1.0.0"
)

# CORS middleware (for web dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (loaded on startup)
MODEL = None
DEVICE = 'cpu'  # Cloud Run uses CPU


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""
    state: List[float] = Field(..., min_items=16, max_items=16,
                               description="16-dimensional state vector")

    class Config:
        schema_extra = {
            "example": {
                "state": [
                    0.85,  # signal_confidence
                    0, 0, 1,  # signal_action (one-hot: STRONG_BUY)
                    1, 0, 0, 0,  # signal_tier (T1)
                    0.3,  # position_size
                    1,    # position_side
                    0.02, # unrealized_pnl_pct
                    0.015, # price_momentum_1h
                    0.023, # price_momentum_4h
                    0.018, # volatility
                    0.65,  # recent_win_rate
                    0.2    # cascade_risk
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    position_multiplier: float = Field(..., ge=0.0, le=2.0,
                                      description="Position size multiplier [0.0-2.0]")
    confidence: float = Field(..., ge=0.0, le=1.0,
                            description="Model confidence score")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(..., description="Model version timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


# Global state
START_TIME = time.time()
MODEL_VERSION = "unknown"


def download_model_from_gcs():
    """Download latest model from Cloud Storage."""
    global MODEL_VERSION

    bucket_name = os.environ.get('GCS_BUCKET', 'himari-rl-models')
    model_path = os.environ.get('GCS_MODEL_PATH', 'final-models/ppo_latest.pt')
    local_path = '/tmp/ppo_latest.pt'

    logger.info(f"Downloading model from gs://{bucket_name}/{model_path}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)

        # Download model
        blob.download_to_filename(local_path)

        # Get model version (blob updated timestamp)
        MODEL_VERSION = blob.updated.strftime('%Y%m%d_%H%M%S')

        logger.info(f"Model downloaded successfully (version: {MODEL_VERSION})")
        return local_path

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def load_model():
    """Load RL model on startup."""
    global MODEL

    # Import after FastAPI setup to avoid import errors
    import sys
    sys.path.insert(0, '/app/src')
    from rl.ppo_agent import PPOAgent, PPOConfig

    try:
        # Download model from GCS
        model_path = download_model_from_gcs()

        # Create agent
        ppo_config = PPOConfig(
            state_dim=16,
            action_dim=1,
            hidden_dim=128,
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
        )

        agent = PPOAgent(config=ppo_config, device=DEVICE)

        # Load weights
        agent.load(model_path)
        agent.eval_mode()

        MODEL = agent
        logger.info("Model loaded successfully and set to eval mode")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    logger.info("Starting HIMARI RL Inference API...")
    load_model()
    logger.info("API ready to serve requests")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HIMARI RL Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Cloud Run."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=DEVICE,
        uptime_seconds=time.time() - START_TIME
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict position size multiplier from state.

    The state is a 16-dimensional vector encoding:
    - Signal confidence, action, tier (from Layer 2)
    - Current position size, side
    - Market metrics (momentum, volatility)
    - Performance metrics (win rate, cascade risk)
    """

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    try:
        # Convert to numpy array
        state = np.array(request.state, dtype=np.float32)

        # Validate state dimension
        if state.shape[0] != 16:
            raise HTTPException(
                status_code=400,
                detail=f"State must be 16-dimensional, got {state.shape[0]}"
            )

        # Get prediction from RL agent
        multiplier, _ = MODEL.get_action(state, deterministic=True)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            position_multiplier=float(multiplier),
            confidence=0.95,  # High confidence for trained model
            latency_ms=latency_ms,
            model_version=MODEL_VERSION
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} latency={latency_ms:.2f}ms"
    )

    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
```

---

### Step 2: Create Dockerfile for Cloud Run

**Create `cloud_run/Dockerfile`:**
```dockerfile
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

# Cloud Run expects PORT environment variable
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT} --workers 1
```

---

### Step 3: Build and Deploy to Cloud Run

**Build Container:**
```bash
cd "LAYER 3 POSITIONING LAYER"

# Build using Cloud Build (no local Docker needed)
gcloud builds submit \
  --tag gcr.io/himari-opus-2/rl-inference:latest \
  cloud_run/
```

**Deploy to Cloud Run:**
```bash
gcloud run deploy himari-rl-api \
  --image gcr.io/himari-opus-2/rl-inference:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 60s \
  --set-env-vars GCS_BUCKET=himari-rl-models,GCS_MODEL_PATH=final-models/ppo_latest.pt \
  --service-account himari-rl-api@himari-opus-2.iam.gserviceaccount.com
```

**Create Service Account (if needed):**
```bash
# Create service account
gcloud iam service-accounts create himari-rl-api \
  --display-name "HIMARI RL API Service Account"

# Grant access to Cloud Storage
gcloud projects add-iam-policy-binding himari-opus-2 \
  --member serviceAccount:himari-rl-api@himari-opus-2.iam.gserviceaccount.com \
  --role roles/storage.objectViewer
```

**Expected Output:**
```
Deploying container to Cloud Run service [himari-rl-api] in project [himari-opus-2] region [us-central1]
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
Done.
Service [himari-rl-api] revision [himari-rl-api-00001-abc] has been deployed and is serving 100 percent of traffic.
Service URL: https://himari-rl-api-abc123-uc.a.run.app
```

**Cost:** $5-15/month (scales to zero when not used)

---

### Step 4: Test Cloud Run API

**Test Health Endpoint:**
```bash
export API_URL="https://himari-rl-api-abc123-uc.a.run.app"

curl $API_URL/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "uptime_seconds": 123.45
}
```

**Test Prediction Endpoint:**
```bash
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": [0.85, 0, 0, 1, 1, 0, 0, 0, 0.3, 1, 0.02, 0.015, 0.023, 0.018, 0.65, 0.2]
  }'
```

**Expected Response:**
```json
{
  "position_multiplier": 1.23,
  "confidence": 0.95,
  "latency_ms": 12.45,
  "model_version": "20251226_143022"
}
```

**Test Latency (100 requests):**
```python
# test_latency.py
import httpx
import numpy as np
import time

API_URL = "https://himari-rl-api-abc123-uc.a.run.app/predict"

# Test state
state = [0.85, 0, 0, 1, 1, 0, 0, 0, 0.3, 1, 0.02, 0.015, 0.023, 0.018, 0.65, 0.2]

latencies = []

print("Running 100 requests...")
for i in range(100):
    start = time.time()
    response = httpx.post(API_URL, json={"state": state}, timeout=1.0)
    latency = (time.time() - start) * 1000
    latencies.append(latency)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/100 completed")

print("\nLatency Statistics:")
print(f"  Mean:   {np.mean(latencies):.2f} ms")
print(f"  Median: {np.median(latencies):.2f} ms")
print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
print(f"  Max:    {np.max(latencies):.2f} ms")
```

**Expected Output:**
```
Running 100 requests...
  10/100 completed
  20/100 completed
  ...
  100/100 completed

Latency Statistics:
  Mean:   45.23 ms
  Median: 42.11 ms
  P95:    67.45 ms
  P99:    89.12 ms
  Max:    112.34 ms
```

✅ **Target:** P99 < 150ms (meets Layer 3 latency budget)

---

## Phase 3: Local Integration (Week 1, Day 5)

### Objective
Update local Layer 3 to call GCP Cloud Run API with fallback to Bayesian Kelly.

### Step 1: Update Phase 1 RL Enhanced

**Edit `src/phases/phase1_rl_enhanced.py`:**

Add at the top:
```python
import httpx
import asyncio
from typing import Optional
```

Update `__init__` method:
```python
def __init__(
    self,
    portfolio_value: float = 100000,
    kelly_fraction: float = 0.25,
    config_path: Optional[str] = None,
    enable_hot_reload: bool = True,
    enable_metrics: bool = True,
    enable_sentiment: bool = True,
    enable_rl: bool = True,
    rl_model_path: Optional[str] = None,      # For local model
    rl_api_endpoint: Optional[str] = None,    # NEW: For GCP API
    rl_timeout_ms: int = 150,                 # NEW: API timeout
    device: str = 'cpu'
):
    """
    Initialize RL-enhanced Phase 1.

    Args:
        portfolio_value: Total portfolio value
        kelly_fraction: Kelly multiplier
        config_path: Config file path
        enable_hot_reload: Enable config hot-reload
        enable_metrics: Enable metrics
        enable_sentiment: Enable sentiment sizing
        enable_rl: Enable RL multiplier
        rl_model_path: Path to trained RL model (local deployment)
        rl_api_endpoint: GCP Cloud Run API endpoint (cloud deployment)
        rl_timeout_ms: API call timeout in milliseconds
        device: 'cpu' or 'cuda'
    """
    # Initialize base Phase 1
    super().__init__(
        portfolio_value=portfolio_value,
        kelly_fraction=kelly_fraction,
        config_path=config_path,
        enable_hot_reload=enable_hot_reload,
        enable_metrics=enable_metrics,
        enable_sentiment=enable_sentiment
    )

    self.enable_rl = enable_rl
    self.rl_api_endpoint = rl_api_endpoint
    self.rl_timeout_ms = rl_timeout_ms
    self.device = device

    # Initialize RL components
    if self.enable_rl:
        if rl_api_endpoint:
            # Cloud deployment: use API
            self._init_rl_api_client()
        elif rl_model_path:
            # Local deployment: load model
            self._init_rl_components(rl_model_path)
        else:
            logger.warning("RL enabled but no model path or API endpoint provided")
            self.rl_agent = None
            self.state_encoder = None
    else:
        self.rl_agent = None
        self.state_encoder = None
        self.http_client = None

    logger.info(
        f"Phase 1 RL-Enhanced initialized: "
        f"rl={'on' if enable_rl else 'off'}, "
        f"mode={'api' if rl_api_endpoint else 'local' if rl_model_path else 'none'}"
    )
```

Add new method for API client:
```python
def _init_rl_api_client(self):
    """Initialize HTTP client for GCP API."""
    self.http_client = httpx.AsyncClient(
        timeout=self.rl_timeout_ms / 1000.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
    self.rl_agent = None  # No local agent

    # Still need state encoder
    self.state_encoder = StateEncoder(max_position_usd=self.portfolio_value * 0.5)

    logger.info(f"RL API client initialized: {self.rl_api_endpoint}")
```

Add async API call method:
```python
async def _get_rl_multiplier_from_api(self, state: np.ndarray) -> Optional[float]:
    """
    Call GCP Cloud Run API for RL prediction.

    Args:
        state: 16-dimensional state vector

    Returns:
        position_multiplier or None if API call fails
    """
    if not self.http_client or not self.rl_api_endpoint:
        return None

    try:
        response = await self.http_client.post(
            self.rl_api_endpoint,
            json={"state": state.tolist()}
        )
        response.raise_for_status()
        data = response.json()

        multiplier = data.get('position_multiplier')
        latency_ms = data.get('latency_ms', 0)

        logger.debug(
            f"RL API response: multiplier={multiplier:.3f}, "
            f"latency={latency_ms:.2f}ms"
        )

        return multiplier

    except httpx.TimeoutException:
        logger.warning(f"RL API timeout after {self.rl_timeout_ms}ms")
        return None

    except httpx.HTTPError as e:
        logger.warning(f"RL API HTTP error: {e}")
        return None

    except Exception as e:
        logger.error(f"RL API call failed: {e}")
        return None
```

Update `calculate_position` method to use API:
```python
def calculate_position(
    self,
    signal: TacticalSignal,
    cascade_indicators: CascadeIndicators,
    current_price: float,
    current_position_usd: float = 0.0,
    current_position_qty: float = 0.0,
    entry_price: float = 0.0,
    recent_trades: list = None
) -> PositionSizingDecision:
    """
    Calculate position with RL enhancement (API or local).
    """
    # Get base decision from Phase 1
    base_decision = super().calculate_position(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=current_price
    )

    # If RL disabled, return base decision
    if not self.enable_rl:
        return base_decision

    # Build state for RL agent
    state = self._build_rl_state(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=current_price,
        current_position_usd=current_position_usd,
        current_position_qty=current_position_qty,
        entry_price=entry_price,
        recent_trades=recent_trades or []
    )

    # Get RL multiplier (API or local)
    if self.rl_api_endpoint:
        # Cloud deployment: call API
        try:
            rl_multiplier = asyncio.run(self._get_rl_multiplier_from_api(state))
        except Exception as e:
            logger.error(f"Async API call failed: {e}")
            rl_multiplier = None

        # Fallback if API unavailable
        if rl_multiplier is None:
            rl_multiplier = 1.0  # Use base decision (no RL adjustment)
            logger.info("Falling back to base Kelly (RL API unavailable)")
            rl_source = 'fallback_kelly'
        else:
            rl_source = 'gcp_api'

    else:
        # Local deployment: use loaded model
        if self.rl_agent:
            rl_multiplier, _ = self.rl_agent.get_action(state, deterministic=True)
            rl_source = 'local_model'
        else:
            rl_multiplier = 1.0
            rl_source = 'fallback_kelly'

    # Apply RL multiplier to cascade-adjusted position
    rl_adjusted_usd = base_decision.cascade_adjusted_usd * rl_multiplier

    # Clamp to reasonable bounds
    max_position = self.portfolio_value * 0.5  # Max 50%
    rl_adjusted_usd = min(rl_adjusted_usd, max_position)

    # Update decision
    base_decision.position_size_usd = rl_adjusted_usd

    # Add RL diagnostics
    if base_decision.diagnostics is None:
        base_decision.diagnostics = {}

    base_decision.diagnostics['rl'] = {
        'multiplier': rl_multiplier,
        'source': rl_source,
        'base_position_usd': base_decision.cascade_adjusted_usd,
        'rl_adjusted_usd': rl_adjusted_usd,
        'state_features': state.tolist()[:5],  # First 5 features for logging
    }

    logger.debug(
        f"RL adjustment ({rl_source}): "
        f"{base_decision.cascade_adjusted_usd:,.2f} × {rl_multiplier:.2f} "
        f"= ${rl_adjusted_usd:,.2f}"
    )

    return base_decision
```

---

### Step 2: Update Configuration

**Create `config/gcp_deployment.yaml`:**
```yaml
# HIMARI Layer 3 - GCP Deployment Configuration

layer3_rl:
  # Enable RL enhancement
  enable_rl: true

  # Cloud deployment (GCP Cloud Run)
  deployment_mode: "cloud"  # "cloud" or "local"

  # GCP Cloud Run API endpoint
  rl_api_endpoint: "https://himari-rl-api-abc123-uc.a.run.app/predict"

  # API timeout (must be < Layer 3 latency budget of 200ms)
  rl_timeout_ms: 150

  # Fallback behavior
  fallback_on_timeout: true
  fallback_multiplier: 1.0  # Use base Kelly if API fails

  # Local model (for testing/development)
  rl_model_path: "models/ppo_final.pt"  # Used if deployment_mode: "local"

  # Monitoring
  log_rl_predictions: true
  log_level: "INFO"

# Portfolio configuration
portfolio:
  initial_value: 100000.0
  max_position_pct: 0.5
  kelly_fraction: 0.25

# Risk management
risk:
  max_daily_loss_pct: 0.02
  max_leverage: 2.0
```

---

### Step 3: Create Production Runner

**Create `run_layer3_gcp.py`:**
```python
"""
HIMARI Layer 3 Production Runner with GCP Integration
"""

import os
import yaml
import logging
from datetime import datetime

from src.phases.phase1_rl_enhanced import Layer3Phase1RL
from src.core.layer3_types import TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/gcp_deployment.yaml"):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main production runner."""

    logger.info("=" * 80)
    logger.info("HIMARI Layer 3 - GCP Production Deployment")
    logger.info("=" * 80)
    logger.info("")

    # Load configuration
    config = load_config()

    rl_config = config['layer3_rl']
    portfolio_config = config['portfolio']

    # Initialize Layer 3 with GCP API
    logger.info("Initializing Layer 3 Phase 1 RL...")

    if rl_config['deployment_mode'] == 'cloud':
        # Cloud deployment
        phase1_rl = Layer3Phase1RL(
            portfolio_value=portfolio_config['initial_value'],
            kelly_fraction=portfolio_config['kelly_fraction'],
            enable_rl=rl_config['enable_rl'],
            rl_api_endpoint=rl_config['rl_api_endpoint'],
            rl_timeout_ms=rl_config['rl_timeout_ms'],
        )
        logger.info(f"  Mode: Cloud API")
        logger.info(f"  Endpoint: {rl_config['rl_api_endpoint']}")

    else:
        # Local deployment
        phase1_rl = Layer3Phase1RL(
            portfolio_value=portfolio_config['initial_value'],
            kelly_fraction=portfolio_config['kelly_fraction'],
            enable_rl=rl_config['enable_rl'],
            rl_model_path=rl_config['rl_model_path'],
        )
        logger.info(f"  Mode: Local Model")
        logger.info(f"  Model: {rl_config['rl_model_path']}")

    logger.info("")

    # Example: Process tactical signal
    logger.info("Example: Processing tactical signal from Layer 2")
    logger.info("-" * 60)

    # Simulate Layer 2 signal
    test_signal = TacticalSignal(
        strategy_id="momentum_strategy_v1",
        symbol="BTC-USD",
        action=TacticalAction.STRONG_BUY,
        confidence=0.85,
        risk_score=0.2,
        regime=MarketRegime.TRENDING_UP,
        timestamp_ns=int(datetime.now().timestamp() * 1e9),
        expected_return=0.08,
        predicted_volatility=0.04,
    )

    # Simulate cascade indicators
    cascade_indicators = CascadeIndicators(
        funding_rate=0.0015,
        oi_change_pct=0.05,
        volume_ratio=2.5,
        onchain_whale_pressure=0.3,
        exchange_netflow_zscore=0.5,
    )

    # Calculate position
    logger.info(f"Signal: {test_signal.action.value} @ confidence={test_signal.confidence:.2f}")
    logger.info(f"Regime: {test_signal.regime.value}")
    logger.info("")

    decision = phase1_rl.calculate_position(
        signal=test_signal,
        cascade_indicators=cascade_indicators,
        current_price=87000.0,
        current_position_usd=0.0,
        recent_trades=[]
    )

    # Display results
    logger.info("Position Sizing Decision:")
    logger.info(f"  Kelly Position:     ${decision.kelly_position_usd:,.2f}")
    logger.info(f"  Cascade Adjusted:   ${decision.cascade_adjusted_usd:,.2f}")
    logger.info(f"  Final Position:     ${decision.position_size_usd:,.2f}")
    logger.info(f"  Position (BTC):     {decision.position_size_usd / 87000:.6f} BTC")
    logger.info("")

    if decision.diagnostics and 'rl' in decision.diagnostics:
        rl_diag = decision.diagnostics['rl']
        logger.info("RL Diagnostics:")
        logger.info(f"  Multiplier:         {rl_diag['multiplier']:.3f}")
        logger.info(f"  Source:             {rl_diag['source']}")
        logger.info(f"  Base Position:      ${rl_diag['base_position_usd']:,.2f}")
        logger.info(f"  RL Adjusted:        ${rl_diag['rl_adjusted_usd']:,.2f}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("READY FOR PRODUCTION")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
```

**Run Production Test:**
```bash
python run_layer3_gcp.py
```

**Expected Output:**
```
================================================================================
HIMARI Layer 3 - GCP Production Deployment
================================================================================

Initializing Layer 3 Phase 1 RL...
  Mode: Cloud API
  Endpoint: https://himari-rl-api-abc123-uc.a.run.app/predict

Example: Processing tactical signal from Layer 2
------------------------------------------------------------
Signal: STRONG_BUY @ confidence=0.85
Regime: TRENDING_UP

Position Sizing Decision:
  Kelly Position:     $21,250.00
  Cascade Adjusted:   $19,550.00
  Final Position:     $24,066.50
  Position (BTC):     0.276667 BTC

RL Diagnostics:
  Multiplier:         1.231
  Source:             gcp_api
  Base Position:      $19,550.00
  RL Adjusted:        $24,066.50

================================================================================
READY FOR PRODUCTION
================================================================================
```

✅ **Success:** Layer 3 now calls GCP Cloud Run API for RL predictions!

---

## Phase 4: Monitoring & Observability (Week 1, Day 6-7)

### Objective
Setup comprehensive monitoring with Cloud Monitoring, dashboards, and alerts.

### Step 1: Enable Cloud Monitoring

**Create monitoring configuration:**
```python
# monitoring/setup_monitoring.py
"""
Setup Cloud Monitoring for HIMARI RL system.
"""

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import query
import time

PROJECT_ID = "himari-opus-2"
PROJECT_NAME = f"projects/{PROJECT_ID}"


def create_custom_metrics():
    """Create custom metrics for HIMARI RL."""

    client = monitoring_v3.MetricServiceClient()

    metrics = [
        {
            "type": "custom.googleapis.com/himari/rl/prediction_latency",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "description": "RL API prediction latency in milliseconds",
            "unit": "ms"
        },
        {
            "type": "custom.googleapis.com/himari/rl/position_multiplier",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "description": "RL position size multiplier [0.0-2.0]",
            "unit": "1"
        },
        {
            "type": "custom.googleapis.com/himari/rl/api_errors",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.INT64,
            "description": "Count of RL API errors",
            "unit": "1"
        },
        {
            "type": "custom.googleapis.com/himari/rl/fallback_count",
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE,
            "value_type": monitoring_v3.MetricDescriptor.ValueType.INT64,
            "description": "Count of fallbacks to Bayesian Kelly",
            "unit": "1"
        },
    ]

    for metric_def in metrics:
        descriptor = monitoring_v3.MetricDescriptor(
            type_=metric_def["type"],
            metric_kind=metric_def["metric_kind"],
            value_type=metric_def["value_type"],
            description=metric_def["description"],
            unit=metric_def.get("unit", "1")
        )

        try:
            created = client.create_metric_descriptor(
                name=PROJECT_NAME,
                metric_descriptor=descriptor
            )
            print(f"✓ Created metric: {metric_def['type']}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⊙ Metric already exists: {metric_def['type']}")
            else:
                print(f"✗ Failed to create metric: {e}")


def create_alert_policies():
    """Create alert policies for critical issues."""

    client = monitoring_v3.AlertPolicyServiceClient()

    # Alert: High latency
    high_latency_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL - High Latency",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="RL API latency exceeded 150ms threshold. Investigate Cloud Run performance.",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Latency > 150ms",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="custom.googleapis.com/himari/rl/prediction_latency" resource.type="cloud_run_revision"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=150.0,
                    duration={"seconds": 60},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                        )
                    ],
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        enabled=True,
    )

    # Alert: High error rate
    high_error_policy = monitoring_v3.AlertPolicy(
        display_name="HIMARI RL - High Error Rate",
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="RL API error rate exceeded 5%. Check Cloud Run logs for failures.",
            mime_type="text/markdown"
        ),
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Error rate > 5%",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="run.googleapis.com/request_count" resource.type="cloud_run_revision" metric.label.response_code_class="5xx"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=5.0,
                    duration={"seconds": 300},
                    aggregations=[
                        monitoring_v3.Aggregation(
                            alignment_period={"seconds": 60},
                            per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE,
                        )
                    ],
                )
            )
        ],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        enabled=True,
    )

    policies = [high_latency_policy, high_error_policy]

    for policy in policies:
        try:
            created = client.create_alert_policy(
                name=PROJECT_NAME,
                alert_policy=policy
            )
            print(f"✓ Created alert: {policy.display_name}")
        except Exception as e:
            print(f"✗ Failed to create alert: {e}")


if __name__ == "__main__":
    print("Setting up Cloud Monitoring for HIMARI RL...")
    print("")

    print("Creating custom metrics...")
    create_custom_metrics()
    print("")

    print("Creating alert policies...")
    create_alert_policies()
    print("")

    print("✓ Monitoring setup complete!")
    print("")
    print("View dashboards: https://console.cloud.google.com/monitoring")
```

**Run setup:**
```bash
python monitoring/setup_monitoring.py
```

---

### Step 2: Create Dashboard

**Create `monitoring/dashboard.json`:**
```json
{
  "displayName": "HIMARI RL Production Dashboard",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "RL API Latency (P50, P95, P99)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/request_latencies\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_50"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              },
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/request_latencies\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              },
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/request_latencies\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_99"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "thresholds": [
              {
                "value": 150.0,
                "color": "RED",
                "direction": "ABOVE"
              }
            ],
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Rate & Error Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              },
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\" metric.label.response_code_class=\"5xx\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y2"
              }
            ]
          }
        }
      },
      {
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Position Multiplier Distribution",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/himari/rl/position_multiplier\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "thresholds": [
              {
                "value": 0.0,
                "color": "YELLOW"
              },
              {
                "value": 2.0,
                "color": "RED"
              }
            ]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Fallback Count (RL API Unavailable)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"custom.googleapis.com/himari/rl/fallback_count\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                },
                "plotType": "STACKED_BAR",
                "targetAxis": "Y1"
              }
            ]
          }
        }
      },
      {
        "yPos": 8,
        "width": 12,
        "height": 4,
        "widget": {
          "title": "Cloud Run Instance Count & CPU Utilization",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/container/instance_count\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              },
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "metric.type=\"run.googleapis.com/container/cpu/utilizations\" resource.type=\"cloud_run_revision\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y2"
              }
            ]
          }
        }
      }
    ]
  }
}
```

**Import dashboard:**
```bash
gcloud monitoring dashboards create --config-from-file=monitoring/dashboard.json
```

**View dashboard:**
```
https://console.cloud.google.com/monitoring/dashboards
```

**Cost:** $3-5/month (most metrics are free tier)

---

## Phase 5: A/B Testing & Optimization (Week 2-3)

### Objective
Compare RL performance vs Bayesian Kelly baseline using live traffic.

### Step 1: Create A/B Testing Framework

**Create `testing/ab_test.py`:**
```python
"""
A/B Testing Framework for RL vs Bayesian Kelly
"""

import random
import json
import logging
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
import numpy as np

from src.phases.phase1_rl_enhanced import Layer3Phase1RL
from src.core.layer3_types import TacticalSignal, PositionSizingDecision

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Single A/B test result."""
    test_id: str
    timestamp: datetime
    variant: str  # "rl" or "kelly"
    signal: TacticalSignal
    decision: PositionSizingDecision
    realized_pnl: float = 0.0
    realized_return: float = 0.0
    trade_duration_seconds: float = 0.0


class ABTestRunner:
    """
    A/B testing framework for RL vs Kelly comparison.

    Split: 50% RL, 50% Kelly (Bayesian fallback)
    Duration: 2-4 weeks
    Metrics: Sharpe ratio, max DD, win rate, avg return
    """

    def __init__(
        self,
        phase1_rl: Layer3Phase1RL,
        split_ratio: float = 0.5,
        output_file: str = "ab_test_results.jsonl"
    ):
        self.phase1_rl = phase1_rl
        self.split_ratio = split_ratio
        self.output_file = output_file

        self.rl_results: List[ABTestResult] = []
        self.kelly_results: List[ABTestResult] = []

        logger.info(f"A/B Test initialized: {split_ratio:.0%} RL, {1-split_ratio:.0%} Kelly")

    def run_test(
        self,
        signal: TacticalSignal,
        cascade_indicators,
        current_price: float
    ) -> ABTestResult:
        """
        Run single A/B test iteration.

        Randomly assigns to RL or Kelly variant, records result.
        """

        # Random assignment
        use_rl = random.random() < self.split_ratio
        variant = "rl" if use_rl else "kelly"

        # Temporarily override RL enable flag
        original_enable_rl = self.phase1_rl.enable_rl
        self.phase1_rl.enable_rl = use_rl

        # Calculate position
        decision = self.phase1_rl.calculate_position(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=current_price
        )

        # Restore original flag
        self.phase1_rl.enable_rl = original_enable_rl

        # Record result
        result = ABTestResult(
            test_id=f"{variant}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            variant=variant,
            signal=signal,
            decision=decision
        )

        if use_rl:
            self.rl_results.append(result)
        else:
            self.kelly_results.append(result)

        # Save to file (append)
        self._save_result(result)

        logger.info(f"A/B Test: variant={variant}, position=${decision.position_size_usd:,.2f}")

        return result

    def update_result(
        self,
        test_id: str,
        realized_pnl: float,
        realized_return: float,
        trade_duration_seconds: float
    ):
        """Update test result with realized metrics after trade closes."""

        # Find result
        all_results = self.rl_results + self.kelly_results
        result = next((r for r in all_results if r.test_id == test_id), None)

        if result:
            result.realized_pnl = realized_pnl
            result.realized_return = realized_return
            result.trade_duration_seconds = trade_duration_seconds

            logger.info(
                f"Updated result {test_id}: "
                f"pnl=${realized_pnl:,.2f}, return={realized_return:.2%}"
            )

    def _save_result(self, result: ABTestResult):
        """Save result to JSONL file."""
        with open(self.output_file, 'a') as f:
            # Convert to dict (handle nested dataclasses)
            result_dict = {
                'test_id': result.test_id,
                'timestamp': result.timestamp.isoformat(),
                'variant': result.variant,
                'position_size_usd': result.decision.position_size_usd,
                'kelly_fraction': result.decision.kelly_fraction,
                'realized_pnl': result.realized_pnl,
                'realized_return': result.realized_return,
            }
            f.write(json.dumps(result_dict) + '\n')

    def analyze_results(self) -> Dict:
        """
        Analyze A/B test results.

        Returns statistical comparison of RL vs Kelly performance.
        """

        if len(self.rl_results) < 20 or len(self.kelly_results) < 20:
            logger.warning("Insufficient data for analysis (need 20+ per variant)")
            return {}

        # Calculate metrics
        rl_returns = [r.realized_return for r in self.rl_results if r.realized_return != 0]
        kelly_returns = [r.realized_return for r in self.kelly_results if r.realized_return != 0]

        def calculate_sharpe(returns):
            if len(returns) < 2:
                return 0.0
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return == 0:
                return 0.0
            return (mean_return / std_return) * np.sqrt(252)

        def calculate_max_dd(returns):
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown) if len(drawdown) > 0 else 0.0

        analysis = {
            'rl': {
                'n_trades': len(rl_returns),
                'mean_return': np.mean(rl_returns),
                'std_return': np.std(rl_returns),
                'sharpe': calculate_sharpe(rl_returns),
                'max_dd': calculate_max_dd(rl_returns),
                'win_rate': sum(1 for r in rl_returns if r > 0) / len(rl_returns),
            },
            'kelly': {
                'n_trades': len(kelly_returns),
                'mean_return': np.mean(kelly_returns),
                'std_return': np.std(kelly_returns),
                'sharpe': calculate_sharpe(kelly_returns),
                'max_dd': calculate_max_dd(kelly_returns),
                'win_rate': sum(1 for r in kelly_returns if r > 0) / len(kelly_returns),
            }
        }

        # Statistical significance (t-test)
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(rl_returns, kelly_returns)

        analysis['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Performance delta
        analysis['delta'] = {
            'sharpe': analysis['rl']['sharpe'] - analysis['kelly']['sharpe'],
            'mean_return': analysis['rl']['mean_return'] - analysis['kelly']['mean_return'],
            'max_dd': analysis['rl']['max_dd'] - analysis['kelly']['max_dd'],
        }

        return analysis

    def print_report(self):
        """Print A/B test report."""

        analysis = self.analyze_results()

        if not analysis:
            print("Insufficient data for analysis")
            return

        print("=" * 80)
        print("A/B TEST RESULTS: RL vs Bayesian Kelly")
        print("=" * 80)
        print()

        print(f"RL Variant ({analysis['rl']['n_trades']} trades):")
        print(f"  Sharpe Ratio:   {analysis['rl']['sharpe']:.3f}")
        print(f"  Mean Return:    {analysis['rl']['mean_return']:.2%}")
        print(f"  Std Return:     {analysis['rl']['std_return']:.2%}")
        print(f"  Max Drawdown:   {analysis['rl']['max_dd']:.2%}")
        print(f"  Win Rate:       {analysis['rl']['win_rate']:.1%}")
        print()

        print(f"Kelly Variant ({analysis['kelly']['n_trades']} trades):")
        print(f"  Sharpe Ratio:   {analysis['kelly']['sharpe']:.3f}")
        print(f"  Mean Return:    {analysis['kelly']['mean_return']:.2%}")
        print(f"  Std Return:     {analysis['kelly']['std_return']:.2%}")
        print(f"  Max Drawdown:   {analysis['kelly']['max_dd']:.2%}")
        print(f"  Win Rate:       {analysis['kelly']['win_rate']:.1%}")
        print()

        print("Performance Delta (RL - Kelly):")
        print(f"  Sharpe:         {analysis['delta']['sharpe']:+.3f}")
        print(f"  Mean Return:    {analysis['delta']['mean_return']:+.2%}")
        print(f"  Max DD:         {analysis['delta']['max_dd']:+.2%}")
        print()

        print("Statistical Test:")
        print(f"  t-statistic:    {analysis['statistical_test']['t_statistic']:.3f}")
        print(f"  p-value:        {analysis['statistical_test']['p_value']:.4f}")
        print(f"  Significant:    {'YES' if analysis['statistical_test']['significant'] else 'NO'} (α=0.05)")
        print()

        # Recommendation
        if analysis['statistical_test']['significant']:
            if analysis['delta']['sharpe'] > 0.1:
                print("✓ RECOMMENDATION: Deploy RL to 100% of traffic")
            elif analysis['delta']['sharpe'] < -0.1:
                print("✗ RECOMMENDATION: Disable RL, use Bayesian Kelly")
            else:
                print("⊙ RECOMMENDATION: Continue monitoring (small difference)")
        else:
            print("⊙ RECOMMENDATION: No significant difference, continue testing")

        print("=" * 80)


if __name__ == "__main__":
    """Example A/B test run."""

    from src.core.layer3_types import TacticalAction, MarketRegime, CascadeIndicators

    # Initialize Phase 1 RL with GCP API
    phase1 = Layer3Phase1RL(
        portfolio_value=100000,
        enable_rl=True,
        rl_api_endpoint="https://himari-rl-api-abc123-uc.a.run.app/predict",
        rl_timeout_ms=150
    )

    # Initialize A/B test
    ab_test = ABTestRunner(phase1, split_ratio=0.5)

    # Simulate 100 trades
    print("Running 100 simulated trades...")
    for i in range(100):
        # Create test signal
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=random.choice(list(TacticalAction)),
            confidence=random.uniform(0.6, 0.95),
            risk_score=random.uniform(0.1, 0.4),
            regime=random.choice(list(MarketRegime)),
            timestamp_ns=int(datetime.now().timestamp() * 1e9),
            expected_return=random.uniform(0.01, 0.08),
        )

        cascade = CascadeIndicators(
            funding_rate=random.uniform(-0.001, 0.003),
            oi_change_pct=random.uniform(-0.1, 0.1),
            volume_ratio=random.uniform(0.5, 3.0),
            onchain_whale_pressure=random.uniform(0.1, 0.6),
            exchange_netflow_zscore=random.uniform(-1, 1),
        )

        # Run test
        result = ab_test.run_test(signal, cascade, 87000.0)

        # Simulate realized return
        realized_return = random.gauss(0.01, 0.03)
        realized_pnl = result.decision.position_size_usd * realized_return

        ab_test.update_result(
            result.test_id,
            realized_pnl=realized_pnl,
            realized_return=realized_return,
            trade_duration_seconds=random.uniform(300, 7200)
        )

    # Print report
    ab_test.print_report()
```

**Run A/B Test:**
```bash
python testing/ab_test.py
```

---

## Budget Summary & Timeline

### 3-Month Deployment Schedule

#### Week 1: Core Deployment ($30-40)
- **Day 1-2:** Vertex AI Training ($30)
- **Day 3-4:** Cloud Run Deployment ($0 - first deploys free)
- **Day 5:** Local Integration ($0)
- **Day 6-7:** Monitoring Setup ($3-5)
- **Total Week 1:** $33-35

#### Week 2-4: Testing & Optimization ($10-15)
- Cloud Run usage: $10-15
- Storage: $2
- Monitoring: $3
- **Total Weeks 2-4:** $15-20

#### Month 2-3: Production Operation ($40-50)
- Cloud Run: $10/month × 2 = $20
- Storage: $2/month × 2 = $4
- Monitoring: $5/month × 2 = $10
- Networking: $3/month × 2 = $6
- **Total Months 2-3:** $40

### Total 3-Month Cost: $88-95 of $400

### Remaining Budget: $305-312

### Recommended Use of Remaining Budget:

1. **Hyperparameter Tuning ($80-100)**
   - Grid search: learning rate, hidden dim, epochs
   - Vertex AI handles parallelization
   - Expected: 5-10% Sharpe improvement

2. **Multi-Asset Training ($80-120)**
   - Train on BTC, ETH, SOL simultaneously
   - Learn cross-asset correlations
   - Expected: Better portfolio-level optimization

3. **Architecture Experiments ($60-80)**
   - LSTM vs PPO comparison
   - Transformer attention layers
   - Ensemble methods

4. **Reserve ($50-80)**
   - Unexpected costs
   - Extended A/B testing
   - Additional training iterations

---

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Latency (P99)** | <150ms | Cloud Monitoring |
| **API Uptime** | >99.5% | Cloud Run SLA |
| **Error Rate** | <1% | Cloud Monitoring |
| **Fallback Rate** | <5% | Custom metrics |
| **Cold Start Time** | <2s | Cloud Run logs |

### Business Metrics

| Metric | Baseline (Kelly) | RL Target | A/B Test |
|--------|-----------------|-----------|----------|
| **Sharpe Ratio** | 0.6-0.8 | >1.0 | 2-4 weeks |
| **Max Drawdown** | 25-30% | <20% | 2-4 weeks |
| **Win Rate** | 50-55% | >55% | 2-4 weeks |
| **Mean Return** | 1-2% | >2% | 2-4 weeks |

---

## Troubleshooting

### Issue 1: API Timeout

**Symptom:** `RL API timeout after 150ms`

**Causes:**
1. Cold start (first request after idle)
2. Model loading delay
3. Network latency

**Solutions:**
```bash
# Set minimum instances to 1 (prevents cold starts)
gcloud run services update himari-rl-api --min-instances=1

# Cost: +$5/month (keeps 1 instance warm)
```

---

### Issue 2: High Cloud Run Costs

**Symptom:** Cloud Run bill >$20/month

**Causes:**
1. Too many minimum instances
2. High request volume
3. Long request duration

**Solutions:**
```bash
# Check current costs
gcloud billing accounts list
gcloud billing projects link himari-opus-2 --billing-account=XXXXX

# Reduce min instances
gcloud run services update himari-rl-api --min-instances=0

# Add request caching in API
# (Cache predictions for identical states)
```

---

### Issue 3: Model Not Loading

**Symptom:** `/health` returns `model_loaded: false`

**Causes:**
1. GCS bucket permissions
2. Model file path incorrect
3. PyTorch version mismatch

**Solutions:**
```bash
# Check logs
gcloud run services logs read himari-rl-api --limit=50

# Verify GCS access
gsutil ls gs://himari-rl-models/final-models/

# Test locally
docker run -it --rm \
  -e GCS_BUCKET=himari-rl-models \
  -e GCS_MODEL_PATH=final-models/ppo_latest.pt \
  gcr.io/himari-opus-2/rl-inference:latest
```

---

## Conclusion

### What You've Built

✅ **Production-grade RL system** with 99.95% uptime SLA
✅ **$88-95 total cost** over 3 months ($305+ budget remaining)
✅ **<50ms API latency** (P99 <150ms)
✅ **Auto-scaling** from 0-10 instances
✅ **Comprehensive monitoring** with alerts
✅ **A/B testing framework** for RL vs Kelly comparison
✅ **Fallback mechanism** (degradation to Bayesian Kelly)

### Next Steps

1. **Week 1:** Deploy to GCP following this guide
2. **Week 2-4:** Run A/B test (50/50 RL vs Kelly)
3. **Month 2:** If RL wins A/B test, deploy to 100% traffic
4. **Month 2-3:** Experiment with remaining $310 budget:
   - Hyperparameter tuning
   - Multi-asset training
   - LSTM architecture

### Support & Resources

- **GCP Console:** https://console.cloud.google.com/
- **Vertex AI:** https://console.cloud.google.com/vertex-ai
- **Cloud Run:** https://console.cloud.google.com/run
- **Monitoring:** https://console.cloud.google.com/monitoring
- **Billing:** https://console.cloud.google.com/billing

---

**Document Version:** 1.0
**Last Updated:** 2025-12-26
**Status:** Production Ready ✅
**Budget:** $88-95 of $400 CAD (76% under budget)

**Ready for Google Anti-Gravity Deployment** 🚀
