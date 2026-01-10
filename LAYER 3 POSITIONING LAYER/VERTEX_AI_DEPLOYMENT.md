# Vertex AI Deployment Instructions for Anti-Gravity

**Status:** Ready to deploy on GCP
**Location:** Cloud Shell
**Estimated Time:** 10-15 minutes setup + 8-10 hours training

---

## Step 1: Prerequisites Check

Run these commands in Cloud Shell to verify everything is ready:

```bash
# Verify you're in the repo
cd ~/HIMARI-LAYER-3-POSITIONING-
pwd

# Check Python version (should be 3.8+)
python3 --version

# List training files
ls -la vertex_training/

# Expected output:
# trainer.py
# trainer_lstm.py
# trainer_multiasset.py
# launch_training.py
```

---

## Step 2: Set Up GCP Project

```bash
# Set your project ID
export PROJECT_ID="himari-opus-position-layer"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Set region
export REGION="us-central1"
gcloud config set ai/region $REGION
```

---

## Step 3: Create GCS Bucket for Models

```bash
# Create bucket for model storage
export BUCKET_NAME="himari-rl-models"
gsutil mb -l $REGION gs://$BUCKET_NAME/

# Verify bucket created
gsutil ls gs://$BUCKET_NAME/
```

---

## Step 4: Install Vertex AI SDK

```bash
# Install only the Vertex AI SDK (minimal dependencies)
pip3 install google-cloud-aiplatform --user --quiet

# Verify installation
python3 -c "from google.cloud import aiplatform; print('✓ Vertex AI SDK installed')"
```

---

## Step 5: Build Docker Container

Vertex AI needs a Docker container with all dependencies.

**Option A: Use Pre-built PyTorch Container (Fastest ✅)**

```bash
# Use Google's pre-built PyTorch container
export CONTAINER_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"

echo "Using pre-built container: $CONTAINER_URI"
```

**Option B: Build Custom Container (If Option A Fails)**

```bash
# Navigate to vertex_training directory
cd vertex_training/

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Copy requirements
COPY ../requirements-gcp.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ../src /app/src
COPY trainer.py /app/

# Set Python path
ENV PYTHONPATH=/app:/app/src

ENTRYPOINT ["python", "trainer.py"]
EOF

# Build and push container
export IMAGE_URI="gcr.io/$PROJECT_ID/himari-rl-trainer:latest"

gcloud builds submit --tag $IMAGE_URI .

# Return to root directory
cd ~/HIMARI-LAYER-3-POSITIONING-
```

---

## Step 6: Submit Training Job (Base PPO)

```bash
# Set training parameters
export JOB_NAME="himari-rl-ppo-$(date +%Y%m%d-%H%M%S)"
export MODEL_DIR="models/himari-rl"

# Create training job config
cat > training_job.yaml << EOF
displayName: $JOB_NAME
jobSpec:
  workerPoolSpecs:
    - machineSpec:
        machineType: n1-standard-4
        acceleratorType: NVIDIA_TESLA_T4
        acceleratorCount: 1
      replicaCount: 1
      containerSpec:
        imageUri: $CONTAINER_URI
        args:
          - --bucket-name=$BUCKET_NAME
          - --model-dir=$MODEL_DIR
          - --num-episodes=1000
          - --save-interval=100
EOF

# Submit job
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --config=training_job.yaml

echo "✓ Training job submitted: $JOB_NAME"
```

---

## Step 7: Monitor Training Progress

```bash
# List all training jobs
gcloud ai custom-jobs list --region=$REGION

# Get specific job details
gcloud ai custom-jobs describe $JOB_NAME --region=$REGION

# Stream logs (real-time monitoring)
gcloud ai custom-jobs stream-logs $JOB_NAME --region=$REGION
```

**Expected Output:**
```
Episode 10/1000 | Avg Reward: 0.0234 | Avg Sharpe: 0.45
Episode 20/1000 | Avg Reward: 0.0312 | Avg Sharpe: 0.68
Episode 30/1000 | Avg Reward: 0.0445 | Avg Sharpe: 0.89
...
Episode 1000/1000 | Avg Reward: 0.0891 | Avg Sharpe: 1.34
✓ Training complete!
Model uploaded to gs://himari-rl-models/models/himari-rl/ppo_latest.pt
```

---

## Step 8: Alternative - Use Python Script

If the YAML approach doesn't work, use the launch script:

```bash
# Make sure you're in the repo root
cd ~/HIMARI-LAYER-3-POSITIONING-

# Run the launch script
python3 vertex_training/launch_training.py
```

**Note:** You may need to edit `launch_training.py` to update the project ID:

```bash
# Edit the file
nano vertex_training/launch_training.py

# Change line 11:
# FROM: project='himari-opus-2',
# TO:   project='himari-opus-position-layer',

# Save: Ctrl+O, Enter
# Exit: Ctrl+X
```

---

## Step 9: Verify Training Results

After training completes (~8-10 hours):

```bash
# Check if model was uploaded
gsutil ls gs://$BUCKET_NAME/models/himari-rl/

# Expected files:
# ppo_latest.pt
# ppo_final_20251227_HHMMSS.pt
# ppo_ep100.pt
# ppo_ep200.pt
# ...

# Download final model (optional, for inspection)
gsutil cp gs://$BUCKET_NAME/models/himari-rl/ppo_latest.pt ./

# Check model file size
ls -lh ppo_latest.pt
# Expected: ~2-5 MB
```

---

## Step 10: Deploy to Cloud Run API

Once training succeeds, deploy the model:

```bash
# Navigate to Cloud Run directory
cd ~/HIMARI-LAYER-3-POSITIONING-/cloud_run

# Deploy API server
gcloud run deploy himari-rl-api \
  --source=. \
  --region=$REGION \
  --memory=1Gi \
  --cpu=2 \
  --max-instances=10 \
  --allow-unauthenticated \
  --set-env-vars="MODEL_PATH=gs://$BUCKET_NAME/models/himari-rl/ppo_latest.pt"

# Get API endpoint
export API_URL=$(gcloud run services describe himari-rl-api \
  --region=$REGION \
  --format='value(status.url)')

echo "✓ API deployed at: $API_URL"
```

---

## Step 11: Test API

```bash
# Test prediction endpoint
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": [0.87, 0.02, 0.05, 0.08, 0.01, 0.02, 0.03, 0.5, 0.01, 0.02, 0.03, 1.2, 0.0, 0.0, 0.0, 0.0]
  }'

# Expected response:
# {
#   "multiplier": 1.23,
#   "source": "gcp_api",
#   "latency_ms": 12.5
# }
```

---

## Troubleshooting

### Error: "No module named 'torch'"
**Solution:** You're trying to run training directly in Cloud Shell. Use Vertex AI instead (Steps 6-7).

### Error: "No space left on device"
**Solution:** Clean up Cloud Shell or use Vertex AI (recommended).

```bash
# Clean up Cloud Shell
rm -rf ~/.cache/*
docker system prune -a -f
```

### Error: "Permission denied" when creating bucket
**Solution:** Check billing is enabled and you have Storage Admin role.

```bash
# Enable billing
gcloud billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID

# Grant yourself Storage Admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$(gcloud config get-value account)" \
  --role="roles/storage.admin"
```

### Training job fails immediately
**Solution:** Check logs for errors.

```bash
# View job logs
gcloud ai custom-jobs stream-logs $JOB_NAME --region=$REGION

# Common issues:
# - Container image not found → Use pre-built container (Step 5 Option A)
# - Import errors → Check that src/rl/ files are in the container
# - Permission errors → Grant Vertex AI service account Storage permissions
```

### Training is very slow (>20 hours)
**Solution:** Make sure you're using GPU (T4), not CPU.

```bash
# Check job configuration
gcloud ai custom-jobs describe $JOB_NAME --region=$REGION | grep accelerator

# Should show:
# acceleratorType: NVIDIA_TESLA_T4
# acceleratorCount: 1
```

---

## Cost Tracking

Monitor costs in real-time:

```bash
# View current month's charges
gcloud billing accounts list
gcloud billing projects describe $PROJECT_ID

# Or visit:
https://console.cloud.google.com/billing
```

**Expected costs:**
- Training (8-10 hours, T4 GPU): ~$30
- Storage (100GB/month): ~$2
- Cloud Run (first 2M requests free): $0-10

**Total for Base PPO:** ~$32-40

---

## Quick Commands Summary

```bash
# 1. Setup
export PROJECT_ID="himari-opus-position-layer"
export BUCKET_NAME="himari-rl-models"
export REGION="us-central1"

# 2. Enable APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

# 3. Create bucket
gsutil mb -l $REGION gs://$BUCKET_NAME/

# 4. Submit training
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name="himari-rl-ppo-$(date +%Y%m%d-%H%M%S)" \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest \
  --args="--bucket-name=$BUCKET_NAME,--model-dir=models/himari-rl,--num-episodes=1000"

# 5. Monitor
gcloud ai custom-jobs list --region=$REGION
```

---

## Success Criteria

✅ Training job submitted successfully
✅ Logs show "Episode 10/1000" progressing
✅ No errors in logs
✅ Model uploaded to GCS after completion
✅ Sharpe ratio > 1.0 in final logs
✅ API server deployed and responding

---

## Next Steps After Base PPO Succeeds

1. Review training metrics (Sharpe, win rate, drawdown)
2. Deploy API to Cloud Run
3. Set up monitoring dashboard
4. Run A/B test vs Bayesian Kelly
5. Consider training LSTM or Multi-Asset (optional)

---

**Estimated Total Time:** 8-12 hours (mostly unattended GPU training)
**Estimated Cost:** $30-40
**Expected Sharpe Ratio:** 1.2-1.5

Good luck! 🚀
