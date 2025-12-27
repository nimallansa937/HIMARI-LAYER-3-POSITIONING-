# Lambda Labs Training - Quick Start Guide

**Cost:** $5-13 (vs $30-60 on GCP)
**Time:** Same (8-15 hours depending on model)
**Setup:** 5 minutes

---

## Step 1: Launch Lambda Labs Instance

1. Go to https://lambdalabs.com/service/gpu-cloud
2. Sign up / Log in
3. Click **"Launch Instance"**
4. Select:
   - **GPU:** A10 (24GB) - $0.60/hour ✅ RECOMMENDED
   - **Region:** US-East or US-West (lowest latency)
   - **OS:** PyTorch (pre-installed)
5. Click **"Launch"**

You'll get SSH credentials:
```
ssh ubuntu@<ip-address>
```

---

## Step 2: Initial Setup (Run Once)

SSH into the instance and run:

```bash
# Download setup script
curl -o setup.sh https://raw.githubusercontent.com/nimallansa937/HIMARI-LAYER-3-POSITIONING-/main/lambda_labs/setup.sh

# Make executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- Clone HIMARI repository
- Install all dependencies
- Verify CUDA and PyTorch
- Test imports

**Time:** 3-5 minutes

---

## Step 3: Train Models

### Option A: Base PPO (Start Here)

```bash
cd ~/HIMARI-LAYER-3-POSITIONING-
chmod +x lambda_labs/train_base_ppo.sh
./lambda_labs/train_base_ppo.sh
```

**Time:** 8-10 hours
**Cost:** ~$5-6
**Expected Sharpe:** 1.2-1.5

### Option B: LSTM-PPO (After Base Succeeds)

```bash
cd ~/HIMARI-LAYER-3-POSITIONING-
chmod +x lambda_labs/train_lstm.sh
./lambda_labs/train_lstm.sh
```

**Time:** 10-12 hours
**Cost:** ~$6-8
**Expected Sharpe:** 1.4-1.8

### Option C: Multi-Asset (Advanced)

```bash
cd ~/HIMARI-LAYER-3-POSITIONING-
chmod +x lambda_labs/train_multiasset.sh
./lambda_labs/train_multiasset.sh
```

**Time:** 12-15 hours
**Cost:** ~$7-10
**Expected Sharpe:** 1.5-2.0

---

## Step 4: Monitor Training

Open a second SSH session and run:

```bash
cd ~/HIMARI-LAYER-3-POSITIONING-
chmod +x lambda_labs/monitor.sh
./lambda_labs/monitor.sh
```

You'll see:
- GPU utilization
- Training progress (Episode X/1000)
- Recent log output
- Estimated completion time

**Or tail logs directly:**
```bash
# Base PPO
tail -f ~/HIMARI-LAYER-3-POSITIONING-/training_base_ppo.log

# LSTM-PPO
tail -f ~/HIMARI-LAYER-3-POSITIONING-/training_lstm_ppo.log

# Multi-Asset
tail -f ~/HIMARI-LAYER-3-POSITIONING-/training_multiasset_ppo.log
```

---

## Step 5: Upload Models to GCS

After training completes:

```bash
cd ~/HIMARI-LAYER-3-POSITIONING-
chmod +x lambda_labs/upload_to_gcs.sh
./lambda_labs/upload_to_gcs.sh
```

This will:
- Install Google Cloud SDK (if needed)
- Upload trained model to `gs://himari-rl-models/`
- Keep timestamped backups

**Alternative (Manual):**
```bash
# Download to your local machine
scp ubuntu@<ip-address>:/tmp/models/ppo_final.pt ./

# Then upload from your machine
gsutil cp ppo_final.pt gs://himari-rl-models/models/himari-rl/ppo_latest.pt
```

---

## Step 6: Terminate Instance

**IMPORTANT:** Terminate instance after training to stop billing!

```bash
# In Lambda Labs dashboard:
# 1. Go to "Instances"
# 2. Click "Terminate" on your instance
# 3. Confirm termination
```

**Billing stops immediately after termination.**

---

## Cost Breakdown

| Model | Training Time | GPU (A10) Cost | Total |
|-------|---------------|----------------|-------|
| **Base PPO** | 8-10 hours | $0.60/hr | **$5-6** |
| **LSTM-PPO** | 10-12 hours | $0.60/hr | **$6-8** |
| **Multi-Asset** | 12-15 hours | $0.60/hr | **$7-10** |
| **All 3 Models** | 30-37 hours | $0.60/hr | **$18-23** |

**Savings vs GCP:** $112-117 (83% cheaper!)

---

## Files Created

| File | Purpose | Usage |
|------|---------|-------|
| **setup.sh** | Initial setup | Run once after SSH |
| **train_base_ppo.sh** | Train Base PPO | Main training script |
| **train_lstm.sh** | Train LSTM-PPO | Advanced model |
| **train_multiasset.sh** | Train Multi-Asset | Portfolio model |
| **monitor.sh** | Monitor training | Real-time monitoring |
| **upload_to_gcs.sh** | Upload to GCS | After training |

---

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# If not showing, restart instance
sudo reboot
```

### Out of Memory Error
```bash
# Reduce batch size in trainer
# Edit vertex_training/trainer.py line 175:
batch_size=32  # Change from 64 to 32
```

### Training Stuck at Episode 1
```bash
# Check if live prices are loading
tail -f training_base_ppo.log | grep "Fetching"

# If stuck, training environment may be downloading price data
# Wait 5-10 minutes for initial data fetch
```

### Model Not Found After Training
```bash
# Check model directory
ls -la /tmp/models/

# Model should be:
# /tmp/models/ppo_final.pt (Base PPO)
# /tmp/models/lstm_ppo_final.pt (LSTM)
# /tmp/models/multiasset_ppo_final.pt (Multi-Asset)
```

---

## Expected Output

```
========================================
HIMARI Layer 3 - Base PPO Training
========================================

Configuration:
  Model: Base PPO
  Episodes: 1000
  GPU: CUDA (auto-detect)
  Estimated time: 8-10 hours
  Estimated cost: $5-6

Starting training...

Episode 10/1000 | Avg Reward: 0.0234 | Avg Sharpe: 0.45
Episode 20/1000 | Avg Reward: 0.0312 | Avg Sharpe: 0.68
Episode 30/1000 | Avg Reward: 0.0445 | Avg Sharpe: 0.89
...
Episode 990/1000 | Avg Reward: 0.0821 | Avg Sharpe: 1.28
Episode 1000/1000 | Avg Reward: 0.0891 | Avg Sharpe: 1.34

========================================
Training Complete!
========================================

Model saved to:
  /tmp/models/ppo_final.pt
```

---

## Quick Commands

```bash
# Setup (run once)
curl -o setup.sh https://raw.githubusercontent.com/nimallansa937/HIMARI-LAYER-3-POSITIONING-/main/lambda_labs/setup.sh && chmod +x setup.sh && ./setup.sh

# Train Base PPO
cd ~/HIMARI-LAYER-3-POSITIONING- && chmod +x lambda_labs/train_base_ppo.sh && ./lambda_labs/train_base_ppo.sh

# Monitor (in separate SSH session)
cd ~/HIMARI-LAYER-3-POSITIONING- && chmod +x lambda_labs/monitor.sh && ./lambda_labs/monitor.sh

# Upload to GCS
cd ~/HIMARI-LAYER-3-POSITIONING- && chmod +x lambda_labs/upload_to_gcs.sh && ./lambda_labs/upload_to_gcs.sh
```

---

## Next Steps After Training

1. ✅ Model uploaded to GCS
2. ⏳ Deploy to Cloud Run (follow [GCP_RL_DEPLOYMENT_GUIDE.md](../GCP_RL_DEPLOYMENT_GUIDE.md) from Step 8)
3. ⏳ Set up monitoring
4. ⏳ Run A/B test vs Bayesian Kelly
5. ⏳ Deploy to production

---

**Total Cost: $5-23 for all 3 models** (vs $130 on GCP)
**Ready to train immediately** (no GPU approval wait)
**Same quality models** (same training code)

Start training now! 🚀
