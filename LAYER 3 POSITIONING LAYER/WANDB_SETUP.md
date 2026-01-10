# Weights & Biases (wandb) Integration ✅

## **Setup Complete**

Your wandb API key has been integrated into all training scripts.

**API Key:** `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`

---

## **What's Logged to W&B**

### **Pre-Training (`himari-layer3-pretraining` project)**
- Episode number
- Total training steps
- Progress percentage
- Average reward (last 100 episodes)
- Average Sharpe ratio (last 100 episodes)
- Per-episode reward
- Per-episode Sharpe ratio
- Scenario type (bull/bear/crash/mixed)

### **Main Training (`himari-layer3-training` project)**
- Same metrics as pre-training
- Plus model-specific metrics (Base PPO, LSTM, Multi-Asset)

---

## **Viewing Your Training**

### **During Training:**

1. Open browser: https://wandb.ai/
2. Login with your account
3. Navigate to projects:
   - **Pre-Training:** `himari-layer3-pretraining`
   - **Main Training:** `himari-layer3-training`

### **Real-Time Monitoring:**

You'll see charts for:
- **Sharpe Ratio** (trending up = learning)
- **Reward** (should increase over time)
- **Progress** (steps/episodes completed)
- **Scenario Distribution** (which scenarios being trained on)

---

## **Automatic Setup**

When you run the training pipeline, wandb is configured automatically:

```bash
./lambda_labs/train_complete_pipeline.sh
```

The script:
1. ✅ Installs wandb
2. ✅ Logs in with your API key
3. ✅ Initializes logging for each training phase
4. ✅ Uploads metrics in real-time

---

## **Manual Setup (if needed)**

```bash
# Install wandb
pip install wandb

# Login
export WANDB_API_KEY="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"
wandb login $WANDB_API_KEY
```

---

## **W&B Dashboard Features**

### **Charts You'll See:**

1. **Sharpe Ratio Over Time**
   - Target: Should reach 1.0+ for Base PPO
   - Target: Should reach 1.4+ for LSTM-PPO

2. **Episode Reward**
   - Shows learning progress
   - Smoothed average (last 100 episodes)

3. **Training Progress**
   - Total steps vs target (500K for pre-training)
   - ETA to completion

4. **Scenario Distribution**
   - Shows mix of bull/bear/crash scenarios
   - Should be balanced (40% mixed, 20% crash, etc.)

### **System Metrics:**

- GPU utilization
- Memory usage
- Training speed (steps/sec)

---

## **Comparing Runs**

W&B allows you to compare multiple training runs:

1. Select multiple runs in dashboard
2. Click "Compare"
3. View side-by-side charts

**Use Cases:**
- Compare Base PPO vs LSTM-PPO performance
- Compare different hyperparameters
- Find best checkpoint

---

## **Files Modified**

1. ✅ `src/rl/pretrain_pipeline.py` - Added wandb logging
2. ✅ `lambda_labs/train_complete_pipeline.sh` - Auto-setup wandb
3. ✅ `setup_wandb.sh` - Standalone setup script

---

## **Training Command with W&B**

```bash
# Pre-training (with wandb)
python src/rl/pretrain_pipeline.py \
    --steps 500000 \
    --device cuda \
    --output-dir /tmp/models/pretrained

# W&B is enabled by default
# To disable: add --no-wandb flag (not recommended)
```

---

## **Expected W&B Output**

```
================================================================
Setting up Weights & Biases
================================================================
✅ W&B configured

================================================================
Starting Pre-Training on Synthetic Data
================================================================
✅ Weights & Biases logging enabled
Target steps: 500,000
Scenarios: 500
Device: cuda
================================================================

Episode   10 | Steps:    5000/500000 (  1.0%) | Reward:  0.023 | Sharpe:  0.45
Episode   20 | Steps:   10000/500000 (  2.0%) | Reward:  0.031 | Sharpe:  0.68
...

✅ W&B run finished
```

---

## **Troubleshooting**

### **"wandb not installed"**
```bash
pip install wandb
```

### **"wandb login failed"**
```bash
# Check API key is correct
echo $WANDB_API_KEY

# Re-login
wandb login $WANDB_API_KEY
```

### **"No metrics showing in dashboard"**
- Wait 30-60 seconds for first metrics
- Check internet connection on Lambda Labs
- Verify wandb.log() is being called (check logs)

---

## **API Key Security**

⚠️ **Your API key is embedded in scripts for convenience during training.**

After training completes, you should:
1. Rotate your API key in W&B settings
2. Remove key from scripts before committing to git

---

## **Cost**

- **W&B:** FREE for individual use
- **Data logged:** ~10MB per training run
- **Retention:** Unlimited

---

## **Quick Links**

- **W&B Dashboard:** https://wandb.ai/
- **W&B Docs:** https://docs.wandb.ai/
- **Your Projects:**
  - Pre-Training: https://wandb.ai/<your-username>/himari-layer3-pretraining
  - Training: https://wandb.ai/<your-username>/himari-layer3-training

---

**Setup Status:** ✅ **READY**

Your training will automatically log to W&B. No additional configuration needed!
