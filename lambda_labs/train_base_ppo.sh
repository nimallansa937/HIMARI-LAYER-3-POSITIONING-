#!/bin/bash
# Lambda Labs - Base PPO Training Script
# Trains PPO agent on BTC-USD

set -e

echo "========================================"
echo "HIMARI Layer 3 - Base PPO Training"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Model: Base PPO"
echo "  Episodes: 1000"
echo "  GPU: CUDA (auto-detect)"
echo "  Estimated time: 8-10 hours"
echo "  Estimated cost: \$5-6"
echo ""
echo "Starting training..."
echo ""

# Navigate to repo
cd ~/HIMARI-LAYER-3-POSITIONING-

# Create local models directory
mkdir -p models

# Run training with logging (using local trainer, no GCS needed)
python3 lambda_labs/trainer_local.py \
    --model-dir /tmp/models \
    --num-episodes 1000 \
    --save-interval 100 \
    --device cuda \
    2>&1 | tee training_base_ppo.log

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Model saved to:"
echo "  /tmp/models/ppo_final.pt"
echo ""
echo "To download model:"
echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):/tmp/models/ppo_final.pt ./"
echo ""
echo "To upload to GCS:"
echo "  gsutil cp /tmp/models/ppo_final.pt gs://himari-rl-models/models/himari-rl/ppo_latest.pt"
echo ""
