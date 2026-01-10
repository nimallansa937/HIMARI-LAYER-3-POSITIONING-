#!/bin/bash
# Lambda Labs - Multi-Asset PPO Training Script
# Trains Multi-Asset PPO agent on BTC/ETH/SOL

set -e

echo "========================================"
echo "HIMARI Layer 3 - Multi-Asset Training"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Model: Multi-Asset PPO"
echo "  Assets: BTC-USD, ETH-USD, SOL-USD"
echo "  Episodes: 1000"
echo "  GPU: CUDA (auto-detect)"
echo "  Estimated time: 12-15 hours"
echo "  Estimated cost: \$7-10"
echo ""
echo "Starting training..."
echo ""

# Navigate to repo
cd ~/HIMARI-LAYER-3-POSITIONING-

# Create local models directory
mkdir -p models

# Run training with logging (using local trainer, no GCS needed)
python3 lambda_labs/trainer_multiasset_local.py \
    --model-dir /tmp/models \
    --symbols BTC-USD ETH-USD SOL-USD \
    --num-episodes 1000 \
    --save-interval 100 \
    --device cuda \
    2>&1 | tee training_multiasset_ppo.log

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Model saved to:"
echo "  /tmp/models/multiasset_ppo_final.pt"
echo ""
echo "To download model:"
echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):/tmp/models/multiasset_ppo_final.pt ./"
echo ""
echo "To upload to GCS:"
echo "  gsutil cp /tmp/models/multiasset_ppo_final.pt gs://himari-rl-models/models/himari-rl-multiasset/multiasset_ppo_latest.pt"
echo ""
