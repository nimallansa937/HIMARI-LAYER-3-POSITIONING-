#!/bin/bash
#================================================================
# HIMARI Layer 3 - Weights & Biases Setup
#================================================================
# Sets up W&B for training monitoring
#================================================================

echo "================================================================"
echo "Setting up Weights & Biases for HIMARI Layer 3 Training"
echo "================================================================"
echo ""

# Install wandb
pip install wandb --quiet

# Login with API key
export WANDB_API_KEY="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"
wandb login $WANDB_API_KEY

echo ""
echo "✅ Weights & Biases configured!"
echo ""
echo "Your training will be logged to:"
echo "  https://wandb.ai/<your-username>/himari-layer3-pretraining"
echo "  https://wandb.ai/<your-username>/himari-layer3-training"
echo ""
echo "================================================================"
