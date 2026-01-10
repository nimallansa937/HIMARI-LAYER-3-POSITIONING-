#!/bin/bash
#================================================================
# HIMARI Layer 3 - Complete Training Pipeline for Lambda Labs A10
#================================================================
# Runs full pipeline: Synthetic Data → Pre-Training → WFO → Final Models
#
# Usage: ./train_complete_pipeline.sh
#
# Timeline: 40-48 hours
# Cost: $24-29 (A10 @ $0.60/hr)
#================================================================

set -e  # Exit on error

echo "================================================================"
echo "HIMARI Layer 3 - Complete Training Pipeline"
echo "================================================================"
echo "Hardware: A10 GPU (24GB VRAM)"
echo "Timeline: 40-48 hours"
echo "Estimated Cost: \$24-29"
echo "================================================================"
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

echo "✅ CUDA detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Directories - Auto-detect repo location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Verify REPO_DIR exists, otherwise use current directory
if [ ! -d "$REPO_DIR" ]; then
    echo "⚠️  Auto-detected path $REPO_DIR not found, using current directory"
    REPO_DIR="$(pwd)"
fi

cd "$REPO_DIR" || { echo "❌ ERROR: Cannot cd to $REPO_DIR"; exit 1; }

SRC_DIR="$REPO_DIR/src"
SYNTHETIC_DATA_DIR="/tmp/synthetic_data"
MODEL_DIR="/tmp/models"
LOG_DIR="/tmp/logs"

# Create directories
mkdir -p "$SYNTHETIC_DATA_DIR"
mkdir -p "$MODEL_DIR/pretrained"
mkdir -p "$MODEL_DIR/wfo"
mkdir -p "$LOG_DIR"

#================================================================
# SETUP: Configure Weights & Biases
#================================================================
echo "================================================================"
echo "Setting up Weights & Biases"
echo "================================================================"
export WANDB_API_KEY="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"
pip install wandb --quiet
wandb login $WANDB_API_KEY
echo "✅ W&B configured"
echo ""

#================================================================
# PHASE 1: Generate Synthetic Stress Scenarios
#================================================================
echo "================================================================"
echo "PHASE 1: Generating Synthetic Stress Scenarios"
echo "================================================================"
echo "Target: 500 scenarios (5M training steps)"
echo "Time: ~10 minutes"
echo ""

START_TIME=$(date +%s)

python src/rl/synthetic_data_generator.py \
    2>&1 | tee "$LOG_DIR/synthetic_generation.log"

if [ ! -f "$SYNTHETIC_DATA_DIR/stress_scenarios.pkl" ]; then
    echo "❌ ERROR: Synthetic data generation failed"
    exit 1
fi

echo ""
echo "✅ Phase 1 Complete: Synthetic data generated"
echo "   Output: $SYNTHETIC_DATA_DIR/stress_scenarios.pkl"
echo "   Size: $(du -h $SYNTHETIC_DATA_DIR/stress_scenarios.pkl | cut -f1)"
echo ""

#================================================================
# PHASE 2: Pre-Training on Synthetic Data
#================================================================
echo "================================================================"
echo "PHASE 2: Pre-Training on Synthetic Data"
echo "================================================================"
echo "Target: 500K steps"
echo "Time: ~2-3 hours"
echo ""

python src/rl/pretrain_pipeline.py \
    --steps 500000 \
    --device cuda \
    --synthetic-data "$SYNTHETIC_DATA_DIR/stress_scenarios.pkl" \
    --output-dir "$MODEL_DIR/pretrained" \
    2>&1 | tee "$LOG_DIR/pretrain.log"

if [ ! -f "$MODEL_DIR/pretrained/pretrained_final.pt" ]; then
    echo "❌ ERROR: Pre-training failed"
    exit 1
fi

echo ""
echo "✅ Phase 2 Complete: Pre-training done"
echo "   Model: $MODEL_DIR/pretrained/pretrained_final.pt"
echo "   Size: $(du -h $MODEL_DIR/pretrained/pretrained_final.pt | cut -f1)"
echo ""

#================================================================
# PHASE 3: Base PPO Training
#================================================================
echo "================================================================"
echo "PHASE 3: Training Base PPO"
echo "================================================================"
echo "Episodes: 1000"
echo "Time: ~8-10 hours"
echo ""

./lambda_labs/train_base_ppo.sh 2>&1 | tee "$LOG_DIR/train_base_ppo.log"

if [ ! -f "/tmp/models/ppo_final.pt" ]; then
    echo "❌ ERROR: Base PPO training failed"
    exit 1
fi

echo ""
echo "✅ Phase 3 Complete: Base PPO trained"
echo "   Model: /tmp/models/ppo_final.pt"
echo ""

#================================================================
# PHASE 4: LSTM-PPO Training
#================================================================
echo "================================================================"
echo "PHASE 4: Training LSTM-PPO"
echo "================================================================"
echo "Episodes: 1000"
echo "Time: ~10-12 hours"
echo ""

./lambda_labs/train_lstm.sh 2>&1 | tee "$LOG_DIR/train_lstm.log"

if [ ! -f "/tmp/models/lstm_ppo_final.pt" ]; then
    echo "❌ ERROR: LSTM-PPO training failed"
    exit 1
fi

echo ""
echo "✅ Phase 4 Complete: LSTM-PPO trained"
echo "   Model: /tmp/models/lstm_ppo_final.pt"
echo ""

#================================================================
# PHASE 5: Multi-Asset PPO Training
#================================================================
echo "================================================================"
echo "PHASE 5: Training Multi-Asset PPO"
echo "================================================================"
echo "Episodes: 1000"
echo "Time: ~12-15 hours"
echo ""

./lambda_labs/train_multiasset.sh 2>&1 | tee "$LOG_DIR/train_multiasset.log"

if [ ! -f "/tmp/models/multiasset_ppo_final.pt" ]; then
    echo "❌ ERROR: Multi-Asset PPO training failed"
    exit 1
fi

echo ""
echo "✅ Phase 5 Complete: Multi-Asset PPO trained"
echo "   Model: /tmp/models/multiasset_ppo_final.pt"
echo ""

#================================================================
# PHASE 6: Walk-Forward Optimization (Optional)
#================================================================
read -p "Run Walk-Forward Optimization? (8 hours, recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "================================================================"
    echo "PHASE 6: Walk-Forward Optimization"
    echo "================================================================"
    echo "Windows: 48 (6-month train, 1-month val)"
    echo "Time: ~8 hours"
    echo ""

    python src/rl/wfo_trainer.py \
        --device cuda \
        --use-pretrained \
        --pretrained-path "$MODEL_DIR/pretrained/pretrained_final.pt" \
        --checkpoint-dir "$MODEL_DIR/wfo" \
        --episodes-per-window 100 \
        2>&1 | tee "$LOG_DIR/wfo_training.log"

    if [ ! -f "$MODEL_DIR/wfo/wfo_final.pt" ]; then
        echo "⚠️  WARNING: WFO training failed, but continuing..."
    else
        echo ""
        echo "✅ Phase 6 Complete: WFO trained"
        echo "   Model: $MODEL_DIR/wfo/wfo_final.pt"
        echo ""
    fi
fi

#================================================================
# SUMMARY
#================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "================================================================"
echo "🎉 TRAINING PIPELINE COMPLETE!"
echo "================================================================"
echo ""
echo "📊 Training Summary:"
echo "  Total Time: ${HOURS}h ${MINUTES}m"
echo "  Estimated Cost: \$$(echo "scale=2; $HOURS * 0.60" | bc)"
echo ""
echo "📁 Trained Models:"
echo "  ✅ Base PPO:       /tmp/models/ppo_final.pt"
echo "  ✅ LSTM-PPO:       /tmp/models/lstm_ppo_final.pt"
echo "  ✅ Multi-Asset:    /tmp/models/multiasset_ppo_final.pt"
if [ -f "$MODEL_DIR/wfo/wfo_final.pt" ]; then
    echo "  ✅ WFO:            $MODEL_DIR/wfo/wfo_final.pt"
fi
echo ""
echo "📈 Synthetic Data:"
echo "  ✅ 500 scenarios:  $SYNTHETIC_DATA_DIR/stress_scenarios.pkl"
echo ""
echo "🧠 Pre-Trained:"
echo "  ✅ Weights:        $MODEL_DIR/pretrained/pretrained_final.pt"
echo ""
echo "📝 Logs:"
echo "  Directory:         $LOG_DIR/"
echo ""
echo "================================================================"
echo "Next Steps:"
echo "================================================================"
echo "1. Test models with paper trading:"
echo "   python src/engines/execution_engine.py --mode paper --model-path /tmp/models/lstm_ppo_final.pt"
echo ""
echo "2. Upload to Google Cloud Storage:"
echo "   gsutil cp /tmp/models/*.pt gs://himari-rl-models/models/"
echo ""
echo "3. Deploy to Cloud Run:"
echo "   Follow GCP_RL_DEPLOYMENT_GUIDE.md"
echo ""
echo "================================================================"
echo "🚀 Ready for deployment!"
echo "================================================================"
