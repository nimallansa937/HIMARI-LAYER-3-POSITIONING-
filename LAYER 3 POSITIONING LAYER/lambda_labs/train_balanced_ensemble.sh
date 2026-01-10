#!/bin/bash
#
# HIMARI Layer 3: Balanced Ensemble Training
# ==========================================
# 
# Trains 5 models on a balanced dataset (40% Bull, 40% Bear, 20% Range).
# Uses regime-aware rewards (Sortino for Bull) to encourage upside capture.
#
# Usage:
#   chmod +x train_balanced_ensemble.sh
#   ./train_balanced_ensemble.sh
#

set -e

# Configuration
STEPS=500000
DEVICE=cuda
DATA_PATH="/tmp/synthetic_data/balanced_scenarios.pkl"
OUTPUT_BASE="/tmp/models/balanced_ensemble"
SEEDS=(42 123 456 789 1024)

echo "============================================================"
echo "🚀 HIMARI Layer 3: Balanced Ensemble Training"
echo "============================================================"
echo "Strategy: Balanced Regime Training (Bull/Bear/Range)"
echo "Reward: Regime-Aware (Sortino for Bull)"
echo "Steps per model: $STEPS"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# Generate Balanced Data
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  Data not found at $DATA_PATH"
    echo "Generating balanced scenarios (40% Bull, 40% Bear, 20% Range)..."
    python src/rl/balanced_data_generator.py \
        --count 1000 \
        --length 1000 \
        --output "$DATA_PATH"
else
    echo "✅ Balanced data already exists at $DATA_PATH"
    echo "Skipping generation (delete file to regenerate)"
fi

echo ""
mkdir -p "$OUTPUT_BASE"

# Train Ensemble
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    MODEL_NUM=$((i + 1))
    OUTPUT_DIR="${OUTPUT_BASE}/model_${MODEL_NUM}_seed_${SEED}"
    
    echo ""
    echo "============================================================"
    echo "📊 Training Model $MODEL_NUM of ${#SEEDS[@]} (Seed: $SEED)"
    echo "============================================================"
    echo "Output: $OUTPUT_DIR"
    echo ""
    
    python src/rl/pretrain_pipeline_v2.py \
        --steps $STEPS \
        --device $DEVICE \
        --data "$DATA_PATH" \
        --output "$OUTPUT_DIR" \
        --seed $SEED
    
    echo ""
    echo "✅ Model $MODEL_NUM complete!"
    echo ""
done

echo ""
echo "============================================================"
echo "🎉 BALANCED ENSEMBLE COMPLETE!"
echo "============================================================"
echo "Trained models saved in: $OUTPUT_BASE"
echo ""
