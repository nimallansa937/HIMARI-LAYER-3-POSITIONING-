#!/bin/bash
#
# HIMARI Layer 3: Ensemble Training Script
# =========================================
# 
# Trains 5 models with different random seeds for ensemble inference.
# Each model will be saved to a separate directory.
#
# Usage:
#   chmod +x train_ensemble.sh
#   ./train_ensemble.sh
#

set -e  # Exit on error

# Configuration
STEPS=500000
DEVICE=cuda
DATA_PATH="/tmp/synthetic_data/stress_scenarios.pkl"
OUTPUT_BASE="/tmp/models/ensemble"
SEEDS=(42 123 456 789 1024)

echo "============================================================"
echo "🚀 HIMARI Layer 3: Ensemble Training"
echo "============================================================"
echo "Training ${#SEEDS[@]} models with different seeds"
echo "Steps per model: $STEPS"
echo "Device: $DEVICE"
echo "============================================================"
echo ""

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  Data not found at $DATA_PATH"
    echo "Generating synthetic scenarios first..."
    python src/rl/synthetic_data_generator.py \
        --scenarios 500 \
        --output "$DATA_PATH" \
        --steps 1000
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Train each model
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
echo "🎉 ENSEMBLE TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Trained models:"
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    MODEL_NUM=$((i + 1))
    echo "  Model $MODEL_NUM: ${OUTPUT_BASE}/model_${MODEL_NUM}_seed_${SEED}/best_model.pt"
done
echo ""
echo "Next steps:"
echo "  1. Use bounded_delta_inference.py with EnsembleInference class"
echo "  2. Pass all 5 model paths to average predictions"
echo "============================================================"
