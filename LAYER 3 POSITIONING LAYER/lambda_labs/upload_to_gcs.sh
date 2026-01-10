#!/bin/bash
# Lambda Labs - Upload trained models to GCS
# Run after training completes

set -e

echo "========================================"
echo "Upload Models to Google Cloud Storage"
echo "========================================"
echo ""

# Check if gcloud is installed
if ! command -v gsutil &> /dev/null; then
    echo "Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
    gcloud init
fi

# Set project
export PROJECT_ID="himari-opus-position-layer"
export BUCKET_NAME="himari-rl-models"

gcloud config set project $PROJECT_ID

echo "Uploading models to gs://$BUCKET_NAME/"
echo ""

# Upload Base PPO if exists
if [ -f /tmp/models/ppo_final.pt ]; then
    echo "1. Uploading Base PPO model..."
    gsutil cp /tmp/models/ppo_final.pt gs://$BUCKET_NAME/models/himari-rl/ppo_latest.pt
    gsutil cp /tmp/models/ppo_final.pt gs://$BUCKET_NAME/models/himari-rl/ppo_final_$(date +%Y%m%d_%H%M%S).pt
    echo "   ✓ Base PPO uploaded"
else
    echo "1. Base PPO model not found, skipping..."
fi

# Upload LSTM-PPO if exists
if [ -f /tmp/models/lstm_ppo_final.pt ]; then
    echo "2. Uploading LSTM-PPO model..."
    gsutil cp /tmp/models/lstm_ppo_final.pt gs://$BUCKET_NAME/models/himari-rl-lstm/lstm_ppo_latest.pt
    gsutil cp /tmp/models/lstm_ppo_final.pt gs://$BUCKET_NAME/models/himari-rl-lstm/lstm_ppo_final_$(date +%Y%m%d_%H%M%S).pt
    echo "   ✓ LSTM-PPO uploaded"
else
    echo "2. LSTM-PPO model not found, skipping..."
fi

# Upload Multi-Asset if exists
if [ -f /tmp/models/multiasset_ppo_final.pt ]; then
    echo "3. Uploading Multi-Asset PPO model..."
    gsutil cp /tmp/models/multiasset_ppo_final.pt gs://$BUCKET_NAME/models/himari-rl-multiasset/multiasset_ppo_latest.pt
    gsutil cp /tmp/models/multiasset_ppo_final.pt gs://$BUCKET_NAME/models/himari-rl-multiasset/multiasset_ppo_final_$(date +%Y%m%d_%H%M%S).pt
    echo "   ✓ Multi-Asset PPO uploaded"
else
    echo "3. Multi-Asset PPO model not found, skipping..."
fi

echo ""
echo "========================================"
echo "Upload Complete!"
echo "========================================"
echo ""
echo "View models at:"
echo "  https://console.cloud.google.com/storage/browser/$BUCKET_NAME/models"
echo ""
echo "List uploaded models:"
echo "  gsutil ls gs://$BUCKET_NAME/models/"
echo ""
