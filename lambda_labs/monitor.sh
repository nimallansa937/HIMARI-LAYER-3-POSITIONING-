#!/bin/bash
# Lambda Labs - Training Monitor
# Monitor GPU usage and training progress

echo "========================================"
echo "HIMARI Training Monitor"
echo "========================================"
echo ""

# Function to show GPU stats
show_gpu() {
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
}

# Function to show training logs
show_logs() {
    if [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_base_ppo.log ]; then
        echo "Base PPO Training (last 20 lines):"
        tail -n 20 ~/HIMARI-LAYER-3-POSITIONING-/training_base_ppo.log
    elif [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_lstm_ppo.log ]; then
        echo "LSTM-PPO Training (last 20 lines):"
        tail -n 20 ~/HIMARI-LAYER-3-POSITIONING-/training_lstm_ppo.log
    elif [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_multiasset_ppo.log ]; then
        echo "Multi-Asset Training (last 20 lines):"
        tail -n 20 ~/HIMARI-LAYER-3-POSITIONING-/training_multiasset_ppo.log
    else
        echo "No training logs found"
    fi
    echo ""
}

# Function to estimate completion time
estimate_time() {
    if [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_base_ppo.log ]; then
        LOG_FILE=~/HIMARI-LAYER-3-POSITIONING-/training_base_ppo.log
    elif [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_lstm_ppo.log ]; then
        LOG_FILE=~/HIMARI-LAYER-3-POSITIONING-/training_lstm_ppo.log
    elif [ -f ~/HIMARI-LAYER-3-POSITIONING-/training_multiasset_ppo.log ]; then
        LOG_FILE=~/HIMARI-LAYER-3-POSITIONING-/training_multiasset_ppo.log
    else
        return
    fi

    # Extract current episode
    CURRENT_EP=$(grep -oP 'Episode \K\d+(?=/1000)' $LOG_FILE | tail -1)
    if [ ! -z "$CURRENT_EP" ]; then
        PROGRESS=$(echo "scale=1; $CURRENT_EP / 10" | bc)
        REMAINING=$(echo "1000 - $CURRENT_EP" | bc)
        echo "Progress: Episode $CURRENT_EP/1000 ($PROGRESS%)"
        echo "Remaining: $REMAINING episodes"
        echo ""
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "========================================"
    echo "HIMARI Training Monitor"
    echo "Refresh every 10 seconds (Ctrl+C to exit)"
    echo "========================================"
    echo ""

    show_gpu
    estimate_time
    show_logs

    echo "Press Ctrl+C to exit monitoring"
    sleep 10
done
