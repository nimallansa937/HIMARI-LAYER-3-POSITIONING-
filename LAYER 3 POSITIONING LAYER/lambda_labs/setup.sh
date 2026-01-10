#!/bin/bash
# Lambda Labs - Initial Setup Script
# Run this once after SSH into Lambda Labs instance

set -e

echo "========================================"
echo "HIMARI Layer 3 - Lambda Labs Setup"
echo "========================================"
echo ""

# Update system
echo "1. Updating system packages..."
sudo apt-get update -qq

# Install required system packages
echo "2. Installing system dependencies..."
sudo apt-get install -y git wget curl nano htop -qq

# Verify CUDA and PyTorch
echo "3. Verifying CUDA installation..."
nvidia-smi
echo ""

# Clone repository
echo "4. Cloning HIMARI repository..."
if [ -d "HIMARI-LAYER-3-POSITIONING-" ]; then
    echo "   Repository already exists, pulling latest changes..."
    cd HIMARI-LAYER-3-POSITIONING-
    git pull
    cd ..
else
    git clone https://github.com/nimallansa937/HIMARI-LAYER-3-POSITIONING-.git
fi

cd HIMARI-LAYER-3-POSITIONING-

# Install Python dependencies
echo "5. Installing Python dependencies..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install numpy pandas matplotlib -q
pip install google-cloud-storage -q
pip install requests pyyaml -q

# Verify installations
echo "6. Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Set PYTHONPATH for imports
export PYTHONPATH="$HOME/HIMARI-LAYER-3-POSITIONING-/src:$PYTHONPATH"
echo "export PYTHONPATH=\"\$HOME/HIMARI-LAYER-3-POSITIONING-/src:\$PYTHONPATH\"" >> ~/.bashrc

# Test imports
echo "7. Testing HIMARI imports..."
python3 -c "from rl.ppo_agent import PPOAgent; print('✓ PPOAgent OK')"
python3 -c "from rl.trading_env import TradingEnvironment; print('✓ TradingEnvironment OK')"
python3 -c "from rl.lstm_ppo_agent import LSTMPPOAgent; print('✓ LSTMPPOAgent OK')"
python3 -c "from rl.multi_asset_env import MultiAssetTradingEnv; print('✓ MultiAssetTradingEnv OK')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run training: ./lambda_labs/train_base_ppo.sh"
echo "2. Monitor progress: tail -f training_base_ppo.log"
echo "3. Download model after completion"
echo ""
