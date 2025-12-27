# HIMARI Layer 3 - Advanced RL Models Guide

**Date:** 2025-12-27
**Models:** LSTM-PPO, Multi-Asset PPO
**Status:** Ready for Training

---

## Overview

This guide covers two advanced RL model variants for HIMARI Layer 3:

1. **LSTM-PPO**: Adds temporal memory to capture market dynamics over time
2. **Multi-Asset PPO**: Trains on multiple cryptocurrencies to learn cross-asset correlations

These are experimental improvements over the base PPO model. Train them AFTER the base model succeeds.

---

## Model 1: LSTM-PPO (Temporal Memory)

### What It Does

Adds LSTM (Long Short-Term Memory) layers to the PPO agent, allowing it to:
- Remember past market states
- Capture temporal patterns (trends, momentum)
- Make decisions based on sequences, not just current state

### Architecture

```
Input State (16-dim)
    ↓
LSTM Layers (2 layers, 64 hidden units)
    ↓
    ├─→ Policy Head → Action (continuous [0,2])
    └─→ Value Head → State Value
```

**Key Differences from Base PPO:**
- **Base PPO**: Feedforward network, memoryless
- **LSTM-PPO**: Recurrent network, remembers past 10-50 timesteps

### Files

| File | Purpose | Location |
|------|---------|----------|
| **lstm_ppo_agent.py** | LSTM agent implementation | [src/rl/lstm_ppo_agent.py](src/rl/lstm_ppo_agent.py) |
| **trainer_lstm.py** | Vertex AI training script | [vertex_training/trainer_lstm.py](vertex_training/trainer_lstm.py) |

### Configuration

```python
LSTMPPOConfig(
    state_dim=16,
    action_dim=1,
    hidden_dim=128,         # Feedforward hidden size
    lstm_hidden_dim=64,     # LSTM hidden size
    lstm_num_layers=2,      # Number of LSTM layers
    learning_rate=3e-4,
    gamma=0.99,
)
```

### Training Command

```bash
# On Vertex AI (GCP Cloud Shell)
python vertex_training/trainer_lstm.py \
    --bucket-name himari-rl-models \
    --model-dir models/himari-rl-lstm \
    --num-episodes 1000 \
    --save-interval 100
```

### Expected Performance

| Metric | Base PPO | LSTM-PPO (Expected) |
|--------|----------|---------------------|
| **Sharpe Ratio** | 1.2-1.5 | 1.4-1.8 |
| **Training Time** | 8-10 hours | 10-12 hours |
| **Memory Usage** | 2GB | 3GB |
| **Latency (inference)** | 5ms | 8ms |

**When to Use LSTM:**
- ✅ Markets with strong momentum/trend
- ✅ When temporal patterns matter
- ❌ High-frequency trading (latency sensitive)
- ❌ Very noisy markets (overfitting risk)

---

## Model 2: Multi-Asset PPO (Portfolio Learning)

### What It Does

Trains a single agent on multiple cryptocurrencies simultaneously:
- Learns cross-asset correlations
- Discovers portfolio-level patterns
- Shares knowledge across assets

### Architecture

```
Input State (16 × 3 assets + 4 correlation features = 52-dim)
    ↓
Feedforward Network (256 hidden units, larger than base)
    ↓
    ├─→ Policy Head → Actions [BTC, ETH, SOL] (3 multipliers)
    └─→ Value Head → Portfolio Value
```

**Key Differences from Base PPO:**
- **Base PPO**: Single asset (BTC-USD)
- **Multi-Asset**: 3+ assets (BTC, ETH, SOL)

### Files

| File | Purpose | Location |
|------|---------|----------|
| **multi_asset_env.py** | Multi-asset environment | [src/rl/multi_asset_env.py](src/rl/multi_asset_env.py) |
| **trainer_multiasset.py** | Vertex AI training script | [vertex_training/trainer_multiasset.py](vertex_training/trainer_multiasset.py) |

### State Space (52-dim for 3 assets)

**Per-Asset Features (16 × 3 = 48):**
- Price, Returns (1h, 4h, 24h), Volatility (1h, 4h, 24h)
- RSI, MACD, Signal, Histogram
- Volume ratio, Position size, Entry distance, Unrealized P&L, Allocation %

**Cross-Asset Features (+4):**
- BTC-ETH correlation
- BTC-SOL correlation
- ETH-SOL correlation
- Portfolio concentration (Herfindahl index)

### Action Space

One position multiplier [0, 2] per asset:
- `[1.5, 0.8, 1.2]` = 1.5× BTC, 0.8× ETH, 1.2× SOL

### Configuration

```python
MultiAssetEnvConfig(
    symbols=['BTC-USD', 'ETH-USD', 'SOL-USD'],
    initial_capital=100000.0,
    max_position_pct=0.5,  # 50% max per asset
    commission_rate=0.001,
    use_live_prices=True,  # Fetch from CoinGecko
)

PPOConfig(
    state_dim=52,  # 16×3 + 4
    action_dim=3,  # One per asset
    hidden_dim=256,  # Larger network
)
```

### Training Command

```bash
# On Vertex AI (GCP Cloud Shell)
python vertex_training/trainer_multiasset.py \
    --bucket-name himari-rl-models \
    --model-dir models/himari-rl-multiasset \
    --symbols BTC-USD ETH-USD SOL-USD \
    --num-episodes 1000 \
    --save-interval 100
```

### Expected Performance

| Metric | Single Asset | Multi-Asset (Expected) |
|--------|--------------|------------------------|
| **Sharpe Ratio** | 1.2-1.5 | 1.5-2.0 |
| **Diversification** | 0% | 30-50% |
| **Training Time** | 8-10 hours | 12-15 hours |
| **Max Drawdown** | 15-20% | 10-15% |

**When to Use Multi-Asset:**
- ✅ Portfolio-level optimization
- ✅ Risk diversification across assets
- ✅ Learning market structure
- ❌ Single-asset focus
- ❌ Limited budget (more expensive to train)

---

## Training Sequence Recommendation

### Phase 1: Base PPO (Week 1)
1. Train base PPO on BTC-USD
2. Verify Sharpe > 1.0
3. Deploy to Cloud Run API
4. Run A/B test vs Bayesian Kelly

**Cost:** ~$30

### Phase 2: LSTM-PPO (Week 2-3)
1. Train LSTM-PPO on BTC-USD
2. Compare Sharpe vs Base PPO
3. If LSTM Sharpe > Base Sharpe + 0.2, deploy
4. Otherwise, keep Base PPO

**Cost:** ~$40

### Phase 3: Multi-Asset (Week 4-6)
1. Train Multi-Asset on BTC+ETH+SOL
2. Verify portfolio Sharpe > 1.5
3. Check diversification (Herfindahl < 0.6)
4. If successful, deploy for portfolio trading

**Cost:** ~$60

**Total Cost:** ~$130 of $400 budget

---

## Cost Comparison

| Model | Episodes | Training Time | GPU Cost | Total Cost |
|-------|----------|---------------|----------|------------|
| **Base PPO** | 1000 | 8-10 hours | T4 GPU | $30 |
| **LSTM-PPO** | 1000 | 10-12 hours | T4 GPU | $40 |
| **Multi-Asset** | 1000 | 12-15 hours | T4 GPU | $60 |
| **All 3 Models** | 3000 | 30-37 hours | T4 GPU | $130 |

**Budget Remaining:** $270 for hyperparameter tuning and retraining

---

## Deployment

### LSTM-PPO Deployment

After training, the model is uploaded to:
```
gs://himari-rl-models/models/himari-rl-lstm/lstm_ppo_latest.pt
```

**To deploy on Cloud Run:**

1. Update [cloud_run/api_server.py](cloud_run/api_server.py) to load LSTM model:
```python
from rl.lstm_ppo_agent import LSTMPPOAgent, LSTMPPOConfig

# Load LSTM model
config = LSTMPPOConfig(state_dim=16, action_dim=1)
agent = LSTMPPOAgent(config, device='cpu')
agent.load('lstm_ppo_latest.pt')
```

2. Deploy:
```bash
gcloud run deploy himari-rl-api-lstm \
    --source=cloud_run \
    --region=us-central1 \
    --memory=1Gi
```

### Multi-Asset Deployment

Multi-Asset model outputs multiple position sizes (one per asset). Integration requires modifying Layer 3 to handle portfolio allocation.

**Integration Steps:**

1. Modify [src/phases/phase1_rl_enhanced.py](src/phases/phase1_rl_enhanced.py):
```python
# Instead of single BTC position
decision = self.calculate_position(signal, ...)

# Use multi-asset allocations
allocations = agent.select_action(state)  # [1.2, 0.8, 1.5] for BTC/ETH/SOL
btc_position = kelly_position * allocations[0]
eth_position = kelly_position * allocations[1]
sol_position = kelly_position * allocations[2]
```

2. Deploy separate API endpoint:
```bash
gcloud run deploy himari-rl-api-multiasset \
    --source=cloud_run \
    --region=us-central1 \
    --memory=2Gi
```

---

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Single BTC trading** | Base PPO | Simplest, fastest |
| **Trending markets** | LSTM-PPO | Captures momentum |
| **Portfolio management** | Multi-Asset PPO | Diversification |
| **High-frequency trading** | Base PPO | Lowest latency |
| **Risk-averse trading** | Multi-Asset PPO | Lower drawdowns |

---

## Hyperparameter Tuning (Optional)

If you have remaining budget, experiment with:

### LSTM-PPO Tuning

```python
# Try different LSTM sizes
lstm_hidden_dim: [32, 64, 128]
lstm_num_layers: [1, 2, 3]

# Try different learning rates
learning_rate: [1e-4, 3e-4, 1e-3]
```

**Cost per experiment:** ~$40

### Multi-Asset Tuning

```python
# Try different asset combinations
symbols:
  - ['BTC-USD', 'ETH-USD']  # 2 assets
  - ['BTC-USD', 'ETH-USD', 'SOL-USD']  # 3 assets
  - ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']  # 4 assets

# Try different network sizes
hidden_dim: [128, 256, 512]
```

**Cost per experiment:** ~$60

---

## Monitoring

Use the same monitoring setup for all models:

```bash
# Setup monitoring (run once)
python monitoring/setup_monitoring.py

# View dashboards
https://console.cloud.google.com/monitoring
```

**Key Metrics to Track:**

| Metric | Base PPO | LSTM-PPO | Multi-Asset |
|--------|----------|----------|-------------|
| **Latency (P99)** | <50ms | <80ms | <100ms |
| **Sharpe Ratio** | >1.0 | >1.2 | >1.5 |
| **Win Rate** | >52% | >55% | >58% |
| **Max Drawdown** | <20% | <18% | <15% |

---

## Troubleshooting

### LSTM Training Issues

**Problem:** LSTM training unstable (loss explodes)

**Solution:**
- Reduce `learning_rate` to 1e-4
- Increase `max_grad_norm` to 1.0
- Reduce `lstm_num_layers` to 1

### Multi-Asset Training Issues

**Problem:** Agent only trades one asset (ignores others)

**Solution:**
- Increase training episodes to 2000
- Add asset diversity reward bonus
- Check price data quality for all assets

**Problem:** High correlation between assets (no diversification)

**Solution:**
- Add more uncorrelated assets (e.g., BTC, ETH, SOL, BNB, XRP)
- Increase `commission_rate` to penalize excessive trading
- Add portfolio concentration penalty

---

## Next Steps

1. ✅ Train Base PPO first (mandatory)
2. ⏳ Review Base PPO performance (Sharpe > 1.0?)
3. ⏳ If successful, train LSTM-PPO or Multi-Asset
4. ⏳ Compare models with A/B testing
5. ⏳ Deploy best performer to production

**Estimated Timeline:** 4-6 weeks
**Estimated Cost:** $130-200 (under $400 budget)

---

## File Summary

### New Files Created

| File | Lines | Purpose |
|------|-------|---------|
| **src/rl/lstm_ppo_agent.py** | 250 | LSTM-PPO agent implementation |
| **src/rl/multi_asset_env.py** | 400 | Multi-asset trading environment |
| **vertex_training/trainer_lstm.py** | 250 | LSTM Vertex AI trainer |
| **vertex_training/trainer_multiasset.py** | 300 | Multi-Asset Vertex AI trainer |
| **ADVANCED_MODELS_GUIDE.md** | This file | Documentation |

---

## Support

For questions or issues:
1. Check base PPO deployment first ([GCP_RL_DEPLOYMENT_GUIDE.md](GCP_RL_DEPLOYMENT_GUIDE.md))
2. Verify all imports work: `python -c "from rl.lstm_ppo_agent import LSTMPPOAgent"`
3. Test locally before GCP deployment
4. Monitor training logs in Cloud Console

**All models are ready for training!** 🚀
