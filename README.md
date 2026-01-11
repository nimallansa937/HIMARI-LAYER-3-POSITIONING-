# HIMARI Layer 3 - Position Sizing & Execution

**RL-Based Position Sizing for Cryptocurrency Trading**

## Overview

HIMARI Layer 3 is a production-grade position sizing system featuring:

- **LSTM-PPO Ensemble Models** for position sizing
- **Regime-Conditional Adjustment** with volatility targeting
- **Bayesian Kelly Position Sizing** with uncertainty quantification
- **Cascade Detection** for crisis protection
- **3-Phase Deployment** architecture

## Model Validation Findings (2026-01-10)

### Key Discovery: Transition Window Edge

| Finding | Status |
|---------|--------|
| RL ensemble adds value broadly | ❌ No |
| Kelly/momentum passes shuffle test | ❌ No |
| **NEUTRAL→BULL transitions** | ✅ **Real edge** |

**Temporal edge found:** Trading Kelly strategy ONLY during NEUTRAL→BULL transition windows (first 3-6 hours) achieves **+18.77 Sharpe** and passes shuffle test.

See [MODEL_VALIDATION_REPORT.md](MODEL_VALIDATION_REPORT.md) for full analysis.

## Project Structure

```
LAYER 3 V1/
├── LAYER 3 POSITIONING LAYER/    # Core positioning code
│   ├── src/                      # Source modules
│   │   ├── core/                 # Type definitions
│   │   ├── engines/              # Position sizing engines
│   │   ├── rl/                   # RL training pipeline
│   │   └── phases/               # Phase orchestrators
│   ├── test_*.py                 # Validation test scripts
│   └── config/                   # Configuration files
│
├── LAYER 3 TRAINED ESSEMBLE MODLES/  # 5-model ensemble
├── pretrain_models/                   # Pretrained checkpoints
├── wfo_models/                        # Walk-forward models
│
├── MODEL_VALIDATION_REPORT.md    # Comprehensive findings
└── test_pretrained_model.py      # Model testing script
```

## Test Scripts

| Script | Purpose |
|--------|---------|
| `test_real_data_ccxt.py` | Real BTC 2Y backtest with shuffle test |
| `test_component_isolation.py` | Isolate vol-targeting vs RL |
| `test_deep_regime_analysis.py` | Regime patterns analysis |
| `test_transition_window.py` | Transition edge validation |
| `test_models_on_transitions.py` | RL contribution in windows |

## Quick Start

```powershell
cd "LAYER 3 V1/LAYER 3 POSITIONING LAYER"
pip install -r requirements.txt

# Run validation
python test_real_data_ccxt.py      # Full backtest
python test_transition_window.py   # Transition edge test
```

## Key Models

### Ensemble (5 models)

- Architecture: LSTM-PPO V2
- State dim: 16, Hidden: 128, LSTM: 2 layers
- Location: `LAYER 3 TRAINED ESSEMBLE MODLES/`

### Pretrained

- Architecture: LSTMPolicyNetwork
- Location: `pretrained_final.pt`

## Path Forward

1. **For production:** Use Kelly in NEUTRAL→BULL windows only (no ML needed)
2. **For ML:** Train transition PREDICTOR (binary classifier)
3. **Use RL for:** Position SIZING, not price prediction

## License

Internal HIMARI OPUS project.
