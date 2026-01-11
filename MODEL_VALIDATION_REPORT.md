# HIMARI Layer 3 - Model Validation Report

**Date:** 2026-01-10  
**Model Tested:** `pretrained_final.pt` and 5-model ensemble

---

## Executive Summary

Comprehensive testing of the Layer 3 positioning models revealed that **the RL ensemble models are NOT providing predictive value**. The high Sharpe ratios observed in earlier tests came primarily from:

1. Volatility targeting mechanism (works on any data)
2. Kelly/momentum logic in bull regimes
3. Test methodology artifacts

---

## Test Results Summary

### 1. Initial Model Test (`pretrained_final.pt`)

| Regime | Sharpe | Return | Max DD | Avg Action |
|--------|--------|--------|--------|------------|
| Bull | -1.07 | -21% | 30% | 0.0 |
| Bear | +1.15 | +66% | 22% | 0.0 |
| Ranging | +0.03 | +0.4% | 16% | 0.0 |

**Finding:** Model always outputs action=0.0 (max short), suggesting degenerate policy or specialized bear hedging.

---

### 2. Full Test Suite Results

| Test Script | Data Type | Sharpe | Status |
|-------------|-----------|--------|--------|
| `test_real_data_ccxt.py` | Real BTC 2Y | +6.34 | ⚠️ See shuffle test |
| `test_pretrain_real_data.py` | Real BTC 2Y | +5.82 | ⚠️ See shuffle test |
| `test_hybrid_strategy.py` | Synthetic 2Y | +3.91 | ✅ |
| `test_extended_backtest.py` | Real BTC 6M | +0.004 | ✅ |
| `test_ensemble.py` | Synthetic | +0.003 | ✅ |
| `test_kelly_baseline.py` | Synthetic | +0.005 | ✅ |

---

### 3. Shuffle Test (Critical Finding)

**Purpose:** If Sharpe survives shuffling, the strategy doesn't exploit real temporal patterns.

| Metric | Original | Shuffled Avg |
|--------|----------|--------------|
| Sharpe | +6.34 | +6.93 |

**Result:** ❌ **FAILED** - Edge SURVIVES shuffling (actually improves!), indicating the high Sharpe is NOT from genuine market prediction.

---

### 4. Component Isolation Test

Isolated contribution of each component:

| Component | Original Sharpe | Shuffled Sharpe | Contribution |
|-----------|-----------------|-----------------|--------------|
| Pure Vol-Targeting | -0.15 | +0.29 | Baseline |
| Pure RL | -0.85 | -0.77 | Negative |
| Hybrid (Vol+RL) | -0.16 | +0.26 | RL adds ~0 |

**Key Findings:**

- **Pure RL = Negative Sharpe (-0.85)** — Models actively hurt performance
- **RL Contribution = -0.003** — Essentially zero value added
- **Vol-targeting survives shuffle** — Works on any random data sequence

---

## Root Cause Analysis

### Why did `test_real_data_ccxt.py` show +6.34 Sharpe?

The high Sharpe came from **additional components** not present in the isolated tests:

1. **Kelly/Momentum in bull markets** — Aggressive sizing during uptrends
2. **Regime-based gating** — Reduces exposure during crisis (RL disabled)
3. **State engineering** — Hardcoded features (state[0]=0.75, etc.)
4. **Vol-targeting mechanics** — Works on any volatility sequence (not just real markets)

### Why do all 5 ensemble models produce identical output?

All models output the same action (0.003 Sharpe with 0% variance reduction), suggesting:

- Training collapsed to similar local minima across all seeds
- Insufficient diversity in training hyperparameters
- Models learned degenerate "always short" or "always neutral" policies

---

## Model Architecture Reference

```
LSTMPolicyNetwork (from lstm_ppo_agent.py):
├── State dim: 16
├── LSTM hidden: 64  
├── Hidden dim: 128
├── LSTM layers: 2
├── Dropout: 0.1
└── Total params: 87,555

LSTMPPONetworkV2 (from pretrain_pipeline_v2.py):
├── State dim: 16
├── Hidden dim: 128
├── LSTM layers: 2
├── Dropout: 0.2
└── Early stopping: patience=3
```

---

## Deep Regime Analysis (2026-01-10)

### Regime Distribution (2Y BTC Data)

| Regime | Hours | Percentage |
|--------|-------|------------|
| NEUTRAL | 10,824 | 62.0% |
| BULL | 2,499 | 14.3% |
| HIGH_VOL | 2,183 | 12.5% |
| BEAR | 1,478 | 8.5% |
| CRISIS | 485 | 2.8% |

### Best Performing 30-Day Windows

All top 5 windows occurred in **NEUTRAL** regime:

1. Day 21-51: Sharpe +8.27, Return +37%
2. Day 28-58: Sharpe +7.65, Return +42%
3. Day 287-317: Sharpe +7.23, Return +38%

### Most Profitable Regime Transitions

| Transition | Count | Avg 6h Return |
|------------|-------|---------------|
| NEUTRAL → BULL | 285 | +0.126% |
| NEUTRAL → BEAR | 217 | +0.088% |
| NEUTRAL → HIGH_VOL | 76 | +0.093% |
| BEAR → HIGH_VOL | 45 | **-0.873%** |

**Key Insight:** Entry AFTER transitions to BULL generates best returns

### RL Performance by Volatility Regime

| Vol Regime | Hours | RL Sharpe | Vol-Target | RL Helps? |
|------------|-------|-----------|------------|-----------|
| Ultra-Low (<0.15) | 1,662 | **-1.47** | +2.07 | ❌ |
| Low (0.15-0.30) | 6,604 | **-0.35** | +0.52 | ❌ |
| Medium (0.30-0.50) | 6,551 | **-0.37** | +0.28 | ❌ |
| High (0.50-0.80) | 2,196 | **-1.57** | +1.53 | ❌ |
| Crisis (>0.80) | 485 | **-4.79** | +4.93 | ❌ |

**Conclusion:** RL **HURTS performance in ALL volatility regimes**

### Kelly/Momentum Shuffle Test

| Metric | Original | Shuffled Avg |
|--------|----------|--------------|
| Sharpe | +10.34 | +10.90 |
| Return | +227% | +285% |

**Result:** ❌ **FAILED** - Kelly actually IMPROVES on shuffled data

**Why Kelly survives shuffle:**

- Position sizing math creates artificial Sharpe
- Not exploiting real temporal patterns
- Works on ANY random return sequence

---

## Strategic Decision Tree

```
┌─────────────────────────────────────────────────┐
│ Kelly/Momentum FAILS shuffle test               │
├─────────────────────────────────────────────────┤
│ ❌ Kelly is NOT exploiting real patterns        │
│                                                 │
│ Next Steps:                                     │
│ 1. Position sizing math creates false Sharpe   │
│ 2. Find different alpha sources:               │
│    • Order flow imbalance                       │
│    • Cross-asset momentum (ETH vs BTC)          │
│    • Funding rate signals                       │
│ 3. No ML until genuine edge is identified      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ RL Does NOT Help in ANY Volatility Regime      │
├─────────────────────────────────────────────────┤
│ Recommendation:                                 │
│ • Remove RL from production                     │
│ • Retrain with different objective:             │
│   - Don't predict price direction              │
│   - Predict optimal position SIZE              │
│ • OR: Use RL for risk management only          │
│   - Binary: "Should I reduce position now?"    │
└─────────────────────────────────────────────────┘
```

---

## Recommendations

### Immediate Actions

1. **Remove RL component** — Vol-targeting alone provides equivalent results
2. **Investigate Kelly/momentum** — This may contain the real (if any) edge
3. **Fix shuffle test** — Ensure strategy fails shuffle test before deployment

### Training Improvements

1. **Retrain models** with truly diverse seeds and hyperparameters
2. **Add entropy regularization** to prevent policy collapse
3. **Use proper validation** - Walk-forward optimization with holdout sets
4. **Implement shuffle validation** during training to detect overfitting

### Architecture Improvements

1. **Separate vol-targeting from RL** — They should be independently testable
2. **Log actual model outputs** — Track what actions are being taken
3. **Add action diversity metrics** — Monitor for policy collapse

---

## Test Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `test_pretrained_model.py` | Test single pretrained model | Project root |
| `test_component_isolation.py` | Isolate vol-targeting vs RL | LAYER 3 POSITIONING LAYER/ |
| `test_real_data_ccxt.py` | Full hybrid with shuffle test | LAYER 3 POSITIONING LAYER/ |
| `test_ensemble.py` | Test 5-model ensemble | LAYER 3 POSITIONING LAYER/ |
| `test_sharpe_diagnostic.py` | Validate Sharpe calculation | LAYER 3 POSITIONING LAYER/ |

---

## Transition Window Test (BREAKTHROUGH!)

### Key Finding: TEMPORAL EDGE EXISTS

| Transition | Sharpe | Shuffled | Drop | Status |
|------------|--------|----------|------|--------|
| **NEUTRAL → BULL** | **+4.10** | +1.85 | **+2.25** | ✅ EDGE |
| BEAR → NEUTRAL | +2.41 | +1.22 | +1.19 | ✅ Marginal |
| NEUTRAL → BEAR | +0.83 | +1.64 | -0.81 | ❌ |

### Optimal Window Length

| Window | Sharpe | Drop | Status |
|--------|--------|------|--------|
| **3h** | +3.78 | **+2.78** | ✅ Best |
| 6h | +4.10 | +2.47 | ✅ |
| 12h | +4.56 | +1.71 | ✅ |
| 48h | +6.43 | +0.03 | ❌ Diluted |

### Why This Matters

- **Only 8.5% of hours traded** (1,481 / 17,519)
- **Edge survives shuffle test** → Real temporal structure
- **Best window: 3-6 hours** post-transition

---

## ML Training Protocol (NEW)

### Phase 1: Transition Predictor

```python
# Binary classifier
INPUT: 16-dim market state
OUTPUT: P(NEUTRAL → BULL in next 1-6 hours)

# Architecture: LSTM (reuse existing)
# Training: Past 2Y of transitions
# Labels: 1 if transition in 6h, else 0
```

### Phase 2: Position Sizer

```python
# Regression model
INPUT: [transition_confidence, volatility, momentum]
OUTPUT: Optimal Kelly fraction [0.0 - 0.5]

# Use RL HERE (not for price prediction)
# Reward = Sharpe during transition windows ONLY
```

### Training Protocol

```python
# Key modifications to pretrain_pipeline.py:

# 1. BINARY CLASSIFICATION, not continuous action
output = P(transition in next 6h)

# 2. ONLY use transition window data
# Ignore 91.5% of non-transition periods

# 3. Shuffle validation DURING training
if epoch % 10 == 0:
    shuffled_acc = test_on_shuffled_transitions()
    if shuffled_acc > 0.55:  # Should be ~0.50 on random
        logger.warning("Model not learning temporal patterns!")

# 4. Class imbalance handling
# 285 transitions / 17,519 hours = 1.6% positive
# Use focal loss or weighted BCE
```

---

## Vast.ai Model Download

```bash
# SSH to Vast.ai
ssh -p 15785 root@ssh5.vast.ai

# Find models
find /workspace -name "*.pth" -o -name "*.pt"
ls -lh /workspace/models/
ls -lh /workspace/checkpoints/

# Download models
cd "C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 3 V1"
scp -P 15785 -r root@ssh5.vast.ai:/workspace/models/ ./models/
scp -P 15785 -r root@ssh5.vast.ai:/workspace/checkpoints/ ./checkpoints/
scp -P 15785 -r root@ssh5.vast.ai:/workspace/logs/ ./logs/
```

### Analysis Tasks

1. [ ] Test existing models on NEUTRAL→BULL windows only
2. [ ] Evaluate early checkpoints (less overfit)
3. [ ] Test on BEAR→NEUTRAL transitions

---

## Conclusion

> **UPDATE:** While RL ensemble models don't add value for general price prediction, we found **REAL temporal edge** in regime transition windows.

### What Works

- ✅ **NEUTRAL → BULL transitions** (Sharpe +4.10, passes shuffle test)
- ✅ **3-6 hour window** after transition is optimal
- ✅ **Kelly momentum** during these windows only

### What Doesn't Work

- ❌ Current RL models (hurt performance in all regimes)
- ❌ Kelly applied broadly (survives shuffle = not predictive)
- ❌ Vol-targeting alone (works on any data)

### Path Forward

1. Train **transition predictor** (binary classifier)
2. Use RL for **position sizing**, not price prediction
3. Trade **only during transition windows**
