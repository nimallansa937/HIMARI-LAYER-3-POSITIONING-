# HIMARI Layer 3 - Phase 3 Deployment Guide

This guide covers deploying Phase 3 with optional Transformer-RL integration.

## Overview

Phase 3 extends Phase 2's multi-asset portfolio management with:

- **Optional Transformer-RL** - Circuit breaker-protected RL predictions
- **Automatic Fallback** - Seamless fallback to Phase 2 when RL unavailable
- **Performance Comparison** - Track RL vs baseline performance

---

## Deployment Modes

### Mode 1: Fallback-Only (Recommended for Start)

Use Phase 2 baseline without RL:

```python
from phases.phase3_hybrid import Layer3Phase3Hybrid

hybrid = Layer3Phase3Hybrid(
    portfolio_value=100000,
    enable_rl=False,  # No RL
    enable_metrics=True
)
```

**Benefits:**

- Zero external dependencies
- Proven Phase 2 performance
- No additional costs

---

### Mode 2: Mock RL (Testing)

Test RL integration without external endpoint:

```python
hybrid = Layer3Phase3Hybrid(
    portfolio_value=100000,
    enable_rl=True,
    use_mock_rl=True,  # Simulated predictions
    enable_metrics=True
)
```

**Benefits:**

- Test full Phase 3 pipeline
- Validate metrics and logging
- No external dependencies

---

### Mode 3: Real RL (Colab Pro)

Connect to real Transformer-RL endpoint:

```python
hybrid = Layer3Phase3Hybrid(
    portfolio_value=100000,
    enable_rl=True,
    rl_endpoint="http://your-colab-endpoint:8888/predict",
    rl_timeout_sec=5.0,
    enable_metrics=True
)
```

**Requirements:**

- Google Colab Pro (~$10/month)
- Trained Transformer model
- FastAPI inference server

---

## Circuit Breaker Configuration

The circuit breaker protects against RL endpoint failures:

```python
from risk.circuit_breaker import ColabProCircuitBreaker

# Custom circuit breaker
cb = ColabProCircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    initial_timeout_sec=30,   # Initial retry delay
    max_timeout_sec=300,      # Max 5 minute delay
    backoff_multiplier=2.0,   # Exponential backoff
    enable_metrics=True
)

# Inject into RL client
from optimization.transformer_rl import TransformerRLClient

client = TransformerRLClient(
    endpoint_url="http://localhost:8888/predict",
    circuit_breaker=cb
)
```

### Circuit Breaker States

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation, all requests allowed |
| **OPEN** | Requests blocked, auto-fallback triggered |
| **HALF_OPEN** | Testing recovery with single request |

### Metrics

| Metric | Description |
|--------|-------------|
| `himari_l3_circuit_breaker_state` | 0=closed, 1=half_open, 2=open |
| `himari_l3_circuit_breaker_failures` | Total failure count |
| `himari_l3_circuit_breaker_fallbacks` | Fallback trigger count |
| `himari_l3_circuit_breaker_timeout_sec` | Current timeout |

---

## Fallback Behavior

When RL is unavailable, Phase 3 automatically falls back to Phase 2:

```
Signal Received
      │
      ▼
┌─────────────────┐
│ RL Enabled?     │──No──▶ Phase 2 Fallback
└────────┬────────┘
         │Yes
         ▼
┌─────────────────┐
│ Circuit Open?   │──Yes──▶ Phase 2 Fallback
└────────┬────────┘
         │No
         ▼
┌─────────────────┐
│ RL Prediction   │──Fail──▶ Phase 2 Fallback
└────────┬────────┘
         │Success
         ▼
   RL-Enhanced Allocation
```

---

## Colab Pro Setup

### Step 1: Create Inference Server

```python
# serve_model.py (run on Colab Pro)
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("position_sizing_model.pt")

@app.post("/predict")
async def predict(request: dict):
    # Process signal
    features = extract_features(request)
    
    with torch.no_grad():
        position_pct, confidence = model(features)
    
    return {
        "position_pct": float(position_pct),
        "confidence": float(confidence),
        "predicted_return": 0.05,
        "predicted_volatility": 0.03,
        "model_version": "v1.0"
    }
```

### Step 2: Expose Endpoint

Use ngrok or Colab tunneling:

```bash
pip install pyngrok
ngrok http 8888
```

### Step 3: Configure Phase 3

```python
hybrid = Layer3Phase3Hybrid(
    enable_rl=True,
    rl_endpoint="https://your-ngrok-url.ngrok.io/predict",
    rl_timeout_sec=5.0
)
```

---

## Monitoring

### Grafana Dashboard

Add these panels to your dashboard:

```json
{
  "title": "Phase 3 RL vs Fallback",
  "targets": [
    {"expr": "rate(himari_l3_position_decisions_total{strategy_id='rl'}[5m])"},
    {"expr": "rate(himari_l3_position_decisions_total{strategy_id='fallback'}[5m])"}
  ]
}
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `himari_l3_position_decisions_total{strategy_id='rl'}` | RL decisions |
| `himari_l3_position_decisions_total{strategy_id='fallback'}` | Fallback decisions |
| `himari_l3_circuit_breaker_state` | Circuit state |

---

## Performance Comparison

Track RL advantage over baseline:

```python
hybrid = Layer3Phase3Hybrid(enable_rl=True, use_mock_rl=True)

# After trades complete, record outcomes
hybrid.record_outcome("BTC-USD", 0.05, used_rl=True)
hybrid.record_outcome("BTC-USD", 0.03, used_rl=False)

# Get comparison
perf = hybrid.get_performance_comparison()
print(f"RL Mean Return: {perf['rl']['mean_return']:.2%}")
print(f"Baseline Mean: {perf['baseline']['mean_return']:.2%}")
print(f"RL Advantage: {perf['rl_advantage']:.2%}")
```

---

## Troubleshooting

### Circuit Breaker Stuck OPEN

```python
# Force reset (use with caution)
hybrid.rl_client.circuit_breaker.reset()
```

### RL Predictions Too Slow

Increase timeout or use fallback:

```python
hybrid = Layer3Phase3Hybrid(
    rl_timeout_sec=10.0,  # Increase timeout
)
```

### No RL Predictions

Check circuit breaker state:

```python
print(hybrid.rl_client.circuit_breaker.get_state())
```

---

## Production Checklist

- [ ] Choose deployment mode (fallback-only / mock / real RL)
- [ ] Configure circuit breaker thresholds
- [ ] Enable Prometheus metrics
- [ ] Add Grafana dashboard panels
- [ ] Test fallback behavior
- [ ] Monitor RL vs baseline performance
