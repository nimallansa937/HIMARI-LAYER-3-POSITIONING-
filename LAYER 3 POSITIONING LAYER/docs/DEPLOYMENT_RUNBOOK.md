# HIMARI OPUS V2 - Layer 3 Deployment Runbook

## Quick Start

### 1. Install Dependencies

```powershell
cd "c:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 3 POSITIONING LAYER"
pip install -r requirements.txt
```

### 2. Run Tests

```powershell
python -m pytest tests/ -v
```

### 3. Run Example

```powershell
python example_phase1.py
```

### 4. Start Monitoring Stack (optional)

```powershell
cd monitoring
docker-compose up -d
```

- Grafana: <http://localhost:3001> (admin/himari123)
- Prometheus: <http://localhost:9090>

---

## Production Deployment

### Step 1: Verify Configuration

```powershell
# Check config file exists
cat config/layer3_config.yaml
```

### Step 2: Start Layer 3

```python
from phases.phase1_core import Layer3Phase1

layer3 = Layer3Phase1(
    portfolio_value=100000,
    kelly_fraction=0.25,
    config_path="config/layer3_config.yaml",
    enable_hot_reload=True,
    enable_metrics=True,
    enable_sentiment=True
)
```

### Step 3: Process Signals

```python
from core.layer3_types import TacticalSignal, MarketRegime, CascadeIndicators

signal = TacticalSignal(
    strategy_id="momentum",
    symbol="BTC-USD",
    confidence=0.75,
    regime=MarketRegime.TRENDING_UP,
    # ... other fields
)

cascade = CascadeIndicators(
    funding_rate=0.001,
    oi_change_pct=0.05,
    volume_ratio=2.0,
    onchain_whale_pressure=0.3,
    exchange_netflow_zscore=0.5
)

decision = layer3.calculate_position(signal, cascade, current_price)
```

### Step 4: Export Metrics

```python
from prometheus_client import start_http_server
start_http_server(8000)  # Expose metrics at :8000/metrics
```

---

## Troubleshooting

### Config Not Loading

```powershell
# Verify path
python -c "import os; print(os.path.exists('config/layer3_config.yaml'))"
```

### Metrics Not Recording

- Check `enable_metrics=True` in constructor
- Verify prometheus_client is installed: `pip install prometheus_client`

### Hot-Reload Not Working

- Check `enable_hot_reload=True`
- Verify file permissions on config file
- Check logs for "Configuration change detected"

### Position Size = 0

- Check cascade risk score in diagnostics
- Verify signal confidence > 0.45
- Check for validation errors in logs

---

## Monitoring

### Key Metrics to Watch

- `himari_l3_cascade_risk_score` - Alert if > 0.8
- `himari_l3_circuit_breaker_state` - Alert if = 2 (OPEN)
- `himari_l3_conformal_null_rejections_total` - Alert if increasing
- `himari_l3_regime_false_flips_total` - Monitor for noisy signals

### Grafana Dashboards

1. Position Sizing Overview
2. Risk Management
3. Regime & Cascade Monitoring

---

## Shutdown

```python
layer3.stop()  # Stops hot-reload watcher
```

Or use Ctrl+C for graceful shutdown.
