# Phase 2 Deployment Guide

**Version:** 3.1 Phase 2 Production
**Status:** Ready for Deployment
**Date:** December 26, 2025

---

## Overview

Phase 2 adds multi-asset portfolio allocation capabilities to Layer 3:

- **Multi-asset Kelly allocation** - Optimal capital allocation across correlated assets
- **Ensemble position aggregation** - Combine multiple strategy signals
- **Hierarchical risk budgets** - Portfolio/strategy/position level limits
- **Correlation monitoring** - Track cross-asset relationships
- **Weight drift tracking** - Post-trade attribution analysis

---

## Prerequisites

### Phase 1 Must Be Deployed

Phase 2 builds on Phase 1. Ensure Phase 1 is working:

```bash
python example_phase1_fixed.py
```

All Phase 1 examples should pass.

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- pyyaml >= 6.0
- prometheus_client >= 0.17.0

---

## Deployment Steps

### 1. Verify Tests Pass

```bash
# Unit tests
pytest tests/unit/test_phase2_portfolio.py -v
pytest tests/unit/test_multi_asset_kelly.py -v
pytest tests/unit/test_ensemble_aggregator.py -v
pytest tests/unit/test_correlation_monitor.py -v
pytest tests/unit/test_risk_budget.py -v

# Integration test
pytest tests/integration/test_phase2_integration.py -v
```

**Expected:** All tests pass

### 2. Run Example

```bash
python example_phase2.py
```

**Expected output:**
```
HIMARI OPUS V2 - Phase 2 Multi-Strategy Example
================================================================================
1. Initializing Phase 2 Portfolio...
   Portfolio: $100,000
   Max position: 15%

...

[SUCCESS] All Phase 2 examples completed successfully!
```

### 3. Configuration

Edit `config/layer3_config.yaml` if needed:

```yaml
risk_management:
  max_portfolio_allocation: 0.80   # 80% max capital deployed
  max_strategy_allocation: 0.40    # 40% per strategy
  max_position_pct: 0.15            # 15% per position
```

### 4. Integration Code

```python
from phases.phase2_portfolio import Layer3Phase2Portfolio
from core.layer3_types import TacticalSignal, CascadeIndicators

# Initialize
portfolio = Layer3Phase2Portfolio(
    portfolio_value=100000,
    max_position_pct=0.15,
    max_strategy_pct=0.40,
    enable_metrics=True
)

# Process signals from multiple strategies
allocation = portfolio.process_multi_strategy_signals(
    signals=[signal1, signal2, signal3],
    cascade_indicators=cascade,
    current_prices={'BTC-USD': 43500, 'ETH-USD': 2280}
)

# Results
print(f"Total Allocated: ${allocation.total_allocated_usd:,.2f}")
print(f"Utilization: {allocation.utilization_pct:.1f}%")
print(f"Allocations: {allocation.allocations}")
```

---

## Key Features

### Hierarchical Risk Budgets

Phase 2 enforces limits at three levels:

1. **Portfolio Level** - Total capital deployed (default 80%)
2. **Strategy Level** - Per-strategy allocation (default 40%)
3. **Position Level** - Per-position size (default 15%)

Example:
```python
# Get remaining budget
budget = portfolio.get_remaining_risk_budget()

print(f"Portfolio: ${budget['portfolio']['remaining_usd']:,.2f} remaining")
for strategy_id, info in budget['strategies'].items():
    print(f"{strategy_id}: ${info['remaining_usd']:,.2f} remaining")
```

### Weight Drift Tracking

Track how strategy weights change over time for attribution:

```python
# Export history
portfolio.export_weight_history('weights.csv')
```

CSV format:
```
timestamp_ns,timestamp_iso,strategy_id,weight,total_allocated_usd,num_strategies
1735226123000000,2025-12-26T12:15:23Z,momentum_btc,0.372000,32250.00,3
```

### Correlation Monitoring

Monitor cross-asset correlations:

```python
# Update after each period
portfolio.update_correlations({
    'BTC-USD': 0.02,
    'ETH-USD': 0.018,
    'SOL-USD': -0.01
})

# Check state
state = portfolio.correlation_monitor.get_state()
print(f"Diversification Score: {state['diversification_score']:.2f}")
print(f"Average Correlation: {state['avg_correlation']:.2f}")
```

---

## Metrics

Phase 2 adds the following Prometheus metrics:

### Ensemble Metrics
- `himari_l3_ensemble_weight_total_usd` - Total allocated capital
- `himari_l3_ensemble_utilization_pct` - Portfolio utilization %
- `himari_l3_ensemble_strategies` - Number of active strategies
- `himari_l3_ensemble_weight_current{strategy_id}` - Per-strategy weights
- `himari_l3_ensemble_weight_drift_pct{strategy_id}` - Weight drift %

### Risk Budget Metrics
- `himari_l3_risk_budget_violations_total{level}` - Violations by level
- `himari_l3_risk_budget_utilization_pct{level,identifier}` - Budget usage

### Correlation Metrics
- `himari_l3_correlation_alerts_total` - High correlation alerts

---

## Monitoring

### Check Metrics

If metrics enabled:
```python
portfolio = Layer3Phase2Portfolio(enable_metrics=True)
```

Metrics exposed on port 8000 (if Prometheus server running).

### View State

```python
state = portfolio.get_state()

print(f"Portfolio Decisions: {state['total_portfolio_decisions']}")
print(f"Ensemble Aggregations: {state['ensemble']['total_aggregations']}")
print(f"Risk Budget Violations: {state['risk_budget']['total_violations']}")
```

---

## Troubleshooting

### Issue: "Metrics recording error"

**Cause:** Metrics not defined or Prometheus not configured

**Fix:** Set `enable_metrics=False` or ensure all metrics are defined in `layer3_metrics.py`

### Issue: "Risk budget violations"

**Cause:** Signals exceed configured limits

**Fix:** Either reduce position sizes or increase limits in config

### Issue: "Empty allocation returned"

**Cause:** All signals failed validation or exceeded budgets

**Fix:** Check logs for validation errors, verify signal fields

### Issue: "Correlation matrix errors"

**Cause:** Insufficient historical data

**Fix:** Need at least 20 samples before correlation calculated

---

## Best Practices

### 1. Start Conservative

Use lower limits initially:
```python
portfolio = Layer3Phase2Portfolio(
    max_position_pct=0.10,      # 10% max
    max_strategy_pct=0.30,      # 30% max
    max_portfolio_pct=0.60      # 60% max
)
```

### 2. Monitor Weight Drift

Export weight history regularly:
```python
# Every hour/day
portfolio.export_weight_history(f'weights_{timestamp}.csv')
```

### 3. Update Correlations Frequently

After each trading period:
```python
portfolio.update_correlations(period_returns)
```

### 4. Check Budget Utilization

Before each batch:
```python
budget = portfolio.get_remaining_risk_budget()
# Verify sufficient budget available
```

---

## Production Checklist

- [ ] All tests pass
- [ ] Example runs successfully
- [ ] Configuration reviewed and updated
- [ ] Metrics recording verified
- [ ] Integration code tested
- [ ] Correlation monitor has sufficient data (20+ samples)
- [ ] Risk budgets configured appropriately
- [ ] Weight history export tested
- [ ] Error handling verified
- [ ] Logging reviewed

---

## Next Steps

1. **Phase 3 (Optional):** RL integration with Colab Pro
2. **Monitoring:** Deploy Grafana dashboards
3. **Attribution:** Analyze weight history for performance attribution
4. **Optimization:** Tune correlation thresholds based on backtest

---

## Support

For issues or questions:
- Check logs for detailed error messages
- Review test files for usage examples
- See `example_phase2.py` for complete working example

**Phase 2 is production-ready!** 🚀
