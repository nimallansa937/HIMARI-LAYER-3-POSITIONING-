# HIMARI Layer 3: Deployment Checklist

**Date:** 2026-01-10
**Status:** ✅ Ready for Paper Trading
**Validated Edge:** NEUTRAL → BULL transitions (Sharpe +4.10, passes shuffle test)

---

## Pre-Deployment Checklist

### ✅ Research & Validation Complete

- [x] Found real temporal edge (Sharpe drops 54% on shuffle)
- [x] Validated on 2Y historical data
- [x] Identified optimal window (3-6 hours)
- [x] Isolated RL contribution (zero - don't use)
- [x] Shuffle test passed for transition timing
- [x] Production code written and tested

### 📋 Before You Start Trading

- [ ] **Install dependencies**
  ```bash
  pip install ccxt pandas numpy
  ```

- [ ] **Verify data availability**
  ```bash
  # Check BTC data exists
  python -c "import pandas as pd; df = pd.read_csv('LAYER 3 POSITIONING LAYER/btc_hourly_real.csv'); print(f'Data: {len(df)} candles')"
  ```

- [ ] **Run validation tests**
  ```bash
  cd "LAYER 3 POSITIONING LAYER"
  python test_transition_window.py
  # Confirm: Sharpe drop > 2.0
  ```

- [ ] **Backtest production strategy**
  ```bash
  python production_transition_strategy.py
  # Verify it runs without errors
  ```

---

## Paper Trading Setup (RECOMMENDED FIRST)

### Step 1: Configure Exchange

```bash
# Create exchange account (Binance, Kraken, etc.)
# Get API keys with READ-ONLY permissions for paper trading
# Enable testnet/sandbox mode if available
```

### Step 2: Start Paper Trading

```bash
cd "LAYER 3 POSITIONING LAYER"

# Run paper trading (1 hour updates)
python live_trading_example.py --paper --interval 3600
```

### Step 3: Monitor for 1-3 Months

Track these metrics:

| Metric | Target | Check Frequency |
|--------|--------|-----------------|
| Sharpe Ratio | > 2.0 | Weekly |
| Win Rate | > 55% | Weekly |
| Drawdown | < 20% | Daily |
| Transition Detection | 4-5 per month | Weekly |

**Exit criteria (STOP if any occur):**
- Sharpe < 0.5 for 3 weeks straight
- Drawdown > 30%
- No transitions detected for 2 months
- Win rate < 45% over 20+ trades

---

## Live Trading Setup (AFTER PAPER TRADING SUCCESS)

### Prerequisites

✅ Paper traded for 1-3 months
✅ Sharpe > 2.0 in paper trading
✅ Understand all code components
✅ Have risk management plan
✅ Tested with small capital first

### Step 1: Risk Management Configuration

```python
# Start VERY conservative
strategy = TransitionStrategy(
    window_hours=3,           # Shorter window = less exposure
    max_position=0.2,         # Only 20% max position
    target_transitions=[("NEUTRAL", "BULL")]  # Only best edge
)
```

### Step 2: Position Sizing

| Account Size | Recommended Position | Max Loss per Trade |
|--------------|---------------------|-------------------|
| $1,000 | 10-20% | $20-40 |
| $10,000 | 10-30% | $200-600 |
| $100,000 | 20-50% | $2,000-10,000 |

### Step 3: Start Small

```bash
# First week: 10% of target position
# Week 2-4: 25% of target position
# Month 2: 50% of target position
# Month 3+: Full position (if performing well)
```

### Step 4: Run Live Trading

```bash
# REAL MONEY - TRIPLE CHECK EVERYTHING!
python live_trading_example.py \
    --live \
    --api-key YOUR_KEY \
    --api-secret YOUR_SECRET \
    --interval 3600 \
    --window 3 \
    --log-file live_trades.log
```

---

## Daily Monitoring Checklist

### Every Day

- [ ] Check `trades.log` for new trades
- [ ] Verify current position matches expected
- [ ] Check equity vs. target
- [ ] Monitor for transition signals

### Every Week

- [ ] Calculate weekly Sharpe ratio
- [ ] Review all trades (wins/losses)
- [ ] Check regime distribution
- [ ] Verify edge still exists

### Every Month

- [ ] Run full backtest on new data
- [ ] **Run shuffle test** (most important!)
- [ ] Adjust position sizing if needed
- [ ] Review overall performance

---

## Warning Signs & Exit Criteria

### 🚨 STOP TRADING IF:

1. **Edge Disappears**
   - Sharpe < 0.5 for 3+ weeks
   - Shuffle test fails (Sharpe survives shuffle)
   - Win rate < 45% over 20+ trades

2. **Market Regime Change**
   - No NEUTRAL→BULL transitions for 2+ months
   - Only CRISIS regime for extended period
   - Volatility stays > 1.0 for weeks

3. **Technical Issues**
   - Exchange API errors
   - Missing price data
   - Strategy execution failures

4. **Risk Limits Breached**
   - Drawdown > 30%
   - 3+ consecutive losing trades
   - Position sizing errors

### Recovery Protocol

If you stop trading:

1. **Analyze what changed**
   ```bash
   # Re-run shuffle test on recent data
   python test_transition_window.py
   ```

2. **Review regime distribution**
   - Has market structure changed?
   - Are transitions still profitable?

3. **Paper trade again**
   - Validate edge still exists
   - Don't restart live until confident

---

## Performance Tracking

### Key Metrics to Log

```python
# Daily
- Equity
- Current position
- Active transitions
- Regime

# Per Trade
- Entry price/time
- Exit price/time
- PnL
- Transition type
- Hold duration

# Weekly
- Sharpe ratio
- Win rate
- Average win/loss
- Max drawdown
```

### Create Performance Dashboard

```python
import pandas as pd

# Load trade log
trades = pd.read_json('trades.log', lines=True)

# Calculate metrics
total_return = (trades['equity'].iloc[-1] - 100000) / 100000
sharpe = trades['equity'].pct_change().mean() / trades['equity'].pct_change().std() * (252*24)**0.5

print(f"Return: {total_return*100:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Trades: {len(trades)}")
```

---

## Emergency Procedures

### If Strategy Goes Wrong

**Scenario 1: Unexpected Large Loss**
```bash
# Immediately exit all positions
# Review what happened
# Check for code bugs
# Verify data feed correctness
```

**Scenario 2: Exchange API Failure**
```bash
# Have backup exchange ready
# Manually close positions if needed
# Don't rely on automated execution alone
```

**Scenario 3: Market Flash Crash**
```bash
# Strategy stays flat during CRISIS regime
# No positions should be open
# Wait for regime to normalize
```

---

## Optimization Timeline

### Month 1-3: Validation Phase

**Goal:** Confirm edge persists in live markets

- Run strategy as-is
- No parameter changes
- Monitor performance
- Collect data

### Month 4-6: Optimization Phase

**If performing well:**

1. **Test secondary edge**
   ```python
   target_transitions=[
       ("NEUTRAL", "BULL"),
       ("BEAR", "NEUTRAL")  # Add this
   ]
   ```

2. **Optimize window length**
   - Test 3h vs 6h vs 12h
   - Use walk-forward validation

3. **Refine position sizing**
   - Analyze which trades were best
   - Adjust Kelly multiplier

### Month 7+: ML Enhancement (Optional)

**Only if edge still strong:**

1. Train transition predictor
2. Optimize entry timing (1-2h early)
3. Use RL for position sizing
4. Add confidence scoring

---

## File Locations

### Production Files
```
LAYER 3 POSITIONING LAYER/
├── production_transition_strategy.py   # Main strategy
├── live_trading_example.py            # Trading wrapper
├── PRODUCTION_STRATEGY_README.md      # Documentation
├── DEPLOYMENT_CHECKLIST.md            # This file
└── btc_hourly_real.csv               # Historical data
```

### Test Files
```
LAYER 3 POSITIONING LAYER/
├── test_transition_window.py          # Validation
├── test_component_isolation.py        # Component tests
├── test_real_data_ccxt.py            # Full backtest
└── MODEL_VALIDATION_REPORT.md        # Analysis
```

### Log Files (Created During Trading)
```
├── trades.log                         # All trades
├── live_trades.log                   # Live trading only
└── performance.csv                   # Daily metrics
```

---

## Quick Reference Commands

### Start Paper Trading
```bash
cd "LAYER 3 POSITIONING LAYER"
python live_trading_example.py --paper --interval 3600
```

### Start Live Trading
```bash
python live_trading_example.py --live --api-key KEY --api-secret SECRET
```

### Run Validation Tests
```bash
python test_transition_window.py
python production_transition_strategy.py
```

### Check Performance
```bash
python -c "
import pandas as pd
trades = pd.read_json('trades.log', lines=True)
print(f'Trades: {len(trades)}')
print(f'Return: {(trades.equity.iloc[-1]-100000)/100000*100:.2f}%')
"
```

---

## Support & Resources

### Documentation
- `PRODUCTION_STRATEGY_README.md` - Full usage guide
- `MODEL_VALIDATION_REPORT.md` - Research findings
- Code comments in Python files

### Validation
- Run `test_transition_window.py` monthly
- Check shuffle test results
- Verify edge persists

### Contact
- Review trade logs for issues
- Analyze performance metrics
- Adjust parameters conservatively

---

## Final Checklist Before Going Live

- [ ] Paper traded successfully for 1-3 months
- [ ] Sharpe > 2.0 in paper trading
- [ ] Understand all strategy components
- [ ] Read all documentation
- [ ] Tested with small capital ($1K-5K)
- [ ] Have risk management plan
- [ ] Know emergency exit procedures
- [ ] Monitoring system in place
- [ ] Backup plan if strategy fails
- [ ] **Shuffle test still passes on recent data**

---

## Remember

1. **Start small** - Only risk what you can afford to lose
2. **Monitor constantly** - Don't set and forget
3. **Validate monthly** - Run shuffle test regularly
4. **Exit if edge dies** - Don't hold losing strategies
5. **Paper trade first** - Always validate before live

**The edge is REAL but fragile. Market conditions change. Stay vigilant.**

Good luck! 🚀
