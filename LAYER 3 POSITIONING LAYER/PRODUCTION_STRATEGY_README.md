# HIMARI Layer 3: Production Transition Strategy

**Status:** ✅ Ready for paper trading
**Date:** 2026-01-10
**Validated Edge:** NEUTRAL → BULL transitions (Sharpe +4.10, passes shuffle test)

---

## Quick Start

### 1. Paper Trading (Recommended First Step)

```bash
# Install dependencies
pip install ccxt pandas numpy

# Run paper trading
python live_trading_example.py --paper --interval 3600
```

### 2. Backtest Validation

```bash
# Validate strategy on historical data
python production_transition_strategy.py
```

### 3. Live Trading (After Paper Trading Success)

```bash
# REAL MONEY - BE CAREFUL!
python live_trading_example.py --live --api-key YOUR_KEY --api-secret YOUR_SECRET
```

---

## Strategy Overview

### Core Concept

The strategy exploits **regime transition timing**:

1. **Detect** NEUTRAL → BULL regime transitions
2. **Trade** for 3-6 hours after transition
3. **Stay flat** outside transition windows (91.5% of time)

### Why This Works

| Test | Result | Interpretation |
|------|--------|----------------|
| Original Sharpe | +4.10 | Strong performance |
| Shuffled Sharpe | +1.85 | **Drops 54%** when timing randomized |
| **Conclusion** | ✅ | **Real temporal edge** (not mathematical artifact) |

### Key Metrics (2Y Backtest)

```
Sharpe Ratio:    +4.10
Total Return:    +24.83%
Active Time:     8.5% (1,481 / 17,519 hours)
Transitions:     285
Win Rate:        ~60-65%
Max Position:    50% of capital
Commission:      0.1% per trade
```

---

## Files Overview

### Production Files

| File | Purpose | Status |
|------|---------|--------|
| `production_transition_strategy.py` | Core strategy implementation | ✅ Production ready |
| `live_trading_example.py` | Live/paper trading wrapper | ✅ Ready |
| `test_transition_window.py` | Validation & shuffle tests | ✅ Complete |

### Strategy Components

```
TransitionStrategy (main class)
├── RegimeDetector
│   ├── Classify market regime
│   └── Uses volatility + momentum
│
├── TransitionDetector
│   ├── Detect regime changes
│   └── Manage trading windows
│
└── KellyPositionSizer
    ├── Calculate Kelly fraction
    └── Apply momentum boost
```

---

## Usage Examples

### Example 1: Simple Backtest

```python
from production_transition_strategy import TransitionStrategy
from datetime import datetime, timedelta
import pandas as pd

# Load data
df = pd.read_csv('btc_hourly_real.csv')
prices = df['close'].values

# Initialize strategy
strategy = TransitionStrategy(
    window_hours=6,
    target_transitions=[("NEUTRAL", "BULL")]
)

# Run backtest
capital = 100000.0
start_time = datetime(2024, 1, 1)

for i, price in enumerate(prices):
    timestamp = start_time + timedelta(hours=i)

    # Get position recommendation
    signal = strategy.get_position(price, timestamp)

    # Your trading logic here...
    if signal['in_transition']:
        print(f"TRADE: {signal['transition_type']} - Position: {signal['position']*100:.1f}%")
```

### Example 2: Live Trading Integration

```python
from production_transition_strategy import TransitionStrategy
import ccxt
from datetime import datetime

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': 'YOUR_KEY',
    'secret': 'YOUR_SECRET',
})

# Initialize strategy
strategy = TransitionStrategy(window_hours=6)

# Trading loop
while True:
    # Get current price
    ticker = exchange.fetch_ticker('BTC/USDT')
    price = ticker['last']

    # Get signal
    signal = strategy.get_position(price, datetime.now())

    # Execute trade
    if signal['in_transition'] and signal['position'] > 0:
        # Place buy order
        print(f"Signal: {signal['transition_type']}")
        print(f"Position: {signal['position']*100:.1f}%")
        # exchange.create_market_buy_order(...)

    # Wait 1 hour
    time.sleep(3600)
```

### Example 3: Risk Management

```python
strategy = TransitionStrategy(
    window_hours=6,
    max_position=0.3,  # Max 30% instead of 50%
    target_transitions=[
        ("NEUTRAL", "BULL"),
        ("BEAR", "NEUTRAL")  # Add mean-reversion edge
    ]
)

signal = strategy.get_position(price, timestamp)

# Conservative sizing
if signal['signal_strength'] == 'STRONG':
    position = signal['position']
elif signal['signal_strength'] == 'MODERATE':
    position = signal['position'] * 0.5  # Half size
else:
    position = 0  # Skip weak signals
```

---

## Configuration Options

### TransitionStrategy Parameters

```python
TransitionStrategy(
    window_hours=6,              # Hours to trade after transition (3-6 recommended)
    max_position=0.5,            # Maximum position size (0.5 = 50%)
    target_transitions=[...],    # Which transitions to trade
    commission_rate=0.001        # Trading fees (0.001 = 0.1%)
)
```

### Recommended Configurations

**Conservative (Lower risk):**
```python
strategy = TransitionStrategy(
    window_hours=3,              # Shorter window
    max_position=0.3,            # Lower max position
    target_transitions=[("NEUTRAL", "BULL")]  # Only best edge
)
```

**Aggressive (Higher risk/reward):**
```python
strategy = TransitionStrategy(
    window_hours=6,
    max_position=0.5,
    target_transitions=[
        ("NEUTRAL", "BULL"),     # Primary edge
        ("BEAR", "NEUTRAL")      # Secondary edge
    ]
)
```

**Research Mode:**
```python
strategy = TransitionStrategy(
    window_hours=6,
    target_transitions=[
        ("NEUTRAL", "BULL"),
        ("NEUTRAL", "BEAR"),
        ("BEAR", "NEUTRAL"),
        ("HIGH_VOL", "NEUTRAL")
    ]
)
# Test all transitions, analyze which work best
```

---

## Regime Classification

### How Regimes Are Defined

```python
# Volatility (annualized, 20-hour lookback)
vol = np.std(returns[-20:]) * sqrt(252 * 24)

# Momentum (50-hour sum of returns)
mom = sum(returns[-50:])

# Classification logic (priority order):
if vol > 0.8:
    regime = "CRISIS"
elif vol > 0.5:
    regime = "HIGH_VOL"
elif mom > 0.03:
    regime = "BULL"
elif mom < -0.03:
    regime = "BEAR"
else:
    regime = "NEUTRAL"
```

### Regime Distribution (2Y BTC Data)

| Regime | Hours | Percentage |
|--------|-------|------------|
| NEUTRAL | 10,824 | 62.0% |
| BULL | 2,499 | 14.3% |
| HIGH_VOL | 2,183 | 12.5% |
| BEAR | 1,478 | 8.5% |
| CRISIS | 485 | 2.8% |

---

## Position Sizing (Kelly Criterion)

### Formula

```python
# Win rate and avg win/loss from recent 20 returns
win_rate = len(wins) / len(returns)
avg_win = mean(wins)
avg_loss = abs(mean(losses))

# Kelly fraction
kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)

# Momentum boost for NEUTRAL->BULL
momentum = sum(returns[-10:])
kelly = kelly * (1 + clip(momentum * 3, 0, 0.5))

# Conservative sizing (50% of Kelly)
position = kelly * 0.5
```

### Position Size Examples

| Scenario | Kelly | Final Position |
|----------|-------|----------------|
| Strong bull transition | 0.6 | 30% (capped at 50%) |
| Moderate transition | 0.3 | 15% |
| Weak signal | 0.1 | 5% |
| No transition | 0.0 | 0% (flat) |

---

## Performance Validation

### Backtest Results (2Y BTC, 2024-2026)

#### NEUTRAL → BULL Transitions

| Window | Sharpe | Shuffled | Drop | Status |
|--------|--------|----------|------|--------|
| **3h** | +3.78 | +1.00 | **+2.78** | ✅ Best edge |
| **6h** | +4.10 | +1.85 | +2.25 | ✅ Strong |
| 12h | +4.56 | +2.86 | +1.71 | ✅ Moderate |
| 24h | +5.35 | +4.53 | +0.82 | ⚠️ Diluted |

#### Other Transitions

| Transition | Sharpe | Drop | Recommended |
|------------|--------|------|-------------|
| NEUTRAL → BULL | +4.10 | +2.13 | ✅ **Primary** |
| BEAR → NEUTRAL | +2.41 | +1.19 | ✅ Secondary |
| NEUTRAL → BEAR | +0.83 | -0.81 | ❌ Avoid |

---

## Risk Management

### Position Limits

```python
# Never exceed max_position (default 50%)
position = min(kelly * 0.5, max_position)

# Scale down on weak signals
if signal_strength == 'WEAK':
    position = position * 0.5
```

### Stop Loss (Optional)

```python
# Track peak equity during transition window
if current_equity < peak_equity * 0.95:  # -5% from peak
    # Exit position early
    position = 0
    print("Stop loss triggered")
```

### Transition Timeout

```python
# Strategy automatically exits after window_hours
# No need for manual position management
```

---

## Monitoring & Logging

### Signal Dictionary

Every `get_position()` call returns:

```python
{
    'position': 0.25,                    # Recommended position size
    'regime': 'BULL',                    # Current regime
    'in_transition': True,               # In trading window?
    'transition_type': ('NEUTRAL', 'BULL'),  # Transition detected
    'hours_remaining': 4,                # Hours left in window
    'kelly_fraction': 0.5,               # Kelly calculation
    'signal_strength': 'STRONG',         # STRONG/MODERATE/WEAK/NO_SIGNAL
    'transition_occurred': False         # True if new transition this update
}
```

### Trade Logging

```python
# Trades automatically logged to trades.log
{
    "timestamp": "2024-01-15T14:30:00",
    "price": 45230.50,
    "old_position": 0.0,
    "new_position": 0.25,
    "reason": "('NEUTRAL', 'BULL') transition",
    "equity": 102340.50,
    "mode": "PAPER"
}
```

---

## Troubleshooting

### Issue: Sharpe Lower Than Expected

**Possible causes:**
1. Different time period (market regime changed)
2. Not enough history (need 50+ hours)
3. Wrong target transitions configured

**Solution:**
```python
# Check your configuration
print(f"Target transitions: {strategy.target_transitions}")
print(f"Window hours: {strategy.window_hours}")

# Validate on known good data
python test_transition_window.py
```

### Issue: No Trades Executing

**Possible causes:**
1. No NEUTRAL→BULL transitions in recent data
2. Insufficient price history (< 50 candles)
3. Transition window expired

**Solution:**
```python
signal = strategy.get_position(price, timestamp)
print(f"Current regime: {signal['regime']}")
print(f"In transition: {signal['in_transition']}")
print(f"History length: {len(strategy.returns_history)}")
```

### Issue: Large Drawdown

**Possible causes:**
1. Position size too large
2. Multiple losing transitions in a row
3. Market regime shift

**Solution:**
```python
# Reduce position size
strategy = TransitionStrategy(max_position=0.3)

# Add stop loss
# Only trade STRONG signals
if signal['signal_strength'] != 'STRONG':
    position = 0
```

---

## Next Steps

### Phase 1: Validation (Current)

- [x] Backtest on 2Y data
- [x] Validate shuffle test
- [x] Production code written
- [ ] Paper trade for 1-3 months
- [ ] Monitor real-time performance

### Phase 2: Optimization (After Validation)

- [ ] Walk-forward optimization
- [ ] Test on different assets (ETH, etc.)
- [ ] Add secondary edges (BEAR→NEUTRAL)
- [ ] Optimize window length dynamically

### Phase 3: ML Enhancement (Optional)

- [ ] Train transition predictor (binary classifier)
- [ ] Train position sizer (RL for Kelly optimization)
- [ ] Add confidence scoring
- [ ] Early entry (1-2h before transition)

---

## References

### Test Files

| File | Purpose |
|------|---------|
| `test_transition_window.py` | Validate transition edge |
| `test_component_isolation.py` | Isolate RL vs vol-targeting |
| `test_real_data_ccxt.py` | Full strategy with shuffle test |
| `MODEL_VALIDATION_REPORT.md` | Complete analysis |

### Key Findings

1. **RL models don't help** (hurt performance in all regimes)
2. **Kelly alone fails shuffle** (not predictive)
3. **Transition timing works** (Sharpe drops 54% when shuffled)
4. **3-6h window optimal** (best risk/reward)

---

## Support

For questions or issues:

1. Check `MODEL_VALIDATION_REPORT.md` for detailed analysis
2. Run `test_transition_window.py` to validate edge still exists
3. Review trade logs in `trades.log`
4. Adjust parameters in `TransitionStrategy(...)` constructor

---

## Disclaimer

**This is experimental software.**

- Past performance does not guarantee future results
- Cryptocurrency trading carries significant risk
- Only trade with capital you can afford to lose
- Always start with paper trading
- Validate the edge persists before going live

**The strategy has been validated on historical data (2024-2026) but market conditions change.**

Test thoroughly in paper trading before risking real money.
