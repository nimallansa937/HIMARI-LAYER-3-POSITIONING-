"""
HIMARI Layer 3: Sharpe Validation Diagnostic
=============================================

Checks for common bugs that inflate Sharpe ratio:
1. Look-ahead bias in regime detection
2. Position sizing timing errors
3. Annualization miscalculation
4. Shuffle test (gold standard)

If Sharpe stays high after shuffle → BUG CONFIRMED
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# =============================================================================
# CONSTANTS
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)  # 77.76
DAILY_ANNUALIZATION = np.sqrt(252)        # 15.87

print("=" * 70)
print("SHARPE VALIDATION DIAGNOSTIC")
print("=" * 70)
print()
print("📊 Annualization Check:")
print(f"   HOURLY_ANNUALIZATION = sqrt(252 * 24) = {HOURLY_ANNUALIZATION:.4f}")
print(f"   DAILY_ANNUALIZATION = sqrt(252) = {DAILY_ANNUALIZATION:.4f}")
print(f"   Ratio: {HOURLY_ANNUALIZATION / DAILY_ANNUALIZATION:.2f}x")
print()

# =============================================================================
# Load Real Data
# =============================================================================

csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
if not os.path.exists(csv_path):
    print("❌ Real data not found. Run test_real_data_ccxt.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)
prices = df['close'].values
print(f"✅ Loaded {len(prices):,} hourly candles")
print()

# =============================================================================
# Calculate Raw Returns
# =============================================================================

returns = np.diff(prices) / prices[:-1]

print("=" * 70)
print("TEST 1: Raw Returns Check")
print("=" * 70)
print(f"   Mean hourly return: {np.mean(returns)*100:.4f}%")
print(f"   Std hourly return: {np.std(returns)*100:.4f}%")
print()

# Buy & Hold Sharpe with DIFFERENT annualizations
bh_sharpe_hourly = np.mean(returns) / np.std(returns) * HOURLY_ANNUALIZATION
bh_sharpe_daily = np.mean(returns) / np.std(returns) * DAILY_ANNUALIZATION
bh_sharpe_none = np.mean(returns) / np.std(returns)

print("📈 Buy & Hold Sharpe (same data, different annualization):")
print(f"   Hourly (sqrt(252*24)): {bh_sharpe_hourly:+.4f}")
print(f"   Daily (sqrt(252)):    {bh_sharpe_daily:+.4f}")
print(f"   None (raw):           {bh_sharpe_none:+.4f}")
print()

# =============================================================================
# TEST 2: Shuffle Test
# =============================================================================

print("=" * 70)
print("TEST 2: SHUFFLE TEST (Gold Standard)")
print("=" * 70)
print("If Sharpe stays high after shuffle → BUG CONFIRMED")
print()

# Simple position strategy (mimic what hybrid does)
def simple_vol_target_backtest(prices, shuffle=False):
    """Simplified vol-targeting backtest."""
    returns = np.diff(prices) / prices[:-1]
    
    if shuffle:
        np.random.shuffle(returns)
    
    # Simple vol-targeting: position = target_vol / realized_vol
    strategy_returns = []
    vol_lookback = 20
    
    for t in range(vol_lookback, len(returns)):
        # Position based on PAST volatility (no look-ahead)
        past_vol = np.std(returns[t-vol_lookback:t]) * HOURLY_ANNUALIZATION
        past_vol = max(past_vol, 0.05)  # Floor
        
        target_vol = 0.15
        position_pct = min(target_vol / past_vol, 0.5)  # Cap at 50%
        
        # Return at time t
        strategy_return = position_pct * returns[t]
        strategy_returns.append(strategy_return)
    
    strategy_returns = np.array(strategy_returns)
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = np.sum(strategy_returns)
    
    return sharpe, total_return


# Run on original data
sharpe_original, ret_original = simple_vol_target_backtest(prices, shuffle=False)
print(f"📊 Original Order:")
print(f"   Sharpe: {sharpe_original:+.4f}")
print(f"   Return: {ret_original*100:+.2f}%")
print()

# Run on shuffled data (5 trials)
print("🔀 Shuffled Order (5 trials):")
shuffled_sharpes = []
for i in range(5):
    np.random.seed(i)
    sharpe_shuffled, ret_shuffled = simple_vol_target_backtest(prices, shuffle=True)
    shuffled_sharpes.append(sharpe_shuffled)
    print(f"   Trial {i+1}: Sharpe {sharpe_shuffled:+.4f}, Return {ret_shuffled*100:+.2f}%")

avg_shuffled_sharpe = np.mean(shuffled_sharpes)
print()
print(f"   Average Shuffled Sharpe: {avg_shuffled_sharpe:+.4f}")
print()

# Verdict
if abs(avg_shuffled_sharpe) < 0.5:
    print("   ✅ Shuffle test PASSED - Edge destroyed by shuffling")
else:
    print("   ❌ Shuffle test FAILED - Edge survives shuffling → BUG LIKELY")
print()

# =============================================================================
# TEST 3: Look-Ahead Bias Check
# =============================================================================

print("=" * 70)
print("TEST 3: LOOK-AHEAD BIAS CHECK")
print("=" * 70)
print()

# Simulate what the backtest does and print timing
print("Position timing check (first 5 decisions):")
print("-" * 50)

vol_lookback = 20
for t in range(vol_lookback, vol_lookback + 5):
    # Data used for decision
    past_returns = returns[t-vol_lookback:t]
    past_vol = np.std(past_returns) * HOURLY_ANNUALIZATION
    
    # Return that this position will capture
    current_return = returns[t]
    
    print(f"t={t}:")
    print(f"   Decision uses returns[{t-vol_lookback}:{t}] (past)")
    print(f"   Position applied to returns[{t}] = {current_return*100:+.4f}%")
    print(f"   Past vol: {past_vol*100:.2f}% (annualized)")
    print()

print("✅ If decision uses [t-20:t] and applies to [t], NO look-ahead bias")
print()

# =============================================================================
# TEST 4: Realistic Sharpe Calculation
# =============================================================================

print("=" * 70)
print("TEST 4: REALISTIC SHARPE CHECK")
print("=" * 70)
print()

# What Sharpe SHOULD be for a simple vol-targeting strategy
print("Expected Sharpe Range for Vol-Targeting on BTC:")
print("   - Pure B&H: ~0.5 - 1.0")
print("   - Vol-Targeting: ~0.8 - 1.5")
print("   - With RL Alpha: ~1.0 - 2.0")
print()

print(f"Your reported Sharpe: 6.34")
print(f"Simple vol-target Sharpe: {sharpe_original:+.4f}")
print()

if sharpe_original > 3.0:
    print("⚠️ WARNING: Simple vol-target > 3.0 suggests calculation error")
    print("   Most likely: Using wrong annualization factor")
elif sharpe_original < 2.0:
    print("✅ Simple vol-target < 2.0 is realistic")
    print("   The 6.34 Sharpe from full strategy may have other issues")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
