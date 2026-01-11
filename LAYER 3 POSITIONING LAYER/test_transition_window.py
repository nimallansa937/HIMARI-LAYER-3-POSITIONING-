"""
HIMARI Layer 3: Transition Window Test
=======================================

Test hypothesis: Edge exists ONLY in specific transition windows

Strategy:
1. Apply Kelly ONLY when regime changes from NEUTRAL -> BULL
2. Only trade within first 6 hours of transition
3. Shuffle ONLY the transition timestamps (keep price intact)

If Sharpe drops after shuffling transition times -> Found temporal structure!

Usage:
    python test_transition_window.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# =============================================================================
# CONSTANTS
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)
COMMISSION_RATE = 0.001


# =============================================================================
# Regime Detection
# =============================================================================

def classify_regime(returns_history: list, vol_lookback: int = 20, mom_lookback: int = 50) -> str:
    """Classify current market regime."""
    if len(returns_history) < mom_lookback:
        return "NEUTRAL"
    
    # Volatility
    vol = np.std(returns_history[-vol_lookback:]) * HOURLY_ANNUALIZATION
    
    # Momentum
    mom = np.sum(returns_history[-mom_lookback:])
    
    if vol > 0.8:
        return "CRISIS"
    elif vol > 0.5:
        return "HIGH_VOL"
    elif mom > 0.03:
        return "BULL"
    elif mom < -0.03:
        return "BEAR"
    else:
        return "NEUTRAL"


def detect_transitions(returns: np.ndarray) -> list:
    """
    Detect all regime transitions and their timestamps.
    Returns list of (timestamp, from_regime, to_regime)
    """
    transitions = []
    returns_history = []
    prev_regime = "NEUTRAL"
    
    for t in range(len(returns)):
        returns_history.append(returns[t])
        curr_regime = classify_regime(returns_history)
        
        if curr_regime != prev_regime:
            transitions.append({
                'timestamp': t,
                'from': prev_regime,
                'to': curr_regime,
                'returns_6h': returns[t:t+6] if t + 6 < len(returns) else returns[t:]
            })
        
        prev_regime = curr_regime
    
    return transitions


# =============================================================================
# Selective Kelly Strategy
# =============================================================================

def run_selective_kelly(
    returns: np.ndarray,
    target_transitions: list = [("NEUTRAL", "BULL")],
    window_hours: int = 6,
    shuffle_timestamps: bool = False
) -> dict:
    """
    Run Kelly strategy ONLY during specific transition windows.
    
    Args:
        returns: Array of hourly returns
        target_transitions: List of (from_regime, to_regime) tuples to trade
        window_hours: How many hours to trade after transition
        shuffle_timestamps: If True, randomize when transitions "happen"
    """
    # Detect transitions
    transitions = detect_transitions(returns)
    
    # Filter to target transitions only
    filtered_trans = [
        t for t in transitions 
        if (t['from'], t['to']) in target_transitions
    ]
    
    if shuffle_timestamps:
        # Shuffle the timestamps while keeping price sequences intact
        # This tests if TIMING matters
        np.random.shuffle(filtered_trans)
        # Reassign timestamps randomly within valid range
        valid_timestamps = np.random.choice(
            range(50, len(returns) - window_hours), 
            size=len(filtered_trans), 
            replace=False
        ) if len(filtered_trans) < len(returns) - 50 - window_hours else []
        
        for i, t in enumerate(filtered_trans):
            if i < len(valid_timestamps):
                t['timestamp'] = valid_timestamps[i]
    
    # Build trading windows
    trading_windows = set()
    for trans in filtered_trans:
        start = trans['timestamp']
        for h in range(window_hours):
            if start + h < len(returns):
                trading_windows.add(start + h)
    
    # Run backtest
    capital = 100000.0
    returns_history = []
    strategy_returns = []
    prev_position = 0.0
    
    trades_executed = 0
    
    for t in range(len(returns)):
        ret = returns[t]
        returns_history.append(ret)
        
        if t < 20:
            strategy_returns.append(0)
            continue
        
        # Only trade during transition windows
        if t in trading_windows:
            trades_executed += 1
            recent = returns_history[-20:]
            
            # Kelly calculation
            wins = [r for r in recent if r > 0]
            losses = [r for r in recent if r < 0]
            
            if not losses or not wins:
                kelly = 0.15
            else:
                win_rate = len(wins) / len(recent)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                
                if avg_loss > 0:
                    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss + 1e-8)
                else:
                    kelly = 0.25
                
                kelly = np.clip(kelly, 0, 0.5)
            
            # Momentum boost (we believe transition to BULL)
            mom = np.sum(recent[-10:])
            mom_factor = 1 + np.clip(mom * 3, 0, 0.5)  # More aggressive on momentum
            
            position = kelly * mom_factor * 0.5
            position = np.clip(position, 0, 0.5)
        else:
            # No position outside transition windows
            position = 0.0
        
        # Apply commission for position changes
        position_change = abs(position - prev_position)
        commission = position_change * COMMISSION_RATE
        
        strategy_return = position * ret - commission
        capital *= (1 + strategy_return)
        
        prev_position = position
        strategy_returns.append(strategy_return)
    
    arr = np.array(strategy_returns)
    sharpe = np.mean(arr) / (np.std(arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - 100000) / 100000
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'final_capital': capital,
        'num_transitions': len(filtered_trans),
        'hours_traded': len(trading_windows),
        'trades_executed': trades_executed
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 3: TRANSITION WINDOW TEST")
    print("=" * 70)
    print()
    print("Hypothesis: Edge exists ONLY in specific transition windows")
    print("Test: Apply Kelly ONLY when NEUTRAL -> BULL, first 6 hours")
    print()
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
    if os.path.exists(csv_path):
        print(f"[+] Loading cached data...")
        df = pd.read_csv(csv_path)
        prices = df['close'].values
        print(f"    Loaded {len(prices):,} hourly candles")
    else:
        print(" No cached data. Run test_real_data_ccxt.py first.")
        return
    
    returns = np.diff(prices) / prices[:-1]
    
    # ==========================================================================
    # Test 1: NEUTRAL -> BULL transitions only
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print(" TEST 1: NEUTRAL -> BULL Transitions (6h window)")
    print("=" * 70)
    
    result_nb = run_selective_kelly(
        returns, 
        target_transitions=[("NEUTRAL", "BULL")],
        window_hours=6
    )
    
    print(f"\n   Transitions found: {result_nb['num_transitions']}")
    print(f"   Hours traded: {result_nb['hours_traded']} / {len(returns)} ({result_nb['hours_traded']/len(returns)*100:.1f}%)")
    print(f"   Sharpe: {result_nb['sharpe']:+.4f}")
    print(f"   Return: {result_nb['total_return']*100:+.2f}%")
    
    # Shuffle test for this specific strategy
    print("\n Shuffle Test (5 trials):")
    shuffled_sharpes = []
    for trial in range(5):
        np.random.seed(trial + 200)
        result_shuf = run_selective_kelly(
            returns,
            target_transitions=[("NEUTRAL", "BULL")],
            window_hours=6,
            shuffle_timestamps=True
        )
        shuffled_sharpes.append(result_shuf['sharpe'])
        print(f"   Trial {trial+1}: Sharpe {result_shuf['sharpe']:+.4f}")
    
    avg_shuf = np.mean(shuffled_sharpes)
    sharpe_drop_nb = result_nb['sharpe'] - avg_shuf
    
    print(f"\n   Original: {result_nb['sharpe']:+.4f}")
    print(f"   Shuffled: {avg_shuf:+.4f}")
    print(f"   Drop: {sharpe_drop_nb:+.4f}")
    
    if sharpe_drop_nb > 0.3:
        print(f"\n    TEMPORAL STRUCTURE FOUND!")
        print(f"      NEUTRAL -> BULL timing matters!")
    else:
        print(f"\n    No temporal structure in NEUTRAL -> BULL")
    
    # ==========================================================================
    # Test 2: Try other transitions
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print(" TEST 2: Other Transition Types")
    print("=" * 70)
    
    transition_types = [
        ("NEUTRAL", "BULL"),
        ("NEUTRAL", "BEAR"),
        ("BEAR", "NEUTRAL"),
        ("HIGH_VOL", "NEUTRAL"),
        ("CRISIS", "NEUTRAL"),
    ]
    
    print(f"\n{'Transition':<25} {'Count':<8} {'Sharpe':<12} {'Shuf Avg':<12} {'Drop':<10}")
    print("-" * 70)
    
    best_transition = None
    best_drop = -999
    
    for trans in transition_types:
        result = run_selective_kelly(returns, target_transitions=[trans], window_hours=6)
        
        if result['num_transitions'] < 10:
            continue
        
        # Quick shuffle test
        shuf_sharpes = []
        for seed in range(3):
            np.random.seed(seed + 300)
            r = run_selective_kelly(returns, target_transitions=[trans], window_hours=6, shuffle_timestamps=True)
            shuf_sharpes.append(r['sharpe'])
        
        avg_s = np.mean(shuf_sharpes)
        drop = result['sharpe'] - avg_s
        
        trans_str = f"{trans[0]} -> {trans[1]}"
        print(f"{trans_str:<25} {result['num_transitions']:<8} {result['sharpe']:+.4f}      {avg_s:+.4f}      {drop:+.4f}")
        
        if drop > best_drop:
            best_drop = drop
            best_transition = trans
    
    # ==========================================================================
    # Test 3: Extended window (12h, 24h)
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print(" TEST 3: Varying Window Length")
    print("=" * 70)
    
    print(f"\n{'Window':<12} {'Sharpe':<12} {'Shuffled':<12} {'Drop':<10} {'Status'}")
    print("-" * 60)
    
    for window in [3, 6, 12, 24, 48]:
        result = run_selective_kelly(returns, target_transitions=[("NEUTRAL", "BULL")], window_hours=window)
        
        shuf_sharpes = []
        for seed in range(3):
            np.random.seed(seed + 400)
            r = run_selective_kelly(returns, target_transitions=[("NEUTRAL", "BULL")], window_hours=window, shuffle_timestamps=True)
            shuf_sharpes.append(r['sharpe'])
        
        avg_s = np.mean(shuf_sharpes)
        drop = result['sharpe'] - avg_s
        status = " EDGE" if drop > 0.3 else ""
        
        print(f"{window}h          {result['sharpe']:+.4f}      {avg_s:+.4f}      {drop:+.4f}    {status}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    if best_transition and best_drop > 0.3:
        print(f"\n    BEST TRANSITION: {best_transition[0]} -> {best_transition[1]}")
        print(f"      Sharpe drop on shuffle: {best_drop:+.4f}")
        print(f"\n    TEMPORAL EDGE FOUND!")
        print(f"      -> Train RL to predict THIS specific transition")
        print(f"      -> Use Kelly ONLY during these windows")
    else:
        print(f"\n    No significant temporal edge found in any transition")
        print(f"      Best drop was: {best_drop:+.4f}")
        print(f"\n   Conclusion: Even selective trading doesn't exploit real patterns")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
