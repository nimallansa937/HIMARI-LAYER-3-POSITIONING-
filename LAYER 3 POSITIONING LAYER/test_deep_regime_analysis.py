"""
HIMARI Layer 3: Deep Regime Analysis
=====================================

Comprehensive analysis to answer:
1. What regime sequences generated the +6.34 Sharpe?
2. Does RL help in specific volatility regimes?
3. Does Kelly/momentum survive shuffle test?
4. Strategic decision tree for system design

Usage:
    python test_deep_regime_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

# =============================================================================
# CONSTANTS
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)
COMMISSION_RATE = 0.001


# =============================================================================
# Configuration and Network (same as other tests)
# =============================================================================

@dataclass
class TrainingConfig:
    state_dim: int = 16
    hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_interval: int = 50000
    patience: int = 3
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5
    commission_rate: float = 0.001

import __main__
__main__.TrainingConfig = TrainingConfig


class LSTMPPONetworkV2(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim, config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.actor(out[:, -1, :])[:, 0:1]


def load_model(path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', TrainingConfig())
    model = LSTMPPONetworkV2(config)
    model.load_state_dict(checkpoint['network_state_dict'], strict=True)
    model.eval()
    return model


def predict_rl_raw(model: nn.Module, state: np.ndarray) -> float:
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_t)
        return torch.tanh(action).item()


# =============================================================================
# ANALYSIS 1: Regime Sequence Analysis
# =============================================================================

def analyze_regime_sequences(prices: np.ndarray, returns: np.ndarray):
    """
    Identify what regime sequences generated the best performance.
    Track Bull→Bear, Bear→Bull transitions and their outcomes.
    """
    print("\n" + "=" * 70)
    print("📊 ANALYSIS 1: REGIME SEQUENCE PATTERNS")
    print("=" * 70)
    
    # Calculate rolling metrics
    vol_lookback = 20
    momentum_lookback = 50
    
    regimes = []
    volatilities = []
    momentums = []
    
    for t in range(momentum_lookback, len(returns)):
        # Realized volatility (annualized)
        vol = np.std(returns[t-vol_lookback:t]) * HOURLY_ANNUALIZATION
        volatilities.append(vol)
        
        # Momentum (cumulative return)
        mom = np.sum(returns[t-momentum_lookback:t])
        momentums.append(mom)
        
        # Classify regime
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
        
        regimes.append(regime)
    
    # Count regime distributions
    regime_counts = defaultdict(int)
    for r in regimes:
        regime_counts[r] += 1
    
    total = len(regimes)
    print("\n📈 Regime Distribution:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"   {regime:<12}: {count:>6} hours ({count/total*100:.1f}%)")
    
    # Analyze transitions
    transitions = defaultdict(lambda: {"count": 0, "avg_return_after": []})
    
    for t in range(1, len(regimes)):
        prev_regime = regimes[t-1]
        curr_regime = regimes[t]
        
        if prev_regime != curr_regime:
            key = f"{prev_regime} → {curr_regime}"
            transitions[key]["count"] += 1
            
            # Calculate return in next 6 hours after transition
            start_idx = momentum_lookback + t
            end_idx = min(start_idx + 6, len(returns))
            if end_idx > start_idx:
                future_return = np.sum(returns[start_idx:end_idx])
                transitions[key]["avg_return_after"].append(future_return)
    
    print("\n🔄 Top Regime Transitions (with avg 6h return after):")
    sorted_trans = sorted(transitions.items(), key=lambda x: -x[1]["count"])[:10]
    
    for trans_key, data in sorted_trans:
        avg_ret = np.mean(data["avg_return_after"]) * 100 if data["avg_return_after"] else 0
        print(f"   {trans_key:<25}: {data['count']:>4} times | Avg 6h return: {avg_ret:+.3f}%")
    
    # Find best performing windows
    print("\n🏆 Best Performing 30-Day Windows:")
    window_size = 24 * 30  # 30 days in hours
    
    window_sharpes = []
    for start in range(0, len(returns) - window_size, 24 * 7):  # Check every week
        window_rets = returns[start:start + window_size]
        sharpe = np.mean(window_rets) / (np.std(window_rets) + 1e-8) * HOURLY_ANNUALIZATION
        
        # Get dominant regime in this window
        if start >= momentum_lookback:
            window_regimes = regimes[start-momentum_lookback:start-momentum_lookback+window_size]
            most_common = max(set(window_regimes), key=window_regimes.count) if window_regimes else "UNKNOWN"
        else:
            most_common = "UNKNOWN"
        
        window_sharpes.append({
            'start': start,
            'sharpe': sharpe,
            'return': np.sum(window_rets) * 100,
            'regime': most_common
        })
    
    top_windows = sorted(window_sharpes, key=lambda x: -x['sharpe'])[:5]
    for i, w in enumerate(top_windows, 1):
        start_day = w['start'] // 24
        print(f"   #{i}: Day {start_day:>4}-{start_day+30} | Sharpe: {w['sharpe']:+.2f} | Return: {w['return']:+.1f}% | Regime: {w['regime']}")
    
    return regimes, volatilities, momentums


# =============================================================================
# ANALYSIS 2: RL Performance by Volatility Regime
# =============================================================================

def analyze_rl_by_volatility(
    models: List[nn.Module],
    prices: np.ndarray,
    returns: np.ndarray
):
    """
    Test if RL models work better in specific volatility regimes.
    """
    print("\n" + "=" * 70)
    print("📊 ANALYSIS 2: RL PERFORMANCE BY VOLATILITY REGIME")
    print("=" * 70)
    
    vol_lookback = 20
    
    # Calculate volatility for each timestep
    volatilities = []
    for t in range(vol_lookback, len(returns)):
        vol = np.std(returns[t-vol_lookback:t]) * HOURLY_ANNUALIZATION
        volatilities.append(vol)
    
    # Define volatility buckets
    vol_buckets = {
        'ultra_low': (0, 0.15),
        'low': (0.15, 0.30),
        'medium': (0.30, 0.50),
        'high': (0.50, 0.80),
        'crisis': (0.80, 10.0)
    }
    
    # Run RL on each bucket
    results_by_vol = {}
    
    for bucket_name, (vol_min, vol_max) in vol_buckets.items():
        bucket_returns_rl = []
        bucket_returns_bh = []
        bucket_returns_vol_target = []
        count = 0
        
        returns_history = list(returns[:vol_lookback])
        
        for t in range(vol_lookback, len(returns) - 1):
            vol = volatilities[t - vol_lookback]
            ret = returns[t]
            returns_history.append(ret)
            
            if vol_min <= vol < vol_max:
                count += 1
                
                # Build state
                state = np.zeros(16, dtype=np.float32)
                recent = returns_history[-20:]
                state[0] = np.mean(recent) * 100
                state[1] = np.std(recent) * 100
                state[2] = ret * 100
                state[3] = np.sum([1 for r in recent if r > 0]) / len(recent)
                state[4] = np.sum(recent[-10:]) * 10
                
                # RL prediction
                rl_outputs = [predict_rl_raw(model, state) for model in models]
                rl_action = np.mean(rl_outputs)
                
                # Position from RL (direct)
                rl_position = np.clip(rl_action * 0.5, -0.5, 0.5)
                
                # Vol-targeting position
                target_vol = 0.15
                vol_position = min(target_vol / (vol + 0.01) * 0.02, 0.5)
                
                # Calculate returns
                bucket_returns_rl.append(rl_position * ret)
                bucket_returns_bh.append(0.5 * ret)  # 50% B&H
                bucket_returns_vol_target.append(vol_position * ret)
        
        if count > 100:  # Only analyze buckets with enough data
            rl_arr = np.array(bucket_returns_rl)
            bh_arr = np.array(bucket_returns_bh)
            vt_arr = np.array(bucket_returns_vol_target)
            
            results_by_vol[bucket_name] = {
                'count': count,
                'vol_range': f"{vol_min:.2f}-{vol_max:.2f}",
                'rl_sharpe': np.mean(rl_arr) / (np.std(rl_arr) + 1e-8) * HOURLY_ANNUALIZATION,
                'bh_sharpe': np.mean(bh_arr) / (np.std(bh_arr) + 1e-8) * HOURLY_ANNUALIZATION,
                'vt_sharpe': np.mean(vt_arr) / (np.std(vt_arr) + 1e-8) * HOURLY_ANNUALIZATION,
                'rl_return': np.sum(rl_arr) * 100,
                'avg_rl_action': np.mean(rl_outputs) if count > 0 else 0
            }
    
    print("\n📈 Performance by Volatility Regime:")
    print(f"{'Regime':<12} {'Hours':<8} {'RL Sharpe':<12} {'Vol-Tgt':<12} {'RL Helps?':<10}")
    print("-" * 60)
    
    for bucket_name, data in results_by_vol.items():
        rl_helps = "✅ YES" if data['rl_sharpe'] > data['vt_sharpe'] and data['rl_sharpe'] > 0 else "❌ NO"
        print(f"{bucket_name:<12} {data['count']:<8} {data['rl_sharpe']:+.4f}      {data['vt_sharpe']:+.4f}      {rl_helps}")
    
    return results_by_vol


# =============================================================================
# ANALYSIS 3: Kelly/Momentum Shuffle Test
# =============================================================================

def analyze_kelly_momentum_with_shuffle(prices: np.ndarray, returns: np.ndarray, n_trials: int = 5):
    """
    Test if Kelly/Momentum component survives shuffle test.
    """
    print("\n" + "=" * 70)
    print("📊 ANALYSIS 3: KELLY/MOMENTUM SHUFFLE TEST")
    print("=" * 70)
    
    def run_kelly_momentum(rets, shuffle=False):
        """Run pure Kelly/Momentum strategy."""
        if shuffle:
            rets = rets.copy()
            np.random.shuffle(rets)
        
        capital = 100000.0
        returns_history = []
        strategy_returns = []
        
        for t in range(len(rets)):
            ret = rets[t]
            returns_history.append(ret)
            
            if t < 20:
                strategy_returns.append(0)
                continue
            
            recent = returns_history[-20:]
            
            # Kelly calculation
            wins = [r for r in recent if r > 0]
            losses = [r for r in recent if r < 0]
            
            if not losses or not wins:
                kelly = 0.1
            else:
                win_rate = len(wins) / len(recent)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                
                if avg_loss > 0:
                    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss + 1e-8)
                else:
                    kelly = 0.25
                
                kelly = np.clip(kelly, 0, 0.5)
            
            # Momentum factor
            momentum = np.sum(recent)
            momentum_factor = 1 + np.clip(momentum * 2, -0.3, 0.3)
            
            # Final position
            position = kelly * momentum_factor * 0.5
            position = np.clip(position, 0, 0.5)
            
            # Apply
            strategy_return = position * ret - abs(position) * COMMISSION_RATE * 0.1
            capital *= (1 + strategy_return)
            strategy_returns.append(strategy_return)
        
        arr = np.array(strategy_returns)
        sharpe = np.mean(arr) / (np.std(arr) + 1e-8) * HOURLY_ANNUALIZATION
        total_ret = (capital - 100000) / 100000
        
        return sharpe, total_ret
    
    # Original
    orig_sharpe, orig_ret = run_kelly_momentum(returns, shuffle=False)
    print(f"\n📈 Original Kelly/Momentum:")
    print(f"   Sharpe: {orig_sharpe:+.4f}")
    print(f"   Return: {orig_ret*100:+.2f}%")
    
    # Shuffled
    print(f"\n🔀 Shuffled Kelly/Momentum ({n_trials} trials):")
    shuffled_sharpes = []
    for trial in range(n_trials):
        np.random.seed(trial + 100)
        shuf_sharpe, shuf_ret = run_kelly_momentum(returns, shuffle=True)
        shuffled_sharpes.append(shuf_sharpe)
        print(f"   Trial {trial+1}: Sharpe {shuf_sharpe:+.4f}, Return {shuf_ret*100:+.2f}%")
    
    avg_shuffled = np.mean(shuffled_sharpes)
    sharpe_drop = orig_sharpe - avg_shuffled
    
    print(f"\n📊 Summary:")
    print(f"   Original Sharpe:  {orig_sharpe:+.4f}")
    print(f"   Shuffled Avg:     {avg_shuffled:+.4f}")
    print(f"   Drop:             {sharpe_drop:+.4f}")
    
    if abs(avg_shuffled) < 0.3 and sharpe_drop > 0.5:
        print(f"\n   ✅ Kelly/Momentum PASSES shuffle test!")
        print(f"      → This component exploits real market patterns")
        kelly_passes = True
    else:
        print(f"\n   ❌ Kelly/Momentum FAILS shuffle test")
        print(f"      → Edge survives shuffling, not predictive")
        kelly_passes = False
    
    return kelly_passes, orig_sharpe, avg_shuffled


# =============================================================================
# ANALYSIS 4: Strategic Decision Tree
# =============================================================================

def print_strategic_decision_tree(kelly_passes: bool, rl_by_vol: Dict):
    """
    Print strategic decision tree based on findings.
    """
    print("\n" + "=" * 70)
    print("🌳 STRATEGIC DECISION TREE")
    print("=" * 70)
    
    # Check if RL helps in any regime
    rl_helps_anywhere = any(
        data['rl_sharpe'] > data['vt_sharpe'] and data['rl_sharpe'] > 0 
        for data in rl_by_vol.values()
    )
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    STRATEGIC DECISION TREE                          │
└─────────────────────────────────────────────────────────────────────┘
    """)
    
    if kelly_passes:
        print("""
    Kelly/Momentum PASSES shuffle test
    ├── ✅ This is your REAL edge
    │
    ├── Next Steps:
    │   ├── 1. Keep Kelly/Momentum as core strategy
    │   ├── 2. Train RL to SIZE Kelly bets (not replace them)
    │   │       └── Input: market state → Output: Kelly multiplier [0.5, 2.0]
    │   │
    │   ├── 3. Train regime classifier
    │   │       └── Question: "Will Kelly work in next 6h?"
    │   │       └── Binary classifier on regime transitions
    │   │
    │   └── 4. Use vol-targeting as risk overlay (not alpha source)
    """)
    else:
        print("""
    Kelly/Momentum FAILS shuffle test
    ├── ❌ Kelly is NOT exploiting real patterns
    │
    ├── Next Steps:
    │   ├── 1. Investigate WHY Kelly survives shuffle
    │   │       └── Possible: Position sizing math creates artificial Sharpe
    │   │
    │   ├── 2. Look for different alpha sources:
    │   │       ├── Order flow imbalance
    │   │       ├── Cross-asset momentum (ETH vs BTC)
    │   │       └── Funding rate signals
    │   │
    │   └── 3. Consider fully rules-based system
    │           └── No ML until genuine edge is identified
    """)
    
    if rl_helps_anywhere:
        best_regime = max(rl_by_vol.items(), key=lambda x: x[1]['rl_sharpe'] - x[1]['vt_sharpe'])
        print(f"""
    RL Helps in Specific Regime: {best_regime[0].upper()}
    ├── RL adds value in {best_regime[0]} volatility conditions
    │
    └── Recommendation:
        ├── Use RL ONLY in {best_regime[0]} volatility regime
        ├── Use pure vol-targeting elsewhere
        └── Implement regime-conditional RL activation
    """)
    else:
        print("""
    RL Does NOT Help in ANY Regime
    ├── ❌ RL models are not adding value anywhere
    │
    └── Recommendation:
        ├── Remove RL from production
        ├── Retrain with different objective:
        │   ├── Don't predict price direction
        │   └── Predict optimal position SIZE given regime
        │
        └── Or: Use RL for risk management only
                └── "Should I reduce position now?" (binary)
    """)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        ACTION ITEMS                                  │
└─────────────────────────────────────────────────────────────────────┘

1. [ ] Document specific regime patterns that generated returns
2. [ ] Build regime transition detector (Rule-based, not ML)
3. [ ] Test Kelly/Momentum on different assets (ETH, SOL)
4. [ ] If Kelly passes: Train RL as Kelly "confidence adjuster"
5. [ ] If Kelly fails: Find new alpha source before using ML
    """)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 3: DEEP REGIME ANALYSIS")
    print("=" * 70)
    print()
    print("This analysis answers:")
    print("  1. What regime sequences generated the +6.34 Sharpe?")
    print("  2. Does RL help in specific volatility regimes?")
    print("  3. Does Kelly/Momentum survive shuffle test?")
    print("  4. What should the strategic direction be?")
    print()
    
    # Load models
    print("Loading RL ensemble models...")
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    model_files = [f"model_{i}.pt" for i in range(1, 6)]
    
    models = []
    for f in model_files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            try:
                model = load_model(path)
                models.append(model)
                print(f"   ✅ {f}")
            except Exception as e:
                print(f"   ❌ {f}: {e}")
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
    if os.path.exists(csv_path):
        print(f"\n📂 Loading cached data...")
        df = pd.read_csv(csv_path)
        prices = df['close'].values
        print(f"   ✅ Loaded {len(prices):,} hourly candles")
    else:
        print("\n❌ No cached data. Run test_real_data_ccxt.py first.")
        return
    
    returns = np.diff(prices) / prices[:-1]
    
    # Run all analyses
    regimes, vols, moms = analyze_regime_sequences(prices, returns)
    
    rl_by_vol = analyze_rl_by_volatility(models, prices, returns)
    
    kelly_passes, kelly_orig, kelly_shuf = analyze_kelly_momentum_with_shuffle(prices, returns)
    
    print_strategic_decision_tree(kelly_passes, rl_by_vol)
    
    # Save summary
    print("\n" + "=" * 70)
    print("📝 Analysis complete. Key findings:")
    print("=" * 70)
    print(f"   • Kelly/Momentum passes shuffle: {'YES ✅' if kelly_passes else 'NO ❌'}")
    print(f"   • RL helps in any regime: {'YES ✅' if any(d['rl_sharpe'] > d['vt_sharpe'] for d in rl_by_vol.values()) else 'NO ❌'}")
    print(f"   • Dominant regime: {max(set(regimes), key=regimes.count)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
