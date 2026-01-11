"""
HIMARI Layer 3: Component Isolation Test
=========================================

Isolates the contribution of each component:
1. PURE Volatility Targeting (no RL)
2. PURE RL (no volatility targeting)
3. Combined Hybrid (current strategy)

This helps identify WHERE the edge actually comes from.

Usage:
    python test_component_isolation.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ccxt
except ImportError:
    print("❌ CCXT not installed. Run: pip install ccxt")
    sys.exit(1)

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

# =============================================================================
# CONSTANTS
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)  # ~77.76
COMMISSION_RATE = 0.001  # 0.1%


# =============================================================================
# Configuration and Network (same as test_real_data_ccxt.py)
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
# STRATEGY 1: PURE VOLATILITY TARGETING (NO RL)
# =============================================================================

def run_pure_vol_targeting(
    prices: np.ndarray,
    initial_capital: float = 100000.0,
    target_volatility: float = 0.15,
    vol_lookback: int = 20,
    max_position_pct: float = 0.5
) -> Dict:
    """
    Pure volatility targeting without any RL component.
    Position = (target_vol / realized_vol) * base_allocation
    """
    returns = np.diff(prices) / prices[:-1]
    
    capital = initial_capital
    returns_history = []
    strategy_returns = []
    prev_position_pct = 0.0
    
    for t in range(len(returns)):
        ret = returns[t]
        returns_history.append(ret)
        
        if t < vol_lookback:
            strategy_returns.append(0)
            continue
        
        # Calculate realized volatility
        realized_vol = np.std(returns_history[-vol_lookback:]) * HOURLY_ANNUALIZATION
        realized_vol = max(realized_vol, 0.05)  # Floor
        
        # Volatility targeting: position = target_vol / realized_vol
        vol_scalar = target_volatility / realized_vol
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)
        
        # Position as percentage of capital
        position_pct = min(vol_scalar * 0.02, max_position_pct)
        
        # Apply commission
        position_change = abs(position_pct - prev_position_pct)
        commission = position_change * COMMISSION_RATE
        
        strategy_return = position_pct * ret - commission
        capital *= (1 + strategy_return)
        
        prev_position_pct = position_pct
        strategy_returns.append(strategy_return)
    
    returns_arr = np.array(strategy_returns)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - initial_capital) / initial_capital
    
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.max(running_max - cumulative)
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'strategy': 'Pure Vol-Targeting'
    }


# =============================================================================
# STRATEGY 2: PURE RL (NO VOLATILITY TARGETING)
# =============================================================================

def run_pure_rl(
    models: List[nn.Module],
    prices: np.ndarray,
    initial_capital: float = 100000.0,
    max_position_pct: float = 0.5
) -> Dict:
    """
    Pure RL-based positioning without volatility targeting.
    Position = RL output * max_position_pct (directly)
    """
    returns = np.diff(prices) / prices[:-1]
    
    capital = initial_capital
    returns_history = []
    price_history = []
    strategy_returns = []
    prev_position_pct = 0.0
    
    for t in range(len(returns)):
        ret = returns[t]
        returns_history.append(ret)
        price_history.append(prices[t])
        
        if t < 60:  # Warm-up period
            strategy_returns.append(0)
            continue
        
        # Build simple state from recent returns
        state = np.zeros(16, dtype=np.float32)
        
        # Fill state with recent price/return info
        recent_returns = returns_history[-20:]
        state[0] = np.mean(recent_returns) * 100
        state[1] = np.std(recent_returns) * 100
        state[2] = ret * 100
        state[3] = np.sum([1 for r in recent_returns if r > 0]) / len(recent_returns)
        state[4] = np.sum(recent_returns[-10:]) * 10 if len(recent_returns) >= 10 else 0
        state[5:16] = (np.array(recent_returns[-11:]) * 10) if len(recent_returns) >= 11 else np.zeros(11)
        
        # Get RL ensemble output
        rl_outputs = [predict_rl_raw(model, state) for model in models]
        ensemble_output = np.mean(rl_outputs)  # Range [-1, 1]
        
        # Directly use RL output for position (no vol targeting)
        position_pct = np.clip(ensemble_output * max_position_pct, -max_position_pct, max_position_pct)
        
        # Apply commission
        position_change = abs(position_pct - prev_position_pct)
        commission = position_change * COMMISSION_RATE
        
        strategy_return = position_pct * ret - commission
        capital *= (1 + strategy_return)
        
        prev_position_pct = position_pct
        strategy_returns.append(strategy_return)
    
    returns_arr = np.array(strategy_returns)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - initial_capital) / initial_capital
    
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.max(running_max - cumulative)
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'strategy': 'Pure RL'
    }


# =============================================================================
# STRATEGY 3: COMBINED HYBRID (Current Strategy)
# =============================================================================

def run_hybrid(
    models: List[nn.Module],
    prices: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict:
    """
    Combined hybrid strategy (vol-targeting + RL delta adjustment).
    This is the current strategy from test_real_data_ccxt.py
    """
    returns = np.diff(prices) / prices[:-1]
    
    capital = initial_capital
    returns_history = []
    price_history = []
    strategy_returns = []
    prev_position_pct = 0.0
    
    vol_lookback = 20
    target_volatility = 0.15
    max_rl_delta = 0.30
    
    for t in range(len(returns)):
        ret = returns[t]
        returns_history.append(ret)
        price_history.append(prices[t])
        
        if t < 60:
            strategy_returns.append(0)
            continue
        
        # Step 1: Calculate base position from volatility targeting
        realized_vol = np.std(returns_history[-vol_lookback:]) * HOURLY_ANNUALIZATION
        realized_vol = max(realized_vol, 0.05)
        
        vol_scalar = target_volatility / realized_vol
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)
        
        base_position_pct = min(vol_scalar * 0.02, 0.5)
        
        # Step 2: Get RL adjustment
        state = np.zeros(16, dtype=np.float32)
        recent_returns = returns_history[-20:]
        state[0] = np.mean(recent_returns) * 100
        state[1] = np.std(recent_returns) * 100
        state[2] = ret * 100
        state[3] = np.sum([1 for r in recent_returns if r > 0]) / len(recent_returns)
        state[4] = np.sum(recent_returns[-10:]) * 10 if len(recent_returns) >= 10 else 0
        
        rl_outputs = [predict_rl_raw(model, state) for model in models]
        ensemble_output = np.mean(rl_outputs)
        
        # Step 3: Apply bounded RL delta
        rl_delta = np.clip(ensemble_output, -1, 1) * max_rl_delta
        position_pct = base_position_pct * (1.0 + rl_delta)
        position_pct = np.clip(position_pct, 0, 0.5)
        
        # Apply commission
        position_change = abs(position_pct - prev_position_pct)
        commission = position_change * COMMISSION_RATE
        
        strategy_return = position_pct * ret - commission
        capital *= (1 + strategy_return)
        
        prev_position_pct = position_pct
        strategy_returns.append(strategy_return)
    
    returns_arr = np.array(strategy_returns)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - initial_capital) / initial_capital
    
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = np.max(running_max - cumulative)
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'strategy': 'Hybrid (Vol+RL)'
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 3: COMPONENT ISOLATION TEST")
    print("=" * 70)
    print()
    print("This test isolates the contribution of each component:")
    print("  1. PURE Vol-Targeting: Position sizing based only on volatility")
    print("  2. PURE RL: Position sizing based only on model output")
    print("  3. HYBRID: Vol-targeting with RL delta adjustment")
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
    
    if not models:
        print("\n⚠️ No models loaded!")
        return
    
    # Load cached data if available
    csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
    if os.path.exists(csv_path):
        print(f"\n📂 Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path)
        prices = df['close'].values
        print(f"   ✅ Loaded {len(prices):,} hourly candles")
    else:
        print("\n❌ No cached data found. Run test_real_data_ccxt.py first.")
        return
    
    # =========================================================================
    # Run All Three Strategies on ORIGINAL Data
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TESTING ON ORIGINAL (UNSHUFFLED) DATA")
    print("=" * 70)
    
    results_original = {
        'vol_only': run_pure_vol_targeting(prices),
        'rl_only': run_pure_rl(models, prices),
        'hybrid': run_hybrid(models, prices)
    }
    
    # =========================================================================
    # Run All Three Strategies on SHUFFLED Data
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TESTING ON SHUFFLED DATA (3 Trials)")
    print("=" * 70)
    
    returns = np.diff(prices) / prices[:-1]
    
    shuffled_results = {'vol_only': [], 'rl_only': [], 'hybrid': []}
    
    for trial in range(3):
        np.random.seed(trial + 42)
        shuffled_returns = returns.copy()
        np.random.shuffle(shuffled_returns)
        
        # Reconstruct shuffled prices
        shuffled_prices = np.zeros(len(prices))
        shuffled_prices[0] = prices[0]
        for i in range(len(shuffled_returns)):
            shuffled_prices[i + 1] = shuffled_prices[i] * (1 + shuffled_returns[i])
        
        print(f"\n   Trial {trial + 1}:")
        
        vol_result = run_pure_vol_targeting(shuffled_prices)
        rl_result = run_pure_rl(models, shuffled_prices)
        hybrid_result = run_hybrid(models, shuffled_prices)
        
        shuffled_results['vol_only'].append(vol_result['sharpe'])
        shuffled_results['rl_only'].append(rl_result['sharpe'])
        shuffled_results['hybrid'].append(hybrid_result['sharpe'])
        
        print(f"      Vol-Only: {vol_result['sharpe']:+.4f} | RL-Only: {rl_result['sharpe']:+.4f} | Hybrid: {hybrid_result['sharpe']:+.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("COMPONENT ISOLATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'Original':<15} {'Shuffled Avg':<15} {'Drop':<15}")
    print("-" * 70)
    
    for key, name in [('vol_only', 'Pure Vol-Targeting'), 
                      ('rl_only', 'Pure RL'),
                      ('hybrid', 'Hybrid (Vol+RL)')]:
        orig = results_original[key]['sharpe']
        shuf = np.mean(shuffled_results[key])
        drop = orig - shuf
        print(f"{name:<25} {orig:+.4f}         {shuf:+.4f}          {drop:+.4f}")
    
    # =========================================================================
    # Verdict
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("🔍 ANALYSIS")
    print("=" * 70)
    
    vol_orig = results_original['vol_only']['sharpe']
    vol_shuf = np.mean(shuffled_results['vol_only'])
    rl_orig = results_original['rl_only']['sharpe']
    rl_shuf = np.mean(shuffled_results['rl_only'])
    hybrid_orig = results_original['hybrid']['sharpe']
    hybrid_shuf = np.mean(shuffled_results['hybrid'])
    
    print(f"\n📊 Vol-Targeting Component:")
    if abs(vol_orig - vol_shuf) < 0.5:
        print(f"   ⚠️ Sharpe SURVIVES shuffle ({vol_orig:+.4f} → {vol_shuf:+.4f})")
        print(f"      → Vol-targeting works on ANY data (not predictive)")
    else:
        print(f"   ✅ Sharpe DROPS after shuffle ({vol_orig:+.4f} → {vol_shuf:+.4f})")
        print(f"      → Vol-targeting exploits real patterns")
    
    print(f"\n📊 RL Component:")
    if abs(rl_orig) < 0.1 and abs(rl_shuf) < 0.1:
        print(f"   ⚠️ RL produces near-zero Sharpe ({rl_orig:+.4f} → {rl_shuf:+.4f})")
        print(f"      → RL models are NOT adding predictive value")
    elif abs(rl_orig - rl_shuf) < 0.5:
        print(f"   ⚠️ RL Sharpe survives shuffle ({rl_orig:+.4f} → {rl_shuf:+.4f})")
        print(f"      → RL may have look-ahead bias")
    else:
        print(f"   ✅ RL Sharpe drops after shuffle ({rl_orig:+.4f} → {rl_shuf:+.4f})")
        print(f"      → RL exploits real patterns")
    
    print(f"\n📊 Combined Hybrid:")
    rl_contribution = hybrid_orig - vol_orig
    print(f"   Hybrid Sharpe: {hybrid_orig:+.4f}")
    print(f"   Vol-Only Sharpe: {vol_orig:+.4f}")
    print(f"   RL Contribution: {rl_contribution:+.4f}")
    
    if abs(rl_contribution) < 0.2:
        print(f"\n   🎯 CONCLUSION: Vol-targeting drives 95%+ of returns")
        print(f"      The RL ensemble adds minimal value over pure vol-targeting.")
    elif rl_contribution > 0.5:
        print(f"\n   🎯 CONCLUSION: RL adds significant alpha!")
    else:
        print(f"\n   🎯 CONCLUSION: RL has modest contribution")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
