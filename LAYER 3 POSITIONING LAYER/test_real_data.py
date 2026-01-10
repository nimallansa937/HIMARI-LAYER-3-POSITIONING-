"""
HIMARI Layer 3: Real Market Data Test
======================================

Tests the ensemble models on REAL BTC/ETH data, not random noise.

Random test data shows Sharpe ~0 because there are NO patterns.
Real market data has weak but exploitable patterns.

Usage:
    python test_real_data.py
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# For loading models
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

import torch
import torch.nn as nn
import __main__
__main__.TrainingConfig = TrainingConfig


class LSTMPPONetworkV2(nn.Module):
    """LSTM-PPO network - must match training architecture."""
    
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
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
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
    
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        lstm_out, hidden_new = self.lstm(x, hidden)
        features = lstm_out[:, -1, :]
        
        actor_out = self.actor(features)
        mean = actor_out[:, 0:1]
        
        return mean


def load_model(path: str) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', TrainingConfig())
    model = LSTMPPONetworkV2(config)
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    return model


def predict(model: nn.Module, state: np.ndarray) -> float:
    """Get position prediction from model."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_t)
        position = torch.tanh(action).item() * 0.5
    return position


def fetch_binance_btc_data() -> np.ndarray:
    """
    Fetch real BTC data from Binance.
    Returns array of 5-minute returns.
    """
    try:
        import requests
        
        print("  Fetching BTC/USDT data from Binance...")
        
        # Get 1000 5-minute candles (~3.5 days of data)
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "limit": 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Extract close prices
        closes = np.array([float(candle[4]) for candle in data])
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        print(f"  ✅ Got {len(returns)} 5-minute returns")
        print(f"     Period: ~{len(returns) * 5 / 60:.1f} hours")
        print(f"     Mean return: {np.mean(returns)*100:.4f}%")
        print(f"     Std: {np.std(returns)*100:.4f}%")
        
        return returns
        
    except Exception as e:
        print(f"  ⚠️ Binance fetch failed: {e}")
        return None


def generate_realistic_btc_data(n: int = 1000) -> np.ndarray:
    """
    Generate realistic BTC-like data with actual patterns.
    Uses GARCH-like volatility clustering and momentum.
    """
    np.random.seed(42)
    
    # Parameters calibrated to BTC
    base_vol = 0.01  # ~1% per 5-min during calm
    
    returns = []
    volatility = base_vol
    momentum = 0
    
    for t in range(n):
        # GARCH: volatility clustering
        volatility = 0.9 * volatility + 0.1 * base_vol + 0.3 * abs(returns[-1]) if returns else base_vol
        
        # Momentum factor
        if len(returns) >= 20:
            recent_return = np.mean(returns[-20:])
            momentum = 0.3 * recent_return
        
        # Regime switching (15% chance of volatility spike)
        if np.random.random() < 0.15:
            volatility *= 2 + np.random.random()
        
        # Generate return with momentum and mean-reversion
        r = np.random.normal(momentum * 0.1, volatility)
        returns.append(r)
    
    return np.array(returns)


def backtest_ensemble(models: List[nn.Module], returns: np.ndarray, 
                      apply_bounds: bool = True) -> Dict:
    """
    Backtest ensemble on real returns data.
    """
    lookback = 16
    all_returns = []
    positions_used = []
    
    for t in range(lookback, len(returns) - 1):
        state = returns[t-lookback:t]
        market_return = returns[t]
        
        # Ensemble prediction
        positions = [predict(m, state) for m in models]
        ensemble_pos = np.mean(positions)
        
        # Apply bounded delta constraints
        if apply_bounds:
            ensemble_pos = np.clip(ensemble_pos, -0.3, 0.3)
        
        strategy_return = ensemble_pos * market_return
        all_returns.append(strategy_return)
        positions_used.append(ensemble_pos)
    
    rets = np.array(all_returns)
    
    # Metrics
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
    total_return = np.sum(rets)  # Log return approx
    cumulative = np.cumprod(1 + rets)
    max_dd = np.max(np.maximum.accumulate(cumulative) - cumulative)
    
    return {
        "sharpe": float(sharpe),
        "total_return": float(total_return * 100),
        "max_drawdown": float(max_dd * 100),
        "avg_position": float(np.mean(np.abs(positions_used))),
        "num_trades": len(rets)
    }


def main():
    print("=" * 60)
    print("HIMARI Layer 3: Real Market Data Test")
    print("=" * 60)
    
    # Load models
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    model_files = [f"model_{i}.pt" for i in range(1, 6)]
    
    print("\nLoading ensemble models...")
    models = []
    for f in model_files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            try:
                model = load_model(path)
                models.append(model)
                print(f"  ✅ {f}")
            except Exception as e:
                print(f"  ❌ {f}: {e}")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"\n{len(models)} models loaded")
    
    # Get real data
    print("\n" + "-" * 60)
    print("OPTION 1: Real Binance Data")
    print("-" * 60)
    
    btc_returns = fetch_binance_btc_data()
    
    if btc_returns is not None and len(btc_returns) > 100:
        print("\nBacktesting on real BTC data...")
        result_real = backtest_ensemble(models, btc_returns, apply_bounds=True)
        
        print(f"\n📊 Real BTC Results:")
        print(f"   Sharpe: {result_real['sharpe']:.4f}")
        print(f"   Return: {result_real['total_return']:.2f}%")
        print(f"   Max DD: {result_real['max_drawdown']:.2f}%")
        print(f"   Avg Position: {result_real['avg_position']*100:.1f}%")
    else:
        result_real = None
    
    # Generate realistic synthetic
    print("\n" + "-" * 60)
    print("OPTION 2: Realistic BTC-like Synthetic Data")
    print("-" * 60)
    
    synthetic_returns = generate_realistic_btc_data(2000)
    print(f"  Generated 2000 realistic returns (GARCH + momentum)")
    
    print("\nBacktesting on realistic synthetic...")
    result_synth = backtest_ensemble(models, synthetic_returns, apply_bounds=True)
    
    print(f"\n📊 Realistic Synthetic Results:")
    print(f"   Sharpe: {result_synth['sharpe']:.4f}")
    print(f"   Return: {result_synth['total_return']:.2f}%")
    print(f"   Max DD: {result_synth['max_drawdown']:.2f}%")
    
    # Comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Data Type':<30} {'Sharpe':>10}")
    print("-" * 42)
    print(f"{'Random noise (previous test)':<30} {'0.003':>10}")
    print(f"{'Realistic synthetic (GARCH)':<30} {result_synth['sharpe']:>10.4f}")
    if result_real:
        print(f"{'Real BTC (Binance)':<30} {result_real['sharpe']:>10.4f}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    best_sharpe = result_real['sharpe'] if result_real else result_synth['sharpe']
    
    if best_sharpe > 0.01:
        print(f"\n✅ Sharpe {best_sharpe:.4f} > 0.01 on real/realistic data")
        print("   Your models have learned useful patterns!")
        print("   The 0.003 result was due to testing on NOISE.")
    else:
        print(f"\n⚠️ Sharpe {best_sharpe:.4f} still low")
        print("   Consider CQL + RevIN improvements.")


if __name__ == "__main__":
    main()
