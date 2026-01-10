"""
HIMARI Layer 3: Extended Backtest (6-12 Months BTC Data)
=========================================================

Fetches 6-12 months of real BTC data from Binance for proper validation.

Usage:
    python test_extended_backtest.py --months 6
"""

import os
import sys
import time
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict
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
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, config.lstm_layers, 
                           batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0)
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
    model.load_state_dict(checkpoint['network_state_dict'])
    model.eval()
    return model


def predict(model: nn.Module, state: np.ndarray) -> float:
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_t)
        return torch.tanh(action).item() * 0.5


def fetch_binance_extended(symbol: str = "BTCUSDT", 
                           interval: str = "5m",
                           months: int = 6) -> np.ndarray:
    """
    Fetch extended historical data from Binance.
    
    5-minute candles:
    - 1 day = 288 candles
    - 1 month = ~8,640 candles
    - 6 months = ~51,840 candles
    
    Binance allows 1000 candles per request, so we need multiple requests.
    """
    import requests
    
    print(f"\n📊 Fetching {months} months of {symbol} data...")
    
    # Calculate how many candles we need
    candles_per_day = 24 * 60 // 5  # 288 for 5-min
    total_candles = candles_per_day * 30 * months
    
    print(f"   Target: {total_candles:,} candles (~{months} months)")
    
    all_closes = []
    
    # Start from months ago
    end_time = int(datetime.now().timestamp() * 1000)
    
    requests_made = 0
    max_requests = (total_candles // 1000) + 1
    
    while len(all_closes) < total_candles and requests_made < max_requests:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or len(data) == 0:
                break
            
            # Extract closes (newest first in this batch)
            batch_closes = [float(candle[4]) for candle in data]
            
            # Prepend to list (so oldest is first at end)
            all_closes = batch_closes + all_closes
            
            # Move end_time back
            end_time = int(data[0][0]) - 1  # Before first candle of this batch
            
            requests_made += 1
            
            if requests_made % 10 == 0:
                print(f"   Fetched {len(all_closes):,} candles ({100*len(all_closes)/total_candles:.0f}%)")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ⚠️ Request failed: {e}")
            break
    
    if len(all_closes) < 1000:
        print(f"   ❌ Not enough data fetched ({len(all_closes)} candles)")
        return None
    
    # Convert to returns
    closes = np.array(all_closes)
    returns = np.diff(closes) / closes[:-1]
    
    # Calculate stats
    print(f"\n   ✅ Got {len(returns):,} 5-minute returns")
    print(f"   Period: {len(returns) / 288:.1f} days")
    
    # Monthly breakdown
    days = len(returns) / 288
    print(f"\n   Monthly Stats (annualized):")
    
    chunk_size = 288 * 30  # ~1 month
    for i in range(0, len(returns), chunk_size):
        chunk = returns[i:i+chunk_size]
        if len(chunk) > 100:
            month_num = i // chunk_size + 1
            monthly_ret = np.sum(chunk) * 100
            monthly_vol = np.std(chunk) * np.sqrt(288 * 365) * 100
            monthly_sharpe = np.mean(chunk) / (np.std(chunk) + 1e-8)
            print(f"   Month {month_num}: Return {monthly_ret:+.1f}%, Vol {monthly_vol:.0f}%, Sharpe {monthly_sharpe:.4f}")
    
    return returns


def backtest_ensemble(models: List[nn.Module], returns: np.ndarray) -> Dict:
    """Backtest ensemble on returns data."""
    lookback = 16
    strategy_returns = []
    buy_hold_returns = []
    positions = []
    
    print(f"\n   Running backtest on {len(returns):,} data points...")
    
    for t in range(lookback, len(returns) - 1):
        state = returns[t-lookback:t]
        market_return = returns[t]
        
        # Ensemble prediction
        preds = [predict(m, state) for m in models]
        pos = np.clip(np.mean(preds), -0.3, 0.3)
        
        strategy_returns.append(pos * market_return)
        buy_hold_returns.append(market_return * 0.3)  # 30% constant position
        positions.append(pos)
        
        if t % 50000 == 0 and t > 0:
            print(f"   Processed {t:,}/{len(returns):,} ({100*t/len(returns):.0f}%)")
    
    strat = np.array(strategy_returns)
    bh = np.array(buy_hold_returns)
    
    # Calculate metrics
    def calc_metrics(rets, name):
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
        total = np.sum(rets) * 100
        
        equity = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown) * 100
        
        return {
            "name": name,
            "sharpe": sharpe,
            "total_return_pct": total,
            "max_drawdown_pct": max_dd,
            "win_rate": np.mean(rets > 0) * 100
        }
    
    return {
        "ensemble": calc_metrics(strat, "Ensemble"),
        "buy_hold": calc_metrics(bh, "Buy & Hold 30%"),
        "avg_position": np.mean(np.abs(positions)),
        "position_std": np.std(positions),
        "num_trades": len(strat)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6, help="Number of months to backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"HIMARI Layer 3: Extended Backtest ({args.months} Months)")
    print("=" * 70)
    
    # Load models
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    
    print("\nLoading ensemble models...")
    models = []
    for i in range(1, 6):
        path = os.path.join(models_dir, f"model_{i}.pt")
        if os.path.exists(path):
            models.append(load_model(path))
            print(f"   ✅ model_{i}.pt")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Fetch extended data
    returns = fetch_binance_extended(args.symbol, "5m", args.months)
    
    if returns is None:
        print("❌ Failed to fetch data!")
        return
    
    # Run backtest
    print("\n" + "=" * 70)
    print("BACKTESTING")
    print("=" * 70)
    
    results = backtest_ensemble(models, returns)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    ens = results["ensemble"]
    bh = results["buy_hold"]
    
    print(f"\n{'Metric':<25} {'Ensemble':>15} {'Buy & Hold':>15}")
    print("-" * 55)
    print(f"{'Sharpe Ratio':<25} {ens['sharpe']:>15.4f} {bh['sharpe']:>15.4f}")
    print(f"{'Total Return':<25} {ens['total_return_pct']:>14.2f}% {bh['total_return_pct']:>14.2f}%")
    print(f"{'Max Drawdown':<25} {ens['max_drawdown_pct']:>14.2f}% {bh['max_drawdown_pct']:>14.2f}%")
    print(f"{'Win Rate':<25} {ens['win_rate']:>14.1f}% {bh['win_rate']:>14.1f}%")
    
    print(f"\n{'Avg Position Size':<25} {results['avg_position']*100:>14.1f}%")
    print(f"{'Position Std':<25} {results['position_std']*100:>14.1f}%")
    print(f"{'Total Steps':<25} {results['num_trades']:>15,}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    outperform = ens['sharpe'] > bh['sharpe']
    positive = ens['sharpe'] > 0
    
    if positive and outperform:
        print(f"\n✅ SUCCESS: Ensemble outperforms buy-hold ({ens['sharpe']:.4f} vs {bh['sharpe']:.4f})")
        print("   Your model adds value on real {args.months}-month BTC data!")
    elif positive:
        print(f"\n✅ POSITIVE: Ensemble Sharpe {ens['sharpe']:.4f} > 0")
        print(f"   But buy-hold did better ({bh['sharpe']:.4f})")
    else:
        print(f"\n⚠️ NEGATIVE: Ensemble Sharpe {ens['sharpe']:.4f}")
        print("   Consider CQL + RevIN improvements")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
