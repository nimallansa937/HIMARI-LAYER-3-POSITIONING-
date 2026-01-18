"""
Test WFO Model on Real BTC Data
================================
Downloads real BTC data and tests the WFO-trained PPO model.
"""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/rl')

import torch
import numpy as np
import ccxt
from datetime import datetime, timedelta

# Import PPO agent
from rl.ppo_agent import PPOAgent, PPOConfig


def download_btc_hourly(days=365):
    """Download BTC hourly data from Binance."""
    print(f"Downloading {days} days of BTC hourly data...")
    
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(all_ohlcv) >= days * 24:
            break
    
    prices = np.array([candle[4] for candle in all_ohlcv])  # Close prices
    print(f"Downloaded {len(prices)} hourly candles")
    return prices


def simple_backtest(prices, model, capital=100000):
    """Run backtest with WFO PPO model."""
    print("\nRunning WFO Model Backtest...")
    print("=" * 60)
    
    returns = np.diff(prices) / prices[:-1]
    
    capital_history = [capital]
    positions = []
    
    # Simple state: last 16 features (returns, vol, momentum, etc.)
    lookback = 20
    
    for t in range(lookback, len(returns)):
        # Create simple state vector (16 features)
        recent_returns = returns[t-lookback:t]
        
        state = np.zeros(16, dtype=np.float32)
        state[0] = np.mean(recent_returns) * 100  # Avg return
        state[1] = np.std(recent_returns) * 100   # Volatility
        state[2] = returns[t-1] * 100 if t > 0 else 0  # Last return
        state[3] = np.sum(recent_returns > 0) / lookback  # Win rate
        state[4] = (prices[t] - np.mean(prices[t-lookback:t])) / np.mean(prices[t-lookback:t])  # Momentum
        state[5:16] = recent_returns[-11:] * 10  # Recent returns
        
        # Get action from model
        action, _ = model.get_action(state, deterministic=True)
        position_pct = (action - 1.0) * 0.3  # Scale to [-0.3, 0.3]
        position_pct = np.clip(position_pct, -0.3, 0.3)
        
        positions.append(position_pct)
        
        # Calculate return
        strategy_return = position_pct * returns[t]
        capital *= (1 + strategy_return)
        capital_history.append(capital)
    
    # Calculate metrics
    strategy_returns = np.diff(capital_history) / capital_history[:-1]
    
    total_return = (capital - 100000) / 100000
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(8760)
    
    # Max drawdown
    peak = np.maximum.accumulate(capital_history)
    drawdown = (peak - capital_history) / peak
    max_dd = np.max(drawdown)
    
    # Buy & Hold comparison
    bh_return = (prices[-1] - prices[lookback]) / prices[lookback]
    
    print(f"\n{'='*60}")
    print("WFO MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Total Return:     {total_return*100:+.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.4f}")
    print(f"Max Drawdown:     {max_dd*100:.2f}%")
    print(f"Avg Position:     {np.mean(np.abs(positions))*100:.1f}%")
    print(f"{'='*60}")
    print(f"Buy & Hold:       {bh_return*100:+.2f}%")
    print(f"Alpha:            {(total_return - bh_return)*100:+.2f}%")
    print(f"{'='*60}")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'buy_hold_return': bh_return
    }


def main():
    print("=" * 60)
    print("WFO MODEL TEST - Real BTC Data")
    print("=" * 60)
    
    # Load WFO model
    model_path = "../wfo_models/checkpoints/rl_policy_final.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Looking for model...")
        # Try alternative paths
        alt_paths = [
            "wfo_models/checkpoints/rl_policy_final.pt",
            "../wfo_models/checkpoints/rl_policy_final.pt",
            "../../wfo_models/checkpoints/rl_policy_final.pt",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break
    
    print(f"\nLoading model from: {model_path}")
    
    # Create agent and load weights
    config = PPOConfig(state_dim=16, hidden_dim=64)
    agent = PPOAgent(config, device='cpu')
    agent.load(model_path)
    agent.eval_mode()
    
    print("Model loaded successfully!")
    
    # Download real data
    prices = download_btc_hourly(days=365)
    
    # Run backtest
    results = simple_backtest(prices, agent)
    
    return results


if __name__ == "__main__":
    main()
