"""
HIMARI Layer 3: Test Existing Models on Transitions Only
=========================================================

Tests if existing RL models already work on NEUTRAL->BULL transitions.
This checks if the ensemble adds value in the specific windows we identified.

Usage:
    python test_models_on_transitions.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)
COMMISSION_RATE = 0.001


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


def predict_rl(model: nn.Module, state: np.ndarray) -> float:
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_t)
        return torch.tanh(action).item()


def classify_regime(returns_history: list, vol_lookback: int = 20, mom_lookback: int = 50) -> str:
    if len(returns_history) < mom_lookback:
        return "NEUTRAL"
    
    vol = np.std(returns_history[-vol_lookback:]) * HOURLY_ANNUALIZATION
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


def test_rl_on_transitions(
    models: List[nn.Module],
    returns: np.ndarray,
    target_transition: tuple = ("NEUTRAL", "BULL"),
    window_hours: int = 6
) -> dict:
    """Test if RL ensemble helps during specific transition windows."""
    
    # Detect transitions
    transitions = []
    returns_history = []
    prev_regime = "NEUTRAL"
    
    for t in range(len(returns)):
        returns_history.append(returns[t])
        curr_regime = classify_regime(returns_history)
        
        if curr_regime != prev_regime and (prev_regime, curr_regime) == target_transition:
            transitions.append(t)
        prev_regime = curr_regime
    
    # Build trading windows
    trading_windows = set()
    for start in transitions:
        for h in range(window_hours):
            if start + h < len(returns):
                trading_windows.add(start + h)
    
    # Run comparison: RL vs Kelly-only
    rl_returns = []
    kelly_returns = []
    
    returns_history = []
    
    for t in range(len(returns)):
        ret = returns[t]
        returns_history.append(ret)
        
        if t not in trading_windows or t < 50:
            continue
        
        recent = returns_history[-20:]
        
        # Kelly-only position
        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r < 0]
        
        if not losses or not wins:
            kelly = 0.15
        else:
            win_rate = len(wins) / len(recent)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss + 1e-8) if avg_loss > 0 else 0.25
            kelly = np.clip(kelly, 0, 0.5)
        
        mom = np.sum(recent[-10:])
        kelly_position = kelly * (1 + np.clip(mom * 3, 0, 0.5)) * 0.5
        kelly_position = np.clip(kelly_position, 0, 0.5)
        
        # RL-enhanced position
        state = np.zeros(16, dtype=np.float32)
        state[0] = np.mean(recent) * 100
        state[1] = np.std(recent) * 100
        state[2] = ret * 100
        state[3] = np.sum([1 for r in recent if r > 0]) / len(recent)
        state[4] = np.sum(recent[-10:]) * 10
        
        rl_outputs = [predict_rl(model, state) for model in models]
        rl_signal = np.mean(rl_outputs)  # [-1, 1]
        
        # RL adjusts Kelly sizing
        rl_adjustment = 1.0 + rl_signal * 0.5  # [0.5, 1.5] multiplier
        rl_position = kelly_position * rl_adjustment
        rl_position = np.clip(rl_position, 0, 0.5)
        
        rl_returns.append(rl_position * ret)
        kelly_returns.append(kelly_position * ret)
    
    if not rl_returns:
        return {'rl_sharpe': 0, 'kelly_sharpe': 0, 'rl_helps': False}
    
    rl_arr = np.array(rl_returns)
    kelly_arr = np.array(kelly_returns)
    
    rl_sharpe = np.mean(rl_arr) / (np.std(rl_arr) + 1e-8) * HOURLY_ANNUALIZATION
    kelly_sharpe = np.mean(kelly_arr) / (np.std(kelly_arr) + 1e-8) * HOURLY_ANNUALIZATION
    
    return {
        'rl_sharpe': rl_sharpe,
        'kelly_sharpe': kelly_sharpe,
        'rl_helps': rl_sharpe > kelly_sharpe,
        'improvement': rl_sharpe - kelly_sharpe,
        'n_windows': len(trading_windows),
        'avg_rl_signal': np.mean([predict_rl(models[0], np.zeros(16)) for _ in range(10)])
    }


def main():
    print("=" * 70)
    print("Test Existing Models on Transition Windows")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    
    models = []
    for i in range(1, 6):
        path = os.path.join(models_dir, f"model_{i}.pt")
        if os.path.exists(path):
            models.append(load_model(path))
            print(f"   [+] model_{i}.pt")
    
    if not models:
        print("No models found!")
        return
    
    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
    df = pd.read_csv(csv_path)
    prices = df['close'].values
    returns = np.diff(prices) / prices[:-1]
    
    print(f"\n   Loaded {len(returns):,} hourly returns")
    
    # Test on different transitions
    print("\n" + "=" * 70)
    print("RL Contribution During Transition Windows")
    print("=" * 70)
    
    transitions = [
        ("NEUTRAL", "BULL"),
        ("BEAR", "NEUTRAL"),
        ("HIGH_VOL", "NEUTRAL"),
    ]
    
    print(f"\n{'Transition':<20} {'Kelly':<12} {'RL+Kelly':<12} {'Diff':<10} {'RL Helps?'}")
    print("-" * 60)
    
    for trans in transitions:
        result = test_rl_on_transitions(models, returns, trans, window_hours=6)
        status = "[+] YES" if result['rl_helps'] else "[-] NO"
        trans_str = f"{trans[0]}->{trans[1]}"
        print(f"{trans_str:<20} {result['kelly_sharpe']:+.4f}      {result['rl_sharpe']:+.4f}      {result['improvement']:+.4f}    {status}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    nb_result = test_rl_on_transitions(models, returns, ("NEUTRAL", "BULL"), 6)
    
    if nb_result['rl_helps']:
        print("\n   [+] RL ADDS VALUE on NEUTRAL->BULL transitions!")
        print(f"      Improvement: {nb_result['improvement']:+.4f} Sharpe")
        print("      -> Can use existing models within transition windows")
    else:
        print("\n   [-] RL does NOT help on transitions")
        print("      -> Need to retrain for transition prediction")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
