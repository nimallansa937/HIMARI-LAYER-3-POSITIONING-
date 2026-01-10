"""
HIMARI Layer 3: Simple Ensemble Test
=====================================

Tests the 5 trained models with minimal dependencies.
"""

import os
import sys
import pickle
import numpy as np

# Need to define TrainingConfig before loading
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration - must match what was saved."""
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

# Now import torch after defining the class
import torch
import torch.nn as nn

# Make TrainingConfig available globally
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


def main():
    print("=" * 60)
    print("HIMARI Layer 3: Ensemble Test")
    print("=" * 60)
    
    # Model paths
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    model_files = [f"model_{i}.pt" for i in range(1, 6)]
    
    # Load models
    print("\nLoading models...")
    models = []
    for f in model_files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            try:
                model = load_model(path)
                models.append(model)
                print(f"  ✅ {f} loaded")
            except Exception as e:
                print(f"  ❌ {f} failed: {e}")
        else:
            print(f"  ⚠️ {f} not found")
    
    if not models:
        print("\n❌ No models loaded!")
        return
    
    print(f"\n{len(models)} models loaded successfully!")
    
    # Generate test scenarios
    print("\n" + "=" * 60)
    print("Running Backtest on Synthetic Data")
    print("=" * 60)
    
    np.random.seed(42)
    n_scenarios = 50
    
    all_individual_returns = [[] for _ in models]
    all_ensemble_returns = []
    
    for scenario_idx in range(n_scenarios):
        # Generate random scenario
        regime = np.random.choice(["bull", "bear", "mixed", "crash"])
        if regime == "bull":
            drift, vol = 0.0002, 0.02
        elif regime == "bear":
            drift, vol = -0.0001, 0.025
        elif regime == "crash":
            drift, vol = -0.001, 0.05
        else:
            drift, vol = 0.0, 0.03
        
        returns = np.random.normal(drift, vol, 1000)
        
        for t in range(16, len(returns) - 1):
            state = returns[t-16:t]
            market_return = returns[t]
            
            # Get predictions from each model
            positions = []
            for i, model in enumerate(models):
                pos = predict(model, state)
                positions.append(pos)
                
                # Track individual returns
                all_individual_returns[i].append(pos * market_return)
            
            # Ensemble average
            ensemble_pos = np.mean(positions)
            all_ensemble_returns.append(ensemble_pos * market_return)
        
        if (scenario_idx + 1) % 10 == 0:
            print(f"  Processed {scenario_idx + 1}/{n_scenarios} scenarios")
    
    # Calculate Sharpe ratios
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nIndividual Model Sharpe Ratios:")
    individual_sharpes = []
    for i, rets in enumerate(all_individual_returns, 1):
        rets = np.array(rets)
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
        individual_sharpes.append(sharpe)
        print(f"  Model {i}: {sharpe:.4f}")
    
    ens_rets = np.array(all_ensemble_returns)
    ensemble_sharpe = np.mean(ens_rets) / (np.std(ens_rets) + 1e-8)
    
    print(f"\n📊 ENSEMBLE SHARPE: {ensemble_sharpe:.4f}")
    print(f"   (Average of individuals: {np.mean(individual_sharpes):.4f})")
    print(f"   (Std of individuals: {np.std(individual_sharpes):.4f})")
    
    # Variance reduction
    individual_std = np.mean([np.std(r) for r in all_individual_returns])
    ensemble_std = np.std(all_ensemble_returns)
    variance_reduction = 1 - (ensemble_std / individual_std)
    
    print(f"\n📉 Variance Reduction: {variance_reduction*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINES")
    print("=" * 60)
    print(f"  Original 500K:     +0.046 (training)")
    print(f"  Bounded Delta:     -0.037 ❌")
    print(f"  Validation:        +0.0334")
    print(f"  Ensemble Test:     {ensemble_sharpe:+.4f} {'✅' if ensemble_sharpe > 0 else '⚠️'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
