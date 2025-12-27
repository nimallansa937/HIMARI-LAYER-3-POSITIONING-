"""
HIMARI Layer 3 - Multi-Asset Training Environment
Training on BTC, ETH, SOL simultaneously with shared encoder.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random


@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset training."""
    assets: List[str] = field(default_factory=lambda: ['BTC-USD', 'ETH-USD', 'SOL-USD'])
    state_dim: int = 16
    action_dim: int = 1
    shared_hidden_dim: int = 128
    asset_hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64


class SharedFeatureEncoder(nn.Module):
    """
    Shared feature encoder across all assets.
    Learns common patterns (volatility, momentum, risk metrics).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class AssetSpecificHead(nn.Module):
    """
    Asset-specific policy head.
    Captures unique characteristics of each asset.
    """
    
    def __init__(self, shared_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output [0, 1], scale to [0, 2]
        )
        
        self.value = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = self.policy(features) * 2.0  # Scale to [0, 2]
        value = self.value(features)
        return action, value


class MultiAssetPolicy(nn.Module):
    """
    Multi-asset policy network with shared encoder and asset-specific heads.
    """
    
    def __init__(self, config: MultiAssetConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.shared_encoder = SharedFeatureEncoder(
            config.state_dim,
            config.shared_hidden_dim
        )
        
        # Asset-specific heads
        self.asset_heads = nn.ModuleDict({
            asset.replace('-', '_'): AssetSpecificHead(
                config.shared_hidden_dim,
                config.asset_hidden_dim,
                config.action_dim
            )
            for asset in config.assets
        })
    
    def forward(self, state: torch.Tensor, asset: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for specific asset.
        
        Args:
            state: (batch, state_dim)
            asset: Asset symbol (e.g., 'BTC-USD')
            
        Returns:
            action, value
        """
        # Shared features
        features = self.shared_encoder(state)
        
        # Asset-specific prediction
        asset_key = asset.replace('-', '_')
        action, value = self.asset_heads[asset_key](features)
        
        return action, value
    
    def get_all_actions(self, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get actions for all assets."""
        actions = {}
        for asset, state in states.items():
            action, _ = self.forward(state, asset)
            actions[asset] = action
        return actions


class MultiAssetEnvironment:
    """
    Simulated multi-asset trading environment.
    Each asset has correlated but distinct dynamics.
    """
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.assets = config.assets
        
        # Asset-specific parameters
        self.asset_params = {
            'BTC-USD': {'volatility': 0.03, 'trend': 0.001, 'correlation': 1.0},
            'ETH-USD': {'volatility': 0.04, 'trend': 0.0015, 'correlation': 0.85},
            'SOL-USD': {'volatility': 0.06, 'trend': 0.002, 'correlation': 0.7}
        }
        
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial states."""
        self.step_count = 0
        self.prices = {asset: 100.0 for asset in self.assets}
        self.positions = {asset: 0.0 for asset in self.assets}
        self.pnl = {asset: 0.0 for asset in self.assets}
        
        return self._get_states()
    
    def _get_states(self) -> Dict[str, np.ndarray]:
        """Generate state vectors for all assets."""
        states = {}
        
        # Generate correlated market conditions
        base_momentum = np.random.randn() * 0.02
        base_volatility = abs(np.random.randn() * 0.01) + 0.01
        
        for asset in self.assets:
            params = self.asset_params.get(asset, {'volatility': 0.03, 'correlation': 1.0})
            
            # State components
            state = np.zeros(self.config.state_dim)
            state[0] = np.random.uniform(0.5, 1.0)  # Signal confidence
            state[1:4] = np.random.multinomial(1, [0.3, 0.4, 0.3])  # Action one-hot
            state[4:8] = np.random.multinomial(1, [0.4, 0.3, 0.2, 0.1])  # Tier one-hot
            state[8] = self.positions[asset]  # Current position
            state[9] = 1 if self.positions[asset] > 0 else -1  # Position side
            state[10] = self.pnl[asset] / 100.0  # PnL %
            state[11] = base_momentum * params['correlation']  # Momentum 1h
            state[12] = base_momentum * params['correlation'] * 0.8  # Momentum 4h
            state[13] = base_volatility * params['volatility'] / 0.03  # Volatility
            state[14] = np.random.uniform(0.4, 0.7)  # Win rate
            state[15] = np.random.uniform(0.0, 0.3)  # Cascade risk
            
            states[asset] = state
        
        return states
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool]:
        """
        Take actions for all assets.
        
        Args:
            actions: Dict of asset -> position multiplier [0, 2]
            
        Returns:
            next_states, rewards, done
        """
        self.step_count += 1
        rewards = {}
        
        for asset in self.assets:
            action = actions.get(asset, 1.0)
            params = self.asset_params.get(asset, {'volatility': 0.03, 'trend': 0.001})
            
            # Simulate price movement
            price_return = (
                params['trend'] 
                + np.random.randn() * params['volatility']
            )
            
            self.prices[asset] *= (1 + price_return)
            
            # Calculate reward based on position sizing
            position_size = action * 0.5  # Scale action to position
            pnl_change = position_size * price_return * 100
            
            # Risk-adjusted reward
            risk_penalty = max(0, (action - 1.5)) * 0.01  # Penalty for extreme sizing
            volatility_adjustment = min(1.0, 0.02 / params['volatility'])
            
            reward = pnl_change * volatility_adjustment - risk_penalty
            
            self.pnl[asset] += pnl_change
            self.positions[asset] = position_size
            rewards[asset] = reward
        
        done = self.step_count >= 500
        next_states = self._get_states()
        
        return next_states, rewards, done


class MultiAssetTrainer:
    """Trainer for multi-asset RL."""
    
    def __init__(self, config: MultiAssetConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        self.policy = MultiAssetPolicy(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        self.env = MultiAssetEnvironment(config)
    
    def train_episode(self) -> Dict[str, float]:
        """Train one episode across all assets."""
        states = self.env.reset()
        episode_rewards = {asset: 0.0 for asset in self.config.assets}
        
        while True:
            # Get actions for all assets
            actions = {}
            for asset, state in states.items():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy(state_tensor, asset)
                actions[asset] = action.squeeze().cpu().item()
            
            # Step environment
            next_states, rewards, done = self.env.step(actions)
            
            # Accumulate rewards
            for asset in self.config.assets:
                episode_rewards[asset] += rewards[asset]
            
            if done:
                break
            
            states = next_states
        
        return episode_rewards
    
    def train(self, num_episodes: int = 1000) -> List[Dict[str, float]]:
        """Train for multiple episodes."""
        all_rewards = []
        
        for episode in range(num_episodes):
            rewards = self.train_episode()
            all_rewards.append(rewards)
            
            if (episode + 1) % 100 == 0:
                avg_rewards = {
                    asset: np.mean([r[asset] for r in all_rewards[-100:]])
                    for asset in self.config.assets
                }
                print(f"Episode {episode + 1}: {avg_rewards}")
        
        return all_rewards
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
