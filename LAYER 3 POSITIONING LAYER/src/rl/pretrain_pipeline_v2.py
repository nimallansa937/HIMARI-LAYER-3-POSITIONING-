"""
HIMARI Layer 3: Improved Pre-Training Pipeline V2
==================================================

V2 Improvements based on diagnosis:
1. Dropout regularization (0.2) to prevent overfitting
2. Early stopping with checkpoint evaluation
3. Simple raw rewards (no complex reward shaping)
4. Validation-based model selection

Expected: Sharpe +0.046 → +0.08 to +0.12
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with regularization."""
    # Architecture
    state_dim: int = 16
    hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2  # KEY: Dropout for regularization
    
    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Early stopping
    checkpoint_interval: int = 50000
    patience: int = 3  # Stop if no improvement for 3 checkpoints
    
    # Environment
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5
    commission_rate: float = 0.001


# ============================================================================
# NEURAL NETWORK WITH DROPOUT
# ============================================================================

class LSTMPPONetworkV2(nn.Module):
    """LSTM-PPO network with dropout regularization."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Input projection with dropout
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)  # DROPOUT 1
        )
        
        # LSTM (has built-in dropout between layers)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        
        # Actor head with dropout
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),  # DROPOUT 2
            nn.Linear(config.hidden_dim // 2, 2)  # mean, log_std
        )
        
        # Critic head with dropout
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),  # DROPOUT 3
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, hidden=None):
        """Forward pass - handles both 2D and 3D input."""
        # Handle 2D input (batch, features) -> expand to (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        x = self.input_proj(x)
        lstm_out, hidden_new = self.lstm(x, hidden)
        
        # Take last timestep output
        features = lstm_out[:, -1, :]
        
        actor_out = self.actor(features)
        mean = actor_out[:, 0:1]
        log_std = actor_out[:, 1:2]
        std = torch.exp(torch.clamp(log_std, -5, 2))
        
        value = self.critic(features)
        
        return mean, std, value, hidden_new
    
    def get_action(self, state, hidden=None, deterministic=False):
        """Sample action."""
        mean, std, value, hidden_new = self.forward(state, hidden)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, hidden_new


# ============================================================================
# SIMPLE REWARD FUNCTION (NO COMPLEX SHAPING)
# ============================================================================

class SimpleReward:
    """
    Simple P&L-based reward.
    
    Key insight: Complex reward shaping HURT performance.
    Simple raw returns worked better (+0.046 Sharpe).
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.returns_history = []
        self.peak_equity = 1.0
        self.current_equity = 1.0
    
    def compute_reward(self, position_return: float, regime: str = "unknown") -> Tuple[float, Dict]:
        """
        Regime-aware reward:
        - Bull: Uses Sortino Ratio (penalize only downside vol) to encourage upside.
        - Bear/Other: Uses Sharpe Ratio (penalize all vol) for safety.
        """
        self.returns_history.append(position_return)
        self.current_equity *= (1 + position_return)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        drawdown = 1 - (self.current_equity / self.peak_equity)
        
        if len(self.returns_history) >= self.window_size:
            recent = np.array(self.returns_history[-self.window_size:])
            mean_ret = np.mean(recent)
            
            if regime == "bull":
                # Sortino: Only penalize negative returns
                downside = recent[recent < 0]
                std_down = np.std(downside) if len(downside) > 0 else 1e-6
                reward = mean_ret / (std_down + 1e-8)
            else:
                # Sharpe: Penalize all volatility
                std_ret = np.std(recent) + 1e-8
                reward = mean_ret / std_ret
        else:
            reward = 0.0
        
        info = {
            "equity": self.current_equity,
            "drawdown": drawdown,
            "reward": reward
        }
        
        return reward, info
    
    def get_sharpe(self) -> float:
        """Calculate episode Sharpe."""
        if len(self.returns_history) < 2:
            return 0.0
        returns = np.array(self.returns_history)
        return float(np.mean(returns) / (np.std(returns) + 1e-8))
    
    def reset(self):
        self.returns_history = []
        self.peak_equity = 1.0
        self.current_equity = 1.0


# ============================================================================
# TRAINER WITH EARLY STOPPING
# ============================================================================

class PPOTrainerV2:
    """PPO trainer with dropout and early stopping."""
    
    def __init__(self, config: TrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        self.network = LSTMPPONetworkV2(config).to(device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        self.reward_fn = SimpleReward()
        self.hidden = None
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Early stopping tracking
        self.best_val_sharpe = -float('inf')
        self.patience_counter = 0
    
    def select_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Select action."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value, self.hidden = self.network.get_action(
                state_t, self.hidden
            )
            
            action_np = action.cpu().numpy()[0, 0]
            log_prob_np = log_prob.cpu().numpy()[0, 0]
            value_np = value.cpu().numpy()[0, 0]
            
            return action_np, log_prob_np, value_np
    
    def step(self, state: np.ndarray, market_return: float, regime: str = "unknown") -> Tuple[float, Dict]:
        """Execute one step."""
        action, log_prob, value = self.select_action(state)
        
        # Simple position sizing from action
        position_pct = np.clip(np.tanh(action) * self.config.max_position_pct, 
                               -self.config.max_position_pct, 
                               self.config.max_position_pct)
        position_return = position_pct * market_return
        
        reward, info = self.reward_fn.compute_reward(position_return, regime)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
        info["action"] = action
        info["position_pct"] = position_pct
        
        return action, info
    
    def end_episode(self) -> Dict:
        """End episode."""
        sharpe = self.reward_fn.get_sharpe()
        equity = self.reward_fn.current_equity
        
        self.reward_fn.reset()
        self.hidden = None
        
        return {"episode_sharpe": sharpe, "final_equity": equity}
    
    def update(self):
        """PPO update."""
        if len(self.states) < 64:
            return {}
        
        # Compute GAE
        rewards = np.array(self.rewards)
        values = np.array(self.values + [0.0])
        
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t+1] - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.actions)).unsqueeze(-1).to(self.device)
        returns_t = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages_t = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(-1).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO update
        self.network.train()  # Enable dropout
        
        for _ in range(4):  # update epochs
            mean, std, values, _ = self.network(states_t)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy()
            
            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                1 + self.config.clip_epsilon) * advantages_t
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy_bonus = entropy.mean()
            
            loss = (policy_loss + 
                    self.config.value_coef * value_loss - 
                    self.config.entropy_coef * entropy_bonus)
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 
                                     self.config.max_grad_norm)
            self.optimizer.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        return {"loss": loss.item()}
    
    def evaluate(self, scenarios: List[Dict]) -> float:
        """Evaluate on validation scenarios."""
        self.network.eval()  # Disable dropout
        
        sharpes = []
        for scenario in scenarios[:50]:  # Use 50 scenarios for validation
            state_idx = 20
            self.reward_fn.reset()
            self.hidden = None
            regime = scenario.get("type", "unknown")
            
            while state_idx < len(scenario['returns']) - 1:
                # Simple state from returns
                if state_idx >= 16:
                    state = scenario['returns'][state_idx-16:state_idx]
                else:
                    state = np.zeros(16)
                
                market_return = scenario['returns'][state_idx]
                self.step(state, market_return, regime=regime)
                state_idx += 1
            
            sharpes.append(self.reward_fn.get_sharpe())
        
        self.network.train()  # Re-enable dropout
        return float(np.mean(sharpes))
    
    def save(self, path: str):
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="/tmp/synthetic_data/stress_scenarios.pkl")
    parser.add_argument("--output", type=str, default="/tmp/models/pretrained_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load scenarios
    with open(args.data, 'rb') as f:
        scenarios = pickle.load(f)
    logger.info(f"Loaded {len(scenarios)} scenarios")
    
    # Split train/val (80/20)
    split_idx = int(len(scenarios) * 0.8)
    train_scenarios = scenarios[:split_idx]
    val_scenarios = scenarios[split_idx:]
    logger.info(f"Train: {len(train_scenarios)}, Val: {len(val_scenarios)}")
    
    # Initialize
    config = TrainingConfig()
    trainer = PPOTrainerV2(config, device=args.device)
    
    # W&B
    if WANDB_AVAILABLE:
        wandb.init(
            project="himari-layer3-v2-regularized",
            name=f"dropout_{config.dropout}_seed_{args.seed}",
            config={"dropout": config.dropout, "seed": args.seed}
        )
    
    # Training
    total_steps = 0
    episode = 0
    recent_sharpes = deque(maxlen=100)
    best_val_sharpe = -float('inf')
    patience = 0
    
    logger.info("=" * 60)
    logger.info("Starting Training V2 (Dropout + Early Stopping)")
    logger.info("=" * 60)
    logger.info(f"Dropout: {config.dropout}")
    logger.info(f"Target steps: {args.steps:,}")
    
    while total_steps < args.steps:
        scenario = random.choice(train_scenarios)
        state_idx = 20
        trainer.reward_fn.reset()
        trainer.hidden = None
        regime = scenario.get("type", "unknown")
        
        while state_idx < len(scenario['returns']) - 1 and total_steps < args.steps:
            if state_idx >= 16:
                state = scenario['returns'][state_idx-16:state_idx]
            else:
                state = np.zeros(16)
            
            market_return = scenario['returns'][state_idx]
            _, info = trainer.step(state, market_return, regime=regime)
            
            state_idx += 1
            total_steps += 1
            
            # Update every 128 steps
            if len(trainer.states) >= 128:
                trainer.update()
        
        # End episode
        episode += 1
        stats = trainer.end_episode()
        recent_sharpes.append(stats["episode_sharpe"])
        
        # Logging
        if episode % 10 == 0:
            avg_sharpe = np.mean(recent_sharpes)
            progress = 100 * total_steps / args.steps
            logger.info(f"Ep {episode:4d} | Steps: {total_steps:7d}/{args.steps} ({progress:.1f}%) | Sharpe: {avg_sharpe:.4f}")
        
        # Checkpoint + Validation
        if total_steps % config.checkpoint_interval == 0 and total_steps > 0:
            # Evaluate on validation
            val_sharpe = trainer.evaluate(val_scenarios)
            logger.info(f"📊 Validation Sharpe: {val_sharpe:.4f}")
            
            # Save checkpoint
            ckpt_path = os.path.join(args.output, f"checkpoint_{total_steps}.pt")
            trainer.save(ckpt_path)
            
            # Early stopping check
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience = 0
                # Save best model
                best_path = os.path.join(args.output, "best_model.pt")
                trainer.save(best_path)
                logger.info(f"✅ New best! Saved to {best_path}")
            else:
                patience += 1
                logger.info(f"⚠️ No improvement. Patience: {patience}/{config.patience}")
                
                if patience >= config.patience:
                    logger.info("🛑 Early stopping triggered!")
                    break
            
            if WANDB_AVAILABLE:
                wandb.log({
                    "val_sharpe": val_sharpe,
                    "best_val_sharpe": best_val_sharpe,
                    "patience": patience
                }, step=total_steps)
    
    # Final save
    final_path = os.path.join(args.output, "final_model.pt")
    trainer.save(final_path)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation Sharpe: {best_val_sharpe:.4f}")
    logger.info(f"Best model: {os.path.join(args.output, 'best_model.pt')}")
    logger.info("=" * 60)
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
