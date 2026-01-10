"""
HIMARI Layer 3 - PPO Agent Implementation
Proximal Policy Optimization for position sizing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    state_dim: int = 16
    action_dim: int = 1
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs mean and std for continuous action (position multiplier).
    Critic outputs state value.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: State value estimate
        """
        features = self.shared(state)
        
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action (clamped to [0, 2])
            log_prob: Log probability of action
        """
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clamp action to valid range [0, 2] for position multiplier
        action = torch.clamp(action, 0.0, 2.0)
        
        return action, log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action for PPO update.
        
        Returns:
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_mean, action_std, value = self.forward(state)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value


class RolloutBuffer:
    """Buffer for storing rollout trajectories."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, lambda_gae: float):
        """Compute GAE advantages and returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_gae * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return returns, advantages
    
    def get_batches(self, batch_size: int, device: str = 'cpu'):
        """Generate mini-batches for training."""
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(np.array([self.states[i] for i in batch_indices])).to(device),
                torch.FloatTensor(np.array([self.actions[i] for i in batch_indices])).to(device),
                torch.FloatTensor(np.array([self.log_probs[i] for i in batch_indices])).to(device),
                batch_indices
            )


class PPOAgent:
    """PPO Agent for position sizing."""
    
    def __init__(self, config: PPOConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Create actor-critic network
        self.actor_critic = ActorCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training stats
        self.total_steps = 0
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """Get action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor_critic.get_action(state_tensor, deterministic)
            _, _, value = self.actor_critic(state_tensor)
        
        return (
            action.squeeze().cpu().numpy(),
            log_prob.squeeze().cpu().item(),
            value.squeeze().cpu().item()
        )
    
    def update(self, last_value: float) -> dict:
        """Perform PPO update."""
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.lambda_gae
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.config.ppo_epochs):
            for states, actions, old_log_probs, indices in self.buffer.get_batches(
                self.config.batch_size, self.device
            ):
                # Get current policy outputs
                log_probs, entropy, values = self.actor_critic.evaluate_action(states, actions)
                
                # Get corresponding returns and advantages
                batch_returns = returns_tensor[indices].unsqueeze(1)
                batch_advantages = advantages_tensor[indices].unsqueeze(1)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs.unsqueeze(1))
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_coef * value_loss 
                    - self.config.entropy_coef * entropy.mean()
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
    
    def eval_mode(self):
        """Set to evaluation mode."""
        self.actor_critic.eval()
    
    def train_mode(self):
        """Set to training mode."""
        self.actor_critic.train()
