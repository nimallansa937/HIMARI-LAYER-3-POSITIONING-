"""
HIMARI Layer 3 - PPO RL Agent
===============================

Proximal Policy Optimization agent for position sizing.

The agent learns to output a position size multiplier [0, 2] that
adjusts the base position size from Layer 3 Phase 1.

Uses PyTorch for neural networks.

Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    state_dim: int = 16
    action_dim: int = 1          # Continuous: position multiplier
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    lambda_gae: float = 0.95     # GAE parameter
    clip_epsilon: float = 0.2    # PPO clip parameter
    value_coef: float = 0.5      # Value loss coefficient
    entropy_coef: float = 0.01   # Entropy bonus coefficient
    max_grad_norm: float = 0.5   # Gradient clipping


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Actor: Outputs mean and std for Gaussian policy over position multiplier
    Critic: Outputs value estimate V(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy): outputs (mean, log_std)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1], will scale to [0, 2]
        )

        # Log std is a learnable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            mean: Action mean (batch_size, 1)
            std: Action std (batch_size, 1)
            value: Value estimate (batch_size, 1)
        """
        features = self.shared(state)

        # Actor: mean in [0, 1], scale to [0, 2]
        mean = self.actor_mean(features) * 2.0

        # Std: clamp to avoid numerical issues
        std = torch.exp(self.actor_log_std).expand_as(mean)
        std = torch.clamp(std, min=0.01, max=1.0)

        # Critic
        value = self.critic(features)

        return mean, std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor
            deterministic: If True, return mean action

        Returns:
            action: Sampled action (batch_size, 1)
            log_prob: Log probability of action
        """
        mean, std, _ = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Clamp action to [0, 2]
        action = torch.clamp(action, 0.0, 2.0)

        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log prob and value for given state-action pairs.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_prob: Log probability of actions
            value: Value estimates
            entropy: Policy entropy
        """
        mean, std, value = self.forward(state)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, value, entropy


class PPOAgent:
    """
    PPO agent for position sizing optimization.

    Action Space: Continuous [0, 2]
        - 0.0 = no position
        - 1.0 = base position size (from Layer 3 Phase 1)
        - 2.0 = 2x base position size

    Reward: Sharpe ratio of recent returns
    """

    def __init__(self, config: Optional[PPOConfig] = None, device: str = 'cpu'):
        """
        Initialize PPO agent.

        Args:
            config: PPO configuration
            device: 'cpu' or 'cuda'
        """
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        # Create network
        self.policy = ActorCritic(
            state_dim=self.config.state_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Training mode
        self.training = True

        logger.info(f"PPO Agent initialized (device={device})")

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """
        Get action for given state.

        Args:
            state: State array (state_dim,)
            deterministic: If True, use mean action

        Returns:
            action: Position multiplier [0, 2]
            log_prob: Log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor, deterministic)

        return float(action.item()), float(log_prob.item())

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        epochs: int = 10
    ) -> dict:
        """
        Update policy using PPO.

        Args:
            states: State batch (batch_size, state_dim)
            actions: Action batch (batch_size, 1)
            old_log_probs: Old log probabilities (batch_size, 1)
            returns: Returns (batch_size, 1)
            advantages: Advantages (batch_size, 1)
            epochs: Number of optimization epochs

        Returns:
            Dictionary with loss statistics
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        updates = 0

        for epoch in range(epochs):
            # Evaluate actions
            log_probs, values, entropy = self.policy.evaluate_actions(states_t, actions_t)

            # Policy loss (PPO clip objective)
            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns_t.squeeze())

            # Entropy bonus (encourage exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            updates += 1

        return {
            'policy_loss': total_policy_loss / updates,
            'value_loss': total_value_loss / updates,
            'entropy': total_entropy / updates,
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"Model loaded from {filepath}")

    def eval_mode(self):
        """Set to evaluation mode."""
        self.policy.eval()
        self.training = False

    def train_mode(self):
        """Set to training mode."""
        self.policy.train()
        self.training = True


def test_ppo_agent():
    """Test PPO agent."""
    print("=" * 80)
    print("HIMARI RL - PPO Agent Test")
    print("=" * 80)
    print()

    # Create agent
    config = PPOConfig(state_dim=16, hidden_dim=128)
    agent = PPOAgent(config, device='cpu')

    print(f"Agent created:")
    print(f"  State dim:   {config.state_dim}")
    print(f"  Hidden dim:  {config.hidden_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    # Test forward pass
    print("Test 1: Forward pass")
    print("-" * 60)
    test_state = np.random.randn(16).astype(np.float32)
    action, log_prob = agent.get_action(test_state, deterministic=False)
    print(f"  State shape:  {test_state.shape}")
    print(f"  Action:       {action:.3f} (should be in [0, 2])")
    print(f"  Log prob:     {log_prob:.3f}")
    print()

    # Test deterministic action
    print("Test 2: Deterministic action")
    print("-" * 60)
    det_action, _ = agent.get_action(test_state, deterministic=True)
    print(f"  Deterministic action: {det_action:.3f}")
    print()

    # Test update
    print("Test 3: Policy update")
    print("-" * 60)
    batch_size = 32
    states = np.random.randn(batch_size, 16).astype(np.float32)
    actions = np.random.rand(batch_size, 1).astype(np.float32) * 2
    old_log_probs = np.random.randn(batch_size, 1).astype(np.float32)
    returns = np.random.randn(batch_size, 1).astype(np.float32)
    advantages = np.random.randn(batch_size, 1).astype(np.float32)

    losses = agent.update(states, actions, old_log_probs, returns, advantages, epochs=5)

    print(f"  Batch size:    {batch_size}")
    print(f"  Policy loss:   {losses['policy_loss']:.4f}")
    print(f"  Value loss:    {losses['value_loss']:.4f}")
    print(f"  Entropy:       {losses['entropy']:.4f}")
    print()

    # Test save/load
    print("Test 4: Save and load model")
    print("-" * 60)
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.pt')
        agent.save(model_path)
        print(f"  Model saved to: {model_path}")

        # Create new agent and load
        agent2 = PPOAgent(config, device='cpu')
        agent2.load(model_path)
        print(f"  Model loaded successfully")

        # Test they produce same action
        action1, _ = agent.get_action(test_state, deterministic=True)
        action2, _ = agent2.get_action(test_state, deterministic=True)
        print(f"  Action match: {np.isclose(action1, action2)}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_ppo_agent()
