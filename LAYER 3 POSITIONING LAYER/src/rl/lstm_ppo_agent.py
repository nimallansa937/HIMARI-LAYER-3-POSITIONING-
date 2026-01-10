"""
LSTM-based PPO Agent for HIMARI Layer 3
Adds temporal memory to capture market dynamics over time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class LSTMPPOConfig:
    """Configuration for LSTM-PPO agent."""
    state_dim: int = 16
    action_dim: int = 1
    hidden_dim: int = 128
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class LSTMPolicyNetwork(nn.Module):
    """LSTM-based policy network with temporal memory."""

    def __init__(self, config: LSTMPPOConfig):
        super().__init__()
        self.config = config

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=config.state_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=0.1 if config.lstm_num_layers > 1 else 0
        )

        # Policy head (actor)
        self.policy_fc1 = nn.Linear(config.lstm_hidden_dim, config.hidden_dim)
        self.policy_fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.policy_mean = nn.Linear(config.hidden_dim // 2, config.action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(config.action_dim))

        # Value head (critic)
        self.value_fc1 = nn.Linear(config.lstm_hidden_dim, config.hidden_dim)
        self.value_fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.value_head = nn.Linear(config.hidden_dim // 2, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM-PPO network.

        Args:
            state: State tensor [batch, seq_len, state_dim] or [batch, state_dim]
            hidden: LSTM hidden state (h, c)

        Returns:
            action_mean, action_log_std, value, new_hidden
        """
        # Add sequence dimension if needed
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # LSTM forward
        lstm_out, new_hidden = self.lstm(state, hidden)

        # Use last timestep output
        lstm_features = lstm_out[:, -1, :]  # [batch, lstm_hidden_dim]

        # Policy head
        policy = F.relu(self.policy_fc1(lstm_features))
        policy = F.relu(self.policy_fc2(policy))
        action_mean = torch.tanh(self.policy_mean(policy)) * 2.0  # Scale to [0, 2]
        action_log_std = self.policy_log_std.expand_as(action_mean)

        # Value head
        value = F.relu(self.value_fc1(lstm_features))
        value = F.relu(self.value_fc2(value))
        value = self.value_head(value)

        return action_mean, action_log_std, value, new_hidden

    def get_action(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from policy."""
        action_mean, action_log_std, value, new_hidden = self.forward(state, hidden)

        if deterministic:
            action = action_mean
        else:
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()

        # Clip action to [0, 2]
        action = torch.clamp(action, 0.0, 2.0)

        # Compute log probability
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value, new_hidden

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        action_mean, action_log_std, value, _ = self.forward(state, hidden)

        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, value


class LSTMPPOAgent:
    """LSTM-based PPO agent with temporal memory."""

    def __init__(self, config: LSTMPPOConfig, device: str = 'cpu'):
        self.config = config
        self.device = device

        self.policy = LSTMPolicyNetwork(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

        # Hidden state for online inference
        self.hidden = None

    def reset_hidden(self, batch_size: int = 1):
        """Reset LSTM hidden state."""
        self.hidden = (
            torch.zeros(self.config.lstm_num_layers, batch_size, self.config.lstm_hidden_dim).to(self.device),
            torch.zeros(self.config.lstm_num_layers, batch_size, self.config.lstm_hidden_dim).to(self.device)
        )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, dict]:
        """Select action given current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.hidden is None:
            self.reset_hidden(batch_size=1)

        with torch.no_grad():
            action, log_prob, value, self.hidden = self.policy.get_action(
                state_tensor,
                self.hidden,
                deterministic
            )

        diagnostics = {
            'action': action.item(),
            'log_prob': log_prob.item(),
            'value': value.item(),
        }

        return action.item(), diagnostics

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> dict:
        """PPO update step."""
        # Evaluate actions
        log_probs, entropy, values = self.policy.evaluate_actions(states, actions)

        # Compute ratio and clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss +
            self.config.value_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': loss.item(),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
