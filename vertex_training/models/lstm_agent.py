"""
HIMARI Layer 3 - LSTM Policy Network
Sequence-based RL for position sizing with temporal dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class LSTMConfig:
    """LSTM Policy hyperparameters."""
    state_dim: int = 16
    action_dim: int = 1
    hidden_dim: int = 128
    lstm_hidden_dim: int = 64
    num_lstm_layers: int = 2
    sequence_length: int = 20  # Look back 20 steps
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    dropout: float = 0.1


class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic for sequential decision making.
    Uses LSTM to capture temporal patterns in market data.
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.num_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0
        )
        
        # Post-LSTM processing
        self.post_lstm = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(config.action_dim))
        
        # Critic head
        self.critic = nn.Linear(config.hidden_dim, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        state_sequence: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass with sequence of states.
        
        Args:
            state_sequence: (batch, seq_len, state_dim)
            hidden: Optional LSTM hidden state
            
        Returns:
            action_mean, action_std, value, new_hidden
        """
        batch_size, seq_len, _ = state_sequence.shape
        
        # Encode features
        features = self.feature_encoder(state_sequence)  # (batch, seq_len, hidden_dim)
        
        # LSTM
        if hidden is None:
            lstm_out, new_hidden = self.lstm(features)
        else:
            lstm_out, new_hidden = self.lstm(features, hidden)
        
        # Use last output for decision
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)
        
        # Post-LSTM
        processed = self.post_lstm(last_out)
        
        # Actor
        action_mean = self.actor_mean(processed)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        # Critic
        value = self.critic(processed)
        
        return action_mean, action_std, value, new_hidden
    
    def get_action(
        self,
        state_sequence: torch.Tensor,
        hidden: Optional[Tuple] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Sample action from policy."""
        action_mean, action_std, _, new_hidden = self.forward(state_sequence, hidden)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clamp to valid range
        action = torch.clamp(action, 0.0, 2.0)
        
        return action, log_prob, new_hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.config.num_lstm_layers, batch_size, self.config.lstm_hidden_dim).to(device)
        c = torch.zeros(self.config.num_lstm_layers, batch_size, self.config.lstm_hidden_dim).to(device)
        return (h, c)


class SequenceBuffer:
    """Buffer for storing sequences for LSTM training."""
    
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.sequences = []  # List of (state_seq, action, reward, done)
        self.current_episode_states = []
    
    def add_state(self, state: np.ndarray):
        """Add state to current episode."""
        self.current_episode_states.append(state)
    
    def add_transition(self, action: float, reward: float, done: bool):
        """Add transition with current sequence."""
        if len(self.current_episode_states) >= 1:
            # Pad sequence if needed
            seq = self.current_episode_states[-self.sequence_length:]
            if len(seq) < self.sequence_length:
                padding = [np.zeros_like(seq[0])] * (self.sequence_length - len(seq))
                seq = padding + seq
            
            self.sequences.append({
                'state_seq': np.array(seq),
                'action': action,
                'reward': reward,
                'done': done
            })
        
        if done:
            self.current_episode_states = []
    
    def clear(self):
        self.sequences = []
        self.current_episode_states = []
    
    def get_batch(self, batch_size: int, device: str = 'cpu'):
        """Get random batch of sequences."""
        indices = np.random.choice(len(self.sequences), min(batch_size, len(self.sequences)), replace=False)
        
        state_seqs = torch.FloatTensor(np.array([self.sequences[i]['state_seq'] for i in indices])).to(device)
        actions = torch.FloatTensor(np.array([self.sequences[i]['action'] for i in indices])).to(device)
        rewards = torch.FloatTensor(np.array([self.sequences[i]['reward'] for i in indices])).to(device)
        
        return state_seqs, actions, rewards


class LSTMAgent:
    """LSTM-based RL Agent for position sizing."""
    
    def __init__(self, config: LSTMConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        self.actor_critic = LSTMActorCritic(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate
        )
        
        self.buffer = SequenceBuffer(config.sequence_length)
        self.hidden = None
        self.state_history = []
    
    def reset_hidden(self):
        """Reset LSTM hidden state for new episode."""
        self.hidden = self.actor_critic.init_hidden(1, self.device)
        self.state_history = []
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """Get action maintaining temporal context."""
        self.state_history.append(state)
        
        # Build sequence
        seq = self.state_history[-self.config.sequence_length:]
        if len(seq) < self.config.sequence_length:
            padding = [np.zeros_like(state)] * (self.config.sequence_length - len(seq))
            seq = padding + seq
        
        state_seq = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, self.hidden = self.actor_critic.get_action(
                state_seq, self.hidden, deterministic
            )
        
        return action.squeeze().cpu().item(), log_prob.squeeze().cpu().item()
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def eval_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()
