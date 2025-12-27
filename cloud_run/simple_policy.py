"""
Simple Policy Network for Cloud Run Inference
Compatible with trained model weights from vertex_ai training.
"""

import torch
import torch.nn as nn


class SimplePolicyNetwork(nn.Module):
    """
    Simple policy network for position sizing.
    Maps 16-dim state to 1-dim action (position multiplier).
    """
    
    def __init__(self, state_dim: int = 16, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output [0, 1], scale to [0, 2] in forward
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Tensor of shape (batch, state_dim)
        
        Returns:
            Position multiplier in range [0, 2]
        """
        # Get raw output [0, 1]
        raw = self.net(state)
        # Scale to [0, 2]
        return raw * 2.0
    
    def get_action(self, state: torch.Tensor, deterministic: bool = True):
        """
        Get action for given state.
        
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
            deterministic: Whether to use deterministic policy
        
        Returns:
            Tuple of (action, log_prob)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(state)
        
        return action.squeeze(), torch.tensor(0.0)
