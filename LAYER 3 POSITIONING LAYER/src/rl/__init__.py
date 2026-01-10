"""
HIMARI Layer 3 - Reinforcement Learning Module
================================================

RL-based position sizing optimization with best practices.

Components:
- PPOAgent: Proximal Policy Optimization agent
- StateEncoder: Feature encoding for RL state
- SyntheticDataGenerator: MJD + GARCH stress scenario generation
- WFOTrainer: Walk-Forward Optimization with transfer learning
"""

from .state_encoder import StateEncoder, TradingState
from .ppo_agent import PPOAgent, PPOConfig

# NEW: Best practices training infrastructure
from .synthetic_data import SyntheticDataGenerator, MJDConfig, GARCHConfig
# from .wfo_trainer import WFOTrainer, WFOConfig, RiskAwareReward

__all__ = [
    # Core RL
    'StateEncoder',
    'TradingState',
    'PPOAgent',
    'PPOConfig',
    
    # Synthetic Data
    'SyntheticDataGenerator',
    'MJDConfig',
    'GARCHConfig',
    
    # WFO Training
    # 'WFOTrainer',
    # 'WFOConfig',
    # 'RiskAwareReward',
]
