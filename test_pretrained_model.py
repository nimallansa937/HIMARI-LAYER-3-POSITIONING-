"""
HIMARI Layer 3 - Pretrained Model Testing Script
=================================================

Tests the pretrained LSTM-PPO model on synthetic scenarios to evaluate:
1. Model loads correctly
2. Inference works (action generation)
3. Performance metrics on bull/bear/ranging scenarios
4. Sharpe ratio comparison across regimes

Usage:
    python test_pretrained_model.py --model pretrained_final.pt
"""

import sys
import os

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "LAYER 3 POSITIONING LAYER", "src")
sys.path.insert(0, src_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE (must match training - from lstm_ppo_agent.py)
# ============================================================================

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

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through LSTM-PPO network."""
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

        return action, value, new_hidden


# ============================================================================
# SYNTHETIC TEST DATA GENERATORS
# ============================================================================

def generate_bull_scenario(length: int = 500) -> Dict:
    """Generate a Bull Market scenario."""
    returns = np.random.normal(0.001, 0.015, length)  # Positive drift
    # Add some momentum
    for i in range(1, length):
        returns[i] += 0.1 * returns[i-1]
    return {"type": "bull", "returns": returns}


def generate_bear_scenario(length: int = 500) -> Dict:
    """Generate a Bear Market scenario."""
    returns = np.random.normal(-0.002, 0.025, length)  # Negative drift, higher vol
    # Add volatility clustering
    for i in range(1, length):
        if returns[i-1] < -0.02:
            returns[i] *= 1.5
    return {"type": "bear", "returns": returns}


def generate_ranging_scenario(length: int = 500) -> Dict:
    """Generate a Ranging scenario."""
    returns = np.random.normal(0, 0.012, length)  # No drift
    # Mean revert
    cumret = np.cumsum(returns)
    for i in range(50, length):
        returns[i] -= 0.1 * cumret[i-1] / i
    return {"type": "ranging", "returns": returns}


# ============================================================================
# MODEL TESTER
# ============================================================================

class ModelTester:
    """Tests a pretrained LSTM-PPO model."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint (weights_only=False for compatibility with custom classes)
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config if present, otherwise use default
        if "config" in self.checkpoint:
            self.config = self.checkpoint["config"]
            logger.info("✓ Config loaded from checkpoint")
            logger.info(f"  - State dim: {self.config.state_dim}")
            logger.info(f"  - Hidden dim: {self.config.hidden_dim}")
            logger.info(f"  - LSTM hidden dim: {self.config.lstm_hidden_dim}")
            logger.info(f"  - LSTM layers: {self.config.lstm_num_layers}")
        else:
            self.config = LSTMPPOConfig()
            logger.info("⚠ Using default config")
        
        # Initialize network
        self.network = LSTMPolicyNetwork(self.config).to(self.device)
        
        # Load weights - handle different checkpoint formats
        if "policy_state_dict" in self.checkpoint:
            self.network.load_state_dict(self.checkpoint["policy_state_dict"])
            logger.info("✓ Model weights loaded from policy_state_dict")
        elif "network_state_dict" in self.checkpoint:
            self.network.load_state_dict(self.checkpoint["network_state_dict"])
            logger.info("✓ Model weights loaded from network_state_dict")
        elif "state_dict" in self.checkpoint:
            self.network.load_state_dict(self.checkpoint["state_dict"])
            logger.info("✓ Model weights loaded from state_dict")
        else:
            # Try loading directly (the checkpoint might be just the state dict)
            try:
                self.network.load_state_dict(self.checkpoint)
                logger.info("✓ Model weights loaded directly")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
                logger.info("Available keys in checkpoint:")
                for key in self.checkpoint.keys():
                    logger.info(f"  - {key}")
                raise
        
        self.network.eval()
        logger.info("✓ Model set to eval mode")
        
        # Show model info
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info(f"✓ Total params: {total_params:,}")
        logger.info(f"✓ Trainable params: {trainable_params:,}")
    
    def run_scenario(self, scenario: Dict, max_position_pct: float = 0.5) -> Dict:
        """Run model on a single scenario."""
        returns = scenario["returns"]
        regime = scenario["type"]
        
        hidden = None
        actions = []
        position_returns = []
        equity_curve = [1.0]
        
        state_dim = self.config.state_dim
        
        for i in range(state_dim, len(returns) - 1):
            # Build state from recent returns
            state = returns[i-state_dim:i]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action
            with torch.no_grad():
                action, value, hidden = self.network.get_action(
                    state_tensor, hidden, deterministic=True
                )
            
            action_val = action.cpu().numpy()[0, 0]
            actions.append(action_val)
            
            # Calculate position return
            # Action is in [0, 2], where 1.0 is neutral (base position)
            # Convert to position percentage: (action - 1) maps to [-1, 1] * max_position
            position_pct = np.clip((action_val - 1.0) * max_position_pct,
                                   -max_position_pct,
                                   max_position_pct)
            market_return = returns[i]
            pos_return = position_pct * market_return
            position_returns.append(pos_return)
            
            # Update equity
            equity_curve.append(equity_curve[-1] * (1 + pos_return))
        
        # Calculate metrics
        position_returns = np.array(position_returns)
        equity_curve = np.array(equity_curve)
        
        # Sharpe ratio (daily -> annualized)
        sharpe = np.mean(position_returns) / (np.std(position_returns) + 1e-8) * np.sqrt(252)
        
        # Sortino ratio
        downside = position_returns[position_returns < 0]
        sortino = np.mean(position_returns) / (np.std(downside) + 1e-8) * np.sqrt(252) if len(downside) > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100
        
        # Final return
        total_return = (equity_curve[-1] - 1) * 100
        
        # Action statistics
        avg_action = np.mean(actions)
        action_std = np.std(actions)
        
        return {
            "regime": regime,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "total_return": total_return,
            "avg_action": avg_action,
            "action_std": action_std,
            "n_steps": len(position_returns)
        }
    
    def run_all_tests(self, n_scenarios: int = 20) -> Dict:
        """Run tests on all scenario types."""
        logger.info("=" * 60)
        logger.info("RUNNING MODEL TESTS")
        logger.info("=" * 60)
        
        results = defaultdict(list)
        
        # Generate and test scenarios for each type
        for scenario_type, generator in [
            ("bull", generate_bull_scenario),
            ("bear", generate_bear_scenario),
            ("ranging", generate_ranging_scenario)
        ]:
            logger.info(f"\n📊 Testing {scenario_type.upper()} scenarios...")
            
            for i in range(n_scenarios):
                scenario = generator(length=500)
                result = self.run_scenario(scenario)
                results[scenario_type].append(result)
            
            # Aggregate results
            sharpes = [r["sharpe"] for r in results[scenario_type]]
            returns = [r["total_return"] for r in results[scenario_type]]
            drawdowns = [r["max_drawdown"] for r in results[scenario_type]]
            actions = [r["avg_action"] for r in results[scenario_type]]
            
            logger.info(f"  Sharpe Ratio:   {np.mean(sharpes):+.4f} ± {np.std(sharpes):.4f}")
            logger.info(f"  Total Return:   {np.mean(returns):+.2f}% ± {np.std(returns):.2f}%")
            logger.info(f"  Max Drawdown:   {np.mean(drawdowns):.2f}% ± {np.std(drawdowns):.2f}%")
            logger.info(f"  Avg Action:     {np.mean(actions):.4f} ± {np.std(actions):.4f} (1.0 = neutral)")
        
        # Overall summary
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 60)
        
        all_sharpes = []
        for regime in ["bull", "bear", "ranging"]:
            sharpes = [r["sharpe"] for r in results[regime]]
            all_sharpes.extend(sharpes)
            
        logger.info(f"Overall Avg Sharpe: {np.mean(all_sharpes):+.4f}")
        
        # Check model behavior
        bull_actions = np.mean([r["avg_action"] for r in results["bull"]])
        bear_actions = np.mean([r["avg_action"] for r in results["bear"]])
        
        logger.info("\n📈 Behavior Analysis:")
        logger.info(f"  Bull market avg action: {bull_actions:.4f}")
        logger.info(f"  Bear market avg action: {bear_actions:.4f}")
        
        if bull_actions > bear_actions:
            logger.info("  ✓ Model is MORE AGGRESSIVE in bull markets (action > 1.0)")
        elif bull_actions < bear_actions:
            logger.info("  ⚠ Model is less aggressive in bull markets")
        else:
            logger.info("  ○ Model actions similar across regimes")
        
        if bull_actions > 1.0:
            logger.info("  ✓ Model goes LONG in bull markets (action > 1.0)")
        
        if bear_actions < 1.0:
            logger.info("  ✓ Model goes SHORT/reduced in bear markets (action < 1.0)")
        
        bear_dd = np.mean([r["max_drawdown"] for r in results["bear"]])
        if bear_dd < 30:
            logger.info(f"  ✓ Max drawdown controlled in bear markets ({bear_dd:.1f}%)")
        else:
            logger.info(f"  ⚠ High drawdowns in bear market scenarios ({bear_dd:.1f}%)")
        
        return dict(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test pretrained LSTM-PPO model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of scenarios per type")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("HIMARI Layer 3 - Pretrained Model Testing")
    print("=" * 60 + "\n")
    
    try:
        tester = ModelTester(args.model, device=args.device)
        results = tester.run_all_tests(n_scenarios=args.scenarios)
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
