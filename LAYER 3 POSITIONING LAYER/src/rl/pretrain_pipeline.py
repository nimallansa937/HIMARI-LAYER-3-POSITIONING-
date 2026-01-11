"""
HIMARI Layer 3 - Pre-Training Pipeline
=======================================

Pre-trains RL agents on synthetic stress scenarios before real data.

Pipeline:
1. Load 500 synthetic stress scenarios
2. Train agent for 500K steps
3. Save pre-trained weights as initialization for WFO

Purpose: Expose agent to black swans before production training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
import numpy as np
import pickle
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

from rl.lstm_ppo_agent import LSTMPPOAgent, LSTMPPOConfig
from rl.trading_env import TradingEnvironment, EnvConfig
from rl.state_encoder import StateEncoder, TradingState

logger = logging.getLogger(__name__)


class SyntheticTradingEnv:
    """
    Trading environment using synthetic data instead of live prices.
    """

    def __init__(self, scenario: Dict, env_config: EnvConfig):
        self.scenario = scenario
        self.config = env_config
        self.state_encoder = StateEncoder()

        # Handle multiple data formats
        if 'prices' in scenario:
            # Original format from synthetic_data_generator.py
            self.prices = scenario['prices']
            self.returns = scenario['returns']
            self.regimes = scenario.get('regimes', ['normal'] * len(self.prices))
        elif 'returns' in scenario:
            # Balanced format from balanced_data_generator.py
            self.returns = np.array(scenario['returns'])
            # Compute prices from returns (starting at 50000)
            self.prices = 50000.0 * np.cumprod(1 + self.returns)
            self.prices = np.insert(self.prices, 0, 50000.0)  # Add initial price
            # Map label to regime
            label = scenario.get('label', scenario.get('type', 'normal'))
            self.regimes = [label] * len(self.prices)
        else:
            raise ValueError(f"Unknown scenario format. Keys: {scenario.keys()}")

        self.step_idx = 0
        self.max_steps = min(len(self.prices) - 1, env_config.max_steps)

        self.capital = env_config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl_history = []

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.step_idx = 0
        self.capital = self.config.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.pnl_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.step_idx >= len(self.prices) - 10:
            # Not enough history
            return np.zeros(16)

        # Get recent prices
        lookback = 20
        start_idx = max(0, self.step_idx - lookback)
        recent_prices = self.prices[start_idx:self.step_idx+1]

        if len(recent_prices) < 2:
            return np.zeros(16)

        # Calculate features
        returns = np.diff(np.log(recent_prices))
        current_price = recent_prices[-1]

        # Calculate momentum
        price_momentum_1h = 0.0
        price_momentum_4h = 0.0
        if len(recent_prices) >= 2:
            price_momentum_1h = ((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]) * 100
        if len(recent_prices) >= 5:
            price_momentum_4h = ((recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]) * 100

        # Calculate volatility
        volatility = float(np.std(returns[-20:]) if len(returns) >= 20 else 0.01)

        # Determine position side
        if self.position > 0:
            position_side = 1  # LONG
        elif self.position < 0:
            position_side = -1  # SHORT
        else:
            position_side = 0  # FLAT

        # Calculate unrealized PnL as percentage
        unrealized_pnl = self._calculate_unrealized_pnl()
        unrealized_pnl_pct = unrealized_pnl / self.capital if self.capital > 0 else 0.0

        # Calculate position size in USD
        position_size_usd = abs(self.position * current_price)

        # Calculate recent performance metrics
        recent_win_rate = 0.5  # Default
        recent_sharpe = 0.0
        total_pnl_pct = (self.capital - self.config.initial_capital) / self.config.initial_capital

        if len(self.pnl_history) >= 5:
            wins = sum(1 for p in self.pnl_history[-10:] if p > 0)
            recent_win_rate = wins / min(len(self.pnl_history), 10)

            recent_pnls = self.pnl_history[-10:]
            if np.std(recent_pnls) > 0:
                recent_sharpe = np.mean(recent_pnls) / np.std(recent_pnls)

        # Get regime from scenario (default to RANGING=2)
        # Map string regimes to integers matching StateEncoder.REGIME_MAP
        regime_map = {
            'bull': 0,        # TRENDING_UP
            'bear': 1,        # TRENDING_DOWN
            'ranging': 2,     # RANGING
            'volatile': 3,    # HIGH_VOLATILITY
            'crisis': 4,      # CRISIS
            'crash': 4,       # CRISIS (alias)
            'recovery': 0,    # TRENDING_UP (alias)
        }
        raw_regime = self.regimes[self.step_idx] if self.step_idx < len(self.regimes) else 'ranging'
        if isinstance(raw_regime, str):
            regime = regime_map.get(raw_regime.lower(), 2)  # Default to RANGING
        else:
            regime = int(raw_regime) if 0 <= raw_regime <= 4 else 2

        # Build state with correct TradingState fields
        trading_state = TradingState(
            signal_confidence=0.7,  # Synthetic default
            signal_action=1,  # HOLD as default
            signal_tier=2,  # T2 as default
            signal_regime=regime,
            position_size_usd=position_size_usd,
            position_side=position_side,
            unrealized_pnl_pct=unrealized_pnl_pct,
            price_momentum_1h=price_momentum_1h,
            price_momentum_4h=price_momentum_4h,
            volatility=volatility,
            recent_win_rate=recent_win_rate,
            recent_sharpe=recent_sharpe,
            total_pnl_pct=total_pnl_pct,
            cascade_risk=min(volatility * 10, 1.0),  # Derived from volatility
            current_drawdown=min(0, total_pnl_pct),  # Negative if in drawdown
            timestamp=float(self.step_idx)
        )

        return self.state_encoder.encode_state(trading_state)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL."""
        if self.position == 0 or self.entry_price == 0:
            return 0.0

        current_price = self.prices[self.step_idx]
        return self.position * (current_price - self.entry_price)

    def step(self, action: float) -> tuple:
        """
        Take action in environment.

        Args:
            action: Position size multiplier [-1, 1]
                    -1 = max short, 0 = flat, +1 = max long

        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.prices[self.step_idx]

        # Close existing position if any
        if self.position != 0:
            pnl = self._calculate_unrealized_pnl()
            self.capital += pnl
            self.pnl_history.append(pnl)
            self.position = 0.0

        # Open new position based on action
        max_position_value = self.capital * self.config.max_position_pct
        position_value = action * max_position_value
        self.position = position_value / current_price
        self.entry_price = current_price

        # Apply commission
        commission = abs(position_value) * self.config.commission_rate
        self.capital -= commission

        # Move to next step
        self.step_idx += 1

        # Check if done
        done = (
            self.step_idx >= self.max_steps or
            self.capital <= self.config.initial_capital * 0.5  # 50% drawdown
        )

        # Calculate reward (Sharpe-based)
        if len(self.pnl_history) > 10:
            recent_pnls = self.pnl_history[-10:]
            mean_pnl = np.mean(recent_pnls)
            std_pnl = np.std(recent_pnls) + 1e-6
            sharpe = mean_pnl / std_pnl
            reward = sharpe
        else:
            reward = 0.0

        # Get next state
        next_state = self._get_state()

        info = {
            'capital': self.capital,
            'position': self.position,
            'pnl': self.pnl_history[-1] if self.pnl_history else 0.0,
            'step': self.step_idx
        }

        return next_state, reward, done, info


class PreTrainer:
    """
    Pre-training pipeline for RL agents on synthetic data.
    """

    def __init__(
        self,
        model_type: str = 'lstm_ppo',
        device: str = 'cuda',
        synthetic_data_path: str = "/tmp/synthetic_data/stress_scenarios.pkl"
    ):
        self.model_type = model_type
        self.device = device
        self.synthetic_data_path = synthetic_data_path

        # Load synthetic scenarios
        logger.info(f"Loading synthetic data from {synthetic_data_path}")
        with open(synthetic_data_path, 'rb') as f:
            self.scenarios = pickle.load(f)
        logger.info(f"Loaded {len(self.scenarios)} synthetic scenarios")

        # Initialize agent
        self.agent_config = LSTMPPOConfig(
            state_dim=16,
            action_dim=1,
            hidden_dim=128,
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
        )

        self.agent = LSTMPPOAgent(self.agent_config, device=device)

        self.env_config = EnvConfig(
            initial_capital=100000.0,
            max_position_pct=0.5,
            commission_rate=0.001,
            slippage_bps=5,
            reward_window=10,
            max_steps=1000,
            symbol='BTC-USD'
        )

    def pretrain(
        self,
        target_steps: int = 500000,
        save_dir: str = "/tmp/models/pretrained",
        use_wandb: bool = True,
        wandb_project: str = "himari-layer3-pretraining"
    ):
        """
        Pre-train agent on synthetic scenarios.

        Args:
            target_steps: Total training steps
            save_dir: Where to save pre-trained weights
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
        """
        os.makedirs(save_dir, exist_ok=True)

        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=wandb_project,
                name=f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "target_steps": target_steps,
                    "scenarios": len(self.scenarios),
                    "device": self.device,
                    "model_type": self.model_type,
                    "learning_rate": self.agent_config.learning_rate,
                    "hidden_dim": self.agent_config.hidden_dim,
                }
            )
            logger.info("✅ Weights & Biases logging enabled")
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("⚠️  wandb requested but not available")

        logger.info("="*60)
        logger.info("Starting Pre-Training on Synthetic Data")
        logger.info("="*60)
        logger.info(f"Target steps: {target_steps:,}")
        logger.info(f"Scenarios: {len(self.scenarios)}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)

        total_steps = 0
        episode = 0
        episode_rewards = []
        episode_sharpes = []

        while total_steps < target_steps:
            # Sample random scenario (use random.choice for list of dicts)
            scenario = random.choice(self.scenarios)

            # Create environment for this scenario
            env = SyntheticTradingEnv(scenario, self.env_config)

            # Reset environment and agent
            state = env.reset()
            self.agent.reset_hidden(batch_size=1)

            episode_reward = 0
            episode_returns = []
            done = False

            # Collect episode data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done and total_steps < target_steps:
                # Get action from agent (select_action handles tensor conversion)
                action, diagnostics = self.agent.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(diagnostics['log_prob'])
                values.append(diagnostics['value'])

                episode_reward += reward
                episode_returns.append(info.get('pnl', 0.0))

                state = next_state
                total_steps += 1

                # Update agent every 128 steps
                if len(states) >= 128:
                    # Convert lists to tensors and compute GAE
                    states_t = torch.FloatTensor(np.array(states)).to(self.device)
                    actions_t = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(self.device)
                    old_log_probs_t = torch.FloatTensor(np.array(log_probs)).unsqueeze(-1).to(self.device)
                    values_arr = np.array(values)
                    rewards_arr = np.array(rewards)

                    # Compute returns and advantages using GAE
                    returns = []
                    advantages = []
                    gae = 0
                    next_value = values_arr[-1] if len(values_arr) > 0 else 0

                    for t in reversed(range(len(rewards_arr))):
                        if t == len(rewards_arr) - 1:
                            next_val = next_value
                        else:
                            next_val = values_arr[t + 1]

                        delta = rewards_arr[t] + self.agent.config.gamma * next_val - values_arr[t]
                        gae = delta + self.agent.config.gamma * self.agent.config.lambda_gae * gae
                        advantages.insert(0, gae)
                        returns.insert(0, gae + values_arr[t])

                    returns_t = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
                    advantages_t = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)

                    # Normalize advantages
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                    # Update agent with tensors
                    self.agent.update(states_t, actions_t, old_log_probs_t, returns_t, advantages_t)
                    states, actions, rewards, log_probs, values = [], [], [], [], []

            # Episode finished
            episode += 1
            episode_rewards.append(episode_reward)

            if len(episode_returns) > 0:
                sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-6)
                episode_sharpes.append(sharpe)
            else:
                episode_sharpes.append(0.0)

            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                avg_sharpe = np.mean(episode_sharpes[-100:]) if len(episode_sharpes) >= 100 else np.mean(episode_sharpes)

                progress_pct = 100 * total_steps / target_steps
                logger.info(
                    f"Episode {episode:4d} | "
                    f"Steps: {total_steps:7d}/{target_steps:7d} ({progress_pct:5.1f}%) | "
                    f"Reward: {avg_reward:6.3f} | "
                    f"Sharpe: {avg_sharpe:6.3f} | "
                    f"Scenario: {scenario['type']}"
                )

                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "episode": episode,
                        "total_steps": total_steps,
                        "progress_pct": progress_pct,
                        "avg_reward_100": avg_reward,
                        "avg_sharpe_100": avg_sharpe,
                        "episode_reward": episode_reward,
                        "episode_sharpe": episode_sharpes[-1],
                        "scenario_type": scenario['type']
                    })

            # Save checkpoint every 50K steps
            if total_steps % 50000 == 0 and total_steps > 0:
                checkpoint_path = os.path.join(save_dir, f"pretrain_checkpoint_{total_steps}.pt")
                self.agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save final pre-trained weights
        final_path = os.path.join(save_dir, "pretrained_final.pt")
        self.agent.save(final_path)

        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        final_avg_sharpe = np.mean(episode_sharpes[-100:]) if len(episode_sharpes) >= 100 else np.mean(episode_sharpes)

        logger.info("="*60)
        logger.info("Pre-Training Complete!")
        logger.info("="*60)
        logger.info(f"Total episodes: {episode}")
        logger.info(f"Total steps: {total_steps:,}")
        logger.info(f"Final avg reward: {final_avg_reward:.3f}")
        logger.info(f"Final avg Sharpe: {final_avg_sharpe:.3f}")
        logger.info(f"Pre-trained weights: {final_path}")
        logger.info("="*60)

        # Final wandb summary
        if use_wandb and WANDB_AVAILABLE:
            wandb.summary["total_episodes"] = episode
            wandb.summary["total_steps"] = total_steps
            wandb.summary["final_avg_reward"] = final_avg_reward
            wandb.summary["final_avg_sharpe"] = final_avg_sharpe
            wandb.finish()
            logger.info("✅ W&B run finished")

        return final_path


def main():
    """Run pre-training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Pre-train RL agent on synthetic data')
    parser.add_argument('--steps', type=int, default=500000, help='Target training steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--synthetic-data', type=str, default='/tmp/synthetic_data/stress_scenarios.pkl')
    parser.add_argument('--output-dir', type=str, default='/tmp/models/pretrained')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Run pre-training
    trainer = PreTrainer(
        model_type='lstm_ppo',
        device=args.device,
        synthetic_data_path=args.synthetic_data
    )

    pretrained_path = trainer.pretrain(
        target_steps=args.steps,
        save_dir=args.output_dir
    )

    print(f"\n✅ Pre-training complete: {pretrained_path}")


if __name__ == "__main__":
    main()
