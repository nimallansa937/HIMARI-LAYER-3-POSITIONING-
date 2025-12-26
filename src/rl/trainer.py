"""
HIMARI Layer 3 - RL Trainer
=============================

Trains PPO agent in trading environment with live price data.

Features:
- Generalized Advantage Estimation (GAE)
- Episodic training
- Checkpointing
- Performance tracking
- TensorBoard logging (optional)

Version: 1.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import List, Dict, Optional
import logging
import time
from dataclasses import dataclass, asdict
import json

from rl.ppo_agent import PPOAgent, PPOConfig
from rl.trading_env import TradingEnvironment, EnvConfig
from rl.state_encoder import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 64
    ppo_epochs: int = 10
    save_interval: int = 50
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    use_live_prices: bool = True


class RolloutBuffer:
    """
    Buffer for storing rollout data.

    Stores (state, action, reward, next_state, done, log_prob, value)
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append([action])
        self.rewards.append([reward])
        self.next_states.append(next_state)
        self.dones.append([1.0 if done else 0.0])
        self.log_probs.append([log_prob])
        self.values.append([value])

    def get(self) -> Dict[str, np.ndarray]:
        """Get all data as numpy arrays."""
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'next_states': np.array(self.next_states, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
        }

    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards array (T, 1)
        values: Value estimates (T, 1)
        dones: Done flags (T, 1)
        gamma: Discount factor
        lambda_: GAE parameter

    Returns:
        advantages: GAE advantages (T, 1)
        returns: Discounted returns (T, 1)
    """
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0

    # Compute GAE backwards
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns


class RLTrainer:
    """
    PPO trainer for trading environment.

    Manages training loop, checkpointing, and logging.
    """

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        env_config: Optional[EnvConfig] = None,
        ppo_config: Optional[PPOConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            training_config: Training configuration
            env_config: Environment configuration
            ppo_config: PPO agent configuration
            device: 'cpu' or 'cuda'
        """
        self.training_config = training_config or TrainingConfig()
        self.env_config = env_config or EnvConfig()
        self.ppo_config = ppo_config or PPOConfig()
        self.device = device

        # Create environment
        self.env = TradingEnvironment(
            config=self.env_config,
            use_live_prices=self.training_config.use_live_prices
        )

        # Create agent
        self.agent = PPOAgent(config=self.ppo_config, device=device)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_sharpes = []
        self.episode_pnls = []

        # Create checkpoint directory
        os.makedirs(self.training_config.checkpoint_dir, exist_ok=True)

        logger.info("RLTrainer initialized")

    def train(self) -> Dict[str, List[float]]:
        """
        Train agent for specified number of episodes.

        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {self.training_config.num_episodes} episodes")

        for episode in range(self.training_config.num_episodes):
            episode_start = time.time()

            # Run episode
            episode_reward, episode_length, episode_stats = self._run_episode()

            # Track statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_sharpes.append(episode_stats.get('sharpe', 0.0))
            self.episode_pnls.append(episode_stats.get('total_pnl_pct', 0.0))

            # Update agent if buffer is large enough
            if len(self.buffer) >= self.training_config.batch_size:
                loss_stats = self._update_agent()
            else:
                loss_stats = {}

            # Log progress
            if (episode + 1) % self.training_config.log_interval == 0:
                self._log_progress(episode + 1, episode_reward, episode_stats, loss_stats, episode_start)

            # Save checkpoint
            if (episode + 1) % self.training_config.save_interval == 0:
                self._save_checkpoint(episode + 1)

        logger.info("Training complete")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_sharpes': self.episode_sharpes,
            'episode_pnls': self.episode_pnls,
        }

    def _run_episode(self) -> tuple:
        """
        Run one episode.

        Returns:
            episode_reward: Total episode reward
            episode_length: Number of steps
            stats: Environment statistics
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(self.training_config.max_steps_per_episode):
            # Get action from agent
            action, log_prob = self.agent.get_action(state, deterministic=False)

            # Get value estimate
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                _, _, value = self.agent.policy.forward(state_tensor)
                value = value.item()

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store in buffer
            self.buffer.add(state, action, reward, next_state, done, log_prob, value)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        stats = self.env.get_statistics()
        return episode_reward, episode_length, stats

    def _update_agent(self) -> Dict[str, float]:
        """
        Update agent using collected rollouts.

        Returns:
            Loss statistics
        """
        # Get rollout data
        rollout_data = self.buffer.get()

        # Compute GAE
        advantages, returns = compute_gae(
            rewards=rollout_data['rewards'],
            values=rollout_data['values'],
            dones=rollout_data['dones'],
            gamma=self.ppo_config.gamma,
            lambda_=self.ppo_config.lambda_gae
        )

        # Update agent
        loss_stats = self.agent.update(
            states=rollout_data['states'],
            actions=rollout_data['actions'],
            old_log_probs=rollout_data['log_probs'],
            returns=returns,
            advantages=advantages,
            epochs=self.training_config.ppo_epochs
        )

        # Clear buffer
        self.buffer.clear()

        return loss_stats

    def _log_progress(
        self,
        episode: int,
        episode_reward: float,
        episode_stats: Dict,
        loss_stats: Dict,
        episode_start: float
    ):
        """Log training progress."""
        elapsed = time.time() - episode_start

        # Recent statistics
        recent_rewards = self.episode_rewards[-10:]
        recent_sharpes = self.episode_sharpes[-10:]
        recent_pnls = self.episode_pnls[-10:]

        avg_reward = np.mean(recent_rewards)
        avg_sharpe = np.mean(recent_sharpes)
        avg_pnl = np.mean(recent_pnls)

        logger.info(
            f"Episode {episode}/{self.training_config.num_episodes} | "
            f"Reward: {episode_reward:.3f} | "
            f"Sharpe: {episode_stats.get('sharpe', 0):.3f} | "
            f"P&L: {episode_stats.get('total_pnl_pct', 0):.2%} | "
            f"Trades: {episode_stats.get('total_trades', 0)} | "
            f"Win Rate: {episode_stats.get('win_rate', 0):.1%} | "
            f"Time: {elapsed:.1f}s"
        )

        if loss_stats:
            logger.info(
                f"  Losses -> Policy: {loss_stats.get('policy_loss', 0):.4f}, "
                f"Value: {loss_stats.get('value_loss', 0):.4f}, "
                f"Entropy: {loss_stats.get('entropy', 0):.4f}"
            )

        logger.info(
            f"  Avg (last 10) -> Reward: {avg_reward:.3f}, "
            f"Sharpe: {avg_sharpe:.3f}, "
            f"P&L: {avg_pnl:.2%}"
        )

    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir,
            f"ppo_episode_{episode}.pt"
        )

        self.agent.save(checkpoint_path)

        # Save training stats
        stats_path = os.path.join(
            self.training_config.checkpoint_dir,
            f"stats_episode_{episode}.json"
        )

        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_sharpes': self.episode_sharpes,
            'episode_pnls': self.episode_pnls,
            'training_config': asdict(self.training_config),
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained agent.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")

        self.agent.eval_mode()

        eval_rewards = []
        eval_sharpes = []
        eval_pnls = []
        eval_win_rates = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.training_config.max_steps_per_episode):
                # Use deterministic action
                action, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

                if done:
                    break

            stats = self.env.get_statistics()
            eval_rewards.append(episode_reward)
            eval_sharpes.append(stats.get('sharpe', 0.0))
            eval_pnls.append(stats.get('total_pnl_pct', 0.0))
            eval_win_rates.append(stats.get('win_rate', 0.0))

        self.agent.train_mode()

        results = {
            'avg_reward': np.mean(eval_rewards),
            'avg_sharpe': np.mean(eval_sharpes),
            'avg_pnl': np.mean(eval_pnls),
            'avg_win_rate': np.mean(eval_win_rates),
            'std_reward': np.std(eval_rewards),
            'std_sharpe': np.std(eval_sharpes),
        }

        logger.info(f"Evaluation results: {results}")
        return results


def test_trainer():
    """Test RL trainer."""
    print("=" * 80)
    print("HIMARI RL - Trainer Test")
    print("=" * 80)
    print()

    # Create configs
    training_config = TrainingConfig(
        num_episodes=5,
        max_steps_per_episode=50,
        batch_size=32,
        log_interval=1,
        save_interval=5,
        use_live_prices=True
    )

    env_config = EnvConfig(
        initial_capital=100000,
        max_steps=50,
        symbol="BTC-USD"
    )

    ppo_config = PPOConfig(
        state_dim=16,
        hidden_dim=64,
        learning_rate=3e-4
    )

    # Create trainer
    trainer = RLTrainer(
        training_config=training_config,
        env_config=env_config,
        ppo_config=ppo_config,
        device='cpu'
    )

    print("Trainer created")
    print(f"  Episodes:     {training_config.num_episodes}")
    print(f"  Max steps:    {training_config.max_steps_per_episode}")
    print(f"  Batch size:   {training_config.batch_size}")
    print(f"  Live prices:  {training_config.use_live_prices}")
    print()

    # Run training
    print("Running training...")
    print("-" * 60)
    stats = trainer.train()
    print()

    # Show results
    print("Training Results:")
    print("-" * 60)
    print(f"  Avg reward (last 3): {np.mean(stats['episode_rewards'][-3:]):.3f}")
    print(f"  Avg Sharpe (last 3): {np.mean(stats['episode_sharpes'][-3:]):.3f}")
    print(f"  Avg P&L (last 3):    {np.mean(stats['episode_pnls'][-3:]):.2%}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_trainer()
