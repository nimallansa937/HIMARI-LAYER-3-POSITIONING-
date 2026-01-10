#!/usr/bin/env python3
"""
HIMARI Layer 3 - Lambda Labs LSTM Local Training Script
Trains LSTM-PPO agent locally without GCS authentication
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rl.lstm_ppo_agent import LSTMPPOAgent, LSTMPPOConfig
from rl.trading_env import TradingEnvironment, EnvConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMLocalTrainer:
    """Local LSTM training wrapper for Lambda Labs"""

    def __init__(
        self,
        model_dir: str = "/tmp/models",
        num_episodes: int = 1000,
        save_interval: int = 100,
        device: str = 'cuda'
    ):
        self.model_dir = Path(model_dir)
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.device = device

        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized LSTM LocalTrainer")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Episodes: {num_episodes}")
        logger.info(f"  Device: {device}")

    def train(self):
        """Main training loop"""
        logger.info("Starting LSTM-PPO training...")

        # Initialize environment and agent
        env_config = EnvConfig(
            initial_capital=100000.0,
            max_position_pct=0.5,
            commission_rate=0.001,
            slippage_bps=5,
            reward_window=10,
            max_steps=500,
            symbol='BTC-USD'
        )

        lstm_ppo_config = LSTMPPOConfig(
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

        env = TradingEnvironment(env_config)
        agent = LSTMPPOAgent(lstm_ppo_config, device=self.device)

        logger.info("Environment and LSTM agent initialized")

        # Training metrics
        episode_rewards = []
        episode_sharpes = []

        for episode in range(self.num_episodes):
            state = env.reset()
            agent.reset_hidden(batch_size=1)

            episode_reward = 0
            episode_returns = []
            done = False

            # Collect episode data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done:
                action, diagnostics = agent.select_action(state)
                next_state, reward, done, info = env.step(np.array([action]))

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(diagnostics['log_prob'])
                values.append(diagnostics['value'])

                episode_reward += reward
                episode_returns.append(info.get('pnl', 0))
                state = next_state

            # Calculate episode Sharpe
            if len(episode_returns) > 1:
                mean_return = np.mean(episode_returns)
                std_return = np.std(episode_returns)
                sharpe = mean_return / (std_return + 1e-8)
            else:
                sharpe = 0.0

            episode_rewards.append(episode_reward)
            episode_sharpes.append(sharpe)

            # PPO update (every 10 episodes)
            if (episode + 1) % 10 == 0 and len(states) > 0:
                returns = self._compute_returns(rewards, values, lstm_ppo_config.gamma)
                advantages = returns - np.array(values)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                states_tensor = torch.FloatTensor(np.array(states)).to(agent.device)
                actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(-1).to(agent.device)
                old_log_probs_tensor = torch.FloatTensor(np.array(log_probs)).unsqueeze(-1).to(agent.device)
                returns_tensor = torch.FloatTensor(returns).unsqueeze(-1).to(agent.device)
                advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(agent.device)

                agent.update(
                    states_tensor,
                    actions_tensor,
                    old_log_probs_tensor,
                    returns_tensor,
                    advantages_tensor
                )

            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_sharpe = np.mean(episode_sharpes[-10:])
                logger.info(
                    f"Episode {episode+1}/{self.num_episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Avg Sharpe: {avg_sharpe:.4f}"
                )

            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                checkpoint_path = self.model_dir / f"lstm_ppo_ep{episode+1}.pt"
                agent.save(str(checkpoint_path))
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info("Training completed!")

        # Save final model
        final_model_path = self.model_dir / "lstm_ppo_final.pt"
        agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        # Save timestamped version
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_path = self.model_dir / f"lstm_ppo_final_{timestamp}.pt"
        agent.save(str(timestamped_path))

        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Total Episodes: {self.num_episodes}")
        logger.info(f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.3f}")
        logger.info(f"Avg Sharpe (last 100): {np.mean(episode_sharpes[-100:]):.3f}")
        logger.info(f"Model saved to: {final_model_path}")
        logger.info("=" * 60)

        return {
            'episode_rewards': episode_rewards,
            'episode_sharpes': episode_sharpes,
        }

    def _compute_returns(self, rewards, values, gamma):
        """Compute discounted returns."""
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        return np.array(returns)


def main():
    parser = argparse.ArgumentParser(description='HIMARI Layer 3 LSTM Local Training')

    parser.add_argument('--model-dir', type=str, default='/tmp/models')
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HIMARI Layer 3 - LSTM Local Training")
    logger.info("=" * 60)

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    trainer = LSTMLocalTrainer(
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval,
        device=args.device
    )

    trainer.train()


if __name__ == '__main__':
    main()
