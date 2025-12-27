#!/usr/bin/env python3
"""
HIMARI Layer 3 - Multi-Asset Vertex AI Training Script
Trains PPO agent on multiple cryptocurrencies simultaneously
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from google.cloud import storage

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rl.ppo_agent import PPOAgent, PPOConfig
from rl.multi_asset_env import MultiAssetTradingEnv, MultiAssetEnvConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetVertexAITrainer:
    """Vertex AI training wrapper for Multi-Asset PPO Agent"""

    def __init__(
        self,
        bucket_name: str,
        model_dir: str,
        symbols: list,
        num_episodes: int = 1000,
        save_interval: int = 100
    ):
        self.bucket_name = bucket_name
        self.model_dir = model_dir
        self.symbols = symbols
        self.num_episodes = num_episodes
        self.save_interval = save_interval

        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # Create local model directory
        self.local_model_dir = Path("/tmp/models")
        self.local_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Multi-Asset VertexAITrainer")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Episodes: {num_episodes}")

    def create_ppo_config(self, state_dim: int, action_dim: int) -> PPOConfig:
        """Create PPO configuration for multi-asset training"""
        return PPOConfig(
            state_dim=state_dim,  # 16 features × num_assets + 4 correlation features
            action_dim=action_dim,  # num_assets (one multiplier per asset)
            hidden_dim=256,  # Larger network for multi-asset
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
        )

    def create_env_config(self) -> MultiAssetEnvConfig:
        """Create multi-asset environment configuration"""
        return MultiAssetEnvConfig(
            symbols=self.symbols,
            initial_capital=100000.0,
            max_position_pct=0.5,
            commission_rate=0.001,
            slippage_bps=5,
            reward_window=10,
            max_steps=500,
            use_live_prices=True
        )

    def upload_to_gcs(self, local_path: Path, gcs_path: str):
        """Upload a file to Google Cloud Storage"""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path.name} to gs://{self.bucket_name}/{gcs_path}")

    def train(self):
        """Main training loop"""
        logger.info("Starting Multi-Asset PPO training...")

        # Initialize environment
        env_config = self.create_env_config()
        env = MultiAssetTradingEnv(env_config)

        # Initialize agent with multi-asset dimensions
        ppo_config = self.create_ppo_config(
            state_dim=env.state_dim,
            action_dim=env.action_dim
        )

        agent = PPOAgent(
            ppo_config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info(f"Environment initialized:")
        logger.info(f"  State dim: {env.state_dim}")
        logger.info(f"  Action dim: {env.action_dim} (one per asset)")
        logger.info(f"  Assets: {', '.join(self.symbols)}")

        # Training metrics
        episode_rewards = []
        episode_sharpes = []
        asset_performances = {symbol: [] for symbol in self.symbols}

        for episode in range(self.num_episodes):
            state = env.reset()

            episode_reward = 0
            episode_returns = []
            done = False

            # Collect episode data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done:
                # Select actions (one per asset)
                action, diagnostics = agent.select_action(state)

                # Ensure action is array with correct dimensions
                if isinstance(action, float):
                    action = np.array([action] * env.action_dim)

                # Environment step
                next_state, reward, done, info = env.step(action)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(diagnostics['log_prob'])
                values.append(diagnostics['value'])

                episode_reward += reward
                episode_returns.append(info.get('pnl', 0))

                # Track per-asset performance
                for symbol in self.symbols:
                    position = info['positions'].get(symbol, 0)
                    if position != 0:
                        asset_performances[symbol].append(1)

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
                # Compute returns and advantages
                returns = self._compute_returns(rewards, values, ppo_config.gamma)
                advantages = returns - np.array(values)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states)).to(agent.device)
                actions_tensor = torch.FloatTensor(np.array(actions)).to(agent.device)
                if actions_tensor.dim() == 1:
                    actions_tensor = actions_tensor.unsqueeze(-1)

                old_log_probs_tensor = torch.FloatTensor(np.array(log_probs)).unsqueeze(-1).to(agent.device)
                returns_tensor = torch.FloatTensor(returns).unsqueeze(-1).to(agent.device)
                advantages_tensor = torch.FloatTensor(advantages).unsqueeze(-1).to(agent.device)

                # PPO update
                update_info = agent.update(
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

                # Log per-asset utilization
                for symbol in self.symbols:
                    utilization = len(asset_performances[symbol]) / ((episode + 1) * env_config.max_steps) * 100
                    logger.info(f"  {symbol} utilization: {utilization:.1f}%")

            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                checkpoint_path = self.local_model_dir / f"multiasset_ppo_ep{episode+1}.pt"
                agent.save(str(checkpoint_path))

                gcs_checkpoint_path = f"{self.model_dir}/multiasset_ppo_ep{episode+1}.pt"
                self.upload_to_gcs(checkpoint_path, gcs_checkpoint_path)

        logger.info("Training completed!")

        # Save final model
        final_model_path = self.local_model_dir / "multiasset_ppo_final.pt"
        agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        # Upload to GCS
        gcs_model_path = f"{self.model_dir}/multiasset_ppo_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        self.upload_to_gcs(final_model_path, gcs_model_path)

        # Upload as latest
        gcs_latest_path = f"{self.model_dir}/multiasset_ppo_latest.pt"
        self.upload_to_gcs(final_model_path, gcs_latest_path)

        logger.info(f"Model uploaded to gs://{self.bucket_name}/{gcs_latest_path}")

        return {
            'episode_rewards': episode_rewards,
            'episode_sharpes': episode_sharpes,
            'asset_performances': asset_performances,
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
    """Main entry point for Multi-Asset Vertex AI training"""
    parser = argparse.ArgumentParser(description='HIMARI Layer 3 Multi-Asset Vertex AI Training')

    parser.add_argument(
        '--bucket-name',
        type=str,
        required=True,
        help='GCS bucket name for model storage'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/himari-rl-multiasset',
        help='Model directory within the bucket'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        help='List of symbols to train on'
    )

    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1000,
        help='Number of training episodes'
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save checkpoint every N episodes'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("HIMARI Layer 3 - Multi-Asset Vertex AI Training")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Bucket: {args.bucket_name}")
    logger.info(f"  Model Dir: {args.model_dir}")
    logger.info(f"  Symbols: {', '.join(args.symbols)}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Save Interval: {args.save_interval}")
    logger.info("="*60)

    # Create trainer and run
    trainer = MultiAssetVertexAITrainer(
        bucket_name=args.bucket_name,
        model_dir=args.model_dir,
        symbols=args.symbols,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval
    )

    # Execute training
    metrics = trainer.train()

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"Total Episodes: {len(metrics['episode_rewards'])}")
    logger.info(f"Avg Reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.3f}")
    logger.info(f"Avg Sharpe (last 100): {np.mean(metrics['episode_sharpes'][-100:]):.3f}")
    logger.info("")
    logger.info("Asset Performance:")
    for symbol, performance in metrics['asset_performances'].items():
        total_trades = len(performance)
        logger.info(f"  {symbol}: {total_trades} trades")
    logger.info("="*60)


if __name__ == '__main__':
    main()
