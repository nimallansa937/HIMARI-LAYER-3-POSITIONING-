#!/usr/bin/env python3
"""
HIMARI Layer 3 - Lambda Labs Local Training Script
Trains PPO agent locally without GCS authentication
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

from rl.trainer import RLTrainer, TrainingConfig
from rl.trading_env import EnvConfig
from rl.ppo_agent import PPOConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalTrainer:
    """Local training wrapper for Lambda Labs (no GCS dependency)"""

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

        # Create local model directory
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized LocalTrainer")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Episodes: {num_episodes}")
        logger.info(f"  Device: {device}")

    def create_ppo_config(self) -> PPOConfig:
        """Create PPO configuration for training"""
        return PPOConfig(
            state_dim=16,
            action_dim=1,
            hidden_dim=128,
            learning_rate=3e-4,
            gamma=0.99,
            lambda_gae=0.95,
            clip_epsilon=0.2,
        )

    def create_env_config(self) -> EnvConfig:
        """Create trading environment configuration"""
        return EnvConfig(
            initial_capital=100000.0,
            max_position_pct=0.5,
            commission_rate=0.001,
            slippage_bps=5,
            reward_window=10,
            max_steps=500,
            symbol='BTC-USD'
        )

    def train(self):
        """Main training loop"""
        logger.info("Starting PPO training...")

        # Initialize environment and agent
        env_config = self.create_env_config()
        ppo_config = self.create_ppo_config()

        training_config = TrainingConfig(
            num_episodes=self.num_episodes,
            max_steps_per_episode=500,
            batch_size=64,
            ppo_epochs=10,
            save_interval=self.save_interval,
            log_interval=10,
            checkpoint_dir=str(self.model_dir / 'checkpoints'),
            use_live_prices=True
        )

        # Create trainer
        trainer = RLTrainer(
            training_config=training_config,
            env_config=env_config,
            ppo_config=ppo_config,
            device=self.device
        )

        logger.info("Trainer initialized")
        logger.info("")

        # Train
        logger.info("Starting training...")
        training_stats = trainer.train()
        logger.info("Training complete!")
        logger.info("")

        # Save final model locally
        final_model_path = self.model_dir / 'ppo_final.pt'
        trainer.agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_path = self.model_dir / f'ppo_final_{timestamp}.pt'
        trainer.agent.save(str(timestamped_path))
        logger.info(f"Timestamped model saved: {timestamped_path}")

        # Print training summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Total Episodes: {self.num_episodes}")
        logger.info(f"Model saved to: {final_model_path}")
        logger.info("")
        logger.info("To upload to GCS, run:")
        logger.info(f"  gsutil cp {final_model_path} gs://himari-rl-models/models/himari-rl/ppo_latest.pt")
        logger.info("=" * 60)

        return training_stats


def main():
    """Main entry point for local training"""
    parser = argparse.ArgumentParser(description='HIMARI Layer 3 Local Training (Lambda Labs)')

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/models',
        help='Local directory for model storage'
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

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HIMARI Layer 3 - Local Training (Lambda Labs)")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Model Dir: {args.model_dir}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Save Interval: {args.save_interval}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 60)
    logger.info("")

    # Verify CUDA availability if requested
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'
        else:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create trainer and run
    trainer = LocalTrainer(
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval,
        device=args.device
    )

    # Execute training
    trainer.train()


if __name__ == '__main__':
    main()
