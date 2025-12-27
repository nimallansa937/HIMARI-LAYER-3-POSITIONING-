#!/usr/bin/env python3
"""
HIMARI Layer 3 - Vertex AI Training Script
Trains the Phase 3 Hybrid RL agent on Google Cloud Vertex AI
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

from rl.trainer import RLTrainer, TrainingConfig
from rl.trading_env import EnvConfig
from rl.ppo_agent import PPOConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VertexAITrainer:
    """Vertex AI training wrapper for HIMARI Layer 3 RL Engine"""
    
    def __init__(
        self,
        bucket_name: str,
        model_dir: str,
        num_episodes: int = 1000,
        save_interval: int = 100
    ):
        self.bucket_name = bucket_name
        self.model_dir = model_dir
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Create local model directory
        self.local_model_dir = Path("/tmp/models")
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VertexAITrainer")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Episodes: {num_episodes}")
    
    def create_ppo_config(self) -> PPOConfig:
        """Create PPO configuration for training"""
        return PPOConfig(
            state_dim=16,  # FIXED: 16-dim state (was 12)
            action_dim=1,  # FIXED: Continuous action (was 3 discrete)
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
    
    def upload_to_gcs(self, local_path: Path, gcs_path: str):
        """Upload a file to Google Cloud Storage"""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path.name} to gs://{self.bucket_name}/{gcs_path}")
    
    def download_from_gcs(self, gcs_path: str, local_path: Path):
        """Download a file from Google Cloud Storage"""
        blob = self.bucket.blob(gcs_path)
        if blob.exists():
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
            return True
        return False
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        # Initialize RL trainer
        ppo_config = self.create_ppo_config()
        env_config = self.create_env_config()
        training_config = TrainingConfig(
            num_episodes=self.num_episodes,
            max_steps_per_episode=500,
            batch_size=64,
            ppo_epochs=10,
            save_interval=self.save_interval,
            log_interval=10,
            checkpoint_dir='./checkpoints',
            use_live_prices=True
        )

        trainer = RLTrainer(
            training_config=training_config,
            env_config=env_config,
            ppo_config=ppo_config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info("Trainer initialized, starting training...")

        # Run training
        training_stats = trainer.train()

        logger.info("Training completed!")

        # Save final model
        final_model_path = self.local_model_dir / "ppo_final.pt"
        trainer.agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        # Upload to GCS
        gcs_model_path = f"{self.model_dir}/ppo_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        self.upload_to_gcs(final_model_path, gcs_model_path)

        # Upload as latest
        gcs_latest_path = f"{self.model_dir}/ppo_latest.pt"
        self.upload_to_gcs(final_model_path, gcs_latest_path)

        logger.info(f"Model uploaded to gs://{self.bucket_name}/{gcs_latest_path}")

        return training_stats
    


def main():
    """Main entry point for Vertex AI training"""
    parser = argparse.ArgumentParser(description='HIMARI Layer 3 Vertex AI Training')
    
    parser.add_argument(
        '--bucket-name',
        type=str,
        required=True,
        help='GCS bucket name for model storage'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/himari-rl',
        help='Model directory within the bucket'
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
    logger.info("HIMARI Layer 3 - Vertex AI Training")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Bucket: {args.bucket_name}")
    logger.info(f"  Model Dir: {args.model_dir}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Save Interval: {args.save_interval}")
    logger.info("="*60)
    
    # Create trainer and run
    trainer = VertexAITrainer(
        bucket_name=args.bucket_name,
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval
    )
    
    # Execute training
    metrics = trainer.train()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    if isinstance(metrics, dict):
        logger.info(f"Total Episodes: {len(metrics.get('episode_rewards', []))}")
        logger.info(f"Avg Reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.3f}")
        logger.info(f"Avg Sharpe (last 100): {np.mean(metrics['episode_sharpes'][-100:]):.3f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
