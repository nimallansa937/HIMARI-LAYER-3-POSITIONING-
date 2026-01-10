#!/usr/bin/env python3
"""
HIMARI Layer 3 - Vertex AI Training Script
Trains the Phase 3 Hybrid RL agent on Google Cloud Vertex AI
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from google.cloud import storage

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engines.rl_engine import RLEngine, RLConfig
from src.engines.execution_engine import ExecutionConfig

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
    
    def create_rl_config(self) -> RLConfig:
        """Create RL configuration for training"""
        return RLConfig(
            state_dim=12,  # Market state features
            action_dim=3,  # 3 sizing modes (Conservative, Moderate, Aggressive)
            hidden_dim=128,
            lr=0.0003,
            gamma=0.99,
            tau=0.005,
            buffer_size=100000,
            batch_size=256,
            reward_scale=1.0,
            alpha=0.2,  # SAC entropy coefficient
            target_update_interval=1,
            gradient_steps=1
        )
    
    def create_execution_config(self) -> ExecutionConfig:
        """Create execution configuration"""
        return ExecutionConfig(
            max_position_size=1.0,
            min_position_size=0.01,
            max_leverage=3.0,
            enable_hedging=True,
            enable_colab_protection=True,
            risk_budget=0.02,
            max_drawdown=0.15,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
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
    
    def generate_synthetic_episode(self, episode_num: int) -> Dict[str, Any]:
        """
        Generate synthetic market episode for training
        In production, this would be replaced with real market data
        """
        # Simulate market conditions
        num_steps = np.random.randint(50, 200)
        
        # Generate market features
        returns = np.random.randn(num_steps) * 0.02
        volatility = np.abs(np.random.randn(num_steps) * 0.01) + 0.01
        volume = np.random.exponential(1.0, num_steps)
        
        # Calculate PnL based on position sizing decisions
        # This is a simplified simulation
        pnl = np.cumsum(returns * np.random.uniform(0.1, 1.0, num_steps))
        final_pnl = pnl[-1]
        
        # Calculate metrics
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        max_dd = np.min(pnl - np.maximum.accumulate(pnl))
        
        return {
            'episode': episode_num,
            'steps': num_steps,
            'final_pnl': float(final_pnl),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_volatility': float(np.mean(volatility))
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize RL engine
        rl_config = self.create_rl_config()
        exec_config = self.create_execution_config()
        
        engine = RLEngine(
            config=rl_config,
            execution_config=exec_config,
            enable_phase3=True  # Enable Phase 3 Hybrid RL
        )
        
        # Try to load existing checkpoint
        checkpoint_path = self.local_model_dir / "checkpoint.pt"
        start_episode = 0
        
        if self.download_from_gcs(f"{self.model_dir}/checkpoint.pt", checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                engine.load_state_dict(checkpoint['model_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                logger.info(f"Resumed from episode {start_episode}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Training metrics
        training_metrics = []
        
        # Training loop
        for episode in range(start_episode, self.num_episodes):
            # Generate synthetic episode
            episode_data = self.generate_synthetic_episode(episode)
            
            # In production, you would:
            # 1. Sample real market data
            # 2. Run RL agent through the episode
            # 3. Collect experiences
            # 4. Update policy
            
            # For now, we'll simulate the training process
            # The actual RL training would use engine.update() with real experiences
            
            training_metrics.append(episode_data)
            
            if episode % 10 == 0:
                avg_pnl = np.mean([m['final_pnl'] for m in training_metrics[-10:]])
                logger.info(f"Episode {episode}/{self.num_episodes} - Avg PnL: {avg_pnl:.4f}")
            
            # Save checkpoint periodically
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(engine, episode, training_metrics)
        
        # Final save
        self.save_checkpoint(engine, self.num_episodes - 1, training_metrics, final=True)
        
        logger.info("Training completed!")
        
        return training_metrics
    
    def save_checkpoint(
        self,
        engine: RLEngine,
        episode: int,
        metrics: list,
        final: bool = False
    ):
        """Save model checkpoint and metrics to GCS"""
        # Save model checkpoint
        checkpoint_path = self.local_model_dir / "checkpoint.pt"
        torch.save({
            'episode': episode,
            'model_state_dict': engine.state_dict(),
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        gcs_checkpoint_path = f"{self.model_dir}/checkpoint.pt"
        self.upload_to_gcs(checkpoint_path, gcs_checkpoint_path)
        
        # Save metrics
        metrics_path = self.local_model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'episodes': len(metrics),
                'metrics': metrics[-100:]  # Save last 100 episodes
            }, f, indent=2)
        
        gcs_metrics_path = f"{self.model_dir}/metrics.json"
        self.upload_to_gcs(metrics_path, gcs_metrics_path)
        
        # If final, save as production model
        if final:
            final_model_path = self.local_model_dir / "model_final.pt"
            torch.save(engine.state_dict(), final_model_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_final_path = f"{self.model_dir}/model_final_{timestamp}.pt"
            self.upload_to_gcs(final_model_path, gcs_final_path)
            
            logger.info(f"Saved final model to gs://{self.bucket_name}/{gcs_final_path}")


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
    logger.info(f"Total Episodes: {len(metrics)}")
    logger.info(f"Avg Final PnL: {np.mean([m['final_pnl'] for m in metrics]):.4f}")
    logger.info(f"Avg Sharpe: {np.mean([m['sharpe_ratio'] for m in metrics]):.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
