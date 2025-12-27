"""
HIMARI Layer 3 - Vertex AI Training Scripts
Unified trainer for PPO, LSTM, and Multi-Asset models.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from google.cloud import storage

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ppo_agent import PPOAgent, PPOConfig
from models.lstm_agent import LSTMAgent, LSTMConfig
from models.multi_asset import MultiAssetTrainer, MultiAssetConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEnvironment:
    """Simple trading environment for training."""
    
    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.position = 0.0
        self.pnl = 0.0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim)
        state[0] = np.random.uniform(0.5, 1.0)  # Confidence
        state[1:4] = np.random.multinomial(1, [0.3, 0.4, 0.3])  # Action
        state[4:8] = np.random.multinomial(1, [0.4, 0.3, 0.2, 0.1])  # Tier
        state[8] = self.position
        state[9] = 1 if self.position > 0 else -1
        state[10] = self.pnl / 100.0
        state[11] = np.random.randn() * 0.02  # Momentum 1h
        state[12] = np.random.randn() * 0.015  # Momentum 4h
        state[13] = abs(np.random.randn() * 0.01) + 0.01  # Volatility
        state[14] = np.random.uniform(0.4, 0.7)  # Win rate
        state[15] = np.random.uniform(0.0, 0.3)  # Cascade risk
        return state
    
    def step(self, action: float) -> tuple:
        self.step_count += 1
        
        # Simulate market
        price_return = np.random.randn() * 0.02 + 0.001
        
        # Calculate reward
        position_size = action * 0.5
        pnl_change = position_size * price_return * 100
        
        # Risk-adjusted reward
        volatility = abs(np.random.randn() * 0.02) + 0.01
        sharpe_adj = pnl_change / (volatility * 100 + 1e-8)
        risk_penalty = max(0, (action - 1.5)) * 0.02
        
        reward = sharpe_adj - risk_penalty
        
        self.pnl += pnl_change
        self.position = position_size
        
        done = self.step_count >= 500
        return self._get_state(), reward, done


def train_ppo(
    num_episodes: int,
    bucket_name: str,
    model_dir: str,
    device: str = 'cpu'
) -> dict:
    """Train PPO agent."""
    logger.info("="*60)
    logger.info("Training PPO Agent")
    logger.info("="*60)
    
    config = PPOConfig(
        state_dim=16,
        action_dim=1,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        batch_size=64
    )
    
    agent = PPOAgent(config, device)
    env = TradingEnvironment()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done = env.step(float(action))
            
            agent.buffer.add(state, action, log_prob, reward, value, done)
            episode_reward += reward
            
            if done:
                # Update agent
                _, _, last_value = agent.get_action(next_state)
                stats = agent.update(last_value)
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.3f}")
    
    # Save model
    save_path = f"/tmp/ppo_model.pt"
    agent.save(save_path)
    
    # Upload to GCS
    upload_to_gcs(bucket_name, save_path, f"{model_dir}/ppo_model.pt")
    
    return {
        'model': 'PPO',
        'episodes': num_episodes,
        'final_avg_reward': np.mean(episode_rewards[-100:])
    }


def train_lstm(
    num_episodes: int,
    bucket_name: str,
    model_dir: str,
    device: str = 'cpu'
) -> dict:
    """Train LSTM agent."""
    logger.info("="*60)
    logger.info("Training LSTM Agent")
    logger.info("="*60)
    
    config = LSTMConfig(
        state_dim=16,
        action_dim=1,
        hidden_dim=128,
        lstm_hidden_dim=64,
        num_lstm_layers=2,
        sequence_length=20
    )
    
    agent = LSTMAgent(config, device)
    env = TradingEnvironment()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_hidden()
        episode_reward = 0
        
        while True:
            action, _ = agent.get_action(state)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.3f}")
    
    # Save model
    save_path = f"/tmp/lstm_model.pt"
    agent.save(save_path)
    
    # Upload to GCS
    upload_to_gcs(bucket_name, save_path, f"{model_dir}/lstm_model.pt")
    
    return {
        'model': 'LSTM',
        'episodes': num_episodes,
        'final_avg_reward': np.mean(episode_rewards[-100:])
    }


def train_multi_asset(
    num_episodes: int,
    bucket_name: str,
    model_dir: str,
    device: str = 'cpu'
) -> dict:
    """Train multi-asset model."""
    logger.info("="*60)
    logger.info("Training Multi-Asset Model")
    logger.info("="*60)
    
    config = MultiAssetConfig(
        assets=['BTC-USD', 'ETH-USD', 'SOL-USD'],
        state_dim=16,
        action_dim=1,
        shared_hidden_dim=128,
        asset_hidden_dim=64
    )
    
    trainer = MultiAssetTrainer(config, device)
    rewards = trainer.train(num_episodes)
    
    # Save model
    save_path = f"/tmp/multi_asset_model.pt"
    trainer.save(save_path)
    
    # Upload to GCS
    upload_to_gcs(bucket_name, save_path, f"{model_dir}/multi_asset_model.pt")
    
    # Calculate final metrics per asset
    final_rewards = {
        asset: np.mean([r[asset] for r in rewards[-100:]])
        for asset in config.assets
    }
    
    return {
        'model': 'MultiAsset',
        'episodes': num_episodes,
        'final_rewards': final_rewards
    }


def upload_to_gcs(bucket_name: str, local_path: str, gcs_path: str):
    """Upload file to GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload: {e}")


def main():
    parser = argparse.ArgumentParser(description='HIMARI RL Training')
    parser.add_argument('--model', type=str, required=True,
                       choices=['ppo', 'lstm', 'multi_asset', 'all'],
                       help='Model to train')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes')
    parser.add_argument('--bucket', type=str, default='himari-rl-models',
                       help='GCS bucket name')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model directory in bucket')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("HIMARI Layer 3 - Vertex AI Training")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info("="*60)
    
    results = []
    
    if args.model in ['ppo', 'all']:
        result = train_ppo(args.episodes, args.bucket, args.model_dir, args.device)
        results.append(result)
    
    if args.model in ['lstm', 'all']:
        result = train_lstm(args.episodes, args.bucket, args.model_dir, args.device)
        results.append(result)
    
    if args.model in ['multi_asset', 'all']:
        result = train_multi_asset(args.episodes, args.bucket, args.model_dir, args.device)
        results.append(result)
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    summary_path = '/tmp/training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    upload_to_gcs(args.bucket, summary_path, f"{args.model_dir}/training_summary.json")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    for r in results:
        logger.info(f"  {r['model']}: {r.get('final_avg_reward', r.get('final_rewards'))}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
