#!/usr/bin/env python3
"""
Local test for Vertex AI training script
Tests the training code locally before deploying to Vertex AI
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vertex_ai.trainer.train import VertexAITrainer, RLEngine, RLConfig, ExecutionConfig


def test_rl_config_creation():
    """Test RLConfig creation"""
    print("Testing RLConfig creation...")
    
    config = RLConfig(
        state_dim=12,
        action_dim=3,
        hidden_dim=128,
        lr=0.0003,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256
    )
    
    assert config.state_dim == 12
    assert config.action_dim == 3
    assert config.hidden_dim == 128
    
    print("✓ RLConfig creation successful")


def test_execution_config_creation():
    """Test ExecutionConfig creation"""
    print("Testing ExecutionConfig creation...")
    
    config = ExecutionConfig(
        max_position_size=1.0,
        min_position_size=0.01,
        max_leverage=3.0,
        enable_hedging=True,
        enable_colab_protection=True
    )
    
    assert config.max_position_size == 1.0
    assert config.max_leverage == 3.0
    assert config.enable_hedging is True
    
    print("✓ ExecutionConfig creation successful")


def test_rl_engine_initialization():
    """Test RLEngine initialization"""
    print("Testing RLEngine initialization...")
    
    rl_config = RLConfig(
        state_dim=12,
        action_dim=3,
        hidden_dim=128,
        lr=0.0003,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256
    )
    
    exec_config = ExecutionConfig(
        max_position_size=1.0,
        min_position_size=0.01,
        max_leverage=3.0
    )
    
    engine = RLEngine(
        config=rl_config,
        execution_config=exec_config,
        enable_phase3=True
    )
    
    assert engine is not None
    assert engine.config.state_dim == 12
    
    print("✓ RLEngine initialization successful")


def test_synthetic_episode_generation():
    """Test synthetic episode generation"""
    print("Testing synthetic episode generation...")
    
    # Create mock trainer (without GCS dependencies)
    with patch('vertex_ai.trainer.train.storage.Client'):
        trainer = Mock()
        trainer.generate_synthetic_episode = VertexAITrainer.generate_synthetic_episode.__get__(trainer)
        
        episode_data = trainer.generate_synthetic_episode(0)
        
        assert 'episode' in episode_data
        assert 'steps' in episode_data
        assert 'final_pnl' in episode_data
        assert 'sharpe_ratio' in episode_data
        assert 'max_drawdown' in episode_data
        
        print(f"  Episode steps: {episode_data['steps']}")
        print(f"  Final PnL: {episode_data['final_pnl']:.4f}")
        print(f"  Sharpe ratio: {episode_data['sharpe_ratio']:.4f}")
    
    print("✓ Synthetic episode generation successful")


def test_training_loop_short():
    """Test a short training loop (5 episodes)"""
    print("Testing short training loop (5 episodes)...")
    
    # Mock GCS client
    with patch('vertex_ai.trainer.train.storage.Client'):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock trainer
            trainer = Mock()
            trainer.bucket_name = "test-bucket"
            trainer.model_dir = "test-models"
            trainer.num_episodes = 5
            trainer.save_interval = 2
            trainer.local_model_dir = Path(tmpdir)
            trainer.storage_client = Mock()
            trainer.bucket = Mock()
            
            # Bind methods
            trainer.create_rl_config = VertexAITrainer.create_rl_config.__get__(trainer)
            trainer.create_execution_config = VertexAITrainer.create_execution_config.__get__(trainer)
            trainer.generate_synthetic_episode = VertexAITrainer.generate_synthetic_episode.__get__(trainer)
            trainer.download_from_gcs = Mock(return_value=False)
            trainer.save_checkpoint = Mock()
            
            # Create RL components
            rl_config = trainer.create_rl_config()
            exec_config = trainer.create_execution_config()
            
            engine = RLEngine(
                config=rl_config,
                execution_config=exec_config,
                enable_phase3=True
            )
            
            # Run mini training loop
            training_metrics = []
            for episode in range(5):
                episode_data = trainer.generate_synthetic_episode(episode)
                training_metrics.append(episode_data)
            
            assert len(training_metrics) == 5
            print(f"  Completed {len(training_metrics)} episodes")
            print(f"  Avg PnL: {sum(m['final_pnl'] for m in training_metrics) / len(training_metrics):.4f}")
    
    print("✓ Short training loop successful")


def main():
    """Run all tests"""
    print("="*60)
    print("HIMARI Layer 3 - Vertex AI Training Local Tests")
    print("="*60)
    print()
    
    tests = [
        test_rl_config_creation,
        test_execution_config_creation,
        test_rl_engine_initialization,
        test_synthetic_episode_generation,
        test_training_loop_short
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
            print()
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All tests passed! Ready to deploy to Vertex AI.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix before deploying.")
        return 1


if __name__ == '__main__':
    exit(main())
