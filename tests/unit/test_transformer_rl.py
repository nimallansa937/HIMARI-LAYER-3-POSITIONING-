"""
Unit tests for Transformer-RL Client
"""

import pytest
import sys
import asyncio
sys.path.insert(0, 'src')

from optimization.transformer_rl import (
    TransformerRLClient, MockTransformerRLClient, RLPrediction
)
from core.layer3_types import TacticalSignal, TacticalAction, MarketRegime
import time


class TestMockTransformerRLClient:
    """Test suite for Mock Transformer-RL Client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = MockTransformerRLClient(success_rate=0.9)
        
        assert client.success_rate == 0.9
        assert client.total_predictions == 0
    
    def test_mock_prediction_success(self):
        """Test mock prediction returns valid result."""
        client = MockTransformerRLClient(success_rate=1.0)  # Always succeed
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.8,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        prediction = asyncio.run(client.predict(signal))
        
        assert prediction is not None
        assert isinstance(prediction, RLPrediction)
        assert prediction.confidence > 0
        assert prediction.optimal_position_pct > 0
    
    def test_mock_prediction_failure(self):
        """Test mock prediction failure."""
        client = MockTransformerRLClient(success_rate=0.0)  # Always fail
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.8,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        prediction = asyncio.run(client.predict(signal))
        
        assert prediction is None
        assert client.fallback_predictions == 1
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        client = MockTransformerRLClient(success_rate=1.0)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.8,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        # Make 5 predictions
        for _ in range(5):
            asyncio.run(client.predict(signal))
        
        assert client.total_predictions == 5
        assert client.successful_predictions == 5
    
    def test_get_state(self):
        """Test state retrieval."""
        client = MockTransformerRLClient()
        
        state = client.get_state()
        
        assert 'total_predictions' in state
        assert 'success_rate' in state
        assert 'circuit_breaker' in state


class TestTransformerRLClient:
    """Test suite for real Transformer-RL Client (offline tests)."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = TransformerRLClient(
            endpoint_url="http://localhost:8888/predict",
            timeout_sec=5.0
        )
        
        assert client.endpoint_url == "http://localhost:8888/predict"
        assert client.timeout_sec == 5.0
    
    def test_circuit_breaker_attached(self):
        """Test that circuit breaker is attached."""
        client = TransformerRLClient()
        
        assert client.circuit_breaker is not None
        assert client.circuit_breaker.failure_threshold == 3
    
    def test_fallback_on_offline_endpoint(self):
        """Test fallback when endpoint is offline."""
        client = TransformerRLClient(
            endpoint_url="http://localhost:99999/predict",  # Invalid port
            timeout_sec=1.0
        )
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.8,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        # Should return None (fallback)
        prediction = client.predict_sync(signal)
        
        assert prediction is None
        assert client.fallback_predictions >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
