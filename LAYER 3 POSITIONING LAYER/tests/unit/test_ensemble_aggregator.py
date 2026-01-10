"""
Unit tests for Ensemble Position Aggregator V2
"""

import pytest
import sys
import os
import tempfile
import numpy as np
sys.path.insert(0, 'src')

from portfolio.ensemble_aggregator import EnsemblePositionAggregatorV2


class TestEnsemblePositionAggregatorV2:
    """Test suite for Ensemble Position Aggregator."""
    
    def test_initialization(self):
        """Test aggregator initialization."""
        agg = EnsemblePositionAggregatorV2(
            portfolio_value=100000,
            max_position_pct=0.10
        )
        
        assert agg.portfolio_value == 100000
        assert agg.max_position_pct == 0.10
    
    def test_single_strategy_aggregation(self):
        """Test aggregation with single strategy."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        strategies = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 5000, 'confidence': 0.8}
        ]
        
        allocations, diag = agg.aggregate_positions(strategies)
        
        assert 'BTC-USD' in allocations
        assert allocations['BTC-USD'] > 0
        assert diag['status'] == 'success'
    
    def test_multi_strategy_aggregation(self):
        """Test aggregation with multiple strategies."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        strategies = [
            {'id': 'momentum_btc', 'symbol': 'BTC-USD', 'size': 5000, 'confidence': 0.8},
            {'id': 'momentum_eth', 'symbol': 'ETH-USD', 'size': 3000, 'confidence': 0.7},
            {'id': 'mean_rev', 'symbol': 'SOL-USD', 'size': 2000, 'confidence': 0.6}
        ]
        
        allocations, diag = agg.aggregate_positions(strategies)
        
        assert len(allocations) == 3
        assert diag['num_strategies'] == 3
    
    def test_same_symbol_aggregation(self):
        """Test that multiple strategies on same symbol aggregate."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        strategies = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 3000, 'confidence': 0.8},
            {'id': 'mean_rev', 'symbol': 'BTC-USD', 'size': 2000, 'confidence': 0.7}
        ]
        
        allocations, diag = agg.aggregate_positions(strategies)
        
        # Should have single BTC-USD entry
        assert 'BTC-USD' in allocations
        assert len(allocations) == 1
    
    def test_weight_tracking(self):
        """Test that weight history is tracked."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        strategies = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 5000, 'confidence': 0.8}
        ]
        
        agg.aggregate_positions(strategies)
        
        assert len(agg.weight_history) == 1
        assert agg.last_weights.get('momentum') > 0
    
    def test_drift_detection(self):
        """Test weight drift detection."""
        agg = EnsemblePositionAggregatorV2(
            portfolio_value=100000,
            drift_alert_threshold=0.20
        )
        
        # First aggregation
        strategies1 = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 5000, 'confidence': 0.8},
            {'id': 'mean_rev', 'symbol': 'ETH-USD', 'size': 5000, 'confidence': 0.8}
        ]
        agg.aggregate_positions(strategies1)
        
        # Second with significant drift
        strategies2 = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 8000, 'confidence': 0.8},
            {'id': 'mean_rev', 'symbol': 'ETH-USD', 'size': 2000, 'confidence': 0.8}
        ]
        _, diag = agg.aggregate_positions(strategies2)
        
        # Should have drift alerts
        assert agg.drift_alerts_triggered > 0
    
    def test_position_limit(self):
        """Test position limit enforcement."""
        agg = EnsemblePositionAggregatorV2(
            portfolio_value=100000,
            max_position_pct=0.05  # 5% = $5000
        )
        
        strategies = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 10000, 'confidence': 1.0}
        ]
        
        allocations, _ = agg.aggregate_positions(strategies)
        
        assert allocations['BTC-USD'] <= 5000
    
    def test_export_weight_history(self):
        """Test CSV export."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        strategies = [
            {'id': 'momentum', 'symbol': 'BTC-USD', 'size': 5000, 'confidence': 0.8}
        ]
        agg.aggregate_positions(strategies)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            result = agg.export_weight_history(temp_path)
            assert result == True
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_state(self):
        """Test state retrieval."""
        agg = EnsemblePositionAggregatorV2(portfolio_value=100000)
        
        state = agg.get_state()
        
        assert 'portfolio_value' in state
        assert 'total_aggregations' in state
        assert 'history_size' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
