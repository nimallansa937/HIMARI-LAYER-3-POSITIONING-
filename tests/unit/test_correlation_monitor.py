"""
Unit tests for Correlation Monitor
"""

import pytest
import sys
import numpy as np
sys.path.insert(0, 'src')

from portfolio.correlation_monitor import CorrelationMonitor


class TestCorrelationMonitor:
    """Test suite for Correlation Monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = CorrelationMonitor(
            window_size=60,
            correlation_alert_threshold=0.7
        )
        
        assert monitor.window_size == 60
        assert monitor.correlation_alert_threshold == 0.7
    
    def test_single_update_insufficient(self):
        """Test that single update returns no correlation."""
        monitor = CorrelationMonitor(min_samples=5)
        
        corr, diag = monitor.update({'BTC-USD': 0.05})
        
        assert corr is None
        assert diag['status'] in ['insufficient_symbols', 'insufficient_data']
    
    def test_multiple_symbols_insufficient_data(self):
        """Test multiple symbols but insufficient data."""
        monitor = CorrelationMonitor(min_samples=20)
        
        # Only 5 samples
        for _ in range(5):
            monitor.update({'BTC-USD': 0.01, 'ETH-USD': 0.02})
        
        corr, diag = monitor.update({'BTC-USD': 0.01, 'ETH-USD': 0.02})
        
        assert diag['status'] == 'insufficient_data'
    
    def test_valid_correlation_calculation(self):
        """Test valid correlation calculation."""
        monitor = CorrelationMonitor(min_samples=10)
        
        # Add 15 correlated samples
        for i in range(15):
            btc_ret = 0.01 * (i % 3)
            eth_ret = 0.01 * (i % 3) + 0.001  # Highly correlated
            monitor.update({'BTC-USD': btc_ret, 'ETH-USD': eth_ret})
        
        corr, diag = monitor.update({'BTC-USD': 0.02, 'ETH-USD': 0.021})
        
        assert corr is not None
        assert diag['status'] == 'success'
        assert corr.shape == (2, 2)
    
    def test_high_correlation_alert(self):
        """Test high correlation alert detection."""
        monitor = CorrelationMonitor(
            min_samples=10,
            correlation_alert_threshold=0.5
        )
        
        # Perfectly correlated returns
        for i in range(15):
            ret = 0.01 * i
            monitor.update({'BTC-USD': ret, 'ETH-USD': ret})
        
        corr, diag = monitor.update({'BTC-USD': 0.15, 'ETH-USD': 0.15})
        
        # Should detect high correlation
        assert len(diag['high_correlation_alerts']) > 0
    
    def test_diversification_score(self):
        """Test diversification score calculation."""
        monitor = CorrelationMonitor(min_samples=5)
        
        # Uncorrelated returns
        np.random.seed(42)
        for _ in range(10):
            monitor.update({
                'BTC-USD': np.random.randn() * 0.01,
                'ETH-USD': np.random.randn() * 0.01,
                'SOL-USD': np.random.randn() * 0.01
            })
        
        state = monitor.get_state()
        
        # Low correlation = high diversification
        assert 'diversification_score' in state
        assert 0 <= state['diversification_score'] <= 1
    
    def test_get_correlation_for_pair(self):
        """Test getting correlation for specific pair."""
        monitor = CorrelationMonitor(min_samples=5)
        
        for _ in range(10):
            monitor.update({'BTC-USD': 0.01, 'ETH-USD': 0.01})
        
        corr = monitor.get_correlation_for_pair('BTC-USD', 'ETH-USD')
        
        assert corr is not None
        assert -1 <= corr <= 1
    
    def test_get_matrix_as_dict(self):
        """Test correlation matrix as dictionary."""
        monitor = CorrelationMonitor(min_samples=5)
        
        for _ in range(10):
            monitor.update({'BTC-USD': 0.01, 'ETH-USD': 0.02})
        
        matrix_dict = monitor.get_matrix_as_dict()
        
        assert 'BTC-USD' in matrix_dict
        assert 'ETH-USD' in matrix_dict['BTC-USD']
    
    def test_get_state(self):
        """Test state retrieval."""
        monitor = CorrelationMonitor()
        
        state = monitor.get_state()
        
        assert 'window_size' in state
        assert 'total_updates' in state
        assert 'n_symbols' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
