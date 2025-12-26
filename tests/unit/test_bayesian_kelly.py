"""
Unit tests for Bayesian Kelly Engine
"""

import pytest
import sys
sys.path.insert(0, 'src')

from engines.bayesian_kelly import BayesianKellyEngine


class TestBayesianKellyEngine:
    """Test suite for Bayesian Kelly position sizing."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = BayesianKellyEngine(portfolio_value=100000, kelly_fraction=0.25)
        
        assert engine.portfolio_value == 100000
        assert engine.kelly_fraction == 0.25
        assert engine.min_win_rate == 0.45
        assert engine.min_edge == 0.02
        assert engine.total_trades == 0
    
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        
        position, diagnostics = engine.calculate_position_size(
            confidence=0.75,
            expected_return=0.08,
            predicted_volatility=0.03
        )
        
        assert position > 0
        assert position <= 100000
        assert 'posterior_win_rate' in diagnostics
        assert 'posterior_edge' in diagnostics
        assert diagnostics['confidence'] == 0.75
    
    def test_below_min_threshold(self):
        """Test that position is zero when below minimum thresholds."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        # Set very low priors
        engine.alpha = 1.0
        engine.beta = 10.0  # Low win rate
        
        position, _ = engine.calculate_position_size(
            confidence=0.5,
            expected_return=0.01,  # Below min_edge
            predicted_volatility=0.02
        )
        
        assert position == 0.0
    
    def test_posterior_update_win(self):
        """Test posterior update with winning trade."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        initial_alpha = engine.alpha
        
        engine.update_posterior(trade_won=True, trade_return=0.05)
        
        assert engine.alpha == initial_alpha + 1.0
        assert engine.winning_trades == 1
        assert engine.total_trades == 1
    
    def test_posterior_update_loss(self):
        """Test posterior update with losing trade."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        initial_beta = engine.beta
        
        engine.update_posterior(trade_won=False, trade_return=-0.02)
        
        assert engine.beta == initial_beta + 1.0
        assert engine.winning_trades == 0
        assert engine.total_trades == 1
    
    def test_get_state(self):
        """Test state retrieval."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        engine.update_posterior(True, 0.05)
        
        state = engine.get_state()
        
        assert 'alpha' in state
        assert 'beta' in state
        assert 'posterior_win_rate' in state
        assert 'total_trades' in state
        assert state['total_trades'] == 1
    
    def test_reset_priors(self):
        """Test prior reset."""
        engine = BayesianKellyEngine(portfolio_value=100000)
        engine.update_posterior(True, 0.05)
        engine.reset_priors(alpha=15.0, beta=15.0)
        
        assert engine.alpha == 15.0
        assert engine.beta == 15.0
        assert engine.total_trades == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
