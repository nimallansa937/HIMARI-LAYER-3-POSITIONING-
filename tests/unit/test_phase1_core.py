"""
Unit tests for Phase 1 Core Pipeline
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from phases.phase1_core import Layer3Phase1
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators,
    InvalidSignalException
)


class TestLayer3Phase1:
    """Test suite for Phase 1 Core Pipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        layer3 = Layer3Phase1(
            portfolio_value=100000,
            kelly_fraction=0.25,
            enable_metrics=False
        )
        
        assert layer3.portfolio_value == 100000
        assert layer3.bayesian_kelly is not None
        assert layer3.conformal_scaler is not None
        assert layer3.regime_adjuster is not None
        assert layer3.sentiment_sizer is not None
        
        layer3.stop()
    
    def test_calculate_position_basic(self):
        """Test basic position calculation."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns(),
            expected_return=0.08,
            predicted_volatility=0.03
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        decision = layer3.calculate_position(signal, cascade, 42000.0)
        
        assert decision.position_size_usd > 0
        assert decision.symbol == "BTC-USD"
        assert decision.strategy_id == "test"
        assert 'pipeline_stages' in decision.diagnostics
        
        layer3.stop()
    
    def test_validation_missing_symbol(self):
        """Test validation rejects missing symbol."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="",  # Empty symbol
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns()
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        with pytest.raises(InvalidSignalException):
            layer3.calculate_position(signal, cascade, 42000.0)
        
        layer3.stop()
    
    def test_validation_invalid_confidence(self):
        """Test validation rejects invalid confidence."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=1.5,  # Invalid: > 1.0
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns()
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        with pytest.raises(InvalidSignalException):
            layer3.calculate_position(signal, cascade, 42000.0)
        
        layer3.stop()
    
    def test_sentiment_integration(self):
        """Test sentiment fields are used."""
        layer3 = Layer3Phase1(
            portfolio_value=100000,
            enable_metrics=False,
            enable_sentiment=True
        )
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns(),
            expected_return=0.08,
            sentiment_score=0.9,  # Very bullish
            sentiment_confidence=1.0
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        decision = layer3.calculate_position(signal, cascade, 42000.0)
        
        # Verify sentiment diagnostics are included
        assert 'sentiment' in decision.diagnostics
        sentiment_diag = decision.diagnostics['sentiment']
        # Sentiment should be enabled and show multiplier > 1 for bullish
        if sentiment_diag.get('enabled', False):
            assert sentiment_diag.get('multiplier', 1.0) >= 1.0
        
        layer3.stop()
    
    def test_cascade_reduction(self):
        """Test high cascade risk reduces position."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns(),
            expected_return=0.08
        )
        
        # HIGH RISK cascade
        cascade = CascadeIndicators(
            funding_rate=0.01,
            oi_change_pct=-0.20,
            volume_ratio=10.0,
            onchain_whale_pressure=0.9,
            exchange_netflow_zscore=4.0
        )
        
        decision = layer3.calculate_position(signal, cascade, 42000.0)
        
        assert decision.cascade_risk_score > 0.6
        assert decision.cascade_recommendation in ["REDUCE_75%", "REDUCE_50%", "EXIT"]
        
        layer3.stop()
    
    def test_get_state(self):
        """Test state retrieval."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        state = layer3.get_state()
        
        assert 'portfolio_value' in state
        assert 'bayesian_kelly' in state
        assert 'conformal_scaler' in state
        assert 'regime_adjuster' in state
        assert 'sentiment_sizer' in state
        
        layer3.stop()
    
    def test_update_from_trade_result(self):
        """Test trade result updates components."""
        layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
        
        layer3.update_from_trade_result(
            predicted_return=0.05,
            actual_return=0.06,
            trade_won=True
        )
        
        assert layer3.bayesian_kelly.total_trades == 1
        assert layer3.bayesian_kelly.winning_trades == 1
        assert len(layer3.conformal_scaler.residuals) == 1
        
        layer3.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
