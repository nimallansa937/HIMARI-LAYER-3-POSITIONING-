"""
Unit tests for Phase 2 Portfolio Orchestrator
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from phases.phase2_portfolio import Layer3Phase2Portfolio
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)


class TestLayer3Phase2Portfolio:
    """Test suite for Phase 2 Portfolio Orchestrator."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        assert portfolio.portfolio_value == 100000
        assert portfolio.phase1 is not None
        assert portfolio.kelly_allocator is not None
        assert portfolio.ensemble is not None
        assert portfolio.correlation_monitor is not None
        
        portfolio.stop()
    
    def test_empty_signals(self):
        """Test handling of empty signals list."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        allocation = portfolio.process_multi_strategy_signals(
            signals=[],
            cascade_indicators=cascade,
            current_prices={}
        )
        
        assert allocation.total_allocated_usd == 0.0
        
        portfolio.stop()
    
    def test_single_signal_processing(self):
        """Test processing single signal."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        signal = TacticalSignal(
            strategy_id="momentum",
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
        
        allocation = portfolio.process_multi_strategy_signals(
            signals=[signal],
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        )
        
        assert allocation.total_allocated_usd > 0
        assert 'BTC-USD' in allocation.allocations
        
        portfolio.stop()
    
    def test_multi_signal_processing(self):
        """Test processing multiple signals."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        signals = [
            TacticalSignal(
                strategy_id="momentum_btc",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.75,
                risk_score=0.3,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.08,
                predicted_volatility=0.03
            ),
            TacticalSignal(
                strategy_id="momentum_eth",
                symbol="ETH-USD",
                action=TacticalAction.BUY,
                confidence=0.70,
                risk_score=0.35,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.10,
                predicted_volatility=0.04
            )
        ]
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        allocation = portfolio.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0, 'ETH-USD': 2200.0}
        )
        
        assert len(allocation.allocations) == 2
        assert allocation.utilization_pct > 0
        
        portfolio.stop()
    
    def test_invalid_price_skipped(self):
        """Test that signals with invalid prices are skipped."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        signal = TacticalSignal(
            strategy_id="momentum",
            symbol="BTC-USD",
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
        
        # Missing price for BTC-USD
        allocation = portfolio.process_multi_strategy_signals(
            signals=[signal],
            cascade_indicators=cascade,
            current_prices={'ETH-USD': 2200.0}  # BTC-USD missing
        )
        
        # Should return empty allocation
        assert allocation.total_allocated_usd == 0.0
        
        portfolio.stop()
    
    def test_get_state(self):
        """Test state retrieval."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        state = portfolio.get_state()
        
        assert 'portfolio_value' in state
        assert 'phase1' in state
        assert 'kelly_allocator' in state
        assert 'ensemble' in state
        assert 'correlation_monitor' in state
        
        portfolio.stop()
    
    def test_update_correlations(self):
        """Test correlation update."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        diag = portfolio.update_correlations({
            'BTC-USD': 0.05,
            'ETH-USD': 0.03
        })
        
        assert 'status' in diag
        
        portfolio.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
