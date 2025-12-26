"""
Integration tests for Phase 3 Hybrid Pipeline
"""

import pytest
import sys
import time
import asyncio
sys.path.insert(0, 'src')

from phases.phase3_hybrid import Layer3Phase3Hybrid
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)


class TestPhase3Integration:
    """Integration tests for Phase 3 pipeline."""
    
    def test_full_phase3_pipeline_with_mock_rl(self):
        """Test complete Phase 3 pipeline with Mock RL."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        signals = [
            TacticalSignal(
                strategy_id="momentum_btc",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.80,
                risk_score=0.25,
                regime=MarketRegime.TRENDING_UP,
                timestamp_ns=time.time_ns(),
                expected_return=0.10,
                predicted_volatility=0.04
            ),
            TacticalSignal(
                strategy_id="momentum_eth",
                symbol="ETH-USD",
                action=TacticalAction.BUY,
                confidence=0.70,
                risk_score=0.30,
                regime=MarketRegime.TRENDING_UP,
                timestamp_ns=time.time_ns(),
                expected_return=0.12,
                predicted_volatility=0.05
            )
        ]
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        prices = {'BTC-USD': 43500.0, 'ETH-USD': 2280.0}
        
        allocation = hybrid.process_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices=prices
        )
        
        # Verify results
        assert allocation.total_allocated_usd > 0
        assert hybrid.total_decisions == 1
        # Either RL or fallback worked
        assert hybrid.rl_decisions + hybrid.fallback_decisions == 1
        
        hybrid.stop()
    
    def test_fallback_when_rl_disabled(self):
        """Test that Phase 2 fallback works when RL disabled."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=False,
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
            expected_return=0.08
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        allocation = hybrid.process_signals(
            signals=[signal],
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        )
        
        assert allocation.total_allocated_usd > 0
        assert hybrid.fallback_decisions == 1
        assert hybrid.rl_decisions == 0
        
        hybrid.stop()
    
    def test_empty_signals_validation(self):
        """Test validation of empty signals list."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=False,
            enable_metrics=False
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        allocation = hybrid.process_signals(
            signals=[],  # Empty list
            cascade_indicators=cascade,
            current_prices={}
        )
        
        assert allocation.total_allocated_usd == 0.0
        assert allocation.allocations == {}
        
        hybrid.stop()
    
    def test_performance_tracking_integration(self):
        """Test performance comparison is tracked."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        # Record some outcomes
        for i in range(10):
            hybrid.record_outcome("BTC-USD", 0.02 + i * 0.005, used_rl=True)
            hybrid.record_outcome("BTC-USD", 0.015 + i * 0.003, used_rl=False)
        
        perf = hybrid.get_performance_comparison()
        
        assert perf['rl']['trades'] == 10
        assert perf['baseline']['trades'] == 10
        assert perf['rl']['mean_return'] > 0
        assert perf['baseline']['mean_return'] > 0
        
        hybrid.stop()
    
    def test_state_includes_all_components(self):
        """Test state retrieval includes all components."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        state = hybrid.get_state()
        
        assert 'portfolio_value' in state
        assert 'enable_rl' in state
        assert 'total_decisions' in state
        assert 'phase2' in state
        assert 'performance' in state
        assert 'rl_client' in state  # Because RL is enabled
        
        hybrid.stop()


class TestPhase3AsyncIntegration:
    """Async integration tests for Phase 3."""
    
    def test_async_processing_with_mock_rl(self):
        """Test async processing works with mock RL."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        signal = TacticalSignal(
            strategy_id="momentum",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.80,
            risk_score=0.25,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.10
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        # Run async method
        allocation = asyncio.run(hybrid.process_signals_async(
            signals=[signal],
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        ))
        
        assert allocation.total_allocated_usd > 0
        
        hybrid.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
