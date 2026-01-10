"""
Unit tests for Phase 3 Hybrid Orchestrator
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from phases.phase3_hybrid import Layer3Phase3Hybrid
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)


class TestLayer3Phase3Hybrid:
    """Test suite for Phase 3 Hybrid Orchestrator."""
    
    def test_initialization_no_rl(self):
        """Test initialization without RL."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=False,
            enable_metrics=False
        )
        
        assert hybrid.portfolio_value == 100000
        assert hybrid.enable_rl == False
        assert hybrid.rl_client is None
        
        hybrid.stop()
    
    def test_initialization_with_mock_rl(self):
        """Test initialization with mock RL."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        assert hybrid.enable_rl == True
        assert hybrid.rl_client is not None
        
        hybrid.stop()
    
    def test_process_without_rl(self):
        """Test processing signals without RL (fallback only)."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=False,
            enable_metrics=False
        )
        
        signals = [
            TacticalSignal(
                strategy_id="momentum",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.75,
                risk_score=0.3,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.08
            )
        ]
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        allocation = hybrid.process_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        )
        
        assert allocation.total_allocated_usd > 0
        assert hybrid.fallback_decisions == 1
        
        hybrid.stop()
    
    def test_process_with_mock_rl(self):
        """Test processing with mock RL."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        signals = [
            TacticalSignal(
                strategy_id="momentum",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.75,
                risk_score=0.3,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.08
            )
        ]
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        allocation = hybrid.process_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        )
        
        assert allocation.total_allocated_usd > 0
        # Either RL or fallback worked
        assert hybrid.total_decisions == 1
        
        hybrid.stop()
    
    def test_performance_tracking(self):
        """Test performance comparison tracking."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=False,
            enable_metrics=False
        )
        
        # Record some outcomes
        hybrid.record_outcome("BTC-USD", 0.05, used_rl=False)
        hybrid.record_outcome("BTC-USD", 0.03, used_rl=False)
        hybrid.record_outcome("ETH-USD", 0.08, used_rl=True)
        
        perf = hybrid.get_performance_comparison()
        
        assert perf['baseline']['trades'] == 2
        assert perf['rl']['trades'] == 1
        
        hybrid.stop()
    
    def test_get_state(self):
        """Test state retrieval."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        state = hybrid.get_state()
        
        assert 'portfolio_value' in state
        assert 'enable_rl' in state
        assert 'phase2' in state
        assert 'performance' in state
        assert 'rl_client' in state
        
        hybrid.stop()
    
    def test_empty_signals_validation(self):
        """Test validation of empty signals list (Issue #13)."""
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
    
    def test_rl_blending_logic(self):
        """Test RL weight blending logic (Issue #8)."""
        hybrid = Layer3Phase3Hybrid(
            portfolio_value=100000,
            enable_rl=True,
            use_mock_rl=True,
            enable_metrics=False
        )
        
        # Run multiple times with mock RL (random success)
        total_runs = 5
        for _ in range(total_runs):
            signals = [
                TacticalSignal(
                    strategy_id="momentum",
                    symbol="BTC-USD",
                    action=TacticalAction.BUY,
                    confidence=0.80,  # High confidence
                    risk_score=0.25,
                    regime=MarketRegime.TRENDING_UP,
                    timestamp_ns=time.time_ns(),
                    expected_return=0.10
                )
            ]
            
            cascade = CascadeIndicators(
                funding_rate=0.001,
                oi_change_pct=0.02,
                volume_ratio=1.5,
                onchain_whale_pressure=0.2,
                exchange_netflow_zscore=0.3
            )
            
            allocation = hybrid.process_signals(
                signals=signals,
                cascade_indicators=cascade,
                current_prices={'BTC-USD': 42000.0}
            )
            
            # Should produce valid allocation
            assert allocation.total_allocated_usd >= 0
        
        # Verify decisions tracked
        assert hybrid.total_decisions == total_runs
        assert hybrid.rl_decisions + hybrid.fallback_decisions == total_runs
        
        hybrid.stop()


class TestPhase3Async:
    """Async tests for Phase 3 (Issue #5)."""
    
    def test_async_processing_with_mock_rl(self):
        """Test async processing works with mock RL."""
        import asyncio
        
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
    
    def test_async_fallback(self):
        """Test async fallback when RL disabled."""
        import asyncio
        
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
            timestamp_ns=time.time_ns()
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        allocation = asyncio.run(hybrid.process_signals_async(
            signals=[signal],
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        ))
        
        assert allocation.total_allocated_usd > 0
        assert hybrid.fallback_decisions == 1
        
        hybrid.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
