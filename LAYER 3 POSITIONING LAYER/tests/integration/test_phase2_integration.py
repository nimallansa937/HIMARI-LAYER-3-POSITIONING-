"""
Integration tests for Phase 2 Portfolio
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from phases.phase2_portfolio import Layer3Phase2Portfolio
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)


class TestPhase2Integration:
    """Integration tests for Phase 2 pipeline."""
    
    def test_full_pipeline_integration(self):
        """Test complete Phase 1 → Phase 2 pipeline."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            max_position_pct=0.15,
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
        
        allocation = portfolio.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 43500.0, 'ETH-USD': 2280.0}
        )
        
        # Verify results
        assert allocation.total_allocated_usd > 0
        assert len(allocation.allocations) == 2
        assert 'BTC-USD' in allocation.allocations
        assert 'ETH-USD' in allocation.allocations
        assert allocation.utilization_pct > 0
        
        portfolio.stop()
    
    def test_hierarchical_risk_budget_integration(self):
        """Test that risk budgets are enforced in pipeline."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            max_position_pct=0.10,  # 10% max
            max_strategy_pct=0.20,  # 20% max per strategy
            max_portfolio_pct=0.50,  # 50% max total
            enable_metrics=False
        )
        
        # Create many signals to exceed limits
        signals = [
            TacticalSignal(
                strategy_id=f"strat_{i}",
                symbol=f"ASSET{i}-USD",
                action=TacticalAction.BUY,
                confidence=0.80,
                risk_score=0.20,
                regime=MarketRegime.TRENDING_UP,
                timestamp_ns=time.time_ns(),
                expected_return=0.15,
                predicted_volatility=0.03
            )
            for i in range(10)
        ]
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        prices = {f"ASSET{i}-USD": 100.0 for i in range(10)}
        
        allocation = portfolio.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices=prices
        )
        
        # Verify portfolio limit enforced (max 50% = $50,000)
        assert allocation.total_allocated_usd <= 50000 + 100  # Small margin
        
        # Verify per-position limit (max 10% = $10,000)
        for usd in allocation.allocations.values():
            assert usd <= 10000 + 100  # Small margin
        
        portfolio.stop()
    
    def test_correlation_monitoring_integration(self):
        """Test correlation updates flow through."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        # Add return samples with varied values (need 20+ for correlation)
        import numpy as np
        np.random.seed(42)
        for i in range(25):
            diag = portfolio.update_correlations({
                'BTC-USD': 0.01 * np.random.randn(),
                'ETH-USD': 0.01 * np.random.randn() + 0.002
            })
        
        state = portfolio.correlation_monitor.get_state()
        
        assert state['n_symbols'] == 2
        assert 'diversification_score' in state
        
        portfolio.stop()
    
    def test_weight_drift_tracking_integration(self):
        """Test weight drift is tracked across allocations."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3
        )
        
        # First allocation
        signals1 = [
            TacticalSignal(
                strategy_id="strat_a",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.80,
                risk_score=0.25,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.10
            )
        ]
        
        portfolio.process_multi_strategy_signals(
            signals=signals1,
            cascade_indicators=cascade,
            current_prices={'BTC-USD': 42000.0}
        )
        
        # Verify weight history tracked
        assert len(portfolio.ensemble.weight_history) == 1
        
        portfolio.stop()
    
    def test_state_includes_all_components(self):
        """Test state includes all Phase 2 components."""
        portfolio = Layer3Phase2Portfolio(
            portfolio_value=100000,
            enable_metrics=False
        )
        
        state = portfolio.get_state()
        
        assert 'phase1' in state
        assert 'kelly_allocator' in state
        assert 'ensemble' in state
        assert 'correlation_monitor' in state
        assert 'risk_budget' in state
        
        portfolio.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
