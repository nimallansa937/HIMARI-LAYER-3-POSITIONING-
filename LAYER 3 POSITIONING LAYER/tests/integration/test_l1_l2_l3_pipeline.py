"""
Integration test for L1→L2→L3 pipeline
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from phases.phase1_core import Layer3Phase1
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from integration.l1_signal_mapper import L1SignalMapper


class TestL1L2L3Pipeline:
    """Integration tests for complete L1→L2→L3 pipeline."""
    
    def test_end_to_end_normal_scenario(self):
        """Test complete pipeline with normal market conditions."""
        # Initialize Phase 1
        layer3 = Layer3Phase1(portfolio_value=100000, kelly_fraction=0.25)
        
        # Simulate Layer 2 tactical signal
        signal = TacticalSignal(
            strategy_id="momentum_btc",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.08,
            predicted_volatility=0.03
        )
        
        # Simulate Layer 1 cascade indicators (normal conditions)
        cascade_indicators = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        # Execute pipeline
        decision = layer3.calculate_position(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=42000.0
        )
        
        # Validate decision
        assert decision.position_size_usd > 0
        assert decision.position_size_usd <= 100000
        assert decision.symbol == "BTC-USD"
        assert decision.strategy_id == "momentum_btc"
        assert decision.cascade_recommendation == "MONITOR"
        # Note: Due to hysteresis (3-period confirmation), first signal keeps default RANGING
        assert decision.current_regime == MarketRegime.RANGING
    
    def test_high_risk_cascade_reduction(self):
        """Test that high cascade risk reduces position."""
        layer3 = Layer3Phase1(portfolio_value=100000)
        
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.70,
            risk_score=0.4,
            regime=MarketRegime.HIGH_VOLATILITY,
            timestamp_ns=time.time_ns(),
            expected_return=0.06,
            predicted_volatility=0.05
        )
        
        # HIGH RISK cascade indicators
        cascade_indicators = CascadeIndicators(
            funding_rate=0.008,
            oi_change_pct=-0.15,
            volume_ratio=8.0,
            onchain_whale_pressure=0.85,
            exchange_netflow_zscore=3.5
        )
        
        decision = layer3.calculate_position(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=42000.0
        )
        
        # Position should be significantly reduced
        assert decision.cascade_risk_score > 0.6
        assert decision.cascade_recommendation in ["REDUCE_75%", "EXIT"]
        assert decision.cascade_adjusted_usd < decision.regime_adjusted_usd
    
    def test_regime_transition_hysteresis(self):
        """Test regime hysteresis over multiple signals."""
        layer3 = Layer3Phase1(portfolio_value=100000)
        
        # Signal 1: TRENDING_UP detected
        signal1 = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.7,
            risk_score=0.3,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        cascade_indicators = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        decision1 = layer3.calculate_position(signal1, cascade_indicators, 42000.0)
        
        # Regime should still be RANGING (not confirmed yet)
        assert decision1.current_regime == MarketRegime.RANGING
        
        # Signal 2: Back to RANGING (false flip)
        signal2 = signal1
        signal2.regime = MarketRegime.RANGING
        signal2.timestamp_ns = time.time_ns() + 1000000
        
        decision2 = layer3.calculate_position(signal2, cascade_indicators, 42000.0)
        
        # Should prevent false flip
        assert decision2.current_regime == MarketRegime.RANGING
        assert layer3.regime_adjuster.false_flip_count >= 1
    
    def test_l1_signal_mapper_integration(self):
        """Test L1 signal mapper produces valid cascade indicators."""
        mapper = L1SignalMapper()
        
        # Simulate Layer 1 antigravity signals
        l1_signal_feed = {
            'antigravity': {
                'coherence': 0.6,  # FSI
                'entropy': 0.55,  # LEI
                'energy_density': 0.3,  # SCSI
                'schwarzschild_radius': 0.4,  # LCI
                'hawking_temperature': 0.6  # CACI
            }
        }
        
        cascade_indicators = mapper.map_to_cascade_indicators(l1_signal_feed)
        
        assert isinstance(cascade_indicators, CascadeIndicators)
        assert cascade_indicators.onchain_whale_pressure == 0.4
        assert -5.0 <= cascade_indicators.exchange_netflow_zscore <= 5.0
    
    def test_trade_result_feedback_loop(self):
        """Test that trade results update components."""
        layer3 = Layer3Phase1(portfolio_value=100000)
        
        # Make initial decision
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.7,
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns(),
            expected_return=0.05
        )
        
        cascade_indicators = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        decision = layer3.calculate_position(signal, cascade_indicators, 42000.0)
        
        # Update with trade result
        layer3.update_from_trade_result(
            predicted_return=0.05,
            actual_return=0.06,
            trade_won=True
        )
        
        # Verify components were updated
        assert layer3.bayesian_kelly.total_trades == 1
        assert layer3.bayesian_kelly.winning_trades == 1
        assert len(layer3.conformal_scaler.residuals) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
