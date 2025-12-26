"""
Unit tests for Regime Conditional Adjuster
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from engines.regime_adjuster import RegimeConditionalAdjuster
from core.layer3_types import MarketRegime


class TestRegimeConditionalAdjuster:
    """Test suite for Regime Conditional Adjuster."""
    
    def test_initialization(self):
        """Test adjuster initialization."""
        adjuster = RegimeConditionalAdjuster(hysteresis_periods=3)
        
        assert adjuster.hysteresis_periods == 3
        assert adjuster.current_regime == MarketRegime.RANGING
        assert adjuster.confirmation_count == 0
        assert adjuster.false_flip_count == 0
    
    def test_same_regime_no_change(self):
        """Test that same regime doesn't trigger change."""
        adjuster = RegimeConditionalAdjuster()
        
        regime, diagnostics = adjuster.update_regime(
            detected_regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns()
        )
        
        assert regime == MarketRegime.RANGING
        assert adjuster.confirmation_count == 0
    
    def test_regime_candidate_detection(self):
        """Test new regime candidate detection."""
        adjuster = RegimeConditionalAdjuster()
        
        regime, diagnostics = adjuster.update_regime(
            detected_regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns()
        )
        
        assert regime == MarketRegime.RANGING  # Still old regime
        assert adjuster.candidate_regime == MarketRegime.TRENDING_UP
        assert adjuster.confirmation_count == 1
        assert diagnostics['new_candidate'] == True
    
    def test_regime_confirmation(self):
        """Test regime transition confirmation after 3 periods."""
        adjuster = RegimeConditionalAdjuster(hysteresis_periods=3)
        
        ts = time.time_ns()
        
        # Period 1: New candidate
        regime, _ = adjuster.update_regime(MarketRegime.TRENDING_UP, ts)
        assert regime == MarketRegime.RANGING
        
        # Period 2: Confirming
        regime, _ = adjuster.update_regime(MarketRegime.TRENDING_UP, ts + 1000000)
        assert regime == MarketRegime.RANGING
        
        # Period 3: Confirmed!
        regime, diagnostics = adjuster.update_regime(MarketRegime.TRENDING_UP, ts + 2000000)
        assert regime == MarketRegime.TRENDING_UP
        assert diagnostics.get('transition_confirmed') == True
    
    def test_false_flip_prevention(self):
        """Test that hysteresis prevents false flips."""
        adjuster = RegimeConditionalAdjuster(hysteresis_periods=3)
        
        ts = time.time_ns()
        
        # Candidate appears
        adjuster.update_regime(MarketRegime.TRENDING_UP, ts)
        
        # Candidate disappears (back to original)
        regime, _ = adjuster.update_regime(MarketRegime.RANGING, ts + 1000000)
        
        assert regime == MarketRegime.RANGING
        assert adjuster.false_flip_count == 1
        assert adjuster.candidate_regime is None
    
    def test_adjust_position_for_regime(self):
        """Test position adjustment for different regimes."""
        adjuster = RegimeConditionalAdjuster()
        
        # TRENDING_UP: 1.2x multiplier
        adjusted, multiplier, diagnostics = adjuster.adjust_position_for_regime(
            base_position_size=10000,
            regime=MarketRegime.TRENDING_UP
        )
        
        assert adjusted == 12000
        assert multiplier == 1.2
        assert diagnostics['increase_pct'] == pytest.approx(20.0)
        
        # CRISIS: 0.2x multiplier
        adjusted, multiplier, diagnostics = adjuster.adjust_position_for_regime(
            base_position_size=10000,
            regime=MarketRegime.CRISIS
        )
        
        assert adjusted == 2000
        assert multiplier == 0.2
        assert diagnostics['reduction_pct'] == pytest.approx(80.0)
    
    def test_get_state(self):
        """Test state retrieval."""
        adjuster = RegimeConditionalAdjuster()
        adjuster.update_regime(MarketRegime.TRENDING_UP, time.time_ns())
        
        state = adjuster.get_state()
        
        assert 'current_regime' in state
        assert 'candidate_regime' in state
        assert 'confirmation_progress' in state
        assert 'false_flip_count' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
