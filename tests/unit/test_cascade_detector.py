"""
Unit tests for Enhanced Cascade Detector
"""

import pytest
import sys
sys.path.insert(0, 'src')

from risk.cascade_detector_v2 import EnhancedCascadeDetector
from core.layer3_types import CascadeIndicators


class TestEnhancedCascadeDetector:
    """Test suite for Enhanced Cascade Detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = EnhancedCascadeDetector()
        
        assert detector.funding_rate_threshold == 0.003
        assert detector.whale_pressure_threshold == 0.7
        assert sum(detector.weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_low_risk_scenario(self):
        """Test low risk scenario returns MONITOR."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        risk_score, recommendation, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert risk_score < 0.4
        assert recommendation == "MONITOR"
        assert 'components' in diagnostics
    
    def test_high_risk_scenario(self):
        """Test high risk scenario triggers reduction."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.008,  # Extreme
            oi_change_pct=-0.15,  # Collapsing
            volume_ratio=8.0,  # Liquidations
            onchain_whale_pressure=0.85,  # Extreme
            exchange_netflow_zscore=3.5  # Anomaly
        )
        
        risk_score, recommendation, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert risk_score > 0.6
        assert recommendation in ["REDUCE_75%", "EXIT"]
        assert len(diagnostics['threshold_breaches']) > 0
    
    def test_funding_rate_component(self):
        """Test funding rate component."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.01,  # 1% - very high
            oi_change_pct=0.0,
            volume_ratio=1.0,
            onchain_whale_pressure=0.0,
            exchange_netflow_zscore=0.0
        )
        
        _, _, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert diagnostics['components']['funding_risk'] > 0
    
    def test_whale_pressure_component(self):
        """Test on-chain whale pressure component."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.0,
            oi_change_pct=0.0,
            volume_ratio=1.0,
            onchain_whale_pressure=0.9,  # High whale activity
            exchange_netflow_zscore=0.0
        )
        
        _, _, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert diagnostics['components']['whale_risk'] > 0.7
    
    def test_netflow_component(self):
        """Test exchange netflow component."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.0,
            oi_change_pct=0.0,
            volume_ratio=1.0,
            onchain_whale_pressure=0.0,
            exchange_netflow_zscore=4.0  # 4-sigma event
        )
        
        _, _, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert diagnostics['components']['netflow_risk'] > 0.5
    
    def test_dominant_factor_identification(self):
        """Test dominant factor identification."""
        detector = EnhancedCascadeDetector()
        
        indicators = CascadeIndicators(
            funding_rate=0.0,
            oi_change_pct=-0.25,  # Massive OI drop
            volume_ratio=1.0,
            onchain_whale_pressure=0.0,
            exchange_netflow_zscore=0.0
        )
        
        _, _, diagnostics = detector.calculate_cascade_risk(indicators)
        
        assert diagnostics['dominant_factor'] == 'oi_risk'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
