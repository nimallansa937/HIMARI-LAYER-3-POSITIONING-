"""
Unit tests for L1 Signal Mapper
"""

import pytest
import sys
sys.path.insert(0, 'src')

from integration.l1_signal_mapper import L1SignalMapper
from core.layer3_types import CascadeIndicators


class TestL1SignalMapper:
    """Test suite for L1 Signal Mapper."""
    
    def test_initialization(self):
        """Test mapper initialization."""
        mapper = L1SignalMapper()
        
        assert mapper.fsi_scale == 0.01
        assert mapper.lei_scale == 0.2
        assert mapper.scsi_scale == 10.0
    
    def test_map_valid_signals(self):
        """Test mapping valid L1 signals."""
        mapper = L1SignalMapper()
        
        signal_feed = {
            'antigravity': {
                'coherence': 0.5,
                'entropy': 0.5,
                'energy_density': 0.2,
                'schwarzschild_radius': 0.6,
                'hawking_temperature': 0.7
            }
        }
        
        indicators = mapper.map_to_cascade_indicators(signal_feed)
        
        assert isinstance(indicators, CascadeIndicators)
        assert indicators.onchain_whale_pressure == 0.6
        # CACI 0.7 -> (0.7 - 0.5) * 4 = 0.8
        assert indicators.exchange_netflow_zscore == pytest.approx(0.8)
    
    def test_map_empty_signals(self):
        """Test mapping with empty/missing signals."""
        mapper = L1SignalMapper()
        
        indicators = mapper.map_to_cascade_indicators({})
        
        # Should return safe defaults
        assert indicators.funding_rate == pytest.approx(0.005)  # 0.5 * 0.01
        assert indicators.onchain_whale_pressure == 0.0
    
    def test_whale_pressure_direct_mapping(self):
        """Test whale pressure is directly mapped."""
        mapper = L1SignalMapper()
        
        signal_feed = {
            'antigravity': {
                'schwarzschild_radius': 0.85
            }
        }
        
        indicators = mapper.map_to_cascade_indicators(signal_feed)
        
        assert indicators.onchain_whale_pressure == 0.85
    
    def test_netflow_zscore_range(self):
        """Test netflow Z-score is within expected range."""
        mapper = L1SignalMapper()
        
        # Test extremes
        low_signal = {'antigravity': {'hawking_temperature': 0.0}}
        high_signal = {'antigravity': {'hawking_temperature': 1.0}}
        
        low_indicators = mapper.map_to_cascade_indicators(low_signal)
        high_indicators = mapper.map_to_cascade_indicators(high_signal)
        
        # Range should be roughly -2 to +2 (4 * ±0.5)
        assert low_indicators.exchange_netflow_zscore == pytest.approx(-2.0)
        assert high_indicators.exchange_netflow_zscore == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
