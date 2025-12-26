"""
HIMARI OPUS V2 - Layer 1 Signal Mapper
=======================================

Maps Layer 1 antigravity signals to cascade risk indicators.

Signal Mapping:
- FSI (Funding Saturation Index) → funding_rate
- LEI (Liquidity Evaporation Index) → oi_change_pct
- SCSI (Stablecoin Stress Index) → volume_ratio
- LCI (Leverage Concentration Index) → onchain_whale_pressure
- CACI (Cross-Asset Contagion Index) → exchange_netflow_zscore

Version: 3.1 Enhanced
"""

from typing import Any, Dict
import logging

# Handle both module and script imports
try:
    from core.layer3_types import CascadeIndicators
except ImportError:
    from ..core.layer3_types import CascadeIndicators

logger = logging.getLogger(__name__)


class L1SignalMapper:
    """Maps Layer 1 antigravity signals to Layer 3 cascade indicators."""
    
    def __init__(self):
        """Initialize signal mapper with scaling factors."""
        # Scaling factors calibrated for cascade detection
        self.fsi_scale = 0.01  # FSI [0,1] → funding rate scale
        self.lei_scale = 0.2   # LEI [0,1] → OI change % scale
        self.scsi_scale = 10.0 # SCSI [0,1] → volume ratio scale
    
    def map_to_cascade_indicators(self, signal_feed: Dict[str, Any]) -> CascadeIndicators:
        """
        Map Layer 1 antigravity signals to cascade risk indicators.
        
        Args:
            signal_feed: Signal feed dictionary from Layer 1
                Expected structure:
                {
                    'antigravity': {
                        'coherence': float,  # FSI proxy
                        'entropy': float,    # LEI proxy
                        'energy_density': float,  # SCSI proxy
                        'schwarzschild_radius': float,  # LCI
                        'hawking_temperature': float    # CACI
                    }
                }
        
        Returns:
            CascadeIndicators with mapped values
        """
        try:
            antigravity = signal_feed.get('antigravity', {})
            
            # FSI → funding_rate proxy
            fsi = antigravity.get('coherence', 0.5)
            funding_rate = fsi * self.fsi_scale
            
            # LEI → oi_change proxy (inverted, centered at 0.5)
            lei = antigravity.get('entropy', 0.5)
            oi_change_pct = (lei - 0.5) * -self.lei_scale
            
            # SCSI → volume_ratio proxy
            scsi = antigravity.get('energy_density', 0.1)
            volume_ratio = scsi * self.scsi_scale
            
            # LCI → whale_pressure (direct mapping)
            whale_pressure = antigravity.get('schwarzschild_radius', 0.0)
            
            # CACI → netflow_zscore (centered at 0.5, scaled to ±4 sigma)
            caci = antigravity.get('hawking_temperature', 0.5)
            netflow_zscore = (caci - 0.5) * 4.0
            
            return CascadeIndicators(
                funding_rate=funding_rate,
                oi_change_pct=oi_change_pct,
                volume_ratio=volume_ratio,
                onchain_whale_pressure=whale_pressure,
                exchange_netflow_zscore=netflow_zscore
            )
        
        except Exception as e:
            logger.error(f"Error mapping L1 signals to cascade indicators: {e}")
            # Return safe defaults
            return CascadeIndicators(
                funding_rate=0.0,
                oi_change_pct=0.0,
                volume_ratio=1.0,
                onchain_whale_pressure=0.0,
                exchange_netflow_zscore=0.0
            )
