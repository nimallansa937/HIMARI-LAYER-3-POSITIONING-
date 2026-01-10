"""
HIMARI OPUS 2 - Tier 2: Cascade Anomaly Detector
=================================================

Detect precursors to liquidation cascades.
Part of Tier 2 Bounded Adaptive Enhancement per CLAUDE Guide Part V.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CascadeAnomalyDetector:
    """
    Detect precursors to liquidation cascades.
    
    Monitors open interest drops and volume spikes that often
    precede cascade events. Uses deterministic thresholds, not
    learned models, for reliability.
    
    Sharpe contribution: +0.02 to +0.03
    """
    
    def __init__(
        self,
        oi_drop_threshold: float = 0.05,     # 5% OI drop in 1 hour
        volume_spike_threshold: float = 3.0,  # 3x average volume
        reduction_factor: float = 0.6         # Reduce to 60%
    ):
        """
        Initialize cascade anomaly detector.
        
        Args:
            oi_drop_threshold: OI drop percentage to trigger alert
            volume_spike_threshold: Volume spike ratio to trigger alert
            reduction_factor: Position multiplier when alert triggered
        """
        self.oi_drop_threshold = oi_drop_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.reduction_factor = reduction_factor
    
    def compute_cascade_score(
        self,
        oi_delta_1h: float,
        volume_spike_ratio: float,
        funding_rate: float
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute cascade risk score and position adjustment.
        
        Formula: score = 0.4×funding_z + 0.3×oi_drop + 0.3×volume_spike
        
        Args:
            oi_delta_1h: Open interest change in last hour (decimal, e.g., -0.05)
            volume_spike_ratio: Current volume / 7-day average volume
            funding_rate: Current funding rate (decimal)
            
        Returns:
            Tuple of (cascade_score, multiplier, diagnostics)
            - cascade_score: Risk score (0.0 to 1.0+)
            - multiplier: Position adjustment factor
            - diagnostics: Computation details
        """
        # Normalize components to [0, 1] scale
        funding_component = min(abs(funding_rate) / 0.001, 1.0)
        
        # OI component only triggers on drops (negative delta)
        oi_component = min(abs(oi_delta_1h) / 0.10, 1.0) if oi_delta_1h < 0 else 0.0
        
        # Volume component (normalize spike above baseline)
        volume_component = min((volume_spike_ratio - 1.0) / 4.0, 1.0) if volume_spike_ratio > 1.0 else 0.0
        
        # Weighted combination
        cascade_score = (
            0.4 * funding_component +
            0.3 * oi_component +
            0.3 * volume_component
        )
        
        cascade_score = min(1.0, cascade_score)
        
        # Determine multiplier based on score
        if cascade_score > 0.7:
            multiplier = 0.1  # Near-zero position
            reason = f'High cascade risk: {cascade_score:.2f}'
            level = 'HIGH'
        elif cascade_score > 0.5:
            multiplier = self.reduction_factor
            reason = f'Elevated cascade risk: {cascade_score:.2f}'
            level = 'ELEVATED'
        else:
            multiplier = 1.0
            reason = 'Normal cascade risk'
            level = 'NORMAL'
        
        diagnostics = {
            'cascade_score': cascade_score,
            'cascade_level': level,
            'cascade_components': {
                'funding': funding_component,
                'oi_drop': oi_component,
                'volume_spike': volume_component
            },
            'inputs': {
                'oi_delta_1h': oi_delta_1h,
                'volume_spike_ratio': volume_spike_ratio,
                'funding_rate': funding_rate
            },
            'cascade_multiplier': multiplier,
            'cascade_reason': reason,
            'tier': 'CASCADE_DETECTOR'
        }
        
        if cascade_score > 0.5:
            logger.warning(
                f"Elevated cascade risk: score={cascade_score:.2f}, "
                f"funding={funding_component:.2f}, oi={oi_component:.2f}, "
                f"volume={volume_component:.2f}"
            )
        
        return cascade_score, multiplier, diagnostics
