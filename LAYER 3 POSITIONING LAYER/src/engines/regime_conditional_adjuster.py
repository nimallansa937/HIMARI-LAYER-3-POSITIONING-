"""
HIMARI OPUS 2 - Tier 3: Regime Conditional Adjuster
====================================================

Applies regime-based position multipliers.
Implements Tier 3 per CLAUDE Guide Part VI.

Multipliers are hard-coded lookup values, not learned.
This ensures predictable behavior during regime transitions.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class RegimeConditionalAdjuster:
    """
    Tier 3: Apply regime-based position multipliers.
    
    Multipliers are hard-coded lookup values, not learned.
    This ensures predictable behavior during regime transitions.
    
    Regime Multipliers (per CLAUDE Guide):
    - NORMAL:   1.0  (Full position allowed)
    - HIGH_VOL: 0.7  (30% reduction for elevated volatility)
    - CRISIS:   0.3  (70% reduction during crisis)
    - CASCADE:  0.05 (Near-zero position during cascades)
    """
    
    REGIME_MULTIPLIERS = {
        'NORMAL':   1.0,
        'HIGH_VOL': 0.7,
        'CRISIS':   0.3,
        'CASCADE':  0.05
    }
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize regime conditional adjuster.
        
        Args:
            confidence_threshold: Below this, assume worse regime
        """
        self.confidence_threshold = confidence_threshold
    
    def compute_regime_adjustment(
        self,
        position: float,
        regime: str,
        regime_confidence: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply regime multiplier to position.
        
        If regime confidence is below threshold, assume worst-case
        regime between current and one step worse.
        
        Args:
            position: Position size from Tier 2
            regime: Current regime classification
            regime_confidence: Confidence in regime classification
            
        Returns:
            Tuple of (adjusted_position, diagnostics)
        """
        multiplier = self.REGIME_MULTIPLIERS.get(regime, 0.5)
        original_multiplier = multiplier
        
        # If low confidence, assume one step worse
        if regime_confidence < self.confidence_threshold:
            worse_regimes = {
                'NORMAL': 'HIGH_VOL',
                'HIGH_VOL': 'CRISIS',
                'CRISIS': 'CASCADE',
                'CASCADE': 'CASCADE'
            }
            worse_regime = worse_regimes.get(regime, 'CRISIS')
            worse_multiplier = self.REGIME_MULTIPLIERS[worse_regime]
            
            # Blend toward worse regime
            blend_factor = (self.confidence_threshold - regime_confidence) / self.confidence_threshold
            blend_factor = max(0.0, min(1.0, blend_factor))
            multiplier = multiplier * (1 - blend_factor) + worse_multiplier * blend_factor
        
        adjusted_position = position * multiplier
        
        diagnostics = {
            'regime': regime,
            'regime_confidence': regime_confidence,
            'confidence_threshold': self.confidence_threshold,
            'low_confidence': regime_confidence < self.confidence_threshold,
            'original_multiplier': original_multiplier,
            'regime_multiplier': multiplier,
            'input_position': position,
            'output_position': adjusted_position,
            'tier': 'REGIME_ADJUSTMENT'
        }
        
        if regime != 'NORMAL':
            logger.info(
                f"Regime adjustment: {regime} (conf={regime_confidence:.2f}) "
                f"→ multiplier={multiplier:.2f}"
            )
        
        return adjusted_position, diagnostics
    
    def get_multiplier_for_regime(self, regime: str) -> float:
        """Get the multiplier for a specific regime."""
        return self.REGIME_MULTIPLIERS.get(regime, 0.5)
