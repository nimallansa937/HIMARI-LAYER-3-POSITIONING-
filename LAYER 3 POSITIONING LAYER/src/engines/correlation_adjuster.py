"""
HIMARI OPUS 2 - Tier 2: Correlation Adjuster
=============================================

Reduces position when cross-asset correlations spike.
Part of Tier 2 Bounded Adaptive Enhancement per CLAUDE Guide Part V.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CorrelationAdjuster:
    """
    Reduce position when cross-asset correlations spike.
    
    High correlation indicates regime stress and reduced
    diversification benefit. Treat highly correlated positions
    as a single concentrated bet.
    
    Sharpe contribution: +0.02 to +0.04
    """
    
    def __init__(
        self,
        threshold_elevated: float = 0.85,
        threshold_extreme: float = 0.95,
        reduction_elevated: float = 0.7,
        reduction_extreme: float = 0.4
    ):
        """
        Initialize correlation adjuster.
        
        Args:
            threshold_elevated: Correlation threshold for elevated warning
            threshold_extreme: Correlation threshold for extreme warning
            reduction_elevated: Multiplier when elevated (0.7 = 30% reduction)
            reduction_extreme: Multiplier when extreme (0.4 = 60% reduction)
        """
        self.threshold_elevated = threshold_elevated
        self.threshold_extreme = threshold_extreme
        self.reduction_elevated = reduction_elevated
        self.reduction_extreme = reduction_extreme
    
    def compute_adjustment(
        self,
        btc_correlation: float,
        symbol: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute position adjustment based on BTC correlation.
        
        Args:
            btc_correlation: Rolling correlation with BTC
            symbol: Current symbol (BTC itself is exempt)
            
        Returns:
            Tuple of (multiplier, diagnostics)
        """
        # BTC itself is exempt
        if 'BTC' in symbol.upper():
            return 1.0, {
                'correlation': btc_correlation,
                'correlation_multiplier': 1.0,
                'correlation_reason': 'BTC exempt',
                'tier': 'CORRELATION_MONITOR'
            }
        
        # Determine multiplier based on correlation level
        if btc_correlation >= self.threshold_extreme:
            multiplier = self.reduction_extreme
            reason = f'Extreme correlation: {btc_correlation:.2f}'
            level = 'EXTREME'
        elif btc_correlation >= self.threshold_elevated:
            multiplier = self.reduction_elevated
            reason = f'Elevated correlation: {btc_correlation:.2f}'
            level = 'ELEVATED'
        else:
            multiplier = 1.0
            reason = 'Normal correlation'
            level = 'NORMAL'
        
        diagnostics = {
            'symbol': symbol,
            'correlation': btc_correlation,
            'correlation_level': level,
            'correlation_multiplier': multiplier,
            'correlation_reason': reason,
            'threshold_elevated': self.threshold_elevated,
            'threshold_extreme': self.threshold_extreme,
            'tier': 'CORRELATION_MONITOR'
        }
        
        if multiplier < 1.0:
            logger.info(f"Correlation adjustment for {symbol}: {reason}")
        
        return multiplier, diagnostics
