"""
HIMARI OPUS 2 - Tier 2: Funding Rate Signal
============================================

Reduces position size when funding rates are extreme.
Part of Tier 2 Bounded Adaptive Enhancement per CLAUDE Guide Part V.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FundingRateSignal:
    """
    Reduce position size when funding rates are extreme.
    
    High funding rates (>0.03%) indicate crowded positioning and
    elevated liquidation risk. This is a deterministic signal
    that reduces the adaptive component's output.
    
    Sharpe contribution: +0.03 to +0.05
    """
    
    def __init__(
        self,
        threshold_reduce: float = 0.0003,   # 0.03%
        threshold_exit: float = 0.001,       # 0.1%
        reduction_factor: float = 0.5        # Reduce by 50%
    ):
        """
        Initialize funding rate signal.
        
        Args:
            threshold_reduce: Funding rate threshold to trigger reduction
            threshold_exit: Funding rate threshold to trigger exit
            reduction_factor: Multiplier when threshold_reduce is breached
        """
        self.threshold_reduce = threshold_reduce
        self.threshold_exit = threshold_exit
        self.reduction_factor = reduction_factor
    
    def compute_adjustment(
        self,
        funding_rate: float,
        direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute position adjustment based on funding rate.
        
        Args:
            funding_rate: Current 8-hour funding rate (decimal)
            direction: Position direction (LONG/SHORT)
            
        Returns:
            Tuple of (multiplier, diagnostics)
            - multiplier: Factor to apply to position (0.0 to 1.0)
            - diagnostics: Computation details
        """
        abs_funding = abs(funding_rate)
        
        # Check for extreme funding
        if abs_funding >= self.threshold_exit:
            # Very extreme: exit signal
            multiplier = 0.0
            reason = f'Extreme funding: {funding_rate:.4%}'
        elif abs_funding >= self.threshold_reduce:
            # Elevated: reduce position
            multiplier = self.reduction_factor
            reason = f'High funding: {funding_rate:.4%}'
        else:
            # Normal: no adjustment
            multiplier = 1.0
            reason = 'Funding normal'
        
        # Direction-aware: only reduce if funding works against us
        # Long + positive funding = we pay, risk of long squeeze
        # Short + negative funding = we pay, risk of short squeeze
        funding_against_us = (
            (direction == 'LONG' and funding_rate > 0) or
            (direction == 'SHORT' and funding_rate < 0)
        )
        
        if not funding_against_us and multiplier < 1.0:
            multiplier = 1.0  # Funding in our favor, no reduction
            reason = 'Funding in our favor'
        
        diagnostics = {
            'funding_rate': funding_rate,
            'funding_rate_pct': f'{funding_rate:.4%}',
            'direction': direction,
            'funding_against_us': funding_against_us,
            'funding_multiplier': multiplier,
            'funding_reason': reason,
            'threshold_reduce': self.threshold_reduce,
            'threshold_exit': self.threshold_exit,
            'tier': 'FUNDING_SIGNAL'
        }
        
        if multiplier < 1.0:
            logger.info(f"Funding signal: {reason}, multiplier={multiplier}")
        
        return multiplier, diagnostics
