"""
HIMARI OPUS V2 - Regime Conditional Adjuster
=============================================

Regime-based position adjustment with hysteresis state machine and diagnostics.

Features:
- Regime multipliers for different market conditions
- Hysteresis state machine (3-period confirmation)
- Transition diagnostics and false flip tracking
- Complete transition history

Version: 3.1 Enhanced
"""

from typing import Tuple, Dict, Optional
from collections import deque
import logging

# Handle both module and script imports
try:
    from core.layer3_types import MarketRegime
except ImportError:
    from ..core.layer3_types import MarketRegime

logger = logging.getLogger(__name__)


class RegimeConditionalAdjuster:
    """
    Regime-based position adjuster with hysteresis diagnostics.
    
    Applies position multipliers based on market regime with hysteresis
    to prevent over-trading on regime flip noise.
    """
    
    def __init__(self, hysteresis_periods: int = 3):
        """
        Initialize regime adjuster.
        
        Args:
            hysteresis_periods: Number of periods to confirm regime change
        """
        self.hysteresis_periods = hysteresis_periods
        
        # Regime multipliers (validated in production)
        self.regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,      # 20% increase in trending up
            MarketRegime.TRENDING_DOWN: 0.8,    # 20% decrease in trending down
            MarketRegime.RANGING: 1.0,          # No change in ranging
            MarketRegime.HIGH_VOLATILITY: 0.6,  # 40% decrease in high vol
            MarketRegime.CRISIS: 0.2            # 80% decrease in crisis
        }
        
        # Hysteresis state
        self.current_regime = MarketRegime.RANGING
        self.candidate_regime: Optional[MarketRegime] = None
        self.confirmation_count = 0
        self.last_update_ns = 0
        
        # NEW: Diagnostic tracking
        self.regime_transition_history = deque(maxlen=10)  # Last 10 transitions
        self.false_flip_count = 0  # Candidate that didn't confirm
    
    def update_regime(
        self, 
        detected_regime: MarketRegime, 
        timestamp_ns: int
    ) -> Tuple[MarketRegime, Dict]:
        """
        Update regime with hysteresis diagnostics.
        
        Args:
            detected_regime: Regime detected by Layer 2
            timestamp_ns: Current timestamp in nanoseconds
            
        Returns:
            (confirmed_regime, diagnostics)
        """
        diagnostics = {
            'detected_regime': detected_regime.value,
            'current_regime': self.current_regime.value,
            'candidate_regime': self.candidate_regime.value if self.candidate_regime else None,
            'confirmation_count': self.confirmation_count,
            'confirmation_progress': f"{self.confirmation_count}/{self.hysteresis_periods}",
            'false_flip_count': self.false_flip_count
        }
        
        if detected_regime == self.current_regime:
            # Same regime: reset candidate
            if self.candidate_regime is not None:
                self.false_flip_count += 1  # Candidate didn't confirm
                logger.debug(
                    f"Regime candidate {self.candidate_regime.value} abandoned. "
                    f"Total false flips: {self.false_flip_count}"
                )
            self.candidate_regime = None
            self.confirmation_count = 0
            self.last_update_ns = timestamp_ns
            return self.current_regime, diagnostics
        
        if detected_regime == self.candidate_regime:
            # Same candidate: increment confirmation
            self.confirmation_count += 1
            diagnostics['confirmation_count'] = self.confirmation_count
            
            if self.confirmation_count >= self.hysteresis_periods:
                # Confirmed transition
                old_regime = self.current_regime
                self.current_regime = detected_regime
                self.candidate_regime = None
                self.confirmation_count = 0
                
                # Log transition
                transition = {
                    'timestamp_ns': timestamp_ns,
                    'from': old_regime.value,
                    'to': detected_regime.value
                }
                self.regime_transition_history.append(transition)
                
                diagnostics['transition_confirmed'] = True
                
                logger.info(
                    f"Regime transition confirmed: {old_regime.value} → {detected_regime.value}"
                )
        else:
            # New candidate: reset counter
            if self.candidate_regime is not None:
                self.false_flip_count += 1  # Previous candidate abandoned
            self.candidate_regime = detected_regime
            self.confirmation_count = 1
            diagnostics['new_candidate'] = True
            
            logger.debug(
                f"New regime candidate: {detected_regime.value} "
                f"(was: {self.current_regime.value})"
            )
        
        self.last_update_ns = timestamp_ns
        return self.current_regime, diagnostics
    
    def adjust_position_for_regime(
        self,
        base_position_size: float,
        regime: Optional[MarketRegime] = None
    ) -> Tuple[float, float, Dict]:
        """
        Apply regime multiplier with diagnostics.
        
        Args:
            base_position_size: Position size before regime adjustment
            regime: Regime to use (defaults to current confirmed regime)
            
        Returns:
            (adjusted_size, regime_multiplier, diagnostics)
        """
        if regime is None:
            regime = self.current_regime
        
        multiplier = self.regime_multipliers.get(regime, 1.0)
        adjusted_size = base_position_size * multiplier
        
        diagnostics = {
            'regime': regime.value,
            'multiplier': multiplier,
            'base_size': base_position_size,
            'adjusted_size': adjusted_size,
            'reduction_pct': (1.0 - multiplier) * 100 if multiplier < 1.0 else 0.0,
            'increase_pct': (multiplier - 1.0) * 100 if multiplier > 1.0 else 0.0
        }
        
        return adjusted_size, multiplier, diagnostics
    
    def get_transition_history(self) -> list:
        """Get regime transition history."""
        return list(self.regime_transition_history)
    
    def get_state(self) -> Dict:
        """Get current state for monitoring."""
        return {
            'current_regime': self.current_regime.value,
            'candidate_regime': self.candidate_regime.value if self.candidate_regime else None,
            'confirmation_count': self.confirmation_count,
            'confirmation_progress': f"{self.confirmation_count}/{self.hysteresis_periods}",
            'false_flip_count': self.false_flip_count,
            'transition_history_size': len(self.regime_transition_history),
            'current_multiplier': self.regime_multipliers[self.current_regime]
        }
    
    def update_multipliers(self, multipliers: Dict[MarketRegime, float]):
        """
        Update regime multipliers (for recalibration).
        
        Args:
            multipliers: Dictionary mapping regime to multiplier
        """
        self.regime_multipliers.update(multipliers)
        logger.info(f"Regime multipliers updated: {self.regime_multipliers}")

    def reset_hysteresis(self):
        """Reset hysteresis state (for debugging)."""
        self.candidate_regime = None
        self.confirmation_count = 0
        self.false_flip_count = 0
        logger.info("Regime hysteresis reset")
