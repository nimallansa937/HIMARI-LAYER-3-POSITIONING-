"""
L3-3: Adaptive Interval Sizing
==============================
Scale position size inversely with prediction uncertainty, using conformal
prediction interval width from L1-4 LSTM-CP.

Core Formula:
    BASE_SIZE = Kelly-optimal or constraint-limited size
    INTERVAL_WIDTH = (upper_bound - lower_bound) / point_prediction
    UNCERTAINTY_PENALTY = 1 / (1 + INTERVAL_WIDTH × SENSITIVITY)
    ADJUSTED_SIZE = BASE_SIZE × UNCERTAINTY_PENALTY

Intuition:
- Narrow interval (high confidence) → penalty ≈ 1 → full size
- Wide interval (low confidence) → penalty < 1 → reduced size
- Very wide interval → penalty → 0 → no trade

File: LAYER 3 V1/LAYER 3 POSITIONING LAYER/src/core/adaptive_interval_sizing.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import math


class RegimeType(Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class PredictionInterval:
    """Prediction interval from L1-4 LSTM-CP."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.90
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def interval_width(self) -> float:
        """Calculate interval width relative to point estimate."""
        if abs(self.point_estimate) < 1e-10:
            return float('inf')
        return (self.upper_bound - self.lower_bound) / abs(self.point_estimate)
    
    @property
    def absolute_width(self) -> float:
        """Calculate absolute interval width."""
        return self.upper_bound - self.lower_bound
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "interval_width": self.interval_width,
            "confidence_level": self.confidence_level
        }


@dataclass
class SizingResult:
    """Result of adaptive sizing calculation."""
    base_size: float
    adjusted_size: float
    uncertainty_penalty: float
    interval_width: float
    regime: RegimeType
    should_trade: bool
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_size": self.base_size,
            "adjusted_size": self.adjusted_size,
            "uncertainty_penalty": self.uncertainty_penalty,
            "interval_width": self.interval_width,
            "regime": self.regime.value,
            "should_trade": self.should_trade,
            "reason": self.reason
        }


class AdaptiveIntervalSizer:
    """
    Scale position size based on prediction uncertainty.
    
    Adaptive Thresholds:
    - INTERVAL_WIDTH < 0.02: UNCERTAINTY_PENALTY = 1.0 (full confidence)
    - INTERVAL_WIDTH 0.02-0.05: UNCERTAINTY_PENALTY = 0.8-1.0 (high confidence)
    - INTERVAL_WIDTH 0.05-0.10: UNCERTAINTY_PENALTY = 0.5-0.8 (moderate confidence)
    - INTERVAL_WIDTH 0.10-0.20: UNCERTAINTY_PENALTY = 0.2-0.5 (low confidence)
    - INTERVAL_WIDTH > 0.20: UNCERTAINTY_PENALTY = 0 (abstain)
    
    Configuration Parameters:
    - interval_sensitivity: Scaling sensitivity factor (default: 5.0)
    - full_confidence_threshold: Interval width for full size (default: 0.02)
    - abstain_threshold: Interval width for zero size (default: 0.20)
    - regime_sensitivity_normal: Sensitivity multiplier in normal (default: 1.0)
    - regime_sensitivity_high_vol: Sensitivity multiplier in high vol (default: 1.5)
    - regime_sensitivity_crisis: Sensitivity multiplier in crisis (default: 2.0)
    """
    
    def __init__(
        self,
        interval_sensitivity: float = 5.0,
        full_confidence_threshold: float = 0.02,
        abstain_threshold: float = 0.20,
        regime_sensitivity_normal: float = 1.0,
        regime_sensitivity_high_vol: float = 1.5,
        regime_sensitivity_crisis: float = 2.0
    ):
        self.base_sensitivity = interval_sensitivity
        self.full_confidence_threshold = full_confidence_threshold
        self.abstain_threshold = abstain_threshold
        
        self.regime_sensitivity = {
            RegimeType.NORMAL: regime_sensitivity_normal,
            RegimeType.HIGH_VOL: regime_sensitivity_high_vol,
            RegimeType.CRISIS: regime_sensitivity_crisis,
            RegimeType.UNKNOWN: regime_sensitivity_high_vol  # Conservative default
        }
        
        # Current regime (updated externally)
        self.current_regime: RegimeType = RegimeType.NORMAL
        
        # History for analysis
        self.sizing_history: List[SizingResult] = []
    
    def set_regime(self, regime: RegimeType) -> None:
        """Update current market regime."""
        self.current_regime = regime
    
    def get_effective_sensitivity(self) -> float:
        """Get sensitivity adjusted for current regime."""
        multiplier = self.regime_sensitivity.get(self.current_regime, 1.0)
        return self.base_sensitivity * multiplier
    
    def calculate_penalty(self, interval_width: float) -> Tuple[float, str]:
        """
        Calculate uncertainty penalty from interval width.
        
        UNCERTAINTY_PENALTY = 1 / (1 + INTERVAL_WIDTH × SENSITIVITY)
        
        Returns:
            (penalty, confidence_level_description)
        """
        # Check for abstain threshold
        if interval_width >= self.abstain_threshold:
            return 0.0, "abstain"
        
        # Check for full confidence
        if interval_width <= self.full_confidence_threshold:
            return 1.0, "full_confidence"
        
        # Get regime-adjusted sensitivity
        sensitivity = self.get_effective_sensitivity()
        
        # Calculate penalty using smooth formula
        penalty = 1.0 / (1.0 + interval_width * sensitivity)
        
        # Determine confidence level description
        if penalty >= 0.8:
            description = "high_confidence"
        elif penalty >= 0.5:
            description = "moderate_confidence"
        elif penalty > 0:
            description = "low_confidence"
        else:
            description = "abstain"
        
        return max(0.0, min(1.0, penalty)), description
    
    def calculate_adjusted_size(
        self,
        base_size: float,
        prediction_interval: PredictionInterval
    ) -> SizingResult:
        """
        Calculate uncertainty-adjusted position size.
        
        Args:
            base_size: Base position size from other constraints
            prediction_interval: Prediction interval from LSTM-CP
        
        Returns:
            SizingResult with adjusted size
        """
        interval_width = prediction_interval.interval_width
        
        # Handle infinite or very large interval widths
        if math.isinf(interval_width) or interval_width > 10.0:
            return SizingResult(
                base_size=base_size,
                adjusted_size=0.0,
                uncertainty_penalty=0.0,
                interval_width=interval_width,
                regime=self.current_regime,
                should_trade=False,
                reason="Invalid or infinite interval width"
            )
        
        # Calculate penalty
        penalty, confidence_level = self.calculate_penalty(interval_width)
        
        # Calculate adjusted size
        adjusted_size = base_size * penalty
        
        # Determine if should trade
        should_trade = penalty > 0 and adjusted_size > 0
        
        result = SizingResult(
            base_size=base_size,
            adjusted_size=adjusted_size,
            uncertainty_penalty=penalty,
            interval_width=interval_width,
            regime=self.current_regime,
            should_trade=should_trade,
            reason=f"{confidence_level} (penalty={penalty:.2f})"
        )
        
        # Record history
        self.sizing_history.append(result)
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]
        
        return result
    
    def quick_adjust(
        self,
        base_size: float,
        point_estimate: float,
        lower_bound: float,
        upper_bound: float
    ) -> float:
        """
        Quick adjustment without full result tracking.
        
        Returns adjusted position size.
        """
        interval = PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
        result = self.calculate_adjusted_size(base_size, interval)
        return result.adjusted_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics on sizing adjustments."""
        if not self.sizing_history:
            return {"count": 0}
        
        penalties = [r.uncertainty_penalty for r in self.sizing_history]
        widths = [r.interval_width for r in self.sizing_history if not math.isinf(r.interval_width)]
        trades = sum(1 for r in self.sizing_history if r.should_trade)
        
        return {
            "count": len(self.sizing_history),
            "avg_penalty": sum(penalties) / len(penalties),
            "min_penalty": min(penalties),
            "max_penalty": max(penalties),
            "avg_interval_width": sum(widths) / len(widths) if widths else 0,
            "trade_rate": trades / len(self.sizing_history),
            "abstain_rate": 1 - (trades / len(self.sizing_history))
        }


class AdaptiveIntervalWithKillSwitch:
    """
    Combines continuous scaling (L3-3) with binary kill-switch (L4).
    
    L3-3 provides continuous scaling based on uncertainty
    L4 Kill-Switch provides binary halt when uncertainty extreme
    Both work together: scaling reduces exposure gradually, kill-switch stops completely
    """
    
    def __init__(
        self,
        adaptive_sizer: Optional[AdaptiveIntervalSizer] = None,
        kill_switch_threshold: float = 0.30
    ):
        self.sizer = adaptive_sizer or AdaptiveIntervalSizer()
        self.kill_switch_threshold = kill_switch_threshold
        self.kill_switch_active = False
    
    def check_kill_switch(self, interval_width: float) -> Tuple[bool, str]:
        """
        Check if kill-switch should be activated.
        
        Returns:
            (should_kill, reason)
        """
        if interval_width >= self.kill_switch_threshold:
            self.kill_switch_active = True
            return True, f"Kill switch activated: interval {interval_width:.2%} >= {self.kill_switch_threshold:.2%}"
        
        # Recovery: require interval well below threshold
        if self.kill_switch_active:
            if interval_width < self.kill_switch_threshold * 0.5:
                self.kill_switch_active = False
                return False, "Kill switch deactivated: uncertainty recovered"
            else:
                return True, "Kill switch remains active: waiting for recovery"
        
        return False, "Kill switch not triggered"
    
    def calculate_with_kill_switch(
        self,
        base_size: float,
        prediction_interval: PredictionInterval
    ) -> SizingResult:
        """
        Calculate size with both scaling and kill-switch.
        """
        interval_width = prediction_interval.interval_width
        
        # Check kill switch first
        should_kill, kill_reason = self.check_kill_switch(interval_width)
        
        if should_kill:
            return SizingResult(
                base_size=base_size,
                adjusted_size=0.0,
                uncertainty_penalty=0.0,
                interval_width=interval_width,
                regime=self.sizer.current_regime,
                should_trade=False,
                reason=f"KILL_SWITCH: {kill_reason}"
            )
        
        # Apply continuous scaling
        return self.sizer.calculate_adjusted_size(base_size, prediction_interval)


if __name__ == "__main__":
    # Quick test
    sizer = AdaptiveIntervalSizer(
        interval_sensitivity=5.0,
        full_confidence_threshold=0.02,
        abstain_threshold=0.20
    )
    
    # Test various interval widths
    test_cases = [
        (0.01, "Very narrow - full confidence"),
        (0.03, "Narrow - high confidence"),
        (0.07, "Moderate - reduced position"),
        (0.15, "Wide - low confidence"),
        (0.25, "Very wide - abstain")
    ]
    
    base_size = 10000.0
    
    print("Adaptive Interval Sizing Tests:")
    print("=" * 60)
    
    for width, description in test_cases:
        interval = PredictionInterval(
            point_estimate=100.0,
            lower_bound=100.0 - (width * 100 / 2),
            upper_bound=100.0 + (width * 100 / 2)
        )
        
        result = sizer.calculate_adjusted_size(base_size, interval)
        
        print(f"\n{description}")
        print(f"  Interval width: {result.interval_width:.2%}")
        print(f"  Base size: ${result.base_size:,.0f}")
        print(f"  Uncertainty penalty: {result.uncertainty_penalty:.2f}")
        print(f"  Adjusted size: ${result.adjusted_size:,.0f}")
        print(f"  Should trade: {result.should_trade}")
    
    # Test regime sensitivity
    print("\n\nRegime Sensitivity Tests:")
    print("=" * 60)
    
    interval = PredictionInterval(
        point_estimate=100.0,
        lower_bound=95.0,  # 10% width
        upper_bound=105.0
    )
    
    for regime in RegimeType:
        sizer.set_regime(regime)
        result = sizer.calculate_adjusted_size(base_size, interval)
        print(f"{regime.name}: penalty={result.uncertainty_penalty:.2f}, size=${result.adjusted_size:,.0f}")
