"""
HIMARI OPUS 2 - Layer 3 Output Types
=====================================

Output dataclasses for Layer 3 Position Sizing Engine.
Implements output contract per CLAUDE Guide Part IX.

Version: 1.0
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import time


@dataclass
class Layer3Output:
    """Output message from Layer 3 to Layer 4."""
    
    # Signal identification
    signal_id: str
    timestamp: int  # Unix ms
    symbol: str
    direction: str  # LONG, SHORT, FLAT
    strategy_id: str
    
    # Position sizing results
    position_size_usd: float
    position_size_pct: float
    leverage: float
    
    # Regime context
    regime: str
    regime_confidence: float
    
    # Constraint information
    constraints_hit: List[str]
    breaker_status: str  # CLEAR or breaker name
    
    # Risk metrics
    stop_distance: float
    stop_price: float
    risk_per_trade_usd: float
    risk_per_trade_pct: float
    
    # Full diagnostics (for audit trail)
    diagnostics: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Serialize for transmission to Layer 4."""
        return asdict(self)
    
    def is_actionable(self) -> bool:
        """Check if this output should result in a trade."""
        return (
            self.position_size_usd > 0 and
            self.breaker_status == 'CLEAR' and
            self.direction != 'FLAT'
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"{self.direction} {self.symbol}: ${self.position_size_usd:,.2f} "
            f"({self.position_size_pct:.1%}) | "
            f"Regime: {self.regime} | "
            f"Breaker: {self.breaker_status}"
        )


def create_rejection_output(
    signal_id: str,
    symbol: str,
    strategy_id: str,
    errors: List[str]
) -> Layer3Output:
    """Create rejection output for invalid signals."""
    return Layer3Output(
        signal_id=signal_id,
        timestamp=int(time.time() * 1000),
        symbol=symbol,
        direction='FLAT',
        strategy_id=strategy_id,
        position_size_usd=0.0,
        position_size_pct=0.0,
        leverage=0.0,
        regime='UNKNOWN',
        regime_confidence=0.0,
        constraints_hit=['INPUT_VALIDATION_FAILED'],
        breaker_status='VALIDATION_ERROR',
        stop_distance=0.0,
        stop_price=0.0,
        risk_per_trade_usd=0.0,
        risk_per_trade_pct=0.0,
        diagnostics={'validation_errors': errors}
    )


def create_zero_position_output(
    signal_id: str,
    symbol: str,
    strategy_id: str,
    regime: str,
    breaker_status: str,
    diagnostics: Dict[str, Any]
) -> Layer3Output:
    """Create zero position output (e.g., circuit breaker triggered)."""
    return Layer3Output(
        signal_id=signal_id,
        timestamp=int(time.time() * 1000),
        symbol=symbol,
        direction='FLAT',
        strategy_id=strategy_id,
        position_size_usd=0.0,
        position_size_pct=0.0,
        leverage=0.0,
        regime=regime,
        regime_confidence=0.0,
        constraints_hit=[],
        breaker_status=breaker_status,
        stop_distance=0.0,
        stop_price=0.0,
        risk_per_trade_usd=0.0,
        risk_per_trade_pct=0.0,
        diagnostics=diagnostics
    )
