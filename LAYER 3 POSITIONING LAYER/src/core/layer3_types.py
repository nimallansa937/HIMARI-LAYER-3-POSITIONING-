"""
HIMARI OPUS V2 - Layer 3 Core Type Definitions
================================================

Core data types, enums, and dataclasses for the Layer 3 Position Sizing system.
Aligned with Layer 2 naming conventions and signal specifications.

Version: 3.1 Enhanced
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time


# ============================================================================
# ENUMS
# ============================================================================

class MarketRegime(Enum):
    """
    Market regime states (aligned with Layer 2 naming).
    
    Alignment with L2 DetectedRegime:
    - L2: TRENDING_UP → L3: TRENDING_UP ✓
    - L2: TRENDING_DOWN → L3: TRENDING_DOWN ✓
    - L2: RANGING → L3: RANGING ✓
    - L2: HIGH_VOLATILITY → L3: HIGH_VOLATILITY ✓
    - L2: CRISIS_FLIGHT → L3: CRISIS (mapped)
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"  # Mapped from L2's CRISIS_FLIGHT
    
    @classmethod
    def from_l2_regime(cls, l2_regime_str: str) -> 'MarketRegime':
        """
        Convert Layer 2 regime string to Layer 3 enum.
        
        Args:
            l2_regime_str: Regime string from Layer 2 (e.g., "CRISIS_FLIGHT")
            
        Returns:
            MarketRegime enum value
        """
        mapping = {
            'TRENDING_UP': cls.TRENDING_UP,
            'TRENDING_DOWN': cls.TRENDING_DOWN,
            'RANGING': cls.RANGING,
            'HIGH_VOLATILITY': cls.HIGH_VOLATILITY,
            'CRISIS_FLIGHT': cls.CRISIS,  # L2 → L3 mapping
            'CRISIS': cls.CRISIS
        }
        return mapping.get(l2_regime_str.upper(), cls.RANGING)  # Default to RANGING


class TacticalAction(Enum):
    """Tactical action recommendations from Layer 2."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class CircuitState(Enum):
    """Circuit breaker states for Colab Pro API."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit tripped, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


# ============================================================================
# DATACLASSES - INPUT SIGNALS
# ============================================================================

@dataclass
class CascadeIndicators:
    """
    Complete cascade risk indicators from L1/L2.
    
    Includes existing exchange-based indicators plus new on-chain signals
    from Layer 1 for enhanced cascade detection.
    """
    # Exchange-based indicators (existing)
    funding_rate: float                     # Perpetual funding rate
    oi_change_pct: float                    # Open interest % change
    volume_ratio: float                     # Current vol / 24h avg
    
    # On-chain indicators (NEW from Layer 1)
    onchain_whale_pressure: float           # Whale netflow pressure [0,1]
    exchange_netflow_zscore: float          # Exchange flow Z-score
    
    # Optional liquidation data
    liquidation_volume_24h: Optional[float] = None


@dataclass
class TacticalSignal:
    """
    Extended tactical signal from Layer 2 with optional sentiment.
    
    Core fields from Layer 2 plus optional sentiment trend for
    sentiment-aware position sizing.
    """
    strategy_id: str
    symbol: str
    action: TacticalAction
    confidence: float                       # [0.0, 1.0]
    risk_score: float                       # [0.0, 1.0]
    regime: MarketRegime
    timestamp_ns: int
    
    # Existing optional fields
    expected_return: Optional[float] = None
    predicted_volatility: Optional[float] = None
    signal_strength: Optional[float] = None
    
    # NEW: Sentiment trend from L2 sentiment processor (optional)
    sentiment_trend: Optional[str] = None   # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_strength: Optional[float] = None  # [0.0, 1.0]
    
    # NEW: Sentiment score for sentiment-aware sizing (Section 11.3)
    sentiment_score: Optional[float] = None  # [0.0, 1.0] where 0=bearish, 1=bullish
    sentiment_confidence: Optional[float] = None  # Confidence in sentiment [0.0, 1.0]


# ============================================================================
# DATACLASSES - OUTPUT DECISIONS
# ============================================================================

@dataclass
class PositionSizingDecision:
    """
    Complete position sizing decision with comprehensive diagnostics.
    
    Contains the final position size in USD along with detailed diagnostics
    from each stage of the sizing pipeline.
    """
    # Core decision
    position_size_usd: float
    symbol: str
    strategy_id: str
    timestamp_ns: int
    
    # Component sizes (pipeline stages)
    kelly_position_usd: float               # Base from Bayesian Kelly
    conformal_adjusted_usd: float           # After conformal scaling
    regime_adjusted_usd: float              # After regime adjustment
    cascade_adjusted_usd: float             # After cascade risk adjustment
    
    # Risk metrics
    cascade_risk_score: float               # [0.0, 1.0]
    cascade_recommendation: str             # EXIT, REDUCE_75%, REDUCE_50%, MONITOR
    current_regime: MarketRegime
    
    # Diagnostics (for monitoring and debugging)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Optional sentiment diagnostics
    sentiment_diagnostics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'position_size_usd': self.position_size_usd,
            'symbol': self.symbol,
            'strategy_id': self.strategy_id,
            'timestamp_ns': self.timestamp_ns,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.timestamp_ns / 1e9)),
            'kelly_position_usd': self.kelly_position_usd,
            'conformal_adjusted_usd': self.conformal_adjusted_usd,
            'regime_adjusted_usd': self.regime_adjusted_usd,
            'cascade_adjusted_usd': self.cascade_adjusted_usd,
            'cascade_risk_score': self.cascade_risk_score,
            'cascade_recommendation': self.cascade_recommendation,
            'current_regime': self.current_regime.value,
            'diagnostics': self.diagnostics,
            'sentiment_diagnostics': self.sentiment_diagnostics
        }


@dataclass
class EnsembleAllocation:
    """
    Portfolio-level allocation across multiple strategies.
    
    Result of ensemble aggregation with weight drift tracking.
    """
    allocations: Dict[str, float]           # {symbol: allocated_usd}
    total_allocated_usd: float
    portfolio_value: float
    utilization_pct: float                  # % of portfolio allocated
    
    # Strategy weights
    strategy_weights: Dict[str, float]      # {strategy_id: weight}
    
    # Weight drift diagnostics
    drifted_strategies: list                # Strategy IDs with >20% drift
    max_drift_pct: float
    
    # Correlation diagnostics
    correlation_penalties: Dict[str, float] # {symbol: penalty_applied}
    
    timestamp_ns: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'allocations': self.allocations,
            'total_allocated_usd': self.total_allocated_usd,
            'portfolio_value': self.portfolio_value,
            'utilization_pct': self.utilization_pct,
            'strategy_weights': self.strategy_weights,
            'drifted_strategies': self.drifted_strategies,
            'max_drift_pct': self.max_drift_pct,
            'correlation_penalties': self.correlation_penalties,
            'timestamp_ns': self.timestamp_ns,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.timestamp_ns / 1e9))
        }


# ============================================================================
# EXCEPTIONS
# ============================================================================

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass


class InvalidSignalException(Exception):
    """Raised when Layer 2 signal fails validation."""
    pass


class InvalidCascadeIndicatorsException(Exception):
    """Raised when cascade indicators fail validation."""
    pass


class ConfigurationException(Exception):
    """Raised when configuration is invalid."""
    pass
