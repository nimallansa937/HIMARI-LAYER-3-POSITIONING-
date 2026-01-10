"""
HIMARI OPUS 2 - Layer 3 Configuration
======================================

Complete configuration for Layer 3 Position Sizing Engine.
Implements all Tier configurations per CLAUDE Guide Part XII.

Version: 1.0
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VolatilityTargetConfig:
    """Configuration for Tier 1 Volatility Targeting."""
    
    # Target annualized volatility (15% is moderate)
    target_vol_annual: float = 0.15
    
    # Lookback periods for volatility calculation
    lookback_short: int = 5    # 5-day for responsiveness
    lookback_long: int = 20    # 20-day for stability
    
    # Base fraction (0.5 = half-Kelly, conservative)
    base_fraction: float = 0.5
    
    # Position size bounds (as fraction of portfolio)
    min_position_pct: float = 0.01  # 1% minimum
    max_position_pct: float = 0.10  # 10% maximum
    
    # ATR parameters
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0


@dataclass
class AdaptiveConfig:
    """Configuration for Tier 2 Bounded Adaptive Enhancement."""
    
    # RL model
    rl_model_path: str = '/models/rl_policy_v3.pt'
    rl_delta_bounds: Tuple[float, float] = (-0.30, +0.30)
    
    # Funding rate thresholds
    funding_threshold_reduce: float = 0.0003   # 0.03%
    funding_threshold_exit: float = 0.001      # 0.1%
    funding_reduction_factor: float = 0.5
    
    # Correlation thresholds
    correlation_threshold_elevated: float = 0.85
    correlation_threshold_extreme: float = 0.95
    correlation_reduction_elevated: float = 0.7
    correlation_reduction_extreme: float = 0.4
    
    # Cascade detection thresholds
    cascade_oi_threshold: float = 0.05        # 5% OI drop
    cascade_volume_threshold: float = 3.0     # 3x volume spike
    cascade_reduction_factor: float = 0.6
    
    # Master bounds for adaptive adjustments
    adaptive_lower_bound: float = 0.7   # Max 30% reduction
    adaptive_upper_bound: float = 1.3   # Max 30% increase


@dataclass
class HardConstraintConfig:
    """Configuration for Tier 4 Hard Constraints."""
    
    # Single position limits
    max_single_position_pct: float = 0.05  # 5% max per position
    
    # Sector concentration
    max_sector_concentration_pct: float = 0.20  # 20% max per sector
    
    # Correlation aggregation
    correlation_aggregation_threshold: float = 0.70  # ρ > 0.7 = same position
    
    # Leverage caps by regime
    leverage_cap_normal: float = 2.0
    leverage_cap_high_vol: float = 1.5
    leverage_cap_crisis: float = 1.0
    leverage_cap_cascade: float = 0.0  # Flat during cascades


@dataclass
class CircuitBreakerConfig:
    """Configuration for Tier 5 Circuit Breakers."""
    
    # Daily drawdown kill switch
    daily_drawdown_limit: float = 0.03  # 3%
    
    # Volatility spike breaker
    vol_spike_threshold: float = 3.0  # 3x average
    vol_spike_reduction: float = 0.1  # Reduce to 10%
    
    # Spread blowout breaker
    spread_threshold: float = 0.005  # 0.5%
    spread_reduction: float = 0.5    # Reduce to 50%
    
    # Data staleness
    data_staleness_limit_ms: int = 5000  # 5 seconds


@dataclass
class Layer3Config:
    """Complete configuration for Layer 3 Position Sizing Engine."""
    
    # Tier 1: Volatility Targeting
    vol_target: VolatilityTargetConfig = field(
        default_factory=lambda: VolatilityTargetConfig(
            target_vol_annual=0.15,
            lookback_short=5,
            lookback_long=20,
            base_fraction=0.5,
            min_position_pct=0.01,
            max_position_pct=0.10,
            atr_period=14,
            atr_stop_multiplier=2.0
        )
    )
    
    # Tier 2: Adaptive Enhancement
    adaptive: AdaptiveConfig = field(
        default_factory=lambda: AdaptiveConfig(
            rl_model_path='/models/rl_policy_v3.pt',
            rl_delta_bounds=(-0.30, +0.30),
            funding_threshold_reduce=0.0003,
            funding_threshold_exit=0.001,
            correlation_threshold_elevated=0.85,
            correlation_threshold_extreme=0.95,
            cascade_oi_threshold=0.05,
            cascade_volume_threshold=3.0
        )
    )
    
    # Tier 4: Hard Constraints
    constraints: HardConstraintConfig = field(
        default_factory=lambda: HardConstraintConfig(
            max_single_position_pct=0.05,
            max_sector_concentration_pct=0.20,
            correlation_aggregation_threshold=0.70,
            leverage_cap_normal=2.0,
            leverage_cap_high_vol=1.5,
            leverage_cap_crisis=1.0,
            leverage_cap_cascade=0.0
        )
    )
    
    # Tier 5: Circuit Breakers
    breakers: CircuitBreakerConfig = field(
        default_factory=lambda: CircuitBreakerConfig(
            daily_drawdown_limit=0.03,
            vol_spike_threshold=3.0,
            spread_threshold=0.005,
            data_staleness_limit_ms=5000
        )
    )
