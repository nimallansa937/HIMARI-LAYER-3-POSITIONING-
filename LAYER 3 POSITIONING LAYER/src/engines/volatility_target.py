"""
HIMARI OPUS 2 - Tier 1: Volatility Targeting Engine
====================================================

Deterministic volatility-targeting position sizing.
This is the foundation of Layer 3 per CLAUDE Guide Part IV.

The key principle: position_size = (target_vol / realized_vol) × equity × base_fraction

Version: 1.0
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any

# Handle both module and script imports
try:
    from core.layer3_config import VolatilityTargetConfig
except ImportError:
    from ..core.layer3_config import VolatilityTargetConfig

logger = logging.getLogger(__name__)


class VolatilityTargetEngine:
    """
    Tier 1: Deterministic volatility-targeting position sizing.
    
    This is the foundation of Layer 3. All other tiers modify the output
    of this engine but cannot increase positions beyond what volatility
    targeting recommends.
    """
    
    def __init__(self, config: VolatilityTargetConfig = None):
        """
        Initialize volatility targeting engine.
        
        Args:
            config: VolatilityTargetConfig with all parameters
        """
        if config is None:
            config = VolatilityTargetConfig()
        
        self.target_vol_annual = config.target_vol_annual  # e.g., 0.15 (15%)
        self.lookback_short = config.lookback_short        # e.g., 5 days
        self.lookback_long = config.lookback_long          # e.g., 20 days
        self.base_fraction = config.base_fraction          # e.g., 0.5 (half-Kelly)
        self.min_position_pct = config.min_position_pct    # e.g., 0.01 (1%)
        self.max_position_pct = config.max_position_pct    # e.g., 0.10 (10%)
        self.atr_stop_multiplier = config.atr_stop_multiplier
    
    def compute_realized_volatility(
        self, 
        returns: np.ndarray, 
        lookback: int
    ) -> float:
        """
        Compute annualized realized volatility from returns.
        
        Uses close-to-close returns with bias correction for small samples.
        Annualization assumes 365 trading days for crypto.
        
        Args:
            returns: Array of daily returns
            lookback: Number of days to look back
            
        Returns:
            Annualized volatility as decimal
        """
        if len(returns) < lookback:
            lookback = len(returns)
        
        if lookback == 0:
            return self.target_vol_annual  # Default to target if no data
        
        recent_returns = returns[-lookback:]
        
        # Standard deviation with Bessel's correction
        vol_daily = np.std(recent_returns, ddof=1) if len(recent_returns) > 1 else 0.02
        
        # Annualize (crypto trades 365 days)
        vol_annual = vol_daily * np.sqrt(365)
        
        return max(vol_annual, 0.01)  # Minimum 1% to avoid division issues
    
    def compute_blended_volatility(
        self,
        vol_short: float,
        vol_long: float,
        regime: str
    ) -> float:
        """
        Blend short and long lookback volatilities based on regime.
        
        In NORMAL regime, weight toward long lookback for stability.
        In HIGH_VOL/CRISIS, weight toward short lookback for responsiveness.
        
        Args:
            vol_short: Short-term volatility (e.g., 5-day)
            vol_long: Long-term volatility (e.g., 20-day)
            regime: Current market regime
            
        Returns:
            Blended volatility
        """
        weights = {
            'NORMAL':   (0.3, 0.7),  # 30% short, 70% long
            'HIGH_VOL': (0.6, 0.4),  # 60% short, 40% long
            'CRISIS':   (0.8, 0.2),  # 80% short, 20% long
            'CASCADE':  (0.9, 0.1),  # 90% short, 10% long
        }
        
        w_short, w_long = weights.get(regime, (0.5, 0.5))
        blended = w_short * vol_short + w_long * vol_long
        
        return max(blended, 0.01)  # Minimum 1%
    
    def compute_base_position_size(
        self,
        portfolio_equity: float,
        realized_vol: float,
        signal_confidence: float
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute base position size using volatility targeting.
        
        Core formula:
        position_pct = (target_vol / realized_vol) × base_fraction × confidence_scalar
        
        Args:
            portfolio_equity: Current portfolio value in USD
            realized_vol: Realized volatility (annualized decimal)
            signal_confidence: Signal confidence [0.0, 1.0]
            
        Returns:
            Tuple of (position_size_usd, diagnostics)
        """
        # Core volatility targeting formula
        vol_ratio = self.target_vol_annual / max(realized_vol, 0.01)
        
        # Apply base fraction (conservative Kelly)
        position_pct = vol_ratio * self.base_fraction
        
        # Scale by signal confidence (0.5 + 0.5 * confidence)
        # Confidence of 0.5 gives 75% of base position
        # Confidence of 1.0 gives 100% of base position
        confidence_scalar = 0.5 + 0.5 * signal_confidence
        position_pct *= confidence_scalar
        
        # Store raw value for diagnostics
        position_pct_raw = position_pct
        
        # Enforce min/max position percentage
        position_pct = np.clip(
            position_pct, 
            self.min_position_pct, 
            self.max_position_pct
        )
        
        # Convert to USD
        position_size_usd = position_pct * portfolio_equity
        
        diagnostics = {
            'target_vol': self.target_vol_annual,
            'realized_vol': realized_vol,
            'vol_ratio': vol_ratio,
            'base_fraction': self.base_fraction,
            'confidence': signal_confidence,
            'confidence_scalar': confidence_scalar,
            'position_pct_raw': position_pct_raw,
            'position_pct_clipped': position_pct,
            'position_size_usd': position_size_usd,
            'portfolio_equity': portfolio_equity,
            'tier': 'VOLATILITY_TARGET'
        }
        
        return position_size_usd, diagnostics
    
    def compute_stop_distance(
        self,
        atr_value: float,
        atr_multiplier: float = None
    ) -> float:
        """
        Compute stop distance as multiple of ATR.
        
        Args:
            atr_value: Current ATR (in price units)
            atr_multiplier: How many ATRs away to place stop (default from config)
            
        Returns:
            Stop distance in price units
        """
        if atr_multiplier is None:
            atr_multiplier = self.atr_stop_multiplier
        
        return atr_value * atr_multiplier
    
    def process(
        self,
        portfolio_equity: float,
        vol_5d: float,
        vol_20d: float,
        signal_confidence: float,
        regime: str,
        atr_value: float = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Complete Tier 1 processing.
        
        Args:
            portfolio_equity: Portfolio value in USD
            vol_5d: 5-day realized volatility
            vol_20d: 20-day realized volatility
            signal_confidence: Signal confidence [0.0, 1.0]
            regime: Current market regime
            atr_value: ATR for stop calculation (optional)
            
        Returns:
            Tuple of (position_size_usd, stop_distance, diagnostics)
        """
        # Blend volatilities
        blended_vol = self.compute_blended_volatility(vol_5d, vol_20d, regime)
        
        # Compute base position
        position_size, diag = self.compute_base_position_size(
            portfolio_equity,
            blended_vol,
            signal_confidence
        )
        
        # Add blending info to diagnostics
        diag['vol_short'] = vol_5d
        diag['vol_long'] = vol_20d
        diag['vol_blended'] = blended_vol
        diag['regime'] = regime
        
        # Compute stop distance
        if atr_value is not None:
            stop_distance = self.compute_stop_distance(atr_value)
        else:
            # Estimate from volatility (rough approximation)
            stop_distance = blended_vol * 0.02 * self.atr_stop_multiplier
        
        diag['stop_distance'] = stop_distance
        
        logger.debug(
            f"Tier 1 output: ${position_size:,.2f} "
            f"(vol={blended_vol:.2%}, conf={signal_confidence:.2f})"
        )
        
        return position_size, stop_distance, diag
