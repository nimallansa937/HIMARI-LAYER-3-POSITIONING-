"""
HIMARI OPUS 2 - Layer 3 Position Sizing Engine
===============================================

Complete Layer 3 Position Sizing Engine.
Integrates all five tiers per CLAUDE Guide Part X.

Tiers:
1. Volatility Targeting (deterministic core)
2. Bounded Adaptive Enhancement
3. Regime-Conditional Adjustment
4. Hard Constraint Enforcement
5. Circuit Breakers

Latency target: <200ms (typical: 150ms)

Version: 1.0
"""

import time
import numpy as np
import logging
from typing import Dict, Any, List, Optional

# Handle imports
try:
    from core.layer3_config import Layer3Config
    from core.layer3_input_validator import Layer3InputValidator
    from core.layer3_output import Layer3Output, create_rejection_output
    from engines.volatility_target import VolatilityTargetEngine
    from engines.bounded_adaptive import BoundedAdaptiveEnhancement
    from engines.regime_conditional_adjuster import RegimeConditionalAdjuster
    from engines.hard_constraints import HardConstraintEnforcer
    from engines.circuit_breaker_system import CircuitBreakerSystem
except ImportError:
    from ..core.layer3_config import Layer3Config
    from ..core.layer3_input_validator import Layer3InputValidator
    from ..core.layer3_output import Layer3Output, create_rejection_output
    from .volatility_target import VolatilityTargetEngine
    from .bounded_adaptive import BoundedAdaptiveEnhancement
    from .regime_conditional_adjuster import RegimeConditionalAdjuster
    from .hard_constraints import HardConstraintEnforcer
    from .circuit_breaker_system import CircuitBreakerSystem

logger = logging.getLogger(__name__)


class Layer3PositionSizingEngine:
    """
    Complete Layer 3 Position Sizing Engine.
    
    Integrates all five tiers:
    1. Volatility Targeting (deterministic core)
    2. Bounded Adaptive Enhancement
    3. Regime-Conditional Adjustment
    4. Hard Constraint Enforcement
    5. Circuit Breakers
    
    Latency target: <200ms (typical: 150ms)
    """
    
    def __init__(self, config: Layer3Config = None):
        """
        Initialize Layer 3 engine.
        
        Args:
            config: Layer3Config with all tier configurations
        """
        if config is None:
            config = Layer3Config()
        
        self.config = config
        
        # Initialize all tiers
        self.validator = Layer3InputValidator()
        self.volatility_target = VolatilityTargetEngine(config.vol_target)
        self.adaptive = BoundedAdaptiveEnhancement(config.adaptive)
        self.regime_adjuster = RegimeConditionalAdjuster()
        self.constraints = HardConstraintEnforcer(config.constraints)
        self.breakers = CircuitBreakerSystem(config.breakers)
        
        # Statistics
        self.signals_processed = 0
        self.signals_rejected = 0
        self.breakers_triggered = 0
        
        logger.info("Layer 3 Position Sizing Engine initialized")
    
    def process_signal(self, signal: dict) -> Layer3Output:
        """
        Process a trading signal through all five tiers.
        
        Args:
            signal: Complete signal from Layer 2
            
        Returns:
            Layer3Output with position size and diagnostics
        """
        start_time = time.time()
        all_diagnostics = {}
        
        # Step 0: Validate inputs
        is_valid, errors = self.validator.validate(signal)
        if not is_valid:
            self.signals_rejected += 1
            return create_rejection_output(
                signal.get('signal_id', 'UNKNOWN'),
                signal.get('symbol', 'UNKNOWN'),
                signal.get('strategy_id', 'UNKNOWN'),
                errors
            )
        
        self.signals_processed += 1
        
        # Extract inputs
        market_ctx = signal['market_context']
        portfolio = signal['portfolio_state']
        regime = signal.get('regime', 'NORMAL')
        regime_conf = signal.get('regime_confidence', 0.5)
        
        # Calculate data age
        data_age_ms = int(time.time() * 1000) - signal['timestamp']
        
        # =====================================================================
        # TIER 1: Volatility Targeting
        # =====================================================================
        vol_short = market_ctx['realized_vol_5d']
        vol_long = market_ctx['realized_vol_20d']
        
        base_position, stop_distance, vol_diag = self.volatility_target.process(
            portfolio['portfolio_equity'],
            vol_short,
            vol_long,
            signal['confidence'],
            regime,
            market_ctx.get('atr_14')
        )
        all_diagnostics['tier_1'] = vol_diag
        
        # =====================================================================
        # TIER 2: Adaptive Enhancement
        # =====================================================================
        features = self._extract_features(signal)
        
        # Add symbol to market context for correlation check
        market_ctx_with_symbol = {**market_ctx, 'symbol': signal['symbol']}
        
        adjusted_position, adaptive_diag = self.adaptive.compute_adaptive_adjustment(
            base_position,
            features,
            market_ctx_with_symbol,
            regime,
            signal['direction']
        )
        all_diagnostics['tier_2'] = adaptive_diag
        
        # =====================================================================
        # TIER 3: Regime Adjustment
        # =====================================================================
        regime_position, regime_diag = self.regime_adjuster.compute_regime_adjustment(
            adjusted_position,
            regime,
            regime_conf
        )
        all_diagnostics['tier_3'] = regime_diag
        
        # =====================================================================
        # TIER 4: Hard Constraints
        # =====================================================================
        sector = self.constraints.get_sector_for_symbol(signal['symbol'])
        correlated_positions = self.constraints.get_correlated_positions(
            signal['symbol'],
            portfolio.get('open_positions', [])
        )
        
        constrained_position, constraint_diag = self.constraints.enforce_constraints(
            regime_position,
            portfolio['portfolio_equity'],
            portfolio.get('sector_exposures', {}),
            sector,
            regime,
            correlated_positions
        )
        all_diagnostics['tier_4'] = constraint_diag
        
        # =====================================================================
        # TIER 5: Circuit Breakers
        # =====================================================================
        final_position, breaker_status, breaker_diag = self.breakers.check_all_breakers(
            constrained_position,
            portfolio,
            market_ctx,
            data_age_ms
        )
        all_diagnostics['tier_5'] = breaker_diag
        
        if breaker_status != 'CLEAR':
            self.breakers_triggered += 1
        
        # =====================================================================
        # Compute derived values
        # =====================================================================
        position_pct = final_position / portfolio['portfolio_equity'] if portfolio['portfolio_equity'] > 0 else 0
        leverage = position_pct  # Simplified: position % = leverage for spot
        
        # Stop price calculation
        current_price = market_ctx.get('current_price', 0)
        if signal['direction'] == 'LONG':
            stop_price = current_price - stop_distance
        elif signal['direction'] == 'SHORT':
            stop_price = current_price + stop_distance
        else:
            stop_price = current_price
        
        # Risk calculation
        if current_price > 0:
            risk_per_trade_usd = final_position * (stop_distance / current_price)
            risk_per_trade_pct = position_pct * (stop_distance / current_price)
        else:
            risk_per_trade_usd = 0
            risk_per_trade_pct = 0
        
        # Latency tracking
        latency_ms = (time.time() - start_time) * 1000
        all_diagnostics['latency_ms'] = latency_ms
        all_diagnostics['data_age_ms'] = data_age_ms
        
        # Build output
        output = Layer3Output(
            signal_id=signal['signal_id'],
            timestamp=int(time.time() * 1000),
            symbol=signal['symbol'],
            direction=signal['direction'],
            strategy_id=signal['strategy_id'],
            position_size_usd=final_position,
            position_size_pct=position_pct,
            leverage=leverage,
            regime=regime,
            regime_confidence=regime_conf,
            constraints_hit=constraint_diag.get('constraints_hit', []),
            breaker_status=breaker_status,
            stop_distance=stop_distance,
            stop_price=stop_price,
            risk_per_trade_usd=risk_per_trade_usd,
            risk_per_trade_pct=risk_per_trade_pct,
            diagnostics=all_diagnostics
        )
        
        logger.info(
            f"L3 processed: {signal['symbol']} {signal['direction']} "
            f"→ ${final_position:,.2f} ({position_pct:.1%}) "
            f"in {latency_ms:.1f}ms"
        )
        
        return output
    
    def _extract_features(self, signal: dict) -> np.ndarray:
        """
        Extract 60-dimensional feature vector for RL policy.
        
        Args:
            signal: Complete signal from Layer 2
            
        Returns:
            Feature vector (60,)
        """
        # Build feature vector from available data
        market_ctx = signal.get('market_context', {})
        portfolio = signal.get('portfolio_state', {})
        
        features = [
            # Volatility features (0-4)
            market_ctx.get('realized_vol_5d', 0.0),
            market_ctx.get('realized_vol_20d', 0.0),
            market_ctx.get('vol_spike_ratio', 1.0),
            market_ctx.get('bid_ask_spread', 0.0),
            market_ctx.get('btc_correlation', 0.0),
            
            # Funding/OI features (5-9)
            market_ctx.get('funding_rate', 0.0),
            market_ctx.get('open_interest', 0.0) / 1e9,  # Normalize to billions
            market_ctx.get('open_interest_delta_1h', 0.0),
            market_ctx.get('volume_24h', 0.0) / 1e9,  # Normalize to billions
            market_ctx.get('volume_spike_ratio', 1.0),
            
            # Portfolio features (10-14)
            portfolio.get('portfolio_equity', 0.0) / 1e6,  # Normalize to millions
            portfolio.get('cash_available', 0.0) / 1e6,
            portfolio.get('daily_pnl_pct', 0.0),
            portfolio.get('drawdown_from_hwm', 0.0),
            signal.get('confidence', 0.5),
        ]
        
        # Pad to 60 dimensions
        while len(features) < 60:
            features.append(0.0)
        
        return np.array(features[:60], dtype=np.float32)
    
    def get_statistics(self) -> dict:
        """Get engine statistics."""
        return {
            'signals_processed': self.signals_processed,
            'signals_rejected': self.signals_rejected,
            'breakers_triggered': self.breakers_triggered,
            'rejection_rate': self.signals_rejected / max(1, self.signals_processed + self.signals_rejected)
        }
    
    def reset_statistics(self):
        """Reset engine statistics."""
        self.signals_processed = 0
        self.signals_rejected = 0
        self.breakers_triggered = 0
