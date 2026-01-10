"""
HIMARI OPUS 2 - Tier 5: Circuit Breaker System
===============================================

Emergency circuit breakers for position sizing.
Implements Tier 5 per CLAUDE Guide Part VIII.

These can override ALL upstream tiers and force position to zero.
They are the last line of defense against catastrophic losses.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any, List

# Handle imports
try:
    from core.layer3_config import CircuitBreakerConfig
except ImportError:
    from ..core.layer3_config import CircuitBreakerConfig

logger = logging.getLogger(__name__)


class CircuitBreakerSystem:
    """
    Tier 5: Emergency circuit breakers.
    
    These can override ALL upstream tiers and force position to zero.
    They are the last line of defense against catastrophic losses.
    
    Breakers:
    1. Daily Drawdown Kill Switch: if daily DD > 3% → position = 0
    2. Volatility Spike: if 5-min vol > 3× average → 10% size
    3. Spread Blowout: if spread > 0.5% → 50% size
    4. Data Staleness: if data > 5 seconds old → hold position
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        """
        Initialize circuit breaker system.
        
        Args:
            config: CircuitBreakerConfig with all parameters
        """
        if config is None:
            config = CircuitBreakerConfig()
        
        self.daily_drawdown_limit = config.daily_drawdown_limit  # 3%
        self.vol_spike_threshold = config.vol_spike_threshold    # 3x
        self.vol_spike_reduction = config.vol_spike_reduction    # 0.1
        self.spread_threshold = config.spread_threshold          # 0.5%
        self.spread_reduction = config.spread_reduction          # 0.5
        self.data_staleness_limit_ms = config.data_staleness_limit_ms  # 5000
    
    def check_all_breakers(
        self,
        position: float,
        portfolio_state: dict,
        market_context: dict,
        data_age_ms: int
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Check all circuit breakers and override position if triggered.
        
        Args:
            position: Position size from Tier 4
            portfolio_state: Portfolio state including daily_pnl_pct
            market_context: Market data including vol_spike_ratio, bid_ask_spread
            data_age_ms: Age of market data in milliseconds
            
        Returns:
            Tuple of (final_position, breaker_status, diagnostics)
            - final_position: Position after breaker checks (may be 0)
            - breaker_status: "CLEAR" or name of triggered breaker
            - diagnostics: Breaker check details
        """
        breakers_checked = []
        breaker_triggered = None
        original_position = position
        
        # Breaker 1: Daily Drawdown Kill Switch
        daily_pnl_pct = portfolio_state.get('daily_pnl_pct', 0.0)
        if daily_pnl_pct < -self.daily_drawdown_limit:
            breaker_triggered = 'DAILY_DRAWDOWN_KILL'
            position = 0  # Force to zero
            breakers_checked.append({
                'name': 'DAILY_DRAWDOWN_KILL',
                'triggered': True,
                'value': daily_pnl_pct,
                'threshold': -self.daily_drawdown_limit,
                'action': 'POSITION_ZERO'
            })
        else:
            breakers_checked.append({
                'name': 'DAILY_DRAWDOWN_KILL',
                'triggered': False,
                'value': daily_pnl_pct,
                'threshold': -self.daily_drawdown_limit
            })
        
        # Breaker 2: Volatility Spike
        vol_spike = market_context.get('vol_spike_ratio', 1.0)
        if vol_spike > self.vol_spike_threshold and breaker_triggered is None:
            breaker_triggered = 'VOLATILITY_SPIKE'
            position = position * self.vol_spike_reduction  # Reduce to 10%
            breakers_checked.append({
                'name': 'VOLATILITY_SPIKE',
                'triggered': True,
                'value': vol_spike,
                'threshold': self.vol_spike_threshold,
                'action': f'REDUCE_TO_{self.vol_spike_reduction:.0%}'
            })
        else:
            breakers_checked.append({
                'name': 'VOLATILITY_SPIKE',
                'triggered': False,
                'value': vol_spike,
                'threshold': self.vol_spike_threshold
            })
        
        # Breaker 3: Spread Blowout
        spread = market_context.get('bid_ask_spread', 0.0)
        if spread > self.spread_threshold and breaker_triggered is None:
            breaker_triggered = 'SPREAD_BLOWOUT'
            position = position * self.spread_reduction  # Reduce to 50%
            breakers_checked.append({
                'name': 'SPREAD_BLOWOUT',
                'triggered': True,
                'value': spread,
                'threshold': self.spread_threshold,
                'action': f'REDUCE_TO_{self.spread_reduction:.0%}'
            })
        else:
            breakers_checked.append({
                'name': 'SPREAD_BLOWOUT',
                'triggered': False,
                'value': spread,
                'threshold': self.spread_threshold
            })
        
        # Breaker 4: Data Staleness
        if data_age_ms > self.data_staleness_limit_ms and breaker_triggered is None:
            breaker_triggered = 'DATA_STALE'
            position = 0  # No new trades on stale data
            breakers_checked.append({
                'name': 'DATA_STALE',
                'triggered': True,
                'value': data_age_ms,
                'threshold': self.data_staleness_limit_ms,
                'action': 'HOLD_POSITION'
            })
        else:
            breakers_checked.append({
                'name': 'DATA_STALE',
                'triggered': False,
                'value': data_age_ms,
                'threshold': self.data_staleness_limit_ms
            })
        
        breaker_status = breaker_triggered if breaker_triggered else 'CLEAR'
        
        diagnostics = {
            'breakers_checked': breakers_checked,
            'breaker_triggered': breaker_triggered,
            'breaker_status': breaker_status,
            'original_position': original_position,
            'final_position': position,
            'tier': 'CIRCUIT_BREAKERS'
        }
        
        if breaker_triggered:
            logger.warning(
                f"Circuit breaker triggered: {breaker_triggered} "
                f"${original_position:,.2f} → ${position:,.2f}"
            )
        
        return position, breaker_status, diagnostics
    
    def get_breaker_states(self) -> Dict[str, bool]:
        """Get current state of all breakers (for monitoring)."""
        return {
            'DAILY_DRAWDOWN_KILL': False,
            'VOLATILITY_SPIKE': False,
            'SPREAD_BLOWOUT': False,
            'DATA_STALE': False
        }
