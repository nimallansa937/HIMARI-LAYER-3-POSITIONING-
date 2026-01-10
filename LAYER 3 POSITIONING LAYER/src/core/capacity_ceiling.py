# src/core/capacity_ceiling.py
"""
#12: Capacity Ceiling
Determine the AUM level where slippage eats the edge
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketMicrostructure:
    """Market microstructure parameters for a trading pair"""
    symbol: str
    avg_spread_bps: float
    avg_depth_usd: float  # Within 10bps of mid
    daily_volume: float
    
    @classmethod
    def get_default(cls, symbol: str) -> 'MarketMicrostructure':
        """Get default parameters for common pairs"""
        defaults = {
            'BTC': cls('BTC', avg_spread_bps=2.0, avg_depth_usd=5_000_000, daily_volume=50_000_000_000),
            'ETH': cls('ETH', avg_spread_bps=3.0, avg_depth_usd=2_000_000, daily_volume=20_000_000_000),
            'SOL': cls('SOL', avg_spread_bps=5.0, avg_depth_usd=500_000, daily_volume=3_000_000_000),
            'BNB': cls('BNB', avg_spread_bps=4.0, avg_depth_usd=800_000, daily_volume=1_500_000_000),
        }
        return defaults.get(symbol.upper(), cls(symbol, 10.0, 100_000, 100_000_000))


class CapacityCeilingCalculator:
    """
    Determine the AUM level where slippage eats the edge.
    Set hard capital limits based on market microstructure.
    """
    
    def __init__(self):
        self.market_params: Dict[str, MarketMicrostructure] = {}
        self._load_default_params()
        
    def _load_default_params(self):
        """Load default market microstructure parameters"""
        for symbol in ['BTC', 'ETH', 'SOL', 'BNB']:
            self.market_params[symbol] = MarketMicrostructure.get_default(symbol)
    
    def update_market_params(self, symbol: str, params: MarketMicrostructure):
        """Update parameters from live orderbook data"""
        self.market_params[symbol] = params
        
    def calculate_capacity_ceiling(self, 
                                   symbol: str,
                                   target_trades_per_day: float,
                                   max_acceptable_slippage_bps: float = 10.0,
                                   max_position_pct: float = 0.25) -> float:
        """
        Calculate maximum AUM before slippage kills edge.
        
        Args:
            symbol: Trading pair (BTC, ETH, etc.)
            target_trades_per_day: Expected trade frequency
            max_acceptable_slippage_bps: Max slippage before edge gone
            max_position_pct: Max position as % of portfolio (default 25%)
            
        Returns:
            Maximum AUM in USD
        """
        params = self.market_params.get(symbol.upper())
        if not params:
            params = MarketMicrostructure.get_default(symbol)
        
        # Model: Slippage = base_spread + (position_size / depth) * impact_coefficient
        base_spread = params.avg_spread_bps
        depth = params.avg_depth_usd
        
        # Solve for max position size given slippage constraint
        remaining_slippage = max_acceptable_slippage_bps - base_spread
        if remaining_slippage <= 0:
            return 0.0  # Spread alone exceeds limit
        
        # Impact coefficient (empirical, typically ~100 for crypto)
        impact_coefficient = 100
        max_position_size = (remaining_slippage * depth) / impact_coefficient
        
        # AUM = position_size / max_position_pct
        max_aum = max_position_size / max_position_pct
        
        return max_aum
    
    def get_current_capacity_utilization(self, 
                                        current_aum: float,
                                        symbol: str,
                                        target_trades_per_day: float = 2.0) -> Dict:
        """
        Check how close to capacity ceiling.
        
        Returns:
            Dict with capacity metrics
        """
        ceiling = self.calculate_capacity_ceiling(
            symbol, 
            target_trades_per_day
        )
        
        utilization = current_aum / ceiling if ceiling > 0 else 1.0
        
        return {
            'current_aum': current_aum,
            'capacity_ceiling': ceiling,
            'utilization_pct': utilization * 100,
            'remaining_capacity': ceiling - current_aum,
            'status': self._get_status(utilization),
            'symbol': symbol,
        }
    
    def _get_status(self, utilization: float) -> str:
        """Get status string based on utilization"""
        if utilization < 0.5:
            return "✅ LOW (healthy)"
        elif utilization < 0.75:
            return "🟡 MEDIUM (monitor)"
        elif utilization < 0.9:
            return "🟠 HIGH (caution)"
        else:
            return "🔴 CRITICAL (stop accepting capital)"
    
    def should_accept_new_capital(self, 
                                  current_aum: float,
                                  new_capital: float,
                                  symbol: str) -> Tuple[bool, str]:
        """
        Determine if can accept new capital without exceeding ceiling.
        
        Returns:
            (should_accept, reason)
        """
        ceiling = self.calculate_capacity_ceiling(symbol, target_trades_per_day=2.0)
        
        if ceiling <= 0:
            return False, "Cannot determine capacity ceiling"
        
        projected_aum = current_aum + new_capital
        utilization = projected_aum / ceiling
        
        if utilization > 0.95:
            return False, f"Would exceed capacity ceiling (${ceiling:,.0f})"
        
        if utilization > 0.85:
            return True, f"⚠️ WARNING: Approaching ceiling ({utilization*100:.0f}% utilized)"
        
        return True, f"✅ Healthy capacity ({utilization*100:.0f}% utilized)"
    
    def calculate_optimal_position_size(self,
                                        current_aum: float,
                                        symbol: str,
                                        target_slippage_bps: float = 5.0) -> float:
        """
        Calculate optimal position size to minimize slippage impact.
        
        Returns:
            Position size as fraction of AUM
        """
        params = self.market_params.get(symbol.upper())
        if not params:
            return 0.1  # Default 10%
        
        depth = params.avg_depth_usd
        base_spread = params.avg_spread_bps
        
        # Max position to stay within target slippage
        remaining = target_slippage_bps - base_spread
        if remaining <= 0:
            return 0.05  # 5% minimum
        
        max_position_usd = (remaining * depth) / 100
        max_position_pct = max_position_usd / current_aum if current_aum > 0 else 0.25
        
        return min(max_position_pct, 0.25)  # Cap at 25%
    
    def generate_capacity_report(self, current_aum: float) -> str:
        """Generate markdown report of capacity status"""
        report = "# Capacity Ceiling Report\n\n"
        
        for symbol in ['BTC', 'ETH', 'SOL']:
            util = self.get_current_capacity_utilization(current_aum, symbol)
            
            report += f"## {symbol} {util['status']}\n"
            report += f"- Capacity Ceiling: ${util['capacity_ceiling']:,.0f}\n"
            report += f"- Current AUM: ${util['current_aum']:,.0f}\n"
            report += f"- Utilization: {util['utilization_pct']:.1f}%\n"
            report += f"- Remaining: ${util['remaining_capacity']:,.0f}\n\n"
        
        return report
