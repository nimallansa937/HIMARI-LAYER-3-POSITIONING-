# src/core/minimum_edge_calculator.py
"""
#10: Minimum Viable Edge Threshold
Calculate the minimum Sharpe ratio needed to cover trading costs
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """All costs of trading"""
    exchange_fee_bps: float = 10.0     # 10 bps (0.1%) for taker
    slippage_bps: float = 5.0           # 5 bps average slippage
    funding_cost_bps: float = 2.0       # Holding cost per trade
    opportunity_cost_bps: float = 3.0   # vs holding BTC/ETH
    
    @property
    def total_cost_bps(self) -> float:
        """Total cost per trade in basis points"""
        return (self.exchange_fee_bps + self.slippage_bps + 
                self.funding_cost_bps + self.opportunity_cost_bps)
    
    @property
    def total_cost_pct(self) -> float:
        """Total cost per trade as decimal"""
        return self.total_cost_bps / 10000
    
    @classmethod
    def from_exchange(cls, exchange: str) -> 'TradingCosts':
        """Get typical costs for specific exchange"""
        exchange_costs = {
            'binance': cls(exchange_fee_bps=10.0, slippage_bps=3.0),
            'bybit': cls(exchange_fee_bps=10.0, slippage_bps=5.0),
            'okx': cls(exchange_fee_bps=8.0, slippage_bps=4.0),
            'dydx': cls(exchange_fee_bps=5.0, slippage_bps=8.0),
        }
        return exchange_costs.get(exchange.lower(), cls())


class MinimumEdgeCalculator:
    """
    Calculate the minimum Sharpe ratio needed to cover trading costs.
    Below this threshold, holding beats trading.
    """
    
    def __init__(self, costs: Optional[TradingCosts] = None,
                 crypto_volatility: float = 0.6,
                 risk_free_rate: float = 0.04):
        """
        Args:
            costs: Trading costs configuration
            crypto_volatility: Assumed annual volatility (60% default)
            risk_free_rate: Risk-free rate (4% default)
        """
        self.costs = costs or TradingCosts()
        self.crypto_volatility = crypto_volatility
        self.risk_free_rate = risk_free_rate
        self.edge_check_log: list = []
        
    def calculate_breakeven_sharpe(self, 
                                   avg_trades_per_day: float,
                                   avg_holding_period_hours: float = 24.0) -> float:
        """
        Calculate Sharpe ratio needed to break even with costs.
        
        Args:
            avg_trades_per_day: Average number of trades per day
            avg_holding_period_hours: Average hours per position
            
        Returns:
            Minimum Sharpe ratio to beat holding
        """
        # Annual trading costs
        trades_per_year = avg_trades_per_day * 252
        cost_per_trade = self.costs.total_cost_pct
        annual_cost = trades_per_year * cost_per_trade
        
        # Breakeven Sharpe = (Cost + Risk-free rate) / Volatility
        breakeven_sharpe = (annual_cost + self.risk_free_rate) / self.crypto_volatility
        
        return breakeven_sharpe
    
    def calculate_required_sharpe(self, 
                                  avg_trades_per_day: float,
                                  buffer_multiplier: float = 1.5) -> float:
        """
        Calculate required Sharpe with safety buffer.
        
        Args:
            avg_trades_per_day: Average number of trades per day
            buffer_multiplier: Safety buffer (default 1.5x breakeven)
            
        Returns:
            Required Sharpe with buffer
        """
        breakeven = self.calculate_breakeven_sharpe(avg_trades_per_day)
        return breakeven * buffer_multiplier
    
    def should_trade(self, current_sharpe: float, 
                    avg_trades_per_day: float,
                    avg_holding_period_hours: float = 24.0) -> bool:
        """
        Determine if edge is sufficient to justify trading.
        
        Returns:
            True if should trade, False if should hold cash
        """
        breakeven = self.calculate_breakeven_sharpe(
            avg_trades_per_day, 
            avg_holding_period_hours
        )
        
        # Add 50% buffer
        threshold = breakeven * 1.5
        
        result = current_sharpe >= threshold
        
        # Log the check
        log_entry = {
            'current_sharpe': current_sharpe,
            'breakeven_sharpe': breakeven,
            'threshold': threshold,
            'should_trade': result,
        }
        self.edge_check_log.append(log_entry)
        
        if not result:
            logger.warning(f"[MINIMUM EDGE VIOLATION] Sharpe {current_sharpe:.2f} < {threshold:.2f}")
            print(f"[MINIMUM EDGE VIOLATION]")
            print(f"  Current Sharpe: {current_sharpe:.2f}")
            print(f"  Breakeven Sharpe: {breakeven:.2f}")
            print(f"  Required Sharpe (1.5x buffer): {threshold:.2f}")
            print(f"  Decision: HOLD CASH")
        
        return result
    
    def calculate_edge_decay_runway(self, 
                                    current_sharpe: float,
                                    decay_rate_per_month: float,
                                    avg_trades_per_day: float) -> int:
        """
        Estimate months until edge decays below minimum threshold.
        
        Returns:
            Number of months until should stop trading
        """
        breakeven = self.calculate_breakeven_sharpe(avg_trades_per_day)
        threshold = breakeven * 1.5
        
        if current_sharpe <= threshold:
            return 0  # Already below threshold
        
        if decay_rate_per_month <= 0:
            return 999  # No decay
        
        months_remaining = (current_sharpe - threshold) / decay_rate_per_month
        return int(months_remaining)
    
    def calculate_max_trades_per_day(self, target_sharpe: float,
                                     required_buffer: float = 1.5) -> float:
        """
        Calculate max trades/day before costs eat the edge.
        
        Args:
            target_sharpe: Expected strategy Sharpe
            required_buffer: Safety buffer multiplier
            
        Returns:
            Maximum trades per day
        """
        # Solve: target_sharpe / buffer = (trades * cost + rf) / vol
        # trades * cost = target_sharpe * vol / buffer - rf
        # trades = (target_sharpe * vol / buffer - rf) / (cost * 252)
        
        cost_per_trade = self.costs.total_cost_pct
        
        numerator = (target_sharpe * self.crypto_volatility / required_buffer) - self.risk_free_rate
        denominator = cost_per_trade * 252
        
        if denominator <= 0:
            return float('inf')
        
        max_trades = numerator / denominator
        return max(0, max_trades)
    
    def generate_edge_report(self, current_sharpe: float,
                            avg_trades_per_day: float) -> str:
        """Generate markdown report of edge analysis"""
        breakeven = self.calculate_breakeven_sharpe(avg_trades_per_day)
        threshold = breakeven * 1.5
        max_trades = self.calculate_max_trades_per_day(current_sharpe)
        
        report = "# Minimum Edge Analysis\n\n"
        
        # Status
        if current_sharpe >= threshold:
            report += "## Status: ✅ EDGE SUFFICIENT\n\n"
        else:
            report += "## Status: ❌ EDGE INSUFFICIENT\n\n"
        
        # Metrics
        report += "## Metrics\n"
        report += f"- Current Sharpe: {current_sharpe:.2f}\n"
        report += f"- Breakeven Sharpe: {breakeven:.2f}\n"
        report += f"- Required Sharpe (1.5x): {threshold:.2f}\n"
        report += f"- Buffer Above Threshold: {(current_sharpe - threshold):.2f}\n\n"
        
        # Trading costs
        report += "## Trading Costs\n"
        report += f"- Exchange Fee: {self.costs.exchange_fee_bps:.1f} bps\n"
        report += f"- Slippage: {self.costs.slippage_bps:.1f} bps\n"
        report += f"- Total per Trade: {self.costs.total_cost_bps:.1f} bps\n"
        report += f"- Trades per Day: {avg_trades_per_day:.1f}\n"
        report += f"- Max Trades at Current Edge: {max_trades:.1f}\n\n"
        
        return report
