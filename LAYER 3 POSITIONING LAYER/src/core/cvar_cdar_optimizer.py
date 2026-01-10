"""
L3-2: CVaR/CDaR Optimization
============================
Replace variance-based risk measures with tail-focused metrics:
- CVaR (Conditional Value-at-Risk): Expected loss given that loss exceeds VaR threshold
- CDaR (Conditional Drawdown-at-Risk): Expected drawdown given drawdown exceeds threshold

Why CVaR > Variance:
- Variance treats upside and downside symmetrically (wrong for trading)
- Variance underweights tail events (exactly what kills accounts)
- CVaR directly measures "how bad can it get?"

File: LAYER 3 V1/LAYER 3 POSITIONING LAYER/src/core/cvar_cdar_optimizer.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import math
import random


@dataclass
class RiskMetrics:
    """Tail risk metrics."""
    var: float          # Value at Risk
    cvar: float         # Conditional Value at Risk
    cdar: float         # Conditional Drawdown at Risk
    max_loss: float     # Maximum loss in scenarios
    max_drawdown: float # Maximum drawdown in scenarios
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "var": self.var,
            "cvar": self.cvar,
            "cdar": self.cdar,
            "max_loss": self.max_loss,
            "max_drawdown": self.max_drawdown
        }


class CVaRCalculator:
    """
    Calculate Conditional Value at Risk.
    
    CVaR_α = E[Loss | Loss > VaR_α]
    """
    
    def __init__(
        self,
        confidence: float = 0.95,
        num_scenarios: int = 1000
    ):
        self.confidence = confidence
        self.num_scenarios = num_scenarios
    
    def calculate(self, returns: List[float]) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR from return distribution.
        
        Returns:
            (VaR, CVaR) as positive loss values
        """
        if not returns:
            return 0.0, 0.0
        
        # Sort returns (worst first, as losses)
        losses = sorted([-r for r in returns], reverse=True)
        
        # VaR: loss at (1-confidence) percentile
        var_index = int(len(losses) * (1 - self.confidence))
        var_index = max(0, min(var_index, len(losses) - 1))
        var = losses[var_index]
        
        # CVaR: mean of losses worse than VaR
        tail_losses = [l for l in losses if l >= var]
        if tail_losses:
            cvar = sum(tail_losses) / len(tail_losses)
        else:
            cvar = var
        
        return max(0, var), max(0, cvar)
    
    def estimate_position_cvar(
        self,
        position_size: float,
        expected_return: float,
        volatility: float,
        holding_period_days: int = 1
    ) -> Tuple[float, float]:
        """
        Estimate CVaR for a position using Monte Carlo simulation.
        
        Returns:
            (VaR, CVaR) for the position
        """
        # Generate scenarios
        scenarios = []
        daily_vol = volatility / math.sqrt(252)
        
        for _ in range(self.num_scenarios):
            # Simulate cumulative return over holding period
            cumulative = 0.0
            for day in range(holding_period_days):
                daily_return = random.gauss(
                    expected_return / 252,
                    daily_vol
                )
                cumulative = (1 + cumulative) * (1 + daily_return) - 1
            
            # Position P&L
            pnl = position_size * cumulative
            scenarios.append(pnl)
        
        return self.calculate(scenarios)


class CDaRCalculator:
    """
    Calculate Conditional Drawdown at Risk.
    
    CDaR_α = E[Drawdown | Drawdown > DD_α]
    """
    
    def __init__(
        self,
        confidence: float = 0.95
    ):
        self.confidence = confidence
    
    def calculate_drawdowns(self, equity_curve: List[float]) -> List[float]:
        """Calculate rolling drawdowns from equity curve."""
        if not equity_curve:
            return []
        
        drawdowns = []
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            drawdowns.append(dd)
        
        return drawdowns
    
    def calculate(self, drawdowns: List[float]) -> Tuple[float, float]:
        """
        Calculate DD@R and CDaR from drawdown distribution.
        
        Returns:
            (DD@R at confidence, CDaR)
        """
        if not drawdowns:
            return 0.0, 0.0
        
        # Sort drawdowns (worst first)
        sorted_dd = sorted(drawdowns, reverse=True)
        
        # DD@R: drawdown at (1-confidence) percentile
        ddar_index = int(len(sorted_dd) * (1 - self.confidence))
        ddar_index = max(0, min(ddar_index, len(sorted_dd) - 1))
        ddar = sorted_dd[ddar_index]
        
        # CDaR: mean of drawdowns worse than DD@R
        tail_dd = [d for d in sorted_dd if d >= ddar]
        if tail_dd:
            cdar = sum(tail_dd) / len(tail_dd)
        else:
            cdar = ddar
        
        return ddar, cdar


class CVaRCDaROptimizer:
    """
    Position sizing optimizer with CVaR and CDaR constraints.
    
    Optimization Framework:
    MAXIMIZE: Expected Return
    SUBJECT TO:
        CVaR_95 ≤ MAX_CVAR_BUDGET
        CDaR_95 ≤ MAX_CDADR_BUDGET
        Position constraints from L3-1
    
    Configuration Parameters:
    - cvar_confidence: CVaR confidence level (default: 0.95)
    - cvar_budget_pct: Max allowed CVaR (% of equity) (default: 0.05)
    - cdar_confidence: CDaR confidence level (default: 0.95)
    - cdar_budget_pct: Max allowed CDaR (% of equity) (default: 0.10)
    - simulation_scenarios: Scenarios for CVaR calculation (default: 1000)
    - lookback_days: Historical data for simulation (default: 252)
    """
    
    def __init__(
        self,
        cvar_confidence: float = 0.95,
        cvar_budget_pct: float = 0.05,
        cdar_confidence: float = 0.95,
        cdar_budget_pct: float = 0.10,
        simulation_scenarios: int = 1000,
        lookback_days: int = 252
    ):
        self.cvar_calc = CVaRCalculator(cvar_confidence, simulation_scenarios)
        self.cdar_calc = CDaRCalculator(cdar_confidence)
        
        self.cvar_budget_pct = cvar_budget_pct
        self.cdar_budget_pct = cdar_budget_pct
        self.lookback_days = lookback_days
        
        # Historical data
        self.historical_returns: List[float] = []
        self.historical_equity: List[float] = []
    
    def update_history(
        self,
        daily_return: float,
        equity: float
    ) -> None:
        """Add daily data point to history."""
        self.historical_returns.append(daily_return)
        self.historical_equity.append(equity)
        
        # Keep only lookback period
        if len(self.historical_returns) > self.lookback_days:
            self.historical_returns = self.historical_returns[-self.lookback_days:]
            self.historical_equity = self.historical_equity[-self.lookback_days:]
    
    def calculate_current_risk(self) -> RiskMetrics:
        """Calculate current risk metrics from history."""
        if not self.historical_returns:
            return RiskMetrics(
                var=0.0, cvar=0.0, cdar=0.0,
                max_loss=0.0, max_drawdown=0.0
            )
        
        # CVaR from returns
        var, cvar = self.cvar_calc.calculate(self.historical_returns)
        
        # CDaR from equity curve
        drawdowns = self.cdar_calc.calculate_drawdowns(self.historical_equity)
        _, cdar = self.cdar_calc.calculate(drawdowns)
        
        return RiskMetrics(
            var=var,
            cvar=cvar,
            cdar=cdar,
            max_loss=max(-r for r in self.historical_returns) if self.historical_returns else 0,
            max_drawdown=max(drawdowns) if drawdowns else 0
        )
    
    def find_cvar_compliant_size(
        self,
        max_position: float,
        expected_return: float,
        volatility: float,
        equity: float,
        holding_period_days: int = 1,
        step_size: float = 0.1
    ) -> Tuple[float, RiskMetrics]:
        """
        Find maximum position size that satisfies CVaR constraint.
        
        Uses binary search to find largest size where CVaR ≤ budget.
        
        Returns:
            (cvar_compliant_size, risk_metrics)
        """
        cvar_budget = equity * self.cvar_budget_pct
        
        # Binary search
        low = 0.0
        high = max_position
        best_size = 0.0
        best_metrics = None
        
        for _ in range(20):  # Max iterations
            mid = (low + high) / 2
            
            var, cvar = self.cvar_calc.estimate_position_cvar(
                mid, expected_return, volatility, holding_period_days
            )
            
            if cvar <= cvar_budget:
                best_size = mid
                best_metrics = RiskMetrics(
                    var=var, cvar=cvar, cdar=0.0,
                    max_loss=cvar * 1.5, max_drawdown=0.0
                )
                low = mid
            else:
                high = mid
            
            # Converged
            if high - low < max_position * 0.01:
                break
        
        if best_metrics is None:
            best_metrics = RiskMetrics(
                var=0.0, cvar=0.0, cdar=0.0,
                max_loss=0.0, max_drawdown=0.0
            )
        
        return best_size, best_metrics
    
    def find_cdar_compliant_size(
        self,
        max_position: float,
        historical_max_dd: float
    ) -> float:
        """
        Adjust position size based on CDaR constraint.
        
        Simple approach: reduce size proportionally if CDaR budget would be exceeded.
        """
        if historical_max_dd <= 0:
            return max_position
        
        # If historical max DD exceeds budget, scale down
        if historical_max_dd > self.cdar_budget_pct:
            scaling = self.cdar_budget_pct / historical_max_dd
            return max_position * scaling
        
        return max_position
    
    def optimize_position(
        self,
        max_from_other_constraints: float,
        expected_return: float,
        volatility: float,
        equity: float,
        holding_period_days: int = 1
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal position size subject to CVaR and CDaR constraints.
        
        Returns:
            (optimal_size, optimization_details)
        """
        # Get CVaR-compliant size
        cvar_size, cvar_metrics = self.find_cvar_compliant_size(
            max_from_other_constraints,
            expected_return,
            volatility,
            equity,
            holding_period_days
        )
        
        # Get CDaR-compliant size
        current_risk = self.calculate_current_risk()
        cdar_size = self.find_cdar_compliant_size(
            cvar_size,
            current_risk.max_drawdown
        )
        
        # Final size is minimum of all constraints
        optimal_size = min(cvar_size, cdar_size)
        
        details = {
            "max_from_other_constraints": max_from_other_constraints,
            "cvar_compliant_size": cvar_size,
            "cdar_compliant_size": cdar_size,
            "optimal_size": optimal_size,
            "cvar_metrics": cvar_metrics.to_dict(),
            "current_portfolio_risk": current_risk.to_dict(),
            "cvar_budget": equity * self.cvar_budget_pct,
            "cdar_budget": self.cdar_budget_pct
        }
        
        return optimal_size, details
    
    def check_constraints(
        self,
        proposed_size: float,
        expected_return: float,
        volatility: float,
        equity: float
    ) -> Tuple[bool, str]:
        """
        Check if proposed position size satisfies CVaR/CDaR constraints.
        
        Returns:
            (is_compliant, message)
        """
        # Check CVaR
        var, cvar = self.cvar_calc.estimate_position_cvar(
            proposed_size, expected_return, volatility
        )
        
        cvar_budget = equity * self.cvar_budget_pct
        if cvar > cvar_budget:
            return False, f"CVaR {cvar:.2f} exceeds budget {cvar_budget:.2f}"
        
        # Check CDaR (using historical)
        current_risk = self.calculate_current_risk()
        if current_risk.cdar > self.cdar_budget_pct:
            return False, f"CDaR {current_risk.cdar:.2%} exceeds budget {self.cdar_budget_pct:.2%}"
        
        return True, f"Position compliant: CVaR={cvar:.2f}, CDaR={current_risk.cdar:.2%}"


if __name__ == "__main__":
    # Quick test
    optimizer = CVaRCDaROptimizer(
        cvar_budget_pct=0.05,
        cdar_budget_pct=0.10
    )
    
    # Simulate some history
    equity = 100000.0
    for _ in range(100):
        daily_ret = random.gauss(0.001, 0.02)
        equity *= (1 + daily_ret)
        optimizer.update_history(daily_ret, equity)
    
    # Find optimal position
    optimal, details = optimizer.optimize_position(
        max_from_other_constraints=50000.0,
        expected_return=0.10,  # 10% annualized
        volatility=0.25,       # 25% annualized
        equity=equity
    )
    
    print(f"Optimal position size: ${optimal:,.2f}")
    print(f"\nDetails:")
    for k, v in details.items():
        print(f"  {k}: {v}")
    
    # Check current risk
    risk = optimizer.calculate_current_risk()
    print(f"\nCurrent portfolio risk: {risk.to_dict()}")
