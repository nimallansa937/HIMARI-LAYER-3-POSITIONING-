"""
L3-3: Multi-Period Path Risk
============================
Calculate path-dependent risk metrics over multiple time horizons.

Evaluates:
- Intra-day drawdown risk
- Multi-day cumulative exposure
- Weekend/holiday gap risk
- Path-dependent maximum loss scenarios

File: LAYER 3 V1/LAYER 3 POSITIONING LAYER/src/core/multi_period_path_risk.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import math


class TimeHorizon(Enum):
    """Time horizons for path risk analysis."""
    INTRADAY = "intraday"       # < 1 day
    SHORT = "short"             # 1-5 days
    MEDIUM = "medium"           # 5-20 days
    LONG = "long"               # > 20 days


@dataclass
class PathRiskMetrics:
    """Path-dependent risk metrics."""
    horizon: TimeHorizon
    max_drawdown_path: float      # Max DD along any path
    prob_ruin: float              # Probability of exceeding ruin threshold
    expected_max_loss: float      # Expected maximum loss
    time_to_max_dd: int           # Periods until worst drawdown
    recovery_time: int            # Periods to recover from DD
    gap_risk: float               # Weekend/holiday gap risk
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon.value,
            "max_drawdown_path": self.max_drawdown_path,
            "prob_ruin": self.prob_ruin,
            "expected_max_loss": self.expected_max_loss,
            "time_to_max_dd": self.time_to_max_dd,
            "recovery_time": self.recovery_time,
            "gap_risk": self.gap_risk
        }


class MultiPeriodPathRisk:
    """
    Calculate path-dependent risk across multiple time horizons.
    
    Unlike single-period VaR, this captures:
    1. Cumulative position risk over holding periods
    2. Path-dependency (early DD vs late DD)
    3. Gap risk from market closures (crypto = 24/7, but exchange maintenance)
    4. Compounding effects over multiple periods
    """
    
    def __init__(
        self,
        ruin_threshold: float = 0.15,         # 15% max acceptable DD
        gap_risk_multiplier: float = 1.5,     # Overnight gap risk multiplier
        num_simulations: int = 1000
    ):
        self.ruin_threshold = ruin_threshold
        self.gap_risk_multiplier = gap_risk_multiplier
        self.num_simulations = num_simulations
    
    def calculate_path_risk(
        self,
        position_size: float,
        volatility: float,
        holding_periods: int,
        drift: float = 0.0
    ) -> PathRiskMetrics:
        """
        Calculate path risk for a position over holding period.
        
        Uses simplified GBM model for paths.
        """
        import random
        
        # Determine horizon
        if holding_periods <= 1:
            horizon = TimeHorizon.INTRADAY
        elif holding_periods <= 5:
            horizon = TimeHorizon.SHORT
        elif holding_periods <= 20:
            horizon = TimeHorizon.MEDIUM
        else:
            horizon = TimeHorizon.LONG
        
        # Monte Carlo path simulation
        max_dds = []
        ruin_count = 0
        max_losses = []
        time_to_max_dds = []
        
        for _ in range(self.num_simulations):
            path_value = 1.0
            peak = 1.0
            max_dd = 0.0
            max_dd_time = 0
            
            for t in range(holding_periods):
                # Daily return (normal approximation)
                daily_vol = volatility / math.sqrt(252)
                daily_drift = drift / 252
                ret = random.gauss(daily_drift, daily_vol)
                
                path_value *= (1 + ret)
                
                # Track peak and drawdown
                if path_value > peak:
                    peak = path_value
                dd = (peak - path_value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_time = t
            
            max_dds.append(max_dd)
            max_losses.append(1 - path_value if path_value < 1 else 0)
            time_to_max_dds.append(max_dd_time)
            
            if max_dd >= self.ruin_threshold:
                ruin_count += 1
        
        # Calculate metrics
        avg_max_dd = sum(max_dds) / len(max_dds)
        prob_ruin = ruin_count / self.num_simulations
        expected_max_loss = sum(max_losses) / len(max_losses) * position_size
        avg_time_to_max_dd = sum(time_to_max_dds) / len(time_to_max_dds)
        
        # Estimate recovery time (simplified)
        recovery_time = int(avg_max_dd * 252 / abs(drift)) if drift > 0.001 else holding_periods
        
        # Gap risk for crypto (maintenance windows)
        gap_risk = volatility * self.gap_risk_multiplier / math.sqrt(252)
        
        return PathRiskMetrics(
            horizon=horizon,
            max_drawdown_path=avg_max_dd,
            prob_ruin=prob_ruin,
            expected_max_loss=expected_max_loss,
            time_to_max_dd=int(avg_time_to_max_dd),
            recovery_time=recovery_time,
            gap_risk=gap_risk
        )
    
    def adjust_position_for_path_risk(
        self,
        base_position: float,
        path_risk: PathRiskMetrics
    ) -> float:
        """Reduce position based on path risk."""
        # Reduce position if prob_ruin is significant
        if path_risk.prob_ruin > 0.05:  # > 5% chance of ruin
            reduction = 1 - (path_risk.prob_ruin * 2)
            return base_position * max(0.2, reduction)
        return base_position


@dataclass
class RegimePositionMultipliers:
    """Position size multipliers by regime."""
    trending_up: float = 1.2
    trending_down: float = 0.8
    ranging: float = 1.0
    high_volatility: float = 0.6
    crisis: float = 0.3


class RegimePositionSizer:
    """
    L3-4: Regime-Dependent Position Sizing
    
    Scale positions based on detected market regime from L1/L2.
    """
    
    def __init__(self, multipliers: Optional[RegimePositionMultipliers] = None):
        self.multipliers = multipliers or RegimePositionMultipliers()
        self.current_regime: str = "ranging"
    
    def set_regime(self, regime: str) -> None:
        """Update current regime from L2 regime detection."""
        self.current_regime = regime.lower().replace(" ", "_")
    
    def get_multiplier(self) -> float:
        """Get position multiplier for current regime."""
        return getattr(self.multipliers, self.current_regime, 1.0)
    
    def adjust_position(self, base_position: float) -> float:
        """Apply regime-based position adjustment."""
        return base_position * self.get_multiplier()


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    timestamp: datetime
    constraint_name: str
    constraint_limit: float
    actual_value: float
    severity: str  # "warning" or "critical"
    action_taken: str


class RealTimeConstraintMonitor:
    """
    L3-5: Real-Time Constraint Monitoring
    
    Continuously monitor all position constraints and alert on violations.
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_weekly_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_position_concentration: float = 0.25,
        max_correlation: float = 0.7
    ):
        self.constraints = {
            "daily_loss": max_daily_loss_pct,
            "weekly_loss": max_weekly_loss_pct,
            "drawdown": max_drawdown_pct,
            "concentration": max_position_concentration,
            "correlation": max_correlation
        }
        
        # State
        self.current_values: Dict[str, float] = {}
        self.violations: List[ConstraintViolation] = []
        self.alerts_enabled = True
    
    def update_value(self, constraint_name: str, value: float) -> Optional[ConstraintViolation]:
        """Update a constraint value and check for violations."""
        self.current_values[constraint_name] = value
        
        if constraint_name not in self.constraints:
            return None
        
        limit = self.constraints[constraint_name]
        
        # Check for violation (assuming constraints are upper limits)
        if value >= limit:
            severity = "critical" if value >= limit else "warning"
            action = "HALT_TRADING" if severity == "critical" else "REDUCE_POSITION"
            
            violation = ConstraintViolation(
                timestamp=datetime.now(),
                constraint_name=constraint_name,
                constraint_limit=limit,
                actual_value=value,
                severity=severity,
                action_taken=action
            )
            self.violations.append(violation)
            return violation
        
        # Warning at 80% of limit
        if value >= limit * 0.8:
            return ConstraintViolation(
                timestamp=datetime.now(),
                constraint_name=constraint_name,
                constraint_limit=limit,
                actual_value=value,
                severity="warning",
                action_taken="MONITOR"
            )
        
        return None
    
    def check_all(self, metrics: Dict[str, float]) -> List[ConstraintViolation]:
        """Check all constraints at once."""
        violations = []
        for name, value in metrics.items():
            v = self.update_value(name, value)
            if v:
                violations.append(v)
        return violations
    
    def get_status(self) -> Dict[str, Any]:
        """Get current constraint status."""
        status = {}
        for name, limit in self.constraints.items():
            current = self.current_values.get(name, 0.0)
            utilization = current / limit if limit > 0 else 0
            status[name] = {
                "limit": limit,
                "current": current,
                "utilization_pct": utilization * 100,
                "status": "OK" if utilization < 0.8 else ("WARNING" if utilization < 1.0 else "CRITICAL")
            }
        return status
    
    def should_halt_trading(self) -> bool:
        """Check if any critical constraint is violated."""
        for name, limit in self.constraints.items():
            current = self.current_values.get(name, 0.0)
            if current >= limit:
                return True
        return False


if __name__ == "__main__":
    # Test Multi-Period Path Risk
    path_risk_calc = MultiPeriodPathRisk()
    
    metrics = path_risk_calc.calculate_path_risk(
        position_size=10000,
        volatility=0.25,
        holding_periods=5,
        drift=0.10
    )
    
    print("Multi-Period Path Risk:")
    print(f"  Horizon: {metrics.horizon.value}")
    print(f"  Max DD Path: {metrics.max_drawdown_path:.2%}")
    print(f"  P(Ruin): {metrics.prob_ruin:.2%}")
    print(f"  Expected Max Loss: ${metrics.expected_max_loss:,.0f}")
    
    # Test Regime Position Sizer
    regime_sizer = RegimePositionSizer()
    regime_sizer.set_regime("high_volatility")
    adjusted = regime_sizer.adjust_position(10000)
    print(f"\nRegime-Adjusted Position: ${adjusted:,.0f}")
    
    # Test Real-Time Constraint Monitor
    monitor = RealTimeConstraintMonitor()
    violations = monitor.check_all({
        "daily_loss": 0.015,  # 1.5% (warning)
        "drawdown": 0.12,     # 12% (warning at 80%)
    })
    print(f"\nConstraint Status: {monitor.get_status()}")
