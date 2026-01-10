"""
L3-1: Inverted Risk Hierarchy
=============================
Enforce a strict hierarchy where capital preservation constraints are satisfied
BEFORE any return optimization occurs. Position sizing becomes a constraint
satisfaction problem, not a maximization problem.

Constraint Hierarchy (evaluated in order):
- Level 1: HARD CAPITAL CONSTRAINTS (non-negotiable)
- Level 2: LIQUIDITY CONSTRAINTS
- Level 3: VOLATILITY TARGETING
- Level 4: ALPHA OPTIMIZATION

File: LAYER 3 V1/LAYER 3 POSITIONING LAYER/src/core/inverted_risk_hierarchy.py
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import math


class ConstraintLevel(Enum):
    """Constraint hierarchy levels."""
    L1_HARD_CAPITAL = 1
    L2_LIQUIDITY = 2
    L3_VOLATILITY = 3
    L4_ALPHA = 4


class ConstraintStatus(Enum):
    """Status of constraint evaluation."""
    PASSED = "passed"
    VIOLATED = "violated"
    WARNING = "warning"


@dataclass
class HardCapitalConstraints:
    """Level 1: Non-negotiable capital constraints."""
    max_daily_loss_pct: float = 0.02      # 2% of equity
    max_weekly_loss_pct: float = 0.05     # 5% of equity
    max_drawdown_pct: float = 0.15        # 15% of peak equity
    
    # Current state
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    
    def update_equity(self, equity: float) -> None:
        """Update equity and recalculate drawdown."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
    
    def check_all(self) -> Tuple[ConstraintStatus, str]:
        """Check all hard capital constraints."""
        # Daily loss check
        daily_loss = -self.daily_pnl / self.current_equity if self.current_equity > 0 else 0
        if daily_loss >= self.max_daily_loss_pct:
            return ConstraintStatus.VIOLATED, f"Daily loss {daily_loss:.2%} >= {self.max_daily_loss_pct:.2%}"
        
        # Weekly loss check
        weekly_loss = -self.weekly_pnl / self.current_equity if self.current_equity > 0 else 0
        if weekly_loss >= self.max_weekly_loss_pct:
            return ConstraintStatus.VIOLATED, f"Weekly loss {weekly_loss:.2%} >= {self.max_weekly_loss_pct:.2%}"
        
        # Drawdown check
        if self.current_drawdown >= self.max_drawdown_pct:
            return ConstraintStatus.VIOLATED, f"Drawdown {self.current_drawdown:.2%} >= {self.max_drawdown_pct:.2%}"
        
        # Warning if approaching limits
        if daily_loss >= self.max_daily_loss_pct * 0.8:
            return ConstraintStatus.WARNING, f"Approaching daily loss limit: {daily_loss:.2%}"
        
        return ConstraintStatus.PASSED, "All capital constraints satisfied"


@dataclass
class LiquidityConstraints:
    """Level 2: Liquidity constraints."""
    max_position_vs_adv: float = 0.01     # 1% of 24h volume
    max_position_vs_depth: float = 0.10   # 10% of book depth (±2%)
    max_slippage_bps: float = 10.0        # 10 basis points
    
    def calculate_max_size(
        self,
        adv_24h: float,
        book_depth: float,
        estimated_slippage_per_unit: float
    ) -> Tuple[float, str]:
        """
        Calculate maximum position size from liquidity constraints.
        
        Returns:
            (max_size, limiting_constraint)
        """
        # ADV constraint
        adv_max = adv_24h * self.max_position_vs_adv
        
        # Book depth constraint
        depth_max = book_depth * self.max_position_vs_depth
        
        # Slippage constraint
        if estimated_slippage_per_unit > 0:
            slippage_max = (self.max_slippage_bps / 10000) / estimated_slippage_per_unit
        else:
            slippage_max = float('inf')
        
        # Find most restrictive
        if adv_max <= depth_max and adv_max <= slippage_max:
            return adv_max, "adv_constraint"
        elif depth_max <= slippage_max:
            return depth_max, "depth_constraint"
        else:
            return slippage_max, "slippage_constraint"


@dataclass
class VolatilityTargeting:
    """Level 3: Volatility targeting."""
    target_portfolio_vol: float = 0.15    # 15% annualized
    vol_lookback_days: int = 30
    min_vol_floor: float = 0.05           # Minimum vol assumption
    max_vol_cap: float = 1.0              # Maximum vol assumption
    
    def calculate_vol_adjusted_size(
        self,
        base_size: float,
        realized_vol: float
    ) -> float:
        """
        Calculate volatility-adjusted position size.
        
        POSITION_SIZE = BASE_SIZE × (TARGET_VOL / REALIZED_VOL)
        """
        # Floor and cap volatility
        vol = max(self.min_vol_floor, min(self.max_vol_cap, realized_vol))
        
        # Calculate scaling factor
        scaling = self.target_portfolio_vol / vol
        
        # Apply scaling (but don't increase above base)
        return base_size * min(1.0, scaling)


@dataclass
class AlphaOptimization:
    """Level 4: Kelly-based alpha optimization."""
    kelly_fraction_cap: float = 0.5       # Half-Kelly maximum
    min_edge_for_trade: float = 0.001     # 0.1% minimum edge
    
    def calculate_kelly_size(
        self,
        edge: float,
        variance: float,
        equity: float,
        remaining_capacity: float
    ) -> float:
        """
        Calculate Kelly-optimal position within remaining capacity.
        
        KELLY_FRACTION = edge / variance
        APPLIED_FRACTION = min(KELLY_FRACTION × 0.5, REMAINING_CAPACITY)
        """
        if edge < self.min_edge_for_trade or variance <= 0:
            return 0.0
        
        # Full Kelly fraction
        full_kelly = edge / variance
        
        # Apply half-Kelly cap
        capped_kelly = min(full_kelly * self.kelly_fraction_cap, self.kelly_fraction_cap)
        
        # Calculate position size
        kelly_size = capped_kelly * equity
        
        # Respect remaining capacity
        return min(kelly_size, remaining_capacity)


@dataclass
class ConstraintEvaluation:
    """Result of constraint evaluation."""
    level: ConstraintLevel
    status: ConstraintStatus
    message: str
    max_size_allowed: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "status": self.status.value,
            "message": self.message,
            "max_size_allowed": self.max_size_allowed
        }


class InvertedRiskHierarchy:
    """
    Inverted Risk Hierarchy position sizer.
    
    Evaluates constraints in strict order:
    1. L1 Hard Capital - if violated, size = 0
    2. L2 Liquidity - sets liquidity_max
    3. L3 Volatility - sets vol_target_size
    4. L4 Alpha - sets kelly_size
    
    FINAL_SIZE = min(liquidity_max, vol_target_size, kelly_size)
    """
    
    def __init__(
        self,
        hard_constraints: Optional[HardCapitalConstraints] = None,
        liquidity_constraints: Optional[LiquidityConstraints] = None,
        volatility_targeting: Optional[VolatilityTargeting] = None,
        alpha_optimization: Optional[AlphaOptimization] = None
    ):
        self.hard = hard_constraints or HardCapitalConstraints()
        self.liquidity = liquidity_constraints or LiquidityConstraints()
        self.volatility = volatility_targeting or VolatilityTargeting()
        self.alpha = alpha_optimization or AlphaOptimization()
        
        # Evaluation history
        self.evaluations: List[Dict[str, Any]] = []
    
    def update_equity_state(
        self,
        current_equity: float,
        daily_pnl: float,
        weekly_pnl: float
    ) -> None:
        """Update equity state for constraint checking."""
        self.hard.current_equity = current_equity
        self.hard.daily_pnl = daily_pnl
        self.hard.weekly_pnl = weekly_pnl
        self.hard.update_equity(current_equity)
    
    def calculate_position_size(
        self,
        signal_edge: float,
        signal_variance: float,
        realized_volatility: float,
        adv_24h: float,
        book_depth: float,
        estimated_slippage_per_unit: float = 0.0001,
        base_position_size: Optional[float] = None
    ) -> Tuple[float, List[ConstraintEvaluation]]:
        """
        Calculate position size through the inverted hierarchy.
        
        Returns:
            (final_size, list_of_evaluations)
        """
        evaluations = []
        
        # Level 1: Hard Capital Constraints
        l1_status, l1_message = self.hard.check_all()
        evaluations.append(ConstraintEvaluation(
            level=ConstraintLevel.L1_HARD_CAPITAL,
            status=l1_status,
            message=l1_message
        ))
        
        if l1_status == ConstraintStatus.VIOLATED:
            # Hard constraint violated - no position allowed
            self._record_evaluation(evaluations, 0.0)
            return 0.0, evaluations
        
        # Level 2: Liquidity Constraints
        liquidity_max, limiting = self.liquidity.calculate_max_size(
            adv_24h, book_depth, estimated_slippage_per_unit
        )
        evaluations.append(ConstraintEvaluation(
            level=ConstraintLevel.L2_LIQUIDITY,
            status=ConstraintStatus.PASSED,
            message=f"Liquidity max: {liquidity_max:.4f} ({limiting})",
            max_size_allowed=liquidity_max
        ))
        
        # Level 3: Volatility Targeting
        base_size = base_position_size or liquidity_max
        vol_target_size = self.volatility.calculate_vol_adjusted_size(
            base_size, realized_volatility
        )
        evaluations.append(ConstraintEvaluation(
            level=ConstraintLevel.L3_VOLATILITY,
            status=ConstraintStatus.PASSED,
            message=f"Vol-adjusted size: {vol_target_size:.4f} (vol={realized_volatility:.2%})",
            max_size_allowed=vol_target_size
        ))
        
        # Level 4: Alpha Optimization
        remaining_capacity = min(liquidity_max, vol_target_size)
        kelly_size = self.alpha.calculate_kelly_size(
            signal_edge,
            signal_variance,
            self.hard.current_equity,
            remaining_capacity
        )
        evaluations.append(ConstraintEvaluation(
            level=ConstraintLevel.L4_ALPHA,
            status=ConstraintStatus.PASSED,
            message=f"Kelly size: {kelly_size:.4f} (edge={signal_edge:.4f})",
            max_size_allowed=kelly_size
        ))
        
        # Final size: minimum of all constraints
        final_size = min(liquidity_max, vol_target_size, kelly_size)
        
        self._record_evaluation(evaluations, final_size)
        return final_size, evaluations
    
    def _record_evaluation(
        self, 
        evaluations: List[ConstraintEvaluation], 
        final_size: float
    ) -> None:
        """Record evaluation for audit."""
        self.evaluations.append({
            "timestamp": datetime.now().isoformat(),
            "final_size": final_size,
            "constraints": [e.to_dict() for e in evaluations]
        })
        # Keep last 1000 evaluations
        if len(self.evaluations) > 1000:
            self.evaluations = self.evaluations[-1000:]
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of current constraint states."""
        return {
            "hard_constraints": {
                "daily_loss_pct": -self.hard.daily_pnl / self.hard.current_equity if self.hard.current_equity > 0 else 0,
                "weekly_loss_pct": -self.hard.weekly_pnl / self.hard.current_equity if self.hard.current_equity > 0 else 0,
                "current_drawdown": self.hard.current_drawdown,
                "max_daily_loss": self.hard.max_daily_loss_pct,
                "max_weekly_loss": self.hard.max_weekly_loss_pct,
                "max_drawdown": self.hard.max_drawdown_pct
            },
            "volatility_targeting": {
                "target_vol": self.volatility.target_portfolio_vol
            },
            "alpha_optimization": {
                "kelly_cap": self.alpha.kelly_fraction_cap
            }
        }


if __name__ == "__main__":
    # Quick test
    hierarchy = InvertedRiskHierarchy()
    
    # Set initial equity state
    hierarchy.update_equity_state(
        current_equity=100000.0,
        daily_pnl=-500.0,  # -0.5% today
        weekly_pnl=-1000.0  # -1% this week
    )
    
    # Calculate position size
    size, evaluations = hierarchy.calculate_position_size(
        signal_edge=0.005,           # 0.5% expected edge
        signal_variance=0.02,        # 2% variance
        realized_volatility=0.25,    # 25% annualized vol
        adv_24h=1000000.0,           # $1M daily volume
        book_depth=500000.0,         # $500K depth
        estimated_slippage_per_unit=0.00005
    )
    
    print(f"Final position size: ${size:,.2f}")
    print("\nConstraint evaluations:")
    for e in evaluations:
        print(f"  {e.level.name}: {e.status.value} - {e.message}")
