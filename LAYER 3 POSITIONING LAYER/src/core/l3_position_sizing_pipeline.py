"""
L3 Complete Position Sizing Pipeline
=====================================
Integrates all L3 enhancements into a single pipeline:

Pipeline Order:
1. L3-5: Minimum Viable Edge Check → If NET_SHARPE < threshold, HOLD CASH
2. L3-1: Inverted Risk Hierarchy → Hard/Liquidity/Vol/Kelly constraints
3. L3-2: CVaR Constraint → Reduce to CVaR-compliant size
4. L3-4: Capacity Ceiling → Cap at safe capacity
5. L3-3: Adaptive Interval Sizing → Apply uncertainty penalty

FINAL_SIZE = min(liquidity_max, vol_target_size, kelly_size, cvar_max, capacity_max) × uncertainty_penalty

File: LAYER 3 V1/LAYER 3 POSITIONING LAYER/src/core/l3_position_sizing_pipeline.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

# Import L3 components
from .inverted_risk_hierarchy import (
    InvertedRiskHierarchy,
    HardCapitalConstraints,
    LiquidityConstraints,
    VolatilityTargeting,
    AlphaOptimization,
    ConstraintEvaluation
)
from .cvar_cdar_optimizer import CVaRCDaROptimizer, RiskMetrics
from .adaptive_interval_sizing import (
    AdaptiveIntervalSizer,
    PredictionInterval,
    SizingResult,
    RegimeType
)
from .capacity_ceiling import CapacityCeilingCalculator
from .minimum_edge_calculator import MinimumEdgeCalculator, TradingCosts


logger = logging.getLogger(__name__)


class TradingState(Enum):
    """System trading state."""
    ACTIVE = "active"           # Full trading
    REDUCED = "reduced"         # Reduced position sizing
    CASH = "cash"              # Holding cash
    HALTED = "halted"          # Trading halted


@dataclass
class PipelineInput:
    """Input to the position sizing pipeline."""
    # Signal information
    signal_edge: float               # Expected return from signal
    signal_variance: float           # Signal variance for Kelly
    
    # Prediction interval (from L1-4 LSTM-CP)
    prediction_point: float          # Point prediction
    prediction_lower: float          # Lower bound
    prediction_upper: float          # Upper bound
    
    # Market data
    realized_volatility: float       # Current annualized vol
    adv_24h: float                   # 24h average daily volume
    book_depth: float                # Order book depth
    estimated_slippage_per_unit: float = 0.0001
    
    # Portfolio state
    current_equity: float
    daily_pnl: float
    weekly_pnl: float
    current_sharpe: float            # Rolling 30-day Sharpe
    
    # Asset
    symbol: str = "BTC"


@dataclass
class PipelineOutput:
    """Output from the position sizing pipeline."""
    # Final decision
    final_size: float
    should_trade: bool
    trading_state: TradingState
    
    # Stage outputs
    edge_sufficient: bool
    hierarchy_size: float
    cvar_size: float
    capacity_size: float
    uncertainty_adjusted_size: float
    
    # Reason chain
    reasons: List[str]
    
    # Detailed metrics
    constraint_evaluations: List[Dict[str, Any]]
    cvar_metrics: Dict[str, Any]
    uncertainty_penalty: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_size": self.final_size,
            "should_trade": self.should_trade,
            "trading_state": self.trading_state.value,
            "edge_sufficient": self.edge_sufficient,
            "hierarchy_size": self.hierarchy_size,
            "cvar_size": self.cvar_size,
            "capacity_size": self.capacity_size,
            "uncertainty_adjusted_size": self.uncertainty_adjusted_size,
            "uncertainty_penalty": self.uncertainty_penalty,
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat()
        }


class L3PositionSizingPipeline:
    """
    Complete L3 Position Sizing Pipeline.
    
    Enforces constraints-first philosophy: define acceptable losses first,
    then optimize returns within those bounds.
    
    Configuration from YAML:
    ```yaml
    l3_inverted_hierarchy:
      hard_constraints:
        max_daily_loss_pct: 0.02
        max_weekly_loss_pct: 0.05
        max_drawdown_pct: 0.15
      liquidity_constraints:
        max_position_vs_adv: 0.01
        max_position_vs_depth: 0.10
        max_slippage_bps: 10
      volatility_targeting:
        target_portfolio_vol: 0.15
      kelly:
        fraction_cap: 0.5
    
    l3_cvar_cdar:
      cvar_confidence: 0.95
      cvar_budget_pct: 0.05
      cdar_confidence: 0.95
      cdar_budget_pct: 0.10
    
    l3_adaptive_interval:
      interval_sensitivity: 5.0
      full_confidence_threshold: 0.02
      abstain_threshold: 0.20
    
    l3_minimum_edge:
      min_sharpe_trading: 0.5
      min_sharpe_marginal: 0.3
    ```
    """
    
    def __init__(
        self,
        # L3-1 Inverted Hierarchy params
        max_daily_loss_pct: float = 0.02,
        max_weekly_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_position_vs_adv: float = 0.01,
        max_position_vs_depth: float = 0.10,
        max_slippage_bps: float = 10.0,
        target_portfolio_vol: float = 0.15,
        kelly_fraction_cap: float = 0.5,
        # L3-2 CVaR params
        cvar_confidence: float = 0.95,
        cvar_budget_pct: float = 0.05,
        cdar_budget_pct: float = 0.10,
        # L3-3 Adaptive Interval params
        interval_sensitivity: float = 5.0,
        full_confidence_threshold: float = 0.02,
        abstain_threshold: float = 0.20,
        # L3-5 Minimum Edge params
        min_sharpe_trading: float = 0.5,
        min_sharpe_marginal: float = 0.3
    ):
        # Initialize L3-1: Inverted Risk Hierarchy
        self.hierarchy = InvertedRiskHierarchy(
            hard_constraints=HardCapitalConstraints(
                max_daily_loss_pct=max_daily_loss_pct,
                max_weekly_loss_pct=max_weekly_loss_pct,
                max_drawdown_pct=max_drawdown_pct
            ),
            liquidity_constraints=LiquidityConstraints(
                max_position_vs_adv=max_position_vs_adv,
                max_position_vs_depth=max_position_vs_depth,
                max_slippage_bps=max_slippage_bps
            ),
            volatility_targeting=VolatilityTargeting(
                target_portfolio_vol=target_portfolio_vol
            ),
            alpha_optimization=AlphaOptimization(
                kelly_fraction_cap=kelly_fraction_cap
            )
        )
        
        # Initialize L3-2: CVaR/CDaR Optimizer
        self.cvar_optimizer = CVaRCDaROptimizer(
            cvar_confidence=cvar_confidence,
            cvar_budget_pct=cvar_budget_pct,
            cdar_budget_pct=cdar_budget_pct
        )
        
        # Initialize L3-3: Adaptive Interval Sizer
        self.interval_sizer = AdaptiveIntervalSizer(
            interval_sensitivity=interval_sensitivity,
            full_confidence_threshold=full_confidence_threshold,
            abstain_threshold=abstain_threshold
        )
        
        # Initialize L3-4: Capacity Ceiling
        self.capacity_calculator = CapacityCeilingCalculator()
        
        # Initialize L3-5: Minimum Edge Calculator
        self.edge_calculator = MinimumEdgeCalculator()
        
        # Edge thresholds
        self.min_sharpe_trading = min_sharpe_trading
        self.min_sharpe_marginal = min_sharpe_marginal
        
        # State
        self.current_regime = RegimeType.NORMAL
        self.trading_state = TradingState.ACTIVE
        
        # History
        self.pipeline_history: List[PipelineOutput] = []
    
    def set_regime(self, regime: RegimeType) -> None:
        """Update market regime for adaptive sizing."""
        self.current_regime = regime
        self.interval_sizer.set_regime(regime)
    
    def calculate_position_size(self, input_data: PipelineInput) -> PipelineOutput:
        """
        Run the complete position sizing pipeline.
        
        Pipeline stages:
        1. Check minimum viable edge (L3-5)
        2. Apply inverted risk hierarchy (L3-1)
        3. Apply CVaR constraint (L3-2)
        4. Apply capacity ceiling (L3-4)
        5. Apply uncertainty penalty (L3-3)
        """
        reasons = []
        
        # ==== STAGE 1: L3-5 Minimum Viable Edge Check ====
        avg_trades_per_day = 2.0  # Assumed
        should_trade_edge = self.edge_calculator.should_trade(
            input_data.current_sharpe,
            avg_trades_per_day
        )
        
        edge_sufficient = input_data.current_sharpe >= self.min_sharpe_trading
        edge_marginal = input_data.current_sharpe >= self.min_sharpe_marginal
        
        if not edge_marginal:
            # NET_SHARPE < 0.3 → HOLD 100% CASH
            reasons.append(f"L3-5: Edge insufficient (Sharpe={input_data.current_sharpe:.2f} < {self.min_sharpe_marginal})")
            return self._create_output(
                final_size=0.0,
                should_trade=False,
                trading_state=TradingState.CASH,
                edge_sufficient=False,
                reasons=reasons
            )
        
        if not edge_sufficient:
            # NET_SHARPE 0.3-0.5 → Trade with 50% position
            reasons.append(f"L3-5: Marginal edge (Sharpe={input_data.current_sharpe:.2f})")
            edge_multiplier = 0.5
        else:
            reasons.append(f"L3-5: Edge sufficient (Sharpe={input_data.current_sharpe:.2f})")
            edge_multiplier = 1.0
        
        # ==== STAGE 2: L3-1 Inverted Risk Hierarchy ====
        self.hierarchy.update_equity_state(
            input_data.current_equity,
            input_data.daily_pnl,
            input_data.weekly_pnl
        )
        
        hierarchy_size, constraint_evals = self.hierarchy.calculate_position_size(
            signal_edge=input_data.signal_edge,
            signal_variance=input_data.signal_variance,
            realized_volatility=input_data.realized_volatility,
            adv_24h=input_data.adv_24h,
            book_depth=input_data.book_depth,
            estimated_slippage_per_unit=input_data.estimated_slippage_per_unit
        )
        
        if hierarchy_size <= 0:
            reasons.append("L3-1: Hard constraint violated, position blocked")
            return self._create_output(
                final_size=0.0,
                should_trade=False,
                trading_state=TradingState.HALTED,
                edge_sufficient=edge_sufficient,
                hierarchy_size=hierarchy_size,
                constraint_evaluations=[e.to_dict() for e in constraint_evals],
                reasons=reasons
            )
        
        reasons.append(f"L3-1: Hierarchy size = ${hierarchy_size:,.0f}")
        
        # ==== STAGE 3: L3-2 CVaR Constraint ====
        cvar_size, cvar_details = self.cvar_optimizer.optimize_position(
            max_from_other_constraints=hierarchy_size,
            expected_return=input_data.signal_edge,
            volatility=input_data.realized_volatility,
            equity=input_data.current_equity
        )
        
        reasons.append(f"L3-2: CVaR size = ${cvar_size:,.0f}")
        
        # ==== STAGE 4: L3-4 Capacity Ceiling ====
        capacity_util = self.capacity_calculator.get_current_capacity_utilization(
            input_data.current_equity,
            input_data.symbol
        )
        capacity_ceiling = capacity_util.get('capacity_ceiling', float('inf'))
        
        # Calculate capacity-constrained size
        if capacity_ceiling > 0:
            capacity_size = min(cvar_size, capacity_ceiling * 0.25)  # 25% max position
        else:
            capacity_size = cvar_size
        
        reasons.append(f"L3-4: Capacity size = ${capacity_size:,.0f}")
        
        # ==== STAGE 5: L3-3 Adaptive Interval Sizing ====
        prediction_interval = PredictionInterval(
            point_estimate=input_data.prediction_point,
            lower_bound=input_data.prediction_lower,
            upper_bound=input_data.prediction_upper
        )
        
        sizing_result = self.interval_sizer.calculate_adjusted_size(
            capacity_size * edge_multiplier,  # Apply edge multiplier
            prediction_interval
        )
        
        uncertainty_adjusted = sizing_result.adjusted_size
        uncertainty_penalty = sizing_result.uncertainty_penalty
        
        reasons.append(f"L3-3: Uncertainty penalty = {uncertainty_penalty:.2f}, final = ${uncertainty_adjusted:,.0f}")
        
        # Determine trading state
        if uncertainty_penalty == 0:
            trading_state = TradingState.CASH
        elif edge_multiplier < 1.0 or uncertainty_penalty < 0.5:
            trading_state = TradingState.REDUCED
        else:
            trading_state = TradingState.ACTIVE
        
        return self._create_output(
            final_size=uncertainty_adjusted,
            should_trade=uncertainty_adjusted > 0,
            trading_state=trading_state,
            edge_sufficient=edge_sufficient,
            hierarchy_size=hierarchy_size,
            cvar_size=cvar_size,
            capacity_size=capacity_size,
            uncertainty_adjusted_size=uncertainty_adjusted,
            uncertainty_penalty=uncertainty_penalty,
            constraint_evaluations=[e.to_dict() for e in constraint_evals],
            cvar_metrics=cvar_details,
            reasons=reasons
        )
    
    def _create_output(
        self,
        final_size: float = 0.0,
        should_trade: bool = False,
        trading_state: TradingState = TradingState.CASH,
        edge_sufficient: bool = False,
        hierarchy_size: float = 0.0,
        cvar_size: float = 0.0,
        capacity_size: float = 0.0,
        uncertainty_adjusted_size: float = 0.0,
        uncertainty_penalty: float = 0.0,
        constraint_evaluations: List[Dict] = None,
        cvar_metrics: Dict = None,
        reasons: List[str] = None
    ) -> PipelineOutput:
        """Create pipeline output."""
        output = PipelineOutput(
            final_size=final_size,
            should_trade=should_trade,
            trading_state=trading_state,
            edge_sufficient=edge_sufficient,
            hierarchy_size=hierarchy_size,
            cvar_size=cvar_size,
            capacity_size=capacity_size,
            uncertainty_adjusted_size=uncertainty_adjusted_size,
            uncertainty_penalty=uncertainty_penalty,
            constraint_evaluations=constraint_evaluations or [],
            cvar_metrics=cvar_metrics or {},
            reasons=reasons or []
        )
        
        # Record history
        self.pipeline_history.append(output)
        if len(self.pipeline_history) > 1000:
            self.pipeline_history = self.pipeline_history[-1000:]
        
        return output
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "trading_state": self.trading_state.value,
            "current_regime": self.current_regime.value,
            "hierarchy_summary": self.hierarchy.get_constraint_summary(),
            "recent_evaluations": len(self.pipeline_history),
            "interval_stats": self.interval_sizer.get_statistics()
        }


# Convenience function
def calculate_position(
    signal_edge: float,
    signal_variance: float,
    prediction_point: float,
    prediction_lower: float,
    prediction_upper: float,
    realized_vol: float,
    equity: float,
    daily_pnl: float,
    current_sharpe: float,
    adv_24h: float = 1000000.0,
    book_depth: float = 500000.0,
    symbol: str = "BTC"
) -> Dict[str, Any]:
    """
    Quick position sizing calculation.
    
    Returns dict with position size and trading decision.
    """
    pipeline = L3PositionSizingPipeline()
    
    input_data = PipelineInput(
        signal_edge=signal_edge,
        signal_variance=signal_variance,
        prediction_point=prediction_point,
        prediction_lower=prediction_lower,
        prediction_upper=prediction_upper,
        realized_volatility=realized_vol,
        adv_24h=adv_24h,
        book_depth=book_depth,
        current_equity=equity,
        daily_pnl=daily_pnl,
        weekly_pnl=daily_pnl * 3,  # Approximation
        current_sharpe=current_sharpe,
        symbol=symbol
    )
    
    output = pipeline.calculate_position_size(input_data)
    return output.to_dict()


if __name__ == "__main__":
    # Quick test
    pipeline = L3PositionSizingPipeline()
    
    # Test input
    input_data = PipelineInput(
        signal_edge=0.005,           # 0.5% expected edge
        signal_variance=0.02,        # 2% variance
        prediction_point=100.0,
        prediction_lower=97.0,       # 6% interval width
        prediction_upper=103.0,
        realized_volatility=0.25,    # 25% annualized vol
        adv_24h=1000000.0,           # $1M daily volume
        book_depth=500000.0,         # $500K depth
        current_equity=100000.0,
        daily_pnl=-500.0,            # -0.5% today
        weekly_pnl=-1000.0,          # -1% this week
        current_sharpe=0.8,          # Sufficient edge
        symbol="BTC"
    )
    
    output = pipeline.calculate_position_size(input_data)
    
    print("=" * 60)
    print("L3 Position Sizing Pipeline Output")
    print("=" * 60)
    print(f"Should Trade: {output.should_trade}")
    print(f"Trading State: {output.trading_state.value}")
    print(f"Final Position Size: ${output.final_size:,.2f}")
    print(f"Uncertainty Penalty: {output.uncertainty_penalty:.2f}")
    print("\nReason Chain:")
    for reason in output.reasons:
        print(f"  - {reason}")
