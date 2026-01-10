# src/core/constraints_first_kelly.py
"""
#5: Invert the Risk Hierarchy
Define acceptable losses first, then optimize returns within those constraints
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConstraints:
    """Non-negotiable risk boundaries"""
    MAX_DRAWDOWN: float = 0.15  # 15% hard ceiling
    MIN_SHARPE: float = 1.0     # Below this, hold cash
    MAX_LEVERAGE: float = 2.0   # Never exceed 2x
    MAX_POSITION_SIZE: float = 0.25  # Never more than 25% in single position
    MIN_LIQUIDITY_DEPTH: float = 10000.0  # Position size must be < depth/10
    FRACTIONAL_KELLY: float = 0.25  # Use 25% of full Kelly for safety
    
    def validate(self) -> bool:
        """Sanity check constraints"""
        assert 0 < self.MAX_DRAWDOWN < 1.0, "MAX_DRAWDOWN must be (0, 1)"
        assert self.MIN_SHARPE > 0, "MIN_SHARPE must be positive"
        assert self.MAX_LEVERAGE >= 1.0, "MAX_LEVERAGE must be >= 1"
        assert 0 < self.MAX_POSITION_SIZE <= 1.0, "MAX_POSITION_SIZE must be (0, 1]"
        return True


class ConstraintsFirstKelly:
    """
    Inverted risk hierarchy: Define acceptable losses first,
    then optimize returns within those constraints.
    
    Philosophy: "What can I afford to lose?" before "What can I gain?"
    """
    
    def __init__(self, constraints: Optional[RiskConstraints] = None):
        self.constraints = constraints or RiskConstraints()
        self.constraints.validate()
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.constraint_violations: Dict[str, int] = {}
        
    def calculate_position_size(self, 
                               win_rate: float,
                               win_loss_ratio: float,
                               signal_strength: float,
                               current_sharpe: float,
                               available_depth: float,
                               current_equity: float = None) -> float:
        """
        Calculate Kelly position size, but only AFTER constraints satisfied.
        
        Args:
            win_rate: Historical win rate [0, 1]
            win_loss_ratio: Avg win / avg loss (>0)
            signal_strength: Current signal strength [-1, 1]
            current_sharpe: Rolling 90-day Sharpe ratio
            available_depth: Liquidity in orderbook (USD)
            current_equity: Current portfolio equity (for drawdown calc)
            
        Returns:
            Position size as fraction of capital, or 0.0 if constraints violated
        """
        # Update drawdown if equity provided
        if current_equity is not None:
            self.update_drawdown(current_equity)
        
        # PHASE 1: CHECK CONSTRAINTS (must pass ALL)
        constraint_result = self._check_constraints(current_sharpe, available_depth)
        if not constraint_result['passed']:
            self._log_violation(constraint_result['reason'])
            return 0.0  # Cash position
        
        # PHASE 2: CALCULATE KELLY (only if constraints satisfied)
        kelly_fraction = self._calculate_kelly(win_rate, win_loss_ratio)
        
        # PHASE 3: APPLY CONSTRAINTS AS HARD CAPS
        position_size = kelly_fraction * abs(signal_strength)
        position_size = self._apply_constraint_caps(position_size, available_depth)
        
        return position_size
    
    def _check_constraints(self, current_sharpe: float, available_depth: float) -> Dict:
        """
        Check if ALL constraints are satisfied.
        
        Returns:
            {'passed': bool, 'reason': str}
        """
        # Constraint 1: Sharpe threshold
        if current_sharpe < self.constraints.MIN_SHARPE:
            return {
                'passed': False,
                'reason': f"Sharpe {current_sharpe:.2f} < {self.constraints.MIN_SHARPE}"
            }
        
        # Constraint 2: Drawdown ceiling
        if self.current_drawdown > self.constraints.MAX_DRAWDOWN:
            return {
                'passed': False,
                'reason': f"Drawdown {self.current_drawdown:.2%} > {self.constraints.MAX_DRAWDOWN:.2%}"
            }
        
        # Constraint 3: Liquidity depth
        if available_depth < self.constraints.MIN_LIQUIDITY_DEPTH:
            return {
                'passed': False,
                'reason': f"Depth ${available_depth:,.0f} < ${self.constraints.MIN_LIQUIDITY_DEPTH:,.0f}"
            }
        
        return {'passed': True, 'reason': ''}
    
    def _calculate_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Standard Kelly criterion with fractional adjustment.
        
        Kelly Formula: f* = (p * b - q) / b
        where:
            p = probability of win
            q = probability of loss (1 - p)
            b = win/loss ratio
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        if not (0 < win_rate < 1):
            return 0.0
        
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
        
        # Fractional Kelly for safety
        return max(0, kelly * self.constraints.FRACTIONAL_KELLY)
    
    def _apply_constraint_caps(self, position_size: float, available_depth: float) -> float:
        """Apply hard caps from constraints"""
        original_size = position_size
        
        # Cap 1: Max position size
        position_size = min(position_size, self.constraints.MAX_POSITION_SIZE)
        
        # Cap 2: Liquidity depth (position must be <10% of available depth)
        # This prevents excessive market impact
        max_size_from_depth = (available_depth * 0.1) / available_depth if available_depth > 0 else 0
        position_size = min(position_size, max_size_from_depth)
        
        if position_size < original_size:
            logger.debug(f"Position capped: {original_size:.2%} -> {position_size:.2%}")
        
        return position_size
    
    def update_drawdown(self, current_equity: float):
        """Update current drawdown for constraint checking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def _log_violation(self, reason: str):
        """Track constraint violations"""
        if reason not in self.constraint_violations:
            self.constraint_violations[reason] = 0
        self.constraint_violations[reason] += 1
        
        logger.warning(f"[CONSTRAINT VIOLATION] {reason}")
        print(f"[CONSTRAINT VIOLATION] {reason}")
    
    def get_constraint_status(self) -> Dict:
        """Get current constraint status"""
        return {
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'max_drawdown_limit': self.constraints.MAX_DRAWDOWN,
            'drawdown_remaining': self.constraints.MAX_DRAWDOWN - self.current_drawdown,
            'violations': self.constraint_violations.copy(),
        }
    
    def generate_constraints_report(self) -> str:
        """Generate markdown report of constraints status"""
        status = self.get_constraint_status()
        
        report = "# Constraints Status Report\n\n"
        
        # Drawdown status
        dd_pct = status['current_drawdown'] * 100
        max_dd = status['max_drawdown_limit'] * 100
        remaining = status['drawdown_remaining'] * 100
        
        if dd_pct > max_dd * 0.8:
            emoji = "🔴"
        elif dd_pct > max_dd * 0.5:
            emoji = "🟡"
        else:
            emoji = "✅"
        
        report += f"## Drawdown {emoji}\n"
        report += f"- Current: {dd_pct:.1f}%\n"
        report += f"- Limit: {max_dd:.1f}%\n"
        report += f"- Remaining: {remaining:.1f}%\n\n"
        
        # Violations
        if status['violations']:
            report += "## Recent Violations\n"
            for reason, count in status['violations'].items():
                report += f"- {reason}: {count}x\n"
        else:
            report += "## Violations\n✅ No constraint violations\n"
        
        return report


class AdaptiveConstraintsKelly(ConstraintsFirstKelly):
    """
    Extended Kelly with adaptive constraints based on market conditions.
    Tightens constraints during high volatility, loosens during calm periods.
    """
    
    def __init__(self, constraints: Optional[RiskConstraints] = None):
        super().__init__(constraints)
        self.volatility_scalar = 1.0  # 1.0 = normal, <1.0 = tighter, >1.0 = looser
    
    def update_volatility_regime(self, current_volatility: float, 
                                  baseline_volatility: float = 0.6):
        """
        Adjust constraints based on current vs baseline volatility.
        
        Args:
            current_volatility: Current annualized volatility
            baseline_volatility: Normal market volatility (default 60% for crypto)
        """
        ratio = current_volatility / baseline_volatility
        
        if ratio > 1.5:
            # High volatility: tighten constraints
            self.volatility_scalar = 0.5
        elif ratio > 1.2:
            self.volatility_scalar = 0.75
        elif ratio < 0.8:
            # Low volatility: slightly loosen
            self.volatility_scalar = 1.1
        else:
            self.volatility_scalar = 1.0
    
    def calculate_position_size(self, *args, **kwargs) -> float:
        """Calculate position with volatility adjustment"""
        base_size = super().calculate_position_size(*args, **kwargs)
        return base_size * self.volatility_scalar
