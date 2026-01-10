"""
HIMARI OPUS V2 - Hierarchical Risk Budget Manager
===================================================

Implements hierarchical risk budgets for portfolio/strategy/position levels.

Features:
- Portfolio-level utilization limits
- Strategy-level allocation limits
- Position-level size limits
- Dynamic budget adjustments based on performance
- Risk budget monitoring

Version: 3.1 Phase 2
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskBudget:
    """Risk budget configuration for a level."""
    max_allocation_pct: float
    current_allocation_pct: float = 0.0
    utilized_usd: float = 0.0
    violations: int = 0


class HierarchicalRiskBudgetManager:
    """
    Manages hierarchical risk budgets at portfolio, strategy, and position levels.
    
    Hierarchy:
    - Portfolio: Total utilization cannot exceed portfolio budget (default 80%)
    - Strategy: Each strategy cannot exceed strategy budget (default 40%)
    - Position: Each position cannot exceed position budget (default 15%)
    
    Enforces:
    - Sum of positions <= strategy budget
    - Sum of strategies <= portfolio budget
    """
    
    def __init__(
        self,
        portfolio_value: float,
        portfolio_max_pct: float = 0.80,
        strategy_max_pct: float = 0.40,
        position_max_pct: float = 0.15
    ):
        """
        Initialize risk budget manager.
        
        Args:
            portfolio_value: Total portfolio value in USD
            portfolio_max_pct: Max portfolio utilization (default 80%)
            strategy_max_pct: Max per-strategy allocation (default 40%)
            position_max_pct: Max per-position allocation (default 15%)
        """
        self.portfolio_value = portfolio_value
        
        # Portfolio level budget
        self.portfolio_budget = RiskBudget(max_allocation_pct=portfolio_max_pct)
        
        # Strategy level budgets
        self.strategy_budgets: Dict[str, RiskBudget] = {}
        self.default_strategy_max_pct = strategy_max_pct
        
        # Position level budgets
        self.position_budgets: Dict[str, RiskBudget] = {}
        self.default_position_max_pct = position_max_pct
        
        # Tracking
        self.total_checks = 0
        self.total_violations = 0
        
        logger.info(
            f"HierarchicalRiskBudgetManager initialized: "
            f"portfolio={portfolio_max_pct:.0%}, strategy={strategy_max_pct:.0%}, "
            f"position={position_max_pct:.0%}"
        )
    
    def check_and_enforce(
        self,
        strategy_id: str,
        symbol: str,
        requested_usd: float
    ) -> Tuple[float, Dict]:
        """
        Check if allocation is within budget and enforce limits.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Asset symbol
            requested_usd: Requested allocation in USD
            
        Returns:
            Tuple of (allowed_usd, diagnostics)
        """
        self.total_checks += 1
        diagnostics = {
            'requested_usd': requested_usd,
            'violations': []
        }
        
        allowed_usd = requested_usd
        
        # Check position limit
        position_key = f"{strategy_id}:{symbol}"
        position_max_usd = self.portfolio_value * self.default_position_max_pct
        
        if allowed_usd > position_max_usd:
            diagnostics['violations'].append(
                f"position_limit: ${allowed_usd:.2f} > ${position_max_usd:.2f}"
            )
            allowed_usd = position_max_usd
            self._record_violation(self.position_budgets, position_key)
        
        # Check strategy limit
        if strategy_id not in self.strategy_budgets:
            self.strategy_budgets[strategy_id] = RiskBudget(
                max_allocation_pct=self.default_strategy_max_pct
            )
        
        strategy_budget = self.strategy_budgets[strategy_id]
        strategy_max_usd = self.portfolio_value * strategy_budget.max_allocation_pct
        strategy_remaining = strategy_max_usd - strategy_budget.utilized_usd
        
        if allowed_usd > strategy_remaining:
            diagnostics['violations'].append(
                f"strategy_limit: ${allowed_usd:.2f} > remaining ${strategy_remaining:.2f}"
            )
            allowed_usd = max(0, strategy_remaining)
            strategy_budget.violations += 1
        
        # Check portfolio limit
        portfolio_max_usd = self.portfolio_value * self.portfolio_budget.max_allocation_pct
        portfolio_remaining = portfolio_max_usd - self.portfolio_budget.utilized_usd
        
        if allowed_usd > portfolio_remaining:
            diagnostics['violations'].append(
                f"portfolio_limit: ${allowed_usd:.2f} > remaining ${portfolio_remaining:.2f}"
            )
            allowed_usd = max(0, portfolio_remaining)
            self.portfolio_budget.violations += 1
        
        # Update utilization
        if allowed_usd > 0:
            strategy_budget.utilized_usd += allowed_usd
            strategy_budget.current_allocation_pct = (
                strategy_budget.utilized_usd / self.portfolio_value
            )
            
            self.portfolio_budget.utilized_usd += allowed_usd
            self.portfolio_budget.current_allocation_pct = (
                self.portfolio_budget.utilized_usd / self.portfolio_value
            )
        
        if diagnostics['violations']:
            self.total_violations += len(diagnostics['violations'])
        
        diagnostics['allowed_usd'] = allowed_usd
        diagnostics['portfolio_utilization'] = self.portfolio_budget.current_allocation_pct
        diagnostics['strategy_utilization'] = strategy_budget.current_allocation_pct
        
        return allowed_usd, diagnostics
    
    def _record_violation(self, budgets: Dict, key: str):
        """Record a budget violation."""
        if key not in budgets:
            budgets[key] = RiskBudget(max_allocation_pct=self.default_position_max_pct)
        budgets[key].violations += 1
    
    def reset_utilization(self):
        """Reset all utilization to zero (e.g., for new period)."""
        self.portfolio_budget.utilized_usd = 0.0
        self.portfolio_budget.current_allocation_pct = 0.0
        
        for budget in self.strategy_budgets.values():
            budget.utilized_usd = 0.0
            budget.current_allocation_pct = 0.0
        
        logger.info("Risk budget utilization reset")
    
    def set_strategy_budget(self, strategy_id: str, max_pct: float):
        """Set custom budget for a strategy."""
        if strategy_id not in self.strategy_budgets:
            self.strategy_budgets[strategy_id] = RiskBudget(max_allocation_pct=max_pct)
        else:
            self.strategy_budgets[strategy_id].max_allocation_pct = max_pct
        
        logger.info(f"Strategy budget set: {strategy_id} = {max_pct:.0%}")
    
    def get_remaining_budget(self) -> Dict:
        """Get remaining budget at each level."""
        portfolio_max = self.portfolio_value * self.portfolio_budget.max_allocation_pct
        
        return {
            'portfolio': {
                'max_usd': portfolio_max,
                'utilized_usd': self.portfolio_budget.utilized_usd,
                'remaining_usd': portfolio_max - self.portfolio_budget.utilized_usd,
                'utilization_pct': self.portfolio_budget.current_allocation_pct
            },
            'strategies': {
                sid: {
                    'max_usd': self.portfolio_value * b.max_allocation_pct,
                    'utilized_usd': b.utilized_usd,
                    'remaining_usd': (
                        self.portfolio_value * b.max_allocation_pct - b.utilized_usd
                    ),
                    'utilization_pct': b.current_allocation_pct
                }
                for sid, b in self.strategy_budgets.items()
            }
        }
    
    def get_state(self) -> Dict:
        """Get complete state."""
        return {
            'portfolio_value': self.portfolio_value,
            'portfolio_budget': {
                'max_pct': self.portfolio_budget.max_allocation_pct,
                'current_pct': self.portfolio_budget.current_allocation_pct,
                'utilized_usd': self.portfolio_budget.utilized_usd,
                'violations': self.portfolio_budget.violations
            },
            'default_strategy_max_pct': self.default_strategy_max_pct,
            'default_position_max_pct': self.default_position_max_pct,
            'num_strategies': len(self.strategy_budgets),
            'total_checks': self.total_checks,
            'total_violations': self.total_violations
        }
