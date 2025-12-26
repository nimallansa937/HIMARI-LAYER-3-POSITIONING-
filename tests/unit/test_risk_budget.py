"""
Unit tests for Hierarchical Risk Budget Manager
"""

import pytest
import sys
sys.path.insert(0, 'src')

from portfolio.risk_budget import HierarchicalRiskBudgetManager


class TestHierarchicalRiskBudgetManager:
    """Test suite for Hierarchical Risk Budget Manager."""

    def test_initialization(self):
        """Test budget manager initialization."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            portfolio_max_pct=0.80,
            strategy_max_pct=0.40,
            position_max_pct=0.15
        )

        assert manager.portfolio_value == 100000
        assert manager.portfolio_budget.max_allocation_pct == 0.80
        assert manager.default_strategy_max_pct == 0.40
        assert manager.default_position_max_pct == 0.15

    def test_position_limit_enforcement(self):
        """Test position limit is enforced."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            position_max_pct=0.10  # 10% = $10k max
        )

        # Request $20k position
        allowed, diag = manager.check_and_enforce(
            strategy_id="momentum",
            symbol="BTC-USD",
            requested_usd=20000
        )

        # Should be capped at $10k
        assert allowed == 10000
        assert 'position_limit' in str(diag['violations'])

    def test_strategy_limit_enforcement(self):
        """Test strategy limit is enforced."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            strategy_max_pct=0.20  # 20% = $20k max per strategy
        )

        # First position
        allowed1, _ = manager.check_and_enforce(
            strategy_id="momentum",
            symbol="BTC-USD",
            requested_usd=12000
        )
        assert allowed1 == 12000

        # Second position for same strategy (total would be $22k)
        allowed2, diag2 = manager.check_and_enforce(
            strategy_id="momentum",
            symbol="ETH-USD",
            requested_usd=10000
        )

        # Should be capped at remaining $8k
        assert allowed2 == 8000
        assert 'strategy_limit' in str(diag2['violations'])

    def test_portfolio_limit_enforcement(self):
        """Test portfolio limit is enforced."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            portfolio_max_pct=0.50,  # 50% = $50k max total
            position_max_pct=0.50   # Allow larger positions for this test
        )

        # Strategy 1: $30k
        manager.check_and_enforce("strat1", "BTC-USD", 30000)

        # Strategy 2: Request $25k (would exceed portfolio limit)
        allowed, diag = manager.check_and_enforce("strat2", "ETH-USD", 25000)

        # Should be capped at remaining $20k (portfolio limit)
        assert allowed == 20000
        assert 'portfolio_limit' in str(diag['violations'])

    def test_reset_utilization(self):
        """Test utilization reset."""
        manager = HierarchicalRiskBudgetManager(portfolio_value=100000)

        # Allocate some budget
        manager.check_and_enforce("strat1", "BTC-USD", 10000)

        assert manager.portfolio_budget.utilized_usd == 10000

        # Reset
        manager.reset_utilization()

        assert manager.portfolio_budget.utilized_usd == 0.0
        assert manager.strategy_budgets['strat1'].utilized_usd == 0.0

    def test_set_strategy_budget(self):
        """Test custom strategy budget."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            strategy_max_pct=0.40,
            position_max_pct=0.60   # Allow larger positions for this test
        )

        # Set custom budget for specific strategy
        manager.set_strategy_budget("high_priority", 0.60)

        # This strategy can now use 60% instead of default 40%
        allowed, _ = manager.check_and_enforce(
            "high_priority",
            "BTC-USD",
            50000
        )

        assert allowed == 50000  # Within 60% limit and within position limit

    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            portfolio_max_pct=0.80,
            position_max_pct=0.25   # Allow $25k positions
        )

        # Use some budget (will be capped at position limit of $25k)
        allowed, _ = manager.check_and_enforce("strat1", "BTC-USD", 20000)

        remaining = manager.get_remaining_budget()

        assert remaining['portfolio']['utilized_usd'] == 20000
        assert remaining['portfolio']['remaining_usd'] == 60000
        assert 'strat1' in remaining['strategies']

    def test_get_state(self):
        """Test state retrieval."""
        manager = HierarchicalRiskBudgetManager(portfolio_value=100000)

        manager.check_and_enforce("strat1", "BTC-USD", 10000)

        state = manager.get_state()

        assert 'portfolio_value' in state
        assert 'portfolio_budget' in state
        assert state['num_strategies'] == 1
        assert state['total_checks'] == 1

    def test_multiple_violations(self):
        """Test multiple limit violations."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            portfolio_max_pct=0.30,
            strategy_max_pct=0.20,
            position_max_pct=0.10
        )

        # Request exceeds all limits
        allowed, diag = manager.check_and_enforce(
            "strat1",
            "BTC-USD",
            50000  # Exceeds all limits
        )

        # Should be capped at position limit (most restrictive)
        assert allowed == 10000
        assert len(diag['violations']) > 0

    def test_zero_budget_remaining(self):
        """Test behavior when budget fully utilized."""
        manager = HierarchicalRiskBudgetManager(
            portfolio_value=100000,
            portfolio_max_pct=0.30,    # 30% = $30k max
            position_max_pct=0.50,     # Allow larger positions
            strategy_max_pct=0.50
        )

        # Use full portfolio budget
        manager.check_and_enforce("strat1", "BTC-USD", 30000)

        # Try to allocate more (portfolio exhausted)
        allowed, diag = manager.check_and_enforce("strat2", "ETH-USD", 10000)

        assert allowed == 0.0
        assert 'portfolio_limit' in str(diag['violations'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
