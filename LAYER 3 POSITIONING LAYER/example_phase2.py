"""
HIMARI OPUS V2 - Phase 2 Example
=================================

Demonstrates Phase 2 multi-asset portfolio allocation with:
- Multiple strategy signals
- Correlation monitoring
- Weight drift tracking
- Hierarchical risk budgets

Version: 3.1 Phase 2
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from phases.phase2_portfolio import Layer3Phase2Portfolio


def example_multi_strategy_allocation():
    """Demonstrate multi-strategy portfolio allocation."""
    
    print("=" * 80)
    print("HIMARI OPUS V2 - Phase 2 Multi-Strategy Example")
    print("=" * 80)
    print()
    
    # Initialize Phase 2 portfolio
    print("1. Initializing Phase 2 Portfolio...")
    portfolio = Layer3Phase2Portfolio(
        portfolio_value=100000,
        kelly_fraction=0.25,
        max_position_pct=0.15,  # Max 15% per position
        max_correlation=0.7,
        enable_metrics=False
    )
    print(f"   Portfolio: $100,000")
    print(f"   Max position: 15%")
    print(f"   Max correlation: 0.7")
    print()
    
    # Create multiple strategy signals
    print("2. Creating strategy signals...")
    signals = [
        TacticalSignal(
            strategy_id="momentum_btc",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.80,
            risk_score=0.25,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.12,
            predicted_volatility=0.04,
            sentiment_score=0.75
        ),
        TacticalSignal(
            strategy_id="momentum_eth",
            symbol="ETH-USD",
            action=TacticalAction.BUY,
            confidence=0.70,
            risk_score=0.30,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.15,
            predicted_volatility=0.05,
            sentiment_score=0.65
        ),
        TacticalSignal(
            strategy_id="mean_reversion_sol",
            symbol="SOL-USD",
            action=TacticalAction.BUY,
            confidence=0.65,
            risk_score=0.40,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns(),
            expected_return=0.20,
            predicted_volatility=0.08,
            sentiment_score=0.55
        )
    ]
    
    for s in signals:
        print(f"   {s.strategy_id}: {s.symbol} (conf={s.confidence:.0%}, exp_ret={s.expected_return:.0%})")
    print()
    
    # Create cascade indicators
    cascade = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.05,
        volume_ratio=2.0,
        onchain_whale_pressure=0.35,
        exchange_netflow_zscore=0.8
    )
    
    # Current prices
    current_prices = {
        'BTC-USD': 43500.0,
        'ETH-USD': 2280.0,
        'SOL-USD': 95.0
    }
    
    print("3. Processing multi-strategy allocation...")
    allocation = portfolio.process_multi_strategy_signals(
        signals=signals,
        cascade_indicators=cascade,
        current_prices=current_prices
    )
    
    print()
    print("-" * 80)
    print("PORTFOLIO ALLOCATION RESULT")
    print("-" * 80)
    print(f"Total Allocated: ${allocation.total_allocated_usd:,.2f}")
    print(f"Utilization: {allocation.utilization_pct:.1f}%")
    print()
    
    print("Individual Allocations:")
    for symbol, usd in allocation.allocations.items():
        pct = usd / portfolio.portfolio_value * 100
        print(f"  {symbol}: ${usd:,.2f} ({pct:.1f}%)")
    print()
    
    print("Strategy Weights:")
    for strategy_id, weight in allocation.strategy_weights.items():
        print(f"  {strategy_id}: {weight:.1%}")
    print()
    
    if allocation.correlation_penalties:
        print("Correlation Penalties Applied:")
        for symbol, penalty in allocation.correlation_penalties.items():
            print(f"  {symbol}: {penalty:.2f}")
        print()
    
    # Simulate some returns and update correlations
    print("4. Updating correlation monitor...")
    for i in range(15):
        portfolio.update_correlations({
            'BTC-USD': 0.01 * (i % 4 - 1.5),
            'ETH-USD': 0.01 * (i % 4 - 1.2),  # Slightly correlated with BTC
            'SOL-USD': 0.015 * (i % 3 - 1)   # Different pattern
        })
    
    corr_state = portfolio.correlation_monitor.get_state()
    print(f"   Diversification Score: {corr_state['diversification_score']:.2f}")
    print(f"   Average Correlation: {corr_state['avg_correlation']:.2f}")
    print()
    
    # Get portfolio state
    print("5. Complete Portfolio State:")
    state = portfolio.get_state()
    print(f"   Total Decisions: {state['total_portfolio_decisions']}")
    print(f"   Ensemble Aggregations: {state['ensemble']['total_aggregations']}")
    print(f"   Kelly Allocations: {state['kelly_allocator']['total_allocations']}")
    print(f"   Drift Alerts: {state['ensemble']['drift_alerts_triggered']}")
    print()
    
    # Export weight history
    print("6. Exporting weight history...")
    export_path = "weight_history.csv"
    if portfolio.export_weight_history(export_path):
        print(f"   [OK] Exported to {export_path}")
    print()

    portfolio.stop()
    print("[OK] Phase 2 example completed!")

    return allocation


def example_rebalancing():
    """Demonstrate rebalancing signals."""
    
    print()
    print("=" * 80)
    print("REBALANCING EXAMPLE")
    print("=" * 80)
    print()
    
    portfolio = Layer3Phase2Portfolio(
        portfolio_value=100000,
        enable_metrics=False
    )
    
    # Current positions (after some market movement)
    current_positions = {
        'BTC-USD': 18000,  # Grew above target
        'ETH-USD': 8000,   # Shrunk below target
        'SOL-USD': 4000
    }
    
    # Target allocations
    target_allocations = {
        'BTC-USD': 12000,
        'ETH-USD': 10000,
        'SOL-USD': 5000
    }
    
    print("Current Positions:")
    for sym, usd in current_positions.items():
        print(f"  {sym}: ${usd:,.2f}")
    
    print()
    print("Target Allocations:")
    for sym, usd in target_allocations.items():
        print(f"  {sym}: ${usd:,.2f}")
    
    print()
    trades, diag = portfolio.kelly_allocator.get_rebalancing_signals(
        current_positions=current_positions,
        target_allocations=target_allocations,
        rebalance_threshold=0.10
    )
    
    print("Rebalancing Trades Required:")
    for sym, delta in trades.items():
        action = "BUY" if delta > 0 else "SELL"
        print(f"  {sym}: {action} ${abs(delta):,.2f}")
    
    print()
    portfolio.stop()


if __name__ == "__main__":
    allocation = example_multi_strategy_allocation()
    example_rebalancing()

    print()
    print("=" * 80)
    print("[SUCCESS] All Phase 2 examples completed successfully!")
    print("=" * 80)
