"""
HIMARI Layer 3 - Phase 3 Example
=================================

Demonstrates Phase 3 hybrid orchestrator with optional RL integration.

Usage:
    python example_phase3.py

Version: 3.1 Phase 3
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from phases.phase3_hybrid import Layer3Phase3Hybrid


def example_phase3_with_mock_rl():
    """Demonstrate Phase 3 with mock RL (no external endpoint needed)."""
    
    print("=" * 80)
    print("HIMARI OPUS V2 - Phase 3 Hybrid Example (Mock RL)")
    print("=" * 80)
    print()
    
    # Initialize Phase 3 with mock RL
    print("1. Initializing Phase 3 Hybrid Orchestrator...")
    hybrid = Layer3Phase3Hybrid(
        portfolio_value=100000,
        kelly_fraction=0.25,
        max_position_pct=0.15,
        enable_rl=True,
        use_mock_rl=True,  # Use mock RL for testing
        enable_metrics=False
    )
    print("   [OK] Phase 3 initialized with Mock RL")
    print()
    
    # Create signals
    print("2. Creating strategy signals...")
    signals = [
        TacticalSignal(
            strategy_id="momentum_btc",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=0.85,
            risk_score=0.25,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.12,
            predicted_volatility=0.04
        ),
        TacticalSignal(
            strategy_id="momentum_eth",
            symbol="ETH-USD",
            action=TacticalAction.BUY,
            confidence=0.75,
            risk_score=0.30,
            regime=MarketRegime.TRENDING_UP,
            timestamp_ns=time.time_ns(),
            expected_return=0.15,
            predicted_volatility=0.05
        )
    ]
    
    for s in signals:
        print(f"   {s.strategy_id}: {s.symbol} (conf={s.confidence:.0%})")
    print()
    
    # Cascade indicators
    cascade = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.05,
        volume_ratio=2.0,
        onchain_whale_pressure=0.30,
        exchange_netflow_zscore=0.5
    )
    
    # Current prices
    prices = {'BTC-USD': 43500.0, 'ETH-USD': 2280.0}
    
    # Process with RL
    print("3. Processing with RL enhancement...")
    allocation = hybrid.process_signals(
        signals=signals,
        cascade_indicators=cascade,
        current_prices=prices
    )
    
    print()
    print("-" * 80)
    print("PHASE 3 ALLOCATION RESULT")
    print("-" * 80)
    print(f"Total Allocated: ${allocation.total_allocated_usd:,.2f}")
    print(f"Utilization: {allocation.utilization_pct:.1f}%")
    print()
    
    print("Allocations:")
    for symbol, usd in allocation.allocations.items():
        pct = usd / hybrid.portfolio_value * 100
        print(f"  {symbol}: ${usd:,.2f} ({pct:.1f}%)")
    print()
    
    # Show RL stats
    print("4. RL Statistics:")
    if hybrid.rl_client:
        rl_state = hybrid.rl_client.get_state()
        print(f"   Total Predictions: {rl_state['total_predictions']}")
        print(f"   Successful: {rl_state['successful_predictions']}")
        print(f"   Fallbacks: {rl_state['fallback_predictions']}")
        print(f"   Success Rate: {rl_state['success_rate']:.0%}")
    print()
    
    # Show hybrid stats
    print("5. Hybrid Orchestrator Stats:")
    print(f"   Total Decisions: {hybrid.total_decisions}")
    print(f"   RL Decisions: {hybrid.rl_decisions}")
    print(f"   Fallback Decisions: {hybrid.fallback_decisions}")
    print(f"   RL Usage Rate: {hybrid.rl_decisions / hybrid.total_decisions * 100:.0f}%")
    print()
    
    hybrid.stop()
    print("[OK] Phase 3 example completed!")
    
    return allocation


def example_phase3_fallback_only():
    """Demonstrate Phase 3 in fallback-only mode (no RL)."""
    
    print()
    print("=" * 80)
    print("PHASE 3 FALLBACK-ONLY MODE")
    print("=" * 80)
    print()
    
    # Initialize without RL
    hybrid = Layer3Phase3Hybrid(
        portfolio_value=100000,
        enable_rl=False,  # No RL
        enable_metrics=False
    )
    print("[OK] Phase 3 initialized in fallback-only mode")
    print()
    
    signal = TacticalSignal(
        strategy_id="momentum",
        symbol="BTC-USD",
        action=TacticalAction.BUY,
        confidence=0.80,
        risk_score=0.25,
        regime=MarketRegime.RANGING,
        timestamp_ns=time.time_ns(),
        expected_return=0.10
    )
    
    cascade = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.02,
        volume_ratio=1.5,
        onchain_whale_pressure=0.25,
        exchange_netflow_zscore=0.3
    )
    
    allocation = hybrid.process_signals(
        signals=[signal],
        cascade_indicators=cascade,
        current_prices={'BTC-USD': 42000.0}
    )
    
    print(f"Allocation: ${allocation.total_allocated_usd:,.2f}")
    print(f"Fallback Decisions: {hybrid.fallback_decisions}")
    print()
    
    hybrid.stop()


def example_performance_comparison():
    """Demonstrate RL vs baseline performance tracking."""
    
    print()
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print()
    
    hybrid = Layer3Phase3Hybrid(
        portfolio_value=100000,
        enable_rl=True,
        use_mock_rl=True,
        enable_metrics=False
    )
    
    # Simulate some trade outcomes
    import random
    random.seed(42)
    
    print("Simulating trade outcomes...")
    for _ in range(20):
        # RL trades slightly better on average
        hybrid.record_outcome("BTC-USD", random.gauss(0.03, 0.02), used_rl=True)
        hybrid.record_outcome("BTC-USD", random.gauss(0.02, 0.025), used_rl=False)
    
    perf = hybrid.get_performance_comparison()
    
    print()
    print("RL Performance:")
    print(f"  Trades: {perf['rl']['trades']}")
    print(f"  Mean Return: {perf['rl']['mean_return']:.2%}")
    print(f"  Std Dev: {perf['rl']['std_return']:.2%}")
    print(f"  Sharpe: {perf['rl']['sharpe']:.2f}")
    print()
    print("Baseline Performance:")
    print(f"  Trades: {perf['baseline']['trades']}")
    print(f"  Mean Return: {perf['baseline']['mean_return']:.2%}")
    print(f"  Std Dev: {perf['baseline']['std_return']:.2%}")
    print(f"  Sharpe: {perf['baseline']['sharpe']:.2f}")
    print()
    print(f"RL Advantage: {perf['rl_advantage']:.2%}")
    print()
    
    hybrid.stop()


if __name__ == "__main__":
    example_phase3_with_mock_rl()
    example_phase3_fallback_only()
    example_performance_comparison()
    
    print()
    print("=" * 80)
    print("[SUCCESS] All Phase 3 examples completed!")
    print("=" * 80)
