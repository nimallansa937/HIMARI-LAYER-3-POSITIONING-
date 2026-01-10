"""
HIMARI Layer 2 → Layer 3 Integration Example
=============================================

Demonstrates the complete flow from Layer 2 tactical decision
to Layer 3 position sizing and paper trade execution.

Usage:
    cd "LAYER 3 POSITIONING LAYER"
    python example_l2_l3_integration.py

Version: 3.1
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from integration.l2_signal_mapper import L2SignalMapper
from engines.execution_engine import ExecutionEngine
from phases.phase1_core import Layer3Phase1


# ============================================================================
# Mock Layer 2 Types
# ============================================================================

class MockTradeAction:
    """Simulates L2 TradeAction enum."""
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name


class MockTier:
    """Simulates L2 Tier enum."""
    def __init__(self, value: int):
        self._value = value
    
    @property
    def value(self):
        return self._value


class MockTacticalDecision:
    """Simulates L2 TacticalDecision dataclass."""
    def __init__(
        self,
        action: str,
        confidence: float,
        tier: int,
        reason: str = "Layer 2 tactical decision",
        metadata: dict = None,
    ):
        self.action = MockTradeAction(action)
        self.confidence = confidence
        self.tier = MockTier(tier)
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = time.time()


# ============================================================================
# Main Example
# ============================================================================

def run_l2_l3_integration_example():
    """Run complete Layer 2 → Layer 3 integration example."""
    
    print("=" * 80)
    print("HIMARI OPUS V2 - Layer 2 to Layer 3 Integration Example")
    print("=" * 80)
    print()
    
    # -------------------------------------------------------------------------
    # STEP 1: Create Layer 2 Tactical Decision
    # -------------------------------------------------------------------------
    print("STEP 1: Layer 2 Tactical Decision (Simulated)")
    print("-" * 60)
    
    l2_decision = MockTacticalDecision(
        action="STRONG_BUY",
        confidence=0.85,
        tier=1,  # T1 = Auto-execute
        reason="High coherence signal with bullish sentiment alignment",
        metadata={
            "regime": "TRENDING_UP",
            "coherence": 0.92,
            "entropy": 2.1,
            "sentiment_trend": 0.7,
        }
    )
    
    print(f"  Action:      {l2_decision.action.name}")
    print(f"  Confidence:  {l2_decision.confidence:.0%}")
    print(f"  Tier:        T{l2_decision.tier.value}")
    print(f"  Reason:      {l2_decision.reason}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 2: Map L2 Decision to L3 Signal
    # -------------------------------------------------------------------------
    print("STEP 2: Signal Mapping (L2 -> L3)")
    print("-" * 60)
    
    mapper = L2SignalMapper(default_strategy_id="momentum_btc")
    
    l3_signal = mapper.map_decision_to_signal(
        decision=l2_decision,
        symbol="BTC-USD",
        expected_return=0.10,
        predicted_volatility=0.04,
    )
    
    print(f"  Strategy ID: {l3_signal.strategy_id}")
    print(f"  Symbol:      {l3_signal.symbol}")
    print(f"  Action:      {l3_signal.action.value}")
    print(f"  Confidence:  {l3_signal.confidence:.0%}")
    print(f"  Risk Score:  {l3_signal.risk_score:.2f}")
    print(f"  Regime:      {l3_signal.regime.value}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 3: Layer 3 Position Sizing
    # -------------------------------------------------------------------------
    print("STEP 3: Layer 3 Position Sizing")
    print("-" * 60)
    
    # Initialize Layer 3 Phase 1
    PORTFOLIO_VALUE = 100_000
    layer3 = Layer3Phase1(
        portfolio_value=PORTFOLIO_VALUE,
        kelly_fraction=0.25,
    )
    
    # Create cascade indicators (simulated Layer 1 data)
    cascade_indicators = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.03,
        volume_ratio=2.0,
        onchain_whale_pressure=0.30,
        exchange_netflow_zscore=0.5,
    )
    
    current_price = 87000.0  # Current BTC price
    
    sizing_decision = layer3.calculate_position(
        signal=l3_signal,
        cascade_indicators=cascade_indicators,
        current_price=current_price,
    )
    
    print(f"  Kelly Position:    ${sizing_decision.kelly_position_usd:,.2f}")
    print(f"  Conformal Adj:     ${sizing_decision.conformal_adjusted_usd:,.2f}")
    print(f"  Regime Adj:        ${sizing_decision.regime_adjusted_usd:,.2f}")
    print(f"  Cascade Adj:       ${sizing_decision.cascade_adjusted_usd:,.2f}")
    print(f"  Final Size:        ${sizing_decision.position_size_usd:,.2f}")
    print(f"  Portfolio %:       {sizing_decision.position_size_usd / PORTFOLIO_VALUE * 100:.1f}%")
    print(f"  Cascade Risk:      {sizing_decision.cascade_risk_score:.2f}")
    print(f"  Recommendation:    {sizing_decision.cascade_recommendation}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 4: Generate Execution Order
    # -------------------------------------------------------------------------
    print("STEP 4: Execution Order")
    print("-" * 60)
    
    # Calculate order quantity
    order_quantity = sizing_decision.position_size_usd / current_price
    
    order = {
        'order_id': f"L2L3_{int(time.time())}",
        'symbol': sizing_decision.symbol,
        'side': 'BUY',
        'order_type': 'MARKET',
        'quantity': order_quantity,
        'quantity_usd': sizing_decision.position_size_usd,
    }
    
    print(f"  Order ID:    {order['order_id']}")
    print(f"  Symbol:      {order['symbol']}")
    print(f"  Side:        {order['side']}")
    print(f"  Type:        {order['order_type']}")
    print(f"  Quantity:    {order['quantity']:.6f} BTC")
    print(f"  USD Value:   ${order['quantity_usd']:,.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 5: Execute (Paper Trading)
    # -------------------------------------------------------------------------
    print("STEP 5: Paper Trade Execution")
    print("-" * 60)
    
    engine = ExecutionEngine(
        exchange="binance",
        paper_trading=True,
        commission_rate=0.001,  # 0.1%
        default_slippage_bps=5,  # 0.05%
    )
    
    report = engine.submit_order(order, current_price=current_price)
    
    print(f"  Status:        {report.status}")
    print(f"  Exchange ID:   {report.exchange_order_id}")
    print(f"  Fill Price:    ${report.fill_price:,.2f}")
    print(f"  Fill Quantity: {report.fill_quantity:.6f} BTC")
    print(f"  Slippage:      {report.slippage_bps} bps")
    print(f"  Commission:    ${report.commission:.2f}")
    print(f"  Latency:       {report.latency_ms:.2f} ms")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 6: Position Status
    # -------------------------------------------------------------------------
    print("STEP 6: Position Status")
    print("-" * 60)
    
    position = engine.get_position(sizing_decision.symbol)
    if position:
        print(f"  Symbol:      {position.symbol}")
        print(f"  Side:        {position.side}")
        print(f"  Quantity:    {position.quantity:.6f} BTC")
        print(f"  Entry Price: ${position.entry_price:,.2f}")
        print(f"  Entry Time:  {position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("  No open position")
    print()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    print()
    print("  Layer 1: Cascade indicators provided (simulated)")
    print(f"  Layer 2: {l2_decision.action.name} decision, {l2_decision.confidence:.0%} confidence")
    print(f"  Layer 3: ${sizing_decision.position_size_usd:,.2f} position sized")
    print(f"  Execution: {report.status} at ${report.fill_price:,.2f}")
    print()
    print("=" * 80)
    print("[SUCCESS] Layer 2 -> Layer 3 Integration Complete!")
    print("=" * 80)
    
    return {
        'l2_decision': l2_decision,
        'l3_signal': l3_signal,
        'sizing_decision': sizing_decision,
        'execution_report': report,
        'position': position,
    }


def demo_multiple_scenarios():
    """Demonstrate multiple trading scenarios."""
    
    print()
    print("=" * 80)
    print("ADDITIONAL SCENARIOS")
    print("=" * 80)
    print()
    
    mapper = L2SignalMapper()
    engine = ExecutionEngine(paper_trading=True)
    layer3 = Layer3Phase1(portfolio_value=100000)
    
    scenarios = [
        ("SELL Signal, T2 Tier", "SELL", 0.72, 2),
        ("HOLD Signal, T1 Tier", "HOLD", 0.80, 1),
        ("STRONG_SELL, High Risk T3", "STRONG_SELL", 0.65, 3),
    ]
    
    for name, action, confidence, tier in scenarios:
        print(f"Scenario: {name}")
        print("-" * 40)
        
        decision = MockTacticalDecision(
            action=action,
            confidence=confidence,
            tier=tier,
        )
        
        signal = mapper.map_decision_to_signal(
            decision=decision,
            symbol="ETH-USD",
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5,
        )
        
        sizing = layer3.calculate_position(
            signal=signal,
            cascade_indicators=cascade,
            current_price=2500.0,
        )
        
        print(f"  L3 Action:     {signal.action.value}")
        print(f"  Risk Score:    {signal.risk_score:.2f}")
        print(f"  Position Size: ${sizing.position_size_usd:,.2f}")
        
        if sizing.position_size_usd > 0 and action != "HOLD":
            order = {
                'order_id': f'DEMO_{int(time.time())}',
                'symbol': 'ETH-USD',
                'side': 'BUY' if action in ['BUY', 'STRONG_BUY'] else 'SELL',
                'quantity': sizing.position_size_usd / 2500.0,
                'order_type': 'MARKET',
            }
            report = engine.submit_order(order, current_price=2500.0)
            print(f"  Execution:     {report.status}")
        else:
            print(f"  Execution:     SKIPPED (HOLD or zero size)")
        print()
    
    # Final stats
    stats = engine.get_performance_stats()
    print("Engine Stats:")
    print(f"  Total Trades:   {stats['total_trades']}")
    print(f"  Open Positions: {stats['open_positions']}")
    print(f"  Commissions:    ${stats['total_commission']:.2f}")


if __name__ == "__main__":
    run_l2_l3_integration_example()
    demo_multiple_scenarios()
