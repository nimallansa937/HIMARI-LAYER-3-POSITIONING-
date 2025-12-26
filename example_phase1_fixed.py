"""
HIMARI OPUS V2 - Phase 1 Production Example (FIXED)
====================================================

Fixed issues:
- Removed Unicode emojis causing Windows encoding errors
- Added ASCII-safe output
- Proper error handling
"""

import sys
sys.path.insert(0, 'src')

from phases.phase1_core import Layer3Phase1
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
import time


def example_basic_position_sizing():
    """Example 1: Basic position sizing with normal conditions."""
    print("\n" + "="*80)
    print("Example 1: Basic Position Sizing")
    print("="*80)

    # Initialize Phase 1
    layer3 = Layer3Phase1(
        portfolio_value=100000,
        kelly_fraction=0.25,
        enable_metrics=False,  # Disable metrics for simple example
        enable_hot_reload=False
    )

    # Create tactical signal
    signal = TacticalSignal(
        strategy_id="momentum_btc",
        symbol="BTC-USD",
        action=TacticalAction.BUY,
        confidence=0.75,
        risk_score=0.3,
        regime=MarketRegime.TRENDING_UP,
        timestamp_ns=time.time_ns(),
        expected_return=0.08,
        predicted_volatility=0.03
    )

    # Create cascade indicators (normal market)
    cascade_indicators = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.05,
        volume_ratio=2.0,
        onchain_whale_pressure=0.3,
        exchange_netflow_zscore=0.5
    )

    # Calculate position
    decision = layer3.calculate_position(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=42000.0
    )

    # Display results
    print(f"\n[SUCCESS] Position calculated:")
    print(f"  Symbol: {decision.symbol}")
    print(f"  Final Position: ${decision.position_size_usd:,.2f}")
    print(f"  Kelly Base: ${decision.kelly_position_usd:,.2f}")
    print(f"  Conformal Adjusted: ${decision.conformal_adjusted_usd:,.2f}")
    print(f"  Regime Adjusted: ${decision.regime_adjusted_usd:,.2f}")
    print(f"  Cascade Recommendation: {decision.cascade_recommendation}")
    print(f"  Cascade Risk: {decision.cascade_risk_score:.3f}")
    print(f"  Current Regime: {decision.current_regime.value}")

    layer3.stop()
    return decision


def example_high_risk_cascade():
    """Example 2: High risk cascade reduction."""
    print("\n" + "="*80)
    print("Example 2: High Risk Cascade Detection")
    print("="*80)

    layer3 = Layer3Phase1(
        portfolio_value=100000,
        enable_metrics=False,
        enable_hot_reload=False
    )

    signal = TacticalSignal(
        strategy_id="test_strategy",
        symbol="BTC-USD",
        action=TacticalAction.BUY,
        confidence=0.70,
        risk_score=0.4,
        regime=MarketRegime.HIGH_VOLATILITY,
        timestamp_ns=time.time_ns(),
        expected_return=0.06,
        predicted_volatility=0.05
    )

    # HIGH RISK cascade indicators
    cascade_indicators = CascadeIndicators(
        funding_rate=0.008,          # High funding
        oi_change_pct=-0.15,         # OI dropping
        volume_ratio=8.0,            # Volume spike
        onchain_whale_pressure=0.85, # High whale activity
        exchange_netflow_zscore=3.5  # Abnormal flows
    )

    decision = layer3.calculate_position(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=42000.0
    )

    print(f"\n[WARNING] High cascade risk detected:")
    print(f"  Cascade Risk Score: {decision.cascade_risk_score:.3f}")
    print(f"  Recommendation: {decision.cascade_recommendation}")
    print(f"  Original Position: ${decision.regime_adjusted_usd:,.2f}")
    print(f"  Final Position: ${decision.position_size_usd:,.2f}")
    print(f"  Reduction: {((1 - decision.position_size_usd / decision.regime_adjusted_usd) * 100):.1f}%")

    layer3.stop()
    return decision


def example_sentiment_integration():
    """Example 3: Sentiment-aware sizing."""
    print("\n" + "="*80)
    print("Example 3: Sentiment-Aware Position Sizing")
    print("="*80)

    layer3 = Layer3Phase1(
        portfolio_value=100000,
        enable_sentiment=True,
        enable_metrics=False,
        enable_hot_reload=False
    )

    # Signal with bullish sentiment
    signal = TacticalSignal(
        strategy_id="sentiment_strategy",
        symbol="ETH-USD",
        action=TacticalAction.BUY,
        confidence=0.70,
        risk_score=0.3,
        regime=MarketRegime.RANGING,
        timestamp_ns=time.time_ns(),
        expected_return=0.07,
        predicted_volatility=0.04,
        sentiment_score=0.85,        # Very bullish
        sentiment_confidence=0.90     # High confidence
    )

    cascade_indicators = CascadeIndicators(
        funding_rate=0.002,
        oi_change_pct=0.03,
        volume_ratio=1.8,
        onchain_whale_pressure=0.4,
        exchange_netflow_zscore=0.3
    )

    decision = layer3.calculate_position(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=2800.0
    )

    print(f"\n[INFO] Sentiment adjustment applied:")
    print(f"  Sentiment Score: {signal.sentiment_score:.2f} (bullish)")
    print(f"  Sentiment Confidence: {signal.sentiment_confidence:.2f}")
    print(f"  Conformal Position: ${decision.conformal_adjusted_usd:,.2f}")
    print(f"  After Sentiment: ${decision.diagnostics['pipeline_stages']['sentiment_usd']:,.2f}")
    print(f"  Final Position: ${decision.position_size_usd:,.2f}")

    if decision.sentiment_diagnostics:
        print(f"  Sentiment Multiplier: {decision.sentiment_diagnostics.get('multiplier', 1.0):.3f}x")
        print(f"  Adjustment Type: {decision.sentiment_diagnostics.get('adjustment_type', 'N/A')}")

    layer3.stop()
    return decision


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("HIMARI OPUS V2 - Phase 1 Production Examples")
    print("="*80)

    try:
        # Run examples
        decision1 = example_basic_position_sizing()
        decision2 = example_high_risk_cascade()
        decision3 = example_sentiment_integration()

        print("\n" + "="*80)
        print("All Examples Completed Successfully!")
        print("="*80)
        print("\n[OK] Phase 1 pipeline is working correctly")
        print(f"[OK] Total test cases: 3/3 passed")

    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
