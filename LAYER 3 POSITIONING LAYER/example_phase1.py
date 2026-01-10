"""
HIMARI OPUS V2 - Complete Phase 1 Example
==========================================

Demonstrates production Phase 1 with ALL integrations:
- Hot-reload configuration
- Prometheus metrics
- Sentiment-aware sizing
- Input validation
- Error handling

Version: 3.1 Production
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from phases.phase1_core import Layer3Phase1
from integration.l1_signal_mapper import L1SignalMapper

# Optional: Start Prometheus metrics server
try:
    from prometheus_client import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def example_full_integration():
    """Complete example with all Phase 1 features."""
    
    print("=" * 80)
    print("HIMARI OPUS V2 - Phase 1 Production Example")
    print("=" * 80)
    print()
    
    # Start Prometheus metrics server (optional)
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(8000)
            print("✅ Prometheus metrics server started on http://localhost:8000/metrics")
        except Exception as e:
            print(f"⚠️  Prometheus server not started: {e}")
    else:
        print("⚠️  prometheus_client not installed, metrics disabled")
    print()
    
    # Initialize Phase 1 with all features enabled
    print("1. Initializing Phase 1 with FULL integration...")
    layer3 = Layer3Phase1(
        portfolio_value=100000,
        kelly_fraction=0.25,
        config_path="config/layer3_config.yaml",  # Hot-reload enabled
        enable_hot_reload=True,
        enable_metrics=True,
        enable_sentiment=True
    )
    
    print(f"   Portfolio: $100,000")
    print(f"   Hot-reload: ENABLED")
    print(f"   Prometheus metrics: ENABLED")
    print(f"   Sentiment sizing: ENABLED")
    print()
    
    # Create tactical signal WITH sentiment (L2)
    print("2. Creating tactical signal WITH sentiment...")
    signal = TacticalSignal(
        strategy_id="momentum_btc",
        symbol="BTC-USD",
        action=TacticalAction.BUY,
        confidence=0.75,
        risk_score=0.3,
        regime=MarketRegime.TRENDING_UP,
        timestamp_ns=time.time_ns(),
        expected_return=0.08,
        predicted_volatility=0.03,
        # Sentiment fields from Layer 2
        sentiment_score=0.75,  # Bullish
        sentiment_confidence=0.8
    )
    print(f"   Symbol: {signal.symbol}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Sentiment Score: {signal.sentiment_score:.2f} (bullish)")
    print(f"   Sentiment Confidence: {signal.sentiment_confidence:.2f}")
    print()
    
    # Create cascade indicators using L1 signal mapper
    print("3. Mapping Layer 1 antigravity signals to cascade indicators...")
    mapper = L1SignalMapper()
    
    l1_signals = {
        'antigravity': {
            'coherence': 0.4,  # FSI - moderate
            'entropy': 0.5,   # LEI - neutral
            'energy_density': 0.2,  # SCSI - low
            'schwarzschild_radius': 0.35,  # LCI - moderate
            'hawking_temperature': 0.55  # CACI - slightly elevated
        }
    }
    
    cascade_indicators = mapper.map_to_cascade_indicators(l1_signals)
    print(f"   Whale Pressure: {cascade_indicators.onchain_whale_pressure:.2f}")
    print(f"   Netflow Z-score: {cascade_indicators.exchange_netflow_zscore:.2f}")
    print()
    
    # Calculate position size
    print("4. Calculating position size through FULL pipeline...")
    decision = layer3.calculate_position(
        signal=signal,
        cascade_indicators=cascade_indicators,
        current_price=42000.0
    )
    
    print()
    print("-" * 80)
    print("POSITION SIZING DECISION (Full Integration)")
    print("-" * 80)
    print(f"Final Position Size: ${decision.position_size_usd:,.2f}")
    print()
    print("Pipeline Breakdown:")
    stages = decision.diagnostics['pipeline_stages']
    print(f"  1. Kelly Base:         ${stages['kelly_usd']:,.2f}")
    print(f"  2. Conformal Adjusted: ${stages['conformal_usd']:,.2f}")
    print(f"  3. Sentiment Adjusted: ${stages['sentiment_usd']:,.2f}")
    print(f"  4. Regime Adjusted:    ${stages['regime_usd']:,.2f}")
    print(f"  5. Cascade Adjusted:   ${stages['cascade_usd']:,.2f}")
    print()
    
    # Show sentiment impact
    sentiment_diag = decision.diagnostics.get('sentiment', {})
    print(f"Sentiment Impact:")
    print(f"  Type: {sentiment_diag.get('adjustment_type', 'N/A')}")
    print(f"  Multiplier: {sentiment_diag.get('multiplier', 1.0):.3f}")
    print()
    
    print(f"Cascade Risk: {decision.cascade_risk_score:.3f}")
    print(f"Recommendation: {decision.cascade_recommendation}")
    print(f"Current Regime: {decision.current_regime.value}")
    print("-" * 80)
    print()
    
    # Simulate trade result feedback
    print("5. Simulating trade result feedback...")
    layer3.update_from_trade_result(
        predicted_return=0.08,
        actual_return=0.07,
        trade_won=True
    )
    print("   ✅ Trade result processed (win, 7% return)")
    print()
    
    # Show component states
    print("6. Component States:")
    state = layer3.get_state()
    print(f"   Bayesian Win Rate: {state['bayesian_kelly']['posterior_win_rate']:.2%}")
    print(f"   Conformal Samples: {state['conformal_scaler']['samples']}")
    print(f"   Sentiment Adjustments: {state['sentiment_sizer']['total_adjustments']}")
    print(f"   False Regime Flips: {state['regime_adjuster']['false_flip_count']}")
    print()
    
    # Show config manager status
    if 'config_manager' in state:
        print(f"   Config Hot-Reload: {'ACTIVE' if state['config_manager']['watcher_running'] else 'INACTIVE'}")
    print()
    
    # Stop background services
    layer3.stop()
    print("✅ Phase 1 stopped cleanly")
    print()
    
    return decision


def example_input_validation():
    """Demonstrate input validation."""
    
    print("=" * 80)
    print("INPUT VALIDATION EXAMPLE")
    print("=" * 80)
    print()
    
    layer3 = Layer3Phase1(portfolio_value=100000, enable_metrics=False)
    
    # Test invalid signal
    print("Testing invalid signal (confidence out of range)...")
    
    try:
        invalid_signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=TacticalAction.BUY,
            confidence=1.5,  # INVALID: > 1.0
            risk_score=0.3,
            regime=MarketRegime.RANGING,
            timestamp_ns=time.time_ns()
        )
        
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.05,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5
        )
        
        decision = layer3.calculate_position(invalid_signal, cascade, 42000.0)
        print(f"   Result: Validation failed, position = ${decision.position_size_usd}")
        
    except Exception as e:
        print(f"   ✅ Validation correctly rejected: {e}")
    
    print()
    layer3.stop()


if __name__ == "__main__":
    decision = example_full_integration()
    print()
    example_input_validation()
    
    print("=" * 80)
    print("✅ All examples completed successfully!")
    print()
    print("Prometheus metrics available at: http://localhost:8000/metrics")
    print("Grafana dashboards at: http://localhost:3001 (admin/himari123)")
    print("=" * 80)
