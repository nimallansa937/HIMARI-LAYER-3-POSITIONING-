"""
HIMARI OPUS 2 - Layer 3 Five-Tier Architecture Example
=======================================================

Demonstrates the complete Layer 3 Position Sizing Engine
using the 5-tier architecture per CLAUDE Guide.

Version: 1.0
"""

import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main engine
from engines.layer3_engine import Layer3PositionSizingEngine
from core.layer3_config import Layer3Config


def create_sample_signal() -> dict:
    """Create a sample trading signal from Layer 2."""
    return {
        'signal_id': f'sig_{int(time.time()*1000)}',
        'timestamp': int(time.time() * 1000),
        'symbol': 'BTC/USDT',
        'direction': 'LONG',
        'confidence': 0.72,
        'strategy_id': 'momentum_5m_v2',
        
        'market_context': {
            'realized_vol_5d': 0.45,      # 45% annualized
            'realized_vol_20d': 0.38,     # 38% annualized
            'funding_rate': 0.0008,       # 0.08%
            'open_interest': 450_000_000, # $450M
            'open_interest_delta_1h': -0.02,  # -2% OI change
            'volume_24h': 12_000_000_000, # $12B
            'volume_spike_ratio': 1.5,    # 1.5x normal
            'bid_ask_spread': 0.0001,     # 0.01%
            'btc_correlation': 1.0,       # BTC is itself
            'current_price': 43000.0,
            'atr_14': 1250.0,
        },
        
        'portfolio_state': {
            'portfolio_equity': 100000.0,
            'cash_available': 50000.0,
            'daily_pnl': 450.0,
            'daily_pnl_pct': 0.0045,      # +0.45%
            'drawdown_from_hwm': 0.021,   # 2.1%
            'open_positions': [],
            'sector_exposures': {},
        },
        
        'regime': 'NORMAL',
        'regime_confidence': 0.82,
    }


def demonstrate_tier_by_tier():
    """Demonstrate each tier in the pipeline."""
    
    print("\n" + "="*70)
    print("HIMARI Layer 3: Five-Tier Position Sizing Architecture")
    print("="*70)
    
    # Initialize engine with default config
    config = Layer3Config()
    engine = Layer3PositionSizingEngine(config)
    
    # Create sample signal
    signal = create_sample_signal()
    
    print(f"\n📊 Input Signal:")
    print(f"   Symbol: {signal['symbol']}")
    print(f"   Direction: {signal['direction']}")
    print(f"   Confidence: {signal['confidence']:.2f}")
    print(f"   Portfolio: ${signal['portfolio_state']['portfolio_equity']:,.2f}")
    print(f"   Regime: {signal['regime']} (conf={signal['regime_confidence']:.2f})")
    print(f"   5d Vol: {signal['market_context']['realized_vol_5d']:.1%}")
    print(f"   20d Vol: {signal['market_context']['realized_vol_20d']:.1%}")
    
    # Process signal
    output = engine.process_signal(signal)
    
    # Display tier-by-tier results
    diag = output.diagnostics
    
    print("\n" + "-"*70)
    print("TIER 1: Volatility Targeting (Deterministic Core)")
    print("-"*70)
    t1 = diag.get('tier_1', {})
    print(f"   Target Vol: {t1.get('target_vol', 0):.1%}")
    print(f"   Realized Vol: {t1.get('realized_vol', 0):.1%}")
    print(f"   Vol Ratio: {t1.get('vol_ratio', 0):.2f}")
    print(f"   Base Position: ${t1.get('position_size_usd', 0):,.2f}")
    
    print("\n" + "-"*70)
    print("TIER 2: Bounded Adaptive Enhancement")
    print("-"*70)
    t2 = diag.get('tier_2', {}).get('combined', {})
    print(f"   RL Delta: {t2.get('rl_delta', 0):+.2%}")
    print(f"   Funding Mult: {t2.get('funding_mult', 1):.2f}")
    print(f"   Correlation Mult: {t2.get('corr_mult', 1):.2f}")
    print(f"   Cascade Mult: {t2.get('cascade_mult', 1):.2f}")
    print(f"   Combined Multiplier: {t2.get('bounded_multiplier', 1):.3f}")
    print(f"   Adjusted Position: ${t2.get('adjusted_position', 0):,.2f}")
    
    print("\n" + "-"*70)
    print("TIER 3: Regime-Conditional Adjustment")
    print("-"*70)
    t3 = diag.get('tier_3', {})
    print(f"   Regime: {t3.get('regime', 'UNKNOWN')}")
    print(f"   Regime Multiplier: {t3.get('regime_multiplier', 1):.2f}")
    print(f"   Regime-Adj Position: ${t3.get('output_position', 0):,.2f}")
    
    print("\n" + "-"*70)
    print("TIER 4: Hard Constraint Enforcement")
    print("-"*70)
    t4 = diag.get('tier_4', {})
    print(f"   Max Single Position: ${t4.get('max_single', 0):,.2f}")
    print(f"   Constraints Hit: {t4.get('constraints_hit', [])}")
    print(f"   Constrained Position: ${t4.get('constrained_position', 0):,.2f}")
    
    print("\n" + "-"*70)
    print("TIER 5: Circuit Breakers")
    print("-"*70)
    t5 = diag.get('tier_5', {})
    print(f"   Breaker Status: {t5.get('breaker_status', 'UNKNOWN')}")
    print(f"   Final Position: ${t5.get('final_position', 0):,.2f}")
    
    print("\n" + "="*70)
    print("FINAL OUTPUT TO LAYER 4")
    print("="*70)
    print(f"   ✅ Position Size: ${output.position_size_usd:,.2f}")
    print(f"   ✅ Position %: {output.position_size_pct:.1%}")
    print(f"   ✅ Leverage: {output.leverage:.2f}x")
    print(f"   ✅ Stop Distance: ${output.stop_distance:,.2f}")
    print(f"   ✅ Stop Price: ${output.stop_price:,.2f}")
    print(f"   ✅ Risk/Trade: ${output.risk_per_trade_usd:,.2f}")
    print(f"   ✅ Breaker Status: {output.breaker_status}")
    print(f"   ✅ Actionable: {output.is_actionable()}")
    print(f"   ⏱️  Latency: {diag.get('latency_ms', 0):.1f}ms")
    
    return output


def demonstrate_regime_scenarios():
    """Demonstrate how regime affects position sizing."""
    
    print("\n" + "="*70)
    print("REGIME SCENARIO COMPARISON")
    print("="*70)
    
    config = Layer3Config()
    engine = Layer3PositionSizingEngine(config)
    
    regimes = ['NORMAL', 'HIGH_VOL', 'CRISIS', 'CASCADE']
    
    for regime in regimes:
        signal = create_sample_signal()
        signal['regime'] = regime
        signal['regime_confidence'] = 0.90
        
        output = engine.process_signal(signal)
        
        print(f"\n{regime:10} → ${output.position_size_usd:>8,.0f} ({output.position_size_pct:>5.1%})")


def demonstrate_circuit_breakers():
    """Demonstrate circuit breaker activation."""
    
    print("\n" + "="*70)
    print("CIRCUIT BREAKER SCENARIOS")
    print("="*70)
    
    config = Layer3Config()
    engine = Layer3PositionSizingEngine(config)
    
    # Scenario 1: Normal
    signal = create_sample_signal()
    output = engine.process_signal(signal)
    print(f"\n✅ Normal: ${output.position_size_usd:,.0f} | Breaker: {output.breaker_status}")
    
    # Scenario 2: Daily drawdown exceeded
    signal = create_sample_signal()
    signal['portfolio_state']['daily_pnl_pct'] = -0.04  # -4% drawdown
    output = engine.process_signal(signal)
    print(f"🛑 Drawdown -4%: ${output.position_size_usd:,.0f} | Breaker: {output.breaker_status}")
    
    # Scenario 3: Volatility spike
    signal = create_sample_signal()
    signal['market_context']['vol_spike_ratio'] = 4.0  # 4x normal
    output = engine.process_signal(signal)
    print(f"⚡ Vol Spike 4x: ${output.position_size_usd:,.0f} | Breaker: {output.breaker_status}")
    
    # Scenario 4: Spread blowout
    signal = create_sample_signal()
    signal['market_context']['bid_ask_spread'] = 0.008  # 0.8%
    output = engine.process_signal(signal)
    print(f"📈 Spread 0.8%: ${output.position_size_usd:,.0f} | Breaker: {output.breaker_status}")


if __name__ == "__main__":
    print("\n🚀 HIMARI Layer 3 Five-Tier Architecture Demo\n")
    
    # Run demonstrations
    demonstrate_tier_by_tier()
    demonstrate_regime_scenarios()
    demonstrate_circuit_breakers()
    
    print("\n" + "="*70)
    print("✅ Layer 3 Five-Tier Architecture Demo Complete!")
    print("="*70 + "\n")
