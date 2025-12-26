"""
HIMARI Layer 3 - Metrics Demo
Runs continuously and exposes metrics for Grafana
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prometheus_client import start_http_server

# Import metrics
from core import layer3_metrics as metrics
from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from phases.phase2_portfolio import Layer3Phase2Portfolio


def main():
    print("=" * 60)
    print("HIMARI Layer 3 - Metrics Demo")
    print("=" * 60)
    print()
    
    # Start metrics server
    port = 8000
    start_http_server(port)
    print(f"[OK] Prometheus metrics server started on port {port}")
    print(f"     Access metrics at: http://localhost:{port}/metrics")
    print()
    
    # Initialize portfolio
    portfolio = Layer3Phase2Portfolio(
        portfolio_value=100000,
        enable_metrics=True  # Metrics enabled
    )
    print("[OK] Phase 2 Portfolio initialized")
    print()
    
    # Cascade indicators
    cascade = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.05,
        volume_ratio=2.0,
        onchain_whale_pressure=0.3,
        exchange_netflow_zscore=0.5
    )
    
    print("Running continuous metrics simulation...")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 60)
    
    iteration = 0
    while True:
        iteration += 1
        
        # Create random signals
        signals = [
            TacticalSignal(
                strategy_id="momentum_btc",
                symbol="BTC-USD",
                action=TacticalAction.BUY,
                confidence=0.6 + random.random() * 0.3,
                risk_score=0.2 + random.random() * 0.3,
                regime=MarketRegime.TRENDING_UP,
                timestamp_ns=time.time_ns(),
                expected_return=0.08 + random.random() * 0.1,
                predicted_volatility=0.03 + random.random() * 0.02
            ),
            TacticalSignal(
                strategy_id="momentum_eth",
                symbol="ETH-USD",
                action=TacticalAction.BUY,
                confidence=0.5 + random.random() * 0.3,
                risk_score=0.25 + random.random() * 0.3,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
                expected_return=0.10 + random.random() * 0.1,
                predicted_volatility=0.04 + random.random() * 0.02
            )
        ]
        
        prices = {
            'BTC-USD': 42000 + random.randint(-1000, 1000),
            'ETH-USD': 2200 + random.randint(-100, 100)
        }
        
        # Process allocation
        allocation = portfolio.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade,
            current_prices=prices
        )
        
        # Update correlation
        portfolio.update_correlations({
            'BTC-USD': random.gauss(0, 0.01),
            'ETH-USD': random.gauss(0, 0.012)
        })
        
        print(f"[{iteration}] Allocated ${allocation.total_allocated_usd:,.2f} "
              f"({allocation.utilization_pct:.1f}%)")
        
        # Wait before next iteration
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
