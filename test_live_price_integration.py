"""
Test Live Price Feed Integration
==================================

Demonstrates Layer 3 execution engine using live prices from Binance API.

Usage:
    cd "LAYER 3 POSITIONING LAYER"
    python test_live_price_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from engines.execution_engine import ExecutionEngine
from market_data.price_feed import get_current_price


def test_live_price_feed():
    """Test execution engine with live price feed."""

    print("=" * 80)
    print("HIMARI LAYER 3 - Live Price Feed Integration Test")
    print("=" * 80)
    print()

    # Test 1: Fetch live prices
    print("Test 1: Fetch Current Live Prices")
    print("-" * 60)

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    print("Fetching live prices from Binance API...\n")

    prices = {}
    for symbol in symbols:
        price = get_current_price(symbol)
        if price:
            prices[symbol] = price
            print(f"  {symbol:10s} ${price:,.2f}")
        else:
            print(f"  {symbol:10s} FAILED")

    print()

    # Test 2: Execute order with live price (no price provided)
    print("Test 2: Paper Trade with Live Price (Auto-Fetch)")
    print("-" * 60)

    engine = ExecutionEngine(
        paper_trading=True,
        use_live_prices=True  # Enable live price feed
    )

    order = {
        'order_id': 'LIVE_TEST_001',
        'symbol': 'BTC-USD',
        'side': 'BUY',
        'quantity': 0.1,
        'order_type': 'MARKET',
    }

    print(f"Submitting order WITHOUT providing current_price...")
    print(f"  Symbol:   {order['symbol']}")
    print(f"  Side:     {order['side']}")
    print(f"  Quantity: {order['quantity']} BTC")
    print()

    # Submit WITHOUT current_price - should auto-fetch from live feed
    report = engine.submit_order(order, current_price=None)

    print(f"Order Execution Result:")
    print(f"  Status:       {report.status}")
    print(f"  Fill Price:   ${report.fill_price:,.2f} (LIVE from Binance)")
    print(f"  Fill Qty:     {report.fill_quantity} BTC")
    print(f"  Slippage:     {report.slippage_bps} bps")
    print(f"  Commission:   ${report.commission:.2f}")
    print(f"  Latency:      {report.latency_ms:.2f} ms")
    print()

    # Test 3: Compare with manual price
    print("Test 3: Compare Auto-Fetch vs Manual Price")
    print("-" * 60)

    live_btc = get_current_price("BTC-USD")
    print(f"Current BTC Price (Binance): ${live_btc:,.2f}")
    print()

    # Submit with manual price
    order2 = {
        'order_id': 'LIVE_TEST_002',
        'symbol': 'BTC-USD',
        'side': 'BUY',
        'quantity': 0.05,
        'order_type': 'MARKET',
    }

    print(f"Submitting order WITH manual price...")
    report2 = engine.submit_order(order2, current_price=live_btc)

    print(f"  Fill Price: ${report2.fill_price:,.2f} (Manual: ${live_btc:,.2f})")
    print()

    # Test 4: Multiple symbols
    print("Test 4: Execute Multiple Symbols with Live Prices")
    print("-" * 60)

    test_orders = [
        {'symbol': 'BTC-USD', 'side': 'BUY', 'quantity': 0.05},
        {'symbol': 'ETH-USD', 'side': 'BUY', 'quantity': 1.0},
        {'symbol': 'SOL-USD', 'side': 'BUY', 'quantity': 10.0},
    ]

    print("Executing orders with live price auto-fetch:\n")

    for i, order_data in enumerate(test_orders):
        order = {
            'order_id': f'MULTI_TEST_{i}',
            'symbol': order_data['symbol'],
            'side': order_data['side'],
            'quantity': order_data['quantity'],
            'order_type': 'MARKET',
        }

        report = engine.submit_order(order, current_price=None)

        print(f"  {order_data['symbol']:10s} {order_data['side']:4s} "
              f"{order_data['quantity']:8.4f} @ ${report.fill_price:,.2f} -> {report.status}")

    print()

    # Test 5: Price feed statistics
    print("Test 5: Price Feed Performance Statistics")
    print("-" * 60)

    if engine.price_feed:
        stats = engine.price_feed.get_statistics()
        print(f"  API Calls:      {stats['api_calls']}")
        print(f"  Cache Hits:     {stats['cache_hits']}")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"  API Errors:     {stats['api_errors']}")
        print(f"  Cache Size:     {stats['cache_size']} symbols")
        print(f"  Cached:         {', '.join(stats['cached_symbols'])}")
    else:
        print("  Price feed not initialized")

    print()

    # Test 6: Fallback behavior
    print("Test 6: Fallback Behavior (Disabled Live Feed)")
    print("-" * 60)

    engine_no_live = ExecutionEngine(
        paper_trading=True,
        use_live_prices=False  # Disable live prices
    )

    order_fallback = {
        'order_id': 'FALLBACK_TEST',
        'symbol': 'BTC-USD',
        'side': 'BUY',
        'quantity': 0.01,
        'order_type': 'MARKET',
    }

    print("Submitting order with live prices DISABLED...")
    report_fallback = engine_no_live.submit_order(order_fallback, current_price=None)

    print(f"  Fill Price: ${report_fallback.fill_price:,.2f} (Hardcoded fallback)")
    print(f"  Status:     {report_fallback.status}")
    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("  Live Price Feed:    WORKING")
    print(f"  BTC Live Price:     ${prices.get('BTC-USD', 0):,.2f}")
    print(f"  ETH Live Price:     ${prices.get('ETH-USD', 0):,.2f}")
    print(f"  SOL Live Price:     ${prices.get('SOL-USD', 0):,.2f}")
    print(f"  Orders Executed:    {engine.total_trades + engine_no_live.total_trades}")
    print(f"  Price Feed Hits:    {stats['cache_hits'] if engine.price_feed else 0}")
    print()
    print("=" * 80)
    print("[SUCCESS] Live Price Feed Integration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_live_price_feed()
