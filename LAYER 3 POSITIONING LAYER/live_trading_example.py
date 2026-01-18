"""
HIMARI Layer 3: Live Trading Example
=====================================

Example script for paper trading or live trading with the transition strategy.

Features:
- Real-time price fetching from exchange
- Position management
- Trade logging
- Risk management
- Performance tracking

Usage:
    # Paper trading (no real orders)
    python live_trading_example.py --paper

    # Live trading (real orders - BE CAREFUL!)
    python live_trading_example.py --live --api-key YOUR_KEY --api-secret YOUR_SECRET
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Optional
import pandas as pd

# Try to import ccxt for exchange connectivity
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("WARNING: ccxt not installed. Install with: pip install ccxt")

from production_transition_strategy import TransitionStrategy


class LiveTrader:
    """
    Live trading wrapper for the transition strategy.
    """

    def __init__(
        self,
        exchange: Optional[object] = None,
        strategy: TransitionStrategy = None,
        paper_trading: bool = True,
        log_file: str = "trades.log"
    ):
        self.exchange = exchange
        self.strategy = strategy or TransitionStrategy(window_hours=6)
        self.paper_trading = paper_trading
        self.log_file = log_file

        # State
        self.current_position = 0.0
        self.equity = 100000.0
        self.trades = []
        self.last_log_time = None

        # Performance tracking
        self.equity_curve = []
        self.start_time = datetime.now()

    def get_current_price(self) -> float:
        """Fetch current BTC price from exchange."""
        if self.exchange:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            return ticker['last']
        else:
            # Fallback: use dummy data for testing
            print("WARNING: No exchange connected. Using dummy price.")
            return 50000.0

    def execute_trade(self, target_position: float, current_price: float, reason: str):
        """Execute a trade (or simulate in paper trading mode)."""
        position_change = target_position - self.current_position

        if abs(position_change) < 0.01:  # No significant change
            return

        timestamp = datetime.now()

        trade = {
            'timestamp': timestamp.isoformat(),
            'price': current_price,
            'old_position': self.current_position,
            'new_position': target_position,
            'change': position_change,
            'reason': reason,
            'equity': self.equity,
            'mode': 'PAPER' if self.paper_trading else 'LIVE'
        }

        # Log trade
        self.log_trade(trade)

        if self.paper_trading:
            print(f"\n[PAPER TRADE] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if position_change > 0:
                print(f"  BUY: {position_change*100:.1f}% -> Total: {target_position*100:.1f}%")
            else:
                print(f"  SELL: {abs(position_change)*100:.1f}% -> Total: {target_position*100:.1f}%")
            print(f"  Price: ${current_price:,.2f}")
            print(f"  Reason: {reason}")
            print(f"  Equity: ${self.equity:,.2f}")
        else:
            print(f"\n[LIVE TRADE] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            # Real trading logic here
            # self.exchange.create_order(...)
            print(f"  WARNING: Live trading not fully implemented!")

        self.current_position = target_position
        self.trades.append(trade)

    def log_trade(self, trade: dict):
        """Append trade to log file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(trade) + '\n')

    def update_equity(self, current_price: float):
        """Update equity based on position and price movement."""
        if len(self.strategy.returns_history) > 0:
            ret = self.strategy.returns_history[-1]
            strategy_return = self.current_position * ret

            # Apply commission if position changed
            if len(self.trades) > 0:
                last_trade = self.trades[-1]
                if last_trade['new_position'] != self.current_position:
                    commission = self.strategy.get_commission_cost(
                        last_trade['new_position'],
                        self.current_position
                    )
                    strategy_return -= commission

            self.equity *= (1 + strategy_return)
            self.equity_curve.append(self.equity)

    def print_status(self, signal: dict, current_price: float):
        """Print current status."""
        timestamp = datetime.now()

        # Only print every hour to avoid spam
        if self.last_log_time is None or \
           (timestamp - self.last_log_time).total_seconds() >= 3600:

            print(f"\n{'='*70}")
            print(f"Status Update: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            print(f"Price: ${current_price:,.2f}")
            print(f"Regime: {signal['regime']}")
            print(f"Position: {self.current_position*100:.1f}%")
            print(f"Equity: ${self.equity:,.2f}")

            if signal['in_transition']:
                print(f"\nACTIVE TRANSITION:")
                print(f"  Type: {signal['transition_type'][0]} -> {signal['transition_type'][1]}")
                print(f"  Hours remaining: {signal['hours_remaining']}")
                print(f"  Signal strength: {signal['signal_strength']}")
                print(f"  Recommended position: {signal['position']*100:.1f}%")

            # Performance metrics
            if len(self.equity_curve) > 24:  # At least 24 hours
                returns = pd.Series(self.equity_curve).pct_change().dropna()
                sharpe = returns.mean() / (returns.std() + 1e-8) * (252 * 24) ** 0.5
                total_return = (self.equity - 100000) / 100000

                print(f"\nPerformance (since start):")
                print(f"  Return: {total_return*100:+.2f}%")
                print(f"  Sharpe: {sharpe:+.2f}")
                print(f"  Trades: {len(self.trades)}")

            self.last_log_time = timestamp

    def run(self, update_interval: int = 3600):
        """
        Main trading loop.

        Args:
            update_interval: Seconds between updates (3600 = 1 hour)
        """
        print("=" * 70)
        print("HIMARI Layer 3: Live Trading Started")
        print("=" * 70)
        print(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        print(f"Update interval: {update_interval}s ({update_interval/3600:.1f}h)")
        print(f"Target transitions: {self.strategy.target_transitions}")
        print(f"Log file: {self.log_file}")
        print("=" * 70)
        print()

        try:
            while True:
                # Get current price
                current_price = self.get_current_price()
                current_time = datetime.now()

                # Get signal from strategy
                signal = self.strategy.get_position(current_price, current_time)

                # Update equity
                self.update_equity(current_price)

                # Check if we need to trade
                if abs(signal['position'] - self.current_position) > 0.01:
                    reason = f"{signal['transition_type']} transition" if signal['in_transition'] else "Exit position"
                    self.execute_trade(signal['position'], current_price, reason)

                # Print status
                self.print_status(signal, current_price)

                # Wait for next update
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n\nTrading stopped by user.")
            self.print_final_summary()

    def print_final_summary(self):
        """Print final trading summary."""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        total_return = (self.equity - 100000) / 100000
        print(f"Starting equity: $100,000.00")
        print(f"Final equity: ${self.equity:,.2f}")
        print(f"Total return: {total_return*100:+.2f}%")
        print(f"Total trades: {len(self.trades)}")

        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe = returns.mean() / (returns.std() + 1e-8) * (252 * 24) ** 0.5
            print(f"Sharpe ratio: {sharpe:+.2f}")

        print(f"\nTrades logged to: {self.log_file}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='HIMARI Layer 3 Live Trading')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (no real orders)')
    parser.add_argument('--live', action='store_true', help='Live trading mode (REAL MONEY!)')
    parser.add_argument('--interval', type=int, default=3600, help='Update interval in seconds (default: 3600 = 1h)')
    parser.add_argument('--window', type=int, default=6, help='Transition window in hours (default: 6)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name (default: binance)')
    parser.add_argument('--api-key', type=str, help='Exchange API key')
    parser.add_argument('--api-secret', type=str, help='Exchange API secret')
    parser.add_argument('--log-file', type=str, default='trades.log', help='Trade log file')

    args = parser.parse_args()

    # Validate mode
    if not args.paper and not args.live:
        print("ERROR: Must specify either --paper or --live mode")
        sys.exit(1)

    if args.live and not args.paper:
        confirm = input("\nWARNING: You are about to start LIVE TRADING with REAL MONEY!\nType 'YES' to confirm: ")
        if confirm != 'YES':
            print("Live trading cancelled.")
            sys.exit(0)

    # Initialize exchange
    exchange = None
    if CCXT_AVAILABLE and (args.api_key or args.paper):
        try:
            exchange_class = getattr(ccxt, args.exchange)
            exchange = exchange_class({
                'apiKey': args.api_key or '',
                'secret': args.api_secret or '',
                'enableRateLimit': True,
            })

            if args.paper:
                exchange.set_sandbox_mode(True)  # Use testnet if available

            print(f"Connected to {args.exchange}")
        except Exception as e:
            print(f"WARNING: Could not connect to exchange: {e}")
            print("Continuing without live price feed...")

    # Initialize strategy
    strategy = TransitionStrategy(
        window_hours=args.window,
        target_transitions=[("NEUTRAL", "BULL")]
    )

    # Initialize trader
    trader = LiveTrader(
        exchange=exchange,
        strategy=strategy,
        paper_trading=args.paper or not args.live,
        log_file=args.log_file
    )

    # Start trading
    trader.run(update_interval=args.interval)


if __name__ == "__main__":
    main()
