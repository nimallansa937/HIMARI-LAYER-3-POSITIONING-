"""
HIMARI Layer 3 - Execution Engine
==================================

Handles order execution on exchanges.
Currently implements paper trading mode for testing.

Version: 3.2 - Live Price Feed Integration
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Report of order execution."""
    order_id: str
    exchange_order_id: str
    symbol: str
    side: str
    status: str                            # FILLED, PARTIALLY_FILLED, REJECTED, CANCELLED
    requested_quantity: float
    fill_price: Optional[float]
    fill_quantity: float
    remaining_quantity: float
    slippage_bps: Optional[int]
    commission: float
    commission_asset: str
    submitted_at: datetime
    filled_at: Optional[datetime]
    latency_ms: float
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'exchange_order_id': self.exchange_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'status': self.status,
            'requested_quantity': self.requested_quantity,
            'fill_price': self.fill_price,
            'fill_quantity': self.fill_quantity,
            'remaining_quantity': self.remaining_quantity,
            'slippage_bps': self.slippage_bps,
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'submitted_at': self.submitted_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'latency_ms': self.latency_ms,
            'realized_pnl': self.realized_pnl,
            'realized_pnl_pct': self.realized_pnl_pct,
        }


@dataclass
class Position:
    """Track current position."""
    symbol: str
    side: str                              # LONG or SHORT
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Update unrealized PnL based on current price."""
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price


class ExecutionEngine:
    """
    Layer 3 Execution Engine.

    Responsibilities:
    1. Connect to exchanges (paper trading mode)
    2. Submit orders
    3. Track fills
    4. Manage positions
    5. Calculate PnL

    NOTE: This implements paper trading for testing.
    Real implementation would use exchange APIs (CCXT, etc.)
    """

    def __init__(
        self,
        exchange: str = "binance",
        paper_trading: bool = True,
        commission_rate: float = 0.001,  # 0.1% default
        default_slippage_bps: int = 5,   # 0.05% default slippage
        use_live_prices: bool = True,    # Use live price feed
    ):
        self.exchange = exchange
        self.paper_trading = paper_trading
        self.commission_rate = commission_rate
        self.default_slippage_bps = default_slippage_bps
        self.use_live_prices = use_live_prices

        # Order and position tracking
        self.orders: Dict[str, ExecutionReport] = {}
        self.positions: Dict[str, Position] = {}

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0

        # Initialize price feed if using live prices
        self.price_feed = None
        if self.use_live_prices:
            try:
                from market_data.price_feed import get_price_feed
                self.price_feed = get_price_feed(cache_seconds=5)
                logger.info("Live price feed enabled")
            except ImportError as e:
                logger.warning(f"Could not import price feed: {e}, using fallback prices")
                self.use_live_prices = False

        logger.info(
            f"Execution engine initialized "
            f"(exchange={exchange}, paper_trading={paper_trading}, "
            f"live_prices={'on' if use_live_prices else 'off'})"
        )

    def submit_order(
        self,
        order: Any,
        current_price: Optional[float] = None,
    ) -> ExecutionReport:
        """
        Submit order to exchange.

        Args:
            order: ExecutionOrder from Layer 2→3 bridge
            current_price: Current market price (for paper trading)

        Returns:
            ExecutionReport with execution details
        """
        submitted_at = datetime.now()
        
        # Get order properties - handle both dataclass and dict
        if hasattr(order, 'order_id'):
            order_id = order.order_id
            symbol = order.symbol
            side = order.side
            quantity = order.quantity
            order_type = order.order_type
            order_price = order.price
        else:
            order_id = order.get('order_id', str(uuid.uuid4()))
            symbol = order.get('symbol', 'BTCUSDT')
            side = order.get('side', 'BUY')
            quantity = order.get('quantity', 0.0)
            order_type = order.get('order_type', 'MARKET')
            order_price = order.get('price')

        if self.paper_trading:
            return self._execute_paper_trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                order_price=order_price,
                current_price=current_price,
                submitted_at=submitted_at,
            )
        else:
            # Real trading: use exchange API
            raise NotImplementedError("Real trading not yet implemented - use CCXT")

    def _execute_paper_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        order_price: Optional[float],
        current_price: Optional[float],
        submitted_at: datetime,
    ) -> ExecutionReport:
        """Execute order in paper trading mode."""

        logger.info(f"[PAPER] Executing {side} {quantity:.6f} {symbol}")

        # Determine fill price
        if order_type == "MARKET":
            if current_price is None:
                # Try to fetch live price
                if self.use_live_prices and self.price_feed:
                    live_price = self.price_feed.get_price(symbol)
                    if live_price:
                        fill_price = live_price
                        logger.info(f"[LIVE PRICE] {symbol}: ${fill_price:,.2f}")
                    else:
                        # Fallback to hardcoded
                        fill_price = 87000.0 if "BTC" in symbol else 2500.0
                        logger.warning(f"[FALLBACK] Using hardcoded price: ${fill_price:,.2f}")
                else:
                    # Use hardcoded price if live prices disabled
                    fill_price = 87000.0 if "BTC" in symbol else 2500.0
            else:
                fill_price = current_price
            
            # Apply slippage
            slippage_mult = 1 + (self.default_slippage_bps / 10000)
            if side == "BUY":
                fill_price *= slippage_mult
            else:
                fill_price /= slippage_mult
        else:
            # LIMIT order
            fill_price = order_price or current_price or 87000.0
        
        # Calculate commission
        trade_value = quantity * fill_price
        commission = trade_value * self.commission_rate
        
        # Simulate execution latency (1-5ms)
        time.sleep(0.002)
        filled_at = datetime.now()
        latency_ms = (filled_at - submitted_at).total_seconds() * 1000
        
        # Calculate realized PnL if closing position
        realized_pnl = None
        realized_pnl_pct = None
        
        if symbol in self.positions:
            position = self.positions[symbol]
            # Check if this closes the position
            closes_position = (
                (position.side == "LONG" and side == "SELL") or
                (position.side == "SHORT" and side == "BUY")
            )
            if closes_position:
                if position.side == "LONG":
                    realized_pnl = (fill_price - position.entry_price) * min(quantity, position.quantity)
                else:
                    realized_pnl = (position.entry_price - fill_price) * min(quantity, position.quantity)
                realized_pnl_pct = realized_pnl / (position.entry_price * min(quantity, position.quantity))
                realized_pnl -= commission  # Subtract commission
                
                # Update position or remove
                if quantity >= position.quantity:
                    del self.positions[symbol]
                else:
                    position.quantity -= quantity
                
                # Update stats
                self.total_pnl += realized_pnl
                self.total_trades += 1
                if realized_pnl > 0:
                    self.winning_trades += 1
        else:
            # Opening new position
            self.positions[symbol] = Position(
                symbol=symbol,
                side="LONG" if side == "BUY" else "SHORT",
                quantity=quantity,
                entry_price=fill_price,
                entry_time=filled_at,
            )
        
        self.total_commission += commission
        
        # Create execution report
        report = ExecutionReport(
            order_id=order_id,
            exchange_order_id=f"PAPER_{uuid.uuid4().hex[:12]}",
            symbol=symbol,
            side=side,
            status="FILLED",
            requested_quantity=quantity,
            fill_price=fill_price,
            fill_quantity=quantity,
            remaining_quantity=0.0,
            slippage_bps=self.default_slippage_bps,
            commission=commission,
            commission_asset="USDT",
            submitted_at=submitted_at,
            filled_at=filled_at,
            latency_ms=latency_ms,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
        )
        
        self.orders[order_id] = report
        
        logger.info(
            f"[PAPER] Order filled: {order_id} @ ${fill_price:,.2f} "
            f"(slippage={self.default_slippage_bps}bps, latency={latency_ms:.1f}ms)"
        )
        
        return report

    def get_order_status(self, order_id: str) -> Optional[ExecutionReport]:
        """Get status of an order."""
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id in self.orders:
            report = self.orders[order_id]
            if report.status in ["NEW", "PARTIALLY_FILLED"]:
                report.status = "CANCELLED"
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return self.positions.copy()

    def update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for a position."""
        if symbol in self.positions:
            self.positions[symbol].update_pnl(current_price)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        win_rate = (
            self.winning_trades / self.total_trades 
            if self.total_trades > 0 else 0.0
        )
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'total_commission': self.total_commission,
            'net_pnl': self.total_pnl - self.total_commission,
            'open_positions': len(self.positions),
        }

    def reset(self):
        """Reset engine state (for testing)."""
        self.orders.clear()
        self.positions.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        logger.info("Execution engine reset")
