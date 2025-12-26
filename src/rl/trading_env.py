"""
HIMARI Layer 3 - Trading Environment
======================================

Gym-like environment for RL training with live price feed.

The environment simulates trading with real market data:
- Uses live prices from Binance
- Executes paper trades
- Tracks P&L and positions
- Calculates rewards based on Sharpe ratio

Version: 1.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time

from rl.state_encoder import StateEncoder, TradingState
from market_data.price_feed import get_price_feed
from engines.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Trading environment configuration."""
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5      # Max 50% of capital per position
    commission_rate: float = 0.001     # 0.1%
    slippage_bps: int = 5              # 0.05%
    reward_window: int = 10            # Window for Sharpe calculation
    max_steps: int = 1000              # Max steps per episode
    symbol: str = "BTC-USD"


class TradingEnvironment:
    """
    Trading environment for RL training.

    State: Market conditions + position + performance
    Action: Position size multiplier [0, 2]
    Reward: Sharpe ratio of recent returns
    """

    def __init__(self, config: Optional[EnvConfig] = None, use_live_prices: bool = True):
        """
        Initialize trading environment.

        Args:
            config: Environment configuration
            use_live_prices: Use live price feed from Binance
        """
        self.config = config or EnvConfig()
        self.use_live_prices = use_live_prices

        # State encoder
        self.state_encoder = StateEncoder(
            max_position_usd=self.config.initial_capital * self.config.max_position_pct
        )

        # Price feed
        if use_live_prices:
            self.price_feed = get_price_feed()
        else:
            self.price_feed = None

        # Execution engine (paper trading)
        self.execution_engine = ExecutionEngine(
            paper_trading=True,
            use_live_prices=use_live_prices,
            commission_rate=self.config.commission_rate,
            default_slippage_bps=self.config.slippage_bps
        )

        # Episode tracking
        self.current_step = 0
        self.capital = self.config.initial_capital
        self.returns = []
        self.trades = []

        # Position tracking
        self.current_position_usd = 0.0
        self.current_position_qty = 0.0
        self.entry_price = 0.0

        logger.info(
            f"TradingEnvironment initialized "
            f"(capital=${self.config.initial_capital:,.0f}, "
            f"live_prices={'on' if use_live_prices else 'off'})"
        )

    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.

        Returns:
            Initial state
        """
        self.current_step = 0
        self.capital = self.config.initial_capital
        self.returns = []
        self.trades = []
        self.current_position_usd = 0.0
        self.current_position_qty = 0.0
        self.entry_price = 0.0
        self.execution_engine.reset()

        logger.debug("Environment reset")
        return self._get_state()

    def _get_current_price(self) -> float:
        """Get current market price."""
        if self.use_live_prices and self.price_feed:
            price = self.price_feed.get_price(self.config.symbol)
            if price:
                return price

        # Fallback
        return 87000.0 if "BTC" in self.config.symbol else 2500.0

    def _get_state(self) -> np.ndarray:
        """
        Get current state.

        Returns:
            Encoded state vector
        """
        current_price = self._get_current_price()

        # Update price history
        self.state_encoder.update_price_history(current_price)

        # Calculate metrics
        momentum_1h = self.state_encoder.calculate_momentum(60)
        momentum_4h = self.state_encoder.calculate_momentum(240)
        volatility = self.state_encoder.calculate_volatility()

        # Calculate unrealized P&L
        unrealized_pnl_pct = 0.0
        if self.current_position_qty != 0 and self.entry_price > 0:
            if self.current_position_qty > 0:  # LONG
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Recent performance
        recent_win_rate = 0.5  # Default
        if len(self.trades) > 0:
            recent_trades = self.trades[-10:]
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            recent_win_rate = wins / len(recent_trades)

        # Total P&L %
        total_pnl_pct = (self.capital - self.config.initial_capital) / self.config.initial_capital

        # Create trading state
        trading_state = TradingState(
            signal_confidence=0.75,  # Simulated (would come from L2)
            signal_action=2,  # BUY (simulated)
            signal_tier=1,
            signal_regime=0,  # TRENDING_UP
            position_size_usd=abs(self.current_position_usd),
            position_side=np.sign(self.current_position_qty),
            unrealized_pnl_pct=unrealized_pnl_pct,
            price_momentum_1h=momentum_1h,
            price_momentum_4h=momentum_4h,
            volatility=volatility,
            recent_win_rate=recent_win_rate,
            recent_sharpe=self._calculate_sharpe(),
            total_pnl_pct=total_pnl_pct,
            cascade_risk=0.2,  # Simulated
            current_drawdown=min(0.0, total_pnl_pct),
            timestamp=time.time()
        )

        return self.state_encoder.encode_state(trading_state)

    def _calculate_sharpe(self) -> float:
        """
        Calculate Sharpe ratio of recent returns.

        Returns:
            Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0.0

        recent_returns = self.returns[-self.config.reward_window:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Position size multiplier [0, 2]

        Returns:
            next_state: Next state
            reward: Reward (Sharpe ratio)
            done: Whether episode is done
            info: Additional information
        """
        self.current_step += 1
        current_price = self._get_current_price()

        # Clamp action
        action = np.clip(action, 0.0, 2.0)

        # Calculate target position size
        base_position = self.capital * self.config.max_position_pct
        target_position_usd = base_position * action

        # Execute trade if position change needed
        position_change = target_position_usd - abs(self.current_position_usd)

        if abs(position_change) > self.capital * 0.01:  # Threshold: 1% of capital
            # Close existing position if necessary
            if self.current_position_qty != 0:
                self._close_position(current_price)

            # Open new position
            if target_position_usd > 0:
                self._open_position(target_position_usd, current_price)

        # Calculate return for this step
        step_return = 0.0
        if self.current_position_qty != 0 and self.entry_price > 0:
            if self.current_position_qty > 0:  # LONG
                step_return = (current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                step_return = (self.entry_price - current_price) / self.entry_price

        self.returns.append(step_return)

        # Calculate reward (Sharpe ratio)
        reward = self._calculate_sharpe()

        # Get next state
        next_state = self._get_state()

        # Check if done
        done = (
            self.current_step >= self.config.max_steps or
            self.capital <= self.config.initial_capital * 0.5  # 50% drawdown
        )

        # Info
        info = {
            'step': self.current_step,
            'capital': self.capital,
            'position_usd': self.current_position_usd,
            'action': action,
            'price': current_price,
            'sharpe': reward,
            'total_pnl_pct': (self.capital - self.config.initial_capital) / self.config.initial_capital,
        }

        return next_state, reward, done, info

    def _open_position(self, position_usd: float, current_price: float):
        """Open a position."""
        quantity = position_usd / current_price

        order = {
            'order_id': f'RL_OPEN_{self.current_step}',
            'symbol': self.config.symbol,
            'side': 'BUY',
            'quantity': quantity,
            'order_type': 'MARKET',
        }

        report = self.execution_engine.submit_order(order, current_price=current_price)

        self.current_position_usd = position_usd
        self.current_position_qty = quantity
        self.entry_price = report.fill_price
        self.capital -= report.commission

        logger.debug(f"Opened position: {quantity:.6f} @ ${report.fill_price:,.2f}")

    def _close_position(self, current_price: float):
        """Close current position."""
        if self.current_position_qty == 0:
            return

        order = {
            'order_id': f'RL_CLOSE_{self.current_step}',
            'symbol': self.config.symbol,
            'side': 'SELL',
            'quantity': abs(self.current_position_qty),
            'order_type': 'MARKET',
        }

        report = self.execution_engine.submit_order(order, current_price=current_price)

        # Calculate P&L
        if report.realized_pnl is not None:
            pnl = report.realized_pnl
        else:
            pnl = (current_price - self.entry_price) * self.current_position_qty

        self.capital += pnl - report.commission

        # Record trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': report.fill_price,
            'quantity': self.current_position_qty,
            'pnl': pnl,
            'pnl_pct': pnl / abs(self.current_position_usd) if self.current_position_usd != 0 else 0,
        })

        logger.debug(f"Closed position: P&L=${pnl:,.2f}")

        self.current_position_usd = 0.0
        self.current_position_qty = 0.0
        self.entry_price = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe': 0.0,
            }

        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades)

        return {
            'total_trades': len(self.trades),
            'win_rate': wins / len(self.trades),
            'total_pnl': total_pnl,
            'total_pnl_pct': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'sharpe': self._calculate_sharpe(),
            'current_capital': self.capital,
        }


def test_trading_env():
    """Test trading environment."""
    print("=" * 80)
    print("HIMARI RL - Trading Environment Test")
    print("=" * 80)
    print()

    # Create environment
    config = EnvConfig(initial_capital=100000, max_steps=100)
    env = TradingEnvironment(config, use_live_prices=True)

    print("Environment created:")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Max position:    {config.max_position_pct:.0%}")
    print(f"  Symbol:          {config.symbol}")
    print()

    # Reset environment
    print("Test 1: Reset environment")
    print("-" * 60)
    state = env.reset()
    print(f"  State shape:  {state.shape}")
    print(f"  State values: {state[:5]}... (showing first 5)")
    print()

    # Run a few steps
    print("Test 2: Run 10 steps with random actions")
    print("-" * 60)

    for step in range(10):
        action = np.random.rand() * 2.0  # Random multiplier [0, 2]
        next_state, reward, done, info = env.step(action)

        print(f"  Step {step+1}: action={action:.2f}, reward={reward:.3f}, "
              f"capital=${info['capital']:,.0f}, done={done}")

        if done:
            print("  Episode terminated early")
            break

    print()

    # Get statistics
    print("Test 3: Environment statistics")
    print("-" * 60)
    stats = env.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:20s} {value:.4f}")
        else:
            print(f"  {key:20s} {value}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_trading_env()
