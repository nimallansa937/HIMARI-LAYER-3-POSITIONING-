"""
HIMARI Layer 3 - Trading Environment (FIXED v2.0)
==================================================

Gym-like environment for RL training with:
- Risk-aware reward function (External Numba-optimized module)
- Bounded delta output (±30% from base)
- Regime gating (disable RL in CRISIS/CASCADE)
- Proper transaction cost penalties

Based on 76-paper systematic literature review best practices. 

Version: 2.1 (External Reward Integration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

from rl.state_encoder import StateEncoder, TradingState
from market_data.price_feed import get_price_feed
from engines.execution_engine import ExecutionEngine
# === NEW: Import advanced reward calculator ===
from rl.reward import RiskAwareRewardCalculator, RewardConfig

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"
    CASCADE = "CASCADE"


@dataclass
class EnvConfig:
    """Trading environment configuration."""
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5      # Max 50% of capital per position
    commission_rate: float = 0.001     # 0.1%
    slippage_bps: int = 5              # 0.05%
    reward_window: int = 20            # Window for volatility calculation
    max_steps: int = 1000              # Max steps per episode
    symbol: str = "BTC-USD"
    
    # === Rewards parameters (mapped to RewardConfig) ===
    lambda_vol: float = 0.5            
    lambda_dd: float = 2.0             
    lambda_cost: float = 0.1           
    
    # === Bounded delta parameters ===
    max_rl_delta: float = 0.30         # RL can only adjust ±30% from base
    
    # === Regime-based leverage caps ===
    leverage_caps: Dict[str, float] = None
    
    def __post_init__(self):
        if self.leverage_caps is None:
            self.leverage_caps = {
                "NORMAL": 2.0,
                "HIGH_VOL": 1.5,
                "CRISIS": 1.0,
                "CASCADE": 0.0  # RL DISABLED
            }


class TradingEnvironment:
    """
    Trading environment for RL training with proper risk management. 

    Versions:
    - v1.0: Baseline
    - v2.0: Risk-aware rewards, Bounded delta, Regime gating
    - v2.1: External numba-optimized reward calculator integration
    """

    def __init__(
        self, 
        config: Optional[EnvConfig] = None, 
        use_live_prices: bool = True,
        # reward_config is now optional override for the internal calculator config
        reward_config: Optional[RewardConfig] = None
    ):
        """
        Initialize trading environment.

        Args:
            config: Environment configuration
            use_live_prices: Use live price feed from Binance
            reward_config: Optional specific reward configuration
        """
        self.config = config or EnvConfig()
        self.use_live_prices = use_live_prices

        # Create reward config from EnvConfig if not provided
        if reward_config is None:
            self.reward_config = RewardConfig(
                lambda_vol=self.config.lambda_vol,
                lambda_dd=self.config.lambda_dd,
                lambda_cost=self.config.lambda_cost,
                # Use strict penalties from best practices
                wrong_action_multiplier=1.5
            )
        else:
            self.reward_config = reward_config

        # Initialize Advanced Reward Calculator
        self.reward_calculator = RiskAwareRewardCalculator(self.reward_config)

        # State encoder
        self.state_encoder = StateEncoder(
            max_position_usd=self.config.initial_capital * self.config.max_position_pct
        )

        # Price feed
        if use_live_prices:
            try:
                self.price_feed = get_price_feed()
            except Exception: 
                self.price_feed = None
                logger.warning("Could not initialize price feed, using fallback")
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
        self.peak_capital = self.config.initial_capital  # For drawdown calculation
        self.returns = []
        self.trades = []

        # Position tracking
        self.current_position_usd = 0.0
        self.current_position_qty = 0.0
        self.prev_position_usd = 0.0  # For transaction cost calculation
        self.entry_price = 0.0
        
        # Track price for step-wise reward
        self.last_price = 0.0
        self.prev_pos_norm = 0.0  # Normalized [-1, 1]

        # === Regime tracking ===
        self.current_regime = MarketRegime.NORMAL
        self.volatility_history = []

        logger.info(
            f"TradingEnvironment v2.1 initialized "
            f"(capital=${self.config.initial_capital:,.0f}, "
            f"max_delta=±{self.config.max_rl_delta:.0%}, "
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
        self.peak_capital = self.config.initial_capital
        self.returns = []
        self.trades = []
        self.current_position_usd = 0.0
        self.current_position_qty = 0.0
        self.prev_position_usd = 0.0
        self.entry_price = 0.0
        self.current_regime = MarketRegime.NORMAL
        self.volatility_history = []
        self.execution_engine.reset()
        
        # Reset reward calculator
        self.reward_calculator.reset(initial_capital=self.capital)
        
        # Initialize last price
        self.last_price = self._get_current_price()
        self.prev_pos_norm = 0.0

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

    def _detect_regime(self) -> MarketRegime:
        """
        Detect current market regime based on volatility. 
        
        Returns:
            MarketRegime enum
        """
        if len(self.volatility_history) < 5:
            return MarketRegime.NORMAL
        
        recent_vol = np.mean(self.volatility_history[-5:])
        
        # Detect regime based on volatility levels
        if recent_vol > 0.08:  # >8% volatility
            return MarketRegime.CASCADE
        elif recent_vol > 0.05:  # >5% volatility
            return MarketRegime.CRISIS
        elif recent_vol > 0.03:  # >3% volatility
            return MarketRegime.HIGH_VOL
        else: 
            return MarketRegime.NORMAL

    def _calculate_volatility_target_position(self) -> float:
        """
        Calculate base position size using volatility targeting.
        
        This is the deterministic core that RL adjusts from.
        Target:  15% annualized volatility
        
        Returns:
            Base position size in USD
        """
        if len(self.returns) < 20:
            # Not enough data, use conservative position
            return self.capital * 0.25
        
        # Calculate recent realized volatility
        # Using numpy std on returns
        recent_vol = np.std(self.returns[-20:]) * np.sqrt(252)  # Annualized
        
        if recent_vol < 0.01:
            recent_vol = 0.01  # Floor to prevent division issues
        
        # Target 15% annualized volatility
        target_vol = 0.15
        vol_scalar = target_vol / recent_vol
        
        # Base position = capital * vol_scalar, capped at max_position_pct
        base_position = self.capital * min(vol_scalar, self.config.max_position_pct)
        
        return base_position

    def _get_state(self) -> np.ndarray:
        """
        Get current state. 

        Returns:
            Encoded state vector
        """
        current_price = self._get_current_price()

        # Update price history
        self.state_encoder.update_price_history(current_price)

        # Calculate metrics using state encoder
        # Note: In production these should come from StateEncoder's internal logic
        # Here we rely on it having enough history
        momentum_1h = self.state_encoder.calculate_momentum(60)
        momentum_4h = self.state_encoder.calculate_momentum(240)
        volatility = self.state_encoder.calculate_volatility()
        
        # Track volatility for regime detection
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)

        # Update regime
        self.current_regime = self._detect_regime()

        # Calculate unrealized P&L
        unrealized_pnl_pct = 0.0
        if self.current_position_qty != 0 and self.entry_price > 0:
            if self.current_position_qty > 0:  # LONG
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:  # SHORT
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Recent performance
        recent_win_rate = 0.5
        if len(self.trades) > 0:
            recent_trades = self.trades[-10:]
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            recent_win_rate = wins / len(recent_trades)

        # Total P&L %
        total_pnl_pct = (self.capital - self.config.initial_capital) / self.config.initial_capital

        # Current drawdown
        current_drawdown = (self.capital - self.peak_capital) / self.peak_capital

        # Map regime to cascade risk for the state
        regime_risk_map = {
            MarketRegime.NORMAL: 0.1,
            MarketRegime.HIGH_VOL: 0.4,
            MarketRegime.CRISIS: 0.7,
            MarketRegime.CASCADE: 1.0
        }
        cascade_risk = regime_risk_map.get(self.current_regime, 0.2)

        # Create trading state
        trading_state = TradingState(
            signal_confidence=0.75,
            signal_action=2,  # BUY (simulated)
            signal_tier=1,
            signal_regime=self.current_regime.value,
            position_size_usd=abs(self.current_position_usd),
            position_side=int(np.sign(self.current_position_qty)),
            unrealized_pnl_pct=unrealized_pnl_pct,
            price_momentum_1h=momentum_1h,
            price_momentum_4h=momentum_4h,
            volatility=volatility,
            recent_win_rate=recent_win_rate,
            recent_sharpe=self._calculate_sharpe(),  # Still used for state feature
            total_pnl_pct=total_pnl_pct,
            cascade_risk=cascade_risk,
            current_drawdown=current_drawdown,
            timestamp=time.time()
        )

        return self.state_encoder.encode_state(trading_state)

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio of recent returns (for state only, NOT reward)."""
        if len(self.returns) < 2:
            return 0.0

        recent_returns = self.returns[-self.config.reward_window:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        if std_return < 1e-8:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(252)
        return np.clip(sharpe, -5.0, 5.0)  # Clip extreme values

    def _apply_bounded_delta(self, raw_action: float) -> float:
        """
        Apply bounded delta constraint to RL output.
        
        RL output is NEVER directly used.  Instead: 
        1. Clip raw action to [-1, 1]
        2. Scale to [-max_delta, +max_delta]
        3. Apply to volatility-targeting base position
        4. Enforce regime-conditional leverage caps
        
        Args:
            raw_action: Raw RL output
            
        Returns:
            Final position size in USD
        """
        # Step 1: Clip raw action
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        
        # Step 2: Scale to bounded delta
        rl_delta = clipped_action * self.config.max_rl_delta  # [-0.30, +0.30]
        
        # Step 3: Get base position from volatility targeting
        base_position = self._calculate_volatility_target_position()
        
        # Step 4: Apply delta adjustment
        adjusted_position = base_position * (1.0 + rl_delta)
        
        # Step 5: Regime gating - CRITICAL SAFETY FEATURE
        regime_name = self.current_regime.value
        leverage_cap = self.config.leverage_caps.get(regime_name, 1.0)
        
        if self.current_regime in [MarketRegime.CRISIS, MarketRegime.CASCADE]: 
            # DISABLE RL in crisis - use pure vol-targeting
            logger.debug(f"Regime={regime_name}:  RL DISABLED, using base position")
            return base_position * leverage_cap
        
        # Step 6: Apply leverage cap
        max_position = self.capital * leverage_cap
        final_position = min(adjusted_position, max_position)
        
        # Ensure non-negative
        final_position = max(0, final_position)
        
        return final_position

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]: 
        """
        Execute one step in the environment.

        Args:
            action: RL delta output [-1, 1] (will be bounded to ±30%)

        Returns:
            next_state: Next state
            reward: Risk-aware reward
            done: Whether episode is done
            info: Additional information
        """
        self.current_step += 1
        current_price = self._get_current_price()
        
        # Store previous position for transaction cost calculation
        self.prev_position_usd = abs(self.current_position_usd)

        # === Apply bounded delta with regime gating ===
        target_position_usd = self._apply_bounded_delta(action)

        # Execute trade if position change needed
        position_change = target_position_usd - abs(self.current_position_usd)
        
        # Execution Engine Handling
        if abs(position_change) > self.capital * 0.01:  # Threshold:  1% of capital
            # Close existing position if necessary (Simplified handling for demo)
            if self.current_position_qty != 0:
                self._close_position(current_price)

            # Open new position
            if target_position_usd > 0:
                self._open_position(target_position_usd, current_price)

        # Calculate STEP return (incremental) strictly for tracking
        # The reward calculator handles its own PnL calculation based on price delta
        step_return = 0.0
        if self.current_position_qty != 0 and self.last_price > 0:
            price_change_pct = (current_price - self.last_price) / self.last_price
            if self.current_position_qty > 0:  # LONG
                step_return = price_change_pct
            else:  # SHORT (if we supported shorting, but here qty is positive for 'LONG')
                # Assuming simple LONG-only for this logic unless _open_position supports sides
                step_return = price_change_pct # Assuming Long
                
            # If we were supporting SHORT, we'd need side tracking. 
            # Based on _open_position: 'side': 'BUY'. So Long only.

        self.returns.append(step_return)

        # Update peak capital for drawdown calculation
        self.peak_capital = max(self.peak_capital, self.capital)

        # === NEW: Use Advanced Reward Calculator ===
        # Calculate volatility for penalty (safe window)
        rolling_vol = 0.0
        if len(self.returns) >= 20:
             rolling_vol = np.std(self.returns[-20:])
        
        # Normalized position [-1, 1] for calculator
        # Assuming Long-Only: [0, 1] mapped relative to max leverage? 
        # Calculator logic treats abs(pos) as exposure.
        # We can use (position_usd / capital).
        current_pos_norm = target_position_usd / (self.capital + 1e-8)
        
        reward, components = self.reward_calculator.calculate_reward(
            previous_price=self.last_price,
            current_price=current_price,
            previous_position=self.prev_pos_norm,
            current_position=current_pos_norm,
            current_capital=self.capital,
            rolling_volatility=rolling_vol
        )

        # Update tracking for next step
        self.last_price = current_price
        self.prev_pos_norm = current_pos_norm

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
            'raw_action': action,
            'bounded_position':  target_position_usd,
            'regime': self.current_regime.value,
            'price': current_price,
            'reward': reward,
            'step_return': step_return,
            'sharpe': self._calculate_sharpe(),
            'total_pnl_pct':  (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'drawdown': (self.peak_capital - self.capital) / self.peak_capital,
            **components # Add breakdown
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
            'order_type':  'MARKET',
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
            'exit_price':  report.fill_price,
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
                'total_pnl':  0.0,
                'sharpe':  0.0,
                'max_drawdown': 0.0,
            }

        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Calculate max drawdown from returns
        cumulative = np.cumsum(self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return {
            'total_trades': len(self.trades),
            'win_rate': wins / len(self.trades),
            'total_pnl':  total_pnl,
            'total_pnl_pct':  (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'sharpe': self._calculate_sharpe(),
            'max_drawdown': max_dd,
            'current_capital': self.capital,
            'final_regime': self.current_regime.value,
        }


def test_trading_env():
    """Test trading environment v2.1."""
    print("=" * 80)
    print("HIMARI RL - Trading Environment v2.1 Test")
    print("=" * 80)
    print()

    # Create environment
    config = EnvConfig(initial_capital=100000, max_steps=100)
    env = TradingEnvironment(config, use_live_prices=False)

    print("Environment created:")
    print(f"  Initial capital:   ${config.initial_capital:,.0f}")
    print(f"  Max RL delta:     ±{config.max_rl_delta:.0%}")
    print()

    # Reset environment
    print("Test 1: Reset environment")
    print("-" * 60)
    state = env.reset()
    print(f"  State shape:   {state.shape}")
    print(f"  State values: {state[:5]}...  (showing first 5)")
    print()

    # Run steps with different action values
    print("Test 2: Run 10 steps testing reward components")
    print("-" * 60)

    test_actions = [0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 0.3, -0.3, 0.0]
    
    for step, action in enumerate(test_actions):
        next_state, reward, done, info = env.step(action)
        
        print(f"  Step {step+1}: acts={action:+.1f} → "
              f"reward={reward:+.4f} | "
              f"PnL={info.get('pnl', 0):.4f} | "
              f"Div={info.get('action_diversity', 0):.4f}")

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
