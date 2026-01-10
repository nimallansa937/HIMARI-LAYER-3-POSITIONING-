"""
HIMARI Layer 3 - RL State Encoder
===================================

Encodes trading state into feature vector for RL agent.

State Components:
- L2 tactical signal (confidence, action, tier, regime)
- Current position (size, side, unrealized P&L)
- Market conditions (price momentum, volatility)
- Recent performance (win rate, Sharpe ratio)
- Risk metrics (cascade risk, drawdown)

Version: 1.0
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time


@dataclass
class TradingState:
    """Complete trading state for RL agent."""

    # L2 Signal
    signal_confidence: float        # [0, 1]
    signal_action: int              # 0=SELL, 1=HOLD, 2=BUY (encoded)
    signal_tier: int                # 1-4
    signal_regime: int              # Regime encoded (0-4)

    # Position
    position_size_usd: float        # Current position value
    position_side: int              # -1=SHORT, 0=FLAT, 1=LONG
    unrealized_pnl_pct: float       # Unrealized P&L as %

    # Market
    price_momentum_1h: float        # 1-hour price change %
    price_momentum_4h: float        # 4-hour price change %
    volatility: float               # Recent volatility

    # Performance
    recent_win_rate: float          # Last 10 trades win rate
    recent_sharpe: float            # Recent Sharpe ratio
    total_pnl_pct: float            # Total P&L %

    # Risk
    cascade_risk: float             # [0, 1]
    current_drawdown: float         # Current drawdown %

    # Meta
    timestamp: float


class StateEncoder:
    """
    Encodes trading state into normalized feature vector.

    Features (16 total):
    1. Signal confidence [0, 1]
    2-4. Signal action (one-hot: SELL, HOLD, BUY)
    5-8. Signal tier (one-hot: T1, T2, T3, T4)
    9. Position size (normalized)
    10. Position side [-1, 1]
    11. Unrealized P&L %
    12. Price momentum 1h
    13. Price momentum 4h
    14. Volatility
    15. Recent win rate
    16. Cascade risk
    """

    STATE_DIM = 16

    # Action encoding
    ACTION_MAP = {
        'STRONG_SELL': 0,
        'SELL': 0,
        'HOLD': 1,
        'BUY': 2,
        'STRONG_BUY': 2,
    }

    # Regime encoding
    REGIME_MAP = {
        'TRENDING_UP': 0,
        'TRENDING_DOWN': 1,
        'RANGING': 2,
        'HIGH_VOLATILITY': 3,
        'CRISIS': 4,
    }

    def __init__(self, max_position_usd: float = 100000.0):
        """
        Initialize state encoder.

        Args:
            max_position_usd: Maximum position size for normalization
        """
        self.max_position_usd = max_position_usd

        # Track history for momentum calculation
        self.price_history: List[float] = []
        self.max_history = 240  # 4 hours at 1min candles

    def encode_action(self, action) -> int:
        """
        Encode L2 action to integer.

        Args:
            action: TradeAction enum or string

        Returns:
            Encoded action: 0=SELL, 1=HOLD, 2=BUY
        """
        action_name = action.name if hasattr(action, 'name') else str(action).upper()
        return self.ACTION_MAP.get(action_name, 1)  # Default: HOLD

    def encode_regime(self, regime) -> int:
        """
        Encode regime to integer.

        Args:
            regime: MarketRegime enum or string

        Returns:
            Encoded regime: 0-4
        """
        regime_name = regime.value if hasattr(regime, 'value') else str(regime).upper()
        return self.REGIME_MAP.get(regime_name, 2)  # Default: RANGING

    def update_price_history(self, current_price: float):
        """Update price history for momentum calculation."""
        self.price_history.append(current_price)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)

    def calculate_momentum(self, periods: int) -> float:
        """
        Calculate price momentum over N periods.

        Args:
            periods: Number of periods to look back

        Returns:
            Momentum as percentage change
        """
        if len(self.price_history) < periods + 1:
            return 0.0

        old_price = self.price_history[-periods - 1]
        new_price = self.price_history[-1]

        if old_price == 0:
            return 0.0

        return ((new_price - old_price) / old_price) * 100.0

    def calculate_volatility(self) -> float:
        """
        Calculate recent volatility (standard deviation of returns).

        Returns:
            Volatility as decimal
        """
        if len(self.price_history) < 20:
            return 0.0

        recent_prices = self.price_history[-20:]
        returns = []

        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)

        return float(np.std(returns))

    def encode_state(self, state: TradingState) -> np.ndarray:
        """
        Encode TradingState to normalized feature vector.

        Args:
            state: TradingState object

        Returns:
            Numpy array of shape (STATE_DIM,)
        """
        features = []

        # 1. Signal confidence [0, 1] - already normalized
        features.append(state.signal_confidence)

        # 2-4. Signal action (one-hot)
        action_onehot = [0, 0, 0]
        action_onehot[state.signal_action] = 1
        features.extend(action_onehot)

        # 5-8. Signal tier (one-hot)
        tier_onehot = [0, 0, 0, 0]
        if 1 <= state.signal_tier <= 4:
            tier_onehot[state.signal_tier - 1] = 1
        features.extend(tier_onehot)

        # 9. Position size (normalized to [0, 1])
        norm_position = min(abs(state.position_size_usd) / self.max_position_usd, 1.0)
        features.append(norm_position)

        # 10. Position side [-1, 1]
        features.append(state.position_side)

        # 11. Unrealized P&L % (clipped to ±50%)
        norm_pnl = np.clip(state.unrealized_pnl_pct, -0.5, 0.5) * 2  # Scale to [-1, 1]
        features.append(norm_pnl)

        # 12. Price momentum 1h (clipped to ±10%)
        norm_mom_1h = np.clip(state.price_momentum_1h / 10.0, -1.0, 1.0)
        features.append(norm_mom_1h)

        # 13. Price momentum 4h (clipped to ±20%)
        norm_mom_4h = np.clip(state.price_momentum_4h / 20.0, -1.0, 1.0)
        features.append(norm_mom_4h)

        # 14. Volatility (clipped to 0.1 = 10%)
        norm_vol = np.clip(state.volatility / 0.1, 0.0, 1.0)
        features.append(norm_vol)

        # 15. Recent win rate [0, 1]
        features.append(state.recent_win_rate)

        # 16. Cascade risk [0, 1]
        features.append(state.cascade_risk)

        return np.array(features, dtype=np.float32)

    def decode_state(self, encoded: np.ndarray) -> Dict[str, Any]:
        """
        Decode feature vector back to readable format.

        Args:
            encoded: Encoded state vector

        Returns:
            Dictionary with decoded features
        """
        if len(encoded) != self.STATE_DIM:
            raise ValueError(f"Expected {self.STATE_DIM} features, got {len(encoded)}")

        # Extract action from one-hot
        action_idx = np.argmax(encoded[1:4])
        action_names = ['SELL', 'HOLD', 'BUY']

        # Extract tier from one-hot
        tier = np.argmax(encoded[4:8]) + 1

        return {
            'signal_confidence': encoded[0],
            'signal_action': action_names[action_idx],
            'signal_tier': tier,
            'position_size_normalized': encoded[8],
            'position_side': encoded[9],
            'unrealized_pnl': encoded[10] / 2,  # Descale
            'price_momentum_1h': encoded[11] * 10,  # Descale
            'price_momentum_4h': encoded[12] * 20,  # Descale
            'volatility': encoded[13] * 0.1,  # Descale
            'recent_win_rate': encoded[14],
            'cascade_risk': encoded[15],
        }

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.STATE_DIM


def test_state_encoder():
    """Test state encoder functionality."""
    print("=" * 80)
    print("HIMARI RL - State Encoder Test")
    print("=" * 80)
    print()

    # Create encoder
    encoder = StateEncoder(max_position_usd=100000)

    # Create test state
    test_state = TradingState(
        signal_confidence=0.85,
        signal_action=2,  # BUY
        signal_tier=1,
        signal_regime=0,  # TRENDING_UP
        position_size_usd=50000,
        position_side=1,  # LONG
        unrealized_pnl_pct=0.05,  # 5%
        price_momentum_1h=2.5,
        price_momentum_4h=8.0,
        volatility=0.03,
        recent_win_rate=0.65,
        recent_sharpe=1.8,
        total_pnl_pct=0.15,
        cascade_risk=0.2,
        current_drawdown=-0.03,
        timestamp=time.time()
    )

    print("Test State:")
    print(f"  Signal: BUY, Confidence=0.85, Tier=T1")
    print(f"  Position: $50,000 LONG, +5% P&L")
    print(f"  Momentum: 1h=+2.5%, 4h=+8.0%")
    print(f"  Volatility: 3%")
    print(f"  Win Rate: 65%")
    print(f"  Cascade Risk: 0.2")
    print()

    # Encode
    encoded = encoder.encode_state(test_state)

    print(f"Encoded State Vector (dim={len(encoded)}):")
    print(f"  {encoded}")
    print()

    # Decode
    decoded = encoder.decode_state(encoded)

    print("Decoded State:")
    for key, value in decoded.items():
        if isinstance(value, float):
            print(f"  {key:25s} {value:.3f}")
        else:
            print(f"  {key:25s} {value}")
    print()

    # Test action encoding
    print("Action Encoding Test:")
    test_actions = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
    for action in test_actions:
        encoded_action = encoder.encode_action(action)
        print(f"  {action:15s} -> {encoded_action}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_state_encoder()
