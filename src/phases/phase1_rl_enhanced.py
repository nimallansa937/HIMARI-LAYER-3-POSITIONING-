"""
HIMARI OPUS V2 - Phase 1 with RL Enhancement
==============================================

Enhanced Phase 1 that uses RL agent to optimize position sizing.

Pipeline:
1. Bayesian Kelly → Base position
2. Conformal scaling → Uncertainty adjustment
3. Sentiment adjustment → L2 sentiment integration
4. Regime adjustment → Market condition scaling
5. Cascade detection → Risk-based reduction
6. **RL Multiplier** → Learned optimization (NEW)

Version: 4.0 - RL Enhanced
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional
import logging
import numpy as np

from phases.phase1_core import Layer3Phase1
from rl.ppo_agent import PPOAgent, PPOConfig
from rl.state_encoder import StateEncoder, TradingState
from core.layer3_types import TacticalSignal, PositionSizingDecision, CascadeIndicators

logger = logging.getLogger(__name__)


class Layer3Phase1RL(Layer3Phase1):
    """
    Phase 1 with RL enhancement.

    Extends base Phase 1 with learned position size multiplier.
    """

    def __init__(
        self,
        portfolio_value: float = 100000,
        kelly_fraction: float = 0.25,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = True,
        enable_metrics: bool = True,
        enable_sentiment: bool = True,
        enable_rl: bool = True,
        rl_model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize RL-enhanced Phase 1.

        Args:
            portfolio_value: Total portfolio value
            kelly_fraction: Kelly multiplier
            config_path: Config file path
            enable_hot_reload: Enable config hot-reload
            enable_metrics: Enable metrics
            enable_sentiment: Enable sentiment sizing
            enable_rl: Enable RL multiplier
            rl_model_path: Path to trained RL model (optional)
            device: 'cpu' or 'cuda'
        """
        # Initialize base Phase 1
        super().__init__(
            portfolio_value=portfolio_value,
            kelly_fraction=kelly_fraction,
            config_path=config_path,
            enable_hot_reload=enable_hot_reload,
            enable_metrics=enable_metrics,
            enable_sentiment=enable_sentiment
        )

        self.enable_rl = enable_rl
        self.device = device

        # Initialize RL components
        if self.enable_rl:
            self._init_rl_components(rl_model_path)
        else:
            self.rl_agent = None
            self.state_encoder = None

        logger.info(
            f"Phase 1 RL-Enhanced initialized: "
            f"rl={'on' if enable_rl else 'off'}, "
            f"model={'loaded' if rl_model_path else 'untrained'}"
        )

    def _init_rl_components(self, rl_model_path: Optional[str]):
        """Initialize RL agent and state encoder."""
        # State encoder
        self.state_encoder = StateEncoder(max_position_usd=self.portfolio_value * 0.5)

        # RL agent
        ppo_config = PPOConfig(state_dim=16, hidden_dim=128)
        self.rl_agent = PPOAgent(config=ppo_config, device=self.device)

        # Load trained model if provided
        if rl_model_path and os.path.exists(rl_model_path):
            self.rl_agent.load(rl_model_path)
            self.rl_agent.eval_mode()
            logger.info(f"RL model loaded from {rl_model_path}")
        else:
            logger.info("RL agent initialized without pretrained weights")

    def calculate_position(
        self,
        signal: TacticalSignal,
        cascade_indicators: CascadeIndicators,
        current_price: float,
        current_position_usd: float = 0.0,
        current_position_qty: float = 0.0,
        entry_price: float = 0.0,
        recent_trades: list = None
    ) -> PositionSizingDecision:
        """
        Calculate position with RL enhancement.

        Args:
            signal: Tactical signal from Layer 2
            cascade_indicators: Cascade risk indicators
            current_price: Current market price
            current_position_usd: Current position value
            current_position_qty: Current position quantity
            entry_price: Position entry price
            recent_trades: Recent trade history (for RL state)

        Returns:
            PositionSizingDecision with RL-adjusted size
        """
        # Get base decision from Phase 1
        base_decision = super().calculate_position(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=current_price
        )

        # If RL disabled, return base decision
        if not self.enable_rl or self.rl_agent is None:
            return base_decision

        # Build state for RL agent
        state = self._build_rl_state(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=current_price,
            current_position_usd=current_position_usd,
            current_position_qty=current_position_qty,
            entry_price=entry_price,
            recent_trades=recent_trades or []
        )

        # Get RL multiplier
        rl_multiplier, _ = self.rl_agent.get_action(state, deterministic=True)

        # Apply RL multiplier to cascade-adjusted position
        rl_adjusted_usd = base_decision.cascade_adjusted_usd * rl_multiplier

        # Clamp to reasonable bounds
        max_position = self.portfolio_value * 0.5  # Max 50%
        rl_adjusted_usd = min(rl_adjusted_usd, max_position)

        # Update decision
        base_decision.position_size_usd = rl_adjusted_usd

        # Add RL diagnostics
        if base_decision.diagnostics is None:
            base_decision.diagnostics = {}

        base_decision.diagnostics['rl'] = {
            'multiplier': rl_multiplier,
            'base_position_usd': base_decision.cascade_adjusted_usd,
            'rl_adjusted_usd': rl_adjusted_usd,
            'state_features': state.tolist()[:5],  # First 5 features for logging
        }

        logger.debug(
            f"RL adjustment: {base_decision.cascade_adjusted_usd:,.2f} × {rl_multiplier:.2f} "
            f"= ${rl_adjusted_usd:,.2f}"
        )

        return base_decision

    def _build_rl_state(
        self,
        signal: TacticalSignal,
        cascade_indicators: CascadeIndicators,
        current_price: float,
        current_position_usd: float,
        current_position_qty: float,
        entry_price: float,
        recent_trades: list
    ) -> np.ndarray:
        """
        Build state for RL agent.

        Args:
            signal: Tactical signal
            cascade_indicators: Cascade indicators
            current_price: Current price
            current_position_usd: Current position
            current_position_qty: Current quantity
            entry_price: Entry price
            recent_trades: Recent trade history

        Returns:
            Encoded state vector
        """
        # Update price history
        self.state_encoder.update_price_history(current_price)

        # Calculate metrics
        momentum_1h = self.state_encoder.calculate_momentum(60)
        momentum_4h = self.state_encoder.calculate_momentum(240)
        volatility = self.state_encoder.calculate_volatility()

        # Calculate unrealized P&L
        unrealized_pnl_pct = 0.0
        if current_position_qty != 0 and entry_price > 0:
            if current_position_qty > 0:  # LONG
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                unrealized_pnl_pct = (entry_price - current_price) / entry_price

        # Recent win rate
        recent_win_rate = 0.5
        if recent_trades:
            wins = sum(1 for t in recent_trades[-10:] if t.get('pnl', 0) > 0)
            recent_win_rate = wins / min(len(recent_trades), 10)

        # Encode action
        action_encoded = self.state_encoder.encode_action(signal.action)

        # Encode regime
        regime_encoded = self.state_encoder.encode_regime(signal.regime)

        # Create trading state
        trading_state = TradingState(
            signal_confidence=signal.confidence,
            signal_action=action_encoded,
            signal_tier=1,  # TODO: Extract from signal.risk_score
            signal_regime=regime_encoded,
            position_size_usd=abs(current_position_usd),
            position_side=int(np.sign(current_position_qty)),
            unrealized_pnl_pct=unrealized_pnl_pct,
            price_momentum_1h=momentum_1h,
            price_momentum_4h=momentum_4h,
            volatility=volatility,
            recent_win_rate=recent_win_rate,
            recent_sharpe=0.0,  # TODO: Calculate from recent trades
            total_pnl_pct=0.0,  # TODO: Track total P&L
            cascade_risk=cascade_indicators.onchain_whale_pressure,
            current_drawdown=0.0,  # TODO: Track drawdown
            timestamp=0.0
        )

        return self.state_encoder.encode_state(trading_state)

    def update_from_trade_result(
        self,
        predicted_return: Optional[float],
        actual_return: Optional[float],
        trade_won: bool
    ):
        """
        Update components from trade result.

        Overrides base method to also update RL agent if training.
        """
        # Update base components
        super().update_from_trade_result(predicted_return, actual_return, trade_won)

        # Note: RL agent updates happen during training, not here
        # This method is for online learning in production (future enhancement)


def test_rl_enhanced_phase1():
    """Test RL-enhanced Phase 1."""
    print("=" * 80)
    print("HIMARI Phase 1 RL-Enhanced Test")
    print("=" * 80)
    print()

    from core.layer3_types import TacticalAction, MarketRegime

    # Create phase 1 with RL (no pretrained model)
    phase1_rl = Layer3Phase1RL(
        portfolio_value=100000,
        kelly_fraction=0.25,
        enable_rl=True,
        rl_model_path=None,  # No pretrained model
        device='cpu'
    )

    print("Phase 1 RL-Enhanced created")
    print(f"  RL enabled: {phase1_rl.enable_rl}")
    print(f"  RL agent:   {'initialized' if phase1_rl.rl_agent else 'None'}")
    print()

    # Create test signal
    test_signal = TacticalSignal(
        strategy_id="test_strategy",
        symbol="BTC-USD",
        action=TacticalAction.STRONG_BUY,
        confidence=0.85,
        risk_score=0.2,
        regime=MarketRegime.TRENDING_UP,
        timestamp_ns=0,
        expected_return=0.10,
        predicted_volatility=0.04,
    )

    # Create cascade indicators
    cascade_indicators = CascadeIndicators(
        funding_rate=0.001,
        oi_change_pct=0.03,
        volume_ratio=2.0,
        onchain_whale_pressure=0.3,
        exchange_netflow_zscore=0.5,
    )

    # Test position calculation
    print("Test: Position calculation with RL")
    print("-" * 60)

    decision = phase1_rl.calculate_position(
        signal=test_signal,
        cascade_indicators=cascade_indicators,
        current_price=87000.0,
        current_position_usd=0.0,
        current_position_qty=0.0,
        entry_price=0.0,
        recent_trades=[]
    )

    print(f"  Kelly Position:     ${decision.kelly_position_usd:,.2f}")
    print(f"  Cascade Adjusted:   ${decision.cascade_adjusted_usd:,.2f}")
    print(f"  RL Adjusted:        ${decision.position_size_usd:,.2f}")
    print()

    if decision.diagnostics and 'rl' in decision.diagnostics:
        rl_diag = decision.diagnostics['rl']
        print(f"  RL Multiplier:      {rl_diag['multiplier']:.3f}")
        print(f"  RL Base:            ${rl_diag['base_position_usd']:,.2f}")
        print(f"  RL Adjusted:        ${rl_diag['rl_adjusted_usd']:,.2f}")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_rl_enhanced_phase1()
