"""
HIMARI Layer 3 Production Runner with GCP Integration
"""

import os
import yaml
import logging
from datetime import datetime

from src.phases.phase1_rl_enhanced import Layer3Phase1RL
from src.core.layer3_types import TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/gcp_deployment.yaml"):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main production runner."""

    logger.info("=" * 80)
    logger.info("HIMARI Layer 3 - GCP Production Deployment")
    logger.info("=" * 80)
    logger.info("")

    # Load configuration
    config = load_config()

    rl_config = config['layer3_rl']
    portfolio_config = config['portfolio']

    # Initialize Layer 3 with GCP API
    logger.info("Initializing Layer 3 Phase 1 RL...")

    if rl_config['deployment_mode'] == 'cloud':
        # Cloud deployment
        phase1_rl = Layer3Phase1RL(
            portfolio_value=portfolio_config['initial_value'],
            kelly_fraction=portfolio_config['kelly_fraction'],
            enable_rl=rl_config['enable_rl'],
            rl_api_endpoint=rl_config['rl_api_endpoint'],
            rl_timeout_ms=rl_config['rl_timeout_ms'],
        )
        logger.info(f"  Mode: Cloud API")
        logger.info(f"  Endpoint: {rl_config['rl_api_endpoint']}")

    else:
        # Local deployment
        phase1_rl = Layer3Phase1RL(
            portfolio_value=portfolio_config['initial_value'],
            kelly_fraction=portfolio_config['kelly_fraction'],
            enable_rl=rl_config['enable_rl'],
            rl_model_path=rl_config['rl_model_path'],
        )
        logger.info(f"  Mode: Local Model")
        logger.info(f"  Model: {rl_config['rl_model_path']}")

    logger.info("")

    # Example: Process tactical signal
    logger.info("Example: Processing tactical signal from Layer 2")
    logger.info("-" * 60)

    # Simulate Layer 2 signal
    test_signal = TacticalSignal(
        strategy_id="momentum_strategy_v1",
        symbol="BTC-USD",
        action=TacticalAction.STRONG_BUY,
        confidence=0.85,
        risk_score=0.2,
        regime=MarketRegime.TRENDING_UP,
        timestamp_ns=int(datetime.now().timestamp() * 1e9),
        expected_return=0.08,
        predicted_volatility=0.04,
    )

    # Simulate cascade indicators
    cascade_indicators = CascadeIndicators(
        funding_rate=0.0015,
        oi_change_pct=0.05,
        volume_ratio=2.5,
        onchain_whale_pressure=0.3,
        exchange_netflow_zscore=0.5,
    )

    # Calculate position
    logger.info(f"Signal: {test_signal.action.value} @ confidence={test_signal.confidence:.2f}")
    logger.info(f"Regime: {test_signal.regime.value}")
    logger.info("")

    decision = phase1_rl.calculate_position(
        signal=test_signal,
        cascade_indicators=cascade_indicators,
        current_price=87000.0,
        current_position_usd=0.0,
        recent_trades=[]
    )

    # Display results
    logger.info("Position Sizing Decision:")
    logger.info(f"  Kelly Position:     ${decision.kelly_position_usd:,.2f}")
    logger.info(f"  Cascade Adjusted:   ${decision.cascade_adjusted_usd:,.2f}")
    logger.info(f"  Final Position:     ${decision.position_size_usd:,.2f}")
    logger.info(f"  Position (BTC):     {decision.position_size_usd / 87000:.6f} BTC")
    logger.info("")

    if decision.diagnostics and 'rl' in decision.diagnostics:
        rl_diag = decision.diagnostics['rl']
        logger.info("RL Diagnostics:")
        logger.info(f"  Multiplier:         {rl_diag['multiplier']:.3f}")
        logger.info(f"  Source:             {rl_diag['source']}")
        logger.info(f"  Base Position:      ${rl_diag['base_position_usd']:,.2f}")
        logger.info(f"  RL Adjusted:        ${rl_diag['rl_adjusted_usd']:,.2f}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("READY FOR PRODUCTION")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
