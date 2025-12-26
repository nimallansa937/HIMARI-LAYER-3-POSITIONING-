"""
HIMARI OPUS V2 - Enhanced Phase 1 Pipeline
===========================================

Production-grade Phase 1 with full integration:
- Sentiment-aware sizing
- Hot-reload configuration
- Prometheus metrics recording
- Complete diagnostics

Version: 3.1 Enhanced
"""

from typing import Optional, Dict
import time
import logging
import os

# Handle both module and script imports
try:
    from core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators
    )
    from core.layer3_config_manager import ConfigManager
    from core.layer3_metrics import (
        record_kelly_metrics, record_conformal_metrics,
        record_regime_metrics, record_cascade_metrics,
        record_position_decision, layer3_conformal_null_rejections,
        layer3_conformal_nan_rejections, layer3_regime_false_flips,
        layer3_regime_transitions
    )
    from engines.bayesian_kelly import BayesianKellyEngine
    from engines.conformal_scaler import ConformalPositionScaler
    from engines.regime_adjuster import RegimeConditionalAdjuster
    from engines.sentiment_sizer import SentimentAwareSizer
    from risk.cascade_detector_v2 import EnhancedCascadeDetector
except ImportError:
    from ..core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators
    )
    from ..core.layer3_config_manager import ConfigManager
    from ..core.layer3_metrics import (
        record_kelly_metrics, record_conformal_metrics,
        record_regime_metrics, record_cascade_metrics,
        record_position_decision, layer3_conformal_null_rejections,
        layer3_conformal_nan_rejections, layer3_regime_false_flips,
        layer3_regime_transitions
    )
    from ..engines.bayesian_kelly import BayesianKellyEngine
    from ..engines.conformal_scaler import ConformalPositionScaler
    from ..engines.regime_adjuster import RegimeConditionalAdjuster
    from ..engines.sentiment_sizer import SentimentAwareSizer
    from ..risk.cascade_detector_v2 import EnhancedCascadeDetector

logger = logging.getLogger(__name__)


class Layer3Phase1Enhanced:
    """
    Enhanced Phase 1 pipeline with full production features.
    
    Integrations:
    - Hot-reload configuration management
    - Sentiment-aware sizing (optional)
    - Prometheus metrics recording
    - Complete diagnostics pipeline
    
    Pipeline stages:
    1. Bayesian Kelly → Base position size
    2. Conformal scaling → Uncertainty adjustment
    3. Sentiment adjustment → L2 sentiment integration
    4. Regime adjustment → Market condition scaling
    5. Cascade detection → Risk-based reduction
    """
    
    def __init__(
        self,
        portfolio_value: float = 100000,
        kelly_fraction: float = 0.25,
        config_path: Optional[str] = None,
        enable_hot_reload: bool = True,
        enable_metrics: bool = True,
        enable_sentiment: bool = True
    ):
        """
        Initialize Enhanced Phase 1 pipeline.
        
        Args:
            portfolio_value: Total portfolio value in USD
            kelly_fraction: Kelly multiplier (0.25 = quarter Kelly)
            config_path: Path to configuration file
            enable_hot_reload: Enable configuration hot-reload
            enable_metrics: Enable Prometheus metrics recording
            enable_sentiment: Enable sentiment-aware sizing
        """
        self.portfolio_value = portfolio_value
        self.enable_metrics = enable_metrics
        self.enable_sentiment = enable_sentiment
        
        # Initialize config manager
        if config_path is None:
            # Try to find config file
            possible_paths = [
                "config/layer3_config.yaml",
                "../config/layer3_config.yaml",
                os.path.join(os.path.dirname(__file__), "../../config/layer3_config.yaml")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        self.config_manager = None
        if config_path and os.path.exists(config_path):
            try:
                self.config_manager = ConfigManager(config_path)
                if enable_hot_reload:
                    self.config_manager.register_callback(self._on_config_change)
                    self.config_manager.start_watcher()
                logger.info(f"Config manager initialized: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize config manager: {e}")
        
        # Load configuration
        self._load_from_config(kelly_fraction)
        
        logger.info(
            f"Layer3Phase1Enhanced initialized: portfolio=${portfolio_value:,.0f}, "
            f"kelly={kelly_fraction}, sentiment={'on' if enable_sentiment else 'off'}, "
            f"metrics={'on' if enable_metrics else 'off'}"
        )
    
    def _load_from_config(self, default_kelly: float):
        """Load component parameters from configuration."""
        if self.config_manager:
            kelly_fraction = self.config_manager.get(
                'position_sizing.bayesian_kelly.kelly_fraction', default_kelly
            )
            coverage = self.config_manager.get(
                'position_sizing.conformal_prediction.coverage', 0.90
            )
            hysteresis = self.config_manager.get(
                'position_sizing.regime_adjustment.hysteresis_periods', 3
            )
            sentiment_weight = self.config_manager.get(
                'sentiment.weight', 0.15
            )
            sentiment_enabled = self.config_manager.get(
                'sentiment.enabled', self.enable_sentiment
            )
        else:
            kelly_fraction = default_kelly
            coverage = 0.90
            hysteresis = 3
            sentiment_weight = 0.15
            sentiment_enabled = self.enable_sentiment
        
        # Initialize components
        self.bayesian_kelly = BayesianKellyEngine(
            portfolio_value=self.portfolio_value,
            kelly_fraction=kelly_fraction
        )
        self.conformal_scaler = ConformalPositionScaler(
            coverage=coverage,
            window_size=200
        )
        self.regime_adjuster = RegimeConditionalAdjuster(
            hysteresis_periods=hysteresis
        )
        self.sentiment_sizer = SentimentAwareSizer(
            sentiment_weight=sentiment_weight,
            enabled=sentiment_enabled
        )
        self.cascade_detector = EnhancedCascadeDetector()
    
    def _on_config_change(self, old_config: Dict, new_config: Dict):
        """Handle configuration change callback."""
        logger.info("Configuration change detected, reloading parameters...")
        
        # Update Kelly fraction
        new_kelly = new_config.get('position_sizing', {}).get('bayesian_kelly', {}).get('kelly_fraction')
        if new_kelly:
            self.bayesian_kelly.kelly_fraction = new_kelly
        
        # Update sentiment
        sentiment_config = new_config.get('sentiment', {})
        if 'enabled' in sentiment_config:
            self.sentiment_sizer.set_enabled(sentiment_config['enabled'])
        if 'weight' in sentiment_config:
            self.sentiment_sizer.update_weight(sentiment_config['weight'])
        
        logger.info("Configuration reloaded successfully")
    
    def calculate_position(
        self,
        signal: TacticalSignal,
        cascade_indicators: CascadeIndicators,
        current_price: float
    ) -> PositionSizingDecision:
        """
        Calculate position size through complete Enhanced Phase 1 pipeline.
        
        Args:
            signal: Tactical signal from Layer 2
            cascade_indicators: Cascade risk indicators (from L1/L2)
            current_price: Current market price
            
        Returns:
            PositionSizingDecision with complete diagnostics
        """
        timestamp_ns = time.time_ns()
        
        # Stage 1: Bayesian Kelly base position
        kelly_position_usd, kelly_diagnostics = self.bayesian_kelly.calculate_position_size(
            confidence=signal.confidence,
            expected_return=signal.expected_return or 0.05,
            predicted_volatility=signal.predicted_volatility or 0.02
        )
        
        if self.enable_metrics:
            record_kelly_metrics(
                kelly_diagnostics.get('applied_fraction', 0.25),
                kelly_position_usd
            )
        
        # Stage 2: Conformal scaling
        conformal_adjusted_usd, conformal_diagnostics = self.conformal_scaler.scale_position(
            base_position_usd=kelly_position_usd,
            predicted_return=signal.expected_return
        )
        
        if self.enable_metrics:
            record_conformal_metrics(
                conformal_diagnostics.get('scale_factor', 1.0),
                len(self.conformal_scaler.residuals)
            )
        
        # Stage 3: Sentiment adjustment (NEW)
        sentiment_adjusted_usd, sentiment_multiplier, sentiment_diagnostics = \
            self.sentiment_sizer.adjust_for_sentiment(
                base_position_usd=conformal_adjusted_usd,
                sentiment_score=signal.sentiment_score,
                sentiment_confidence=signal.sentiment_confidence
            )
        
        # Stage 4: Regime adjustment
        confirmed_regime, regime_diagnostics = self.regime_adjuster.update_regime(
            detected_regime=signal.regime,
            timestamp_ns=timestamp_ns
        )
        
        regime_adjusted_usd, regime_multiplier, regime_adj_diagnostics = \
            self.regime_adjuster.adjust_position_for_regime(
                base_position_size=sentiment_adjusted_usd,
                regime=confirmed_regime
            )
        
        regime_diagnostics.update(regime_adj_diagnostics)
        
        if self.enable_metrics:
            record_regime_metrics(confirmed_regime.value, regime_multiplier)
            if regime_diagnostics.get('transition_confirmed'):
                layer3_regime_transitions.labels(
                    from_regime=regime_diagnostics.get('from', 'unknown'),
                    to_regime=confirmed_regime.value
                ).inc()
        
        # Stage 5: Cascade risk detection
        cascade_risk_score, cascade_recommendation, cascade_diagnostics = \
            self.cascade_detector.calculate_cascade_risk(cascade_indicators)
        
        # Apply cascade recommendation
        if cascade_recommendation == "EXIT":
            cascade_adjusted_usd = 0.0
        elif cascade_recommendation == "REDUCE_75%":
            cascade_adjusted_usd = regime_adjusted_usd * 0.25
        elif cascade_recommendation == "REDUCE_50%":
            cascade_adjusted_usd = regime_adjusted_usd * 0.50
        else:
            cascade_adjusted_usd = regime_adjusted_usd
        
        if self.enable_metrics:
            record_cascade_metrics(
                cascade_risk_score,
                cascade_diagnostics.get('components', {}),
                cascade_recommendation
            )
        
        # Final position size
        final_position_usd = cascade_adjusted_usd
        
        if self.enable_metrics:
            record_position_decision(signal.symbol, signal.strategy_id, final_position_usd)
        
        # Comprehensive diagnostics
        diagnostics = {
            'kelly': kelly_diagnostics,
            'conformal': conformal_diagnostics,
            'sentiment': sentiment_diagnostics,
            'regime': regime_diagnostics,
            'cascade': cascade_diagnostics,
            'pipeline_stages': {
                'kelly_usd': kelly_position_usd,
                'conformal_usd': conformal_adjusted_usd,
                'sentiment_usd': sentiment_adjusted_usd,
                'regime_usd': regime_adjusted_usd,
                'cascade_usd': cascade_adjusted_usd,
                'final_usd': final_position_usd
            }
        }
        
        # Create decision
        decision = PositionSizingDecision(
            position_size_usd=final_position_usd,
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
            timestamp_ns=timestamp_ns,
            kelly_position_usd=kelly_position_usd,
            conformal_adjusted_usd=conformal_adjusted_usd,
            regime_adjusted_usd=regime_adjusted_usd,
            cascade_adjusted_usd=cascade_adjusted_usd,
            cascade_risk_score=cascade_risk_score,
            cascade_recommendation=cascade_recommendation,
            current_regime=confirmed_regime,
            diagnostics=diagnostics,
            sentiment_diagnostics=sentiment_diagnostics
        )
        
        logger.info(
            f"Position calculated: {signal.symbol} ${final_position_usd:,.2f} "
            f"(Kelly: ${kelly_position_usd:,.2f}, Sentiment: {sentiment_multiplier:.2f}x, "
            f"Cascade: {cascade_recommendation}, Regime: {confirmed_regime.value})"
        )
        
        return decision
    
    def update_from_trade_result(
        self,
        predicted_return: Optional[float],
        actual_return: Optional[float],
        trade_won: bool
    ):
        """Update components based on trade result."""
        if actual_return is not None:
            self.bayesian_kelly.update_posterior(trade_won, actual_return)
        
        self.conformal_scaler.update(predicted_return, actual_return)
        
        if self.enable_metrics:
            stats = self.conformal_scaler.get_statistics()
            layer3_conformal_null_rejections._value.set(stats['null_rejections'])
            layer3_conformal_nan_rejections._value.set(stats['nan_rejections'])
        
        logger.debug("Phase 1 Enhanced components updated from trade result")
    
    def get_state(self) -> Dict:
        """Get current state of all components."""
        state = {
            'portfolio_value': self.portfolio_value,
            'bayesian_kelly': self.bayesian_kelly.get_state(),
            'conformal_scaler': self.conformal_scaler.get_statistics(),
            'regime_adjuster': self.regime_adjuster.get_state(),
            'sentiment_sizer': self.sentiment_sizer.get_statistics()
        }
        
        if self.config_manager:
            state['config_manager'] = self.config_manager.get_state()
        
        return state
    
    def stop(self):
        """Stop background services."""
        if self.config_manager:
            self.config_manager.stop_watcher()
        logger.info("Layer3Phase1Enhanced stopped")
