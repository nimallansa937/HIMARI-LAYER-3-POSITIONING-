"""
HIMARI OPUS V2 - Production Phase 1 Core Pipeline
===================================================

Production-ready Phase 1 with FULL integration:
- Bayesian Kelly position sizing
- Conformal prediction scaling
- Sentiment-aware adjustment (Section 11.3)
- Regime conditional adjustment
- Enhanced cascade detection
- Hot-reload configuration
- Prometheus metrics recording
- Comprehensive input validation
- Error handling with graceful degradation

Version: 3.1 Production
"""

from typing import Optional, Dict, Tuple
import time
import logging
import os
import traceback

# Handle both module and script imports
try:
    from core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        InvalidSignalException, InvalidCascadeIndicatorsException
    )
    from core.layer3_config_manager import ConfigManager
    from core import layer3_metrics as metrics
    from engines.bayesian_kelly import BayesianKellyEngine
    from engines.conformal_scaler import ConformalPositionScaler
    from engines.regime_adjuster import RegimeConditionalAdjuster
    from engines.sentiment_sizer import SentimentAwareSizer
    from risk.cascade_detector_v2 import EnhancedCascadeDetector
except ImportError:
    from ..core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        InvalidSignalException, InvalidCascadeIndicatorsException
    )
    from ..core.layer3_config_manager import ConfigManager
    from ..core import layer3_metrics as metrics
    from ..engines.bayesian_kelly import BayesianKellyEngine
    from ..engines.conformal_scaler import ConformalPositionScaler
    from ..engines.regime_adjuster import RegimeConditionalAdjuster
    from ..engines.sentiment_sizer import SentimentAwareSizer
    from ..risk.cascade_detector_v2 import EnhancedCascadeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Layer3Phase1:
    """
    Production-ready Phase 1 position sizing pipeline.
    
    Pipeline stages:
    1. Input validation
    2. Bayesian Kelly → Base position size
    3. Conformal scaling → Uncertainty adjustment
    4. Sentiment adjustment → L2 sentiment integration
    5. Regime adjustment → Market condition scaling
    6. Cascade detection → Risk-based reduction
    
    Features:
    - Hot-reload configuration
    - Prometheus metrics recording
    - Comprehensive error handling
    - Input validation
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
        Initialize Phase 1 pipeline.
        
        Args:
            portfolio_value: Total portfolio value in USD
            kelly_fraction: Kelly multiplier (0.25 = quarter Kelly)
            config_path: Path to configuration file (optional)
            enable_hot_reload: Enable configuration hot-reload
            enable_metrics: Enable Prometheus metrics recording
            enable_sentiment: Enable sentiment-aware sizing
        """
        # Validate portfolio value
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be > 0, got {portfolio_value}")

        self.portfolio_value = portfolio_value
        self.enable_metrics = enable_metrics
        self.enable_sentiment = enable_sentiment
        self.enable_hot_reload = enable_hot_reload
        
        # Initialize config manager (hot-reload)
        self.config_manager: Optional[ConfigManager] = None
        self._init_config_manager(config_path, kelly_fraction)
        
        logger.info(
            f"Phase 1 initialized: portfolio=${portfolio_value:,.0f}, "
            f"kelly={kelly_fraction}, sentiment={'on' if enable_sentiment else 'off'}, "
            f"metrics={'on' if enable_metrics else 'off'}, "
            f"hot_reload={'on' if enable_hot_reload else 'off'}"
        )
    
    def _init_config_manager(self, config_path: Optional[str], default_kelly: float):
        """Initialize configuration manager with hot-reload."""
        # Find config file
        if config_path is None:
            possible_paths = [
                "config/layer3_config.yaml",
                "../config/layer3_config.yaml",
                os.path.join(os.path.dirname(__file__), "../../config/layer3_config.yaml")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        # Load config if available
        if config_path and os.path.exists(config_path):
            try:
                self.config_manager = ConfigManager(config_path)
                
                # Register callback for hot-reload
                if self.enable_hot_reload:
                    self.config_manager.register_callback(self._on_config_change)
                    self.config_manager.start_watcher()
                    logger.info(f"Hot-reload config enabled: {config_path}")

                if self.enable_metrics:
                    metrics.layer3_config_reloads.labels(status='success').inc()
                    metrics.layer3_config_callbacks.set(len(self.config_manager._callbacks))
            except Exception as e:
                logger.warning(f"Failed to init config manager: {e}")
                if self.enable_metrics:
                    metrics.layer3_config_reloads.labels(status='error').inc()
        
        # Initialize components from config or defaults
        self._init_components(default_kelly)
    
    def _init_components(self, default_kelly: float):
        """Initialize all pipeline components."""
        # Get config values or use defaults
        if self.config_manager:
            kelly = self.config_manager.get('position_sizing.bayesian_kelly.kelly_fraction', default_kelly)
            coverage = self.config_manager.get('position_sizing.conformal_prediction.coverage', 0.90)
            hysteresis = self.config_manager.get('position_sizing.regime_adjustment.hysteresis_periods', 3)
            sentiment_weight = self.config_manager.get('sentiment.weight', 0.15)
            sentiment_enabled = self.config_manager.get('sentiment.enabled', self.enable_sentiment)
        else:
            kelly = default_kelly
            coverage = 0.90
            hysteresis = 3
            sentiment_weight = 0.15
            sentiment_enabled = self.enable_sentiment
        
        # Initialize engines
        self.bayesian_kelly = BayesianKellyEngine(
            portfolio_value=self.portfolio_value,
            kelly_fraction=kelly
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
        """Callback for configuration changes (hot-reload)."""
        logger.info("Configuration change detected, applying updates...")
        
        try:
            # Update Kelly fraction
            new_kelly = new_config.get('position_sizing', {}).get('bayesian_kelly', {}).get('kelly_fraction')
            if new_kelly and 0 < new_kelly <= 0.5:
                self.bayesian_kelly.kelly_fraction = new_kelly
                logger.info(f"Kelly fraction updated to {new_kelly}")
            
            # Update sentiment
            sentiment_config = new_config.get('sentiment', {})
            if 'enabled' in sentiment_config:
                self.sentiment_sizer.set_enabled(sentiment_config['enabled'])
            if 'weight' in sentiment_config:
                self.sentiment_sizer.update_weight(sentiment_config['weight'])
            
            # Update hysteresis
            new_hysteresis = new_config.get('position_sizing', {}).get('regime_adjustment', {}).get('hysteresis_periods')
            if new_hysteresis:
                self.regime_adjuster.hysteresis_periods = new_hysteresis
            
            if self.enable_metrics:
                metrics.layer3_config_reloads.labels(status='success').inc()
            
            logger.info("Configuration reload complete")
        except Exception as e:
            logger.error(f"Config reload error: {e}")
            if self.enable_metrics:
                metrics.layer3_config_reloads.labels(status='error').inc()
    
    def _validate_signal(self, signal: TacticalSignal) -> Tuple[bool, str]:
        """
        Validate tactical signal before processing.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if signal is None:
            return False, "Signal is None"
        
        if not signal.symbol:
            return False, "Signal missing symbol"
        
        if signal.confidence is None:
            return False, "Signal missing confidence"
        
        if not 0 <= signal.confidence <= 1:
            return False, f"Invalid confidence: {signal.confidence}"
        
        if signal.risk_score is not None and not 0 <= signal.risk_score <= 1:
            return False, f"Invalid risk_score: {signal.risk_score}"
        
        return True, ""
    
    def _validate_cascade_indicators(self, indicators: CascadeIndicators) -> Tuple[bool, str]:
        """Validate cascade indicators."""
        if indicators is None:
            return False, "Cascade indicators is None"

        # Validate field ranges
        if not -1.0 <= indicators.funding_rate <= 1.0:
            return False, f"Invalid funding_rate: {indicators.funding_rate}"

        if not -1.0 <= indicators.oi_change_pct <= 1.0:
            return False, f"Invalid oi_change_pct: {indicators.oi_change_pct}"

        if indicators.volume_ratio < 0:
            return False, f"Invalid volume_ratio: {indicators.volume_ratio}"

        if not 0 <= indicators.onchain_whale_pressure <= 1.0:
            return False, f"Invalid onchain_whale_pressure: {indicators.onchain_whale_pressure}"

        if not -10.0 <= indicators.exchange_netflow_zscore <= 10.0:
            return False, f"Invalid exchange_netflow_zscore: {indicators.exchange_netflow_zscore}"

        return True, ""
    
    def calculate_position(
        self,
        signal: TacticalSignal,
        cascade_indicators: CascadeIndicators,
        current_price: float
    ) -> PositionSizingDecision:
        """
        Calculate position size through complete Phase 1 pipeline.
        
        Args:
            signal: Tactical signal from Layer 2
            cascade_indicators: Cascade risk indicators (from L1/L2)
            current_price: Current market price
            
        Returns:
            PositionSizingDecision with complete diagnostics
            
        Raises:
            InvalidSignalException: If signal validation fails
        """
        timestamp_ns = time.time_ns()
        
        # Stage 0: Input validation
        valid, error = self._validate_signal(signal)
        if not valid:
            logger.error(f"Signal validation failed: {error}")
            raise InvalidSignalException(error)
        
        valid, error = self._validate_cascade_indicators(cascade_indicators)
        if not valid:
            logger.error(f"Cascade validation failed: {error}")
            raise InvalidCascadeIndicatorsException(error)

        # Validate current_price
        if current_price is None or current_price <= 0:
            error = f"Invalid current_price: {current_price}"
            logger.error(error)
            raise InvalidSignalException(error)
        
        try:
            # Stage 1: Bayesian Kelly base position
            kelly_position_usd, kelly_diagnostics = self.bayesian_kelly.calculate_position_size(
                confidence=signal.confidence,
                expected_return=signal.expected_return or 0.05,
                predicted_volatility=signal.predicted_volatility or 0.02
            )
            
            if self.enable_metrics:
                metrics.layer3_kelly_fraction.observe(kelly_diagnostics.get('applied_fraction', 0.25))
                metrics.layer3_kelly_position_size.observe(kelly_position_usd)
            
            # Stage 2: Conformal scaling
            conformal_adjusted_usd, conformal_diagnostics = self.conformal_scaler.scale_position(
                base_position_usd=kelly_position_usd,
                predicted_return=signal.expected_return
            )
            
            if self.enable_metrics:
                metrics.layer3_conformal_scale_factor.observe(conformal_diagnostics.get('scale_factor', 1.0))
                metrics.layer3_conformal_samples.set(len(self.conformal_scaler.residuals))
            
            # Stage 3: Sentiment adjustment (Section 11.3)
            sentiment_adjusted_usd, sentiment_multiplier, sentiment_diagnostics = \
                self.sentiment_sizer.adjust_for_sentiment(
                    base_position_usd=conformal_adjusted_usd,
                    sentiment_score=signal.sentiment_score,
                    sentiment_confidence=signal.sentiment_confidence
                )

            # Record sentiment metrics
            if self.enable_metrics and sentiment_multiplier != 1.0:
                if sentiment_multiplier > 1.0:
                    metrics.layer3_position_reductions.labels(reason='sentiment_boost').inc()
                else:
                    metrics.layer3_position_reductions.labels(reason='sentiment_reduction').inc()
            
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
                metrics.record_regime_metrics(confirmed_regime.value, regime_multiplier)
                if regime_diagnostics.get('transition_confirmed'):
                    metrics.layer3_regime_transitions.labels(
                        from_regime=regime_diagnostics.get('from', 'unknown'),
                        to_regime=confirmed_regime.value
                    ).inc()
                if self.regime_adjuster.false_flip_count > 0:
                    metrics.layer3_regime_false_flips.inc()
            
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
                metrics.record_cascade_metrics(
                    cascade_risk_score,
                    cascade_diagnostics.get('components', {}),
                    cascade_recommendation
                )
            
            # Final position size
            final_position_usd = cascade_adjusted_usd
            
            if self.enable_metrics:
                metrics.record_position_decision(signal.symbol, signal.strategy_id, final_position_usd)
                metrics.layer3_final_position_size.observe(final_position_usd)
            
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
                f"Position: {signal.symbol} ${final_position_usd:,.2f} "
                f"(Kelly: ${kelly_position_usd:,.2f}, Sentiment: {sentiment_multiplier:.2f}x, "
                f"Cascade: {cascade_recommendation}, Regime: {confirmed_regime.value})"
            )
            
            return decision
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")

            # Record error metrics
            if self.enable_metrics:
                metrics.layer3_position_reductions.labels(reason='pipeline_error').inc()

            # Return safe zero position on error
            return PositionSizingDecision(
                position_size_usd=0.0,
                symbol=signal.symbol if signal else "UNKNOWN",
                strategy_id=signal.strategy_id if signal else "UNKNOWN",
                timestamp_ns=timestamp_ns,
                kelly_position_usd=0.0,
                conformal_adjusted_usd=0.0,
                regime_adjusted_usd=0.0,
                cascade_adjusted_usd=0.0,
                cascade_risk_score=1.0,
                cascade_recommendation="EXIT",
                current_regime=MarketRegime.CRISIS,
                diagnostics={'error': str(e)}
            )
    
    def update_from_trade_result(
        self,
        predicted_return: Optional[float],
        actual_return: Optional[float],
        trade_won: bool
    ):
        """Update components based on trade result."""
        try:
            if actual_return is not None:
                self.bayesian_kelly.update_posterior(trade_won, actual_return)
                if self.enable_metrics:
                    metrics.layer3_bayesian_trades.labels(
                        outcome='win' if trade_won else 'loss'
                    ).inc()
            
            self.conformal_scaler.update(predicted_return, actual_return)
            
            if self.enable_metrics:
                stats = self.conformal_scaler.get_statistics()
                metrics.layer3_bayesian_win_rate.set(
                    self.bayesian_kelly.get_state()['posterior_win_rate']
                )
                metrics.layer3_bayesian_edge.set(
                    self.bayesian_kelly.get_state()['posterior_edge']
                )
            
            logger.debug("Components updated from trade result")
        except Exception as e:
            logger.error(f"Error updating from trade result: {e}")
    
    def get_state(self) -> Dict:
        """Get current state of all components."""
        state = {
            'portfolio_value': self.portfolio_value,
            'enable_metrics': self.enable_metrics,
            'enable_sentiment': self.enable_sentiment,
            'enable_hot_reload': self.enable_hot_reload,
            'bayesian_kelly': self.bayesian_kelly.get_state(),
            'conformal_scaler': self.conformal_scaler.get_statistics(),
            'regime_adjuster': self.regime_adjuster.get_state(),
            'sentiment_sizer': self.sentiment_sizer.get_statistics()
        }
        
        if self.config_manager:
            state['config_manager'] = self.config_manager.get_state()
        
        return state
    
    def stop(self):
        """Stop background services (hot-reload watcher)."""
        if self.config_manager:
            self.config_manager.stop_watcher()
        logger.info("Phase 1 stopped")
