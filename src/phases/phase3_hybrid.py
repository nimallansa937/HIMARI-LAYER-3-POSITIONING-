"""
HIMARI OPUS V2 - Phase 3 Hybrid Orchestrator
=============================================

Integrates Phase 1 + Phase 2 + Optional RL optimization with
automatic fallback on RL failure.

Features:
- Full Phase 2 portfolio management
- Optional Transformer-RL enhancement
- Circuit breaker protection
- Automatic fallback switching
- Performance comparison metrics

Version: 3.1 Phase 3
"""

from typing import Dict, List, Tuple, Optional
import time
import logging
import asyncio

try:
    from core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        EnsembleAllocation, CircuitBreakerOpenException
    )
    from core import layer3_metrics as metrics
    from phases.phase2_portfolio import Layer3Phase2Portfolio
    from optimization.transformer_rl import (
        TransformerRLClient, MockTransformerRLClient, RLPrediction
    )
    from risk.circuit_breaker import ColabProCircuitBreaker
except ImportError:
    from ..core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        EnsembleAllocation, CircuitBreakerOpenException
    )
    from ..core import layer3_metrics as metrics
    from ..phases.phase2_portfolio import Layer3Phase2Portfolio
    from ..optimization.transformer_rl import (
        TransformerRLClient, MockTransformerRLClient, RLPrediction
    )
    from ..risk.circuit_breaker import ColabProCircuitBreaker

logger = logging.getLogger(__name__)


class Layer3Phase3Hybrid:
    """
    Phase 3 Hybrid Orchestrator.
    
    Combines:
    - Phase 2: Multi-asset portfolio management
    - Phase 3: Optional Transformer-RL optimization
    
    Provides automatic fallback when RL is unavailable:
    1. Try RL prediction if enabled
    2. If RL fails/unavailable → fallback to Phase 2
    3. Track performance comparison
    """
    
    def __init__(
        self,
        portfolio_value: float = 100000,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.15,
        enable_rl: bool = False,
        rl_endpoint: str = "http://localhost:8888/predict",
        rl_timeout_sec: float = 5.0,
        use_mock_rl: bool = False,
        enable_metrics: bool = True
    ):
        """
        Initialize Phase 3 hybrid orchestrator.
        
        Args:
            portfolio_value: Total portfolio value in USD
            kelly_fraction: Kelly multiplier
            max_position_pct: Max single position as % of portfolio
            enable_rl: Enable Transformer-RL predictions
            rl_endpoint: RL prediction endpoint URL
            rl_timeout_sec: RL request timeout
            use_mock_rl: Use mock RL for testing
            enable_metrics: Enable Prometheus metrics
        """
        self.portfolio_value = portfolio_value
        self.enable_rl = enable_rl
        self.enable_metrics = enable_metrics
        
        # Initialize Phase 2 (always available as fallback)
        self.phase2 = Layer3Phase2Portfolio(
            portfolio_value=portfolio_value,
            kelly_fraction=kelly_fraction,
            max_position_pct=max_position_pct,
            enable_metrics=enable_metrics
        )
        
        # Initialize RL client (optional)
        self.rl_client: Optional[TransformerRLClient] = None
        if enable_rl:
            if use_mock_rl:
                self.rl_client = MockTransformerRLClient(success_rate=0.9)
                logger.info("Using Mock Transformer-RL client for testing")
            else:
                self.rl_client = TransformerRLClient(
                    endpoint_url=rl_endpoint,
                    timeout_sec=rl_timeout_sec,
                    enable_metrics=enable_metrics
                )
        
        # Statistics
        self.total_decisions = 0
        self.rl_decisions = 0
        self.fallback_decisions = 0
        
        # Performance tracking
        self.rl_returns: List[float] = []
        self.baseline_returns: List[float] = []
        
        logger.info(
            f"Layer3Phase3Hybrid initialized: portfolio=${portfolio_value:,.0f}, "
            f"RL={'enabled' if enable_rl else 'disabled'}"
        )
    
    def process_signals(
        self,
        signals: List[TacticalSignal],
        cascade_indicators: CascadeIndicators,
        current_prices: Dict[str, float],
        market_data: Optional[Dict] = None
    ) -> EnsembleAllocation:
        """
        Process signals through Phase 3 hybrid pipeline.
        
        Args:
            signals: List of tactical signals
            cascade_indicators: Cascade risk indicators
            current_prices: Current prices by symbol
            market_data: Optional additional market context for RL
            
        Returns:
            EnsembleAllocation with final portfolio decision
        """
        self.total_decisions += 1
        start_time = time.time()
        
        # Validate inputs (Issue #13)
        if not signals:
            logger.warning("Empty signals list in Phase 3")
            self.fallback_decisions += 1
            self._record_metrics(used_rl=False, latency_ms=0)
            return self._create_empty_allocation()
        
        # Try RL-enhanced prediction if enabled
        if self.enable_rl and self.rl_client is not None:
            try:
                allocation = self._process_with_rl(
                    signals=signals,
                    cascade_indicators=cascade_indicators,
                    current_prices=current_prices,
                    market_data=market_data
                )
                
                if allocation is not None:
                    self.rl_decisions += 1
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_metrics(used_rl=True, latency_ms=latency_ms)
                    return allocation
                    
            except CircuitBreakerOpenException as e:
                logger.warning(f"Circuit breaker open: {e}")
                # Record circuit breaker fallback
                if self.rl_client and hasattr(self.rl_client, 'circuit_breaker'):
                    self.rl_client.circuit_breaker.record_fallback()
                    
            except Exception as e:
                logger.error(f"RL processing failed: {e}")
        
        # Fallback to Phase 2
        self.fallback_decisions += 1
        latency_ms = (time.time() - start_time) * 1000
        self._record_metrics(used_rl=False, latency_ms=latency_ms)
        
        return self.phase2.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade_indicators,
            current_prices=current_prices
        )
    
    def _create_empty_allocation(self) -> EnsembleAllocation:
        """Create empty allocation for invalid inputs."""
        return EnsembleAllocation(
            allocations={},
            total_allocated_usd=0.0,
            portfolio_value=self.portfolio_value,
            utilization_pct=0.0,
            strategy_weights={},
            drifted_strategies=[],
            max_drift_pct=0.0,
            correlation_penalties={},
            timestamp_ns=time.time_ns()
        )
    
    def _record_metrics(self, used_rl: bool, latency_ms: float):
        """Record Phase 3 metrics to Prometheus."""
        if not self.enable_metrics:
            return
        
        try:
            # Record decision counts
            if used_rl:
                metrics.layer3_position_decisions_total.labels(
                    strategy_id='rl', symbol='aggregate'
                ).inc()
            else:
                metrics.layer3_position_decisions_total.labels(
                    strategy_id='fallback', symbol='aggregate'
                ).inc()
            
            # Record RL usage rate as gauge
            if self.total_decisions > 0:
                rl_rate = self.rl_decisions / self.total_decisions
                # Use existing metric or log
                logger.debug(f"RL usage rate: {rl_rate:.1%}")
                
        except Exception as e:
            logger.debug(f"Phase 3 metrics skipped: {e}")
    
    def _process_with_rl(
        self,
        signals: List[TacticalSignal],
        cascade_indicators: CascadeIndicators,
        current_prices: Dict[str, float],
        market_data: Optional[Dict]
    ) -> Optional[EnsembleAllocation]:
        """
        Process with RL enhancement.
        
        Returns None if RL unavailable (triggers fallback).
        """
        # First get Phase 2 baseline
        baseline = self.phase2.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade_indicators,
            current_prices=current_prices
        )
        
        # Get RL predictions for each signal
        rl_adjustments = {}
        
        for signal in signals:
            # Synchronous call for simplicity
            prediction = self.rl_client.predict_sync(signal, market_data)
            
            if prediction is not None:
                # RL suggests adjustment
                rl_adjustments[signal.symbol] = {
                    'position_pct': prediction.optimal_position_pct,
                    'confidence': prediction.confidence,
                    'latency_ms': prediction.latency_ms
                }
        
        # If RL failed for all signals, return None to trigger fallback
        if not rl_adjustments:
            return None
        
        # Apply RL adjustments to baseline allocations
        adjusted_allocations = {}
        
        for symbol, base_usd in baseline.allocations.items():
            if symbol in rl_adjustments:
                rl_info = rl_adjustments[symbol]
                
                # Blend RL suggestion with baseline
                # RL suggests position as % of portfolio
                rl_usd = rl_info['position_pct'] * self.portfolio_value
                
                # Weight by RL confidence
                rl_weight = rl_info['confidence']
                blended_usd = (
                    rl_weight * rl_usd +
                    (1 - rl_weight) * base_usd
                )
                
                adjusted_allocations[symbol] = blended_usd
            else:
                # Keep baseline for symbols without RL prediction
                adjusted_allocations[symbol] = base_usd
        
        # Create adjusted allocation
        total_allocated = sum(adjusted_allocations.values())
        
        return EnsembleAllocation(
            allocations=adjusted_allocations,
            total_allocated_usd=total_allocated,
            portfolio_value=self.portfolio_value,
            utilization_pct=total_allocated / self.portfolio_value * 100,
            strategy_weights=baseline.strategy_weights,
            drifted_strategies=baseline.drifted_strategies,
            max_drift_pct=baseline.max_drift_pct,
            correlation_penalties=baseline.correlation_penalties,
            timestamp_ns=time.time_ns()
        )
    
    async def process_signals_async(
        self,
        signals: List[TacticalSignal],
        cascade_indicators: CascadeIndicators,
        current_prices: Dict[str, float],
        market_data: Optional[Dict] = None
    ) -> EnsembleAllocation:
        """
        Async version of process_signals for better RL performance.
        """
        self.total_decisions += 1
        
        # Try RL-enhanced prediction if enabled
        if self.enable_rl and self.rl_client is not None:
            try:
                allocation = await self._process_with_rl_async(
                    signals=signals,
                    cascade_indicators=cascade_indicators,
                    current_prices=current_prices,
                    market_data=market_data
                )
                
                if allocation is not None:
                    self.rl_decisions += 1
                    return allocation
                    
            except Exception as e:
                logger.error(f"RL async processing failed: {e}")
        
        # Fallback to Phase 2
        self.fallback_decisions += 1
        return self.phase2.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade_indicators,
            current_prices=current_prices
        )
    
    async def _process_with_rl_async(
        self,
        signals: List[TacticalSignal],
        cascade_indicators: CascadeIndicators,
        current_prices: Dict[str, float],
        market_data: Optional[Dict]
    ) -> Optional[EnsembleAllocation]:
        """Async RL processing."""
        # Get baseline
        baseline = self.phase2.process_multi_strategy_signals(
            signals=signals,
            cascade_indicators=cascade_indicators,
            current_prices=current_prices
        )
        
        # Parallel RL predictions
        tasks = [
            self.rl_client.predict(signal, market_data)
            for signal in signals
        ]
        
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        rl_adjustments = {}
        for signal, pred in zip(signals, predictions):
            if isinstance(pred, RLPrediction):
                rl_adjustments[signal.symbol] = {
                    'position_pct': pred.optimal_position_pct,
                    'confidence': pred.confidence
                }
        
        if not rl_adjustments:
            return None
        
        # Apply adjustments (same as sync version)
        adjusted_allocations = {}
        for symbol, base_usd in baseline.allocations.items():
            if symbol in rl_adjustments:
                rl_info = rl_adjustments[symbol]
                rl_usd = rl_info['position_pct'] * self.portfolio_value
                rl_weight = rl_info['confidence']
                adjusted_allocations[symbol] = (
                    rl_weight * rl_usd + (1 - rl_weight) * base_usd
                )
            else:
                adjusted_allocations[symbol] = base_usd
        
        total_allocated = sum(adjusted_allocations.values())
        
        return EnsembleAllocation(
            allocations=adjusted_allocations,
            total_allocated_usd=total_allocated,
            portfolio_value=self.portfolio_value,
            utilization_pct=total_allocated / self.portfolio_value * 100,
            strategy_weights=baseline.strategy_weights,
            drifted_strategies=baseline.drifted_strategies,
            max_drift_pct=baseline.max_drift_pct,
            correlation_penalties=baseline.correlation_penalties,
            timestamp_ns=time.time_ns()
        )
    
    def record_outcome(
        self,
        symbol: str,
        actual_return: float,
        used_rl: bool
    ):
        """
        Record trade outcome for performance comparison.
        
        Args:
            symbol: Asset symbol
            actual_return: Realized return
            used_rl: Whether RL was used for this trade
        """
        if used_rl:
            self.rl_returns.append(actual_return)
        else:
            self.baseline_returns.append(actual_return)
    
    def get_performance_comparison(self) -> Dict:
        """Get RL vs baseline performance comparison."""
        import numpy as np
        
        rl_mean = np.mean(self.rl_returns) if self.rl_returns else 0.0
        rl_std = np.std(self.rl_returns) if len(self.rl_returns) > 1 else 0.0
        
        baseline_mean = np.mean(self.baseline_returns) if self.baseline_returns else 0.0
        baseline_std = np.std(self.baseline_returns) if len(self.baseline_returns) > 1 else 0.0
        
        return {
            'rl': {
                'trades': len(self.rl_returns),
                'mean_return': rl_mean,
                'std_return': rl_std,
                'sharpe': rl_mean / rl_std if rl_std > 0 else 0.0
            },
            'baseline': {
                'trades': len(self.baseline_returns),
                'mean_return': baseline_mean,
                'std_return': baseline_std,
                'sharpe': baseline_mean / baseline_std if baseline_std > 0 else 0.0
            },
            'rl_advantage': rl_mean - baseline_mean
        }
    
    def get_state(self) -> Dict:
        """Get complete Phase 3 state."""
        state = {
            'portfolio_value': self.portfolio_value,
            'enable_rl': self.enable_rl,
            'total_decisions': self.total_decisions,
            'rl_decisions': self.rl_decisions,
            'fallback_decisions': self.fallback_decisions,
            'rl_usage_rate': (
                self.rl_decisions / self.total_decisions
                if self.total_decisions > 0 else 0.0
            ),
            'phase2': self.phase2.get_state(),
            'performance': self.get_performance_comparison()
        }
        
        if self.rl_client is not None:
            state['rl_client'] = self.rl_client.get_state()
        
        return state
    
    def stop(self):
        """Stop all background services."""
        self.phase2.stop()
        logger.info("Layer3Phase3Hybrid stopped")
