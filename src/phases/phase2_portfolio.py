"""
HIMARI OPUS V2 - Phase 2 Portfolio Orchestrator (Updated)
==========================================================

Integrates Phase 1 position sizing with Phase 2 multi-asset
portfolio allocation. Full production features.

Features:
- Multi-asset Kelly allocation
- Ensemble position aggregation
- Correlation monitoring
- Hierarchical risk budgets
- Prometheus metrics integration
- Input validation

Version: 3.1 Phase 2 Production
"""

from typing import Dict, List, Tuple, Optional
import time
import logging
import math
import numpy as np

try:
    from core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        EnsembleAllocation, InvalidSignalException
    )
    from core.layer3_config_manager import ConfigManager
    from core import layer3_metrics as metrics
    from phases.phase1_core import Layer3Phase1
    from portfolio.multi_asset_kelly import MultiAssetKellyAllocator
    from portfolio.ensemble_aggregator import EnsemblePositionAggregatorV2
    from portfolio.correlation_monitor import CorrelationMonitor
    from portfolio.risk_budget import HierarchicalRiskBudgetManager
except ImportError:
    from ..core.layer3_types import (
        TacticalSignal, PositionSizingDecision, MarketRegime, CascadeIndicators,
        EnsembleAllocation, InvalidSignalException
    )
    from ..core.layer3_config_manager import ConfigManager
    from ..core import layer3_metrics as metrics
    from ..phases.phase1_core import Layer3Phase1
    from ..portfolio.multi_asset_kelly import MultiAssetKellyAllocator
    from ..portfolio.ensemble_aggregator import EnsemblePositionAggregatorV2
    from ..portfolio.correlation_monitor import CorrelationMonitor
    from ..portfolio.risk_budget import HierarchicalRiskBudgetManager

logger = logging.getLogger(__name__)


class Layer3Phase2Portfolio:
    """
    Phase 2 Portfolio Orchestrator (Production).
    
    Combines Phase 1 position sizing with Phase 2 portfolio features:
    1. Input validation
    2. Process multiple signals through Phase 1 pipeline
    3. Apply hierarchical risk budgets
    4. Aggregate using ensemble aggregator
    5. Apply multi-asset Kelly allocation
    6. Monitor correlations
    7. Record Prometheus metrics
    """
    
    def __init__(
        self,
        portfolio_value: float = 100000,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.15,
        max_strategy_pct: float = 0.40,
        max_portfolio_pct: float = 0.80,
        max_correlation: float = 0.7,
        config_path: Optional[str] = None,
        enable_metrics: bool = True
    ):
        """
        Initialize Phase 2 orchestrator.
        
        Args:
            portfolio_value: Total portfolio value in USD
            kelly_fraction: Kelly multiplier
            max_position_pct: Max single position as % of portfolio
            max_strategy_pct: Max per-strategy allocation
            max_portfolio_pct: Max total portfolio utilization
            max_correlation: Maximum acceptable correlation
            config_path: Path to configuration file
            enable_metrics: Enable Prometheus metrics
        """
        # Validate inputs
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be > 0, got {portfolio_value}")
        if not 0 < kelly_fraction <= 0.5:
            raise ValueError(f"Kelly fraction must be in (0, 0.5], got {kelly_fraction}")
        
        self.portfolio_value = portfolio_value
        self.enable_metrics = enable_metrics
        
        # Initialize config manager
        self.config_manager = None
        if config_path:
            try:
                self.config_manager = ConfigManager(config_path)
                logger.info(f"Config loaded: {config_path}")
            except Exception as e:
                logger.warning(f"Config load failed: {e}")
        
        # Initialize Phase 1
        self.phase1 = Layer3Phase1(
            portfolio_value=portfolio_value,
            kelly_fraction=kelly_fraction,
            enable_metrics=enable_metrics,
            enable_sentiment=True
        )
        
        # Initialize Phase 2 components
        self.kelly_allocator = MultiAssetKellyAllocator(
            portfolio_value=portfolio_value,
            max_position_pct=max_position_pct,
            max_portfolio_utilization=max_portfolio_pct,
            kelly_fraction=kelly_fraction
        )
        
        self.ensemble = EnsemblePositionAggregatorV2(
            portfolio_value=portfolio_value,
            max_position_pct=max_position_pct,
            max_correlation=max_correlation
        )
        
        self.correlation_monitor = CorrelationMonitor(
            window_size=60,
            correlation_alert_threshold=max_correlation
        )
        
        # Hierarchical risk budgets
        self.risk_budget = HierarchicalRiskBudgetManager(
            portfolio_value=portfolio_value,
            portfolio_max_pct=max_portfolio_pct,
            strategy_max_pct=max_strategy_pct,
            position_max_pct=max_position_pct
        )
        
        # Statistics
        self.total_portfolio_decisions = 0
        
        logger.info(
            f"Layer3Phase2Portfolio initialized: portfolio=${portfolio_value:,.0f}, "
            f"max_pos={max_position_pct:.0%}, max_strat={max_strategy_pct:.0%}"
        )
    
    def _validate_signals(self, signals: List[TacticalSignal]) -> List[str]:
        """Validate signal list. Returns list of error messages."""
        errors = []
        
        if signals is None:
            errors.append("Signals list is None")
            return errors
        
        for i, signal in enumerate(signals):
            if signal is None:
                errors.append(f"Signal[{i}] is None")
                continue
            if not signal.symbol:
                errors.append(f"Signal[{i}] missing symbol")
            if signal.confidence is None or not 0 <= signal.confidence <= 1:
                errors.append(f"Signal[{i}] invalid confidence: {signal.confidence}")
        
        return errors
    
    def _validate_prices(
        self, 
        prices: Dict[str, float],
        required_symbols: List[str]
    ) -> List[str]:
        """Validate prices dict. Returns list of error messages."""
        errors = []
        
        if prices is None:
            errors.append("Prices dict is None")
            return errors
        
        for symbol in required_symbols:
            if symbol not in prices:
                errors.append(f"Missing price for {symbol}")
            elif prices[symbol] is None or prices[symbol] <= 0:
                errors.append(f"Invalid price for {symbol}: {prices.get(symbol)}")
            elif math.isnan(prices[symbol]) or math.isinf(prices[symbol]):
                errors.append(f"NaN/Inf price for {symbol}")
        
        return errors
    
    def process_multi_strategy_signals(
        self,
        signals: List[TacticalSignal],
        cascade_indicators: CascadeIndicators,
        current_prices: Dict[str, float]
    ) -> EnsembleAllocation:
        """
        Process multiple strategy signals through complete Phase 2 pipeline.
        
        Args:
            signals: List of tactical signals from multiple strategies
            cascade_indicators: Shared cascade indicators
            current_prices: {symbol: price} for all symbols
            
        Returns:
            EnsembleAllocation with portfolio-level decision
        """
        self.total_portfolio_decisions += 1
        timestamp_ns = time.time_ns()
        
        # Validate inputs
        if not signals:
            logger.warning("Empty signals list")
            return self._create_empty_allocation(timestamp_ns)
        
        signal_errors = self._validate_signals(signals)
        if signal_errors:
            for err in signal_errors:
                logger.error(f"Signal validation: {err}")
            return self._create_empty_allocation(timestamp_ns)
        
        required_symbols = [s.symbol for s in signals]
        price_errors = self._validate_prices(current_prices, required_symbols)
        if price_errors:
            for err in price_errors:
                logger.warning(f"Price validation: {err}")
            # Continue with valid symbols only
        
        # Reset risk budget utilization for this batch
        self.risk_budget.reset_utilization()
        
        # Phase 1: Process each signal individually
        phase1_decisions = []
        strategy_data = []
        
        for signal in signals:
            price = current_prices.get(signal.symbol, 0.0)
            
            if price <= 0 or math.isnan(price) or math.isinf(price):
                logger.warning(f"Invalid price for {signal.symbol}, skipping")
                continue
            
            try:
                decision = self.phase1.calculate_position(
                    signal=signal,
                    cascade_indicators=cascade_indicators,
                    current_price=price
                )
                
                # Apply hierarchical risk budget
                allowed_usd, budget_diag = self.risk_budget.check_and_enforce(
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    requested_usd=decision.position_size_usd
                )
                
                if budget_diag['violations']:
                    logger.info(f"Risk budget violations: {budget_diag['violations']}")
                
                phase1_decisions.append(decision)
                
                strategy_data.append({
                    'id': signal.strategy_id,
                    'symbol': signal.symbol,
                    'size': allowed_usd,  # Use budget-enforced size
                    'confidence': signal.confidence,
                    'expected_return': signal.expected_return or 0.05,
                    'volatility': signal.predicted_volatility or 0.02
                })
                
            except InvalidSignalException as e:
                logger.error(f"Signal invalid for {signal.symbol}: {e}")
            except Exception as e:
                logger.error(f"Phase 1 failed for {signal.symbol}: {e}")
        
        if not strategy_data:
            return self._create_empty_allocation(timestamp_ns)
        
        # Build correlation matrix
        symbols = list(set(s['symbol'] for s in strategy_data))
        n_symbols = len(symbols)
        corr_matrix = self._build_correlation_matrix(symbols)
        
        # Phase 2: Ensemble aggregation
        allocations, ensemble_diag = self.ensemble.aggregate_positions(
            strategies=strategy_data,
            correlation_matrix=corr_matrix
        )
        
        # Calculate total and utilization
        total_allocated = sum(allocations.values())
        
        # Calculate weights
        strategy_weights = {}
        for s in strategy_data:
            alloc = allocations.get(s['symbol'], 0.0)
            strategy_weights[s['id']] = alloc / total_allocated if total_allocated > 0 else 0.0
        
        # Detect drifted strategies
        drift_alerts = ensemble_diag.get('drift_alerts', [])
        drifted = [a.split(':')[0] for a in drift_alerts] if drift_alerts else []
        
        # Calculate max drift percentage
        max_drift = 0.0
        if drift_alerts:
            # Parse drift from alert strings
            for alert in drift_alerts:
                try:
                    pct_str = alert.split('(')[1].replace('% change)', '')
                    max_drift = max(max_drift, float(pct_str))
                except:
                    pass
        
        # Record metrics
        if self.enable_metrics:
            try:
                metrics.layer3_ensemble_weight_total.set(total_allocated)
                for sym, alloc in allocations.items():
                    metrics.layer3_ensemble_weight_current.labels(strategy_id=sym).set(
                        alloc / self.portfolio_value if self.portfolio_value > 0 else 0
                    )
                if max_drift > 0:
                    metrics.layer3_ensemble_weight_drift.labels(strategy_id='max').observe(max_drift)
            except Exception as e:
                logger.debug(f"Metrics recording skipped: {e}")
        
        # Create ensemble allocation
        allocation = EnsembleAllocation(
            allocations=allocations,
            total_allocated_usd=total_allocated,
            portfolio_value=self.portfolio_value,
            utilization_pct=total_allocated / self.portfolio_value * 100,
            strategy_weights=strategy_weights,
            drifted_strategies=drifted,
            max_drift_pct=max_drift,
            correlation_penalties=ensemble_diag.get('correlation_penalties', {}),
            timestamp_ns=timestamp_ns
        )
        
        logger.info(
            f"Portfolio decision: {len(signals)} strategies → "
            f"${total_allocated:,.2f} allocated ({allocation.utilization_pct:.1f}%)"
        )
        
        return allocation
    
    def _build_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Build correlation matrix for symbols from monitor."""
        n_symbols = len(symbols)
        corr_matrix = np.eye(n_symbols)
        
        if self.correlation_monitor.current_correlation is not None:
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    if i != j:
                        corr = self.correlation_monitor.get_correlation_for_pair(s1, s2)
                        if corr is not None:
                            corr_matrix[i, j] = corr
        
        return corr_matrix
    
    def _create_empty_allocation(self, timestamp_ns: int) -> EnsembleAllocation:
        """Create empty allocation when no valid strategies."""
        return EnsembleAllocation(
            allocations={},
            total_allocated_usd=0.0,
            portfolio_value=self.portfolio_value,
            utilization_pct=0.0,
            strategy_weights={},
            drifted_strategies=[],
            max_drift_pct=0.0,
            correlation_penalties={},
            timestamp_ns=timestamp_ns
        )
    
    def update_correlations(
        self,
        returns: Dict[str, float]
    ) -> Dict:
        """
        Update correlation monitor with new returns.
        
        Args:
            returns: {symbol: period_return}
            
        Returns:
            Diagnostics dict
        """
        # Validate returns
        if returns is None:
            return {'status': 'error', 'error': 'returns is None'}
        
        clean_returns = {}
        for sym, ret in returns.items():
            if ret is not None and not math.isnan(ret) and not math.isinf(ret):
                clean_returns[sym] = ret
        
        corr_matrix, diag = self.correlation_monitor.update(clean_returns)
        return diag
    
    def export_weight_history(self, output_path: str) -> bool:
        """Export ensemble weight history to CSV."""
        return self.ensemble.export_weight_history(output_path)
    
    def get_remaining_risk_budget(self) -> Dict:
        """Get remaining risk budget at all levels."""
        return self.risk_budget.get_remaining_budget()
    
    def get_state(self) -> Dict:
        """Get complete Phase 2 state."""
        return {
            'portfolio_value': self.portfolio_value,
            'total_portfolio_decisions': self.total_portfolio_decisions,
            'phase1': self.phase1.get_state(),
            'kelly_allocator': self.kelly_allocator.get_statistics(),
            'ensemble': self.ensemble.get_state(),
            'correlation_monitor': self.correlation_monitor.get_state(),
            'risk_budget': self.risk_budget.get_state()
        }
    
    def stop(self):
        """Stop all background services."""
        self.phase1.stop()
        logger.info("Layer3Phase2Portfolio stopped")
