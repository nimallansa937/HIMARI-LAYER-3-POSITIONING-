"""
HIMARI OPUS V2 - Prometheus Metrics Module
===========================================

Comprehensive Prometheus metrics for Layer 3 monitoring.

30+ metrics covering:
- Position sizing
- Cascade detection
- Regime transitions
- Circuit breaker
- Ensemble (Phase 2)
- Configuration changes

Version: 3.1 Enhanced
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# POSITION SIZING METRICS
# ============================================================================

# Kelly fraction
layer3_kelly_fraction = Histogram(
    'himari_l3_kelly_fraction',
    'Kelly fraction applied to position sizing',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
)

layer3_kelly_position_size = Histogram(
    'himari_l3_kelly_position_size_usd',
    'Kelly position size in USD',
    buckets=[100, 500, 1000, 5000, 10000, 25000, 50000, 100000]
)

# Conformal scaling
layer3_conformal_scale_factor = Histogram(
    'himari_l3_conformal_scale_factor',
    'Conformal prediction scale factor',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

layer3_conformal_residuals = Summary(
    'himari_l3_conformal_residuals',
    'Conformal prediction residuals'
)

layer3_conformal_samples = Gauge(
    'himari_l3_conformal_samples',
    'Number of samples in conformal scaler'
)

layer3_conformal_null_rejections = Counter(
    'himari_l3_conformal_null_rejections_total',
    'NULL value rejections in conformal scaler'
)

layer3_conformal_nan_rejections = Counter(
    'himari_l3_conformal_nan_rejections_total',
    'NaN/Inf rejections in conformal scaler'
)

# Bayesian Kelly
layer3_bayesian_win_rate = Gauge(
    'himari_l3_bayesian_win_rate',
    'Bayesian posterior win rate'
)

layer3_bayesian_edge = Gauge(
    'himari_l3_bayesian_edge',
    'Bayesian posterior edge estimate'
)

layer3_bayesian_trades = Counter(
    'himari_l3_bayesian_trades_total',
    'Total trades for Bayesian update',
    ['outcome']  # win, loss
)


# ============================================================================
# REGIME METRICS
# ============================================================================

layer3_current_regime = Gauge(
    'himari_l3_current_regime',
    'Current market regime (encoded)',
    ['regime']  # trending_up, trending_down, ranging, high_volatility, crisis
)

layer3_regime_multiplier = Gauge(
    'himari_l3_regime_multiplier',
    'Current regime multiplier',
    ['regime']
)

layer3_regime_transitions = Counter(
    'himari_l3_regime_transitions_total',
    'Regime transitions',
    ['from_regime', 'to_regime']
)

layer3_regime_false_flips = Counter(
    'himari_l3_regime_false_flips_total',
    'Regime candidate flips that didn\'t confirm'
)

layer3_regime_confirmation_progress = Gauge(
    'himari_l3_regime_confirmation_progress',
    'Regime confirmation progress (0-3)',
    ['candidate_regime']
)


# ============================================================================
# CASCADE DETECTION METRICS
# ============================================================================

layer3_cascade_risk_score = Gauge(
    'himari_l3_cascade_risk_score',
    'Aggregate cascade risk score [0-1]'
)

layer3_cascade_risk_components = Gauge(
    'himari_l3_cascade_risk_components',
    'Individual cascade risk components',
    ['component']  # funding, oi_drop, volume_spike, whale_pressure, netflow
)

layer3_cascade_recommendations = Counter(
    'himari_l3_cascade_recommendations_total',
    'Cascade recommendations issued',
    ['recommendation']  # EXIT, REDUCE_75%, REDUCE_50%, MONITOR
)

layer3_cascade_threshold_breaches = Counter(
    'himari_l3_cascade_threshold_breaches_total',
    'Cascade component threshold breaches',
    ['component']
)


# ============================================================================
# CIRCUIT BREAKER METRICS
# ============================================================================

layer3_circuit_breaker_state = Gauge(
    'himari_l3_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)'
)

layer3_circuit_breaker_failures = Counter(
    'himari_l3_circuit_breaker_failures_total',
    'Circuit breaker failures'
)

layer3_circuit_breaker_fallbacks = Counter(
    'himari_l3_circuit_breaker_fallbacks_total',
    'Fallbacks triggered by circuit breaker'
)

layer3_circuit_breaker_timeout = Gauge(
    'himari_l3_circuit_breaker_timeout_sec',
    'Current circuit breaker timeout in seconds'
)

layer3_circuit_breaker_success_rate = Gauge(
    'himari_l3_circuit_breaker_success_rate',
    'Circuit breaker success rate [0-1]'
)


# ============================================================================
# ENSEMBLE METRICS (Phase 2)
# ============================================================================

layer3_ensemble_weight_drift = Histogram(
    'himari_l3_ensemble_weight_drift_pct',
    'Ensemble weight drift percentage',
    ['strategy_id'],
    buckets=[1, 5, 10, 20, 50, 100]
)

layer3_ensemble_weight_current = Gauge(
    'himari_l3_ensemble_weight_current',
    'Current ensemble weight per strategy',
    ['strategy_id']
)

layer3_ensemble_weight_total = Gauge(
    'himari_l3_ensemble_weight_total_usd',
    'Total ensemble allocation in USD'
)

layer3_ensemble_strategies = Gauge(
    'himari_l3_ensemble_strategies',
    'Number of active strategies in ensemble'
)

layer3_ensemble_utilization = Gauge(
    'himari_l3_ensemble_utilization_pct',
    'Portfolio utilization percentage'
)

# Risk budget metrics
layer3_risk_budget_violations = Counter(
    'himari_l3_risk_budget_violations_total',
    'Risk budget violations',
    ['level']  # portfolio, strategy, position
)

layer3_risk_budget_utilization = Gauge(
    'himari_l3_risk_budget_utilization_pct',
    'Risk budget utilization percentage',
    ['level', 'identifier']
)

# Correlation alerts
layer3_correlation_alerts = Counter(
    'himari_l3_correlation_alerts_total',
    'High correlation alert count'
)


# ============================================================================
# CONFIGURATION METRICS
# ============================================================================

layer3_config_reloads = Counter(
    'himari_l3_config_reloads_total',
    'Configuration reload attempts',
    ['status']  # success, validation_failed, error
)

layer3_config_callbacks = Gauge(
    'himari_l3_config_callbacks',
    'Number of registered config callbacks'
)


# ============================================================================
# DECISION METRICS
# ============================================================================

layer3_position_decisions = Counter(
    'himari_l3_position_decisions_total',
    'Position sizing decisions made',
    ['symbol', 'strategy_id']
)

layer3_final_position_size = Histogram(
    'himari_l3_final_position_size_usd',
    'Final position size in USD',
    buckets=[0, 100, 500, 1000, 5000, 10000, 25000, 50000, 100000]
)

layer3_position_reductions = Counter(
    'himari_l3_position_reductions_total',
    'Position reductions applied',
    ['reason']  # cascade, regime, conformal
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_kelly_metrics(kelly_fraction: float, position_size: float):
    """Record Kelly engine metrics."""
    layer3_kelly_fraction.observe(kelly_fraction)
    layer3_kelly_position_size.observe(position_size)


def record_conformal_metrics(scale_factor: float, samples: int, residual: float = None):
    """Record conformal scaler metrics."""
    layer3_conformal_scale_factor.observe(scale_factor)
    layer3_conformal_samples.set(samples)
    if residual is not None:
        layer3_conformal_residuals.observe(residual)


def record_regime_metrics(regime: str, multiplier: float):
    """Record regime adjustment metrics."""
    # Only set current regime (no need to clear others)
    layer3_current_regime.labels(regime=regime).set(1)
    layer3_regime_multiplier.labels(regime=regime).set(multiplier)


def record_cascade_metrics(risk_score: float, components: dict, recommendation: str):
    """Record cascade detection metrics."""
    layer3_cascade_risk_score.set(risk_score)
    
    for component, value in components.items():
        layer3_cascade_risk_components.labels(component=component).set(value)
        if value > 0.5:
            layer3_cascade_threshold_breaches.labels(component=component).inc()
    
    layer3_cascade_recommendations.labels(recommendation=recommendation).inc()


def record_circuit_breaker_metrics(state: str, timeout: float, success_rate: float):
    """Record circuit breaker metrics."""
    state_map = {'closed': 0, 'half_open': 1, 'open': 2}
    layer3_circuit_breaker_state.set(state_map.get(state, 0))
    layer3_circuit_breaker_timeout.set(timeout)
    layer3_circuit_breaker_success_rate.set(success_rate)


def record_position_decision(symbol: str, strategy_id: str, position_size: float):
    """Record position sizing decision."""
    layer3_position_decisions.labels(symbol=symbol, strategy_id=strategy_id).inc()
    layer3_final_position_size.observe(position_size)


logger.info("Prometheus metrics module initialized (30+ metrics)")
