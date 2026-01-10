"""Position sizing engines - 5-Tier Architecture"""

# Legacy components (kept for compatibility)
from .bayesian_kelly import BayesianKellyEngine
from .conformal_scaler import ConformalPositionScaler
from .regime_adjuster import RegimeConditionalAdjuster
from .sentiment_sizer import SentimentAwareSizer
from .execution_engine import ExecutionEngine, ExecutionReport, Position

# NEW: 5-Tier Architecture (per CLAUDE Guide)
# Tier 1
from .volatility_target import VolatilityTargetEngine

# Tier 2
from .funding_rate_signal import FundingRateSignal
from .rl_directional_delta import RLDirectionalDelta
from .correlation_adjuster import CorrelationAdjuster
from .cascade_anomaly_detector import CascadeAnomalyDetector
from .bounded_adaptive import BoundedAdaptiveEnhancement

# Tier 3
from .regime_conditional_adjuster import RegimeConditionalAdjuster as Tier3RegimeAdjuster

# Tier 4
from .hard_constraints import HardConstraintEnforcer

# Tier 5
from .circuit_breaker_system import CircuitBreakerSystem

# Main Engine
from .layer3_engine import Layer3PositionSizingEngine
