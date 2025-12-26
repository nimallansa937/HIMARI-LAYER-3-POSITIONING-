"""
HIMARI OPUS V2 - Enhanced Cascade Detector
===========================================

Production-grade cascade detector integrating:
- Existing OPUS 2 EnhancedCascadeDetector logic
- Layer 1 on-chain whale pressure signals
- Exchange netflow Z-scores
- Multi-factor risk scoring

Cost: $0 (reuses existing OPUS 2 logic + L1 signals)
Version: 3.1 Enhanced
"""

from typing import Tuple, Dict
import logging

# Handle both module and script imports
try:
    from core.layer3_types import CascadeIndicators
except ImportError:
    from ..core.layer3_types import CascadeIndicators

logger = logging.getLogger(__name__)


class EnhancedCascadeDetector:
    """
    Production-grade cascade detector (reused from OPUS 2).
    
    Enhancements over RuleBasedCascadeDetector:
    - Integrates Layer 1 on-chain whale pressure
    - Exchange netflow Z-scores for early warning
    - Multi-factor risk scoring with weights
    - Historical calibration support
    
    Cost: $0 (reuses existing OPUS 2 logic + L1 signals)
    """
    
    def __init__(self):
        # Original thresholds (validated in OPUS 2)
        self.funding_rate_threshold = 0.003  # 0.3% absolute
        self.oi_drop_threshold = 0.10        # 10% decline
        self.volume_spike_threshold = 5.0    # 5× average
        
        # NEW: On-chain thresholds
        self.whale_pressure_threshold = 0.7   # Normalized [0,1]
        self.netflow_zscore_threshold = 2.0   # 2-sigma event
        
        # Risk component weights
        self.weights = {
            'funding': 0.20,
            'oi_drop': 0.25,
            'volume_spike': 0.15,
            'whale_pressure': 0.25,  # NEW
            'netflow': 0.15          # NEW
        }
    
    def calculate_cascade_risk(
        self,
        indicators: CascadeIndicators
    ) -> Tuple[float, str, Dict]:
        """
        Calculate multi-factor liquidation cascade risk.
        
        Args:
            indicators: Complete cascade indicators from L1/L2
            
        Returns:
            (risk_score, recommendation, diagnostics)
            - risk_score: [0.0, 1.0]
            - recommendation: EXIT, REDUCE_75%, REDUCE_50%, MONITOR
            - diagnostics: Component breakdown for debugging
        """
        components = {}
        
        # Component 1: Funding rate pressure
        funding_risk = 0.0
        if abs(indicators.funding_rate) > self.funding_rate_threshold:
            funding_risk = min(1.0, abs(indicators.funding_rate) / 0.01)  # Cap at 1% funding
        components['funding_risk'] = funding_risk
        
        # Component 2: Open interest collapse
        oi_risk = 0.0
        if indicators.oi_change_pct < -self.oi_drop_threshold:
            oi_risk = min(1.0, abs(indicators.oi_change_pct) / 0.30)  # Cap at -30% OI drop
        components['oi_risk'] = oi_risk
        
        # Component 3: Volume spike (liquidation activity)
        volume_risk = 0.0
        if indicators.volume_ratio > self.volume_spike_threshold:
            volume_risk = min(1.0, indicators.volume_ratio / 10.0)  # Cap at 10× volume
        components['volume_risk'] = volume_risk
        
        # Component 4: On-chain whale pressure (NEW)
        whale_risk = 0.0
        if indicators.onchain_whale_pressure > self.whale_pressure_threshold:
            whale_risk = indicators.onchain_whale_pressure
        components['whale_risk'] = whale_risk
        
        # Component 5: Exchange netflow anomaly (NEW)
        netflow_risk = 0.0
        if abs(indicators.exchange_netflow_zscore) > self.netflow_zscore_threshold:
            netflow_risk = min(1.0, abs(indicators.exchange_netflow_zscore) / 4.0)  # Cap at 4-sigma
        components['netflow_risk'] = netflow_risk
        
        # Weighted aggregate risk score
        aggregate_risk = (
            self.weights['funding'] * funding_risk +
            self.weights['oi_drop'] * oi_risk +
            self.weights['volume_spike'] * volume_risk +
            self.weights['whale_pressure'] * whale_risk +
            self.weights['netflow'] * netflow_risk
        )
        
        aggregate_risk = min(1.0, aggregate_risk)
        
        # Recommendation thresholds
        if aggregate_risk > 0.8:
            recommendation = "EXIT"
        elif aggregate_risk > 0.6:
            recommendation = "REDUCE_75%"
        elif aggregate_risk > 0.4:
            recommendation = "REDUCE_50%"
        else:
            recommendation = "MONITOR"
        
        # Diagnostics for debugging/monitoring
        diagnostics = {
            'components': components,
            'aggregate_risk': aggregate_risk,
            'dominant_factor': max(components, key=components.get),
            'threshold_breaches': [
                k for k, v in components.items() if v > 0.5
            ]
        }
        
        # Log high risk scenarios
        if aggregate_risk > 0.6:
            logger.warning(
                f"High cascade risk detected: {aggregate_risk:.3f} - {recommendation}. "
                f"Dominant factor: {diagnostics['dominant_factor']} = {components[diagnostics['dominant_factor']]:.3f}"
            )
        
        return aggregate_risk, recommendation, diagnostics
    
    def update_thresholds(
        self,
        funding_rate_threshold: float = None,
        oi_drop_threshold: float = None,
        volume_spike_threshold: float = None,
        whale_pressure_threshold: float = None,
        netflow_zscore_threshold: float = None
    ):
        """
        Update detection thresholds (for calibration).
        
        Args:
            funding_rate_threshold: Absolute funding rate threshold
            oi_drop_threshold: OI drop % threshold
            volume_spike_threshold: Volume spike multiplier threshold
            whale_pressure_threshold: Whale pressure threshold [0,1]
            netflow_zscore_threshold: Netflow Z-score threshold
        """
        if funding_rate_threshold is not None:
            self.funding_rate_threshold = funding_rate_threshold
        if oi_drop_threshold is not None:
            self.oi_drop_threshold = oi_drop_threshold
        if volume_spike_threshold is not None:
            self.volume_spike_threshold = volume_spike_threshold
        if whale_pressure_threshold is not None:
            self.whale_pressure_threshold = whale_pressure_threshold
        if netflow_zscore_threshold is not None:
            self.netflow_zscore_threshold = netflow_zscore_threshold
        
        logger.info("Cascade detector thresholds updated")
    
    def update_weights(
        self,
        funding: float = None,
        oi_drop: float = None,
        volume_spike: float = None,
        whale_pressure: float = None,
        netflow: float = None
    ):
        """
        Update component weights (for calibration).
        Weight sum should equal 1.0.
        
        Args:
            funding: Funding rate component weight
            oi_drop: OI drop component weight
            volume_spike: Volume spike component weight
            whale_pressure: Whale pressure component weight
            netflow: Netflow component weight
        """
        if funding is not None:
            self.weights['funding'] = funding
        if oi_drop is not None:
            self.weights['oi_drop'] = oi_drop
        if volume_spike is not None:
            self.weights['volume_spike'] = volume_spike
        if whale_pressure is not None:
            self.weights['whale_pressure'] = whale_pressure
        if netflow is not None:
            self.weights['netflow'] = netflow
        
        # Validate weights sum to 1.0 (approximately)
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Component weights sum to {weight_sum:.3f}, expected 1.0")
        
        logger.info(f"Cascade detector weights updated: {self.weights}")
