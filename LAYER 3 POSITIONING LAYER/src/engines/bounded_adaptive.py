"""
HIMARI OPUS 2 - Tier 2: Bounded Adaptive Enhancement
=====================================================

Combines all adaptive components with strict bounds.
Implements Tier 2 per CLAUDE Guide Part V.

The key invariant: Tier 2 output is always in [0.7 × base, 1.3 × base],
ensuring adaptive components cannot dramatically change positions.

Version: 1.0
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any

# Handle both module and script imports
try:
    from engines.funding_rate_signal import FundingRateSignal
    from engines.rl_directional_delta import RLDirectionalDelta
    from engines.correlation_adjuster import CorrelationAdjuster
    from engines.cascade_anomaly_detector import CascadeAnomalyDetector
    from core.layer3_config import AdaptiveConfig
except ImportError:
    from .funding_rate_signal import FundingRateSignal
    from .rl_directional_delta import RLDirectionalDelta
    from .correlation_adjuster import CorrelationAdjuster
    from .cascade_anomaly_detector import CascadeAnomalyDetector
    from ..core.layer3_config import AdaptiveConfig

logger = logging.getLogger(__name__)


class BoundedAdaptiveEnhancement:
    """
    Tier 2: Combine all adaptive components with strict bounds.
    
    The key invariant is that Tier 2 output is always in the range
    [0.7 × base, 1.3 × base], ensuring that adaptive components
    cannot dramatically increase or decrease positions.
    
    Components:
    1. RL Directional Delta: ±30% based on learned policy
    2. Funding Rate Signal: Reduce when funding extreme
    3. Correlation Monitor: Reduce when BTC correlation high
    4. Cascade Detector: Reduce on cascade precursors
    """
    
    def __init__(self, config: AdaptiveConfig = None):
        """
        Initialize bounded adaptive enhancement.
        
        Args:
            config: AdaptiveConfig with all parameters
        """
        if config is None:
            config = AdaptiveConfig()
        
        # Initialize components
        self.rl_delta = RLDirectionalDelta(
            model_path=config.rl_model_path,
            bounds=config.rl_delta_bounds
        )
        
        self.funding_signal = FundingRateSignal(
            threshold_reduce=config.funding_threshold_reduce,
            threshold_exit=config.funding_threshold_exit,
            reduction_factor=config.funding_reduction_factor
        )
        
        self.correlation_adjuster = CorrelationAdjuster(
            threshold_elevated=config.correlation_threshold_elevated,
            threshold_extreme=config.correlation_threshold_extreme,
            reduction_elevated=config.correlation_reduction_elevated,
            reduction_extreme=config.correlation_reduction_extreme
        )
        
        self.cascade_detector = CascadeAnomalyDetector(
            oi_drop_threshold=config.cascade_oi_threshold,
            volume_spike_threshold=config.cascade_volume_threshold,
            reduction_factor=config.cascade_reduction_factor
        )
        
        # Master bounds: adaptive output cannot exceed these
        self.adaptive_lower_bound = config.adaptive_lower_bound  # 0.7
        self.adaptive_upper_bound = config.adaptive_upper_bound  # 1.3
    
    def compute_adaptive_adjustment(
        self,
        base_position: float,
        features: np.ndarray,
        market_context: dict,
        regime: str,
        direction: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply all adaptive components to base position.
        
        Args:
            base_position: Position size from Tier 1
            features: Feature vector for RL policy
            market_context: Market data (funding, correlation, OI, volume)
            regime: Current market regime
            direction: Trade direction (LONG/SHORT)
            
        Returns:
            Tuple of (adjusted_position, all_diagnostics)
        """
        all_diagnostics = {}
        
        # Component 1: RL Delta (only in NORMAL regime)
        rl_delta, rl_diag = self.rl_delta.compute_delta(features, regime)
        all_diagnostics['rl'] = rl_diag
        
        # Component 2: Funding Rate
        funding_mult, funding_diag = self.funding_signal.compute_adjustment(
            market_context.get('funding_rate', 0.0),
            direction
        )
        all_diagnostics['funding'] = funding_diag
        
        # Component 3: Correlation
        corr_mult, corr_diag = self.correlation_adjuster.compute_adjustment(
            market_context.get('btc_correlation', 0.0),
            market_context.get('symbol', '')
        )
        all_diagnostics['correlation'] = corr_diag
        
        # Component 4: Cascade Detection
        _, cascade_mult, cascade_diag = self.cascade_detector.compute_cascade_score(
            market_context.get('open_interest_delta_1h', 0.0),
            market_context.get('volume_spike_ratio', 1.0),
            market_context.get('funding_rate', 0.0)
        )
        all_diagnostics['cascade'] = cascade_diag
        
        # Combine adjustments
        # RL delta is additive: (1 + delta)
        # Other components are multiplicative
        raw_multiplier = (1 + rl_delta) * funding_mult * corr_mult * cascade_mult
        
        # Enforce master bounds (critical safety mechanism)
        combined_multiplier = np.clip(
            raw_multiplier,
            self.adaptive_lower_bound,
            self.adaptive_upper_bound
        )
        
        adjusted_position = base_position * combined_multiplier
        
        was_bounded = raw_multiplier != combined_multiplier
        
        all_diagnostics['combined'] = {
            'rl_delta': rl_delta,
            'funding_mult': funding_mult,
            'corr_mult': corr_mult,
            'cascade_mult': cascade_mult,
            'raw_multiplier': raw_multiplier,
            'bounded_multiplier': combined_multiplier,
            'was_bounded': was_bounded,
            'bounds': (self.adaptive_lower_bound, self.adaptive_upper_bound),
            'base_position': base_position,
            'adjusted_position': adjusted_position,
            'tier': 'ADAPTIVE_ENHANCEMENT'
        }
        
        if was_bounded:
            logger.info(
                f"Adaptive multiplier bounded: {raw_multiplier:.3f} → {combined_multiplier:.3f}"
            )
        
        logger.debug(
            f"Tier 2 output: ${adjusted_position:,.2f} "
            f"(multiplier={combined_multiplier:.3f})"
        )
        
        return adjusted_position, all_diagnostics
