"""
HIMARI OPUS 2 - Tier 2: RL Directional Delta
=============================================

Bounded RL adjustment to base position size.
Part of Tier 2 Bounded Adaptive Enhancement per CLAUDE Guide Part V.

Version: 1.0
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional torch import for RL model
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, RL delta will use fallback mode")


class RLDirectionalDelta:
    """
    Bounded RL adjustment to base position size.
    
    The RL policy is trained to maximize Sharpe ratio but its output
    is strictly bounded to prevent catastrophic recommendations.
    
    Sharpe contribution: +0.08 to +0.12
    """
    
    def __init__(
        self,
        model_path: str = None,
        bounds: Tuple[float, float] = (-0.30, +0.30),
        enabled_regimes: set = None
    ):
        """
        Initialize RL directional delta.
        
        Args:
            model_path: Path to pre-trained RL policy
            bounds: (lower, upper) bounds for delta
            enabled_regimes: Regimes where RL is active (default: NORMAL only)
        """
        self.model_path = model_path
        self.lower_bound, self.upper_bound = bounds
        self.enabled_regimes = enabled_regimes or {'NORMAL'}
        
        # Model state
        self.model = None
        self.model_loaded = False
        
        # Try to load model if path provided
        if model_path and TORCH_AVAILABLE:
            self._load_model(model_path)
    
    def _load_model(self, path: str) -> bool:
        """
        Load pre-trained RL policy.
        
        Model is expected to be a simple MLP: 60-dim features → 64 → 64 → 1
        Output is tanh-activated, range [-1, 1]
        """
        try:
            self.model = torch.load(path, map_location='cpu')
            self.model.eval()
            self.model_loaded = True
            logger.info(f"RL model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RL model from {path}: {e}")
            self.model_loaded = False
            return False
    
    def compute_delta(
        self,
        features: np.ndarray,
        regime: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute position adjustment from RL policy.
        
        Args:
            features: 60-dimensional feature vector from Layer 2
            regime: Current market regime
            
        Returns:
            Tuple of (delta, diagnostics)
            - delta: Adjustment factor (e.g., +0.15 means +15%)
            - diagnostics: Computation details
        """
        # Disable in non-NORMAL regimes
        if regime not in self.enabled_regimes:
            return 0.0, {
                'rl_raw_output': None,
                'rl_delta': 0.0,
                'rl_disabled_reason': f'Regime {regime} not in enabled set',
                'rl_enabled': False,
                'tier': 'RL_DELTA'
            }
        
        # Get raw policy output
        raw_output = self._get_model_output(features)
        
        # Clip to bounds (this is the critical safety mechanism)
        delta = np.clip(raw_output, self.lower_bound, self.upper_bound)
        
        was_clipped = raw_output != delta
        
        diagnostics = {
            'rl_raw_output': raw_output,
            'rl_delta': delta,
            'rl_was_clipped': was_clipped,
            'rl_bounds': (self.lower_bound, self.upper_bound),
            'rl_model_loaded': self.model_loaded,
            'rl_enabled': True,
            'regime': regime,
            'tier': 'RL_DELTA'
        }
        
        if was_clipped:
            logger.warning(
                f"RL output clipped: {raw_output:.3f} → {delta:.3f}"
            )
        
        return delta, diagnostics
    
    def _get_model_output(self, features: np.ndarray) -> float:
        """
        Get output from RL model or fallback.
        
        Args:
            features: Feature vector
            
        Returns:
            Raw model output in [-1, 1] range
        """
        if self.model_loaded and TORCH_AVAILABLE:
            try:
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    raw_output = self.model(features_tensor).item()
                return raw_output
            except Exception as e:
                logger.error(f"RL model inference failed: {e}")
        
        # Fallback: conservative zero delta
        # In production, this should be replaced with a trained model
        return 0.0
    
    def set_enabled_regimes(self, regimes: set):
        """Update enabled regimes."""
        self.enabled_regimes = regimes
        logger.info(f"RL enabled regimes updated: {regimes}")
