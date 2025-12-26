"""
HIMARI OPUS V2 - Conformal Position Scaler
===========================================

Conformal prediction-based position scaling with NULL safety and defensive checks.

Features:
- Residuals tracking (deque of last 200 trades)
- Quantile-based scaling at 90% coverage
- NULL safety checks for predicted and actual returns
- NaN/Inf validation to prevent crashes
- Comprehensive diagnostics

Version: 3.1 Enhanced
"""

from typing import Optional, Tuple, Dict
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConformalPositionScaler:
    """
    Conformal prediction-based position scaler with defensive NULL safety.
    
    Tracks prediction residuals and scales positions based on uncertainty
    quantification. Enhanced with robust NULL checking to prevent crashes.
    """
    
    def __init__(
        self,
        coverage: float = 0.90,
        window_size: int = 200,
        min_samples: int = 20
    ):
        """
        Initialize conformal scaler.
        
        Args:
            coverage: Target coverage level (e.g., 0.90 = 90%)
            window_size: Number of recent residuals to track
            min_samples: Minimum samples before scaling activates
        """
        self.coverage = coverage
        self.alpha = 1.0 - coverage  # Miscoverage rate
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Residuals tracking
        self.residuals = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        
        # Statistics
        self.total_updates = 0
        self.null_rejections = 0
        self.nan_rejections = 0
    
    def scale_position(
        self,
        base_position_usd: float,
        predicted_return: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        Scale position based on conformal prediction uncertainty.
        
        Args:
            base_position_usd: Base position size from Kelly
            predicted_return: Predicted return (optional, for diagnostics)
            
        Returns:
            (scaled_position_usd, diagnostics)
        """
        if len(self.residuals) < self.min_samples:
            # Not enough samples: no scaling
            return base_position_usd, {
                'scale_factor': 1.0,
                'samples': len(self.residuals),
                'min_samples': self.min_samples,
                'status': 'insufficient_samples'
            }
        
        # Calculate quantile at (1 - alpha) coverage
        quantile_value = np.quantile(list(self.residuals), 1.0 - self.alpha)
        
        # Scale factor (inverse of quantile)
        # Higher uncertainty (larger quantile) → smaller scale factor
        if quantile_value > 0:
            scale_factor = min(1.0, 1.0 / (1.0 + quantile_value))
        else:
            scale_factor = 1.0
        
        scaled_position_usd = base_position_usd * scale_factor
        
        # Diagnostics
        diagnostics = {
            'scale_factor': scale_factor,
            'quantile_value': quantile_value,
            'coverage': self.coverage,
            'samples': len(self.residuals),
            'mean_residual': np.mean(list(self.residuals)),
            'std_residual': np.std(list(self.residuals)),
            'status': 'active'
        }
        
        return scaled_position_usd, diagnostics
    
    def update(
        self,
        predicted_return: Optional[float],
        actual_return: Optional[float]
    ):
        """
        Update residuals with NULL safety and validation.
        
        Args:
            predicted_return: Predicted return
            actual_return: Actual return
        """
        # NEW: Defensive NULL checks
        if predicted_return is None or actual_return is None:
            self.null_rejections += 1
            logger.warning(
                f"Skipping conformal update: predicted={predicted_return}, actual={actual_return}. "
                f"Total NULL rejections: {self.null_rejections}"
            )
            return
        
        # Validate numeric types
        if not isinstance(predicted_return, (int, float)) or not isinstance(actual_return, (int, float)):
            self.null_rejections += 1
            logger.error(
                f"Invalid types for conformal update: "
                f"predicted type={type(predicted_return)}, actual type={type(actual_return)}"
            )
            return
        
        # Check for NaN/Inf
        if np.isnan(predicted_return) or np.isinf(predicted_return):
            self.nan_rejections += 1
            logger.error(f"Invalid predicted_return: {predicted_return}. Total NaN rejections: {self.nan_rejections}")
            return
        
        if np.isnan(actual_return) or np.isinf(actual_return):
            self.nan_rejections += 1
            logger.error(f"Invalid actual_return: {actual_return}. Total NaN rejections: {self.nan_rejections}")
            return
        
        # Safe to update
        residual = abs(actual_return - predicted_return)
        self.residuals.append(residual)
        self.predictions.append(predicted_return)
        self.actuals.append(actual_return)
        self.total_updates += 1
        
        logger.debug(
            f"Conformal updated: residual={residual:.4f}, "
            f"samples={len(self.residuals)}/{self.window_size}"
        )
    
    def get_statistics(self) -> Dict:
        """Get current statistics for monitoring."""
        if len(self.residuals) == 0:
            return {
                'samples': 0,
                'total_updates': self.total_updates,
                'null_rejections': self.null_rejections,
                'nan_rejections': self.nan_rejections,
                'status': 'no_data'
            }
        
        residuals_list = list(self.residuals)
        predictions_list = list(self.predictions)
        actuals_list = list(self.actuals)
        
        # Calculate coverage achieved
        quantile_value = np.quantile(residuals_list, 1.0 - self.alpha)
        coverage_achieved = np.mean([r <= quantile_value for r in residuals_list])
        
        return {
            'samples': len(self.residuals),
            'total_updates': self.total_updates,
            'null_rejections': self.null_rejections,
            'nan_rejections': self.nan_rejections,
            'mean_residual': np.mean(residuals_list),
            'median_residual': np.median(residuals_list),
            'std_residual': np.std(residuals_list),
            'max_residual': np.max(residuals_list),
            'min_residual': np.min(residuals_list),
            'quantile_value': quantile_value,
            'target_coverage': self.coverage,
            'achieved_coverage': coverage_achieved,
            'mean_prediction': np.mean(predictions_list),
            'mean_actual': np.mean(actuals_list),
            'prediction_bias': np.mean(predictions_list) - np.mean(actuals_list),
            'status': 'active'
        }
    
    def reset(self):
        """Reset all tracking (for recalibration)."""
        self.residuals.clear()
        self.predictions.clear()
        self.actuals.clear()
        self.total_updates = 0
        self.null_rejections = 0
        self.nan_rejections = 0
        logger.info("Conformal scaler reset")
