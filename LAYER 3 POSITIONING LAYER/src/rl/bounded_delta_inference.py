"""
HIMARI Layer 3: Bounded Delta Inference Wrapper
=================================================

Key Insight: Apply safety bounds at INFERENCE time, not during training.

This wrapper:
1. Takes the original model (trained with simple rewards)
2. Applies bounded delta constraints at runtime
3. Integrates with Layer 3 Tier 1 volatility targeting

Usage:
    wrapper = BoundedDeltaInference(model_path, device="cuda")
    final_position = wrapper.get_position(features, volatility, equity)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference-time bounding."""
    
    # Delta bounds (Layer 3 Tier 2 specification)
    delta_lower: float = -0.30
    delta_upper: float = +0.30
    
    # Tier 1 volatility targeting
    target_vol_annual: float = 0.15
    base_fraction: float = 0.5  # Half-Kelly
    min_position_pct: float = 0.01
    max_position_pct: float = 0.10
    
    # Regime adjustments (applied AFTER model prediction)
    regime_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                "bull": 1.0,          # No adjustment
                "bear": 0.8,          # Reduce 20%
                "ranging": 1.0,       # No adjustment
                "crisis": 0.5,        # Reduce 50%
                "crash": 0.3,         # Reduce 70%
                "volatility_cluster": 0.7,  # Reduce 30%
                "cascade": 0.3,       # Reduce 70%
            }


class Tier1VolatilityTargeter:
    """
    Tier 1: Deterministic volatility-targeted base position.
    
    This is the FOUNDATION that the Tier 2 delta adjusts.
    Pure arithmetic, no neural networks.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._vol_buffer = []
        self._lookback = 20
    
    def compute_base_position(
        self,
        realized_vol: float,
        portfolio_equity: float
    ) -> Tuple[float, Dict]:
        """
        Compute base position using volatility targeting.
        
        Formula: position_pct = (target_vol / realized_vol) * base_fraction
        Clamped to [min_position_pct, max_position_pct]
        """
        # Update rolling volatility
        self._vol_buffer.append(realized_vol)
        if len(self._vol_buffer) > self._lookback:
            self._vol_buffer.pop(0)
        
        # Blended volatility (recent-weighted)
        if len(self._vol_buffer) >= 5:
            short_vol = np.mean(self._vol_buffer[-5:])
            long_vol = np.mean(self._vol_buffer)
            blended_vol = 0.7 * short_vol + 0.3 * long_vol
        else:
            blended_vol = realized_vol
        
        blended_vol = max(blended_vol, 0.001)  # Prevent div by zero
        
        # Volatility targeting formula
        raw_position = (self.config.target_vol_annual / blended_vol) * self.config.base_fraction
        
        # Clamp to bounds
        base_position = np.clip(
            raw_position,
            self.config.min_position_pct,
            self.config.max_position_pct
        )
        
        diagnostics = {
            "realized_vol": realized_vol,
            "blended_vol": blended_vol,
            "raw_position_pct": raw_position,
            "base_position_pct": base_position
        }
        
        return base_position, diagnostics
    
    def reset(self):
        self._vol_buffer = []


class BoundedDeltaInference:
    """
    Apply bounded delta to trained model at inference time.
    
    The model outputs raw position preferences.
    We convert to delta format and apply safety bounds.
    """
    
    def __init__(
        self,
        model_path: str,
        config: InferenceConfig = None,
        device: str = "cuda"
    ):
        self.config = config or InferenceConfig()
        self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Disable dropout for inference
        
        # Tier 1 base position calculator
        self.tier1 = Tier1VolatilityTargeter(self.config)
        
        # LSTM hidden state
        self.hidden = None
        
        logger.info(f"✅ Loaded model from {model_path}")
        logger.info(f"Delta bounds: [{self.config.delta_lower}, {self.config.delta_upper}]")
    
    def _load_model(self, path: str) -> nn.Module:
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct network (matches pretrain_pipeline architecture)
        from pretrain_pipeline_v2 import LSTMPPONetworkV2, TrainingConfig
        
        config = checkpoint.get("config", TrainingConfig())
        model = LSTMPPONetworkV2(config).to(self.device)
        model.load_state_dict(checkpoint["network_state_dict"])
        
        return model
    
    def get_position(
        self,
        features: np.ndarray,
        volatility: float,
        portfolio_equity: float,
        regime: str = "ranging"
    ) -> Tuple[float, Dict]:
        """
        Get final position with bounded delta applied.
        
        Args:
            features: Market features (state vector)
            volatility: Current realized volatility
            portfolio_equity: Current portfolio value
            regime: Market regime classification
        
        Returns:
            final_position_pct: Position as % of equity (with bounds applied)
            diagnostics: Dict with all intermediate values
        """
        
        # Step 1: Tier 1 base position (deterministic)
        base_position, tier1_diag = self.tier1.compute_base_position(
            volatility, portfolio_equity
        )
        
        # Step 2: Model prediction (raw)
        with torch.no_grad():
            state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            mean, std, value, self.hidden = self.model(state_t, self.hidden)
            raw_prediction = mean.cpu().numpy()[0, 0]
        
        # Step 3: Convert to delta
        # Model outputs position preference, convert to delta from base
        model_position = np.tanh(raw_prediction) * self.config.max_position_pct
        raw_delta = (model_position / base_position) - 1.0 if base_position > 0 else 0.0
        
        # Step 4: Apply bounds (KEY STEP - safety at runtime)
        bounded_delta = np.clip(
            raw_delta,
            self.config.delta_lower,
            self.config.delta_upper
        )
        
        # Step 5: Apply regime multiplier (additional safety)
        regime_mult = self.config.regime_multipliers.get(regime, 1.0)
        
        # Step 6: Calculate final position
        final_position = base_position * (1.0 + bounded_delta) * regime_mult
        
        # Clamp final position to absolute bounds
        final_position = np.clip(
            final_position,
            self.config.min_position_pct,
            self.config.max_position_pct
        )
        
        diagnostics = {
            **tier1_diag,
            "raw_prediction": raw_prediction,
            "model_position": model_position,
            "raw_delta": raw_delta,
            "bounded_delta": bounded_delta,
            "regime": regime,
            "regime_multiplier": regime_mult,
            "final_position_pct": final_position
        }
        
        return final_position, diagnostics
    
    def reset(self):
        """Reset for new trading session."""
        self.tier1.reset()
        self.hidden = None


# ============================================================================
# ENSEMBLE INFERENCE
# ============================================================================

class EnsembleInference:
    """
    Ensemble of multiple models for more stable predictions.
    
    Average predictions from 5 models trained with different seeds.
    """
    
    def __init__(
        self,
        model_paths: list,
        config: InferenceConfig = None,
        device: str = "cuda"
    ):
        self.config = config or InferenceConfig()
        self.device = device
        
        # Load all models
        self.models = []
        for path in model_paths:
            wrapper = BoundedDeltaInference(path, config, device)
            self.models.append(wrapper)
        
        logger.info(f"✅ Loaded ensemble of {len(self.models)} models")
    
    def get_position(
        self,
        features: np.ndarray,
        volatility: float,
        portfolio_equity: float,
        regime: str = "ranging"
    ) -> Tuple[float, Dict]:
        """
        Get ensemble position (average of all models).
        """
        positions = []
        all_diagnostics = []
        
        for model in self.models:
            pos, diag = model.get_position(
                features, volatility, portfolio_equity, regime
            )
            positions.append(pos)
            all_diagnostics.append(diag)
        
        # Average position
        ensemble_position = np.mean(positions)
        
        diagnostics = {
            "individual_positions": positions,
            "ensemble_position": ensemble_position,
            "position_std": np.std(positions),
            "regime": regime
        }
        
        return ensemble_position, diagnostics
    
    def reset(self):
        for model in self.models:
            model.reset()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the inference wrapper."""
    
    # Single model
    wrapper = BoundedDeltaInference(
        model_path="/tmp/models/pretrained_v2/best_model.pt",
        device="cuda"
    )
    
    # Simulated market state
    features = np.random.randn(16)  # 16-dim state
    volatility = 0.25  # 25% annualized vol
    equity = 100000.0  # $100k portfolio
    regime = "bull"
    
    # Get bounded position
    position, diagnostics = wrapper.get_position(
        features=features,
        volatility=volatility,
        portfolio_equity=equity,
        regime=regime
    )
    
    print(f"Final position: {position:.2%}")
    print(f"Base position: {diagnostics['base_position_pct']:.2%}")
    print(f"Bounded delta: {diagnostics['bounded_delta']:.2%}")
    print(f"Regime multiplier: {diagnostics['regime_multiplier']:.2f}")


if __name__ == "__main__":
    example_usage()
