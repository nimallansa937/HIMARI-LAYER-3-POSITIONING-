"""
HIMARI Layer 3: Kelly Criterion Baseline
=========================================

Tests if position sizing is too simple for RL.
If Kelly Sharpe > 0.025, use deterministic Kelly + volatility targeting instead of RL.

This is the sanity check before investing time in CQL/RevIN.
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class KellyConfig:
    """Kelly Criterion configuration."""
    fractional_kelly: float = 0.5  # Half-Kelly for safety
    target_vol: float = 0.15       # 15% target annual vol
    max_position: float = 0.5      # Max 50% of capital
    min_position: float = 0.01     # Min 1% of capital
    lookback: int = 20             # Rolling window for stats


class KellyPositionSizer:
    """
    Deterministic Kelly Criterion position sizing.
    
    Formula: f* = (p*W - (1-p)*L) / W
    Where:
        p = win probability
        W = average win
        L = average loss
    """
    
    def __init__(self, config: KellyConfig = None):
        self.config = config or KellyConfig()
        self.returns_buffer = []
    
    def compute_kelly_fraction(self, returns: np.ndarray) -> float:
        """Compute optimal Kelly fraction from recent returns."""
        if len(returns) < 5:
            return 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_prob = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_win == 0:
            return 0.0
        
        # Kelly formula
        f_star = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        
        # Apply fractional Kelly for safety
        f_kelly = f_star * self.config.fractional_kelly
        
        return np.clip(f_kelly, -self.config.max_position, self.config.max_position)
    
    def compute_volatility_position(self, realized_vol: float) -> float:
        """Volatility targeting position."""
        if realized_vol < 0.001:
            return self.config.max_position
        
        position = self.config.target_vol / realized_vol
        return np.clip(position, self.config.min_position, self.config.max_position)
    
    def get_position(self, recent_returns: np.ndarray, realized_vol: float) -> Tuple[float, Dict]:
        """
        Get position size using Kelly + Volatility targeting.
        
        Returns:
            position: Position size as fraction of capital
            diagnostics: Dict with intermediate values
        """
        # Kelly fraction
        kelly_f = self.compute_kelly_fraction(recent_returns)
        
        # Volatility target
        vol_position = self.compute_volatility_position(realized_vol)
        
        # Blend: use smaller of the two for safety
        final_position = min(abs(kelly_f), vol_position)
        
        # Apply sign from Kelly (direction)
        if kelly_f < 0:
            final_position = -final_position
        
        diagnostics = {
            "kelly_fraction": kelly_f,
            "vol_position": vol_position,
            "final_position": final_position,
            "realized_vol": realized_vol
        }
        
        return final_position, diagnostics


def backtest_kelly(scenarios: List[Dict], config: KellyConfig = None) -> Dict:
    """
    Backtest Kelly Criterion on synthetic scenarios.
    
    Args:
        scenarios: List of scenario dicts with 'returns' key
        config: Kelly configuration
    
    Returns:
        Dict with Sharpe ratio and other metrics
    """
    if config is None:
        config = KellyConfig()
    
    sizer = KellyPositionSizer(config)
    all_returns = []
    
    for scenario in scenarios:
        returns = scenario.get('returns', np.random.randn(1000) * 0.02)
        
        for t in range(config.lookback, len(returns) - 1):
            # Recent returns for Kelly calculation
            recent = returns[t - config.lookback:t]
            
            # Rolling volatility
            realized_vol = np.std(recent) * np.sqrt(252)
            
            # Get position
            position, _ = sizer.get_position(recent, realized_vol)
            
            # Strategy return
            market_return = returns[t]
            strategy_return = position * market_return
            all_returns.append(strategy_return)
    
    # Calculate metrics
    all_returns = np.array(all_returns)
    
    if len(all_returns) < 2:
        return {"sharpe": 0.0, "total_return": 0.0}
    
    sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8)
    total_return = np.prod(1 + all_returns) - 1
    max_dd = np.max(np.maximum.accumulate(np.cumprod(1 + all_returns)) - np.cumprod(1 + all_returns))
    
    return {
        "sharpe": float(sharpe),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "num_trades": len(all_returns),
        "avg_position": float(np.mean(np.abs(all_returns) / (np.abs(all_returns / np.array([r for r in all_returns if r != 0]) if any(all_returns != 0) else [1]))))
    }


def generate_test_scenarios(n: int = 100, seed: int = 42) -> List[Dict]:
    """Generate diverse test scenarios."""
    np.random.seed(seed)
    scenarios = []
    
    regime_params = {
        "bull": (0.0003, 0.015),
        "bear": (-0.0002, 0.020),
        "mixed": (0.0, 0.025),
        "crash": (-0.002, 0.050),
        "volatility_cluster": (0.0, 0.040)
    }
    
    for i in range(n):
        regime = np.random.choice(list(regime_params.keys()))
        drift, vol = regime_params[regime]
        length = np.random.randint(500, 1500)
        
        returns = np.random.normal(drift, vol, length)
        scenarios.append({"type": regime, "returns": returns})
    
    return scenarios


def main():
    print("=" * 60)
    print("HIMARI Layer 3: Kelly Criterion Baseline Test")
    print("=" * 60)
    print("\nThis tests if position sizing is too simple for RL.")
    print("If Kelly Sharpe > 0.025, use deterministic Kelly instead of RL.\n")
    
    # Generate test scenarios
    print("Generating 100 test scenarios...")
    scenarios = generate_test_scenarios(100)
    
    # Test different Kelly configurations
    configs = [
        ("Full Kelly", KellyConfig(fractional_kelly=1.0)),
        ("Half Kelly (0.5)", KellyConfig(fractional_kelly=0.5)),
        ("Quarter Kelly (0.25)", KellyConfig(fractional_kelly=0.25)),
        ("Conservative (0.1)", KellyConfig(fractional_kelly=0.1)),
    ]
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = []
    for name, config in configs:
        result = backtest_kelly(scenarios, config)
        results.append((name, result))
        
        print(f"\n{name}:")
        print(f"  Sharpe Ratio: {result['sharpe']:.4f}")
        print(f"  Total Return: {result['total_return']*100:.2f}%")
        print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    
    # Find best Kelly
    best_name, best_result = max(results, key=lambda x: x[1]['sharpe'])
    
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)
    
    if best_result['sharpe'] > 0.025:
        print(f"\n✅ Kelly Sharpe ({best_result['sharpe']:.4f}) > 0.025")
        print(f"\n→ RECOMMENDATION: Use {best_name} + Volatility Targeting")
        print("   Position sizing is TOO SIMPLE for RL.")
        print("   RL models will overfit to synthetic patterns.")
        print("\n   STOP RL experiments. Use deterministic Kelly baseline.")
    else:
        print(f"\n⚠️ Kelly Sharpe ({best_result['sharpe']:.4f}) < 0.025")
        print("\n→ RECOMMENDATION: Proceed with CQL + RevIN")
        print("   Position sizing is complex enough for RL.")
        print("   Implement Conservative Q-Learning to avoid extrapolation errors.")
    
    print("\n" + "=" * 60)
    print("COMPARISON TO RL MODELS")
    print("=" * 60)
    print(f"  Original 500K PPO:    +0.046 (training)")
    print(f"  Bounded Delta:        -0.037 ❌")
    print(f"  Dropout Ensemble:     +0.003 (test)")
    print(f"  Kelly Criterion:      {best_result['sharpe']:+.4f} {'✅' if best_result['sharpe'] > 0.003 else '⚠️'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
