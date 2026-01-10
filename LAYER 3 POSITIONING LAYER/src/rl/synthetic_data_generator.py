"""
HIMARI Layer 3 - Synthetic Data Generator
==========================================

Generates synthetic price data for pre-training with:
- Merton Jump-Diffusion (MJD) for black swan events
- GARCH(1,1) for volatility clustering
- Regime switching (bull/bear/range)

Purpose: Pre-train RL agents on 500 stress scenarios before real data.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    initial_price: float = 50000.0
    n_steps: int = 10000
    dt: float = 1.0 / 365.0  # Daily timesteps

    # Merton Jump-Diffusion parameters
    mu: float = 0.10  # Drift (annualized)
    sigma: float = 0.60  # Volatility (annualized)
    lambda_jump: float = 5.0  # Jump frequency (per year)
    jump_mean: float = -0.05  # Average jump size (-5%)
    jump_std: float = 0.10  # Jump volatility (10%)

    # GARCH(1,1) parameters
    omega: float = 0.00001  # Constant term
    alpha: float = 0.10  # ARCH term
    beta: float = 0.85  # GARCH term

    # Regime parameters
    regime_durations: Dict[str, int] = None

    def __post_init__(self):
        if self.regime_durations is None:
            self.regime_durations = {
                'bull': 180,    # 180 days
                'bear': 90,     # 90 days
                'range': 60,    # 60 days
                'crash': 14     # 14 days (rare)
            }


class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion model for crypto prices.

    dS = μS dt + σS dW + J dN

    Where:
    - μ: drift
    - σ: diffusion volatility
    - J: jump size (log-normal)
    - N: Poisson process with intensity λ
    """

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate(self, S0: float, n_steps: int, dt: float) -> np.ndarray:
        """
        Generate price path with jumps.

        Args:
            S0: Initial price
            n_steps: Number of timesteps
            dt: Time increment (fraction of year)

        Returns:
            Price path array
        """
        mu = self.config.mu
        sigma = self.config.sigma
        lambda_jump = self.config.lambda_jump
        jump_mean = self.config.jump_mean
        jump_std = self.config.jump_std

        # Continuous component (Geometric Brownian Motion)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn(n_steps)

        # Jump component (Poisson process)
        n_jumps = np.random.poisson(lambda_jump * dt, n_steps)
        jumps = np.zeros(n_steps)

        for i in range(n_steps):
            if n_jumps[i] > 0:
                # Log-normal jump sizes
                jump_sizes = np.random.normal(jump_mean, jump_std, n_jumps[i])
                jumps[i] = np.sum(jump_sizes)

        # Combine components
        log_returns = drift + diffusion + jumps

        # Generate prices
        prices = np.zeros(n_steps + 1)
        prices[0] = S0

        for i in range(n_steps):
            prices[i+1] = prices[i] * np.exp(log_returns[i])

        return prices


class GARCH11:
    """
    GARCH(1,1) volatility model.

    σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

    Captures volatility clustering (high vol follows high vol).
    """

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate_volatility(self, n_steps: int) -> np.ndarray:
        """
        Generate time-varying volatility.

        Args:
            n_steps: Number of timesteps

        Returns:
            Volatility path array
        """
        omega = self.config.omega
        alpha = self.config.alpha
        beta = self.config.beta

        # Initialize
        sigma2 = np.zeros(n_steps)
        sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
        epsilon = np.random.randn(n_steps)

        # GARCH recursion
        for t in range(1, n_steps):
            sigma2[t] = omega + alpha * epsilon[t-1]**2 * sigma2[t-1] + beta * sigma2[t-1]

        return np.sqrt(sigma2)


class RegimeSwitching:
    """
    Regime-switching model for market states.

    Regimes:
    - BULL: High drift, low vol
    - BEAR: Negative drift, high vol
    - RANGE: Zero drift, medium vol
    - CRASH: Extreme negative drift, extreme vol
    """

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def generate_regimes(self, n_steps: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Generate regime sequence and regime-specific parameters.

        Args:
            n_steps: Number of timesteps

        Returns:
            (regime_labels, drift_array, vol_array)
        """
        regimes = []
        drifts = np.zeros(n_steps)
        vols = np.zeros(n_steps)

        # Regime parameters
        regime_params = {
            'bull': {'mu': 0.30, 'sigma': 0.40},
            'bear': {'mu': -0.20, 'sigma': 0.70},
            'range': {'mu': 0.0, 'sigma': 0.50},
            'crash': {'mu': -0.80, 'sigma': 1.50}
        }

        # Transition probabilities
        transitions = {
            'bull': {'bull': 0.85, 'bear': 0.05, 'range': 0.09, 'crash': 0.01},
            'bear': {'bull': 0.10, 'bear': 0.70, 'range': 0.15, 'crash': 0.05},
            'range': {'bull': 0.30, 'bear': 0.20, 'range': 0.48, 'crash': 0.02},
            'crash': {'bull': 0.05, 'bear': 0.50, 'range': 0.30, 'crash': 0.15}
        }

        # Start in bull regime
        current_regime = 'bull'

        for t in range(n_steps):
            # Store current regime
            regimes.append(current_regime)
            drifts[t] = regime_params[current_regime]['mu']
            vols[t] = regime_params[current_regime]['sigma']

            # Transition to next regime
            next_regimes = list(transitions[current_regime].keys())
            probs = list(transitions[current_regime].values())
            current_regime = np.random.choice(next_regimes, p=probs)

        return regimes, drifts, vols


class SyntheticDataGenerator:
    """
    Complete synthetic data generator combining MJD + GARCH + Regimes.
    """

    def __init__(self, config: SyntheticConfig = None, seed: int = None):
        if config is None:
            config = SyntheticConfig()

        self.config = config
        self.mjd = MertonJumpDiffusion(config)
        self.garch = GARCH11(config)
        self.regime = RegimeSwitching(config)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_scenario(
        self,
        scenario_type: str = 'mixed'
    ) -> Dict[str, np.ndarray]:
        """
        Generate single synthetic scenario.

        Args:
            scenario_type: Type of scenario
                - 'bull': Mostly bullish
                - 'bear': Mostly bearish
                - 'crash': Black swan event
                - 'mixed': Random regimes
                - 'volatility_cluster': High GARCH effect

        Returns:
            Dictionary with prices, returns, regime labels
        """
        n_steps = self.config.n_steps
        dt = self.config.dt

        # Generate regime-dependent parameters
        regimes, drifts, vols = self.regime.generate_regimes(n_steps)

        # Override for specific scenarios
        if scenario_type == 'bull':
            drifts[:] = 0.30
            vols[:] = 0.40
        elif scenario_type == 'bear':
            drifts[:] = -0.20
            vols[:] = 0.70
        elif scenario_type == 'crash':
            drifts[:] = -0.80
            vols[:] = 1.50
        elif scenario_type == 'volatility_cluster':
            # Use GARCH volatility
            vols = self.garch.generate_volatility(n_steps)

        # Generate prices with jumps
        prices = np.zeros(n_steps + 1)
        prices[0] = self.config.initial_price

        for i in range(n_steps):
            # Adjust config for this timestep
            self.config.mu = drifts[i]
            self.config.sigma = vols[i]

            # Generate single step with MJD
            step_prices = self.mjd.generate(prices[i], 1, dt)
            prices[i+1] = step_prices[1]

        # Calculate returns
        returns = np.diff(np.log(prices))

        # Compute features for each timestep
        features = self._compute_features(prices, returns, regimes)

        return {
            'prices': prices,
            'returns': returns,
            'regimes': regimes,
            'drifts': drifts,
            'vols': vols,
            'features': features,  # Added pre-computed features
            'timestamps': [
                datetime.now() + timedelta(days=i)
                for i in range(len(prices))
            ]
        }

    def _compute_features(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        regimes: List[str]
    ) -> np.ndarray:
        """
        Compute 32-dimensional features from price data.

        Features match bounded_delta_training.py expectations (32-dim).
        """
        n_steps = len(prices) - 1  # returns is 1 shorter
        features = np.zeros((n_steps, 32))

        for i in range(n_steps):
            # Get recent price history
            lookback = min(20, i + 1)
            recent_prices = prices[max(0, i - lookback + 1):i + 2]

            if len(recent_prices) < 2:
                continue

            recent_returns = np.diff(np.log(recent_prices))
            current_price = recent_prices[-1]

            # Calculate momentum
            price_momentum_1h = 0.0
            price_momentum_4h = 0.0
            price_momentum_24h = 0.0
            if len(recent_prices) >= 2:
                price_momentum_1h = ((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]) * 100
            if len(recent_prices) >= 5:
                price_momentum_4h = ((recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]) * 100
            if len(recent_prices) >= 10:
                price_momentum_24h = ((recent_prices[-1] - recent_prices[-10]) / recent_prices[-10]) * 100

            # Calculate volatility
            volatility = float(np.std(recent_returns) if len(recent_returns) > 1 else 0.01)

            # Map regime to integer
            regime_map = {
                'bull': 0, 'bear': 1, 'ranging': 2,
                'volatile': 3, 'crisis': 4, 'crash': 4,
                'recovery': 0
            }
            regime_int = regime_map.get(regimes[i] if i < len(regimes) else 'ranging', 2)

            # Build 32-dim feature vector (16 base + 16 extended)
            features[i, 0] = 0.7  # signal_confidence (placeholder)
            features[i, 1] = 1    # signal_action (HOLD)
            features[i, 2] = 2    # signal_tier (T2)
            features[i, 3] = regime_int
            features[i, 4] = 0.0  # position_size_usd (flat)
            features[i, 5] = 0    # position_side (flat)
            features[i, 6] = 0.0  # unrealized_pnl_pct
            features[i, 7] = price_momentum_1h
            features[i, 8] = price_momentum_4h
            features[i, 9] = volatility
            features[i, 10] = 0.5  # recent_win_rate (default)
            features[i, 11] = 0.0  # recent_sharpe
            features[i, 12] = 0.0  # total_pnl_pct
            features[i, 13] = min(volatility * 10, 1.0)  # cascade_risk
            features[i, 14] = 0.0  # current_drawdown
            features[i, 15] = float(i)  # timestamp

            # Extended features (16-31)
            features[i, 16] = price_momentum_24h
            features[i, 17] = returns[i] if i < len(returns) else 0.0  # Current return
            features[i, 18] = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else 0.0  # MA returns
            features[i, 19] = np.max(recent_returns) if len(recent_returns) > 0 else 0.0  # Max recent return
            features[i, 20] = np.min(recent_returns) if len(recent_returns) > 0 else 0.0  # Min recent return
            # Fill remaining with derived features
            for j in range(21, 32):
                features[i, j] = volatility * (j - 20) * 0.1  # Scaled volatility features

        return features

    def generate_stress_scenarios(
        self,
        n_scenarios: int = 500
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate multiple stress test scenarios.

        Args:
            n_scenarios: Number of scenarios to generate

        Returns:
            List of scenario dictionaries
        """
        logger.info(f"Generating {n_scenarios} stress scenarios...")

        scenarios = []

        # Scenario distribution
        scenario_mix = {
            'mixed': int(n_scenarios * 0.40),      # 40% mixed regimes
            'crash': int(n_scenarios * 0.20),      # 20% crashes
            'bear': int(n_scenarios * 0.20),       # 20% bear markets
            'volatility_cluster': int(n_scenarios * 0.10),  # 10% vol clusters
            'bull': int(n_scenarios * 0.10),       # 10% bull markets
        }

        for scenario_type, count in scenario_mix.items():
            for i in range(count):
                scenario = self.generate_scenario(scenario_type)
                scenario['type'] = scenario_type
                scenario['id'] = f"{scenario_type}_{i}"
                scenarios.append(scenario)

                if (len(scenarios) % 50) == 0:
                    logger.info(f"Generated {len(scenarios)}/{n_scenarios} scenarios")

        logger.info(f"Stress scenario generation complete: {len(scenarios)} scenarios")
        return scenarios

    def save_scenarios(
        self,
        scenarios: List[Dict],
        output_dir: str = "/tmp/synthetic_data"
    ):
        """Save scenarios to disk."""
        import os
        import pickle

        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, 'stress_scenarios.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(scenarios, f)

        logger.info(f"Saved {len(scenarios)} scenarios to {filepath}")

        # Save summary
        summary_path = os.path.join(output_dir, 'scenarios_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Synthetic Stress Scenarios\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Total scenarios: {len(scenarios)}\n\n")

            scenario_types = {}
            for s in scenarios:
                stype = s['type']
                scenario_types[stype] = scenario_types.get(stype, 0) + 1

            for stype, count in scenario_types.items():
                pct = 100 * count / len(scenarios)
                f.write(f"{stype:20s}: {count:4d} ({pct:5.1f}%)\n")

        logger.info(f"Saved summary to {summary_path}")


def main():
    """Generate and save stress scenarios."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic stress scenarios')
    parser.add_argument('--scenarios', type=int, default=500, help='Number of scenarios to generate')
    parser.add_argument('--output', type=str, default='/tmp/synthetic_data/stress_scenarios.pkl', 
                        help='Output file path')
    parser.add_argument('--steps', type=int, default=10000, help='Steps per scenario')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)

    config = SyntheticConfig(
        n_steps=args.steps,  # ~27 years of daily data
        initial_price=50000.0
    )

    generator = SyntheticDataGenerator(config)
    scenarios = generator.generate_stress_scenarios(n_scenarios=args.scenarios)
    
    # Extract output directory from path
    import os
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to specified path
    import pickle
    with open(args.output, 'wb') as f:
        pickle.dump(scenarios, f)
    logger.info(f"Saved {len(scenarios)} scenarios to {args.output}")

    total_steps = args.scenarios * args.steps
    print("\n" + "="*60)
    print("Synthetic Data Generation Complete!")
    print("="*60)
    print(f"Generated: {args.scenarios} scenarios")
    print(f"Steps per scenario: {args.steps:,}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Output: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
