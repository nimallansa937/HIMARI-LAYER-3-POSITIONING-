"""
HIMARI Layer 3 - Monte Carlo Validation Framework
===================================================

Implements comprehensive statistical validation based on:
- joel-saucedo/Crypto-Strategy-Lab (DSR, PSR, bootstrap, reality check)

Validation Methods:
1. Deflated Sharpe Ratio (DSR)
2. Probabilistic Sharpe Ratio (PSR)
3. Bootstrap Confidence Intervals
4. White's Reality Check (Bootstrap p-value)
5. Permutation Tests
6. Regime Analysis

Version: 1.0
"""

import numpy as np
from scipy import stats
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Numba fallback
try:
    from numba import jit, prange
except ImportError:
    # Dummy decorator if numba is missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    # Dummy prange
    def prange(n):
        return range(n)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for Monte Carlo validation."""
    # DSR/PSR thresholds
    min_dsr: float = 0.95
    min_psr: float = 0.95
    
    # Monte Carlo parameters
    n_trials: int = 10000
    n_bootstrap: int = 10000
    
    # Bootstrap parameters
    block_size_factor: float = 0.33  # T^(1/3) rule
    
    # Significance levels
    alpha: float = 0.05


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

@jit(nopython=True)
def calculate_sharpe_fast(returns: np.ndarray) -> float:
    """Fast Sharpe ratio calculation."""
    if len(returns) == 0:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret < 1e-9:
        return 0.0
    
    return mean_ret / std_ret * np.sqrt(252.0)


@jit(nopython=True)
def calculate_max_drawdown_fast(returns: np.ndarray) -> float:
    """Fast maximum drawdown calculation."""
    if len(returns) == 0:
        return 0.0
    
    # Build equity curve
    equity = np.zeros(len(returns) + 1)
    equity[0] = 1.0
    
    for i in range(len(returns)):
        equity[i + 1] = equity[i] * (1.0 + returns[i])
    
    # Calculate drawdown
    running_max = equity[0]
    max_dd = 0.0
    
    for i in range(1, len(equity)):
        if equity[i] > running_max:
            running_max = equity[i]
        
        dd = (running_max - equity[i]) / running_max
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


@jit(nopython=True)
def bootstrap_sample(returns: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate bootstrap sample using random replacement."""
    indices = np.random.randint(0, len(returns), n_samples)
    # Manual indexing for numba compatibility in older versions if needed, 
    # but returns[indices] usually works.
    out = np.empty(n_samples, dtype=returns.dtype)
    for i in range(n_samples):
        out[i] = returns[indices[i]]
    return out


@jit(nopython=True)
def block_bootstrap_sample(
    returns: np.ndarray,
    block_size: int,
    n_samples: int
) -> np.ndarray:
    """Generate block bootstrap sample to preserve autocorrelation."""
    n_obs = len(returns)
    n_blocks = (n_samples + block_size - 1) // block_size
    
    result = np.zeros(n_samples)
    pos = 0
    
    for _ in range(n_blocks):
        if pos >= n_samples:
            break
        
        # Random starting position
        start = np.random.randint(0, max(1, n_obs - block_size + 1))
        
        # Copy block
        for j in range(block_size):
            if pos >= n_samples:
                break
            result[pos] = returns[(start + j) % n_obs]
            pos += 1
    
    return result


@jit(nopython=True, parallel=True)
def bootstrap_sharpe_distribution(
    returns: np.ndarray,
    n_bootstrap: int
) -> np.ndarray:
    """Generate bootstrap distribution of Sharpe ratios."""
    sharpes = np.zeros(n_bootstrap)
    n_obs = len(returns)
    
    for i in prange(n_bootstrap):
        sample = bootstrap_sample(returns, n_obs)
        sharpes[i] = calculate_sharpe_fast(sample)
    
    return sharpes


@jit(nopython=True, parallel=True)
def permutation_test_distribution(
    returns: np.ndarray,
    n_trials: int
) -> np.ndarray:
    """
    Generate permutation distribution of Sharpe ratios.
    Shuffling destroys serial correlation and effectively tests if mean != 0 
    assuming i.i.d under null.
    """
    sharpes = np.zeros(n_trials)
    n_obs = len(returns)
    
    # Pre-allocate array for shuffling key
    scratch = np.copy(returns)
    
    for i in prange(n_trials):
        # Fisher-Yates shuffle equivalent via numpy
        # Note: np.random.shuffle in numba parallel can be tricky, 
        # usually assume thread-safety or use local state.
        # For simplicity, we just random sample without replacement (permutation)
        # Or just random permutation logic:
        # np.random.shuffle(scratch) <-- this modifies inplace, tricky in parallel.
        # So we use random choice without replacement strategy or just shuffle logic.
        # Simpler for numba parallel: random generator per thread.
        # Let's simply generate random noise with same mean/std? 
        # No, permutation must use exact data. 
        # As safe bet: standard bootstrap is robust enough, but let's try simple shuffle.
        
        # Using a simple swap shuffle for 'scratch' copy might not work in parallel loop 
        # if 'scratch' is shared. We need local copies.
        local_sample = np.random.permutation(returns)
        sharpes[i] = calculate_sharpe_fast(local_sample)
        
    return sharpes


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class MonteCarloValidator:
    """
    Comprehensive Monte Carlo validation framework.
    
    Based on: joel-saucedo/Crypto-Strategy-Lab MonteCarloValidator
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        
    def validate_strategy(
        self,
        returns: np.ndarray,
        n_strategies_tested: int = 1
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation on strategy returns.
        
        Args:
            returns: Array of strategy returns
            n_strategies_tested: Number of strategies tested (for DSR)
            
        Returns: 
            Dictionary with all validation results
        """
        if len(returns) < 30:
            return {
                'validation_passed': False,
                'error': 'Insufficient data (need ≥30 observations)'
            }
        
        returns = np.array(returns, dtype=np.float64)
        
        results = {
            'basic_metrics': self._calculate_basic_metrics(returns),
            'dsr_analysis': self._calculate_dsr(returns, n_strategies_tested),
            'psr_analysis': self._calculate_psr(returns),
            'bootstrap_analysis': self._bootstrap_validation(returns),
            'permutation_test': self._permutation_test(returns),
            'reality_check': self._white_reality_check(returns),
            'regime_analysis': self._regime_analysis(returns)
        }
        
        # Overall validation
        results['validation_passed'] = self._evaluate_validation(results)
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _calculate_basic_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        sharpe = calculate_sharpe_fast(returns)
        max_dd = calculate_max_drawdown_fast(returns)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            sortino = np.mean(returns) * 252 / downside_std if downside_std > 0 else 0
        else: 
            sortino = float('inf')
        
        # Calmar ratio
        annual_return = np.mean(returns) * 252
        calmar = annual_return / max_dd if max_dd > 0 else float('inf')
        
        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'total_return': float(np.prod(1 + returns) - 1),
            'annual_return': float(annual_return),
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'max_drawdown': float(max_dd),
            'win_rate': float(np.mean(returns > 0)),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns))
        }
    
    def _calculate_dsr(
        self,
        returns: np.ndarray,
        n_strategies_tested: int
    ) -> Dict[str, float]: 
        """
        Calculate Deflated Sharpe Ratio. 
        
        DSR accounts for multiple testing bias by adjusting for the expected
        maximum Sharpe ratio under the null hypothesis.
        """
        observed_sharpe = calculate_sharpe_fast(returns)
        n_obs = len(returns)
        
        # Expected maximum Sharpe under null (asymptotic approximation)
        if n_strategies_tested > 100:
            # Use asymptotic formula
            # gamma = 0.5772156649  # Euler-Mascheroni constant
            expected_max_sr = (
                np.sqrt(2 * np.log(n_strategies_tested)) -
                (np.log(np.log(n_strategies_tested)) + np.log(4 * np.pi)) /
                (2 * np.sqrt(2 * np.log(n_strategies_tested)))
            )
        else:
            # Simulate for small n
            expected_max_sr = self._simulate_expected_max_sharpe(
                n_strategies_tested, n_obs
            )
        
        # Standard deviation of max Sharpe approximation
        # Ideally this assumes returns are normal 
        std_max_sr = 1 / np.sqrt(2 * np.log(max(n_strategies_tested, 2)))
        
        # Adjusted Sharpe (Haircut)
        # Note: DSR formula actually checks probability that SR > Expected Max
        sr_diff = observed_sharpe - expected_max_sr
        
        # DSR statistic (simplified Z-score approach)
        dsr_stat = sr_diff / std_max_sr
        
        # Convert to probability
        dsr_probability = float(stats.norm.cdf(dsr_stat))
        
        return {
            'dsr': float(dsr_stat),
            'dsr_probability': dsr_probability,
            'observed_sharpe': float(observed_sharpe),
            'expected_max_sharpe': float(expected_max_sr),
            'std_max_sharpe': float(std_max_sr),
            'n_strategies_tested': n_strategies_tested,
            'passes_dsr': dsr_probability >= self.config.min_dsr
        }
    
    def _simulate_expected_max_sharpe(
        self,
        n_strategies: int,
        n_obs: int,
        n_simulations: int = 100
    ) -> float:
        """Simulate expected maximum Sharpe for small n_strategies."""
        max_sharpes = []
        
        for _ in range(n_simulations):
            strategy_sharpes = []
            for _ in range(n_strategies):
                # Generate random returns with mean=0, std=1/sqrt(252) => Daily standard noise
                random_returns = np.random.normal(0, 1, n_obs)
                # Sharpe of noise ~ 0
                sharpe = calculate_sharpe_fast(random_returns)
                strategy_sharpes.append(sharpe)
            max_sharpes.append(max(strategy_sharpes))
        
        return float(np.mean(max_sharpes))
    
    def _calculate_psr(
        self,
        returns: np.ndarray,
        benchmark_sharpe: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        PSR = P[SR* > benchmark] accounting for estimation error.
        """
        observed_sharpe = calculate_sharpe_fast(returns)
        n_obs = len(returns)
        
        # Higher moments for correction
        skew = stats.skew(returns) if len(returns) > 2 else 0
        kurt = stats.kurtosis(returns, fisher=True) if len(returns) > 2 else 0
        
        # Standard error with higher moment corrections (Opdyke 2007)
        # Variance of SR
        sr_variance = (
            1 + 0.5 * observed_sharpe**2 -
            skew * observed_sharpe +
            (kurt / 4) * observed_sharpe**2
        ) / max(1, n_obs - 1)
        
        sr_std_error = np.sqrt(max(sr_variance, 1e-8))
        
        # PSR calculation
        if sr_std_error > 0:
            psr_statistic = (observed_sharpe - benchmark_sharpe) / sr_std_error
            psr = float(stats.norm.cdf(psr_statistic))
        else:
            psr = 1.0 if observed_sharpe > benchmark_sharpe else 0.0
            psr_statistic = float('inf') if observed_sharpe > benchmark_sharpe else float('-inf')
        
        return {
            'psr': psr,
            'psr_statistic': float(psr_statistic),
            'observed_sharpe': float(observed_sharpe),
            'sharpe_std_error': float(sr_std_error),
            'benchmark_sharpe': benchmark_sharpe,
            'passes_psr': psr >= self.config.min_psr
        }
    
    def _bootstrap_validation(self, returns: np.ndarray) -> Dict[str, Any]: 
        """
        Bootstrap confidence intervals for Sharpe ratio.
        """
        n_obs = len(returns)
        n_bootstrap = self.config.n_bootstrap
        
        # Standard bootstrap
        standard_sharpes = bootstrap_sharpe_distribution(returns, n_bootstrap)
        
        # Block bootstrap (preserves autocorrelation)
        block_size = max(1, int(n_obs ** self.config.block_size_factor))
        block_sharpes = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = block_bootstrap_sample(returns, block_size, n_obs)
            block_sharpes[i] = calculate_sharpe_fast(sample)
        
        # Calculate confidence intervals
        ci_lower = float(np.percentile(standard_sharpes, self.config.alpha / 2 * 100))
        ci_upper = float(np.percentile(standard_sharpes, (1 - self.config.alpha / 2) * 100))
        
        block_ci_lower = float(np.percentile(block_sharpes, self.config.alpha / 2 * 100))
        block_ci_upper = float(np.percentile(block_sharpes, (1 - self.config.alpha / 2) * 100))
        
        original_sharpe = float(calculate_sharpe_fast(returns))
        
        return {
            'mean_bootstrap_sharpe': float(np.mean(standard_sharpes)),
            'std_bootstrap_sharpe': float(np.std(standard_sharpes)),
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
            'block_ci_lower_95': block_ci_lower,
            'block_ci_upper_95': block_ci_upper,
            'probability_loss': float(np.mean(standard_sharpes < 0)),
            'passes_significance': ci_lower > 0,
            'bias': float(np.mean(standard_sharpes) - original_sharpe)
        }
        
    def _permutation_test(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Perform Permutation Test (Null Hypothesis: SR <= 0).
        Shuffling returns destroys the signal. We check if original SR is significantly
        better than random shuffles.
        """
        n_trials = self.config.n_trials
        original_sharpe = calculate_sharpe_fast(returns)
        
        if self.config.n_trials < 100:
            return {'error': 'Low trials'}
            
        permuted_sharpes = permutation_test_distribution(returns, n_trials)
        
        # P-value: Fraction of permuted trials that beat original result
        p_value = np.mean(permuted_sharpes >= original_sharpe)
        
        return {
            'original_sharpe': float(original_sharpe),
            'permuted_mean': float(np.mean(permuted_sharpes)),
            'p_value': float(p_value),
            'significant': p_value < self.config.alpha
        }

    def _white_reality_check(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Simplified "Reality Check".
        Tests if the strategy mean return is significantly > 0 using bootstrap p-value.
        Strictly speaking WRC compares against a universe, but here we test the single strategy
        against the Null of 0 mean.
        """
        n_bootstrap = self.config.n_bootstrap
        mean_ret = np.mean(returns)
        
        # Offset returns to have mean 0 (Enforce Null Hypothesis)
        null_returns = returns - mean_ret
        
        # Bootstrap under null
        bootstrap_means = np.zeros(n_bootstrap)
        n_obs = len(returns)
        
        # We can re-use block bootstrap or simple bootstrap
        # Simple:
        for i in range(n_bootstrap):
            sample = bootstrap_sample(null_returns, n_obs)
            bootstrap_means[i] = np.mean(sample)
            
        # P-value: Probability that null distribution generates a mean >= observed mean
        p_value = np.mean(bootstrap_means >= mean_ret)
        
        return {
            'observed_mean': float(mean_ret),
            'bootstrap_p_value': float(p_value),
            'significant': p_value < self.config.alpha
        }

    def _regime_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Analyze performance across different volatility regimes.
        Divides data into Low, Medium, High volatility buckets.
        """
        if len(returns) < 100:
            return {'note': 'Insufficient data for regime split'}
            
        # Calculate rolling volatility
        vol_window = 20
        rolling_std = np.zeros(len(returns))
        for i in range(vol_window, len(returns)):
            rolling_std[i] = np.std(returns[i-vol_window:i])
            
        # Skip warm-up
        valid_returns = returns[vol_window:]
        valid_vol = rolling_std[vol_window:]
        
        # Define quartiles
        low_thresh = np.percentile(valid_vol, 33)
        high_thresh = np.percentile(valid_vol, 66)
        
        low_vol_rets = valid_returns[valid_vol <= low_thresh]
        med_vol_rets = valid_returns[(valid_vol > low_thresh) & (valid_vol <= high_thresh)]
        high_vol_rets = valid_returns[valid_vol > high_thresh]
        
        def safe_sharpe(r):
            if len(r) < 5: return 0.0
            return calculate_sharpe_fast(r)
            
        return {
            'low_vol_sharpe': float(safe_sharpe(low_vol_rets)),
            'med_vol_sharpe': float(safe_sharpe(med_vol_rets)),
            'high_vol_sharpe': float(safe_sharpe(high_vol_rets)),
            'low_vol_count': len(low_vol_rets),
            'high_vol_count': len(high_vol_rets)
        }

    def _evaluate_validation(self, results: Dict[str, Any]) -> bool:
        """Determines if the strategy PASSES validation."""
        metrics = results['basic_metrics']
        
        # Hard Requirements
        dsr_pass = results['dsr_analysis'].get('passes_dsr', False)
        psr_pass = results['psr_analysis'].get('passes_psr', False)
        
        # Soft Requirements
        positive_sharpe = metrics['sharpe_ratio'] > 0.5
        significant_bootstrap = results['bootstrap_analysis'].get('passes_significance', False)
        
        # Final Decision
        # Just requiring PSR/DSR pass might be strict depending on config
        # Default config requires 95% confidence
        return bool(dsr_pass and psr_pass and positive_sharpe and significant_bootstrap)

    def _create_summary(self, results: Dict[str, Any]) -> str:
        """Create human-readable summary string."""
        m = results['basic_metrics']
        
        dsr_res = results['dsr_analysis']
        psr_res = results['psr_analysis']
        boot = results['bootstrap_analysis']
        
        passed = "✅ PASSED" if results['validation_passed'] else "❌ FAILED"
        
        summary = [
            f"Validation Status: {passed}",
            f"=" * 40,
            f"Sharpe Ratio:     {m['sharpe_ratio']:.4f}",
            f"Sortino Ratio:    {m['sortino_ratio']:.4f}",
            f"Max Drawdown:     {m['max_drawdown']:.2%}",
            f"-" * 40,
            f"DSR (Deflated SR): {dsr_res['dsr_probability']:.4f} (Threshold: {self.config.min_dsr}) -> {'PASS' if dsr_res['passes_dsr'] else 'FAIL'}",
            f"PSR (Prob. SR):    {psr_res['psr']:.4f} (Threshold: {self.config.min_psr}) -> {'PASS' if psr_res['passes_psr'] else 'FAIL'}",
            f"-" * 40,
            f"95% CI (Bootstrap): [{boot['ci_lower_95']:.4f}, {boot['ci_upper_95']:.4f}]",
            f"Block Bootstrap CI: [{boot['block_ci_lower_95']:.4f}, {boot['block_ci_upper_95']:.4f}]"
        ]
        
        return "\n".join(summary)


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Monte Carlo Validation Framework Test")
    print("=" * 70)
    
    np.random.seed(42)
    validator = MonteCarloValidator()
    
    # 1. Generate profitable synthetic strategy
    print("\nGenerating Synthetic Profitable Strategy...")
    # Mean daily return 0.05%, daily vol 1% -> Annual Sharpe ~ 0.8
    returns_good = np.random.normal(0.0005, 0.01, 1000)
    
    # Add some autocorrelation / regime
    returns_good[0:200] += 0.001  # Bull run
    
    results_good = validator.validate_strategy(returns_good, n_strategies_tested=5)
    print("\n[Profitable Strategy Results]")
    print(results_good['summary'])
    
    # 2. Generate random noise strategy
    print("\nGenerating Random Noise Strategy...")
    returns_bad = np.random.normal(0.0, 0.01, 1000)
    
    results_bad = validator.validate_strategy(returns_bad, n_strategies_tested=5)
    print("\n[Random Strategy Results]")
    print(results_bad['summary'])

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
