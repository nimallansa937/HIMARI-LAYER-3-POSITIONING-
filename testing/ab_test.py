"""
A/B Testing Framework for RL vs Bayesian Kelly
"""

import random
import json
import logging
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
import numpy as np

from src.phases.phase1_rl_enhanced import Layer3Phase1RL
from src.core.layer3_types import TacticalSignal, PositionSizingDecision

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Single A/B test result."""
    test_id: str
    timestamp: datetime
    variant: str  # "rl" or "kelly"
    signal: TacticalSignal
    decision: PositionSizingDecision
    realized_pnl: float = 0.0
    realized_return: float = 0.0
    trade_duration_seconds: float = 0.0


class ABTestRunner:
    """
    A/B testing framework for RL vs Kelly comparison.

    Split: 50% RL, 50% Kelly (Bayesian fallback)
    Duration: 2-4 weeks
    Metrics: Sharpe ratio, max DD, win rate, avg return
    """

    def __init__(
        self,
        phase1_rl: Layer3Phase1RL,
        split_ratio: float = 0.5,
        output_file: str = "ab_test_results.jsonl"
    ):
        self.phase1_rl = phase1_rl
        self.split_ratio = split_ratio
        self.output_file = output_file

        self.rl_results: List[ABTestResult] = []
        self.kelly_results: List[ABTestResult] = []

        logger.info(f"A/B Test initialized: {split_ratio:.0%} RL, {1-split_ratio:.0%} Kelly")

    def run_test(
        self,
        signal: TacticalSignal,
        cascade_indicators,
        current_price: float
    ) -> ABTestResult:
        """
        Run single A/B test iteration.

        Randomly assigns to RL or Kelly variant, records result.
        """

        # Random assignment
        use_rl = random.random() < self.split_ratio
        variant = "rl" if use_rl else "kelly"

        # Temporarily override RL enable flag
        original_enable_rl = self.phase1_rl.enable_rl
        self.phase1_rl.enable_rl = use_rl

        # Calculate position
        decision = self.phase1_rl.calculate_position(
            signal=signal,
            cascade_indicators=cascade_indicators,
            current_price=current_price
        )

        # Restore original flag
        self.phase1_rl.enable_rl = original_enable_rl

        # Record result
        result = ABTestResult(
            test_id=f"{variant}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            variant=variant,
            signal=signal,
            decision=decision
        )

        if use_rl:
            self.rl_results.append(result)
        else:
            self.kelly_results.append(result)

        # Save to file (append)
        self._save_result(result)

        logger.info(f"A/B Test: variant={variant}, position=${decision.position_size_usd:,.2f}")

        return result

    def update_result(
        self,
        test_id: str,
        realized_pnl: float,
        realized_return: float,
        trade_duration_seconds: float
    ):
        """Update test result with realized metrics after trade closes."""

        # Find result
        all_results = self.rl_results + self.kelly_results
        result = next((r for r in all_results if r.test_id == test_id), None)

        if result:
            result.realized_pnl = realized_pnl
            result.realized_return = realized_return
            result.trade_duration_seconds = trade_duration_seconds

            logger.info(
                f"Updated result {test_id}: "
                f"pnl=${realized_pnl:,.2f}, return={realized_return:.2%}"
            )

    def _save_result(self, result: ABTestResult):
        """Save result to JSONL file."""
        with open(self.output_file, 'a') as f:
            # Convert to dict (handle nested dataclasses)
            result_dict = {
                'test_id': result.test_id,
                'timestamp': result.timestamp.isoformat(),
                'variant': result.variant,
                'position_size_usd': result.decision.position_size_usd,
                'kelly_fraction': result.decision.kelly_fraction,
                'realized_pnl': result.realized_pnl,
                'realized_return': result.realized_return,
            }
            f.write(json.dumps(result_dict) + '\n')

    def analyze_results(self) -> Dict:
        """
        Analyze A/B test results.

        Returns statistical comparison of RL vs Kelly performance.
        """

        if len(self.rl_results) < 20 or len(self.kelly_results) < 20:
            logger.warning("Insufficient data for analysis (need 20+ per variant)")
            return {}

        # Calculate metrics
        rl_returns = [r.realized_return for r in self.rl_results if r.realized_return != 0]
        kelly_returns = [r.realized_return for r in self.kelly_results if r.realized_return != 0]

        def calculate_sharpe(returns):
            if len(returns) < 2:
                return 0.0
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return == 0:
                return 0.0
            return (mean_return / std_return) * np.sqrt(252)

        def calculate_max_dd(returns):
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown) if len(drawdown) > 0 else 0.0

        analysis = {
            'rl': {
                'n_trades': len(rl_returns),
                'mean_return': np.mean(rl_returns),
                'std_return': np.std(rl_returns),
                'sharpe': calculate_sharpe(rl_returns),
                'max_dd': calculate_max_dd(rl_returns),
                'win_rate': sum(1 for r in rl_returns if r > 0) / len(rl_returns),
            },
            'kelly': {
                'n_trades': len(kelly_returns),
                'mean_return': np.mean(kelly_returns),
                'std_return': np.std(kelly_returns),
                'sharpe': calculate_sharpe(kelly_returns),
                'max_dd': calculate_max_dd(kelly_returns),
                'win_rate': sum(1 for r in kelly_returns if r > 0) / len(kelly_returns),
            }
        }

        # Statistical significance (t-test)
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(rl_returns, kelly_returns)

        analysis['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Performance delta
        analysis['delta'] = {
            'sharpe': analysis['rl']['sharpe'] - analysis['kelly']['sharpe'],
            'mean_return': analysis['rl']['mean_return'] - analysis['kelly']['mean_return'],
            'max_dd': analysis['rl']['max_dd'] - analysis['kelly']['max_dd'],
        }

        return analysis

    def print_report(self):
        """Print A/B test report."""

        analysis = self.analyze_results()

        if not analysis:
            print("Insufficient data for analysis")
            return

        print("=" * 80)
        print("A/B TEST RESULTS: RL vs Bayesian Kelly")
        print("=" * 80)
        print()

        print(f"RL Variant ({analysis['rl']['n_trades']} trades):")
        print(f"  Sharpe Ratio:   {analysis['rl']['sharpe']:.3f}")
        print(f"  Mean Return:    {analysis['rl']['mean_return']:.2%}")
        print(f"  Std Return:     {analysis['rl']['std_return']:.2%}")
        print(f"  Max Drawdown:   {analysis['rl']['max_dd']:.2%}")
        print(f"  Win Rate:       {analysis['rl']['win_rate']:.1%}")
        print()

        print(f"Kelly Variant ({analysis['kelly']['n_trades']} trades):")
        print(f"  Sharpe Ratio:   {analysis['kelly']['sharpe']:.3f}")
        print(f"  Mean Return:    {analysis['kelly']['mean_return']:.2%}")
        print(f"  Std Return:     {analysis['kelly']['std_return']:.2%}")
        print(f"  Max Drawdown:   {analysis['kelly']['max_dd']:.2%}")
        print(f"  Win Rate:       {analysis['kelly']['win_rate']:.1%}")
        print()

        print("Performance Delta (RL - Kelly):")
        print(f"  Sharpe:         {analysis['delta']['sharpe']:+.3f}")
        print(f"  Mean Return:    {analysis['delta']['mean_return']:+.2%}")
        print(f"  Max DD:         {analysis['delta']['max_dd']:+.2%}")
        print()

        print("Statistical Test:")
        print(f"  t-statistic:    {analysis['statistical_test']['t_statistic']:.3f}")
        print(f"  p-value:        {analysis['statistical_test']['p_value']:.4f}")
        print(f"  Significant:    {'YES' if analysis['statistical_test']['significant'] else 'NO'} (α=0.05)")
        print()

        # Recommendation
        if analysis['statistical_test']['significant']:
            if analysis['delta']['sharpe'] > 0.1:
                print("✓ RECOMMENDATION: Deploy RL to 100% of traffic")
            elif analysis['delta']['sharpe'] < -0.1:
                print("✗ RECOMMENDATION: Disable RL, use Bayesian Kelly")
            else:
                print("⊙ RECOMMENDATION: Continue monitoring (small difference)")
        else:
            print("⊙ RECOMMENDATION: No significant difference, continue testing")

        print("=" * 80)


if __name__ == "__main__":
    """Example A/B test run."""

    from src.core.layer3_types import TacticalAction, MarketRegime, CascadeIndicators

    # Initialize Phase 1 RL with GCP API
    phase1 = Layer3Phase1RL(
        portfolio_value=100000,
        enable_rl=True,
        rl_api_endpoint="https://himari-rl-api-abc123-uc.a.run.app/predict",
        rl_timeout_ms=150
    )

    # Initialize A/B test
    ab_test = ABTestRunner(phase1, split_ratio=0.5)

    # Simulate 100 trades
    print("Running 100 simulated trades...")
    for i in range(100):
        # Create test signal
        signal = TacticalSignal(
            strategy_id="test",
            symbol="BTC-USD",
            action=random.choice(list(TacticalAction)),
            confidence=random.uniform(0.6, 0.95),
            risk_score=random.uniform(0.1, 0.4),
            regime=random.choice(list(MarketRegime)),
            timestamp_ns=int(datetime.now().timestamp() * 1e9),
            expected_return=random.uniform(0.01, 0.08),
        )

        cascade = CascadeIndicators(
            funding_rate=random.uniform(-0.001, 0.003),
            oi_change_pct=random.uniform(-0.1, 0.1),
            volume_ratio=random.uniform(0.5, 3.0),
            onchain_whale_pressure=random.uniform(0.1, 0.6),
            exchange_netflow_zscore=random.uniform(-1, 1),
        )

        # Run test
        result = ab_test.run_test(signal, cascade, 87000.0)

        # Simulate realized return
        realized_return = random.gauss(0.01, 0.03)
        realized_pnl = result.decision.position_size_usd * realized_return

        ab_test.update_result(
            result.test_id,
            realized_pnl=realized_pnl,
            realized_return=realized_return,
            trade_duration_seconds=random.uniform(300, 7200)
        )

    # Print report
    ab_test.print_report()
