"""
HIMARI OPUS V2 - Bayesian Kelly Position Sizing Engine
=======================================================

Bayesian Kelly criterion with posterior tracking for uncertainty quantification.

Features:
- Bayesian posterior tracking for win rate and edge
- Kelly fraction calculation with uncertainty
- Prior updates based on historical performance
- Comprehensive diagnostics

Version: 3.1 Enhanced
"""

from typing import Tuple, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BayesianKellyEngine:
    """
    Bayesian Kelly criterion position sizing with posterior tracking.
    
    Uses Bayesian inference to estimate win rate and edge with uncertainty
    quantification. Updates beliefs based on historical outcomes.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        kelly_fraction: float = 0.25,
        min_win_rate: float = 0.45,
        min_edge: float = 0.02
    ):
        """
        Initialize Bayesian Kelly engine.
        
        Args:
            portfolio_value: Total portfolio value in USD
            kelly_fraction: Kelly multiplier (0.25 = quarter Kelly)
            min_win_rate: Minimum win rate threshold
            min_edge: Minimum edge threshold
        """
        self.portfolio_value = portfolio_value
        self.kelly_fraction = kelly_fraction
        self.min_win_rate = min_win_rate
        self.min_edge = min_edge
        
        # Bayesian priors (Beta distribution for win rate)
        self.alpha = 10.0  # Prior wins
        self.beta = 10.0   # Prior losses
        
        # Edge priors (Normal distribution)
        self.edge_mean = 0.05  # 5% expected edge
        self.edge_std = 0.03   # 3% uncertainty
        
        # Historical tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_return = 0.0
        
    def calculate_position_size(
        self,
        confidence: float,
        expected_return: float,
        predicted_volatility: float = 0.02
    ) -> Tuple[float, Dict]:
        """
        Calculate position size using Bayesian Kelly criterion.
        
        Args:
            confidence: Signal confidence [0.0, 1.0]
            expected_return: Expected return for this trade
            predicted_volatility: Predicted return volatility
            
        Returns:
            (position_size_usd, diagnostics)
        """
        # Posterior win rate (Beta distribution)
        posterior_win_rate = self.alpha / (self.alpha + self.beta)
        win_rate_uncertainty = np.sqrt(
            (self.alpha * self.beta) / 
            ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        )
        
        # Posterior edge (using confidence-adjusted expected return)
        confidence_adjusted_edge = confidence * expected_return
        
        # Confidence-adjusted edge with Bayesian prior
        posterior_edge = (
            0.7 * confidence_adjusted_edge +  # 70% weight on current signal
            0.3 * self.edge_mean                # 30% weight on prior
        )
        
        # Edge uncertainty (combines signal uncertainty and prior)
        edge_uncertainty = np.sqrt(
            (1 - confidence)**2 * predicted_volatility**2 +
            self.edge_std**2
        )
        
        # Kelly fraction calculation
        # f = (p * (1 + b) - 1) / b, where p = win_rate, b = edge
        # Simplified: f = (p * edge) for small edges
        
        if posterior_win_rate < self.min_win_rate or posterior_edge < self.min_edge:
            # Below thresholds: return zero position
            position_size_usd = 0.0
            kelly_f = 0.0
        else:
            # Kelly fraction (simplified for small edges)
            kelly_f = posterior_win_rate * posterior_edge / predicted_volatility**2
            
            # Apply kelly_fraction multiplier (e.g., quarter Kelly)
            kelly_f *= self.kelly_fraction
            
            # Cap at reasonable maximum (50% of portfolio)
            kelly_f = min(kelly_f, 0.5)
            
            # Position size in USD
            position_size_usd = kelly_f * self.portfolio_value
        
        # Diagnostics
        diagnostics = {
            'posterior_win_rate': posterior_win_rate,
            'win_rate_uncertainty': win_rate_uncertainty,
            'posterior_edge': posterior_edge,
            'edge_uncertainty': edge_uncertainty,
            'kelly_fraction_raw': kelly_f / self.kelly_fraction if kelly_f > 0 else 0.0,
            'kelly_fraction_applied': kelly_f,
            'kelly_multiplier': self.kelly_fraction,
            'confidence': confidence,
            'expected_return': expected_return,
            'total_trades': self.total_trades,
            'historical_win_rate': self.winning_trades / max(1, self.total_trades)
        }
        
        return position_size_usd, diagnostics
    
    def update_posterior(self, trade_won: bool, trade_return: float):
        """
        Update Bayesian posterior based on trade outcome.
        
        Args:
            trade_won: True if trade was profitable
            trade_return: Actual return from trade
        """
        # Update Beta distribution for win rate
        if trade_won:
            self.alpha += 1.0
            self.winning_trades += 1
        else:
            self.beta += 1.0
        
        self.total_trades += 1
        self.total_return += trade_return
        
        # Update edge estimate (exponential moving average)
        avg_return = self.total_return / max(1, self.total_trades)
        self.edge_mean = 0.9 * self.edge_mean + 0.1 * avg_return
        
        # Update edge uncertainty (decreases with more samples)
        self.edge_std = max(0.01, self.edge_std * 0.99)  # Slowly decrease uncertainty
        
        logger.debug(
            f"Posterior updated: win_rate={self.alpha/(self.alpha+self.beta):.3f}, "
            f"edge={self.edge_mean:.4f}, trades={self.total_trades}"
        )
    
    def reset_priors(self, alpha: float = 10.0, beta: float = 10.0):
        """Reset Bayesian priors (for recalibration)."""
        self.alpha = alpha
        self.beta = beta
        self.total_trades = 0
        self.winning_trades = 0
        self.total_return = 0.0
        logger.info("Bayesian priors reset")
    
    def get_state(self) -> Dict:
        """Get current state for monitoring."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'posterior_win_rate': self.alpha / (self.alpha + self.beta),
            'edge_mean': self.edge_mean,
            'edge_std': self.edge_std,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'historical_win_rate': self.winning_trades / max(1, self.total_trades),
            'total_return': self.total_return
        }
