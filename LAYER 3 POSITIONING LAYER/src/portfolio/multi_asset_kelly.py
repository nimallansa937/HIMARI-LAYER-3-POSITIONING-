"""
HIMARI OPUS V2 - Multi-Asset Kelly Allocator
=============================================

Portfolio-level Kelly allocation across multiple assets with 
correlation-aware position sizing.

Features:
- Multi-asset Kelly criterion
- Correlation matrix integration
- Position limit enforcement
- Rebalancing signals
- Risk budget constraints

Version: 3.1 Phase 2
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiAssetKellyAllocator:
    """
    Multi-asset Kelly position allocator.
    
    Implements the multi-asset Kelly criterion which accounts for
    correlations between assets when determining optimal position sizes.
    
    The classic single-asset Kelly formula is:
        f* = (μ - r) / σ²
    
    For multiple correlated assets:
        f* = Σ⁻¹ @ (μ - r)
    
    Where Σ is the covariance matrix of returns.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = 0.25,
        max_portfolio_utilization: float = 0.80,
        risk_free_rate: float = 0.05,
        kelly_fraction: float = 0.25
    ):
        """
        Initialize multi-asset Kelly allocator.
        
        Args:
            portfolio_value: Total portfolio value in USD
            max_position_pct: Maximum single position as % of portfolio
            max_portfolio_utilization: Maximum total utilization %
            risk_free_rate: Annual risk-free rate (default 5%)
            kelly_fraction: Kelly multiplier (0.25 = quarter Kelly)
        """
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_portfolio_utilization = max_portfolio_utilization
        self.risk_free_rate = risk_free_rate
        self.kelly_fraction = kelly_fraction
        
        # Tracking
        self.total_allocations = 0
        self.position_limit_hits = 0
        self.utilization_limit_hits = 0
        
        logger.info(
            f"MultiAssetKellyAllocator initialized: "
            f"portfolio=${portfolio_value:,.0f}, kelly={kelly_fraction}"
        )
    
    def allocate(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Allocate capital across multiple assets using Kelly criterion.
        
        Args:
            expected_returns: {symbol: expected_return} annualized
            volatilities: {symbol: volatility} annualized std dev
            correlation_matrix: NxN correlation matrix (optional)
            symbols: Order of symbols in correlation matrix
            
        Returns:
            Tuple of (allocations_usd, diagnostics)
        """
        self.total_allocations += 1
        
        # Validate inputs
        if not expected_returns:
            return {}, {'status': 'no_assets', 'error': 'No expected returns provided'}
        
        # Validate and clean inputs
        valid_symbols = []
        for sym, ret in expected_returns.items():
            vol = volatilities.get(sym, 0.01)
            
            # Skip invalid values
            if ret is None or np.isnan(ret) or np.isinf(ret):
                logger.warning(f"Invalid expected return for {sym}: {ret}")
                continue
            if vol is None or np.isnan(vol) or np.isinf(vol) or vol <= 0:
                logger.warning(f"Invalid volatility for {sym}: {vol}")
                continue
            
            valid_symbols.append(sym)
        
        if not valid_symbols:
            return {}, {'status': 'no_valid_assets', 'error': 'All inputs failed validation'}
        
        # Get ordered symbol list
        if symbols is None:
            symbols = valid_symbols
        else:
            # Filter to valid symbols only
            symbols = [s for s in symbols if s in valid_symbols]
        
        n_assets = len(symbols)
        
        # Build return and volatility vectors
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])
        sigma = np.array([volatilities.get(s, 0.01) for s in symbols])
        
        # Build covariance matrix
        if correlation_matrix is not None and correlation_matrix.shape == (n_assets, n_assets):
            # Convert correlation to covariance: Cov = diag(σ) @ Corr @ diag(σ)
            sigma_diag = np.diag(sigma)
            cov_matrix = sigma_diag @ correlation_matrix @ sigma_diag
        else:
            # Assume uncorrelated (diagonal covariance)
            cov_matrix = np.diag(sigma ** 2)
        
        # Multi-asset Kelly: f* = Σ⁻¹ @ (μ - r)
        try:
            excess_returns = mu - self.risk_free_rate
            
            # Use pseudo-inverse for numerical stability
            cov_inv = np.linalg.pinv(cov_matrix)
            kelly_weights = cov_inv @ excess_returns
            
            # Apply Kelly fraction (quarter-Kelly, etc.)
            kelly_weights *= self.kelly_fraction
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Covariance matrix inversion failed: {e}")
            # Fallback to single-asset Kelly
            kelly_weights = (mu - self.risk_free_rate) / (sigma ** 2) * self.kelly_fraction
        
        # Apply constraints
        allocations_usd = {}
        total_allocated = 0.0
        max_single_position = self.portfolio_value * self.max_position_pct
        max_total = self.portfolio_value * self.max_portfolio_utilization
        
        constraint_hits = []
        
        for i, symbol in enumerate(symbols):
            raw_weight = kelly_weights[i]
            
            # Skip negative weights (short positions not supported in Phase 2)
            if raw_weight <= 0:
                allocations_usd[symbol] = 0.0
                continue
            
            # Calculate raw USD allocation
            raw_usd = raw_weight * self.portfolio_value
            
            # Apply single position limit
            if raw_usd > max_single_position:
                raw_usd = max_single_position
                self.position_limit_hits += 1
                constraint_hits.append(f"{symbol}: position_limit")
            
            # Check total utilization limit
            if total_allocated + raw_usd > max_total:
                # Scale down to fit
                remaining = max_total - total_allocated
                if remaining > 0:
                    raw_usd = remaining
                    self.utilization_limit_hits += 1
                    constraint_hits.append(f"{symbol}: utilization_limit")
                else:
                    raw_usd = 0.0
            
            allocations_usd[symbol] = raw_usd
            total_allocated += raw_usd
        
        # Diagnostics
        diagnostics = {
            'n_assets': n_assets,
            'symbols': symbols,
            'kelly_weights': dict(zip(symbols, kelly_weights.tolist())),
            'total_allocated_usd': total_allocated,
            'utilization_pct': total_allocated / self.portfolio_value * 100,
            'constraint_hits': constraint_hits,
            'status': 'success'
        }
        
        logger.info(
            f"Multi-asset allocation: {n_assets} assets, "
            f"${total_allocated:,.2f} allocated ({diagnostics['utilization_pct']:.1f}%)"
        )
        
        return allocations_usd, diagnostics
    
    def get_rebalancing_signals(
        self,
        current_positions: Dict[str, float],
        target_allocations: Dict[str, float],
        rebalance_threshold: float = 0.10
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate rebalancing trades needed.
        
        Args:
            current_positions: {symbol: current_usd}
            target_allocations: {symbol: target_usd}
            rebalance_threshold: Minimum deviation to trigger rebalance
            
        Returns:
            Tuple of (trades, diagnostics) where trades = {symbol: delta_usd}
        """
        trades = {}
        deviations = {}
        
        all_symbols = set(current_positions.keys()) | set(target_allocations.keys())
        
        for symbol in all_symbols:
            current = current_positions.get(symbol, 0.0)
            target = target_allocations.get(symbol, 0.0)
            
            delta = target - current
            
            # Calculate deviation percentage
            if target > 0:
                deviation_pct = abs(delta) / target
            elif current > 0:
                deviation_pct = 1.0  # 100% deviation if current but no target
            else:
                deviation_pct = 0.0
            
            deviations[symbol] = deviation_pct
            
            # Only generate trade if above threshold
            if deviation_pct >= rebalance_threshold:
                trades[symbol] = delta
        
        diagnostics = {
            'deviations': deviations,
            'rebalance_threshold': rebalance_threshold,
            'trades_generated': len(trades)
        }
        
        return trades, diagnostics
    
    def get_statistics(self) -> Dict:
        """Get allocator statistics."""
        return {
            'portfolio_value': self.portfolio_value,
            'kelly_fraction': self.kelly_fraction,
            'max_position_pct': self.max_position_pct,
            'max_portfolio_utilization': self.max_portfolio_utilization,
            'total_allocations': self.total_allocations,
            'position_limit_hits': self.position_limit_hits,
            'utilization_limit_hits': self.utilization_limit_hits
        }
